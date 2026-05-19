import json
import logging
import time
import random
import traceback
from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from groq import Groq
import groq

from core.config import settings
from db.models import QueryLog
from rag.pipeline import retrieve_relevant_chunks
from tools import TOOL_REGISTRY, execute_tool, get_tool_definitions

logger = logging.getLogger("helpdesk_assistant.agent")

MAX_REACT_STEPS = 4
MAX_SQL_CORRECTIONS = 3


# ---------------------------------------------------------------------------
# Pydantic schemas for the ReAct agent decision cycle
# ---------------------------------------------------------------------------

class AgentAction(BaseModel):
    action_type: Literal["tool", "sql_query", "rag_search"] = Field(
        description="Type of action: 'tool' to run calculator/file_search, "
                    "'sql_query' for a read-only database query, "
                    "'rag_search' for semantic document retrieval."
    )
    tool_name: Optional[str] = Field(
        None,
        description="Name of the tool: 'calculator' or 'file_search'. Required when action_type='tool'."
    )
    tool_args_json: Optional[str] = Field(
        None,
        description="JSON string of tool arguments, e.g. '{\"expression\": \"2+2\"}'. Required when action_type='tool'."
    )
    sql_query: Optional[str] = Field(
        None,
        description="A read-only PostgreSQL SELECT query. Required when action_type='sql_query'."
    )
    rag_query: Optional[str] = Field(
        None,
        description="Natural-language search query for RAG retrieval. Required when action_type='rag_search'."
    )


class AgentDecision(BaseModel):
    step_type: Literal["action", "finalize"] = Field(
        description="'action' to gather more information; 'finalize' to deliver the final answer."
    )
    thought: str = Field(
        description="Step-by-step reasoning: what is known so far, what is still needed, "
                    "and why this step was chosen."
    )
    action: Optional[AgentAction] = Field(
        None,
        description="The action to execute next. Required when step_type='action'."
    )
    final_response: Optional[str] = Field(
        None,
        description="The complete, synthesized final answer for the user. "
                    "Required when step_type='finalize'."
    )


# ---------------------------------------------------------------------------
# Groq client helpers  (unchanged public signatures)
# ---------------------------------------------------------------------------

def get_groq_client() -> Groq:
    """Instantiate standard Groq API client using API key."""
    return Groq(api_key=settings.GROQ_API_KEY)


def call_groq_with_retry(
    client: Groq,
    model: str,
    messages: List[Dict[str, str]],
    response_format: Optional[Dict[str, Any]] = None,
    temperature: float = 0.2,
    max_retries: int = 5,
    initial_delay: float = 3.0
) -> Any:
    """
    Execute a Groq API call with robust exponential backoff, jitter,
    and automatic rate limit recovery.
    """
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=temperature
            )
            return response
        except groq.RateLimitError as e:
            sleep_time = delay * (2 ** attempt) + random.uniform(0.5, 1.5)
            logger.warning(
                f"Groq API rate limited on attempt {attempt+1}/{max_retries}. "
                f"Retrying in {sleep_time:.2f} seconds..."
            )
            time.sleep(sleep_time)
            continue
        except groq.APIError as e:
            sleep_time = delay + random.uniform(0.2, 0.8)
            logger.warning(
                f"Groq API error ({type(e).__name__}) on attempt {attempt+1}/{max_retries}. "
                f"Retrying in {sleep_time:.2f} seconds..."
            )
            time.sleep(sleep_time)
            continue
        except Exception as e:
            sleep_time = delay + random.uniform(0.2, 0.8)
            logger.warning(
                f"Transient error ({type(e).__name__}) on attempt {attempt+1}/{max_retries}. "
                f"Retrying in {sleep_time:.2f} seconds..."
            )
            time.sleep(sleep_time)
            continue

    # Final failover attempt
    logger.error("Groq API call failed after max retries due to quota limits.")
    return client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format,
        temperature=temperature
    )


# ---------------------------------------------------------------------------
# Safe SQL execution  (unchanged)
# ---------------------------------------------------------------------------

def execute_safe_select_query(db: Session, sql_query: str) -> List[Dict[str, Any]]:
    """Execute a SELECT SQL query safely, ensuring it is read-only."""
    query_upper = sql_query.strip().upper()

    # Simple syntax protection check
    if not query_upper.startswith("SELECT"):
        raise ValueError("Security violation: Only SELECT queries are permitted.")

    for forbidden in ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "REPLACE"]:
        if forbidden in query_upper:
            raise ValueError(f"Security violation: Forbidden keyword '{forbidden}' detected in query.")

    logger.info(f"Running safe PostgreSQL SELECT: {sql_query}")
    result = db.execute(text(sql_query))

    # Convert result rows to dictionary representation
    columns = result.keys()
    rows = [dict(zip(columns, row)) for row in result.fetchall()]
    return rows


# ---------------------------------------------------------------------------
# Final synthesis fallback  (unchanged, used when ReAct loop exhausts steps)
# ---------------------------------------------------------------------------

def synthesize_answer(query: str, routing_decision: str, context_details: dict) -> str:
    """
    Final answer synthesis:
    Injects the retrieved data, SQL output, or tool results into the final LLM prompt 
    to output a beautiful, helpful response.
    """
    client = get_groq_client()
    
    prompt = f"""
    You are an elite Senior IT Helpdesk Specialist and Systems Architect. Your objective is to formulate a precise, highly structured, and authoritative technical response to the user's inquiry based strictly on the provided context.
    
    User Inquiry: "{query}"
    Routing Path Selected: {routing_decision}
    
    Context Information:
    {json.dumps(context_details, indent=2, default=str)}
    
    Instructions:
    1. Base your response strictly on the validated context details. If the context is insufficient, state so with professional transparency and directness (e.g., "The retrieved technical documentation does not specify...").
    2. Adhere to a premium, executive-grade corporate tone: use articulate technical vocabulary, precise phrasing, and professional structure. Avoid colloquialisms, contractions, or generic filler text.
    3. Structure your response clearly using professional Markdown formatting (bullet points, clear headers, bold terminology, and code blocks for logs/queries/commands) to ensure high readability.
    4. Maintain technical accuracy. If listing database rows or values, present them in a neat, well-organized format.
    5. Do not expose internal system routing jargon (e.g., "routing pathway selected", "RAG query", "SQL execution pipeline") to the end user unless they explicitly ask for system mechanics. Maintain a seamless, professional interface.
    """
    
    try:
        response = call_groq_with_retry(
            client=client,
            model=settings.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a Senior IT Helpdesk Specialist and Systems Architect. You communicate with absolute clarity, elite corporate courtesy, and precise technical vocabulary, organizing your responses with professional structure and Markdown formatting."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error during final response synthesis: {e}")
        return f"I processed your query, but encountered an error generating the final summary. Raw Context: {str(context_details)}"


# ---------------------------------------------------------------------------
# ReAct prompt builders
# ---------------------------------------------------------------------------

def _build_tool_schema_text() -> str:
    """Build a plain-text description of all registered tools for the system prompt."""
    tools = get_tool_definitions()
    if not tools:
        return "No tools registered."
    lines = []
    for t in tools:
        lines.append(f"- {t['name']}: {t['description']}")
        lines.append(f"  Parameters: {json.dumps(t['parameters'], default=str)}")
    return "\n".join(lines)


def _format_scratchpad(scratchpad: List[Dict[str, Any]]) -> str:
    """Render the execution scratchpad as readable text for the LLM prompt."""
    if not scratchpad:
        return "(No actions taken yet — this is the first step.)"

    parts = []
    for entry in scratchpad:
        step_num = entry.get("step", "?")
        parts.append(f"Step {step_num}:")
        parts.append(f"  Thought: {entry.get('thought', 'N/A')}")
        parts.append(f"  Action: {entry.get('action_summary', 'N/A')}")
        obs = entry.get("observation", "N/A")
        # Truncate very long observations to avoid blowing up the prompt
        if isinstance(obs, str) and len(obs) > 2000:
            obs = obs[:2000] + "...[truncated]"
        elif isinstance(obs, (list, dict)):
            obs_str = json.dumps(obs, default=str)
            if len(obs_str) > 2000:
                obs = obs_str[:2000] + "...[truncated]"
            else:
                parts.append(f"  Observation: {obs_str}")
                continue
        parts.append(f"  Observation: {obs}")
    return "\n".join(parts)


def _build_react_system_prompt(query: str, scratchpad: List[Dict[str, Any]]) -> str:
    """Build the full system prompt for one ReAct decision step."""
    tool_schema = _build_tool_schema_text()
    scratchpad_text = _format_scratchpad(scratchpad)

    return f"""You are an advanced ReAct (Reasoning + Action) IT Helpdesk Agent. You operate in an iterative loop: at each step you either take ONE action to gather information, or finalize with a complete answer.

=== USER QUERY ===
{query}

=== SCRATCHPAD (actions taken so far) ===
{scratchpad_text}

=== AVAILABLE ACTIONS ===
1. rag_search — Search internal IT documentation semantically. Provide a 'rag_query' string.
2. sql_query — Execute a read-only PostgreSQL SELECT query against the helpdesk database.
   Available tables:
     - users (id INTEGER, username VARCHAR(100), email VARCHAR(255), status VARCHAR(50) ['active','offline','suspended'], role VARCHAR(50) ['admin','developer','user'], created_at TIMESTAMPTZ)
     - documents (id INTEGER, content TEXT, metadata JSONB, created_at TIMESTAMPTZ)
     - queries (id INTEGER, input_query TEXT, output_response TEXT, routing_decision VARCHAR(50), tools_used JSONB, latency DOUBLE PRECISION, timestamp TIMESTAMPTZ)
   Provide a 'sql_query' string containing ONLY a valid PostgreSQL read-only SELECT statement.
3. tool — Execute a registered tool. Available tools:
{tool_schema}
   Provide 'tool_name' and 'tool_args_json' (JSON string of the arguments object).

=== RULES ===
- If the scratchpad already contains enough information to fully answer the user query, set step_type to "finalize" and write the complete answer in final_response.
- If you still need information, set step_type to "action" and choose exactly ONE action.
- You have a maximum of {MAX_REACT_STEPS} steps total (including this one). Be efficient.
- SQL queries MUST be read-only SELECT statements. Never generate INSERT/UPDATE/DELETE/DROP.
- When using 'tool' action_type, tool_args_json must be a valid JSON string like '{{"expression": "2+2"}}'.
- Maintain an objective, analytical, and highly professional senior technician tone throughout. Your thoughts should detail precise system indicators, query requirements, and logical deductions step-by-step.
- Output ONLY a valid JSON object matching the schema below. No markdown fences, no preamble.

=== OUTPUT SCHEMA ===
{{
  "step_type": "action" | "finalize",
  "thought": "Your step-by-step reasoning string.",
  "action": {{
    "action_type": "tool" | "sql_query" | "rag_search",
    "tool_name": "calculator" | "file_search" | null,
    "tool_args_json": "JSON string of args" | null,
    "sql_query": "SELECT ..." | null,
    "rag_query": "search string" | null
  }} | null,
  "final_response": "Complete answer text" | null
}}"""


# ---------------------------------------------------------------------------
# SQL self-healing loop
# ---------------------------------------------------------------------------

def _sql_self_healing(client: Groq, db: Session, sql_query: str) -> tuple:
    """
    Execute a SQL query with up to MAX_SQL_CORRECTIONS automatic correction
    attempts via Groq.  Returns (final_sql, results_or_None, error_or_None).
    """
    current_sql = sql_query

    for attempt in range(MAX_SQL_CORRECTIONS + 1):
        try:
            results = execute_safe_select_query(db, current_sql)
            return current_sql, results, None
        except Exception as exc:
            error_trace = traceback.format_exc()
            logger.warning(
                f"SQL execution failed (attempt {attempt+1}/{MAX_SQL_CORRECTIONS+1}): {exc}"
            )

            if attempt >= MAX_SQL_CORRECTIONS:
                return current_sql, None, (
                    f"SQL query failed after {MAX_SQL_CORRECTIONS} self-correction "
                    f"attempts.\nLast SQL: {current_sql}\nError: {error_trace}"
                )

            # Build micro-correction prompt
            correction_prompt = f"""The following PostgreSQL SELECT query failed with an error.

FAILED SQL:
{current_sql}

ERROR:
{error_trace}

Please output ONLY the corrected, valid PostgreSQL read-only SELECT query.
Do not include any markdown fencing, explanation, or commentary — just the raw corrected SQL statement.
Available tables: users (id, username, email, status, role, created_at), documents (id, content, metadata, created_at), queries (id, input_query, output_response, routing_decision, tools_used, latency, timestamp)."""

            try:
                response = call_groq_with_retry(
                    client=client,
                    model=settings.LLM_MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a PostgreSQL expert. Output only raw, corrected SQL. No markdown, no explanation."},
                        {"role": "user", "content": correction_prompt}
                    ],
                    response_format=None,
                    temperature=0.0
                )
                corrected = response.choices[0].message.content.strip()
                # Strip any markdown code fences the model may have emitted
                for fence in ("```sql", "```", "~~~"):
                    corrected = corrected.replace(fence, "")
                corrected = corrected.strip().rstrip(";").strip()
                if corrected:
                    current_sql = corrected
                    logger.info(f"SQL self-correction attempt {attempt+1}: new query = {current_sql}")
                else:
                    logger.error("Groq returned an empty SQL correction.")
                    return current_sql, None, f"Empty correction response at attempt {attempt+1}."
            except Exception as llm_err:
                logger.error(f"LLM call during SQL self-healing failed: {llm_err}")
                return current_sql, None, f"LLM error during SQL self-correction: {llm_err}"

    return current_sql, None, "SQL self-healing exhausted all correction attempts."


# ---------------------------------------------------------------------------
# Single ReAct decision call
# ---------------------------------------------------------------------------

def _call_agent_decision(client: Groq, query: str, scratchpad: List[Dict[str, Any]]) -> AgentDecision:
    """Call Groq to get the next AgentDecision (action or finalize)."""
    system_prompt = _build_react_system_prompt(query, scratchpad)

    response = call_groq_with_retry(
        client=client,
        model=settings.LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a precise ReAct agent that outputs ONLY valid JSON matching the requested schema. No markdown, no commentary."},
            {"role": "user", "content": system_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )

    res_text = response.choices[0].message.content.strip()
    data = json.loads(res_text)
    return AgentDecision(**data)


# ---------------------------------------------------------------------------
# Main ReAct orchestrator  (backward-compatible signature & return keys)
# ---------------------------------------------------------------------------

def process_pipeline(db: Session, query: str) -> dict:
    """
    ReAct (Reasoning + Action) agent loop that replaces the old one-shot router.

    1. Iterate up to MAX_REACT_STEPS times, maintaining an execution scratchpad.
    2. At each step, call Groq for an AgentDecision: either take an action
       (tool / SQL with self-healing / RAG) or finalize with a response.
    3. After the loop, synthesize a fallback answer if no final_response was produced.
    4. Log latency, tools used, and outcomes to the 'queries' table.

    Returns the exact same dict keys as the original implementation:
      query, routing_decision, response, retrieved_documents, tools_used,
      latency_seconds, sql_query, sql_results, tool_results
    """
    start_time = time.time()

    # --- tracking variables (same names & semantics as original) ---
    routing_decision = "REACT_AGENT"
    retrieved_docs: List[Any] = []
    tools_used: List[Dict[str, Any]] = []
    sql_query_used: Optional[str] = None
    sql_results: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None
    scratchpad: List[Dict[str, Any]] = []

    client = get_groq_client()

    # --- High-Performance Fast-Path Elite Greeting & Capability Engine ---
    clean_query = query.strip().lower().rstrip("?.!")
    
    # 1. Pure Greetings Check
    simple_greetings = {"hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy", "hola"}
    is_greeting = clean_query in simple_greetings or any(clean_query.startswith(g + " ") for g in simple_greetings) or clean_query.startswith("hello ") or clean_query.startswith("hi ") or clean_query.startswith("hey ")
    
    # 2. Capabilities Check
    capability_phrases = ["who are you", "what can you do", "how can you help", "help me", "what is your role", "what are your capabilities", "what are your features", "capabilities", "features"]
    is_capability = any(phrase in clean_query for phrase in capability_phrases) or (clean_query == "help")
    
    # 3. Appreciation Check
    thanks_phrases = {"thanks", "thank you", "much appreciated", "thanks a lot", "thank you very much", "thx"}
    is_thanks = clean_query in thanks_phrases or any(clean_query.startswith(t + " ") for t in thanks_phrases) or clean_query.startswith("thank you")

    if is_greeting or is_capability or is_thanks:
        if is_capability:
            routing_decision = "FAST_PATH_CAPABILITY"
            final_answer = (
                "Greetings. As your Senior IT Helpdesk Specialist and Systems Architect, "
                "I stand ready to facilitate a wide range of analytical and diagnostic operations. "
                "My engineered capabilities include:\n\n"
                "- **Semantic RAG Search**: Perform high-precision query lookups across internal IT policies, networking manuals, and security guidelines using vector embedding similarity.\n"
                "- **Database Diagnostics (SQL)**: Execute secure, read-only PostgreSQL SELECT queries to monitor live tables (e.g., query logs, system user roles, account statuses) with automatic self-healing syntax correction.\n"
                "- **Specialized Tooling**: Leverage local system modules, including a high-performance mathematical calculator and a local codebase keyword search crawler.\n"
                "- **Multi-Source Synthesis**: Dynamically coordinate and cross-reference retrieved documentation and live database records to synthesize single, high-fidelity technical briefs.\n\n"
                "Please specify the database logs, system documentation, or analytical parameters you would like to examine, and I will execute the optimal processing pathway immediately."
            )
        elif is_thanks:
            routing_decision = "FAST_PATH_THANKS"
            final_answer = (
                "You are very welcome. It is my professional priority to ensure your technical operations "
                "remain seamless and optimal. Should you require further system diagnostics, query logs, "
                "or internal documentation searches, I remain online and ready to assist.\n\n"
                "Have a productive and successful day."
            )
        else:
            routing_decision = "FAST_PATH_GREETING"
            final_answer = (
                "Good day. I am your Senior IT Helpdesk and Systems Assistant. "
                "My diagnostic systems are fully operational, and I am prepared to assist you with "
                "IT infrastructure queries, PostgreSQL database metrics, or technical process documentation.\n\n"
                "Please state your inquiry or select a diagnostic path, and I will facilitate immediate technical support."
            )

        latency = time.time() - start_time
        return {
            "query": query,
            "routing_decision": routing_decision,
            "response": final_answer,
            "retrieved_documents": [],
            "tools_used": [],
            "latency_seconds": latency,
            "sql_query": None,
            "sql_results": None,
            "tool_results": None
        }

    try:
        # ---- ReAct loop ----
        for step_idx in range(1, MAX_REACT_STEPS + 1):
            logger.info(f"ReAct step {step_idx}/{MAX_REACT_STEPS} for query: '{query}'")

            try:
                decision = _call_agent_decision(client, query, scratchpad)
            except Exception as decide_err:
                logger.error(f"Agent decision call failed at step {step_idx}: {decide_err}", exc_info=True)
                # Append error to scratchpad so the next iteration knows about it
                scratchpad.append({
                    "step": step_idx,
                    "thought": "Decision call failed",
                    "action_summary": "LLM error",
                    "observation": f"Error: {str(decide_err)}"
                })
                continue

            thought = decision.thought
            logger.info(f"ReAct step {step_idx} decision: {decision.step_type} — {thought[:120]}")

            # --- FINALIZE branch ---
            if decision.step_type == "finalize" and decision.final_response:
                final_answer = decision.final_response
                routing_decision = "REACT_AGENT_FINALIZED"
                logger.info("Agent chose to finalize.")
                break

            # --- ACTION branch ---
            if decision.step_type == "action" and decision.action:
                action = decision.action
                action_summary = f"{action.action_type}"
                observation: Any = None

                # -- RAG search --
                if action.action_type == "rag_search":
                    rag_q = action.rag_query or query
                    action_summary = f"rag_search({rag_q!r})"
                    try:
                        retrieved = retrieve_relevant_chunks(db, rag_q, top_k=3)
                        retrieved_docs.extend(retrieved)
                        observation = retrieved
                        if not tools_used or tools_used[-1].get("action_type") != "rag_search":
                            tools_used.append({"action_type": "rag_search", "rag_query": rag_q})
                    except Exception as rag_err:
                        observation = {"error": str(rag_err)}
                        logger.error(f"RAG search failed: {rag_err}")

                # -- SQL query  (with self-healing) --
                elif action.action_type == "sql_query":
                    raw_sql = action.sql_query
                    action_summary = f"sql_query({raw_sql!r})"
                    if raw_sql:
                        healed_sql, healed_results, healed_error = _sql_self_healing(
                            client, db, raw_sql
                        )
                        sql_query_used = healed_sql
                        sql_results = healed_results
                        tools_used.append({"action_type": "sql_query", "sql": healed_sql})
                        if healed_error:
                            observation = {"sql_error": healed_error}
                        else:
                            observation = healed_results
                    else:
                        observation = {"error": "sql_query action had no SQL text."}
                        logger.warning("Agent chose sql_query but provided no SQL.")

                # -- Tool execution --
                elif action.action_type == "tool":
                    tool_name = action.tool_name
                    tool_args_str = action.tool_args_json or "{}"
                    action_summary = f"tool({tool_name}, {tool_args_str!r})"

                    tool_args: Dict[str, Any] = {}
                    try:
                        tool_args = json.loads(tool_args_str)
                    except json.JSONDecodeError:
                        tool_args = {"raw_args": tool_args_str}

                    if tool_name:
                        try:
                            tool_res = execute_tool(tool_name, tool_args)
                            tool_results = tool_res
                            observation = tool_res
                            tools_used.append({"tool": tool_name, "args": tool_args})
                        except Exception as tool_err:
                            observation = {"error": str(tool_err)}
                            logger.error(f"Tool execution failed: {tool_err}")
                    else:
                        observation = {"error": "tool action had no tool_name."}

                else:
                    observation = {"error": f"Unknown action_type: {action.action_type}"}

                # Append step to scratchpad
                scratchpad.append({
                    "step": step_idx,
                    "thought": thought,
                    "action_summary": action_summary,
                    "observation": observation,
                })

            else:
                # Malformed decision — neither valid action nor finalize
                logger.warning(f"Agent returned malformed decision at step {step_idx}: {decision.model_dump()}")
                scratchpad.append({
                    "step": step_idx,
                    "thought": thought,
                    "action_summary": "malformed_decision",
                    "observation": "Decision was neither a valid action nor a finalize with content."
                })

        # ---- Post-loop: synthesize fallback if no final_answer yet ----
        if final_answer is None:
            routing_decision = "REACT_AGENT_FALLBACK_SYNTHESIS"
            context_details: Dict[str, Any] = {
                "scratchpad": scratchpad,
                "retrieved_documents": retrieved_docs,
            }
            if sql_results is not None:
                context_details["database_results"] = sql_results
            if tool_results is not None:
                context_details["tool_results"] = tool_results
            final_answer = synthesize_answer(query, routing_decision, context_details)

    except Exception as pipeline_err:
        logger.error(f"Error in ReAct orchestration pipeline: {pipeline_err}", exc_info=True)
        routing_decision = "ERROR_FALLBACK"
        final_answer = (
            f"I'm sorry, an internal server error occurred while processing your request: "
            f"{str(pipeline_err)}"
        )

    # ---- Metrics & audit logging ----
    latency = time.time() - start_time

    log_tools_payload = list(tools_used)
    if sql_query_used:
        log_tools_payload.append({"sql_executed": sql_query_used})

    try:
        query_log = QueryLog(
            input_query=query,
            output_response=final_answer,
            routing_decision=routing_decision,
            tools_used=log_tools_payload,
            latency=latency
        )
        db.add(query_log)
        db.commit()
        logger.info(f"ReAct pipeline completed in {latency:.4f}s. Decision logged to queries table.")
    except Exception as log_err:
        logger.error(f"Failed to save query audit log to PostgreSQL: {log_err}")
        db.rollback()

    return {
        "query": query,
        "routing_decision": routing_decision,
        "response": final_answer,
        "retrieved_documents": retrieved_docs,
        "tools_used": log_tools_payload,
        "latency_seconds": latency,
        "sql_query": sql_query_used,
        "sql_results": sql_results,
        "tool_results": tool_results
    }
