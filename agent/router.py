import json
import logging
import time
import random
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.orm import Session

from groq import Groq
import groq

from core.config import settings
from db.models import QueryLog
from rag.pipeline import retrieve_relevant_chunks
from tools import TOOL_REGISTRY, execute_tool

logger = logging.getLogger("helpdesk_assistant.agent")

# Pydantic schema for strict JSON output from Groq router
class RouterDecision(BaseModel):
    decision: str = Field(
        description="The routing path to choose. Must be exactly one of: RAG_RETRIEVAL, DB_QUERY, TOOL_EXECUTION, MULTI_SOURCE, DIRECT_LLM"
    )
    rationale: str = Field(
        description="A concise rationale explaining why this routing decision was made."
    )
    rag_query: Optional[str] = Field(
        None,
        description="Search query optimized for RAG semantic search. Required if decision is RAG_RETRIEVAL or MULTI_SOURCE."
    )
    db_sql_query: Optional[str] = Field(
        None,
        description="A read-only SELECT SQL statement. Required if decision is DB_QUERY or MULTI_SOURCE. "
        "Available tables: users (id, username, email, status, role, created_at), documents (id, content, metadata, created_at), queries (id, input_query, output_response, routing_decision, tools_used, latency, timestamp)."
    )
    tool_name: Optional[str] = Field(
        None,
        description="The name of the tool to execute. Must be exactly 'calculator' or 'file_search'. Required if decision is TOOL_EXECUTION."
    )
    tool_args_json: Optional[str] = Field(
        None,
        description="Arguments passed to the tool as a JSON string, e.g. '{\"expression\": \"500 * 133.4\"}' for calculator, or '{\"query\": \"vpn\"}' for file_search. Required if decision is TOOL_EXECUTION."
    )

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

def route_query(query: str) -> RouterDecision:
    """
    LLM-driven router that analyzes the incoming query and decides the optimal execution path.
    """
    client = get_groq_client()
    
    prompt = f"""
    You are an advanced IT Helpdesk Agent Router. Your job is to analyze the incoming user query and route it to the correct processing path.
    
    User Query: "{query}"
    
    Available routing paths:
    1. RAG_RETRIEVAL: Use for questions asking about internal IT policies, processes, configurations, deployment steps, network SSID/VPN setup, printer guides, or email server details.
    2. DB_QUERY: Use for structured numerical or statistic questions regarding users, documents, or logs (e.g., "How many active users are in the system?", "Find the role of user bob_dev", "Show the average query latency", "List all offline users").
       - SQL SCHEMA INFO:
         - Table `users`:
           - id: integer (primary key)
           - username: varchar(100) (e.g. 'alice_admin', 'bob_dev')
           - email: varchar(255)
           - status: varchar(50) (values: 'active', 'offline', 'suspended')
           - role: varchar(50) (values: 'admin', 'developer', 'user')
           - created_at: timestamptz
         - Table `documents`:
           - id: integer
           - content: text
           - metadata: jsonb
           - created_at: timestamptz
         - Table `queries`:
           - id: integer
           - input_query: text
           - output_response: text
           - routing_decision: varchar(50)
           - tools_used: jsonb
           - latency: double precision
           - timestamp: timestamptz
    3. TOOL_EXECUTION: Use when the user asks for a mathematical calculation, translation/conversion calculation, OR wants to scan/search local files.
       - Available Tools:
         - 'calculator': Evaluates arithmetic expressions (parameters: expression). E.g. "Convert 500 USD to NPR" or "What is 500 * 133.4".
         - 'file_search': Searches local workspace and fallback files for keywords (parameters: query). E.g. "Search file logs for printer" or "Find email configurations in files".
    4. MULTI_SOURCE: Use when a question requires BOTH querying database counts/statistics AND retrieving process documentation (e.g. "How many developers are there, and how do they deploy the backend?").
    5. DIRECT_LLM: Use for general greetings, general IT explanations that do not require company-specific knowledge, or generic tech questions (e.g., "Hi", "What is DNS?", "Explain the difference between TCP and UDP").

    Strict Constraint: For DB_QUERY and MULTI_SOURCE, you MUST generate a valid, standard PostgreSQL read-only SELECT query. Do not perform any modifications (INSERT, UPDATE, DELETE). Ensure you double-quote table columns if needed or write clean standard SQL.

    You must output a JSON object adhering exactly to the following structure:
    {{
      "decision": "RAG_RETRIEVAL" | "DB_QUERY" | "TOOL_EXECUTION" | "MULTI_SOURCE" | "DIRECT_LLM",
      "rationale": "A concise explanation",
      "rag_query": "Optimized RAG query, required if decision is RAG_RETRIEVAL or MULTI_SOURCE, else null",
      "db_sql_query": "PostgreSQL read-only SELECT query, required if decision is DB_QUERY or MULTI_SOURCE, else null",
      "tool_name": "'calculator' or 'file_search', required if decision is TOOL_EXECUTION, else null",
      "tool_args_json": "JSON string containing tool arguments, required if decision is TOOL_EXECUTION, else null"
    }}
    """
    
    logger.info(f"Routing user query: '{query}'")
    
    try:
        response = call_groq_with_retry(
            client=client,
            model=settings.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a precise router that outputs valid JSON only matching the requested schema. Do not output any markdown fencing, preamble, or comments. Just the raw JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0 # Strict deterministic routing
        )
        
        # Parse the JSON response
        res_text = response.choices[0].message.content.strip()
        data = json.loads(res_text)
        decision = RouterDecision(**data)
        logger.info(f"Routing decision made: {decision.decision} (Rationale: {decision.rationale})")
        return decision
    except Exception as e:
        logger.error(f"Failed to route query due to LLM error: {e}", exc_info=True)
        # Safe fallback path
        return RouterDecision(
            decision="DIRECT_LLM",
            rationale=f"Fallback to Direct LLM due to routing error: {str(e)}"
        )

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

def synthesize_answer(query: str, routing_decision: str, context_details: dict) -> str:
    """
    Final answer synthesis:
    Injects the retrieved data, SQL output, or tool results into the final LLM prompt 
    to output a beautiful, helpful response.
    """
    client = get_groq_client()
    
    prompt = f"""
    You are an expert IT Helpdesk and Knowledge Assistant. Your goal is to provide a clear, professional, and accurate response to the user's inquiry based on the context provided.
    
    User Inquiry: "{query}"
    Routing Path Selected: {routing_decision}
    
    Context Information:
    {json.dumps(context_details, indent=2, default=str)}
    
    Instructions:
    1. Base your answer strictly on the context details. If the context does not contain enough information, explain that honestly.
    2. Maintain a highly professional and friendly IT specialist tone.
    3. Do not mention system-level routing terms like "RAG", "SQL", or "Tool execution" in your response unless directly relevant to explaining a command. Keep the response seamless for the end user.
    """
    
    try:
        response = call_groq_with_retry(
            client=client,
            model=settings.LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a professional IT Helpdesk Assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error during final response synthesis: {e}")
        return f"I processed your query, but encountered an error generating the final summary. Raw Context: {str(context_details)}"

def process_pipeline(db: Session, query: str) -> dict:
    """
    Main Orchestrator representing the complete IT Helpdesk core pipeline:
    1. Route the query using the LLM router.
    2. Execute the designated pathway (RAG, DB, Tool, Multi-source, Direct).
    3. Synthesize the final answer.
    4. Log latency, tools used, and outcomes to the 'queries' table in PostgreSQL.
    """
    start_time = time.time()
    
    # Initialize response tracking
    routing_decision = "DIRECT_LLM"
    retrieved_docs = []
    tools_used = []
    sql_query_used = None
    sql_results = None
    tool_results = None
    context_details = {}
    
    try:
        # Step 1: Decision routing
        decision: RouterDecision = route_query(query)
        routing_decision = decision.decision
        
        # Step 2: Path Execution
        if routing_decision == "RAG_RETRIEVAL":
            rag_q = decision.rag_query or query
            retrieved = retrieve_relevant_chunks(db, rag_q, top_k=3)
            retrieved_docs = retrieved
            context_details["retrieved_documents"] = retrieved
            
        elif routing_decision == "DB_QUERY":
            sql_query_used = decision.db_sql_query
            if sql_query_used:
                try:
                    sql_results = execute_safe_select_query(db, sql_query_used)
                    context_details["database_query"] = {
                        "sql": sql_query_used,
                        "results": sql_results
                    }
                except Exception as db_err:
                    logger.error(f"SQL execution failed: {db_err}")
                    context_details["database_query_error"] = str(db_err)
            else:
                context_details["error"] = "DB_QUERY selected but no SQL query generated by router."
                
        elif routing_decision == "TOOL_EXECUTION":
            tool_name = decision.tool_name
            tool_args_str = decision.tool_args_json or "{}"
            
            # Parse the JSON string arguments
            tool_args = {}
            if tool_args_str:
                try:
                    tool_args = json.loads(tool_args_str)
                except Exception as parse_err:
                    logger.error(f"Failed to parse tool arguments JSON: {parse_err}")
                    tool_args = {"raw_args": tool_args_str}
                    
            if tool_name:
                tools_used.append({"tool": tool_name, "args": tool_args})
                tool_res = execute_tool(tool_name, tool_args)
                tool_results = tool_res
                context_details["tool_execution"] = tool_res
            else:
                context_details["error"] = "TOOL_EXECUTION selected but no tool name was provided by router."
                
        elif routing_decision == "MULTI_SOURCE":
            # Co-ordinate DB Query and RAG Search
            rag_q = decision.rag_query or query
            sql_q = decision.db_sql_query
            
            # Run RAG
            retrieved = retrieve_relevant_chunks(db, rag_q, top_k=3)
            retrieved_docs = retrieved
            context_details["retrieved_documents"] = retrieved
            
            # Run SQL
            sql_query_used = sql_q
            if sql_q:
                try:
                    sql_results = execute_safe_select_query(db, sql_q)
                    context_details["database_query"] = {
                        "sql": sql_q,
                        "results": sql_results
                    }
                except Exception as db_err:
                    logger.error(f"SQL execution failed in Multi-source: {db_err}")
                    context_details["database_query_error"] = str(db_err)
                    
        elif routing_decision == "DIRECT_LLM":
            context_details["general_knowledge"] = "Direct model completion utilized."
            
        else:
            logger.warning(f"Unknown routing decision '{routing_decision}'. Defaulting to Direct LLM.")
            routing_decision = "DIRECT_LLM"
            
        # Step 3: Synthesis
        final_answer = synthesize_answer(query, routing_decision, context_details)
        
    except Exception as pipeline_err:
        logger.error(f"Error in main orchestration pipeline: {pipeline_err}", exc_info=True)
        routing_decision = "ERROR_FALLBACK"
        final_answer = f"I'm sorry, an internal server error occurred while processing your request: {str(pipeline_err)}"
        context_details["pipeline_error"] = str(pipeline_err)
        
    # Step 4: Metrics and database logging
    latency = time.time() - start_time
    
    # Store standard log details
    log_tools_payload = tools_used
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
        logger.info(f"Request pipeline completed in {latency:.4f}s. Decision logged to queries table.")
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
