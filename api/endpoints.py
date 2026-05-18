from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from db.session import get_db
from rag.pipeline import ingest_document
from agent.router import process_pipeline

router = APIRouter()

# ---------------------------------------------------------------------------
#  PYDANTIC REQUEST / RESPONSE SCHEMAS
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    content: str = Field(..., description="The raw textual content of the document to ingest.")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata key-value pairs (e.g. title, category).")

class IngestResponse(BaseModel):
    status: str
    document_id: int
    chunks_count: int

class QueryRequest(BaseModel):
    query: str = Field(..., description="The IT helpdesk query to submit to the assistant.")

class QueryResponse(BaseModel):
    query: str
    routing_decision: str
    response: str
    latency_seconds: float
    tools_used: List[Any]
    retrieved_documents: List[Any]

class EvalRequest(BaseModel):
    queries: List[str] = Field(..., description="A list of test queries to run through the evaluation suite (typically 10 queries).")

class EvalItemResult(BaseModel):
    query: str
    routing_decision: str
    response: str
    latency_seconds: float
    tools_used: List[Any]
    retrieved_documents: List[Any]

class EvalResponse(BaseModel):
    total_queries: int
    average_latency_seconds: float
    results: List[EvalItemResult]

# ---------------------------------------------------------------------------
#  ROUTER ENDPOINTS
# ---------------------------------------------------------------------------

@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
def api_ingest(payload: IngestRequest, db: Session = Depends(get_db)):
    """
    Ingest a document into the RAG Pipeline:
    Splits the text into chunks, generates vector embeddings locally,
    and indexes them in the PostgreSQL database for similarity search.
    """
    if not payload.content.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document content cannot be empty."
        )
    try:
        doc = ingest_document(db, payload.content, payload.metadata)
        return IngestResponse(
            status="success",
            document_id=doc.id,
            chunks_count=len(doc.chunks)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}"
        )

@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
def api_query(payload: QueryRequest, db: Session = Depends(get_db)):
    """
    Submit a query to the IT Helpdesk and Knowledge Assistant:
    Runs the LLM router, executes the optimal path (RAG, SQL DB search, Tool invocation, etc.),
    synthesizes a final response, and registers request metrics in PostgreSQL.
    """
    if not payload.query.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query string cannot be empty."
        )
    try:
        pipeline_result = process_pipeline(db, payload.query)
        return QueryResponse(
            query=pipeline_result["query"],
            routing_decision=pipeline_result["routing_decision"],
            response=pipeline_result["response"],
            latency_seconds=pipeline_result["latency_seconds"],
            tools_used=pipeline_result["tools_used"],
            retrieved_documents=pipeline_result["retrieved_documents"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while executing the query pipeline: {str(e)}"
        )

@router.post("/eval", response_model=EvalResponse, status_code=status.HTTP_200_OK)
def api_eval(payload: EvalRequest, db: Session = Depends(get_db)):
    """
    Run a batch dataset through the IT Helpdesk evaluation suite.
    Processes a list of queries sequentially, tracking latencies, decisions, tools, and outputs.
    """
    if not payload.queries:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Queries list cannot be empty."
        )
        
    results = []
    total_latency = 0.0
    
    for q in payload.queries:
        try:
            pipeline_result = process_pipeline(db, q)
            latency = pipeline_result["latency_seconds"]
            total_latency += latency
            
            results.append(EvalItemResult(
                query=pipeline_result["query"],
                routing_decision=pipeline_result["routing_decision"],
                response=pipeline_result["response"],
                latency_seconds=latency,
                tools_used=pipeline_result["tools_used"],
                retrieved_documents=pipeline_result["retrieved_documents"]
            ))
        except Exception as e:
            # Continue evaluation even if one query fails, logging the error in response
            results.append(EvalItemResult(
                query=q,
                routing_decision="ERROR",
                response=f"Evaluation execution failure: {str(e)}",
                latency_seconds=0.0,
                tools_used=[],
                retrieved_documents=[]
            ))
            
    avg_latency = total_latency / len(payload.queries) if payload.queries else 0.0
    
    return EvalResponse(
        total_queries=len(payload.queries),
        average_latency_seconds=avg_latency,
        results=results
    )
