import os
import logging
from sqlalchemy import text
from sqlalchemy.orm import Session
from core.config import settings
from db.models import Document, DocumentChunk
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("helpdesk_assistant.rag")

# Singleton holder for the embedding model
_embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    """Lazy load the local SentenceTransformer embedding model."""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Activating Hugging Face Offline Mode to bypass network checks...")
        # Force offline mode so that it loads strictly from local disk cache, avoiding network timeouts
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        logger.info(f"Loading local embedding model: {settings.EMBEDDING_MODEL_NAME}...")
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded successfully from local disk cache.")
    return _embedding_model

def chunk_text(text_content: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    """
    Manually chunk text into overlapping segments, respecting word and sentence boundaries
    to maintain semantic integrity without external agent frameworks.
    """
    if not text_content:
        return []
        
    chunks = []
    start = 0
    text_len = len(text_content)
    
    while start < text_len:
        # Initial guess for the end of the chunk
        end = min(start + chunk_size, text_len)
        
        # If we are not at the absolute end, try to find a natural boundary backwards
        if end < text_len:
            boundary = -1
            # Search up to 80 characters backwards for a sentence/word boundary
            for i in range(end, max(start, end - 80), -1):
                if text_content[i] in ['\n', '.', '?', '!', ';', ' ']:
                    boundary = i
                    break
            if boundary != -1:
                end = boundary + 1 # Include the boundary character
                
        chunk = text_content[start:end].strip()
        if chunk:
            chunks.append(chunk)
            
        # If we have reached the end of the text, break to avoid infinite loops
        if end >= text_len:
            break
            
        # Move the window forward, respecting the overlap
        start = end - chunk_overlap
        # Safety check: if start is not progressing or overlap is invalid, advance start to end
        if start <= 0 or (end - start) <= 0:
            start = end
            
    return chunks

def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate vector embeddings for a list of texts using the local model."""
    if not texts:
        return []
    
    model = get_embedding_model()
    # encode outputs numpy array, convert to standard Python float list for PostgreSQL array compatibility
    embeddings = model.encode(texts)
    return embeddings.tolist()

def ingest_document(db: Session, content: str, metadata: dict = None) -> Document:
    """
    RAG Ingestion:
    1. Save the raw parent document.
    2. Split the document into overlapping chunks.
    3. Generate vector embeddings locally for all chunks.
    4. Store chunks and their high-dimensional vectors in PostgreSQL.
    """
    if metadata is None:
        metadata = {}
        
    logger.info(f"Ingesting new document (length: {len(content)} chars)...")
    
    # 1. Create and add parent document
    doc = Document(content=content, doc_metadata=metadata)
    db.add(doc)
    db.flush() # Populate the ID
    
    # 2. Chunk text
    chunks = chunk_text(content)
    logger.info(f"Split document into {len(chunks)} chunks.")
    
    if chunks:
        # 3. Batch generate embeddings for speed
        embeddings = generate_embeddings(chunks)
        
        # 4. Save chunks
        for idx, (chunk_text_data, emb) in enumerate(zip(chunks, embeddings)):
            db_chunk = DocumentChunk(
                document_id=doc.id,
                chunk_index=idx,
                content=chunk_text_data,
                embedding=emb
            )
            db.add(db_chunk)
            
    db.commit()
    logger.info(f"Successfully ingested and indexed Document ID {doc.id}.")
    return doc

def retrieve_relevant_chunks(db: Session, query: str, top_k: int = 3) -> list[dict]:
    """
    RAG Retrieval:
    1. Embed user query locally.
    2. Run custom cosine similarity function in PostgreSQL database.
    3. Retrieve the top-K matching chunks along with parent metadata.
    """
    logger.info(f"Retrieving top {top_k} relevant chunks for query: '{query}'")
    
    # Embed query
    query_emb = generate_embeddings([query])[0]
    
    # PostgreSQL raw query calling custom cosine_similarity function
    sql = text("""
        SELECT dc.id, dc.document_id, dc.content, 
               cosine_similarity(dc.embedding, :query_emb) AS similarity,
               d.metadata AS doc_metadata
        FROM document_chunks dc
        JOIN documents d ON dc.document_id = d.id
        ORDER BY similarity DESC
        LIMIT :top_k;
    """)
    
    results = db.execute(sql, {"query_emb": query_emb, "top_k": top_k}).fetchall()
    
    retrieved_chunks = []
    for row in results:
        retrieved_chunks.append({
            "id": row[0],
            "document_id": row[1],
            "content": row[2],
            "similarity": float(row[3]) if row[3] is not None else 0.0,
            "metadata": row[4]
        })
        
    logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks. Top similarity: {retrieved_chunks[0]['similarity']:.4f}" if retrieved_chunks else "No chunks found.")
    return retrieved_chunks
