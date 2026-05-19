import json
from sqlalchemy import text
from db.session import SessionLocal

def generate_sql_dump():
    db = SessionLocal()
    try:
        sql_lines = []
        
        # 1. Custom functions
        sql_lines.append("-- ========================================================")
        sql_lines.append("-- 1. CUSTOM VECTOR MATH PL/pgSQL FUNCTIONS")
        sql_lines.append("-- ========================================================")
        
        sql_lines.append("""
CREATE OR REPLACE FUNCTION dot_product(a double precision[], b double precision[])
RETURNS double precision AS $$
DECLARE
  s double precision := 0;
  i integer;
  len_a integer;
  len_b integer;
  BEGIN
  len_a := cardinality(a);
  len_b := cardinality(b);
  IF len_a IS NULL OR len_b IS NULL OR len_a <> len_b THEN
    RETURN 0;
  END IF;
  FOR i IN 1..len_a LOOP
    s := s + a[i] * b[i];
  END LOOP;
  RETURN s;
END;
$$ LANGUAGE plpgsql IMMUTABLE;
""")

        sql_lines.append("""
CREATE OR REPLACE FUNCTION magnitude(a double precision[])
RETURNS double precision AS $$
DECLARE
  s double precision := 0;
  i integer;
  len integer;
BEGIN
  len := cardinality(a);
  IF len IS NULL THEN
    RETURN 0;
  END IF;
  FOR i IN 1..len LOOP
    s := s + a[i] * a[i];
  END LOOP;
  RETURN sqrt(s);
END;
$$ LANGUAGE plpgsql IMMUTABLE;
""")

        sql_lines.append("""
CREATE OR REPLACE FUNCTION cosine_similarity(a double precision[], b double precision[])
RETURNS double precision AS $$
DECLARE
  dp double precision;
  mag_a double precision;
  mag_b double precision;
BEGIN
  dp := dot_product(a, b);
  mag_a := magnitude(a);
  mag_b := magnitude(b);
  IF mag_a = 0 OR mag_b = 0 THEN
    RETURN 0;
  END IF;
  RETURN dp / (mag_a * mag_b);
END;
$$ LANGUAGE plpgsql IMMUTABLE;
""")

        # 2. Table Creation DDL
        sql_lines.append("\n-- ========================================================")
        sql_lines.append("-- 2. TABLE CREATION DDL (SCHEMAS)")
        sql_lines.append("-- ========================================================")
        
        sql_lines.append("""
DROP TABLE IF EXISTS queries CASCADE;
DROP TABLE IF EXISTS document_chunks CASCADE;
DROP TABLE IF EXISTS documents CASCADE;
DROP TABLE IF EXISTS users CASCADE;

CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    role VARCHAR(50) NOT NULL DEFAULT 'user',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE document_chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding DOUBLE PRECISION[] NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE queries (
    id SERIAL PRIMARY KEY,
    input_query TEXT NOT NULL,
    output_response TEXT NOT NULL,
    routing_decision VARCHAR(50) NOT NULL,
    tools_used JSONB,
    latency DOUBLE PRECISION NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
""")

        # 3. Insert Users
        sql_lines.append("\n-- ========================================================")
        sql_lines.append("-- 3. SEED DATA FOR USERS TABLE")
        sql_lines.append("-- ========================================================")
        
        users = db.execute(text("SELECT username, email, status, role FROM users")).fetchall()
        for user in users:
            sql_lines.append(
                f"INSERT INTO users (username, email, status, role) VALUES "
                f"('{user.username}', '{user.email}', '{user.status}', '{user.role}');"
            )

        # 4. Insert Documents and Document Chunks
        sql_lines.append("\n-- ========================================================")
        sql_lines.append("-- 4. SEED DATA FOR DOCUMENTS & DOCUMENT_CHUNKS (WITH EMBEDDINGS)")
        sql_lines.append("-- ========================================================")
        
        docs = db.execute(text("SELECT id, content, metadata FROM documents ORDER BY id")).fetchall()
        for doc in docs:
            # Escape single quotes in text content
            content_escaped = doc.content.replace("'", "''")
            metadata_str = json.dumps(doc.metadata)
            sql_lines.append(f"\n-- Inserting Document {doc.id}")
            sql_lines.append(
                f"INSERT INTO documents (id, content, metadata) VALUES "
                f"({doc.id}, '{content_escaped}', '{metadata_str}'::jsonb);"
            )
            
            # Fetch chunks for this document
            chunks = db.execute(text(f"SELECT chunk_index, content, embedding FROM document_chunks WHERE document_id = {doc.id} ORDER BY chunk_index")).fetchall()
            for chunk in chunks:
                chunk_content_escaped = chunk.content.replace("'", "''")
                # Format embedding float array for postgres: ARRAY[0.1, 0.2, ...]
                emb_str = ", ".join(map(str, chunk.embedding))
                sql_lines.append(
                    f"INSERT INTO document_chunks (document_id, chunk_index, content, embedding) VALUES "
                    f"({doc.id}, {chunk.chunk_index}, '{chunk_content_escaped}', ARRAY[{emb_str}]::double precision[]);"
                )
                
        # Fix sequence values for SERIAL columns so new records don't hit duplicate key errors
        sql_lines.append("\n-- Reset primary key sequences")
        sql_lines.append("SELECT setval('users_id_seq', COALESCE((SELECT MAX(id) FROM users), 1));")
        sql_lines.append("SELECT setval('documents_id_seq', COALESCE((SELECT MAX(id) FROM documents), 1));")
        sql_lines.append("SELECT setval('document_chunks_id_seq', COALESCE((SELECT MAX(id) FROM document_chunks), 1));")
        
        # Write to file
        with open("init_db.sql", "w", encoding="utf-8") as f:
            f.write("\n".join(sql_lines))
            
        print("SQL Dump successfully generated in init_db.sql")
        
    finally:
        db.close()

if __name__ == "__main__":
    generate_sql_dump()
