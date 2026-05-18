from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, JSON
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from db.session import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False, default="active") # active, suspended, offline
    role = Column(String(50), nullable=False, default="user") # admin, developer, user
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    # Map the Python property doc_metadata to the actual database column named "metadata"
    doc_metadata = Column("metadata", JSONB, nullable=True, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship to chunks
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(ARRAY(Float), nullable=False) # Represents double precision[] in Postgres
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Parent relationship
    document = relationship("Document", back_populates="chunks")

class QueryLog(Base):
    __tablename__ = "queries"

    id = Column(Integer, primary_key=True, index=True)
    input_query = Column(Text, nullable=False)
    output_response = Column(Text, nullable=False)
    routing_decision = Column(String(50), nullable=False)
    tools_used = Column(JSONB, nullable=True, default=[])
    latency = Column(Float, nullable=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
