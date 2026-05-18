import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from the .env file in the current working directory
load_dotenv()

class Settings(BaseSettings):
    # App Settings
    PROJECT_NAME: str = "Internal IT Helpdesk & Knowledge Assistant"
    VERSION: str = "1.0.0"
    
    # Core LLM settings
    LLM_MODEL_NAME: str = "llama-3.3-70b-versatile"
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # RAG Settings (Local Lightweight Models suitable for 4GB VRAM)
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    
    # Database Configuration (PostgreSQL)
    # Prefer DATABASE_URL from .env if present
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    
    # Individual fallbacks
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "postgres")
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: str = os.getenv("DB_PORT", "5432")
    DB_NAME: str = os.getenv("DB_NAME", "helpdesk_assistant")
    
    def get_database_url(self) -> str:
        """Return DATABASE_URL if defined, otherwise compile from components."""
        if self.DATABASE_URL:
            # SQLAlchemy expects 'postgresql://' instead of 'postgres://' if that is in .env
            if self.DATABASE_URL.startswith("postgres://"):
                return self.DATABASE_URL.replace("postgres://", "postgresql://", 1)
            return self.DATABASE_URL
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

settings = Settings()
