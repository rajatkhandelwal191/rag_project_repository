from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    """All values read from environment variables (.env file)"""
    
    # Environment
    env: str = os.environ.get("ENV", "LOCAL")
    
    # Application
    app_name: str = os.environ.get("APP_NAME", "RAG Pipeline API")
    debug: bool = os.environ.get("DEBUG", "true").lower() == "true"
    upload_dir: str = os.environ.get("UPLOAD_DIR", "uploads")
    max_file_size: int = int(os.environ.get("MAX_FILE_SIZE", "10485760"))
    
    # Chunking
    chunk_size: int = int(os.environ.get("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.environ.get("CHUNK_OVERLAP", "200"))
    
    # Embedding
    default_embedding_model: str = os.environ.get("DEFAULT_EMBEDDING_MODEL", "gemini-embedding-001")
    embedding_dim: int = int(os.environ.get("EMBEDDING_DIM", "768"))
    
    # LLM
    default_llm_model: str = os.environ.get("DEFAULT_LLM_MODEL", "gpt-3.5-turbo")
    max_tokens: int = int(os.environ.get("MAX_TOKENS", "2048"))
    temperature: float = float(os.environ.get("TEMPERATURE", "0.7"))
    
    # Qdrant
    qdrant_url: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str = os.environ.get("QDRANT_API_KEY", "")
    qdrant_cluster_name: str = os.environ.get("QDRANT_CLUSTER_NAME", "rag_project_cluster")
    collection_name: str = os.environ.get("COLLECTION_NAME", "rag_documents")
    
    # Qdrant Cloud
    qdrant_cloud_url: str = os.environ.get("QDRANT_CLOUD_URL", "")
    qdrant_cloud_api_key: str = os.environ.get("QDRANT_CLOUD_API_KEY", "")
    
    # Supabase
    supabase_url: str = os.environ.get("SUPABASE_URL", "")
    
    # Supabase S3 Storage
    supabase_s3_bucket: str = os.environ.get("SUPABASE_S3_BUCKET", "RAG_INGEST_DOCS")
    supabase_s3_access_key_id: str = os.environ.get("SUPABASE_S3_ACCESS_KEY_ID", "")
    supabase_s3_secret_access_key: str = os.environ.get("SUPABASE_S3_SECRET_ACCESS_KEY", "")
    supabase_s3_endpoint: str = os.environ.get("SUPABASE_S3_ENDPOINT", "")
    
    # API Keys
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    cohere_api_key: str = os.environ.get("COHERE_API_KEY", "")
    google_api_key: str = os.environ.get("GOOGLE_API_KEY", "")
    gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "")
    
    # Groq
    groq_api_key: str = os.environ.get("GROQ_API_KEY", "")
    groq_model: str = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
    
    # Gemini
    gemini_chat_model: str = os.environ.get("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
    gemini_embed_model: str = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    
    # Backend URL
    backend_url: str = os.environ.get("BACKEND_URL", "http://localhost:8080")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
