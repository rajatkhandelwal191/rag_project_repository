from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Environment
    env: str = "LOCAL"  # LOCAL or CLOUD
    
    app_name: str = "RAG Pipeline API"
    debug: bool = True
    upload_dir: str = "uploads"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Embedding (Gemini default: 768 dims)
    default_embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768  # Gemini embedding-001 dimension
    
    # LLM
    default_llm_model: str = "gpt-3.5-turbo"
    max_tokens: int = 2048
    temperature: float = 0.7
    
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_cluster_name: str = "rag_project_cluster"
    collection_name: str = "rag_documents"
    
    # Qdrant Cloud
    qdrant_cloud_url: str = ""
    qdrant_cloud_api_key: str = ""
    
    # Supabase
    supabase_url: str = ""
    
    # Supabase S3 Storage
    supabase_s3_bucket: str = "RAG_INGEST_DOCS"
    supabase_s3_access_key_id: str = ""
    supabase_s3_secret_access_key: str = ""
    supabase_s3_endpoint: str = ""
    
    # API Keys
    openai_api_key: str = ""
    cohere_api_key: str = ""
    google_api_key: str = ""
    
    # Groq
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    
    # Gemini
    gemini_chat_model: str = "gemini-2.0-flash"
    gemini_embed_model: str = "gemini-embedding-001"
    
    # Backend URL (for CORS and redirects)
    backend_url: str = "http://localhost:8080"  # Local default
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
