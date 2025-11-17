"""
Application configuration settings.
"""

import os
from typing import Optional, List
from pydantic import field_validator
from pydantic_settings import BaseSettings
from loguru import logger


class Settings(BaseSettings):
    """Application settings."""
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/knowledge_db"
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # LLM Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    DEFAULT_MODEL: str = "llama3.2:1b"  # Smallest model for Mac compatibility (~1GB, best for 8GB Mac)
    # Alternative models: "llama3.2:3b" (~2GB), "phi3:mini" (~2GB), "gemma:2b" (~1.5GB)
    # For more powerful systems: "llama2" (~4GB), "mistral:7b" (~4GB), "llama3.2" (~4GB)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    # Alternative embedding models: "all-mpnet-base-v2" (better quality), "multilingual-mpnet-base-v2" (multilingual)
    EMBEDDING_MODEL_OPTIONS: List[str] = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multilingual-mpnet-base-v2"]
    
    # ChromaDB
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "knowledge_base"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Application
    DEBUG: bool = True
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # Data Sources
    GITLAB_URL: Optional[str] = None
    GITLAB_TOKEN: Optional[str] = None
    CONFLUENCE_URL: Optional[str] = None
    CONFLUENCE_USER: Optional[str] = None
    CONFLUENCE_API_TOKEN: Optional[str] = None
    
    # Celery
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./data/logs/app.log"
    
    # Chat Configuration
    MAX_CONTEXT_LENGTH: int = 4000
    MAX_RESPONSE_LENGTH: int = 1000
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    
    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_SEARCH_RESULTS: int = 5
    
    # RAG Configuration
    RAG_HYBRID_SEARCH_ENABLED: bool = True
    RAG_HYBRID_SEARCH_ALPHA: float = 0.7  # Semantic weight (0.0 = keyword only, 1.0 = semantic only)
    RAG_RERANKING_ENABLED: bool = True
    RAG_RERANKING_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RAG_RERANKING_TOP_K: int = 5
    RAG_MAX_CONTEXT_TOKENS: int = 4000
    RAG_MIN_RELEVANCE_SCORE: float = 0.3
    RAG_CHUNKING_STRATEGY: str = "semantic"  # semantic or fixed
    RAG_QUERY_EXPANSION_ENABLED: bool = True
    RAG_MMR_ENABLED: bool = True
    RAG_MMR_LAMBDA: float = 0.5  # Balance between relevance (1.0) and diversity (0.0)
    RAG_DEDUPLICATION_ENABLED: bool = True
    RAG_DEDUPLICATION_THRESHOLD: float = 0.95  # Similarity threshold for considering duplicates
    
    # MinIO Object Storage
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET_NAME: str = "documents"
    MINIO_USE_SSL: bool = False
    MINIO_PRESIGNED_URL_EXPIRY: int = 3600  # 1 hour in seconds
    MINIO_PROXY_BASE_URL: Optional[str] = None  # Base URL for nginx proxy (e.g., "http://localhost:3000/minio")
    
    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def validate_database_url(cls, v):
        if not v or v == "":
            raise ValueError("DATABASE_URL must be set")
        return v
    
    @field_validator("SECRET_KEY", mode="before")
    @classmethod
    def validate_secret_key(cls, v):
        if v == "your-secret-key-change-in-production":
            logger.warning("Using default SECRET_KEY. Change this in production!")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Configure logging
logger.remove()  # Remove default handler
logger.add(
    settings.LOG_FILE,
    level=settings.LOG_LEVEL,
    rotation="10 MB",
    retention="30 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
)
logger.add(
    lambda msg: print(msg, end=""),
    level=settings.LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | {message}"
)


