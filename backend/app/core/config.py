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

    # Database pool tuning (async engine)
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 40
    DB_POOL_TIMEOUT_SECONDS: int = 10
    DB_POOL_RECYCLE_SECONDS: int = 300
    # Backpressure: limit concurrent DB sessions per API instance
    DB_SESSION_CONCURRENCY_LIMIT: Optional[int] = None  # default: pool_size + max_overflow
    DB_SESSION_ACQUIRE_TIMEOUT_SECONDS: int = 2

    # Celery task DB pool tuning (fresh engine per task invocation)
    CELERY_DB_USE_NULLPOOL: bool = True
    CELERY_DB_POOL_SIZE: int = 2
    CELERY_DB_MAX_OVERFLOW: int = 5
    CELERY_DB_POOL_TIMEOUT_SECONDS: int = 10
    
    # LLM Configuration
    # Provider can be 'ollama' (local) or 'deepseek' (external OpenAI-compatible API)
    LLM_PROVIDER: str = "ollama"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    DEFAULT_MODEL: str = "llama3.2:1b"  # Smallest model for Mac compatibility (~1GB, best for 8GB Mac)
    # Alternative models: "llama3.2:3b" (~2GB), "phi3:mini" (~2GB), "gemma:2b" (~1.5GB)
    # For more powerful systems: "llama2" (~4GB), "mistral:7b" (~4GB), "llama3.2" (~4GB)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    # Alternative embedding models: "all-mpnet-base-v2" (better quality), "multilingual-mpnet-base-v2" (multilingual)
    EMBEDDING_MODEL_OPTIONS: List[str] = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "multilingual-mpnet-base-v2"]

    # DeepSeek (external) — optional
    DEEPSEEK_API_BASE: str = "https://api.deepseek.com/v1"
    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_TIMEOUT_SECONDS: int = 120
    DEEPSEEK_MAX_RESPONSE_TOKENS: int = 2000
    
    # ChromaDB
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "knowledge_base"

    # Vector store provider
    # Supported: "chroma" (embedded), "qdrant" (service)
    VECTOR_STORE_PROVIDER: str = "qdrant"

    # Qdrant (when VECTOR_STORE_PROVIDER="qdrant")
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "knowledge_base"
    
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
    
    # Transcription
    WHISPER_MODEL_SIZE: str = "base"  # Options: tiny, base, small, medium, large
    WHISPER_DEVICE: str = "auto"  # Options: cpu, cuda, auto
    TRANSCRIPTION_LANGUAGE: str = "ru"  # Default language for transcription
    TRANSCRIPTION_SPEAKER_DIARIZATION: bool = True  # Enable speaker labels in transcripts
    TRANSCRIPTION_DIARIZATION_MODEL: str = "pyannote/speaker-diarization-3.1"  # Pyannote diarization model
    HUGGINGFACE_TOKEN: Optional[str] = None  # HF token required to download some pyannote models
    
    # File Upload Limits
    MAX_FILE_SIZE: int = 500 * 1024 * 1024  # 500MB default (videos can be large)
    MAX_VIDEO_SIZE: int = 2000 * 1024 * 1024  # 2GB for videos specifically
    
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

    # Backpressure (global concurrency caps)
    LLM_MAX_CONCURRENCY: int = 4
    
    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_SEARCH_RESULTS: int = 5
    
    # Summarization
    SUMMARIZATION_HEAVY_THRESHOLD_CHARS: int = 30000  # Above this, treat as heavy and prefer external provider
    SUMMARIZATION_CHUNK_SIZE_CHARS: int = 12000       # Per-chunk size for document summarization
    SUMMARIZATION_CHUNK_OVERLAP_CHARS: int = 800      # Overlap between chunks to preserve continuity
    KNOWLEDGE_GRAPH_ENABLED: bool = True
    SUMMARIZATION_ENABLED: bool = True
    AUTO_SUMMARIZE_ON_PROCESS: bool = False

    # Knowledge Graph Extraction
    KG_LLM_EXTRACTION_ENABLED: bool = True  # Use LLM for better entity/relationship extraction
    KG_EXTRACTION_MODEL: Optional[str] = None  # Model for KG extraction (None = use default)
    KG_EXTRACTION_BATCH_SIZE: int = 3  # Chunks to batch per LLM call
    KG_EXTRACTION_MAX_TEXT_LENGTH: int = 3000  # Max chars per extraction call

    # Unsafe code execution (disabled by default)
    # Enables running generated demo scripts for "paper algorithm" projects.
    # WARNING: This executes untrusted code. Only enable in an isolated sandbox environment.
    ENABLE_UNSAFE_CODE_EXECUTION: bool = False
    UNSAFE_CODE_EXEC_TIMEOUT_SECONDS: int = 10
    UNSAFE_CODE_EXEC_MAX_STDOUT_CHARS: int = 20000
    UNSAFE_CODE_EXEC_MAX_STDERR_CHARS: int = 20000
    UNSAFE_CODE_EXEC_MAX_MEMORY_MB: int = 512
    # Execution backend: 'subprocess' (best-effort local) or 'docker' (recommended).
    UNSAFE_CODE_EXEC_BACKEND: str = "subprocess"
    # Docker backend settings (only used when UNSAFE_CODE_EXEC_BACKEND='docker')
    UNSAFE_CODE_EXEC_DOCKER_IMAGE: str = "python:3.11-slim"
    UNSAFE_CODE_EXEC_DOCKER_CPUS: float = 1.0
    UNSAFE_CODE_EXEC_DOCKER_PIDS_LIMIT: int = 128

    # RAG Knowledge Graph Integration
    RAG_KG_CONTEXT_ENABLED: bool = True  # Inject KG context into chat responses
    RAG_KG_MAX_ENTITIES: int = 10  # Max entities to include in context
    RAG_KG_MAX_RELATIONSHIPS: int = 15  # Max relationships to include
    
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
    
    # Kroki (local diagram rendering)
    KROKI_URL: str = "http://localhost:8001"  # Local Kroki Docker container
    KROKI_FALLBACK_URL: str = "https://kroki.io"  # External fallback
    KROKI_USE_FALLBACK: bool = True  # Fall back to external if local fails

    # LaTeX Studio
    # Security note: compiling arbitrary TeX on the server can be dangerous (file reads, resource usage).
    # Keep disabled by default; enable only in trusted environments.
    LATEX_COMPILER_ENABLED: bool = False
    LATEX_COMPILER_ADMIN_ONLY: bool = True
    LATEX_COMPILER_TIMEOUT_SECONDS: int = 20
    LATEX_COMPILER_MAX_SOURCE_CHARS: int = 200000
    LATEX_PROJECT_MAX_FILE_SIZE: int = 25 * 1024 * 1024  # 25MB per asset
    LATEX_COMPILER_RUN_BIBTEX: bool = True
    LATEX_COMPILER_USE_CELERY: bool = False
    LATEX_COMPILER_CELERY_QUEUE: str = "latex"
    LATEX_COMPILER_JOB_QUEUED_STALE_SECONDS: int = 10 * 60
    LATEX_COMPILER_JOB_RUNNING_STALE_SECONDS: int = 5 * 60

    # MinIO Object Storage
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET_NAME: str = "documents"
    MINIO_USE_SSL: bool = False
    MINIO_PRESIGNED_URL_EXPIRY: int = 3600  # 1 hour in seconds
    MINIO_PROXY_BASE_URL: Optional[str] = None  # Base URL for nginx proxy (e.g., "http://localhost:3000/minio")

    # Secrets vault
    SECRETS_ENCRYPTION_KEY: Optional[str] = None  # Optional Fernet key (urlsafe base64, 32 bytes)

    # Agent governance
    AGENT_REQUIRE_TOOL_APPROVAL: bool = True
    AGENT_DANGEROUS_TOOLS: List[str] = [
        "delete_document",
        "batch_delete_documents",
        "delete_entity",
        "merge_entities",
        "run_custom_tool",
    ]
    # If enabled, autonomous agent jobs may directly apply code patches to the KB (writes).
    # Strongly recommended to keep disabled and use PatchPR review/merge instead.
    AGENT_KB_PATCH_APPLY_ENABLED: bool = False

    # Custom tools
    # Docker-based tools require access to a Docker daemon (often via host docker socket).
    # Keep disabled by default for safety.
    CUSTOM_TOOL_DOCKER_ENABLED: bool = False

    # AI Hub Training Configuration
    TRAINING_ENABLED: bool = True
    TRAINING_MAX_CONCURRENT_JOBS: int = 2
    TRAINING_DEFAULT_BACKEND: str = "local"  # local, modal, runpod
    TRAINING_LOCAL_DEVICE: str = "auto"  # cuda, cpu, mps, auto
    TRAINING_LOCAL_MAX_GPU_MEMORY_GB: float = 24.0
    TRAINING_CHECKPOINT_INTERVAL_STEPS: int = 100
    TRAINING_OUTPUT_DIR: str = "./data/training_outputs"
    AI_HUB_EVAL_TEMPLATES_DIR: Optional[str] = None  # Optional override for eval template "plugins"
    AI_HUB_EVAL_ENABLED_TEMPLATE_IDS: Optional[str] = None  # Comma-separated template IDs allowed for non-admin users
    AI_HUB_DATASET_PRESETS_DIR: Optional[str] = None  # Optional override for dataset preset "plugins"
    AI_HUB_DATASET_ENABLED_PRESET_IDS: Optional[str] = None  # Comma-separated preset IDs allowed for non-admin users

    # Cloud Training (future - optional)
    MODAL_API_KEY: Optional[str] = None
    RUNPOD_API_KEY: Optional[str] = None

    # Dataset Limits
    DATASET_MAX_SIZE_MB: int = 500
    DATASET_MAX_SAMPLES: int = 100000
    DATASET_MAX_TOKEN_COUNT: int = 50000000  # 50M tokens
    
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

    # Validators
    @field_validator("OLLAMA_BASE_URL", mode="before")
    @classmethod
    def default_ollama_base_url_for_docker(cls, v):
        # Prefer explicit env var (including docker-compose `environment:` entries).
        env_val = os.getenv("OLLAMA_BASE_URL")
        if env_val:
            return env_val

        # If value is localhost but we're running inside Docker, localhost points at the container.
        # Default to the docker-compose service name `ollama`.
        try:
            in_docker = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")
        except Exception:
            in_docker = False

        if in_docker and isinstance(v, str) and ("localhost:11434" in v or "127.0.0.1:11434" in v):
            return "http://ollama:11434"

        return v

    @field_validator("CELERY_BROKER_URL", mode="before")
    @classmethod
    def default_celery_broker_from_redis(cls, v):
        # If not explicitly set, default to REDIS_URL to avoid localhost in containers
        env_val = os.getenv("CELERY_BROKER_URL")
        if env_val:
            return env_val
        return os.getenv("REDIS_URL", v or "redis://localhost:6379/0")

    @field_validator("CELERY_RESULT_BACKEND", mode="before")
    @classmethod
    def default_celery_backend_from_redis(cls, v):
        env_val = os.getenv("CELERY_RESULT_BACKEND")
        if env_val:
            return env_val
        return os.getenv("REDIS_URL", v or "redis://localhost:6379/0")


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
