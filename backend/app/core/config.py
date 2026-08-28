"""
Application configuration settings.
"""

import os
from typing import List, Optional

from loguru import logger
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    DB_SESSION_CONCURRENCY_LIMIT: Optional[
        int
    ] = None  # default: pool_size + max_overflow
    DB_SESSION_ACQUIRE_TIMEOUT_SECONDS: int = 2

    # Celery task DB pool tuning (fresh engine per task invocation)
    CELERY_DB_USE_NULLPOOL: bool = True
    CELERY_DB_POOL_SIZE: int = 2
    CELERY_DB_MAX_OVERFLOW: int = 5
    CELERY_DB_POOL_TIMEOUT_SECONDS: int = 10

    # LLM Configuration
    # Provider: 'ollama' (local), 'deepseek', 'openai', 'anthropic',
    # 'qwen' (DashScope), or 'kimi' (Moonshot AI)
    LLM_PROVIDER: str = "ollama"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    DEFAULT_MODEL: str = (
        "llama3.2:1b"  # Smallest model for Mac compatibility (~1GB, best for 8GB Mac)
    )
    # Alternative models: "llama3.2:3b" (~2GB), "phi3:mini" (~2GB), "gemma:2b" (~1.5GB)
    # For more powerful systems: "llama2" (~4GB), "mistral:7b" (~4GB), "llama3.2" (~4GB)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    # Alternative embedding models: "all-mpnet-base-v2" (better quality), "multilingual-mpnet-base-v2" (multilingual)
    EMBEDDING_MODEL_OPTIONS: List[str] = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "multilingual-mpnet-base-v2",
    ]

    # DeepSeek (external) — optional
    DEEPSEEK_API_BASE: str = "https://api.deepseek.com/v1"
    DEEPSEEK_API_KEY: Optional[str] = None
    # deepseek-chat was retired; the API now accepts deepseek-v4-pro,
    # deepseek-v4-flash and deepseek-v4-flash-vision-exp, and rejects
    # anything else with a 400 naming the three.
    DEEPSEEK_MODEL: str = "deepseek-v4-pro"
    DEEPSEEK_TIMEOUT_SECONDS: int = 120
    # These models reason before answering and charge it to max_tokens, so
    # this has to fit the thinking as well as the answer. At 2000 an agent
    # decision came back empty once the prompt grew.
    DEEPSEEK_MAX_RESPONSE_TOKENS: int = 8000
    # A floor under whatever a caller asks for. Call sites across this
    # codebase name budgets between 200 and 4096, all written when a model
    # emitted its answer directly. These models reason first and charge it
    # to the same budget, so a number that sizes the answer truncates the
    # thinking and the call returns empty. The caller's number stays the
    # intent; this is the headroom it did not know it needed.
    DEEPSEEK_MIN_COMPLETION_TOKENS: int = 4000

    # Where reproducibility bundles are written. Hardcoded to the container
    # path until a run outside it failed every record_entry with "Read-only
    # file system: '/app'" -- a warning, so the run continued and simply
    # produced no bundle. That is the quietest possible version of the gap
    # this machinery exists to close.
    AGENT_BUNDLE_ROOT: str = "/app/data/agent-bundles"

    # OpenAI (external, official SDK) — optional
    OPENAI_API_BASE: str = "https://api.openai.com/v1"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"

    # Anthropic (external, official SDK) — optional
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-opus-4-8"
    ANTHROPIC_MAX_TOKENS: int = 16000
    # Add cache_control breakpoints (system + newest message) so stable
    # prompt prefixes are served from Anthropic's prompt cache (~0.1x cost).
    ANTHROPIC_PROMPT_CACHE_ENABLED: bool = True

    # Qwen via DashScope OpenAI-compatible mode (external) — optional
    QWEN_API_BASE: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_API_KEY: Optional[str] = None
    QWEN_MODEL: str = "qwen-plus"

    # Kimi / Moonshot AI (external, OpenAI-compatible) — optional
    KIMI_API_BASE: str = "https://api.moonshot.cn/v1"
    KIMI_API_KEY: Optional[str] = None
    KIMI_MODEL: str = "kimi-latest"

    # LLM call snapshots (replay/debug observability). Opt-in: snapshots
    # store full prompt and response text in the llm_call_snapshots table.
    LLM_CALL_SNAPSHOT_ENABLED: bool = False
    LLM_CALL_SNAPSHOT_MAX_CHARS: int = 20000

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
    # Ed25519 seed is base64/base64url-encoded raw 32-byte private key material.
    # When omitted, a stable development seed is domain-derived from SECRET_KEY.
    AUTONOMOUS_RND_AUDIT_SIGNING_KEY_ID: str = "knowledgeops-ed25519-v1"
    AUTONOMOUS_RND_AUDIT_SIGNING_PRIVATE_KEY: Optional[str] = None
    # Launching an evaluation suite fans out into many unattended agent jobs,
    # so it stays opt-in and hard-capped on total trial jobs per launch.
    AUTONOMOUS_RND_EVAL_LAUNCH_ENABLED: bool = False
    AUTONOMOUS_RND_EVAL_MAX_TRIAL_JOBS: int = 30
    AUTONOMOUS_RND_EVAL_TRIAL_MAX_ITERATIONS: int = 25
    AUTONOMOUS_RND_EVAL_TRIAL_MAX_RUNTIME_MINUTES: int = 30
    # Exact hostnames permitted to resolve to private/loopback addresses through
    # the external-system gateway (for example a Docker-internal CompOps API).
    EXTERNAL_GATEWAY_PRIVATE_HOST_ALLOWLIST: str = ""

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

    # Vision
    VISION_MODEL: str = (
        "llava"  # Vision-capable model for image analysis (e.g. llava, llava:13b)
    )

    # Transcription
    WHISPER_MODEL_SIZE: str = "base"  # Options: tiny, base, small, medium, large
    WHISPER_DEVICE: str = "auto"  # Options: cpu, cuda, auto
    TRANSCRIPTION_LANGUAGE: str = "auto"  # Default language for transcription ("auto" enables Whisper language detection)
    TRANSCRIPTION_SPEAKER_DIARIZATION: bool = (
        False  # Enable speaker labels in transcripts
    )
    TRANSCRIPTION_DIARIZATION_MODEL: str = (
        "pyannote/speaker-diarization-3.1"  # Pyannote diarization model
    )
    HUGGINGFACE_TOKEN: Optional[
        str
    ] = None  # HF token required to download some pyannote models
    TRANSCRIPTION_FILTER_INTRO_JUNK: bool = True
    TRANSCRIPTION_INTRO_MAX_SECONDS: float = 12.0
    TRANSCRIPTION_INTRO_NO_SPEECH_PROB: float = 0.30
    TRANSCRIPTION_SKIP_INITIAL_SECONDS: float = 0.0

    # LDAP (optional)
    LDAP_ENABLED: bool = False
    LDAP_URI: Optional[
        str
    ] = None  # e.g. "ldap://ldap.example.com:389" or "ldaps://ldap.example.com:636"
    LDAP_START_TLS: bool = False
    LDAP_INSECURE_SKIP_TLS_VERIFY: bool = False
    LDAP_CONNECT_TIMEOUT_SECONDS: int = 8

    # Service account used to search for user DN (recommended). If not set, DN template is used.
    LDAP_BIND_DN: Optional[str] = None
    LDAP_BIND_PASSWORD: Optional[str] = None

    # User search
    LDAP_BASE_DN: Optional[str] = None  # e.g. "dc=example,dc=com"
    LDAP_USER_DN_TEMPLATE: Optional[
        str
    ] = None  # e.g. "uid={username},ou=People,dc=example,dc=com"
    LDAP_USER_SEARCH_FILTER: str = (
        "(|(uid={username})(sAMAccountName={username})(userPrincipalName={username}))"
    )
    LDAP_IMPORT_FILTER: str = (
        "(|(objectClass=person)(objectClass=inetOrgPerson)(objectClass=user))"
    )
    LDAP_SEARCH_PAGE_SIZE: int = 200

    # Attribute mapping
    LDAP_USERNAME_ATTRIBUTE: str = "uid"
    LDAP_EMAIL_ATTRIBUTE: str = "mail"
    LDAP_FULL_NAME_ATTRIBUTE: str = "displayName"
    LDAP_GROUPS_ATTRIBUTE: str = "memberOf"
    LDAP_DEFAULT_EMAIL_DOMAIN: Optional[
        str
    ] = None  # if LDAP has no email, we can synthesize `${username}@domain`
    LDAP_USER_ATTRIBUTES: str = (
        "uid,sAMAccountName,userPrincipalName,mail,cn,displayName,memberOf"
    )

    # Role mapping (comma-separated group DNs)
    LDAP_ADMIN_GROUP_DNS: Optional[str] = None
    LDAP_VIEWER_GROUP_DNS: Optional[str] = None
    LDAP_SYNC_ON_LOGIN: bool = True
    LDAP_CREATE_USER_ON_LOGIN: bool = True

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
    SUMMARIZATION_HEAVY_THRESHOLD_CHARS: int = (
        30000  # Above this, treat as heavy and prefer external provider
    )
    SUMMARIZATION_CHUNK_SIZE_CHARS: int = (
        12000  # Per-chunk size for document summarization
    )
    SUMMARIZATION_CHUNK_OVERLAP_CHARS: int = (
        800  # Overlap between chunks to preserve continuity
    )
    KNOWLEDGE_GRAPH_ENABLED: bool = True
    SUMMARIZATION_ENABLED: bool = True
    AUTO_SUMMARIZE_ON_PROCESS: bool = False

    # Knowledge Graph Extraction
    KG_LLM_EXTRACTION_ENABLED: bool = (
        True  # Use LLM for better entity/relationship extraction
    )
    KG_EXTRACTION_MODEL: Optional[
        str
    ] = None  # Model for KG extraction (None = use default)
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
    SCIENTIFIC_VALIDATION_ALLOWED_DOCKER_IMAGES: str = (
        "ghcr.io/al3x3n0/kdbc-compiler-research:latest,"
        "ghcr.io/al3x3n0/kdbc-microarch-research:latest,"
        "ghcr.io/al3x3n0/kdbc-axis-research:latest,"
        "ghcr.io/al3x3n0/kdbc-profiling-research:latest,"
        "ghcr.io/al3x3n0/kdbc-gem5-research:latest,"
        "python:3.11-slim"
    )
    SCIENTIFIC_VALIDATION_ALLOWED_CAPABILITIES: str = (
        "repo_reconstruction,perf_counters"
    )
    SCIENTIFIC_VALIDATION_ALLOWED_BENCHMARK_FAMILIES: str = (
        "compiler_regression,codegen_quality,kernel_compile,"
        "perf_counter_regression,cache_branch_analysis,throughput_latency,generic_validation"
    )
    SCIENTIFIC_VALIDATION_ALLOWED_PERF_COLLECTORS: str = (
        "benchmark_output,compile_time,artifact_diff,perf_stat,cache_miss,branch_miss"
    )
    SCIENTIFIC_VALIDATION_MAX_TIMEOUT_SECONDS: int = 1800
    SCIENTIFIC_VALIDATION_MAX_MEMORY_MB: int = 8192
    SCIENTIFIC_VALIDATION_MAX_CPUS: float = 8.0
    SCIENTIFIC_VALIDATION_MAX_PIDS_LIMIT: int = 1024
    SCIENTIFIC_VALIDATION_MAX_BUDGET_PER_RUN: float = 10000.0

    # RAG Knowledge Graph Integration
    RAG_KG_CONTEXT_ENABLED: bool = True  # Inject KG context into chat responses
    RAG_KG_MAX_ENTITIES: int = 10  # Max entities to include in context
    RAG_KG_MAX_RELATIONSHIPS: int = 15  # Max relationships to include

    # RAG Configuration
    RAG_HYBRID_SEARCH_ENABLED: bool = True
    RAG_HYBRID_SEARCH_ALPHA: float = (
        0.7  # Semantic weight (0.0 = keyword only, 1.0 = semantic only)
    )
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
    RAG_DEDUPLICATION_THRESHOLD: float = (
        0.95  # Similarity threshold for considering duplicates
    )

    # Kroki (local diagram rendering)
    KROKI_URL: str = "http://localhost:8001"  # Local Kroki Docker container
    KROKI_FALLBACK_URL: str = "https://kroki.io"  # External fallback
    KROKI_USE_FALLBACK: bool = True  # Fall back to external if local fails
    # Whether KROKI_URL points at a Kroki *companion* (one renderer, raw POST
    # to /svg) rather than the full gateway. The gateway bundles every diagram
    # backend at 3.76 GB; the only caller here renders Mermaid, so the stack
    # runs the companion alone. Set False when pointing at a real gateway.
    KROKI_LOCAL_IS_COMPANION: bool = True

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
    MINIO_PROXY_BASE_URL: Optional[
        str
    ] = None  # Base URL for nginx proxy (e.g., "http://localhost:3000/minio")

    # Secrets vault
    SECRETS_ENCRYPTION_KEY: Optional[
        str
    ] = None  # Optional Fernet key (urlsafe base64, 32 bytes)

    # Agent governance
    AGENT_REQUIRE_TOOL_APPROVAL: bool = True
    AGENT_DANGEROUS_TOOLS: List[str] = [
        "delete_document",
        "batch_delete_documents",
        "delete_entity",
        "merge_entities",
        "run_custom_tool",
    ]
    # Native tool-calling loop in the agent think phase (opt-in globally;
    # per-job override via job config key `native_tool_loop`). Requires a
    # provider with native tool calling (all providers in llm_providers).
    AGENT_NATIVE_TOOL_LOOP_ENABLED: bool = False
    AGENT_NATIVE_TOOL_LOOP_MAX_TOOL_CALLS: int = 5
    AGENT_NATIVE_TOOL_LOOP_MAX_LLM_CALLS: int = 6
    # Automatic context compaction: when serialized iteration state exceeds
    # the threshold, older actions are summarized into compressed_history
    # (same contract as the agent-invoked compress_history tool). Per-job
    # override via job config key `auto_compaction`.
    AGENT_AUTO_COMPACTION_ENABLED: bool = True
    AGENT_AUTO_COMPACTION_THRESHOLD_CHARS: int = 60000
    AGENT_AUTO_COMPACTION_KEEP_RECENT_ACTIONS: int = 5
    AGENT_AUTO_COMPACTION_MIN_ITERATIONS_BETWEEN: int = 3
    REPO_SYMBOL_RETRIEVAL_ENABLED: bool = False
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
    AI_HUB_EVAL_TEMPLATES_DIR: Optional[
        str
    ] = None  # Optional override for eval template "plugins"
    AI_HUB_EVAL_ENABLED_TEMPLATE_IDS: Optional[
        str
    ] = None  # Comma-separated template IDs allowed for non-admin users
    AI_HUB_DATASET_PRESETS_DIR: Optional[
        str
    ] = None  # Optional override for dataset preset "plugins"
    AI_HUB_DATASET_ENABLED_PRESET_IDS: Optional[
        str
    ] = None  # Comma-separated preset IDs allowed for non-admin users

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

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

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
            in_docker = os.path.exists("/.dockerenv") or os.path.exists(
                "/run/.containerenv"
            )
        except Exception:
            in_docker = False

        if (
            in_docker
            and isinstance(v, str)
            and ("localhost:11434" in v or "127.0.0.1:11434" in v)
        ):
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
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
)
logger.add(
    lambda msg: print(msg, end=""),
    level=settings.LOG_LEVEL,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | {message}",
)
