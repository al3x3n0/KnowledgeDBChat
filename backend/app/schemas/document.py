"""
Document-related Pydantic schemas.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from uuid import UUID
from pydantic import BaseModel, Field, computed_field
from pydantic import field_validator, model_validator

from app.schemas.persona import PersonaResponse, DocumentPersonaDetectionResponse


class DocumentSourceCreate(BaseModel):
    """Schema for creating a document source."""
    name: str = Field(..., min_length=1, max_length=100)
    source_type: str = Field(..., pattern="^(gitlab|github|confluence|web|file|arxiv)$")
    config: Dict[str, Any] = Field(..., description="Source-specific configuration")


class DocumentSourceResponse(BaseModel):
    """Schema for document source response."""
    id: UUID
    name: str
    source_type: str
    config: Dict[str, Any]
    is_active: bool
    is_syncing: bool | None = None
    last_sync: Optional[datetime]
    last_error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    class Config:
        from_attributes = True

    @computed_field(return_type=Dict[str, Any])
    def display_config(self) -> Dict[str, Any]:
        cfg = self.config or {}
        if isinstance(cfg, dict):
            display = cfg.get("display")
        if isinstance(display, dict):
            return display
        return {}


class ActiveSourceStatus(BaseModel):
    """Schema describing the active/pending state for a document source."""
    source: DocumentSourceResponse
    pending: bool = False
    task_id: Optional[str] = None


class DocumentChunkResponse(BaseModel):
    """Schema for document chunk response."""
    id: UUID
    content: str
    chunk_index: int
    start_pos: Optional[int]
    end_pos: Optional[int]
    embedding_id: Optional[str]
    extra_metadata: Optional[Dict[str, Any]]
    created_at: datetime
    
    class Config:
        from_attributes = True


class DocumentResponse(BaseModel):
    """Schema for document response."""
    id: UUID
    title: str
    content: Optional[str]
    content_hash: str
    url: Optional[str]
    file_path: Optional[str]
    file_type: Optional[str]
    file_size: Optional[int]
    source_identifier: str
    author: Optional[str]
    tags: Optional[List[str]]
    extra_metadata: Optional[Dict[str, Any]]
    is_processed: bool
    processing_error: Optional[str]
    # Summarization
    summary: Optional[str]
    summary_model: Optional[str]
    summary_generated_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    last_modified: Optional[datetime]
    source: DocumentSourceResponse
    owner_persona: Optional[PersonaResponse] = None
    persona_detections: Optional[List[DocumentPersonaDetectionResponse]] = []
    chunks: Optional[List[DocumentChunkResponse]] = []
    download_url: Optional[str] = None  # Presigned download URL (generated on demand)
    
    class Config:
        from_attributes = True


class DocumentUpload(BaseModel):
    """Schema for document upload."""
    title: Optional[str] = None
    tags: Optional[List[str]] = []
    extra_metadata: Optional[Dict[str, Any]] = {}


class GitRepoSourceRequest(BaseModel):
    """Schema for user-submitted Git repository processing."""
    provider: Literal["github", "gitlab"] = "github"
    name: Optional[str] = Field(
        default=None,
        description="Optional display name for the source"
    )
    token: Optional[str] = Field(
        default=None,
        min_length=10,
        description="Access token or PAT (required for GitLab or private GitHub repos)"
    )
    repositories: List[str] = Field(
        ...,
        min_length=1,
        description="List of owner/repo (GitHub) or project paths (GitLab)"
    )
    include_files: bool = True
    include_issues: bool = True
    include_pull_requests: bool = False
    include_wiki: bool = False
    incremental_files: bool = True
    use_gitignore: bool = True
    max_pages: int = Field(default=10, ge=1, le=100)
    gitlab_url: Optional[str] = Field(
        default=None,
        description="GitLab base URL (required when provider=gitlab if not configured globally)"
    )
    auto_sync: bool = Field(
        default=True,
        description="Start ingestion immediately after creating the source"
    )

    @field_validator("repositories", mode="after")
    @classmethod
    def _normalize_repos(cls, repos: List[str]) -> List[str]:
        cleaned = [repo.strip().strip("/") for repo in repos if repo and repo.strip()]
        if not cleaned:
            raise ValueError("At least one repository must be provided")
        return cleaned


class ArxivSourceRequest(BaseModel):
    """Schema for requesting ingestion of ArXiv papers."""
    name: Optional[str] = Field(
        default=None,
        description="Optional source name to display"
    )
    search_queries: Optional[List[str]] = Field(
        default=None,
        description="List of ArXiv API search_query expressions"
    )
    paper_ids: Optional[List[str]] = Field(
        default=None,
        description="Explicit ArXiv identifiers (e.g. 2401.12345)"
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="Category codes (e.g. cs.CL, cs.AI) that will be ANDed with each query"
    )
    max_results: int = Field(default=50, ge=1, le=200, description="Max results per query")
    start: int = Field(default=0, ge=0, le=1000, description="Offset for paginated queries")
    sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = "submittedDate"
    sort_order: Literal["ascending", "descending"] = "descending"
    auto_sync: bool = Field(default=True, description="Start ingestion immediately")
    auto_summarize: bool = Field(default=True, description="Queue summarization for ingested papers")
    auto_literature_review: bool = Field(default=False, description="Generate a literature review document after ingestion")
    auto_enrich_metadata: bool = Field(default=True, description="Enrich papers with BibTeX/DOI metadata after ingestion")
    topic: Optional[str] = Field(default=None, description="Optional topic label for literature review/reporting")

    @field_validator("search_queries", "paper_ids", "categories", mode="after")
    @classmethod
    def _normalize_lists(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return None
        cleaned = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return cleaned or None

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() if isinstance(value, str) and value.strip() else None

    @field_validator("topic")
    @classmethod
    def _strip_topic(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() if isinstance(value, str) and value.strip() else None

    @field_validator("paper_ids", mode="after")
    @classmethod
    def _normalize_ids(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if not value:
            return None
        normalized: List[str] = []
        for entry in value:
            entry = entry.strip()
            if not entry:
                continue
            if entry.lower().startswith("arxiv:"):
                entry = entry.split(":", 1)[1]
            normalized.append(entry)
        return normalized or None

    @model_validator(mode="after")
    def _ensure_inputs(cls, values: "ArxivSourceRequest") -> "ArxivSourceRequest":
        queries = values.search_queries
        ids = values.paper_ids
        categories = values.categories
        if (not queries or len(queries) == 0) and categories:
            combined = " OR ".join(f"cat:{cat}" for cat in categories)
            values.search_queries = [combined]
            queries = values.search_queries
        if (not queries or len(queries) == 0) and (not ids or len(ids) == 0):
            raise ValueError("Provide at least one search query, category, or arXiv ID")
        return values


class DocumentSearch(BaseModel):
    """Schema for document search."""
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=50)
    source_types: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    
    
class DocumentStats(BaseModel):
    """Schema for document statistics."""
    total_documents: int
    total_chunks: int
    processed_documents: int
    failed_documents: int
    sources_count: int
    last_sync: Optional[datetime]


# =============================================================================
# Instant URL Ingestion Schemas
# =============================================================================

class IngestUrlRequest(BaseModel):
    """Request to scrape a URL and ingest it as document(s)."""
    url: str = Field(..., description="URL to ingest (http/https)", min_length=1, max_length=2000)
    title: Optional[str] = Field(default=None, description="Optional title override (single-document mode)")
    tags: Optional[List[str]] = Field(default=None, description="Optional tags to apply")

    follow_links: bool = Field(default=False, description="Crawl a small link graph from the start page")
    max_pages: int = Field(default=1, ge=1, le=25, description="Max pages to fetch when crawling")
    max_depth: int = Field(default=0, ge=0, le=5, description="Max crawl depth")
    same_domain_only: bool = Field(default=True, description="Only follow links on the same domain")
    one_document_per_page: bool = Field(default=False, description="Create/update one document per page URL")

    allow_private_networks: bool = Field(
        default=False,
        description="Allow private-network hosts (admin only, or allowlisted web sources)",
    )
    max_content_chars: int = Field(default=50000, ge=1000, le=500000, description="Max characters per page to ingest")


class IngestUrlResponse(BaseModel):
    """Response after URL ingestion."""
    action: str = "ingested"
    root_url: Optional[str] = None
    total_pages_scraped: int = 0
    created: List[Dict[str, Any]] = Field(default_factory=list)
    updated: List[Dict[str, Any]] = Field(default_factory=list)
    skipped: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)


class IngestUrlJobResponse(BaseModel):
    """Response after scheduling background URL ingestion."""
    job_id: str
    progress_key: str


# =============================================================================
# Instant ArXiv Ingestion Schemas
# =============================================================================

class InstantArxivIngestRequest(BaseModel):
    """Request to instantly ingest an arXiv paper for immediate chat."""
    arxiv_input: str = Field(
        ...,
        description="arXiv URL (https://arxiv.org/abs/2401.12345) or paper ID (2401.12345)",
        min_length=1,
        max_length=500
    )
    auto_summarize: bool = Field(
        default=False,
        description="Queue background summarization after ingestion"
    )
    auto_enrich: bool = Field(
        default=False,
        description="Queue background metadata enrichment (BibTeX, DOI)"
    )

    @field_validator("arxiv_input")
    @classmethod
    def extract_arxiv_id(cls, v: str) -> str:
        """Extract and validate arXiv ID from URL or raw ID."""
        import re
        v = v.strip()

        # Handle various URL formats
        # https://arxiv.org/abs/2401.12345
        # https://arxiv.org/pdf/2401.12345.pdf
        # http://arxiv.org/abs/2401.12345v2
        # arxiv:2401.12345
        # 2401.12345

        patterns = [
            r'arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)',  # URL format
            r'arxiv:(\d{4}\.\d{4,5}(?:v\d+)?)',  # arxiv: prefix
            r'^(\d{4}\.\d{4,5}(?:v\d+)?)$',  # Raw new-style ID
            r'arxiv\.org/(?:abs|pdf)/([\w\-\.]+/\d+(?:v\d+)?)',  # Old-style ID in URL
            r'^([\w\-\.]+/\d+(?:v\d+)?)$',  # Raw old-style ID (e.g., cs.CL/0001234)
        ]

        for pattern in patterns:
            match = re.search(pattern, v, re.IGNORECASE)
            if match:
                return match.group(1)

        raise ValueError(
            f"Could not parse arXiv ID from '{v}'. "
            "Expected format: arXiv URL, 'arxiv:ID', or paper ID like '2401.12345'"
        )


class InstantArxivIngestResponse(BaseModel):
    """Response after instant arXiv paper ingestion."""
    document_id: UUID
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    categories: List[str]
    url: str
    pdf_url: str
    chunks_created: int
    ready_for_chat: bool = True
    background_tasks: List[str] = Field(
        default_factory=list,
        description="List of queued background tasks (summarize, enrich)"
    )


class ResearchPresentationRequest(BaseModel):
    """Request to generate a presentation from research topic."""
    topic: str = Field(
        ...,
        description="Research topic for the presentation",
        min_length=3,
        max_length=500
    )
    slide_count: int = Field(
        default=10,
        ge=5,
        le=25,
        description="Number of slides to generate"
    )
    include_arxiv: bool = Field(
        default=True,
        description="Search arXiv for relevant papers to include"
    )
    arxiv_max_papers: int = Field(
        default=5,
        ge=0,
        le=15,
        description="Maximum arXiv papers to incorporate"
    )
    style: str = Field(
        default="technical",
        description="Presentation style (professional, technical, casual, modern)"
    )
    include_diagrams: bool = Field(
        default=True,
        description="Include auto-generated diagrams"
    )
    source_document_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific document IDs to include as sources"
    )
