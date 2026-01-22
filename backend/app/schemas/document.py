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
