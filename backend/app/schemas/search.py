"""
Search schemas for request/response models.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Individual search result."""
    id: str
    title: str
    source: str
    source_type: str
    file_type: Optional[str] = None
    author: Optional[str] = None
    snippet: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    created_at: str
    updated_at: str
    url: Optional[str] = None
    download_url: Optional[str] = None
    chunk_id: Optional[str] = None


class SearchResponse(BaseModel):
    """Search response with results and metadata."""
    results: List[SearchResult]
    total: int
    page: int
    page_size: int
    query: str
    mode: str
    took_ms: int
