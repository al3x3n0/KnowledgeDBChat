"""
Pydantic schemas for research notes.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ResearchNoteCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content_markdown: str = Field(..., min_length=1)
    tags: Optional[List[str]] = None
    source_synthesis_job_id: Optional[UUID] = None
    source_document_ids: Optional[List[UUID]] = None


class ResearchNoteUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=500)
    content_markdown: Optional[str] = Field(default=None, min_length=1)
    tags: Optional[List[str]] = None


class ResearchNoteResponse(BaseModel):
    id: UUID
    user_id: UUID
    title: str
    content_markdown: str
    tags: Optional[List[str]] = None
    attribution: Optional[Dict[str, Any]] = None
    source_synthesis_job_id: Optional[UUID] = None
    source_document_ids: Optional[List[UUID]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ResearchNoteListResponse(BaseModel):
    items: List[ResearchNoteResponse]
    total: int
    limit: int
    offset: int


class ResearchNoteAttributionTrimOptions(BaseModel):
    include_details: bool = False


class ResearchNoteEnforceCitationsRequest(BaseModel):
    policy: str = Field(default="sentence", pattern="^(sentence|paragraph)$")
    update_content: bool = False
    append_bibliography: bool = False
    max_sources: int = Field(default=10, ge=1, le=25)
    max_source_chars: int = Field(default=2000, ge=200, le=8000)
    max_note_chars: int = Field(default=12000, ge=500, le=60000)
    use_vector_snippets: bool = True
    chunks_per_source: int = Field(default=3, ge=1, le=8)
    chunk_max_chars: int = Field(default=600, ge=100, le=2000)
    chunk_query: Optional[str] = None
    strict: bool = False
    max_uncited_examples: int = Field(default=10, ge=0, le=50)
    document_ids: Optional[List[UUID]] = None


class ResearchNoteLintCitationsRequest(BaseModel):
    max_sources: int = Field(default=10, ge=1, le=25)
    max_uncited_examples: int = Field(default=10, ge=0, le=50)
    document_ids: Optional[List[UUID]] = None


class ResearchNotesLintRecentRequest(BaseModel):
    window_hours: int = Field(default=24, ge=1, le=24 * 30)
    max_notes: int = Field(default=200, ge=1, le=2000)
    max_sources: int = Field(default=10, ge=1, le=25)
    max_uncited_examples: int = Field(default=10, ge=0, le=50)


class ResearchNotesLintRecentResponse(BaseModel):
    processed: int
    updated: int
    skipped: int
    missing_sources: int
