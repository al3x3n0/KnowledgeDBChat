"""
Pydantic schemas for structured paper extraction.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class PaperClaimResponse(BaseModel):
    id: UUID
    kind: str
    statement: str
    mechanism: Optional[str] = None
    target_layer: str
    conditions: Optional[List[str]] = None
    assumptions: Optional[List[str]] = None
    expected_effect: Optional[str] = None
    evidence_summary: Optional[str] = None
    confidence: Optional[float] = None
    tags: Optional[List[str]] = None
    rank: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class PaperExtractionJobResponse(BaseModel):
    id: UUID
    user_id: UUID
    document_id: UUID
    source_id: Optional[UUID] = None
    paper_id: Optional[UUID] = None
    status: str
    extractor_version: Optional[str] = None
    error: Optional[str] = None
    request_payload: Optional[Dict[str, Any]] = None
    result_summary: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ResearchPaperResponse(BaseModel):
    id: UUID
    user_id: UUID
    document_id: UUID
    source_id: Optional[UUID] = None
    arxiv_id: str
    title: str
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    published_at: Optional[datetime] = None
    categories: Optional[List[str]] = None
    paper_url: Optional[str] = None
    pdf_url: Optional[str] = None
    extraction_status: str
    extracted_at: Optional[datetime] = None
    extractor_version: Optional[str] = None
    summary: Optional[str] = None
    mechanisms: Optional[List[str]] = None
    assumptions: Optional[List[str]] = None
    benchmarks: Optional[List[str]] = None
    metrics: Optional[List[str]] = None
    limitations: Optional[List[str]] = None
    raw_extraction_payload: Optional[Dict[str, Any]] = None
    claims: List[PaperClaimResponse] = Field(default_factory=list)
    latest_job: Optional[PaperExtractionJobResponse] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ResearchPaperListResponse(BaseModel):
    items: List[ResearchPaperResponse]
    total: int
    limit: int
    offset: int


class PaperExtractionRequest(BaseModel):
    document_ids: List[UUID] = Field(default_factory=list, max_length=50)
    source_id: Optional[UUID] = None
    force: bool = False
    limit: int = Field(default=50, ge=1, le=500)


class SaveResearchPaperAsNoteRequest(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=500)
    tags: Optional[List[str]] = None
