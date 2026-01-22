"""
Template-related Pydantic schemas.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field


class TemplateSectionResponse(BaseModel):
    """Schema for a template section."""
    title: str
    level: int
    placeholder_text: Optional[str] = None


class TemplateJobCreate(BaseModel):
    """Schema for creating a template fill job."""
    source_document_ids: List[UUID] = Field(
        ...,
        min_length=1,
        description="List of document IDs to use as sources"
    )


class TemplateJobResponse(BaseModel):
    """Schema for template job response."""
    id: UUID
    template_filename: str
    sections: Optional[List[Dict[str, Any]]] = None
    source_document_ids: List[str]
    status: str
    progress: int
    current_section: Optional[str] = None
    filled_filename: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    download_url: Optional[str] = None

    class Config:
        from_attributes = True


class TemplateJobListResponse(BaseModel):
    """Schema for listing template jobs."""
    jobs: List[TemplateJobResponse]
    total: int


class TemplateProgressUpdate(BaseModel):
    """Schema for WebSocket progress updates."""
    type: str  # progress, complete, error
    job_id: str
    stage: Optional[str] = None
    progress: Optional[int] = None
    current_section: Optional[str] = None
    section_index: Optional[int] = None
    total_sections: Optional[int] = None
    error: Optional[str] = None
    filled_filename: Optional[str] = None
