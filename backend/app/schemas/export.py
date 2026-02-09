"""
Pydantic schemas for export functionality (DOCX/PDF generation).
"""

from typing import Optional, Dict, Any, List, Literal
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field


class ExportTheme(BaseModel):
    """Theme configuration for export."""
    title_color: Optional[str] = Field(None, description="Title color hex code")
    heading_color: Optional[str] = Field(None, description="Heading color hex code")
    text_color: Optional[str] = Field(None, description="Body text color hex code")
    title_font: Optional[str] = Field(None, description="Title font family")
    body_font: Optional[str] = Field(None, description="Body font family")
    title_size: Optional[int] = Field(None, description="Title font size in points")
    heading_size: Optional[int] = Field(None, description="Heading font size in points")
    body_size: Optional[int] = Field(None, description="Body font size in points")


# Request schemas

class ExportChatRequest(BaseModel):
    """Request to export a chat session."""
    format: Literal["docx", "pdf"] = Field("docx", description="Output format")
    title: Optional[str] = Field(None, description="Document title (defaults to chat topic)")
    style: Literal["professional", "casual", "technical"] = Field("professional", description="Document style")
    custom_theme: Optional[ExportTheme] = Field(None, description="Custom theme settings")
    include_timestamps: bool = Field(True, description="Include message timestamps")
    include_sources: bool = Field(True, description="Include RAG source citations")


class ExportDocumentSummaryRequest(BaseModel):
    """Request to export a document summary."""
    format: Literal["docx", "pdf"] = Field("docx", description="Output format")
    title: Optional[str] = Field(None, description="Document title (defaults to source document title)")
    style: Literal["professional", "casual", "technical"] = Field("professional", description="Document style")
    custom_theme: Optional[ExportTheme] = Field(None, description="Custom theme settings")
    include_metadata: bool = Field(True, description="Include source document metadata")


class ExportCustomRequest(BaseModel):
    """Request to export custom/LLM-generated content."""
    format: Literal["docx", "pdf"] = Field("docx", description="Output format")
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Content to export (markdown or HTML)")
    content_format: Literal["markdown", "html", "plain"] = Field("markdown", description="Content format")
    style: Literal["professional", "casual", "technical"] = Field("professional", description="Document style")
    custom_theme: Optional[ExportTheme] = Field(None, description="Custom theme settings")


class ExportBatchRequest(BaseModel):
    """Request to export multiple items."""
    items: List[Dict[str, Any]] = Field(..., description="List of items to export")
    format: Literal["docx", "pdf"] = Field("docx", description="Output format")
    style: Literal["professional", "casual", "technical"] = Field("professional", description="Document style")


# Response schemas

class ExportJobResponse(BaseModel):
    """Response with export job details."""
    id: UUID
    user_id: UUID
    export_type: str
    output_format: str
    source_type: str
    source_id: Optional[UUID] = None
    title: str
    style: str
    custom_theme: Optional[Dict[str, Any]] = None
    status: str
    progress: int
    current_stage: Optional[str] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ExportJobCreatedResponse(BaseModel):
    """Response when export job is created."""
    job_id: UUID
    status: str
    message: str


class ExportJobStatusResponse(BaseModel):
    """Response with export job status."""
    id: UUID
    status: str
    progress: int
    current_stage: Optional[str] = None
    error: Optional[str] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    completed_at: Optional[datetime] = None


class ExportJobListResponse(BaseModel):
    """Response with list of export jobs."""
    items: List[ExportJobResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
