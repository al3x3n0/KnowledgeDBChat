"""
Pydantic schemas for presentation generation.
"""

from datetime import datetime
from typing import Optional, List, Literal, Any, Dict
from uuid import UUID
from pydantic import BaseModel, Field


# =============================================================================
# Theme Configuration Schemas
# =============================================================================

class ThemeColors(BaseModel):
    """Color configuration for a presentation theme."""
    title_color: str = Field(default="#1a365d", description="Title text color (hex)")
    accent_color: str = Field(default="#2e86ab", description="Accent/highlight color (hex)")
    text_color: str = Field(default="#333333", description="Body text color (hex)")
    bg_color: str = Field(default="#ffffff", description="Background color (hex)")


class ThemeFonts(BaseModel):
    """Font configuration for a presentation theme."""
    title_font: str = Field(default="Calibri", description="Font for titles")
    body_font: str = Field(default="Calibri", description="Font for body text")


class ThemeSizes(BaseModel):
    """Font size configuration for a presentation theme."""
    title_size: int = Field(default=44, ge=20, le=72, description="Title font size (pt)")
    subtitle_size: int = Field(default=24, ge=12, le=48, description="Subtitle font size (pt)")
    heading_size: int = Field(default=36, ge=16, le=60, description="Heading font size (pt)")
    body_size: int = Field(default=20, ge=10, le=36, description="Body font size (pt)")
    bullet_size: int = Field(default=18, ge=10, le=32, description="Bullet point font size (pt)")


class ThemeConfig(BaseModel):
    """Complete theme configuration."""
    colors: ThemeColors = Field(default_factory=ThemeColors)
    fonts: ThemeFonts = Field(default_factory=ThemeFonts)
    sizes: ThemeSizes = Field(default_factory=ThemeSizes)


# =============================================================================
# Template Schemas
# =============================================================================

class PresentationTemplateCreate(BaseModel):
    """Request schema for creating a presentation template."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    template_type: Literal["theme", "pptx"] = Field(default="theme")
    theme_config: Optional[ThemeConfig] = None
    is_public: bool = Field(default=False, description="Share template with all users")


class PresentationTemplateUpdate(BaseModel):
    """Request schema for updating a presentation template."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    theme_config: Optional[ThemeConfig] = None
    is_public: Optional[bool] = None
    is_active: Optional[bool] = None


class PresentationTemplateResponse(BaseModel):
    """Response schema for presentation templates."""
    id: UUID
    user_id: Optional[UUID] = None
    name: str
    description: Optional[str] = None
    template_type: str
    theme_config: Optional[Dict[str, Any]] = None
    preview_url: Optional[str] = None
    is_system: bool
    is_public: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# =============================================================================
# Slide Content Schemas
# =============================================================================

class SlideContent(BaseModel):
    """Content for a single slide."""
    slide_number: int
    slide_type: Literal["title", "content", "diagram", "summary", "two_column"]
    title: str
    content: List[str] = Field(default_factory=list)  # Bullet points or paragraphs
    diagram_code: Optional[str] = None  # Mermaid code for diagram slides
    diagram_description: Optional[str] = None  # Description for generating diagram
    notes: Optional[str] = None  # Speaker notes
    subtitle: Optional[str] = None  # For title slides


class PresentationOutline(BaseModel):
    """Complete presentation outline with all slides."""
    title: str
    subtitle: Optional[str] = None
    slides: List[SlideContent]


# =============================================================================
# Request/Response Schemas
# =============================================================================

class PresentationJobCreate(BaseModel):
    """Request schema for creating a presentation job."""
    title: str = Field(..., min_length=1, max_length=255)
    topic: str = Field(..., min_length=1, description="Topic or description for the presentation")
    source_document_ids: List[UUID] = Field(
        default_factory=list,
        description="Document IDs to use as context. Empty means search all documents."
    )
    slide_count: int = Field(default=10, ge=3, le=30, description="Number of slides to generate")
    style: Literal["professional", "casual", "technical", "modern", "minimal", "corporate", "creative"] = Field(
        default="professional",
        description="Built-in visual style (ignored if template_id or custom_theme is provided)"
    )
    include_diagrams: bool = Field(
        default=True,
        description="Whether to include Mermaid diagrams"
    )
    template_id: Optional[UUID] = Field(
        default=None,
        description="ID of a saved template to use"
    )
    custom_theme: Optional[ThemeConfig] = Field(
        default=None,
        description="Custom inline theme configuration"
    )


class PresentationJobUpdate(BaseModel):
    """Schema for updating a presentation job (internal use)."""
    status: Optional[str] = None
    progress: Optional[int] = None
    current_stage: Optional[str] = None
    generated_outline: Optional[dict] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class PresentationJobResponse(BaseModel):
    """Response schema for presentation jobs."""
    id: UUID
    user_id: UUID
    title: str
    topic: str
    source_document_ids: List[str] = Field(default_factory=list)
    slide_count: int
    style: str
    include_diagrams: bool
    template_id: Optional[UUID] = None
    template_name: Optional[str] = None  # For display
    custom_theme: Optional[Dict[str, Any]] = None
    status: str
    progress: int
    current_stage: Optional[str] = None
    generated_outline: Optional[PresentationOutline] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    download_url: Optional[str] = None

    class Config:
        from_attributes = True


class PresentationJobListResponse(BaseModel):
    """Response schema for listing presentation jobs."""
    jobs: List[PresentationJobResponse]
    total: int


# =============================================================================
# Progress WebSocket Messages
# =============================================================================

class PresentationProgressMessage(BaseModel):
    """WebSocket message for presentation generation progress."""
    type: Literal["progress", "stage", "complete", "error"]
    job_id: UUID
    status: Optional[str] = None
    progress: Optional[int] = None
    stage: Optional[str] = None
    message: Optional[str] = None
    error: Optional[str] = None
