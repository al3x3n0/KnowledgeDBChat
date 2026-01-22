"""
Presentation job models for AI-powered presentation generation.
"""

from datetime import datetime
from typing import Optional, List
from uuid import uuid4

from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class PresentationTemplate(Base):
    """
    User-uploaded or system presentation templates.

    Templates can be:
    - Custom PPTX files uploaded by users
    - Built-in theme configurations (color schemes, fonts)
    """
    __tablename__ = "presentation_templates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True)  # NULL = system template

    # Template info
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    # Template type: 'pptx' (uploaded file) or 'theme' (color/font config)
    template_type = Column(String(50), default="theme")

    # For PPTX templates: MinIO file path
    file_path = Column(String(500), nullable=True)

    # Theme configuration (colors, fonts)
    theme_config = Column(JSON, nullable=True)
    # Example theme_config:
    # {
    #   "title_color": "#1a365d",
    #   "accent_color": "#2e86ab",
    #   "text_color": "#333333",
    #   "bg_color": "#ffffff",
    #   "title_font": "Calibri",
    #   "body_font": "Calibri",
    #   "title_size": 44,
    #   "body_size": 20
    # }

    # Preview image (thumbnail)
    preview_path = Column(String(500), nullable=True)

    # Flags
    is_system = Column(Boolean, default=False)  # Built-in templates
    is_public = Column(Boolean, default=False)  # Shared with all users
    is_active = Column(Boolean, default=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="presentation_templates")
    jobs = relationship("PresentationJob", back_populates="template")

    def __repr__(self):
        return f"<PresentationTemplate {self.id} - {self.name}>"


class PresentationJob(Base):
    """
    Represents a presentation generation job.

    Tracks the status and progress of AI-powered presentation creation,
    including the generated outline, source documents, and output file.
    """
    __tablename__ = "presentation_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    template_id = Column(UUID(as_uuid=True), ForeignKey("presentation_templates.id", ondelete="SET NULL"), nullable=True)

    # Request parameters
    title = Column(String(255), nullable=False)
    topic = Column(Text, nullable=False)
    source_document_ids = Column(JSON, default=list)  # List of document UUIDs for RAG
    slide_count = Column(Integer, default=10)
    style = Column(String(50), default="professional")  # professional, casual, technical (used if no template)
    include_diagrams = Column(Integer, default=1)  # Boolean as int for SQLite compatibility

    # Custom theme override (if not using a template)
    custom_theme = Column(JSON, nullable=True)  # Inline theme config

    # Job status
    status = Column(String(50), default="pending")  # pending, generating, rendering, completed, failed
    progress = Column(Integer, default=0)  # 0-100
    current_stage = Column(String(100), nullable=True)  # Current processing stage

    # Generated content
    generated_outline = Column(JSON, nullable=True)  # PresentationOutline as JSON

    # Output
    file_path = Column(String(500), nullable=True)  # MinIO path to generated PPTX
    file_size = Column(Integer, nullable=True)  # File size in bytes

    # Error handling
    error = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="presentation_jobs")
    template = relationship("PresentationTemplate", back_populates="jobs")

    def __repr__(self):
        return f"<PresentationJob {self.id} - {self.title} ({self.status})>"
