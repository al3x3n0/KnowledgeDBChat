"""
Export job models for DOCX/PDF document generation.
"""

from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class ExportJob(Base):
    """
    Represents a document export job.

    Tracks the status and progress of exporting content to DOCX or PDF format.
    Content can come from chat sessions, document summaries, or custom LLM-generated text.
    """
    __tablename__ = "export_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # Export type: what kind of content is being exported
    export_type = Column(String(50), nullable=False)  # 'chat', 'document_summary', 'custom'

    # Output format
    output_format = Column(String(20), nullable=False)  # 'docx', 'pdf'

    # Source reference (what content to export)
    source_type = Column(String(50), nullable=False)  # 'chat_session', 'document', 'llm_content'
    source_id = Column(UUID(as_uuid=True), nullable=True)  # ID of chat session or document

    # For custom/LLM content export - the actual content
    content = Column(Text, nullable=True)

    # Content format hint
    content_format = Column(String(20), default="markdown")  # 'markdown', 'html', 'plain'

    # Document metadata
    title = Column(String(255), nullable=False)

    # Style configuration
    style = Column(String(50), default="professional")  # 'professional', 'casual', 'technical'
    custom_theme = Column(JSON, nullable=True)
    # Example custom_theme:
    # {
    #   "title_color": "#1a365d",
    #   "heading_color": "#2e86ab",
    #   "text_color": "#333333",
    #   "title_font": "Arial",
    #   "body_font": "Times New Roman",
    #   "title_size": 24,
    #   "body_size": 12
    # }

    # Job status
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    progress = Column(Integer, default=0)  # 0-100
    current_stage = Column(String(100), nullable=True)  # Current processing stage

    # Output
    file_path = Column(String(500), nullable=True)  # MinIO path to generated file
    file_size = Column(Integer, nullable=True)  # File size in bytes

    # Error handling
    error = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="export_jobs")

    def __repr__(self):
        return f"<ExportJob {self.id} - {self.title} ({self.status})>"
