"""
Template job database models.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.core.database import Base


class TemplateJob(Base):
    """Template filling job record."""

    __tablename__ = "template_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)

    # Template file info
    template_file_path = Column(String(1000), nullable=False)  # MinIO path
    template_filename = Column(String(500), nullable=False)

    # Detected sections from template
    sections = Column(JSON, nullable=True)  # List of {title, level, placeholder_text}

    # Source documents (list of document IDs)
    source_document_ids = Column(JSON, nullable=False)  # List of UUID strings

    # Processing status
    status = Column(String(50), default="pending")  # pending, analyzing, extracting, filling, completed, failed
    progress = Column(Integer, default=0)  # 0-100
    current_section = Column(String(500), nullable=True)

    # Result
    filled_file_path = Column(String(1000), nullable=True)  # MinIO path to filled document
    filled_filename = Column(String(500), nullable=True)

    # Error tracking
    error_message = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", backref="template_jobs")

    def __repr__(self):
        return f"<TemplateJob(id={self.id}, status='{self.status}', template='{self.template_filename}')>"
