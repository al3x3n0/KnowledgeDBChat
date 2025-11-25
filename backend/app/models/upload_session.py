"""
Upload session model for chunked file uploads.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, Integer, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.core.database import Base


class UploadSession(Base):
    """Tracks chunked file upload sessions for resume capability."""
    
    __tablename__ = "upload_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    
    # File metadata
    filename = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)  # Total file size in bytes
    file_type = Column(String(50), nullable=True)  # MIME type
    content_type = Column(String(100), nullable=True)  # Content type
    
    # Upload metadata
    chunk_size = Column(Integer, nullable=False, default=5 * 1024 * 1024)  # 5MB chunks
    total_chunks = Column(Integer, nullable=False)  # Total number of chunks
    uploaded_chunks = Column(JSON, nullable=False, default=list)  # List of uploaded chunk indices
    uploaded_bytes = Column(Integer, nullable=False, default=0)  # Total bytes uploaded
    
    # MinIO multipart upload
    minio_upload_id = Column(String(200), nullable=True)  # MinIO multipart upload ID
    minio_part_etags = Column(JSON, nullable=True)  # Map of part number -> ETag
    
    # Document metadata (for final document creation)
    title = Column(String(500), nullable=True)
    tags = Column(JSON, nullable=True)  # List of tags
    extra_metadata = Column(JSON, nullable=True)  # Additional metadata (temp paths, etc.)
    
    # Status
    status = Column(String(50), nullable=False, default="pending")  # pending, uploading, completed, failed
    error_message = Column(String(1000), nullable=True)
    
    # Result
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True)  # Created document ID
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime(timezone=True), nullable=True)  # Session expiration (24 hours)
    
    def __repr__(self):
        return f"<UploadSession(id={self.id}, filename='{self.filename}', status='{self.status}', progress={self.uploaded_bytes}/{self.file_size})>"
    
    @property
    def progress_percentage(self) -> float:
        """Calculate upload progress percentage."""
        if self.file_size == 0:
            return 0.0
        return min((self.uploaded_bytes / self.file_size) * 100, 100.0)
    
    @property
    def is_complete(self) -> bool:
        """Check if upload is complete."""
        return self.status == "completed" and self.uploaded_bytes >= self.file_size
    
    @property
    def can_resume(self) -> bool:
        """Check if upload can be resumed."""
        return self.status in ["pending", "uploading"] and self.uploaded_bytes < self.file_size

