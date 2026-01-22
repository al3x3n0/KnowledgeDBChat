"""
Document-related database models.
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.core.database import Base


class DocumentSource(Base):
    """Document source configuration."""
    
    __tablename__ = "document_sources"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(100), nullable=False, unique=True)
    source_type = Column(String(50), nullable=False)  # gitlab, confluence, web, file
    config = Column(JSON, nullable=False)  # Source-specific configuration
    is_active = Column(Boolean, default=True)
    is_syncing = Column(Boolean, default=False, nullable=False)
    last_error = Column(Text, nullable=True)
    last_sync = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    documents = relationship("Document", back_populates="source", cascade="all, delete-orphan")
    sync_logs = relationship("DocumentSourceSyncLog", back_populates="source", cascade="all, delete-orphan")


class Document(Base):
    """Document metadata and content."""

    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=True)  # Full document content
    content_hash = Column(String(64), nullable=False, index=True)  # SHA256 hash for change detection
    url = Column(String(1000), nullable=True)  # Original URL
    file_path = Column(String(1000), nullable=True)  # Local file path
    file_type = Column(String(50), nullable=True)  # pdf, docx, txt, html, etc.
    file_size = Column(Integer, nullable=True)  # File size in bytes

    # Source information
    source_id = Column(UUID(as_uuid=True), ForeignKey("document_sources.id"), nullable=False, index=True)
    source_identifier = Column(String(500), nullable=False)  # Source-specific ID

    # Metadata
    author = Column(String(200), nullable=True)
    tags = Column(JSON, nullable=True)  # List of tags
    extra_metadata = Column(JSON, nullable=True)  # Additional metadata
    owner_persona_id = Column(UUID(as_uuid=True), ForeignKey("personas.id", ondelete="SET NULL"), nullable=True, index=True)
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    processing_error = Column(Text, nullable=True)
    
    # Summarization
    summary = Column(Text, nullable=True)
    summary_model = Column(String(100), nullable=True)
    summary_generated_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    last_modified = Column(DateTime(timezone=True), nullable=True)  # From source
    
    # Relationships
    source = relationship("DocumentSource", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    owner_persona = relationship("Persona", back_populates="owned_documents", foreign_keys=[owner_persona_id])
    persona_detections = relationship("DocumentPersonaDetection", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, title='{self.title[:50]}...')>"


class DocumentSourceSyncLog(Base):
    """Sync history for document sources."""

    __tablename__ = "document_source_sync_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(UUID(as_uuid=True), ForeignKey("document_sources.id"), nullable=False, index=True)
    task_id = Column(String(100), nullable=True)
    status = Column(String(20), nullable=False, default="running")  # running, success, failed, canceled
    started_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    total_documents = Column(Integer, nullable=True)
    processed = Column(Integer, nullable=True)
    created = Column(Integer, nullable=True)
    updated = Column(Integer, nullable=True)
    errors = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)

    source = relationship("DocumentSource", back_populates="sync_logs")


class DocumentChunk(Base):
    """Document chunks for vector search."""

    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True)
    
    # Chunk content
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)  # SHA256 hash
    
    # Position information
    chunk_index = Column(Integer, nullable=False)  # Order within document
    start_pos = Column(Integer, nullable=True)  # Start position in original content
    end_pos = Column(Integer, nullable=True)  # End position in original content
    
    # Embedding information
    embedding_id = Column(String(100), nullable=True)  # ChromaDB document ID
    embedding_hash = Column(String(64), nullable=True)  # Hash of embedding model + content
    
    # Metadata
    extra_metadata = Column(JSON, nullable=True)  # Additional chunk metadata
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"


class GitBranch(Base):
    """Cache table for git branch metadata per source."""

    __tablename__ = "git_branches"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(UUID(as_uuid=True), ForeignKey("document_sources.id"), nullable=False, index=True)
    repository = Column(String(255), nullable=False)
    name = Column(String(120), nullable=False)
    head_sha = Column(String(100), nullable=True)
    head_timestamp = Column(DateTime(timezone=True), nullable=True)
    branch_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    source = relationship("DocumentSource", backref="git_branches")


class GitBranchDiff(Base):
    """Records git branch comparison jobs and their results."""

    __tablename__ = "git_branch_diffs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(UUID(as_uuid=True), ForeignKey("document_sources.id"), nullable=False, index=True)
    repository = Column(String(255), nullable=False)
    base_branch = Column(String(120), nullable=False)
    compare_branch = Column(String(120), nullable=False)
    status = Column(String(20), nullable=False, default="queued")
    task_id = Column(String(120), nullable=True)
    diff_summary = Column(JSON, nullable=True)
    llm_summary = Column(Text, nullable=True)
    options = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    source = relationship("DocumentSource", backref="branch_diffs")
