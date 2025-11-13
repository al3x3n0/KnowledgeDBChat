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
    last_sync = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    documents = relationship("Document", back_populates="source", cascade="all, delete-orphan")


class Document(Base):
    """Document metadata and content."""
    
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=True)  # Full document content
    content_hash = Column(String(64), nullable=False)  # SHA256 hash for change detection
    url = Column(String(1000), nullable=True)  # Original URL
    file_path = Column(String(1000), nullable=True)  # Local file path
    file_type = Column(String(50), nullable=True)  # pdf, docx, txt, html, etc.
    file_size = Column(Integer, nullable=True)  # File size in bytes
    
    # Source information
    source_id = Column(UUID(as_uuid=True), ForeignKey("document_sources.id"), nullable=False)
    source_identifier = Column(String(500), nullable=False)  # Source-specific ID
    
    # Metadata
    author = Column(String(200), nullable=True)
    tags = Column(JSON, nullable=True)  # List of tags
    extra_metadata = Column(JSON, nullable=True)  # Additional metadata
    
    # Processing status
    is_processed = Column(Boolean, default=False)
    processing_error = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    last_modified = Column(DateTime(timezone=True), nullable=True)  # From source
    
    # Relationships
    source = relationship("DocumentSource", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, title='{self.title[:50]}...')>"


class DocumentChunk(Base):
    """Document chunks for vector search."""
    
    __tablename__ = "document_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    
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


