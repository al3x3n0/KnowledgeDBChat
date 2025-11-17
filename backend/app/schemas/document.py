"""
Document-related Pydantic schemas.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field


class DocumentSourceCreate(BaseModel):
    """Schema for creating a document source."""
    name: str = Field(..., min_length=1, max_length=100)
    source_type: str = Field(..., pattern="^(gitlab|confluence|web|file)$")
    config: Dict[str, Any] = Field(..., description="Source-specific configuration")


class DocumentSourceResponse(BaseModel):
    """Schema for document source response."""
    id: UUID
    name: str
    source_type: str
    config: Dict[str, Any]
    is_active: bool
    last_sync: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class DocumentChunkResponse(BaseModel):
    """Schema for document chunk response."""
    id: UUID
    content: str
    chunk_index: int
    start_pos: Optional[int]
    end_pos: Optional[int]
    embedding_id: Optional[str]
    extra_metadata: Optional[Dict[str, Any]]
    created_at: datetime
    
    class Config:
        from_attributes = True


class DocumentResponse(BaseModel):
    """Schema for document response."""
    id: UUID
    title: str
    content: Optional[str]
    content_hash: str
    url: Optional[str]
    file_path: Optional[str]
    file_type: Optional[str]
    file_size: Optional[int]
    source_identifier: str
    author: Optional[str]
    tags: Optional[List[str]]
    extra_metadata: Optional[Dict[str, Any]]
    is_processed: bool
    processing_error: Optional[str]
    created_at: datetime
    updated_at: datetime
    last_modified: Optional[datetime]
    source: DocumentSourceResponse
    chunks: Optional[List[DocumentChunkResponse]] = []
    download_url: Optional[str] = None  # Presigned download URL (generated on demand)
    
    class Config:
        from_attributes = True


class DocumentUpload(BaseModel):
    """Schema for document upload."""
    title: Optional[str] = None
    tags: Optional[List[str]] = []
    extra_metadata: Optional[Dict[str, Any]] = {}


class DocumentSearch(BaseModel):
    """Schema for document search."""
    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=50)
    source_types: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    
    
class DocumentStats(BaseModel):
    """Schema for document statistics."""
    total_documents: int
    total_chunks: int
    processed_documents: int
    failed_documents: int
    sources_count: int
    last_sync: Optional[datetime]


