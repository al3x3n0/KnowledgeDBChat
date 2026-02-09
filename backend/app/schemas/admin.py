"""
Admin-related Pydantic schemas.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class HealthCheckServiceResponse(BaseModel):
    """Schema for individual service health status."""
    status: str
    message: Optional[str] = None
    error: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Schema for system health check response."""
    timestamp: str
    overall_status: str
    services: Dict[str, HealthCheckServiceResponse]


class DocumentStatsResponse(BaseModel):
    """Schema for document statistics."""
    total: int
    processed: int
    failed: int
    pending: int
    success_rate: float


class ChatStatsResponse(BaseModel):
    """Schema for chat statistics."""
    total_sessions: int
    active_sessions_24h: int
    total_messages: int
    avg_messages_per_session: float


class SourceStatsResponse(BaseModel):
    """Schema for source statistics."""
    total: int
    active: int
    by_type: Dict[str, int]


class VectorStoreStatsResponse(BaseModel):
    """Schema for vector store statistics."""
    total_chunks: Optional[int] = None
    collection_name: Optional[str] = None
    embedding_model: Optional[str] = None
    error: Optional[str] = None


class ProcessingStatsResponse(BaseModel):
    """Schema for processing statistics."""
    documents_last_7_days: List[Dict[str, Any]]
    total_documents_last_7_days: int


class SystemStatsResponse(BaseModel):
    """Schema for comprehensive system statistics."""
    timestamp: str
    documents: Optional[DocumentStatsResponse] = None
    chat: Optional[ChatStatsResponse] = None
    sources: Optional[SourceStatsResponse] = None
    vector_store: Optional[VectorStoreStatsResponse] = None
    processing: Optional[ProcessingStatsResponse] = None
    error: Optional[str] = None


class TaskTriggerResponse(BaseModel):
    """Schema for task trigger response."""
    task_id: str
    message: str
    status: str


class TaskStatusResponse(BaseModel):
    """Schema for task status response."""
    active_tasks: Optional[Dict[str, Any]] = None
    scheduled_tasks: Optional[Dict[str, Any]] = None
    reserved_tasks: Optional[Dict[str, Any]] = None


class LogsResponse(BaseModel):
    """Schema for system logs response."""
    logs: List[str]
    total_lines: int
    returned_lines: int
    message: Optional[str] = None


class IngestionSourceStatusResponse(BaseModel):
    source_id: str
    name: str
    source_type: str
    is_active: bool
    is_syncing: bool
    last_sync: Optional[str] = None
    last_error: Optional[str] = None

    docs_total: int
    docs_processed: int
    docs_pending: int
    docs_failed: int

    chunks_total: int
    chunks_embedded: int
    chunks_missing_embedding: int

    last_sync_log: Optional[Dict[str, Any]] = None


class IngestionDBStatusResponse(BaseModel):
    documents_total: int
    documents_processed: int
    documents_pending: int
    documents_failed: int
    documents_without_chunks: int

    chunks_total: int
    chunks_embedded: int
    chunks_missing_embedding: int


class IngestionVectorStoreStatusResponse(BaseModel):
    provider: str
    collection_name: Optional[str] = None
    collection_exists: Optional[bool] = None
    points_total: Optional[int] = None
    error: Optional[str] = None


class IngestionStatusResponse(BaseModel):
    timestamp: str
    db: IngestionDBStatusResponse
    vector_store: IngestionVectorStoreStatusResponse
    sources: List[IngestionSourceStatusResponse]
    recent_document_errors: List[Dict[str, Any]]







