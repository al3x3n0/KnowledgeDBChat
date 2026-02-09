"""
Pydantic schemas for conversation memory system.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID
from pydantic import BaseModel, Field, validator

class MemoryBase(BaseModel):
    """Base memory schema."""
    memory_type: str = Field(..., description="Type of memory: fact, preference, context, summary")
    content: str = Field(..., description="Memory content")
    importance_score: float = Field(0.5, ge=0.0, le=1.0, description="Importance score (0.0 to 1.0)")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    tags: Optional[List[str]] = Field(None, description="Memory tags")

    @validator('memory_type')
    def validate_memory_type(cls, v):
        allowed_types = ['fact', 'preference', 'context', 'summary', 'goal', 'constraint']
        if v not in allowed_types:
            raise ValueError(f'memory_type must be one of {allowed_types}')
        return v

class MemoryCreate(MemoryBase):
    """Schema for creating a new memory."""
    session_id: Optional[UUID] = Field(None, description="Associated chat session")
    source_message_id: Optional[UUID] = Field(None, description="Source message ID")

class MemoryUpdate(BaseModel):
    """Schema for updating a memory."""
    content: Optional[str] = None
    importance_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    context: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    is_active: Optional[bool] = None

class MemoryResponse(MemoryBase):
    """Schema for memory response."""
    id: UUID
    user_id: UUID
    session_id: Optional[UUID]
    source_message_id: Optional[UUID]
    created_at: datetime
    last_accessed_at: datetime
    access_count: int
    is_active: bool

    class Config:
        from_attributes = True

class MemoryInteractionBase(BaseModel):
    """Base memory interaction schema."""
    interaction_type: str = Field(..., description="Type of interaction: retrieved, updated, reinforced, contradicted")
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Relevance score")
    usage_context: Optional[Dict[str, Any]] = Field(None, description="Usage context")

    @validator('interaction_type')
    def validate_interaction_type(cls, v):
        allowed_types = ['retrieved', 'updated', 'reinforced', 'contradicted', 'created', 'deleted']
        if v not in allowed_types:
            raise ValueError(f'interaction_type must be one of {allowed_types}')
        return v

class MemoryInteractionCreate(MemoryInteractionBase):
    """Schema for creating a memory interaction."""
    memory_id: UUID
    session_id: UUID
    message_id: Optional[UUID] = None

class MemoryInteractionResponse(MemoryInteractionBase):
    """Schema for memory interaction response."""
    id: UUID
    memory_id: UUID
    session_id: UUID
    message_id: Optional[UUID]
    created_at: datetime

    class Config:
        from_attributes = True

class UserPreferencesBase(BaseModel):
    """Base user preferences schema."""
    memory_retention_days: int = Field(90, ge=1, le=365, description="Memory retention period in days")
    max_memories_per_session: int = Field(10, ge=1, le=100, description="Max memories to load per session")
    memory_importance_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Minimum importance to store")
    auto_summarize_sessions: bool = Field(True, description="Auto-create session summaries")
    allow_cross_session_memory: bool = Field(True, description="Share memories across sessions")
    allow_personal_data_storage: bool = Field(True, description="Store personal information")

    # LLM preferences (per-user overrides)
    llm_provider: Optional[str] = Field(None, description="LLM provider override: ollama, deepseek, openai, or custom")
    llm_model: Optional[str] = Field(None, description="Model name override (default for chat)")
    llm_api_url: Optional[str] = Field(None, description="Custom API URL override")
    llm_api_key: Optional[str] = Field(None, description="User's own API key for external providers")
    llm_temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature override")
    llm_max_tokens: Optional[int] = Field(None, ge=1, le=32000, description="Max response tokens override")

    # Per-task model overrides
    llm_task_models: Optional[Dict[str, str]] = Field(
        None,
        description="Task-specific model overrides: title_generation, summarization, query_expansion, memory_extraction"
    )

    # Paper algorithm agent defaults
    paper_algo_default_run_demo_check: bool = Field(False, description="Default: run demo check after generating paper algorithm projects (when available)")

class UserPreferencesCreate(UserPreferencesBase):
    """Schema for creating user preferences."""
    user_id: UUID

class UserPreferencesUpdate(BaseModel):
    """Schema for updating user preferences."""
    memory_retention_days: Optional[int] = Field(None, ge=1, le=365)
    max_memories_per_session: Optional[int] = Field(None, ge=1, le=100)
    memory_importance_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    auto_summarize_sessions: Optional[bool] = None
    allow_cross_session_memory: Optional[bool] = None
    allow_personal_data_storage: Optional[bool] = None

    # LLM preferences (per-user overrides)
    llm_provider: Optional[str] = Field(None, description="LLM provider override")
    llm_model: Optional[str] = Field(None, description="Model name override")
    llm_api_url: Optional[str] = Field(None, description="Custom API URL override")
    llm_api_key: Optional[str] = Field(None, description="User's own API key")
    llm_temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    llm_max_tokens: Optional[int] = Field(None, ge=1, le=32000)

    # Per-task model overrides
    llm_task_models: Optional[Dict[str, str]] = Field(None, description="Task-specific model overrides")

    # Paper algorithm agent defaults
    paper_algo_default_run_demo_check: Optional[bool] = None

class UserPreferencesResponse(UserPreferencesBase):
    """Schema for user preferences response."""
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class MemorySearchRequest(BaseModel):
    """Schema for memory search request."""
    query: str = Field(..., description="Search query")
    memory_types: Optional[List[str]] = Field(None, description="Filter by memory types")
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    min_importance: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum importance score")
    limit: int = Field(10, ge=1, le=50, description="Maximum number of results")

class MemorySearchResponse(BaseModel):
    """Schema for memory search response."""
    memories: List[MemoryResponse]
    total_count: int
    query: str

class MemorySummaryRequest(BaseModel):
    """Schema for memory summary request."""
    session_id: Optional[UUID] = Field(None, description="Session to summarize")
    time_range_days: Optional[int] = Field(7, ge=1, le=30, description="Time range in days")
    include_types: Optional[List[str]] = Field(None, description="Memory types to include")

class MemorySummaryResponse(BaseModel):
    """Schema for memory summary response."""
    summary: str
    key_facts: List[str]
    preferences: List[str]
    context_items: List[str]
    memory_count: int
    time_range: str

class MemoryStatsResponse(BaseModel):
    """Schema for memory statistics response."""
    total_memories: int
    memories_by_type: Dict[str, int]
    recent_memories: int  # Last 7 days
    most_accessed_memories: List[MemoryResponse]
    memory_usage_trend: List[Dict[str, Any]]  # Usage over time








