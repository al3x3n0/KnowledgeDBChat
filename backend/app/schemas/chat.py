"""
Chat-related Pydantic schemas.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field


class ChatSessionCreate(BaseModel):
    """Schema for creating a chat session."""
    title: Optional[str] = None
    extra_metadata: Optional[Dict[str, Any]] = None


class ChatSessionBase(BaseModel):
    """Base chat session schema."""
    id: UUID
    title: Optional[str]
    is_active: bool
    extra_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime
    
    class Config:
        from_attributes = True


class ChatMessageBase(BaseModel):
    """Base chat message schema."""
    id: UUID
    content: str
    role: str
    message_type: str = "text"
    model_used: Optional[str]
    response_time: Optional[float]
    token_count: Optional[int]
    source_documents: Optional[List[Dict[str, Any]]]
    context_used: Optional[str]
    search_query: Optional[str]
    groundedness_score: Optional[float] = None
    retrieval_trace_id: Optional[UUID] = None
    user_rating: Optional[int]
    user_feedback: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class ChatMessageCreate(BaseModel):
    """Schema for creating a chat message."""
    content: str = Field(..., min_length=1, max_length=5000)


class ChatMessageResponse(ChatMessageBase):
    """Schema for chat message response."""
    pass


class ChatSessionResponse(ChatSessionBase):
    """Schema for chat session response with messages."""
    messages: Optional[List[ChatMessageResponse]] = []


class ChatSessionUpdate(BaseModel):
    """Schema for updating a chat session."""
    title: Optional[str] = None
    extra_metadata: Optional[Dict[str, Any]] = None


class ChatQuery(BaseModel):
    """Schema for chat query."""
    message: str = Field(..., min_length=1, max_length=5000)
    session_id: Optional[UUID] = None
    include_context: bool = True
    max_results: int = Field(default=5, ge=1, le=20)


class ChatFeedback(BaseModel):
    """Schema for chat feedback."""
    rating: int = Field(..., ge=1, le=5)
    feedback: Optional[str] = Field(None, max_length=1000)





