"""
Chat-related database models.
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.core.database import Base


class ChatSession(Base):
    """Chat session for organizing conversations."""
    
    __tablename__ = "chat_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(200), nullable=True)  # Auto-generated or user-defined
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Session metadata
    is_active = Column(Boolean, default=True)
    extra_metadata = Column(JSON, nullable=True)  # Additional session data
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    last_message_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    memories = relationship("ConversationMemory", back_populates="session", cascade="all, delete-orphan")
    memory_interactions = relationship("MemoryInteraction", back_populates="session", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ChatSession(id={self.id}, title='{self.title}', user_id={self.user_id})>"


class ChatMessage(Base):
    """Individual chat messages."""
    
    __tablename__ = "chat_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False)
    
    # Message content
    content = Column(Text, nullable=False)
    role = Column(String(20), nullable=False)  # user, assistant, system
    message_type = Column(String(50), default="text")  # text, error, system
    
    # Response metadata (for assistant messages)
    model_used = Column(String(100), nullable=True)
    response_time = Column(Float, nullable=True)  # Response time in seconds
    token_count = Column(Integer, nullable=True)  # Token count for the response
    
    # Source references (for assistant messages)
    source_documents = Column(JSON, nullable=True)  # List of referenced document IDs
    context_used = Column(Text, nullable=True)  # Context that was provided to LLM
    search_query = Column(String(500), nullable=True)  # Query used for document retrieval
    
    # Processing status
    is_processed = Column(Boolean, default=True)
    processing_error = Column(Text, nullable=True)
    
    # Feedback
    user_rating = Column(Integer, nullable=True)  # 1-5 rating
    user_feedback = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")
    
    def __repr__(self):
        return f"<ChatMessage(id={self.id}, role='{self.role}', session_id={self.session_id})>"
