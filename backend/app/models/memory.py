"""
Memory models for conversation context retention.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from uuid import UUID, uuid4
from sqlalchemy import Column, String, Text, DateTime, JSON, ForeignKey, Integer, Float, Boolean
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

from app.core.database import Base

class ConversationMemory(Base):
    """Stores long-term conversation memories for users."""
    __tablename__ = "conversation_memories"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PostgresUUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    session_id = Column(PostgresUUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=True, index=True)
    
    # Memory content
    memory_type = Column(String(50), nullable=False, index=True)  # 'fact', 'preference', 'context', 'summary'
    content = Column(Text, nullable=False)
    importance_score = Column(Float, default=0.5)  # 0.0 to 1.0, higher = more important
    
    # Context and metadata
    context = Column(JSON, nullable=True)  # Additional context about the memory
    tags = Column(JSON, nullable=True)  # Tags for categorization
    source_message_id = Column(PostgresUUID(as_uuid=True), nullable=True)  # Original message that created this memory
    
    # Memory lifecycle
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_accessed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    access_count = Column(Integer, default=0, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="memories")
    session = relationship("ChatSession", back_populates="memories")

    def __repr__(self):
        return f"<ConversationMemory(id={self.id}, type={self.memory_type}, importance={self.importance_score})>"

class MemoryInteraction(Base):
    """Tracks how memories are used in conversations."""
    __tablename__ = "memory_interactions"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    memory_id = Column(PostgresUUID(as_uuid=True), ForeignKey("conversation_memories.id"), nullable=False)
    session_id = Column(PostgresUUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False)
    message_id = Column(PostgresUUID(as_uuid=True), nullable=True)
    
    # Interaction details
    interaction_type = Column(String(50), nullable=False)  # 'retrieved', 'updated', 'reinforced', 'contradicted'
    relevance_score = Column(Float, nullable=True)  # How relevant was this memory to the conversation
    usage_context = Column(JSON, nullable=True)  # Context about how the memory was used
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    memory = relationship("ConversationMemory")
    session = relationship("ChatSession")

    def __repr__(self):
        return f"<MemoryInteraction(id={self.id}, memory_id={self.memory_id}, type={self.interaction_type})>"

class UserPreferences(Base):
    """Stores user preferences and settings for memory system."""
    __tablename__ = "user_preferences"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PostgresUUID(as_uuid=True), ForeignKey("users.id"), nullable=False, unique=True, index=True)
    
    # Memory preferences
    memory_retention_days = Column(Integer, default=90, nullable=False)  # How long to keep memories
    max_memories_per_session = Column(Integer, default=10, nullable=False)  # Max memories to load per session
    memory_importance_threshold = Column(Float, default=0.3, nullable=False)  # Min importance to store
    auto_summarize_sessions = Column(Boolean, default=True, nullable=False)  # Auto-create session summaries
    
    # Privacy settings
    allow_cross_session_memory = Column(Boolean, default=True, nullable=False)  # Share memories across sessions
    allow_personal_data_storage = Column(Boolean, default=True, nullable=False)  # Store personal information
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="preferences")

    def __repr__(self):
        return f"<UserPreferences(user_id={self.user_id}, retention_days={self.memory_retention_days})>"




