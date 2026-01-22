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
    agent_injections = relationship("AgentMemoryInjection", back_populates="memory", cascade="all, delete-orphan")

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

    # LLM preferences (per-user overrides)
    llm_provider = Column(String(50), nullable=True)  # "ollama", "deepseek", "openai", or custom
    llm_model = Column(String(100), nullable=True)  # Model name override (default for chat)
    llm_api_url = Column(String(500), nullable=True)  # Custom API URL override
    llm_api_key = Column(String(500), nullable=True)  # User's own API key (for external providers)
    llm_temperature = Column(Float, nullable=True)  # Temperature override (0.0-2.0)
    llm_max_tokens = Column(Integer, nullable=True)  # Max response tokens override

    # Per-task model overrides (JSON: {"title_generation": "model", "summarization": "model", ...})
    # Supported tasks: title_generation, summarization, query_expansion, memory_extraction
    llm_task_models = Column(JSON, nullable=True)

    # Agent memory integration settings
    enable_agent_memory = Column(Boolean, default=True, nullable=False)  # Inject memories into agent prompts
    memory_injection_types = Column(JSON, default=lambda: ["fact", "preference", "context"], nullable=False)  # Types to inject
    max_injected_memories = Column(Integer, default=5, nullable=False)  # Max memories per turn

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="preferences")

    def __repr__(self):
        return f"<UserPreferences(user_id={self.user_id}, retention_days={self.memory_retention_days})>"


class AgentConversation(Base):
    """Stores agent chat conversation sessions for persistence across page reloads."""
    __tablename__ = "agent_conversations"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PostgresUUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)

    # Conversation metadata
    title = Column(String(255), nullable=True)  # Auto-generated title
    status = Column(String(50), default="active", nullable=False)  # 'active', 'completed', 'archived'

    # Messages stored as JSON array for simplicity
    # Each message: {id, role, content, tool_calls, created_at}
    messages = Column(JSON, default=list, nullable=False)

    # Summary for quick context loading
    summary = Column(Text, nullable=True)

    # Stats
    message_count = Column(Integer, default=0, nullable=False)
    tool_calls_count = Column(Integer, default=0, nullable=False)

    # Multi-agent tracking
    active_agent_id = Column(PostgresUUID(as_uuid=True), ForeignKey("agent_definitions.id", ondelete="SET NULL"), nullable=True)
    agent_handoffs = Column(Integer, default=0, nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_message_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="agent_conversations")
    tool_executions = relationship("AgentToolExecution", back_populates="conversation", cascade="all, delete-orphan")
    active_agent = relationship("AgentDefinition", foreign_keys=[active_agent_id])
    agent_contexts = relationship("AgentConversationContext", back_populates="conversation", cascade="all, delete-orphan")
    memory_injections = relationship("AgentMemoryInjection", back_populates="conversation", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<AgentConversation(id={self.id}, user_id={self.user_id}, messages={self.message_count})>"


class AgentToolExecution(Base):
    """Tracks individual tool executions within agent conversations."""
    __tablename__ = "agent_tool_executions"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id = Column(PostgresUUID(as_uuid=True), ForeignKey("agent_conversations.id"), nullable=False, index=True)

    # Tool details
    tool_name = Column(String(100), nullable=False, index=True)
    tool_input = Column(JSON, nullable=True)
    tool_output = Column(JSON, nullable=True)

    # Execution metadata
    status = Column(String(50), nullable=False)  # 'completed', 'failed'
    error = Column(Text, nullable=True)
    execution_time_ms = Column(Integer, nullable=True)

    # Context
    message_id = Column(String(100), nullable=True)  # ID of the agent message this belonged to

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    conversation = relationship("AgentConversation", back_populates="tool_executions")

    def __repr__(self):
        return f"<AgentToolExecution(id={self.id}, tool={self.tool_name}, status={self.status})>"









