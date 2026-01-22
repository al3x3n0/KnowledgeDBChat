"""
Agent definition and multi-agent tracking models.

Supports:
- Specialized agent configurations
- Agent conversation context tracking
- Memory injection tracking
"""

from datetime import datetime
from typing import Optional, List
from uuid import uuid4

from sqlalchemy import (
    Column, String, Text, Boolean, Integer, DateTime, Float, ForeignKey, Index
)
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class AgentDefinition(Base):
    """
    Specialized agent configuration.

    Defines agent personality, capabilities, and tool access.
    """
    __tablename__ = "agent_definitions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    name = Column(String(100), unique=True, nullable=False, index=True)
    display_name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    system_prompt = Column(Text, nullable=False)

    # Capabilities this agent has (for routing)
    # e.g., ["document_search", "rag_qa", "workflow_exec"]
    capabilities = Column(JSON, nullable=False, default=list)

    # Tools this agent can use (null = all tools)
    # e.g., ["search_documents", "answer_question"]
    tool_whitelist = Column(JSON, nullable=True)

    # Routing priority (higher = preferred when multiple agents match)
    priority = Column(Integer, default=50, nullable=False)

    # State
    is_active = Column(Boolean, default=True, nullable=False)
    is_system = Column(Boolean, default=True, nullable=False)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    conversation_contexts = relationship("AgentConversationContext", back_populates="agent_definition")

    __table_args__ = (
        Index("ix_agent_definitions_is_active", "is_active"),
    )

    def has_tool(self, tool_name: str) -> bool:
        """Check if this agent can use a specific tool."""
        if self.tool_whitelist is None:
            return True  # All tools allowed
        return tool_name in self.tool_whitelist

    def has_capability(self, capability: str) -> bool:
        """Check if this agent has a specific capability."""
        return capability in (self.capabilities or [])


class AgentConversationContext(Base):
    """
    Tracks which agents participated in each turn of a conversation.

    Enables:
    - Viewing conversation history with agent attribution
    - Analyzing handoff patterns
    - Debugging agent routing decisions
    """
    __tablename__ = "agent_conversation_contexts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("agent_conversations.id", ondelete="CASCADE"), nullable=False)
    agent_definition_id = Column(UUID(as_uuid=True), ForeignKey("agent_definitions.id", ondelete="SET NULL"), nullable=True)

    # Which turn in the conversation (0-indexed)
    turn_number = Column(Integer, nullable=False)

    # Why this agent was selected
    routing_reason = Column(Text, nullable=True)

    # Context passed during handoff (if any)
    handoff_context = Column(JSON, nullable=True)

    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    agent_definition = relationship("AgentDefinition", back_populates="conversation_contexts")
    conversation = relationship("AgentConversation", back_populates="agent_contexts")

    __table_args__ = (
        Index("ix_agent_conversation_contexts_conversation_id", "conversation_id"),
    )


class AgentMemoryInjection(Base):
    """
    Tracks which memories were injected into which conversation turns.

    Enables:
    - Auditing memory usage
    - Improving memory relevance scoring
    - User transparency about what context was used
    """
    __tablename__ = "agent_memory_injections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("agent_conversations.id", ondelete="CASCADE"), nullable=False)
    memory_id = Column(UUID(as_uuid=True), ForeignKey("conversation_memories.id", ondelete="CASCADE"), nullable=False)

    # Which turn in the conversation
    turn_number = Column(Integer, nullable=False)

    # How relevant this memory was considered
    relevance_score = Column(Float, nullable=True)

    # How the memory was injected
    # "automatic" = system selected based on relevance
    # "manual" = user explicitly injected
    # "shared" = passed from another agent during handoff
    injection_type = Column(String(50), nullable=False, default="automatic")

    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    conversation = relationship("AgentConversation", back_populates="memory_injections")
    memory = relationship("ConversationMemory", back_populates="agent_injections")

    __table_args__ = (
        Index("ix_agent_memory_injections_conversation_id", "conversation_id"),
        Index("ix_agent_memory_injections_memory_id", "memory_id"),
    )
