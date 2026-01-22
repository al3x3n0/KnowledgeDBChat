"""
Pydantic schemas for the agentic chat system.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field


class AgentToolCall(BaseModel):
    """Schema for a tool call made by the agent."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    tool_name: str = Field(..., description="Name of the tool to execute")
    tool_input: Dict[str, Any] = Field(default_factory=dict, description="Input parameters for the tool")
    tool_output: Optional[Any] = Field(None, description="Output from tool execution")
    status: str = Field("pending", description="Execution status: pending, running, completed, failed")
    error: Optional[str] = Field(None, description="Error message if execution failed")
    execution_time_ms: Optional[int] = Field(None, description="Execution time in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "abc123",
                "tool_name": "search_documents",
                "tool_input": {"query": "python tutorials", "limit": 5},
                "tool_output": [{"id": "doc1", "title": "Python Basics"}],
                "status": "completed",
                "execution_time_ms": 150
            }
        }


class AgentMessage(BaseModel):
    """Schema for agent chat messages."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: str = Field(..., description="Message role: user, assistant, system, tool")
    content: str = Field(..., description="Message content")
    tool_calls: Optional[List[AgentToolCall]] = Field(None, description="Tool calls made in this message")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "msg123",
                "role": "assistant",
                "content": "I found 3 documents matching your query.",
                "tool_calls": [],
                "created_at": "2024-01-15T10:30:00Z"
            }
        }


class AgentChatRequest(BaseModel):
    """Request schema for agent chat."""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    conversation_history: Optional[List[AgentMessage]] = Field(
        None,
        description="Previous messages in the conversation for context"
    )
    conversation_id: Optional[UUID] = Field(
        None,
        description="ID of the conversation for memory tracking"
    )
    turn_number: int = Field(
        0,
        description="Turn number in the conversation"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Find all documents about machine learning",
                "conversation_history": [],
                "conversation_id": None,
                "turn_number": 0
            }
        }


class AgentRoutingInfo(BaseModel):
    """Information about which agent handled the request."""
    agent_id: UUID = Field(..., description="ID of the selected agent")
    agent_name: str = Field(..., description="Internal name of the agent")
    agent_display_name: str = Field(..., description="Display name of the agent")
    routing_reason: str = Field(..., description="Why this agent was selected")
    handoff_from: Optional[str] = Field(
        None,
        description="Name of agent that handed off (if this was a handoff)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "550e8400-e29b-41d4-a716-446655440000",
                "agent_name": "document_expert",
                "agent_display_name": "Document Expert",
                "routing_reason": "Matched capabilities: document_search",
                "handoff_from": None
            }
        }


class AgentChatResponse(BaseModel):
    """Response schema for agent chat."""
    message: AgentMessage = Field(..., description="The assistant's response message")
    tool_results: Optional[List[AgentToolCall]] = Field(
        None,
        description="Results from tool executions"
    )
    requires_user_action: bool = Field(
        False,
        description="Whether the response requires user action (e.g., file upload)"
    )
    action_type: Optional[str] = Field(
        None,
        description="Type of action required: upload_file, confirm_delete, etc."
    )
    routing_info: Optional[AgentRoutingInfo] = Field(
        None,
        description="Information about which agent handled the request"
    )
    injected_memories: Optional[List[str]] = Field(
        None,
        description="IDs of memories that were injected into the agent context"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": {
                    "id": "msg456",
                    "role": "assistant",
                    "content": "I found 5 documents matching 'machine learning'.",
                    "created_at": "2024-01-15T10:30:00Z"
                },
                "tool_results": [
                    {
                        "id": "tool123",
                        "tool_name": "search_documents",
                        "tool_input": {"query": "machine learning"},
                        "status": "completed"
                    }
                ],
                "requires_user_action": False,
                "routing_info": {
                    "agent_id": "550e8400-e29b-41d4-a716-446655440000",
                    "agent_name": "qa_specialist",
                    "agent_display_name": "Q&A Specialist",
                    "routing_reason": "Matched capabilities: rag_qa"
                },
                "injected_memories": ["mem-uuid-1", "mem-uuid-2"]
            }
        }


class DocumentSearchResult(BaseModel):
    """Schema for document search results returned by tools."""
    id: str
    title: str
    content_preview: Optional[str] = None
    score: float
    source_type: Optional[str] = None
    created_at: Optional[datetime] = None


class DocumentDetails(BaseModel):
    """Schema for detailed document information."""
    id: str
    title: str
    content_preview: Optional[str] = None
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    author: Optional[str] = None
    tags: Optional[List[str]] = None
    summary: Optional[str] = None
    is_processed: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


# ============================================================================
# Agent Conversation Memory Schemas
# ============================================================================

class AgentConversationCreate(BaseModel):
    """Request to create a new agent conversation."""
    title: Optional[str] = Field(None, description="Optional title for the conversation")


class AgentConversationUpdate(BaseModel):
    """Request to update an agent conversation."""
    title: Optional[str] = None
    status: Optional[str] = Field(None, description="Conversation status: active, completed, archived")
    messages: Optional[List[Dict[str, Any]]] = Field(None, description="Messages to update")


class AgentConversationResponse(BaseModel):
    """Response schema for agent conversation."""
    id: UUID
    title: Optional[str] = None
    status: str
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    summary: Optional[str] = None
    message_count: int = 0
    tool_calls_count: int = 0
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime

    class Config:
        from_attributes = True


class AgentConversationListItem(BaseModel):
    """Summary item for conversation list."""
    id: UUID
    title: Optional[str] = None
    status: str
    message_count: int = 0
    tool_calls_count: int = 0
    summary: Optional[str] = None
    last_message_at: datetime
    created_at: datetime

    class Config:
        from_attributes = True


class AgentConversationListResponse(BaseModel):
    """Response for listing conversations."""
    conversations: List[AgentConversationListItem]
    total: int
    has_more: bool = False


class AgentMessageAppend(BaseModel):
    """Request to append a message to a conversation."""
    message: AgentMessage
    tool_calls: Optional[List[AgentToolCall]] = None


# ============================================================================
# Agent Definition Admin Schemas
# ============================================================================

class AgentDefinitionBase(BaseModel):
    """Base schema for agent definition."""
    name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        pattern=r'^[a-z][a-z0-9_]*$',
        description="Unique agent identifier (lowercase, underscores allowed)"
    )
    display_name: str = Field(
        ...,
        min_length=2,
        max_length=255,
        description="Human-readable agent name"
    )
    description: Optional[str] = Field(
        None,
        description="Brief description of the agent's purpose"
    )
    system_prompt: str = Field(
        ...,
        min_length=10,
        description="System prompt defining the agent's personality and instructions"
    )
    capabilities: List[str] = Field(
        default_factory=list,
        description="List of capabilities for routing (e.g., 'rag_qa', 'document_search')"
    )
    tool_whitelist: Optional[List[str]] = Field(
        None,
        description="List of allowed tools (null = all tools)"
    )
    priority: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Routing priority (higher = preferred when multiple agents match)"
    )
    is_active: bool = Field(
        default=True,
        description="Whether the agent is active and available for routing"
    )


class AgentDefinitionCreate(AgentDefinitionBase):
    """Request schema for creating an agent."""
    pass


class AgentDefinitionUpdate(BaseModel):
    """Request schema for updating an agent (all fields optional)."""
    display_name: Optional[str] = Field(None, min_length=2, max_length=255)
    description: Optional[str] = None
    system_prompt: Optional[str] = Field(None, min_length=10)
    capabilities: Optional[List[str]] = None
    tool_whitelist: Optional[List[str]] = None
    priority: Optional[int] = Field(None, ge=1, le=100)
    is_active: Optional[bool] = None


class AgentDefinitionResponse(AgentDefinitionBase):
    """Response schema for agent definition."""
    id: UUID
    is_system: bool = Field(description="Whether this is a built-in system agent")
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "name": "code_expert",
                "display_name": "Code Expert",
                "description": "Specialized in analyzing and explaining code",
                "system_prompt": "You are a Code Expert...",
                "capabilities": ["code_analysis", "rag_qa"],
                "tool_whitelist": ["read_document_content", "search_documents"],
                "priority": 65,
                "is_active": True,
                "is_system": False,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:00Z"
            }
        }


class AgentDefinitionListResponse(BaseModel):
    """Response schema for listing agents."""
    agents: List[AgentDefinitionResponse]
    total: int


class CapabilityInfo(BaseModel):
    """Information about an available capability."""
    name: str
    description: str
    keywords: List[str]


class CapabilitiesListResponse(BaseModel):
    """Response schema for listing available capabilities."""
    capabilities: List[CapabilityInfo]
