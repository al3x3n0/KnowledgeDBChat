"""
Agent chat API endpoints.

Provides agentic chat functionality with tool calling for document operations.
Supports both REST and WebSocket interfaces.
"""

import json
import asyncio
from typing import List, Optional, Callable, Any
from datetime import datetime
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func, or_
from loguru import logger

from app.core.database import get_db, AsyncSessionLocal
from app.models.user import User
from app.models.memory import UserPreferences, AgentConversation, AgentToolExecution
from app.services.auth_service import get_current_user
from app.services.agent_service import AgentService
from app.services.agent_tools import AGENT_TOOLS
from app.services.llm_service import UserLLMSettings
from app.services.llm_routing import (
    coerce_routing_config,
    compute_attempt_tiers,
    resolve_effective_provider_model,
    resolve_feature_default_model,
    resolve_tier_overrides,
)
from app.schemas.agent import (
    AgentChatRequest,
    AgentChatResponse,
    AgentMessage,
    AgentToolCall,
    AgentConversationCreate,
    AgentConversationUpdate,
    AgentConversationResponse,
    AgentConversationListItem,
    AgentConversationListResponse,
    AgentMessageAppend,
)


router = APIRouter()
agent_service = AgentService()


@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(
    request: AgentChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Agentic chat endpoint that processes user messages
    and executes document operations automatically.

    Multi-agent architecture routes requests to specialized agents:
    - Document Expert: Search, CRUD, tag management
    - Q&A Specialist: RAG-based Q&A, summarization
    - Workflow Assistant: Automation, templates, diagrams
    - Generalist: Fallback for general queries

    Memory integration injects relevant user context from previous
    conversations into agent prompts.

    Args:
        request: Chat request with message and optional conversation history
        current_user: Authenticated user
        db: Database session

    Returns:
        AgentChatResponse with assistant message, tool results, routing info
    """
    try:
        # Load user LLM preferences
        user_settings = None
        try:
            prefs_result = await db.execute(
                select(UserPreferences).where(UserPreferences.user_id == current_user.id)
            )
            user_prefs = prefs_result.scalar_one_or_none()
            if user_prefs:
                user_settings = UserLLMSettings.from_preferences(user_prefs)
        except Exception as e:
            logger.warning(f"Failed to load user preferences: {e}")

        response = await agent_service.process_message(
            message=request.message,
            conversation_history=request.conversation_history,
            user_id=current_user.id,
            db=db,
            user_settings=user_settings,
            conversation_id=request.conversation_id,
            turn_number=request.turn_number,
            agent_id=request.agent_id,
        )

        logger.info(
            f"Agent processed message for user {current_user.id}: "
            f"agent={response.routing_info.agent_name if response.routing_info else 'unknown'}, "
            f"tools_called={len(response.tool_results or [])}, "
            f"memories_injected={len(response.injected_memories or [])}, "
            f"requires_action={response.requires_user_action}"
        )

        return response

    except Exception as e:
        logger.error(f"Error in agent chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")


@router.get("/tools")
async def list_available_tools(
    current_user: User = Depends(get_current_user)
):
    """
    List all available tools that the agent can use.

    Returns:
        List of tool definitions with their parameters
    """
    return {
        "tools": [
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
            for tool in AGENT_TOOLS
        ]
    }


@router.post("/confirm-delete/{document_id}")
async def confirm_document_deletion(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Confirm and execute document deletion.

    This endpoint is called when the user confirms deletion
    after the agent requests confirmation.

    Args:
        document_id: UUID of the document to delete
        current_user: Authenticated user
        db: Database session

    Returns:
        Deletion result
    """
    try:
        result = await agent_service._tool_delete_document(
            params={"document_id": document_id, "confirm": True},
            db=db
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error confirming document deletion: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")


# ============================================================================
# Agent Conversation Memory Endpoints
# ============================================================================

@router.get("/conversations", response_model=AgentConversationListResponse)
async def list_agent_conversations(
    status: Optional[str] = Query(None, description="Filter by status: active, completed, archived"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List agent conversations for the current user.
    Returns conversations sorted by last_message_at (most recent first).
    """
    try:
        query = select(AgentConversation).where(
            AgentConversation.user_id == current_user.id
        )

        if status:
            query = query.where(AgentConversation.status == status)

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        # Get paginated results
        query = query.order_by(desc(AgentConversation.last_message_at))
        query = query.offset(offset).limit(limit)

        result = await db.execute(query)
        conversations = result.scalars().all()

        return AgentConversationListResponse(
            conversations=[
                AgentConversationListItem(
                    id=conv.id,
                    title=conv.title,
                    status=conv.status,
                    message_count=conv.message_count,
                    tool_calls_count=conv.tool_calls_count,
                    summary=conv.summary,
                    last_message_at=conv.last_message_at,
                    created_at=conv.created_at
                )
                for conv in conversations
            ],
            total=total,
            has_more=offset + limit < total
        )

    except Exception as e:
        logger.error(f"Error listing agent conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations", response_model=AgentConversationResponse)
async def create_agent_conversation(
    request: AgentConversationCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new agent conversation.
    """
    try:
        conversation = AgentConversation(
            user_id=current_user.id,
            title=request.title,
            status="active",
            messages=[],
            message_count=0,
            tool_calls_count=0
        )
        db.add(conversation)
        await db.commit()
        await db.refresh(conversation)

        logger.info(f"Created agent conversation {conversation.id} for user {current_user.id}")

        return AgentConversationResponse(
            id=conversation.id,
            title=conversation.title,
            status=conversation.status,
            messages=conversation.messages or [],
            summary=conversation.summary,
            message_count=conversation.message_count,
            tool_calls_count=conversation.tool_calls_count,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            last_message_at=conversation.last_message_at
        )

    except Exception as e:
        logger.error(f"Error creating agent conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/active", response_model=Optional[AgentConversationResponse])
async def get_active_conversation(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get the most recent active conversation for the current user.
    Creates a new one if none exists.
    """
    try:
        # Find most recent active conversation
        query = select(AgentConversation).where(
            AgentConversation.user_id == current_user.id,
            AgentConversation.status == "active"
        ).order_by(desc(AgentConversation.last_message_at)).limit(1)

        result = await db.execute(query)
        conversation = result.scalar_one_or_none()

        if not conversation:
            # Create a new conversation
            conversation = AgentConversation(
                user_id=current_user.id,
                status="active",
                messages=[],
                message_count=0,
                tool_calls_count=0
            )
            db.add(conversation)
            await db.commit()
            await db.refresh(conversation)
            logger.info(f"Created new active conversation {conversation.id} for user {current_user.id}")

        return AgentConversationResponse(
            id=conversation.id,
            title=conversation.title,
            status=conversation.status,
            messages=conversation.messages or [],
            summary=conversation.summary,
            message_count=conversation.message_count,
            tool_calls_count=conversation.tool_calls_count,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            last_message_at=conversation.last_message_at
        )

    except Exception as e:
        logger.error(f"Error getting active conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}", response_model=AgentConversationResponse)
async def get_agent_conversation(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific agent conversation by ID.
    """
    try:
        query = select(AgentConversation).where(
            AgentConversation.id == conversation_id,
            AgentConversation.user_id == current_user.id
        )
        result = await db.execute(query)
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return AgentConversationResponse(
            id=conversation.id,
            title=conversation.title,
            status=conversation.status,
            messages=conversation.messages or [],
            summary=conversation.summary,
            message_count=conversation.message_count,
            tool_calls_count=conversation.tool_calls_count,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            last_message_at=conversation.last_message_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/conversations/{conversation_id}", response_model=AgentConversationResponse)
async def update_agent_conversation(
    conversation_id: UUID,
    request: AgentConversationUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update an agent conversation (title, status, or messages).
    """
    try:
        query = select(AgentConversation).where(
            AgentConversation.id == conversation_id,
            AgentConversation.user_id == current_user.id
        )
        result = await db.execute(query)
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        if request.title is not None:
            conversation.title = request.title
        if request.status is not None:
            conversation.status = request.status
        if request.messages is not None:
            conversation.messages = request.messages
            conversation.message_count = len(request.messages)
            # Count tool calls
            tool_count = sum(
                len(msg.get("tool_calls", []) or [])
                for msg in request.messages
                if msg.get("role") == "assistant"
            )
            conversation.tool_calls_count = tool_count

        conversation.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(conversation)

        return AgentConversationResponse(
            id=conversation.id,
            title=conversation.title,
            status=conversation.status,
            messages=conversation.messages or [],
            summary=conversation.summary,
            message_count=conversation.message_count,
            tool_calls_count=conversation.tool_calls_count,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            last_message_at=conversation.last_message_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating agent conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations/{conversation_id}/messages")
async def append_message_to_conversation(
    conversation_id: UUID,
    request: AgentMessageAppend,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Append a message to an existing conversation.
    """
    try:
        query = select(AgentConversation).where(
            AgentConversation.id == conversation_id,
            AgentConversation.user_id == current_user.id
        )
        result = await db.execute(query)
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Add message to list
        messages = conversation.messages or []
        message_dict = {
            "id": request.message.id,
            "role": request.message.role,
            "content": request.message.content,
            "created_at": request.message.created_at.isoformat(),
        }
        if request.tool_calls:
            message_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "tool_name": tc.tool_name,
                    "tool_input": tc.tool_input,
                    "tool_output": tc.tool_output,
                    "status": tc.status,
                    "error": tc.error,
                    "execution_time_ms": tc.execution_time_ms
                }
                for tc in request.tool_calls
            ]

        messages.append(message_dict)
        conversation.messages = messages
        conversation.message_count = len(messages)

        # Update tool count
        if request.tool_calls:
            conversation.tool_calls_count += len(request.tool_calls)

            # Log tool executions
            for tc in request.tool_calls:
                execution = AgentToolExecution(
                    conversation_id=conversation.id,
                    tool_name=tc.tool_name,
                    tool_input=tc.tool_input,
                    tool_output=tc.tool_output,
                    status=tc.status,
                    error=tc.error,
                    execution_time_ms=tc.execution_time_ms,
                    message_id=request.message.id
                )
                db.add(execution)

        conversation.last_message_at = datetime.utcnow()
        conversation.updated_at = datetime.utcnow()

        await db.commit()

        return {"status": "ok", "message_count": conversation.message_count}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error appending message to conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversations/{conversation_id}")
async def delete_agent_conversation(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete an agent conversation.
    """
    try:
        query = select(AgentConversation).where(
            AgentConversation.id == conversation_id,
            AgentConversation.user_id == current_user.id
        )
        result = await db.execute(query)
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        await db.delete(conversation)
        await db.commit()

        logger.info(f"Deleted agent conversation {conversation_id} for user {current_user.id}")

        return {"status": "ok", "deleted": str(conversation_id)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting agent conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations/{conversation_id}/archive")
async def archive_agent_conversation(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Archive an agent conversation.
    """
    try:
        query = select(AgentConversation).where(
            AgentConversation.id == conversation_id,
            AgentConversation.user_id == current_user.id
        )
        result = await db.execute(query)
        conversation = result.scalar_one_or_none()

        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        conversation.status = "archived"
        conversation.updated_at = datetime.utcnow()
        await db.commit()

        return {"status": "ok", "conversation_id": str(conversation_id)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error archiving agent conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations/new")
async def start_new_conversation(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Archive the current active conversation and start a new one.
    """
    try:
        # Archive current active conversations
        query = select(AgentConversation).where(
            AgentConversation.user_id == current_user.id,
            AgentConversation.status == "active"
        )
        result = await db.execute(query)
        active_conversations = result.scalars().all()

        for conv in active_conversations:
            conv.status = "archived"
            conv.updated_at = datetime.utcnow()

        # Create new conversation
        new_conversation = AgentConversation(
            user_id=current_user.id,
            status="active",
            messages=[],
            message_count=0,
            tool_calls_count=0
        )
        db.add(new_conversation)
        await db.commit()
        await db.refresh(new_conversation)

        logger.info(f"Started new conversation {new_conversation.id} for user {current_user.id}")

        return AgentConversationResponse(
            id=new_conversation.id,
            title=new_conversation.title,
            status=new_conversation.status,
            messages=[],
            summary=None,
            message_count=0,
            tool_calls_count=0,
            created_at=new_conversation.created_at,
            updated_at=new_conversation.updated_at,
            last_message_at=new_conversation.last_message_at
        )

    except Exception as e:
        logger.error(f"Error starting new conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Agent Definition Endpoints
# ============================================================================

from app.models.agent_definition import AgentDefinition, AgentConversationContext, AgentMemoryInjection
from app.models.memory import ConversationMemory


@router.get("/agents")
async def list_agents(
    active_only: bool = Query(True, description="Only return active agents"),
    search: Optional[str] = Query(None, description="Search by name or display name"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List available specialized agents.
    """
    try:
        query = select(AgentDefinition)
        if active_only:
            query = query.where(AgentDefinition.is_active == True)
        if search:
            search_term = f"%{search.strip()}%"
            query = query.where(
                or_(
                    AgentDefinition.name.ilike(search_term),
                    AgentDefinition.display_name.ilike(search_term),
                )
            )
        query = query.order_by(desc(AgentDefinition.priority))

        result = await db.execute(query)
        agents = result.scalars().all()

        return {
            "agents": [
                {
                    "id": str(agent.id),
                    "name": agent.name,
                    "display_name": agent.display_name,
                    "description": agent.description,
                    "capabilities": agent.capabilities,
                    "priority": agent.priority,
                    "is_active": agent.is_active,
                    "is_system": agent.is_system,
                    "owner_user_id": str(agent.owner_user_id) if getattr(agent, "owner_user_id", None) else None,
                    "version": getattr(agent, "version", 1),
                    "lifecycle_status": getattr(agent, "lifecycle_status", "published"),
            "routing_defaults": getattr(agent, "routing_defaults", None),
                    "created_at": agent.created_at.isoformat() if agent.created_at else None,
                    "updated_at": agent.updated_at.isoformat() if agent.updated_at else None,
                }
                for agent in agents
            ],
            "total": len(agents)
        }

    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}")
async def get_agent(
    agent_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get details of a specific agent.
    """
    try:
        result = await db.execute(
            select(AgentDefinition).where(AgentDefinition.id == agent_id)
        )
        agent = result.scalar_one_or_none()

        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        return {
            "id": str(agent.id),
            "name": agent.name,
            "display_name": agent.display_name,
            "description": agent.description,
            "system_prompt": agent.system_prompt if not agent.is_system else None,  # Hide system prompts for built-in agents
            "capabilities": agent.capabilities,
            "tool_whitelist": agent.tool_whitelist,
            "priority": agent.priority,
            "is_active": agent.is_active,
            "is_system": agent.is_system,
            "owner_user_id": str(agent.owner_user_id) if getattr(agent, "owner_user_id", None) else None,
            "version": getattr(agent, "version", 1),
            "lifecycle_status": getattr(agent, "lifecycle_status", "published"),
            "created_at": agent.created_at.isoformat() if agent.created_at else None,
            "updated_at": agent.updated_at.isoformat() if agent.updated_at else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


from app.schemas.agent import (
    AgentDefinitionCreate,
    AgentDefinitionUpdate,
    AgentDefinitionResponse,
    AgentDefinitionListResponse,
    CapabilitiesListResponse,
    CapabilityInfo,
    AgentRoutingPreviewRequest,
    AgentRoutingPreviewResponse,
)
from app.services.agent_router import CAPABILITY_KEYWORDS


def require_admin(user: User) -> User:
    """Check if user is admin."""
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def _validate_agent_definition_fields(
    *,
    capabilities: Optional[List[str]] = None,
    tool_whitelist: Optional[List[str]] = None,
) -> None:
    valid_capabilities = set(CAPABILITY_KEYWORDS.keys()) | {"general"}

    if capabilities is not None:
        unknown_caps = sorted({c for c in capabilities if c not in valid_capabilities})
        if unknown_caps:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown capabilities: {', '.join(unknown_caps)}",
            )

    if tool_whitelist is not None:
        valid_tool_names = {t.get("name") for t in AGENT_TOOLS if t.get("name")}
        unknown_tools = sorted({t for t in tool_whitelist if t not in valid_tool_names})
        if unknown_tools:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown tools in whitelist: {', '.join(unknown_tools)}",
            )




@router.post("/agents/{agent_id}/routing-preview", response_model=AgentRoutingPreviewResponse)
async def preview_agent_routing(
    agent_id: UUID,
    request: AgentRoutingPreviewRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Preview effective LLM routing for this agent for the current user.

    Combines:
      - agent.routing_defaults
      - optional request.agent_routing_overrides
      - optional job config (job_id or job_config_overrides)
      - user preferences (custom api_url/task models)
      - feature-flag tier resolution (provider/model per tier)
    """
    from app.core.feature_flags import get_str as get_feature_str
    from app.models.agent_job import AgentJob

    res = await db.execute(select(AgentDefinition).where(AgentDefinition.id == agent_id))
    agent = res.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    task_type = str(getattr(request, "task_type", None) or "chat").strip() or "chat"

    # Load user preferences
    prefs_result = await db.execute(select(UserPreferences).where(UserPreferences.user_id == current_user.id))
    prefs = prefs_result.scalar_one_or_none()
    user_settings = UserLLMSettings.from_preferences(prefs) if prefs else UserLLMSettings()

    def _normalize_routing(obj: Any) -> dict | None:
        if not isinstance(obj, dict):
            return None
        tier = str(obj.get("tier") or obj.get("llm_tier") or "").strip().lower() or None
        fb = obj.get("fallback_tiers") or obj.get("llm_fallback_tiers")
        if not isinstance(fb, list):
            fb = []
        fallback_tiers = [str(x).strip().lower() for x in fb if str(x).strip()]
        out: dict[str, Any] = {"tier": tier, "fallback_tiers": fallback_tiers}
        for k_in, k_out in [
            ("timeout_seconds", "timeout_seconds"),
            ("llm_timeout_seconds", "timeout_seconds"),
            ("max_tokens_cap", "max_tokens_cap"),
            ("llm_max_tokens_cap", "max_tokens_cap"),
            ("cooldown_seconds", "cooldown_seconds"),
            ("llm_unhealthy_cooldown_seconds", "cooldown_seconds"),
        ]:
            if obj.get(k_in) is not None:
                out[k_out] = obj.get(k_in)
        if out.get("tier") is None and not out.get("fallback_tiers") and all(out.get(k) is None for k in ("timeout_seconds", "max_tokens_cap", "cooldown_seconds")):
            return None
        return out

    routing_agent = _normalize_routing(getattr(agent, "routing_defaults", None))

    overrides = _normalize_routing(request.agent_routing_overrides)
    if overrides:
        routing_agent = dict(routing_agent or {})
        # overrides win
        for k, v in overrides.items():
            if v is None and k in routing_agent:
                continue
            if k == "fallback_tiers" and isinstance(v, list):
                routing_agent[k] = v
            elif v is not None:
                routing_agent[k] = v

    job_cfg: dict[str, Any] | None = None
    if request.job_id:
        j = await db.get(AgentJob, request.job_id)
        if j and j.user_id == current_user.id:
            job_cfg = j.config if isinstance(j.config, dict) else None

    if request.job_config_overrides and isinstance(request.job_config_overrides, dict):
        job_cfg = dict(job_cfg or {})
        job_cfg.update(request.job_config_overrides)

    routing_job = None
    if isinstance(job_cfg, dict):
        routing_job = _normalize_routing(job_cfg)

    # Merge job > agent
    routing_effective = dict(routing_agent or {}) if routing_agent else {}
    if routing_job:
        for k, v in routing_job.items():
            if v is None:
                continue
            routing_effective[k] = v

    if not routing_effective:
        routing_effective = None

    # Tier resolution from feature flags
    tier_resolution: dict[str, Any] = {}
    for t in ("fast", "balanced", "deep"):
        p, m = await resolve_tier_overrides(get_feature_str, t)
        tier_resolution[t] = {"provider": p, "model": m}

    # Determine attempt order (same as runtime)
    routing_effective_cfg = coerce_routing_config(routing_effective)
    attempt_tiers = compute_attempt_tiers(
        tier=routing_effective_cfg.get("tier"),
        fallback_tiers=routing_effective_cfg.get("fallback_tiers") or [],
    )

    # User overrides summary
    user_llm = {
        "provider": user_settings.get_provider_for_task(task_type),
        "model": user_settings.get_model_for_task(task_type),
        "api_url": user_settings.api_url,
        "has_custom": user_settings.has_custom_settings(),
    }

    notes: list[str] = []
    if user_settings.api_url:
        notes.append("User has custom api_url configured; runtime will route via custom OpenAI-compatible endpoint.")
    if any(t in {"fast", "balanced", "deep"} for t in attempt_tiers if t):
        notes.append("Tier provider/model are resolved from feature flags (admin-configured).")

    feature_default_model = await resolve_feature_default_model(get_feature_str)

    # Compute effective provider/model per attempt exactly like runtime.
    system_provider = (getattr(settings, "LLM_PROVIDER", None) or "ollama")
    system_default_model = getattr(settings, "DEFAULT_MODEL", None) or ""

    attempts: list[dict[str, Any]] = []
    for i, t in enumerate(attempt_tiers):
        tier_provider, tier_model = await resolve_tier_overrides(get_feature_str, t)

        # Matches generate_response: attempt_provider/model are per-call overrides.
        attempt_provider = tier_provider
        attempt_model = tier_model

        eff = resolve_effective_provider_model(
            system_provider=str(system_provider).strip().lower(),
            system_default_model=str(system_default_model),
            feature_default_model=feature_default_model,
            user_settings=user_settings,
            task_type=task_type,
            model=attempt_model,
            provider_override=attempt_provider,
            api_url_override=None,
            api_key_override=None,
            prefer_deepseek=False,
        )

        attempts.append(
            {
                "attempt": i + 1,
                "tier": (str(t).strip().lower() if t else None),
                "tier_provider": tier_provider,
                "tier_model": tier_model,
                "effective_provider": eff.get("provider_used"),
                "effective_model": eff.get("effective_model"),
            }
        )

    return AgentRoutingPreviewResponse(
        agent_id=agent.id,
        task_type=task_type,
        user_llm=user_llm,
        routing_agent=routing_agent,
        routing_job=routing_job,
        routing_effective=routing_effective,
        tier_resolution=tier_resolution,
        attempts=attempts,
        notes=notes,
    )
@router.post("/agents", response_model=AgentDefinitionResponse)
async def create_agent(
    request: AgentDefinitionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new agent definition (admin only).
    """
    require_admin(current_user)

    try:
        _validate_agent_definition_fields(
            capabilities=request.capabilities,
            tool_whitelist=request.tool_whitelist,
            routing_defaults=getattr(request, "routing_defaults", None),
        )

        # Check if name already exists
        existing = await db.execute(
            select(AgentDefinition).where(AgentDefinition.name == request.name)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=400, detail=f"Agent with name '{request.name}' already exists")

        # Create new agent
        agent = AgentDefinition(
            name=request.name,
            display_name=request.display_name,
            description=request.description,
            system_prompt=request.system_prompt,
            capabilities=request.capabilities,
            tool_whitelist=request.tool_whitelist,
            priority=request.priority,
            is_active=request.is_active,
            is_system=False,  # User-created agents are never system agents
            owner_user_id=current_user.id,
            version=1,
            lifecycle_status="draft",
        )
        db.add(agent)
        await db.commit()
        await db.refresh(agent)

        logger.info(f"Created agent '{agent.name}' by user {current_user.id}")

        return AgentDefinitionResponse(
            id=agent.id,
            name=agent.name,
            display_name=agent.display_name,
            description=agent.description,
            system_prompt=agent.system_prompt,
            capabilities=agent.capabilities or [],
            tool_whitelist=agent.tool_whitelist,
            routing_defaults=getattr(agent, "routing_defaults", None),
            priority=agent.priority,
            is_active=agent.is_active,
            lifecycle_status=getattr(agent, "lifecycle_status", "draft"),
            is_system=agent.is_system,
            owner_user_id=getattr(agent, "owner_user_id", None),
            version=getattr(agent, "version", 1),
            created_at=agent.created_at,
            updated_at=agent.updated_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating agent: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/agents/{agent_id}", response_model=AgentDefinitionResponse)
async def update_agent(
    agent_id: UUID,
    request: AgentDefinitionUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update an agent definition (admin only).
    System agents can only have is_active and priority modified.
    """
    require_admin(current_user)

    try:
        result = await db.execute(
            select(AgentDefinition).where(AgentDefinition.id == agent_id)
        )
        agent = result.scalar_one_or_none()

        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # System agents have limited editability
        if agent.is_system:
            # Only allow toggling active and priority for system agents
            if request.is_active is not None:
                agent.is_active = request.is_active
            if request.priority is not None:
                agent.priority = request.priority
            # Reject other changes for system agents
            if any([
                request.display_name is not None,
                request.description is not None,
                request.system_prompt is not None,
                request.capabilities is not None,
                request.tool_whitelist is not None
            ]):
                raise HTTPException(
                    status_code=400,
                    detail="System agents can only have 'is_active' and 'priority' modified"
                )
        else:
            # Non-system agents can be fully edited
            _validate_agent_definition_fields(
                capabilities=request.capabilities,
                tool_whitelist=request.tool_whitelist,
            )
            changed = False
            if request.display_name is not None:
                agent.display_name = request.display_name
                changed = True
            if request.description is not None:
                agent.description = request.description
                changed = True
            if request.system_prompt is not None:
                agent.system_prompt = request.system_prompt
                changed = True
            if request.capabilities is not None:
                agent.capabilities = request.capabilities
                changed = True
            if request.tool_whitelist is not None:
                agent.tool_whitelist = request.tool_whitelist
            if request.routing_defaults is not None:
                agent.routing_defaults = request.routing_defaults
                changed = True
            if request.priority is not None:
                agent.priority = request.priority
                changed = True
            if request.is_active is not None:
                agent.is_active = request.is_active
                changed = True
            if request.lifecycle_status is not None:
                agent.lifecycle_status = request.lifecycle_status
                changed = True

            if changed:
                agent.version = (agent.version or 1) + 1

        await db.commit()
        await db.refresh(agent)

        logger.info(f"Updated agent '{agent.name}' by user {current_user.id}")

        return AgentDefinitionResponse(
            id=agent.id,
            name=agent.name,
            display_name=agent.display_name,
            description=agent.description,
            system_prompt=agent.system_prompt,
            capabilities=agent.capabilities or [],
            tool_whitelist=agent.tool_whitelist,
            priority=agent.priority,
            is_active=agent.is_active,
            lifecycle_status=getattr(agent, "lifecycle_status", "published"),
            routing_defaults=getattr(agent, "routing_defaults", None),
            is_system=agent.is_system,
            owner_user_id=getattr(agent, "owner_user_id", None),
            version=getattr(agent, "version", 1),
            created_at=agent.created_at,
            updated_at=agent.updated_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating agent: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/agents/{agent_id}")
async def delete_agent(
    agent_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete an agent definition (admin only).
    System agents cannot be deleted.
    """
    require_admin(current_user)

    try:
        result = await db.execute(
            select(AgentDefinition).where(AgentDefinition.id == agent_id)
        )
        agent = result.scalar_one_or_none()

        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        if agent.is_system:
            raise HTTPException(status_code=400, detail="System agents cannot be deleted")

        agent_name = agent.name
        await db.delete(agent)
        await db.commit()

        logger.info(f"Deleted agent '{agent_name}' by user {current_user.id}")

        return {"status": "ok", "deleted": str(agent_id), "name": agent_name}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting agent: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/{agent_id}/duplicate", response_model=AgentDefinitionResponse)
async def duplicate_agent(
    agent_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Duplicate an existing agent with a new name (admin only).
    """
    require_admin(current_user)

    try:
        result = await db.execute(
            select(AgentDefinition).where(AgentDefinition.id == agent_id)
        )
        source_agent = result.scalar_one_or_none()

        if not source_agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Generate unique name
        base_name = f"copy_of_{source_agent.name}"
        new_name = base_name
        counter = 1
        while True:
            existing = await db.execute(
                select(AgentDefinition).where(AgentDefinition.name == new_name)
            )
            if not existing.scalar_one_or_none():
                break
            counter += 1
            new_name = f"{base_name}_{counter}"

        # Create duplicate
        new_agent = AgentDefinition(
            name=new_name,
            display_name=f"Copy of {source_agent.display_name}",
            description=source_agent.description,
            system_prompt=source_agent.system_prompt,
            capabilities=source_agent.capabilities.copy() if source_agent.capabilities else [],
            tool_whitelist=source_agent.tool_whitelist.copy() if source_agent.tool_whitelist else None,
            priority=source_agent.priority,
            is_active=False,  # Start inactive
            is_system=False  # Duplicates are never system agents
        )
        db.add(new_agent)
        await db.commit()
        await db.refresh(new_agent)

        logger.info(f"Duplicated agent '{source_agent.name}' to '{new_agent.name}' by user {current_user.id}")

        return AgentDefinitionResponse(
            id=new_agent.id,
            name=new_agent.name,
            display_name=new_agent.display_name,
            description=new_agent.description,
            system_prompt=new_agent.system_prompt,
            capabilities=new_agent.capabilities or [],
            tool_whitelist=new_agent.tool_whitelist,
            priority=new_agent.priority,
            is_active=new_agent.is_active,
            is_system=new_agent.is_system,
            created_at=new_agent.created_at,
            updated_at=new_agent.updated_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error duplicating agent: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/capabilities", response_model=CapabilitiesListResponse)
async def list_capabilities(
    current_user: User = Depends(get_current_user)
):
    """
    List all available capabilities that can be assigned to agents.
    """
    capabilities = []
    for cap_name, keywords in CAPABILITY_KEYWORDS.items():
        # Generate description from capability name
        description = cap_name.replace("_", " ").title()
        if cap_name == "document_search":
            description = "Search and find documents in the knowledge base"
        elif cap_name == "document_crud":
            description = "Create, update, and delete documents"
        elif cap_name == "document_compare":
            description = "Compare and diff documents"
        elif cap_name == "tag_management":
            description = "Manage document tags and categories"
        elif cap_name == "rag_qa":
            description = "Answer questions using RAG (Retrieval-Augmented Generation)"
        elif cap_name == "summarization":
            description = "Summarize documents and content"
        elif cap_name == "knowledge_synthesis":
            description = "Analyze patterns, relationships, and synthesize insights"
        elif cap_name == "workflow_exec":
            description = "Execute automated workflows"
        elif cap_name == "template_fill":
            description = "Fill document templates with content"
        elif cap_name == "diagram_gen":
            description = "Generate diagrams and visualizations"
        elif cap_name == "automation":
            description = "Schedule and automate batch processes"
        elif cap_name == "code_analysis":
            description = "Analyze code structure and patterns"
        elif cap_name == "code_explanation":
            description = "Explain and document code"

        capabilities.append(CapabilityInfo(
            name=cap_name,
            description=description,
            keywords=keywords[:5]  # Show first 5 keywords as examples
        ))

    capabilities.append(CapabilityInfo(
        name="general",
        description="General assistance (fallback when no specialized capability matches)",
        keywords=[]
    ))

    return CapabilitiesListResponse(capabilities=capabilities)


@router.get("/conversations/{conversation_id}/agents")
async def get_conversation_agents(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get which agents participated in a conversation and why.
    """
    try:
        # Verify conversation belongs to user
        conv_result = await db.execute(
            select(AgentConversation).where(
                AgentConversation.id == conversation_id,
                AgentConversation.user_id == current_user.id
            )
        )
        conversation = conv_result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Get agent contexts
        result = await db.execute(
            select(AgentConversationContext, AgentDefinition)
            .join(AgentDefinition, AgentConversationContext.agent_definition_id == AgentDefinition.id)
            .where(AgentConversationContext.conversation_id == conversation_id)
            .order_by(AgentConversationContext.turn_number)
        )

        contexts = []
        for ctx, agent in result.all():
            contexts.append({
                "turn_number": ctx.turn_number,
                "agent_id": str(agent.id),
                "agent_name": agent.name,
                "agent_display_name": agent.display_name,
                "routing_reason": ctx.routing_reason,
                "handoff_context": ctx.handoff_context,
                "created_at": ctx.created_at.isoformat() if ctx.created_at else None
            })

        return {
            "conversation_id": str(conversation_id),
            "agent_participations": contexts,
            "total_handoffs": conversation.agent_handoffs or 0,
            "active_agent_id": str(conversation.active_agent_id) if conversation.active_agent_id else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{conversation_id}/memories")
async def get_conversation_memories(
    conversation_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get which memories were injected into a conversation.
    """
    try:
        # Verify conversation belongs to user
        conv_result = await db.execute(
            select(AgentConversation).where(
                AgentConversation.id == conversation_id,
                AgentConversation.user_id == current_user.id
            )
        )
        conversation = conv_result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Get memory injections with memory details
        result = await db.execute(
            select(AgentMemoryInjection, ConversationMemory)
            .join(ConversationMemory, AgentMemoryInjection.memory_id == ConversationMemory.id)
            .where(AgentMemoryInjection.conversation_id == conversation_id)
            .order_by(AgentMemoryInjection.turn_number, desc(AgentMemoryInjection.relevance_score))
        )

        injections = []
        for injection, memory in result.all():
            injections.append({
                "injection_id": str(injection.id),
                "turn_number": injection.turn_number,
                "relevance_score": injection.relevance_score,
                "injection_type": injection.injection_type,
                "memory": {
                    "id": str(memory.id),
                    "memory_type": memory.memory_type,
                    "content": memory.content,
                    "importance_score": memory.importance_score,
                    "tags": memory.tags
                },
                "created_at": injection.created_at.isoformat() if injection.created_at else None
            })

        return {
            "conversation_id": str(conversation_id),
            "memory_injections": injections,
            "total": len(injections)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting conversation memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/conversations/{conversation_id}/inject-memory")
async def manually_inject_memory(
    conversation_id: UUID,
    memory_id: UUID = Query(..., description="ID of the memory to inject"),
    turn_number: int = Query(0, description="Turn number to inject into"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Manually inject a specific memory into a conversation's context.
    Used when user wants to explicitly add context.
    """
    try:
        # Verify conversation belongs to user
        conv_result = await db.execute(
            select(AgentConversation).where(
                AgentConversation.id == conversation_id,
                AgentConversation.user_id == current_user.id
            )
        )
        conversation = conv_result.scalar_one_or_none()
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")

        # Verify memory belongs to user
        mem_result = await db.execute(
            select(ConversationMemory).where(
                ConversationMemory.id == memory_id,
                ConversationMemory.user_id == current_user.id
            )
        )
        memory = mem_result.scalar_one_or_none()
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")

        # Create injection record
        injection = AgentMemoryInjection(
            conversation_id=conversation_id,
            memory_id=memory_id,
            turn_number=turn_number,
            relevance_score=1.0,  # Manual injection = max relevance
            injection_type="manual"
        )
        db.add(injection)
        await db.commit()
        await db.refresh(injection)

        return {
            "status": "ok",
            "injection_id": str(injection.id),
            "memory_content": memory.content,
            "memory_type": memory.memory_type
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error injecting memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WebSocket Streaming Endpoint
# ============================================================================

@router.websocket("/ws")
async def agent_chat_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for streaming agent chat with real-time tool execution feedback.

    Message Types (Client -> Server):
        - {"type": "message", "content": "user message", "conversation_history": [...]}
        - {"type": "ping"}

    Message Types (Server -> Client):
        - {"type": "connected", "message": "Connected to agent"}
        - {"type": "thinking", "message": "Processing your request..."}
        - {"type": "planning", "message": "Determining actions...", "tool_count": n}
        - {"type": "tool_start", "tool": {...}}
        - {"type": "tool_progress", "tool_id": "...", "status": "running"}
        - {"type": "tool_complete", "tool": {...}}
        - {"type": "tool_error", "tool_id": "...", "error": "..."}
        - {"type": "generating", "message": "Generating response..."}
        - {"type": "response", "message": {...}, "tool_results": [...]}
        - {"type": "error", "message": "Error description"}
        - {"type": "pong"}
    """
    from app.utils.websocket_auth import require_websocket_auth

    # Authenticate WebSocket connection
    try:
        user = await require_websocket_auth(websocket)
        logger.info(f"Agent WebSocket authenticated for user {user.id}")
    except WebSocketDisconnect:
        return

    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to document assistant"
        })

        while True:
            try:
                # Receive message from client
                data = await websocket.receive_json()

                # Handle ping
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue

                # Handle chat message
                if data.get("type") == "message":
                    content = data.get("content", "").strip()
                    conversation_history = data.get("conversation_history", [])

                    if not content:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Empty message received"
                        })
                        continue

                    # Process message with streaming
                    await _process_message_with_streaming(
                        websocket=websocket,
                        message=content,
                        conversation_history=conversation_history,
                        user=user
                    )

            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON message"
                })

    except WebSocketDisconnect:
        logger.info(f"Agent WebSocket disconnected for user {user.id}")
    except Exception as e:
        logger.error(f"Agent WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Connection error: {str(e)}"
            })
        except:
            pass
        await websocket.close()


async def _process_message_with_streaming(
    websocket: WebSocket,
    message: str,
    conversation_history: List[dict],
    user: User
):
    """Process agent message with real-time streaming updates."""

    # Send thinking indicator
    await websocket.send_json({
        "type": "thinking",
        "message": "Processing your request..."
    })

    async with AsyncSessionLocal() as db:
        try:
            # Load user settings
            user_settings = None
            try:
                prefs_result = await db.execute(
                    select(UserPreferences).where(UserPreferences.user_id == user.id)
                )
                user_prefs = prefs_result.scalar_one_or_none()
                if user_prefs:
                    user_settings = UserLLMSettings.from_preferences(user_prefs)
            except Exception as e:
                logger.warning(f"Failed to load user preferences: {e}")

            # Create agent service with streaming callback
            # Reuse shared service to avoid re-initializing models per message

            # Convert conversation history to AgentMessage objects
            history = []
            for msg in conversation_history:
                if isinstance(msg, dict):
                    history.append(AgentMessage(
                        role=msg.get("role", "user"),
                        content=msg.get("content", ""),
                        created_at=datetime.fromisoformat(msg["created_at"]) if msg.get("created_at") else datetime.utcnow()
                    ))

            # Send planning indicator
            await websocket.send_json({
                "type": "planning",
                "message": "Determining actions..."
            })

            # Step 1: Plan tool calls
            tool_calls = await agent_service._plan_tool_calls(message, history, user_settings)

            # Send planning result
            await websocket.send_json({
                "type": "planning",
                "message": f"Found {len(tool_calls)} action(s) to perform",
                "tool_count": len(tool_calls)
            })

            # Step 2: Execute tools with streaming
            tool_results = []
            requires_user_action = False
            action_type = None

            for tool_call in tool_calls:
                # Notify tool start
                await websocket.send_json({
                    "type": "tool_start",
                    "tool": {
                        "id": tool_call.id,
                        "tool_name": tool_call.tool_name,
                        "tool_input": tool_call.tool_input,
                        "status": "pending"
                    }
                })

                # Small delay to allow UI to update
                await asyncio.sleep(0.1)

                # Send running status
                await websocket.send_json({
                    "type": "tool_progress",
                    "tool_id": tool_call.id,
                    "status": "running"
                })

                try:
                    # Execute the tool
                    result = await agent_service._execute_tool(tool_call, user.id, db)
                    tool_results.append(result)

                    # Check for user action requirements
                    if result.tool_name == "request_file_upload" and result.status == "completed":
                        requires_user_action = True
                        action_type = "upload_file"

                    # Send tool completion
                    await websocket.send_json({
                        "type": "tool_complete",
                        "tool": {
                            "id": result.id,
                            "tool_name": result.tool_name,
                            "tool_input": result.tool_input,
                            "tool_output": result.tool_output,
                            "status": result.status,
                            "execution_time_ms": result.execution_time_ms
                        }
                    })

                except Exception as e:
                    logger.error(f"Tool execution error: {e}")
                    tool_call.status = "failed"
                    tool_call.error = str(e)
                    tool_results.append(tool_call)

                    await websocket.send_json({
                        "type": "tool_error",
                        "tool_id": tool_call.id,
                        "error": str(e)
                    })

            # Step 3: Generate response
            await websocket.send_json({
                "type": "generating",
                "message": "Generating response..."
            })

            response_content = await agent_service._generate_response(
                message, tool_results, history, user_settings
            )

            # Create response message
            response_message = AgentMessage(
                role="assistant",
                content=response_content,
                tool_calls=tool_results if tool_results else None,
                created_at=datetime.utcnow()
            )

            # Send final response
            await websocket.send_json({
                "type": "response",
                "message": {
                    "id": response_message.id,
                    "role": response_message.role,
                    "content": response_message.content,
                    "created_at": response_message.created_at.isoformat(),
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "tool_name": tc.tool_name,
                            "tool_input": tc.tool_input,
                            "tool_output": tc.tool_output,
                            "status": tc.status,
                            "error": tc.error,
                            "execution_time_ms": tc.execution_time_ms
                        }
                        for tc in (tool_results or [])
                    ] if tool_results else None
                },
                "tool_results": [
                    {
                        "id": tc.id,
                        "tool_name": tc.tool_name,
                        "tool_output": tc.tool_output,
                        "status": tc.status,
                        "execution_time_ms": tc.execution_time_ms
                    }
                    for tc in (tool_results or [])
                ] if tool_results else None,
                "requires_user_action": requires_user_action,
                "action_type": action_type
            })

            logger.info(
                f"Agent WebSocket processed message for user {user.id}: "
                f"tools_called={len(tool_results)}, requires_action={requires_user_action}"
            )

        except Exception as e:
            logger.error(f"Error processing agent message via WebSocket: {e}")
            await websocket.send_json({
                "type": "error",
                "message": f"Failed to process message: {str(e)}"
            })


# =============================================================================
# Agent Builder Endpoints
# =============================================================================

@router.get("/templates", tags=["agent-builder"])
async def list_agent_templates(
    category: Optional[str] = Query(None, description="Filter by category"),
    current_user: User = Depends(get_current_user),
):
    """
    List all available agent templates for the Agent Builder.

    Templates provide pre-configured agent roles that can be customized:
    - support: Customer support, onboarding guides
    - content: Technical writers, documentation
    - analysis: Research analysts, data insights
    - productivity: Project managers, meeting assistants
    - development: Code reviewers
    - management: Content curators
    - governance: Compliance checkers
    """
    from app.services.agent_templates import (
        AGENT_TEMPLATES,
        get_templates_by_category,
        get_template_summary,
        list_template_categories,
    )

    if category:
        templates = get_templates_by_category(category)
        summary = [
            {
                "template_id": t["template_id"],
                "name": t["name"],
                "display_name": t["display_name"],
                "description": t["description"],
                "category": t.get("category", "other"),
                "capabilities": t.get("capabilities", []),
                "tool_count": len(t.get("tool_whitelist") or []),
                "use_cases": t.get("use_cases", []),
            }
            for t in templates
        ]
    else:
        summary = get_template_summary()

    return {
        "templates": summary,
        "categories": list_template_categories(),
        "total": len(summary)
    }


@router.get("/templates/{template_id}", tags=["agent-builder"])
async def get_agent_template(
    template_id: str,
    current_user: User = Depends(get_current_user),
):
    """Get full details of an agent template including system prompt."""
    from app.services.agent_templates import get_template_by_id

    template = get_template_by_id(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    return template


@router.post("/templates/{template_id}/create", tags=["agent-builder"])
async def create_agent_from_template(
    template_id: str,
    name: Optional[str] = Query(None, description="Custom agent name"),
    display_name: Optional[str] = Query(None, description="Custom display name"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new agent from a template.

    The agent will be created in 'draft' status so you can customize it
    before publishing.
    """
    from app.services.agent_templates import get_template_by_id
    import time

    template = get_template_by_id(template_id)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    # Generate unique name if not provided
    agent_name = name or f"{template['name']}_{int(time.time()) % 10000}"

    # Check for duplicate name
    existing = await db.execute(
        select(AgentDefinition).where(AgentDefinition.name == agent_name)
    )
    if existing.scalar_one_or_none():
        agent_name = f"{agent_name}_{int(time.time()) % 10000}"

    # Create agent from template
    agent = AgentDefinition(
        name=agent_name,
        display_name=display_name or template["display_name"],
        description=template["description"],
        system_prompt=template["system_prompt"],
        capabilities=template["capabilities"],
        tool_whitelist=template.get("tool_whitelist"),
        priority=template.get("priority", 50),
        is_active=False,  # Start inactive until published
        is_system=False,
        owner_user_id=current_user.id,
        lifecycle_status="draft",
    )

    db.add(agent)
    await db.commit()
    await db.refresh(agent)

    return {
        "message": f"Agent created from template '{template['display_name']}'",
        "agent": {
            "id": str(agent.id),
            "name": agent.name,
            "display_name": agent.display_name,
            "description": agent.description,
            "capabilities": agent.capabilities,
            "tool_whitelist": agent.tool_whitelist,
            "priority": agent.priority,
            "is_active": agent.is_active,
            "lifecycle_status": agent.lifecycle_status,
        },
        "template_id": template_id
    }


@router.post("/agents/{agent_id}/test", tags=["agent-builder"])
async def test_agent_routing(
    agent_id: UUID,
    message: str = Query(..., description="Test message to route"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Test how an agent would handle a message without executing tools.

    Returns:
    - Whether this agent would be selected for the message
    - The routing score and reason
    - Which tools the agent would have available
    - What the agent's response approach would be
    """
    from app.services.agent_router import AgentRouter
    from app.services.llm_service import LLMService
    from app.services.agent_tools import AGENT_TOOLS

    # Get the agent
    result = await db.execute(
        select(AgentDefinition).where(AgentDefinition.id == agent_id)
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Check ownership for non-system agents
    if not agent.is_system and agent.owner_user_id != current_user.id:
        if not current_user.is_admin():
            raise HTTPException(status_code=403, detail="Access denied")

    # Analyze intent
    llm_service = LLMService()
    router = AgentRouter(llm_service)
    await router.load_agents(db)

    intent_analysis = await router.analyze_intent(message)

    # Score this agent against the intent
    needed_caps = set(intent_analysis.get("capabilities_needed", []))
    agent_caps = set(agent.capabilities or [])

    if needed_caps:
        capability_score = len(needed_caps & agent_caps) / len(needed_caps)
    else:
        capability_score = 0.5  # Neutral if no specific capabilities needed

    priority_score = (agent.priority or 50) / 100
    total_score = (capability_score * 0.7) + (priority_score * 0.3)

    # Get available tools for this agent
    if agent.tool_whitelist:
        available_tools = [
            {"name": t["name"], "description": t["description"][:100]}
            for t in AGENT_TOOLS
            if t["name"] in agent.tool_whitelist
        ]
    else:
        available_tools = [
            {"name": t["name"], "description": t["description"][:100]}
            for t in AGENT_TOOLS
        ]

    # Get routing decision
    selected_agent, routing_reason = await router.select_agent(
        intent_analysis, db
    )

    would_be_selected = selected_agent and selected_agent.id == agent.id

    return {
        "agent": {
            "id": str(agent.id),
            "name": agent.name,
            "display_name": agent.display_name,
            "capabilities": agent.capabilities,
        },
        "test_message": message,
        "intent_analysis": intent_analysis,
        "scoring": {
            "capability_score": round(capability_score, 3),
            "priority_score": round(priority_score, 3),
            "total_score": round(total_score, 3),
            "matching_capabilities": list(needed_caps & agent_caps),
            "missing_capabilities": list(needed_caps - agent_caps),
        },
        "routing_decision": {
            "would_be_selected": would_be_selected,
            "selected_agent": selected_agent.display_name if selected_agent else None,
            "routing_reason": routing_reason,
        },
        "available_tools": available_tools[:20],  # Limit for readability
        "total_tools_available": len(available_tools),
    }


@router.get("/agents/{agent_id}/analytics", tags=["agent-builder"])
async def get_agent_analytics(
    agent_id: UUID,
    days: int = Query(30, ge=1, le=365, description="Days to analyze"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get usage analytics for an agent.

    Returns:
    - Total conversations handled
    - Tool usage statistics
    - Handoff patterns
    - Performance metrics
    """
    from datetime import datetime, timedelta
    from sqlalchemy import func

    # Get the agent
    result = await db.execute(
        select(AgentDefinition).where(AgentDefinition.id == agent_id)
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Check ownership for non-system agents
    if not agent.is_system and agent.owner_user_id != current_user.id:
        if not current_user.is_admin():
            raise HTTPException(status_code=403, detail="Access denied")

    cutoff_date = datetime.utcnow() - timedelta(days=days)

    # Get conversation context stats
    context_query = select(
        func.count(AgentConversationContext.id).label('total_turns'),
        func.count(func.distinct(AgentConversationContext.conversation_id)).label('unique_conversations'),
    ).where(
        AgentConversationContext.agent_definition_id == agent_id,
        AgentConversationContext.created_at >= cutoff_date
    )
    context_result = await db.execute(context_query)
    context_stats = context_result.one()

    # Get tool execution stats from ToolExecutionAudit
    from app.models.tool_audit import ToolExecutionAudit

    tool_query = select(
        ToolExecutionAudit.tool_name,
        func.count(ToolExecutionAudit.id).label('count'),
        func.avg(ToolExecutionAudit.execution_time_ms).label('avg_time'),
    ).where(
        ToolExecutionAudit.agent_definition_id == agent_id,
        ToolExecutionAudit.created_at >= cutoff_date
    ).group_by(ToolExecutionAudit.tool_name).order_by(func.count(ToolExecutionAudit.id).desc())

    tool_result = await db.execute(tool_query)
    tool_stats = [
        {
            "tool_name": row.tool_name,
            "call_count": row.count,
            "avg_execution_time_ms": round(row.avg_time, 2) if row.avg_time else None
        }
        for row in tool_result.fetchall()
    ]

    # Get status distribution
    status_query = select(
        ToolExecutionAudit.status,
        func.count(ToolExecutionAudit.id)
    ).where(
        ToolExecutionAudit.agent_definition_id == agent_id,
        ToolExecutionAudit.created_at >= cutoff_date
    ).group_by(ToolExecutionAudit.status)

    status_result = await db.execute(status_query)
    status_distribution = dict(status_result.fetchall())

    # Get handoff stats (from context routing_reason)
    handoff_query = select(
        func.count(AgentConversationContext.id)
    ).where(
        AgentConversationContext.agent_definition_id == agent_id,
        AgentConversationContext.created_at >= cutoff_date,
        AgentConversationContext.handoff_context != None
    )
    handoff_result = await db.execute(handoff_query)
    handoff_count = handoff_result.scalar() or 0

    # Daily usage trend
    daily_query = select(
        func.date(AgentConversationContext.created_at).label('date'),
        func.count(AgentConversationContext.id).label('turns')
    ).where(
        AgentConversationContext.agent_definition_id == agent_id,
        AgentConversationContext.created_at >= cutoff_date
    ).group_by(func.date(AgentConversationContext.created_at)).order_by('date')

    daily_result = await db.execute(daily_query)
    daily_trend = [
        {"date": str(row.date), "turns": row.turns}
        for row in daily_result.fetchall()
    ]

    return {
        "agent": {
            "id": str(agent.id),
            "name": agent.name,
            "display_name": agent.display_name,
            "is_active": agent.is_active,
        },
        "period_days": days,
        "summary": {
            "total_turns": context_stats.total_turns or 0,
            "unique_conversations": context_stats.unique_conversations or 0,
            "handoffs_received": handoff_count,
            "total_tool_calls": sum(t["call_count"] for t in tool_stats),
        },
        "tool_usage": tool_stats[:15],  # Top 15 tools
        "status_distribution": status_distribution,
        "daily_trend": daily_trend,
    }


@router.post("/agents/{agent_id}/publish", tags=["agent-builder"])
async def publish_agent(
    agent_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Publish a draft agent, making it active and available for routing.
    """
    result = await db.execute(
        select(AgentDefinition).where(AgentDefinition.id == agent_id)
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Check ownership
    if agent.is_system:
        raise HTTPException(status_code=403, detail="Cannot modify system agents")
    if agent.owner_user_id != current_user.id and not current_user.is_admin():
        raise HTTPException(status_code=403, detail="Access denied")

    agent.lifecycle_status = "published"
    agent.is_active = True
    await db.commit()

    return {
        "message": f"Agent '{agent.display_name}' published successfully",
        "agent_id": str(agent.id),
        "is_active": agent.is_active,
        "lifecycle_status": agent.lifecycle_status
    }


@router.post("/agents/{agent_id}/archive", tags=["agent-builder"])
async def archive_agent(
    agent_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Archive an agent, making it inactive but preserving it for historical reference.
    """
    result = await db.execute(
        select(AgentDefinition).where(AgentDefinition.id == agent_id)
    )
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Check ownership
    if agent.is_system:
        raise HTTPException(status_code=403, detail="Cannot archive system agents")
    if agent.owner_user_id != current_user.id and not current_user.is_admin():
        raise HTTPException(status_code=403, detail="Access denied")

    agent.lifecycle_status = "archived"
    agent.is_active = False
    await db.commit()

    return {
        "message": f"Agent '{agent.display_name}' archived",
        "agent_id": str(agent.id),
        "lifecycle_status": agent.lifecycle_status
    }
