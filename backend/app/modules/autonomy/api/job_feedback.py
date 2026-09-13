"""HTTP boundary for human feedback used by autonomous-job learning."""

import json
from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.core.feature_flags import get_str as get_feature_str
from app.models.agent_job import AgentJob
from app.models.memory import ConversationMemory
from app.models.user import User
from app.modules.autonomy.application import feedback_presenters
from app.schemas.agent_job import (
    AgentJobFeedbackCreate,
    AgentJobFeedbackListResponse,
    AgentJobFeedbackResponse,
)
from app.schemas.customer_profile import CustomerProfile
from app.services.agent_job_memory_service import agent_job_memory_service

router = APIRouter()


async def _get_owned_job(
    *,
    job_id: UUID,
    user_id: UUID,
    db: AsyncSession,
) -> AgentJob:
    result = await db.execute(
        select(AgentJob).where(and_(AgentJob.id == job_id, AgentJob.user_id == user_id))
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code=404, detail="Agent job not found")
    return job


def _is_feedback_memory(memory: ConversationMemory) -> bool:
    tags = [
        str(tag).strip().lower()
        for tag in (memory.tags if isinstance(memory.tags, list) else [])
        if str(tag).strip()
    ]
    context = memory.context if isinstance(memory.context, dict) else {}
    return "human_feedback" in tags or (
        str(context.get("feedback_type") or "").strip().lower() == "human"
    )


@router.post("/{job_id}/feedback", response_model=AgentJobFeedbackResponse)
async def create_agent_job_feedback(
    job_id: UUID,
    payload: AgentJobFeedbackCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Persist human feedback as a lesson memory for future agent runs."""
    job = await _get_owned_job(
        job_id=job_id,
        user_id=current_user.id,
        db=db,
    )

    scope = str(payload.scope or "user").strip().lower()
    if scope not in {"user", "customer", "team"}:
        raise HTTPException(
            status_code=400,
            detail="scope must be one of: user, customer, team",
        )

    preferred_tools = feedback_presenters.sanitize_tool_names(payload.preferred_tools)
    discouraged_tools = feedback_presenters.sanitize_tool_names(
        payload.discouraged_tools
    )
    overlap = set(preferred_tools).intersection(discouraged_tools)
    if overlap:
        discouraged_tools = [tool for tool in discouraged_tools if tool not in overlap]

    target_type = str(payload.target_type or "job").strip().lower()
    if target_type not in {"job", "checkpoint", "finding", "action", "tool"}:
        raise HTTPException(
            status_code=400,
            detail="target_type must be job, checkpoint, finding, action, or tool",
        )

    team_key = str(payload.team_key or "").strip()
    if scope == "team" and not team_key:
        raise HTTPException(
            status_code=400,
            detail="team_key is required when scope=team",
        )

    scope_marker = f"user:{current_user.id}"
    if scope == "customer":
        customer = str((job.config or {}).get("customer") or "").strip()
        if not customer:
            raw_profile = await get_feature_str("ai_hub_customer_profile")
            if raw_profile:
                try:
                    profile = CustomerProfile.model_validate(json.loads(raw_profile))
                    customer = str(profile.id or profile.name or "").strip()
                except Exception:
                    customer = ""
        if not customer:
            raise HTTPException(
                status_code=400,
                detail=(
                    "customer scope requires job.config.customer or "
                    "ai_hub_customer_profile"
                ),
            )
        scope_marker = f"customer:{customer[:120]}"
    elif scope == "team":
        scope_marker = f"team:{team_key[:120]}"

    rating = max(1, min(int(payload.rating), 5))
    sentiment = (
        "positive" if rating >= 4 else ("negative" if rating <= 2 else "neutral")
    )
    feedback_text = str(payload.feedback or "").strip()
    target_id = str(payload.target_id or "").strip()
    checkpoint = str(payload.checkpoint or "").strip()
    content = feedback_text or f"User rated {target_type} as {rating}/5."
    importance = min(1.0, max(0.35, 0.55 + abs(rating - 3) * 0.1))

    tags = [
        "human_feedback",
        "feedback",
        f"feedback:{sentiment}",
        f"rating:{rating}",
        f"job_type:{job.job_type}",
        f"target:{target_type}",
        f"scope:{scope}",
        scope_marker,
    ]
    tags.extend(f"prefer_tool:{tool}" for tool in preferred_tools)
    tags.extend(f"avoid_tool:{tool}" for tool in discouraged_tools)
    tags = list(dict.fromkeys(tag for tag in tags if str(tag).strip()))

    context = {
        "feedback_type": "human",
        "rating": rating,
        "feedback_text": feedback_text,
        "target_type": target_type,
        "target_id": target_id or None,
        "checkpoint": checkpoint or None,
        "scope": scope,
        "scope_marker": scope_marker,
        "preferred_tools": preferred_tools,
        "discouraged_tools": discouraged_tools,
        "job_id": str(job.id),
        "job_name": job.name,
        "job_type": job.job_type,
        "job_status": job.status,
        "recorded_at": datetime.utcnow().isoformat(),
    }
    memory = ConversationMemory(
        user_id=current_user.id,
        job_id=job.id,
        memory_type="lesson",
        content=content,
        importance_score=importance,
        tags=tags,
        context=context,
    )
    db.add(memory)
    job.add_log_entry(
        {
            "phase": "human_feedback_recorded",
            "rating": rating,
            "target_type": target_type,
            "scope": scope,
            "preferred_tools": preferred_tools[:8],
            "discouraged_tools": discouraged_tools[:8],
        }
    )
    await db.commit()
    await db.refresh(memory)

    try:
        await agent_job_memory_service.link_memories_into_task_graph(
            new_memories=[memory],
            user_id=str(current_user.id),
            db=db,
        )
        await db.refresh(memory)
    except Exception as error:
        logger.warning(
            f"Failed linking feedback memory {memory.id} into task graph: {error}"
        )
    return feedback_presenters.memory_to_feedback_response(memory)


@router.get(
    "/{job_id}/feedback",
    response_model=AgentJobFeedbackListResponse,
)
async def list_agent_job_feedback(
    job_id: UUID,
    limit: int = Query(50, ge=1, le=200, description="Max feedback entries"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List feedback captured for one user-owned job."""
    await _get_owned_job(job_id=job_id, user_id=current_user.id, db=db)
    result = await db.execute(
        select(ConversationMemory)
        .where(
            and_(
                ConversationMemory.user_id == current_user.id,
                ConversationMemory.job_id == job_id,
                ConversationMemory.is_active.is_(True),
                ConversationMemory.memory_type == "lesson",
            )
        )
        .order_by(ConversationMemory.created_at.desc())
        .limit(max(20, limit * 3))
    )
    rows = []
    for memory in result.scalars().all():
        if _is_feedback_memory(memory):
            rows.append(feedback_presenters.memory_to_feedback_response(memory))
        if len(rows) >= limit:
            break
    return AgentJobFeedbackListResponse(items=rows, total=len(rows))


@router.get("/memory/feedback", response_model=AgentJobFeedbackListResponse)
async def list_learning_feedback(
    scope: Optional[str] = Query(
        None,
        description="Optional scope filter: user|customer|team",
    ),
    limit: int = Query(100, ge=1, le=300, description="Max feedback entries"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List recent human feedback memories used by the learning loop."""
    scope_filter = str(scope or "").strip().lower()
    if scope_filter and scope_filter not in {"user", "customer", "team"}:
        raise HTTPException(
            status_code=400,
            detail="scope must be user, customer, or team",
        )

    result = await db.execute(
        select(ConversationMemory)
        .where(
            and_(
                ConversationMemory.user_id == current_user.id,
                ConversationMemory.is_active.is_(True),
                ConversationMemory.memory_type == "lesson",
            )
        )
        .order_by(ConversationMemory.created_at.desc())
        .limit(max(50, limit * 3))
    )
    items = []
    for memory in result.scalars().all():
        context = memory.context if isinstance(memory.context, dict) else {}
        if not _is_feedback_memory(memory):
            continue
        if (
            scope_filter
            and str(context.get("scope") or "").strip().lower() != scope_filter
        ):
            continue
        items.append(feedback_presenters.memory_to_feedback_response(memory))
        if len(items) >= limit:
            break
    return AgentJobFeedbackListResponse(items=items, total=len(items))
