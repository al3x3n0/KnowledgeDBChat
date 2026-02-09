"""
API endpoints for autonomous agent jobs.

Provides CRUD operations and control actions for autonomous agent jobs.
"""

import asyncio
import json
import re
import uuid
from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from loguru import logger
from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import redis.asyncio as redis

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.agent_job import (
    AgentJob,
    AgentJobStatus,
    AgentJobTemplate,
    AgentJobCheckpoint,
    AgentJobChainDefinition,
)
from app.models.agent_definition import AgentDefinition
from app.models.memory import ConversationMemory
from app.models.user import User
from app.schemas.agent_job import (
    AgentJobCreate,
    AgentJobFromTemplate,
    AgentJobUpdate,
    AgentJobResponse,
    AgentJobListResponse,
    AgentJobDetailResponse,
    AgentJobTemplateResponse,
    AgentJobTemplateListResponse,
    AgentJobActionRequest,
    AgentJobStatsResponse,
    AgentJobCheckpointResponse,
    # Chain schemas
    AgentJobChainDefinitionCreate,
    AgentJobChainDefinitionUpdate,
    AgentJobChainDefinitionResponse,
    AgentJobChainDefinitionListResponse,
    AgentJobFromChainCreate,
    AgentJobChainStatusResponse,
    AgentJobSaveAsChainRequest,
    AgentJobFeedbackCreate,
    AgentJobFeedbackResponse,
    AgentJobFeedbackListResponse,
)
from app.tasks.agent_job_tasks import execute_agent_job_task, generate_job_summary
from app.models.ai_hub_recommendation_feedback import AIHubRecommendationFeedback
from app.core.feature_flags import get_str as get_feature_str
from app.schemas.customer_profile import CustomerProfile
from app.schemas.ai_hub_recommendation_feedback import (
    AIHubRecommendationFeedbackCreate,
    AIHubRecommendationFeedbackResponse,
    AIHubRecommendationFeedbackListResponse,
)
from app.services.agent_job_templates import (
    get_builtin_agent_job_template,
    list_builtin_agent_job_templates,
)
from app.services.agent_job_chain_templates import (
    get_builtin_agent_job_chain_definition,
    list_builtin_agent_job_chain_definitions,
)

# NOTE: api/routes.py mounts this router under `/agent-jobs`.
# Do not set a prefix here, otherwise routes become `/agent-jobs/agent-jobs/...`.
router = APIRouter()


def _extract_swarm_summary(job: AgentJob) -> Optional[dict]:
    """Build a compact swarm/fan-in summary for API consumers."""
    results = job.results if isinstance(job.results, dict) else {}
    execution_strategy = results.get("execution_strategy") if isinstance(results.get("execution_strategy"), dict) else {}
    swarm_exec = execution_strategy.get("swarm") if isinstance(execution_strategy.get("swarm"), dict) else {}
    fan_in = results.get("swarm_fan_in") if isinstance(results.get("swarm_fan_in"), dict) else {}

    cfg = job.config if isinstance(job.config, dict) else {}
    enabled = bool(cfg.get("swarm_child_jobs_enabled", False) or swarm_exec.get("enabled", False))
    configured = bool(swarm_exec.get("configured", False) or swarm_exec)
    fan_in_enabled = bool(swarm_exec.get("fan_in_enabled", False))

    expected_siblings = int(fan_in.get("expected_siblings", 0) or 0)
    received_siblings = int(fan_in.get("received_siblings", 0) or 0)
    terminal_siblings = int(fan_in.get("terminal_siblings", 0) or 0)
    if expected_siblings <= 0:
        expected_siblings = int(swarm_exec.get("child_jobs_count", 0) or 0)
    if received_siblings <= 0 and expected_siblings > 0:
        received_siblings = expected_siblings
    if terminal_siblings <= 0 and received_siblings > 0:
        terminal_siblings = received_siblings

    roles = []
    raw_roles = fan_in.get("roles")
    if isinstance(raw_roles, list) and raw_roles:
        roles = [str(r).strip() for r in raw_roles if str(r).strip()][:20]
    elif isinstance(swarm_exec.get("roles_assigned"), list):
        roles = [str(r).strip() for r in swarm_exec.get("roles_assigned", []) if str(r).strip()][:20]

    confidence = fan_in.get("confidence") if isinstance(fan_in.get("confidence"), dict) else {}
    consensus_rows = fan_in.get("consensus_findings") if isinstance(fan_in.get("consensus_findings"), list) else []
    consensus_findings = [
        str(row.get("finding") or "").strip()[:280]
        for row in consensus_rows
        if isinstance(row, dict) and str(row.get("finding") or "").strip()
    ][:10]
    conflicts = fan_in.get("conflicts") if isinstance(fan_in.get("conflicts"), list) else []
    action_plan = fan_in.get("action_plan") if isinstance(fan_in.get("action_plan"), list) else []

    if not any([enabled, configured, fan_in, swarm_exec]):
        return None

    return {
        "enabled": enabled,
        "configured": configured,
        "fan_in_enabled": fan_in_enabled,
        "fan_in_group_id": str(fan_in.get("fan_in_group_id") or swarm_exec.get("fan_in_group_id") or "").strip(),
        "roles": roles,
        "role_count": len(roles),
        "expected_siblings": expected_siblings,
        "received_siblings": received_siblings,
        "terminal_siblings": terminal_siblings,
        "consensus_count": len(consensus_rows),
        "consensus_findings": consensus_findings,
        "conflict_count": len(conflicts),
        "conflicts": conflicts[:10],
        "action_plan": action_plan[:10],
        "confidence": confidence,
    }


def _extract_goal_contract_summary(job: AgentJob) -> Optional[dict]:
    """Build compact goal-contract status for quick UI rendering."""
    results = job.results if isinstance(job.results, dict) else {}
    contract = results.get("goal_contract") if isinstance(results.get("goal_contract"), dict) else {}
    if not contract:
        return None

    enabled = bool(contract.get("enabled", False))
    if not enabled and not contract:
        return None
    missing = contract.get("missing") if isinstance(contract.get("missing"), list) else []
    contract_cfg = contract.get("contract") if isinstance(contract.get("contract"), dict) else {}
    metrics = contract.get("metrics") if isinstance(contract.get("metrics"), dict) else {}
    return {
        "enabled": enabled,
        "satisfied": bool(contract.get("satisfied", True)),
        "missing_count": len(missing),
        "missing": [str(x)[:120] for x in missing[:10]],
        "strict_completion": bool(contract_cfg.get("strict_completion", False)),
        "satisfied_iteration": int(contract.get("satisfied_iteration", 0) or 0),
        "metrics": metrics,
    }


def _extract_approval_checkpoint(job: AgentJob) -> Optional[dict]:
    """Extract pending approval checkpoint summary for paused jobs."""
    results = job.results if isinstance(job.results, dict) else {}
    direct = results.get("approval_checkpoint") if isinstance(results.get("approval_checkpoint"), dict) else None
    execution = results.get("execution_strategy") if isinstance(results.get("execution_strategy"), dict) else {}
    approval = execution.get("approval_checkpoints") if isinstance(execution.get("approval_checkpoints"), dict) else {}
    pending = approval.get("pending") if isinstance(approval.get("pending"), dict) else None
    data = direct or pending
    if not isinstance(data, dict):
        return None
    return {
        "required": True,
        "status": "pending" if str(job.status or "") == AgentJobStatus.PAUSED.value else "stale",
        "current_phase": str(job.current_phase or ""),
        "message": str(data.get("message") or job.phase_details or "").strip()[:300],
        "iteration": int(data.get("iteration", 0) or 0),
        "reasons": [str(x)[:140] for x in (data.get("reasons") if isinstance(data.get("reasons"), list) else [])[:8]],
        "action": data.get("action") if isinstance(data.get("action"), dict) else {},
        "created_at": data.get("created_at"),
    }


def _extract_executive_digest(job: AgentJob) -> Optional[dict]:
    """Extract deterministic executive digest payload when present."""
    results = job.results if isinstance(job.results, dict) else {}
    digest = results.get("executive_digest") if isinstance(results.get("executive_digest"), dict) else None
    return digest


def _sanitize_tool_names(values: Optional[list[str]], *, limit: int = 12) -> list[str]:
    out: list[str] = []
    if not isinstance(values, list):
        return out
    for raw in values:
        tool = str(raw or "").strip()
        if not tool:
            continue
        if not re.match(r"^[a-zA-Z0-9_:\\-]{2,80}$", tool):
            continue
        if tool not in out:
            out.append(tool)
        if len(out) >= max(1, min(limit, 40)):
            break
    return out


def _memory_to_feedback_response(memory: ConversationMemory) -> AgentJobFeedbackResponse:
    context = memory.context if isinstance(memory.context, dict) else {}
    preferred = context.get("preferred_tools") if isinstance(context.get("preferred_tools"), list) else []
    discouraged = context.get("discouraged_tools") if isinstance(context.get("discouraged_tools"), list) else []
    try:
        rating = int(context.get("rating", 0) or 0)
    except Exception:
        rating = 0
    rating = max(1, min(5, rating)) if rating else 3
    return AgentJobFeedbackResponse(
        id=memory.id,
        job_id=memory.job_id,
        rating=rating,
        feedback=str(context.get("feedback_text") or memory.content or "").strip() or None,
        target_type=str(context.get("target_type") or "job"),
        target_id=str(context.get("target_id") or "").strip() or None,
        scope=str(context.get("scope") or "user"),
        preferred_tools=[str(x) for x in preferred[:20]],
        discouraged_tools=[str(x) for x in discouraged[:20]],
        checkpoint=str(context.get("checkpoint") or "").strip() or None,
        created_at=memory.created_at,
    )


def _job_to_response(job: AgentJob) -> AgentJobResponse:
    """Convert AgentJob model to response schema."""
    return AgentJobResponse(
        id=job.id,
        name=job.name,
        description=job.description,
        job_type=job.job_type,
        goal=job.goal,
        goal_criteria=job.goal_criteria,
        config=job.config,
        agent_definition_id=job.agent_definition_id,
        agent_definition_name=job.agent_definition.name if job.agent_definition else None,
        user_id=job.user_id,
        status=job.status,
        progress=job.progress,
        current_phase=job.current_phase,
        phase_details=job.phase_details,
        iteration=job.iteration,
        max_iterations=job.max_iterations,
        max_tool_calls=job.max_tool_calls,
        max_llm_calls=job.max_llm_calls,
        max_runtime_minutes=job.max_runtime_minutes,
        tool_calls_used=job.tool_calls_used,
        llm_calls_used=job.llm_calls_used,
        tokens_used=job.tokens_used,
        error=job.error,
        error_count=job.error_count,
        schedule_type=job.schedule_type,
        schedule_cron=job.schedule_cron,
        next_run_at=job.next_run_at,
        results=job.results,
        output_artifacts=job.output_artifacts,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        last_activity_at=job.last_activity_at,
        celery_task_id=job.celery_task_id,
        # Chain fields
        parent_job_id=job.parent_job_id,
        root_job_id=job.root_job_id,
        chain_depth=job.chain_depth,
        chain_triggered=job.chain_triggered,
        chain_config=job.chain_config,
        swarm_summary=_extract_swarm_summary(job),
        goal_contract_summary=_extract_goal_contract_summary(job),
        approval_checkpoint=_extract_approval_checkpoint(job),
        executive_digest=_extract_executive_digest(job),
    )


@router.post("", response_model=AgentJobResponse, status_code=status.HTTP_201_CREATED)
async def create_agent_job(
    job_create: AgentJobCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Create a new autonomous agent job.

    Creates a background job that will work autonomously toward the specified goal.
    """
    # Validate agent definition if specified
    if job_create.agent_definition_id:
        result = await db.execute(
            select(AgentDefinition).where(AgentDefinition.id == job_create.agent_definition_id)
        )
        agent_def = result.scalar_one_or_none()
        if not agent_def:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent definition not found",
            )

    # Validate parent job if specified (for manual chaining)
    if job_create.parent_job_id:
        parent_result = await db.execute(
            select(AgentJob).where(
                and_(
                    AgentJob.id == job_create.parent_job_id,
                    AgentJob.user_id == current_user.id,
                )
            )
        )
        parent_job = parent_result.scalar_one_or_none()
        if not parent_job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Parent job not found",
            )

    # Create the job
    job = AgentJob(
        name=job_create.name,
        description=job_create.description,
        job_type=job_create.job_type,
        goal=job_create.goal,
        goal_criteria=job_create.goal_criteria,
        config=job_create.config,
        agent_definition_id=job_create.agent_definition_id,
        user_id=current_user.id,
        status=AgentJobStatus.PENDING.value,
        max_iterations=job_create.max_iterations or 100,
        max_tool_calls=job_create.max_tool_calls or 500,
        max_llm_calls=job_create.max_llm_calls or 200,
        max_runtime_minutes=job_create.max_runtime_minutes or 60,
        schedule_type=job_create.schedule_type,
        schedule_cron=job_create.schedule_cron,
        # Chain fields
        chain_config=job_create.chain_config,
        parent_job_id=job_create.parent_job_id,
        chain_depth=parent_job.chain_depth + 1 if job_create.parent_job_id and parent_job else 0,
        root_job_id=parent_job.root_job_id or parent_job.id if job_create.parent_job_id and parent_job else None,
    )

    # Set next_run_at for scheduled jobs
    if job_create.schedule_type and job_create.schedule_cron:
        try:
            from croniter import croniter
            cron = croniter(job_create.schedule_cron, datetime.utcnow())
            job.next_run_at = cron.get_next(datetime)
        except Exception as e:
            logger.warning(f"Invalid cron expression: {e}")
    elif job_create.schedule_type == "continuous" and not job.next_run_at:
        # Continuous jobs use a simple interval (handled by scheduler task).
        job.next_run_at = datetime.utcnow()

    db.add(job)
    await db.commit()
    await db.refresh(job)

    logger.info(f"Created agent job {job.id} for user {current_user.id}")

    # Start immediately if requested (including scheduled jobs).
    # For scheduled jobs, we also advance `next_run_at` to avoid an immediate duplicate run by the scheduler.
    if job_create.start_immediately:
        execute_agent_job_task.delay(str(job.id), str(current_user.id))
        logger.info(f"Queued agent job {job.id} for immediate execution")

        if job.schedule_type == "continuous":
            try:
                interval = int(((job.config or {}).get("interval_minutes") or 30))
            except Exception:
                interval = 30
            interval = max(1, min(interval, 24 * 60))
            job.next_run_at = datetime.utcnow() + timedelta(minutes=interval)
            await db.commit()
        elif job.schedule_type == "once":
            job.next_run_at = None
            await db.commit()

    return _job_to_response(job)


@router.post("/from-template", response_model=AgentJobResponse, status_code=status.HTTP_201_CREATED)
async def create_job_from_template(
    request: AgentJobFromTemplate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Create a new agent job from a template.

    Uses the template's default configuration with optional overrides.
    """
    # Load template (builtin first, then DB)
    builtin = get_builtin_agent_job_template(request.template_id)
    template = None
    if builtin is None:
        result = await db.execute(
            select(AgentJobTemplate).where(
                and_(
                    AgentJobTemplate.id == request.template_id,
                    AgentJobTemplate.is_active == True,
                )
            )
        )
        template = result.scalar_one_or_none()

    if not builtin and not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job template not found or not active",
        )

    # Merge config
    base_config = builtin.default_config if builtin else (template.default_config or {})
    config = dict(base_config) if base_config else {}
    if request.config:
        config.update(request.config)

    # Create job from template
    job = AgentJob(
        name=request.name,
        description=(builtin.description if builtin else template.description),
        job_type=(builtin.job_type if builtin else template.job_type),
        goal=request.goal or (builtin.default_goal if builtin else template.default_goal),
        config=config,
        agent_definition_id=(builtin.agent_definition_id if builtin else template.agent_definition_id),
        user_id=current_user.id,
        status=AgentJobStatus.PENDING.value,
        max_iterations=(builtin.default_max_iterations if builtin else template.default_max_iterations),
        max_tool_calls=(builtin.default_max_tool_calls if builtin else template.default_max_tool_calls),
        max_llm_calls=(builtin.default_max_llm_calls if builtin else template.default_max_llm_calls),
        max_runtime_minutes=(builtin.default_max_runtime_minutes if builtin else template.default_max_runtime_minutes),
        chain_config=(
            request.chain_config
            if request.chain_config
            else (builtin.default_chain_config if builtin else None)
        ),
    )

    db.add(job)
    await db.commit()
    await db.refresh(job)

    logger.info(f"Created agent job {job.id} from template {template.name}")

    # Start immediately if requested
    if request.start_immediately:
        execute_agent_job_task.delay(str(job.id), str(current_user.id))

    return _job_to_response(job)


@router.get("", response_model=AgentJobListResponse)
async def list_agent_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    job_type: Optional[str] = Query(None, description="Filter by job type"),
    swarm_only: bool = Query(False, description="Only return jobs with swarm summary data"),
    swarm_min_consensus: int = Query(0, ge=0, le=100, description="Minimum swarm consensus findings"),
    sort_by: str = Query(
        "created_desc",
        description="Sort mode: created_desc|created_asc|swarm_confidence_desc|swarm_consensus_desc|swarm_conflicts_desc",
    ),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    List agent jobs for the current user.

    Returns paginated list of jobs with optional filtering.
    """
    # Build query
    query = select(AgentJob).where(AgentJob.user_id == current_user.id)

    if status:
        query = query.where(AgentJob.status == status)
    if job_type:
        query = query.where(AgentJob.job_type == job_type)

    sort_mode = str(sort_by or "created_desc").strip().lower()
    allowed_sort_modes = {
        "created_desc",
        "created_asc",
        "swarm_confidence_desc",
        "swarm_consensus_desc",
        "swarm_conflicts_desc",
    }
    if sort_mode not in allowed_sort_modes:
        sort_mode = "created_desc"

    requires_swarm_projection = bool(swarm_only) or int(swarm_min_consensus or 0) > 0 or sort_mode.startswith("swarm_") or sort_mode == "created_asc"

    if not requires_swarm_projection:
        # Fast SQL path for common listing case.
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        page_query = query.options(selectinload(AgentJob.agent_definition))
        page_query = page_query.order_by(AgentJob.created_at.desc())
        page_query = page_query.offset((page - 1) * page_size).limit(page_size)
        result = await db.execute(page_query)
        jobs = result.scalars().all()
    else:
        all_query = query.options(selectinload(AgentJob.agent_definition))
        all_query = all_query.order_by(AgentJob.created_at.desc())
        all_result = await db.execute(all_query)
        jobs_all = all_result.scalars().all()

        rows = []
        for job in jobs_all:
            swarm_summary = _extract_swarm_summary(job)
            if swarm_only and not swarm_summary:
                continue
            if int(swarm_min_consensus or 0) > 0:
                consensus_count = int((swarm_summary or {}).get("consensus_count", 0) or 0)
                if consensus_count < int(swarm_min_consensus or 0):
                    continue
            rows.append((job, swarm_summary))

        def _created_ts(job: AgentJob) -> float:
            try:
                return float(job.created_at.timestamp()) if job.created_at else 0.0
            except Exception:
                return 0.0

        if sort_mode == "created_asc":
            rows.sort(key=lambda x: _created_ts(x[0]))
        elif sort_mode == "swarm_confidence_desc":
            rows.sort(
                key=lambda x: (
                    float((((x[1] or {}).get("confidence") or {}).get("overall") or 0.0)),
                    int((x[1] or {}).get("consensus_count", 0) or 0),
                    _created_ts(x[0]),
                ),
                reverse=True,
            )
        elif sort_mode == "swarm_consensus_desc":
            rows.sort(
                key=lambda x: (
                    int((x[1] or {}).get("consensus_count", 0) or 0),
                    float((((x[1] or {}).get("confidence") or {}).get("overall") or 0.0)),
                    _created_ts(x[0]),
                ),
                reverse=True,
            )
        elif sort_mode == "swarm_conflicts_desc":
            rows.sort(
                key=lambda x: (
                    int((x[1] or {}).get("conflict_count", 0) or 0),
                    int((x[1] or {}).get("consensus_count", 0) or 0),
                    _created_ts(x[0]),
                ),
                reverse=True,
            )

        total = len(rows)
        offset = (page - 1) * page_size
        jobs = [j for j, _ in rows[offset : offset + page_size]]

    return AgentJobListResponse(
        jobs=[_job_to_response(job) for job in jobs],
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )


@router.get("/stats", response_model=AgentJobStatsResponse)
async def get_job_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get statistics for the current user's agent jobs.
    """
    base_query = select(AgentJob).where(AgentJob.user_id == current_user.id)

    # Count by status
    status_counts = {}
    for s in AgentJobStatus:
        count_result = await db.execute(
            select(func.count()).where(
                and_(AgentJob.user_id == current_user.id, AgentJob.status == s.value)
            )
        )
        status_counts[s.value] = count_result.scalar()

    # Total counts
    total_result = await db.execute(
        select(
            func.sum(AgentJob.iteration),
            func.sum(AgentJob.tool_calls_used),
            func.sum(AgentJob.llm_calls_used),
        ).where(AgentJob.user_id == current_user.id)
    )
    totals = total_result.one()

    # Average completion time
    completed_jobs = await db.execute(
        select(AgentJob).where(
            and_(
                AgentJob.user_id == current_user.id,
                AgentJob.status == AgentJobStatus.COMPLETED.value,
                AgentJob.started_at.isnot(None),
                AgentJob.completed_at.isnot(None),
            )
        )
    )
    completed = completed_jobs.scalars().all()

    avg_time = None
    if completed:
        durations = [
            (job.completed_at - job.started_at).total_seconds() / 60
            for job in completed
        ]
        avg_time = sum(durations) / len(durations)

    # Success rate
    total_finished = status_counts.get("completed", 0) + status_counts.get("failed", 0)
    success_rate = None
    if total_finished > 0:
        success_rate = status_counts.get("completed", 0) / total_finished

    return AgentJobStatsResponse(
        total_jobs=sum(status_counts.values()),
        running_jobs=status_counts.get("running", 0),
        pending_jobs=status_counts.get("pending", 0),
        completed_jobs=status_counts.get("completed", 0),
        failed_jobs=status_counts.get("failed", 0),
        total_iterations=totals[0] or 0,
        total_tool_calls=totals[1] or 0,
        total_llm_calls=totals[2] or 0,
        avg_completion_time_minutes=avg_time,
        success_rate=success_rate,
    )


@router.get("/templates", response_model=AgentJobTemplateListResponse)
async def list_job_templates(
    category: Optional[str] = Query(None, description="Filter by category"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    List available job templates.

    Returns system templates and user's own templates.
    """
    query = select(AgentJobTemplate).where(
        and_(
            AgentJobTemplate.is_active == True,
            or_(
                AgentJobTemplate.is_system == True,
                AgentJobTemplate.owner_user_id == current_user.id,
            )
        )
    )

    if category:
        query = query.where(AgentJobTemplate.category == category)

    query = query.order_by(AgentJobTemplate.is_system.desc(), AgentJobTemplate.name)

    result = await db.execute(query)
    templates = result.scalars().all()

    builtin = list_builtin_agent_job_templates(category)

    return AgentJobTemplateListResponse(
        templates=[
            AgentJobTemplateResponse.model_validate(t)
            for t in templates
        ]
        + [
            AgentJobTemplateResponse(
                id=t.id,
                name=t.name,
                display_name=t.display_name,
                description=t.description,
                category=t.category,
                job_type=t.job_type,
                default_goal=t.default_goal,
                default_config=t.default_config,
                default_chain_config=t.default_chain_config,
                agent_definition_id=t.agent_definition_id,
                default_max_iterations=t.default_max_iterations,
                default_max_tool_calls=t.default_max_tool_calls,
                default_max_llm_calls=t.default_max_llm_calls,
                default_max_runtime_minutes=t.default_max_runtime_minutes,
                is_system=t.is_system,
                is_active=t.is_active,
                owner_user_id=t.owner_user_id,
                created_at=t.created_at,
                updated_at=t.updated_at,
            )
            for t in builtin
        ],
        total=len(templates) + len(builtin),
    )


# ============================================================================
# Chain Definition Endpoints
#
# IMPORTANT: Keep these static routes above `/{job_id}`. FastAPI matches routes
# in declaration order, and `/{job_id}` would otherwise capture "/chains".
# ============================================================================

@router.get("/chains", response_model=AgentJobChainDefinitionListResponse)
async def list_chain_definitions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    List available job chain definitions.

    Returns system chains and user's own chains.
    """
    query = select(AgentJobChainDefinition).where(
        and_(
            AgentJobChainDefinition.is_active == True,
            or_(
                AgentJobChainDefinition.is_system == True,
                AgentJobChainDefinition.owner_user_id == current_user.id,
            )
        )
    )
    query = query.order_by(AgentJobChainDefinition.is_system.desc(), AgentJobChainDefinition.name)

    result = await db.execute(query)
    chains = result.scalars().all()
    builtin = list_builtin_agent_job_chain_definitions()

    return AgentJobChainDefinitionListResponse(
        chains=[AgentJobChainDefinitionResponse.model_validate(c) for c in chains]
        + [AgentJobChainDefinitionResponse.model_validate(c) for c in builtin],
        total=len(chains) + len(builtin),
    )


@router.post("/chains", response_model=AgentJobChainDefinitionResponse, status_code=status.HTTP_201_CREATED)
async def create_chain_definition(
    chain_create: AgentJobChainDefinitionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Create a new job chain definition.

    Chain definitions can be used to create multi-step job sequences.
    """
    # Check for duplicate name
    existing = await db.execute(
        select(AgentJobChainDefinition).where(AgentJobChainDefinition.name == chain_create.name)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chain definition with this name already exists",
        )

    # Convert chain_steps to list of dicts
    chain_steps = [step.model_dump() for step in chain_create.chain_steps]

    chain = AgentJobChainDefinition(
        name=chain_create.name,
        display_name=chain_create.display_name,
        description=chain_create.description,
        chain_steps=chain_steps,
        default_settings=chain_create.default_settings,
        owner_user_id=current_user.id,
        is_system=False,
        is_active=True,
    )

    db.add(chain)
    await db.commit()
    await db.refresh(chain)

    logger.info(f"Created chain definition {chain.id} for user {current_user.id}")

    return AgentJobChainDefinitionResponse.model_validate(chain)


@router.get("/chains/{chain_id}", response_model=AgentJobChainDefinitionResponse)
async def get_chain_definition(
    chain_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get a specific chain definition.
    """
    builtin = get_builtin_agent_job_chain_definition(chain_id)
    chain = None
    if builtin is None:
        result = await db.execute(
            select(AgentJobChainDefinition).where(
                and_(
                    AgentJobChainDefinition.id == chain_id,
                    or_(
                        AgentJobChainDefinition.is_system == True,
                        AgentJobChainDefinition.owner_user_id == current_user.id,
                    )
                )
            )
        )
        chain = result.scalar_one_or_none()
    else:
        chain = builtin

    if not chain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chain definition not found",
        )

    return AgentJobChainDefinitionResponse.model_validate(chain)


@router.patch("/chains/{chain_id}", response_model=AgentJobChainDefinitionResponse)
async def update_chain_definition(
    chain_id: UUID,
    chain_update: AgentJobChainDefinitionUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Update a chain definition.

    Only the owner can update non-system chains.
    """
    result = await db.execute(
        select(AgentJobChainDefinition).where(
            and_(
                AgentJobChainDefinition.id == chain_id,
                AgentJobChainDefinition.owner_user_id == current_user.id,
                AgentJobChainDefinition.is_system == False,
            )
        )
    )
    chain = result.scalar_one_or_none()

    if not chain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chain definition not found or not editable",
        )

    # Apply updates
    update_data = chain_update.model_dump(exclude_unset=True)
    if "chain_steps" in update_data and update_data["chain_steps"]:
        update_data["chain_steps"] = [step.model_dump() for step in chain_update.chain_steps]

    for field, value in update_data.items():
        setattr(chain, field, value)

    chain.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(chain)

    return AgentJobChainDefinitionResponse.model_validate(chain)


@router.delete("/chains/{chain_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chain_definition(
    chain_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Delete a chain definition.

    Only the owner can delete non-system chains.
    """
    result = await db.execute(
        select(AgentJobChainDefinition).where(
            and_(
                AgentJobChainDefinition.id == chain_id,
                AgentJobChainDefinition.owner_user_id == current_user.id,
                AgentJobChainDefinition.is_system == False,
            )
        )
    )
    chain = result.scalar_one_or_none()

    if not chain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chain definition not found or not deletable",
        )

    await db.delete(chain)
    await db.commit()


@router.post("/from-chain", response_model=AgentJobResponse, status_code=status.HTTP_201_CREATED)
async def create_job_from_chain(
    request: AgentJobFromChainCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Create and start a job chain from a chain definition.

    Creates the first job in the chain. Subsequent jobs will be created
    automatically as each job completes based on trigger conditions.
    """
    # Load chain definition (builtin first, then DB)
    builtin = get_builtin_agent_job_chain_definition(request.chain_definition_id)
    chain = None
    if builtin is None:
        result = await db.execute(
            select(AgentJobChainDefinition).where(
                and_(
                    AgentJobChainDefinition.id == request.chain_definition_id,
                    AgentJobChainDefinition.is_active == True,
                    or_(
                        AgentJobChainDefinition.is_system == True,
                        AgentJobChainDefinition.owner_user_id == current_user.id,
                    )
                )
            )
        )
        chain = result.scalar_one_or_none()
    else:
        chain = builtin

    if not chain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chain definition not found or not active",
        )

    if not chain.chain_steps:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chain definition has no steps",
        )


@router.get("/{job_id}", response_model=AgentJobDetailResponse)
async def get_agent_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get details of a specific agent job.

    Includes full execution log.
    """
    result = await db.execute(
        select(AgentJob)
        .options(selectinload(AgentJob.agent_definition))
        .where(and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id))
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    response = _job_to_response(job)
    return AgentJobDetailResponse(
        **response.model_dump(),
        execution_log=job.execution_log,
    )


@router.get("/{job_id}/ai-hub/recommendation-feedback", response_model=AIHubRecommendationFeedbackListResponse)
async def list_ai_hub_recommendation_feedback(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List feedback entries for this AI Scientist job."""
    job_result = await db.execute(
        select(AgentJob).where(
            and_(
                AgentJob.id == job_id,
                AgentJob.user_id == current_user.id,
            )
        )
    )
    job = job_result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Agent job not found")

    res = await db.execute(
        select(AIHubRecommendationFeedback)
        .where(
            and_(
                AIHubRecommendationFeedback.agent_job_id == job_id,
                AIHubRecommendationFeedback.user_id == current_user.id,
            )
        )
        .order_by(AIHubRecommendationFeedback.created_at.desc())
    )
    items = res.scalars().all()
    return AIHubRecommendationFeedbackListResponse(
        items=[AIHubRecommendationFeedbackResponse.model_validate(x) for x in items],
        total=len(items),
    )


@router.post("/{job_id}/ai-hub/recommendation-feedback", response_model=AIHubRecommendationFeedbackResponse)
async def create_ai_hub_recommendation_feedback(
    job_id: UUID,
    payload: AIHubRecommendationFeedbackCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create an accept/reject feedback entry for an AI Scientist recommendation."""
    job_result = await db.execute(
        select(AgentJob).where(
            and_(
                AgentJob.id == job_id,
                AgentJob.user_id == current_user.id,
            )
        )
    )
    job = job_result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Agent job not found")

    customer_profile_name = None
    customer_profile_id = None
    customer_keywords = None
    raw_profile = await get_feature_str("ai_hub_customer_profile")
    if raw_profile:
        try:
            cp = CustomerProfile.model_validate(json.loads(raw_profile))
            customer_profile_id = cp.id
            customer_profile_name = cp.name
            customer_keywords = cp.keywords
        except Exception:
            pass

    item_id = payload.item_id.strip()
    if not item_id:
        raise HTTPException(status_code=400, detail="item_id required")

    # De-dupe: if user already provided feedback for this exact tuple in this job, update the latest entry.
    existing = await db.execute(
        select(AIHubRecommendationFeedback)
        .where(
            and_(
                AIHubRecommendationFeedback.agent_job_id == job_id,
                AIHubRecommendationFeedback.user_id == current_user.id,
                AIHubRecommendationFeedback.workflow == payload.workflow,
                AIHubRecommendationFeedback.item_type == payload.item_type,
                AIHubRecommendationFeedback.item_id == item_id,
            )
        )
        .order_by(AIHubRecommendationFeedback.created_at.desc())
        .limit(1)
    )
    row = existing.scalar_one_or_none()

    if row:
        row.decision = payload.decision
        row.reason = payload.reason
        row.customer_profile_id = customer_profile_id
        row.customer_profile_name = customer_profile_name
        row.customer_keywords = customer_keywords
        await db.commit()
        await db.refresh(row)
        return AIHubRecommendationFeedbackResponse.model_validate(row)

    fb = AIHubRecommendationFeedback(
        user_id=current_user.id,
        agent_job_id=job_id,
        customer_profile_id=customer_profile_id,
        customer_profile_name=customer_profile_name,
        customer_keywords=customer_keywords,
        workflow=payload.workflow,
        item_type=payload.item_type,
        item_id=item_id,
        decision=payload.decision,
        reason=payload.reason,
    )
    db.add(fb)
    await db.commit()
    await db.refresh(fb)
    return AIHubRecommendationFeedbackResponse.model_validate(fb)


@router.patch("/{job_id}", response_model=AgentJobResponse)
async def update_agent_job(
    job_id: UUID,
    job_update: AgentJobUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Update an agent job.

    Only pending or paused jobs can be updated.
    """
    result = await db.execute(
        select(AgentJob)
        .options(selectinload(AgentJob.agent_definition))
        .where(and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id))
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    if job.status not in [AgentJobStatus.PENDING.value, AgentJobStatus.PAUSED.value]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot update job in status: {job.status}",
        )

    # Apply updates
    update_data = job_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(job, field, value)

    await db.commit()
    await db.refresh(job)

    return _job_to_response(job)


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Delete an agent job.

    Running jobs must be cancelled first.
    """
    result = await db.execute(
        select(AgentJob).where(
            and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id)
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    if job.status == AgentJobStatus.RUNNING.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete running job. Cancel it first.",
        )

    # Delete checkpoints
    await db.execute(
        select(AgentJobCheckpoint).where(AgentJobCheckpoint.job_id == job_id)
    )

    await db.delete(job)
    await db.commit()


@router.post("/{job_id}/action", response_model=AgentJobResponse)
async def job_action(
    job_id: UUID,
    request: AgentJobActionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Perform an action on an agent job.

    Actions: pause, resume, cancel, restart
    """
    result = await db.execute(
        select(AgentJob)
        .options(selectinload(AgentJob.agent_definition))
        .where(and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id))
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    action = request.action.lower()

    if action == "pause":
        if job.status != AgentJobStatus.RUNNING.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Can only pause running jobs",
            )
        job.status = AgentJobStatus.PAUSED.value
        job.add_log_entry({"phase": "paused", "reason": "user_request"})

    elif action == "resume":
        if job.status != AgentJobStatus.PAUSED.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Can only resume paused jobs",
            )
        results = job.results if isinstance(job.results, dict) else {}
        execution = results.get("execution_strategy") if isinstance(results.get("execution_strategy"), dict) else {}
        approval = execution.get("approval_checkpoints") if isinstance(execution.get("approval_checkpoints"), dict) else {}
        pending_checkpoint = (
            approval.get("pending")
            if isinstance(approval.get("pending"), dict)
            else (results.get("approval_checkpoint") if isinstance(results.get("approval_checkpoint"), dict) else None)
        )
        if pending_checkpoint:
            approval["pending"] = None
            approvals = approval.get("approvals") if isinstance(approval.get("approvals"), list) else []
            approvals.append(
                {
                    "at": datetime.utcnow().isoformat(),
                    "approved_by": str(current_user.id),
                    "method": "resume_action",
                    "checkpoint": {
                        "iteration": int(pending_checkpoint.get("iteration", 0) or 0),
                        "action_tool": str(((pending_checkpoint.get("action") or {}).get("tool") or "")).strip(),
                    },
                }
            )
            approval["approvals"] = approvals[-50:]
            execution["approval_checkpoints"] = approval
            results["execution_strategy"] = execution
            results["approval_checkpoint"] = None
            job.results = results
            job.add_log_entry(
                {
                    "phase": "approval_checkpoint_approved",
                    "reason": "resume_action",
                    "action_tool": str(((pending_checkpoint.get("action") or {}).get("tool") or "")).strip(),
                }
            )
        job.status = AgentJobStatus.PENDING.value
        job.add_log_entry({"phase": "resumed", "reason": "user_request"})
        # Queue for execution
        execute_agent_job_task.delay(str(job.id), str(current_user.id))

    elif action == "cancel":
        if job.status not in [
            AgentJobStatus.PENDING.value,
            AgentJobStatus.RUNNING.value,
            AgentJobStatus.PAUSED.value,
        ]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel job in status: {job.status}",
            )
        job.status = AgentJobStatus.CANCELLED.value
        job.completed_at = datetime.utcnow()
        job.add_log_entry({"phase": "cancelled", "reason": "user_request"})

    elif action == "restart":
        if job.status not in [
            AgentJobStatus.COMPLETED.value,
            AgentJobStatus.FAILED.value,
            AgentJobStatus.CANCELLED.value,
        ]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Can only restart completed, failed, or cancelled jobs",
            )
        # Reset job state
        job.status = AgentJobStatus.PENDING.value
        job.progress = 0
        job.iteration = 0
        job.tool_calls_used = 0
        job.llm_calls_used = 0
        job.tokens_used = 0
        job.error = None
        job.error_count = 0
        job.started_at = None
        job.completed_at = None
        job.current_phase = None
        job.phase_details = None
        job.execution_log = []
        job.results = None
        job.output_artifacts = None
        job.add_log_entry({"phase": "restarted", "reason": "user_request"})
        # Queue for execution
        execute_agent_job_task.delay(str(job.id), str(current_user.id))

    elif action == "generate_summary":
        if job.status != AgentJobStatus.COMPLETED.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Can only generate summary for completed jobs",
            )
        generate_job_summary.delay(str(job.id))

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown action: {action}. Valid actions: pause, resume, cancel, restart, generate_summary",
        )

    await db.commit()
    await db.refresh(job)

    return _job_to_response(job)


@router.get("/{job_id}/log")
async def get_job_log(
    job_id: UUID,
    limit: int = Query(50, ge=1, le=500, description="Number of log entries"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get execution log for an agent job.

    Returns paginated log entries.
    """
    result = await db.execute(
        select(AgentJob).where(
            and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id)
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    log = job.execution_log or []
    total = len(log)

    # Apply pagination
    paginated_log = log[offset:offset + limit]

    return {
        "entries": paginated_log,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + limit) < total,
    }


@router.get("/{job_id}/export")
async def export_job_results(
    job_id: UUID,
    format: str = Query("docx", description="Export format: docx, pdf, or pptx"),
    style: str = Query("professional", description="Visual style: professional, technical, or casual"),
    include_log: bool = Query(False, description="Include execution log in export"),
    include_metadata: bool = Query(True, description="Include job metadata in export"),
    enhance: bool = Query(False, description="Use LLM to generate executive summary and insights"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Export agent job results to a document or presentation.

    Supported formats:
    - docx: Microsoft Word document
    - pdf: PDF document
    - pptx: PowerPoint presentation

    When enhance=true, uses LLM to generate:
    - Executive summary
    - Key insights
    - Recommendations

    Returns the file as a downloadable attachment.
    """
    from app.services.job_results_exporter import JobResultsExporter

    # Validate format
    if format not in ["docx", "pdf", "pptx"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format: {format}. Use docx, pdf, or pptx.",
        )

    # Validate style
    if style not in ["professional", "technical", "casual"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported style: {style}. Use professional, technical, or casual.",
        )

    # Get job
    result = await db.execute(
        select(AgentJob).where(
            and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id)
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    # Generate export
    try:
        exporter = JobResultsExporter(style=style)
        if enhance:
            # Apply per-user LLM settings to enhanced export generation.
            user_settings = None
            try:
                from app.models.memory import UserPreferences
                from app.services.llm_service import UserLLMSettings
                prefs_res = await db.execute(select(UserPreferences).where(UserPreferences.user_id == current_user.id))
                prefs = prefs_res.scalar_one_or_none()
                user_settings = UserLLMSettings.from_preferences(prefs) if prefs else None
            except Exception:
                user_settings = None
            # Use async LLM-enhanced export
            file_bytes = await exporter.export_enhanced(
                job=job,
                format=format,
                include_log=include_log,
                include_metadata=include_metadata,
                user_id=current_user.id,
                user_settings=user_settings,
            )
        else:
            # Use standard export
            file_bytes = exporter.export(
                job=job,
                format=format,
                include_log=include_log,
                include_metadata=include_metadata,
            )
    except Exception as e:
        logger.error(f"Failed to export job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export failed: {str(e)}",
        )

    # Determine content type and filename
    content_types = {
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pdf": "application/pdf",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }

    # Sanitize job name for filename
    safe_name = "".join(c for c in job.name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_name = safe_name[:50] or "agent_job"
    filename = f"{safe_name}_report.{format}"

    logger.info(f"Exported job {job_id} as {format} for user {current_user.id}")

    return Response(
        content=file_bytes,
        media_type=content_types[format],
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@router.get("/{job_id}/checkpoints", response_model=list[AgentJobCheckpointResponse])
async def get_job_checkpoints(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get checkpoints for an agent job.

    Useful for debugging and understanding job progress.
    """
    # Verify job belongs to user
    job_result = await db.execute(
        select(AgentJob).where(
            and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id)
        )
    )
    job = job_result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    result = await db.execute(
        select(AgentJobCheckpoint)
        .where(AgentJobCheckpoint.job_id == job_id)
        .order_by(AgentJobCheckpoint.created_at.desc())
    )
    checkpoints = result.scalars().all()

    return [AgentJobCheckpointResponse.model_validate(cp) for cp in checkpoints]


@router.websocket("/{job_id}/progress")
async def agent_job_progress_websocket(
    websocket: WebSocket,
    job_id: str,
    token: str = Query(...),
):
    """
    WebSocket endpoint for real-time agent job progress updates.

    Subscribes to Redis pub/sub for progress updates from the Celery worker.
    """
    from app.core.config import settings
    from app.api.endpoints.auth import get_user_from_token
    from app.core.database import async_session_factory
    from app.utils.websocket_manager import websocket_manager

    # Authenticate
    try:
        user = await get_user_from_token(token)
        if not user:
            await websocket.close(code=4001, reason="Invalid token")
            return
    except Exception:
        await websocket.close(code=4001, reason="Authentication failed")
        return

    # Verify job ownership
    async with async_session_factory() as db:
        job_result = await db.execute(
            select(AgentJob).where(
                and_(AgentJob.id == UUID(job_id), AgentJob.user_id == user.id)
            )
        )
        job = job_result.scalar_one_or_none()

        if not job:
            await websocket.accept()
            await websocket.send_json({"type": "error", "error": "Job not found"})
            await websocket.close(code=4004, reason="Job not found")
            return

    # Accept connection
    await websocket.accept()

    # Send initial state
    await websocket.send_json({
        "type": "connected",
        "job_id": job_id,
        "status": job.status,
        "progress": job.progress,
    })

    # Connect to Redis pub/sub
    redis_client = None
    pubsub = None

    try:
        redis_client = redis.from_url(settings.REDIS_URL)
        pubsub = redis_client.pubsub()
        channel = f"agent_job:{job_id}:progress"

        await pubsub.subscribe(channel)
        logger.info(f"WebSocket subscribed to {channel}")

        # Listen for messages
        while True:
            try:
                # Check for WebSocket messages (ping/close)
                try:
                    msg = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=0.1
                    )
                    if msg == "ping":
                        await websocket.send_text("pong")
                except asyncio.TimeoutError:
                    pass

                # Check for Redis messages
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0
                )

                if message and message["type"] == "message":
                    data = json.loads(message["data"])
                    await websocket.send_json(data)

                    # Close on completion
                    if data.get("status") in ["completed", "failed", "cancelled"]:
                        logger.info(f"Job {job_id} finished, closing WebSocket")
                        break

            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for job {job_id}")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket loop: {e}")
                break

    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except:
            pass

    finally:
        # Cleanup
        if pubsub:
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.close()
            except:
                pass
        if redis_client:
            try:
                await redis_client.close()
            except:
                pass
        try:
            await websocket.close()
        except:
            pass


    # Get first step configuration
    job_config = chain.create_job_config_for_step(
        step_index=0,
        variables=request.variables,
        parent_results=None,
    )

    if not job_config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create job configuration from chain",
        )

    # Merge with config overrides
    default_settings = chain.default_settings or {}
    if request.config_overrides:
        default_settings.update(request.config_overrides)

    # Build chain config for next step trigger
    chain_config = None
    if len(chain.chain_steps) > 1:
        next_step = chain.chain_steps[1]
        chain_config = {
            "trigger_condition": chain.chain_steps[0].get("trigger_condition", "on_complete"),
            "inherit_results": default_settings.get("inherit_results", True),
            "chain_definition_id": str(chain.id),
            "current_step_index": 0,
            "total_steps": len(chain.chain_steps),
            "variables": request.variables,
            "child_jobs": [{
                "name": f"{request.name_prefix} - {next_step.get('step_name', 'Step 2')}",
                "job_type": next_step.get("job_type", "custom"),
                "goal": _substitute_variables(next_step.get("goal_template", ""), request.variables),
                "config": {**default_settings, **next_step.get("config", {})},
                "chain_config": _build_chain_config_for_step(chain, 1, request.variables, default_settings) if len(chain.chain_steps) > 2 else None,
            }],
        }
        if chain.chain_steps[0].get("trigger_thresholds"):
            chain_config.update(chain.chain_steps[0]["trigger_thresholds"])

    # Create the first job
    job = AgentJob(
        name=f"{request.name_prefix} - {job_config.get('name', 'Step 1')}",
        description=f"Chain: {chain.display_name}",
        job_type=job_config.get("job_type", "custom"),
        goal=job_config.get("goal", ""),
        config={**default_settings, **job_config.get("config", {})},
        user_id=current_user.id,
        status=AgentJobStatus.PENDING.value,
        max_iterations=default_settings.get("max_iterations", 100),
        max_tool_calls=default_settings.get("max_tool_calls", 500),
        max_llm_calls=default_settings.get("max_llm_calls", 200),
        max_runtime_minutes=default_settings.get("max_runtime_minutes", 60),
        chain_config=chain_config,
        chain_depth=0,
    )

    db.add(job)
    await db.commit()
    await db.refresh(job)

    logger.info(f"Created chain job {job.id} from chain {chain.name} for user {current_user.id}")

    # Start immediately if requested
    if request.start_immediately:
        execute_agent_job_task.delay(str(job.id), str(current_user.id))

    return _job_to_response(job)


@router.get("/{job_id}/chain-status", response_model=AgentJobChainStatusResponse)
async def get_chain_status(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get the status of an entire job chain.

    Works with any job in the chain - finds the root and returns full chain status.
    """
    # Get the job
    result = await db.execute(
        select(AgentJob)
        .options(selectinload(AgentJob.agent_definition))
        .where(and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id))
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    # Find root job
    root_job_id = job.root_job_id or job.id

    # Get all jobs in chain
    chain_result = await db.execute(
        select(AgentJob)
        .options(selectinload(AgentJob.agent_definition))
        .where(
            and_(
                AgentJob.user_id == current_user.id,
                or_(
                    AgentJob.id == root_job_id,
                    AgentJob.root_job_id == root_job_id,
                )
            )
        )
        .order_by(AgentJob.chain_depth, AgentJob.created_at)
    )
    chain_jobs = chain_result.scalars().all()

    # Calculate chain status
    total_steps = len(chain_jobs)
    completed_steps = len([j for j in chain_jobs if j.status == AgentJobStatus.COMPLETED.value])
    failed_jobs = [j for j in chain_jobs if j.status == AgentJobStatus.FAILED.value]
    running_jobs = [j for j in chain_jobs if j.status == AgentJobStatus.RUNNING.value]

    # Determine current step
    current_step = 0
    for i, j in enumerate(chain_jobs):
        if j.status in [AgentJobStatus.RUNNING.value, AgentJobStatus.PENDING.value]:
            current_step = i
            break
        elif j.status == AgentJobStatus.COMPLETED.value:
            current_step = i + 1

    # Calculate overall progress
    overall_progress = 0
    if total_steps > 0:
        for j in chain_jobs:
            overall_progress += j.progress
        overall_progress = overall_progress // total_steps

    # Determine chain status
    if failed_jobs:
        chain_status = "failed"
    elif completed_steps == total_steps:
        chain_status = "completed"
    elif running_jobs:
        chain_status = "running"
    elif completed_steps > 0:
        chain_status = "partially_completed"
    else:
        chain_status = "pending"

    # Get chain definition ID if available
    chain_definition_id = None
    if chain_jobs and chain_jobs[0].chain_config:
        chain_definition_id = chain_jobs[0].chain_config.get("chain_definition_id")

    return AgentJobChainStatusResponse(
        root_job_id=root_job_id,
        chain_definition_id=UUID(chain_definition_id) if chain_definition_id else None,
        total_steps=total_steps,
        completed_steps=completed_steps,
        current_step=current_step,
        overall_progress=overall_progress,
        status=chain_status,
        jobs=[_job_to_response(j) for j in chain_jobs],
    )


@router.post("/{job_id}/save-as-chain", response_model=AgentJobChainDefinitionResponse, status_code=status.HTTP_201_CREATED)
async def save_job_as_chain_definition(
    job_id: UUID,
    request: AgentJobSaveAsChainRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Save a completed (or in-progress) job chain as a reusable chain definition ("playbook").

    This primarily uses `chain_config.child_jobs` to reconstruct a linear chain, falling back
    to the persisted job hierarchy when needed.
    """
    result = await db.execute(
        select(AgentJob)
        .options(selectinload(AgentJob.agent_definition))
        .where(and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id))
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent job not found")

    # Prefer the root job if available
    root_job_id = job.root_job_id or job.id
    root_job = await db.get(AgentJob, root_job_id)
    if root_job is None:
        root_job = job

    def _safe_step_name(name: str) -> str:
        s = (name or "").strip()
        return s[:200] if s else "Step"

    def _step_from_payload(payload: dict, chain_cfg: Optional[dict]) -> dict:
        trig = (chain_cfg or {}).get("trigger_condition") or "on_complete"
        thresholds = None
        if isinstance(chain_cfg, dict) and isinstance(chain_cfg.get("progress_threshold"), int):
            thresholds = {"progress_threshold": int(chain_cfg["progress_threshold"])}
        if isinstance(chain_cfg, dict) and isinstance(chain_cfg.get("findings_threshold"), int):
            thresholds = {**(thresholds or {}), "findings_threshold": int(chain_cfg["findings_threshold"])}

        step = {
            "step_name": _safe_step_name(str(payload.get("name") or "Step")),
            "template_id": None,
            "job_type": str(payload.get("job_type") or "custom"),
            "goal_template": str(payload.get("goal") or ""),
            "config": payload.get("config") if isinstance(payload.get("config"), dict) else None,
            "trigger_condition": str(trig),
            "trigger_thresholds": thresholds,
        }
        return step

    # Build chain steps (linear) from chain_config child_jobs nesting
    steps: list[dict] = []
    cur_payload: dict = {
        "name": root_job.name,
        "job_type": root_job.job_type,
        "goal": root_job.goal,
        "config": root_job.config,
    }
    cur_chain_cfg: Optional[dict] = root_job.chain_config if isinstance(root_job.chain_config, dict) else None
    seen = set()

    while True:
        steps.append(_step_from_payload(cur_payload, cur_chain_cfg))
        if len(steps) >= 25:
            break

        child_jobs = (cur_chain_cfg or {}).get("child_jobs") if isinstance(cur_chain_cfg, dict) else None
        if not isinstance(child_jobs, list) or not child_jobs:
            break

        child0 = child_jobs[0]
        if not isinstance(child0, dict):
            break
        key = json.dumps(child0, sort_keys=True, default=str)[:2000]
        if key in seen:
            break
        seen.add(key)

        cur_payload = child0
        cur_chain_cfg = child0.get("chain_config") if isinstance(child0.get("chain_config"), dict) else None

    # If we only captured one step and the job tree exists in DB, attempt to extend using persisted hierarchy.
    if len(steps) <= 1:
        chain_result = await db.execute(
            select(AgentJob)
            .where(
                and_(
                    AgentJob.user_id == current_user.id,
                    or_(
                        AgentJob.id == root_job_id,
                        AgentJob.root_job_id == root_job_id,
                    ),
                )
            )
            .order_by(AgentJob.chain_depth, AgentJob.created_at)
        )
        chain_jobs = list(chain_result.scalars().all())
        if chain_jobs:
            by_parent: dict[UUID, list[AgentJob]] = {}
            for j in chain_jobs:
                if j.parent_job_id:
                    by_parent.setdefault(j.parent_job_id, []).append(j)
            for kids in by_parent.values():
                kids.sort(key=lambda x: (x.created_at or datetime.utcnow(), str(x.id)))

            linear: list[AgentJob] = []
            cur = chain_jobs[0]
            linear.append(cur)
            while len(linear) < 25:
                kids = by_parent.get(cur.id) or []
                if not kids:
                    break
                cur = kids[0]
                linear.append(cur)

            steps = []
            for j in linear:
                cfg = j.chain_config if isinstance(j.chain_config, dict) else None
                steps.append(
                    _step_from_payload(
                        {"name": j.name, "job_type": j.job_type, "goal": j.goal, "config": j.config},
                        cfg,
                    )
                )

    # Ensure last step doesn't imply a trigger if no child exists (cosmetic)
    if steps:
        steps[-1]["trigger_condition"] = "on_complete"
        steps[-1]["trigger_thresholds"] = None

    now = datetime.utcnow()

    def _slugify(s: str) -> str:
        s2 = re.sub(r"[^a-z0-9_]+", "_", (s or "").strip().lower())
        s2 = re.sub(r"_+", "_", s2).strip("_")
        return s2[:40] or "job"

    requested_name = (request.name or "").strip() if request.name else ""
    if requested_name:
        existing = await db.execute(select(AgentJobChainDefinition).where(AgentJobChainDefinition.name == requested_name))
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Chain definition name already exists")
        name = requested_name
    else:
        base = _slugify(root_job.name)
        name = f"playbook_{base}_{now.strftime('%Y%m%d_%H%M%S')}"
        name = name[:100]
        # Ensure uniqueness
        for _ in range(5):
            existing = await db.execute(select(AgentJobChainDefinition).where(AgentJobChainDefinition.name == name))
            if not existing.scalar_one_or_none():
                break
            name = (name[:90] + "_" + uuid.uuid4().hex[:8])[:100]

    display_name = (request.display_name or "").strip() if request.display_name else ""
    if not display_name:
        display_name = f"{root_job.name} (Playbook)"

    description = (request.description or "").strip() if request.description else ""
    if not description:
        description = f"Saved from job {root_job_id} on {now.isoformat()}."

    chain = AgentJobChainDefinition(
        name=name,
        display_name=display_name,
        description=description,
        chain_steps=steps,
        default_settings={
            "inherit_results": True,
            "inherit_config": True,
            "max_iterations": int(getattr(root_job, "max_iterations", 100) or 100),
            "max_tool_calls": int(getattr(root_job, "max_tool_calls", 500) or 500),
            "max_llm_calls": int(getattr(root_job, "max_llm_calls", 200) or 200),
            "max_runtime_minutes": int(getattr(root_job, "max_runtime_minutes", 60) or 60),
        },
        owner_user_id=current_user.id,
        is_system=False,
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    db.add(chain)
    await db.commit()
    await db.refresh(chain)

    return AgentJobChainDefinitionResponse.model_validate(chain)


def _substitute_variables(template: str, variables: dict) -> str:
    """Substitute {variable} placeholders in a template string."""
    result = template
    for key, value in variables.items():
        result = result.replace(f"{{{key}}}", value)
    return result


def _build_chain_config_for_step(
    chain: AgentJobChainDefinition,
    step_index: int,
    variables: dict,
    default_settings: dict,
) -> Optional[dict]:
    """Build chain config for a specific step in the chain."""
    if step_index >= len(chain.chain_steps):
        return None

    step = chain.chain_steps[step_index]

    config = {
        "trigger_condition": step.get("trigger_condition", "on_complete"),
        "inherit_results": default_settings.get("inherit_results", True),
        "chain_definition_id": str(chain.id),
        "current_step_index": step_index,
        "total_steps": len(chain.chain_steps),
        "variables": variables,
    }

    if step.get("trigger_thresholds"):
        config.update(step["trigger_thresholds"])

    # Add next step as child job
    if step_index + 1 < len(chain.chain_steps):
        next_step = chain.chain_steps[step_index + 1]
        config["child_jobs"] = [{
            "name": next_step.get("step_name", f"Step {step_index + 2}"),
            "job_type": next_step.get("job_type", "custom"),
            "goal": _substitute_variables(next_step.get("goal_template", ""), variables),
            "config": {**default_settings, **next_step.get("config", {})},
            "chain_config": _build_chain_config_for_step(chain, step_index + 1, variables, default_settings),
        }]

    return config


# ============================================================================
# Job Memory Endpoints
# ============================================================================

@router.get("/{job_id}/memories")
async def get_job_memories(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get all memories created from a specific job.

    Returns memories extracted from the job's results.
    """
    from app.services.agent_job_memory_service import agent_job_memory_service

    # Verify job belongs to user
    result = await db.execute(
        select(AgentJob).where(
            and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id)
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    memories = await agent_job_memory_service.get_job_memories(
        job_id=job_id,
        user_id=str(current_user.id),
        db=db,
    )

    return {
        "job_id": str(job_id),
        "memories": [
            {
                "id": str(m.id),
                "type": m.memory_type,
                "content": m.content,
                "importance_score": m.importance_score,
                "tags": m.tags,
                "context": m.context,
                "access_count": m.access_count,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in memories
        ],
        "total": len(memories),
    }


@router.post("/{job_id}/memories/extract")
async def extract_job_memories(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Manually trigger memory extraction from a completed job.

    Useful for re-extracting memories or extracting from older jobs.
    """
    from app.services.agent_job_memory_service import agent_job_memory_service

    # Verify job belongs to user
    result = await db.execute(
        select(AgentJob).where(
            and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id)
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    if job.status not in ["completed", "failed"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only extract memories from completed or failed jobs",
        )

    try:
        memories = await agent_job_memory_service.extract_memories_from_job(
            job=job,
            user_id=str(current_user.id),
            db=db,
        )

        logger.info(f"Manually extracted {len(memories)} memories from job {job_id}")

        return {
            "job_id": str(job_id),
            "memories_created": len(memories),
            "memories": [
                {
                    "id": str(m.id),
                    "type": m.memory_type,
                    "content": m.content,
                    "importance_score": m.importance_score,
                    "tags": m.tags,
                }
                for m in memories
            ],
        }
    except Exception as e:
        logger.error(f"Failed to extract memories from job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory extraction failed: {str(e)}",
        )


@router.post("/{job_id}/memories")
async def create_job_memory(
    job_id: UUID,
    memory_type: str = Query(..., description="Memory type: finding, insight, pattern, or lesson"),
    content: str = Query(..., description="Memory content"),
    importance: float = Query(0.5, ge=0.0, le=1.0, description="Importance score"),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Manually create a memory from a job.

    Allows users to add custom memories derived from job insights.
    """
    from app.services.agent_job_memory_service import agent_job_memory_service

    # Validate memory type
    if memory_type not in ["finding", "insight", "pattern", "lesson", "fact", "context"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid memory type. Use: finding, insight, pattern, lesson, fact, or context",
        )

    # Verify job belongs to user
    result = await db.execute(
        select(AgentJob).where(
            and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id)
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    # Parse tags
    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    try:
        memory = await agent_job_memory_service.create_memory_from_job(
            job=job,
            memory_type=memory_type,
            content=content,
            user_id=str(current_user.id),
            db=db,
            importance=importance,
            tags=tag_list,
        )

        return {
            "id": str(memory.id),
            "job_id": str(job_id),
            "type": memory.memory_type,
            "content": memory.content,
            "importance_score": memory.importance_score,
            "tags": memory.tags,
            "created_at": memory.created_at.isoformat() if memory.created_at else None,
        }
    except Exception as e:
        logger.error(f"Failed to create memory for job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory creation failed: {str(e)}",
        )


@router.delete("/{job_id}/memories")
async def delete_job_memories(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Delete all memories created from a job.

    Performs a soft delete (memories marked inactive, not removed).
    """
    from app.services.agent_job_memory_service import agent_job_memory_service

    # Verify job belongs to user
    result = await db.execute(
        select(AgentJob).where(
            and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id)
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    deleted_count = await agent_job_memory_service.delete_job_memories(
        job_id=job_id,
        user_id=str(current_user.id),
        db=db,
    )

    return {
        "job_id": str(job_id),
        "deleted_count": deleted_count,
    }


@router.post("/{job_id}/feedback", response_model=AgentJobFeedbackResponse)
async def create_agent_job_feedback(
    job_id: UUID,
    payload: AgentJobFeedbackCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Store human feedback for a job/checkpoint and convert it into learning memory.

    Feedback is persisted as a `lesson` memory tagged with `human_feedback`, then
    used by the autonomous executor to bias future tool choices and prompts.
    """
    from app.services.agent_job_memory_service import agent_job_memory_service

    job_res = await db.execute(
        select(AgentJob).where(
            and_(
                AgentJob.id == job_id,
                AgentJob.user_id == current_user.id,
            )
        )
    )
    job = job_res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Agent job not found")

    scope = str(payload.scope or "user").strip().lower()
    if scope not in {"user", "customer", "team"}:
        raise HTTPException(status_code=400, detail="scope must be one of: user, customer, team")

    preferred_tools = _sanitize_tool_names(payload.preferred_tools)
    discouraged_tools = _sanitize_tool_names(payload.discouraged_tools)
    if preferred_tools and discouraged_tools:
        overlap = [t for t in preferred_tools if t in set(discouraged_tools)]
        if overlap:
            discouraged_tools = [t for t in discouraged_tools if t not in set(overlap)]

    target_type = str(payload.target_type or "job").strip().lower()
    if target_type not in {"job", "checkpoint", "finding", "action", "tool"}:
        raise HTTPException(status_code=400, detail="target_type must be job, checkpoint, finding, action, or tool")

    team_key = str(payload.team_key or "").strip()
    if scope == "team" and not team_key:
        raise HTTPException(status_code=400, detail="team_key is required when scope=team")

    scope_marker = f"user:{current_user.id}"
    if scope == "customer":
        customer = str((job.config or {}).get("customer") or "").strip()
        if not customer:
            raw_profile = await get_feature_str("ai_hub_customer_profile")
            if raw_profile:
                try:
                    cp = CustomerProfile.model_validate(json.loads(raw_profile))
                    customer = str(cp.id or cp.name or "").strip()
                except Exception:
                    customer = ""
        if not customer:
            raise HTTPException(status_code=400, detail="customer scope requires job.config.customer or ai_hub_customer_profile")
        scope_marker = f"customer:{customer[:120]}"
    elif scope == "team":
        scope_marker = f"team:{team_key[:120]}"

    rating = max(1, min(int(payload.rating), 5))
    sentiment = "positive" if rating >= 4 else ("negative" if rating <= 2 else "neutral")
    feedback_text = str(payload.feedback or "").strip()
    target_id = str(payload.target_id or "").strip()
    checkpoint = str(payload.checkpoint or "").strip()

    content = feedback_text or f"User rated {target_type} as {rating}/5."
    importance = min(1.0, max(0.35, 0.55 + (abs(rating - 3) * 0.1)))

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
    tags.extend([f"prefer_tool:{t}" for t in preferred_tools])
    tags.extend([f"avoid_tool:{t}" for t in discouraged_tools])
    tags = list(dict.fromkeys([t for t in tags if str(t).strip()]))

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
    except Exception as graph_exc:
        logger.warning(f"Failed linking feedback memory {memory.id} into task graph: {graph_exc}")

    return _memory_to_feedback_response(memory)


@router.get("/{job_id}/feedback", response_model=AgentJobFeedbackListResponse)
async def list_agent_job_feedback(
    job_id: UUID,
    limit: int = Query(50, ge=1, le=200, description="Max feedback entries"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List human feedback entries captured for a specific job."""
    job_res = await db.execute(
        select(AgentJob).where(
            and_(
                AgentJob.id == job_id,
                AgentJob.user_id == current_user.id,
            )
        )
    )
    job = job_res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Agent job not found")

    res = await db.execute(
        select(ConversationMemory)
        .where(
            and_(
                ConversationMemory.user_id == current_user.id,
                ConversationMemory.job_id == job_id,
                ConversationMemory.is_active == True,
                ConversationMemory.memory_type == "lesson",
            )
        )
        .order_by(ConversationMemory.created_at.desc())
        .limit(max(20, limit * 3))
    )
    memories = list(res.scalars().all())
    rows = []
    for memory in memories:
        tags = [str(t).strip().lower() for t in (memory.tags if isinstance(memory.tags, list) else []) if str(t).strip()]
        context = memory.context if isinstance(memory.context, dict) else {}
        is_feedback = ("human_feedback" in tags) or str(context.get("feedback_type") or "").strip().lower() == "human"
        if is_feedback:
            rows.append(_memory_to_feedback_response(memory))
        if len(rows) >= limit:
            break
    return AgentJobFeedbackListResponse(items=rows, total=len(rows))


@router.get("/memory/feedback", response_model=AgentJobFeedbackListResponse)
async def list_learning_feedback(
    scope: Optional[str] = Query(None, description="Optional scope filter: user|customer|team"),
    limit: int = Query(100, ge=1, le=300, description="Max feedback entries"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List recent human feedback memories used by the learning loop."""
    scope_filter = str(scope or "").strip().lower()
    if scope_filter and scope_filter not in {"user", "customer", "team"}:
        raise HTTPException(status_code=400, detail="scope must be user, customer, or team")

    res = await db.execute(
        select(ConversationMemory)
        .where(
            and_(
                ConversationMemory.user_id == current_user.id,
                ConversationMemory.is_active == True,
                ConversationMemory.memory_type == "lesson",
            )
        )
        .order_by(ConversationMemory.created_at.desc())
        .limit(max(50, limit * 3))
    )
    memories = list(res.scalars().all())
    items = []
    for memory in memories:
        tags = [str(t).strip().lower() for t in (memory.tags if isinstance(memory.tags, list) else []) if str(t).strip()]
        context = memory.context if isinstance(memory.context, dict) else {}
        if "human_feedback" not in tags and str(context.get("feedback_type") or "").strip().lower() != "human":
            continue
        if scope_filter and str(context.get("scope") or "").strip().lower() != scope_filter:
            continue
        items.append(_memory_to_feedback_response(memory))
        if len(items) >= limit:
            break
    return AgentJobFeedbackListResponse(items=items, total=len(items))


@router.get("/memory/graph")
async def get_task_memory_graph(
    limit: int = Query(120, ge=20, le=300, description="Max memories to include as graph nodes"),
    min_link_score: float = Query(1.0, ge=0.2, le=10.0, description="Minimum edge score"),
    max_edges: int = Query(800, ge=50, le=3000, description="Maximum graph edges"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Return reusable cross-job task memory graph (lessons, failed paths, successful strategies)."""
    from app.services.agent_job_memory_service import agent_job_memory_service

    graph = await agent_job_memory_service.get_task_memory_graph(
        user_id=str(current_user.id),
        db=db,
        limit=limit,
        min_link_score=min_link_score,
        max_edges=max_edges,
    )
    return graph


@router.get("/{job_id}/memories/graph")
async def get_job_memory_graph(
    job_id: UUID,
    neighbor_depth: int = Query(1, ge=1, le=2, description="Neighborhood depth around this job's memory nodes"),
    limit: int = Query(180, ge=20, le=300, description="Max nodes to scan"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Return task-memory subgraph centered on memories created by a specific job."""
    from app.services.agent_job_memory_service import agent_job_memory_service

    job_res = await db.execute(
        select(AgentJob).where(
            and_(
                AgentJob.id == job_id,
                AgentJob.user_id == current_user.id,
            )
        )
    )
    job = job_res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Agent job not found")

    graph = await agent_job_memory_service.get_task_memory_graph(
        user_id=str(current_user.id),
        db=db,
        limit=limit,
        min_link_score=1.0,
        max_edges=1200,
    )
    nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []
    job_node_ids = {str(n.get("id")) for n in nodes if str(n.get("job_id") or "") == str(job_id)}
    if not job_node_ids:
        return {"job_id": str(job_id), "nodes": [], "edges": [], "stats": {"memory_count": 0, "edge_count": 0}}

    selected = set(job_node_ids)
    hops = max(1, min(int(neighbor_depth or 1), 2))
    for _ in range(hops):
        expanded = set(selected)
        for e in edges:
            src = str(e.get("source") or "")
            dst = str(e.get("target") or "")
            if src in selected or dst in selected:
                expanded.add(src)
                expanded.add(dst)
        selected = expanded

    sub_nodes = [n for n in nodes if str(n.get("id")) in selected]
    sub_edges = [e for e in edges if str(e.get("source")) in selected and str(e.get("target")) in selected]
    return {
        "job_id": str(job_id),
        "nodes": sub_nodes,
        "edges": sub_edges,
        "stats": {
            "memory_count": len(sub_nodes),
            "edge_count": len(sub_edges),
            "job_memory_count": len(job_node_ids),
            "neighbor_depth": hops,
        },
    }


@router.get("/memory/stats")
async def get_memory_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get memory statistics for the current user.

    Includes breakdown by type, source, and most accessed/important memories.
    """
    from app.services.agent_job_memory_service import agent_job_memory_service

    stats = await agent_job_memory_service.get_memory_stats_for_user(
        user_id=str(current_user.id),
        db=db,
    )

    return stats


@router.get("/memory/search")
async def search_memories(
    query: str = Query(..., description="Search query"),
    memory_types: Optional[str] = Query(None, description="Comma-separated memory types to filter"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Search memories relevant to a query.

    Returns memories ranked by relevance to the query.
    """
    from app.services.agent_job_memory_service import agent_job_memory_service
    from app.models.memory import ConversationMemory
    from sqlalchemy import desc

    # Build query
    type_list = [t.strip() for t in memory_types.split(",")] if memory_types else None

    base_query = (
        select(ConversationMemory)
        .where(
            and_(
                ConversationMemory.user_id == current_user.id,
                ConversationMemory.is_active == True,
            )
        )
    )

    if type_list:
        base_query = base_query.where(ConversationMemory.memory_type.in_(type_list))

    # Simple text search (for now, could be enhanced with vector search)
    base_query = base_query.where(
        ConversationMemory.content.ilike(f"%{query}%")
    )

    base_query = base_query.order_by(desc(ConversationMemory.importance_score)).limit(limit)

    result = await db.execute(base_query)
    memories = list(result.scalars().all())

    return {
        "query": query,
        "memories": [
            {
                "id": str(m.id),
                "type": m.memory_type,
                "content": m.content,
                "importance_score": m.importance_score,
                "tags": m.tags,
                "job_id": str(m.job_id) if m.job_id else None,
                "access_count": m.access_count,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in memories
        ],
        "total": len(memories),
    }
