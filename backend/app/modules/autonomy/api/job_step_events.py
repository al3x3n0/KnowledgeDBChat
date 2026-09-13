"""HTTP boundary and presentation policy for autonomous-job step events."""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.agent_job import AgentJob
from app.models.user import User

CheckpointLoader = Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class JobStepEventApi:
    router: APIRouter
    get_job_step_events: Callable[..., Any]


def build_step_event_page(
    *,
    job_results: Any,
    checkpoint_state: Any,
    offset: int,
    limit: int,
) -> dict[str, Any]:
    """Choose the richer event stream and return its requested page."""
    results_payload = job_results if isinstance(job_results, dict) else {}
    execution = (
        results_payload.get("execution_strategy")
        if isinstance(results_payload.get("execution_strategy"), dict)
        else {}
    )
    result_rows = (
        execution.get("step_events")
        if isinstance(execution.get("step_events"), list)
        else []
    )

    normalized_checkpoint_state = (
        checkpoint_state if isinstance(checkpoint_state, dict) else {}
    )
    checkpoint_rows = (
        normalized_checkpoint_state.get("step_events")
        if isinstance(normalized_checkpoint_state.get("step_events"), list)
        else []
    )

    if checkpoint_rows and len(checkpoint_rows) >= len(result_rows):
        source = "checkpoint_state"
        rows = checkpoint_rows
    else:
        source = "results_execution_strategy"
        rows = result_rows

    normalized_rows = [row for row in rows if isinstance(row, dict)]
    total = len(normalized_rows)
    return {
        "items": normalized_rows[offset : offset + limit],
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + limit) < total,
        "source": source,
    }


def build_job_step_event_api(
    *,
    load_latest_checkpoint: CheckpointLoader,
) -> JobStepEventApi:
    """Build the owned-job step-event query route."""
    router = APIRouter()

    @router.get("/{job_id}/step-events")
    async def get_job_step_events(
        job_id: UUID,
        limit: int = Query(100, ge=1, le=500, description="Number of step events"),
        offset: int = Query(0, ge=0, description="Offset for pagination"),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ) -> dict[str, Any]:
        """Get per-step execution and approval events for an owned job."""
        result = await db.execute(
            select(AgentJob).where(
                and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id)
            )
        )
        job = result.scalar_one_or_none()
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent job not found",
            )

        checkpoint = await load_latest_checkpoint(job.id, db)
        checkpoint_state = checkpoint.state if checkpoint is not None else None
        return build_step_event_page(
            job_results=job.results,
            checkpoint_state=checkpoint_state,
            offset=offset,
            limit=limit,
        )

    return JobStepEventApi(
        router=router,
        get_job_step_events=get_job_step_events,
    )
