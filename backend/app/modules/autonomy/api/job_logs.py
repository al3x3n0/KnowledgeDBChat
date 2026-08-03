"""Owned-job execution-log query and pagination boundary."""

from dataclasses import dataclass
from typing import Any, Callable
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.agent_job import AgentJob
from app.models.user import User


@dataclass(frozen=True)
class JobLogApi:
    router: APIRouter
    get_job_log: Callable[..., Any]


def build_job_log_page(
    execution_log: Any,
    *,
    offset: int,
    limit: int,
) -> dict[str, Any]:
    """Return one page from the job's ordered execution log."""
    rows = execution_log or []
    total = len(rows)
    return {
        "entries": rows[offset : offset + limit],
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + limit) < total,
    }


def build_job_log_api() -> JobLogApi:
    """Build the ownership-scoped job execution-log route."""
    router = APIRouter()

    @router.get("/{job_id}/log")
    async def get_job_log(
        job_id: UUID,
        limit: int = Query(50, ge=1, le=500, description="Number of log entries"),
        offset: int = Query(0, ge=0, description="Offset for pagination"),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ) -> dict[str, Any]:
        """Get paginated execution-log entries for an owned agent job."""
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
        return build_job_log_page(
            job.execution_log,
            offset=offset,
            limit=limit,
        )

    return JobLogApi(
        router=router,
        get_job_log=get_job_log,
    )
