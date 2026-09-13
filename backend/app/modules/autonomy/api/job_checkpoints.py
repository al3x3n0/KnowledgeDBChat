"""Owned-job checkpoint query boundary."""

from dataclasses import dataclass
from typing import Any, Callable
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.agent_job import AgentJob, AgentJobCheckpoint
from app.models.user import User
from app.schemas.agent_job import AgentJobCheckpointResponse


@dataclass(frozen=True)
class JobCheckpointApi:
    router: APIRouter
    get_job_checkpoints: Callable[..., Any]


def build_job_checkpoint_api() -> JobCheckpointApi:
    """Build the ownership-scoped checkpoint history route."""
    router = APIRouter()

    @router.get(
        "/{job_id}/checkpoints",
        response_model=list[AgentJobCheckpointResponse],
    )
    async def get_job_checkpoints(
        job_id: UUID,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ) -> list[AgentJobCheckpointResponse]:
        """Return an owned job's checkpoints from newest to oldest."""
        job_result = await db.execute(
            select(AgentJob).where(
                and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id)
            )
        )
        if job_result.scalar_one_or_none() is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent job not found",
            )

        result = await db.execute(
            select(AgentJobCheckpoint)
            .where(AgentJobCheckpoint.job_id == job_id)
            .order_by(AgentJobCheckpoint.created_at.desc())
        )
        return [
            AgentJobCheckpointResponse.model_validate(checkpoint)
            for checkpoint in result.scalars().all()
        ]

    return JobCheckpointApi(
        router=router,
        get_job_checkpoints=get_job_checkpoints,
    )
