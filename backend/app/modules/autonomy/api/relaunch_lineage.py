"""HTTP boundary for autonomous-job relaunch lineage."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.agent_job import AgentJob
from app.models.user import User
from app.modules.autonomy.application.relaunch_lineage import build_lineage
from app.schemas.agent_job import AgentJobRelaunchLineageResponse

router = APIRouter()


@router.get(
    "/{job_id}/relaunch-lineage",
    response_model=AgentJobRelaunchLineageResponse,
)
async def get_agent_job_relaunch_lineage(
    job_id: UUID,
    ancestor_limit: int = Query(
        100,
        ge=1,
        le=300,
        description="Max ancestor nodes to include",
    ),
    descendant_limit: int = Query(
        500,
        ge=1,
        le=2000,
        description="Max descendant nodes to include",
    ),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Return user-scoped relaunch ancestry and descendants for one job."""
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

    jobs_result = await db.execute(
        select(AgentJob).where(AgentJob.user_id == current_user.id)
    )
    jobs = jobs_result.scalars().all()
    jobs_by_id = {item.id: item for item in jobs}
    return build_lineage(
        job,
        jobs_by_id,
        max_ancestors=ancestor_limit,
        max_descendants=descendant_limit,
    )
