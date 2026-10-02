"""The lookup that makes a job route about the caller's own job.

Three route modules each carried this query. It is the whole of their
ownership check, so a copy that drifted would have been a route answering for
somebody else's job. A job that exists but belongs to another user is reported
as not found, the same as one that does not exist.
"""

from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob


async def get_owned_job(*, job_id: UUID, user_id: UUID, db: AsyncSession) -> AgentJob:
    result = await db.execute(
        select(AgentJob).where(and_(AgentJob.id == job_id, AgentJob.user_id == user_id))
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )
    return job
