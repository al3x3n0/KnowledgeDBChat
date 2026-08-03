"""Read-model service for the status of an autonomous job chain."""

from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.agent_job import AgentJob, AgentJobStatus


class AgentChainStatusError(RuntimeError):
    """Domain error translated at the API boundary."""

    def __init__(self, detail: str, *, status_code: int) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


@dataclass(frozen=True)
class AgentChainStatusSnapshot:
    root_job_id: UUID
    chain_definition_id: Optional[UUID]
    total_steps: int
    completed_steps: int
    current_step: int
    overall_progress: int
    status: str
    jobs: list[AgentJob]


class AgentChainStatusService:
    """Load an owned chain and derive its aggregate runtime status."""

    async def get_snapshot(
        self,
        *,
        job_id: UUID,
        user_id: UUID,
        db: AsyncSession,
    ) -> AgentChainStatusSnapshot:
        job = (
            await db.execute(
                select(AgentJob)
                .options(selectinload(AgentJob.agent_definition))
                .where(
                    and_(
                        AgentJob.id == job_id,
                        AgentJob.user_id == user_id,
                    )
                )
            )
        ).scalar_one_or_none()
        if job is None:
            raise AgentChainStatusError(
                "Agent job not found",
                status_code=404,
            )

        root_job_id = job.root_job_id or job.id
        jobs = list(
            (
                await db.execute(
                    select(AgentJob)
                    .options(selectinload(AgentJob.agent_definition))
                    .where(
                        and_(
                            AgentJob.user_id == user_id,
                            or_(
                                AgentJob.id == root_job_id,
                                AgentJob.root_job_id == root_job_id,
                            ),
                        )
                    )
                    .order_by(AgentJob.chain_depth, AgentJob.created_at)
                )
            )
            .scalars()
            .all()
        )

        completed_steps = sum(
            job.status == AgentJobStatus.COMPLETED.value for job in jobs
        )
        failed = any(job.status == AgentJobStatus.FAILED.value for job in jobs)
        running = any(job.status == AgentJobStatus.RUNNING.value for job in jobs)
        current_step = self._current_step(jobs)
        total_steps = len(jobs)
        overall_progress = (
            sum(int(job.progress or 0) for job in jobs) // total_steps
            if total_steps
            else 0
        )

        if failed:
            status = "failed"
        elif completed_steps == total_steps:
            status = "completed"
        elif running:
            status = "running"
        elif completed_steps:
            status = "partially_completed"
        else:
            status = "pending"

        return AgentChainStatusSnapshot(
            root_job_id=root_job_id,
            chain_definition_id=self._chain_definition_id(jobs),
            total_steps=total_steps,
            completed_steps=completed_steps,
            current_step=current_step,
            overall_progress=overall_progress,
            status=status,
            jobs=jobs,
        )

    @staticmethod
    def _current_step(jobs: list[AgentJob]) -> int:
        current_step = 0
        for index, job in enumerate(jobs):
            if job.status in {
                AgentJobStatus.RUNNING.value,
                AgentJobStatus.PENDING.value,
            }:
                return index
            if job.status == AgentJobStatus.COMPLETED.value:
                current_step = index + 1
        return current_step

    @staticmethod
    def _chain_definition_id(jobs: list[AgentJob]) -> Optional[UUID]:
        if not jobs or not isinstance(jobs[0].chain_config, dict):
            return None
        value = jobs[0].chain_config.get("chain_definition_id")
        if not value:
            return None
        try:
            return UUID(str(value))
        except (TypeError, ValueError):
            return None


agent_chain_status_service = AgentChainStatusService()
