"""Checkpoint persistence helpers extracted from AutonomousAgentExecutor."""

from __future__ import annotations

from typing import Any, Dict, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import select

from app.models.agent_job import AgentJobCheckpoint


class AgentCheckpointService:
    """Persist and load runtime checkpoints for autonomous jobs."""

    async def save_checkpoint(
        self,
        *,
        job: Any,
        state: Dict[str, Any],
        db: Any,
    ) -> None:
        """Save a checkpoint for job resumption."""
        checkpoint = AgentJobCheckpoint(
            job_id=job.id,
            iteration=job.iteration,
            phase=job.current_phase,
            state=state,
            context={"progress": job.progress},
        )
        db.add(checkpoint)
        await db.commit()
        logger.debug(f"Saved checkpoint for job {job.id} at iteration {job.iteration}")

    async def load_latest_checkpoint(
        self,
        *,
        job_id: UUID,
        db: Any,
    ) -> Optional[AgentJobCheckpoint]:
        """Load the latest checkpoint for a job."""
        result = await db.execute(
            select(AgentJobCheckpoint)
            .where(AgentJobCheckpoint.job_id == job_id)
            .order_by(AgentJobCheckpoint.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()
