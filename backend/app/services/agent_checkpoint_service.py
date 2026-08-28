"""Checkpoint persistence helpers extracted from AutonomousAgentExecutor."""

from __future__ import annotations

import inspect
import math
from typing import Any, Dict, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import select

from app.models.agent_job import AgentJobCheckpoint


def strip_non_finite(value: Any) -> Any:
    """The same structure, with values a JSON column will accept.

    Named for what it removes, not for what it protects: there is another
    `_json_safe` in the executor that converts UUIDs and datetimes, which is a
    different concern with a confusingly similar name.

    Postgres JSON has no NaN and no Infinity; Python's json.dumps emits them
    happily as bare `NaN`, so the mismatch surfaces as
    `invalid input syntax for type json: Token "NaN" is invalid` at INSERT --
    after the work is done. That killed a run at iteration 10 of 14: the state
    being checkpointed carried tool results derived from a gem5 stats file,
    and a plain gem5 run emits fourteen NaN statistics for averages whose
    denominator was zero.

    Replaced with None rather than dropped: a key vanishing from a checkpoint
    changes the shape a resume reads back, and a missing average and an
    unmeasurable one are the same claim -- there is no number here.
    """
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if isinstance(value, dict):
        return {key: strip_non_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strip_non_finite(item) for item in value]
    return value


class AgentCheckpointService:
    """Persist and load runtime checkpoints for autonomous jobs."""

    async def save_checkpoint(
        self,
        *,
        job: Any,
        state: Dict[str, Any],
        db: Any,
        reason: str = "runtime_checkpoint",
    ) -> None:
        """Save a checkpoint for job resumption."""
        checkpoint = AgentJobCheckpoint(
            job_id=job.id,
            iteration=job.iteration,
            phase=job.current_phase,
            state=strip_non_finite(state),
            context=strip_non_finite(
                {
                    "progress": job.progress,
                    "reason": reason,
                    "journal_cursor": state.get("execution_journal_cursor"),
                }
            ),
        )
        add_result = db.add(checkpoint)
        if inspect.isawaitable(add_result):
            await add_result
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
