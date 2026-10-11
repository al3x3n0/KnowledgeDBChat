"""The jobs an operator queue could show, asked of the database.

The queue is built by projecting each job into a row or into nothing
(`checkpoint_queue_jobs`), and most jobs become nothing: of everything a
person has run, a handful are waiting on them. Both callers still loaded
every job, whole, on each call -- results and execution log included.

A job can become a row for four reasons. Two the database can see directly
(it is paused on a person; it is a recurring job that failed, paused or went
quiet). Two are inside JSON (an approval checkpoint in `results`; a
contract-unmet entry in the log), and two flags on `AgentJob` carry those. This selects the jobs any of the four could apply to. **It may select
too many, never too few**: the projector still decides, from the row, what
each job becomes. `tests/test_checkpoint_queue_candidates.py` builds the
queue both ways over every combination of state and requires the same rows.

Jobs named in `also` are included whatever their state: the monitor
snapshot looks up the job behind each accepted inbox item.
"""

from __future__ import annotations

from typing import Iterable, List
from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.agent_job import AgentJob, AgentJobStatus
from app.modules.autonomy.application.checkpoint_queue_jobs import BLOCKED_PHASE

#: As the projector reads them: lowered and stripped.
_RECURRING = ("recurring", "continuous")
_RECOVERABLE = (
    AgentJobStatus.FAILED.value,
    AgentJobStatus.PAUSED.value,
    AgentJobStatus.RUNNING.value,
)


def could_become_a_row():
    """The condition a job must meet to be worth projecting."""
    status = func.lower(func.trim(func.coalesce(AgentJob.status, "")))
    schedule = func.lower(func.trim(func.coalesce(AgentJob.schedule_type, "")))
    phase = func.trim(func.coalesce(AgentJob.current_phase, ""))
    return or_(
        AgentJob.has_approval_checkpoint.is_(True),
        AgentJob.has_contract_unmet_entry.is_(True),
        and_(status == AgentJobStatus.PAUSED.value, phase == BLOCKED_PHASE),
        and_(schedule.in_(_RECURRING), status.in_(_RECOVERABLE)),
    )


async def load_queue_jobs(
    db: AsyncSession, user_id: UUID, *, also: Iterable[UUID] = ()
) -> List[AgentJob]:
    """One person's jobs that could appear in their queue, newest first."""
    wanted = could_become_a_row()
    named = [job_id for job_id in also if job_id is not None]
    if named:
        wanted = or_(wanted, AgentJob.id.in_(named))
    result = await db.execute(
        select(AgentJob)
        .options(selectinload(AgentJob.agent_definition))
        .where(AgentJob.user_id == user_id, wanted)
        .order_by(AgentJob.created_at.desc())
    )
    return list(result.scalars().all())


__all__ = ["could_become_a_row", "load_queue_jobs"]
