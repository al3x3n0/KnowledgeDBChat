"""Persist normalized operator actions taken on autonomous jobs."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob
from app.models.user import User


@dataclass(frozen=True)
class JobOperatorEventDependencies:
    record_event: Callable[..., Awaitable[Any]]
    queue_customer_for_job: Callable[[AgentJob], str | None]
    reason_label: Callable[[str | None], str | None]
    utcnow: Callable[[], datetime] = datetime.utcnow


async def record_job_operator_event(
    *,
    db: AsyncSession,
    job: AgentJob,
    current_user: User,
    action: str,
    note: str | None,
    previous_status: str | None,
    next_status: str | None,
    deps: JobOperatorEventDependencies,
    scheduler_state: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    summary: str | None = None,
) -> None:
    """Build and persist one stable job operator-action trace event."""
    job_id = str(job.id) if job.id else None
    job_label = str(job.name or "Agent job").strip() or "Agent job"
    reason_code = str((metadata or {}).get("reason_code") or "").strip() or None
    normalized_action = str(action or "").strip().lower() or "operator_intervention"
    await deps.record_event(
        db,
        user_id=current_user.id,
        event_type="job_operator_action",
        event_time=deps.utcnow(),
        source_kind="job",
        source_id=job_id,
        source_label=job_label,
        customer=deps.queue_customer_for_job(job),
        decision_type=normalized_action,
        reason_code=reason_code,
        status=str(next_status or job.status or "").strip() or None,
        severity="medium",
        actor_mode="operator",
        summary=(
            summary
            or f"{job_label}: {str(action or 'operator action').replace('_', ' ')}"
        ),
        operator_note=note,
        reason_label=deps.reason_label(reason_code),
        scheduler_state=scheduler_state if isinstance(scheduler_state, dict) else None,
        before_state={"job_status": previous_status} if previous_status else None,
        after_state={"job_status": next_status} if next_status else None,
        deep_link={
            "target_tab": "jobs",
            "job_id": job_id,
            "params": {"job": job_id} if job_id else {},
            "label": "Open Job",
        },
        metadata=metadata or None,
    )
