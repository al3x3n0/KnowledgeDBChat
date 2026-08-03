"""Persist normalized operator decisions for queued follow-up work."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class FollowUpQueueEventDependencies:
    record_event: Callable[..., Awaitable[Any]]
    utcnow: Callable[[], datetime] = datetime.utcnow


async def record_follow_up_queue_decision(
    *,
    db: AsyncSession,
    user_id: Any,
    action: str,
    operator_note: str | None,
    source_kind: str,
    source_id: str,
    source_label: str,
    customer: str | None,
    reason_code: str | None,
    reason_label: str | None,
    scheduler_state: dict[str, Any] | None,
    follow_up_launch_status: str | None,
    deep_link: dict[str, Any],
    metadata: dict[str, Any],
    after_state: dict[str, Any],
    deps: FollowUpQueueEventDependencies,
) -> None:
    """Normalize and persist an approved or rejected queue decision."""
    approved = str(action or "").strip().lower() == "approve_launch"
    decision_type = "follow_up_approved" if approved else "follow_up_rejected"
    normalized_scheduler_state = (
        {
            key: value
            for key, value in scheduler_state.items()
            if value not in (None, "", 0)
        }
        if isinstance(scheduler_state, dict)
        else None
    )
    await deps.record_event(
        db,
        user_id=user_id,
        event_type=decision_type,
        event_time=deps.utcnow(),
        source_kind=source_kind,
        source_id=source_id,
        source_label=source_label,
        customer=customer,
        decision_type=decision_type,
        reason_code=reason_code,
        status=str(follow_up_launch_status or "").strip() or None,
        severity="medium",
        actor_mode="operator",
        summary=(
            f"{source_label}: "
            f"{'approved' if approved else 'rejected'} queued follow-up"
        ),
        operator_note=operator_note,
        reason_label=reason_label,
        scheduler_state=normalized_scheduler_state,
        after_state=after_state,
        deep_link=deep_link,
        metadata=metadata,
    )
