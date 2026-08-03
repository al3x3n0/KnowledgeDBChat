"""Priority, SLA, and escalation policy for checkpoint queue items."""

from datetime import datetime
from typing import Any

from app.services.agent_job_queue_helpers import queue_age_minutes


def queue_priority_fields(
    *,
    item_type: str,
    reason_code: str | None,
    created_at: datetime | None,
    next_run_at: datetime | None,
    backoff_until: datetime | None,
    stale: bool,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Calculate sortable priority and operator-facing SLA fields."""
    reference = now or datetime.utcnow()
    age_minutes = queue_age_minutes(created_at, now=reference)
    priority_score = 0.0
    sla_bucket = "normal"
    escalation_level = "normal"
    is_overdue = False
    is_stale = bool(stale)
    normalized_reason = str(reason_code or "").strip().lower()

    if item_type == "approval_checkpoint":
        priority_score = 100 + min(age_minutes, 720) / 8
        if age_minutes >= 240:
            sla_bucket = "overdue"
            escalation_level = "high"
            is_overdue = True
        elif age_minutes >= 60:
            sla_bucket = "at_risk"
            escalation_level = "medium"
    elif item_type == "job_recovery":
        priority_score = 80 + min(age_minutes, 720) / 12
        if normalized_reason == "execution_failure":
            priority_score += 16
        elif normalized_reason == "stalled_run":
            priority_score += 20
        elif normalized_reason in {"scheduler_backoff", "scheduled_recovery"}:
            priority_score += 10
        if backoff_until and backoff_until <= reference:
            priority_score += 14
            is_overdue = True
        if next_run_at and next_run_at <= reference:
            priority_score += 8
            is_overdue = True
        if is_stale:
            priority_score += 18
            is_overdue = True
        if is_overdue or age_minutes >= 180:
            sla_bucket = "overdue"
            escalation_level = "high"
        elif age_minutes >= 45:
            sla_bucket = "at_risk"
            escalation_level = "medium"
    elif item_type == "policy_review":
        priority_score = 90 + min(age_minutes, 720) / 10
        if age_minutes >= 180:
            priority_score += 10
            sla_bucket = "overdue"
            escalation_level = "high"
            is_overdue = True
        elif age_minutes >= 30:
            sla_bucket = "at_risk"
            escalation_level = "medium"
    elif item_type == "budget_review":
        priority_score = 74 + min(age_minutes, 720) / 14
        if age_minutes >= 240:
            priority_score += 8
            sla_bucket = "overdue"
            escalation_level = "high"
            is_overdue = True
        elif age_minutes >= 60:
            sla_bucket = "at_risk"
            escalation_level = "medium"
    else:
        priority_score = 60 + min(age_minutes, 720) / 18
        if age_minutes >= 1440:
            priority_score += 12
            sla_bucket = "at_risk"
            escalation_level = "medium"

    return {
        "priority_score": round(float(priority_score), 2),
        "age_minutes": age_minutes,
        "sla_bucket": sla_bucket,
        "escalation_level": escalation_level,
        "is_overdue": is_overdue,
        "is_stale": is_stale,
    }
