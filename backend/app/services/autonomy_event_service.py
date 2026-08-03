"""
Helpers for persisted autonomy/operator decision events.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.autonomy_decision_event import AutonomyDecisionEvent
from app.models.notification import Notification, NotificationType
from app.services.notification_service import notification_service

HIGH_SIGNAL_EVENT_NOTIFICATIONS: dict[str, tuple[str, str, str]] = {
    "validation_blocked": (
        NotificationType.QUEUE_URGENCY_ALERT,
        "high",
        "Validation blocked",
    ),
    "policy_guardrail_triggered": (
        NotificationType.POLICY_GUARDRAIL_ALERT,
        "high",
        "Policy guardrail triggered",
    ),
    "budget_clamped": (
        NotificationType.AUTONOMY_BUDGET_ALERT,
        "high",
        "Autonomy budget clamp active",
    ),
    "job_recovery_queued": (
        NotificationType.QUEUE_URGENCY_ALERT,
        "high",
        "Job recovery queued",
    ),
    "opportunity_blocked": (
        NotificationType.QUEUE_URGENCY_ALERT,
        "high",
        "Research opportunity blocked",
    ),
}

ESCALATION_RULES = {
    "pinned_warning_hours": 2,
    "high_warning_hours": 4,
    "high_escalated_hours": 24,
    "investigating_escalated_hours": 48,
}

TRACE_TEAM_BUCKETS = {
    "monitor": "monitor",
    "domain_profile": "domain_profiles",
    "portfolio": "research_fleet",
    "validation_run": "validation",
    "queue": "queue",
    "job": "jobs",
}

SCHEDULER_STATE_KEYS = (
    "queue_reason",
    "last_run_status",
    "failure_streak",
    "last_scheduled_at",
    "last_dispatched_at",
    "current_run_started_at",
    "last_completed_run_at",
    "last_successful_run_at",
    "last_failure_at",
    "backoff_until",
    "backoff_seconds",
)


def _clean_text(value: Any, *, limit: int = 255) -> Optional[str]:
    text = str(value or "").strip()
    return text[:limit] if text else None


def _clean_json(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict) or not value:
        return None
    return deepcopy(value)


def _trace_reason_label(reason_code: Any) -> Optional[str]:
    code = str(reason_code or "").strip().lower()
    if not code:
        return None
    mapping = {
        "accepted_inbox_item": "Accepted inbox signal",
        "approval_required": "Approval required",
        "budget_clamped": "Autonomy budget review",
        "execution_failure": "Execution failure",
        "follow_up_blocked": "Follow-up blocked by policy",
        "follow_up_launch_failed": "Follow-up launch failed",
        "follow_up_launch_approval": "Follow-up launch approval",
        "policy_guardrail": "Policy safeguard review",
        "scheduled_recovery": "Scheduled recovery",
        "stalled_run": "Stalled run",
        "validation_blocked": "Validation blocked",
    }
    return mapping.get(code) or code.replace("_", " ").capitalize()


def _extract_scheduler_state(metadata: Any) -> Optional[dict[str, Any]]:
    if not isinstance(metadata, dict):
        return None
    scheduler_state = metadata.get("scheduler_state")
    if isinstance(scheduler_state, dict) and scheduler_state:
        return deepcopy(scheduler_state)
    extracted: dict[str, Any] = {}
    for key in SCHEDULER_STATE_KEYS:
        if key in metadata and metadata.get(key) is not None:
            extracted[key] = deepcopy(metadata.get(key))
    return extracted or None


def _extract_trace_context(
    metadata: Any, *, reason_code: Any = None
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    if not isinstance(metadata, dict):
        return _trace_reason_label(reason_code), None
    reason_label = _clean_text(
        metadata.get("reason_label"), limit=255
    ) or _trace_reason_label(metadata.get("queue_reason") or reason_code)
    scheduler_state = _extract_scheduler_state(metadata)
    return reason_label, scheduler_state


def _normalize_trace_metadata(
    metadata: Optional[dict[str, Any]],
    *,
    reason_code: Any = None,
    reason_label: Optional[str] = None,
    scheduler_state: Any = None,
) -> Optional[dict[str, Any]]:
    normalized = _clean_json(metadata) or {}
    if (
        not normalized
        and not reason_label
        and scheduler_state is None
        and not reason_code
    ):
        return None

    extracted_reason_label, extracted_scheduler_state = _extract_trace_context(
        normalized, reason_code=reason_code
    )
    next_reason_label = _clean_text(reason_label, limit=255) or extracted_reason_label
    next_scheduler_state = (
        _normalize_scheduler_state(scheduler_state) or extracted_scheduler_state
    )

    if next_reason_label:
        normalized["reason_label"] = next_reason_label
    else:
        normalized.pop("reason_label", None)
    if next_scheduler_state is not None:
        normalized["scheduler_state"] = next_scheduler_state
    else:
        normalized.pop("scheduler_state", None)
    return normalized or None


def _normalize_scheduler_state(value: Any) -> Optional[dict[str, Any]]:
    if not isinstance(value, dict) or not value:
        return None
    return deepcopy(value)


def event_to_trace_payload(event: AutonomyDecisionEvent) -> dict[str, Any]:
    (
        escalation_state,
        escalation_reason,
        escalated_at,
    ) = compute_decision_trace_escalation(event)
    metadata = _clean_json(event.event_metadata)
    reason_label, scheduler_state = _extract_trace_context(
        metadata, reason_code=event.reason_code
    )
    return {
        "event_id": str(event.id),
        "event_type": str(event.event_type or "").strip(),
        "event_time": event.event_time,
        "source_kind": str(event.source_kind or "").strip(),
        "source_id": _clean_text(event.source_id, limit=128),
        "source_label": _clean_text(event.source_label),
        "customer": _clean_text(event.customer),
        "decision_type": str(event.decision_type or "").strip(),
        "reason_code": _clean_text(event.reason_code, limit=128),
        "reason_label": reason_label,
        "scheduler_state": scheduler_state,
        "status": _clean_text(event.status, limit=64),
        "severity": _clean_text(event.severity, limit=32),
        "actor_mode": _clean_text(event.actor_mode, limit=24),
        "summary": str(event.summary or "").strip(),
        "operator_note": _clean_text(event.operator_note, limit=4000),
        "before_state": _clean_json(event.before_state),
        "after_state": _clean_json(event.after_state),
        "deep_link": _clean_json(event.deep_link),
        "metadata": metadata,
        "is_derived": bool(event.is_derived),
        "record_origin": _clean_text(event.record_origin, limit=24) or "persisted",
        "triage_status": _clean_text(event.triage_status, limit=24) or "new",
        "acknowledged_at": event.acknowledged_at,
        "acknowledged_by_user_id": event.acknowledged_by_user_id,
        "resolved_at": event.resolved_at,
        "resolved_by_user_id": event.resolved_by_user_id,
        "resolution_note": _clean_text(event.resolution_note, limit=4000),
        "pinned": bool(event.pinned),
        "last_viewed_at": event.last_viewed_at,
        "owner_user_id": event.user_id,
        "owner_label": None,
        "assigned_to_user_id": event.assigned_to_user_id,
        "assigned_at": event.assigned_at,
        "assigned_by_user_id": event.assigned_by_user_id,
        "assignee_label": None,
        "is_owned_by_current_user": False,
        "is_assigned_to_current_user": False,
        "team_bucket": _clean_text(event.team_bucket, limit=64),
        "due_at": event.due_at,
        "escalation_state": escalation_state,
        "escalation_reason": _clean_text(escalation_reason, limit=255),
        "escalated_at": escalated_at,
    }


async def _maybe_create_event_notification(
    db: AsyncSession,
    event: AutonomyDecisionEvent,
    *,
    force: bool = False,
) -> None:
    event_key = str(event.event_type or "").strip().lower()
    escalation_state, escalation_reason, _ = compute_decision_trace_escalation(event)
    target_user_id = event.assigned_to_user_id or event.user_id
    rule = HIGH_SIGNAL_EVENT_NOTIFICATIONS.get(event_key)
    if escalation_state in {"warning", "escalated"}:
        rule = (
            NotificationType.QUEUE_URGENCY_ALERT,
            "urgent" if escalation_state == "escalated" else "high",
            "Decision trace escalation",
        )
    if not rule or not hasattr(db, "execute"):
        return

    notification_type, priority, title = rule
    existing_notification = (
        (
            await db.execute(
                select(Notification).where(
                    Notification.user_id == target_user_id,
                    Notification.related_entity_type == "autonomy_decision_event",
                    Notification.related_entity_id == event.id,
                )
            )
        )
        .scalars()
        .first()
    )
    if existing_notification is not None and not force:
        return
    if existing_notification is not None and force:
        existing_state = (
            str((existing_notification.data or {}).get("escalation_state") or "")
            .strip()
            .lower()
        )
        if (
            existing_state == escalation_state
            and existing_notification.notification_type == rule[0]
        ):
            return

    action_url = f"/autonomous-agents?tab=trace&trace_event={event.id}"
    message = str(event.summary or "").strip()[:2000] or title
    if escalation_state in {"warning", "escalated"}:
        reason_suffix = f" ({escalation_reason})" if escalation_reason else ""
        title = (
            "Decision trace escalated"
            if escalation_state == "escalated"
            else "Decision trace warning"
        )
        message = f"{message}{reason_suffix}".strip()
    reason_label, scheduler_state = _extract_trace_context(
        event.event_metadata, reason_code=event.reason_code
    )
    await notification_service.create_notification(
        db=db,
        user_id=target_user_id,
        notification_type=notification_type,
        title=title,
        message=message,
        priority=priority,
        related_entity_type="autonomy_decision_event",
        related_entity_id=event.id,
        data={
            "trace_event_id": str(event.id),
            "event_type": str(event.event_type or "").strip(),
            "decision_type": str(event.decision_type or "").strip(),
            "source_kind": str(event.source_kind or "").strip(),
            "source_label": _clean_text(event.source_label),
            "reason_label": reason_label,
            "scheduler_state": scheduler_state,
            "escalation_state": escalation_state or "none",
        },
        action_url=action_url,
        commit=False,
    )


def _is_unresolved(event: AutonomyDecisionEvent) -> bool:
    return str(event.triage_status or "new").strip().lower() != "resolved"


def derive_trace_team_bucket(source_kind: Any) -> str:
    return TRACE_TEAM_BUCKETS.get(str(source_kind or "").strip().lower(), "jobs")


def compute_decision_trace_escalation(
    event: AutonomyDecisionEvent, *, now: Optional[datetime] = None
) -> tuple[str, Optional[str], Optional[datetime]]:
    now = now or datetime.utcnow()
    if not _is_unresolved(event):
        return "none", None, None

    due_at = event.due_at
    if due_at is not None and due_at <= now:
        return "escalated", "due_at_expired", due_at

    event_time = event.event_time or event.created_at or now
    age = now - event_time
    severity = str(event.severity or "").strip().lower()
    triage_status = str(event.triage_status or "new").strip().lower()

    if triage_status == "investigating" and age >= timedelta(
        hours=ESCALATION_RULES["investigating_escalated_hours"]
    ):
        escalated_at = event_time + timedelta(
            hours=ESCALATION_RULES["investigating_escalated_hours"]
        )
        return "escalated", "investigation_stale", escalated_at
    if severity in {"high", "urgent"} and age >= timedelta(
        hours=ESCALATION_RULES["high_escalated_hours"]
    ):
        escalated_at = event_time + timedelta(
            hours=ESCALATION_RULES["high_escalated_hours"]
        )
        return "escalated", "high_severity_stale", escalated_at
    if bool(event.pinned) and age >= timedelta(
        hours=ESCALATION_RULES["pinned_warning_hours"]
    ):
        escalated_at = event_time + timedelta(
            hours=ESCALATION_RULES["pinned_warning_hours"]
        )
        return "warning", "pinned_stale", escalated_at
    if severity in {"high", "urgent"} and age >= timedelta(
        hours=ESCALATION_RULES["high_warning_hours"]
    ):
        escalated_at = event_time + timedelta(
            hours=ESCALATION_RULES["high_warning_hours"]
        )
        return "warning", "high_severity_warning", escalated_at
    return "none", None, None


def apply_decision_trace_escalation(
    event: AutonomyDecisionEvent, *, now: Optional[datetime] = None
) -> str:
    (
        escalation_state,
        escalation_reason,
        escalated_at,
    ) = compute_decision_trace_escalation(event, now=now)
    event.escalation_state = escalation_state
    event.escalation_reason = escalation_reason
    event.escalated_at = escalated_at
    return escalation_state


async def record_autonomy_decision_event(
    db: AsyncSession,
    *,
    user_id: UUID | str,
    event_type: str,
    source_kind: str,
    decision_type: str,
    summary: str,
    event_time: Optional[datetime] = None,
    source_id: Optional[str] = None,
    source_label: Optional[str] = None,
    customer: Optional[str] = None,
    reason_code: Optional[str] = None,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    actor_mode: Optional[str] = None,
    operator_note: Optional[str] = None,
    before_state: Optional[dict[str, Any]] = None,
    after_state: Optional[dict[str, Any]] = None,
    deep_link: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
    reason_label: Optional[str] = None,
    scheduler_state: Any = None,
    is_derived: bool = False,
    record_origin: str = "persisted",
    emit_notification: bool = True,
) -> AutonomyDecisionEvent:
    normalized_metadata = _normalize_trace_metadata(
        metadata,
        reason_code=reason_code,
        reason_label=reason_label,
        scheduler_state=scheduler_state,
    )
    row = AutonomyDecisionEvent(
        user_id=user_id,
        event_time=event_time or datetime.utcnow(),
        event_type=str(event_type or "").strip()[:80] or "event",
        source_kind=str(source_kind or "").strip()[:64] or "unknown",
        source_id=_clean_text(source_id, limit=128),
        source_label=_clean_text(source_label),
        customer=_clean_text(customer),
        decision_type=str(decision_type or "").strip()[:80] or "event",
        reason_code=_clean_text(reason_code, limit=128),
        status=_clean_text(status, limit=64),
        severity=_clean_text(severity, limit=32),
        actor_mode=_clean_text(actor_mode, limit=24),
        summary=str(summary or "").strip()[:4000] or "Autonomy event",
        operator_note=_clean_text(operator_note, limit=4000),
        before_state=_clean_json(before_state),
        after_state=_clean_json(after_state),
        deep_link=_clean_json(deep_link),
        event_metadata=normalized_metadata,
        is_derived=bool(is_derived),
        record_origin=_clean_text(record_origin, limit=24) or "persisted",
        triage_status="new",
        pinned=False,
        team_bucket=derive_trace_team_bucket(source_kind),
        escalation_state="none",
    )
    apply_decision_trace_escalation(row)
    if hasattr(db, "add"):
        db.add(row)
    if hasattr(db, "flush"):
        await db.flush()
    if emit_notification and hasattr(db, "add") and hasattr(db, "execute"):
        await _maybe_create_event_notification(db, row)
    return row


async def maybe_reopen_event_notification(
    db: AsyncSession, event: AutonomyDecisionEvent
) -> None:
    if not hasattr(db, "execute"):
        return
    await _maybe_create_event_notification(db, event, force=True)


async def maybe_emit_escalation_transition_notification(
    db: AsyncSession,
    event: AutonomyDecisionEvent,
    *,
    previous_state: Optional[str],
) -> None:
    next_state = apply_decision_trace_escalation(event)
    previous = str(previous_state or "").strip().lower() or "none"
    if next_state not in {"warning", "escalated"}:
        return
    if previous == next_state:
        return
    await _maybe_create_event_notification(db, event, force=True)
