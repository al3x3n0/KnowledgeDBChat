from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.notification import Notification, NotificationType
from app.models.research_note import ResearchNote
from app.services.notification_service import notification_service


def _text(value: Any) -> str:
    return str(value or "").strip()


def _clean_string_list(value: Any, *, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: set[str] = set()
    items: list[str] = []
    for raw in value:
        item = _text(raw)
        if not item or item in seen:
            continue
        seen.add(item)
        items.append(item)
        if len(items) >= limit:
            break
    return items


def _iter_autonomous_origins(payload: dict[str, Any]) -> list[dict[str, str]]:
    hypotheses = payload.get("hypotheses") if isinstance(payload.get("hypotheses"), list) else []
    seen: set[tuple[str, str, str]] = set()
    origins: list[dict[str, str]] = []
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, dict):
            continue
        candidates = [hypothesis.get("autonomous_origin")]
        evidence = hypothesis.get("experiment_evidence") if isinstance(hypothesis.get("experiment_evidence"), list) else []
        for item in evidence:
            if isinstance(item, dict):
                candidates.append(item.get("autonomous_origin"))
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            source_kind = _text(candidate.get("source_kind")).lower()
            source_id = _text(candidate.get("source_id"))
            opportunity_id = _text(candidate.get("opportunity_id"))
            if source_kind not in {"profile", "portfolio"} or not source_id or not opportunity_id:
                continue
            key = (source_kind, source_id, opportunity_id)
            if key in seen:
                continue
            seen.add(key)
            origins.append(
                {
                    "source_kind": source_kind,
                    "source_id": source_id,
                    "opportunity_id": opportunity_id,
                }
            )
    return origins


def _primary_origin(payload: dict[str, Any]) -> Optional[dict[str, str]]:
    origins = _iter_autonomous_origins(payload)
    return origins[0] if origins else None


def _build_action_url(*, status: str, note_id: str, reevaluation_job_id: str) -> Optional[str]:
    if status == "failed" and reevaluation_job_id:
        return f"/synthesis?job={reevaluation_job_id}"
    if note_id:
        return f"/research-notes?note={note_id}"
    if reevaluation_job_id:
        return f"/synthesis?job={reevaluation_job_id}"
    return None


def _build_origin_action_url(origin: Optional[dict[str, str]]) -> Optional[str]:
    if not origin:
        return None
    if origin["source_kind"] == "profile":
        return (
            f"/autonomous-agents?tab=domain&profileId={origin['source_id']}"
            f"&opportunityId={origin['opportunity_id']}"
        )
    return (
        f"/autonomous-agents?tab=fleet&fleetId={origin['source_id']}"
        f"&opportunityId={origin['opportunity_id']}"
    )


def _build_notification_content(
    *,
    note_title: str,
    status: str,
    summary: str,
    source_run_ids: list[str],
) -> tuple[str, str, str]:
    label = note_title or "Research note"
    summary_text = summary or "Evidence-aware reevaluation state changed."
    if status == "completed":
        message = summary_text
        if source_run_ids:
            message = f"{summary_text} Source runs: {', '.join(source_run_ids[:3])}."
        return f"Reevaluation ready: {label}", message, "normal"
    if status == "stale":
        return (
            f"Reevaluation stale: {label}",
            summary_text or "A queued reevaluation draft is stale because newer experiment evidence arrived.",
            "high",
        )
    return (
        f"Reevaluation failed: {label}",
        summary_text or "The queued hypothesis reevaluation failed.",
        "high",
    )


async def _notification_exists(
    db: AsyncSession,
    *,
    user_id: UUID,
    note_id: UUID,
    reevaluation_job_id: str,
    status: str,
) -> bool:
    result = await db.execute(
        select(Notification).where(
            Notification.user_id == user_id,
            Notification.notification_type == NotificationType.HYPOTHESIS_REEVALUATION_UPDATE,
            Notification.related_entity_type == "research_note",
            Notification.related_entity_id == note_id,
        )
    )
    for notification in result.scalars().all():
        payload = notification.data if isinstance(notification.data, dict) else {}
        if (
            _text(payload.get("reevaluation_job_id")) == reevaluation_job_id
            and _text(payload.get("reevaluation_status")).lower() == status
        ):
            return True
    return False


async def maybe_emit_reevaluation_notification(
    db: AsyncSession,
    *,
    note: ResearchNote,
    user_id: UUID,
    reevaluation_job_id: str,
    status: str,
    summary: str = "",
    source_run_ids: Optional[list[str]] = None,
    error: str = "",
    created_at: Optional[str] = None,
    completed_at: Optional[str] = None,
    commit: bool = True,
    push: bool = True,
) -> Optional[Notification]:
    normalized_status = _text(status).lower()
    normalized_job_id = _text(reevaluation_job_id)
    if normalized_status not in {"completed", "failed", "stale"} or not normalized_job_id:
        return None

    if await _notification_exists(
        db,
        user_id=user_id,
        note_id=note.id,
        reevaluation_job_id=normalized_job_id,
        status=normalized_status,
    ):
        return None

    payload = note.structured_payload if isinstance(note.structured_payload, dict) else {}
    origin = _primary_origin(payload)
    run_ids = _clean_string_list(source_run_ids if source_run_ids is not None else payload.get("pending_reevaluation_source_run_ids"))
    message_summary = _text(summary)
    if normalized_status == "failed":
        message_summary = _text(error) or message_summary
    elif normalized_status == "stale" and not message_summary:
        message_summary = "Newer experiment evidence arrived after the queued reevaluation draft was created."
    elif normalized_status == "completed" and not message_summary:
        message_summary = _text(payload.get("reprioritization_summary")) or "Evidence-aware hypothesis reevaluation finished and is ready to review."

    title, message, priority = _build_notification_content(
        note_title=_text(note.title),
        status=normalized_status,
        summary=message_summary,
        source_run_ids=run_ids,
    )

    data = {
        "note_id": str(note.id),
        "note_title": _text(note.title) or None,
        "reevaluation_job_id": normalized_job_id,
        "reevaluation_status": normalized_status,
        "source_run_ids": run_ids,
        "reprioritization_summary": message_summary or None,
        "pending_reevaluation_created_at": _text(created_at) or None,
        "pending_reevaluation_completed_at": _text(completed_at) or None,
        "reevaluation_error": _text(error) or None,
        "origin_source_kind": origin["source_kind"] if origin else None,
        "origin_source_id": origin["source_id"] if origin else None,
        "origin_opportunity_id": origin["opportunity_id"] if origin else None,
        "origin_action_url": _build_origin_action_url(origin),
    }

    return await notification_service.create_notification(
        db=db,
        user_id=user_id,
        notification_type=NotificationType.HYPOTHESIS_REEVALUATION_UPDATE,
        title=title,
        message=message,
        priority=priority,
        related_entity_type="research_note",
        related_entity_id=note.id,
        data=data,
        action_url=_build_action_url(
            status=normalized_status,
            note_id=str(note.id),
            reevaluation_job_id=normalized_job_id,
        ),
        commit=commit,
        push=push,
    )


async def resolve_reevaluation_notifications(
    db: AsyncSession,
    *,
    user_id: UUID,
    reevaluation_job_id: str,
    note_id: Optional[UUID] = None,
    review_outcome_status: str,
    review_recorded_at: Optional[str] = None,
    resolved_target_note_id: Optional[str] = None,
    review_note: Optional[str] = None,
    commit: bool = True,
) -> int:
    normalized_job_id = _text(reevaluation_job_id)
    normalized_outcome = _text(review_outcome_status)
    if not normalized_job_id or not normalized_outcome:
        return 0

    query = select(Notification).where(
        Notification.user_id == user_id,
        Notification.notification_type == NotificationType.HYPOTHESIS_REEVALUATION_UPDATE,
    )
    if note_id is not None:
        query = query.where(
            (Notification.related_entity_type == "research_note")
            & (Notification.related_entity_id == note_id)
        )

    result = await db.execute(query)
    notifications = result.scalars().all()
    resolved_at = _text(review_recorded_at) or datetime.utcnow().isoformat()
    resolved_count = 0

    for notification in notifications:
        payload = notification.data if isinstance(notification.data, dict) else {}
        if _text(payload.get("reevaluation_job_id")) != normalized_job_id:
            continue
        next_payload = dict(payload)
        next_payload["review_outcome_status"] = normalized_outcome
        next_payload["review_recorded_at"] = resolved_at
        next_payload["resolved_by_review"] = True
        if _text(resolved_target_note_id):
            next_payload["resolved_target_note_id"] = _text(resolved_target_note_id)
        if _text(review_note):
            next_payload["review_note"] = _text(review_note)
        notification.data = next_payload
        notification.is_dismissed = True
        if not notification.is_read:
            notification.is_read = True
            notification.read_at = datetime.utcnow()
        resolved_count += 1

    if commit:
        await db.commit()
    return resolved_count
