"""
Helpers for syncing launched follow-up job outcomes back to Research Inbox items.
"""

from __future__ import annotations

import re
from datetime import datetime
from uuid import UUID

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.domain_research_profile import DomainResearchProfile
from app.models.notification import Notification, NotificationType
from app.models.research_inbox import ResearchInboxItem
from app.models.research_portfolio import ResearchPortfolio
from app.services.notification_service import notification_service
from app.services.research_opportunity_service import (
    list_normalized_research_opportunities,
)


def _normalize_summary_text(value: object, *, limit: int = 280) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return None
    return normalized[:limit]


def summarize_follow_up_job_outcome(job: AgentJob) -> str:
    """Build a compact deterministic outcome summary from current job state."""
    status = str(job.status or "").strip().lower()
    results = job.results if isinstance(job.results, dict) else {}
    executive_digest = (
        results.get("executive_digest")
        if isinstance(results.get("executive_digest"), dict)
        else {}
    )

    if status == AgentJobStatus.COMPLETED.value:
        for candidate in (
            results.get("summary"),
            executive_digest.get("outcome"),
            executive_digest.get("summary"),
            executive_digest.get("headline"),
            results.get("note"),
            job.phase_details,
        ):
            summary = _normalize_summary_text(candidate)
            if summary:
                return summary
        return (
            f"{str(job.name or 'Follow-up job').strip() or 'Follow-up job'} completed."
        )

    if status == AgentJobStatus.FAILED.value:
        summary = _normalize_summary_text(job.error) or _normalize_summary_text(
            job.phase_details
        )
        if summary:
            return summary
        return f"{str(job.name or 'Follow-up job').strip() or 'Follow-up job'} failed."

    if status == AgentJobStatus.CANCELLED.value:
        summary = _normalize_summary_text(job.phase_details) or _normalize_summary_text(
            job.error
        )
        if summary:
            return summary
        return f"{str(job.name or 'Follow-up job').strip() or 'Follow-up job'} was cancelled."

    return f"{str(job.name or 'Follow-up job').strip() or 'Follow-up job'} ended with status {status or 'unknown'}."


def _follow_up_outcome_decision_type(status: str) -> str:
    normalized = str(status or "").strip().lower()
    if normalized == AgentJobStatus.FAILED.value:
        return "follow_up_failed"
    if normalized == AgentJobStatus.CANCELLED.value:
        return "follow_up_cancelled"
    return "follow_up_completed"


async def _get_model_by_id(db: AsyncSession, model: type, value: str):
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return await db.get(model, UUID(text))
    except Exception:
        return None


def _resolve_follow_up_opportunity_origin(
    job: AgentJob,
) -> tuple[str | None, str | None, str | None]:
    job_config = getattr(job, "config", None)
    config = job_config if isinstance(job_config, dict) else {}
    follow_up = (
        config.get("domain_research_follow_up")
        if isinstance(config.get("domain_research_follow_up"), dict)
        else {}
    )
    idea = follow_up.get("idea") if isinstance(follow_up.get("idea"), dict) else {}
    metadata = (
        config.get("autonomy_metadata")
        if isinstance(config.get("autonomy_metadata"), dict)
        else {}
    )
    origin = (
        idea.get("autonomous_origin")
        if isinstance(idea.get("autonomous_origin"), dict)
        else {}
    )
    opportunity_id = (
        str(origin.get("opportunity_id") or "").strip()
        or str(
            metadata.get("profile_opportunity_id")
            or metadata.get("portfolio_opportunity_id")
            or ""
        ).strip()
        or str(idea.get("opportunity_id") or "").strip()
    )
    source_kind = str(origin.get("source_kind") or "").strip().lower() or (
        "profile"
        if str(metadata.get("domain_research_profile_id") or "").strip()
        else "portfolio"
        if str(metadata.get("portfolio_id") or "").strip()
        else ""
    )
    source_id = (
        str(origin.get("source_id") or "").strip()
        or str(
            metadata.get("domain_research_profile_id")
            or metadata.get("portfolio_id")
            or ""
        ).strip()
    )
    if (
        source_kind not in {"profile", "portfolio"}
        or not source_id
        or not opportunity_id
    ):
        return None, None, None
    return source_kind, source_id, opportunity_id


async def _project_follow_up_outcome_to_originating_opportunity(
    db: AsyncSession,
    *,
    job: AgentJob,
    status: str,
    summary: str,
    recorded_at: datetime,
) -> bool:
    source_kind, source_id, opportunity_id = _resolve_follow_up_opportunity_origin(job)
    if not source_kind or not source_id or not opportunity_id:
        return False

    decision_type = _follow_up_outcome_decision_type(status)
    if source_kind == "profile":
        profile = await _get_model_by_id(db, DomainResearchProfile, source_id)
        if profile is None:
            return False
        profile_summary = (
            dict(profile.latest_summary)
            if isinstance(profile.latest_summary, dict)
            else {}
        )
        rows = list_normalized_research_opportunities(
            profile_summary.get("opportunities")
            or profile_summary.get("idea_candidates")
            or []
        )
        idx = next(
            (
                i
                for i, row in enumerate(rows)
                if str(row.get("opportunity_id") or "").strip() == opportunity_id
            ),
            -1,
        )
        if idx < 0:
            return False
        row = dict(rows[idx])
        current_status = str(row.get("follow_up_outcome_status") or "").strip().lower()
        current_summary = str(row.get("follow_up_outcome_summary") or "").strip()
        current_recorded_at = str(
            row.get("follow_up_outcome_recorded_at") or ""
        ).strip()
        next_recorded_at = recorded_at.isoformat()
        if (
            current_status == status
            and current_summary == summary
            and current_recorded_at == next_recorded_at
            and str(row.get("follow_up_last_job_id") or "").strip() == str(job.id)
        ):
            return False
        row["follow_up_outcome_status"] = status
        row["follow_up_outcome_recorded_at"] = next_recorded_at
        row["follow_up_outcome_summary"] = summary
        row["follow_up_last_job_id"] = str(job.id)
        if not str(row.get("follow_up_launched_at") or "").strip():
            launched_at = job.created_at or recorded_at
            row["follow_up_launched_at"] = (
                launched_at.isoformat() if launched_at else next_recorded_at
            )
        row["last_activity_at"] = next_recorded_at
        row["last_decision_type"] = decision_type
        row["last_decision_reason_code"] = decision_type
        rows[idx] = row
        profile_summary["opportunities"] = rows
        if isinstance(profile_summary.get("idea_candidates"), list):
            profile_summary["idea_candidates"] = rows
        profile.latest_summary = profile_summary
        profile.updated_at = recorded_at
        return True

    portfolio = await _get_model_by_id(db, ResearchPortfolio, source_id)
    if portfolio is None:
        return False
    rows = list_normalized_research_opportunities(portfolio.opportunities or [])
    idx = next(
        (
            i
            for i, row in enumerate(rows)
            if str(row.get("opportunity_id") or "").strip() == opportunity_id
        ),
        -1,
    )
    if idx < 0:
        return False
    row = dict(rows[idx])
    current_status = str(row.get("follow_up_outcome_status") or "").strip().lower()
    current_summary = str(row.get("follow_up_outcome_summary") or "").strip()
    current_recorded_at = str(row.get("follow_up_outcome_recorded_at") or "").strip()
    next_recorded_at = recorded_at.isoformat()
    if (
        current_status == status
        and current_summary == summary
        and current_recorded_at == next_recorded_at
        and str(row.get("follow_up_last_job_id") or "").strip() == str(job.id)
    ):
        return False
    row["follow_up_outcome_status"] = status
    row["follow_up_outcome_recorded_at"] = next_recorded_at
    row["follow_up_outcome_summary"] = summary
    row["follow_up_last_job_id"] = str(job.id)
    if not str(row.get("follow_up_launched_at") or "").strip():
        launched_at = job.created_at or recorded_at
        row["follow_up_launched_at"] = (
            launched_at.isoformat() if launched_at else next_recorded_at
        )
    row["last_activity_at"] = next_recorded_at
    row["last_decision_type"] = decision_type
    row["last_decision_reason_code"] = decision_type
    rows[idx] = row
    portfolio.opportunities = rows
    portfolio.updated_at = recorded_at
    return True


async def project_follow_up_relaunch_to_originating_opportunity(
    db: AsyncSession,
    *,
    job: AgentJob,
    launched_at: datetime,
) -> bool:
    source_kind, source_id, opportunity_id = _resolve_follow_up_opportunity_origin(job)
    if not source_kind or not source_id or not opportunity_id:
        return False

    next_launched_at = launched_at.isoformat()

    def _apply_relaunch(row: dict) -> bool:
        current_job_id = str(row.get("follow_up_last_job_id") or "").strip()
        current_launched_at = str(row.get("follow_up_launched_at") or "").strip()
        child_job_ids = [
            str(value or "").strip()
            for value in (
                row.get("child_job_ids")
                if isinstance(row.get("child_job_ids"), list)
                else []
            )
            if str(value or "").strip()
        ]
        if str(job.id) not in child_job_ids:
            child_job_ids.append(str(job.id))
        row_changed = (
            current_job_id != str(job.id)
            or current_launched_at != next_launched_at
            or str(row.get("follow_up_outcome_status") or "").strip()
            or str(row.get("follow_up_outcome_recorded_at") or "").strip()
            or str(row.get("follow_up_outcome_summary") or "").strip()
            or str(row.get("last_decision_type") or "").strip() != "follow_up_launched"
            or str(row.get("last_decision_reason_code") or "").strip()
            != "follow_up_relaunched"
            or str(row.get("autonomy_state") or "").strip() != "active"
            or str(row.get("stage") or "").strip() != "accepted"
            or row.get("child_job_ids") != child_job_ids
        )
        if not row_changed:
            return False
        row["follow_up_last_job_id"] = str(job.id)
        row["follow_up_launched_at"] = next_launched_at
        row["follow_up_outcome_status"] = None
        row["follow_up_outcome_recorded_at"] = None
        row["follow_up_outcome_summary"] = None
        row["child_job_ids"] = child_job_ids
        row["last_activity_at"] = next_launched_at
        row["last_decision_type"] = "follow_up_launched"
        row["last_decision_reason_code"] = "follow_up_relaunched"
        row["autonomy_state"] = "active"
        row["stage"] = "accepted"
        return True

    if source_kind == "profile":
        profile = await _get_model_by_id(db, DomainResearchProfile, source_id)
        if profile is None:
            return False
        profile_summary = (
            dict(profile.latest_summary)
            if isinstance(profile.latest_summary, dict)
            else {}
        )
        rows = list_normalized_research_opportunities(
            profile_summary.get("opportunities")
            or profile_summary.get("idea_candidates")
            or []
        )
        idx = next(
            (
                i
                for i, row in enumerate(rows)
                if str(row.get("opportunity_id") or "").strip() == opportunity_id
            ),
            -1,
        )
        if idx < 0:
            return False
        row = dict(rows[idx])
        if not _apply_relaunch(row):
            return False
        rows[idx] = row
        profile_summary["opportunities"] = rows
        if isinstance(profile_summary.get("idea_candidates"), list):
            profile_summary["idea_candidates"] = rows
        profile.latest_summary = profile_summary
        profile.updated_at = launched_at
        return True

    portfolio = await _get_model_by_id(db, ResearchPortfolio, source_id)
    if portfolio is None:
        return False
    rows = list_normalized_research_opportunities(portfolio.opportunities or [])
    idx = next(
        (
            i
            for i, row in enumerate(rows)
            if str(row.get("opportunity_id") or "").strip() == opportunity_id
        ),
        -1,
    )
    if idx < 0:
        return False
    row = dict(rows[idx])
    if not _apply_relaunch(row):
        return False
    rows[idx] = row
    portfolio.opportunities = rows
    portfolio.updated_at = launched_at
    return True


def _build_follow_up_outcome_action_url(
    inbox_item_id: object | None,
    *,
    origin_source_kind: str | None = None,
    origin_source_id: object | None = None,
    origin_opportunity_id: object | None = None,
) -> str:
    source_kind = str(origin_source_kind or "").strip().lower()
    source_id = str(origin_source_id or "").strip()
    opportunity_id = str(origin_opportunity_id or "").strip()
    if source_kind == "profile" and source_id and opportunity_id:
        return f"/autonomous-agents?tab=domain&profileId={source_id}&opportunityId={opportunity_id}"
    if source_kind == "portfolio" and source_id and opportunity_id:
        return f"/autonomous-agents?tab=fleet&fleetId={source_id}&opportunityId={opportunity_id}"
    item_id = str(inbox_item_id or "").strip()
    if item_id:
        return f"/autonomous-agents?tab=inbox&inbox={item_id}"
    return "/autonomous-agents?tab=inbox"


def _build_follow_up_outcome_notification_content(
    item: ResearchInboxItem,
    job: AgentJob,
    *,
    status: str,
    summary: str,
) -> tuple[str, str, str]:
    title_text = str(item.title or job.name or "Follow-up").strip() or "Follow-up"
    if status == AgentJobStatus.FAILED.value:
        title = f"Follow-up failed: {title_text[:120]}"
        message = (
            summary or "A launched follow-up failed and may need operator relaunch."
        )
        priority = "high"
    elif status == AgentJobStatus.CANCELLED.value:
        title = f"Follow-up cancelled: {title_text[:120]}"
        message = (
            summary
            or "A launched follow-up was cancelled and can be relaunched from the inbox."
        )
        priority = "high"
    else:
        title = f"Follow-up completed: {title_text[:120]}"
        message = summary or "A launched follow-up completed successfully."
        priority = "normal"
    return title, message[:500], priority


async def _maybe_emit_follow_up_outcome_notification(
    db: AsyncSession,
    *,
    item: ResearchInboxItem,
    job: AgentJob,
    status: str,
    summary: str,
) -> None:
    (
        origin_source_kind,
        origin_source_id,
        origin_opportunity_id,
    ) = _resolve_follow_up_opportunity_origin(job)
    existing_result = await db.execute(
        select(Notification)
        .where(
            and_(
                Notification.user_id == item.user_id,
                Notification.notification_type
                == NotificationType.FOLLOW_UP_OUTCOME_ALERT,
                Notification.related_entity_id == item.id,
                Notification.is_dismissed.is_(False),
            )
        )
        .order_by(Notification.created_at.desc())
        .limit(20)
    )
    existing_notifications = list(existing_result.scalars().all())
    for notification in existing_notifications:
        data = notification.data if isinstance(notification.data, dict) else {}
        if str(
            data.get("follow_up_outcome_status") or ""
        ).strip().lower() == status and str(
            data.get("follow_up_job_id") or ""
        ).strip() == str(
            job.id
        ):
            return

    title, message, priority = _build_follow_up_outcome_notification_content(
        item,
        job,
        status=status,
        summary=summary,
    )
    await notification_service.create_notification(
        db=db,
        user_id=item.user_id,
        notification_type=NotificationType.FOLLOW_UP_OUTCOME_ALERT,
        title=title,
        message=message,
        priority=priority,
        related_entity_type="research_inbox_item",
        related_entity_id=item.id,
        data={
            "inbox_item_id": str(item.id),
            "follow_up_job_id": str(job.id),
            "follow_up_last_job_id": str(job.id),
            "follow_up_recommendation_key": str(
                item.follow_up_recommendation_key or ""
            ).strip()
            or None,
            "follow_up_outcome_status": status,
            "follow_up_outcome_summary": summary,
            "customer": str(item.customer or "").strip() or None,
            "follow_up_policy_mode": str(item.follow_up_policy_mode or "").strip()
            or None,
            "origin_source_kind": origin_source_kind,
            "origin_source_id": origin_source_id,
            "origin_opportunity_id": origin_opportunity_id,
        },
        action_url=_build_follow_up_outcome_action_url(
            item.id,
            origin_source_kind=origin_source_kind,
            origin_source_id=origin_source_id,
            origin_opportunity_id=origin_opportunity_id,
        ),
        commit=False,
        push=True,
    )


async def sync_follow_up_outcome_for_job(db: AsyncSession, job: AgentJob) -> int:
    """
    Propagate a terminal follow-up AgentJob outcome back to linked accepted inbox items.
    """
    status = str(job.status or "").strip().lower()
    if status not in {
        AgentJobStatus.COMPLETED.value,
        AgentJobStatus.FAILED.value,
        AgentJobStatus.CANCELLED.value,
    }:
        return 0

    result = await db.execute(
        select(ResearchInboxItem).where(
            ResearchInboxItem.follow_up_job_id == job.id,
            ResearchInboxItem.status == "accepted",
        )
    )
    items = list(result.scalars().all())
    if not items:
        return 0

    recorded_at = job.completed_at or datetime.utcnow()
    summary = summarize_follow_up_job_outcome(job)
    updated = 0
    for item in items:
        current_status = str(item.follow_up_outcome_status or "").strip().lower()
        current_summary = str(item.follow_up_outcome_summary or "").strip()
        current_recorded_at = item.follow_up_outcome_recorded_at
        if (
            current_status == status
            and current_summary == summary
            and current_recorded_at == recorded_at
        ):
            continue
        item.follow_up_outcome_status = status
        item.follow_up_outcome_recorded_at = recorded_at
        item.follow_up_outcome_summary = summary
        await _maybe_emit_follow_up_outcome_notification(
            db,
            item=item,
            job=job,
            status=status,
            summary=summary,
        )
        updated += 1
    await _project_follow_up_outcome_to_originating_opportunity(
        db,
        job=job,
        status=status,
        summary=summary,
        recorded_at=recorded_at,
    )
    return updated
