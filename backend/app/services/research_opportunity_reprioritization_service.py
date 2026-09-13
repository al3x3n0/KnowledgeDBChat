from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob
from app.models.domain_research_profile import DomainResearchProfile
from app.models.research_note import ResearchNote
from app.models.research_portfolio import ResearchPortfolio
from app.services.autonomous_agent_executor import AutonomousAgentExecutor
from app.services.autonomy_event_service import record_autonomy_decision_event
from app.services.autonomy_service import (
    build_autonomy_summary,
    current_domain_profile_policy_snapshot,
    resolve_domain_profile_automation_contract,
)
from app.services.research_opportunity_service import (
    collect_research_opportunity_linked_ids,
    compute_research_opportunity_evidence_revision,
    list_normalized_research_opportunities,
    normalize_research_opportunity,
)
from app.services.scientific_validation_service import (
    normalize_portfolio_automation_profile,
    resolve_portfolio_automation_policy,
)
from app.tasks.agent_job_tasks import execute_agent_job_task


def _text(value: Any) -> str:
    return str(value or "").strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _string_list(value: Any, *, limit: int = 12) -> list[str]:
    rows = value if isinstance(value, list) else []
    out: list[str] = []
    for row in rows:
        text = _text(row)
        if text and text not in out:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _dict_list(value: Any, *, limit: int = 12) -> list[dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        marker = repr(sorted(row.items(), key=lambda item: str(item[0])))
        if marker in seen:
            continue
        seen.add(marker)
        out.append(dict(row))
        if len(out) >= limit:
            break
    return out


def _merge_strings(*collections: Any, limit: int = 12) -> list[str]:
    out: list[str] = []
    for collection in collections:
        for item in _string_list(collection, limit=limit * 2):
            if item not in out:
                out.append(item)
            if len(out) >= limit:
                return out
    return out


def _merge_dicts(*collections: Any, limit: int = 12) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for collection in collections:
        for item in _dict_list(collection, limit=limit * 2):
            marker = repr(sorted(item.items(), key=lambda row: str(row[0])))
            if marker in seen:
                continue
            seen.add(marker)
            out.append(dict(item))
            if len(out) >= limit:
                return out
    return out


def _extract_scheduler_state(job: AgentJob | None) -> dict[str, Any] | None:
    results = job.results if job and isinstance(job.results, dict) else {}
    execution = (
        results.get("execution_strategy")
        if isinstance(results.get("execution_strategy"), dict)
        else {}
    )
    raw = (
        execution.get("scheduler_state")
        if isinstance(execution.get("scheduler_state"), dict)
        else {}
    )
    if not raw:
        return None
    return {
        "last_run_status": _text(raw.get("last_run_status")) or None,
        "failure_streak": max(0, _safe_int(raw.get("failure_streak"), 0)),
        "last_scheduled_at": raw.get("last_scheduled_at"),
        "last_dispatched_at": raw.get("last_dispatched_at"),
        "current_run_started_at": raw.get("current_run_started_at"),
        "last_successful_run_at": raw.get("last_successful_run_at"),
        "last_completed_run_at": raw.get("last_completed_run_at"),
        "last_failure_at": raw.get("last_failure_at"),
        "backoff_until": raw.get("backoff_until"),
        "backoff_seconds": max(0, _safe_int(raw.get("backoff_seconds"), 0)),
        "queue_reason": _text(raw.get("queue_reason")) or None,
    }


def _hypotheses_from_note(note: ResearchNote) -> list[dict[str, Any]]:
    payload = (
        note.structured_payload if isinstance(note.structured_payload, dict) else {}
    )
    if _text(payload.get("artifact_type")) not in {
        "hypothesis_reevaluation",
        "hypothesis_synthesis",
    }:
        return []
    rows = (
        payload.get("hypotheses") if isinstance(payload.get("hypotheses"), list) else []
    )
    return [dict(row) for row in rows if isinstance(row, dict)]


def _priority_delta_reason(note: ResearchNote, hypothesis_id: str) -> str | None:
    payload = (
        note.structured_payload if isinstance(note.structured_payload, dict) else {}
    )
    rows = (
        payload.get("priority_deltas")
        if isinstance(payload.get("priority_deltas"), list)
        else []
    )
    for row in rows:
        if not isinstance(row, dict):
            continue
        if _text(row.get("hypothesis_id")) == hypothesis_id:
            reason = _text(row.get("reason"))
            if reason:
                return reason[:400]
    summary = _text(payload.get("reprioritization_summary") or payload.get("summary"))
    return summary[:400] or None


def _hypothesis_run_ids(hypothesis: dict[str, Any]) -> list[str]:
    evidence_rows = (
        hypothesis.get("experiment_evidence")
        if isinstance(hypothesis.get("experiment_evidence"), list)
        else []
    )
    return _merge_strings(
        [_text(row.get("run_id")) for row in evidence_rows if isinstance(row, dict)],
        limit=8,
    )


def _hypothesis_supporting_sources(hypothesis: dict[str, Any]) -> list[dict[str, Any]]:
    evidence_rows = (
        hypothesis.get("experiment_evidence")
        if isinstance(hypothesis.get("experiment_evidence"), list)
        else []
    )
    evidence_sources = [
        item
        for row in evidence_rows
        for item in (
            (row.get("supporting_sources") if isinstance(row, dict) else None) or []
        )
        if isinstance(item, dict)
    ]
    return _merge_dicts(hypothesis.get("supporting_sources"), evidence_sources, limit=8)


def _hypothesis_supporting_evidence(hypothesis: dict[str, Any]) -> list[str]:
    evidence_rows = (
        hypothesis.get("experiment_evidence")
        if isinstance(hypothesis.get("experiment_evidence"), list)
        else []
    )
    text_rows: list[str] = []
    for row in evidence_rows:
        if not isinstance(row, dict):
            continue
        summary = _text(row.get("summary"))
        if summary:
            text_rows.append(summary)
        for item in _string_list(row.get("result_highlights"), limit=4):
            text_rows.append(item)
    next_step = _text(hypothesis.get("recommended_next_step"))
    if next_step:
        text_rows.append(next_step)
    return _merge_strings(hypothesis.get("supporting_evidence"), text_rows, limit=8)


def _collect_autonomous_origins(hypothesis: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    candidate_rows: list[Any] = []
    if isinstance(hypothesis.get("autonomous_origin"), dict):
        candidate_rows.append(hypothesis.get("autonomous_origin"))
    evidence_rows = (
        hypothesis.get("experiment_evidence")
        if isinstance(hypothesis.get("experiment_evidence"), list)
        else []
    )
    for row in evidence_rows:
        if isinstance(row, dict) and isinstance(row.get("autonomous_origin"), dict):
            candidate_rows.append(row.get("autonomous_origin"))

    seen: set[str] = set()
    for row in candidate_rows:
        if not isinstance(row, dict):
            continue
        source_kind = _text(row.get("source_kind")).lower()
        source_id = _text(row.get("source_id"))
        opportunity_id = _text(row.get("opportunity_id"))
        evidence_revision_at_launch = (
            _text(row.get("evidence_revision_at_launch")) or None
        )
        if (
            source_kind not in {"profile", "portfolio"}
            or not source_id
            or not opportunity_id
        ):
            continue
        marker = f"{source_kind}:{source_id}:{opportunity_id}"
        if marker in seen:
            continue
        seen.add(marker)
        out.append(
            {
                "source_kind": source_kind,
                "source_id": source_id,
                "opportunity_id": opportunity_id,
                "evidence_revision_at_launch": evidence_revision_at_launch,
            }
        )
    return out


def _matches_origin(
    *,
    row: dict[str, Any],
    note: ResearchNote,
    hypothesis_id: str,
    entity_kind: str,
    entity_id: str,
    origins: list[dict[str, str]],
) -> bool:
    opportunity_id = _text(row.get("opportunity_id"))
    if not opportunity_id:
        return False
    if any(
        origin["source_kind"] == entity_kind
        and origin["source_id"] == entity_id
        and origin["opportunity_id"] == opportunity_id
        for origin in origins
    ):
        return True
    note_id = str(note.id)
    source_note_ids = _string_list(row.get("source_note_ids"), limit=8)
    return opportunity_id == hypothesis_id and note_id in source_note_ids


def _resolve_follow_up_review_mode(
    effective_policy: dict[str, Any],
    *,
    explicit_policy: Any,
    default: str = "auto_launch_safe",
) -> str:
    """Resolve the follow-up review mode for auto-dispatch.

    A domain profile has no dedicated ``follow_up_review_mode`` column (only the
    legacy ``auto_launch_follow_up`` boolean), so the resolved effective policy
    falls back to the portfolio default of ``queue_for_approval``. Only honor an
    explicitly configured mode; otherwise default to ``auto_launch_safe`` so that
    ``auto_launch_follow_up`` profiles actually launch.
    """
    explicit = explicit_policy if isinstance(explicit_policy, dict) else {}
    raw_mode = _text(explicit.get("follow_up_review_mode")).lower()
    if raw_mode not in {"auto_launch_safe", "queue_for_approval", "manual_only"}:
        raw_mode = _text(default).lower()
    if raw_mode not in {"auto_launch_safe", "queue_for_approval", "manual_only"}:
        raw_mode = "auto_launch_safe"
    return raw_mode


def _current_follow_up_review_applies(
    row: dict[str, Any], *, evidence_revision: str
) -> bool:
    return _text(row.get("follow_up_review_evidence_revision")) == evidence_revision


def _follow_up_staging_baseline(row: dict[str, Any]) -> str:
    return (
        "planned"
        if _string_list(row.get("linked_experiment_plan_ids"), limit=1)
        else "accepted"
    )


async def _resolve_parent_job_for_profile(
    db: AsyncSession, profile: DomainResearchProfile
) -> AgentJob | None:
    parent_job_id = profile.latest_run_job_id or profile.active_job_id
    if not parent_job_id:
        return None
    parent_job = await db.get(AgentJob, parent_job_id)
    if not parent_job or parent_job.user_id != profile.user_id:
        return None
    return parent_job


async def _resolve_parent_job_for_portfolio(
    db: AsyncSession, portfolio: ResearchPortfolio
) -> AgentJob | None:
    parent_job_id = portfolio.latest_run_job_id or portfolio.active_job_id
    if not parent_job_id:
        return None
    parent_job = await db.get(AgentJob, parent_job_id)
    if not parent_job or parent_job.user_id != portfolio.user_id:
        return None
    return parent_job


async def _record_follow_up_dispatch_event(
    *,
    db: AsyncSession,
    entity_kind: str,
    entity_id: str,
    entity_label: str,
    user_id: Any,
    note: ResearchNote,
    opportunity_before: dict[str, Any],
    opportunity_after: dict[str, Any],
    event_type: str,
    reason_code: str | None,
    summary: str,
    scheduler_state: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> None:
    deep_link = (
        {
            "target_tab": "domain",
            "params": {"tab": "domain"},
            "label": "Open Domain Profiles",
        }
        if entity_kind == "domain_profile"
        else {
            "target_tab": "fleet",
            "params": {"tab": "fleet", "fleetId": entity_id},
            "label": "Open Research Fleet",
        }
    )
    await record_autonomy_decision_event(
        db=db,
        user_id=user_id,
        event_type=event_type,
        event_time=datetime.utcnow(),
        source_kind=entity_kind,
        source_id=entity_id,
        source_label=entity_label,
        customer=str(getattr(note, "title", "") or "").strip() or None,
        decision_type=event_type,
        reason_code=reason_code,
        status=_text(opportunity_after.get("stage")) or None,
        severity="medium",
        actor_mode="system",
        summary=summary[:500],
        reason_label=reason_code.replace("_", " ").strip().title()
        if reason_code
        else None,
        scheduler_state=scheduler_state,
        before_state=opportunity_before,
        after_state=opportunity_after,
        deep_link=deep_link,
        metadata=metadata
        or {
            "opportunity_id": opportunity_after.get("opportunity_id"),
            "source_note_id": str(note.id),
        },
    )


async def _record_reevaluation_closeout_event(
    *,
    db: AsyncSession,
    entity_kind: str,
    entity_id: str,
    entity_label: str,
    user_id: Any,
    note: ResearchNote,
    opportunity_before: dict[str, Any],
    opportunity_after: dict[str, Any],
    outcome_status: str,
    review_job_id: str,
    review_note: str | None,
    reviewed_at: str | None = None,
    source_note_id: str | None = None,
    target_note_id: str | None = None,
) -> None:
    event_type = f"reevaluation_{_text(outcome_status).lower()}"
    opportunity_id = _text(opportunity_after.get("opportunity_id"))
    deep_link = (
        {
            "target_tab": "domain",
            "params": {
                "tab": "domain",
                "profileId": entity_id,
                "opportunityId": opportunity_id,
            },
            "label": "Open Domain Opportunity",
        }
        if entity_kind == "domain_profile"
        else {
            "target_tab": "fleet",
            "params": {
                "tab": "fleet",
                "fleetId": entity_id,
                "opportunityId": opportunity_id,
            },
            "label": "Open Research Fleet Opportunity",
        }
    )
    summary = f"{entity_label}: reevaluation {_text(outcome_status).replace('_', ' ')} for {_text(opportunity_after.get('title') or 'opportunity')}".strip()[
        :500
    ]
    await record_autonomy_decision_event(
        db=db,
        user_id=user_id,
        event_type=event_type,
        event_time=datetime.utcnow(),
        source_kind=entity_kind,
        source_id=entity_id,
        source_label=entity_label,
        customer=str(getattr(note, "title", "") or "").strip() or None,
        decision_type=event_type,
        reason_code="reevaluation_review_closeout",
        status=_text(opportunity_after.get("stage")) or None,
        severity="low",
        actor_mode="operator",
        summary=summary,
        operator_note=review_note,
        before_state=opportunity_before,
        after_state=opportunity_after,
        deep_link=deep_link,
        metadata={
            "opportunity_id": opportunity_id,
            "source_note_id": _text(source_note_id) or str(note.id),
            "target_note_id": _text(target_note_id) or None,
            "review_job_id": review_job_id,
            "reevaluation_job_id": review_job_id,
            "review_outcome_status": outcome_status,
            "review_recorded_at": _text(reviewed_at) or None,
        },
    )


async def _auto_dispatch_profile_follow_ups(
    *,
    db: AsyncSession,
    note: ResearchNote,
    profile: DomainResearchProfile,
    opportunities: list[dict[str, Any]],
    effective_policy: dict[str, Any],
    changed_opportunity_ids: set[str],
    follow_up_review_mode: str,
) -> bool:
    if not changed_opportunity_ids or not bool(
        effective_policy.get("auto_launch_follow_up", True)
    ):
        return False

    if follow_up_review_mode not in {
        "auto_launch_safe",
        "queue_for_approval",
        "manual_only",
    }:
        follow_up_review_mode = "auto_launch_safe"

    max_auto_launches = max(
        0, _safe_int(effective_policy.get("max_auto_follow_up_launches"), 1)
    )
    parent_job = (
        await _resolve_parent_job_for_profile(db, profile)
        if follow_up_review_mode == "auto_launch_safe"
        else None
    )
    scheduler_state = _extract_scheduler_state(parent_job)
    executor = (
        AutonomousAgentExecutor()
        if follow_up_review_mode == "auto_launch_safe"
        else None
    )
    launched_count = 0
    changed_any = False

    for idx, row in enumerate(opportunities):
        opportunity_id = _text(row.get("opportunity_id"))
        if opportunity_id not in changed_opportunity_ids:
            continue
        current = normalize_research_opportunity(row)
        evidence_revision = _text(current.get("evidence_revision"))
        review_status = _text(current.get("follow_up_review_status")).lower()
        baseline_stage = _follow_up_staging_baseline(current)
        if _text(current.get("autonomy_state")) != "eligible":
            continue
        if _string_list(current.get("child_job_ids"), limit=1):
            continue
        if review_status == "pending_approval" and _current_follow_up_review_applies(
            current, evidence_revision=evidence_revision
        ):
            continue
        if review_status == "rejected" and _current_follow_up_review_applies(
            current, evidence_revision=evidence_revision
        ):
            continue

        updated = deepcopy(current)
        event_type: str | None = None
        reason_code: str | None = None
        summary: str | None = None

        if follow_up_review_mode == "queue_for_approval":
            updated["follow_up_review_status"] = "pending_approval"
            updated["follow_up_review_evidence_revision"] = evidence_revision
            updated["stage"] = baseline_stage
            updated["last_decision_type"] = "follow_up_queued_for_approval"
            updated["last_decision_reason_code"] = "follow_up_pending_approval"
            event_type = "follow_up_queued_for_approval"
            reason_code = "follow_up_pending_approval"
            summary = f"{str(profile.title or 'Domain profile').strip()}: queued follow-up approval for {str(updated.get('title') or 'opportunity').strip()}"
        elif follow_up_review_mode == "manual_only":
            updated["follow_up_review_status"] = "manual_recommendation"
            updated["follow_up_review_evidence_revision"] = evidence_revision
            updated["stage"] = baseline_stage
            updated["last_decision_type"] = "follow_up_manual_recommendation"
            updated["last_decision_reason_code"] = "manual_follow_up_recommendation"
            event_type = "follow_up_manual_recommendation"
            reason_code = "manual_follow_up_recommendation"
            summary = f"{str(profile.title or 'Domain profile').strip()}: marked {str(updated.get('title') or 'opportunity').strip()} for manual follow-up review"
        elif (
            parent_job is not None
            and launched_count < max_auto_launches
            and executor is not None
        ):
            child_job = await executor._create_domain_research_follow_up_job(
                db=db,
                job=parent_job,
                domain=profile.domain,
                objective=profile.objective,
                customer_context=str(profile.customer_context or ""),
                track_type=str(profile.track_type or "generic"),
                source_scope=str(profile.source_scope or "kb_plus_arxiv"),
                top_idea=updated,
                docs=[],
                repo_documents=[],
                papers=[],
                repo_source_ids=[
                    str(v) for v in (profile.repo_source_ids or []) if _text(v)
                ],
                benchmark_queries=[
                    str(v) for v in (profile.benchmark_queries or []) if _text(v)
                ],
                automation_profile=profile.automation_profile,
                automation_policy=effective_policy,
                sandbox_profile_id=profile.sandbox_profile_id,
                profile_id=str(profile.id),
            )
            if child_job is None:
                continue
            child_id = str(child_job.id)
            execute_agent_job_task.delay(child_id, str(profile.user_id))
            updated["child_job_ids"] = _merge_strings(
                updated.get("child_job_ids"), [child_id], limit=8
            )
            updated["decision_state"] = "auto_accepted"
            updated["stage"] = "validating"
            updated["follow_up_review_status"] = "approved_launch"
            updated["follow_up_review_evidence_revision"] = evidence_revision
            updated["last_decision_type"] = "follow_up_launched"
            updated["last_decision_reason_code"] = None
            event_type = "follow_up_launched"
            summary = f"{str(profile.title or 'Domain profile').strip()}: auto-launched follow-up for {str(updated.get('title') or 'opportunity').strip()}"
            launched_count += 1
        else:
            continue

        normalized = normalize_research_opportunity(updated)
        opportunities[idx] = normalized
        changed_any = True
        await _record_follow_up_dispatch_event(
            db=db,
            entity_kind="domain_profile",
            entity_id=str(profile.id),
            entity_label=str(profile.title or "Domain profile").strip(),
            user_id=profile.user_id,
            note=note,
            opportunity_before=current,
            opportunity_after=normalized,
            event_type=event_type,
            reason_code=reason_code,
            summary=summary
            or "Opportunity follow-up status updated from reevaluation.",
            scheduler_state=scheduler_state,
        )
    return changed_any


async def _auto_dispatch_portfolio_follow_ups(
    *,
    db: AsyncSession,
    note: ResearchNote,
    portfolio: ResearchPortfolio,
    opportunities: list[dict[str, Any]],
    effective_policy: dict[str, Any],
    changed_opportunity_ids: set[str],
    follow_up_review_mode: str,
) -> bool:
    if not changed_opportunity_ids or not bool(
        effective_policy.get("auto_launch_follow_up", True)
    ):
        return False

    if follow_up_review_mode not in {
        "auto_launch_safe",
        "queue_for_approval",
        "manual_only",
    }:
        follow_up_review_mode = "auto_launch_safe"

    max_auto_launches = max(
        0, _safe_int(effective_policy.get("max_auto_follow_up_launches"), 1)
    )
    parent_job = (
        await _resolve_parent_job_for_portfolio(db, portfolio)
        if follow_up_review_mode == "auto_launch_safe"
        else None
    )
    scheduler_state = _extract_scheduler_state(parent_job)
    executor = (
        AutonomousAgentExecutor()
        if follow_up_review_mode == "auto_launch_safe"
        else None
    )
    launched_count = 0
    changed_any = False

    for idx, row in enumerate(opportunities):
        opportunity_id = _text(row.get("opportunity_id"))
        if opportunity_id not in changed_opportunity_ids:
            continue
        current = normalize_research_opportunity(row)
        evidence_revision = _text(current.get("evidence_revision"))
        review_status = _text(current.get("follow_up_review_status")).lower()
        baseline_stage = _follow_up_staging_baseline(current)
        if _text(current.get("autonomy_state")) != "eligible":
            continue
        if _string_list(current.get("child_job_ids"), limit=1):
            continue
        if review_status == "pending_approval" and _current_follow_up_review_applies(
            current, evidence_revision=evidence_revision
        ):
            continue
        if review_status == "rejected" and _current_follow_up_review_applies(
            current, evidence_revision=evidence_revision
        ):
            continue

        updated = deepcopy(current)
        event_type: str | None = None
        reason_code: str | None = None
        summary: str | None = None

        if follow_up_review_mode == "queue_for_approval":
            updated["follow_up_review_status"] = "pending_approval"
            updated["follow_up_review_evidence_revision"] = evidence_revision
            updated["stage"] = baseline_stage
            updated["last_decision_type"] = "follow_up_queued_for_approval"
            updated["last_decision_reason_code"] = "follow_up_pending_approval"
            event_type = "follow_up_queued_for_approval"
            reason_code = "follow_up_pending_approval"
            summary = f"{str(portfolio.title or 'Research fleet').strip()}: queued follow-up approval for {str(updated.get('title') or 'opportunity').strip()}"
        elif follow_up_review_mode == "manual_only":
            updated["follow_up_review_status"] = "manual_recommendation"
            updated["follow_up_review_evidence_revision"] = evidence_revision
            updated["stage"] = baseline_stage
            updated["last_decision_type"] = "follow_up_manual_recommendation"
            updated["last_decision_reason_code"] = "manual_follow_up_recommendation"
            event_type = "follow_up_manual_recommendation"
            reason_code = "manual_follow_up_recommendation"
            summary = f"{str(portfolio.title or 'Research fleet').strip()}: marked {str(updated.get('title') or 'opportunity').strip()} for manual follow-up review"
        elif (
            parent_job is not None
            and launched_count < max_auto_launches
            and executor is not None
        ):
            child_job = await executor._create_domain_research_follow_up_job(
                db=db,
                job=parent_job,
                domain=str(updated.get("title") or portfolio.title),
                objective=portfolio.objective,
                customer_context="research_portfolio",
                track_type=str(updated.get("track_type") or "generic"),
                source_scope="kb_plus_arxiv_plus_repo"
                if updated.get("source_repo_ids")
                else "kb_plus_arxiv",
                top_idea=updated,
                docs=[],
                repo_documents=[],
                papers=[],
                repo_source_ids=[
                    str(v) for v in (updated.get("source_repo_ids") or []) if _text(v)
                ],
                benchmark_queries=[],
                automation_profile=portfolio.automation_profile,
                automation_policy=effective_policy,
                sandbox_profile_id=portfolio.sandbox_profile_id,
            )
            if child_job is None:
                continue
            child_id = str(child_job.id)
            execute_agent_job_task.delay(child_id, str(portfolio.user_id))
            updated["child_job_ids"] = _merge_strings(
                updated.get("child_job_ids"), [child_id], limit=8
            )
            updated["decision_state"] = "auto_accepted"
            updated["stage"] = "validating"
            updated["follow_up_review_status"] = "approved_launch"
            updated["follow_up_review_evidence_revision"] = evidence_revision
            updated["last_decision_type"] = "follow_up_launched"
            updated["last_decision_reason_code"] = None
            event_type = "follow_up_launched"
            summary = f"{str(portfolio.title or 'Research fleet').strip()}: auto-launched follow-up for {str(updated.get('title') or 'opportunity').strip()}"
            launched_count += 1
        else:
            continue

        normalized = normalize_research_opportunity(updated)
        opportunities[idx] = normalized
        changed_any = True
        await _record_follow_up_dispatch_event(
            db=db,
            entity_kind="portfolio",
            entity_id=str(portfolio.id),
            entity_label=str(portfolio.title or "Research fleet").strip(),
            user_id=portfolio.user_id,
            note=note,
            opportunity_before=current,
            opportunity_after=normalized,
            event_type=event_type,
            reason_code=reason_code,
            summary=summary or "Portfolio follow-up status updated from reevaluation.",
            scheduler_state=scheduler_state,
        )
    return changed_any


def _reprioritize_opportunity(
    *,
    row: dict[str, Any],
    note: ResearchNote,
    hypothesis: dict[str, Any],
    entity_kind: str,
    entity_id: str,
    effective_policy: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    now_iso = datetime.utcnow().isoformat()
    previous = normalize_research_opportunity(row)
    hypothesis_id = _text(hypothesis.get("id"))
    next_confidence = round(
        max(
            0.0,
            min(
                _safe_float(
                    hypothesis.get("evidence_score"),
                    _safe_float(
                        hypothesis.get("overall_score"),
                        previous.get("confidence") or 0.0,
                    ),
                ),
                1.0,
            ),
        ),
        4,
    )
    next_readiness = round(
        max(
            0.0,
            min(
                _safe_float(
                    hypothesis.get("overall_score"), previous.get("readiness") or 0.0
                ),
                1.0,
            ),
        ),
        4,
    )
    next_novelty = round(
        max(
            0.0,
            min(
                _safe_float(
                    hypothesis.get("novelty_score"), previous.get("novelty") or 0.0
                ),
                1.0,
            ),
        ),
        4,
    )
    source_run_ids = _hypothesis_run_ids(hypothesis)
    supporting_sources = _hypothesis_supporting_sources(hypothesis)
    supporting_evidence = _hypothesis_supporting_evidence(hypothesis)
    reprioritization_reason = _priority_delta_reason(note, hypothesis_id)
    merged_source_note_ids = _merge_strings(
        previous.get("source_note_ids"), [str(note.id)], limit=8
    )

    if (
        (_text(hypothesis.get("title")) or previous.get("title"))
        == previous.get("title")
        and (_text(hypothesis.get("claim")) or previous.get("hypothesis"))
        == previous.get("hypothesis")
        and next_confidence == previous.get("confidence")
        and next_readiness == previous.get("readiness")
        and next_novelty == previous.get("novelty")
        and supporting_sources == previous.get("supporting_sources")
        and supporting_evidence == previous.get("supporting_evidence")
        and merged_source_note_ids == previous.get("source_note_ids")
        and source_run_ids == previous.get("reprioritization_source_run_ids")
        and reprioritization_reason == previous.get("reprioritization_reason")
    ):
        return previous, False

    updated = deepcopy(previous)
    updated["title"] = _text(hypothesis.get("title")) or previous.get("title")
    updated["hypothesis"] = _text(hypothesis.get("claim")) or previous.get("hypothesis")
    updated["confidence"] = next_confidence
    updated["readiness"] = next_readiness
    updated["novelty"] = next_novelty
    updated["supporting_sources"] = supporting_sources
    updated["supporting_evidence"] = supporting_evidence
    updated["source_note_ids"] = merged_source_note_ids
    updated["updated_at"] = now_iso
    updated["last_material_change_at"] = now_iso
    updated["reprioritized_at"] = now_iso
    updated["reprioritization_reason"] = reprioritization_reason
    updated["reprioritization_source_run_ids"] = source_run_ids
    updated["prior_confidence"] = round(_safe_float(previous.get("confidence"), 0.0), 4)
    updated["prior_readiness"] = round(_safe_float(previous.get("readiness"), 0.0), 4)
    updated["autonomous_origin"] = {
        "source_kind": entity_kind,
        "source_id": entity_id,
        "opportunity_id": _text(previous.get("opportunity_id")),
        "evidence_revision_at_launch": _text(
            (previous.get("autonomous_origin") or {}).get("evidence_revision_at_launch")
            if isinstance(previous.get("autonomous_origin"), dict)
            else None
        )
        or previous.get("evidence_revision"),
    }

    existing_revision = _text(previous.get("evidence_revision"))
    updated["evidence_revision"] = compute_research_opportunity_evidence_revision(
        updated
    )
    evidence_changed = existing_revision != _text(updated.get("evidence_revision"))

    if evidence_changed and not _string_list(previous.get("child_job_ids"), limit=1):
        updated["follow_up_review_status"] = None
        updated["follow_up_reviewed_at"] = None
        updated["follow_up_reviewed_by_user_id"] = None
        updated["follow_up_review_note"] = None
        updated["follow_up_review_evidence_revision"] = None

    confidence_threshold = max(
        0.0, min(_safe_float(effective_policy.get("confidence_threshold"), 0.72), 1.0)
    )
    readiness_threshold = max(
        0.0,
        min(
            _safe_float(effective_policy.get("experiment_readiness_threshold"), 0.8),
            1.0,
        ),
    )
    has_active_links = any(
        _string_list(previous.get(key), limit=1)
        for key in (
            "linked_experiment_plan_ids",
            "linked_validation_run_ids",
            "child_job_ids",
        )
    )

    below_policy = (
        next_confidence < confidence_threshold or next_readiness < readiness_threshold
    )
    if (
        below_policy
        and not has_active_links
        and str(previous.get("decision_state") or "") != "suppressed"
    ):
        updated["stage"] = "blocked"
        updated["last_blocked_reason_code"] = "reprioritized_below_policy_threshold"
        updated["last_decision_reason_code"] = "reprioritized_from_experiment_evidence"
    else:
        if (
            str(previous.get("stage") or "") in {"completed", "blocked"}
            and not has_active_links
            and str(previous.get("decision_state") or "") != "suppressed"
        ):
            updated["stage"] = "accepted"
        updated["last_blocked_reason_code"] = None
        updated["last_decision_reason_code"] = "reprioritized_from_experiment_evidence"

    candidate = normalize_research_opportunity(updated)
    for key in (
        "reprioritized_at",
        "reprioritization_reason",
        "reprioritization_source_run_ids",
        "prior_confidence",
        "prior_readiness",
        "autonomous_origin",
    ):
        candidate[key] = updated.get(key)

    changed = any(
        candidate.get(key) != previous.get(key)
        for key in (
            "title",
            "hypothesis",
            "confidence",
            "readiness",
            "novelty",
            "supporting_sources",
            "supporting_evidence",
            "stage",
            "autonomy_state",
            "evidence_revision",
            "last_blocked_reason_code",
            "last_decision_reason_code",
            "follow_up_review_status",
            "reprioritization_source_run_ids",
        )
    )
    return candidate, changed


async def _sync_profile(
    profile: DomainResearchProfile, opportunities: list[dict[str, Any]]
) -> None:
    automation_profile, effective_policy = resolve_domain_profile_automation_contract(
        automation_profile=getattr(profile, "automation_profile", None),
        automation_policy=getattr(profile, "automation_policy", None),
        current_snapshot=current_domain_profile_policy_snapshot(profile),
    )
    normalized = list_normalized_research_opportunities(opportunities)
    linked_ids = collect_research_opportunity_linked_ids(normalized)
    summary = (
        dict(profile.latest_summary) if isinstance(profile.latest_summary, dict) else {}
    )
    profile.latest_summary = build_autonomy_summary(
        raw_summary=summary,
        opportunities=normalized,
        automation_profile=automation_profile,
        effective_policy=effective_policy,
        sandbox_profile_id=profile.sandbox_profile_id,
        config_revision_key="profile_config_revision",
    )
    profile.latest_note_ids = list(
        dict.fromkeys(
            [
                *(_string_list(profile.latest_note_ids, limit=32)),
                *linked_ids["note_ids"],
            ]
        )
    )[:20]
    profile.latest_experiment_plan_ids = linked_ids["plan_ids"][:20]
    profile.latest_validation_run_ids = linked_ids["run_ids"][:20]


async def _sync_portfolio(
    portfolio: ResearchPortfolio, opportunities: list[dict[str, Any]]
) -> None:
    normalized = list_normalized_research_opportunities(opportunities)
    linked_ids = collect_research_opportunity_linked_ids(normalized)
    portfolio.opportunities = normalized
    summary = (
        dict(portfolio.latest_summary)
        if isinstance(portfolio.latest_summary, dict)
        else {}
    )
    effective_policy = resolve_portfolio_automation_policy(
        portfolio.automation_profile, portfolio.automation_policy
    )
    portfolio.latest_summary = build_autonomy_summary(
        raw_summary=summary,
        opportunities=normalized,
        automation_profile=normalize_portfolio_automation_profile(
            portfolio.automation_profile, default="balanced"
        ),
        effective_policy=effective_policy,
        sandbox_profile_id=portfolio.sandbox_profile_id,
        config_revision_key="portfolio_config_revision",
    )
    portfolio.latest_note_ids = list(
        dict.fromkeys(
            [
                *(_string_list(portfolio.latest_note_ids, limit=48)),
                *linked_ids["note_ids"],
            ]
        )
    )[:30]
    portfolio.latest_experiment_plan_ids = linked_ids["plan_ids"][:30]
    portfolio.latest_validation_run_ids = linked_ids["run_ids"][:30]
    portfolio.child_job_ids = linked_ids["child_job_ids"][:50]


async def project_note_reevaluation_to_autonomous_opportunities(
    *,
    db: AsyncSession,
    note: ResearchNote,
) -> dict[str, int]:
    hypotheses = _hypotheses_from_note(note)
    if not hypotheses:
        return {
            "profiles_updated": 0,
            "portfolios_updated": 0,
            "opportunities_updated": 0,
        }

    hypothesis_by_id = {
        _text(hypothesis.get("id")): hypothesis
        for hypothesis in hypotheses
        if _text(hypothesis.get("id"))
    }
    origin_rows = [
        origin
        for hypothesis in hypotheses
        for origin in _collect_autonomous_origins(hypothesis)
    ]
    profile_origin_ids = list(
        dict.fromkeys(
            origin["source_id"]
            for origin in origin_rows
            if origin.get("source_kind") == "profile" and _text(origin.get("source_id"))
        )
    )
    portfolio_origin_ids = list(
        dict.fromkeys(
            origin["source_id"]
            for origin in origin_rows
            if origin.get("source_kind") == "portfolio"
            and _text(origin.get("source_id"))
        )
    )
    if profile_origin_ids:
        profile_rows: list[DomainResearchProfile] = []
        for value in profile_origin_ids:
            try:
                profile = await db.get(DomainResearchProfile, UUID(value))
            except (TypeError, ValueError, AttributeError):
                continue
            if profile and profile.user_id == note.user_id:
                profile_rows.append(profile)
    else:
        profile_rows = list(
            (
                await db.execute(
                    select(DomainResearchProfile).where(
                        DomainResearchProfile.user_id == note.user_id
                    )
                )
            )
            .scalars()
            .all()
        )
    if portfolio_origin_ids:
        portfolio_rows: list[ResearchPortfolio] = []
        for value in portfolio_origin_ids:
            try:
                portfolio = await db.get(ResearchPortfolio, UUID(value))
            except (TypeError, ValueError, AttributeError):
                continue
            if portfolio and portfolio.user_id == note.user_id:
                portfolio_rows.append(portfolio)
    else:
        portfolio_rows = list(
            (
                await db.execute(
                    select(ResearchPortfolio).where(
                        ResearchPortfolio.user_id == note.user_id
                    )
                )
            )
            .scalars()
            .all()
        )

    profiles_updated = 0
    portfolios_updated = 0
    opportunities_updated = 0

    for profile in profile_rows:
        summary = (
            profile.latest_summary if isinstance(profile.latest_summary, dict) else {}
        )
        opportunities = list_normalized_research_opportunities(
            summary.get("opportunities") or summary.get("idea_candidates")
        )
        if not opportunities:
            continue
        (
            _automation_profile,
            effective_policy,
        ) = resolve_domain_profile_automation_contract(
            automation_profile=getattr(profile, "automation_profile", None),
            automation_policy=getattr(profile, "automation_policy", None),
            current_snapshot=current_domain_profile_policy_snapshot(profile),
        )
        changed_any = False
        changed_opportunity_ids: set[str] = set()
        next_rows: list[dict[str, Any]] = []
        for row in opportunities:
            updated_row = row
            for hypothesis_id, hypothesis in hypothesis_by_id.items():
                origins = _collect_autonomous_origins(hypothesis)
                if not _matches_origin(
                    row=row,
                    note=note,
                    hypothesis_id=hypothesis_id,
                    entity_kind="profile",
                    entity_id=str(profile.id),
                    origins=origins,
                ):
                    continue
                updated_row, changed = _reprioritize_opportunity(
                    row=updated_row,
                    note=note,
                    hypothesis=hypothesis,
                    entity_kind="profile",
                    entity_id=str(profile.id),
                    effective_policy=effective_policy,
                )
                changed_any = changed_any or changed
                if changed:
                    changed_opportunity_ids.add(
                        _text(updated_row.get("opportunity_id"))
                    )
                    opportunities_updated += 1
                break
            next_rows.append(updated_row)
        if changed_any:
            dispatch_changed = await _auto_dispatch_profile_follow_ups(
                db=db,
                note=note,
                profile=profile,
                opportunities=next_rows,
                effective_policy=effective_policy,
                changed_opportunity_ids=changed_opportunity_ids,
                follow_up_review_mode=_resolve_follow_up_review_mode(
                    effective_policy,
                    explicit_policy=getattr(profile, "automation_policy", None),
                ),
            )
            changed_any = changed_any or dispatch_changed
            await _sync_profile(profile, next_rows)
            profiles_updated += 1

    for portfolio in portfolio_rows:
        opportunities = list_normalized_research_opportunities(portfolio.opportunities)
        if not opportunities:
            continue
        effective_policy = resolve_portfolio_automation_policy(
            portfolio.automation_profile, portfolio.automation_policy
        )
        changed_any = False
        changed_opportunity_ids: set[str] = set()
        next_rows: list[dict[str, Any]] = []
        for row in opportunities:
            updated_row = row
            for hypothesis_id, hypothesis in hypothesis_by_id.items():
                origins = _collect_autonomous_origins(hypothesis)
                if not _matches_origin(
                    row=row,
                    note=note,
                    hypothesis_id=hypothesis_id,
                    entity_kind="portfolio",
                    entity_id=str(portfolio.id),
                    origins=origins,
                ):
                    continue
                updated_row, changed = _reprioritize_opportunity(
                    row=updated_row,
                    note=note,
                    hypothesis=hypothesis,
                    entity_kind="portfolio",
                    entity_id=str(portfolio.id),
                    effective_policy=effective_policy,
                )
                changed_any = changed_any or changed
                if changed:
                    changed_opportunity_ids.add(
                        _text(updated_row.get("opportunity_id"))
                    )
                    opportunities_updated += 1
                break
            next_rows.append(updated_row)
        if changed_any:
            dispatch_changed = await _auto_dispatch_portfolio_follow_ups(
                db=db,
                note=note,
                portfolio=portfolio,
                opportunities=next_rows,
                effective_policy=effective_policy,
                changed_opportunity_ids=changed_opportunity_ids,
                follow_up_review_mode=_resolve_follow_up_review_mode(
                    effective_policy,
                    explicit_policy=getattr(portfolio, "automation_policy", None),
                ),
            )
            changed_any = changed_any or dispatch_changed
            await _sync_portfolio(portfolio, next_rows)
            portfolios_updated += 1

    return {
        "profiles_updated": profiles_updated,
        "portfolios_updated": portfolios_updated,
        "opportunities_updated": opportunities_updated,
    }


async def project_reevaluation_review_to_autonomous_opportunities(
    *,
    db: AsyncSession,
    note: ResearchNote,
    review_outcome_status: str,
    review_job_id: str,
    review_note: str | None = None,
    reviewed_at: str | None = None,
    source_note_id: str | None = None,
    target_note_id: str | None = None,
) -> dict[str, int]:
    hypotheses = _hypotheses_from_note(note)
    review_outcome = _text(review_outcome_status)
    review_job = _text(review_job_id)
    reviewed_at_value = _text(reviewed_at) or datetime.utcnow().isoformat()
    source_note_value = _text(source_note_id) or str(note.id)
    target_note_value = _text(target_note_id) or None
    if not hypotheses or not review_outcome or not review_job:
        return {
            "profiles_updated": 0,
            "portfolios_updated": 0,
            "opportunities_updated": 0,
        }

    origin_rows = [
        origin
        for hypothesis in hypotheses
        for origin in _collect_autonomous_origins(hypothesis)
    ]
    profile_origin_ids = list(
        dict.fromkeys(
            origin["source_id"]
            for origin in origin_rows
            if origin.get("source_kind") == "profile" and _text(origin.get("source_id"))
        )
    )
    portfolio_origin_ids = list(
        dict.fromkeys(
            origin["source_id"]
            for origin in origin_rows
            if origin.get("source_kind") == "portfolio"
            and _text(origin.get("source_id"))
        )
    )

    profile_rows: list[DomainResearchProfile] = []
    for value in profile_origin_ids:
        try:
            profile = await db.get(DomainResearchProfile, UUID(value))
        except (TypeError, ValueError, AttributeError):
            continue
        if profile and profile.user_id == note.user_id:
            profile_rows.append(profile)

    portfolio_rows: list[ResearchPortfolio] = []
    for value in portfolio_origin_ids:
        try:
            portfolio = await db.get(ResearchPortfolio, UUID(value))
        except (TypeError, ValueError, AttributeError):
            continue
        if portfolio and portfolio.user_id == note.user_id:
            portfolio_rows.append(portfolio)

    profiles_updated = 0
    portfolios_updated = 0
    opportunities_updated = 0

    for profile in profile_rows:
        summary = (
            profile.latest_summary if isinstance(profile.latest_summary, dict) else {}
        )
        opportunities = list_normalized_research_opportunities(
            summary.get("opportunities") or summary.get("idea_candidates")
        )
        if not opportunities:
            continue
        changed_any = False
        next_rows: list[dict[str, Any]] = []
        for row in opportunities:
            updated_row = row
            for hypothesis in hypotheses:
                hypothesis_id = _text(hypothesis.get("id"))
                origins = _collect_autonomous_origins(hypothesis)
                if not _matches_origin(
                    row=updated_row,
                    note=note,
                    hypothesis_id=hypothesis_id,
                    entity_kind="profile",
                    entity_id=str(profile.id),
                    origins=origins,
                ):
                    continue
                updated = deepcopy(updated_row)
                updated["last_reevaluation_review_outcome"] = review_outcome
                updated["last_reevaluation_reviewed_at"] = reviewed_at_value
                updated["last_reevaluation_review_job_id"] = review_job
                updated["last_reevaluation_review_note"] = _text(review_note) or None
                updated["last_reevaluation_review_source_note_id"] = source_note_value
                updated["last_reevaluation_review_target_note_id"] = target_note_value
                normalized = normalize_research_opportunity(updated)
                await _record_reevaluation_closeout_event(
                    db=db,
                    entity_kind="domain_profile",
                    entity_id=str(profile.id),
                    entity_label=str(profile.title or "Domain profile").strip(),
                    user_id=profile.user_id,
                    note=note,
                    opportunity_before=updated_row,
                    opportunity_after=normalized,
                    outcome_status=review_outcome,
                    review_job_id=review_job,
                    review_note=_text(review_note) or None,
                    reviewed_at=reviewed_at_value,
                    source_note_id=source_note_value,
                    target_note_id=target_note_value,
                )
                updated_row = normalized
                changed_any = True
                opportunities_updated += 1
                break
            next_rows.append(updated_row)
        if changed_any:
            await _sync_profile(profile, next_rows)
            profiles_updated += 1

    for portfolio in portfolio_rows:
        opportunities = list_normalized_research_opportunities(portfolio.opportunities)
        if not opportunities:
            continue
        changed_any = False
        next_rows: list[dict[str, Any]] = []
        for row in opportunities:
            updated_row = row
            for hypothesis in hypotheses:
                hypothesis_id = _text(hypothesis.get("id"))
                origins = _collect_autonomous_origins(hypothesis)
                if not _matches_origin(
                    row=updated_row,
                    note=note,
                    hypothesis_id=hypothesis_id,
                    entity_kind="portfolio",
                    entity_id=str(portfolio.id),
                    origins=origins,
                ):
                    continue
                updated = deepcopy(updated_row)
                updated["last_reevaluation_review_outcome"] = review_outcome
                updated["last_reevaluation_reviewed_at"] = reviewed_at_value
                updated["last_reevaluation_review_job_id"] = review_job
                updated["last_reevaluation_review_note"] = _text(review_note) or None
                updated["last_reevaluation_review_source_note_id"] = source_note_value
                updated["last_reevaluation_review_target_note_id"] = target_note_value
                normalized = normalize_research_opportunity(updated)
                await _record_reevaluation_closeout_event(
                    db=db,
                    entity_kind="portfolio",
                    entity_id=str(portfolio.id),
                    entity_label=str(portfolio.title or "Research fleet").strip(),
                    user_id=portfolio.user_id,
                    note=note,
                    opportunity_before=updated_row,
                    opportunity_after=normalized,
                    outcome_status=review_outcome,
                    review_job_id=review_job,
                    review_note=_text(review_note) or None,
                    reviewed_at=reviewed_at_value,
                    source_note_id=source_note_value,
                    target_note_id=target_note_value,
                )
                updated_row = normalized
                changed_any = True
                opportunities_updated += 1
                break
            next_rows.append(updated_row)
        if changed_any:
            portfolio.opportunities = next_rows
            await _sync_portfolio(portfolio, next_rows)
            portfolios_updated += 1

    return {
        "profiles_updated": profiles_updated,
        "portfolios_updated": portfolios_updated,
        "opportunities_updated": opportunities_updated,
    }
