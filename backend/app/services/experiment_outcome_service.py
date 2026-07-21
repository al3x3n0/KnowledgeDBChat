"""
Helpers for projecting experiment-run outcomes back to originating research opportunities.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.domain_research_profile import DomainResearchProfile
from app.models.experiment import ExperimentPlan, ExperimentRun
from app.models.research_portfolio import ResearchPortfolio
from app.services.research_opportunity_service import (
    list_normalized_research_opportunities,
    normalize_research_opportunity,
)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _clean_list(values: Any, *, limit: int = 30) -> list[str]:
    rows: list[str] = []
    for value in values if isinstance(values, list) else []:
        text = _text(value)
        if text and text not in rows:
            rows.append(text)
        if len(rows) >= limit:
            break
    return rows


def _normalize_summary_text(value: object, *, limit: int = 280) -> str | None:
    text = _text(value)
    if not text:
        return None
    normalized = re.sub(r"\s+", " ", text).strip()
    return normalized[:limit] if normalized else None


def _get_model_id(db_value: object | None) -> str | None:
    text = _text(db_value)
    return text or None


def _scientific_validation_payload(run: ExperimentRun) -> dict[str, Any]:
    config = run.config if isinstance(run.config, dict) else {}
    payload = config.get("scientific_validation")
    return dict(payload) if isinstance(payload, dict) else {}


def _execution_handoff(run: ExperimentRun) -> dict[str, Any]:
    config = run.config if isinstance(run.config, dict) else {}
    payload = config.get("execution_handoff")
    return dict(payload) if isinstance(payload, dict) else {}


def _resolve_origin(run: ExperimentRun) -> tuple[str | None, str | None, str | None]:
    scientific_validation = _scientific_validation_payload(run)
    execution_handoff = _execution_handoff(run)
    autonomous_origin = (
        execution_handoff.get("autonomous_origin")
        if isinstance(execution_handoff.get("autonomous_origin"), dict)
        else {}
    )
    selected_hypothesis_ids = (
        execution_handoff.get("selected_hypothesis_ids")
        if isinstance(execution_handoff.get("selected_hypothesis_ids"), list)
        else []
    )
    source_kind = _text(autonomous_origin.get("source_kind")).lower()
    source_id = _text(autonomous_origin.get("source_id"))
    opportunity_id = _text(autonomous_origin.get("opportunity_id"))

    if source_kind not in {"profile", "portfolio"}:
        source_kind = (
            "profile"
            if _text(scientific_validation.get("domain_research_profile_id"))
            else "portfolio"
            if _text(scientific_validation.get("research_portfolio_id"))
            else ""
        )
    if not source_id:
        source_id = (
            _text(scientific_validation.get("domain_research_profile_id"))
            or _text(scientific_validation.get("research_portfolio_id"))
        )
    if not opportunity_id:
        opportunity_id = (
            _text(scientific_validation.get("hypothesis_id"))
            or next((str(item).strip() for item in selected_hypothesis_ids if str(item).strip()), "")
        )
    if source_kind not in {"profile", "portfolio"} or not source_id or not opportunity_id:
        return None, None, None
    return source_kind, source_id, opportunity_id


def summarize_experiment_run_outcome(run: ExperimentRun) -> str | None:
    scientific_validation = _scientific_validation_payload(run)
    results = run.results if isinstance(run.results, dict) else {}
    for candidate in (
        run.summary,
        results.get("summary"),
        results.get("note"),
        scientific_validation.get("decision_summary"),
    ):
        summary = _normalize_summary_text(candidate)
        if summary:
            return summary
    status = _text(run.status).lower()
    blocked_reason = _text(
        scientific_validation.get("blocked_reason_code")
        or scientific_validation.get("blocked_reason")
        or getattr(run, "blocked_reason_code", None)
    )
    if status == "blocked":
        return _normalize_summary_text(blocked_reason or "Validation blocked.")
    if status in {"completed", "succeeded"}:
        return "Validation completed."
    if status == "failed":
        return "Validation failed."
    if status == "cancelled":
        return "Validation cancelled."
    if status in {"running", "queued", "provisioning", "planned"}:
        return f"Validation {status}."
    return None


def _apply_experiment_outcome_to_row(
    row: dict[str, Any],
    *,
    run: ExperimentRun,
    plan: ExperimentPlan,
    recorded_at: datetime,
) -> dict[str, Any]:
    updated = dict(row)
    scientific_validation = _scientific_validation_payload(run)
    status = _text(run.status).lower()
    recorded_at_iso = recorded_at.isoformat()
    blocked_reason_code = _text(
        scientific_validation.get("blocked_reason_code")
        or scientific_validation.get("blocked_reason")
        or getattr(run, "blocked_reason_code", None)
    ) or None
    run_id = _get_model_id(run.id)
    plan_id = _get_model_id(plan.id)
    note_id = _get_model_id(plan.research_note_id)
    agent_job_id = _get_model_id(run.agent_job_id)

    linked_plan_ids = _clean_list([*(updated.get("linked_experiment_plan_ids") or []), plan_id], limit=8)
    linked_run_ids = _clean_list([*(updated.get("linked_validation_run_ids") or []), run_id], limit=8)
    source_note_ids = _clean_list([*(updated.get("source_note_ids") or []), note_id], limit=8)

    updated["linked_experiment_plan_ids"] = linked_plan_ids
    updated["linked_validation_run_ids"] = linked_run_ids
    updated["source_note_ids"] = source_note_ids
    updated["latest_experiment_plan_id"] = plan_id
    updated["latest_validation_run_id"] = run_id
    updated["latest_validation_job_id"] = agent_job_id
    updated["latest_validation_status"] = status or None
    updated["latest_validation_blocked_reason_code"] = blocked_reason_code
    updated["last_activity_at"] = recorded_at_iso
    updated["updated_at"] = recorded_at_iso
    updated["follow_up_last_job_id"] = agent_job_id

    summary = summarize_experiment_run_outcome(run)
    if status in {"planned", "queued", "provisioning", "running"}:
        updated["follow_up_outcome_status"] = None
        updated["follow_up_outcome_recorded_at"] = None
        updated["follow_up_outcome_summary"] = None
        updated["last_blocked_reason_code"] = None
        updated["autonomy_state"] = "active"
        updated["stage"] = "planned" if status == "planned" else "validating"
    elif status in {"completed", "succeeded"}:
        updated["follow_up_outcome_status"] = "completed"
        updated["follow_up_outcome_recorded_at"] = recorded_at_iso
        updated["follow_up_outcome_summary"] = summary
        updated["last_blocked_reason_code"] = None
        updated["last_decision_type"] = "validation_completed"
        updated["last_decision_reason_code"] = "completed_current_evidence"
        updated["autonomy_state"] = "completed_waiting_change"
        updated["stage"] = "completed"
    elif status == "blocked":
        updated["follow_up_outcome_status"] = "blocked"
        updated["follow_up_outcome_recorded_at"] = recorded_at_iso
        updated["follow_up_outcome_summary"] = summary
        updated["last_decision_type"] = "validation_blocked"
        updated["last_decision_reason_code"] = blocked_reason_code or "validation_blocked"
        updated["last_blocked_reason_code"] = blocked_reason_code
        updated["autonomy_state"] = "blocked_structural"
        updated["stage"] = "blocked"
    elif status == "failed":
        updated["follow_up_outcome_status"] = "failed"
        updated["follow_up_outcome_recorded_at"] = recorded_at_iso
        updated["follow_up_outcome_summary"] = summary
        updated["last_decision_type"] = "validation_failed"
        updated["last_decision_reason_code"] = "validation_failed"
        updated["last_blocked_reason_code"] = None
        updated["autonomy_state"] = "eligible"
        updated["stage"] = "accepted"
    elif status == "cancelled":
        updated["follow_up_outcome_status"] = "cancelled"
        updated["follow_up_outcome_recorded_at"] = recorded_at_iso
        updated["follow_up_outcome_summary"] = summary
        updated["last_decision_type"] = "validation_cancelled"
        updated["last_decision_reason_code"] = "validation_cancelled"
        updated["last_blocked_reason_code"] = None
        updated["autonomy_state"] = "eligible"
        updated["stage"] = "accepted"
    return normalize_research_opportunity(updated)


async def _get_model_by_id(db: AsyncSession, model: type, value: str):
    text = _text(value)
    if not text:
        return None
    try:
        return await db.get(model, UUID(text))
    except Exception:
        return None


async def reconcile_experiment_run_outcome_to_originating_opportunity(
    db: AsyncSession,
    *,
    run: ExperimentRun,
    plan: ExperimentPlan,
    recorded_at: datetime | None = None,
) -> bool:
    source_kind, source_id, opportunity_id = _resolve_origin(run)
    if not source_kind or not source_id or not opportunity_id:
        return False

    at = recorded_at or getattr(run, "completed_at", None) or datetime.utcnow()
    plan_id = _get_model_id(plan.id)
    run_id = _get_model_id(run.id)
    note_id = _get_model_id(plan.research_note_id)

    if source_kind == "profile":
        profile = await _get_model_by_id(db, DomainResearchProfile, source_id)
        if profile is None:
            return False
        summary = dict(profile.latest_summary) if isinstance(profile.latest_summary, dict) else {}
        rows = list_normalized_research_opportunities(summary.get("opportunities") or summary.get("idea_candidates") or [])
        idx = next((i for i, row in enumerate(rows) if _text(row.get("opportunity_id")) == opportunity_id), -1)
        if idx < 0:
            return False
        next_row = _apply_experiment_outcome_to_row(rows[idx], run=run, plan=plan, recorded_at=at)
        if next_row == rows[idx]:
            return False
        rows[idx] = next_row
        summary["opportunities"] = rows
        if isinstance(summary.get("idea_candidates"), list):
            summary["idea_candidates"] = rows
        profile.latest_summary = summary
        profile.latest_experiment_plan_ids = _clean_list([*(profile.latest_experiment_plan_ids or []), plan_id], limit=20)
        profile.latest_validation_run_ids = _clean_list([*(profile.latest_validation_run_ids or []), run_id], limit=20)
        profile.latest_note_ids = _clean_list([*(profile.latest_note_ids or []), note_id], limit=20)
        profile.updated_at = at
        return True

    portfolio = await _get_model_by_id(db, ResearchPortfolio, source_id)
    if portfolio is None:
        return False
    rows = list_normalized_research_opportunities(portfolio.opportunities or [])
    idx = next((i for i, row in enumerate(rows) if _text(row.get("opportunity_id")) == opportunity_id), -1)
    if idx < 0:
        return False
    next_row = _apply_experiment_outcome_to_row(rows[idx], run=run, plan=plan, recorded_at=at)
    if next_row == rows[idx]:
        return False
    rows[idx] = next_row
    portfolio.opportunities = rows
    portfolio.latest_experiment_plan_ids = _clean_list([*(portfolio.latest_experiment_plan_ids or []), plan_id], limit=30)
    portfolio.latest_validation_run_ids = _clean_list([*(portfolio.latest_validation_run_ids or []), run_id], limit=30)
    portfolio.latest_note_ids = _clean_list([*(portfolio.latest_note_ids or []), note_id], limit=30)
    portfolio.updated_at = at
    return True
