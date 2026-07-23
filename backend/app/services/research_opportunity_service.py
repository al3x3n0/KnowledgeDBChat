from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob
from app.models.experiment import ExperimentPlan, ExperimentRun


RESEARCH_OPPORTUNITY_STAGE_VALUES = {
    "discovered",
    "accepted",
    "suppressed",
    "planned",
    "validating",
    "completed",
    "blocked",
}

RESEARCH_OPPORTUNITY_DECISION_VALUES = {
    "pending_review",
    "accepted",
    "suppressed",
    "auto_accepted",
}

RESEARCH_OPPORTUNITY_DECISION_SOURCE_VALUES = {
    "system",
    "operator",
}

RESEARCH_OPPORTUNITY_AUTONOMY_STATES = {
    "eligible",
    "cooldown",
    "blocked_structural",
    "completed_waiting_change",
    "active",
}

PORTFOLIO_OPERATOR_REVIEW_TYPE_VALUES = {
    "follow_up_recommendation",
    "policy_review",
    "budget_review",
}

PORTFOLIO_BUDGET_REASON_CODES = {
    "budget_limit_exceeded",
    "budget_limit_rejected",
    "budget_exhausted",
    "portfolio_budget_exhausted",
    "validation_budget_exceeded",
}

PORTFOLIO_POLICY_REVIEW_REASON_CODES = {
    "sandbox_rejected",
    "sandbox_missing",
    "sandbox_mismatch",
    "sandbox_policy_rejected",
    "policy_rejected",
    "disallowed_image",
    "missing_required_runtime",
    "missing_required_source_material",
    "missing_required_capability",
    "capability_mismatch",
    "runtime_capability_mismatch",
}


def _text(value: Any, *, default: str = "") -> str:
    return str(value or "").strip() or default


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


def normalize_research_opportunity_key(value: Any) -> str:
    text = _text(value).lower()
    out = []
    last_sep = False
    for char in text:
        if char.isalnum():
            out.append(char)
            last_sep = False
        elif not last_sep:
            out.append("_")
            last_sep = True
    return "".join(out).strip("_")


def _clean_string_list(value: Any, *, limit: int = 12) -> List[str]:
    rows = value if isinstance(value, list) else []
    out: List[str] = []
    for row in rows:
        text = _text(row)
        if not text or text in out:
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _clean_dict_list(value: Any, *, limit: int = 12) -> List[Dict[str, Any]]:
    rows = value if isinstance(value, list) else []
    out: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(dict(row))
        if len(out) >= limit:
            break
    return out


def _merge_string_lists(first: Any, second: Any, *, limit: int) -> List[str]:
    return list(
        dict.fromkeys(
            [
                *_clean_string_list(first, limit=max(limit * 2, limit)),
                *_clean_string_list(second, limit=max(limit * 2, limit)),
            ]
        )
    )[:limit]


def _merge_dict_lists(first: Any, second: Any, *, limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in [
        *_clean_dict_list(first, limit=max(limit * 2, limit)),
        *_clean_dict_list(second, limit=max(limit * 2, limit)),
    ]:
        marker = "|".join(
            f"{str(key)}={repr(value)}"
            for key, value in sorted(row.items(), key=lambda item: str(item[0]))
        )
        if marker in seen:
            continue
        seen.add(marker)
        out.append(dict(row))
        if len(out) >= limit:
            break
    return out


def compute_research_opportunity_evidence_revision(row: Dict[str, Any]) -> str:
    payload = {
        "canonical_key": _text(row.get("canonical_key") or row.get("title") or row.get("hypothesis")).lower(),
        "source_note_ids": sorted(_clean_string_list(row.get("source_note_ids"), limit=16)),
        "supporting_evidence": sorted(_clean_string_list(row.get("supporting_evidence"), limit=16)),
        "supporting_sources": sorted(
            [
                json.dumps(entry, sort_keys=True, separators=(",", ":"))
                for entry in _clean_dict_list(row.get("supporting_sources"), limit=16)
            ]
        ),
        "confidence": round(max(0.0, min(_safe_float(row.get("confidence"), 0.0), 1.0)), 4),
        "readiness": round(max(0.0, min(_safe_float(row.get("readiness"), 0.0), 1.0)), 4),
        "novelty": round(max(0.0, min(_safe_float(row.get("novelty"), 0.0), 1.0)), 4),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def compute_research_portfolio_config_revision(
    automation_profile: Any,
    effective_policy: Any,
    sandbox_profile_id: Any,
) -> str:
    payload = {
        "automation_profile": _text(automation_profile, default="balanced").lower(),
        "effective_policy": effective_policy if isinstance(effective_policy, dict) else {},
        "sandbox_profile_id": _text(sandbox_profile_id) or None,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def _infer_autonomy_state(opportunity: Dict[str, Any]) -> str:
    explicit = _text(opportunity.get("autonomy_state")).lower()
    if explicit in RESEARCH_OPPORTUNITY_AUTONOMY_STATES:
        return explicit
    stage = _text(opportunity.get("stage")).lower()
    if stage == "completed":
        return "completed_waiting_change"
    if stage == "blocked" and _text(opportunity.get("last_blocked_reason_code")):
        return "blocked_structural"
    if stage in {"planned", "validating"} or _clean_string_list(opportunity.get("linked_validation_run_ids"), limit=8):
        return "active"
    return "eligible"


def derive_research_opportunity_stage(
    opportunity: Dict[str, Any],
    *,
    validation_status_by_id: Optional[Dict[str, str]] = None,
) -> str:
    validation_status_by_id = validation_status_by_id or {}
    explicit_stage = _text(opportunity.get("stage")).lower()
    decision_state = _text(opportunity.get("decision_state"), default="pending_review")
    linked_validation_run_ids = _clean_string_list(opportunity.get("linked_validation_run_ids"), limit=16)
    linked_experiment_plan_ids = _clean_string_list(opportunity.get("linked_experiment_plan_ids"), limit=16)
    child_job_ids = _clean_string_list(opportunity.get("child_job_ids"), limit=16)

    if explicit_stage == "suppressed" or decision_state == "suppressed":
        return "suppressed"
    if explicit_stage == "blocked" and _text(opportunity.get("last_blocked_reason_code")):
        return "blocked"
    if explicit_stage == "completed":
        return "completed"
    for run_id in linked_validation_run_ids:
        status = _text(validation_status_by_id.get(run_id)).lower()
        if status in {"queued", "provisioning", "running"}:
            return "validating"
    for run_id in linked_validation_run_ids:
        status = _text(validation_status_by_id.get(run_id)).lower()
        if status in {"completed", "succeeded"}:
            return "completed"
        if status in {"blocked", "failed", "cancelled"}:
            return "blocked"
    if linked_validation_run_ids:
        return "validating"
    if child_job_ids:
        return "validating"
    if linked_experiment_plan_ids:
        return "planned"
    if decision_state in {"accepted", "auto_accepted"}:
        return "accepted"
    return "discovered"


def normalize_research_opportunity(
    row: Dict[str, Any],
    *,
    default_stage: str = "discovered",
    default_decision_state: str = "pending_review",
    default_decision_source: str = "system",
    validation_status_by_id: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    title = _text(row.get("title") or row.get("idea_title") or row.get("hypothesis"))
    canonical_key = _text(row.get("canonical_key")) or normalize_research_opportunity_key(title)
    opportunity_id = _text(row.get("opportunity_id")) or (f"opp_{canonical_key[:48]}" if canonical_key else "")
    decision_state = _text(row.get("decision_state"), default=default_decision_state).lower()
    if decision_state not in RESEARCH_OPPORTUNITY_DECISION_VALUES:
        decision_state = default_decision_state
    decision_source = _text(row.get("decision_source"), default=default_decision_source).lower()
    if decision_source not in RESEARCH_OPPORTUNITY_DECISION_SOURCE_VALUES:
        decision_source = default_decision_source

    stage = _text(row.get("stage"), default=default_stage).lower()
    if stage not in RESEARCH_OPPORTUNITY_STAGE_VALUES:
        stage = default_stage

    normalized = {
        "opportunity_id": opportunity_id,
        "canonical_key": canonical_key,
        "title": title,
        "hypothesis": _text(row.get("hypothesis") or row.get("claim") or title),
        "stage": stage,
        "decision_state": decision_state,
        "decision_source": decision_source,
        "operator_note": _text(row.get("operator_note")) or None,
        "supporting_evidence": _clean_string_list(row.get("supporting_evidence"), limit=8),
        "supporting_sources": _clean_dict_list(row.get("supporting_sources"), limit=8),
        "next_steps": _clean_string_list(row.get("next_steps"), limit=6),
        "source_profile_ids": _clean_string_list(row.get("source_profile_ids"), limit=8),
        "source_job_ids": _clean_string_list(row.get("source_job_ids"), limit=8),
        "source_note_ids": _clean_string_list(row.get("source_note_ids"), limit=8),
        "linked_experiment_plan_ids": _clean_string_list(row.get("linked_experiment_plan_ids"), limit=8),
        "linked_validation_run_ids": _clean_string_list(row.get("linked_validation_run_ids"), limit=8),
        "latest_experiment_plan_id": _text(row.get("latest_experiment_plan_id")) or None,
        "latest_validation_run_id": _text(row.get("latest_validation_run_id")) or None,
        "latest_validation_job_id": _text(row.get("latest_validation_job_id")) or None,
        "latest_validation_status": _text(row.get("latest_validation_status")) or None,
        "latest_validation_blocked_reason_code": _text(row.get("latest_validation_blocked_reason_code")) or None,
        "child_job_ids": _clean_string_list(row.get("child_job_ids"), limit=8),
        "source_repo_ids": _clean_string_list(row.get("source_repo_ids"), limit=8),
        "confidence": round(max(0.0, min(_safe_float(row.get("confidence"), 0.0), 1.0)), 4),
        "novelty": round(max(0.0, min(_safe_float(row.get("novelty") or row.get("novelty_score"), 0.0), 1.0)), 4),
        "readiness": round(max(0.0, min(_safe_float(row.get("readiness") or row.get("overall_score"), 0.0), 1.0)), 4),
        "track_type": _text(row.get("track_type"), default="generic"),
        "autonomy_state": None,
        "last_evaluated_at": _text(row.get("last_evaluated_at")) or None,
        "next_eligible_at": _text(row.get("next_eligible_at")) or None,
        "evidence_revision": _text(row.get("evidence_revision")) or None,
        "last_material_change_at": _text(row.get("last_material_change_at")) or None,
        "last_decision_type": _text(row.get("last_decision_type")) or None,
        "last_decision_reason_code": _text(row.get("last_decision_reason_code")) or None,
        "portfolio_config_revision": _text(row.get("portfolio_config_revision")) or None,
        "last_skip_reason_code": _text(row.get("last_skip_reason_code")) or None,
        "last_blocked_reason_code": _text(row.get("last_blocked_reason_code")) or None,
        "follow_up_review_status": _text(row.get("follow_up_review_status")) or None,
        "follow_up_reviewed_at": _text(row.get("follow_up_reviewed_at")) or None,
        "follow_up_reviewed_by_user_id": _text(row.get("follow_up_reviewed_by_user_id")) or None,
        "follow_up_review_note": _text(row.get("follow_up_review_note")) or None,
        "follow_up_review_evidence_revision": _text(row.get("follow_up_review_evidence_revision")) or None,
        "last_reevaluation_review_outcome": _text(row.get("last_reevaluation_review_outcome")) or None,
        "last_reevaluation_reviewed_at": _text(row.get("last_reevaluation_reviewed_at")) or None,
        "last_reevaluation_review_job_id": _text(row.get("last_reevaluation_review_job_id")) or None,
        "last_reevaluation_review_note": _text(row.get("last_reevaluation_review_note")) or None,
        "last_reevaluation_review_source_note_id": _text(row.get("last_reevaluation_review_source_note_id")) or None,
        "last_reevaluation_review_target_note_id": _text(row.get("last_reevaluation_review_target_note_id")) or None,
        "follow_up_outcome_status": _text(row.get("follow_up_outcome_status")) or None,
        "follow_up_outcome_recorded_at": _text(row.get("follow_up_outcome_recorded_at")) or None,
        "follow_up_outcome_summary": _text(row.get("follow_up_outcome_summary")) or None,
        "follow_up_last_job_id": _text(row.get("follow_up_last_job_id")) or None,
        "follow_up_launched_at": _text(row.get("follow_up_launched_at")) or None,
        "last_activity_at": _text(row.get("last_activity_at")) or None,
        "reprioritized_at": _text(row.get("reprioritized_at")) or None,
        "reprioritization_reason": _text(row.get("reprioritization_reason")) or None,
        "reprioritization_source_run_ids": _clean_string_list(row.get("reprioritization_source_run_ids"), limit=8),
        "prior_confidence": round(max(0.0, min(_safe_float(row.get("prior_confidence"), 0.0), 1.0)), 4) if row.get("prior_confidence") is not None else None,
        "prior_readiness": round(max(0.0, min(_safe_float(row.get("prior_readiness"), 0.0), 1.0)), 4) if row.get("prior_readiness") is not None else None,
        "autonomous_origin": dict(row.get("autonomous_origin")) if isinstance(row.get("autonomous_origin"), dict) else None,
        "updated_at": _text(row.get("updated_at")) or datetime.utcnow().isoformat(),
    }
    normalized["stage"] = derive_research_opportunity_stage(
        normalized,
        validation_status_by_id=validation_status_by_id,
    )
    normalized["evidence_revision"] = normalized["evidence_revision"] or compute_research_opportunity_evidence_revision(normalized)
    normalized["autonomy_state"] = _infer_autonomy_state({**normalized, **row})
    return normalized


def build_validation_status_map(
    latest_validation_runs: Optional[Iterable[Any]] = None,
    summary_validation_runs: Optional[Iterable[Any]] = None,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for collection in (latest_validation_runs or [], summary_validation_runs or []):
        for row in collection if isinstance(collection, list) else []:
            if not isinstance(row, dict):
                continue
            run_id = _text(row.get("id") or row.get("run_id"))
            status = _text(row.get("status")).lower()
            if run_id and status:
                out[run_id] = status
    return out


def list_normalized_research_opportunities(
    rows: Any,
    *,
    validation_status_by_id: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    items = rows if isinstance(rows, list) else []
    normalized: List[Dict[str, Any]] = []
    seen_ids = set()
    for row in items:
        if not isinstance(row, dict):
            continue
        item = normalize_research_opportunity(row, validation_status_by_id=validation_status_by_id)
        if not item["opportunity_id"] or item["opportunity_id"] in seen_ids:
            continue
        seen_ids.add(item["opportunity_id"])
        normalized.append(item)
    return normalized


def merge_operator_fields(
    current: Dict[str, Any],
    previous: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    merged = dict(current)
    if not isinstance(previous, dict):
        return normalize_research_opportunity(merged)
    for key in (
        "decision_state",
        "decision_source",
        "operator_note",
        "updated_at",
        "autonomy_state",
        "last_evaluated_at",
        "next_eligible_at",
        "evidence_revision",
        "last_material_change_at",
        "last_decision_type",
        "last_decision_reason_code",
        "portfolio_config_revision",
        "last_skip_reason_code",
        "last_blocked_reason_code",
        "follow_up_review_status",
        "follow_up_reviewed_at",
        "follow_up_reviewed_by_user_id",
        "follow_up_review_note",
        "follow_up_review_evidence_revision",
        "last_reevaluation_review_outcome",
        "last_reevaluation_reviewed_at",
        "last_reevaluation_review_job_id",
        "last_reevaluation_review_note",
        "last_reevaluation_review_source_note_id",
        "last_reevaluation_review_target_note_id",
        "follow_up_outcome_status",
        "follow_up_outcome_recorded_at",
        "follow_up_outcome_summary",
        "follow_up_last_job_id",
        "follow_up_launched_at",
        "latest_experiment_plan_id",
        "latest_validation_run_id",
        "latest_validation_job_id",
        "latest_validation_status",
        "latest_validation_blocked_reason_code",
        "last_activity_at",
        "reprioritized_at",
        "reprioritization_reason",
        "prior_confidence",
        "prior_readiness",
        "autonomous_origin",
    ):
        if previous.get(key) is not None:
            merged[key] = previous.get(key)
    for key, limit in (
        ("linked_experiment_plan_ids", 8),
        ("linked_validation_run_ids", 8),
        ("child_job_ids", 8),
        ("source_note_ids", 8),
        ("source_profile_ids", 8),
        ("source_job_ids", 8),
        ("source_repo_ids", 8),
        ("supporting_evidence", 8),
        ("next_steps", 6),
        ("reprioritization_source_run_ids", 8),
    ):
        merged[key] = _merge_string_lists(previous.get(key), merged.get(key), limit=limit)
    merged["supporting_sources"] = _merge_dict_lists(
        previous.get("supporting_sources"),
        merged.get("supporting_sources"),
        limit=8,
    )
    return normalize_research_opportunity(merged)


def summarize_research_opportunity_stages(rows: Any) -> Dict[str, int]:
    opportunities = list_normalized_research_opportunities(rows)
    return {
        "discovered": sum(1 for row in opportunities if str(row.get("stage") or "") == "discovered"),
        "accepted": sum(1 for row in opportunities if str(row.get("stage") or "") == "accepted"),
        "suppressed": sum(1 for row in opportunities if str(row.get("stage") or "") == "suppressed"),
        "planned": sum(1 for row in opportunities if str(row.get("stage") or "") == "planned"),
        "validating": sum(1 for row in opportunities if str(row.get("stage") or "") == "validating"),
        "completed": sum(1 for row in opportunities if str(row.get("stage") or "") == "completed"),
        "blocked": sum(1 for row in opportunities if str(row.get("stage") or "") == "blocked"),
    }


def summarize_research_opportunity_autonomy_states(rows: Any) -> Dict[str, int]:
    opportunities = list_normalized_research_opportunities(rows)
    return {
        "eligible": sum(1 for row in opportunities if str(row.get("autonomy_state") or "") == "eligible"),
        "cooldown": sum(1 for row in opportunities if str(row.get("autonomy_state") or "") == "cooldown"),
        "blocked_structural": sum(1 for row in opportunities if str(row.get("autonomy_state") or "") == "blocked_structural"),
        "completed_waiting_change": sum(1 for row in opportunities if str(row.get("autonomy_state") or "") == "completed_waiting_change"),
        "active": sum(1 for row in opportunities if str(row.get("autonomy_state") or "") == "active"),
    }


def collect_research_opportunity_linked_ids(rows: Any) -> Dict[str, List[str]]:
    opportunities = list_normalized_research_opportunities(rows)
    plan_ids: List[str] = []
    run_ids: List[str] = []
    child_job_ids: List[str] = []
    note_ids: List[str] = []
    for row in opportunities:
        for key, target in (
            ("linked_experiment_plan_ids", plan_ids),
            ("linked_validation_run_ids", run_ids),
            ("child_job_ids", child_job_ids),
            ("source_note_ids", note_ids),
        ):
            for value in row.get(key) or []:
                text = _text(value)
                if text and text not in target:
                    target.append(text)
    return {
        "plan_ids": plan_ids[:50],
        "run_ids": run_ids[:50],
        "child_job_ids": child_job_ids[:50],
        "note_ids": note_ids[:50],
    }


def apply_materialized_experiment_metadata(
    opportunity: Dict[str, Any],
    *,
    owner_kind: str,
    owner_id: str,
    plan_ids: list[str],
    run_id: Optional[str],
    job_id: Optional[str],
    validation_status: Optional[str],
    blocked_reason_code: Optional[str],
    materialized_at: Optional[str] = None,
) -> Dict[str, Any]:
    updated = dict(opportunity)
    now_iso = materialized_at or datetime.utcnow().isoformat()
    normalized_plan_ids = _clean_string_list(plan_ids, limit=8)
    normalized_run_ids = _clean_string_list(
        [*(updated.get("linked_validation_run_ids") or []), run_id] if run_id else updated.get("linked_validation_run_ids"),
        limit=8,
    )
    updated["linked_experiment_plan_ids"] = normalized_plan_ids
    updated["linked_validation_run_ids"] = normalized_run_ids
    updated["latest_experiment_plan_id"] = normalized_plan_ids[-1] if normalized_plan_ids else None
    updated["latest_validation_run_id"] = run_id or (normalized_run_ids[-1] if normalized_run_ids else None)
    updated["latest_validation_job_id"] = _text(job_id) or None
    updated["latest_validation_status"] = _text(validation_status) or None
    updated["latest_validation_blocked_reason_code"] = _text(blocked_reason_code) or None
    updated["decision_state"] = "accepted"
    updated["decision_source"] = "operator"
    updated["updated_at"] = now_iso
    updated["last_activity_at"] = now_iso
    updated["last_decision_type"] = "materialize_experiment"
    updated["last_decision_reason_code"] = (
        "validation_blocked"
        if _text(validation_status).lower() == "blocked"
        else ("validation_reused" if run_id and not job_id and _text(validation_status) else "validation_queued")
    )
    updated["last_blocked_reason_code"] = _text(blocked_reason_code) or None
    updated["autonomous_origin"] = {
        "source_kind": "profile" if owner_kind == "profile" else "portfolio",
        "source_id": _text(owner_id) or None,
        "opportunity_id": _text(updated.get("opportunity_id")) or None,
        "evidence_revision_at_launch": _text(updated.get("evidence_revision")) or None,
    }
    status = _text(validation_status).lower()
    if status in {"queued", "provisioning", "running"}:
        updated["stage"] = "validating"
    elif status in {"blocked", "failed", "cancelled"}:
        updated["stage"] = "blocked"
    elif updated["latest_experiment_plan_id"]:
        updated["stage"] = "planned"
    else:
        updated["stage"] = "accepted"
    return normalize_research_opportunity(updated)


async def materialize_research_opportunity_experiment(
    *,
    db: AsyncSession,
    parent_job: AgentJob,
    owner_kind: str,
    owner_id: str,
    user_id: str,
    opportunity: Dict[str, Any],
    title: str,
    hypothesis: str,
    note_ids: list[str],
    track_type: str,
    objective: str,
    validation_policy: Dict[str, Any],
    sandbox_profile_id: Optional[str],
    repo_source_ids: list[str],
    benchmark_queries: list[str],
    ensure_plan_ids: Callable[[list[str]], Awaitable[list[str]]],
    profile_id: Optional[str] = None,
    portfolio_id: Optional[str] = None,
    originating_job_id: Optional[str] = None,
    start_immediately: bool = True,
) -> Dict[str, Any]:
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    existing_plan_ids = _clean_string_list(opportunity.get("linked_experiment_plan_ids"), limit=8)
    existing_run_ids = _clean_string_list(opportunity.get("linked_validation_run_ids"), limit=8)

    # Idempotent requeue: when the opportunity is already linked to a validation
    # run, reuse it instead of creating a duplicate. This must run before we
    # force a plan to exist, so a requeue never fabricates a fresh plan.
    if existing_run_ids:
        run_id = _text(existing_run_ids[-1])
        target_note_id = next((item for item in note_ids if _text(item)), "")
        run = None
        try:
            run = await db.get(ExperimentRun, UUID(run_id))
        except Exception:
            run = None
        if run is not None:
            config = deepcopy(run.config) if isinstance(run.config, dict) else {}
            post_run_actions = config.get("post_run_actions") if isinstance(config.get("post_run_actions"), dict) else {}
            post_run_actions["auto_append_to_note"] = True
            if not target_note_id:
                target_note_id = str(getattr(run, "research_note_id", "") or "")
            if target_note_id:
                post_run_actions["target_note_id"] = target_note_id
            config["post_run_actions"] = post_run_actions
            run.config = config
            scientific_validation = config.get("scientific_validation") if isinstance(config.get("scientific_validation"), dict) else {}
            return {
                "plan_ids": existing_plan_ids,
                "run_id": str(run.id),
                "job_id": str(run.agent_job_id) if run.agent_job_id else None,
                "validation_status": _text(run.status) or _text(scientific_validation.get("status")) or "planned",
                "blocked_reason_code": _text(
                    scientific_validation.get("blocked_reason_code")
                    or scientific_validation.get("blocked_reason")
                ) or None,
                "reused_run": True,
                "reused_plan": bool(existing_plan_ids),
            }
        # The linked validation run row cannot be loaded (missing, or a non-UUID
        # identifier). Still treat this as an idempotent requeue of that run.
        return {
            "plan_ids": existing_plan_ids,
            "run_id": run_id,
            "job_id": _text(opportunity.get("latest_validation_job_id")) or None,
            "validation_status": _text(opportunity.get("latest_validation_status")) or "planned",
            "blocked_reason_code": _text(opportunity.get("latest_validation_blocked_reason_code")) or None,
            "reused_run": True,
            "reused_plan": bool(existing_plan_ids),
        }

    plan_ids = await ensure_plan_ids(existing_plan_ids)
    if not plan_ids:
        raise HTTPException(status_code=400, detail="Could not resolve an experiment plan for this opportunity")
    plan_id = _text(plan_ids[0] if plan_ids else "")
    try:
        plan_uuid = UUID(plan_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Linked experiment plan is unavailable") from exc
    experiment_plan = await db.get(ExperimentPlan, plan_uuid)
    if experiment_plan is None:
        raise HTTPException(status_code=400, detail="Linked experiment plan is unavailable")

    target_note_id = next((item for item in note_ids if _text(item)), "") or str(experiment_plan.research_note_id or "")

    executor = AutonomousAgentExecutor()
    decision = await executor._create_scientific_validation_run(
        db=db,
        parent_job=parent_job,
        experiment_plan=experiment_plan,
        track_type=_text(track_type, default="generic"),
        objective=_text(objective) or title,
        hypothesis_title=_text(title) or "Research opportunity",
        hypothesis_text=_text(hypothesis) or _text(title) or "Research opportunity",
        validation_policy=validation_policy if isinstance(validation_policy, dict) else {},
        sandbox_profile_id=_text(sandbox_profile_id) or None,
        repo_source_ids=[item for item in _clean_string_list(repo_source_ids, limit=8) if item],
        benchmark_queries=[item for item in _clean_string_list(benchmark_queries, limit=8) if item],
        supporting_evidence=_clean_string_list(opportunity.get("supporting_evidence"), limit=8),
        supporting_sources=_clean_dict_list(opportunity.get("supporting_sources"), limit=8),
        profile_id=_text(profile_id) or None,
        portfolio_id=_text(portfolio_id) or None,
        hypothesis_id=_text(opportunity.get("opportunity_id")) or None,
        originating_job_id=_text(originating_job_id) or str(parent_job.id),
    )
    run_id = _text(decision.get("run_id")) or None
    job_id = _text(decision.get("job_id")) or None
    if run_id:
        try:
            run = await db.get(ExperimentRun, UUID(run_id))
        except Exception:
            run = None
        if run is not None:
            config = deepcopy(run.config) if isinstance(run.config, dict) else {}
            post_run_actions = config.get("post_run_actions") if isinstance(config.get("post_run_actions"), dict) else {}
            post_run_actions["auto_append_to_note"] = True
            if target_note_id:
                post_run_actions["target_note_id"] = target_note_id
            config["post_run_actions"] = post_run_actions
            run.config = config
    if start_immediately and job_id and _text(decision.get("status")).lower() == "queued":
        # Deferred import: agent_job_tasks imports the executor, which imports
        # this module — a top-level import here is a circular import.
        from app.tasks.agent_job_tasks import execute_agent_job_task

        execute_agent_job_task.delay(job_id, str(user_id))
    return {
        "plan_ids": plan_ids,
        "run_id": run_id,
        "job_id": job_id,
        "validation_status": _text(decision.get("status")) or None,
        "blocked_reason_code": _text(decision.get("reason_code")) or None,
        "reused_run": False,
        "reused_plan": bool(existing_plan_ids),
    }


def get_opportunity_lookup(rows: Any) -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get("opportunity_id") or ""): item
        for item in list_normalized_research_opportunities(rows)
        if str(item.get("opportunity_id") or "").strip()
    }


def classify_portfolio_operator_review(
    row: Dict[str, Any],
    *,
    effective_policy: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    opportunity = normalize_research_opportunity(row)
    reason_code = _text(opportunity.get("last_blocked_reason_code") or opportunity.get("last_decision_reason_code") or opportunity.get("last_skip_reason_code")).lower()
    evidence_revision = _text(opportunity.get("evidence_revision")) or compute_research_opportunity_evidence_revision(opportunity)
    policy = effective_policy if isinstance(effective_policy, dict) else {}
    review_mode = _text(policy.get("follow_up_review_mode"), default="auto_launch_safe").lower()
    if review_mode not in {"auto_launch_safe", "queue_for_approval", "manual_only"}:
        review_mode = "auto_launch_safe"

    if (
        review_mode == "queue_for_approval"
        and bool(policy.get("auto_launch_follow_up", True))
        and str(opportunity.get("autonomy_state") or "") == "eligible"
        and not _clean_string_list(opportunity.get("child_job_ids"), limit=1)
        and _text(opportunity.get("follow_up_review_status")).lower() != "approved_launch"
        and not (
            _text(opportunity.get("follow_up_review_status")).lower() == "rejected"
            and _text(opportunity.get("follow_up_review_evidence_revision")) == evidence_revision
        )
    ):
        return {
            "review_type": "follow_up_recommendation",
            "reason_code": "follow_up_launch_approval",
            "reason_label": "Follow-up launch approval",
            "evidence_revision": evidence_revision,
        }

    if str(opportunity.get("autonomy_state") or "") != "blocked_structural" or not reason_code:
        return None

    if reason_code in PORTFOLIO_BUDGET_REASON_CODES:
        return {
            "review_type": "budget_review",
            "reason_code": reason_code,
            "reason_label": "Autonomy budget review",
            "evidence_revision": evidence_revision,
        }
    if reason_code in PORTFOLIO_POLICY_REVIEW_REASON_CODES:
        return {
            "review_type": "policy_review",
            "reason_code": reason_code,
            "reason_label": "Policy review",
            "evidence_revision": evidence_revision,
        }
    return None


def summarize_portfolio_operator_reviews(
    rows: Any,
    *,
    effective_policy: Optional[Dict[str, Any]] = None,
    limit: int = 6,
) -> Dict[str, Any]:
    opportunities = list_normalized_research_opportunities(rows)
    review_rows: List[Dict[str, Any]] = []
    counts = {review_type: 0 for review_type in PORTFOLIO_OPERATOR_REVIEW_TYPE_VALUES}
    for row in opportunities:
        review = classify_portfolio_operator_review(row, effective_policy=effective_policy)
        if not review:
            continue
        review_type = str(review.get("review_type") or "").strip()
        if review_type in counts:
            counts[review_type] += 1
        review_rows.append(
            {
                "review_type": review_type,
                "reason_code": review.get("reason_code"),
                "reason_label": review.get("reason_label"),
                "opportunity_id": row.get("opportunity_id"),
                "canonical_key": row.get("canonical_key"),
                "title": row.get("title"),
                "evidence_revision": review.get("evidence_revision"),
                "autonomy_state": row.get("autonomy_state"),
            }
        )
    non_zero_counts = {key: value for key, value in counts.items() if value}
    return {
        "queued_operator_reviews_count": sum(non_zero_counts.values()),
        "queued_operator_reviews_by_type": non_zero_counts,
        "queued_operator_reviews": review_rows[: max(1, limit)],
    }
