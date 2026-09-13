"""
Research monitor profiles endpoints.

These profiles are learned from Research Inbox triage and can be inspected/edited.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from sqlalchemy import and_, desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.agent_job import AgentJob
from app.models.research_inbox import ResearchInboxItem
from app.models.research_monitor_profile import ResearchMonitorProfile
from app.models.user import User
from app.schemas.research_monitor_profile import (
    ResearchMonitorAnalyticsResponse,
    ResearchMonitorBudgetConfigResponse,
    ResearchMonitorBudgetHistoryEntryResponse,
    ResearchMonitorBudgetUpdateRequest,
    ResearchMonitorBudgetUpdateResponse,
    ResearchMonitorCustomerBudgetUpdateRequest,
    ResearchMonitorCustomerBudgetUpdateResponse,
    ResearchMonitorCustomerRebalanceApplyRequest,
    ResearchMonitorCustomerRebalanceApplyResponse,
    ResearchMonitorCustomerRebalanceEvaluationDetailResponse,
    ResearchMonitorCustomerRebalancePreviewRequest,
    ResearchMonitorCustomerRebalancePreviewResponse,
    ResearchMonitorPolicyConfigResponse,
    ResearchMonitorPolicyEvaluationDetailResponse,
    ResearchMonitorPolicyHistoryEntryResponse,
    ResearchMonitorPolicyRollbackRequest,
    ResearchMonitorPolicySimulationRequest,
    ResearchMonitorPolicySimulationResponse,
    ResearchMonitorPolicyUpdateRequest,
    ResearchMonitorPolicyUpdateResponse,
    ResearchMonitorProfileResponse,
    ResearchMonitorProfileUpdateRequest,
)
from app.services.agent_job_scheduler_state import (
    extract_scheduler_state as _extract_scheduler_state,
)
from app.services.auth_service import get_current_user
from app.services.autonomy_event_service import record_autonomy_decision_event
from app.services.autonomy_service import (
    build_monitor_policy_compat_fields,
    build_monitor_policy_history_compat_entry,
)
from app.services.research_monitor_profile_service import (
    research_monitor_profile_service,
)

router = APIRouter()
POLICY_HISTORY_KEY = "follow_up_policy_history"
POLICY_HISTORY_LIMIT = 20
POLICY_EVALUATION_TARGET_COUNT = 8
DEFAULT_AUTONOMY_BUDGET = {
    "auto_launch_limit_24h": 3,
    "approval_queue_limit_24h": 6,
    "alert_limit_24h": 4,
    "queue_backlog_cap": 8,
}


def _normalize_policy_mode(raw: object) -> str:
    mode = str(raw or "").strip().lower()
    if mode in {"manual_only", "auto_launch_safe", "queue_for_approval"}:
        return mode
    return "manual_only"


def _normalize_allowed_recommendations(raw: object) -> list[str]:
    values = raw if isinstance(raw, list) else []
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out or ["deep_dive_chain", "single_research_job"]


def _normalize_policy_config(raw: object) -> dict[str, Any]:
    policy = raw if isinstance(raw, dict) else {}
    return {
        "mode": _normalize_policy_mode(policy.get("mode")),
        "allowed_recommendations": _normalize_allowed_recommendations(
            policy.get("allowed_recommendations")
        ),
    }


def _resolve_monitor_automation_contract(config: object) -> dict[str, Any]:
    return research_monitor_profile_service.resolve_monitor_automation_config(config)


def _resolve_monitor_policy_request(
    *,
    current_config: dict[str, Any],
    current_automation: dict[str, Any],
    automation_profile: object = None,
    automation_policy: object = None,
    mode: object = None,
    allowed_recommendations: object = None,
    reset_to_default: bool = False,
) -> dict[str, Any]:
    if reset_to_default:
        return _resolve_monitor_automation_contract(
            {
                **current_config,
                "automation_profile": "balanced",
                "automation_policy": {
                    "follow_up_review_mode": "manual_only",
                    "allowed_recommendations": [
                        "deep_dive_chain",
                        "single_research_job",
                    ],
                },
            }
        )

    next_automation_profile = (
        str(automation_profile).strip()
        if automation_profile is not None
        else current_automation["automation_profile"]
    )
    next_automation_policy = dict(current_automation["automation_policy"])
    if isinstance(automation_policy, dict):
        next_automation_policy.update(automation_policy)
    if mode is not None:
        next_automation_policy["follow_up_review_mode"] = _normalize_policy_mode(mode)
    if allowed_recommendations is not None:
        next_automation_policy[
            "allowed_recommendations"
        ] = _normalize_allowed_recommendations(allowed_recommendations)
    return _resolve_monitor_automation_contract(
        {
            **current_config,
            "automation_profile": next_automation_profile,
            "automation_policy": next_automation_policy,
        }
    )


def _resolve_monitor_policy_history_snapshot(
    *,
    entry: dict[str, Any],
    phase: str,
    fallback_automation: dict[str, Any],
) -> dict[str, Any]:
    automation_profile = (
        str(
            entry.get(f"{phase}_automation_profile")
            or fallback_automation["automation_profile"]
        ).strip()
        or fallback_automation["automation_profile"]
    )
    automation_policy = (
        entry.get(f"{phase}_automation_policy")
        if isinstance(entry.get(f"{phase}_automation_policy"), dict)
        else None
    )
    compat_policy_raw = entry.get(f"{phase}_follow_up_autonomy")
    compat_policy = (
        _normalize_policy_config(compat_policy_raw)
        if isinstance(compat_policy_raw, dict)
        else None
    )
    if automation_policy is None:
        automation_policy = dict(fallback_automation["automation_policy"])
        if compat_policy is not None:
            automation_policy["follow_up_review_mode"] = compat_policy["mode"]
            automation_policy["allowed_recommendations"] = compat_policy[
                "allowed_recommendations"
            ]
    return _resolve_monitor_automation_contract(
        {
            "automation_profile": automation_profile,
            "automation_policy": automation_policy,
        }
    )


def _normalize_budget_int(raw: object, *, fallback: int) -> int:
    try:
        value = int(raw)
    except Exception:
        value = fallback
    return max(0, min(value, 10000))


def _normalize_budget_config(raw: object) -> dict[str, int]:
    budget = raw if isinstance(raw, dict) else {}
    return {
        key: _normalize_budget_int(budget.get(key), fallback=value)
        for key, value in DEFAULT_AUTONOMY_BUDGET.items()
    }


def _normalize_customer_budget_config(raw: object) -> dict[str, int]:
    budget = raw if isinstance(raw, dict) else {}
    return {
        key: _normalize_budget_int(budget.get(key), fallback=0)
        for key in DEFAULT_AUTONOMY_BUDGET.keys()
    }


def _normalize_budget_history_entry(raw: object) -> Optional[dict[str, Any]]:
    return research_monitor_profile_service._normalize_budget_history_entry(raw)


def _get_budget_history(job: AgentJob) -> list[dict[str, Any]]:
    return research_monitor_profile_service._budget_history_for_job(job)


def _sanitize_policy_analytics_context(raw: object) -> dict[str, Any]:
    allowed_keys = {
        "health_bucket",
        "policy_confidence",
        "accepted_count",
        "blocked_count",
        "follow_up_completed_count",
        "follow_up_failed_count",
        "follow_up_cancelled_count",
    }
    context = raw if isinstance(raw, dict) else {}
    sanitized: dict[str, Any] = {}
    for key in allowed_keys:
        if key not in context:
            continue
        value = context.get(key)
        if value is None:
            continue
        if key.endswith("_count"):
            try:
                sanitized[key] = int(value)
            except Exception:
                continue
        else:
            text = str(value).strip()
            if text:
                sanitized[key] = text
    return sanitized


def _normalize_policy_history_entry(raw: object) -> Optional[dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    at_value = raw.get("at")
    parsed_at: Optional[datetime] = None
    if isinstance(at_value, datetime):
        parsed_at = at_value
    elif isinstance(at_value, str):
        try:
            parsed_at = datetime.fromisoformat(at_value.replace("Z", "+00:00"))
        except Exception:
            parsed_at = None
    if parsed_at is None:
        return None
    previous_snapshot = _resolve_monitor_policy_history_snapshot(
        entry=raw,
        phase="previous",
        fallback_automation={
            "automation_profile": str(
                raw.get("previous_automation_profile") or "balanced"
            ).strip()
            or "balanced",
            "automation_policy": raw.get("previous_automation_policy")
            if isinstance(raw.get("previous_automation_policy"), dict)
            else {},
        },
    )
    next_snapshot = _resolve_monitor_policy_history_snapshot(
        entry=raw,
        phase="next",
        fallback_automation={
            "automation_profile": str(
                raw.get("next_automation_profile") or "balanced"
            ).strip()
            or "balanced",
            "automation_policy": raw.get("next_automation_policy")
            if isinstance(raw.get("next_automation_policy"), dict)
            else {},
        },
    )
    return {
        "id": str(raw.get("id") or "").strip() or str(uuid4()),
        "at": parsed_at,
        "actor_user_id": (str(raw.get("actor_user_id") or "").strip() or None),
        "change_source": (
            str(raw.get("change_source") or "").strip() or "manual_override"
        ),
        "change_reason": (str(raw.get("change_reason") or "").strip() or None),
        **build_monitor_policy_history_compat_entry(
            previous_snapshot=previous_snapshot,
            next_snapshot=next_snapshot,
        ),
        "previous_automation_profile": previous_snapshot["automation_profile"],
        "next_automation_profile": next_snapshot["automation_profile"],
        "previous_automation_policy": previous_snapshot["automation_policy"],
        "next_automation_policy": next_snapshot["automation_policy"],
        "previous_effective_policy": previous_snapshot["effective_policy"],
        "next_effective_policy": next_snapshot["effective_policy"],
        "effective_clamp_state": (
            str(raw.get("effective_clamp_state") or "").strip() or None
        ),
        "effective_clamp_reasons": [
            str(reason).strip()
            for reason in (raw.get("effective_clamp_reasons") or [])
            if str(reason).strip()
        ],
        "analytics_context": _sanitize_policy_analytics_context(
            raw.get("analytics_context")
        ),
        "evaluation_target_count": max(
            3, int(raw.get("evaluation_target_count") or POLICY_EVALUATION_TARGET_COUNT)
        ),
        "evaluation_state": (
            str(raw.get("evaluation_state") or "").strip().lower() or "active"
        ),
    }


def _get_policy_history(job: AgentJob) -> list[dict[str, Any]]:
    results = dict(job.results or {}) if isinstance(job.results, dict) else {}
    raw_entries = (
        results.get(POLICY_HISTORY_KEY)
        if isinstance(results.get(POLICY_HISTORY_KEY), list)
        else []
    )
    entries: list[dict[str, Any]] = []
    for raw in raw_entries:
        entry = _normalize_policy_history_entry(raw)
        if entry is not None:
            entries.append(entry)
    entries.sort(key=lambda row: row["at"], reverse=True)
    return entries[:POLICY_HISTORY_LIMIT]


def _set_policy_history(
    job: AgentJob, entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    def _history_sort_key(row: dict[str, Any]) -> datetime:
        value = row.get("at")
        if isinstance(value, datetime):
            return (
                value
                if value.tzinfo is not None
                else value.replace(tzinfo=timezone.utc)
            )
        return datetime.min.replace(tzinfo=timezone.utc)

    normalized = sorted(entries, key=_history_sort_key, reverse=True)[
        :POLICY_HISTORY_LIMIT
    ]
    results = dict(job.results or {}) if isinstance(job.results, dict) else {}
    results[POLICY_HISTORY_KEY] = [
        {
            **entry,
            "at": entry["at"].isoformat(),
        }
        for entry in normalized
    ]
    job.results = results
    return normalized


def _append_policy_history_entry(
    job: AgentJob,
    *,
    previous_automation_profile: str,
    next_automation_profile: str,
    previous_automation_policy: dict[str, Any],
    next_automation_policy: dict[str, Any],
    previous_effective_policy: dict[str, Any],
    next_effective_policy: dict[str, Any],
    previous_policy: dict[str, Any],
    next_policy: dict[str, Any],
    actor_user_id: object,
    change_source: Optional[str],
    change_reason: Optional[str],
    analytics_context: object,
    effective_clamp_state: Optional[str] = None,
    effective_clamp_reasons: Optional[list[str]] = None,
) -> dict[str, Any]:
    history = _get_policy_history(job)
    entry = {
        "id": str(uuid4()),
        "at": datetime.now(timezone.utc),
        "actor_user_id": (str(actor_user_id or "").strip() or None),
        "change_source": (str(change_source or "").strip() or "manual_override"),
        "change_reason": (str(change_reason or "").strip() or None),
        **build_monitor_policy_history_compat_entry(
            previous_snapshot={
                "follow_up_autonomy": _normalize_policy_config(previous_policy),
            },
            next_snapshot={
                "follow_up_autonomy": _normalize_policy_config(next_policy),
            },
        ),
        "previous_automation_profile": previous_automation_profile,
        "next_automation_profile": next_automation_profile,
        "previous_automation_policy": dict(previous_automation_policy),
        "next_automation_policy": dict(next_automation_policy),
        "previous_effective_policy": dict(previous_effective_policy),
        "next_effective_policy": dict(next_effective_policy),
        "effective_clamp_state": (str(effective_clamp_state or "").strip() or None),
        "effective_clamp_reasons": [
            str(reason).strip()
            for reason in (effective_clamp_reasons or [])
            if str(reason).strip()
        ],
        "analytics_context": _sanitize_policy_analytics_context(analytics_context),
        "evaluation_target_count": POLICY_EVALUATION_TARGET_COUNT,
        "evaluation_state": "active",
    }
    history.insert(0, entry)
    _set_policy_history(job, history)
    return entry


def _latest_policy_history_entry(job: AgentJob) -> Optional[dict[str, Any]]:
    history = _get_policy_history(job)
    return history[0] if history else None


@router.get("", response_model=list[ResearchMonitorProfileResponse])
async def list_monitor_profiles(
    customer: Optional[str] = Query(None, description="Optional customer tag"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(ResearchMonitorProfile).where(
        ResearchMonitorProfile.user_id == current_user.id
    )
    if customer:
        stmt = stmt.where(ResearchMonitorProfile.customer == customer)
    stmt = stmt.order_by(desc(ResearchMonitorProfile.updated_at))
    res = await db.execute(stmt)
    profiles = list(res.scalars().all())
    return [ResearchMonitorProfileResponse.model_validate(p) for p in profiles]


@router.get("/analytics", response_model=ResearchMonitorAnalyticsResponse)
async def get_monitor_analytics(
    customer: Optional[str] = Query(None, description="Optional customer tag"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    snapshot = await research_monitor_profile_service.build_effectiveness_analytics(
        db=db,
        user_id=current_user.id,
        customer=(customer or "").strip() or None,
    )
    return ResearchMonitorAnalyticsResponse.model_validate(snapshot)


@router.post(
    "/{monitor_job_id}/policy/simulate",
    response_model=ResearchMonitorPolicySimulationResponse,
)
async def simulate_monitor_policy(
    monitor_job_id: str,
    payload: ResearchMonitorPolicySimulationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        from uuid import UUID

        job_uuid = UUID(monitor_job_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid monitor job id")

    monitor_job = await db.get(AgentJob, job_uuid)
    if not monitor_job or monitor_job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Monitor job not found")
    if str(monitor_job.job_type or "").strip().lower() != "monitor":
        raise HTTPException(
            status_code=400, detail="Only monitor jobs support policy simulation"
        )

    current_config = (
        dict(monitor_job.config or {})
        if isinstance(getattr(monitor_job, "config", None), dict)
        else {}
    )
    current_automation = _resolve_monitor_automation_contract(current_config)
    proposed_policy = _resolve_monitor_policy_request(
        current_config=current_config,
        current_automation=current_automation,
        automation_profile=payload.automation_profile,
        automation_policy=payload.automation_policy,
        mode=payload.mode,
        allowed_recommendations=payload.allowed_recommendations,
    )
    (
        monitor_job,
        snapshot,
    ) = await research_monitor_profile_service.build_policy_simulation(
        db=db,
        user_id=current_user.id,
        monitor_job_id=job_uuid,
        proposed_policy=proposed_policy,
        history_limit=payload.history_limit,
    )
    return ResearchMonitorPolicySimulationResponse.model_validate(snapshot)


@router.post(
    "/{monitor_job_id}/policy", response_model=ResearchMonitorPolicyUpdateResponse
)
async def update_monitor_policy(
    monitor_job_id: str,
    payload: ResearchMonitorPolicyUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        from uuid import UUID

        job_uuid = UUID(monitor_job_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid monitor job id")

    job = await db.get(AgentJob, job_uuid)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Monitor job not found")
    if str(job.job_type or "").strip().lower() != "monitor":
        raise HTTPException(
            status_code=400, detail="Only monitor jobs support follow-up policy tuning"
        )
    source_scheduler_state = _extract_scheduler_state(job)

    config = dict(job.config or {}) if isinstance(job.config, dict) else {}
    existing_automation = _resolve_monitor_automation_contract(config)
    resolved_next = _resolve_monitor_policy_request(
        current_config=config,
        current_automation=existing_automation,
        automation_profile=payload.automation_profile,
        automation_policy=payload.automation_policy,
        mode=payload.mode,
        allowed_recommendations=payload.allowed_recommendations,
        reset_to_default=payload.reset_to_default,
    )
    compat_fields = build_monitor_policy_compat_fields(
        automation_profile=resolved_next["automation_profile"],
        automation_policy=resolved_next["automation_policy"],
        effective_policy=resolved_next["effective_policy"],
        default_allowed=["deep_dive_chain", "single_research_job"],
    )
    next_policy = compat_fields["follow_up_autonomy"]
    config["automation_profile"] = resolved_next["automation_profile"]
    config["automation_policy"] = resolved_next["automation_policy"]
    config["follow_up_autonomy"] = next_policy
    job.config = config
    latest_history_entry = None
    if (
        resolved_next["automation_profile"] != existing_automation["automation_profile"]
        or resolved_next["automation_policy"]
        != existing_automation["automation_policy"]
        or resolved_next["effective_policy"] != existing_automation["effective_policy"]
    ):
        latest_history_entry = _append_policy_history_entry(
            job,
            previous_automation_profile=existing_automation["automation_profile"],
            next_automation_profile=resolved_next["automation_profile"],
            previous_automation_policy=existing_automation["automation_policy"],
            next_automation_policy=resolved_next["automation_policy"],
            previous_effective_policy=existing_automation["effective_policy"],
            next_effective_policy=resolved_next["effective_policy"],
            previous_policy=existing_automation["follow_up_autonomy"],
            next_policy=next_policy,
            actor_user_id=current_user.id,
            change_source=payload.change_source
            or ("reset_to_default" if payload.reset_to_default else "manual_override"),
            change_reason=payload.change_reason,
            analytics_context=payload.analytics_context,
        )
        await record_autonomy_decision_event(
            db,
            user_id=current_user.id,
            event_type="policy_updated",
            event_time=datetime.utcnow(),
            source_kind="monitor",
            source_id=str(job.id),
            source_label=str(job.name or "Research monitor").strip(),
            customer=None,
            decision_type="policy_updated",
            reason_code=payload.change_source
            or ("reset_to_default" if payload.reset_to_default else "manual_override"),
            status="active",
            severity="medium",
            actor_mode="operator",
            summary=f"{str(job.name or 'Research monitor').strip()}: monitor policy updated",
            operator_note=payload.change_reason,
            reason_label=str(
                payload.change_source
                or (
                    "reset_to_default"
                    if payload.reset_to_default
                    else "manual_override"
                )
            )
            .replace("_", " ")
            .strip()
            .capitalize(),
            scheduler_state=source_scheduler_state,
            before_state={"effective_policy": existing_automation["effective_policy"]},
            after_state={"effective_policy": resolved_next["effective_policy"]},
            deep_link={
                "target_tab": "health",
                "params": {"tab": "health"},
                "label": "Open Autonomy Health",
            },
            metadata={
                "history_entry_id": latest_history_entry.get("id")
                if isinstance(latest_history_entry, dict)
                else None
            },
        )
    await db.commit()

    return ResearchMonitorPolicyUpdateResponse(
        monitor_job_id=job.id,
        follow_up_autonomy=ResearchMonitorPolicyConfigResponse.model_validate(
            next_policy
        ),
        automation_profile=resolved_next["automation_profile"],
        automation_policy=resolved_next["automation_policy"],
        effective_policy=resolved_next["effective_policy"],
        latest_history_entry=(
            ResearchMonitorPolicyHistoryEntryResponse.model_validate(
                latest_history_entry
            )
            if latest_history_entry is not None
            else (
                ResearchMonitorPolicyHistoryEntryResponse.model_validate(
                    _latest_policy_history_entry(job)
                )
                if _latest_policy_history_entry(job) is not None
                else None
            )
        ),
        policy_history_count=len(_get_policy_history(job)),
    )


@router.post(
    "/{monitor_job_id}/policy/rollback",
    response_model=ResearchMonitorPolicyUpdateResponse,
)
async def rollback_monitor_policy(
    monitor_job_id: str,
    payload: ResearchMonitorPolicyRollbackRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        from uuid import UUID

        job_uuid = UUID(monitor_job_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid monitor job id")

    job = await db.get(AgentJob, job_uuid)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Monitor job not found")
    if str(job.job_type or "").strip().lower() != "monitor":
        raise HTTPException(
            status_code=400,
            detail="Only monitor jobs support follow-up policy rollback",
        )
    source_scheduler_state = _extract_scheduler_state(job)

    history = _get_policy_history(job)
    target = next(
        (entry for entry in history if entry["id"] == payload.history_entry_id), None
    )
    if target is None:
        raise HTTPException(status_code=404, detail="Policy history entry not found")
    results = dict(job.results or {}) if isinstance(job.results, dict) else {}
    raw_history = (
        results.get(POLICY_HISTORY_KEY)
        if isinstance(results.get(POLICY_HISTORY_KEY), list)
        else []
    )
    raw_target = next(
        (
            entry
            for entry in raw_history
            if isinstance(entry, dict)
            and str(entry.get("id") or "").strip() == payload.history_entry_id
        ),
        {},
    )

    config = dict(job.config or {}) if isinstance(job.config, dict) else {}
    existing_automation = _resolve_monitor_automation_contract(config)
    resolved_restored = _resolve_monitor_policy_history_snapshot(
        entry=raw_target if isinstance(raw_target, dict) and raw_target else target,
        phase="previous",
        fallback_automation=existing_automation,
    )
    compat_fields = build_monitor_policy_compat_fields(
        automation_profile=resolved_restored["automation_profile"],
        automation_policy=resolved_restored["automation_policy"],
        effective_policy=resolved_restored["effective_policy"],
        default_allowed=["deep_dive_chain", "single_research_job"],
    )
    if (
        resolved_restored["automation_profile"]
        == existing_automation["automation_profile"]
        and resolved_restored["automation_policy"]
        == existing_automation["automation_policy"]
        and resolved_restored["effective_policy"]
        == existing_automation["effective_policy"]
    ):
        raise HTTPException(
            status_code=400,
            detail="Selected history entry already matches the current policy",
        )

    config["automation_profile"] = resolved_restored["automation_profile"]
    config["automation_policy"] = resolved_restored["automation_policy"]
    config["follow_up_autonomy"] = compat_fields["follow_up_autonomy"]
    job.config = config
    latest_history_entry = _append_policy_history_entry(
        job,
        previous_automation_profile=existing_automation["automation_profile"],
        next_automation_profile=resolved_restored["automation_profile"],
        previous_automation_policy=existing_automation["automation_policy"],
        next_automation_policy=resolved_restored["automation_policy"],
        previous_effective_policy=existing_automation["effective_policy"],
        next_effective_policy=resolved_restored["effective_policy"],
        previous_policy=existing_automation["follow_up_autonomy"],
        next_policy=compat_fields["follow_up_autonomy"],
        actor_user_id=current_user.id,
        change_source="rollback",
        change_reason=payload.change_reason
        or f"Rollback of history entry {payload.history_entry_id}",
        analytics_context={},
    )
    await record_autonomy_decision_event(
        db,
        user_id=current_user.id,
        event_type="policy_rollback",
        event_time=datetime.utcnow(),
        source_kind="monitor",
        source_id=str(job.id),
        source_label=str(job.name or "Research monitor").strip(),
        decision_type="policy_rollback",
        reason_code="rollback",
        status="active",
        severity="medium",
        actor_mode="operator",
        summary=f"{str(job.name or 'Research monitor').strip()}: monitor policy rolled back",
        operator_note=payload.change_reason,
        reason_label="Policy rollback",
        scheduler_state=source_scheduler_state,
        before_state={"effective_policy": existing_automation["effective_policy"]},
        after_state={"effective_policy": resolved_restored["effective_policy"]},
        deep_link={
            "target_tab": "health",
            "params": {"tab": "health"},
            "label": "Open Autonomy Health",
        },
        metadata={
            "history_entry_id": latest_history_entry.get("id")
            if isinstance(latest_history_entry, dict)
            else None
        },
    )
    await db.commit()

    return ResearchMonitorPolicyUpdateResponse(
        monitor_job_id=job.id,
        follow_up_autonomy=ResearchMonitorPolicyConfigResponse.model_validate(
            compat_fields["follow_up_autonomy"]
        ),
        automation_profile=resolved_restored["automation_profile"],
        automation_policy=resolved_restored["automation_policy"],
        effective_policy=resolved_restored["effective_policy"],
        latest_history_entry=ResearchMonitorPolicyHistoryEntryResponse.model_validate(
            latest_history_entry
        ),
        policy_history_count=len(_get_policy_history(job)),
    )


@router.post(
    "/{monitor_job_id}/budget", response_model=ResearchMonitorBudgetUpdateResponse
)
async def update_monitor_budget(
    monitor_job_id: str,
    payload: ResearchMonitorBudgetUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        from uuid import UUID

        job_uuid = UUID(monitor_job_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid monitor job id")

    job = await db.get(AgentJob, job_uuid)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Monitor job not found")
    if str(job.job_type or "").strip().lower() != "monitor":
        raise HTTPException(
            status_code=400, detail="Only monitor jobs support autonomy budgets"
        )
    source_scheduler_state = _extract_scheduler_state(job)

    config = dict(job.config or {}) if isinstance(job.config, dict) else {}
    existing_budget = _normalize_budget_config(config.get("autonomy_budget"))
    next_budget = dict(existing_budget)
    if payload.reset_to_default:
        next_budget = dict(DEFAULT_AUTONOMY_BUDGET)
    else:
        for key in DEFAULT_AUTONOMY_BUDGET.keys():
            value = getattr(payload, key, None)
            if value is not None:
                next_budget[key] = _normalize_budget_int(
                    value, fallback=DEFAULT_AUTONOMY_BUDGET[key]
                )

    config["autonomy_budget"] = next_budget
    job.config = config
    latest_history_entry = None
    if next_budget != existing_budget:
        latest_history_entry = (
            research_monitor_profile_service.append_budget_history_entry(
                job=job,
                previous_budget=existing_budget,
                next_budget=next_budget,
                actor_user_id=current_user.id,
                change_source=payload.change_source
                or (
                    "reset_to_default"
                    if payload.reset_to_default
                    else "manual_override"
                ),
                change_reason=payload.change_reason,
            )
        )
        await record_autonomy_decision_event(
            db,
            user_id=current_user.id,
            event_type="budget_clamped",
            event_time=datetime.utcnow(),
            source_kind="monitor",
            source_id=str(job.id),
            source_label=str(job.name or "Research monitor").strip(),
            decision_type="budget_clamped",
            reason_code=payload.change_source
            or ("reset_to_default" if payload.reset_to_default else "manual_override"),
            status="budget_updated",
            severity="medium",
            actor_mode="operator",
            summary=f"{str(job.name or 'Research monitor').strip()}: monitor budget updated",
            operator_note=payload.change_reason,
            reason_label=str(
                payload.change_source
                or (
                    "reset_to_default"
                    if payload.reset_to_default
                    else "manual_override"
                )
            )
            .replace("_", " ")
            .strip()
            .capitalize(),
            scheduler_state=source_scheduler_state,
            before_state={"autonomy_budget": existing_budget},
            after_state={"autonomy_budget": next_budget},
            deep_link={
                "target_tab": "health",
                "params": {"tab": "health"},
                "label": "Open Autonomy Health",
            },
            metadata={
                "history_entry_id": latest_history_entry.get("id")
                if isinstance(latest_history_entry, dict)
                else None
            },
        )
    await db.commit()

    return ResearchMonitorBudgetUpdateResponse(
        monitor_job_id=job.id,
        autonomy_budget=ResearchMonitorBudgetConfigResponse.model_validate(next_budget),
        latest_history_entry=(
            ResearchMonitorBudgetHistoryEntryResponse.model_validate(
                latest_history_entry
            )
            if latest_history_entry is not None
            else (
                ResearchMonitorBudgetHistoryEntryResponse.model_validate(
                    _get_budget_history(job)[0]
                )
                if _get_budget_history(job)
                else None
            )
        ),
    )


@router.post(
    "/customer-budget", response_model=ResearchMonitorCustomerBudgetUpdateResponse
)
async def update_customer_budget(
    payload: ResearchMonitorCustomerBudgetUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    customer = str(payload.customer or "").strip()
    if not customer:
        raise HTTPException(status_code=400, detail="Customer is required")

    stmt = (
        select(ResearchMonitorProfile)
        .where(
            and_(
                ResearchMonitorProfile.user_id == current_user.id,
                ResearchMonitorProfile.customer == customer,
            )
        )
        .limit(1)
    )
    profile = (await db.execute(stmt)).scalar_one_or_none()
    if profile is None:
        profile = ResearchMonitorProfile(
            user_id=current_user.id,
            customer=customer,
            token_scores={},
            phrase_scores={},
            recommendation_scores={},
            source_type_scores={},
            outcome_counters={},
            customer_budget_config=_normalize_customer_budget_config(None),
            customer_rebalance_history=[],
            muted_tokens=[],
            muted_patterns=[],
            notes=None,
        )
        db.add(profile)
        await db.flush()

    current_budget = _normalize_customer_budget_config(
        getattr(profile, "customer_budget_config", None)
    )
    if payload.reset_to_default:
        next_budget = _normalize_customer_budget_config(None)
    else:
        next_budget = {
            "auto_launch_limit_24h": current_budget["auto_launch_limit_24h"]
            if payload.auto_launch_limit_24h is None
            else int(payload.auto_launch_limit_24h),
            "approval_queue_limit_24h": current_budget["approval_queue_limit_24h"]
            if payload.approval_queue_limit_24h is None
            else int(payload.approval_queue_limit_24h),
            "alert_limit_24h": current_budget["alert_limit_24h"]
            if payload.alert_limit_24h is None
            else int(payload.alert_limit_24h),
            "queue_backlog_cap": current_budget["queue_backlog_cap"]
            if payload.queue_backlog_cap is None
            else int(payload.queue_backlog_cap),
        }
        next_budget = _normalize_customer_budget_config(next_budget)

    profile.customer_budget_config = next_budget
    await db.commit()
    await db.refresh(profile)
    return ResearchMonitorCustomerBudgetUpdateResponse(
        customer=customer,
        customer_budget=ResearchMonitorBudgetConfigResponse.model_validate(next_budget),
    )


@router.post(
    "/customer-rebalance/preview",
    response_model=ResearchMonitorCustomerRebalancePreviewResponse,
)
async def preview_customer_rebalance(
    payload: ResearchMonitorCustomerRebalancePreviewRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    customer = str(payload.customer or "").strip()
    if not customer:
        raise HTTPException(status_code=400, detail="Customer is required")

    updates: list[dict[str, Any]] = []
    for row in payload.monitor_budget_updates:
        updates.append(
            {
                "monitor_job_id": row.monitor_job_id,
                "auto_launch_limit_24h": row.auto_launch_limit_24h,
                "approval_queue_limit_24h": row.approval_queue_limit_24h,
                "alert_limit_24h": row.alert_limit_24h,
                "queue_backlog_cap": row.queue_backlog_cap,
            }
        )
    preview = await research_monitor_profile_service.build_customer_rebalance_preview(
        db=db,
        user_id=current_user.id,
        customer=customer,
        monitor_budget_updates=updates,
    )
    return ResearchMonitorCustomerRebalancePreviewResponse.model_validate(preview)


@router.post(
    "/customer-rebalance/apply",
    response_model=ResearchMonitorCustomerRebalanceApplyResponse,
)
async def apply_customer_rebalance(
    payload: ResearchMonitorCustomerRebalanceApplyRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    customer = str(payload.customer or "").strip()
    if not customer:
        raise HTTPException(status_code=400, detail="Customer is required")

    stmt = (
        select(ResearchMonitorProfile)
        .where(
            ResearchMonitorProfile.user_id == current_user.id,
            ResearchMonitorProfile.customer == customer,
        )
        .limit(1)
    )
    profile = (await db.execute(stmt)).scalar_one_or_none()
    if profile is None:
        profile = ResearchMonitorProfile(
            user_id=current_user.id,
            customer=customer,
            token_scores={},
            phrase_scores={},
            recommendation_scores={},
            source_type_scores={},
            outcome_counters={},
            customer_budget_config=_normalize_customer_budget_config(None),
            customer_rebalance_history=[],
            muted_tokens=[],
            muted_patterns=[],
            notes=None,
        )
        db.add(profile)
        await db.flush()

    before_preview = (
        await research_monitor_profile_service.build_customer_rebalance_preview(
            db=db,
            user_id=current_user.id,
            customer=customer,
            monitor_budget_updates=[
                {
                    "monitor_job_id": row.monitor_job_id,
                    "auto_launch_limit_24h": row.auto_launch_limit_24h,
                    "approval_queue_limit_24h": row.approval_queue_limit_24h,
                    "alert_limit_24h": row.alert_limit_24h,
                    "queue_backlog_cap": row.queue_backlog_cap,
                }
                for row in payload.monitor_budget_updates
            ],
        )
    )

    latest_history_entries: list[dict[str, Any]] = []
    updated_monitor_ids: list[Any] = []
    for row in payload.monitor_budget_updates:
        job = await db.get(AgentJob, row.monitor_job_id)
        if not job or job.user_id != current_user.id:
            raise HTTPException(
                status_code=404, detail=f"Monitor job not found: {row.monitor_job_id}"
            )
        if str(job.job_type or "").strip().lower() != "monitor":
            raise HTTPException(
                status_code=400, detail="Only monitor jobs support autonomy budgets"
            )

        config = dict(job.config or {}) if isinstance(job.config, dict) else {}
        previous_budget = _normalize_budget_config(config.get("autonomy_budget"))
        next_budget = _normalize_budget_config(
            {
                "auto_launch_limit_24h": row.auto_launch_limit_24h,
                "approval_queue_limit_24h": row.approval_queue_limit_24h,
                "alert_limit_24h": row.alert_limit_24h,
                "queue_backlog_cap": row.queue_backlog_cap,
            }
        )
        if next_budget == previous_budget:
            continue
        config["autonomy_budget"] = next_budget
        job.config = config
        entry = research_monitor_profile_service.append_budget_history_entry(
            job=job,
            previous_budget=previous_budget,
            next_budget=next_budget,
            actor_user_id=current_user.id,
            change_source="customer_rebalance_guidance",
            change_reason=payload.change_reason
            or f"Customer rebalance applied for {customer}",
            guidance_context={"customer": customer},
        )
        latest_history_entries.append(entry)
        updated_monitor_ids.append(job.id)

    if latest_history_entries:
        research_monitor_profile_service.append_customer_rebalance_history_entry(
            profile=profile,
            actor_user_id=current_user.id,
            change_source="customer_rebalance_guidance",
            change_reason=payload.change_reason
            or f"Customer rebalance applied for {customer}",
            changes=list(before_preview.get("changes") or []),
            before_capacity=dict(before_preview.get("before_capacity") or {}),
            after_capacity=dict(before_preview.get("after_capacity") or {}),
        )
        await record_autonomy_decision_event(
            db,
            user_id=current_user.id,
            event_type="customer_rebalanced",
            event_time=datetime.utcnow(),
            source_kind="monitor",
            source_id=customer,
            source_label=customer,
            customer=customer,
            decision_type="customer_rebalanced",
            reason_code="customer_rebalance_guidance",
            reason_label="Customer rebalance guidance",
            scheduler_state=None,
            status="applied",
            severity="medium",
            actor_mode="operator",
            summary=f"{customer}: customer rebalance applied",
            operator_note=payload.change_reason,
            before_state={"before_capacity": before_preview.get("before_capacity")},
            after_state={"after_capacity": before_preview.get("after_capacity")},
            deep_link={
                "target_tab": "health",
                "params": {"tab": "health", "health_customer": customer},
                "label": "Open Autonomy Health",
            },
            metadata={"updated_monitor_ids": [str(v) for v in updated_monitor_ids]},
        )

    await db.commit()
    preview = await research_monitor_profile_service.build_customer_rebalance_preview(
        db=db,
        user_id=current_user.id,
        customer=customer,
    )
    return ResearchMonitorCustomerRebalanceApplyResponse(
        customer=customer,
        updated_monitor_ids=updated_monitor_ids,
        guidance_status=str(preview.get("guidance_status") or "none"),
        guidance_summary=str(preview.get("guidance_summary") or "") or None,
        latest_history_entries=[
            ResearchMonitorBudgetHistoryEntryResponse.model_validate(entry)
            for entry in latest_history_entries
        ],
    )


@router.get(
    "/customer-rebalance/{customer}/history/{history_entry_id}/evaluation",
    response_model=ResearchMonitorCustomerRebalanceEvaluationDetailResponse,
)
async def get_customer_rebalance_evaluation(
    customer: str,
    history_entry_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    customer_name = str(customer or "").strip()
    if not customer_name:
        raise HTTPException(status_code=400, detail="Customer is required")

    stmt = (
        select(ResearchMonitorProfile)
        .where(
            ResearchMonitorProfile.user_id == current_user.id,
            ResearchMonitorProfile.customer == customer_name,
        )
        .limit(1)
    )
    profile = (await db.execute(stmt)).scalar_one_or_none()
    if profile is None:
        raise HTTPException(
            status_code=404, detail="Customer rebalance history not found"
        )

    history = research_monitor_profile_service._customer_rebalance_history_for_profile(
        profile
    )
    target = next(
        (entry for entry in history if str(entry.get("id") or "") == history_entry_id),
        None,
    )
    if target is None:
        raise HTTPException(
            status_code=404, detail="Customer rebalance history entry not found"
        )

    inbox_stmt = (
        select(ResearchInboxItem)
        .where(
            ResearchInboxItem.user_id == current_user.id,
            ResearchInboxItem.customer == customer_name,
            ResearchInboxItem.status == "accepted",
        )
        .order_by(ResearchInboxItem.updated_at.desc())
    )
    items = list((await db.execute(inbox_stmt)).scalars().all())
    job_ids = [item.job_id for item in items if item.job_id is not None]
    jobs_by_id: dict[Any, AgentJob] = {}
    if job_ids:
        jobs_stmt = select(AgentJob).where(AgentJob.id.in_(job_ids))
        jobs_by_id = {
            job.id: job for job in (await db.execute(jobs_stmt)).scalars().all()
        }
    snapshot = await research_monitor_profile_service.build_effectiveness_analytics(
        db=db,
        user_id=current_user.id,
        customer=customer_name,
    )
    monitor_rows = [
        row
        for row in snapshot.get("monitors", [])
        if str(row.get("customer") or "").strip() == customer_name
    ]
    detail = (
        research_monitor_profile_service.build_customer_rebalance_evaluation_detail(
            customer=customer_name,
            history_entry=target,
            items=items,
            monitor_rows=monitor_rows,
            jobs_by_id=jobs_by_id,
        )
    )
    return ResearchMonitorCustomerRebalanceEvaluationDetailResponse.model_validate(
        detail
    )


@router.get(
    "/{monitor_job_id}/policy-history/{history_entry_id}/evaluation",
    response_model=ResearchMonitorPolicyEvaluationDetailResponse,
)
async def get_monitor_policy_evaluation(
    monitor_job_id: str,
    history_entry_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        from uuid import UUID

        job_uuid = UUID(monitor_job_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid monitor job id")

    job = await db.get(AgentJob, job_uuid)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Monitor job not found")
    if str(job.job_type or "").strip().lower() != "monitor":
        raise HTTPException(
            status_code=400, detail="Only monitor jobs support policy evaluation"
        )

    history = _get_policy_history(job)
    target = next((entry for entry in history if entry["id"] == history_entry_id), None)
    if target is None:
        raise HTTPException(status_code=404, detail="Policy history entry not found")

    stmt = (
        select(ResearchInboxItem)
        .where(
            ResearchInboxItem.user_id == current_user.id,
            ResearchInboxItem.job_id == job.id,
            ResearchInboxItem.status == "accepted",
        )
        .order_by(ResearchInboxItem.updated_at.desc())
    )
    res = await db.execute(stmt)
    items = list(res.scalars().all())
    evaluation = research_monitor_profile_service.build_policy_evaluation_detail(
        monitor_job_id=job.id,
        history_entry=target,
        items=items,
    )
    return ResearchMonitorPolicyEvaluationDetailResponse.model_validate(evaluation)


@router.patch("/{profile_id}", response_model=ResearchMonitorProfileResponse)
async def update_monitor_profile(
    profile_id: str,
    payload: ResearchMonitorProfileUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        from uuid import UUID

        pid = UUID(profile_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid profile id")

    profile = await db.get(ResearchMonitorProfile, pid)
    if not profile or profile.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Profile not found")

    if payload.muted_tokens is not None:
        profile.muted_tokens = [
            str(x).strip().lower()
            for x in (payload.muted_tokens or [])
            if str(x).strip()
        ]
    if payload.muted_patterns is not None:
        profile.muted_patterns = [
            str(x).strip() for x in (payload.muted_patterns or []) if str(x).strip()
        ]
    if payload.notes is not None:
        profile.notes = (payload.notes or "").strip() or None

    await db.commit()
    await db.refresh(profile)
    return ResearchMonitorProfileResponse.model_validate(profile)


class ResearchMonitorProfileUpsertRequest(ResearchMonitorProfileUpdateRequest):
    customer: Optional[str] = None
    merge_lists: bool = True


@router.post("/upsert", response_model=ResearchMonitorProfileResponse)
async def upsert_monitor_profile(
    payload: ResearchMonitorProfileUpsertRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create or update a monitor profile for the current user (+ optional customer).

    This enables users to mute tokens/patterns even before they have enough triage history.
    """
    customer = (payload.customer or "").strip() or None
    try:
        stmt = select(ResearchMonitorProfile).where(
            ResearchMonitorProfile.user_id == current_user.id
        )
        if customer:
            stmt = stmt.where(ResearchMonitorProfile.customer == customer)
        else:
            stmt = stmt.where(ResearchMonitorProfile.customer.is_(None))
        res = await db.execute(stmt.limit(1))
        profile = res.scalar_one_or_none()

        if not profile:
            profile = ResearchMonitorProfile(
                user_id=current_user.id,
                customer=customer,
                token_scores={},
                customer_budget_config=_normalize_customer_budget_config(None),
                customer_rebalance_history=[],
                muted_tokens=[],
                muted_patterns=[],
                notes=None,
            )
            db.add(profile)
            await db.commit()
            await db.refresh(profile)

        if payload.muted_tokens is not None:
            incoming = [
                str(x).strip().lower()
                for x in (payload.muted_tokens or [])
                if str(x).strip()
            ]
            if payload.merge_lists:
                existing = (
                    profile.muted_tokens
                    if isinstance(profile.muted_tokens, list)
                    else []
                )
                merged = list(
                    dict.fromkeys(
                        [
                            *(
                                str(x).strip().lower()
                                for x in existing
                                if str(x).strip()
                            ),
                            *incoming,
                        ]
                    )
                )
                profile.muted_tokens = merged
            else:
                profile.muted_tokens = incoming
        if payload.muted_patterns is not None:
            incoming = [
                str(x).strip() for x in (payload.muted_patterns or []) if str(x).strip()
            ]
            if payload.merge_lists:
                existing = (
                    profile.muted_patterns
                    if isinstance(profile.muted_patterns, list)
                    else []
                )
                merged = list(
                    dict.fromkeys(
                        [
                            *(str(x).strip() for x in existing if str(x).strip()),
                            *incoming,
                        ]
                    )
                )
                profile.muted_patterns = merged
            else:
                profile.muted_patterns = incoming
        if payload.notes is not None:
            profile.notes = (payload.notes or "").strip() or None

        await db.commit()
        await db.refresh(profile)
        return ResearchMonitorProfileResponse.model_validate(profile)
    except Exception as exc:
        logger.error(f"Failed to upsert monitor profile: {exc}")
        raise HTTPException(status_code=500, detail="Failed to upsert profile")
