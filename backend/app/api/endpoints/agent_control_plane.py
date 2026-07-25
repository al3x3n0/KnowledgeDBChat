from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Iterable, Optional
from urllib.parse import urlencode
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.endpoints.agent_jobs import (
    _build_decision_trace_event,
    _build_checkpoint_queue_items,
    _build_memory_graph_response,
    _job_matches_bulk_queue_item_type,
    _perform_job_action,
    _perform_follow_up_queue_action,
    checkpoint_queue_bulk_action,
    checkpoint_queue_bulk_follow_up_action,
)
from app.api.endpoints.domain_research_profiles import act_on_domain_research_opportunity
from app.api.endpoints.research_portfolios import act_on_research_portfolio_opportunity
from app.api.endpoints.research_monitor_profiles import rollback_monitor_policy, update_monitor_policy
from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.agent_control_plane_view import AgentControlPlaneView
from app.models.agent_job import AgentJob
from app.models.autonomy_decision_event import AutonomyDecisionEvent
from app.models.domain_research_profile import DomainResearchProfile
from app.models.research_portfolio import ResearchPortfolio
from app.models.user import User
from app.models.workflow import WorkflowExecution
from app.schemas.agent_control_plane import (
    AgentControlRunDetail,
    AgentControlRunBulkReviewActionRequest,
    AgentControlRunBulkReviewActionResponse,
    AgentControlRunBulkReviewActionResultResponse,
    AgentControlRunEdge,
    AgentControlRunLinkResponse,
    AgentControlRunListResponse,
    AgentControlRunReviewListResponse,
    AgentControlRunReviewActionRequest,
    AgentControlRunReviewActionResponse,
    AgentControlRunNode,
    AgentControlRunReplaySummary,
    AgentControlRunReviewItemResponse,
    AgentControlRunRoutingSummary,
    AgentControlRunSummary,
    AgentControlRunViewCreate,
    AgentControlRunViewListResponse,
    AgentControlRunViewResponse,
    AgentControlRunViewUpdate,
)
from app.schemas.agent_job import AgentCheckpointQueueBulkActionRequest, AgentCheckpointQueueBulkFollowUpActionRequest, AgentJobActionRequest
from app.schemas.domain_research_profile import ResearchOpportunityActionRequest
from app.schemas.research_monitor_profile import ResearchMonitorPolicyRollbackRequest, ResearchMonitorPolicyUpdateRequest
from app.schemas.research_portfolio import ResearchPortfolioOpportunityActionRequest

router = APIRouter()
_CONTROL_RUN_VIEW_ALLOWED_FILTERS = {
    "source_type",
    "outcome",
    "routing_tier",
    "selected_run_id",
    "has_operator_review",
    "review_type",
    "review_status",
    "queue_status",
    "queue_customer",
    "queue_sla",
    "queue_escalation",
    "queue_health_drilldown",
    "queue_preset",
    "queue_scope",
    "queue_sort",
}


def _normalize_string(value: Any) -> Optional[str]:
    token = str(value or "").strip()
    return token or None


def _normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            token = _normalize_string(item)
            if token:
                out.append(token)
        return out
    token = _normalize_string(value)
    return [token] if token else []


def _normalize_float(value: Any) -> Optional[float]:
    if value is None:
        return None


def _normalize_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    token = str(value or "").strip().lower()
    if token in {"true", "1", "yes"}:
        return True
    if token in {"false", "0", "no"}:
        return False
    return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_control_run_view_filters(filters: Optional[dict[str, Any]]) -> dict[str, str]:
    payload = filters if isinstance(filters, dict) else {}
    normalized: dict[str, str] = {}
    for raw_key, raw_value in payload.items():
        key = str(raw_key or "").strip().lower()
        if key not in _CONTROL_RUN_VIEW_ALLOWED_FILTERS:
            continue
        value = _normalize_string(raw_value)
        if value:
            normalized[key] = value
    return normalized


def _normalize_review_sort(value: Any) -> str:
    token = str(value or "").strip().lower()
    if token in {"created_at_desc", "age_desc"}:
        return token
    return "priority"


def _review_priority_sort_key(review: AgentControlRunReviewItemResponse) -> tuple[Any, ...]:
    sla_rank = {"overdue": 3, "at_risk": 2, "normal": 1}
    escalation_rank = {"critical": 4, "high": 3, "medium": 2, "normal": 1, "low": 0}
    return (
        sla_rank.get(str(review.sla_bucket or "").strip().lower(), 0),
        escalation_rank.get(str(review.escalation_level or "").strip().lower(), 0),
        float(review.priority_score or 0.0),
        int(review.age_minutes or 0),
        review.created_at or datetime.min,
        review.run_id or "",
        review.queue_item_key or review.canonical_key or "",
    )


def _review_created_desc_sort_key(review: AgentControlRunReviewItemResponse) -> tuple[Any, ...]:
    return (
        review.created_at or datetime.min,
        float(review.priority_score or 0.0),
        int(review.age_minutes or 0),
        review.run_id or "",
        review.queue_item_key or review.canonical_key or "",
    )


def _review_age_desc_sort_key(review: AgentControlRunReviewItemResponse) -> tuple[Any, ...]:
    return (
        int(review.age_minutes or 0),
        float(review.priority_score or 0.0),
        review.created_at or datetime.min,
        review.run_id or "",
        review.queue_item_key or review.canonical_key or "",
    )


def _sort_control_reviews(
    items: list[AgentControlRunReviewItemResponse],
    *,
    sort_mode: str,
) -> list[AgentControlRunReviewItemResponse]:
    if sort_mode == "created_at_desc":
        return sorted(items, key=_review_created_desc_sort_key, reverse=True)
    if sort_mode == "age_desc":
        return sorted(items, key=_review_age_desc_sort_key, reverse=True)
    return sorted(items, key=_review_priority_sort_key, reverse=True)


def _primary_review_status(review: AgentControlRunReviewItemResponse) -> Optional[str]:
    for value in (review.status, review.follow_up_launch_status, review.follow_up_review_status, review.review_status):
        token = _normalize_string(value)
        if token:
            return token
    return None


def _build_control_review_summary(items: list[AgentControlRunReviewItemResponse]) -> dict[str, Any]:
    by_type = Counter(_normalize_string(item.review_type or item.item_type) or "unknown" for item in items)
    by_sla_bucket = Counter(_normalize_string(item.sla_bucket) or "unknown" for item in items)
    by_status = Counter(_primary_review_status(item) or "unknown" for item in items)
    by_customer = Counter(_normalize_string(item.customer) or "unassigned" for item in items)
    by_escalation = Counter(_normalize_string(item.escalation_level) or "unknown" for item in items)
    return {
        "total": len(items),
        "by_type": dict(by_type),
        "by_sla_bucket": dict(by_sla_bucket),
        "by_status": dict(by_status),
        "by_customer": dict(by_customer),
        "by_escalation": dict(by_escalation),
    }


def _serialize_control_run_view(row: AgentControlPlaneView) -> AgentControlRunViewResponse:
    return AgentControlRunViewResponse(
        id=str(row.id),
        user_id=str(row.user_id),
        name=str(row.name or "").strip(),
        filters=_normalize_control_run_view_filters(row.filters if isinstance(row.filters, dict) else {}),
        is_default=bool(row.is_default),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _build_autonomous_review_action_path(
    *,
    source_kind: Optional[str],
    source_id: Optional[str],
    opportunity_id: Optional[str],
) -> Optional[str]:
    kind = _normalize_string(source_kind)
    owner_id = _normalize_string(source_id)
    review_opportunity_id = _normalize_string(opportunity_id)
    if not kind or not owner_id:
        return None
    params: dict[str, str] = {}
    if kind == "profile":
        params["tab"] = "domain"
        params["profileId"] = owner_id
    elif kind == "portfolio":
        params["tab"] = "fleet"
        params["fleetId"] = owner_id
    else:
        return None
    if review_opportunity_id:
        params["opportunityId"] = review_opportunity_id
    return f"/autonomous-agents?{urlencode(params)}"


def _build_autonomous_queue_path(
    *,
    review_type: Optional[str],
    source_kind: Optional[str],
    source_id: Optional[str],
    opportunity_id: Optional[str],
    customer: Optional[str] = None,
    job_id: Optional[str] = None,
) -> str:
    params: dict[str, str] = {"tab": "queue"}
    review_token = _normalize_string(review_type)
    if review_token == "follow_up_recommendation":
        params["queue_item_type"] = "follow_up_recommendation"
        params["queue_health_drilldown"] = "pending_follow_up_approvals"
    elif review_token == "manual_follow_up_recommendation":
        params["queue_item_type"] = "follow_up_recommendation"
        params["queue_health_drilldown"] = "manual_follow_up_recommendations"
    elif review_token in {"approval_checkpoint", "job_recovery"}:
        params["queue_item_type"] = review_token
    elif review_token in {"policy_review", "budget_review"}:
        params["queue_item_type"] = review_token
    owner_kind = _normalize_string(source_kind)
    owner_id = _normalize_string(source_id)
    review_opportunity_id = _normalize_string(opportunity_id)
    if owner_kind == "profile" and owner_id:
        params["profileId"] = owner_id
    elif owner_kind == "portfolio" and owner_id:
        params["fleetId"] = owner_id
    if review_opportunity_id:
        params["opportunityId"] = review_opportunity_id
    queue_customer = _normalize_string(customer)
    queue_job = _normalize_string(job_id)
    if queue_customer:
        params["queue_customer"] = queue_customer
    if queue_job:
        params["queue_job"] = queue_job
    return f"/autonomous-agents?{urlencode(params)}"


def _build_job_queue_path(*, item_type: Optional[str], job_id: Optional[str]) -> Optional[str]:
    normalized_type = _normalize_string(item_type)
    normalized_job_id = _normalize_string(job_id)
    if not normalized_type:
        return None
    params: dict[str, str] = {
        "tab": "queue",
        "queue_item_type": normalized_type,
    }
    if normalized_job_id:
        params["queue_job"] = normalized_job_id
    return f"/autonomous-agents?{urlencode(params)}"


def _parse_run_id(run_id: str) -> tuple[str, UUID]:
    prefix, sep, raw_id = str(run_id or "").partition(":")
    if sep != ":" or prefix not in {"job", "workflow"}:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent control run not found",
        )
    try:
        return prefix, UUID(raw_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent control run not found",
        ) from exc


def _collect_named_ids(payload: Any, *, keys: set[str]) -> set[str]:
    found: set[str] = set()

    def walk(node: Any):
        if isinstance(node, dict):
            for key, value in node.items():
                key_token = str(key or "").strip().lower()
                if key_token in keys:
                    for item in _normalize_string_list(value):
                        found.add(item)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return found


def _collect_job_linkage(job: AgentJob) -> dict[str, set[str]]:
    keys_map = {
        "note_ids": {"note_id", "note_ids", "research_note_id", "research_note_ids"},
        "plan_ids": {"experiment_plan_id", "experiment_plan_ids"},
        "run_ids": {"experiment_run_id", "experiment_run_ids", "validation_run_id", "validation_run_ids"},
        "workflow_execution_ids": {"workflow_execution_id", "workflow_execution_ids"},
        "synthesis_job_ids": {
            "synthesis_job_id",
            "synthesis_job_ids",
            "source_synthesis_job_id",
            "source_synthesis_job_ids",
            "reevaluation_job_id",
            "reevaluation_job_ids",
            "explanation_synthesis_job_id",
            "proposal_synthesis_job_id",
            "patch_draft_synthesis_job_id",
        },
    }
    payloads = [job.config, job.results, job.output_artifacts]
    out: dict[str, set[str]] = {key: set() for key in keys_map}
    for payload in payloads:
        for bucket, names in keys_map.items():
            out[bucket].update(_collect_named_ids(payload, keys=names))
    return out


def _collect_workflow_inferred_job_ids(execution: WorkflowExecution) -> set[str]:
    return _collect_named_ids(
        {
            "trigger_data": execution.trigger_data,
            "context": execution.context,
        },
        keys={"job_id", "job_ids", "agent_job_id", "agent_job_ids", "root_job_id"},
    )


def _collect_routing_lineage(payloads: Iterable[Any]) -> dict[str, set[str]]:
    experiment_ids: set[str] = set()
    variant_ids: set[str] = set()
    for payload in payloads:
        experiment_ids.update(
            _collect_named_ids(
                payload,
                keys={"experiment_id", "experiment_ids", "routing_experiment_id", "routing_experiment_ids"},
            )
        )
        variant_ids.update(
            _collect_named_ids(
                payload,
                keys={"variant_id", "variant_ids", "routing_experiment_variant_id", "routing_experiment_variant_ids"},
            )
        )
    return {
        "experiment_ids": experiment_ids,
        "variant_ids": variant_ids,
    }


def _build_event_response(row: AutonomyDecisionEvent):
    metadata = row.event_metadata if isinstance(row.event_metadata, dict) else {}
    return _build_decision_trace_event(
        event_type=row.event_type,
        event_time=row.event_time,
        source_kind=row.source_kind,
        source_id=row.source_id,
        source_label=row.source_label,
        decision_type=row.decision_type,
        summary=row.summary,
        customer=row.customer,
        reason_code=row.reason_code,
        reason_label=_normalize_string(metadata.get("reason_label")),
        status=row.status,
        severity=row.severity,
        actor_mode=row.actor_mode,
        operator_note=row.operator_note,
        before_state=row.before_state if isinstance(row.before_state, dict) else None,
        after_state=row.after_state if isinstance(row.after_state, dict) else None,
        deep_link=row.deep_link if isinstance(row.deep_link, dict) else None,
        metadata=metadata or None,
        is_derived=bool(row.is_derived),
        record_origin=row.record_origin,
        suffix=str(row.id),
        operator_context=metadata.get("operator_context") if isinstance(metadata.get("operator_context"), dict) else metadata,
    )


def _derive_routing_summary(
    *,
    events: Iterable[Any],
    root_metadata: Optional[dict[str, Any]] = None,
) -> Optional[AgentControlRunRoutingSummary]:
    providers: Counter[str] = Counter()
    models: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    requested_tiers: Counter[str] = Counter()
    request_count = 0

    def absorb(payload: Any):
        nonlocal request_count
        if not isinstance(payload, dict):
            return
        provider = _normalize_string(payload.get("provider") or payload.get("routing_tier_provider"))
        model = _normalize_string(payload.get("model") or payload.get("routing_tier_model"))
        routing_tier = _normalize_string(payload.get("routing_tier"))
        requested_tier = _normalize_string(payload.get("routing_requested_tier"))
        if provider:
            providers[provider] += 1
        if model:
            models[model] += 1
        if routing_tier:
            tiers[routing_tier] += 1
        if requested_tier:
            requested_tiers[requested_tier] += 1
        if provider or model or routing_tier or requested_tier:
            request_count += 1

    for event in events:
        metadata = getattr(event, "metadata", None)
        absorb(metadata)
        absorb(getattr(event, "after_state", None))
        absorb(getattr(event, "before_state", None))

    absorb(root_metadata or {})

    if request_count == 0 and not any((providers, models, tiers, requested_tiers)):
        return None

    provider = providers.most_common(1)[0][0] if providers else None
    model = models.most_common(1)[0][0] if models else None
    routing_tier = tiers.most_common(1)[0][0] if tiers else None
    requested_tier = requested_tiers.most_common(1)[0][0] if requested_tiers else None
    parts = [part for part in [routing_tier, provider, model] if part]
    summary = " / ".join(parts) if parts else None
    return AgentControlRunRoutingSummary(
        provider=provider,
        model=model,
        routing_tier=routing_tier,
        requested_tier=requested_tier,
        request_count=request_count,
        summary=summary,
    )


def _build_replay_summary(
    *,
    source_type: str,
    title: str,
    status: str,
    current_phase: Optional[str],
    routing: Optional[AgentControlRunRoutingSummary],
    ended_at: Optional[datetime],
    child_count: int,
    decision_count: int,
) -> AgentControlRunReplaySummary:
    planner_summary = f"{source_type.title()} run '{title}' targeted phase '{current_phase or 'planning'}'."
    router_summary = routing.summary if routing and routing.summary else "No routing metadata captured for this run."
    executor_summary = (
        f"Status {status}; {child_count} downstream tasks and {decision_count} persisted decisions were linked."
    )
    replayability_status = "full_lineage" if decision_count > 0 and child_count > 0 else "partial_lineage"
    return AgentControlRunReplaySummary(
        replayability_status=replayability_status,
        planner_summary=planner_summary,
        router_summary=router_summary,
        executor_summary=executor_summary,
        ended_at=ended_at,
    )


def _build_control_run_review_item(
    *,
    row: dict[str, Any],
    opportunity: Optional[dict[str, Any]] = None,
    source_kind: str,
    source_id: str,
    latest_note_ids: Iterable[str] | None = None,
) -> AgentControlRunReviewItemResponse:
    note_id = next((item for item in (latest_note_ids or []) if _normalize_string(item)), None)
    review_type = _normalize_string(row.get("review_type"))
    available_actions: list[str] = []
    follow_up_outcome_status = _normalize_string((opportunity or {}).get("follow_up_outcome_status"))
    if review_type == "follow_up_recommendation":
        available_actions = ["approve_follow_up", "reject_follow_up"]
    elif review_type == "manual_follow_up_recommendation":
        if follow_up_outcome_status and follow_up_outcome_status.lower() in {"failed", "cancelled"}:
            available_actions = ["relaunch_follow_up"]
        else:
            available_actions = ["launch_follow_up"]
    elif review_type == "policy_review":
        available_actions = ["apply_guardrail"]
    customer = _normalize_string(
        row.get("customer") if row.get("customer") is not None else (opportunity or {}).get("customer")
    )
    monitor_job_id = _normalize_string(
        row.get("monitor_job_id") if row.get("monitor_job_id") is not None else (opportunity or {}).get("monitor_job_id")
    )
    policy_update_payload = row.get("policy_update_payload")
    if not isinstance(policy_update_payload, dict):
        policy_update_payload = (opportunity or {}).get("policy_update_payload")
    if not isinstance(policy_update_payload, dict):
        policy_update_payload = None
    policy_rollback_payload = row.get("policy_rollback_payload")
    if not isinstance(policy_rollback_payload, dict):
        policy_rollback_payload = (opportunity or {}).get("policy_rollback_payload")
    if not isinstance(policy_rollback_payload, dict):
        policy_rollback_payload = None
    metadata = {
        "canonical_key": _normalize_string(row.get("canonical_key")),
        "source_kind": source_kind,
        "source_id": source_id,
        "follow_up_outcome_status": follow_up_outcome_status,
        "customer": customer,
        "monitor_job_id": monitor_job_id,
    }
    opportunity_id = _normalize_string(row.get("opportunity_id"))
    queue_item_key = "::".join(
        [
            _normalize_string(source_kind) or "review",
            _normalize_string(source_id) or "owner",
            opportunity_id or "opportunity",
            review_type or "review",
        ]
    )
    return AgentControlRunReviewItemResponse(
        review_type=review_type,
        review_status="queued",
        reason_code=_normalize_string(row.get("reason_code")),
        reason_label=_normalize_string(row.get("reason_label")),
        source_kind=source_kind,
        source_id=source_id,
        opportunity_id=opportunity_id,
        canonical_key=_normalize_string(row.get("canonical_key")),
        title=_normalize_string(row.get("title")),
        evidence_revision=_normalize_string(row.get("evidence_revision") or (opportunity or {}).get("evidence_revision")),
        autonomy_state=_normalize_string(row.get("autonomy_state") or (opportunity or {}).get("autonomy_state")),
        operator_note=_normalize_string(row.get("operator_note") or (opportunity or {}).get("operator_note")),
        item_type="follow_up_recommendation" if review_type in {"follow_up_recommendation", "manual_follow_up_recommendation"} else review_type,
        queue_item_key=queue_item_key,
        customer=customer,
        job_id=monitor_job_id,
        follow_up_launch_status=_normalize_string((opportunity or {}).get("follow_up_launch_status") or (opportunity or {}).get("follow_up_outcome_status")),
        follow_up_review_status=_normalize_string((opportunity or {}).get("follow_up_review_status")),
        follow_up_recommendation_key=_normalize_string((opportunity or {}).get("follow_up_recommendation_key")),
        recommendation_score=_normalize_float(
            row.get("recommendation_score") if row.get("recommendation_score") is not None else (opportunity or {}).get("recommendation_score")
        ),
        follow_up_block_reason=_normalize_string((opportunity or {}).get("follow_up_block_reason")),
        follow_up_budget_decision=_normalize_string((opportunity or {}).get("follow_up_budget_decision")),
        follow_up_budget_reason=_normalize_string((opportunity or {}).get("follow_up_budget_reason")),
        follow_up_customer_budget_decision=_normalize_string((opportunity or {}).get("follow_up_customer_budget_decision")),
        follow_up_customer_budget_reason=_normalize_string((opportunity or {}).get("follow_up_customer_budget_reason")),
        recommended_action=_normalize_string(
            row.get("recommended_action") if row.get("recommended_action") is not None else (opportunity or {}).get("recommended_action")
        ),
        policy_update_payload=policy_update_payload,
        policy_rollback_payload=policy_rollback_payload,
        policy_guardrail_action=_normalize_string(
            row.get("policy_guardrail_action")
            if row.get("policy_guardrail_action") is not None
            else (opportunity or {}).get("policy_guardrail_action")
        ),
        policy_guardrail_target_history_entry_id=_normalize_string(
            row.get("policy_guardrail_target_history_entry_id")
            if row.get("policy_guardrail_target_history_entry_id") is not None
            else (opportunity or {}).get("policy_guardrail_target_history_entry_id")
        ),
        policy_guardrail_reasons=_normalize_string_list(
            row.get("policy_guardrail_reasons")
            if row.get("policy_guardrail_reasons") is not None
            else (opportunity or {}).get("policy_guardrail_reasons")
        ),
        budget_throttle_state=_normalize_string(
            row.get("budget_throttle_state")
            if row.get("budget_throttle_state") is not None
            else (opportunity or {}).get("budget_throttle_state")
        ),
        budget_reason=_normalize_string(
            row.get("budget_reason")
            if row.get("budget_reason") is not None
            else (opportunity or {}).get("budget_reason")
        ),
        customer_budget_throttle_state=_normalize_string(
            row.get("customer_budget_throttle_state")
            if row.get("customer_budget_throttle_state") is not None
            else (opportunity or {}).get("customer_budget_throttle_state")
        ),
        customer_budget_reason=_normalize_string(
            row.get("customer_budget_reason")
            if row.get("customer_budget_reason") is not None
            else (opportunity or {}).get("customer_budget_reason")
        ),
        action_path=_build_autonomous_review_action_path(
            source_kind=source_kind,
            source_id=source_id,
            opportunity_id=opportunity_id,
        ),
        queue_path=_build_autonomous_queue_path(
            review_type=review_type,
            source_kind=source_kind,
            source_id=source_id,
            opportunity_id=opportunity_id,
            customer=customer,
            job_id=monitor_job_id,
        ),
        note_path=f"/research-notes?note={note_id}" if note_id else None,
        available_actions=available_actions,
        can_acknowledge="mark_reviewed" in available_actions,
        can_approve="approve_follow_up" in available_actions,
        can_reject="reject_follow_up" in available_actions,
        can_defer="defer_review" in available_actions,
        can_launch_follow_up="launch_follow_up" in available_actions,
        can_relaunch_follow_up="relaunch_follow_up" in available_actions,
        metadata=metadata,
    )


def _build_job_queue_review_item(
    row: Any,
) -> AgentControlRunReviewItemResponse:
    item_type = _normalize_string(getattr(row, "item_type", None))
    job_id = _normalize_string(getattr(row, "job_id", None))
    available_actions = [
        str(getattr(action, "action", "")).strip()
        for action in (getattr(row, "actions", None) or [])
        if str(getattr(action, "kind", "")).strip() == "job_action"
        and str(getattr(action, "action", "")).strip()
    ]
    checkpoint_payload = getattr(row, "checkpoint", None) if isinstance(getattr(row, "checkpoint", None), dict) else None
    checkpoint_action = checkpoint_payload.get("action") if isinstance(checkpoint_payload, dict) and isinstance(checkpoint_payload.get("action"), dict) else {}
    checkpoint_action_draft = None
    if checkpoint_action:
        checkpoint_action_draft = {
            "tool": _normalize_string(checkpoint_action.get("tool")),
            "purpose": _normalize_string(checkpoint_action.get("purpose")),
            "params": checkpoint_action.get("params") if isinstance(checkpoint_action.get("params"), dict) else {},
        }
    metadata = {
        "queue_key": _normalize_string(getattr(row, "queue_key", None)),
        "job_id": job_id,
        "item_type": item_type,
        "checkpoint": checkpoint_payload,
        "checkpoint_action_draft": checkpoint_action_draft,
        "scheduler_state": getattr(row, "scheduler_state", None) if isinstance(getattr(row, "scheduler_state", None), dict) else None,
    }
    return AgentControlRunReviewItemResponse(
        review_type=item_type,
        review_status="queued",
        reason_code=_normalize_string(getattr(row, "reason_code", None)),
        reason_label=_normalize_string(getattr(row, "reason_label", None)),
        source_kind="job",
        source_id=job_id,
        opportunity_id=job_id,
        canonical_key=_normalize_string(getattr(row, "queue_key", None)),
        title=_normalize_string(getattr(row, "title", None)),
        created_at=getattr(row, "created_at", None),
        action_path=f"/autonomous-agents?job={job_id}" if job_id else None,
        queue_path=_build_job_queue_path(item_type=item_type, job_id=job_id),
        item_type=item_type,
        queue_item_key=_normalize_string(getattr(row, "queue_key", None)),
        status=_normalize_string(getattr(row, "status", None)),
        summary=_normalize_string(getattr(row, "summary", None)),
        evidence_summary=_normalize_string(getattr(row, "evidence_summary", None)),
        customer=_normalize_string(getattr(row, "customer", None)),
        job_id=job_id,
        job_name=_normalize_string(getattr(row, "job_name", None)),
        job_type=_normalize_string(getattr(row, "job_type", None)),
        age_minutes=getattr(row, "age_minutes", None),
        priority_score=_normalize_float(getattr(row, "priority_score", None)),
        sla_bucket=_normalize_string(getattr(row, "sla_bucket", None)),
        escalation_level=_normalize_string(getattr(row, "escalation_level", None)),
        next_run_at=getattr(row, "next_run_at", None),
        backoff_until=getattr(row, "backoff_until", None),
        checkpoint=checkpoint_payload,
        checkpoint_action_draft=checkpoint_action_draft,
        scheduler_state=getattr(row, "scheduler_state", None) if isinstance(getattr(row, "scheduler_state", None), dict) else None,
        available_actions=available_actions,
        can_approve="approve" in available_actions,
        can_reject="reject" in available_actions,
        can_defer=False,
        can_skip="skip" in available_actions,
        can_restart="restart" in available_actions,
        can_resume="resume" in available_actions,
        can_cancel="cancel" in available_actions,
        metadata=metadata,
    )


async def _get_control_review_target(
    *,
    db: AsyncSession,
    current_user: User,
    source_kind: str,
    source_id: str,
) -> DomainResearchProfile | ResearchPortfolio:
    if source_kind == "profile":
        row = await db.get(DomainResearchProfile, UUID(source_id))
    elif source_kind == "portfolio":
        row = await db.get(ResearchPortfolio, UUID(source_id))
    else:
        raise HTTPException(status_code=400, detail="Unsupported review source kind")
    if row is None or str(getattr(row, "user_id", "")) != str(current_user.id):
        raise HTTPException(status_code=404, detail="Review target not found")
    return row


def _build_job_summary(
    *,
    job: AgentJob,
    child_job_count: int,
    linked_note_count: int,
    linked_experiment_count: int,
    decision_count: int,
    routing: Optional[AgentControlRunRoutingSummary],
) -> AgentControlRunSummary:
    config = job.config if isinstance(job.config, dict) else {}
    replayability_status = "full_lineage" if child_job_count > 0 and decision_count > 0 else "partial_lineage"
    return AgentControlRunSummary(
        id=f"job:{job.id}",
        source_type="job",
        title=job.name,
        subtitle=_normalize_string(job.goal),
        status=job.status,
        outcome=_normalize_string(job.status),
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        root_job_id=str(job.id),
        child_job_count=child_job_count,
        linked_note_count=linked_note_count,
        linked_experiment_count=linked_experiment_count,
        decision_count=decision_count,
        replayability_status=replayability_status,
        automation_profile=_normalize_string(config.get("automation_profile")),
        routing=routing,
    )


def _build_workflow_summary(
    *,
    execution: WorkflowExecution,
    child_execution_count: int,
    child_job_count: int,
    linked_note_count: int,
    linked_experiment_count: int,
    decision_count: int,
    routing: Optional[AgentControlRunRoutingSummary],
) -> AgentControlRunSummary:
    workflow_name = execution.workflow.name if execution.workflow else f"Workflow {execution.workflow_id}"
    replayability_status = "full_lineage" if (child_execution_count or child_job_count) and decision_count > 0 else "partial_lineage"
    return AgentControlRunSummary(
        id=f"workflow:{execution.id}",
        source_type="workflow",
        title=workflow_name,
        subtitle=_normalize_string(execution.trigger_type),
        status=execution.status,
        outcome=_normalize_string(execution.status),
        created_at=execution.created_at,
        started_at=execution.started_at,
        completed_at=execution.completed_at,
        workflow_execution_id=str(execution.id),
        child_job_count=child_job_count,
        child_execution_count=child_execution_count,
        linked_note_count=linked_note_count,
        linked_experiment_count=linked_experiment_count,
        decision_count=decision_count,
        replayability_status=replayability_status,
        routing=routing,
    )


async def _collect_control_run_review_mappings(
    *,
    db: AsyncSession,
    current_user: User,
    job_ids: set[str],
) -> tuple[dict[str, list[AgentControlRunReviewItemResponse]], dict[str, Counter[str]]]:
    if not job_ids:
        return {}, {}

    parsed_job_ids: set[UUID] = set()
    for raw_id in job_ids:
        try:
            parsed_job_ids.add(UUID(raw_id))
        except ValueError:
            continue
    if not parsed_job_ids:
        return {}, {}

    rows_by_job_id: dict[str, list[AgentControlRunReviewItemResponse]] = {}
    counts_by_job_id: dict[str, Counter[str]] = {}

    job_rows = list(
        (
            await db.execute(
                select(AgentJob)
                .options(selectinload(AgentJob.agent_definition))
                .where(
                    AgentJob.user_id == current_user.id,
                    AgentJob.id.in_(list(parsed_job_ids)),
                )
            )
        ).scalars().all()
    )
    queue_items = _build_checkpoint_queue_items(job_rows, [])
    for queue_item in queue_items:
        item_type = _normalize_string(getattr(queue_item, "item_type", None))
        if item_type not in {"approval_checkpoint", "job_recovery"}:
            continue
        job_token = _normalize_string(getattr(queue_item, "job_id", None))
        if not job_token or job_token not in job_ids:
            continue
        item = _build_job_queue_review_item(queue_item)
        rows_by_job_id.setdefault(job_token, []).append(item)
        if item.review_type:
            counts_by_job_id.setdefault(job_token, Counter())[item.review_type] += 1

    def _opportunity_lookup(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
        rows = summary.get("opportunities") if isinstance(summary.get("opportunities"), list) else summary.get("idea_candidates")
        lookup: dict[str, dict[str, Any]] = {}
        if not isinstance(rows, list):
            return lookup
        for row in rows:
            if not isinstance(row, dict):
                continue
            opportunity_id = _normalize_string(row.get("opportunity_id"))
            if opportunity_id:
                lookup[opportunity_id] = row
        return lookup

    def _append_review_rows(
        *,
        summary: dict[str, Any],
        job_tokens: set[str],
        source_kind: str,
        source_id: str,
        latest_note_ids: list[str],
    ) -> None:
        opportunity_by_id = _opportunity_lookup(summary)
        merged_rows: list[dict[str, Any]] = []
        queued_review_rows = summary.get("queued_operator_reviews") if isinstance(summary.get("queued_operator_reviews"), list) else []
        for row in queued_review_rows:
            if isinstance(row, dict):
                merged_rows.append(row)
        manual_review_rows = (
            summary.get("manual_follow_up_recommendations")
            if isinstance(summary.get("manual_follow_up_recommendations"), list)
            else []
        )
        for row in manual_review_rows:
            if not isinstance(row, dict):
                continue
            merged_rows.append(
                {
                    **row,
                    "review_type": "manual_follow_up_recommendation",
                    "reason_label": row.get("reason_label") or "Manual follow-up recommendation",
                }
            )
        if not merged_rows or not job_tokens:
            return
        for job_token in job_tokens:
            bucket = rows_by_job_id.setdefault(job_token, [])
            count_bucket = counts_by_job_id.setdefault(job_token, Counter())
            for row in merged_rows:
                opportunity = opportunity_by_id.get(_normalize_string(row.get("opportunity_id")) or "")
                item = _build_control_run_review_item(
                    row=row,
                    opportunity=opportunity,
                    source_kind=source_kind,
                    source_id=source_id,
                    latest_note_ids=latest_note_ids,
                )
                bucket.append(item)
                review_type = _normalize_string(item.review_type)
                if review_type:
                    count_bucket[review_type] += 1

    profile_rows = list(
        (
            await db.execute(
                select(DomainResearchProfile).where(
                    DomainResearchProfile.user_id == current_user.id,
                    or_(
                        DomainResearchProfile.active_job_id.in_(list(parsed_job_ids)),
                        DomainResearchProfile.latest_run_job_id.in_(list(parsed_job_ids)),
                    ),
                )
            )
        ).scalars().all()
    )
    portfolio_rows = list(
        (
            await db.execute(
                select(ResearchPortfolio).where(
                    ResearchPortfolio.user_id == current_user.id,
                    or_(
                        ResearchPortfolio.active_job_id.in_(list(parsed_job_ids)),
                        ResearchPortfolio.latest_run_job_id.in_(list(parsed_job_ids)),
                    ),
                )
            )
        ).scalars().all()
    )

    for profile in profile_rows:
        summary = profile.latest_summary if isinstance(profile.latest_summary, dict) else {}
        job_tokens = {
            _normalize_string(profile.active_job_id),
            _normalize_string(profile.latest_run_job_id),
        }
        job_tokens = {token for token in job_tokens if token}
        _append_review_rows(
            summary=summary,
            job_tokens=job_tokens,
            source_kind="profile",
            source_id=str(profile.id),
            latest_note_ids=profile.latest_note_ids if isinstance(profile.latest_note_ids, list) else [],
        )

    for portfolio in portfolio_rows:
        summary = portfolio.latest_summary if isinstance(portfolio.latest_summary, dict) else {}
        job_tokens = {
            _normalize_string(portfolio.active_job_id),
            _normalize_string(portfolio.latest_run_job_id),
        }
        child_job_ids = portfolio.child_job_ids if isinstance(portfolio.child_job_ids, list) else []
        job_tokens.update(_normalize_string_list(child_job_ids))
        job_tokens = {token for token in job_tokens if token in job_ids}
        _append_review_rows(
            summary=summary,
            job_tokens=job_tokens,
            source_kind="portfolio",
            source_id=str(portfolio.id),
            latest_note_ids=portfolio.latest_note_ids if isinstance(portfolio.latest_note_ids, list) else [],
        )

    return rows_by_job_id, counts_by_job_id


def _review_matches_filters(
    *,
    review: AgentControlRunReviewItemResponse,
    review_type: Optional[str],
    review_status: Optional[str],
    queue_status: Optional[str],
    queue_customer: Optional[str],
    queue_sla: Optional[str],
    queue_escalation: Optional[str],
    queue_health_drilldown: Optional[str],
    queue_preset: Optional[str],
) -> bool:
    normalized_review_type = _normalize_string(review_type)
    normalized_review_status = _normalize_string(review_status)
    normalized_queue_status = _normalize_string(queue_status)
    normalized_queue_customer = _normalize_string(queue_customer)
    normalized_queue_sla = _normalize_string(queue_sla)
    normalized_queue_escalation = _normalize_string(queue_escalation)
    normalized_queue_health_drilldown = _normalize_string(queue_health_drilldown)
    normalized_queue_preset = _normalize_string(queue_preset)

    if normalized_review_type and _normalize_string(review.review_type) != normalized_review_type:
        return False
    if normalized_review_status and _normalize_string(review.review_status) != normalized_review_status:
        return False

    status_tokens = {
        token.lower()
        for token in [
            _normalize_string(review.status),
            _normalize_string(review.follow_up_launch_status),
            _normalize_string(review.follow_up_review_status),
        ]
        if token
    }
    if normalized_queue_status and normalized_queue_status.lower() not in status_tokens:
        return False
    if normalized_queue_customer and (_normalize_string(review.customer) or "").lower() != normalized_queue_customer.lower():
        return False
    if normalized_queue_sla and (_normalize_string(review.sla_bucket) or "").lower() != normalized_queue_sla.lower():
        return False
    if normalized_queue_escalation and (_normalize_string(review.escalation_level) or "").lower() != normalized_queue_escalation.lower():
        return False

    if normalized_queue_health_drilldown == "pending_follow_up_approvals":
        if not (
            _normalize_string(review.review_type) == "follow_up_recommendation"
            and (_normalize_string(review.follow_up_launch_status) or "").lower() == "pending_approval"
        ):
            return False
    elif normalized_queue_health_drilldown == "manual_follow_up_recommendations":
        if _normalize_string(review.review_type) != "manual_follow_up_recommendation":
            return False
    elif normalized_queue_health_drilldown == "blocked_follow_up":
        blocked = bool(_normalize_string(review.follow_up_block_reason)) or _normalize_string(review.review_type) in {"budget_review", "policy_review"}
        if not blocked:
            return False

    if normalized_queue_preset == "approval_required":
        is_approval_required = _normalize_string(review.review_type) == "approval_checkpoint" or (
            _normalize_string(review.review_type) == "follow_up_recommendation"
            and (_normalize_string(review.follow_up_launch_status) or "").lower() == "pending_approval"
        )
        if not is_approval_required:
            return False
    elif normalized_queue_preset == "failed_recovery":
        if _normalize_string(review.review_type) != "job_recovery":
            return False
    elif normalized_queue_preset == "compiler":
        compiler_tokens = [
            _normalize_string(review.customer),
            _normalize_string(review.title),
            _normalize_string(review.summary),
            _normalize_string(review.evidence_summary),
            _normalize_string(review.job_name),
            _normalize_string(review.job_type),
            _normalize_string(review.run_title),
        ]
        if not any(token and "compiler" in token.lower() for token in compiler_tokens):
            return False

    return True


async def _build_control_run_summaries(
    *,
    db: AsyncSession,
    current_user: User,
    source_type: Optional[str],
    has_operator_review: Optional[bool],
    review_type: Optional[str],
    review_status: Optional[str],
    limit: int,
) -> tuple[
    list[AgentControlRunSummary],
    dict[str, set[str]],
    dict[str, set[str]],
    dict[str, list[AgentControlRunReviewItemResponse]],
]:
    items: list[AgentControlRunSummary] = []
    normalized_type = _normalize_string(source_type)
    normalized_review_type = _normalize_string(review_type)
    normalized_review_status = _normalize_string(review_status)
    root_job_ids_for_reviews: set[str] = set()
    job_linked_job_ids_by_root: dict[str, set[str]] = {}
    workflow_linked_job_ids_by_root: dict[str, set[str]] = {}

    if normalized_type in {None, "job"}:
        job_rows = list(
            (
                await db.execute(
                    select(AgentJob)
                    .where(
                        AgentJob.user_id == current_user.id,
                        # A control-run root has no parent (mirrors
                        # AgentJob.is_control_run_root). chain_depth defaults to
                        # 0 even for children, so it cannot be used to identify
                        # roots or child jobs would be double-counted as roots.
                        AgentJob.parent_job_id.is_(None),
                    )
                    .order_by(AgentJob.created_at.desc())
                    .limit(limit)
                )
            ).scalars().all()
        )
        if job_rows:
            # Parentless jobs whose only control-plane relevance is a pending
            # approval checkpoint or a recovery candidate are queue items, not
            # standalone runs. Fold them into the newest genuine (non-queue-only)
            # root run so their reviews surface there instead of spawning empty
            # duplicate runs.
            queue_only_item_types = {"approval_checkpoint", "job_recovery"}
            queue_item_types_by_job: dict[str, set[str]] = {}
            for queue_item in _build_checkpoint_queue_items(job_rows, []):
                q_type = _normalize_string(getattr(queue_item, "item_type", None))
                q_job = _normalize_string(getattr(queue_item, "job_id", None))
                if q_type and q_job:
                    queue_item_types_by_job.setdefault(q_job, set()).add(q_type)

            def _is_queue_only_root(job: AgentJob) -> bool:
                types = queue_item_types_by_job.get(str(job.id), set())
                if not types or not types.issubset(queue_only_item_types):
                    return False
                # A job that anchors a lineage (children) is a real run.
                return True

            primary_job_rows = [job for job in job_rows if not _is_queue_only_root(job)]
            queue_only_job_rows = [job for job in job_rows if _is_queue_only_root(job)]
            # Anchor queue-only jobs to the newest primary root (job_rows are
            # ordered created_at desc, so the first primary is the newest).
            anchor_root_id = str(primary_job_rows[0].id) if primary_job_rows else None
            if anchor_root_id is None:
                # No primary root: treat queue-only jobs as their own runs.
                primary_job_rows = job_rows
                queue_only_job_rows = []
            job_rows = primary_job_rows

            root_ids = [job.id for job in job_rows]
            job_linked_job_ids_by_root = {str(job.id): {str(job.id)} for job in job_rows}
            if anchor_root_id is not None:
                for queue_job in queue_only_job_rows:
                    job_linked_job_ids_by_root.setdefault(anchor_root_id, {anchor_root_id}).add(str(queue_job.id))
            lineage_rows = list(
                (
                    await db.execute(
                        select(AgentJob).where(
                            AgentJob.user_id == current_user.id,
                            or_(
                                AgentJob.root_job_id.in_(root_ids),
                                AgentJob.parent_job_id.in_(root_ids),
                            ),
                        )
                    )
                ).scalars().all()
            )
            root_id_tokens = {str(root_id) for root_id in root_ids}
            lineage_ids = {str(row.id) for row in lineage_rows}
            lineage_ids.update(root_id_tokens)
            event_rows = list(
                (
                    await db.execute(
                        select(AutonomyDecisionEvent).where(
                            AutonomyDecisionEvent.user_id == current_user.id,
                            AutonomyDecisionEvent.source_id.in_(lineage_ids),
                        )
                    )
                ).scalars().all()
            )
            child_count_by_root: Counter[str] = Counter()
            note_count_by_root: Counter[str] = Counter()
            experiment_count_by_root: Counter[str] = Counter()
            decision_count_by_root: Counter[str] = Counter()
            lineage_root_lookup: dict[str, str] = {}
            for row in lineage_rows:
                root_key = _normalize_string(row.root_job_id or row.parent_job_id)
                if not root_key:
                    continue
                lineage_root_lookup[str(row.id)] = root_key
                job_linked_job_ids_by_root.setdefault(root_key, {root_key}).add(str(row.id))
                child_count_by_root[root_key] += 1
                linkage = _collect_job_linkage(row)
                note_count_by_root[root_key] += len(linkage["note_ids"])
                experiment_count_by_root[root_key] += len(linkage["plan_ids"]) + len(linkage["run_ids"])
            for event in event_rows:
                event_source_id = _normalize_string(event.source_id)
                if not event_source_id:
                    continue
                root_key = lineage_root_lookup.get(event_source_id, event_source_id if event_source_id in root_id_tokens else None)
                if root_key:
                    decision_count_by_root[root_key] += 1
            for job in job_rows:
                root_key = str(job.id)
                routing = _derive_routing_summary(events=[], root_metadata=job.config if isinstance(job.config, dict) else {})
                items.append(
                    _build_job_summary(
                        job=job,
                        child_job_count=child_count_by_root.get(root_key, 0),
                        linked_note_count=note_count_by_root.get(root_key, 0),
                        linked_experiment_count=experiment_count_by_root.get(root_key, 0),
                        decision_count=decision_count_by_root.get(root_key, 0),
                        routing=routing,
                    )
                )
                root_job_ids_for_reviews.update(job_linked_job_ids_by_root.get(root_key, {root_key}))

    if normalized_type in {None, "workflow"}:
        workflow_rows = list(
            (
                await db.execute(
                    select(WorkflowExecution)
                    .options(selectinload(WorkflowExecution.workflow))
                    .where(
                        WorkflowExecution.user_id == current_user.id,
                        WorkflowExecution.parent_execution_id.is_(None),
                    )
                    .order_by(WorkflowExecution.created_at.desc())
                    .limit(limit)
                )
            ).scalars().all()
        )
        if workflow_rows:
            root_ids = [row.id for row in workflow_rows]
            child_rows = list(
                (
                    await db.execute(
                        select(WorkflowExecution).where(
                            WorkflowExecution.user_id == current_user.id,
                            WorkflowExecution.parent_execution_id.in_(root_ids),
                        )
                    )
                ).scalars().all()
            )
            execution_id_tokens = {str(row.id) for row in workflow_rows}
            execution_id_tokens.update(str(row.id) for row in child_rows)
            event_rows = list(
                (
                    await db.execute(
                        select(AutonomyDecisionEvent).where(
                            AutonomyDecisionEvent.user_id == current_user.id,
                            AutonomyDecisionEvent.source_id.in_(execution_id_tokens),
                        )
                    )
                ).scalars().all()
            )
            child_count_by_root: Counter[str] = Counter(str(row.parent_execution_id) for row in child_rows if row.parent_execution_id)
            execution_root_lookup: dict[str, str] = {str(row.id): str(row.parent_execution_id) for row in child_rows if row.parent_execution_id}
            inferred_job_ids_by_root: dict[str, set[str]] = {str(row.id): _collect_workflow_inferred_job_ids(row) for row in workflow_rows}
            all_inferred_job_ids: set[UUID] = set()
            for raw_ids in inferred_job_ids_by_root.values():
                for raw_id in raw_ids:
                    try:
                        all_inferred_job_ids.add(UUID(raw_id))
                    except ValueError:
                        continue
            existing_job_ids: set[str] = set()
            if all_inferred_job_ids:
                existing_job_rows = list(
                    (
                        await db.execute(
                            select(AgentJob.id).where(
                                AgentJob.user_id == current_user.id,
                                AgentJob.id.in_(list(all_inferred_job_ids)),
                            )
                        )
                    ).scalars().all()
                )
                existing_job_ids = {str(row) for row in existing_job_rows}
            valid_job_count_by_root: Counter[str] = Counter()
            for root_key, raw_ids in inferred_job_ids_by_root.items():
                valid_job_count_by_root[root_key] = sum(1 for raw_id in raw_ids if raw_id in existing_job_ids)
            decision_count_by_root: Counter[str] = Counter()
            root_id_tokens = {str(root_id) for root_id in root_ids}
            for event in event_rows:
                event_source_id = _normalize_string(event.source_id)
                if not event_source_id:
                    continue
                root_key = execution_root_lookup.get(event_source_id, event_source_id if event_source_id in root_id_tokens else None)
                if root_key:
                    decision_count_by_root[root_key] += 1
            for row in workflow_rows:
                workflow_linked_job_ids_by_root[str(row.id)] = {
                    raw_id for raw_id in inferred_job_ids_by_root.get(str(row.id), set()) if raw_id in existing_job_ids
                }
                root_job_ids_for_reviews.update(workflow_linked_job_ids_by_root[str(row.id)])
                items.append(
                    _build_workflow_summary(
                        execution=row,
                        child_execution_count=child_count_by_root.get(str(row.id), 0),
                        child_job_count=valid_job_count_by_root.get(str(row.id), 0),
                        linked_note_count=0,
                        linked_experiment_count=0,
                        decision_count=decision_count_by_root.get(str(row.id), 0),
                        routing=None,
                    )
                )

    review_rows_by_job_id, review_counts_by_job_id = await _collect_control_run_review_mappings(
        db=db,
        current_user=current_user,
        job_ids=root_job_ids_for_reviews,
    )

    filtered_items: list[AgentControlRunSummary] = []
    for item in items:
        queued_reviews: list[AgentControlRunReviewItemResponse] = []
        review_counts: Counter[str] = Counter()
        if item.source_type == "job" and item.root_job_id:
            for linked_job_id in job_linked_job_ids_by_root.get(item.root_job_id, {item.root_job_id}):
                queued_reviews.extend(review_rows_by_job_id.get(linked_job_id, []))
                review_counts.update(review_counts_by_job_id.get(linked_job_id, Counter()))
        elif item.source_type == "workflow" and item.workflow_execution_id:
            for linked_job_id in workflow_linked_job_ids_by_root.get(item.workflow_execution_id, set()):
                queued_reviews.extend(review_rows_by_job_id.get(linked_job_id, []))
                review_counts.update(review_counts_by_job_id.get(linked_job_id, Counter()))
        item.queued_operator_review_count = len(queued_reviews)
        item.queued_operator_reviews_by_type = dict(review_counts)
        if has_operator_review is True and item.queued_operator_review_count <= 0:
            continue
        if has_operator_review is False and item.queued_operator_review_count > 0:
            continue
        if normalized_review_type and review_counts.get(normalized_review_type, 0) <= 0:
            continue
        if normalized_review_status and normalized_review_status != "queued":
            continue
        filtered_items.append(item)

    filtered_items.sort(key=lambda item: item.created_at, reverse=True)
    trimmed = filtered_items[:limit]
    return trimmed, job_linked_job_ids_by_root, workflow_linked_job_ids_by_root, review_rows_by_job_id


async def _slice_memory_graph_for_job_ids(
    *,
    db: AsyncSession,
    current_user: User,
    job_ids: set[str],
):
    if not job_ids:
        return None
    from app.services.agent_job_memory_service import agent_job_memory_service

    graph = await agent_job_memory_service.get_task_memory_graph(
        user_id=str(current_user.id),
        db=db,
        limit=180,
        min_link_score=1.0,
        max_edges=1200,
    )
    nodes = graph.get("nodes") if isinstance(graph, dict) else []
    edges = graph.get("edges") if isinstance(graph, dict) else []
    if not isinstance(nodes, list) or not isinstance(edges, list):
        return None

    selected = {str(node.get("id")) for node in nodes if str(node.get("job_id") or "") in job_ids}
    if not selected:
        return None

    expanded = set(selected)
    for edge in edges:
        src = str(edge.get("source") or "")
        dst = str(edge.get("target") or "")
        if src in selected or dst in selected:
            expanded.add(src)
            expanded.add(dst)

    filtered_nodes = [node for node in nodes if str(node.get("id") or "") in expanded]
    filtered_edges = [
        edge
        for edge in edges
        if str(edge.get("source") or "") in expanded and str(edge.get("target") or "") in expanded
    ]
    return _build_memory_graph_response(
        graph={
            "nodes": filtered_nodes,
            "edges": filtered_edges,
            "stats": {
                "memory_count": len(filtered_nodes),
                "edge_count": len(filtered_edges),
                "job_count": len(job_ids),
            },
        }
    )


def _collect_event_linkage(events: Iterable[Any]) -> dict[str, set[str]]:
    note_ids: set[str] = set()
    plan_ids: set[str] = set()
    run_ids: set[str] = set()
    child_job_ids: set[str] = set()
    synthesis_job_ids: set[str] = set()
    for event in events:
        note_ids.update(_normalize_string_list(getattr(event, "linked_note_ids", None)))
        plan_ids.update(_normalize_string_list(getattr(event, "linked_experiment_plan_ids", None)))
        run_ids.update(_normalize_string_list(getattr(event, "linked_validation_run_ids", None)))
        child_job_ids.update(_normalize_string_list(getattr(event, "child_job_ids", None)))
        metadata = getattr(event, "metadata", None)
        if isinstance(metadata, dict):
            synthesis_job_ids.update(
                _collect_named_ids(
                    metadata,
                    keys={
                        "synthesis_job_id",
                        "synthesis_job_ids",
                        "source_synthesis_job_id",
                        "source_synthesis_job_ids",
                        "reevaluation_job_id",
                        "reevaluation_job_ids",
                        "review_job_id",
                        "explanation_synthesis_job_id",
                        "proposal_synthesis_job_id",
                        "patch_draft_synthesis_job_id",
                    },
                )
            )
    return {
        "note_ids": note_ids,
        "plan_ids": plan_ids,
        "run_ids": run_ids,
        "child_job_ids": child_job_ids,
        "synthesis_job_ids": synthesis_job_ids,
    }


def _build_routing_query_params(
    *,
    routing: Optional[AgentControlRunRoutingSummary],
    experiment_ids: Optional[Iterable[str]] = None,
    variant_ids: Optional[Iterable[str]] = None,
) -> str:
    params: dict[str, str] = {}
    if routing:
        provider = _normalize_string(routing.provider)
        model = _normalize_string(routing.model)
        tier = _normalize_string(routing.routing_tier)
        if provider:
            params["provider"] = provider
        if model:
            params["model"] = model
        if tier:
            params["routing_tier"] = tier
    experiment_id = next((value for value in (experiment_ids or []) if _normalize_string(value)), None)
    variant_id = next((value for value in (variant_ids or []) if _normalize_string(value)), None)
    if experiment_id:
        params["experiment_id"] = experiment_id
    if variant_id:
        params["variant_id"] = variant_id
    if not params:
        return "/usage/routing"
    return f"/usage/routing?{urlencode(params)}"


def _build_related_links(
    *,
    source_type: str,
    root_job_id: Optional[str],
    workflow_execution_id: Optional[str],
    note_ids: Iterable[str],
    plan_ids: Iterable[str],
    run_ids: Iterable[str],
    synthesis_job_ids: Optional[Iterable[str]] = None,
    routing: Optional[AgentControlRunRoutingSummary] = None,
    routing_experiment_ids: Optional[Iterable[str]] = None,
    routing_variant_ids: Optional[Iterable[str]] = None,
) -> list[AgentControlRunLinkResponse]:
    links: list[AgentControlRunLinkResponse] = []
    if root_job_id:
        links.append(
            AgentControlRunLinkResponse(
                label="Autonomous Agents",
                path=f"/autonomous-agents?job={root_job_id}",
            )
        )
    if workflow_execution_id:
        links.append(
            AgentControlRunLinkResponse(
                label="Workflows",
                path=f"/workflows?executionId={workflow_execution_id}",
            )
        )
    first_note = next(iter(note_ids), None)
    if first_note:
        links.append(
            AgentControlRunLinkResponse(
                label="Research Notes",
                path=f"/research-notes?note={first_note}",
            )
        )
    synthesis_job_id = next(iter(synthesis_job_ids or []), None)
    if synthesis_job_id:
        links.append(AgentControlRunLinkResponse(label="Synthesis", path=f"/synthesis?job={synthesis_job_id}"))
    elif next(iter(run_ids), None):
        links.append(AgentControlRunLinkResponse(label="Synthesis", path="/synthesis"))
    if next(iter(plan_ids), None) or source_type == "job" or routing:
        links.append(
            AgentControlRunLinkResponse(
                label="Routing Observability",
                path=_build_routing_query_params(
                    routing=routing,
                    experiment_ids=routing_experiment_ids,
                    variant_ids=routing_variant_ids,
                ),
            )
        )
    return links


@router.get("/views", response_model=AgentControlRunViewListResponse)
async def list_agent_control_run_views(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    rows = list(
        (
            await db.execute(
                select(AgentControlPlaneView)
                .where(AgentControlPlaneView.user_id == current_user.id)
                .order_by(AgentControlPlaneView.is_default.desc(), AgentControlPlaneView.updated_at.desc())
            )
        ).scalars().all()
    )
    return AgentControlRunViewListResponse(
        items=[_serialize_control_run_view(row) for row in rows],
        total=len(rows),
    )


@router.post("/views", response_model=AgentControlRunViewResponse, status_code=status.HTTP_201_CREATED)
async def create_agent_control_run_view(
    request: AgentControlRunViewCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    name = _normalize_string(request.name)
    if not name:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Control plane view name is required")
    if request.is_default:
        await db.execute(
            AgentControlPlaneView.__table__.update()
            .where(AgentControlPlaneView.user_id == current_user.id)
            .values(is_default=False)
        )
    row = AgentControlPlaneView(
        user_id=current_user.id,
        name=name,
        filters=_normalize_control_run_view_filters(request.filters),
        is_default=bool(request.is_default),
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return _serialize_control_run_view(row)


@router.patch("/views/{view_id}", response_model=AgentControlRunViewResponse)
async def update_agent_control_run_view(
    view_id: UUID,
    request: AgentControlRunViewUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    row = (
        await db.execute(
            select(AgentControlPlaneView).where(
                AgentControlPlaneView.id == view_id,
                AgentControlPlaneView.user_id == current_user.id,
            )
        )
    ).scalars().first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Control plane view not found")

    if request.name is not None:
        next_name = _normalize_string(request.name)
        if not next_name:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Control plane view name is required")
        row.name = next_name
    if request.filters is not None:
        row.filters = _normalize_control_run_view_filters(request.filters)
    if request.is_default is not None:
        if bool(request.is_default):
            await db.execute(
                AgentControlPlaneView.__table__.update()
                .where(
                    AgentControlPlaneView.user_id == current_user.id,
                    AgentControlPlaneView.id != row.id,
                )
                .values(is_default=False)
            )
        row.is_default = bool(request.is_default)
    row.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    return _serialize_control_run_view(row)


@router.delete("/views/{view_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent_control_run_view(
    view_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    row = (
        await db.execute(
            select(AgentControlPlaneView).where(
                AgentControlPlaneView.id == view_id,
                AgentControlPlaneView.user_id == current_user.id,
            )
        )
    ).scalars().first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Control plane view not found")
    await db.delete(row)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/reviews/action", response_model=AgentControlRunReviewActionResponse)
async def act_on_agent_control_review(
    payload: AgentControlRunReviewActionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    review_type = _normalize_string(payload.review_type)
    source_kind = _normalize_string(payload.source_kind)
    source_id = _normalize_string(payload.source_id)
    opportunity_id = _normalize_string(payload.opportunity_id)
    action = _normalize_string(payload.action)
    operator_note = _normalize_string(payload.operator_note)
    checkpoint_action_patch = payload.checkpoint_action_patch if isinstance(payload.checkpoint_action_patch, dict) else None

    if not review_type or not source_kind or not source_id or not opportunity_id or not action:
        raise HTTPException(status_code=400, detail="Review action payload is incomplete")

    target: DomainResearchProfile | ResearchPortfolio | None = None
    follow_up_launch_status: Optional[str] = None
    follow_up_operator_decision: Optional[str] = None
    follow_up_job_id: Optional[str] = None
    monitor_job_id: Optional[str] = None
    detail: Optional[str] = None

    if review_type == "follow_up_recommendation":
        target = await _get_control_review_target(
            db=db,
            current_user=current_user,
            source_kind=source_kind,
            source_id=source_id,
        )
        if action not in {"approve_follow_up", "reject_follow_up"}:
            raise HTTPException(status_code=400, detail="Unsupported control-plane review action")
        helper_action = "approve_launch" if action == "approve_follow_up" else "reject_launch"
        if source_kind == "profile":
            helper_response = await _perform_follow_up_queue_action(
                profile=target,
                profile_opportunity_id=opportunity_id,
                action=helper_action,
                operator_note=operator_note,
                db=db,
                current_user=current_user,
            )
        else:
            helper_response = await _perform_follow_up_queue_action(
                portfolio=target,
                portfolio_opportunity_id=opportunity_id,
                action=helper_action,
                operator_note=operator_note,
                db=db,
                current_user=current_user,
            )
        detail = helper_response.detail
        follow_up_launch_status = helper_response.follow_up_launch_status
        follow_up_operator_decision = helper_response.follow_up_operator_decision
        follow_up_job_id = str(helper_response.follow_up_job_id) if helper_response.follow_up_job_id else None
    elif review_type == "manual_follow_up_recommendation":
        target = await _get_control_review_target(
            db=db,
            current_user=current_user,
            source_kind=source_kind,
            source_id=source_id,
        )
        if action not in {"launch_follow_up", "relaunch_follow_up"}:
            raise HTTPException(status_code=400, detail="Unsupported control-plane review action")
        if source_kind == "profile":
            response = await act_on_domain_research_opportunity(
                profile_id=UUID(source_id),
                opportunity_id=opportunity_id,
                payload=ResearchOpportunityActionRequest(action=action, operator_note=operator_note),
                db=db,
                current_user=current_user,
            )
            detail = "Follow-up launched from manual recommendation" if action == "launch_follow_up" else "Follow-up relaunched from manual recommendation"
            if action == "launch_follow_up":
                opportunities = response.latest_summary.get("opportunities") if isinstance(response.latest_summary, dict) and isinstance(response.latest_summary.get("opportunities"), list) else response.latest_summary.get("idea_candidates") if isinstance(response.latest_summary, dict) and isinstance(response.latest_summary.get("idea_candidates"), list) else []
                matched = next((row for row in opportunities if str(row.get("opportunity_id") or "").strip() == opportunity_id), None)
                if isinstance(matched, dict):
                    child_ids = [str(v).strip() for v in (matched.get("child_job_ids") or []) if str(v).strip()]
                    follow_up_job_id = child_ids[-1] if child_ids else None
                follow_up_launch_status = "launched"
                follow_up_operator_decision = "approved_launch"
        else:
            response = await act_on_research_portfolio_opportunity(
                portfolio_id=UUID(source_id),
                opportunity_id=opportunity_id,
                payload=ResearchPortfolioOpportunityActionRequest(action=action, operator_note=operator_note),
                db=db,
                current_user=current_user,
            )
            detail = "Follow-up launched from manual recommendation" if action == "launch_follow_up" else "Follow-up relaunched from manual recommendation"
            if action == "launch_follow_up":
                opportunities = response.opportunities if isinstance(response.opportunities, list) else []
                matched = next((row for row in opportunities if str(row.get("opportunity_id") or "").strip() == opportunity_id), None)
                if isinstance(matched, dict):
                    child_ids = [str(v).strip() for v in (matched.get("child_job_ids") or []) if str(v).strip()]
                    follow_up_job_id = child_ids[-1] if child_ids else None
                follow_up_launch_status = "launched"
                follow_up_operator_decision = "approved_launch"
    elif review_type in {"approval_checkpoint", "job_recovery"}:
        if source_kind != "job":
            raise HTTPException(status_code=400, detail="Job queue actions require source_kind=job")
        if action not in {"approve", "edit", "reject", "skip", "restart", "resume", "cancel"}:
            raise HTTPException(status_code=400, detail="Unsupported control-plane review action")
        if action == "edit" and review_type != "approval_checkpoint":
            raise HTTPException(status_code=400, detail="Only approval checkpoints support edit actions")
        job = await db.get(AgentJob, UUID(source_id))
        if job is None or str(job.user_id) != str(current_user.id):
            raise HTTPException(status_code=404, detail="Review target not found")
        matches, mismatch_reason = _job_matches_bulk_queue_item_type(job, review_type)
        if not matches:
            raise HTTPException(status_code=400, detail=mismatch_reason or "Job does not match this queue item type")
        updated_job = await _perform_job_action(
            job,
            AgentJobActionRequest(
                action=action,
                checkpoint_note=operator_note,
                checkpoint_action_patch=checkpoint_action_patch,
            ),
            db=db,
            current_user=current_user,
        )
        detail = f"{review_type.replace('_', ' ')} action '{action}' applied"
        follow_up_job_id = str(updated_job.id)
    elif review_type == "policy_review":
        target = await _get_control_review_target(
            db=db,
            current_user=current_user,
            source_kind=source_kind,
            source_id=source_id,
        )
        if action != "apply_guardrail":
            raise HTTPException(status_code=400, detail="Unsupported control-plane review action")
        latest_summary = target.latest_summary if isinstance(target.latest_summary, dict) else {}
        queued_review_rows = latest_summary.get("queued_operator_reviews") if isinstance(latest_summary.get("queued_operator_reviews"), list) else []
        opportunity_rows = latest_summary.get("opportunities") if isinstance(latest_summary.get("opportunities"), list) else latest_summary.get("idea_candidates")
        if not isinstance(opportunity_rows, list):
            opportunity_rows = []
        review_row = next(
            (
                row for row in queued_review_rows
                if isinstance(row, dict)
                and _normalize_string(row.get("review_type")) == "policy_review"
                and _normalize_string(row.get("opportunity_id")) == opportunity_id
            ),
            None,
        )
        opportunity_row = next(
            (
                row for row in opportunity_rows
                if isinstance(row, dict) and _normalize_string(row.get("opportunity_id")) == opportunity_id
            ),
            None,
        )
        if review_row is None and opportunity_row is None:
            raise HTTPException(status_code=404, detail="Policy review target not found")
        policy_update_payload = (
            review_row.get("policy_update_payload")
            if isinstance(review_row, dict) and isinstance(review_row.get("policy_update_payload"), dict)
            else opportunity_row.get("policy_update_payload")
            if isinstance(opportunity_row, dict) and isinstance(opportunity_row.get("policy_update_payload"), dict)
            else None
        )
        policy_rollback_payload = (
            review_row.get("policy_rollback_payload")
            if isinstance(review_row, dict) and isinstance(review_row.get("policy_rollback_payload"), dict)
            else opportunity_row.get("policy_rollback_payload")
            if isinstance(opportunity_row, dict) and isinstance(opportunity_row.get("policy_rollback_payload"), dict)
            else None
        )
        monitor_job_id = _normalize_string(
            review_row.get("monitor_job_id") if isinstance(review_row, dict) else None
        ) or _normalize_string(
            opportunity_row.get("monitor_job_id") if isinstance(opportunity_row, dict) else None
        )
        if not monitor_job_id:
            raise HTTPException(status_code=400, detail="Policy review is missing a monitor job id")
        if isinstance(policy_rollback_payload, dict) and _normalize_string(policy_rollback_payload.get("history_entry_id")):
            await rollback_monitor_policy(
                monitor_job_id=monitor_job_id,
                payload=ResearchMonitorPolicyRollbackRequest(
                    history_entry_id=str(policy_rollback_payload.get("history_entry_id")).strip(),
                    change_reason=operator_note or "Applied from control-plane policy safeguard review",
                ),
                current_user=current_user,
                db=db,
            )
            detail = "Policy safeguard rollback applied"
        elif isinstance(policy_update_payload, dict):
            await update_monitor_policy(
                monitor_job_id=monitor_job_id,
                payload=ResearchMonitorPolicyUpdateRequest(
                    automation_profile=_normalize_string(policy_update_payload.get("automation_profile")),
                    automation_policy=policy_update_payload.get("automation_policy") if isinstance(policy_update_payload.get("automation_policy"), dict) else None,
                    mode=_normalize_string(policy_update_payload.get("mode")),
                    allowed_recommendations=policy_update_payload.get("allowed_recommendations") if isinstance(policy_update_payload.get("allowed_recommendations"), list) else None,
                    change_source="policy_guardrail",
                    change_reason=operator_note or "Applied from control-plane policy safeguard review",
                ),
                current_user=current_user,
                db=db,
            )
            detail = "Policy safeguard update applied"
        else:
            raise HTTPException(status_code=400, detail="Policy review is missing an actionable safeguard payload")
    else:
        raise HTTPException(status_code=400, detail="This review type is currently read-only in the control plane")
    await db.commit()
    return AgentControlRunReviewActionResponse(
        ok=True,
        action=action,
        review_type=review_type,
        source_kind=source_kind,
        source_id=source_id,
        opportunity_id=opportunity_id,
        detail=detail,
        monitor_job_id=monitor_job_id,
        follow_up_launch_status=follow_up_launch_status,
        follow_up_operator_decision=follow_up_operator_decision,
        follow_up_job_id=follow_up_job_id,
    )


@router.post("/reviews/bulk-action", response_model=AgentControlRunBulkReviewActionResponse)
async def bulk_act_on_agent_control_reviews(
    payload: AgentControlRunBulkReviewActionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    item_type = _normalize_string(payload.item_type)
    action = _normalize_string(payload.action)
    operator_note = _normalize_string(payload.operator_note)
    if not item_type or not action:
        raise HTTPException(status_code=400, detail="Bulk review action payload is incomplete")

    if item_type in {"approval_checkpoint", "job_recovery"}:
        request = AgentCheckpointQueueBulkActionRequest(
            item_type=item_type,
            action=action,
            job_ids=[str(value).strip() for value in (payload.job_ids or []) if str(value).strip()],
            checkpoint_note=operator_note if item_type == "approval_checkpoint" else None,
        )
        response = await checkpoint_queue_bulk_action(
            request=request,
            db=db,
            current_user=current_user,
        )
        return AgentControlRunBulkReviewActionResponse(
            ok=True,
            item_type=item_type,
            action=action,
            requested_count=response.requested_count,
            applied=response.applied,
            failed=response.failed,
            results=[
                AgentControlRunBulkReviewActionResultResponse(
                    item_key=row.queue_key,
                    job_id=str(row.job_id) if row.job_id else None,
                    ok=bool(row.ok),
                    error=row.error,
                    status=row.status,
                )
                for row in response.results
            ],
        )

    if item_type == "follow_up_recommendation":
        domain_research_profile_id = _normalize_string(payload.domain_research_profile_id)
        portfolio_id = _normalize_string(payload.portfolio_id)
        request = AgentCheckpointQueueBulkFollowUpActionRequest(
            domain_research_profile_id=UUID(domain_research_profile_id) if domain_research_profile_id else None,
            profile_opportunity_ids=[str(value).strip() for value in (payload.profile_opportunity_ids or []) if str(value).strip()],
            portfolio_id=UUID(portfolio_id) if portfolio_id else None,
            portfolio_opportunity_ids=[str(value).strip() for value in (payload.portfolio_opportunity_ids or []) if str(value).strip()],
            action=action,
            operator_note=operator_note,
        )
        response = await checkpoint_queue_bulk_follow_up_action(
            request=request,
            db=db,
            current_user=current_user,
        )
        return AgentControlRunBulkReviewActionResponse(
            ok=True,
            item_type=item_type,
            action=action,
            requested_count=response.requested_count,
            applied=response.applied,
            failed=response.failed,
            results=[
                AgentControlRunBulkReviewActionResultResponse(
                    opportunity_id=row.profile_opportunity_id or row.portfolio_opportunity_id,
                    ok=bool(row.ok),
                    error=row.error,
                    detail=row.detail,
                    follow_up_launch_status=row.follow_up_launch_status,
                    follow_up_operator_decision=row.follow_up_operator_decision,
                    follow_up_job_id=str(row.follow_up_job_id) if row.follow_up_job_id else None,
                )
                for row in response.results
            ],
        )

    raise HTTPException(status_code=400, detail="Bulk actions are not supported for this control-plane item type")


@router.get("/runs", response_model=AgentControlRunListResponse)
async def list_agent_control_runs(
    source_type: Optional[str] = Query(None, description="Filter by control run root type: job|workflow"),
    has_operator_review: Optional[bool] = Query(None),
    review_type: Optional[str] = Query(None),
    review_status: Optional[str] = Query(None),
    limit: int = Query(30, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AgentControlRunListResponse:
    trimmed, _job_linked_job_ids_by_root, _workflow_linked_job_ids_by_root, _review_rows_by_job_id = await _build_control_run_summaries(
        db=db,
        current_user=current_user,
        source_type=source_type,
        has_operator_review=has_operator_review,
        review_type=review_type,
        review_status=review_status,
        limit=limit,
    )
    return AgentControlRunListResponse(items=trimmed, total=len(trimmed))


@router.get("/reviews", response_model=AgentControlRunReviewListResponse)
async def list_agent_control_reviews(
    source_type: Optional[str] = Query(None, description="Filter by control run root type: job|workflow"),
    has_operator_review: Optional[bool] = Query(None),
    review_type: Optional[str] = Query(None),
    review_status: Optional[str] = Query(None),
    queue_status: Optional[str] = Query(None),
    queue_customer: Optional[str] = Query(None),
    queue_sla: Optional[str] = Query(None),
    queue_escalation: Optional[str] = Query(None),
    queue_health_drilldown: Optional[str] = Query(None),
    queue_preset: Optional[str] = Query(None),
    sort: str = Query("priority"),
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AgentControlRunReviewListResponse:
    runs, job_linked_job_ids_by_root, workflow_linked_job_ids_by_root, review_rows_by_job_id = await _build_control_run_summaries(
        db=db,
        current_user=current_user,
        source_type=source_type,
        has_operator_review=has_operator_review,
        review_type=review_type,
        review_status=review_status,
        limit=100,
    )

    flattened: list[AgentControlRunReviewItemResponse] = []
    for run in runs:
        linked_job_ids: set[str] = set()
        if run.source_type == "job" and run.root_job_id:
            linked_job_ids.update(job_linked_job_ids_by_root.get(run.root_job_id, {run.root_job_id}))
        elif run.source_type == "workflow" and run.workflow_execution_id:
            linked_job_ids.update(workflow_linked_job_ids_by_root.get(run.workflow_execution_id, set()))
        for linked_job_id in sorted(linked_job_ids):
            for review in review_rows_by_job_id.get(linked_job_id, []):
                review_copy = review.model_copy(update={
                    "run_id": run.id,
                    "run_title": run.title,
                    "run_source_type": run.source_type,
                    "run_status": run.status,
                })
                if _review_matches_filters(
                    review=review_copy,
                    review_type=review_type,
                    review_status=review_status,
                    queue_status=queue_status,
                    queue_customer=queue_customer,
                    queue_sla=queue_sla,
                    queue_escalation=queue_escalation,
                    queue_health_drilldown=queue_health_drilldown,
                    queue_preset=queue_preset,
                ):
                    flattened.append(review_copy)

    sort_mode = _normalize_review_sort(sort)
    sorted_items = _sort_control_reviews(flattened, sort_mode=sort_mode)
    trimmed = sorted_items[offset : offset + limit]
    return AgentControlRunReviewListResponse(
        items=trimmed,
        total=len(sorted_items),
        summary=_build_control_review_summary(sorted_items),
        offset=offset,
        limit=limit,
        has_more=offset + len(trimmed) < len(sorted_items),
    )


@router.get("/runs/{run_id}", response_model=AgentControlRunDetail)
async def get_agent_control_run_detail(
    run_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AgentControlRunDetail:
    source_type, source_uuid = _parse_run_id(run_id)

    if source_type == "job":
        root_job = (
            await db.execute(
                select(AgentJob).where(
                    AgentJob.id == source_uuid,
                    AgentJob.user_id == current_user.id,
                )
            )
        ).scalar_one_or_none()
        if not root_job:
            raise HTTPException(status_code=404, detail="Agent control run not found")

        job_rows = list(
            (
                await db.execute(
                    select(AgentJob)
                    .where(
                        AgentJob.user_id == current_user.id,
                        or_(
                            AgentJob.id == root_job.id,
                            AgentJob.root_job_id == root_job.id,
                            AgentJob.parent_job_id == root_job.id,
                        ),
                    )
                    .order_by(AgentJob.created_at.asc())
                )
            ).scalars().all()
        )
        job_ids = {str(row.id) for row in job_rows}
        event_rows = list(
            (
                await db.execute(
                    select(AutonomyDecisionEvent)
                    .where(
                        AutonomyDecisionEvent.user_id == current_user.id,
                        AutonomyDecisionEvent.source_id.in_(job_ids),
                    )
                    .order_by(AutonomyDecisionEvent.event_time.desc())
                    .limit(80)
                )
            ).scalars().all()
        )
        decision_trace = [_build_event_response(row) for row in event_rows]
        linkage = {
            "note_ids": set(),
            "plan_ids": set(),
            "run_ids": set(),
            "workflow_execution_ids": set(),
            "synthesis_job_ids": set(),
        }
        routing_lineage = _collect_routing_lineage(
            [root_job.config, root_job.results, root_job.output_artifacts]
        )
        for row in job_rows:
            row_linkage = _collect_job_linkage(row)
            for key in linkage:
                linkage[key].update(row_linkage.get(key, set()))
            row_routing_lineage = _collect_routing_lineage([row.config, row.results, row.output_artifacts])
            routing_lineage["experiment_ids"].update(row_routing_lineage["experiment_ids"])
            routing_lineage["variant_ids"].update(row_routing_lineage["variant_ids"])
        event_linkage = _collect_event_linkage(decision_trace)
        linkage["note_ids"].update(event_linkage["note_ids"])
        linkage["plan_ids"].update(event_linkage["plan_ids"])
        linkage["run_ids"].update(event_linkage["run_ids"])
        linkage["synthesis_job_ids"].update(event_linkage["synthesis_job_ids"])
        job_ids.update(event_linkage["child_job_ids"])
        event_routing_lineage = _collect_routing_lineage([event.metadata for event in decision_trace])
        routing_lineage["experiment_ids"].update(event_routing_lineage["experiment_ids"])
        routing_lineage["variant_ids"].update(event_routing_lineage["variant_ids"])

        workflow_ids: set[UUID] = set()
        for raw_id in linkage["workflow_execution_ids"]:
            try:
                workflow_ids.add(UUID(raw_id))
            except ValueError:
                continue
        workflow_rows = []
        if workflow_ids:
            workflow_rows = list(
                (
                    await db.execute(
                        select(WorkflowExecution)
                        .options(selectinload(WorkflowExecution.workflow))
                        .where(
                            WorkflowExecution.user_id == current_user.id,
                            WorkflowExecution.id.in_(workflow_ids),
                        )
                    )
                ).scalars().all()
            )
        routing = _derive_routing_summary(events=decision_trace, root_metadata=root_job.config if isinstance(root_job.config, dict) else {})
        review_rows_by_job_id, _review_counts_by_job_id = await _collect_control_run_review_mappings(
            db=db,
            current_user=current_user,
            job_ids=job_ids,
        )
        queued_operator_reviews: list[AgentControlRunReviewItemResponse] = []
        for linked_job_id in sorted(job_ids):
            queued_operator_reviews.extend(review_rows_by_job_id.get(linked_job_id, []))
        summary = _build_job_summary(
            job=root_job,
            child_job_count=max(0, len(job_rows) - 1),
            linked_note_count=len(linkage["note_ids"]),
            linked_experiment_count=len(linkage["plan_ids"]) + len(linkage["run_ids"]),
            decision_count=len(decision_trace),
            routing=routing,
        )
        summary.queued_operator_review_count = len(queued_operator_reviews)
        summary.queued_operator_reviews_by_type = dict(Counter(item.review_type for item in queued_operator_reviews if _normalize_string(item.review_type)))
        replay = _build_replay_summary(
            source_type="job",
            title=root_job.name,
            status=root_job.status,
            current_phase=root_job.current_phase,
            routing=routing,
            ended_at=root_job.completed_at,
            child_count=max(0, len(job_rows) - 1),
            decision_count=len(decision_trace),
        )
        memory_graph = await _slice_memory_graph_for_job_ids(db=db, current_user=current_user, job_ids=job_ids)

        nodes: list[AgentControlRunNode] = [
            AgentControlRunNode(
                id=f"job:{row.id}",
                kind="agent_job",
                label=row.name,
                status=row.status,
                stage="planner" if row.id == root_job.id else "executor",
                timestamp=row.created_at,
                metadata={
                    "agent_job_id": str(row.id),
                    "goal": row.goal,
                    "job_type": row.job_type,
                    "current_phase": row.current_phase,
                },
            )
            for row in job_rows
        ]
        edges: list[AgentControlRunEdge] = []
        for row in job_rows:
            if row.parent_job_id:
                edges.append(
                    AgentControlRunEdge(
                        source=f"job:{row.parent_job_id}",
                        target=f"job:{row.id}",
                        relation="delegates_to",
                    )
                )
        for workflow_row in workflow_rows:
            nodes.append(
                AgentControlRunNode(
                    id=f"workflow:{workflow_row.id}",
                    kind="workflow_execution",
                    label=workflow_row.workflow.name if workflow_row.workflow else f"Workflow {workflow_row.workflow_id}",
                    status=workflow_row.status,
                    stage="executor",
                    timestamp=workflow_row.created_at,
                    metadata={
                        "workflow_execution_id": str(workflow_row.id),
                        "workflow_id": str(workflow_row.workflow_id),
                        "trigger_type": workflow_row.trigger_type,
                    },
                )
            )
            edges.append(
                AgentControlRunEdge(
                    source=f"job:{root_job.id}",
                    target=f"workflow:{workflow_row.id}",
                    relation="executes_workflow",
                )
            )
        for note_id in sorted(linkage["note_ids"]):
            nodes.append(
                AgentControlRunNode(
                    id=f"note:{note_id}",
                    kind="research_note",
                    label=f"Research note {note_id[:8]}",
                    stage="planner",
                    metadata={"research_note_id": note_id, "note_id": note_id},
                )
            )
            edges.append(AgentControlRunEdge(source=f"job:{root_job.id}", target=f"note:{note_id}", relation="references_note"))
        for plan_id in sorted(linkage["plan_ids"]):
            nodes.append(
                AgentControlRunNode(
                    id=f"experiment-plan:{plan_id}",
                    kind="experiment_plan",
                    label=f"Experiment plan {plan_id[:8]}",
                    stage="executor",
                    metadata={"experiment_plan_id": plan_id, "plan_id": plan_id},
                )
            )
            edges.append(
                AgentControlRunEdge(
                    source=f"job:{root_job.id}",
                    target=f"experiment-plan:{plan_id}",
                    relation="materializes_plan",
                )
            )
        for validation_run_id in sorted(linkage["run_ids"]):
            nodes.append(
                AgentControlRunNode(
                    id=f"experiment-run:{validation_run_id}",
                    kind="experiment_run",
                    label=f"Validation run {validation_run_id[:8]}",
                    stage="executor",
                    metadata={"experiment_run_id": validation_run_id, "run_id": validation_run_id},
                )
            )
            edges.append(
                AgentControlRunEdge(
                    source=f"job:{root_job.id}",
                    target=f"experiment-run:{validation_run_id}",
                    relation="launches_validation",
                )
            )
        for synthesis_job_id in sorted(linkage["synthesis_job_ids"]):
            nodes.append(
                AgentControlRunNode(
                    id=f"synthesis:{synthesis_job_id}",
                    kind="synthesis_job",
                    label=f"Synthesis job {synthesis_job_id[:8]}",
                    stage="operator_review",
                    metadata={"synthesis_job_id": synthesis_job_id},
                )
            )
            edges.append(
                AgentControlRunEdge(
                    source=f"job:{root_job.id}",
                    target=f"synthesis:{synthesis_job_id}",
                    relation="queues_synthesis",
                )
            )
        for index, review in enumerate(queued_operator_reviews):
            review_node_id = f"review:{root_job.id}:{index}"
            nodes.append(
                AgentControlRunNode(
                    id=review_node_id,
                    kind="operator_review",
                    label=review.title or review.reason_label or review.review_type or "Operator review",
                    status=review.review_status,
                    stage="operator_review",
                    metadata={
                        "review_type": review.review_type,
                        "review_status": review.review_status,
                        "reason_code": review.reason_code,
                        "reason_label": review.reason_label,
                        "source_kind": review.source_kind,
                        "source_id": review.source_id,
                        "opportunity_id": review.opportunity_id,
                        "action_path": review.action_path,
                        "queue_path": review.queue_path,
                        "note_path": review.note_path,
                        "synthesis_path": review.synthesis_path,
                        "job_id": review.job_id,
                        "item_type": review.item_type,
                    },
                )
            )
            edges.append(
                AgentControlRunEdge(
                    source=f"job:{root_job.id}",
                    target=review_node_id,
                    relation="queues_operator_review",
                )
            )
        for event in decision_trace[:30]:
            event_metadata = event.metadata if isinstance(event.metadata, dict) else {}
            nodes.append(
                AgentControlRunNode(
                    id=f"event:{event.event_id}",
                    kind="decision_event",
                    label=event.summary,
                    status=event.status,
                    stage="router" if event.actor_mode == "autonomous" else "planner",
                    timestamp=event.event_time,
                    metadata={
                        "decision_type": event.decision_type,
                        "source_id": event.source_id,
                        "source_kind": event.source_kind,
                        "research_note_ids": event.linked_note_ids or [],
                        "experiment_plan_ids": event.linked_experiment_plan_ids or [],
                        "experiment_run_ids": event.linked_validation_run_ids or [],
                        "synthesis_job_id": _normalize_string(
                            event_metadata.get("reevaluation_job_id")
                            or event_metadata.get("review_job_id")
                            or event_metadata.get("synthesis_job_id")
                            or event_metadata.get("source_synthesis_job_id")
                        ),
                        "routing_experiment_id": _normalize_string(
                            event_metadata.get("routing_experiment_id") or event_metadata.get("experiment_id")
                        ),
                        "routing_experiment_variant_id": _normalize_string(
                            event_metadata.get("routing_experiment_variant_id") or event_metadata.get("variant_id")
                        ),
                    },
                )
            )
            if event.source_id:
                edges.append(
                    AgentControlRunEdge(
                        source=f"job:{event.source_id}",
                        target=f"event:{event.event_id}",
                        relation="emits_decision",
                    )
                )

        return AgentControlRunDetail(
            run=summary,
            nodes=nodes,
            edges=edges,
            decision_trace=decision_trace,
            memory_graph=memory_graph,
            routing=routing,
            replay=replay,
            related_links=_build_related_links(
                source_type="job",
                root_job_id=str(root_job.id),
                workflow_execution_id=str(workflow_rows[0].id) if workflow_rows else None,
                note_ids=sorted(linkage["note_ids"]),
                plan_ids=sorted(linkage["plan_ids"]),
                run_ids=sorted(linkage["run_ids"]),
                synthesis_job_ids=sorted(linkage["synthesis_job_ids"]),
                routing=routing,
                routing_experiment_ids=sorted(routing_lineage["experiment_ids"]),
                routing_variant_ids=sorted(routing_lineage["variant_ids"]),
            ),
            queued_operator_review_count=len(queued_operator_reviews),
            queued_operator_reviews=queued_operator_reviews,
            policy_summary={
                "automation_profile": _normalize_string((root_job.config or {}).get("automation_profile"))
                if isinstance(root_job.config, dict)
                else None,
                "effective_policy": (root_job.config or {}).get("effective_policy")
                if isinstance(root_job.config, dict)
                else None,
            },
            metadata={
                "root_job_id": str(root_job.id),
                "workflow_execution_ids": sorted(linkage["workflow_execution_ids"]),
                "linked_note_ids": sorted(linkage["note_ids"]),
                "linked_experiment_plan_ids": sorted(linkage["plan_ids"]),
                "linked_validation_run_ids": sorted(linkage["run_ids"]),
            },
        )

    execution = (
        await db.execute(
            select(WorkflowExecution)
            .options(selectinload(WorkflowExecution.workflow), selectinload(WorkflowExecution.node_executions))
            .where(
                WorkflowExecution.id == source_uuid,
                WorkflowExecution.user_id == current_user.id,
            )
        )
    ).scalar_one_or_none()
    if not execution:
        raise HTTPException(status_code=404, detail="Agent control run not found")

    execution_rows = list(
        (
            await db.execute(
                select(WorkflowExecution)
                .options(selectinload(WorkflowExecution.workflow))
                .where(
                    WorkflowExecution.user_id == current_user.id,
                    or_(
                        WorkflowExecution.id == execution.id,
                        WorkflowExecution.parent_execution_id == execution.id,
                    ),
                )
                .order_by(WorkflowExecution.created_at.asc())
            )
        ).scalars().all()
    )
    workflow_source_ids = {str(row.id) for row in execution_rows}
    event_rows = list(
        (
            await db.execute(
                select(AutonomyDecisionEvent)
                .where(
                    AutonomyDecisionEvent.user_id == current_user.id,
                    AutonomyDecisionEvent.source_id.in_(workflow_source_ids),
                )
                .order_by(AutonomyDecisionEvent.event_time.desc())
                .limit(80)
            )
        ).scalars().all()
    )
    decision_trace = [_build_event_response(row) for row in event_rows]

    inferred_job_ids = _collect_workflow_inferred_job_ids(execution)
    job_rows = []
    if inferred_job_ids:
        valid_job_uuids: list[UUID] = []
        for raw_id in inferred_job_ids:
            try:
                valid_job_uuids.append(UUID(raw_id))
            except ValueError:
                continue
        if valid_job_uuids:
            job_rows = list(
                (
                    await db.execute(
                        select(AgentJob).where(
                            AgentJob.user_id == current_user.id,
                            AgentJob.id.in_(valid_job_uuids),
                        )
                    )
                ).scalars().all()
            )
    job_ids = {str(job.id) for job in job_rows}
    memory_graph = await _slice_memory_graph_for_job_ids(db=db, current_user=current_user, job_ids=job_ids)
    event_linkage = _collect_event_linkage(decision_trace)
    routing_lineage = _collect_routing_lineage([execution.context, execution.trigger_data, *(event.metadata for event in decision_trace)])
    routing = _derive_routing_summary(events=decision_trace, root_metadata=execution.context if isinstance(execution.context, dict) else {})
    review_rows_by_job_id, _review_counts_by_job_id = await _collect_control_run_review_mappings(
        db=db,
        current_user=current_user,
        job_ids=job_ids,
    )
    queued_operator_reviews: list[AgentControlRunReviewItemResponse] = []
    for linked_job_id in sorted(job_ids):
        queued_operator_reviews.extend(review_rows_by_job_id.get(linked_job_id, []))
    summary = _build_workflow_summary(
        execution=execution,
        child_execution_count=max(0, len(execution_rows) - 1),
        child_job_count=len(job_rows),
        linked_note_count=len(event_linkage["note_ids"]),
        linked_experiment_count=len(event_linkage["plan_ids"]) + len(event_linkage["run_ids"]),
        decision_count=len(decision_trace),
        routing=routing,
    )
    summary.queued_operator_review_count = len(queued_operator_reviews)
    summary.queued_operator_reviews_by_type = dict(Counter(item.review_type for item in queued_operator_reviews if _normalize_string(item.review_type)))
    replay = _build_replay_summary(
        source_type="workflow",
        title=execution.workflow.name if execution.workflow else f"Workflow {execution.workflow_id}",
        status=execution.status,
        current_phase=execution.current_node_id,
        routing=routing,
        ended_at=execution.completed_at,
        child_count=max(0, len(execution_rows) - 1) + len(job_rows),
        decision_count=len(decision_trace),
    )

    nodes = [
        AgentControlRunNode(
            id=f"workflow:{row.id}",
            kind="workflow_execution",
            label=row.workflow.name if row.workflow else f"Workflow {row.workflow_id}",
            status=row.status,
            stage="executor" if row.id != execution.id else "planner",
            timestamp=row.created_at,
            metadata={
                "workflow_execution_id": str(row.id),
                "workflow_id": str(row.workflow_id),
                "trigger_type": row.trigger_type,
                "current_node_id": row.current_node_id,
            },
        )
        for row in execution_rows
    ]
    edges: list[AgentControlRunEdge] = []
    for row in execution_rows:
        if row.parent_execution_id:
            edges.append(
                AgentControlRunEdge(
                    source=f"workflow:{row.parent_execution_id}",
                    target=f"workflow:{row.id}",
                    relation="spawns_subworkflow",
                )
            )
    for job in job_rows:
        nodes.append(
            AgentControlRunNode(
                id=f"job:{job.id}",
                kind="agent_job",
                label=job.name,
                status=job.status,
                stage="executor",
                timestamp=job.created_at,
                metadata={"agent_job_id": str(job.id), "goal": job.goal},
            )
        )
        edges.append(
            AgentControlRunEdge(
                source=f"workflow:{execution.id}",
                target=f"job:{job.id}",
                relation="invokes_job",
            )
        )
    for note_id in sorted(event_linkage["note_ids"]):
        nodes.append(
            AgentControlRunNode(
                id=f"note:{note_id}",
                kind="research_note",
                label=f"Research note {note_id[:8]}",
                stage="planner",
                metadata={"research_note_id": note_id, "note_id": note_id},
            )
        )
        edges.append(
            AgentControlRunEdge(
                source=f"workflow:{execution.id}",
                target=f"note:{note_id}",
                relation="references_note",
            )
        )
    for synthesis_job_id in sorted(event_linkage["synthesis_job_ids"]):
        nodes.append(
            AgentControlRunNode(
                id=f"synthesis:{synthesis_job_id}",
                kind="synthesis_job",
                label=f"Synthesis job {synthesis_job_id[:8]}",
                stage="operator_review",
                metadata={"synthesis_job_id": synthesis_job_id},
            )
        )
        edges.append(
            AgentControlRunEdge(
                source=f"workflow:{execution.id}",
                target=f"synthesis:{synthesis_job_id}",
                relation="queues_synthesis",
            )
        )
    for index, review in enumerate(queued_operator_reviews):
        review_node_id = f"review:{execution.id}:{index}"
        nodes.append(
            AgentControlRunNode(
                id=review_node_id,
                kind="operator_review",
                label=review.title or review.reason_label or review.review_type or "Operator review",
                status=review.review_status,
                stage="operator_review",
                metadata={
                    "review_type": review.review_type,
                    "review_status": review.review_status,
                    "reason_code": review.reason_code,
                    "reason_label": review.reason_label,
                    "source_kind": review.source_kind,
                    "source_id": review.source_id,
                    "opportunity_id": review.opportunity_id,
                    "action_path": review.action_path,
                    "queue_path": review.queue_path,
                    "note_path": review.note_path,
                    "synthesis_path": review.synthesis_path,
                    "job_id": review.job_id,
                    "item_type": review.item_type,
                },
            )
        )
        edges.append(
            AgentControlRunEdge(
                source=f"workflow:{execution.id}",
                target=review_node_id,
                relation="queues_operator_review",
            )
        )

    return AgentControlRunDetail(
        run=summary,
        nodes=nodes,
        edges=edges,
        decision_trace=decision_trace,
        memory_graph=memory_graph,
        routing=routing,
        replay=replay,
        related_links=_build_related_links(
            source_type="workflow",
            root_job_id=str(job_rows[0].id) if job_rows else None,
            workflow_execution_id=str(execution.id),
            note_ids=sorted(event_linkage["note_ids"]),
            plan_ids=sorted(event_linkage["plan_ids"]),
            run_ids=sorted(event_linkage["run_ids"]),
            synthesis_job_ids=sorted(event_linkage["synthesis_job_ids"]),
            routing=routing,
            routing_experiment_ids=sorted(routing_lineage["experiment_ids"]),
            routing_variant_ids=sorted(routing_lineage["variant_ids"]),
        ),
        queued_operator_review_count=len(queued_operator_reviews),
        queued_operator_reviews=queued_operator_reviews,
        metadata={
            "workflow_execution_id": str(execution.id),
            "linked_job_ids": sorted(job_ids),
            "linked_note_ids": sorted(event_linkage["note_ids"]),
            "linked_experiment_plan_ids": sorted(event_linkage["plan_ids"]),
            "linked_validation_run_ids": sorted(event_linkage["run_ids"]),
        },
    )
