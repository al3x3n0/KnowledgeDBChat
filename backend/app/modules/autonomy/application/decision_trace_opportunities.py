"""Decision-state policy for opportunity trace projection."""

from dataclasses import dataclass
from typing import Any, Callable

from app.schemas.agent_job import AgentDecisionTraceEventResponse


@dataclass(frozen=True)
class OpportunityEventState:
    event_type: str
    decision_type: str
    status: str | None
    actor_mode: str
    severity: str


@dataclass(frozen=True)
class OpportunityDecisionTraceDependencies:
    parse_time: Callable[..., Any]
    reason_label: Callable[..., Any]
    build_event: Callable[..., AgentDecisionTraceEventResponse]
    build_operator_context: Callable[..., dict[str, Any]]


def bind_opportunity_decision_trace(
    *,
    deps: OpportunityDecisionTraceDependencies,
) -> Callable[..., list[AgentDecisionTraceEventResponse]]:
    """Bind projection dependencies once and return the endpoint-facing builder."""

    def build(**kwargs: Any) -> list[AgentDecisionTraceEventResponse]:
        return build_opportunity_decision_trace(
            **kwargs,
            deps=deps,
        )

    return build


def classify_opportunity_event(row: dict) -> OpportunityEventState:
    autonomy = str(row.get("autonomy_state") or "").strip().lower()
    review = str(row.get("follow_up_review_status") or "").strip().lower()
    outcome = str(row.get("follow_up_outcome_status") or "").strip().lower()
    decision = str(row.get("last_decision_type") or "").strip().lower()
    event_type = decision or "opportunity_updated"
    status = autonomy or str(row.get("stage") or "").strip().lower() or None
    if outcome in {"completed", "failed", "cancelled"}:
        event_type = f"follow_up_{outcome}"
        status = outcome
    elif review == "approved_launch":
        event_type = "follow_up_approved"
    elif review == "rejected":
        event_type = "follow_up_rejected"
    elif autonomy == "cooldown":
        event_type = "opportunity_cooldown"
    elif autonomy == "blocked_structural":
        event_type = "opportunity_blocked"
    elif autonomy == "completed_waiting_change":
        event_type = "opportunity_completed_waiting_change"
    elif review == "pending_approval":
        event_type = "follow_up_queued"
    elif autonomy == "active":
        event_type = "follow_up_launched"
    return OpportunityEventState(
        event_type=event_type,
        decision_type=event_type,
        status=status,
        actor_mode=(
            "operator" if review in {"approved_launch", "rejected"} else "autonomous"
        ),
        severity=(
            "high"
            if autonomy == "blocked_structural"
            else "medium"
            if autonomy == "cooldown"
            else "normal"
        ),
    )


def build_opportunity_decision_trace(
    *,
    source_kind: str,
    source_id: str,
    source_label: str,
    customer: str | None,
    opportunities: list[dict[str, Any]],
    deep_link_params: dict[str, str],
    deps: OpportunityDecisionTraceDependencies,
    domain: str | None = None,
    objective: str | None = None,
    track_type: str | None = None,
    source_scope: str | None = None,
    repo_source_ids: Any = None,
    benchmark_queries: Any = None,
    sandbox_profile_id: str | None = None,
    automation_profile: str | None = None,
    effective_policy: dict[str, Any] | None = None,
) -> list[AgentDecisionTraceEventResponse]:
    events: list[AgentDecisionTraceEventResponse] = []
    for row in opportunities:
        event_time = deps.parse_time(
            row.get("follow_up_outcome_recorded_at")
            or row.get("follow_up_launched_at")
            or row.get("follow_up_reviewed_at")
            or row.get("last_material_change_at")
            or row.get("last_evaluated_at")
            or row.get("updated_at")
        )
        if event_time is None:
            continue
        autonomy_state = str(row.get("autonomy_state") or "").strip().lower()
        review_status = str(row.get("follow_up_review_status") or "").strip().lower()
        outcome_status = str(row.get("follow_up_outcome_status") or "").strip().lower()
        reason_code = (
            str(
                row.get("last_decision_reason_code")
                or row.get("last_blocked_reason_code")
                or row.get("last_skip_reason_code")
                or ""
            ).strip()
            or None
        )
        event_state = classify_opportunity_event(row)
        title = str(
            row.get("title")
            or row.get("canonical_key")
            or row.get("opportunity_id")
            or "Opportunity"
        ).strip()
        summary = (
            f"{source_label}: {title} is " f"{event_state.event_type.replace('_', ' ')}"
        )
        outcome_summary = str(row.get("follow_up_outcome_summary") or "").strip()
        if outcome_status and outcome_summary:
            summary = f"{summary} - {outcome_summary}"
        link_params = dict(deep_link_params)
        opportunity_id = str(row.get("opportunity_id") or "").strip()
        if opportunity_id:
            link_params["opportunityId"] = opportunity_id
        if source_kind == "domain_profile" and source_id:
            link_params.setdefault("profileId", source_id)
        if source_kind == "portfolio" and source_id:
            link_params.setdefault("fleetId", source_id)
        events.append(
            deps.build_event(
                event_type=event_state.event_type,
                event_time=event_time,
                source_kind=source_kind,
                source_id=source_id,
                source_label=source_label,
                customer=customer,
                decision_type=event_state.decision_type,
                reason_code=reason_code,
                reason_label=deps.reason_label(reason_code),
                status=event_state.status,
                severity=event_state.severity,
                actor_mode=event_state.actor_mode,
                summary=summary,
                operator_note=str(
                    row.get("follow_up_review_note") or row.get("operator_note") or ""
                ).strip()
                or None,
                before_state=None,
                after_state={
                    "autonomy_state": autonomy_state or None,
                    "stage": row.get("stage"),
                    "review_status": review_status or None,
                    "follow_up_outcome_status": outcome_status or None,
                    "follow_up_last_job_id": row.get("follow_up_last_job_id"),
                },
                deep_link={
                    "target_tab": deep_link_params.get("tab") or source_kind,
                    "params": link_params,
                    "label": f"Open {source_label}",
                },
                metadata={
                    "opportunity_id": row.get("opportunity_id"),
                    "canonical_key": row.get("canonical_key"),
                    "evidence_revision": row.get("evidence_revision"),
                    "follow_up_outcome_summary": row.get("follow_up_outcome_summary"),
                },
                suffix=str(row.get("opportunity_id") or row.get("canonical_key") or ""),
                operator_context=deps.build_operator_context(
                    objective=str(row.get("objective") or objective or "").strip()
                    or None,
                    domain=str(row.get("domain") or domain or "").strip() or None,
                    track_type=str(row.get("track_type") or track_type or "").strip()
                    or None,
                    source_scope=str(
                        row.get("source_scope") or source_scope or ""
                    ).strip()
                    or None,
                    repo_source_ids=row.get("repo_source_ids") or repo_source_ids,
                    benchmark_queries=(
                        row.get("benchmark_queries") or benchmark_queries
                    ),
                    sandbox_profile_id=str(
                        row.get("sandbox_profile_id") or sandbox_profile_id or ""
                    ).strip()
                    or None,
                    automation_profile=str(
                        row.get("automation_profile") or automation_profile or ""
                    ).strip()
                    or None,
                    effective_policy=(
                        row.get("effective_policy")
                        if isinstance(row.get("effective_policy"), dict)
                        else effective_policy
                    ),
                    confidence=row.get("confidence"),
                    readiness=row.get("readiness"),
                    linked_note_ids=row.get("linked_note_ids"),
                    linked_experiment_plan_ids=row.get("linked_experiment_plan_ids"),
                    linked_validation_run_ids=row.get("linked_validation_run_ids"),
                    child_job_ids=row.get("child_job_ids"),
                ),
            )
        )
    return events
