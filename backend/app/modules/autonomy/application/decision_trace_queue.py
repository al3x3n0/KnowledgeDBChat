"""Project checkpoint queue items into derived decision-trace events."""

from dataclasses import dataclass
from typing import Any, Callable

from app.schemas.agent_job import (
    AgentCheckpointQueueItemResponse,
    AgentDecisionTraceEventResponse,
)

QUEUE_EVENT_TYPES = {
    "follow_up_recommendation": "follow_up_queued",
    "policy_review": "policy_guardrail_triggered",
    "budget_review": "budget_clamped",
    "job_recovery": "job_recovery_queued",
    "approval_checkpoint": "approval_required",
}


@dataclass(frozen=True)
class QueueDecisionTraceDependencies:
    build_event: Callable[..., AgentDecisionTraceEventResponse]
    build_operator_context: Callable[..., dict[str, Any]]


def build_queue_decision_trace(
    items: list[AgentCheckpointQueueItemResponse],
    *,
    deps: QueueDecisionTraceDependencies,
) -> list[AgentDecisionTraceEventResponse]:
    """Build decision events for actionable checkpoint queue items."""
    events: list[AgentDecisionTraceEventResponse] = []
    for item in items:
        event_time = item.created_at or item.backoff_until or item.next_run_at
        if event_time is None:
            continue
        source_kind, source_id, source_label = _source_fields(item)
        target_tab, params = _deep_link_fields(item)
        event_type = QUEUE_EVENT_TYPES.get(item.item_type, "queue_item_open")
        events.append(
            deps.build_event(
                event_type=event_type,
                event_time=event_time,
                source_kind=source_kind,
                source_id=source_id,
                source_label=source_label,
                customer=item.customer,
                decision_type=event_type,
                reason_code=item.reason_code,
                reason_label=item.reason_label,
                scheduler_state=item.scheduler_state,
                status=item.status,
                severity=item.escalation_level,
                actor_mode="autonomous",
                summary=item.summary or item.title,
                operator_note=item.follow_up_operator_note,
                deep_link={
                    "target_tab": target_tab,
                    "job_id": item.job_id,
                    "params": params,
                    "label": f"Open {source_label or 'source'}",
                },
                metadata={
                    "queue_key": item.queue_key,
                    "item_type": item.item_type,
                    "reason_label": item.reason_label,
                    "scheduler_state": item.scheduler_state,
                    "profile_opportunity_id": item.profile_opportunity_id,
                    "portfolio_opportunity_id": item.portfolio_opportunity_id,
                },
                suffix=item.queue_key,
                operator_context=deps.build_operator_context(
                    objective=item.objective,
                    domain=item.domain,
                    track_type=item.track_type,
                    source_scope=item.source_scope,
                    repo_source_ids=item.repo_source_ids,
                    benchmark_queries=item.benchmark_queries,
                    sandbox_profile_id=item.sandbox_profile_id,
                    automation_profile=item.automation_profile,
                    effective_policy=item.effective_policy,
                    confidence=item.confidence,
                    readiness=item.readiness,
                    linked_note_ids=item.linked_note_ids,
                    linked_experiment_plan_ids=item.linked_experiment_plan_ids,
                    linked_validation_run_ids=item.linked_validation_run_ids,
                    child_job_ids=item.child_job_ids,
                ),
            )
        )
    return events


def _source_fields(
    item: AgentCheckpointQueueItemResponse,
) -> tuple[str, str, str | None]:
    source_kind = (
        "portfolio"
        if item.portfolio_id
        else "domain_profile"
        if item.domain_research_profile_id
        else "queue"
    )
    source_id = (
        str(item.portfolio_id)
        if item.portfolio_id
        else str(item.domain_research_profile_id)
        if item.domain_research_profile_id
        else str(item.job_id or item.inbox_item_id or item.queue_key)
    )
    source_label = (
        item.portfolio_title
        or item.domain_research_profile_title
        or item.job_name
        or item.title
    )
    return source_kind, source_id, source_label


def _deep_link_fields(
    item: AgentCheckpointQueueItemResponse,
) -> tuple[str, dict[str, str]]:
    params: dict[str, str] = {}
    target_tab = "queue"
    if item.portfolio_id:
        target_tab = "fleet"
        params["fleetId"] = str(item.portfolio_id)
        if item.portfolio_opportunity_id:
            params["opportunityId"] = str(item.portfolio_opportunity_id)
    elif item.domain_research_profile_id:
        target_tab = "domain"
        params["profileId"] = str(item.domain_research_profile_id)
        if item.profile_opportunity_id:
            params["opportunityId"] = str(item.profile_opportunity_id)
    elif item.customer:
        params["health_customer"] = str(item.customer)
    if item.job_id:
        params["job"] = str(item.job_id)
    return target_tab, params
