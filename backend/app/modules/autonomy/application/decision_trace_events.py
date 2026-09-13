"""Build normalized decision-trace event response contracts."""

import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from app.schemas.agent_job import (
    AgentDecisionTraceDeepLinkResponse,
    AgentDecisionTraceEventResponse,
)


@dataclass(frozen=True)
class DecisionTraceEventDependencies:
    build_operator_context: Callable[..., dict[str, Any]]


def decision_trace_event_id(
    source_kind: str,
    source_id: str | None,
    decision_type: str,
    event_time: datetime,
    suffix: str | None = None,
) -> str:
    """Build a stable identifier for a derived decision event."""
    raw = "|".join(
        [
            str(source_kind or "").strip(),
            str(source_id or "").strip(),
            str(decision_type or "").strip(),
            event_time.isoformat(),
            str(suffix or "").strip(),
        ]
    )
    return uuid.uuid5(uuid.NAMESPACE_URL, raw).hex


def build_decision_trace_event(
    *,
    event_type: str,
    event_time: datetime,
    source_kind: str,
    source_id: str | None,
    source_label: str | None,
    decision_type: str,
    summary: str,
    deps: DecisionTraceEventDependencies,
    customer: str | None = None,
    reason_code: str | None = None,
    reason_label: str | None = None,
    scheduler_state: dict[str, Any] | None = None,
    status: str | None = None,
    severity: str | None = None,
    actor_mode: str | None = None,
    operator_note: str | None = None,
    before_state: dict[str, Any] | None = None,
    after_state: dict[str, Any] | None = None,
    deep_link: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    is_derived: bool = False,
    record_origin: str | None = None,
    suffix: str | None = None,
    operator_context: dict[str, Any] | None = None,
) -> AgentDecisionTraceEventResponse:
    """Normalize source data into the public decision-trace event schema."""
    normalized_context = (
        deps.build_operator_context(
            objective=(operator_context or {}).get("objective"),
            domain=(operator_context or {}).get("domain"),
            track_type=(operator_context or {}).get("track_type"),
            source_scope=(operator_context or {}).get("source_scope"),
            repo_source_ids=(operator_context or {}).get("repo_source_ids"),
            benchmark_queries=(operator_context or {}).get("benchmark_queries"),
            sandbox_profile_id=(operator_context or {}).get("sandbox_profile_id"),
            automation_profile=(operator_context or {}).get("automation_profile"),
            effective_policy=(operator_context or {}).get("effective_policy"),
            confidence=(operator_context or {}).get("confidence"),
            readiness=(operator_context or {}).get("readiness"),
            linked_note_ids=(operator_context or {}).get("linked_note_ids"),
            linked_experiment_plan_ids=(operator_context or {}).get(
                "linked_experiment_plan_ids"
            ),
            linked_validation_run_ids=(operator_context or {}).get(
                "linked_validation_run_ids"
            ),
            child_job_ids=(operator_context or {}).get("child_job_ids"),
        )
        if isinstance(operator_context, dict)
        else {}
    )
    return AgentDecisionTraceEventResponse(
        event_id=decision_trace_event_id(
            source_kind,
            source_id,
            decision_type,
            event_time,
            suffix=suffix,
        ),
        event_type=(
            str(event_type or "").strip() or str(decision_type or "").strip() or "event"
        ),
        event_time=event_time,
        source_kind=str(source_kind or "").strip() or "unknown",
        source_id=str(source_id or "").strip() or None,
        source_label=str(source_label or "").strip() or None,
        customer=str(customer or "").strip() or None,
        decision_type=str(decision_type or "").strip() or "event",
        reason_code=str(reason_code or "").strip() or None,
        reason_label=str(reason_label or "").strip() or None,
        scheduler_state=(
            deepcopy(scheduler_state) if isinstance(scheduler_state, dict) else None
        ),
        status=str(status or "").strip() or None,
        severity=str(severity or "").strip() or None,
        actor_mode=str(actor_mode or "").strip() or None,
        summary=str(summary or "").strip() or "Autonomy event",
        operator_note=str(operator_note or "").strip() or None,
        before_state=before_state or None,
        after_state=after_state or None,
        deep_link=(
            AgentDecisionTraceDeepLinkResponse.model_validate(deep_link)
            if isinstance(deep_link, dict)
            else None
        ),
        metadata=metadata or None,
        is_derived=bool(is_derived),
        record_origin=(
            str(record_origin or "").strip()
            or ("derived" if is_derived else "persisted")
        ),
        domain=normalized_context.get("domain"),
        objective=normalized_context.get("objective"),
        track_type=normalized_context.get("track_type"),
        source_scope=normalized_context.get("source_scope"),
        repo_source_ids=normalized_context.get("repo_source_ids"),
        benchmark_queries=normalized_context.get("benchmark_queries"),
        sandbox_profile_id=normalized_context.get("sandbox_profile_id"),
        automation_profile=normalized_context.get("automation_profile"),
        effective_policy=normalized_context.get("effective_policy"),
        confidence=normalized_context.get("confidence"),
        readiness=normalized_context.get("readiness"),
        linked_note_ids=normalized_context.get("linked_note_ids"),
        linked_experiment_plan_ids=normalized_context.get("linked_experiment_plan_ids"),
        linked_validation_run_ids=normalized_context.get("linked_validation_run_ids"),
        child_job_ids=normalized_context.get("child_job_ids"),
    )
