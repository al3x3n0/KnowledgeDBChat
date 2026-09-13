"""Project autonomous-job state into derived decision-trace events."""

from dataclasses import dataclass
from typing import Any, Callable

from app.models.agent_job import AgentJob
from app.schemas.agent_job import AgentDecisionTraceEventResponse
from app.services.operator_interventions import (
    derive_operator_interventions_with_outcomes,
)


@dataclass(frozen=True)
class JobDecisionTraceDependencies:
    parse_time: Callable[..., Any]
    build_event: Callable[..., AgentDecisionTraceEventResponse]
    queue_customer_for_job: Callable[[AgentJob], str | None]
    reason_label: Callable[[str | None], str | None]


def build_job_decision_trace(
    job: AgentJob,
    *,
    deps: JobDecisionTraceDependencies,
) -> list[AgentDecisionTraceEventResponse]:
    """Build operator-intervention and scheduler-recovery events for one job."""
    events: list[AgentDecisionTraceEventResponse] = []
    execution_strategy = (
        job.results.get("execution_strategy")
        if isinstance(getattr(job, "results", None), dict)
        and isinstance(job.results.get("execution_strategy"), dict)
        else {}
    )
    job_reference_time = (
        job.last_activity_at or job.completed_at or job.started_at or job.created_at
    )
    operator_interventions = derive_operator_interventions_with_outcomes(
        execution_strategy.get("operator_interventions")
        if isinstance(execution_strategy.get("operator_interventions"), list)
        else [],
        current_status=job.status,
        completed_at=getattr(job, "completed_at", None),
    )
    customer = deps.queue_customer_for_job(job)
    for index, row in enumerate(operator_interventions):
        if not isinstance(row, dict):
            continue
        event_time = deps.parse_time(row.get("at"), fallback=job_reference_time)
        if event_time is None:
            continue
        action = str(row.get("action") or "operator_intervention").strip().lower()
        events.append(
            deps.build_event(
                event_type="job_operator_action",
                event_time=event_time,
                source_kind="job",
                source_id=str(job.id) if job.id else None,
                source_label=job.name,
                customer=customer,
                decision_type=action or "operator_intervention",
                reason_code=str(row.get("outcome_status") or "").strip() or None,
                reason_label=None,
                status=str(
                    row.get("job_status_after")
                    or row.get("job_status_before")
                    or job.status
                    or ""
                ).strip()
                or None,
                severity="medium",
                actor_mode="operator",
                summary=f"{job.name}: {action.replace('_', ' ')}",
                operator_note=str(row.get("note") or "").strip() or None,
                before_state=(
                    {"job_status": row.get("job_status_before")}
                    if row.get("job_status_before")
                    else None
                ),
                after_state=(
                    {"job_status": row.get("job_status_after")}
                    if row.get("job_status_after")
                    else None
                ),
                deep_link={
                    "target_tab": "jobs",
                    "job_id": job.id,
                    "params": {"job": str(job.id)},
                    "label": "Open Job",
                },
                metadata={
                    "outcome_status": row.get("outcome_status"),
                    "outcome_reason": row.get("outcome_reason"),
                    "metadata": row.get("metadata"),
                },
                suffix=str(index),
            )
        )

    scheduler_state = (
        execution_strategy.get("scheduler_state")
        if isinstance(execution_strategy.get("scheduler_state"), dict)
        else {}
    )
    queue_reason = str(scheduler_state.get("queue_reason") or "").strip().lower()
    scheduled_at = deps.parse_time(
        scheduler_state.get("last_dispatched_at")
        or scheduler_state.get("last_scheduled_at")
        or scheduler_state.get("backoff_until"),
        fallback=job_reference_time,
    )
    if queue_reason and scheduled_at is not None:
        reason_label = deps.reason_label(queue_reason)
        events.append(
            deps.build_event(
                event_type="job_recovery_queued",
                event_time=scheduled_at,
                source_kind="job",
                source_id=str(job.id) if job.id else None,
                source_label=job.name,
                customer=customer,
                decision_type="job_recovery_queued",
                reason_code=queue_reason,
                reason_label=reason_label,
                scheduler_state=scheduler_state,
                status=job.status,
                severity=(
                    "high"
                    if queue_reason in {"execution_failure", "stalled_run"}
                    else "medium"
                ),
                actor_mode="autonomous",
                summary=f"{job.name}: queued for scheduler recovery",
                deep_link={
                    "target_tab": "queue",
                    "job_id": job.id,
                    "params": {"tab": "queue", "job": str(job.id)},
                    "label": "Open Checkpoint Queue",
                },
                metadata={
                    "scheduler_state": scheduler_state,
                    "reason_label": reason_label,
                },
                suffix="scheduler",
            )
        )
    return events
