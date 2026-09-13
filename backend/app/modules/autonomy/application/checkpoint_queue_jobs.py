"""Project job approvals and recurring recoveries into checkpoint queue rows."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

from app.models.agent_job import AgentJob, AgentJobStatus
from app.schemas.agent_job import (
    AgentCheckpointQueueActionResponse,
    AgentCheckpointQueueItemResponse,
)


@dataclass(frozen=True)
class JobCheckpointQueueDependencies:
    extract_approval_checkpoint: Callable[..., Any]
    extract_scheduler_state: Callable[..., Any]
    queue_customer_for_job: Callable[..., Any]
    present_job: Callable[..., Any]
    queue_priority_fields: Callable[..., Any]
    queue_evidence_summary_for_job: Callable[..., Any]
    queue_reason_label: Callable[..., Any]
    parse_optional_datetime: Callable[..., Any]
    extract_launch_mode: Callable[..., Any]


def build_job_checkpoint_queue_items(
    jobs: list[AgentJob],
    *,
    now: datetime,
    deps: JobCheckpointQueueDependencies,
) -> list[AgentCheckpointQueueItemResponse]:
    items: list[AgentCheckpointQueueItemResponse] = []
    for job in jobs:
        checkpoint = deps.extract_approval_checkpoint(job)
        scheduler_state = deps.extract_scheduler_state(job)
        customer = deps.queue_customer_for_job(job)
        job_response = deps.present_job(job)
        if checkpoint:
            created_at = (
                job.last_activity_at
                or job.completed_at
                or job.started_at
                or job.created_at
            )
            urgency = deps.queue_priority_fields(
                item_type="approval_checkpoint",
                reason_code="approval_required",
                created_at=created_at,
                next_run_at=job.next_run_at,
                backoff_until=None,
                stale=False,
                now=now,
            )
            action_rows = [
                AgentCheckpointQueueActionResponse(
                    kind="job_action",
                    label="Approve",
                    action="approve",
                    recommended=True,
                ),
                AgentCheckpointQueueActionResponse(
                    kind="job_action", label="Edit + Approve", action="edit"
                ),
                AgentCheckpointQueueActionResponse(
                    kind="job_action", label="Reject", action="reject"
                ),
                AgentCheckpointQueueActionResponse(
                    kind="job_action", label="Skip Step", action="skip"
                ),
            ]
            items.append(
                AgentCheckpointQueueItemResponse(
                    queue_key=f"approval:{job.id}",
                    item_type="approval_checkpoint",
                    priority=100,
                    title=job.name,
                    summary=str(checkpoint.get("message") or job.goal or "").strip()[
                        :320
                    ]
                    or None,
                    evidence_summary=deps.queue_evidence_summary_for_job(job),
                    status=job.status,
                    customer=customer,
                    job_name=job.name,
                    job_type=str(job.job_type or "").strip() or None,
                    reason_code="approval_required",
                    reason_label=deps.queue_reason_label("approval_required"),
                    recommended_action="approve",
                    priority_score=urgency["priority_score"],
                    age_minutes=urgency["age_minutes"],
                    sla_bucket=urgency["sla_bucket"],
                    escalation_level=urgency["escalation_level"],
                    is_overdue=urgency["is_overdue"],
                    is_stale=urgency["is_stale"],
                    next_run_at=job.next_run_at,
                    backoff_until=None,
                    action_count=len(action_rows),
                    created_at=created_at,
                    job_id=job.id,
                    job=job_response,
                    checkpoint=checkpoint,
                    scheduler_state=scheduler_state,
                    actions=action_rows,
                )
            )
            continue

        is_recurring = str(job.schedule_type or "").strip().lower() in {
            "recurring",
            "continuous",
        }
        failed_or_paused = str(job.status or "").strip().lower() in {
            AgentJobStatus.FAILED.value,
            AgentJobStatus.PAUSED.value,
        }
        stale_running = (
            str(job.status or "").strip().lower() == AgentJobStatus.RUNNING.value
            and job.last_activity_at is not None
            and (now - job.last_activity_at) > timedelta(minutes=30)
        )
        if not is_recurring or not (failed_or_paused or stale_running):
            continue

        reason = str((scheduler_state or {}).get("queue_reason") or "").strip() or (
            "stalled_run" if stale_running else "scheduled_recovery"
        )
        created_at = (
            job.last_activity_at or job.completed_at or job.started_at or job.created_at
        )
        backoff_until = deps.parse_optional_datetime(
            (scheduler_state or {}).get("backoff_until")
        )
        urgency = deps.queue_priority_fields(
            item_type="job_recovery",
            reason_code=reason,
            created_at=created_at,
            next_run_at=job.next_run_at,
            backoff_until=backoff_until,
            stale=stale_running,
            now=now,
        )
        launch_mode = deps.extract_launch_mode(
            job.config if isinstance(job.config, dict) else None
        )
        is_repo_bug_triage = launch_mode == "quick_start_repo_bug_triage"
        action_rows = [
            AgentCheckpointQueueActionResponse(
                kind="job_action",
                label="Retry with refined plan" if is_repo_bug_triage else "Restart",
                action="restart",
                recommended=True,
            ),
            AgentCheckpointQueueActionResponse(
                kind="job_action",
                label="Resume verification" if is_repo_bug_triage else "Resume",
                action="resume",
            ),
            AgentCheckpointQueueActionResponse(
                kind="job_action", label="Cancel", action="cancel"
            ),
        ]
        items.append(
            AgentCheckpointQueueItemResponse(
                queue_key=f"recovery:{job.id}",
                item_type="job_recovery",
                priority=80,
                title=job.name,
                summary=(
                    job.error
                    or job.phase_details
                    or f"Recurring job requires operator recovery ({reason})."
                )[:320],
                evidence_summary=deps.queue_evidence_summary_for_job(job),
                status=job.status,
                customer=customer,
                job_name=job.name,
                job_type=str(job.job_type or "").strip() or None,
                reason_code=reason,
                reason_label=deps.queue_reason_label(reason),
                recommended_action=(
                    "restart"
                    if reason
                    in {"execution_failure", "stalled_run", "scheduled_recovery"}
                    else "resume"
                ),
                priority_score=urgency["priority_score"],
                age_minutes=urgency["age_minutes"],
                sla_bucket=urgency["sla_bucket"],
                escalation_level=urgency["escalation_level"],
                is_overdue=urgency["is_overdue"],
                is_stale=urgency["is_stale"],
                next_run_at=job.next_run_at,
                backoff_until=backoff_until,
                action_count=len(action_rows),
                created_at=created_at,
                job_id=job.id,
                job=job_response,
                scheduler_state={**(scheduler_state or {}), "queue_reason": reason},
                actions=action_rows,
            )
        )
    return items
