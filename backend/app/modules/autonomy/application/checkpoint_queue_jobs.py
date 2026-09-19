"""Project job approvals and recurring recoveries into checkpoint queue rows."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

from app.models.agent_job import AgentJob, AgentJobStatus
from app.schemas.agent_job import (
    AgentCheckpointQueueActionResponse,
    AgentCheckpointQueueItemResponse,
)
from app.utils.datetimes import as_aware_utc


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


#: The phase the runtime sets when a run stops because it needs a person
#: (``agent_runtime_finalizer``). It is not an approval: nothing is proposed,
#: something is missing.
BLOCKED_PHASE = "blocked_needs_input"


def _is_blocked_on_a_person(job: AgentJob) -> bool:
    return (
        str(job.status or "").strip().lower() == AgentJobStatus.PAUSED.value
        and str(job.current_phase or "").strip() == BLOCKED_PHASE
    )


#: The log entry a run leaves when it finishes without satisfying its contract.
#: Distinct from blocking: the run did not stop to ask, it ran out of road.
CONTRACT_UNMET_PHASE = "completed_contract_unmet"


def _finished_without_meeting_its_contract(job: AgentJob) -> dict[str, Any]:
    """The entry a completed-but-unsatisfied run left, or {}.

    A run that gives up early pauses as blocked_needs_input and is visible. A
    run that exhausts its iteration budget with the contract unmet is marked
    **completed**, so the worse outcome carried the better-looking status and
    nothing surfaced it: 28 such runs in a fortnight, all reporting success
    while delivering nothing the contract asked for.
    """
    if str(job.status or "").strip().lower() != AgentJobStatus.COMPLETED.value:
        return {}
    entries = job.execution_log if isinstance(job.execution_log, list) else []
    for entry in reversed(entries):
        if isinstance(entry, dict) and entry.get("phase") == CONTRACT_UNMET_PHASE:
            return entry
    return {}


def _blocked_payload(job: AgentJob) -> dict[str, Any]:
    """What the run recorded about why it gave up, if anything."""
    results = job.results if isinstance(job.results, dict) else {}
    blocked = results.get("blocked")
    return blocked if isinstance(blocked, dict) else {}


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

        # A run that stopped because it needs a person is not an approval:
        # nothing is proposed for sign-off, something is missing. It carries no
        # approval checkpoint, and the recurring branch below skips it because
        # it is a one-shot job, so before this it appeared in no queue at all --
        # six were found waiting 8 to 11 days, one of them a pipeline stage,
        # which is the whole DAG behind it stopped with nobody told.
        if _is_blocked_on_a_person(job):
            blocked = _blocked_payload(job)
            created_at = (
                job.last_activity_at
                or job.completed_at
                or job.started_at
                or job.created_at
            )
            urgency = deps.queue_priority_fields(
                item_type="blocked_run",
                reason_code="needs_input",
                created_at=created_at,
                next_run_at=job.next_run_at,
                backoff_until=None,
                stale=False,
                now=now,
            )
            # Resume is offered only when the run said it could be resumed;
            # the rest is a decision a person has to make with the job open.
            action_rows = (
                [
                    AgentCheckpointQueueActionResponse(
                        kind="job_action",
                        label="Resume",
                        action="resume",
                        recommended=True,
                    )
                ]
                if blocked.get("resumable")
                else []
            )
            missing = [str(m) for m in (blocked.get("missing") or []) if str(m).strip()]
            items.append(
                AgentCheckpointQueueItemResponse(
                    queue_key=f"blocked:{job.id}",
                    item_type="blocked_run",
                    priority=100,
                    title=job.name,
                    # The run usually knows why it gave up. That sentence is the
                    # single most useful thing in this row.
                    summary=str(
                        blocked.get("reason") or job.phase_details or job.goal or ""
                    ).strip()[:320]
                    or None,
                    evidence_summary=deps.queue_evidence_summary_for_job(job),
                    status=job.status,
                    customer=customer,
                    job_name=job.name,
                    job_type=str(job.job_type or "").strip() or None,
                    reason_code="needs_input",
                    reason_label=deps.queue_reason_label("needs_input"),
                    recommended_action="resume" if action_rows else None,
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
                    # What it lacks, named, so the row can be acted on without
                    # opening the run.
                    checkpoint={
                        "kind": "blocked",
                        "reason": blocked.get("reason"),
                        "missing": missing,
                        "resumable": bool(blocked.get("resumable")),
                    },
                    scheduler_state=scheduler_state,
                    actions=action_rows,
                )
            )
            continue

        unmet = _finished_without_meeting_its_contract(job)
        if unmet:
            created_at = (
                job.completed_at
                or job.last_activity_at
                or job.started_at
                or job.created_at
            )
            urgency = deps.queue_priority_fields(
                item_type="contract_unmet",
                reason_code="contract_unmet",
                created_at=created_at,
                next_run_at=job.next_run_at,
                backoff_until=None,
                stale=False,
                now=now,
            )
            missing = [
                str(item) for item in (unmet.get("missing") or []) if str(item).strip()
            ]
            # restart resets iteration and progress, so the run gets its budget
            # back rather than re-hitting the cap it just hit; relaunch starts a
            # fresh one. Both are valid on a completed job.
            action_rows = [
                AgentCheckpointQueueActionResponse(
                    kind="job_action",
                    label="Restart",
                    action="restart",
                    recommended=True,
                ),
                AgentCheckpointQueueActionResponse(
                    kind="job_action", label="Relaunch", action="relaunch"
                ),
            ]
            items.append(
                AgentCheckpointQueueItemResponse(
                    queue_key=f"contract_unmet:{job.id}",
                    item_type="contract_unmet",
                    # Below an approval or a blocked run on purpose: nobody is
                    # waiting on this one. It is a quality signal about work
                    # already reported as done, not a request for a decision.
                    priority=60,
                    title=job.name,
                    summary=str(unmet.get("reason") or "").strip()[:320] or None,
                    evidence_summary=deps.queue_evidence_summary_for_job(job),
                    status=job.status,
                    customer=customer,
                    job_name=job.name,
                    job_type=str(job.job_type or "").strip() or None,
                    reason_code="contract_unmet",
                    reason_label=deps.queue_reason_label("contract_unmet"),
                    recommended_action="restart",
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
                    checkpoint={
                        "kind": "contract_unmet",
                        "reason": unmet.get("reason"),
                        "missing": missing,
                    },
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
        # Both sides normalised before subtracting. Every timestamp column on
        # a job is DateTime(timezone=True) while the Python defaults are naive
        # utcnow(), so what goes in naive comes back from Postgres AWARE --
        # and `now` here is whatever the composer passed, which is naive. One
        # such row turned the whole checkpoint queue into a 500:
        # "can't subtract offset-naive and offset-aware datetimes".
        last_activity = as_aware_utc(job.last_activity_at)
        stale_running = (
            str(job.status or "").strip().lower() == AgentJobStatus.RUNNING.value
            and last_activity is not None
            and (as_aware_utc(now) - last_activity) > timedelta(minutes=30)
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
