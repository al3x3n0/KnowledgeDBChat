"""Basic pause, cancel, and summary actions for autonomous jobs."""

from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.user import User
from app.modules.autonomy.application.job_action_contracts import (
    JobActionDependencies,
    JobActionError,
)
from app.services.research_inbox_follow_up_service import sync_follow_up_outcome_for_job

LIFECYCLE_ACTIONS = frozenset({"pause", "cancel", "generate_summary"})


async def perform_lifecycle_action(
    job: AgentJob,
    action: str,
    checkpoint_note: str | None,
    *,
    deps: JobActionDependencies,
    db: AsyncSession,
    current_user: User,
) -> AgentJob:
    if action == "pause":
        if job.status != AgentJobStatus.RUNNING.value:
            raise JobActionError(
                status_code=400,
                detail="Can only pause running jobs",
            )
        results_payload = job.results if isinstance(job.results, dict) else {}
        deps.append_operator_intervention(
            results_payload,
            action="pause",
            actor_user_id=current_user.id,
            note=checkpoint_note,
            job_status_before=job.status,
            job_status_after=AgentJobStatus.PAUSED.value,
        )
        job.results = results_payload
        flag_modified(job, "results")
        job.status = AgentJobStatus.PAUSED.value
        job.add_log_entry({"phase": "paused", "reason": "user_request"})
        return job

    if action == "cancel":
        if job.status not in {
            AgentJobStatus.PENDING.value,
            AgentJobStatus.RUNNING.value,
            AgentJobStatus.PAUSED.value,
        }:
            raise JobActionError(
                status_code=400,
                detail=f"Cannot cancel job in status: {job.status}",
            )
        results_payload = job.results if isinstance(job.results, dict) else {}
        deps.append_operator_intervention(
            results_payload,
            action="cancel",
            actor_user_id=current_user.id,
            note=checkpoint_note,
            job_status_before=job.status,
            job_status_after=AgentJobStatus.CANCELLED.value,
        )
        job.results = results_payload
        flag_modified(job, "results")
        job.status = AgentJobStatus.CANCELLED.value
        job.completed_at = datetime.utcnow()
        job.add_log_entry({"phase": "cancelled", "reason": "user_request"})
        await sync_follow_up_outcome_for_job(db, job)
        return job

    if job.status != AgentJobStatus.COMPLETED.value:
        raise JobActionError(
            status_code=400,
            detail="Can only generate summary for completed jobs",
        )
    deps.generate_job_summary.delay(str(job.id))
    return job
