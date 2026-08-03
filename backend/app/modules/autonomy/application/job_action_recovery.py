"""Restart and relaunch actions for autonomous jobs."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.user import User
from app.modules.autonomy.application.job_action_contracts import (
    JobActionDependencies,
    JobActionError,
)

RECOVERY_ACTIONS = frozenset({"restart", "relaunch"})


async def perform_recovery_action(
    job: AgentJob,
    action: str,
    checkpoint_note: str | None,
    *,
    deps: JobActionDependencies,
    db: AsyncSession,
    current_user: User,
) -> AgentJob:
    if job.status not in {
        AgentJobStatus.COMPLETED.value,
        AgentJobStatus.FAILED.value,
        AgentJobStatus.CANCELLED.value,
    }:
        raise JobActionError(
            status_code=400,
            detail=(
                "Can only restart completed, failed, or cancelled jobs"
                if action == "restart"
                else "Can only relaunch completed, failed, or cancelled jobs"
            ),
        )

    if action == "restart":
        retry_outcome = await deps.quick_start_relaunch_dispatcher.refined_repo_retry(
            job,
            db=db,
            current_user=current_user,
        )
        if retry_outcome is not None:
            new_job = retry_outcome.job
            results_payload = job.results if isinstance(job.results, dict) else {}
            recovery = retry_outcome.recovery or {}
            deps.append_operator_intervention(
                results_payload,
                action="restart",
                actor_user_id=current_user.id,
                note=checkpoint_note,
                job_status_before=job.status,
                job_status_after=job.status,
                metadata={
                    "new_job_id": str(new_job.id),
                    "launch_mode": retry_outcome.launch_mode,
                    "recovery_strategy": retry_outcome.recovery_strategy,
                    "retry_reason": str(recovery.get("retry_reason") or "").strip()
                    or None,
                },
            )
            job.results = results_payload
            job.add_log_entry(
                {
                    "phase": "restart_requested",
                    "reason": "user_request",
                    "result": {
                        "new_job_id": str(new_job.id),
                        "launch_mode": retry_outcome.launch_mode,
                        "recovery_strategy": retry_outcome.recovery_strategy,
                    },
                }
            )
            await db.commit()
            return new_job

        previous_status = job.status
        job.status = AgentJobStatus.PENDING.value
        job.progress = 0
        job.iteration = 0
        job.tool_calls_used = 0
        job.llm_calls_used = 0
        job.tokens_used = 0
        job.error = None
        job.error_count = 0
        job.started_at = None
        job.completed_at = None
        job.current_phase = None
        job.phase_details = None
        job.execution_log = []
        results_payload = {}
        deps.append_operator_intervention(
            results_payload,
            action="restart",
            actor_user_id=current_user.id,
            note=checkpoint_note,
            job_status_before=previous_status,
            job_status_after=AgentJobStatus.PENDING.value,
        )
        job.results = results_payload
        job.output_artifacts = None
        job.add_log_entry({"phase": "restarted", "reason": "user_request"})
        deps.execute_agent_job_task.delay(str(job.id), str(current_user.id))
        return job

    relaunch_outcome = await deps.quick_start_relaunch_dispatcher.relaunch(
        job,
        db=db,
        current_user=current_user,
    )
    if relaunch_outcome is None:
        raise JobActionError(
            status_code=422,
            detail=(
                "Relaunch is only supported for quick-start Claude backend, "
                "domain research, repo bug triage, coding swarm quick starts, "
                "or role-workflow jobs with valid launch configuration"
            ),
        )
    new_job = relaunch_outcome.job
    relaunch_mode = relaunch_outcome.launch_mode
    job.add_log_entry(
        {
            "phase": "relaunch_requested",
            "reason": "user_request",
            "result": {
                "new_job_id": str(new_job.id),
                "launch_mode": relaunch_mode,
            },
        }
    )
    results_payload = job.results if isinstance(job.results, dict) else {}
    deps.append_operator_intervention(
        results_payload,
        action="relaunch",
        actor_user_id=current_user.id,
        note=checkpoint_note,
        job_status_before=job.status,
        job_status_after=job.status,
        metadata={
            "new_job_id": str(new_job.id),
            "launch_mode": relaunch_mode,
            "recovery_strategy": relaunch_outcome.recovery_strategy,
        },
    )
    job.results = results_payload
    await db.commit()
    return new_job
