"""Application state machine for operator actions on autonomous jobs."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob
from app.models.user import User
from app.modules.autonomy.application.job_action_checkpoint_decisions import (
    CHECKPOINT_DECISION_ACTIONS,
    perform_checkpoint_decision,
)
from app.modules.autonomy.application.job_action_checkpoint_resume import (
    perform_resume_action,
)
from app.modules.autonomy.application.job_action_contracts import (
    JobActionDependencies,
    JobActionError,
)
from app.modules.autonomy.application.job_action_lifecycle import (
    LIFECYCLE_ACTIONS,
    perform_lifecycle_action,
)
from app.modules.autonomy.application.job_action_recovery import (
    RECOVERY_ACTIONS,
    perform_recovery_action,
)
from app.modules.autonomy.application.job_action_swarm import (
    SWARM_ACTIONS,
    perform_swarm_action,
)
from app.schemas.agent_job import AgentJobActionRequest


async def perform_job_action(
    job: AgentJob,
    request: AgentJobActionRequest,
    *,
    deps: JobActionDependencies,
    db: AsyncSession,
    current_user: User,
) -> AgentJob:
    action = request.action.lower()
    checkpoint_note = str(request.checkpoint_note or "").strip() or None
    action_payload = (
        request.action_payload if isinstance(request.action_payload, dict) else {}
    )
    if action in SWARM_ACTIONS and not deps.is_job_visible(job, current_user):
        raise JobActionError(status_code=404, detail="Agent job not found")

    if action == "resume":
        return await perform_resume_action(
            job,
            checkpoint_note,
            deps=deps,
            db=db,
            current_user=current_user,
        )

    elif action in CHECKPOINT_DECISION_ACTIONS:
        return await perform_checkpoint_decision(
            job,
            action,
            request,
            checkpoint_note,
            deps=deps,
            db=db,
            current_user=current_user,
        )

    elif action in LIFECYCLE_ACTIONS:
        return await perform_lifecycle_action(
            job,
            action,
            checkpoint_note,
            deps=deps,
            db=db,
            current_user=current_user,
        )

    elif action in RECOVERY_ACTIONS:
        return await perform_recovery_action(
            job,
            action,
            checkpoint_note,
            deps=deps,
            db=db,
            current_user=current_user,
        )

    elif action in SWARM_ACTIONS:
        return await perform_swarm_action(
            job,
            action,
            action_payload,
            checkpoint_note,
            deps=deps,
            db=db,
            current_user=current_user,
        )

    else:
        raise JobActionError(
            status_code=400,
            detail=(
                "Unknown action: "
                f"{action}. Valid actions: pause, resume, cancel, restart, relaunch, "
                "generate_summary, approve, reject, edit, skip, launch_tie_breaker, "
                "promote_swarm_candidate, assign_swarm_review, clear_swarm_assignment, "
                "update_swarm_review_note"
            ),
        )
