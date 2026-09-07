"""Resume handling for paused autonomous jobs and approval checkpoints."""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services import agent_loop_policy
from app.models.user import User
from app.modules.autonomy.application.job_action_contracts import (
    JobActionDependencies,
    JobActionError,
)


async def perform_resume_action(
    job: AgentJob,
    checkpoint_note: str | None,
    *,
    deps: JobActionDependencies,
    db: AsyncSession,
    current_user: User,
) -> AgentJob:
    if job.status != AgentJobStatus.PAUSED.value:
        raise JobActionError(
            status_code=400,
            detail="Can only resume paused jobs",
        )

    (
        results_payload,
        approval_payload,
        pending_checkpoint,
    ) = deps.approval_payload_from_results(job.results)
    checkpoint_row = await deps.load_latest_checkpoint(job.id, db)
    state = (
        dict(checkpoint_row.state)
        if checkpoint_row and isinstance(checkpoint_row.state, dict)
        else {}
    )
    if (
        isinstance(pending_checkpoint, dict)
        and pending_checkpoint.get("checkpoint_type") == "execution_reconciliation"
    ):
        raise JobActionError(
            status_code=400,
            detail=(
                "Interrupted tool calls require an explicit approve, edit, "
                "skip, or reject action"
            ),
        )

    if pending_checkpoint:
        approval_payload["pending"] = None
        deps.append_step_event(
            state,
            {
                "type": "checkpoint_approved",
                "method": "resume_action",
                "iteration": int(pending_checkpoint.get("iteration", 0) or 0),
                "plan_step_id": str(
                    pending_checkpoint.get("plan_step_id") or ""
                ).strip()
                or None,
                "plan_step_index": int(
                    pending_checkpoint.get("plan_step_index", -1) or -1
                ),
                "tool": str(
                    ((pending_checkpoint.get("action") or {}).get("tool") or "")
                ).strip()
                or None,
                "note": checkpoint_note,
                "actor_user_id": str(current_user.id),
            },
        )
        deps.append_approval_event(
            approval_payload,
            pending_checkpoint,
            method="resume_action",
            user_id=current_user.id,
            note=checkpoint_note,
        )
        job.add_log_entry(
            {
                "phase": "approval_checkpoint_approved",
                "reason": "resume_action",
                "action_tool": str(
                    ((pending_checkpoint.get("action") or {}).get("tool") or "")
                ).strip(),
            }
        )
        if checkpoint_row:
            state["approval_checkpoint_pending"] = None
            deps.set_current_plan_step_status(
                state,
                status="in_progress",
                advance_next=False,
            )
            checkpoint_row.state = state
            db.add(checkpoint_row)
        deps.sync_execution_strategy_state(
            results_payload,
            approval_payload=approval_payload,
            state=state,
        )
        results_payload["approval_checkpoint"] = None

    # The note is the whole point of resuming a run that stopped because it
    # was stuck: it carries the correction -- the right path, a credential, an
    # assumption the run got wrong. It was recorded in the audit trail and
    # nowhere the agent reads, so a person could answer a blocked run and the
    # run would resume from its checkpoint and repeat the same mistake.
    note_text = str(checkpoint_note or "").strip()
    if note_text:
        clue = {
            "note": note_text[:2000],
            "iteration": int(state.get("iteration", 0) or 0),
            "actor_user_id": str(current_user.id),
        }

        # The job config, not the checkpoint, is the durable home. A run that
        # blocks early has no checkpoint at all -- the first live test of this
        # had zero rows -- and a clue written only to a checkpoint row that
        # does not exist is dropped in silence, which looks exactly like an
        # agent that ignored the correction.
        config = dict(job.config or {})
        pending = config.get("operator_clues")
        if not isinstance(pending, list):
            pending = []
        config["operator_clues"] = ([*pending, clue])[-8:]
        job.config = config
        flag_modified(job, "config")

        if checkpoint_row is not None:
            clues = state.get("operator_clues")
            if not isinstance(clues, list):
                clues = []
            clues.append(clue)
            state["operator_clues"] = clues[-8:]
            # And drop the evidence that made it give up. The run stopped
            # because of what it believed, and it has just been told that
            # belief was wrong.
            agent_loop_policy.clear_give_up_state(state)
            # `state` is a copy of the row's dict, so assigning it is a real
            # attribute set and SQLAlchemy marks the column dirty -- same as
            # the approval branch above.
            checkpoint_row.state = state
            db.add(checkpoint_row)

    # Whatever it was waiting for, it is no longer waiting: clearing the phase
    # is what lets the stalled-job sweep look after this job again.
    if str(job.current_phase or "") in ("blocked_needs_input", "awaiting_approval"):
        job.current_phase = "resuming"
        job.phase_details = "Resumed by an operator" + (
            " with a note" if note_text else ""
        )

    deps.append_operator_intervention(
        results_payload,
        action="resume",
        actor_user_id=current_user.id,
        note=checkpoint_note,
        job_status_before=job.status,
        job_status_after=AgentJobStatus.PENDING.value,
        metadata={
            "approval_checkpoint_pending": bool(pending_checkpoint),
        },
    )
    job.results = results_payload
    flag_modified(job, "results")
    job.status = AgentJobStatus.PENDING.value
    job.add_log_entry({"phase": "resumed", "reason": "user_request"})
    deps.execute_agent_job_task.delay(str(job.id), str(current_user.id))
    return job
