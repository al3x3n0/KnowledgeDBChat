"""Explicit approve, edit, skip, and reject checkpoint decisions."""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.user import User
from app.modules.autonomy.application.job_action_contracts import (
    JobActionDependencies,
    JobActionError,
)
from app.schemas.agent_job import AgentJobActionRequest

CHECKPOINT_DECISION_ACTIONS = frozenset({"approve", "edit", "skip", "reject"})


def _pending_tool(pending_checkpoint: dict) -> str | None:
    return (
        str(((pending_checkpoint.get("action") or {}).get("tool") or "")).strip()
        or None
    )


def _pending_step_metadata(pending_checkpoint: dict) -> dict[str, Any]:
    return {
        "plan_step_id": str(pending_checkpoint.get("plan_step_id") or "").strip()
        or None,
        "plan_step_index": int(pending_checkpoint.get("plan_step_index", -1) or -1),
    }


async def perform_checkpoint_decision(
    job: AgentJob,
    action: str,
    request: AgentJobActionRequest,
    checkpoint_note: str | None,
    *,
    deps: JobActionDependencies,
    db: AsyncSession,
    current_user: User,
) -> AgentJob:
    if job.status != AgentJobStatus.PAUSED.value:
        raise JobActionError(
            status_code=400,
            detail="Approval checkpoint actions require paused status",
        )

    (
        results_payload,
        approval_payload,
        pending_checkpoint,
    ) = deps.approval_payload_from_results(job.results)
    if not isinstance(pending_checkpoint, dict):
        raise JobActionError(
            status_code=400,
            detail="No pending approval checkpoint for this job",
        )
    checkpoint_row = await deps.load_latest_checkpoint(job.id, db)
    state = (
        dict(checkpoint_row.state)
        if checkpoint_row and isinstance(checkpoint_row.state, dict)
        else {}
    )

    if pending_checkpoint.get("checkpoint_type") == "execution_reconciliation":
        journal_summary = (
            dict(results_payload.get("execution_journal") or {})
            if isinstance(results_payload.get("execution_journal"), dict)
            else {}
        )
        journal_summary["reconciliation_pending"] = False
        results_payload["execution_journal"] = journal_summary

    action_patch: dict[str, Any] = {}
    if request.checkpoint_action_patch is not None:
        try:
            action_patch = deps.normalize_checkpoint_action_patch(
                request.checkpoint_action_patch
            )
        except ValueError as exc:
            raise JobActionError(status_code=400, detail=str(exc))

    if action == "edit":
        _perform_edit(
            job,
            pending_checkpoint,
            action_patch,
            checkpoint_note,
            results_payload,
            approval_payload,
            state,
            deps=deps,
            current_user=current_user,
        )
    elif action == "approve":
        _perform_approve(
            job,
            pending_checkpoint,
            action_patch,
            checkpoint_note,
            results_payload,
            approval_payload,
            state,
            deps=deps,
            current_user=current_user,
        )
    else:
        _perform_skip_or_reject(
            job,
            action,
            pending_checkpoint,
            checkpoint_note,
            results_payload,
            approval_payload,
            state,
            deps=deps,
            current_user=current_user,
        )

    if checkpoint_row:
        checkpoint_row.state = state
        db.add(checkpoint_row)
    if action != "reject":
        deps.execute_agent_job_task.delay(str(job.id), str(current_user.id))
    return job


def _clear_pending_checkpoint(
    results_payload: dict,
    approval_payload: dict,
    state: dict,
    *,
    deps: JobActionDependencies,
) -> None:
    state["approval_checkpoint_pending"] = None
    state["execution_reconciliation_pending"] = None
    approval_payload["pending"] = None
    deps.sync_execution_strategy_state(
        results_payload,
        approval_payload=approval_payload,
        state=state,
    )
    results_payload["approval_checkpoint"] = None


def _perform_edit(
    job: AgentJob,
    pending_checkpoint: dict,
    action_patch: dict[str, Any],
    checkpoint_note: str | None,
    results_payload: dict,
    approval_payload: dict,
    state: dict,
    *,
    deps: JobActionDependencies,
    current_user: User,
) -> None:
    if not action_patch:
        raise JobActionError(
            status_code=400,
            detail="edit action requires checkpoint_action_patch",
        )
    edited_action = deps.apply_checkpoint_action_patch(
        pending_checkpoint,
        action_patch,
    )
    state["approval_override_action"] = edited_action
    deps.set_current_plan_step_status(
        state,
        status="in_progress",
        advance_next=False,
    )
    step_metadata = _pending_step_metadata(pending_checkpoint)
    deps.append_step_event(
        state,
        {
            "type": "checkpoint_approved",
            "method": "edit_action",
            "iteration": int(pending_checkpoint.get("iteration", 0) or 0),
            **step_metadata,
            "tool": str(edited_action.get("tool") or "").strip() or None,
            "note": checkpoint_note,
            "actor_user_id": str(current_user.id),
        },
    )
    deps.append_approval_event(
        approval_payload,
        pending_checkpoint,
        method="edit_action",
        user_id=current_user.id,
        note=checkpoint_note,
        edited_action=edited_action,
    )
    _clear_pending_checkpoint(
        results_payload,
        approval_payload,
        state,
        deps=deps,
    )
    deps.append_operator_intervention(
        results_payload,
        action="edit",
        actor_user_id=current_user.id,
        note=checkpoint_note,
        job_status_before=job.status,
        job_status_after=AgentJobStatus.PENDING.value,
        metadata={
            **step_metadata,
            "tool": str(edited_action.get("tool") or "").strip() or None,
        },
    )
    job.results = results_payload
    job.status = AgentJobStatus.PENDING.value
    job.current_phase = "approval_edited"
    job.phase_details = "Checkpoint action edited and approved"
    job.add_log_entry(
        {
            "phase": "approval_checkpoint_edited",
            "reason": "edit_action",
            "action_tool": str(edited_action.get("tool") or "").strip(),
        }
    )


def _perform_approve(
    job: AgentJob,
    pending_checkpoint: dict,
    action_patch: dict[str, Any],
    checkpoint_note: str | None,
    results_payload: dict,
    approval_payload: dict,
    state: dict,
    *,
    deps: JobActionDependencies,
    current_user: User,
) -> None:
    edited_action: dict[str, Any] = {}
    if action_patch:
        edited_action = deps.apply_checkpoint_action_patch(
            pending_checkpoint,
            action_patch,
        )
        state["approval_override_action"] = edited_action
    elif pending_checkpoint.get("checkpoint_type") == "execution_reconciliation":
        if not bool(pending_checkpoint.get("retryable_from_journal", False)):
            raise JobActionError(
                status_code=400,
                detail=(
                    "The interrupted action contains redacted parameters; "
                    "use edit with explicit replacement parameters or skip it"
                ),
            )
        edited_action = dict(pending_checkpoint.get("action") or {})
        state["approval_override_action"] = edited_action

    deps.set_current_plan_step_status(
        state,
        status="in_progress",
        advance_next=False,
    )
    step_metadata = _pending_step_metadata(pending_checkpoint)
    deps.append_step_event(
        state,
        {
            "type": "checkpoint_approved",
            "method": "approve_action",
            "iteration": int(pending_checkpoint.get("iteration", 0) or 0),
            **step_metadata,
            "tool": _pending_tool(pending_checkpoint),
            "note": checkpoint_note,
            "actor_user_id": str(current_user.id),
        },
    )
    deps.append_approval_event(
        approval_payload,
        pending_checkpoint,
        method="approve_action",
        user_id=current_user.id,
        note=checkpoint_note,
        edited_action=edited_action or None,
    )
    _clear_pending_checkpoint(
        results_payload,
        approval_payload,
        state,
        deps=deps,
    )
    deps.append_operator_intervention(
        results_payload,
        action="approve",
        actor_user_id=current_user.id,
        note=checkpoint_note,
        job_status_before=job.status,
        job_status_after=AgentJobStatus.PENDING.value,
        metadata={
            **step_metadata,
            "tool": _pending_tool(pending_checkpoint),
            "edited_action": bool(edited_action),
        },
    )
    job.results = results_payload
    job.status = AgentJobStatus.PENDING.value
    job.current_phase = "approval_approved"
    job.phase_details = "Checkpoint approved"
    job.add_log_entry(
        {
            "phase": "approval_checkpoint_approved",
            "reason": "approve_action",
            "action_tool": _pending_tool(pending_checkpoint) or "",
        }
    )


def _perform_skip_or_reject(
    job: AgentJob,
    action: str,
    pending_checkpoint: dict,
    checkpoint_note: str | None,
    results_payload: dict,
    approval_payload: dict,
    state: dict,
    *,
    deps: JobActionDependencies,
    current_user: User,
) -> None:
    is_skip = action == "skip"
    step_meta = deps.set_current_plan_step_status(
        state,
        status="skipped" if is_skip else "failed",
        advance_next=is_skip,
    )
    event_type = "step_skipped" if is_skip else "checkpoint_rejected"
    method = "skip_action" if is_skip else "reject_action"
    deps.append_step_event(
        state,
        {
            "type": event_type,
            "method": method,
            "iteration": int(pending_checkpoint.get("iteration", 0) or 0),
            "plan_step_id": str(step_meta.get("step_id") or "") or None,
            "plan_step_index": int(step_meta.get("plan_step_index", -1) or -1),
            "tool": _pending_tool(pending_checkpoint),
            "note": checkpoint_note,
            "actor_user_id": str(current_user.id),
        },
    )
    deps.append_approval_event(
        approval_payload,
        pending_checkpoint,
        method=method,
        user_id=current_user.id,
        note=checkpoint_note,
    )
    _clear_pending_checkpoint(
        results_payload,
        approval_payload,
        state,
        deps=deps,
    )
    target_status = (
        AgentJobStatus.PENDING.value if is_skip else AgentJobStatus.PAUSED.value
    )
    deps.append_operator_intervention(
        results_payload,
        action=action,
        actor_user_id=current_user.id,
        note=checkpoint_note,
        job_status_before=job.status,
        job_status_after=target_status,
        metadata={
            "step_id": str(step_meta.get("step_id") or ""),
            "plan_step_index": int(step_meta.get("plan_step_index", -1) or -1),
        },
    )
    job.results = results_payload
    job.status = target_status
    job.current_phase = "approval_skipped" if is_skip else "approval_rejected"
    if is_skip:
        job.phase_details = "Skipped current plan step and resumed"
    else:
        job.phase_details = str(
            checkpoint_note
            or "Checkpoint rejected. Edit, approve, skip, or resume when ready."
        )[:280]
    job.add_log_entry(
        {
            "phase": (
                "approval_checkpoint_skipped"
                if is_skip
                else "approval_checkpoint_rejected"
            ),
            "reason": method,
            "step_id": str(step_meta.get("step_id") or ""),
            "plan_step_index": int(step_meta.get("plan_step_index", -1) or -1),
            **({} if is_skip else {"note": str(checkpoint_note or "")[:300] or None}),
        }
    )
