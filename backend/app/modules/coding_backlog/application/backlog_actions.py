"""What an operator can do to a coding backlog item.

Thirteen actions in three groups: who the item belongs to (assign, clear,
note), its life (start, resume, pause, cancel, close), and a decision about one
slice of it (apply the proposal, open a patch PR, keep it as a proposal,
relaunch, skip). This was one 650-line route handler; each action is a
function here, and ``ACTIONS`` says in one place who may do it and whether it
is about a slice.

Nothing here knows about HTTP. A refusal is an ``ActionRefused`` with a kind,
which the endpoint turns into a status. ``perform`` changes the item and adds
what it creates to the session; committing is the caller's.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.code_patch_proposal import CodePatchProposal
from app.models.coding_backlog import CodingBacklogItem
from app.models.patch_pr import PatchPR
from app.models.user import User
from app.modules.coding_backlog.application.backlog_jobs import (
    spawn_slice_apply_job,
    spawn_slice_repair_job,
)
from app.modules.coding_backlog.application.backlog_slices import (
    append_manual_promotion_history,
    clear_waiting_metadata,
    normalize_closure_reason,
    normalize_collaboration,
    set_latest_summary,
)
from app.modules.coding_backlog.application.errors import ActionRefused
from app.services.coding_backlog_decomposition import (
    append_artifact_history,
    append_backlog_timeline,
    append_lineage_id,
    append_slice_timeline,
    append_unique,
    find_slice,
    timeline_entry,
    upsert_promotion_decision,
)
from app.services.coding_backlog_items import (
    create_orchestrator_job,
    normalize_item_decomposition,
    recompute_portfolio_progress,
)


@dataclass
class ActionContext:
    """One action on one item, with what every action needs already read."""

    db: AsyncSession
    item: CodingBacklogItem
    actor_id: UUID
    decomposition: Dict[str, Any]
    collaboration: Dict[str, Any]
    slice_state: Optional[Dict[str, Any]] = None
    note: Optional[str] = None
    closure_reason: Optional[str] = None
    assigned_user_id: Optional[UUID] = None


async def _assign_backlog(ctx: ActionContext) -> None:
    item = ctx.item
    db = ctx.db
    decomposition = ctx.decomposition
    collaboration = ctx.collaboration
    operator_note = ctx.note
    assigned_user_id = str(ctx.assigned_user_id or ctx.actor_id).strip()
    try:
        assigned_user = await db.get(User, UUID(assigned_user_id))
    except Exception:
        assigned_user = None
    if assigned_user is None or not bool(getattr(assigned_user, "is_active", False)):
        raise ActionRefused("unknown_user", detail="Assigned user not found")
    item.assigned_user_id = assigned_user.id
    item.assigned_by_user_id = ctx.actor_id
    item.assigned_at = datetime.utcnow()
    collaboration = normalize_collaboration(
        {
            **collaboration,
            "visibility": "shared"
            if bool(collaboration.get("shared_with_user_ids"))
            or str(item.user_id) != assigned_user_id
            else collaboration.get("visibility"),
            "assigned_user_id": assigned_user_id,
            "assigned_by_user_id": str(ctx.actor_id),
            "assigned_at": item.assigned_at.isoformat(),
            "shared_with_user_ids": [
                *list(collaboration.get("shared_with_user_ids") or []),
                assigned_user_id,
            ],
            "note": operator_note or collaboration.get("note"),
        },
        fallback_owner_user_id=str(item.user_id),
    )
    item.visibility = str(
        collaboration.get("visibility") or item.visibility or "private"
    )
    item.shared_with_user_ids = (
        list(collaboration.get("shared_with_user_ids") or []) or None
    )
    item.collaboration = collaboration
    append_backlog_timeline(
        decomposition,
        timeline_entry(
            actor="user",
            action="assign_backlog",
            note=operator_note,
            metadata={"assigned_user_id": assigned_user_id},
        ),
    )
    item.decomposition = decomposition


async def _clear_backlog_assignment(ctx: ActionContext) -> None:
    item = ctx.item
    decomposition = ctx.decomposition
    collaboration = ctx.collaboration
    operator_note = ctx.note
    item.assigned_user_id = None
    item.assigned_by_user_id = None
    item.assigned_at = None
    collaboration = normalize_collaboration(
        {
            **collaboration,
            "assigned_user_id": None,
            "assigned_by_user_id": None,
            "assigned_at": None,
            "note": operator_note or collaboration.get("note"),
        },
        fallback_owner_user_id=str(item.user_id),
    )
    item.collaboration = collaboration
    append_backlog_timeline(
        decomposition,
        timeline_entry(
            actor="user", action="clear_backlog_assignment", note=operator_note
        ),
    )
    item.decomposition = decomposition


async def _update_backlog_note(ctx: ActionContext) -> None:
    item = ctx.item
    decomposition = ctx.decomposition
    collaboration = ctx.collaboration
    operator_note = ctx.note
    collaboration = normalize_collaboration(
        {
            **collaboration,
            "note": operator_note,
        },
        fallback_owner_user_id=str(item.user_id),
    )
    item.collaboration = collaboration
    # A new dict, not the stored one edited in place: assigning a JSON column
    # the object it already holds is no change as far as the session can
    # tell, so a second note was never written and the first one stayed.
    summary = dict(item.latest_summary) if isinstance(item.latest_summary, dict) else {}
    summary["operator_note"] = operator_note
    item.latest_summary = summary
    append_backlog_timeline(
        decomposition,
        timeline_entry(actor="user", action="update_backlog_note", note=operator_note),
    )
    item.decomposition = decomposition


async def _start(ctx: ActionContext) -> None:
    item = ctx.item
    db = ctx.db
    await create_orchestrator_job(item, db=db, start_immediately=True)


async def _pause(ctx: ActionContext) -> None:
    item = ctx.item
    decomposition = ctx.decomposition
    operator_note = ctx.note
    previous_status = str(item.status or "").strip() or None
    item.status = "paused"
    item.updated_at = datetime.utcnow()
    append_backlog_timeline(
        decomposition,
        timeline_entry(
            actor="user",
            action="pause",
            previous_status=previous_status,
            new_status="paused",
            note=operator_note,
        ),
    )
    item.decomposition = decomposition


async def _cancel(ctx: ActionContext) -> None:
    item = ctx.item
    decomposition = ctx.decomposition
    operator_note = ctx.note
    closure_reason = ctx.closure_reason
    if not closure_reason:
        raise ActionRefused("invalid", detail="closure_reason is required for cancel")
    previous_status = str(item.status or "").strip() or None
    item.status = "cancelled"
    item.completed_at = datetime.utcnow()
    item.updated_at = datetime.utcnow()
    append_backlog_timeline(
        decomposition,
        timeline_entry(
            actor="user",
            action="cancel",
            previous_status=previous_status,
            new_status="cancelled",
            note=operator_note,
            metadata={"closure_reason": closure_reason},
        ),
    )
    item.decomposition = decomposition
    set_latest_summary(
        item,
        decomposition,
        status_value="cancelled",
        note=operator_note,
        extra={"closure_reason": closure_reason},
    )


async def _close(ctx: ActionContext) -> None:
    item = ctx.item
    decomposition = ctx.decomposition
    operator_note = ctx.note
    closure_reason = ctx.closure_reason
    if not closure_reason:
        raise ActionRefused("invalid", detail="closure_reason is required for close")
    previous_status = str(item.status or "").strip() or None
    item.status = (
        "completed"
        if closure_reason in {"fixed_through_backlog", "promoted_to_repair"}
        else "cancelled"
    )
    item.completed_at = datetime.utcnow()
    item.updated_at = datetime.utcnow()
    append_backlog_timeline(
        decomposition,
        timeline_entry(
            actor="user",
            action="close",
            previous_status=previous_status,
            new_status=item.status,
            note=operator_note,
            metadata={"closure_reason": closure_reason},
        ),
    )
    item.decomposition = decomposition
    set_latest_summary(
        item,
        decomposition,
        status_value="closed",
        note=operator_note,
        extra={"closure_reason": closure_reason},
    )


async def _apply_override(ctx: ActionContext) -> None:
    item = ctx.item
    db = ctx.db
    decomposition = ctx.decomposition
    slice_state = ctx.slice_state
    operator_note = ctx.note
    proposal_id = str(
        slice_state.get("selected_proposal_id") or item.latest_proposal_id or ""
    ).strip()
    proposal_uuid = None
    try:
        proposal_uuid = UUID(proposal_id)
    except Exception:
        raise ActionRefused(
            "invalid", detail="Slice does not have a valid proposal to apply"
        )
    proposal = await db.get(CodePatchProposal, proposal_uuid)
    if not proposal or proposal.user_id != ctx.actor_id:
        raise ActionRefused("not_found", detail="Proposal not found")
    apply_job = await spawn_slice_apply_job(
        item,
        slice_state,
        db=db,
        proposal_id=proposal_id,
        operator_note=operator_note,
    )
    item.child_job_ids = append_unique(item.child_job_ids, str(apply_job.id))
    item.current_job_id = apply_job.id
    item.latest_apply_job_id = apply_job.id
    item.status = "running"
    previous_status = str(slice_state.get("status") or "").strip() or None
    slice_state["status"] = "applying"
    slice_state["apply_job_id"] = str(apply_job.id)
    slice_state["operator_decision"] = "apply_override"
    slice_state["operator_note"] = operator_note
    slice_state["operator_acted_at"] = datetime.utcnow().isoformat()
    clear_waiting_metadata(slice_state)
    append_slice_timeline(
        slice_state,
        timeline_entry(
            actor="user",
            action="apply_override_confirmed",
            previous_status=previous_status,
            new_status="applying",
            note=operator_note,
            related_job_id=str(apply_job.id),
            related_proposal_id=proposal_id,
        ),
    )
    append_backlog_timeline(
        decomposition,
        timeline_entry(
            actor="user",
            action="apply_override",
            previous_status="awaiting_operator",
            new_status="running",
            note=operator_note,
            related_job_id=str(apply_job.id),
            related_proposal_id=proposal_id,
        ),
    )
    decomposition["active_slice_id"] = slice_state.get("slice_id")
    decomposition["portfolio_progress"] = recompute_portfolio_progress(decomposition)
    item.decomposition = decomposition
    set_latest_summary(
        item,
        decomposition,
        status_value="apply_started",
        slice_state=slice_state,
        extra={
            "current_child_job_id": str(apply_job.id),
            "promotion_decision": "auto_applied",
            "selected_proposal_id": proposal_id,
            "waiting_on_operator_action": False,
        },
    )


async def _create_patch_pr(ctx: ActionContext) -> None:
    item = ctx.item
    db = ctx.db
    decomposition = ctx.decomposition
    slice_state = ctx.slice_state
    operator_note = ctx.note
    proposal_id = str(
        slice_state.get("selected_proposal_id") or item.latest_proposal_id or ""
    ).strip()
    proposal_uuid = None
    try:
        proposal_uuid = UUID(proposal_id)
    except Exception:
        raise ActionRefused(
            "invalid",
            detail="Slice does not have a valid proposal to promote",
        )
    proposal = await db.get(CodePatchProposal, proposal_uuid)
    if not proposal or proposal.user_id != ctx.actor_id:
        raise ActionRefused("not_found", detail="Proposal not found")
    pr = PatchPR(
        user_id=ctx.actor_id,
        source_id=proposal.source_id,
        title=f"{item.title}: {str(slice_state.get('title') or 'Patch')}"[:500],
        description=operator_note,
        status="draft",
        selected_proposal_id=proposal.id,
        proposal_ids=[str(proposal.id)],
        approvals=[],
        checks={
            "coding_backlog": {
                "item_id": str(item.id),
                "slice_id": str(slice_state.get("slice_id") or ""),
                "operator_note": operator_note,
            }
        },
    )
    db.add(pr)
    await db.flush()
    previous_status = str(slice_state.get("status") or "").strip() or None
    slice_state["status"] = "patch_pr"
    slice_state["promotion_decision"] = "patch_pr"
    slice_state["patch_pr_id"] = str(pr.id)
    slice_state["operator_decision"] = "create_patch_pr"
    slice_state["operator_note"] = operator_note
    slice_state["operator_acted_at"] = datetime.utcnow().isoformat()
    slice_state["completed_at"] = (
        slice_state.get("completed_at") or datetime.utcnow().isoformat()
    )
    clear_waiting_metadata(slice_state)
    append_slice_timeline(
        slice_state,
        timeline_entry(
            actor="user",
            action="create_patch_pr",
            previous_status=previous_status,
            new_status="patch_pr",
            note=operator_note,
            related_proposal_id=proposal_id,
            related_patch_pr_id=str(pr.id),
        ),
    )
    append_lineage_id(slice_state, "patch_pr_ids", str(pr.id))
    append_lineage_id(slice_state, "proposal_ids", proposal_id)
    append_artifact_history(slice_state, "patch_pr", str(pr.id), "Patch PR")
    append_artifact_history(slice_state, "proposal", proposal_id, "Selected proposal")
    append_manual_promotion_history(
        slice_state,
        action="create_patch_pr",
        operator_note=operator_note,
        proposal_id=proposal_id,
        patch_pr_id=str(pr.id),
    )
    append_backlog_timeline(
        decomposition,
        timeline_entry(
            actor="user",
            action="create_patch_pr",
            previous_status="awaiting_operator",
            new_status="completed",
            note=operator_note,
            related_proposal_id=proposal_id,
            related_patch_pr_id=str(pr.id),
        ),
    )
    decomposition["completed_slices"] = append_unique(
        decomposition.get("completed_slices"), slice_state.get("slice_id")
    )
    upsert_promotion_decision(
        decomposition,
        {
            "slice_id": str(slice_state.get("slice_id") or ""),
            "title": str(slice_state.get("title") or ""),
            "decision": "patch_pr",
            "proposal_id": proposal_id,
            "blocked_reason": str(slice_state.get("blocked_reason") or "").strip()
            or None,
            "proposal_confidence": float(
                slice_state.get("proposal_confidence", 0.0) or 0.0
            ),
            "patch_pr_id": str(pr.id),
        },
    )
    decomposition["active_slice_id"] = None
    decomposition["portfolio_progress"] = recompute_portfolio_progress(decomposition)
    item.decomposition = decomposition
    item.status = "completed"
    item.completed_at = datetime.utcnow()
    set_latest_summary(
        item,
        decomposition,
        status_value="completed",
        slice_state=slice_state,
        note="Operator created a patch PR for this slice.",
        extra={
            "promotion_decision": "patch_pr",
            "selected_proposal_id": proposal_id,
            "patch_pr_id": str(pr.id),
            "waiting_on_operator_action": False,
        },
    )


async def _keep_proposal_only(ctx: ActionContext) -> None:
    item = ctx.item
    decomposition = ctx.decomposition
    slice_state = ctx.slice_state
    operator_note = ctx.note
    proposal_id = (
        str(
            slice_state.get("selected_proposal_id") or item.latest_proposal_id or ""
        ).strip()
        or None
    )
    previous_status = str(slice_state.get("status") or "").strip() or None
    slice_state["status"] = "proposal_only"
    slice_state["promotion_decision"] = "proposal_only"
    slice_state["operator_decision"] = "keep_proposal_only"
    slice_state["operator_note"] = operator_note
    slice_state["operator_acted_at"] = datetime.utcnow().isoformat()
    slice_state["completed_at"] = (
        slice_state.get("completed_at") or datetime.utcnow().isoformat()
    )
    clear_waiting_metadata(slice_state)
    append_slice_timeline(
        slice_state,
        timeline_entry(
            actor="user",
            action="keep_proposal_only",
            previous_status=previous_status,
            new_status="proposal_only",
            note=operator_note,
            related_proposal_id=proposal_id,
        ),
    )
    append_lineage_id(slice_state, "proposal_ids", proposal_id)
    append_artifact_history(slice_state, "proposal", proposal_id, "Selected proposal")
    append_manual_promotion_history(
        slice_state,
        action="keep_proposal_only",
        operator_note=operator_note,
        proposal_id=proposal_id,
    )
    append_backlog_timeline(
        decomposition,
        timeline_entry(
            actor="user",
            action="keep_proposal_only",
            previous_status="awaiting_operator",
            new_status="completed",
            note=operator_note,
            related_proposal_id=proposal_id,
        ),
    )
    decomposition["completed_slices"] = append_unique(
        decomposition.get("completed_slices"), slice_state.get("slice_id")
    )
    upsert_promotion_decision(
        decomposition,
        {
            "slice_id": str(slice_state.get("slice_id") or ""),
            "title": str(slice_state.get("title") or ""),
            "decision": "proposal_only",
            "proposal_id": proposal_id,
            "blocked_reason": str(slice_state.get("blocked_reason") or "").strip()
            or None,
            "proposal_confidence": float(
                slice_state.get("proposal_confidence", 0.0) or 0.0
            ),
        },
    )
    decomposition["active_slice_id"] = None
    decomposition["portfolio_progress"] = recompute_portfolio_progress(decomposition)
    item.decomposition = decomposition
    item.status = "completed"
    item.completed_at = datetime.utcnow()
    set_latest_summary(
        item,
        decomposition,
        status_value="completed",
        slice_state=slice_state,
        note="Operator kept this slice as a reviewable proposal only.",
        extra={
            "promotion_decision": "proposal_only",
            "selected_proposal_id": proposal_id,
            "waiting_on_operator_action": False,
        },
    )


async def _relaunch_slice(ctx: ActionContext) -> None:
    item = ctx.item
    db = ctx.db
    decomposition = ctx.decomposition
    slice_state = ctx.slice_state
    operator_note = ctx.note
    previous_job_id = str(slice_state.get("child_job_id") or "").strip() or None
    repair_job = await spawn_slice_repair_job(
        item, slice_state, db=db, operator_note=operator_note
    )
    item.child_job_ids = append_unique(item.child_job_ids, str(repair_job.id))
    item.current_job_id = repair_job.id
    item.status = "running"
    slice_state["status"] = "retrying"
    slice_state["retry_count"] = max(0, int(slice_state.get("retry_count", 0) or 0)) + 1
    slice_state["child_job_id"] = str(repair_job.id)
    slice_state["operator_decision"] = "relaunch_slice"
    slice_state["operator_note"] = operator_note
    slice_state["operator_acted_at"] = datetime.utcnow().isoformat()
    clear_waiting_metadata(slice_state)
    append_lineage_id(slice_state, "retry_from_job_ids", previous_job_id)
    append_backlog_timeline(
        decomposition,
        timeline_entry(
            actor="user",
            action="relaunch_slice",
            previous_status="awaiting_operator",
            new_status="running",
            note=operator_note,
            related_job_id=str(repair_job.id),
            metadata={"retried_from_job_id": previous_job_id},
        ),
    )
    decomposition["active_slice_id"] = slice_state.get("slice_id")
    decomposition["portfolio_progress"] = recompute_portfolio_progress(decomposition)
    item.decomposition = decomposition
    set_latest_summary(
        item,
        decomposition,
        status_value="repair_started",
        slice_state=slice_state,
        extra={
            "current_child_job_id": str(repair_job.id),
            # The run it retries. This read the slice's child job after it
            # had been replaced, and so named the run just started.
            "retry_from_job_id": previous_job_id or "",
            "waiting_on_operator_action": False,
        },
    )


async def _skip_slice(ctx: ActionContext) -> None:
    item = ctx.item
    db = ctx.db
    decomposition = ctx.decomposition
    slice_state = ctx.slice_state
    operator_note = ctx.note
    previous_status = str(slice_state.get("status") or "").strip() or None
    slice_state["status"] = "deferred"
    slice_state["operator_decision"] = "skip_slice"
    slice_state["operator_note"] = operator_note
    slice_state["operator_acted_at"] = datetime.utcnow().isoformat()
    clear_waiting_metadata(slice_state)
    append_slice_timeline(
        slice_state,
        timeline_entry(
            actor="user",
            action="skip_slice",
            previous_status=previous_status,
            new_status="deferred",
            note=operator_note,
        ),
    )
    append_manual_promotion_history(
        slice_state, action="skip_slice", operator_note=operator_note
    )
    append_backlog_timeline(
        decomposition,
        timeline_entry(
            actor="user",
            action="skip_slice",
            previous_status="awaiting_operator",
            new_status="running",
            note=operator_note,
        ),
    )
    decomposition["active_slice_id"] = None
    decomposition["portfolio_progress"] = recompute_portfolio_progress(decomposition)
    next_slice = None
    for row in decomposition.get("planned_slices") or []:
        if str((row or {}).get("status") or "").strip().lower() == "pending":
            next_slice = row
            break
    if next_slice is not None:
        repair_job = await spawn_slice_repair_job(
            item, next_slice, db=db, operator_note=operator_note
        )
        item.child_job_ids = append_unique(item.child_job_ids, str(repair_job.id))
        item.current_job_id = repair_job.id
        item.status = "running"
        next_slice["status"] = "repairing"
        next_slice["child_job_id"] = str(repair_job.id)
        decomposition["active_slice_id"] = next_slice.get("slice_id")
        decomposition["portfolio_progress"] = recompute_portfolio_progress(
            decomposition
        )
        item.decomposition = decomposition
        set_latest_summary(
            item,
            decomposition,
            status_value="repair_started",
            slice_state=next_slice,
            note="Previous slice deferred by operator; continuing with next pending slice.",
            extra={
                "current_child_job_id": str(repair_job.id),
                "waiting_on_operator_action": False,
            },
        )
    else:
        item.decomposition = decomposition
        item.status = "completed"
        item.completed_at = datetime.utcnow()
        set_latest_summary(
            item,
            decomposition,
            status_value="completed",
            slice_state=slice_state,
            note="Slice deferred by operator; no further pending slices remain.",
            extra={"waiting_on_operator_action": False},
        )


@dataclass(frozen=True)
class Action:
    run: Callable[[ActionContext], Awaitable[None]]
    #: Only the item's owner, not somebody it is shared with.
    owner_only: bool = False
    #: A decision about one slice, which must be named and must exist.
    on_slice: bool = False
    #: Allowed on a slice nobody is waiting on.
    any_slice_state: bool = False


ACTIONS: Dict[str, Action] = {
    "assign_backlog": Action(_assign_backlog),
    "clear_backlog_assignment": Action(_clear_backlog_assignment),
    "update_backlog_note": Action(_update_backlog_note),
    "start": Action(_start),
    "resume": Action(_start),
    "pause": Action(_pause),
    "cancel": Action(_cancel, owner_only=True),
    "close": Action(_close, owner_only=True),
    "apply_override": Action(_apply_override, owner_only=True, on_slice=True),
    "create_patch_pr": Action(_create_patch_pr, owner_only=True, on_slice=True),
    "keep_proposal_only": Action(_keep_proposal_only, owner_only=True, on_slice=True),
    "relaunch_slice": Action(
        _relaunch_slice, owner_only=True, on_slice=True, any_slice_state=True
    ),
    "skip_slice": Action(
        _skip_slice, owner_only=True, on_slice=True, any_slice_state=True
    ),
}


async def perform(
    db: AsyncSession,
    item: CodingBacklogItem,
    actor_id: UUID,
    *,
    action: Any,
    slice_id: Any = None,
    operator_note: Any = None,
    closure_reason: Any = None,
    assigned_user_id: Optional[UUID] = None,
) -> None:
    """Apply one action to an item the actor is already known to see.

    The checks run in the order an operator would want them explained: is
    this an action, may you do it, which slice, is that slice waiting.
    """
    name = str(action or "").strip().lower()
    chosen = ACTIONS.get(name)
    if chosen is None:
        raise ActionRefused("invalid", "Unsupported action")
    decomposition = normalize_item_decomposition(item)
    slice_state = find_slice(decomposition, str(slice_id or "").strip() or None)

    if chosen.owner_only and str(item.user_id) != str(actor_id):
        raise ActionRefused(
            "forbidden", "Only the backlog owner can perform this action"
        )
    if chosen.on_slice and slice_state is None:
        raise ActionRefused("invalid", "slice_id is required for this action")
    if (
        slice_state is not None
        and not slice_state.get("allowed_slice_actions")
        and not bool(slice_state.get("awaiting_operator_action"))
        and not chosen.any_slice_state
    ):
        raise ActionRefused("conflict", "Slice is not awaiting operator action")

    await chosen.run(
        ActionContext(
            db=db,
            item=item,
            actor_id=actor_id,
            decomposition=decomposition,
            collaboration=normalize_collaboration(
                item.collaboration, fallback_owner_user_id=str(item.user_id)
            ),
            slice_state=slice_state,
            note=str(operator_note or "").strip() or None,
            closure_reason=normalize_closure_reason(closure_reason),
            assigned_user_id=assigned_user_id,
        )
    )


__all__ = ["ACTIONS", "Action", "ActionContext", "ActionRefused", "perform"]
