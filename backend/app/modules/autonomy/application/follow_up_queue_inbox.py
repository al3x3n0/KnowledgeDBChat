"""Approve or reject queued research-inbox follow-up launches."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.research_inbox import ResearchInboxItem
from app.models.user import User
from app.schemas.agent_job import (
    AgentCheckpointQueueFollowUpActionResponse,
    AgentJobFromChainCreate,
)


class FollowUpQueueActionError(Exception):
    def __init__(self, *, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@dataclass(frozen=True)
class InboxFollowUpActionDependencies:
    load_learning_profile: Callable[..., Awaitable[dict[str, Any]]]
    build_follow_up_actions: Callable[..., Any]
    launch_follow_up_action: Callable[..., Awaitable[Any]]


async def perform_inbox_follow_up_queue_action(
    *,
    item: ResearchInboxItem,
    action: str,
    operator_note: str | None,
    db: AsyncSession,
    current_user: User,
    deps: InboxFollowUpActionDependencies,
) -> AgentCheckpointQueueFollowUpActionResponse:
    if str(item.status or "").strip().lower() != "accepted":
        raise FollowUpQueueActionError(
            status_code=400,
            detail="Only accepted inbox items support follow-up queue actions",
        )
    if str(item.follow_up_launch_status or "").strip().lower() != "pending_approval":
        raise FollowUpQueueActionError(
            status_code=400,
            detail="Follow-up is not currently waiting for approval",
        )
    if str(item.follow_up_operator_decision or "").strip():
        raise FollowUpQueueActionError(
            status_code=400,
            detail="Follow-up already has an operator decision",
        )

    normalized_action = str(action or "").strip().lower()
    acted_at = datetime.utcnow()
    item.follow_up_operator_acted_at = acted_at
    item.follow_up_operator_user_id = current_user.id
    item.follow_up_operator_note = (operator_note or "").strip() or None

    if normalized_action == "reject_launch":
        item.follow_up_operator_decision = "rejected"
        item.follow_up_decision = "rejected"
        item.follow_up_launch_status = "rejected"
        item.follow_up_block_reason = (
            item.follow_up_operator_note
            or "Operator rejected the queued follow-up launch."
        )
        return AgentCheckpointQueueFollowUpActionResponse(
            inbox_item_id=item.id,
            follow_up_launch_status=item.follow_up_launch_status,
            follow_up_operator_decision=item.follow_up_operator_decision,
            detail=item.follow_up_block_reason,
        )

    learning_profile = await deps.load_learning_profile(
        db=db,
        user_id=current_user.id,
        customer=str(item.customer or "").strip() or None,
    )
    actions = deps.build_follow_up_actions(
        item,
        learning_profile=learning_profile,
    )
    action_row = next(
        (
            row
            for row in actions
            if str(row.recommendation_key or "").strip()
            == str(item.follow_up_recommendation_key or "").strip()
        ),
        None,
    )
    if action_row is None:
        raise FollowUpQueueActionError(
            status_code=422,
            detail="Queued recommendation can no longer be resolved",
        )
    if str(action_row.autonomy_eligibility or "").strip().lower() != "auto_launchable":
        raise FollowUpQueueActionError(
            status_code=422,
            detail="Queued recommendation is not safe to approve-launch",
        )

    launched = await deps.launch_follow_up_action(
        action_row,
        db=db,
        current_user=current_user,
    )
    item.follow_up_operator_decision = "approved_launch"
    item.follow_up_decision = "approved_and_launched"
    item.follow_up_launch_status = "launched"
    item.follow_up_job_id = launched.id
    if action_row.chain_create_payload:
        item.follow_up_chain_definition_id = AgentJobFromChainCreate.model_validate(
            action_row.chain_create_payload
        ).chain_definition_id
    else:
        item.follow_up_chain_definition_id = None
    item.follow_up_launched_at = acted_at
    item.follow_up_block_reason = None
    item.follow_up_outcome_status = None
    item.follow_up_outcome_recorded_at = None
    item.follow_up_outcome_summary = None
    return AgentCheckpointQueueFollowUpActionResponse(
        inbox_item_id=item.id,
        follow_up_launch_status=item.follow_up_launch_status,
        follow_up_operator_decision=item.follow_up_operator_decision,
        follow_up_job_id=item.follow_up_job_id,
        follow_up_chain_definition_id=item.follow_up_chain_definition_id,
        detail="Follow-up launched from queue approval",
    )
