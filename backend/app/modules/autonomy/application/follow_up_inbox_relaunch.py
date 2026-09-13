"""Relaunch failed or cancelled research-inbox follow-up work."""

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

from .follow_up_queue_inbox import FollowUpQueueActionError


@dataclass(frozen=True)
class InboxFollowUpRelaunchDependencies:
    load_learning_profile: Callable[..., Awaitable[dict[str, Any]]]
    build_follow_up_actions: Callable[..., Any]
    launch_follow_up_action: Callable[..., Awaitable[Any]]
    project_relaunch_to_originating_opportunity: Callable[..., Awaitable[None]]


async def relaunch_inbox_follow_up(
    *,
    item: ResearchInboxItem,
    operator_note: str | None,
    db: AsyncSession,
    current_user: User,
    deps: InboxFollowUpRelaunchDependencies,
) -> AgentCheckpointQueueFollowUpActionResponse:
    """Validate and relaunch the item's stored, still-safe recommendation."""
    if str(item.status or "").strip().lower() != "accepted":
        raise FollowUpQueueActionError(
            status_code=400,
            detail="Only accepted inbox items can relaunch a follow-up",
        )

    outcome_status = str(item.follow_up_outcome_status or "").strip().lower()
    launch_status = str(item.follow_up_launch_status or "").strip().lower()
    if outcome_status not in {"failed", "cancelled"} or launch_status != "launched":
        raise FollowUpQueueActionError(
            status_code=400,
            detail="Only failed or cancelled launched follow-ups can be relaunched",
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
            detail="Stored follow-up recommendation can no longer be resolved",
        )
    if str(action_row.autonomy_eligibility or "").strip().lower() != "auto_launchable":
        raise FollowUpQueueActionError(
            status_code=422,
            detail="Stored follow-up recommendation is no longer safe to relaunch",
        )

    launched = await deps.launch_follow_up_action(
        action_row,
        db=db,
        current_user=current_user,
    )
    item.follow_up_decision = "relaunched"
    item.follow_up_launch_status = "launched"
    item.follow_up_job_id = launched.id
    item.follow_up_chain_definition_id = None
    if action_row.chain_create_payload:
        item.follow_up_chain_definition_id = AgentJobFromChainCreate.model_validate(
            action_row.chain_create_payload
        ).chain_definition_id
    launched_at = datetime.utcnow()
    item.follow_up_launched_at = launched_at
    item.follow_up_block_reason = (operator_note or "").strip() or None
    item.follow_up_budget_decision = None
    item.follow_up_budget_reason = None
    item.follow_up_budget_throttle_state = None
    item.follow_up_customer_budget_decision = None
    item.follow_up_customer_budget_reason = None
    item.follow_up_customer_budget_throttle_state = None
    item.follow_up_outcome_status = None
    item.follow_up_outcome_recorded_at = None
    item.follow_up_outcome_summary = None
    await deps.project_relaunch_to_originating_opportunity(
        db=db,
        job=launched,
        launched_at=launched_at,
    )
    return AgentCheckpointQueueFollowUpActionResponse(
        inbox_item_id=item.id,
        follow_up_launch_status=item.follow_up_launch_status,
        follow_up_operator_decision=item.follow_up_operator_decision,
        follow_up_job_id=item.follow_up_job_id,
        follow_up_chain_definition_id=item.follow_up_chain_definition_id,
        detail="Follow-up relaunched",
    )
