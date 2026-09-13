"""Dispatch follow-up queue decisions to inbox, portfolio, or profile handlers."""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.domain_research_profile import DomainResearchProfile
from app.models.research_inbox import ResearchInboxItem
from app.models.research_portfolio import ResearchPortfolio
from app.models.user import User
from app.schemas.agent_job import AgentCheckpointQueueFollowUpActionResponse

from . import (
    follow_up_queue_inbox,
    follow_up_queue_portfolios,
    follow_up_queue_profiles,
)


@dataclass(frozen=True)
class FollowUpQueueDispatcherDependencies:
    load_learning_profile: Callable[..., Awaitable[dict[str, Any]]]
    build_follow_up_actions: Callable[..., Any]
    launch_follow_up_action: Callable[..., Awaitable[Any]]
    build_portfolio_summary: Callable[..., dict[str, Any]]
    build_profile_summary: Callable[..., dict[str, Any]]
    classify_operator_review: Callable[..., Any]
    sync_portfolio_queue_state: Callable[..., Awaitable[None]]
    sync_profile_queue_state: Callable[..., Awaitable[None]]
    resolve_portfolio_parent_job: Callable[..., Awaitable[Any]]
    resolve_profile_parent_job: Callable[..., Awaitable[Any]]
    execute_agent_job_task: Any


async def dispatch_follow_up_queue_action(
    *,
    action: str,
    operator_note: str | None,
    db: AsyncSession,
    current_user: User,
    deps: FollowUpQueueDispatcherDependencies,
    item: ResearchInboxItem | None = None,
    portfolio: ResearchPortfolio | None = None,
    portfolio_opportunity_id: str | None = None,
    profile: DomainResearchProfile | None = None,
    profile_opportunity_id: str | None = None,
) -> AgentCheckpointQueueFollowUpActionResponse:
    """Validate one target and dispatch its normalized queue decision."""
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in {"approve_launch", "reject_launch"}:
        raise follow_up_queue_inbox.FollowUpQueueActionError(
            status_code=400,
            detail="Unknown follow-up queue action",
        )
    target_count = sum(1 for target in (item, portfolio, profile) if target is not None)
    if target_count > 1:
        raise follow_up_queue_inbox.FollowUpQueueActionError(
            status_code=400,
            detail="Queue action target is ambiguous",
        )
    if target_count == 0:
        raise follow_up_queue_inbox.FollowUpQueueActionError(
            status_code=400,
            detail="Queue action target is required",
        )

    if portfolio is not None:
        return (
            await follow_up_queue_portfolios.perform_portfolio_follow_up_queue_action(
                portfolio=portfolio,
                opportunity_id=portfolio_opportunity_id,
                action=normalized_action,
                operator_note=operator_note,
                db=db,
                current_user=current_user,
                deps=follow_up_queue_portfolios.PortfolioFollowUpActionDependencies(
                    build_summary_payload=deps.build_portfolio_summary,
                    classify_operator_review=deps.classify_operator_review,
                    sync_queue_state=deps.sync_portfolio_queue_state,
                    resolve_parent_job=deps.resolve_portfolio_parent_job,
                    execute_agent_job_task=deps.execute_agent_job_task,
                ),
            )
        )
    if profile is not None:
        return await follow_up_queue_profiles.perform_profile_follow_up_queue_action(
            profile=profile,
            opportunity_id=profile_opportunity_id,
            action=normalized_action,
            operator_note=operator_note,
            db=db,
            current_user=current_user,
            deps=follow_up_queue_profiles.ProfileFollowUpActionDependencies(
                build_summary_payload=deps.build_profile_summary,
                classify_operator_review=deps.classify_operator_review,
                sync_queue_state=deps.sync_profile_queue_state,
                resolve_parent_job=deps.resolve_profile_parent_job,
                execute_agent_job_task=deps.execute_agent_job_task,
            ),
        )

    assert item is not None
    return await follow_up_queue_inbox.perform_inbox_follow_up_queue_action(
        item=item,
        action=normalized_action,
        operator_note=operator_note,
        db=db,
        current_user=current_user,
        deps=follow_up_queue_inbox.InboxFollowUpActionDependencies(
            load_learning_profile=deps.load_learning_profile,
            build_follow_up_actions=deps.build_follow_up_actions,
            launch_follow_up_action=deps.launch_follow_up_action,
        ),
    )
