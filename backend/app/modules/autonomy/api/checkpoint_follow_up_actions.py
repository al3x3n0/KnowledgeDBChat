"""Individual and bulk follow-up checkpoint action boundaries."""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.agent_job import AgentJob
from app.models.domain_research_profile import DomainResearchProfile
from app.models.research_inbox import ResearchInboxItem
from app.models.research_portfolio import ResearchPortfolio
from app.models.user import User
from app.schemas.agent_job import (
    AgentCheckpointQueueBulkFollowUpActionRequest,
    AgentCheckpointQueueBulkFollowUpActionResponse,
    AgentCheckpointQueueBulkFollowUpActionResultResponse,
    AgentCheckpointQueueFollowUpActionRequest,
    AgentCheckpointQueueFollowUpActionResponse,
)

FollowUpActionPerformer = Callable[
    ...,
    Awaitable[AgentCheckpointQueueFollowUpActionResponse],
]
DecisionEventRecorder = Callable[..., Awaitable[Any]]
ProfileParentResolver = Callable[..., Awaitable[AgentJob | None]]
PortfolioParentResolver = Callable[..., Awaitable[AgentJob | None]]
SchedulerStateExtractor = Callable[[AgentJob | None], dict[str, Any] | None]
OpportunityReasonLabel = Callable[..., str | None]
QueueReasonLabel = Callable[[str], str | None]


@dataclass(frozen=True)
class CheckpointFollowUpActionApi:
    router: APIRouter
    checkpoint_queue_follow_up_action: Callable[..., Any]
    checkpoint_queue_bulk_follow_up_action: Callable[..., Any]


def build_checkpoint_follow_up_action_api(
    *,
    router: APIRouter,
    perform_follow_up_action: FollowUpActionPerformer,
    record_decision_event: DecisionEventRecorder,
    resolve_profile_parent_job: ProfileParentResolver,
    resolve_portfolio_parent_job: PortfolioParentResolver,
    extract_scheduler_state: SchedulerStateExtractor,
    follow_up_reason_label: OpportunityReasonLabel,
    queue_reason_label: QueueReasonLabel,
) -> CheckpointFollowUpActionApi:
    """Register operator decisions for follow-up queue recommendations."""

    async def load_profile(
        *,
        profile_id,
        db: AsyncSession,
        current_user: User,
    ) -> DomainResearchProfile:
        profile = (
            await db.execute(
                select(DomainResearchProfile).where(
                    and_(
                        DomainResearchProfile.id == profile_id,
                        DomainResearchProfile.user_id == current_user.id,
                    )
                )
            )
        ).scalar_one_or_none()
        if profile is None:
            raise HTTPException(
                status_code=404,
                detail="Domain research profile not found",
            )
        return profile

    async def load_portfolio(
        *,
        portfolio_id,
        db: AsyncSession,
        current_user: User,
    ) -> ResearchPortfolio:
        portfolio = (
            await db.execute(
                select(ResearchPortfolio).where(
                    and_(
                        ResearchPortfolio.id == portfolio_id,
                        ResearchPortfolio.user_id == current_user.id,
                    )
                )
            )
        ).scalar_one_or_none()
        if portfolio is None:
            raise HTTPException(
                status_code=404,
                detail="Research portfolio not found",
            )
        return portfolio

    @router.post(
        "/checkpoint-queue/follow-up-action",
        response_model=AgentCheckpointQueueFollowUpActionResponse,
    )
    async def checkpoint_queue_follow_up_action(
        request: AgentCheckpointQueueFollowUpActionRequest,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        if request.inbox_item_id is not None:
            item = (
                await db.execute(
                    select(ResearchInboxItem).where(
                        and_(
                            ResearchInboxItem.id == request.inbox_item_id,
                            ResearchInboxItem.user_id == current_user.id,
                        )
                    )
                )
            ).scalar_one_or_none()
            if item is None:
                raise HTTPException(status_code=404, detail="Inbox item not found")
            source_job = await db.get(AgentJob, item.job_id) if item.job_id else None
            if source_job is not None and (
                str(source_job.user_id) != str(current_user.id)
            ):
                source_job = None
            scheduler_state = (
                extract_scheduler_state(source_job) if source_job is not None else None
            )
            response = await perform_follow_up_action(
                item=item,
                action=request.action,
                operator_note=request.operator_note,
                db=db,
                current_user=current_user,
            )
            reason_code = str(
                item.follow_up_recommendation_key or item.follow_up_block_reason or ""
            ).strip()
            await record_decision_event(
                db=db,
                current_user=current_user,
                action=request.action,
                operator_note=request.operator_note,
                source_kind="queue",
                source_id=str(item.id),
                source_label=str(item.title or "Research inbox item").strip(),
                customer=str(item.customer or "").strip() or None,
                reason_code=reason_code or None,
                reason_label=queue_reason_label(reason_code),
                scheduler_state=scheduler_state,
                follow_up_launch_status=response.follow_up_launch_status,
                follow_up_operator_decision=response.follow_up_operator_decision,
                deep_link={
                    "target_tab": "queue",
                    "params": {"tab": "queue"},
                    "label": "Open Checkpoint Queue",
                },
                metadata={"inbox_item_id": str(item.id)},
                after_state={
                    "follow_up_launch_status": response.follow_up_launch_status,
                    "follow_up_operator_decision": (
                        response.follow_up_operator_decision
                    ),
                },
            )
            await db.commit()
            return response

        if request.domain_research_profile_id is not None:
            profile = await load_profile(
                profile_id=request.domain_research_profile_id,
                db=db,
                current_user=current_user,
            )
            parent_job = await resolve_profile_parent_job(
                db=db,
                profile=profile,
            )
            scheduler_state = extract_scheduler_state(parent_job)
            opportunity_id = str(request.profile_opportunity_id or "").strip()
            response = await perform_follow_up_action(
                profile=profile,
                profile_opportunity_id=request.profile_opportunity_id,
                action=request.action,
                operator_note=request.operator_note,
                db=db,
                current_user=current_user,
            )
            await record_decision_event(
                db=db,
                current_user=current_user,
                action=request.action,
                operator_note=request.operator_note,
                source_kind="domain_profile",
                source_id=str(profile.id),
                source_label=str(profile.title or "Domain profile").strip(),
                customer=str(profile.customer_context or "").strip() or None,
                reason_code=opportunity_id or None,
                reason_label=follow_up_reason_label(
                    profile=profile,
                    opportunity_id=opportunity_id,
                ),
                scheduler_state=scheduler_state,
                follow_up_launch_status=response.follow_up_launch_status,
                follow_up_operator_decision=response.follow_up_operator_decision,
                deep_link={
                    "target_tab": "domain",
                    "params": {"tab": "domain"},
                    "label": "Open Domain Profiles",
                },
                metadata={"profile_opportunity_id": opportunity_id},
                after_state={
                    "profile_opportunity_id": opportunity_id,
                    "follow_up_launch_status": response.follow_up_launch_status,
                    "follow_up_operator_decision": (
                        response.follow_up_operator_decision
                    ),
                },
            )
            await db.commit()
            return response

        portfolio = await load_portfolio(
            portfolio_id=request.portfolio_id,
            db=db,
            current_user=current_user,
        )
        parent_job = await resolve_portfolio_parent_job(
            db=db,
            portfolio=portfolio,
        )
        scheduler_state = extract_scheduler_state(parent_job)
        opportunity_id = str(request.portfolio_opportunity_id or "").strip()
        response = await perform_follow_up_action(
            portfolio=portfolio,
            portfolio_opportunity_id=request.portfolio_opportunity_id,
            action=request.action,
            operator_note=request.operator_note,
            db=db,
            current_user=current_user,
        )
        await record_decision_event(
            db=db,
            current_user=current_user,
            action=request.action,
            operator_note=request.operator_note,
            source_kind="portfolio",
            source_id=str(portfolio.id),
            source_label=str(portfolio.title or "Research fleet").strip(),
            customer=None,
            reason_code=opportunity_id or None,
            reason_label=follow_up_reason_label(
                portfolio=portfolio,
                opportunity_id=opportunity_id,
            ),
            scheduler_state=scheduler_state,
            follow_up_launch_status=response.follow_up_launch_status,
            follow_up_operator_decision=response.follow_up_operator_decision,
            deep_link={
                "target_tab": "fleet",
                "params": {"tab": "fleet", "fleetId": str(portfolio.id)},
                "label": "Open Research Fleet",
            },
            metadata={"portfolio_opportunity_id": opportunity_id},
            after_state={
                "portfolio_opportunity_id": opportunity_id,
                "follow_up_launch_status": response.follow_up_launch_status,
                "follow_up_operator_decision": (response.follow_up_operator_decision),
            },
        )
        await db.commit()
        return response

    @router.post(
        "/checkpoint-queue/follow-up-bulk-action",
        response_model=AgentCheckpointQueueBulkFollowUpActionResponse,
    )
    async def checkpoint_queue_bulk_follow_up_action(
        request: AgentCheckpointQueueBulkFollowUpActionRequest,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        action = str(request.action or "").strip().lower()
        if action not in {"approve_launch", "reject_launch"}:
            raise HTTPException(
                status_code=400,
                detail="Unknown follow-up queue action",
            )

        results = []
        if request.domain_research_profile_id is not None:
            profile = await load_profile(
                profile_id=request.domain_research_profile_id,
                db=db,
                current_user=current_user,
            )
            parent_job = await resolve_profile_parent_job(
                db=db,
                profile=profile,
            )
            scheduler_state = extract_scheduler_state(parent_job)
            requested_ids = list(dict.fromkeys(request.profile_opportunity_ids))
            for opportunity_id in requested_ids:
                try:
                    response = await perform_follow_up_action(
                        profile=profile,
                        profile_opportunity_id=opportunity_id,
                        action=action,
                        operator_note=request.operator_note,
                        db=db,
                        current_user=current_user,
                    )
                    await record_decision_event(
                        db=db,
                        current_user=current_user,
                        action=action,
                        operator_note=request.operator_note,
                        source_kind="domain_profile",
                        source_id=str(profile.id),
                        source_label=str(profile.title or "Domain profile").strip(),
                        customer=(str(profile.customer_context or "").strip() or None),
                        reason_code=opportunity_id,
                        reason_label=follow_up_reason_label(
                            profile=profile,
                            opportunity_id=opportunity_id,
                        ),
                        scheduler_state=scheduler_state,
                        follow_up_launch_status=(response.follow_up_launch_status),
                        follow_up_operator_decision=(
                            response.follow_up_operator_decision
                        ),
                        deep_link={
                            "target_tab": "domain",
                            "params": {"tab": "domain"},
                            "label": "Open Domain Profiles",
                        },
                        metadata={"profile_opportunity_id": opportunity_id},
                        after_state={
                            "profile_opportunity_id": opportunity_id,
                            "follow_up_launch_status": (
                                response.follow_up_launch_status
                            ),
                            "follow_up_operator_decision": (
                                response.follow_up_operator_decision
                            ),
                        },
                    )
                    results.append(
                        AgentCheckpointQueueBulkFollowUpActionResultResponse(
                            domain_research_profile_id=profile.id,
                            profile_opportunity_id=opportunity_id,
                            ok=True,
                            follow_up_launch_status=(response.follow_up_launch_status),
                            follow_up_operator_decision=(
                                response.follow_up_operator_decision
                            ),
                            follow_up_job_id=response.follow_up_job_id,
                            detail=response.detail,
                        )
                    )
                except HTTPException as exc:
                    results.append(
                        AgentCheckpointQueueBulkFollowUpActionResultResponse(
                            domain_research_profile_id=profile.id,
                            profile_opportunity_id=opportunity_id,
                            ok=False,
                            error=str(exc.detail),
                        )
                    )
        else:
            portfolio = await load_portfolio(
                portfolio_id=request.portfolio_id,
                db=db,
                current_user=current_user,
            )
            parent_job = await resolve_portfolio_parent_job(
                db=db,
                portfolio=portfolio,
            )
            scheduler_state = extract_scheduler_state(parent_job)
            requested_ids = list(dict.fromkeys(request.portfolio_opportunity_ids))
            for opportunity_id in requested_ids:
                try:
                    response = await perform_follow_up_action(
                        portfolio=portfolio,
                        portfolio_opportunity_id=opportunity_id,
                        action=action,
                        operator_note=request.operator_note,
                        db=db,
                        current_user=current_user,
                    )
                    await record_decision_event(
                        db=db,
                        current_user=current_user,
                        action=action,
                        operator_note=request.operator_note,
                        source_kind="portfolio",
                        source_id=str(portfolio.id),
                        source_label=str(portfolio.title or "Research fleet").strip(),
                        customer=None,
                        reason_code=opportunity_id,
                        reason_label=follow_up_reason_label(
                            portfolio=portfolio,
                            opportunity_id=opportunity_id,
                        ),
                        scheduler_state=scheduler_state,
                        follow_up_launch_status=(response.follow_up_launch_status),
                        follow_up_operator_decision=(
                            response.follow_up_operator_decision
                        ),
                        deep_link={
                            "target_tab": "fleet",
                            "params": {
                                "tab": "fleet",
                                "fleetId": str(portfolio.id),
                            },
                            "label": "Open Research Fleet",
                        },
                        metadata={"portfolio_opportunity_id": opportunity_id},
                        after_state={
                            "portfolio_opportunity_id": opportunity_id,
                            "follow_up_launch_status": (
                                response.follow_up_launch_status
                            ),
                            "follow_up_operator_decision": (
                                response.follow_up_operator_decision
                            ),
                        },
                    )
                    results.append(
                        AgentCheckpointQueueBulkFollowUpActionResultResponse(
                            portfolio_id=portfolio.id,
                            portfolio_opportunity_id=opportunity_id,
                            ok=True,
                            follow_up_launch_status=(response.follow_up_launch_status),
                            follow_up_operator_decision=(
                                response.follow_up_operator_decision
                            ),
                            follow_up_job_id=response.follow_up_job_id,
                            detail=response.detail,
                        )
                    )
                except HTTPException as exc:
                    results.append(
                        AgentCheckpointQueueBulkFollowUpActionResultResponse(
                            portfolio_id=portfolio.id,
                            portfolio_opportunity_id=opportunity_id,
                            ok=False,
                            error=str(exc.detail),
                        )
                    )

        await db.commit()
        applied = sum(row.ok for row in results)
        return AgentCheckpointQueueBulkFollowUpActionResponse(
            requested_count=len(requested_ids),
            applied=applied,
            failed=len(results) - applied,
            results=results,
        )

    return CheckpointFollowUpActionApi(
        router=router,
        checkpoint_queue_follow_up_action=checkpoint_queue_follow_up_action,
        checkpoint_queue_bulk_follow_up_action=(checkpoint_queue_bulk_follow_up_action),
    )
