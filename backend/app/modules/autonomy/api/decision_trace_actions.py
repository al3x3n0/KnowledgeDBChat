"""Operator mutation boundary for autonomous decision-trace events."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.autonomy_decision_event import AutonomyDecisionEvent
from app.models.domain_research_profile import DomainResearchProfile
from app.models.research_inbox import ResearchInboxItem
from app.models.research_portfolio import ResearchPortfolio
from app.models.user import User
from app.schemas.agent_job import (
    AgentCheckpointQueueFollowUpActionResponse,
    AgentDecisionTraceActionRequest,
    AgentDecisionTraceActionResponse,
    AgentDecisionTraceEventResponse,
)
from app.services.autonomy_event_service import (
    apply_decision_trace_escalation,
    compute_decision_trace_escalation,
    event_to_trace_payload,
    maybe_emit_escalation_transition_notification,
    maybe_reopen_event_notification,
)

PersistedEventLoader = Callable[..., Awaitable[AutonomyDecisionEvent | None]]
FollowUpTargetResolver = Callable[[AutonomyDecisionEvent], tuple[str, str, str]]
FollowUpQueueAction = Callable[
    ...,
    Awaitable[AgentCheckpointQueueFollowUpActionResponse],
]
FollowUpJobResolver = Callable[[AutonomyDecisionEvent], str]
InboxFollowUpRelauncher = Callable[
    ...,
    Awaitable[AgentCheckpointQueueFollowUpActionResponse],
]
AssigneeValidator = Callable[..., Awaitable[UUID | None]]
VisibleUserIdsLoader = Callable[..., Awaitable[set[UUID]]]
TraceEventDecorator = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class DecisionTraceActionApi:
    router: APIRouter
    act_on_decision_trace_event: Callable[..., Any]


def build_decision_trace_action_api(
    *,
    router: APIRouter,
    allowed_actions: set[str],
    load_persisted_event: PersistedEventLoader,
    resolve_follow_up_target: FollowUpTargetResolver,
    perform_follow_up_queue_action: FollowUpQueueAction,
    resolve_follow_up_job_id: FollowUpJobResolver,
    relaunch_follow_up_inbox_item: InboxFollowUpRelauncher,
    validate_assignee: AssigneeValidator,
    list_visible_user_ids: VisibleUserIdsLoader,
    decorate_trace_event_payload: TraceEventDecorator,
) -> DecisionTraceActionApi:
    """Register the ownership-scoped operator action route."""

    @router.post(
        "/decision-trace/{event_id}/action",
        response_model=AgentDecisionTraceActionResponse,
    )
    async def act_on_decision_trace_event(
        event_id: UUID,
        request: AgentDecisionTraceActionRequest,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        action = str(request.action or "").strip().lower()
        if action not in allowed_actions:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Unsupported decision trace action",
            )
        event = await load_persisted_event(
            db,
            event_id=event_id,
            current_user=current_user,
        )
        if event is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Decision trace event not found",
            )
        if bool(event.is_derived):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Derived fallback events are read-only",
            )

        now = datetime.utcnow()
        note = str(request.note or "").strip() or None
        current_status = str(event.triage_status or "new").strip().lower() or "new"
        previous_escalation_state = (
            str(compute_decision_trace_escalation(event)[0] or "none").strip().lower()
            or "none"
        )

        if action in {"approve_launch", "reject_launch"}:
            source_kind, source_id, opportunity_id = resolve_follow_up_target(event)
            try:
                owner_id = UUID(source_id)
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        "Decision trace event has an invalid follow-up owner identifier"
                    ),
                ) from exc
            if source_kind == "domain_profile":
                profile = (
                    await db.execute(
                        select(DomainResearchProfile).where(
                            and_(
                                DomainResearchProfile.id == owner_id,
                                DomainResearchProfile.user_id == current_user.id,
                            )
                        )
                    )
                ).scalar_one_or_none()
                if profile is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Domain research profile not found",
                    )
                response = await perform_follow_up_queue_action(
                    profile=profile,
                    profile_opportunity_id=opportunity_id,
                    action=action,
                    operator_note=note,
                    db=db,
                    current_user=current_user,
                )
            else:
                portfolio = (
                    await db.execute(
                        select(ResearchPortfolio).where(
                            and_(
                                ResearchPortfolio.id == owner_id,
                                ResearchPortfolio.user_id == current_user.id,
                            )
                        )
                    )
                ).scalar_one_or_none()
                if portfolio is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Research portfolio not found",
                    )
                response = await perform_follow_up_queue_action(
                    portfolio=portfolio,
                    portfolio_opportunity_id=opportunity_id,
                    action=action,
                    operator_note=note,
                    db=db,
                    current_user=current_user,
                )

            approved = action == "approve_launch"
            prior_after_state = (
                event.after_state if isinstance(event.after_state, dict) else {}
            )
            next_after_state = {
                **prior_after_state,
                "opportunity_id": opportunity_id,
                "follow_up_launch_status": response.follow_up_launch_status,
                "follow_up_operator_decision": response.follow_up_operator_decision,
            }
            if response.follow_up_job_id:
                next_after_state["follow_up_job_id"] = str(response.follow_up_job_id)
            event.event_type = (
                "follow_up_approved" if approved else "follow_up_rejected"
            )
            event.decision_type = event.event_type
            event.reason_code = (
                "operator_approved_follow_up"
                if approved
                else "operator_rejected_follow_up"
            )
            event.status = str(response.follow_up_launch_status or "").strip() or None
            event.actor_mode = "operator"
            event.summary = (
                f"{str(event.source_label or 'Autonomy source').strip()}: "
                f"{'approved' if approved else 'rejected'} queued follow-up"
            )
            event.before_state = prior_after_state or event.before_state
            event.after_state = next_after_state
            event.operator_note = note or event.operator_note
            event.acknowledged_at = event.acknowledged_at or now
            event.acknowledged_by_user_id = (
                event.acknowledged_by_user_id or current_user.id
            )
            event.triage_status = "resolved"
            event.resolved_at = now
            event.resolved_by_user_id = current_user.id
            event.resolution_note = note or event.resolution_note
        elif action == "relaunch_follow_up":
            prior_after_state = (
                event.after_state if isinstance(event.after_state, dict) else {}
            )
            try:
                follow_up_job_uuid = UUID(resolve_follow_up_job_id(event))
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        "Decision trace event has an invalid follow-up job identifier"
                    ),
                ) from exc
            inbox_item = (
                await db.execute(
                    select(ResearchInboxItem).where(
                        and_(
                            ResearchInboxItem.user_id == current_user.id,
                            ResearchInboxItem.follow_up_job_id == follow_up_job_uuid,
                        )
                    )
                )
            ).scalar_one_or_none()
            if inbox_item is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        "Decision trace event could not resolve a relaunchable "
                        "inbox follow-up"
                    ),
                )
            response = await relaunch_follow_up_inbox_item(
                item=inbox_item,
                operator_note=note,
                db=db,
                current_user=current_user,
            )
            event.event_type = "follow_up_launched"
            event.decision_type = "follow_up_launched"
            event.reason_code = "operator_relaunched_follow_up"
            event.status = "active"
            event.actor_mode = "operator"
            event.summary = (
                f"{str(event.source_label or 'Autonomy source').strip()}: "
                "relaunched terminal follow-up"
            )
            event.before_state = prior_after_state or event.before_state
            event.after_state = {
                **prior_after_state,
                "follow_up_launch_status": response.follow_up_launch_status,
                "follow_up_outcome_status": None,
                "follow_up_last_job_id": str(response.follow_up_job_id or "") or None,
            }
            event.operator_note = note or event.operator_note
            event.acknowledged_at = event.acknowledged_at or now
            event.acknowledged_by_user_id = (
                event.acknowledged_by_user_id or current_user.id
            )
            event.triage_status = "resolved"
            event.resolved_at = now
            event.resolved_by_user_id = current_user.id
            event.resolution_note = note or event.resolution_note
        elif action == "acknowledge":
            event.triage_status = "acknowledged"
            event.acknowledged_at = now
            event.acknowledged_by_user_id = current_user.id
        elif action == "start_investigation":
            event.triage_status = "investigating"
            event.acknowledged_at = event.acknowledged_at or now
            event.acknowledged_by_user_id = (
                event.acknowledged_by_user_id or current_user.id
            )
        elif action == "resolve":
            event.triage_status = "resolved"
            event.acknowledged_at = event.acknowledged_at or now
            event.acknowledged_by_user_id = (
                event.acknowledged_by_user_id or current_user.id
            )
            event.resolved_at = now
            event.resolved_by_user_id = current_user.id
            event.resolution_note = note or event.resolution_note
        elif action == "reopen":
            event.triage_status = "new"
            event.resolved_at = None
            event.resolved_by_user_id = None
            event.resolution_note = None
            if note:
                event.operator_note = note
            if current_status == "resolved":
                await maybe_reopen_event_notification(db, event)
        elif action == "toggle_pin":
            event.pinned = not bool(event.pinned)
        elif action == "assign":
            assignee_id = await validate_assignee(
                db,
                current_user=current_user,
                assigned_to_user_id=request.assigned_to_user_id,
            )
            if assignee_id is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Valid assignee is required",
                )
            event.assigned_to_user_id = assignee_id
            event.assigned_at = now
            event.assigned_by_user_id = current_user.id
        elif action == "unassign":
            event.assigned_to_user_id = None
            event.assigned_at = None
            event.assigned_by_user_id = None
        elif action == "set_due_at":
            if request.due_at is None:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Due date is required",
                )
            event.due_at = request.due_at
        elif action == "clear_due_at":
            event.due_at = None

        event.last_viewed_at = now
        event.updated_at = now
        if note and action not in {"resolve", "reopen"}:
            event.operator_note = note
        apply_decision_trace_escalation(event, now=now)
        await db.commit()
        await maybe_emit_escalation_transition_notification(
            db,
            event,
            previous_state=previous_escalation_state,
        )
        await db.commit()
        await db.refresh(event)

        visible_user_ids = await list_visible_user_ids(
            db,
            current_user=current_user,
        )
        if current_user.id not in visible_user_ids:
            visible_user_ids.add(current_user.id)
        visible_users = list(
            (await db.execute(select(User).where(User.id.in_(visible_user_ids))))
            .scalars()
            .all()
        )
        user_lookup = {str(user.id): user for user in visible_users}
        return AgentDecisionTraceActionResponse(
            event=AgentDecisionTraceEventResponse.model_validate(
                decorate_trace_event_payload(
                    event_to_trace_payload(event),
                    user_lookup=user_lookup,
                    current_user_id=current_user.id,
                )
            )
        )

    return DecisionTraceActionApi(
        router=router,
        act_on_decision_trace_event=act_on_decision_trace_event,
    )
