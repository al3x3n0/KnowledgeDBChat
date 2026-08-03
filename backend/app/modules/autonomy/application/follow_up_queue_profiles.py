"""Approve or reject queued domain-profile opportunity follow-ups."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.domain_research_profile import DomainResearchProfile
from app.models.user import User
from app.modules.autonomy.application.follow_up_queue_inbox import (
    FollowUpQueueActionError,
)
from app.schemas.agent_job import AgentCheckpointQueueFollowUpActionResponse
from app.services.autonomous_agent_executor import AutonomousAgentExecutor


@dataclass(frozen=True)
class ProfileFollowUpActionDependencies:
    build_summary_payload: Callable[..., dict[str, Any]]
    classify_operator_review: Callable[..., Any]
    sync_queue_state: Callable[..., Awaitable[None]]
    resolve_parent_job: Callable[..., Awaitable[Any]]
    execute_agent_job_task: Any


async def perform_profile_follow_up_queue_action(
    *,
    profile: DomainResearchProfile,
    opportunity_id: str | None,
    action: str,
    operator_note: str | None,
    db: AsyncSession,
    current_user: User,
    deps: ProfileFollowUpActionDependencies,
) -> AgentCheckpointQueueFollowUpActionResponse:
    normalized_opportunity_id = str(opportunity_id or "").strip()
    if not normalized_opportunity_id:
        raise FollowUpQueueActionError(
            status_code=400,
            detail="profile_opportunity_id is required",
        )
    payload = deps.build_summary_payload(profile)
    opportunities = payload["opportunities"]
    effective_policy = payload["effective_policy"]
    opportunity_index = next(
        (
            index
            for index, row in enumerate(opportunities)
            if str(row.get("opportunity_id") or "").strip() == normalized_opportunity_id
        ),
        -1,
    )
    if opportunity_index < 0:
        raise FollowUpQueueActionError(
            status_code=404,
            detail="Profile opportunity not found",
        )
    opportunity = dict(opportunities[opportunity_index])
    review = deps.classify_operator_review(
        opportunity,
        effective_policy=effective_policy,
    )
    if not review or review.get("review_type") != "follow_up_recommendation":
        raise FollowUpQueueActionError(
            status_code=400,
            detail=(
                "Profile opportunity is not currently waiting for " "follow-up approval"
            ),
        )
    acted_at = datetime.utcnow()
    opportunity["follow_up_reviewed_at"] = acted_at.isoformat()
    opportunity["follow_up_reviewed_by_user_id"] = str(current_user.id)
    opportunity["follow_up_review_note"] = (operator_note or "").strip() or None
    opportunity["updated_at"] = acted_at.isoformat()
    opportunity["decision_source"] = "operator"
    opportunity["operator_note"] = opportunity["follow_up_review_note"]
    opportunity["follow_up_review_evidence_revision"] = (
        str(
            opportunity.get("evidence_revision")
            or review.get("evidence_revision")
            or ""
        ).strip()
        or None
    )

    if str(action or "").strip().lower() == "reject_launch":
        opportunity["follow_up_review_status"] = "rejected"
        opportunity["last_decision_type"] = "follow_up_rejected"
        opportunity["last_decision_reason_code"] = "operator_rejected_follow_up"
        opportunities[opportunity_index] = opportunity
        await deps.sync_queue_state(
            profile=profile,
            opportunities=opportunities,
        )
        return AgentCheckpointQueueFollowUpActionResponse(
            domain_research_profile_id=profile.id,
            profile_opportunity_id=normalized_opportunity_id,
            follow_up_launch_status="rejected",
            follow_up_operator_decision="rejected",
            detail=opportunity["follow_up_review_note"]
            or "Operator rejected the queued follow-up launch.",
        )

    if opportunity.get("child_job_ids"):
        raise FollowUpQueueActionError(
            status_code=400,
            detail="Follow-up already launched for this opportunity",
        )
    parent_job = await deps.resolve_parent_job(db=db, profile=profile)
    executor = AutonomousAgentExecutor()
    child_job = await executor._create_domain_research_follow_up_job(
        db=db,
        job=parent_job,
        domain=profile.domain,
        objective=profile.objective,
        customer_context=str(profile.customer_context or ""),
        track_type=str(profile.track_type or "generic"),
        source_scope=str(profile.source_scope or "kb_plus_arxiv"),
        top_idea=opportunity,
        docs=[],
        repo_documents=[],
        papers=[],
        repo_source_ids=[
            str(value)
            for value in (profile.repo_source_ids or [])
            if str(value).strip()
        ],
        benchmark_queries=[
            str(value)
            for value in (profile.benchmark_queries or [])
            if str(value).strip()
        ],
        automation_profile=profile.automation_profile,
        automation_policy=effective_policy,
        sandbox_profile_id=profile.sandbox_profile_id,
        profile_id=str(profile.id),
    )
    if child_job is None:
        raise FollowUpQueueActionError(
            status_code=400,
            detail="Failed to launch follow-up job",
        )
    opportunity["child_job_ids"] = list(
        dict.fromkeys(
            [
                *[
                    str(value)
                    for value in (opportunity.get("child_job_ids") or [])
                    if str(value).strip()
                ],
                str(child_job.id),
            ]
        )
    )[:8]
    opportunity["decision_state"] = "accepted"
    opportunity["stage"] = "validating"
    opportunity["follow_up_review_status"] = "approved_launch"
    opportunity["last_decision_type"] = "follow_up_approved_launch"
    opportunity["last_decision_reason_code"] = "operator_approved_follow_up"
    opportunities[opportunity_index] = opportunity
    await deps.sync_queue_state(
        profile=profile,
        opportunities=opportunities,
    )
    deps.execute_agent_job_task.delay(str(child_job.id), str(profile.user_id))
    return AgentCheckpointQueueFollowUpActionResponse(
        domain_research_profile_id=profile.id,
        profile_opportunity_id=normalized_opportunity_id,
        follow_up_launch_status="launched",
        follow_up_operator_decision="approved_launch",
        follow_up_job_id=child_job.id,
        detail="Follow-up launched from queue approval",
    )
