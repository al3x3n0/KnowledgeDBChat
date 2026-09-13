"""Project domain-profile opportunity reviews into checkpoint queue rows."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from app.models.domain_research_profile import DomainResearchProfile
from app.schemas.agent_job import (
    AgentCheckpointQueueActionResponse,
    AgentCheckpointQueueItemResponse,
)


@dataclass(frozen=True)
class ProfileCheckpointQueueDependencies:
    build_summary_payload: Callable[..., dict[str, Any]]
    classify_operator_review: Callable[..., Any]
    parse_optional_datetime: Callable[..., Any]
    queue_priority_fields: Callable[..., Any]
    queue_reason_label: Callable[..., str]
    build_operator_context: Callable[..., dict[str, Any]]


def build_profile_checkpoint_queue_items(
    profiles: list[DomainResearchProfile],
    *,
    now: datetime,
    deps: ProfileCheckpointQueueDependencies,
) -> list[AgentCheckpointQueueItemResponse]:
    items: list[AgentCheckpointQueueItemResponse] = []
    for profile in profiles:
        payload = deps.build_summary_payload(profile)
        effective_policy = payload["effective_policy"]
        for opportunity in payload["opportunities"]:
            review = deps.classify_operator_review(
                opportunity,
                effective_policy=effective_policy,
            )
            if not review:
                continue
            items.append(
                _build_profile_review_item(
                    profile,
                    opportunity,
                    review,
                    payload=payload,
                    effective_policy=effective_policy,
                    now=now,
                    deps=deps,
                )
            )
    return items


def _build_profile_review_item(
    profile: DomainResearchProfile,
    opportunity: dict[str, Any],
    review: dict[str, Any],
    *,
    payload: dict[str, Any],
    effective_policy: dict[str, Any],
    now: datetime,
    deps: ProfileCheckpointQueueDependencies,
) -> AgentCheckpointQueueItemResponse:
    created_at = None
    if str(opportunity.get("follow_up_reviewed_at") or "").strip():
        created_at = deps.parse_optional_datetime(
            opportunity.get("follow_up_reviewed_at")
        )
    if created_at is None:
        created_at = (
            deps.parse_optional_datetime(opportunity.get("updated_at"))
            or profile.updated_at
            or profile.created_at
        )
    review_type = str(review.get("review_type") or "").strip()
    reason_code = str(review.get("reason_code") or "").strip()
    urgency = deps.queue_priority_fields(
        item_type=review_type,
        reason_code=reason_code,
        created_at=created_at,
        next_run_at=None,
        backoff_until=None,
        stale=False,
        now=now,
    )
    opportunity_id = str(opportunity.get("opportunity_id") or "").strip()
    queue_key = (
        f"profile:{review_type}:{profile.id}:{opportunity_id}:"
        f"{str(review.get('evidence_revision') or '').strip()}:"
        f"{str((payload['summary'] or {}).get('profile_config_revision') or '').strip() or 'current'}"
    )
    common = dict(
        queue_key=queue_key,
        item_type=review_type,
        priority=(
            90
            if review_type == "policy_review"
            else 70
            if review_type == "budget_review"
            else 60
        ),
        title=str(
            opportunity.get("title") or profile.title or "Domain profile review"
        ).strip(),
        summary=(
            str(opportunity.get("follow_up_review_note") or "").strip()
            or str(opportunity.get("operator_note") or "").strip()
            or (
                "Queued follow-up is ready for approval."
                if review_type == "follow_up_recommendation"
                else "Open the domain profile card to review this blocked opportunity."
            )
        )[:320],
        evidence_summary=" · ".join(
            str(row).strip()
            for row in (opportunity.get("supporting_evidence") or [])
            if str(row).strip()
        )[:320]
        or str(opportunity.get("hypothesis") or "").strip()[:320]
        or None,
        status=str(profile.status or "").strip() or None,
        customer=None,
        job_name=str(profile.title or "").strip() or None,
        job_type="research",
        reason_code=reason_code,
        reason_label=str(
            review.get("reason_label") or deps.queue_reason_label(reason_code)
        ),
        priority_score=urgency["priority_score"],
        age_minutes=urgency["age_minutes"],
        sla_bucket=urgency["sla_bucket"],
        escalation_level=urgency["escalation_level"],
        is_overdue=urgency["is_overdue"],
        is_stale=urgency["is_stale"],
        created_at=created_at,
        job_id=profile.active_job_id or profile.latest_run_job_id,
        domain_research_profile_id=profile.id,
        domain_research_profile_title=str(profile.title or "").strip() or None,
        profile_opportunity_id=opportunity_id or None,
        profile_opportunity_key=str(opportunity.get("canonical_key") or "").strip()
        or None,
        follow_up_operator_note=str(
            opportunity.get("follow_up_review_note") or ""
        ).strip()
        or None,
        **deps.build_operator_context(
            objective=str(profile.objective or "").strip() or None,
            domain=str(profile.domain or "").strip() or None,
            track_type=str(
                profile.track_type or opportunity.get("track_type") or "generic"
            ).strip()
            or None,
            source_scope=str(profile.source_scope or "kb_plus_arxiv").strip() or None,
            repo_source_ids=profile.repo_source_ids
            or opportunity.get("source_repo_ids"),
            benchmark_queries=profile.benchmark_queries,
            sandbox_profile_id=str(profile.sandbox_profile_id or "").strip() or None,
            automation_profile=payload["automation_profile"],
            effective_policy=effective_policy,
            confidence=opportunity.get("confidence"),
            readiness=opportunity.get("readiness"),
            linked_note_ids=profile.latest_note_ids,
            linked_experiment_plan_ids=(
                opportunity.get("linked_experiment_plan_ids")
                or profile.latest_experiment_plan_ids
            ),
            linked_validation_run_ids=(
                opportunity.get("linked_validation_run_ids")
                or profile.latest_validation_run_ids
            ),
            child_job_ids=opportunity.get("child_job_ids"),
        ),
    )
    if review_type == "follow_up_recommendation":
        actions = _follow_up_actions(profile, opportunity_id)
        return AgentCheckpointQueueItemResponse(
            **common,
            recommended_action="approve_launch",
            action_count=len(actions),
            follow_up_launch_status="pending_approval",
            follow_up_policy_mode=str(
                effective_policy.get("follow_up_review_mode") or ""
            ).strip()
            or None,
            follow_up_operator_decision=str(
                opportunity.get("follow_up_review_status") or ""
            ).strip()
            or None,
            actions=actions,
        )
    return AgentCheckpointQueueItemResponse(
        **common,
        recommended_action="open_fleet",
        action_count=1,
        budget_reason=reason_code if review_type == "budget_review" else None,
        actions=[
            AgentCheckpointQueueActionResponse(
                kind="policy_action",
                label="Open Profile",
                action="open_fleet",
                description=(
                    "Open this domain profile and inspect the targeted opportunity."
                ),
            )
        ],
    )


def _follow_up_actions(
    profile: DomainResearchProfile,
    opportunity_id: str,
) -> list[AgentCheckpointQueueActionResponse]:
    payload = {
        "domain_research_profile_id": str(profile.id),
        "profile_opportunity_id": opportunity_id,
    }
    return [
        AgentCheckpointQueueActionResponse(
            kind="follow_up_action",
            label="Approve & Launch",
            action="approve_launch",
            description=(
                "Approve this bounded profile follow-up and launch it immediately."
            ),
            recommended=True,
            recommendation_key="profile_follow_up",
            follow_up_action_payload=payload,
        ),
        AgentCheckpointQueueActionResponse(
            kind="follow_up_action",
            label="Reject Launch",
            action="reject_launch",
            description=(
                "Reject this queued profile follow-up for the current evidence revision."
            ),
            recommendation_key="profile_follow_up",
            follow_up_action_payload=payload,
        ),
    ]
