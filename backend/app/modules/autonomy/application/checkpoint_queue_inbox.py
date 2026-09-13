"""Project accepted research-inbox follow-ups into checkpoint queue rows."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from app.models.research_inbox import ResearchInboxItem
from app.schemas.agent_job import (
    AgentCheckpointQueueActionResponse,
    AgentCheckpointQueueItemResponse,
)


@dataclass(frozen=True)
class InboxCheckpointQueueDependencies:
    build_follow_up_actions: Callable[..., Any]
    customer_profile_key: Callable[..., str]
    queue_priority_fields: Callable[..., Any]
    queue_reason_label: Callable[..., str]


def build_inbox_checkpoint_queue_items(
    inbox_items: list[ResearchInboxItem],
    *,
    learning_profiles: dict[str, dict[str, Any]],
    now: datetime,
    deps: InboxCheckpointQueueDependencies,
) -> list[AgentCheckpointQueueItemResponse]:
    items: list[AgentCheckpointQueueItemResponse] = []
    for item in inbox_items:
        launch_status = str(item.follow_up_launch_status or "").strip().lower()
        operator_decision = str(item.follow_up_operator_decision or "").strip().lower()
        if (
            launch_status == "launched"
            or operator_decision == "rejected"
            or launch_status == "rejected"
        ):
            continue
        actions = deps.build_follow_up_actions(
            item,
            learning_profile=learning_profiles.get(
                deps.customer_profile_key(item.customer)
            ),
        )
        created_at = item.updated_at or item.discovered_at
        follow_up_decision = str(item.follow_up_decision or "").strip() or None
        follow_up_policy_mode = str(item.follow_up_policy_mode or "").strip() or None
        follow_up_launch_status = (
            str(item.follow_up_launch_status or "").strip() or None
        )
        follow_up_block_reason = str(item.follow_up_block_reason or "").strip() or None
        reason_code, reason_label = _follow_up_reason(
            follow_up_decision,
            follow_up_launch_status,
            deps=deps,
        )
        urgency = deps.queue_priority_fields(
            item_type="follow_up_recommendation",
            reason_code=reason_code,
            created_at=created_at,
            next_run_at=None,
            backoff_until=None,
            stale=False,
            now=now,
        )
        if follow_up_launch_status == "pending_approval":
            actions = _pending_approval_actions(item)
        items.append(
            AgentCheckpointQueueItemResponse(
                queue_key=f"followup:{item.id}",
                item_type="follow_up_recommendation",
                priority=60,
                title=item.title,
                summary=(
                    follow_up_block_reason
                    or item.summary
                    or f"Accepted {item.item_type} signal ready for follow-up."
                )[:320],
                evidence_summary=(item.summary or item.url or item.title)[:320],
                status=item.status,
                customer=str(item.customer or "").strip() or None,
                job_name=None,
                job_type="research",
                reason_code=reason_code,
                reason_label=reason_label,
                recommended_action=next(
                    (
                        action.action or action.launch_label or action.label
                        for action in actions
                        if action.recommended
                    ),
                    None,
                )
                or (actions[0].launch_label if actions else None),
                priority_score=urgency["priority_score"],
                age_minutes=urgency["age_minutes"],
                sla_bucket=urgency["sla_bucket"],
                escalation_level=urgency["escalation_level"],
                is_overdue=urgency["is_overdue"],
                is_stale=urgency["is_stale"],
                next_run_at=None,
                backoff_until=None,
                action_count=len(actions),
                created_at=created_at,
                inbox_item_id=item.id,
                inbox_item={
                    "id": str(item.id),
                    "item_type": item.item_type,
                    "item_key": item.item_key,
                    "title": item.title,
                    "summary": item.summary,
                    "url": item.url,
                    "customer": item.customer,
                },
                follow_up_decision=follow_up_decision,
                follow_up_policy_mode=follow_up_policy_mode,
                follow_up_launch_status=follow_up_launch_status,
                follow_up_block_reason=follow_up_block_reason,
                follow_up_budget_decision=_text(item.follow_up_budget_decision),
                follow_up_budget_reason=_text(item.follow_up_budget_reason),
                follow_up_budget_throttle_state=_text(
                    item.follow_up_budget_throttle_state
                ),
                follow_up_customer_budget_decision=_text(
                    item.follow_up_customer_budget_decision
                ),
                follow_up_customer_budget_reason=_text(
                    item.follow_up_customer_budget_reason
                ),
                follow_up_customer_budget_throttle_state=_text(
                    item.follow_up_customer_budget_throttle_state
                ),
                follow_up_recommendation_key=_text(item.follow_up_recommendation_key),
                follow_up_job_id=item.follow_up_job_id,
                follow_up_chain_definition_id=item.follow_up_chain_definition_id,
                follow_up_operator_decision=_text(item.follow_up_operator_decision),
                follow_up_operator_note=_text(item.follow_up_operator_note),
                follow_up_operator_acted_at=item.follow_up_operator_acted_at,
                follow_up_operator_user_id=item.follow_up_operator_user_id,
                budget_throttle_state=_text(item.follow_up_budget_throttle_state),
                budget_reason=_text(item.follow_up_budget_reason),
                customer_budget_throttle_state=_text(
                    item.follow_up_customer_budget_throttle_state
                ),
                customer_budget_reason=_text(item.follow_up_customer_budget_reason),
                actions=actions,
            )
        )
    return items


def _follow_up_reason(
    decision: str | None,
    launch_status: str | None,
    *,
    deps: InboxCheckpointQueueDependencies,
) -> tuple[str, str]:
    if decision == "queued_for_approval":
        return "follow_up_launch_approval", "Follow-up launch approval"
    if launch_status == "blocked":
        return "follow_up_blocked", "Follow-up blocked by policy"
    if launch_status == "failed":
        return "follow_up_launch_failed", "Follow-up launch failed"
    return "accepted_inbox_item", deps.queue_reason_label("accepted_inbox_item")


def _pending_approval_actions(
    item: ResearchInboxItem,
) -> list[AgentCheckpointQueueActionResponse]:
    recommendation_key = _text(item.follow_up_recommendation_key)
    payload = {"inbox_item_id": str(item.id)}
    return [
        AgentCheckpointQueueActionResponse(
            kind="follow_up_action",
            label="Approve & Launch",
            action="approve_launch",
            description=(
                "Approve this bounded safe follow-up and launch it immediately."
            ),
            recommended=True,
            recommendation_key=recommendation_key,
            follow_up_action_payload=payload,
        ),
        AgentCheckpointQueueActionResponse(
            kind="follow_up_action",
            label="Reject Launch",
            action="reject_launch",
            description=(
                "Reject this queued safe follow-up without creating a downstream job."
            ),
            recommendation_key=recommendation_key,
            follow_up_action_payload=payload,
        ),
    ]


def _text(value: Any) -> str | None:
    return str(value or "").strip() or None
