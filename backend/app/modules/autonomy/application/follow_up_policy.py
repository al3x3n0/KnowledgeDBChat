"""Apply monitor follow-up policy when a research-inbox item is accepted."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob
from app.models.research_inbox import ResearchInboxItem
from app.models.user import User
from app.schemas.agent_job import AgentJobFromChainCreate

MANUAL_ONLY = "manual_only"
AUTO_LAUNCH_SAFE = "auto_launch_safe"
QUEUE_FOR_APPROVAL = "queue_for_approval"


@dataclass(frozen=True)
class FollowUpPolicyDependencies:
    get_policy_from_job: Callable[[AgentJob | None], dict[str, Any]]
    budget_service: Any
    load_learning_profile: Callable[..., Awaitable[dict[str, Any]]]
    build_follow_up_actions: Callable[..., Any]
    launch_follow_up_action: Callable[..., Awaitable[Any]]


async def apply_follow_up_policy_on_accept(
    *,
    item: ResearchInboxItem,
    current_user: User,
    db: AsyncSession,
    deps: FollowUpPolicyDependencies,
) -> None:
    if item.follow_up_launch_status == "launched":
        return

    source_job = await _load_source_job(item, current_user=current_user, db=db)
    policy = deps.get_policy_from_job(source_job)
    mode = policy["mode"]
    allowed_recommendations = set(policy["allowed_recommendations"])
    budget_snapshot = await _monitor_budget_snapshot(
        source_job,
        current_user=current_user,
        db=db,
        budget_service=deps.budget_service,
    )
    customer = str(item.customer or "").strip() or None
    customer_budget_snapshot = await deps.budget_service.build_customer_budget_snapshot(
        db=db,
        user_id=current_user.id,
        customer=customer,
    )
    learning_profile = await deps.load_learning_profile(
        db=db,
        user_id=current_user.id,
        customer=customer,
    )
    actions = deps.build_follow_up_actions(
        item,
        learning_profile=learning_profile,
    )
    recommended_action = _select_recommended_action(
        actions,
        mode=mode,
        allowed_recommendations=allowed_recommendations,
    )
    _reset_follow_up_state(
        item,
        mode=mode,
        budget_snapshot=budget_snapshot,
        customer_budget_snapshot=customer_budget_snapshot,
    )
    if recommended_action is None:
        _block(
            item,
            "No supported follow-up recommendation is available for this inbox item.",
        )
        item.follow_up_recommendation_key = None
        return

    recommendation_key = (
        str(recommended_action.recommendation_key or "").strip() or None
    )
    eligibility = (
        str(recommended_action.autonomy_eligibility or "").strip().lower()
        or MANUAL_ONLY
    )
    item.follow_up_recommendation_key = recommendation_key
    if mode == MANUAL_ONLY:
        _block(item, "Monitor policy is set to manual follow-up launches.")
        return
    if eligibility != "auto_launchable":
        _block(item, "Recommended follow-up is outside the safe auto-launch allowlist.")
        return
    if recommendation_key and recommendation_key not in allowed_recommendations:
        _block(item, "Recommendation is not allowlisted by this monitor policy.")
        return

    effective_mode = _apply_budget_throttles(
        item,
        mode=mode,
        budget_snapshot=budget_snapshot,
        customer_budget_snapshot=customer_budget_snapshot,
    )
    if effective_mode == QUEUE_FOR_APPROVAL:
        item.follow_up_decision = "queued_for_approval"
        item.follow_up_launch_status = "pending_approval"
        item.follow_up_block_reason = (
            item.follow_up_customer_budget_reason
            or item.follow_up_budget_reason
            or "Safe follow-up is prepared and waiting for operator approval."
        )
        return
    if effective_mode == MANUAL_ONLY:
        _block(
            item,
            item.follow_up_customer_budget_reason
            or item.follow_up_budget_reason
            or "Autonomy budgets currently clamp follow-ups to manual mode.",
        )
        return

    try:
        launched = await deps.launch_follow_up_action(
            recommended_action,
            db=db,
            current_user=current_user,
        )
        item.follow_up_decision = "auto_launched"
        item.follow_up_launch_status = "launched"
        item.follow_up_job_id = launched.id
        if recommended_action.chain_create_payload:
            item.follow_up_chain_definition_id = AgentJobFromChainCreate.model_validate(
                recommended_action.chain_create_payload
            ).chain_definition_id
        else:
            item.follow_up_chain_definition_id = None
        item.follow_up_launched_at = datetime.utcnow()
        item.follow_up_block_reason = None
    except Exception as exc:
        detail = getattr(exc, "detail", None)
        if detail is None:
            logger.warning(
                f"Failed to auto-launch follow-up for inbox item {item.id}: {exc}"
            )
        item.follow_up_decision = "launch_failed"
        item.follow_up_launch_status = "failed"
        item.follow_up_block_reason = (
            str(detail) if detail is not None else str(exc)[:500]
        )


async def _load_source_job(
    item: ResearchInboxItem,
    *,
    current_user: User,
    db: AsyncSession,
) -> AgentJob | None:
    if not item.job_id:
        return None
    source_job = await db.get(AgentJob, item.job_id)
    if source_job and source_job.user_id != current_user.id:
        return None
    return source_job


async def _monitor_budget_snapshot(
    source_job: AgentJob | None,
    *,
    current_user: User,
    db: AsyncSession,
    budget_service: Any,
) -> dict[str, Any]:
    if source_job:
        return await budget_service.build_monitor_budget_snapshot(
            db=db,
            user_id=current_user.id,
            monitor_job=source_job,
        )
    return {
        "autonomy_budget": budget_service._normalize_budget_config(None),
        "budget_usage": budget_service._empty_budget_usage(),
        "budget_remaining": budget_service._empty_budget_usage(),
        "budget_throttle_state": "normal",
        "budget_throttle_reasons": [],
    }


def _select_recommended_action(
    actions: list[Any],
    *,
    mode: str,
    allowed_recommendations: set[str],
) -> Any:
    preferred_action = next(
        (action for action in actions if action.recommended),
        actions[0] if actions else None,
    )
    if mode == MANUAL_ONLY:
        return preferred_action
    for action in actions:
        recommendation_key = str(action.recommendation_key or "").strip()
        eligibility = (
            str(action.autonomy_eligibility or "").strip().lower() or MANUAL_ONLY
        )
        if (
            eligibility == "auto_launchable"
            and recommendation_key in allowed_recommendations
        ):
            return action
    return preferred_action


def _reset_follow_up_state(
    item: ResearchInboxItem,
    *,
    mode: str,
    budget_snapshot: dict[str, Any],
    customer_budget_snapshot: dict[str, Any],
) -> None:
    item.follow_up_policy_mode = mode
    for field in (
        "follow_up_job_id",
        "follow_up_chain_definition_id",
        "follow_up_launched_at",
        "follow_up_operator_decision",
        "follow_up_operator_note",
        "follow_up_operator_acted_at",
        "follow_up_operator_user_id",
        "follow_up_outcome_status",
        "follow_up_outcome_recorded_at",
        "follow_up_outcome_summary",
        "follow_up_budget_decision",
        "follow_up_budget_reason",
        "follow_up_customer_budget_decision",
        "follow_up_customer_budget_reason",
    ):
        setattr(item, field, None)
    item.follow_up_budget_throttle_state = str(
        budget_snapshot.get("budget_throttle_state") or "normal"
    )
    item.follow_up_customer_budget_throttle_state = str(
        customer_budget_snapshot.get("customer_budget_throttle_state") or "normal"
    )


def _apply_budget_throttles(
    item: ResearchInboxItem,
    *,
    mode: str,
    budget_snapshot: dict[str, Any],
    customer_budget_snapshot: dict[str, Any],
) -> str:
    throttle_state = str(budget_snapshot.get("budget_throttle_state") or "normal")
    throttle_reasons = _reasons(budget_snapshot.get("budget_throttle_reasons"))
    customer_state = str(
        customer_budget_snapshot.get("customer_budget_throttle_state") or "normal"
    )
    customer_reasons = _reasons(
        customer_budget_snapshot.get("customer_budget_throttle_reasons")
    )
    effective_mode = mode
    if mode == AUTO_LAUNCH_SAFE and throttle_state == "auto_launch_throttled":
        effective_mode = QUEUE_FOR_APPROVAL
        item.follow_up_budget_decision = "downgraded_to_queue"
    elif throttle_state == "manual_only_clamped":
        effective_mode = MANUAL_ONLY
        item.follow_up_budget_decision = "clamped_to_manual"
    if mode == AUTO_LAUNCH_SAFE and customer_state == "auto_launch_throttled":
        effective_mode = QUEUE_FOR_APPROVAL
        item.follow_up_customer_budget_decision = "downgraded_to_queue"
    elif customer_state == "manual_only_clamped":
        effective_mode = MANUAL_ONLY
        item.follow_up_customer_budget_decision = "clamped_to_manual"
    if effective_mode != mode:
        if item.follow_up_budget_decision:
            item.follow_up_budget_reason = (
                "; ".join(throttle_reasons[:3])
                or "Monitor autonomy budget is currently exhausted."
            )
            item.follow_up_budget_throttle_state = throttle_state
        if item.follow_up_customer_budget_decision:
            item.follow_up_customer_budget_reason = (
                "; ".join(customer_reasons[:3])
                or "Customer autonomy budget is currently exhausted."
            )
            item.follow_up_customer_budget_throttle_state = customer_state
        item.follow_up_block_reason = (
            item.follow_up_customer_budget_reason or item.follow_up_budget_reason
        )
    return effective_mode


def _block(item: ResearchInboxItem, reason: str) -> None:
    item.follow_up_decision = "manual"
    item.follow_up_launch_status = "blocked"
    item.follow_up_block_reason = reason


def _reasons(values: Any) -> list[str]:
    return [str(value).strip() for value in (values or []) if str(value).strip()]
