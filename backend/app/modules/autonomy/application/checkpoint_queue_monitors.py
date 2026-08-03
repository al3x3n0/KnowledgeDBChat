"""Project monitor policy and budget reviews into checkpoint queue rows."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from app.models.agent_job import AgentJob
from app.schemas.agent_job import (
    AgentCheckpointQueueActionResponse,
    AgentCheckpointQueueItemResponse,
)


@dataclass(frozen=True)
class MonitorCheckpointQueueDependencies:
    queue_customer_for_job: Callable[..., Any]
    present_job: Callable[..., Any]
    queue_priority_fields: Callable[..., Any]
    build_policy_compat_fields: Callable[..., Any]
    safe_autonomy_recommendations: tuple[str, ...]


def build_monitor_checkpoint_queue_items(
    jobs: list[AgentJob],
    monitor_health_rows: list[dict[str, Any]],
    *,
    now: datetime,
    deps: MonitorCheckpointQueueDependencies,
) -> list[AgentCheckpointQueueItemResponse]:
    items: list[AgentCheckpointQueueItemResponse] = []
    jobs_by_id = {job.id: job for job in jobs}
    for monitor in monitor_health_rows:
        monitor_job_id = monitor.get("monitor_job_id")
        if not monitor_job_id:
            continue
        job = jobs_by_id.get(monitor_job_id)
        if job is None:
            continue
        customer = str(
            monitor.get("customer") or ""
        ).strip() or deps.queue_customer_for_job(job)
        created_at = (
            monitor.get("latest_policy_changed_at")
            or job.last_activity_at
            or job.completed_at
            or job.started_at
            or job.created_at
        )
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except Exception:
                created_at = (
                    job.last_activity_at
                    or job.completed_at
                    or job.started_at
                    or job.created_at
                )
        guardrail_status = (
            str(monitor.get("policy_guardrail_status") or "").strip().lower()
        )
        guardrail_action = (
            str(monitor.get("policy_guardrail_action") or "").strip().lower()
        )
        guardrail_target_policy = monitor.get("policy_guardrail_target_policy")
        guardrail_compat_fields = deps.build_policy_compat_fields(
            automation_profile=monitor.get("automation_profile")
            or monitor.get("autonomy_mode"),
            automation_policy=monitor.get("automation_policy"),
            effective_policy=monitor.get("effective_policy"),
            default_allowed=list(deps.safe_autonomy_recommendations),
            target_policy=guardrail_target_policy,
        )
        guardrail_policy = guardrail_compat_fields.get(
            "policy_guardrail_follow_up_autonomy"
        )
        history_entry_id = (
            str(monitor.get("policy_guardrail_target_history_entry_id") or "").strip()
            or None
        )
        if guardrail_status == "active":
            items.append(
                _build_policy_review_item(
                    monitor,
                    job,
                    customer=customer,
                    created_at=created_at,
                    guardrail_action=guardrail_action,
                    guardrail_target_policy=guardrail_target_policy,
                    guardrail_policy=guardrail_policy,
                    history_entry_id=history_entry_id,
                    now=now,
                    deps=deps,
                )
            )
        budget_state = str(monitor.get("budget_throttle_state") or "").strip().lower()
        if budget_state != "normal":
            items.append(
                _build_budget_review_item(
                    monitor,
                    job,
                    customer=customer,
                    created_at=created_at,
                    budget_state=budget_state,
                    now=now,
                    deps=deps,
                )
            )
    return items


def _build_policy_review_item(
    monitor: dict[str, Any],
    job: AgentJob,
    *,
    customer: str | None,
    created_at: Any,
    guardrail_action: str,
    guardrail_target_policy: Any,
    guardrail_policy: Any,
    history_entry_id: str | None,
    now: datetime,
    deps: MonitorCheckpointQueueDependencies,
) -> AgentCheckpointQueueItemResponse:
    urgency = deps.queue_priority_fields(
        item_type="policy_review",
        reason_code="policy_guardrail",
        created_at=created_at,
        next_run_at=None,
        backoff_until=None,
        stale=False,
        now=now,
    )
    actions = [
        AgentCheckpointQueueActionResponse(
            kind="policy_action",
            label="Apply safeguard",
            action="apply_guardrail",
            recommended=True,
            description=(
                "Apply the recommended rollback or downgrade for this "
                "degrading monitor policy."
            ),
            policy_rollback_payload=(
                {"history_entry_id": history_entry_id}
                if guardrail_action == "rollback" and history_entry_id
                else None
            ),
            policy_update_payload=(
                {
                    "automation_profile": str(
                        monitor.get("automation_profile")
                        or monitor.get("autonomy_mode")
                        or "balanced"
                    ).strip()
                    or "balanced",
                    "automation_policy": (
                        {
                            "follow_up_review_mode": str(
                                (guardrail_target_policy or {}).get(
                                    "follow_up_review_mode"
                                )
                                or ""
                            ).strip()
                            or None,
                            "allowed_recommendations": list(
                                (guardrail_target_policy or {}).get(
                                    "allowed_recommendations"
                                )
                                or []
                            ),
                        }
                        if isinstance(guardrail_target_policy, dict)
                        else None
                    ),
                    "mode": str((guardrail_policy or {}).get("mode") or "").strip()
                    or None,
                    "allowed_recommendations": list(
                        (guardrail_policy or {}).get("allowed_recommendations") or []
                    ),
                    "change_source": "policy_guardrail",
                }
                if guardrail_action == "downgrade"
                and isinstance(guardrail_policy, dict)
                else None
            ),
        ),
        AgentCheckpointQueueActionResponse(
            kind="policy_action",
            label="Compare Before/After",
            action="compare_before_after",
            description="Open the latest policy comparison for this degrading rollout.",
            policy_rollback_payload=(
                {"history_entry_id": history_entry_id} if history_entry_id else None
            ),
        ),
    ]
    return AgentCheckpointQueueItemResponse(
        queue_key=f"policy_review:{job.id}:{history_entry_id or 'current'}",
        item_type="policy_review",
        priority=90,
        title=str(
            monitor.get("monitor_name") or job.name or "Monitor policy review"
        ).strip(),
        summary=(
            "Latest policy evaluation is degrading. Suggested safeguard: "
            f"{'roll back to the previous policy' if guardrail_action == 'rollback' else 'downgrade autonomy mode'}."
        ),
        evidence_summary=" · ".join(
            str(reason).strip()
            for reason in (monitor.get("policy_guardrail_reasons") or [])
            if str(reason).strip()
        )[:320]
        or None,
        status=str(job.status or "").strip() or None,
        customer=customer or None,
        job_name=str(job.name or "").strip() or None,
        job_type=str(job.job_type or "").strip() or None,
        reason_code="policy_guardrail",
        reason_label="Policy safeguard review",
        recommended_action="apply_guardrail",
        priority_score=urgency["priority_score"],
        age_minutes=urgency["age_minutes"],
        sla_bucket=urgency["sla_bucket"],
        escalation_level=urgency["escalation_level"],
        is_overdue=urgency["is_overdue"],
        is_stale=urgency["is_stale"],
        action_count=len(actions),
        created_at=created_at,
        job_id=job.id,
        job=deps.present_job(job),
        policy_guardrail_status="active",
        policy_guardrail_action=guardrail_action or None,
        policy_guardrail_target_history_entry_id=history_entry_id,
        policy_guardrail_reasons=list(monitor.get("policy_guardrail_reasons") or []),
        policy_guardrail_target_policy=(
            guardrail_target_policy
            if isinstance(guardrail_target_policy, dict)
            else None
        ),
        policy_guardrail_follow_up_autonomy=(
            guardrail_policy if isinstance(guardrail_policy, dict) else None
        ),
        actions=actions,
    )


def _build_budget_review_item(
    monitor: dict[str, Any],
    job: AgentJob,
    *,
    customer: str | None,
    created_at: Any,
    budget_state: str,
    now: datetime,
    deps: MonitorCheckpointQueueDependencies,
) -> AgentCheckpointQueueItemResponse:
    budget_reasons = [
        str(reason).strip()
        for reason in (monitor.get("budget_throttle_reasons") or [])
        if str(reason).strip()
    ]
    urgency = deps.queue_priority_fields(
        item_type="budget_review",
        reason_code="budget_throttle",
        created_at=created_at,
        next_run_at=None,
        backoff_until=None,
        stale=False,
        now=now,
    )
    return AgentCheckpointQueueItemResponse(
        queue_key=f"budget_review:{job.id}:{budget_state}",
        item_type="budget_review",
        priority=70,
        title=str(
            monitor.get("monitor_name") or job.name or "Monitor budget review"
        ).strip(),
        summary=(
            "Autonomy is temporarily throttled to "
            f"{budget_state.replace('_', ' ')} for this monitor."
        ),
        evidence_summary=" · ".join(budget_reasons[:3])[:320] or None,
        status=str(job.status or "").strip() or None,
        customer=customer or None,
        job_name=str(job.name or "").strip() or None,
        job_type=str(job.job_type or "").strip() or None,
        reason_code="budget_throttle",
        reason_label="Autonomy budget review",
        recommended_action="open_monitor",
        priority_score=urgency["priority_score"],
        age_minutes=urgency["age_minutes"],
        sla_bucket=urgency["sla_bucket"],
        escalation_level=urgency["escalation_level"],
        is_overdue=urgency["is_overdue"],
        is_stale=urgency["is_stale"],
        action_count=1,
        created_at=created_at,
        job_id=job.id,
        job=deps.present_job(job),
        budget_throttle_state=budget_state,
        budget_reason="; ".join(budget_reasons[:3]) or None,
        actions=[
            AgentCheckpointQueueActionResponse(
                kind="policy_action",
                label="Open Monitor",
                action="open_monitor",
                description=(
                    "Open this monitor in Autonomy Health to inspect budget pressure "
                    "and adjust limits."
                ),
            )
        ],
    )
