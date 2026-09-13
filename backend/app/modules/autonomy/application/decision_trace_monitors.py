"""Project monitor policy, budget, and rebalance history into decision events."""

from dataclasses import dataclass
from typing import Any, Callable

from app.schemas.agent_job import AgentDecisionTraceEventResponse


@dataclass(frozen=True)
class MonitorDecisionTraceDependencies:
    parse_time: Callable[..., Any]
    reason_label: Callable[..., Any]
    build_event: Callable[..., AgentDecisionTraceEventResponse]


def build_monitor_decision_trace(
    snapshot: dict[str, Any],
    *,
    deps: MonitorDecisionTraceDependencies,
) -> list[AgentDecisionTraceEventResponse]:
    events: list[AgentDecisionTraceEventResponse] = []
    for row in snapshot.get("monitors") or []:
        if not isinstance(row, dict):
            continue
        source_id = (
            str(
                row.get("monitor_job_id")
                or row.get("customer")
                or row.get("monitor_name")
                or ""
            ).strip()
            or None
        )
        source_label = str(
            row.get("monitor_name") or row.get("customer") or "Research monitor"
        ).strip()
        customer = str(row.get("customer") or "").strip() or None
        deep_link = _health_link(customer)
        for entry in row.get("recent_policy_history") or []:
            if not isinstance(entry, dict):
                continue
            event_time = deps.parse_time(entry.get("at"))
            if event_time is None:
                continue
            change_source = str(entry.get("change_source") or "").strip().lower()
            decision_type = (
                "policy_rollback" if "rollback" in change_source else "policy_updated"
            )
            events.append(
                deps.build_event(
                    event_type=decision_type,
                    event_time=event_time,
                    source_kind="monitor",
                    source_id=source_id,
                    source_label=source_label,
                    customer=customer,
                    decision_type=decision_type,
                    reason_code=str(entry.get("change_source") or "").strip() or None,
                    reason_label=deps.reason_label(
                        str(entry.get("change_source") or "").strip() or None
                    ),
                    status=str(
                        row.get("policy_guardrail_state")
                        or row.get("policy_guardrail_status")
                        or ""
                    ).strip()
                    or None,
                    severity="medium",
                    actor_mode="operator",
                    summary=f"{source_label}: {decision_type.replace('_', ' ')}",
                    operator_note=str(entry.get("change_reason") or "").strip() or None,
                    before_state={
                        "effective_policy": entry.get("previous_effective_policy")
                    },
                    after_state={
                        "effective_policy": entry.get("next_effective_policy")
                    },
                    deep_link=deep_link,
                    metadata={"history_entry_id": entry.get("id")},
                    suffix=str(entry.get("id") or ""),
                )
            )
        if str(row.get("policy_guardrail_status") or "").strip().lower() == "active":
            event_time = deps.parse_time(
                row.get("latest_policy_changed_at"),
                fallback=deps.parse_time(snapshot.get("generated_at")),
            )
            if event_time is not None:
                action = str(row.get("policy_guardrail_action") or "").strip() or None
                events.append(
                    deps.build_event(
                        event_type="policy_guardrail_triggered",
                        event_time=event_time,
                        source_kind="monitor",
                        source_id=source_id,
                        source_label=source_label,
                        customer=customer,
                        decision_type="policy_guardrail_triggered",
                        reason_code=action,
                        reason_label=deps.reason_label(action),
                        status=str(
                            row.get("policy_guardrail_state")
                            or row.get("policy_guardrail_status")
                            or ""
                        ).strip()
                        or None,
                        severity="high",
                        actor_mode="autonomous",
                        summary=f"{source_label}: policy guardrail triggered",
                        before_state={"effective_policy": row.get("effective_policy")},
                        after_state={
                            "target_policy": row.get("policy_guardrail_target_policy")
                        },
                        deep_link=deep_link,
                        metadata={"reasons": row.get("policy_guardrail_reasons")},
                        suffix="guardrail",
                    )
                )
        clamp_state = str(row.get("budget_clamp_state") or "").strip().lower()
        if clamp_state not in {"", "normal", "none"}:
            event_time = deps.parse_time(
                row.get("latest_budget_changed_at"),
                fallback=deps.parse_time(snapshot.get("generated_at")),
            )
            if event_time is not None:
                reason = (
                    str((row.get("budget_clamp_reasons") or [None])[0] or "").strip()
                    or None
                )
                events.append(
                    deps.build_event(
                        event_type="budget_clamped",
                        event_time=event_time,
                        source_kind="monitor",
                        source_id=source_id,
                        source_label=source_label,
                        customer=customer,
                        decision_type="budget_clamped",
                        reason_code=reason,
                        reason_label=deps.reason_label(reason),
                        status=str(row.get("budget_clamp_state") or "").strip() or None,
                        severity="high",
                        actor_mode="autonomous",
                        summary=f"{source_label}: budget clamp active",
                        after_state={
                            "budget_clamp_state": row.get("budget_clamp_state")
                        },
                        deep_link=deep_link,
                        metadata={
                            "budget_clamp_reasons": row.get("budget_clamp_reasons")
                        },
                        suffix="budget",
                    )
                )
    events.extend(_build_customer_rebalance_events(snapshot, deps=deps))
    return events


def _build_customer_rebalance_events(
    snapshot: dict[str, Any],
    *,
    deps: MonitorDecisionTraceDependencies,
) -> list[AgentDecisionTraceEventResponse]:
    events: list[AgentDecisionTraceEventResponse] = []
    for row in snapshot.get("customers") or []:
        if not isinstance(row, dict):
            continue
        customer = str(row.get("customer") or "").strip() or None
        for entry in row.get("recent_rebalance_history") or []:
            if not isinstance(entry, dict):
                continue
            event_time = deps.parse_time(entry.get("at"))
            if event_time is None:
                continue
            reason = str(entry.get("change_source") or "").strip() or None
            events.append(
                deps.build_event(
                    event_type="customer_rebalanced",
                    event_time=event_time,
                    source_kind="monitor",
                    source_id=customer or str(entry.get("id") or ""),
                    source_label=customer or "Customer portfolio",
                    customer=customer,
                    decision_type="customer_rebalanced",
                    reason_code=reason,
                    reason_label=deps.reason_label(reason),
                    status=str(
                        entry.get("evaluation_status")
                        or entry.get("evaluation_state")
                        or ""
                    ).strip()
                    or None,
                    severity="medium",
                    actor_mode="operator",
                    summary=f"{customer or 'Customer'}: customer rebalance applied",
                    operator_note=str(entry.get("change_reason") or "").strip() or None,
                    before_state={"before_capacity": entry.get("before_capacity")},
                    after_state={"after_capacity": entry.get("after_capacity")},
                    deep_link=_health_link(customer),
                    metadata={"history_entry_id": entry.get("id")},
                    suffix=str(entry.get("id") or ""),
                )
            )
    return events


def _health_link(customer: str | None) -> dict[str, Any]:
    return {
        "target_tab": "health",
        "params": {"tab": "health", "health_customer": customer or ""},
        "label": "Open Autonomy Health",
    }
