from __future__ import annotations

from typing import Any, Mapping

from app.services.research_opportunity_service import (
    compute_research_portfolio_config_revision,
    summarize_portfolio_operator_reviews,
    summarize_research_opportunity_autonomy_states,
    summarize_research_opportunity_stages,
)
from app.services.scientific_validation_service import (
    normalize_portfolio_automation_profile,
    resolve_portfolio_automation_policy,
)

PROFILE_POLICY_COMPAT_FIELDS = (
    "validation_policy",
    "auto_launch_follow_up",
    "auto_create_experiment_plans",
    "confidence_threshold",
)


def legacy_profile_policy_overrides(
    payload: Mapping[str, Any] | None
) -> dict[str, Any]:
    source = payload or {}
    overrides: dict[str, Any] = {}
    validation_policy = (
        source.get("validation_policy")
        if isinstance(source.get("validation_policy"), dict)
        else {}
    )
    if validation_policy:
        overrides.update(validation_policy)
    for key in (
        "auto_launch_follow_up",
        "auto_create_experiment_plans",
        "confidence_threshold",
        "max_auto_follow_up_launches",
        "auto_execute_validation_runs",
        "auto_launch_experiment_runs",
        "max_concurrent_validation_runs",
        "max_validation_runtime_minutes",
        "max_validation_budget_per_run",
        "experiment_readiness_threshold",
        "follow_up_review_mode",
        "duplicate_window_items",
        "validation_backoff_policy",
    ):
        if key in source and source.get(key) is not None:
            overrides[key] = source.get(key)
    return overrides


def current_domain_profile_policy_snapshot(profile: Any) -> dict[str, Any]:
    return {
        "validation_policy": getattr(profile, "validation_policy", None)
        if isinstance(getattr(profile, "validation_policy", None), dict)
        else None,
        "auto_launch_follow_up": (
            bool(getattr(profile, "auto_launch_follow_up"))
            if getattr(profile, "auto_launch_follow_up", None) is not None
            else True
        ),
        "auto_create_experiment_plans": (
            bool(getattr(profile, "auto_create_experiment_plans"))
            if getattr(profile, "auto_create_experiment_plans", None) is not None
            else True
        ),
        "confidence_threshold": (
            float(getattr(profile, "confidence_threshold"))
            if getattr(profile, "confidence_threshold", None) is not None
            else 0.7
        ),
    }


def build_domain_profile_compat_policy(effective_policy: object) -> dict[str, Any]:
    return dict(effective_policy) if isinstance(effective_policy, dict) else {}


def resolve_domain_profile_automation_contract(
    *,
    automation_profile: Any,
    automation_policy: Any,
    current_snapshot: Mapping[str, Any] | None = None,
    explicit_updates: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    normalized_profile = normalize_portfolio_automation_profile(
        automation_profile, default="balanced"
    )
    resolved_policy = resolve_portfolio_automation_policy(
        normalized_profile,
        {
            **legacy_profile_policy_overrides(current_snapshot),
            **(automation_policy if isinstance(automation_policy, dict) else {}),
            **legacy_profile_policy_overrides(explicit_updates),
        },
    )
    return normalized_profile, resolved_policy


def build_autonomy_summary(
    *,
    raw_summary: Mapping[str, Any] | None,
    opportunities: list[dict[str, Any]],
    automation_profile: str,
    effective_policy: dict[str, Any],
    sandbox_profile_id: str | None,
    config_revision_key: str,
    include_autonomy_mode: bool = True,
) -> dict[str, Any]:
    summary = dict(raw_summary) if isinstance(raw_summary, dict) else {}
    summary["effective_policy"] = effective_policy
    summary[config_revision_key] = summary.get(
        config_revision_key
    ) or compute_research_portfolio_config_revision(
        automation_profile,
        effective_policy,
        sandbox_profile_id,
    )
    summary["opportunities"] = opportunities
    summary["stage_counts"] = summarize_research_opportunity_stages(opportunities)
    summary["autonomy_state_counts"] = summarize_research_opportunity_autonomy_states(
        opportunities
    )
    summary.update(
        summarize_portfolio_operator_reviews(
            opportunities, effective_policy=effective_policy
        )
    )
    if include_autonomy_mode:
        summary["autonomy_mode"] = automation_profile
    return summary


def normalize_monitor_policy_mode(raw: object, *, default: str = "manual_only") -> str:
    mode = str(raw or "").strip().lower()
    if mode in {"manual_only", "auto_launch_safe", "queue_for_approval"}:
        return mode
    return default


def normalize_monitor_allowed_recommendations(
    raw: object,
    *,
    default_allowed: list[str],
) -> list[str]:
    values = raw if isinstance(raw, list) else []
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out or list(default_allowed)


def build_monitor_follow_up_autonomy_compat(
    *,
    automation_profile: Any,
    automation_policy: object,
    effective_policy: object,
    default_allowed: list[str],
) -> dict[str, Any]:
    resolved_effective = effective_policy if isinstance(effective_policy, dict) else {}
    raw_policy = automation_policy if isinstance(automation_policy, dict) else {}
    allowed_recommendations = normalize_monitor_allowed_recommendations(
        raw_policy.get("allowed_recommendations")
        if isinstance(raw_policy.get("allowed_recommendations"), list)
        else resolved_effective.get("allowed_recommendations"),
        default_allowed=default_allowed,
    )
    mode = normalize_monitor_policy_mode(
        resolved_effective.get("follow_up_review_mode")
    )
    return {
        "mode": mode,
        "allowed_recommendations": allowed_recommendations,
        "automation_profile": normalize_portfolio_automation_profile(
            automation_profile, default="balanced"
        ),
        "automation_policy": dict(raw_policy) if isinstance(raw_policy, dict) else {},
        "effective_policy": dict(resolved_effective),
    }


def build_monitor_policy_compat_fields(
    *,
    automation_profile: Any,
    automation_policy: object,
    effective_policy: object,
    default_allowed: list[str],
    target_policy: object | None = None,
) -> dict[str, Any]:
    compat = build_monitor_follow_up_autonomy_compat(
        automation_profile=automation_profile,
        automation_policy=automation_policy,
        effective_policy=effective_policy,
        default_allowed=default_allowed,
    )
    fields = {
        "follow_up_autonomy": compat,
        "current_policy_mode": compat["mode"],
        "current_allowed_recommendations": list(compat["allowed_recommendations"]),
        "policy_guardrail_follow_up_autonomy": None,
    }
    if isinstance(target_policy, dict):
        fields[
            "policy_guardrail_follow_up_autonomy"
        ] = build_monitor_follow_up_autonomy_compat(
            automation_profile=automation_profile,
            automation_policy={
                "follow_up_review_mode": target_policy.get("follow_up_review_mode"),
                "allowed_recommendations": target_policy.get("allowed_recommendations"),
            },
            effective_policy=target_policy,
            default_allowed=default_allowed,
        )
    return fields


def build_monitor_policy_history_compat_entry(
    *,
    previous_snapshot: Mapping[str, Any],
    next_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "previous_follow_up_autonomy": previous_snapshot.get("follow_up_autonomy"),
        "next_follow_up_autonomy": next_snapshot.get("follow_up_autonomy"),
    }


def resolve_monitor_automation_contract(
    config: object,
    *,
    default_allowed: list[str],
) -> dict[str, Any]:
    raw_config = config if isinstance(config, dict) else {}
    automation_profile = normalize_portfolio_automation_profile(
        raw_config.get("automation_profile"), default="balanced"
    )
    legacy_policy = {
        "mode": normalize_monitor_policy_mode(
            (raw_config.get("follow_up_autonomy") or {}).get("mode")
        ),
        "allowed_recommendations": normalize_monitor_allowed_recommendations(
            (raw_config.get("follow_up_autonomy") or {}).get("allowed_recommendations"),
            default_allowed=default_allowed,
        ),
    }
    raw_automation_policy = (
        raw_config.get("automation_policy")
        if isinstance(raw_config.get("automation_policy"), dict)
        else {}
    )
    policy_input = {
        **legacy_policy,
        **raw_automation_policy,
        "follow_up_review_mode": raw_automation_policy.get(
            "follow_up_review_mode", legacy_policy.get("mode")
        ),
        "allowed_recommendations": normalize_monitor_allowed_recommendations(
            raw_automation_policy.get("allowed_recommendations")
            if isinstance(raw_automation_policy.get("allowed_recommendations"), list)
            else legacy_policy.get("allowed_recommendations"),
            default_allowed=default_allowed,
        ),
    }
    effective_policy = resolve_portfolio_automation_policy(
        automation_profile, policy_input
    )
    effective_policy[
        "allowed_recommendations"
    ] = normalize_monitor_allowed_recommendations(
        policy_input.get("allowed_recommendations"),
        default_allowed=default_allowed,
    )
    automation_policy = {
        **raw_automation_policy,
        "follow_up_review_mode": normalize_monitor_policy_mode(
            policy_input.get("follow_up_review_mode"), default="auto_launch_safe"
        ),
        "allowed_recommendations": list(effective_policy["allowed_recommendations"]),
    }
    follow_up_autonomy = build_monitor_follow_up_autonomy_compat(
        automation_profile=automation_profile,
        automation_policy=automation_policy,
        effective_policy=effective_policy,
        default_allowed=default_allowed,
    )
    return {
        "automation_profile": automation_profile,
        "automation_policy": automation_policy,
        "effective_policy": effective_policy,
        "follow_up_autonomy": follow_up_autonomy,
    }
