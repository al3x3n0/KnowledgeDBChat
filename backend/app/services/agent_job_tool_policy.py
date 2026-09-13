"""Job-type tool policy.

Which tools a job type may use, and how tool selection is tuned for a job.
Pure policy tables extracted from AutonomousAgentExecutor: they read only the
job type and its config, so they are unit-tested directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.agent_core import tool_specs
from app.models.agent_job import AgentJob


def get_tools_for_job_type(
    job_type: str,
    config: Optional[Dict[str, Any]],
    profile: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Get available tools based on job type.

    The per-job-type lists that used to sit here -- a base list plus one list
    per job type, 326 tool names maintained by hand -- are gone. Each tool
    declares which job types may call it in ``agent_core.tool_specs``, and a
    name listed here but nowhere else was not an error, only a capability the
    job type quietly did without.
    """
    proposed = sorted(tool_specs.tools_for_job_type(job_type))

    cfg = config if isinstance(config, dict) else {}

    def _as_list(value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(x).strip() for x in value if str(x).strip()]
        if isinstance(value, str):
            return [str(x).strip() for x in value.split(",") if str(x).strip()]
        return []

    allowlist = set(_as_list(cfg.get("allowed_tools") or cfg.get("tool_allowlist")))
    denylist = set(_as_list(cfg.get("blocked_tools") or cfg.get("tool_denylist")))

    if allowlist:
        proposed = [t for t in proposed if t in allowlist]
    if denylist:
        proposed = [t for t in proposed if t not in denylist]

    role_profile = profile if isinstance(profile, dict) else {}
    blocked = set(_as_list(role_profile.get("blocked_tools")))
    preferred = [
        t for t in _as_list(role_profile.get("preferred_tools")) if t in proposed
    ]
    discouraged = [
        t for t in _as_list(role_profile.get("discouraged_tools")) if t in proposed
    ]
    if blocked:
        proposed = [t for t in proposed if t not in blocked]

    preferred_seen = set()
    preferred_ordered: List[str] = []
    for t in preferred:
        if t not in preferred_seen and t in proposed:
            preferred_seen.add(t)
            preferred_ordered.append(t)

    discouraged_set = set(discouraged)
    middle = [
        t for t in proposed if t not in preferred_seen and t not in discouraged_set
    ]
    tail = []
    for t in discouraged:
        if t in proposed and t not in preferred_seen and t not in tail:
            tail.append(t)

    ordered = preferred_ordered + middle + tail

    try:
        max_tools = int(cfg.get("skill_profile_max_tools", 0) or 0)
    except Exception:
        max_tools = 0
    if max_tools > 0:
        ordered = ordered[: max(1, min(max_tools, len(ordered)))]

    return ordered


def get_tool_selection_config(job: AgentJob) -> Dict[str, Any]:
    """Get adaptive selection settings for tool ranking."""
    cfg = job.config if isinstance(job.config, dict) else {}

    def _as_float(key: str, default: float, lo: float, hi: float) -> float:
        try:
            val = float(cfg.get(key, default))
        except Exception:
            val = default
        return max(lo, min(val, hi))

    def _as_int(key: str, default: int, lo: int, hi: int) -> int:
        try:
            val = int(cfg.get(key, default))
        except Exception:
            val = default
        return max(lo, min(val, hi))

    def _as_mode(key: str, default: str) -> str:
        val = str(cfg.get(key, default) or default).strip().lower()
        return val if val in {"baseline", "adaptive", "thompson"} else default

    policy_mode = _as_mode("tool_selection_policy_mode", "adaptive")

    return {
        "policy_mode": policy_mode,
        "exploration_enabled": bool(
            cfg.get("tool_selection_exploration_enabled", True)
        ),
        "exploration_bonus": _as_float(
            "tool_selection_exploration_bonus", 0.15, 0.0, 2.0
        ),
        "cold_start_bonus": _as_float(
            "tool_selection_cold_start_bonus", 0.05, 0.0, 1.0
        ),
        "min_trials": _as_int("tool_selection_min_trials", 3, 0, 100),
        "failure_penalty": _as_float("tool_selection_failure_penalty", 0.08, 0.0, 1.0),
        "thompson_alpha_prior": _as_float(
            "tool_selection_thompson_alpha_prior", 1.0, 0.1, 100.0
        ),
        "thompson_beta_prior": _as_float(
            "tool_selection_thompson_beta_prior", 1.0, 0.1, 100.0
        ),
        "thompson_temperature": _as_float(
            "tool_selection_thompson_temperature", 1.0, 0.1, 5.0
        ),
        "ab_test_enabled": bool(cfg.get("tool_selection_ab_test_enabled", False)),
        "ab_test_split": _as_float("tool_selection_ab_test_split", 0.5, 0.0, 1.0),
        "ab_test_variant_a": _as_mode("tool_selection_ab_test_variant_a", "adaptive"),
        "ab_test_variant_b": _as_mode("tool_selection_ab_test_variant_b", "thompson"),
        "live_fallback_enabled": bool(
            cfg.get("tool_selection_live_fallback_enabled", True)
        ),
        "live_fallback_min_samples": _as_int(
            "tool_selection_live_fallback_min_samples", 8, 1, 10_000
        ),
        "live_fallback_min_success_rate": _as_float(
            "tool_selection_live_fallback_min_success_rate", 0.2, 0.0, 1.0
        ),
        "live_fallback_to_mode": _as_mode(
            "tool_selection_live_fallback_to_mode", "adaptive"
        ),
        "live_fallback_reset_enabled": bool(
            cfg.get("tool_selection_live_fallback_reset_enabled", True)
        ),
        "live_fallback_reset_min_samples": _as_int(
            "tool_selection_live_fallback_reset_min_samples", 10, 1, 10_000
        ),
        "live_fallback_reset_min_success_rate": _as_float(
            "tool_selection_live_fallback_reset_min_success_rate", 0.55, 0.0, 1.0
        ),
        "stage_schedule_enabled": bool(
            cfg.get("tool_selection_stage_schedule_enabled", False)
        ),
        "stage_discovery_mode": _as_mode(
            "tool_selection_stage_discovery_mode", "thompson"
        ),
        "stage_consolidation_mode": _as_mode(
            "tool_selection_stage_consolidation_mode", "adaptive"
        ),
        "stage_finish_mode": _as_mode("tool_selection_stage_finish_mode", "baseline"),
        "stage_rescue_mode": _as_mode("tool_selection_stage_rescue_mode", "adaptive"),
        "stage_rescue_stall_threshold": _as_int(
            "tool_selection_stage_rescue_stall_threshold", 3, 1, 100
        ),
        "stage_finish_progress": _as_int(
            "tool_selection_stage_finish_progress", 80, 10, 100
        ),
        "stage_discovery_progress": _as_int(
            "tool_selection_stage_discovery_progress", 35, 0, 90
        ),
        "family_diversification_enabled": bool(
            cfg.get("tool_selection_family_diversification_enabled", True)
        ),
        "family_diversification_window": _as_int(
            "tool_selection_family_diversification_window", 6, 1, 100
        ),
        "family_diversification_bonus": _as_float(
            "tool_selection_family_diversification_bonus", 0.08, 0.0, 1.0
        ),
        "family_diversification_target_unique": _as_int(
            "tool_selection_family_diversification_target_unique", 3, 1, 20
        ),
        "feedback_learning_enabled": bool(cfg.get("feedback_learning_enabled", True)),
        "feedback_learning_weight": _as_float(
            "feedback_learning_weight", 0.08, 0.0, 0.6
        ),
        "feedback_learning_max_abs_bias": _as_float(
            "feedback_learning_max_abs_bias", 0.3, 0.0, 1.0
        ),
    }
