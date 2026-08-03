"""Pure scoring and ranking math for adaptive tool selection.

Extracted from ``autonomous_agent_executor`` so the selection policy can be read,
tested, and changed without touching the runtime loop. Everything here is a
function of its arguments: no database, no service state, no ``AgentJob``. The
executor keeps the orchestration (config resolution, mode assignment, cooldown
bookkeeping) and delegates the arithmetic here.

Determinism matters — replay and routing experiments compare rankings across
runs — so any randomness is seeded from stable identity, never from the clock.
"""

from __future__ import annotations

import hashlib
import math
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

VISUALIZATION_TOKENS = ("chart", "diagram", "heatmap", "flowchart", "gantt", "drawio")
RETRIEVAL_PREFIXES = ("search_", "find_", "get_", "list_")
INGESTION_PREFIXES = ("ingest_", "batch_ingest_", "load_", "monitor_")
ANALYSIS_PREFIXES = (
    "read_",
    "summarize_",
    "extract_",
    "analyze_",
    "compare_",
    "identify_",
    "describe_",
    "query_",
    "filter_",
    "aggregate_",
    "join_",
    "transform_",
    "detect_",
    "calculate_",
)
SYNTHESIS_PREFIXES = (
    "create_",
    "generate_",
    "write_",
    "save_",
    "link_",
    "add_",
    "export_",
    "suggest_",
)


def stable_fraction(key: str) -> float:
    """Map a key to a stable [0,1) fraction."""
    digest = hashlib.sha256(str(key).encode("utf-8")).hexdigest()
    bucket = int(digest[:12], 16)
    return float(bucket % 1_000_000) / 1_000_000.0


def normalize_tool_stats_map(raw: Any) -> Dict[str, Dict[str, Any]]:
    """Normalize a ``{tool: {success, failure, last_error}}`` map."""
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for tool, val in raw.items():
        tool_name = str(tool or "").strip()
        if not tool_name or not isinstance(val, dict):
            continue
        out[tool_name] = {
            "success": int(val.get("success", 0) or 0),
            "failure": int(val.get("failure", 0) or 0),
            "last_error": str(val.get("last_error") or "").strip()[:200],
        }
    return out


def merge_tool_stats(
    *stats_maps: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Merge tool stat maps by summing success and failure counts."""
    merged: Dict[str, Dict[str, Any]] = {}
    for smap in stats_maps:
        for tool, val in normalize_tool_stats_map(smap).items():
            cur = merged.get(tool) or {"success": 0, "failure": 0, "last_error": ""}
            cur["success"] = int(cur.get("success", 0) or 0) + int(
                val.get("success", 0) or 0
            )
            cur["failure"] = int(cur.get("failure", 0) or 0) + int(
                val.get("failure", 0) or 0
            )
            if val.get("last_error"):
                cur["last_error"] = str(val.get("last_error") or "")[:200]
            merged[tool] = cur
    return merged


def tool_success_ratio(stat: Dict[str, Any]) -> float:
    """Laplace-smoothed success ratio, so early samples do not dominate."""
    if not isinstance(stat, dict):
        return 0.0
    successes = int(stat.get("success", 0) or 0)
    failures = int(stat.get("failure", 0) or 0)
    return (successes + 1.0) / float(successes + failures + 2.0)


def tool_observation_count(stat: Dict[str, Any]) -> int:
    """Total observed outcomes for a tool."""
    if not isinstance(stat, dict):
        return 0
    successes = int(stat.get("success", 0) or 0)
    failures = int(stat.get("failure", 0) or 0)
    return max(0, successes + failures)


def tool_family(tool: str) -> str:
    """Map a tool to a coarse family for diversification incentives."""
    name = str(tool or "").strip().lower()
    if not name:
        return "unknown"
    if any(token in name for token in VISUALIZATION_TOKENS):
        return "visualization"
    if name.startswith(RETRIEVAL_PREFIXES):
        return "retrieval"
    if name.startswith(INGESTION_PREFIXES):
        return "ingestion"
    if name.startswith(ANALYSIS_PREFIXES):
        return "analysis"
    if name.startswith(SYNTHESIS_PREFIXES):
        return "synthesis"
    return "other"


def feedback_tool_bias(
    tool_name: str,
    state: Optional[Dict[str, Any]],
    *,
    weight: float = 0.08,
    max_abs: float = 0.30,
    enabled: bool = True,
) -> float:
    """Map feedback signals to a bounded additive tool-priority adjustment."""
    if not enabled:
        return 0.0
    tool = str(tool_name or "").strip()
    if not tool or not isinstance(state, dict):
        return 0.0
    feedback = state.get("feedback_learning")
    if not isinstance(feedback, dict):
        return 0.0
    bias_map = feedback.get("tool_bias")
    if not isinstance(bias_map, dict):
        return 0.0
    try:
        signal = float(bias_map.get(tool) or 0.0)
    except Exception:
        signal = 0.0
    signal = max(-1.0, min(1.0, signal))
    adjustment = signal * max(0.0, float(weight))
    return max(-abs(float(max_abs)), min(abs(float(max_abs)), adjustment))


def family_diversification_bonus(
    tool: str,
    *,
    state: Optional[Dict[str, Any]],
    selection_cfg: Optional[Dict[str, Any]],
) -> float:
    """Boost underrepresented tool families based on recent action history."""
    cfg = selection_cfg if isinstance(selection_cfg, dict) else {}
    if not bool(cfg.get("family_diversification_enabled", True)):
        return 0.0
    if not isinstance(state, dict):
        return 0.0
    actions = state.get("actions_taken")
    if not isinstance(actions, list) or not actions:
        return 0.0

    window = max(1, int(cfg.get("family_diversification_window", 6) or 6))
    family_counts: Dict[str, int] = {}
    for row in actions[-window:]:
        if not isinstance(row, dict):
            continue
        action = row.get("action")
        if not isinstance(action, dict):
            continue
        used_tool = str(action.get("tool") or "").strip()
        if not used_tool:
            continue
        family = tool_family(used_tool)
        family_counts[family] = int(family_counts.get(family, 0) or 0) + 1
    if not family_counts:
        return 0.0

    target_unique = max(1, int(cfg.get("family_diversification_target_unique", 3) or 3))
    raw_bonus = float(cfg.get("family_diversification_bonus", 0.08) or 0.08)
    used_count = int(family_counts.get(tool_family(tool), 0) or 0)
    unique_used = len(family_counts)
    diversity_pressure = max(
        0.0, float(target_unique - unique_used) / float(target_unique)
    )

    if used_count <= 0:
        return raw_bonus * (1.0 + 0.5 * diversity_pressure)
    return raw_bonus * diversity_pressure / float(used_count + 1)


def tool_priority_score(
    stat: Dict[str, Any],
    *,
    total_trials: int = 0,
    selection_cfg: Optional[Dict[str, Any]] = None,
    mode: str = "adaptive",
    tool_name: str = "",
    job_id: str = "",
    iteration: int = 0,
    state: Optional[Dict[str, Any]] = None,
    context_tag: str = "",
) -> float:
    """Score a tool for adaptive selection.

    Base quality is the smoothed success ratio. Optional exploration adds an
    uncertainty bonus and a mild cold-start boost, then subtracts a failure
    penalty. Thompson sampling seeds its RNG from job, tool, and iteration so a
    replay of the same state produces the same draw.
    """
    ratio = tool_success_ratio(stat)
    cfg = selection_cfg if isinstance(selection_cfg, dict) else {}
    feedback_adjustment = feedback_tool_bias(
        tool_name,
        state,
        weight=float(cfg.get("feedback_learning_weight", 0.08) or 0.08),
        max_abs=float(cfg.get("feedback_learning_max_abs_bias", 0.3) or 0.3),
        enabled=bool(cfg.get("feedback_learning_enabled", True)),
    )
    mode_norm = str(mode or "adaptive").strip().lower()
    if mode_norm == "baseline":
        return ratio + feedback_adjustment

    if mode_norm == "thompson":
        alpha_prior = float(cfg.get("thompson_alpha_prior", 1.0) or 1.0)
        beta_prior = float(cfg.get("thompson_beta_prior", 1.0) or 1.0)
        temperature = max(0.1, float(cfg.get("thompson_temperature", 1.0) or 1.0))
        successes = max(0, int((stat or {}).get("success", 0) or 0))
        failures = max(0, int((stat or {}).get("failure", 0) or 0))
        forced_part = (
            int((state or {}).get("forced_exploration_attempts", 0) or 0)
            if isinstance(state, dict)
            else 0
        )
        seed_key = (
            f"{job_id}:{tool_name}:{context_tag}:"
            f"{int(iteration or 0)}:{forced_part}:{total_trials}"
        )
        seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:16], 16)
        sample = float(
            random.Random(seed).betavariate(
                alpha_prior + successes, beta_prior + failures
            )
        )
        # Temperature scales exploitation pressure while preserving rank ordering.
        score = max(0.0, min(1.0, math.pow(sample, 1.0 / temperature)))
        return score + feedback_adjustment

    if not bool(cfg.get("exploration_enabled", True)):
        return ratio + feedback_adjustment

    observations = tool_observation_count(stat)
    failures = int((stat or {}).get("failure", 0) or 0) if isinstance(stat, dict) else 0

    exploration_bonus = float(cfg.get("exploration_bonus", 0.15) or 0.15)
    cold_start_bonus = float(cfg.get("cold_start_bonus", 0.05) or 0.05)
    min_trials = int(cfg.get("min_trials", 3) or 3)
    failure_penalty = float(cfg.get("failure_penalty", 0.08) or 0.08)

    uncertainty_bonus = exploration_bonus / math.sqrt(float(observations) + 1.0)
    ucb_bonus = 0.0
    if total_trials > 0:
        ucb_bonus = (
            0.5
            * exploration_bonus
            * math.sqrt(
                max(
                    0.0,
                    math.log(float(total_trials) + 1.0) / (float(observations) + 1.0),
                )
            )
        )
    cold_bonus = cold_start_bonus if observations < min_trials else 0.0
    penalty = failure_penalty * (float(failures) / float(observations + 1))

    return (
        ratio
        + uncertainty_bonus
        + ucb_bonus
        + cold_bonus
        - penalty
        + feedback_adjustment
    )


def rank_tools_for_selection(
    tools: List[str],
    combined_stats: Dict[str, Dict[str, Any]],
    *,
    selection_cfg: Optional[Dict[str, Any]] = None,
    mode: str = "adaptive",
    job_id: str = "",
    iteration: int = 0,
    state: Optional[Dict[str, Any]] = None,
    context_tag: str = "",
) -> List[str]:
    """Rank candidate tools by score, then by quality, then deterministically."""
    if not isinstance(tools, list) or not tools:
        return []
    stats = combined_stats if isinstance(combined_stats, dict) else {}
    total_trials = sum(tool_observation_count(stats.get(t, {})) for t in tools)

    scored: List[Tuple[str, float, float]] = []
    for tool in [str(t).strip() for t in tools if str(t).strip()]:
        base_score = tool_priority_score(
            stats.get(tool, {}),
            total_trials=total_trials,
            selection_cfg=selection_cfg,
            mode=mode,
            tool_name=tool,
            job_id=job_id,
            iteration=iteration,
            state=state,
            context_tag=context_tag,
        )
        family_bonus = family_diversification_bonus(
            tool, state=state, selection_cfg=selection_cfg
        )
        scored.append((tool, base_score + family_bonus, base_score))

    ranked = sorted(
        scored,
        key=lambda row: (
            -float(row[1]),
            -float(row[2]),
            -tool_success_ratio(stats.get(row[0], {})),
            tool_observation_count(stats.get(row[0], {})),
            row[0],
        ),
    )
    return [row[0] for row in ranked]


def is_tool_in_cooldown(
    tool: str, cooldowns: Dict[str, Any], current_iteration: int
) -> bool:
    """Return true while a tool is still cooling down at the given iteration."""
    if not isinstance(cooldowns, dict):
        return False
    key = str(tool)
    if key not in cooldowns:
        return False
    try:
        until = int(cooldowns.get(key, 0) or 0)
    except Exception:
        return False
    if until <= 0:
        return False
    return until >= int(current_iteration or 0)


def apply_decay_to_prior_counts(
    success_count: int,
    failure_count: int,
    updated_at: Optional[datetime],
    *,
    now: Optional[datetime] = None,
    enabled: bool = True,
    half_life_days: float = 45.0,
    min_factor: float = 0.01,
) -> Tuple[int, int]:
    """Exponentially decay prior counts by age, so stale evidence loses weight."""
    successes = max(0, int(success_count or 0))
    failures = max(0, int(failure_count or 0))
    if not enabled or updated_at is None:
        return successes, failures

    now_dt = now or datetime.utcnow()

    def _to_utc_naive(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value
        return value.astimezone(timezone.utc).replace(tzinfo=None)

    try:
        age_days = (
            _to_utc_naive(now_dt) - _to_utc_naive(updated_at)
        ).total_seconds() / 86400.0
    except Exception:
        return successes, failures
    if age_days <= 0:
        return successes, failures

    half_life = max(1.0, float(half_life_days))
    factor = math.pow(0.5, age_days / half_life)
    factor = max(float(min_factor), max(0.0, min(1.0, factor)))
    return (
        max(0, int(round(float(successes) * factor))),
        max(0, int(round(float(failures) * factor))),
    )
