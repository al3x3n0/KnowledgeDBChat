"""Runtime policy helpers extracted from AutonomousAgentExecutor."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from app.services import agent_tool_scoring


class AgentRuntimePolicyService:
    """Shared runtime policy decisions for the autonomous executor."""

    def resolve_tool_selection_mode(
        self,
        executor: Any,
        job: Any,
        *,
        state: Optional[Dict[str, Any]] = None,
        selection_cfg: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Resolve configured tool-selection mode with optional A/B override."""
        cfg = (
            selection_cfg
            if isinstance(selection_cfg, dict)
            else executor._get_tool_selection_config(job)
        )
        base_mode = (
            str(cfg.get("policy_mode", "adaptive") or "adaptive").strip().lower()
        )
        if base_mode not in {"baseline", "adaptive", "thompson"}:
            base_mode = "adaptive"

        assignment = {
            "enabled": bool(cfg.get("ab_test_enabled", False)),
            "bucket": 0.0,
            "variant": "configured",
            "mode": base_mode,
        }
        if isinstance(state, dict):
            executor._maybe_reset_live_mode_override(job, state, cfg)

        if not assignment["enabled"]:
            override_mode = (
                str((state or {}).get("tool_selection_mode_override") or "")
                .strip()
                .lower()
                if isinstance(state, dict)
                else ""
            )
            if override_mode in {"baseline", "adaptive", "thompson"}:
                base_mode = override_mode
                assignment["override_active"] = True
                assignment["mode"] = base_mode
            elif isinstance(state, dict):
                base_mode = executor._apply_goal_stage_policy_mode(
                    state, base_mode, cfg
                )
            if isinstance(state, dict):
                state["tool_selection_effective_mode"] = base_mode
                state["tool_selection_ab_assignment"] = assignment
                base_mode = executor._apply_live_mode_fallback_guardrail(
                    job, state, base_mode, cfg
                )
                state["tool_selection_effective_mode"] = base_mode
                assignment["mode"] = base_mode
                assignment["goal_stage"] = str(
                    state.get("tool_selection_goal_stage") or ""
                )
            return base_mode, assignment

        variant_a = (
            str(cfg.get("ab_test_variant_a", "adaptive") or "adaptive").strip().lower()
        )
        variant_b = (
            str(cfg.get("ab_test_variant_b", "thompson") or "thompson").strip().lower()
        )
        if variant_a not in {"baseline", "adaptive", "thompson"}:
            variant_a = "adaptive"
        if variant_b not in {"baseline", "adaptive", "thompson"}:
            variant_b = "thompson"
        split_raw = cfg.get("ab_test_split", 0.5)
        try:
            split = float(0.5 if split_raw is None else split_raw)
        except Exception:
            split = 0.5
        split = max(0.0, min(1.0, split))

        key = f"{job.user_id}:{job.id}:{job.job_type}"
        bucket = self._stable_fraction(key)
        mode = variant_a if bucket < split else variant_b
        assignment = {
            "enabled": True,
            "bucket": bucket,
            "split": split,
            "variant": "A" if bucket < split else "B",
            "mode": mode,
            "variant_a": variant_a,
            "variant_b": variant_b,
            "configured_mode": base_mode,
        }
        override_mode = (
            str((state or {}).get("tool_selection_mode_override") or "").strip().lower()
            if isinstance(state, dict)
            else ""
        )
        if override_mode in {"baseline", "adaptive", "thompson"}:
            mode = override_mode
            assignment["override_active"] = True
            assignment["mode"] = mode
        elif isinstance(state, dict):
            mode = executor._apply_goal_stage_policy_mode(state, mode, cfg)
        if isinstance(state, dict):
            mode = executor._apply_live_mode_fallback_guardrail(job, state, mode, cfg)
            state["tool_selection_effective_mode"] = mode
            assignment["mode"] = mode
            assignment["goal_stage"] = str(state.get("tool_selection_goal_stage") or "")
            state["tool_selection_ab_assignment"] = assignment
        return mode, assignment

    def _stable_fraction(self, key: str) -> float:
        """Map a key to stable [0,1) fraction."""
        return agent_tool_scoring.stable_fraction(key)
