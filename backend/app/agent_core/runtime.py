"""Generic runtime loop primitives for autonomous agents."""

from __future__ import annotations

import time
from typing import Any


def _ms_since(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


class AgentRuntimeRunner:
    """Drives a generic observe/think/act/evaluate loop through an adapter.

    Each phase is timed and the times ride on the action bundle as
    ``phase_ms``, for the iteration's log entry: a run's log said when each
    iteration ended and not where its minutes went, so a profile could not
    tell a run waiting on the model from one waiting on a simulator.
    """

    async def run(self, adapter: Any) -> Any:
        while await adapter.can_continue():
            await adapter.on_iteration_start()
            try:
                phase_ms = {}
                start = time.monotonic()
                observation = await adapter.observe_phase()
                phase_ms["observe"] = _ms_since(start)
                start = time.monotonic()
                decision = await adapter.think_phase(observation)
                phase_ms["think"] = _ms_since(start)
                if bool(decision.get("goal_achieved")) or bool(
                    decision.get("should_stop")
                ):
                    break

                start = time.monotonic()
                action_bundle = await adapter.act_phase(decision)
                phase_ms["act"] = _ms_since(start)
                if (
                    isinstance(action_bundle, dict)
                    and action_bundle.get("terminal_result") is not None
                ):
                    return action_bundle.get("terminal_result")

                start = time.monotonic()
                evaluation = await adapter.evaluate_phase(decision, action_bundle)
                phase_ms["evaluate"] = _ms_since(start)
                if isinstance(action_bundle, dict):
                    action_bundle["phase_ms"] = phase_ms
                await adapter.on_iteration_complete(
                    observation, decision, action_bundle, evaluation
                )
                if bool((evaluation or {}).get("should_stop")):
                    break
            except Exception as exc:
                if bool(getattr(exc, "fatal", False)):
                    raise
                should_continue = await adapter.on_iteration_error(exc)
                if not should_continue:
                    break
        return await adapter.build_run_result()
