"""Generic runtime loop primitives for autonomous agents."""

from __future__ import annotations

from typing import Any


class AgentRuntimeRunner:
    """Drives a generic observe/think/act/evaluate loop through an adapter."""

    async def run(self, adapter: Any) -> Any:
        while await adapter.can_continue():
            await adapter.on_iteration_start()
            try:
                observation = await adapter.observe_phase()
                decision = await adapter.think_phase(observation)
                if bool(decision.get("goal_achieved")) or bool(decision.get("should_stop")):
                    break

                action_bundle = await adapter.act_phase(decision)
                if isinstance(action_bundle, dict) and action_bundle.get("terminal_result") is not None:
                    return action_bundle.get("terminal_result")

                evaluation = await adapter.evaluate_phase(decision, action_bundle)
                await adapter.on_iteration_complete(observation, decision, action_bundle, evaluation)
                if bool((evaluation or {}).get("should_stop")):
                    break
            except Exception as exc:
                should_continue = await adapter.on_iteration_error(exc)
                if not should_continue:
                    break
        return await adapter.build_run_result()
