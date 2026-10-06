"""Autonomous-job tools: the ``sandbox_skill`` provider.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

from typing import Any, Dict

from app.services.agent_tool_providers.base import (
    AgentToolExecutionContext,
    FunctionToolProvider,
)


def build_autonomous_sandbox_skill_provider(executor: Any) -> FunctionToolProvider:
    """Sandbox skills: find one, read it, run it, propose another.

    A table and nothing else. The handlers live in
    ``agent_sandbox_skill_tools`` and are imported when called, the way every
    other provider here reaches its service.

    Answers in chat as well as in autonomous jobs. Every spec is advertised to
    chat whatever its provider's modes, so an autonomous-only provider is a
    tool chat is offered and then told is unknown. The handlers need a user
    and something to key a working directory on, and a conversation supplies
    both.
    """

    def _handler(name: str):
        async def _call(params: Dict[str, Any], ctx: AgentToolExecutionContext) -> Any:
            from app.services import agent_sandbox_skill_tools

            return await getattr(agent_sandbox_skill_tools, name)(params, ctx)

        return _call

    return FunctionToolProvider(
        name="sandbox_skill_tools",
        modes={"autonomous", "chat"},
        handlers={
            name: _handler(name)
            for name in (
                "list_sandbox_skills",
                "load_sandbox_skill",
                "run_sandbox_skill",
                "propose_sandbox_skill",
            )
        },
    )
