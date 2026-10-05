"""MCP tools for sandbox skills.

An external agent holding an API key gets the same four operations an
autonomous job and chat have: list, load, run, propose. The handlers are the
ones those surfaces use (`services/agent_sandbox_skill_tools.py`) and the input
schemas are read from the tool specs, so a skill behaves the same whichever
way it is reached and there is no second description to keep in step.

Two things are specific to this surface. The scope a call needs follows what
it does -- reading a skill is `read`, running one or proposing one is `write`
-- and the working directory is keyed on the API key, so two keys belonging to
one user do not share files.

Compared with `docker_execute`, which this server already offers, this is the
narrower door: the image is one the server allows, the command runs with no
network and no capabilities, and only a skill whose control has passed can be
run at all.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent_core import tool_specs
from app.mcp.auth import MCPAuthContext

#: Each tool, and the API-key scope a call to it needs.
SCOPES: Dict[str, str] = {
    "list_sandbox_skills": "read",
    "load_sandbox_skill": "read",
    "run_sandbox_skill": "write",
    "propose_sandbox_skill": "write",
}


class SandboxSkillTool:
    names: Tuple[str, ...] = tuple(SCOPES)

    @staticmethod
    def description(name: str) -> str:
        spec = tool_specs.spec_for(name)
        return spec.description if spec else ""

    @staticmethod
    def input_schema(name: str) -> Dict[str, Any]:
        spec = tool_specs.spec_for(name)
        return dict(spec.parameters) if spec else {"type": "object", "properties": {}}

    async def execute(
        self,
        name: str,
        auth: MCPAuthContext,
        db: AsyncSession,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        from app.services import agent_sandbox_skill_tools

        if name not in SCOPES:
            return {"error": f"Unknown sandbox skill tool: {name}"}
        auth.require_scope(SCOPES[name])

        context = SimpleNamespace(
            db=db,
            job=None,
            user_id=auth.user_id,
            state={},
            extra={"workdir_key": f"mcp-{auth.api_key.id}"},
        )
        handler = getattr(agent_sandbox_skill_tools, name)
        return await handler(dict(arguments or {}), context)
