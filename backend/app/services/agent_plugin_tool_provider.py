"""Executing a tool a plugin contributed.

This is a provider in the sense ``agent_tool_dispatch`` already means: it sits
in the same ``AgentToolRegistry`` as the built-in providers, receives the same
``AgentToolExecutionContext``, and is asked the same two questions. Nothing in
the dispatch layer had to learn what a plugin is.

``can_handle`` is synchronous and resolution needs the database, so the
provider claims the reserved ``p_`` namespace on sight and does the real lookup
in ``execute``. That is honest rather than approximate: the prefix is reserved
for contributed tools and refused to built-ins at install, so a ``p_`` name is
always this provider's to answer -- including when the answer is "you do not
have that plugin enabled", which is a better error than the silent fallthrough
a name-set check would produce.

Execution delegates to ``CustomToolService``, which is what applies the tool
policy engine, the approval gate and the audit record. A contributed tool is
therefore governed by the machinery that governs every other custom tool, and
its classification comes from its executor type rather than from anything its
author declared.
"""

from __future__ import annotations

from typing import Any, Dict

from loguru import logger
from sqlalchemy import select

from app.agent_core.plugin_specs import NAME_PREFIX
from app.services.agent_tool_dispatch import AgentToolExecutionContext


class PluginToolProvider:
    """Resolves and runs tools contributed by the caller's enabled plugins."""

    name = "plugins"

    @property
    def supported_tools(self) -> set[str]:
        # Which tools exist depends on who is asking, and this property has no
        # context to answer with. Nothing in the codebase reads it -- the
        # registry resolves through `can_handle` -- so the honest answer is the
        # empty set rather than a guess that would be wrong for every user.
        return set()

    def can_handle(self, tool_name: str, context: AgentToolExecutionContext) -> bool:
        return str(tool_name or "").startswith(NAME_PREFIX)

    async def execute(
        self,
        tool_name: str,
        params: Dict[str, Any],
        context: AgentToolExecutionContext,
    ) -> Any:
        from app.models.user import User
        from app.services import plugin_registry
        from app.services.custom_tool_service import (
            CustomToolService,
            ToolExecutionError,
        )

        owner_id = getattr(context, "user_id", None)
        if owner_id is None:
            job = getattr(context, "job", None)
            owner_id = getattr(job, "user_id", None)
        if owner_id is None:
            return {
                "success": False,
                "error": (
                    f"Cannot run {tool_name!r}: no owning user for this "
                    "execution, and a contributed tool is always somebody's."
                ),
            }

        contributed = await plugin_registry.resolve_tool(
            context.db, owner_id, tool_name
        )
        if contributed is None:
            # Say which of the two reasons it is, because they need different
            # fixes: an unknown tool is a planning error, a disabled plugin is
            # a settings one.
            return {
                "success": False,
                "error": (
                    f"No enabled plugin contributes {tool_name!r}. It may not "
                    "be installed, or its plugin may be disabled."
                ),
            }

        user = (
            await context.db.execute(select(User).where(User.id == owner_id))
        ).scalar_one_or_none()

        inputs = params if isinstance(params, dict) else {}
        try:
            return await CustomToolService().execute_tool(
                tool=contributed, inputs=inputs, user=user, db=context.db
            )
        except ToolExecutionError as exc:
            # An approval gate raises through here, and its message carries the
            # approval id the caller needs. Passing the text through unchanged
            # keeps that usable.
            return {"success": False, "error": str(exc)}
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(f"Contributed tool {tool_name!r} failed")
            return {"success": False, "error": f"{type(exc).__name__}: {exc}"}
