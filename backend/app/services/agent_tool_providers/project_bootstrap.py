"""Autonomous-job tools: the ``project_bootstrap`` provider.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

from typing import Any, Dict

from app.services.agent_tool_providers.base import (
    AgentToolExecutionContext,
    FunctionToolProvider,
)


def build_autonomous_project_bootstrap_provider(executor: Any) -> FunctionToolProvider:
    """Project bootstrap tool for AutonomousAgentExecutor."""

    async def _project_bootstrap(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services.project_profile_service import build_project_profile

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        source_id = str(
            params.get("source_id") or ""
        ).strip() or executor._resolve_default_source_scope(job)
        max_files = int(params.get("max_files", 400) or 400)
        profile = await build_project_profile(
            job,
            ctx.db,
            source_id=source_id,
            max_files=max_files,
        )
        if not profile.get("sampled_files"):
            return {"error": "No repository-like files found to build project profile."}
        state["project_profile"] = profile
        return {
            "success": True,
            "data": profile,
            "findings": [
                {
                    "type": "project_profile",
                    "title": "Project bootstrap profile generated",
                    "source_id": profile.get("source_id"),
                    "detected_stack": profile.get("detected_stack", []),
                    "sampled_files": profile.get("sampled_files", 0),
                }
            ],
            "artifacts": [
                {
                    "type": "project_profile",
                    "source_id": profile.get("source_id"),
                    "sampled_files": profile.get("sampled_files", 0),
                    "detected_stack": profile.get("detected_stack", []),
                }
            ],
        }

    return FunctionToolProvider(
        name="autonomous_project_bootstrap_tools",
        modes={"autonomous"},
        handlers={
            "project_bootstrap": _project_bootstrap,
        },
    )
