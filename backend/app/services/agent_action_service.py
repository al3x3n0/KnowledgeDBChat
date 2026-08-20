"""Action phase helpers extracted from AutonomousAgentExecutor."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from loguru import logger

from app.services.agent_execution_journal_service import agent_execution_journal_service
from app.services.agent_tool_dispatch import AgentToolExecutionContext
from app.services.agent_tool_validation import (
    coerce_tool_params,
    validate_tool_params,
)


def _surface_nested_error(result: Dict[str, Any]) -> None:
    """Lift a nested failure reason to the top level.

    A sweep of the tool catalog found fourteen data-analysis tools reporting
    {"success": False, "data": {"error": ...}} while the rest report a
    top-level "error". Everything that reacts to a failure keys off the
    top-level field, so those tools failed without a reason attached: no
    nothing for a reader to act on.
    """
    if not isinstance(result, dict) or result.get("error"):
        return
    if result.get("success"):
        return
    nested = result.get("data")
    if isinstance(nested, dict) and nested.get("error"):
        result["error"] = str(nested["error"])[:500]


class AgentActionService:
    """Execute autonomous tool actions."""

    async def act(
        self,
        executor: Any,
        job: Any,
        action: Dict[str, Any],
        state: Dict[str, Any],
        db: Any,
    ) -> Dict[str, Any]:
        """Execute the decided action and return normalized results."""
        intent = await agent_execution_journal_service.begin_tool_call(
            executor=executor,
            job=job,
            state=state,
            action=action,
            db=db,
        )
        try:
            result = await self._act_unjournaled(executor, job, action, state, db)
        except Exception as exc:
            await agent_execution_journal_service.complete_tool_call(
                executor=executor,
                job=job,
                state=state,
                intent=intent,
                result={"success": False, "error": str(exc)},
                db=db,
            )
            raise
        await agent_execution_journal_service.complete_tool_call(
            executor=executor,
            job=job,
            state=state,
            intent=intent,
            result=result,
            db=db,
        )
        return result

    async def _act_unjournaled(
        self,
        executor: Any,
        job: Any,
        action: Dict[str, Any],
        state: Dict[str, Any],
        db: Any,
    ) -> Dict[str, Any]:
        """Execute one action after its durable intent has been recorded."""
        action = executor._apply_default_scope_to_action(dict(action), job)
        tool_name = action.get("tool")
        params = action.get("params", {})

        result = {
            "tool": tool_name,
            "success": False,
            "findings": [],
            "artifacts": [],
            "timestamp": datetime.utcnow().isoformat(),
        }

        scope_violation = executor._validate_action_scope(job, action)
        scope_guard_cfg = executor._get_scope_guard_config(job)
        if scope_violation:
            event = {
                "timestamp": datetime.utcnow().isoformat(),
                "tool": str(tool_name or ""),
                "error": scope_violation,
                "default_source_id": executor._resolve_default_source_scope(job),
                "action_source_id": str((params or {}).get("source_id") or "").strip()
                or None,
                "enforced": bool(scope_guard_cfg.get("enforce", True)),
            }
            events = state.get("scope_guard_events")
            if not isinstance(events, list):
                events = []
            events.append(event)
            state["scope_guard_events"] = events[-100:]
            state["scope_guard_blocks"] = (
                int(state.get("scope_guard_blocks", 0) or 0) + 1
            )

            if bool(scope_guard_cfg.get("enforce", True)):
                result["error"] = scope_violation
                result["scope_guard"] = event
                return result

        # Check the call against the tool's own schema before running it, so a
        # malformed call is rejected with the offending field named rather than
        # failing somewhere inside the tool.
        # A lone string where the schema wants a list of strings is repaired
        # first: it cannot mean anything else, and refusing it costs an
        # iteration to re-send the same value inside brackets.
        coerced = coerce_tool_params(str(tool_name or ""), params)
        if coerced:
            result["coerced_params"] = coerced
        invalid = validate_tool_params(str(tool_name or ""), params)
        if invalid:
            result["error"] = invalid
            return result

        try:
            handled, handled_result = await executor.tool_registry.try_execute(
                str(tool_name or ""),
                params if isinstance(params, dict) else {},
                AgentToolExecutionContext(
                    mode="autonomous",
                    db=db,
                    service=executor,
                    job=job,
                    state=state,
                    idempotency_key=str(action.get("_idempotency_key") or "") or None,
                ),
            )
            if handled:
                result.update(
                    handled_result if isinstance(handled_result, dict) else {}
                )
                _surface_nested_error(result)
                return result

            if job.job_type == "research":
                prefer_sources = (job.config or {}).get("prefer_sources") or []
                if isinstance(prefer_sources, str):
                    prefer_sources = [
                        x.strip() for x in prefer_sources.split(",") if x.strip()
                    ]
                prefer_sources = [
                    str(x).strip().lower() for x in prefer_sources if str(x).strip()
                ]

                external_tools = {
                    "search_arxiv",
                    "monitor_arxiv_topic",
                    "ingest_paper_by_id",
                    "batch_ingest_papers",
                }
                if tool_name in external_tools and "arxiv" not in prefer_sources:
                    result[
                        "error"
                    ] = "External paper research is disabled for this job (prefer_sources excludes arxiv)."
                    return result

                if (
                    tool_name in external_tools
                    and prefer_sources
                    and prefer_sources[0] == "documents"
                    and "arxiv" in prefer_sources
                ):
                    has_internal_attempt = any(
                        (a.get("action") or {}).get("tool")
                        in {"search_documents", "search_with_filters"}
                        for a in (state.get("actions_taken") or [])
                    )
                    if not has_internal_attempt and int(job.iteration or 0) <= 2:
                        result[
                            "error"
                        ] = "Prefer internal documents first (run a document search before arXiv)."
                        return result
            else:
                result["error"] = f"Unknown or unimplemented tool: {tool_name}"
                logger.warning(f"Tool not implemented: {tool_name}")
        except Exception as exc:
            logger.error(f"Error executing tool {tool_name}: {exc}")
            result["error"] = str(exc)

        return result
