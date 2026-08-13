"""Action phase helpers extracted from AutonomousAgentExecutor."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger

from app.services.agent_execution_journal_service import agent_execution_journal_service
from app.services.agent_tool_dispatch import AgentToolExecutionContext


def _surface_nested_error(result: Dict[str, Any]) -> None:
    """Lift a nested failure reason to the top level.

    A sweep of the tool catalog found fourteen data-analysis tools reporting
    {"success": False, "data": {"error": ...}} while the rest report a
    top-level "error". Everything that reacts to a failure keys off the
    top-level field, so those tools failed without a reason attached: no
    fallback suggestion, and nothing for a reader to act on.
    """
    if not isinstance(result, dict) or result.get("error"):
        return
    if result.get("success"):
        return
    nested = result.get("data")
    if isinstance(nested, dict) and nested.get("error"):
        result["error"] = str(nested["error"])[:500]


class AgentActionService:
    """Execute autonomous tool actions and fallback routing."""

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

        cfg = job.config if isinstance(job.config, dict) else {}
        fallback_enabled = cfg.get("tool_fallback_enabled")
        if fallback_enabled is None:
            # Off by default. Running a different tool and reporting success
            # under the requested tool's name turns "this tool is broken" into
            # "the agent got an answer": a broken arXiv search became a
            # knowledge-base search whose unrelated documents were recorded as
            # research findings. A failing tool now reports its failure, and
            # the agent chooses what to do with it.
            fallback_enabled = False

        if not fallback_enabled and not result.get("success") and result.get("error"):
            # Name an alternative without running it, so recovery stays the
            # agent's decision and appears in its trajectory.
            suggestion = self._fallback_action_for(
                executor, job, tool_name=str(tool_name or "").strip(), params=params
            )
            suggested_tool = str((suggestion or {}).get("tool") or "").strip()
            if suggested_tool and suggested_tool != tool_name:
                result["suggested_alternative_tool"] = suggested_tool

        if fallback_enabled and not result.get("success") and result.get("error"):
            depth = 0
            try:
                depth = int(params.get("_fallback_depth") or 0)
            except Exception:
                depth = 0

            max_depth = 1
            try:
                max_depth = int(cfg.get("tool_fallback_max_depth") or 1)
            except Exception:
                max_depth = 1
            max_depth = max(0, min(max_depth, 2))

            if depth < max_depth:
                fb_action_base = self._fallback_action_for(
                    executor, job, tool_name=str(tool_name or "").strip(), params=params
                )
                if fb_action_base and isinstance(fb_action_base, dict):
                    try:
                        fb_action = {
                            "tool": fb_action_base.get("tool"),
                            "params": {
                                **(fb_action_base.get("params") or {}),
                                "_fallback_depth": depth + 1,
                                "_fallback_from": tool_name,
                            },
                        }
                        fb = await self.act(executor, job, fb_action, state, db)
                        result["primary_tool"] = tool_name
                        result["primary_error"] = result.get("error")
                        result["fallback"] = fb
                        if isinstance(fb, dict) and fb.get("success"):
                            result["success"] = True
                            result["tool"] = fb.get("tool") or fb_action.get("tool")
                            result["data"] = fb.get("data")
                            result["findings"] = fb.get("findings") or result.get(
                                "findings"
                            )
                            result["artifacts"] = fb.get("artifacts") or result.get(
                                "artifacts"
                            )
                            result[
                                "note"
                            ] = f"Primary tool failed; used fallback tool: {result['tool']}"
                    except Exception:
                        pass

        return result

    def _fallback_action_for(
        self,
        executor: Any,
        job: Any,
        *,
        tool_name: str,
        params: Any,
    ) -> Optional[Dict[str, Any]]:
        cfg = job.config if isinstance(job.config, dict) else {}
        fallback_map = (
            cfg.get("tool_fallback_map")
            if isinstance(cfg.get("tool_fallback_map"), dict)
            else {}
        )
        if (
            isinstance(fallback_map, dict)
            and tool_name in fallback_map
            and isinstance(fallback_map.get(tool_name), dict)
        ):
            entry = fallback_map.get(tool_name) or {}
            fallback_tool = str(entry.get("tool") or "").strip()
            fallback_params = (
                entry.get("params") if isinstance(entry.get("params"), dict) else {}
            )
            if fallback_tool:
                return {"tool": fallback_tool, "params": dict(fallback_params)}

        if tool_name == "search_documents":
            return None

        policies = executor._get_tool_fallback_policies()
        policy = dict(policies.get("_default") or {})
        policy.update(policies.get(str(getattr(job, "job_type", "") or "")) or {})
        entry = policy.get(tool_name) or policy.get("__default__")
        if not isinstance(entry, dict):
            entry = None

        def _goal_query() -> str:
            return str(getattr(job, "goal", "") or "").strip()

        def _query_from(param: str) -> str:
            if param == "goal":
                return _goal_query()
            return str((params or {}).get(param) or "").strip()

        if entry:
            fallback_tool = str(entry.get("tool") or "").strip()
            param = str(entry.get("param") or "goal").strip()
            if fallback_tool == "search_documents":
                query = _query_from(param)
                if not query and param != "goal":
                    query = _goal_query()
                if query:
                    return {
                        "tool": "search_documents",
                        "params": {"query": query, "limit": 5},
                    }

        goal_query = _goal_query()
        if goal_query:
            return {
                "tool": "search_documents",
                "params": {"query": goal_query, "limit": 5},
            }
        return None
