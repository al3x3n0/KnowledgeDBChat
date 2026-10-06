"""Autonomous-job tools: the ``observability`` provider.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from sqlalchemy import select

from app.services.agent_tool_providers.base import (
    AgentToolExecutionContext,
    FunctionToolProvider,
)
from app.services.agent_tool_providers.common import _tool_snapshot_context


def build_autonomous_observability_provider(executor: Any) -> FunctionToolProvider:
    """Observability, analytics, and conditional tools for AutonomousAgentExecutor."""

    async def _recall_prior_findings(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services import agent_prior_findings

        job = ctx.job
        types = params.get("finding_types")
        if isinstance(types, str):
            types = [t.strip() for t in types.split(",") if t.strip()]

        outcome = await agent_prior_findings.recall(
            db=ctx.db,
            user_id=job.user_id,
            exclude_job_id=job.id,
            finding_types=[str(t) for t in types] if isinstance(types, list) else None,
            subject=str(params.get("subject") or ""),
            job_type=str(params.get("job_type") or ""),
            limit=int(params.get("limit", 10) or 10),
        )

        # Asked with no filter and nothing matched, the useful answer is the
        # vocabulary rather than an empty list: a caller cannot guess type
        # names that are whatever earlier tools happened to emit.
        if not outcome["findings"]:
            available = await agent_prior_findings.available_types(
                db=ctx.db, user_id=job.user_id, exclude_job_id=job.id
            )
            # Name the types that do not exist, rather than reporting a
            # generic miss beside a list. A live run asked for 'measurement',
            # then 'record_measurement', then 'benchmark' -- none of which is
            # a type anything emits -- while the list of real ones was sitting
            # in the reply each time. Saying "you asked for X and there is no
            # X" is a different sentence from "nothing matched".
            asked = [t for t in (types or []) if isinstance(t, str)]
            unknown = [t for t in asked if t not in available]
            catalogue = (
                ", ".join(f"{k} ({v})" for k, v in available.items()) or "none yet"
            )
            if unknown:
                message = (
                    f"No such evidence type: {', '.join(unknown)}. Earlier runs "
                    f"produced these and only these: {catalogue}. Ask again "
                    "with one of those names."
                )
            elif asked:
                message = (
                    f"{', '.join(asked)} exist, but nothing matched the "
                    "subject filter. Try fewer words, or drop `subject` to see "
                    "everything of that type."
                )
            else:
                message = (
                    "No earlier finding matched. Evidence types this user's "
                    f"previous runs did produce: {catalogue}"
                )
            return {
                "success": True,
                "data": {"findings": [], "count": 0, "available_types": available},
                "message": message,
            }

        # Returned under `findings` so they enter state the way every other
        # tool's findings do -- that is what makes them citable in
        # derived_from. Each carries recalled=True, which keeps them out of
        # the goal contract's count.
        return {
            "success": True,
            "data": {
                "count": outcome["count"],
                "types_found": outcome["types_found"],
                "jobs_scanned": outcome["jobs_scanned"],
                "note": outcome["note"],
            },
            "findings": outcome["findings"],
        }

    def _tool_calls(log: Any) -> Iterable[tuple[str, Optional[str], Dict[str, Any]]]:
        """(tool, error, entry) for each tool call in an execution log.

        Only the executor's per-iteration entry is a call. Operator decisions
        also carry an `action` ("approve"), and were being counted as calls
        to a tool of that name. A call failed if it recorded an error or
        recorded that it did not succeed.
        """
        for entry in log or []:
            if not isinstance(entry, dict):
                continue
            if entry.get("phase") not in (None, "iteration_complete"):
                continue
            tool = entry.get("action")
            if not tool:
                continue
            error = entry.get("error")
            if not error and entry.get("success") is False:
                error = "failed"
            yield str(tool), (str(error) if error else None), entry

    async def _get_job_history(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.models.agent_job import AgentJob as AgentJobModel

        job = ctx.job
        try:
            stmt = (
                select(AgentJobModel)
                .where(
                    AgentJobModel.user_id == job.user_id,
                    AgentJobModel.id != job.id,
                )
                .order_by(AgentJobModel.created_at.desc())
            )
            jt_filter = str(params.get("job_type", "")).strip()
            if jt_filter:
                stmt = stmt.where(AgentJobModel.job_type == jt_filter)
            status_filter = str(params.get("status", "")).strip()
            if status_filter:
                stmt = stmt.where(AgentJobModel.status == status_filter)
            limit = max(1, min(int(params.get("limit", 10) or 10), 50))
            stmt = stmt.limit(limit)

            past_jobs = (await ctx.db.execute(stmt)).scalars().all()
            return {
                "success": True,
                "data": {
                    "jobs": [
                        {
                            "id": str(j.id),
                            "goal": (j.goal or "")[:200],
                            "job_type": j.job_type,
                            "status": j.status,
                            "iteration": j.iteration,
                            "tool_calls_used": j.tool_calls_used,
                            "llm_calls_used": j.llm_calls_used,
                            "tokens_used": j.tokens_used,
                            "error": (j.error or "")[:200] if j.error else None,
                            "created_at": j.created_at.isoformat()
                            if j.created_at
                            else None,
                            "completed_at": j.completed_at.isoformat()
                            if j.completed_at
                            else None,
                            "duration_minutes": round(
                                (j.completed_at - j.started_at).total_seconds() / 60, 1
                            )
                            if j.started_at and j.completed_at
                            else None,
                        }
                        for j in past_jobs
                    ],
                    "count": len(past_jobs),
                },
            }
        except Exception as exc:
            return {"error": f"Failed to get job history: {exc}"}

    async def _get_job_metrics(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID as _UUID

        from app.models.agent_job import AgentJob as AgentJobModel

        job = ctx.job
        try:
            target_id_str = str(params.get("job_id", "")).strip()
            if target_id_str:
                target_job = await ctx.db.get(AgentJobModel, _UUID(target_id_str))
            else:
                target_job = job
            if not target_job:
                return {"error": f"Job not found: {target_id_str}"}
            if target_job.user_id != job.user_id:
                return {"error": "Not authorized to view this job's metrics"}

            duration = None
            if target_job.started_at and target_job.completed_at:
                duration = round(
                    (target_job.completed_at - target_job.started_at).total_seconds()
                    / 60,
                    2,
                )
            tool_counts = {}
            if target_job.execution_log:
                for tool, _error, _entry in _tool_calls(target_job.execution_log):
                    tool_counts[tool] = tool_counts.get(tool, 0) + 1

            return {
                "success": True,
                "data": {
                    "id": str(target_job.id),
                    "goal": (target_job.goal or "")[:200],
                    "status": target_job.status,
                    "job_type": target_job.job_type,
                    "iterations": target_job.iteration,
                    "tool_calls_used": target_job.tool_calls_used,
                    "llm_calls_used": target_job.llm_calls_used,
                    "tokens_used": target_job.tokens_used,
                    "max_tool_calls": target_job.max_tool_calls,
                    "max_llm_calls": target_job.max_llm_calls,
                    "max_runtime_minutes": target_job.max_runtime_minutes,
                    "duration_minutes": duration,
                    "error_count": target_job.error_count,
                    "tool_usage_breakdown": tool_counts,
                    "created_at": target_job.created_at.isoformat()
                    if target_job.created_at
                    else None,
                    "started_at": target_job.started_at.isoformat()
                    if target_job.started_at
                    else None,
                    "completed_at": target_job.completed_at.isoformat()
                    if target_job.completed_at
                    else None,
                },
            }
        except ValueError:
            return {"error": "Invalid job_id format"}
        except Exception as exc:
            return {"error": f"Failed to get job metrics: {exc}"}

    async def _get_tool_usage_stats(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime, timedelta, timezone

        from app.models.agent_job import AgentJob as AgentJobModel

        job = ctx.job
        try:
            days = max(1, min(int(params.get("days", 7) or 7), 30))
            tool_name_filter = str(params.get("tool_name", "")).strip() or None
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            stmt = select(AgentJobModel).where(
                AgentJobModel.user_id == job.user_id,
                AgentJobModel.created_at >= cutoff,
                AgentJobModel.execution_log.isnot(None),
            )
            analyzed_jobs = (await ctx.db.execute(stmt)).scalars().all()

            tool_stats = {}
            for row in analyzed_jobs:
                for tool, error, _entry in _tool_calls(row.execution_log):
                    if tool_name_filter and tool != tool_name_filter:
                        continue
                    stats = tool_stats.setdefault(
                        tool, {"calls": 0, "successes": 0, "failures": 0}
                    )
                    stats["calls"] += 1
                    if error:
                        stats["failures"] += 1
                    else:
                        stats["successes"] += 1
            sorted_tools = sorted(
                tool_stats.items(), key=lambda x: x[1]["calls"], reverse=True
            )
            return {
                "success": True,
                "data": {
                    "period_days": days,
                    "total_jobs_analyzed": len(analyzed_jobs),
                    "tools": [
                        {
                            "name": name,
                            "calls": stats["calls"],
                            "successes": stats["successes"],
                            "failures": stats["failures"],
                            "success_rate": round(
                                stats["successes"] / stats["calls"], 3
                            )
                            if stats["calls"] > 0
                            else 0.0,
                        }
                        for name, stats in sorted_tools[:50]
                    ],
                },
            }
        except Exception as exc:
            return {"error": f"Failed to get tool usage stats: {exc}"}

    async def _get_tool_failure_analysis(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime, timedelta, timezone

        from app.models.agent_job import AgentJob as AgentJobModel

        job = ctx.job
        analysis_tool_name = str(params.get("tool_name", "")).strip()
        if not analysis_tool_name:
            return {"error": "tool_name is required"}
        try:
            days = max(1, min(int(params.get("days", 7) or 7), 30))
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            stmt = (
                select(AgentJobModel)
                .where(
                    AgentJobModel.user_id == job.user_id,
                    AgentJobModel.created_at >= cutoff,
                    AgentJobModel.execution_log.isnot(None),
                )
                .order_by(AgentJobModel.created_at.asc())
            )
            analyzed_jobs = (await ctx.db.execute(stmt)).scalars().all()
            total_calls = 0
            errors = []
            for row in analyzed_jobs:
                for tool, error, entry in _tool_calls(row.execution_log):
                    if tool != analysis_tool_name:
                        continue
                    total_calls += 1
                    if error:
                        errors.append(
                            {
                                "job_id": str(row.id),
                                "job_type": row.job_type,
                                "error": error[:200],
                                "timestamp": entry.get("timestamp"),
                                "iteration": entry.get("iteration"),
                            }
                        )
            error_patterns = {}
            for err in errors:
                key = err["error"][:80]
                pattern = error_patterns.setdefault(key, {"count": 0, "examples": []})
                pattern["count"] += 1
                if len(pattern["examples"]) < 3:
                    pattern["examples"].append(err)
            sorted_patterns = sorted(
                error_patterns.items(), key=lambda x: x[1]["count"], reverse=True
            )
            return {
                "success": True,
                "data": {
                    "tool_name": analysis_tool_name,
                    "period_days": days,
                    "total_calls": total_calls,
                    "total_failures": len(errors),
                    "failure_rate": round(len(errors) / total_calls, 3)
                    if total_calls > 0
                    else 0.0,
                    "error_patterns": [
                        {
                            "pattern": pat,
                            "count": info["count"],
                            "examples": info["examples"],
                        }
                        for pat, info in sorted_patterns[:10]
                    ],
                    "recent_failures": errors[-10:],
                },
            }
        except Exception as exc:
            return {"error": f"Failed to analyze tool failures: {exc}"}

    async def _batch_search(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        queries_raw = params.get("queries")
        if not queries_raw or not isinstance(queries_raw, list) or not queries_raw:
            return {"error": "queries is required and must be a non-empty array"}
        try:
            queries = [str(q).strip() for q in queries_raw if str(q).strip()][:10]
            if not queries:
                return {"error": "No valid queries provided"}
            limit_per = max(1, min(int(params.get("limit_per_query", 5) or 5), 20))
            source_id_filter = str(params.get("source_id", "")).strip() or None
            dedup = params.get("deduplicate", True)
            if dedup is None:
                dedup = True
            all_results = []
            seen_ids = set()
            findings = []
            for query in queries:
                try:
                    results_list, total, _took = await executor.search_service.search(
                        query=query,
                        mode="smart",
                        page=1,
                        page_size=limit_per,
                        source_id=source_id_filter,
                        db=ctx.db,
                    )
                    query_results = []
                    for row in results_list:
                        doc_id = row.get("id")
                        if dedup and doc_id in seen_ids:
                            continue
                        if doc_id:
                            seen_ids.add(doc_id)
                        query_results.append(row)
                    all_results.append(
                        {"query": query, "results": query_results, "total": total}
                    )
                except Exception as exc:
                    all_results.append(
                        {
                            "query": query,
                            "results": [],
                            "total": 0,
                            "error": str(exc)[:200],
                        }
                    )
            for qr in all_results:
                for row in qr.get("results", [])[:5]:
                    findings.append(
                        {
                            "type": "document",
                            "title": row.get("title"),
                            "id": row.get("id"),
                            "score": row.get("relevance_score", row.get("score")),
                            "query": qr.get("query"),
                        }
                    )
            return {
                "success": True,
                "data": {
                    "queries_executed": len(queries),
                    "results": all_results,
                    "total_unique_documents": len(seen_ids),
                },
                "findings": findings,
            }
        except Exception as exc:
            return {"error": f"Failed to execute batch search: {exc}"}

    async def _batch_summarize(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID as _UUID

        from app.models.document import Document

        job = ctx.job
        doc_ids_raw = params.get("document_ids")
        if not doc_ids_raw or not isinstance(doc_ids_raw, list) or not doc_ids_raw:
            return {"error": "document_ids is required and must be a non-empty array"}
        try:
            doc_ids = [str(d).strip() for d in doc_ids_raw if str(d).strip()][:20]
            generate_missing = bool(params.get("generate_missing", False))
            summaries = []
            for doc_id_str in doc_ids:
                try:
                    doc = await ctx.db.get(Document, _UUID(doc_id_str))
                    if not doc:
                        summaries.append(
                            {"document_id": doc_id_str, "status": "not_found"}
                        )
                        continue
                    if doc.summary:
                        summaries.append(
                            {
                                "document_id": doc_id_str,
                                "title": doc.title,
                                "summary": doc.summary,
                                "status": "available",
                            }
                        )
                    elif generate_missing:
                        try:
                            from app.services.llm_service import load_user_llm_settings

                            summary_text = (
                                await executor.document_service.summarize_document(
                                    doc.id,
                                    ctx.db,
                                    user_settings=await load_user_llm_settings(
                                        ctx.db, job.user_id
                                    ),
                                )
                            )
                            if not summary_text:
                                raise ValueError("the summariser returned nothing")
                            summaries.append(
                                {
                                    "document_id": doc_id_str,
                                    "title": doc.title,
                                    "summary": summary_text,
                                    "status": "generated",
                                }
                            )
                        except Exception as exc:
                            summaries.append(
                                {
                                    "document_id": doc_id_str,
                                    "title": doc.title,
                                    "status": "generation_failed",
                                    "error": str(exc)[:200],
                                }
                            )
                    else:
                        summaries.append(
                            {
                                "document_id": doc_id_str,
                                "title": doc.title,
                                "status": "no_summary",
                            }
                        )
                except Exception:
                    summaries.append({"document_id": doc_id_str, "status": "error"})
            available = sum(
                1 for s in summaries if s.get("status") in ("available", "generated")
            )
            return {
                "success": True,
                "data": {
                    "summaries": summaries,
                    "total_requested": len(doc_ids),
                    "available": available,
                    "missing": len(doc_ids) - available,
                },
            }
        except Exception as exc:
            return {"error": f"Failed to execute batch summarize: {exc}"}

    async def _evaluate_condition(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        try:
            condition = str(params.get("condition", "")).strip()
            # 0 is a threshold; `or 1` turned it into 1.
            raw_threshold = params.get("threshold")
            threshold = 1 if raw_threshold is None else int(raw_threshold)
            data: Dict[str, Any]
            if condition == "findings_count":
                count = len(state.get("findings", []))
                data = {
                    "met": count >= threshold,
                    "actual": count,
                    "threshold": threshold,
                    "condition": condition,
                }
            elif condition == "findings_has_category":
                cat = str(params.get("category") or "").strip()
                if not cat:
                    return {
                        "error": "category parameter required for "
                        "findings_has_category condition"
                    }
                matches = [
                    f
                    for f in state.get("findings", [])
                    if isinstance(f, dict) and f.get("category") == cat
                ]
                data = {
                    "met": len(matches) >= threshold,
                    "actual": len(matches),
                    "category": cat,
                    "threshold": threshold,
                    "condition": condition,
                }
            elif condition == "documents_count":
                # The search reports as its total only what it fetched, so
                # the page must be at least as large as the threshold: with a
                # page of one the total was at most two, and a threshold of
                # three or more could never be met. `actual` is therefore a
                # floor once it reaches the threshold, not an exact count.
                source_id = str(params.get("source_id", "")).strip() or None
                _, total, _ = await executor.search_service.search(
                    query="*",
                    mode="smart",
                    page=1,
                    page_size=max(1, min(threshold, 100)),
                    source_id=source_id,
                    db=ctx.db,
                )
                data = {
                    "met": total >= threshold,
                    "actual": total,
                    "threshold": threshold,
                    "condition": condition,
                }
            elif condition == "search_has_results":
                query = str(params.get("query", "")).strip()
                source_id = str(params.get("source_id", "")).strip() or None
                if not query:
                    return {
                        "error": "query parameter required for search_has_results condition"
                    }
                # The search reports as its total only what it fetched, so
                # the page has to be at least as large as the threshold.
                _, total, _ = await executor.search_service.search(
                    query=query,
                    mode="smart",
                    page=1,
                    page_size=max(1, min(threshold, 100)),
                    source_id=source_id,
                    db=ctx.db,
                )
                data = {
                    "met": total >= threshold,
                    "actual": total,
                    "query": query,
                    "threshold": threshold,
                    "condition": condition,
                }
            elif condition == "actions_count":
                count = len(state.get("actions_taken", []))
                data = {
                    "met": count >= threshold,
                    "actual": count,
                    "threshold": threshold,
                    "condition": condition,
                }
            elif condition == "progress_above":
                progress = state.get("goal_progress", 0)
                data = {
                    "met": progress >= threshold,
                    "actual": progress,
                    "threshold": threshold,
                    "condition": condition,
                }
            else:
                return {
                    "error": f"Unknown condition: {condition}. Valid: findings_count, findings_has_category, documents_count, search_has_results, actions_count, progress_above"
                }
            return {"success": True, "data": data}
        except Exception as exc:
            return {"error": f"Failed to evaluate condition: {exc}"}

    async def _count_findings(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        try:
            findings = state.get("findings", [])
            min_conf = float(params.get("min_confidence", 0.0) or 0.0)
            cat_filter = str(params.get("category", "")).strip() or None
            filtered = [
                f
                for f in findings
                if (0.8 if f.get("confidence") is None else float(f.get("confidence")))
                >= min_conf
            ]
            if cat_filter:
                filtered = [f for f in filtered if f.get("category") == cat_filter]
            by_category: dict[str, int] = {}
            for finding in filtered:
                category = str(finding.get("category") or "uncategorized")
                by_category[category] = by_category.get(category, 0) + 1
            return {
                "success": True,
                "data": {
                    "total": len(filtered),
                    "by_category": by_category,
                    "categories": list(by_category.keys()),
                },
            }
        except Exception as exc:
            return {"error": f"Failed to count findings: {exc}"}

    def _contract_status(state: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize the executor's most recent goal-contract evaluation.

        Reuses `goal_contract_last` rather than re-evaluating: the executor
        writes it every iteration, and a second evaluation here could disagree
        with the one that actually gates completion.
        """
        from app.services import agent_measurement_validity

        last = state.get("goal_contract_last")
        if not isinstance(last, dict) or not last.get("enabled"):
            return {"goal_contract_enabled": False}

        missing = last.get("missing") if isinstance(last.get("missing"), list) else []
        validity = (last.get("metrics") or {}).get("validity") or {}
        status: Dict[str, Any] = {
            "goal_contract_enabled": True,
            "goal_contract_satisfied": bool(last.get("satisfied")),
            "goal_contract_missing": [str(x) for x in missing][:10],
        }
        remedies = agent_measurement_validity.explain(
            missing, validity.get("details") or {}
        )
        if remedies:
            status["goal_contract_remedies"] = remedies[:4]
        return status

    async def _request_stage_rerun(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        """Ask for an earlier stage to be redone on a stated correction.

        Records the request rather than acting on it. The stage has to finish
        the iteration it is in -- there is a checkpoint to write and a result
        to return -- and the finaliser is the one place that already decides
        what happens when a stage ends, so putting the decision anywhere else
        would give a run two ways to end and one of them would drift.
        """
        from app.services import agent_stage_rerun

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        results = job.results if isinstance(job.results, dict) else {}

        verdict = agent_stage_rerun.evaluate(
            stage=str(params.get("stage") or ""),
            reason=str(params.get("reason") or ""),
            config=job.config,
            results=results,
        )
        if not verdict.ok:
            return {"error": verdict.error}

        state["stage_rerun_request"] = {
            "stage": verdict.stage,
            "reason": verdict.reason,
            "iteration": int(job.iteration or 0),
        }
        return {
            "success": True,
            "data": {
                "stage": verdict.stage,
                "queued": True,
                "note": (
                    f"This stage will end and {verdict.stage!r} will run again "
                    "with your reason attached. Everything after it is "
                    "re-derived, so do not keep working on the current "
                    "attempt."
                ),
            },
        }

    async def _check_goal_status(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        try:
            exec_plan = state.get("execution_plan")
            # The plan is stored as a list of steps; only a dict was counted,
            # so every real plan reported zero steps.
            if isinstance(exec_plan, list):
                plan_steps_total = len(exec_plan)
            elif isinstance(exec_plan, dict):
                plan_steps_total = len(exec_plan.get("steps", []))
            else:
                plan_steps_total = 0
            return {
                "success": True,
                "data": {
                    "iteration": job.iteration,
                    "max_iterations": job.max_iterations,
                    "iterations_remaining": job.max_iterations - job.iteration,
                    "tool_calls_used": job.tool_calls_used,
                    "max_tool_calls": job.max_tool_calls,
                    "tool_calls_remaining": job.max_tool_calls - job.tool_calls_used,
                    "goal_progress": state.get("goal_progress", 0),
                    "findings_count": len(state.get("findings", [])),
                    "actions_count": len(state.get("actions_taken", [])),
                    "has_execution_plan": bool(exec_plan),
                    "plan_steps_completed": state.get("plan_step_index", 0),
                    "plan_steps_total": plan_steps_total,
                    # What the run still has to satisfy, from the executor's
                    # own last evaluation. Without this the contract is only
                    # discoverable by trying to finish and being refused,
                    # which wastes the iteration that discovers it.
                    **_contract_status(state),
                },
            }
        except Exception as exc:
            return {"error": f"Failed to check goal status: {exc}"}

    async def _compress_history(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        try:
            actions = state.get("actions_taken", [])
            raw_keep = params.get("keep_last")
            keep_last = max(0, min(5 if raw_keep is None else int(raw_keep), 20))
            if len(actions) <= keep_last:
                return {
                    "success": True,
                    "data": {
                        "message": "Not enough history to compress",
                        "actions_count": len(actions),
                    },
                }
            to_compress = actions[:-keep_last] if keep_last > 0 else list(actions)
            actions_text = ""
            for action in to_compress:
                tool = (
                    action.get("action", {}).get("tool", "unknown")
                    if isinstance(action.get("action"), dict)
                    else "unknown"
                )
                res_summary = ""
                act_result = action.get("result", {})
                if isinstance(act_result, dict):
                    if act_result.get("success"):
                        data_keys = (
                            list(act_result.get("data", {}).keys())
                            if isinstance(act_result.get("data"), dict)
                            else []
                        )
                        res_summary = f"success, data keys: {data_keys}"
                    else:
                        res_summary = (
                            f"failed: {str(act_result.get('error', ''))[:100]}"
                        )
                actions_text += f"- Iteration {action.get('iteration', '?')}: {tool} → {res_summary}\n"
            existing_compressed = state.get("compressed_history", "")
            compress_prompt = (
                "Summarize the following agent action history into a concise narrative (max 500 words).\n"
                "Focus on: what was discovered, what worked/failed, key decisions made, and current trajectory.\n\n"
            )
            if existing_compressed:
                compress_prompt += (
                    f"Previous compressed history:\n{existing_compressed}\n\n"
                )
            compress_prompt += f"New actions to compress:\n{actions_text}\n\nWrite a concise summary in past tense."
            # Loaded for the job's owner. The executor has no such attribute
            # (only its runtime adapter does), so reading it there always
            # gave None and the owner's provider and model were ignored.
            from app.services.llm_service import load_user_llm_settings

            user_settings = await load_user_llm_settings(
                ctx.db, getattr(ctx.job, "user_id", None)
            )
            summary_resp = await executor.llm_service.generate_response(
                system_prompt="You are a concise summarizer. Output only the summary, no preamble.",
                user_message=compress_prompt,
                user_settings=user_settings,
                db=ctx.db,
                snapshot_context=_tool_snapshot_context(ctx, "compress_history"),
            )
            summary_text = str(summary_resp or "").strip()[:2000]
            if not summary_text:
                # Nothing is dropped for a summary that is not there: storing
                # it erased the earlier summary and the actions together.
                return {
                    "error": "The model returned no summary; history was left "
                    "as it was"
                }
            state["compressed_history"] = summary_text
            state["actions_taken"] = actions[-keep_last:] if keep_last > 0 else []
            return {
                "success": True,
                "data": {
                    "compressed_actions": len(to_compress),
                    "kept_actions": len(state["actions_taken"]),
                    "summary_length": len(summary_text),
                },
            }
        except Exception as exc:
            return {"error": f"Failed to compress history: {exc}"}

    async def _summarize_findings(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import uuid
        from datetime import datetime

        state = ctx.state if isinstance(ctx.state, dict) else {}
        job = ctx.job
        try:
            findings = state.get("findings", [])
            cat_filter = str(params.get("category", "")).strip() or None
            consolidate = bool(params.get("consolidate", False))
            target = (
                [f for f in findings if f.get("category") == cat_filter]
                if cat_filter
                else list(findings)
            )
            if not target:
                return {
                    "success": True,
                    "data": {"message": "No findings to summarize", "count": 0},
                }
            findings_text = ""
            for finding in target:
                findings_text += f"- [{finding.get('category', 'general')}] {finding.get('title', 'Untitled')}: {str(finding.get('content', ''))[:300]}\n"
            synth_prompt = (
                f"Synthesize these {len(target)} research findings into a coherent summary (max 800 words).\n"
                "Group related findings, identify themes, note contradictions, and highlight the most important insights.\n\n"
                f"Findings:\n{findings_text}\n\nWrite a structured synthesis."
            )
            # Loaded for the job's owner. The executor has no such attribute
            # (only its runtime adapter does), so reading it there always
            # gave None and the owner's provider and model were ignored.
            from app.services.llm_service import load_user_llm_settings

            user_settings = await load_user_llm_settings(
                ctx.db, getattr(ctx.job, "user_id", None)
            )
            synthesis_resp = await executor.llm_service.generate_response(
                system_prompt="You are a research synthesizer. Output only the synthesis, no preamble.",
                user_message=synth_prompt,
                user_settings=user_settings,
                db=ctx.db,
                snapshot_context=_tool_snapshot_context(ctx, "summarize_findings"),
            )
            synthesis_text = str(synthesis_resp or "").strip()[:3000]
            if not synthesis_text:
                return {
                    "error": "The model returned no synthesis; the findings "
                    "were left as they were"
                }
            out: Dict[str, Any] = {
                "success": True,
                "data": {
                    "synthesis": synthesis_text,
                    "findings_summarized": len(target),
                    "consolidated": consolidate,
                },
            }
            if consolidate:
                # Only narrative findings are folded into the synthesis. A
                # finding with a `type` is evidence: goal contracts count
                # them, bounds are checked on them and later tools cite
                # them. Replacing them with one untyped summary let a run
                # un-satisfy a contract it had already met.
                folded = [f for f in target if not f.get("type")]
                kept_typed = len(target) - len(folded)
                folded_ids = {id(f) for f in folded}
                state["findings"] = [f for f in findings if id(f) not in folded_ids]
                out["data"]["findings_folded"] = len(folded)
                out["data"]["typed_findings_kept"] = kept_typed
                consolidated = {
                    "id": str(uuid.uuid4()),
                    "title": f"Synthesis: {cat_filter or 'all findings'} ({len(target)} items)",
                    "content": synthesis_text,
                    "category": "synthesis",
                    "confidence": 0.9,
                    "tags": ["synthesized", "compressed"],
                    "created_at": datetime.utcnow().isoformat(),
                }
                # Returned, not appended: the executor adds a result's
                # `findings` to state itself, so appending here as well
                # recorded the synthesis twice.
                executor._job_findings.setdefault(str(job.id), []).append(consolidated)
                out["findings"] = [consolidated]
            return out
        except Exception as exc:
            return {"error": f"Failed to summarize findings: {exc}"}

    return FunctionToolProvider(
        name="autonomous_observability_tools",
        modes={"autonomous"},
        handlers={
            "get_job_history": _get_job_history,
            "recall_prior_findings": _recall_prior_findings,
            "get_job_metrics": _get_job_metrics,
            "get_tool_usage_stats": _get_tool_usage_stats,
            "get_tool_failure_analysis": _get_tool_failure_analysis,
            "batch_search": _batch_search,
            "batch_summarize": _batch_summarize,
            "evaluate_condition": _evaluate_condition,
            "count_findings": _count_findings,
            "check_goal_status": _check_goal_status,
            "request_stage_rerun": _request_stage_rerun,
            "compress_history": _compress_history,
            "summarize_findings": _summarize_findings,
        },
    )
