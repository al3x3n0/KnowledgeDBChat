"""Autonomous-job tools: the ``memory`` provider.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

from typing import Any, Dict

from app.services.agent_tool_providers.base import (
    AgentToolExecutionContext,
    FunctionToolProvider,
)


def build_autonomous_memory_provider(executor: Any) -> FunctionToolProvider:
    """Memory tools for AutonomousAgentExecutor."""

    async def _record_method(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services import agent_method_record

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        findings = (
            state.get("findings") if isinstance(state.get("findings"), list) else []
        )
        available = sorted(
            {
                str(f.get("type")).strip()
                for f in findings
                if isinstance(f, dict) and str(f.get("type") or "").strip()
            }
        )

        try:
            record = agent_method_record.build_record(
                name=params.get("name"),
                procedure=params.get("procedure"),
                prevents=params.get("prevents"),
                derived_from=params.get("derived_from"),
                available_finding_types=available,
                applies_to=params.get("applies_to"),
                limits=str(params.get("limits") or ""),
            )
        except agent_method_record.MethodRecordError as exc:
            return {"error": str(exc)}

        content = agent_method_record.render(record)
        try:
            # Constructed directly rather than through MemoryCreate: that
            # schema's types exclude "pattern", and a method stored under a
            # type the job-memory filter does not inject would be written and
            # never recalled -- the one outcome that makes this tool pointless.
            from app.models.memory import ConversationMemory

            stored = ConversationMemory(
                user_id=job.user_id,
                job_id=getattr(job, "id", None),
                memory_type=agent_method_record.MEMORY_TYPE,
                content=content,
                # A method outranks an observation about one subject: it is
                # what a later job on a different subject can still use.
                importance_score=(
                    0.9 if record["status"] == agent_method_record.VALIDATED else 0.6
                ),
                tags=agent_method_record.tags_for(record),
                context={
                    "record": "method",
                    "status": record["status"],
                    "evidence": record["evidence"],
                },
            )
            ctx.db.add(stored)
            await ctx.db.commit()
            await ctx.db.refresh(stored)
        except Exception as exc:
            return {"error": f"Failed to record the method: {str(exc)[:200]}"}

        # Also kept on the run, which is where the run's own record reads it:
        # the finaliser's "what this run established" document lists methods
        # verbatim from results["methods"], and nothing ever wrote that key,
        # so a run's methods reached later jobs' memory and never its record.
        state.setdefault("recorded_methods", []).append(
            {
                "name": record["name"],
                "procedure": record["procedure"],
                "prevents": record["prevents"],
                "status": record["status"],
                "evidence": record["evidence"],
            }
        )

        return {
            "success": True,
            "data": {
                "memory_id": str(stored.id),
                "name": record["name"],
                "status": record["status"],
                "evidence": record["evidence"],
                "note": (
                    "Stored where later jobs recall it. A method recorded "
                    "unvalidated stays that way until a run demonstrates it."
                ),
            },
            "findings": [
                {
                    "type": "method_recorded",
                    "title": f"Method: {record['name']} ({record['status']})",
                    "content": content,
                    "category": "insight",
                }
            ],
        }

    async def _create_memory(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.schemas.memory import MemoryCreate

        job = ctx.job
        content_str = str(params.get("content", "")).strip()
        if not content_str:
            return {"error": "content is required"}

        raw_importance = params.get("importance")
        try:
            # 0.0 is a value, not an absence: `or 0.5` stored it as 0.5.
            importance = 0.5 if raw_importance is None else float(raw_importance)
        except (TypeError, ValueError):
            return {"error": "importance must be a number between 0.0 and 1.0"}
        importance = max(0.0, min(1.0, importance))
        category = str(params.get("category", "fact") or "fact")
        metadata = (
            params.get("metadata") if isinstance(params.get("metadata"), dict) else None
        )
        tags = []
        if metadata and isinstance(metadata.get("tags"), list):
            metadata = dict(metadata)
            tags = metadata.pop("tags")
        try:
            memory_data = MemoryCreate(
                memory_type=category,
                content=content_str,
                importance_score=importance,
                context=metadata,
                tags=tags or None,
            )
            mem_resp = await executor.memory_service.create_memory(
                job.user_id, memory_data, ctx.db
            )
            # Say which run remembered it; the column exists for that.
            try:
                from app.models.memory import ConversationMemory

                row = await ctx.db.get(ConversationMemory, mem_resp.id)
                if row is not None and row.job_id is None and job.id is not None:
                    row.job_id = job.id
                    await ctx.db.commit()
            except Exception:
                await ctx.db.rollback()
            return {
                "success": True,
                "data": {"memory_id": str(mem_resp.id), "content": content_str[:200]},
            }
        except Exception as exc:
            return {"error": f"Failed to create memory: {exc}"}

    async def _search_memories(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.schemas.memory import MemorySearchRequest

        job = ctx.job
        query_str = str(params.get("query", "")).strip()
        if not query_str:
            return {"error": "query is required"}

        try:
            limit = max(1, min(int(params.get("limit", 10) or 10), 50))
        except (TypeError, ValueError):
            limit = 10
        cat_filter = params.get("category_filter")
        min_imp = params.get("min_importance")
        memory_types = [cat_filter] if cat_filter else None
        try:
            search_req = MemorySearchRequest(
                query=query_str,
                limit=limit,
                memory_types=memory_types,
                min_importance=float(min_imp) if min_imp is not None else None,
            )
            memories = await executor.memory_service.search_memories(
                job.user_id, search_req, ctx.db
            )
            from app.services.memory_service import lexical_relevance

            return {
                "success": True,
                "data": {
                    "memories": [
                        {
                            "id": str(m.id),
                            "content": m.content,
                            "importance": m.importance_score,
                            "type": m.memory_type,
                            # How much of the query this memory shares, by
                            # words. The ranking is lexical, so a memory
                            # scoring 0.0 was returned for its importance,
                            # not because it matched.
                            "relevance": round(
                                lexical_relevance(query_str, m.content or ""), 2
                            ),
                        }
                        for m in memories
                    ],
                    "count": len(memories),
                },
            }
        except Exception as exc:
            return {"error": f"Memory search failed: {exc}"}

    async def _recall_memories(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.schemas.memory import MemorySearchRequest

        job = ctx.job
        topic = str(params.get("topic", "")).strip()
        if not topic:
            return {"error": "topic is required"}

        try:
            limit = max(1, min(int(params.get("limit", 10) or 10), 50))
        except (TypeError, ValueError):
            limit = 10
        try:
            search_req = MemorySearchRequest(query=topic, limit=limit)
            memories = await executor.memory_service.search_memories(
                job.user_id, search_req, ctx.db
            )
            return {
                "success": True,
                "data": {
                    "memories": [
                        {
                            "id": str(m.id),
                            "content": m.content,
                            "importance": m.importance_score,
                            "type": m.memory_type,
                        }
                        for m in memories
                    ],
                    "count": len(memories),
                },
            }
        except Exception as exc:
            return {"error": f"Memory recall failed: {exc}"}

    async def _get_memory_stats(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        job = ctx.job
        try:
            stats = await executor.memory_service.get_memory_stats(job.user_id, ctx.db)
            return {
                "success": True,
                "data": {
                    "total_memories": stats.total_memories,
                    "memories_by_type": stats.memories_by_type,
                    "recent_memories": stats.recent_memories,
                    "most_accessed_memories": [
                        {
                            "id": str(m.id),
                            "content": (m.content or "")[:200],
                            "type": m.memory_type,
                            "access_count": m.access_count,
                        }
                        for m in (stats.most_accessed_memories or [])[:10]
                    ],
                },
            }
        except Exception as exc:
            return {"error": f"Failed to get memory stats: {exc}"}

    return FunctionToolProvider(
        name="autonomous_memory_tools",
        modes={"autonomous"},
        handlers={
            "create_memory": _create_memory,
            "record_method": _record_method,
            "search_memories": _search_memories,
            "recall_memories": _recall_memories,
            "get_memory_stats": _get_memory_stats,
        },
    )
