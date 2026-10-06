"""Autonomous-job tools: the ``output_state`` provider.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

from typing import Any, Dict

from sqlalchemy import select

from app.services.agent_tool_providers.base import (
    AgentToolExecutionContext,
    FunctionToolProvider,
)


def build_autonomous_output_state_provider(executor: Any) -> FunctionToolProvider:
    """Output shaping, strategy, and handoff tools for AutonomousAgentExecutor."""

    async def _create_handoff(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.models.agent_job import AgentJob, AgentJobStatus
        from app.services.job_dispatch import enqueue_agent_job

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        try:
            child_goal = str(params.get("goal", "")).strip()
            expected_outputs = params.get("expected_outputs", [])
            if not child_goal:
                return {"error": "goal parameter is required"}
            if not isinstance(expected_outputs, list) or not expected_outputs:
                return {
                    "error": "expected_outputs must be a non-empty array of strings"
                }
            # A pipeline stage with a declared backward edge must use it rather
            # than hand the work to a job outside the pipeline. Measured: a
            # `mine` stage correctly judged its profile too coarse to mine and
            # created a handoff to re-profile -- work that would have run, and
            # produced evidence no stage could consume, because a handoff job
            # carries no `pipeline_stage` and nothing re-derives the stages
            # after it. The two look equivalent and are not.
            #
            # Steered rather than forbidden: a handoff to genuinely new work is
            # still the right tool, and only the stages this one may revisit
            # are named.
            revisit_targets = (job.config or {}).get("may_revisit")
            if isinstance(revisit_targets, list) and revisit_targets:
                return {
                    "error": (
                        "This is a pipeline stage and it may send work back to "
                        f"{', '.join(str(t) for t in revisit_targets)} with "
                        "request_stage_rerun. Use that instead of a handoff if "
                        "an earlier stage is what needs redoing: a handoff job "
                        "runs outside the pipeline, so nothing re-derives the "
                        "stages after it and its result reaches no contract. "
                        "If the work is genuinely NEW rather than a redo, say "
                        "so in the goal and it will be clear which you meant."
                    )
                }

            chain_depth = int(getattr(job, "chain_depth", 0) or 0)
            if chain_depth >= 3:
                return {
                    "error": "Maximum chain depth (3) reached — cannot create further handoffs"
                }
            existing_children = state.get("delegated_subtask_ids", [])
            if len(existing_children) >= 5:
                return {"error": "Maximum child jobs (5) reached for this parent"}
            child_type = str(params.get("job_type", "research")).strip()
            if child_type not in {"research", "analysis", "synthesis", "custom"}:
                child_type = "research"
            child_max = max(1, min(int(params.get("max_iterations", 10) or 10), 20))
            share = params.get("share_findings", True)
            if share is None:
                share = True
            child_config: dict = {
                "handoff_contract": {
                    "from_job_id": str(job.id),
                    "from_job_name": job.name or "unknown",
                    "context": str(params.get("context", ""))[:2000],
                    "expected_outputs": [str(o)[:200] for o in expected_outputs[:10]],
                },
            }
            if share:
                # Under the key the child's prompt reads. These went to
                # `inherited_findings`, which nothing reads, so a child told
                # "findings shared" started blind.
                child_config.setdefault("inherited_data", {})["parent_findings"] = (
                    state.get("findings") or []
                )[-20:]
            source_scope_id = executor._resolve_default_source_scope(job)
            if source_scope_id:
                child_config["default_source_id"] = source_scope_id
            child_name = f"Handoff from {job.name or 'parent'}: {child_goal[:80]}"
            child = AgentJob(
                name=child_name[:200],
                description=f"Structured handoff from {job.name}: {child_goal[:500]}",
                job_type=child_type,
                goal=child_goal[:2000],
                config=child_config,
                status=AgentJobStatus.PENDING.value,
                user_id=job.user_id,
                parent_job_id=job.id,
                chain_depth=chain_depth + 1,
                root_job_id=getattr(job, "root_job_id", None) or job.id,
                max_iterations=child_max,
                max_tool_calls=min(child_max * 5, job.max_tool_calls or 500),
                max_llm_calls=min(child_max * 3, job.max_llm_calls or 200),
                max_runtime_minutes=min(30, job.max_runtime_minutes or 60),
                results={},
            )
            async with ctx.db.begin_nested():
                ctx.db.add(child)
                await ctx.db.flush()
            state.setdefault("delegated_subtask_ids", []).append(str(child.id))
            # Committed before it is queued: a worker in another process
            # cannot see a row that has only been flushed, and one that
            # picked the task up first found no job to run.
            await ctx.db.commit()
            enqueue_agent_job(ctx.db, str(child.id), str(job.user_id))
            return {
                "success": True,
                "data": {
                    "child_job_id": str(child.id),
                    "child_name": child_name[:200],
                    "job_type": child_type,
                    "expected_outputs": [str(o)[:200] for o in expected_outputs[:10]],
                    "max_iterations": child_max,
                    "findings_shared": bool(share),
                },
            }
        except Exception as exc:
            return {"error": f"Failed to create handoff: {exc}"}

    async def _get_sibling_status(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.models.agent_job import AgentJob

        job = ctx.job
        try:
            if not job.parent_job_id:
                return {"error": "This job has no parent — no siblings exist"}
            include_findings = bool(params.get("include_findings", False))
            siblings_q = await ctx.db.execute(
                select(AgentJob).where(
                    AgentJob.parent_job_id == job.parent_job_id,
                    AgentJob.id != job.id,
                    AgentJob.user_id == job.user_id,
                )
            )
            sibling_data = []
            for sibling in siblings_q.scalars().all():
                entry: dict = {
                    "job_id": str(sibling.id),
                    "name": sibling.name,
                    "job_type": sibling.job_type,
                    "status": sibling.status,
                    "iteration": sibling.iteration,
                    "max_iterations": sibling.max_iterations,
                }
                if include_findings and isinstance(sibling.results, dict):
                    findings = sibling.results.get("findings", [])
                    if isinstance(findings, list):
                        entry["findings_count"] = len(findings)
                        entry["finding_titles"] = [
                            str(f.get("title", ""))[:100]
                            for f in findings[:10]
                            if isinstance(f, dict)
                        ]
                sibling_data.append(entry)
            return {
                "success": True,
                "data": {"siblings": sibling_data, "count": len(sibling_data)},
            }
        except Exception as exc:
            return {"error": f"Failed to get sibling status: {exc}"}

    async def _broadcast_to_siblings(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime

        from sqlalchemy.orm.attributes import flag_modified

        from app.models.agent_job import AgentJob

        job = ctx.job
        try:
            message = str(params.get("message", "")).strip()
            if not message:
                return {"error": "message parameter is required"}
            if not job.parent_job_id:
                return {"error": "This job has no parent — no siblings to broadcast to"}
            category = str(params.get("category", "broadcast")).strip()[:100]
            msg_entry = {
                "from_job_id": str(job.id),
                "from_job_name": job.name or "unknown",
                "message": message[:2000],
                "category": category,
                "sent_at": datetime.utcnow().isoformat(),
                "broadcast": True,
            }
            siblings_q = await ctx.db.execute(
                select(AgentJob).where(
                    AgentJob.parent_job_id == job.parent_job_id,
                    AgentJob.id != job.id,
                    AgentJob.user_id == job.user_id,
                )
            )
            delivered = 0
            for sibling in siblings_q.scalars().all():
                s_results = sibling.results if isinstance(sibling.results, dict) else {}
                agent_msgs = s_results.get("agent_messages", [])
                if not isinstance(agent_msgs, list):
                    agent_msgs = []
                agent_msgs.append(msg_entry)
                s_results["agent_messages"] = agent_msgs[-100:]
                sibling.results = s_results
                flag_modified(sibling, "results")
                delivered += 1
            await ctx.db.flush()
            return {
                "success": True,
                "data": {"recipients": delivered, "message_length": len(message)},
            }
        except Exception as exc:
            return {"error": f"Failed to broadcast to siblings: {exc}"}

    async def _switch_strategy(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        try:
            valid_roles = {
                "researcher",
                "critic",
                "synthesizer",
                "verifier",
                "coder",
                "author",
            }
            role = str(params.get("role", "")).strip().lower()
            if role not in valid_roles:
                return {
                    "error": f"Invalid role: {role}. Valid: {', '.join(sorted(valid_roles))}"
                }
            old_role = (
                state.get("skill_profile", {}).get("role", "unknown")
                if isinstance(state.get("skill_profile"), dict)
                else "unknown"
            )
            new_profile = executor._resolve_agent_skill_profile(
                job, state=state, override_role=role
            )
            state["skill_profile"] = new_profile
            state.setdefault("strategy_switches", []).append(
                {
                    "from": old_role,
                    "to": role,
                    "reason": str(params.get("reason", ""))[:500],
                    "iteration": job.iteration,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            return {
                "success": True,
                "data": {
                    "previous_role": old_role,
                    "new_role": role,
                    "display_name": new_profile.get("display_name", role),
                    "preferred_tools": new_profile.get("preferred_tools", [])[:5],
                },
            }
        except Exception as exc:
            return {"error": f"Failed to switch strategy: {exc}"}

    async def _set_focus_directive(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        try:
            directive = str(params.get("directive") or "").strip()[:1000]
            if not directive:
                return {"error": "directive parameter is required"}
            append = bool(params.get("append", False))
            if append:
                existing = str(state.get("focus_directive") or "")
                combined = (existing + "\n" + directive).strip()
                if len(combined) > 2000:
                    # Cutting the tail dropped the new text and still
                    # answered "appended".
                    return {
                        "error": "The focus directive is full (2000 characters). "
                        "Replace it instead of appending."
                    }
                state["focus_directive"] = combined
            else:
                state["focus_directive"] = directive
            return {
                "success": True,
                "data": {
                    "directive": state["focus_directive"],
                    "mode": "appended" if append else "replaced",
                },
            }
        except Exception as exc:
            return {"error": f"Failed to set focus directive: {exc}"}

    async def _get_available_strategies(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        try:
            strategies = []
            for role_name in [
                "researcher",
                "critic",
                "synthesizer",
                "verifier",
                "coder",
                "author",
            ]:
                profile = executor._resolve_agent_skill_profile(
                    job, state=state, override_role=role_name
                )
                strategies.append(
                    {
                        "role": role_name,
                        "display_name": profile.get("display_name", role_name),
                        "guidance": "; ".join(profile.get("prompt_directives", []))[
                            :300
                        ],
                        "preferred_tools": profile.get("preferred_tools", [])[:5],
                        "discouraged_tools": profile.get("discouraged_tools", []),
                    }
                )
            current = (
                state.get("skill_profile", {}).get("role", "researcher")
                if isinstance(state.get("skill_profile"), dict)
                else "researcher"
            )
            return {
                "success": True,
                "data": {"strategies": strategies, "current_role": current},
            }
        except Exception as exc:
            return {"error": f"Failed to get available strategies: {exc}"}

    async def _format_as_table(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        try:
            title = str(params.get("title", "")).strip()
            source = str(params.get("source", "custom")).strip()
            columns = params.get("columns", [])
            rows = params.get("rows", [])
            if not title:
                return {"error": "title parameter is required"}
            if source == "findings":
                findings = state.get("findings", [])
                fields = params.get(
                    "finding_fields", ["title", "category", "confidence"]
                )
                if not isinstance(fields, list):
                    fields = ["title", "category", "confidence"]
                fields = [str(f).strip() for f in fields if str(f).strip()][:10]
                columns = fields
                rows = []
                for finding in findings:
                    if isinstance(finding, dict):
                        rows.append(
                            [str(finding.get(field, ""))[:200] for field in fields]
                        )
            elif (
                not isinstance(columns, list)
                or not columns
                or not isinstance(rows, list)
                or not rows
            ):
                return {"error": "columns and rows are required for custom tables"}
            md = f"## {title}\n\n"
            col_headers = [str(c) for c in columns]
            md += "| " + " | ".join(col_headers) + " |\n"
            md += "| " + " | ".join("---" for _ in col_headers) + " |\n"
            row_count = 0
            for row in rows[:100]:
                cells = [
                    str(c).replace("|", "\\|")[:200]
                    for c in (row if isinstance(row, list) else [])
                ]
                while len(cells) < len(col_headers):
                    cells.append("")
                cells = cells[: len(col_headers)]
                md += "| " + " | ".join(cells) + " |\n"
                row_count += 1
            state.setdefault("formatted_outputs", []).append(
                {
                    "type": "table",
                    "title": title,
                    "markdown": md,
                    "columns": col_headers,
                    "row_count": row_count,
                }
            )
            return {
                "success": True,
                "data": {
                    "markdown": md,
                    "row_count": row_count,
                    "columns": len(col_headers),
                },
                "artifacts": [{"type": "formatted_table", "title": title}],
            }
        except Exception as exc:
            return {"error": f"Failed to format as table: {exc}"}

    async def _format_as_report(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from loguru import logger

        from app.models.document import Document

        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        try:
            title = str(params.get("title", "")).strip()
            if not title:
                return {"error": "title parameter is required"}
            md = f"# {title}\n\n"
            exec_summary = str(params.get("executive_summary", "")).strip()
            if exec_summary:
                md += f"## Executive Summary\n\n{exec_summary[:3000]}\n\n"
            sections = params.get("sections", [])
            if isinstance(sections, list):
                for sec in sections[:20]:
                    if isinstance(sec, dict):
                        heading = str(sec.get("heading", "Section"))[:200]
                        content = str(sec.get("content", ""))[:5000]
                        md += f"## {heading}\n\n{content}\n\n"
            include_findings = params.get("include_findings", True)
            if include_findings is None:
                include_findings = True
            if include_findings:
                findings = state.get("findings", [])
                if findings:
                    md += "## Findings\n\n"
                    for i, finding in enumerate(findings[:30], 1):
                        if not isinstance(finding, dict):
                            continue
                        md += f"### {i}. {finding.get('title', 'Untitled')}\n\n"
                        md += f"{str(finding.get('content', ''))[:1000]}\n\n"
                        meta_parts = []
                        if finding.get("category"):
                            meta_parts.append(f"Category: {finding['category']}")
                        if finding.get("confidence"):
                            meta_parts.append(f"Confidence: {finding['confidence']}")
                        if meta_parts:
                            md += f"*{' | '.join(meta_parts)}*\n\n"
            include_progress = params.get("include_progress", True)
            if include_progress is None:
                include_progress = True
            if include_progress:
                reports = state.get("progress_reports", [])
                if reports:
                    md += "## Progress History\n\n"
                    for report in reports[-5:]:
                        if isinstance(report, dict):
                            md += f"### Iteration {report.get('iteration', '?')}\n\n"
                            if report.get("summary"):
                                md += f"{report['summary']}\n\n"
            md = md[:50000]
            state.setdefault("formatted_outputs", []).append(
                {"type": "report", "title": title, "markdown": md}
            )
            doc_id = None
            if params.get("persist", False):
                try:
                    import hashlib
                    import uuid

                    notes_source = await executor.document_service._get_or_create_agent_notes_source(
                        ctx.db
                    )
                    doc = Document(
                        title=title[:500],
                        content=md,
                        content_hash=hashlib.sha256(md.encode("utf-8")).hexdigest(),
                        file_type="text/markdown",
                        file_size=len(md.encode("utf-8")),
                        source_id=notes_source.id,
                        source_identifier=f"agent_report:{uuid.uuid4().hex}",
                        tags=["autonomous_job", "report"],
                        extra_metadata={
                            "origin": "autonomous_job",
                            "job_id": str(job.id),
                        },
                    )
                    ctx.db.add(doc)
                    await ctx.db.flush()
                    doc_id = str(doc.id)
                    state.setdefault("artifacts", []).append(
                        {"type": "document", "id": doc_id, "title": title[:500]}
                    )
                except Exception as doc_exc:
                    logger.warning(f"Failed to persist report document: {doc_exc}")
            return {
                "success": True,
                "data": {"markdown": md, "length": len(md), "document_id": doc_id},
                "artifacts": [{"type": "formatted_report", "title": title}],
            }
        except Exception as exc:
            return {"error": f"Failed to format as report: {exc}"}

    async def _set_output_schema(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        try:
            schema = params.get("schema")
            if not isinstance(schema, dict) or not schema:
                return {"error": "schema must be a non-empty object"}
            merge = params.get("merge", True)
            if merge is None:
                merge = True
            if merge:
                existing = state.get("output_schema", {})
                if not isinstance(existing, dict):
                    existing = {}
                existing.update(schema)
                state["output_schema"] = existing
            else:
                state["output_schema"] = dict(schema)
            return {
                "success": True,
                "data": {
                    "schema_keys": list(state["output_schema"].keys()),
                    "total_keys": len(state["output_schema"]),
                    "mode": "merged" if merge else "replaced",
                },
            }
        except Exception as exc:
            return {"error": f"Failed to set output schema: {exc}"}

    return FunctionToolProvider(
        name="autonomous_output_state_tools",
        modes={"autonomous"},
        handlers={
            "create_handoff": _create_handoff,
            "get_sibling_status": _get_sibling_status,
            "broadcast_to_siblings": _broadcast_to_siblings,
            "switch_strategy": _switch_strategy,
            "set_focus_directive": _set_focus_directive,
            "get_available_strategies": _get_available_strategies,
            "format_as_table": _format_as_table,
            "format_as_report": _format_as_report,
            "set_output_schema": _set_output_schema,
        },
    )
