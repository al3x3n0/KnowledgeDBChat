"""Autonomous-job tools: the ``workspace_read`` provider.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

from typing import Any, Dict

from app.services.agent_tool_providers.base import (
    AgentToolExecutionContext,
    FunctionToolProvider,
)


def build_autonomous_workspace_read_provider(executor: Any) -> FunctionToolProvider:
    """Read-oriented workspace tools for AutonomousAgentExecutor."""

    async def _document_source_exists(db: Any, source_id: str) -> bool:
        from uuid import UUID

        from sqlalchemy import select

        from app.models.document import DocumentSource

        try:
            key = UUID(str(source_id))
        except (TypeError, ValueError):
            return False
        row = await db.execute(
            select(DocumentSource.id).where(DocumentSource.id == key)
        )
        return row.scalar_one_or_none() is not None

    async def _clone_and_index_repo(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        source_id = str(params.get("source_id") or "").strip()
        repo_url = str(params.get("repo_url") or "").strip()
        branch = str(params.get("branch") or "").strip() or None
        job = ctx.job
        state = ctx.state if isinstance(ctx.state, dict) else {}
        if not source_id and not repo_url:
            source_id = str((job.config or {}).get("source_id") or "").strip()
        if not source_id and not repo_url:
            return {"error": "Either source_id or repo_url is required"}
        fallback_note = ""
        if source_id:
            # A source id that names nothing used to load zero documents and
            # report success with files_count 0. Measured: a run's critic
            # suggested a placeholder UUID, the clone "succeeded" empty, the
            # repo_url passed in the SAME call was ignored, and three
            # iterations went to confirming the workspace was empty.
            if not await _document_source_exists(ctx.db, source_id):
                if not repo_url:
                    return {
                        "error": (
                            f"source_id {source_id!r} names no document source in "
                            "the knowledge base. To clone from git, pass repo_url "
                            "(and branch) and leave source_id out."
                        )
                    }
                fallback_note = (
                    f"source_id {source_id!r} names no document source; cloned "
                    "repo_url instead. Leave source_id out when giving a URL."
                )
                source_id = ""
        try:
            if source_id:
                ws = await executor.workspace_manager.create_from_source(
                    source_id, ctx.db
                )
            else:
                from app.core.config import settings as app_settings
                from app.core.feature_flags import get_flag

                enabled = await get_flag("unsafe_code_execution_enabled")
                if enabled is None:
                    enabled = bool(
                        getattr(app_settings, "ENABLE_UNSAFE_CODE_EXECUTION", False)
                    )
                if not enabled:
                    return {"error": "Git clone requires unsafe_code_execution_enabled"}
                ws = await executor.workspace_manager.create_from_url(repo_url, branch)
            if not ws.original_hashes:
                # An empty workspace is never a success: every later tool
                # would report "0 files" as if that were the repository.
                executor.workspace_manager.cleanup(ws.workspace_id)
                return {
                    "error": (
                        "the workspace came out EMPTY -- "
                        + (
                            f"source {source_id} has no documents"
                            if source_id
                            else f"cloning {repo_url} produced no files"
                        )
                        + ". Nothing was indexed; do not proceed as if it were."
                    )
                }
            state["coding_workspace_id"] = ws.workspace_id
            ws.owner_job_id = str(job.id)
            ws.session_id = (
                str((job.config or {}).get("coding_workspace_session_id") or "").strip()
                or None
            )
            # Register it now, while the provenance is in hand. Both branches
            # above converge here, so a workspace cannot be created by this
            # tool without being findable -- which is the whole point: the
            # process that made it is not the process anyone will read it from.
            await executor.workspace_manager.persist_record(
                ws, ctx.db, user_id=job.user_id, job_id=job.id
            )
            from app.services.agent_coding_harness_service import (
                agent_coding_harness_service,
            )

            instruction_context = (
                agent_coding_harness_service.discover_project_instructions(ws)
            )
            state["coding_harness_context"] = instruction_context
            baseline_checkpoint = None
            if bool((job.config or {}).get("coding_harness_may_mutate")):
                (
                    baseline_checkpoint,
                    checkpoint_error,
                ) = executor.workspace_manager.create_checkpoint(
                    ws,
                    label="Automatic baseline before mutation",
                    kind="baseline",
                )
                if checkpoint_error:
                    executor.workspace_manager.cleanup(ws.workspace_id)
                    state.pop("coding_workspace_id", None)
                    return {
                        "error": (
                            "Failed to create mandatory pre-mutation checkpoint: "
                            f"{checkpoint_error}"
                        )
                    }
                state["coding_pre_mutation_checkpoint_id"] = str(
                    baseline_checkpoint.get("checkpoint_id") or ""
                )
            restored_durable_checkpoint = None
            durable_checkpoint_id = str(
                state.get("coding_last_durable_checkpoint_id") or ""
            ).strip()
            if durable_checkpoint_id:
                try:
                    from app.services.agent_coding_durable_checkpoint_service import (
                        agent_coding_durable_checkpoint_service,
                    )

                    restored_durable_checkpoint = (
                        await agent_coding_durable_checkpoint_service.restore(
                            executor,
                            job,
                            state,
                            checkpoint_id=durable_checkpoint_id,
                        )
                    )
                except Exception as exc:
                    executor.workspace_manager.cleanup(ws.workspace_id)
                    state.pop("coding_workspace_id", None)
                    return {
                        "error": (
                            "Failed to restore durable coding session checkpoint: "
                            f"{exc}"
                        )
                    }
            return {
                "success": True,
                **({"note": fallback_note} if fallback_note else {}),
                "data": {
                    "workspace_id": ws.workspace_id,
                    "files_count": len(ws.original_hashes),
                    "source": "kb_source" if source_id else "git_clone",
                    "instruction_files": [
                        str(item.get("path") or "")
                        for item in instruction_context.get("files", [])
                        if isinstance(item, dict)
                        and str(item.get("path") or "").strip()
                    ],
                    "baseline_checkpoint": baseline_checkpoint,
                    "restored_durable_checkpoint": restored_durable_checkpoint,
                },
                "findings": [
                    {
                        "type": "repo_workspace",
                        "workspace_id": ws.workspace_id,
                        "files_count": len(ws.original_hashes),
                        "source": "kb_source" if source_id else "git_clone",
                    }
                ],
            }
        except Exception as exc:
            return {"error": f"Failed to create workspace: {exc}"}

    async def _browse_repo_files(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state, job=ctx.job
        )
        if not ws:
            return {
                "error": "No active coding workspace. Use clone_and_index_repo first."
            }
        entries = executor.workspace_manager.browse_files(
            ws,
            path=str(params.get("path", ".") or "."),
            glob_pattern=params.get("glob_pattern"),
            max_results=min(int(params.get("max_results", 200) or 200), 500),
        )
        return {
            "success": True,
            "data": {"files": entries, "count": len(entries)},
            "findings": [
                {
                    "type": "repo_listing",
                    "count": len(entries),
                    "files": entries[:50],
                }
            ],
        }

    async def _read_file(params: Dict[str, Any], ctx: AgentToolExecutionContext) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state, job=ctx.job
        )
        if not ws:
            return {"error": "No active coding workspace"}
        path = str(params.get("path", "")).strip()
        if not path:
            return {"error": "path is required"}
        content, err = executor.workspace_manager.read_file(
            ws,
            path,
            start_line=params.get("start_line"),
            end_line=params.get("end_line"),
            max_chars=min(int(params.get("max_chars", 20000) or 20000), 50000),
        )
        if err:
            return {"error": err}
        return {
            "success": True,
            "data": {"path": path, "content": content, "length": len(content or "")},
        }

    async def _search_code(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state, job=ctx.job
        )
        if not ws:
            return {"error": "No active coding workspace"}
        pattern = str(params.get("pattern", "")).strip()
        if not pattern:
            return {"error": "pattern is required"}
        matches = executor.workspace_manager.search_code(
            ws,
            pattern,
            path=str(params.get("path", ".") or "."),
            file_glob=params.get("file_glob"),
            max_results=min(int(params.get("max_results", 50) or 50), 200),
            context_lines=min(int(params.get("context_lines", 2) or 2), 10),
        )
        return {
            "success": True,
            "data": {"matches": matches, "count": len(matches)},
            "findings": [
                {
                    "type": "code_search_result",
                    "count": len(matches),
                    "matches": matches[:25],
                }
            ],
        }

    async def _get_workspace_status(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state, job=ctx.job
        )
        if not ws:
            return {"error": "No active coding workspace"}
        status = executor.workspace_manager.get_status(ws)
        return {"success": True, "data": status}

    async def _list_workspace_checkpoints(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state, job=ctx.job
        )
        if not ws:
            return {"error": "No active coding workspace"}
        checkpoints = executor.workspace_manager.list_checkpoints(ws)
        return {
            "success": True,
            "data": {
                "workspace_id": ws.workspace_id,
                "checkpoints": checkpoints,
                "count": len(checkpoints),
            },
        }

    async def _list_durable_workspace_checkpoints(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        from app.services.agent_coding_durable_checkpoint_service import (
            agent_coding_durable_checkpoint_service,
        )

        checkpoints = agent_coding_durable_checkpoint_service.list_checkpoints(
            ctx.job,
            state,
        )
        rows = [
            {
                key: item.get(key)
                for key in (
                    "checkpoint_id",
                    "session_id",
                    "workspace_state_digest",
                    "changes_summary",
                    "persistence_complete",
                    "label",
                    "reason",
                    "persisted_at",
                )
            }
            for item in checkpoints
        ]
        return {"success": True, "data": {"checkpoints": rows, "count": len(rows)}}

    async def _get_workspace_artifact_url(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        ws_job_id = str(params.get("job_id", "")).strip()
        ws_file_path = str(params.get("file_path", "")).strip()
        if not ws_job_id or not ws_file_path:
            return {"error": "job_id and file_path are required"}
        from app.services.storage_service import storage_service

        full_path = f"workspaces/{ws_job_id}/{ws_file_path}"
        try:
            await storage_service.initialize()
            url = await storage_service.get_presigned_download_url(full_path)
            return {"success": True, "data": {"url": url, "object_path": full_path}}
        except Exception as exc:
            return {"error": f"Failed to get download URL: {exc}"}

    return FunctionToolProvider(
        name="autonomous_workspace_read_tools",
        modes={"autonomous"},
        handlers={
            "clone_and_index_repo": _clone_and_index_repo,
            "browse_repo_files": _browse_repo_files,
            "read_file": _read_file,
            "search_code": _search_code,
            "get_workspace_status": _get_workspace_status,
            "list_workspace_checkpoints": _list_workspace_checkpoints,
            "list_durable_workspace_checkpoints": (_list_durable_workspace_checkpoints),
            "get_workspace_artifact_url": _get_workspace_artifact_url,
        },
    )
