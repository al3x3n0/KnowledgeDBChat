"""Autonomous-job tools: the ``symbol_retrieval`` provider.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

from typing import Any, Dict

from app.services.agent_tool_providers.base import (
    AgentToolExecutionContext,
    FunctionToolProvider,
)


def build_autonomous_symbol_retrieval_provider(executor: Any) -> FunctionToolProvider:
    """Symbol-aware retrieval tools for AutonomousAgentExecutor."""

    async def _retrieve_repo_symbols(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import asyncio as _asyncio

        state = ctx.state if isinstance(ctx.state, dict) else {}
        query_str = str(params.get("query", "")).strip()
        if not query_str:
            return {"error": "query is required"}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state, job=ctx.job
        )
        if not ws:
            return {
                "error": "No active coding workspace. Use clone_and_index_repo first."
            }
        lang_filter = params.get("language_filter")
        max_results = min(int(params.get("max_results", 20) or 20), 50)
        query_keywords = [
            t.strip()
            for t in query_str.replace("-", " ").replace("_", " ").split()
            if t.strip()
        ]
        try:
            retrieve_result = await _asyncio.to_thread(
                executor.symbol_index_service.retrieve,
                repo_root=ws.base_path,
                query_keywords=query_keywords,
                include_paths=[],
                max_symbols=max_results,
                max_snippets=min(max_results, 10),
            )
            if lang_filter:
                ext_map = {
                    "python": {".py"},
                    "typescript": {".ts", ".tsx"},
                    "javascript": {".js", ".jsx"},
                }
                allowed_exts = ext_map.get(lang_filter, set())
                if allowed_exts:
                    retrieve_result["symbol_matches"] = [
                        s
                        for s in retrieve_result.get("symbol_matches", [])
                        if any(
                            str(s.get("path", "")).endswith(ext) for ext in allowed_exts
                        )
                    ]
            return {
                "success": True,
                "data": retrieve_result,
                "findings": [
                    {
                        "type": "symbol_index",
                        "summary": {
                            k: v
                            for k, v in (retrieve_result or {}).items()
                            if isinstance(v, (int, float, str, bool))
                        },
                    }
                ],
            }
        except Exception as exc:
            return {"error": f"Symbol retrieval failed: {exc}"}

    async def _get_symbol_context(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import asyncio as _asyncio

        state = ctx.state if isinstance(ctx.state, dict) else {}
        symbol_name = str(params.get("symbol_name", "")).strip()
        file_path_param = str(params.get("file_path", "")).strip()
        if not symbol_name or not file_path_param:
            return {"error": "symbol_name and file_path are required"}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state, job=ctx.job
        )
        if not ws:
            return {"error": "No active coding workspace"}
        from app.services.repo_symbol_index_service import RepoSymbolIndexService

        if not RepoSymbolIndexService.reads(file_path_param):
            # "Not found" from an index that cannot read the file is a claim
            # about the file it has no grounds for.
            return {
                "error": (
                    f"the symbol index does not read {file_path_param!r} files; "
                    f"use search_code with a pattern for {symbol_name!r} in that "
                    "file, then read_file with the line range it reports"
                )
            }
        try:
            retrieve_result = await _asyncio.to_thread(
                executor.symbol_index_service.retrieve,
                repo_root=ws.base_path,
                query_keywords=[symbol_name],
                include_paths=[file_path_param],
                max_symbols=20,
                max_snippets=10,
            )
            matches = [
                s
                for s in retrieve_result.get("symbol_matches", [])
                if s.get("path") == file_path_param
            ]
            exact = [s for s in matches if s.get("symbol") == symbol_name]
            target = exact[0] if exact else (matches[0] if matches else None)
            if not target:
                return {
                    "error": f"Symbol '{symbol_name}' not found in {file_path_param}"
                }
            code_content, _ = executor.workspace_manager.read_file(
                ws,
                file_path_param,
                start_line=max(1, target.get("start_line", 1) - 5),
                end_line=(target.get("end_line") or target.get("start_line", 1)) + 5,
                max_chars=10000,
            )
            related = [s for s in matches if s.get("symbol") != symbol_name][:5]
            return {
                "success": True,
                "data": {
                    "symbol": target,
                    "code_context": code_content,
                    "related_symbols": related,
                    "file_path": file_path_param,
                },
                "findings": [
                    {
                        "type": "symbol_context",
                        "symbol": target,
                        "file_path": file_path_param,
                        "related_symbols": related[:20]
                        if isinstance(related, list)
                        else related,
                    }
                ],
            }
        except Exception as exc:
            return {"error": f"Symbol context retrieval failed: {exc}"}

    async def _find_tests_for_symbol(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import asyncio as _asyncio

        state = ctx.state if isinstance(ctx.state, dict) else {}
        symbol_name = str(params.get("symbol_name", "")).strip()
        if not symbol_name:
            return {"error": "symbol_name is required"}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state, job=ctx.job
        )
        if not ws:
            return {"error": "No active coding workspace"}
        try:
            retrieve_result = await _asyncio.to_thread(
                executor.symbol_index_service.retrieve,
                repo_root=ws.base_path,
                query_keywords=[symbol_name, "test"],
                include_paths=[],
                max_symbols=30,
                max_snippets=10,
            )
            test_matches = list(retrieve_result.get("related_tests", []))
            for sym in retrieve_result.get("symbol_matches", []):
                path_lower = str(sym.get("path", "")).lower()
                if executor.symbol_index_service._looks_like_test(path_lower):
                    entry = {
                        "path": sym.get("path"),
                        "symbol": sym.get("symbol"),
                        "score": sym.get("score", 0),
                    }
                    if entry not in test_matches:
                        test_matches.append(entry)
            return {
                "success": True,
                "data": {
                    "tests": test_matches[:20],
                    "count": len(test_matches[:20]),
                    "symbol_searched": symbol_name,
                },
                "findings": [
                    {
                        "type": "test_targets",
                        "symbol": symbol_name,
                        "tests": test_matches[:20],
                        "count": len(test_matches[:20]),
                    }
                ],
            }
        except Exception as exc:
            return {"error": f"Test search failed: {exc}"}

    return FunctionToolProvider(
        name="autonomous_symbol_retrieval_tools",
        modes={"autonomous"},
        handlers={
            "retrieve_repo_symbols": _retrieve_repo_symbols,
            "get_symbol_context": _get_symbol_context,
            "find_tests_for_symbol": _find_tests_for_symbol,
        },
    )
