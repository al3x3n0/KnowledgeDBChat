"""Autonomous-job tools: the ``workspace_mutation`` provider.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import select

from app.services.agent_tool_providers.base import (
    AgentToolExecutionContext,
    FunctionToolProvider,
)
from app.services.config_values import bounded_int
from app.utils.tree_copy import without_git_or_links


def _replay_finding_summary(replay: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Bound details carried by findings, retaining full counts for long runs."""
    if not replay:
        return None
    diagnostics = replay.get("program_diagnostics") or []
    issues = [row for row in diagnostics if row.get("status") != "match"]
    errors = replay.get("errored") or []
    return {
        "expected_comparisons": int(replay.get("expected_comparisons") or 0),
        "judged": int(replay.get("judged") or 0),
        "reproduced": int(replay.get("reproduced") or 0),
        "differed": len(replay.get("differed") or []),
        "skipped": len(replay.get("skipped") or []),
        "error_count": len(errors),
        "errors": deepcopy(errors[:10]),
        "program_status_counts": dict(
            sorted(Counter(row["status"] for row in diagnostics).items())
        ),
        "program_issue_count": len(issues),
        "program_issues": deepcopy(issues[:10]),
    }


def _as_float(value: Any) -> Optional[float]:
    """A number, or None. Never 0.0 for a missing value.

    The distinction matters here: a comparison that reads an absent claimed
    value as zero divides by it, and one that reads an absent measurement as
    zero scores a perfect failure against a claim nothing was measured for.
    None reaches the comparison as "not supplied" and comes back as a named
    blocker.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _diff_files(diff: str) -> List[str]:
    """Every path a unified diff touches, deletions included.

    Reading only `+++ b/` lines missed a deleted file, whose new side is
    `/dev/null`; its old side is the only place it is named.
    """
    files: List[str] = []
    for line in diff.splitlines():
        path = None
        if line.startswith("+++ b/"):
            path = line[len("+++ b/") :]
        elif line.startswith("--- a/"):
            path = line[len("--- a/") :]
        if path and path not in files:
            files.append(path)
    return files


def _new_file_diffs(ws: Any, manager: Any, diff: str) -> str:
    """Unified-diff hunks for files the run created that git does not track."""
    try:
        added = manager.get_status(ws).get("added") or []
    except Exception:  # noqa: BLE001
        return ""
    known = set(_diff_files(diff))
    chunks: List[str] = []
    for rel in added:
        if rel in known:
            continue
        try:
            text = (Path(ws.base_path) / rel).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        lines = text.splitlines()
        body = "".join(f"+{line}\n" for line in lines)
        chunks.append(
            f"diff --git a/{rel} b/{rel}\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            f"+++ b/{rel}\n"
            f"@@ -0,0 +1,{len(lines)} @@\n"
            f"{body}"
        )
    return "".join(chunks)


async def _store_patch_proposal(
    ctx: "AgentToolExecutionContext", proposal: Dict[str, Any]
) -> Any:
    """Write (or revise) the job's CodePatchProposal row; returns its id.

    One row per job (`uq_code_patch_proposals_job_id`): a second proposal from
    the same run revises the first while it is still awaiting review, and is
    refused once a person has decided on it.
    """
    job = ctx.job
    user_id = getattr(job, "user_id", None) or ctx.user_id
    if ctx.db is None or user_id is None:
        return None
    from app.models.code_patch_proposal import CodePatchProposal

    metadata = {
        "files_touched": proposal["files"],
        "rationale": proposal["rationale"],
        "lines_added": proposal["lines_added"],
        "lines_removed": proposal["lines_removed"],
        "workspace_id": proposal["workspace_id"],
        "origin": "propose_code_patch",
    }
    if "diff_truncated_from" in proposal:
        metadata["diff_truncated_from"] = proposal["diff_truncated_from"]
    row = None
    if job is not None:
        row = (
            await ctx.db.execute(
                select(CodePatchProposal).where(CodePatchProposal.job_id == job.id)
            )
        ).scalar_one_or_none()
    if row is not None and row.status != "proposed":
        return {
            "error": (
                f"This run's proposal was already {row.status}; a decided "
                "proposal is not overwritten."
            )
        }
    if row is None:
        row = CodePatchProposal(
            user_id=user_id,
            job_id=getattr(job, "id", None),
            status="proposed",
            title=proposal["title"][:500],
            diff_unified=proposal["diff"],
        )
        ctx.db.add(row)
    row.title = proposal["title"][:500]
    row.summary = proposal["rationale"] or None
    row.diff_unified = proposal["diff"]
    row.proposal_metadata = metadata
    await ctx.db.commit()
    return row.id


#: Where `execute_python` puts a copy of the repository, under /work.
WORKSPACE_COPY_DIR = "workspace"


def build_autonomous_workspace_mutation_provider(executor: Any) -> FunctionToolProvider:
    """Workspace mutation and code-execution tools for AutonomousAgentExecutor."""

    async def _resolve_user(ctx: AgentToolExecutionContext) -> Any:
        from app.models.user import User

        job = ctx.job
        user_result = await ctx.db.execute(select(User).where(User.id == job.user_id))
        user = user_result.scalar_one_or_none()
        if not user:
            raise ValueError("User not found for code execution")
        return user

    async def _sandbox_workdir(ctx: AgentToolExecutionContext) -> Any:
        """This run's sandbox directory, shared with the skills it uses."""
        from app.services.agent_sandbox_skill_tools import working_directory

        try:
            return await working_directory(ctx)
        except Exception as exc:  # noqa: BLE001 - the tool can still run
            logger.warning(f"No shared sandbox directory for this call: {exc}")
            return None

    async def _copy_workspace_in(
        params: Dict[str, Any],
        ctx: AgentToolExecutionContext,
        state: Dict[str, Any],
        workdir: Any,
    ) -> Dict[str, Any]:
        """Put a copy of the run's repository at `workspace/` in the sandbox.

        A copy, because the sandbox daemon resolves mounts on its own
        filesystem and cannot see where coding workspaces live -- the same
        reason the scan tools copy. It is the repository as it is now,
        without `.git`, replacing whatever an earlier call copied. **Nothing
        written there goes back**: a run changes code through `write_file`
        and `apply_patch`, which are the calls a patch proposal is built
        from, and a second way in would be one the proposal cannot see.
        """
        import asyncio
        import shutil

        manager = getattr(executor, "workspace_manager", None)
        ws = (
            manager.get_or_default(params.get("workspace_id"), state, job=ctx.job)
            if manager is not None
            else None
        )
        if ws is None:
            return {
                "error": "include_workspace needs a coding workspace, and this "
                "run has none. Call clone_and_index_repo first."
            }
        if workdir is None:
            return {"error": "This call has no sandbox directory to copy into"}
        target = Path(workdir) / WORKSPACE_COPY_DIR

        def copy() -> int:
            shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(
                ws.base_path,
                target,
                ignore=without_git_or_links,
            )
            return sum(1 for p in target.rglob("*") if p.is_file())

        try:
            files = await asyncio.to_thread(copy)
        except OSError as exc:
            return {"error": f"The workspace could not be copied in: {exc}"}
        return {
            "path": WORKSPACE_COPY_DIR,
            "files": files,
            "note": "A copy. Changes made here are not written back to the "
            "repository; use write_file or apply_patch for those.",
        }

    async def _execute_python(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from datetime import datetime

        from app.services.custom_tool_service import CustomToolService

        state = ctx.state if isinstance(ctx.state, dict) else {}
        code = str(params.get("code", ""))
        timeout = min(int(params.get("timeout_seconds", 10) or 10), 30)
        if not code.strip():
            return {"error": "No code provided"}
        workdir = await _sandbox_workdir(ctx)
        copied = None
        if params.get("include_workspace") is True:
            copied = await _copy_workspace_in(params, ctx, state, workdir)
            if copied.get("error"):
                return copied
        try:
            cts = CustomToolService()
            user = await _resolve_user(ctx)
            exec_result = await cts._execute_python(
                config={"code": code, "timeout_seconds": timeout},
                inputs={},
                user=user,
                workdir=workdir,
            )
            if copied:
                exec_result = {**exec_result, "workspace": copied}
            history = state.get("code_execution_history")
            if not isinstance(history, list):
                history = []
            history.append(
                {
                    "tool": "execute_python",
                    "success": True,
                    "code_preview": code[:200],
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            state["code_execution_history"] = history[-50:]
            return {"success": True, "data": exec_result}
        except Exception as exc:
            return {"error": f"Python execution failed: {exc}"}

    async def _execute_data_pipeline(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import json
        from datetime import datetime

        from app.core.config import settings
        from app.services.custom_tool_service import CustomToolService

        state = ctx.state if isinstance(ctx.state, dict) else {}
        code = str(params.get("code", ""))
        timeout = min(int(params.get("timeout_seconds", 60) or 60), 300)
        input_data = (
            params.get("input_data")
            if isinstance(params.get("input_data"), dict)
            else {}
        )
        if not code.strip():
            return {"error": "No code provided"}
        try:
            cts = CustomToolService()
            user = await _resolve_user(ctx)
            if getattr(settings, "CUSTOM_TOOL_DOCKER_ENABLED", False):
                wrapper = (
                    "import json, sys\n"
                    "input_data = json.loads(sys.stdin.read()) if not sys.stdin.isatty() else {}\n"
                    f"{code}\n"
                    "if 'result' in dir():\n"
                    "    print(json.dumps(result, default=str))\n"
                )
                exec_result = await cts._execute_docker(
                    config={
                        "image": "python:3.11-slim",
                        "command": ["python", "-c", wrapper],
                        "timeout_seconds": timeout,
                        "memory_limit": "512m",
                        "network_enabled": False,
                    },
                    inputs={"stdin": json.dumps(input_data, default=str)},
                    user=user,
                )
            else:
                exec_result = await cts._execute_python(
                    config={
                        "code": f"input_data = {repr(input_data)}\n{code}",
                        "timeout_seconds": timeout,
                    },
                    inputs={},
                    user=user,
                    workdir=await _sandbox_workdir(ctx),
                )
            history = state.get("code_execution_history")
            if not isinstance(history, list):
                history = []
            history.append(
                {
                    "tool": "execute_data_pipeline",
                    "success": True,
                    "code_preview": code[:200],
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            state["code_execution_history"] = history[-50:]
            return {"success": True, "data": exec_result}
        except Exception as exc:
            return {"error": f"Data pipeline execution failed: {exc}"}

    async def _write_and_run_script(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import json
        from datetime import datetime

        from app.core.config import settings
        from app.services.custom_tool_service import CustomToolService

        state = ctx.state if isinstance(ctx.state, dict) else {}
        # The name becomes a file in the container's working directory, so it
        # is a bare file name and nothing else.
        script_name = os.path.basename(str(params.get("script_name") or "script.py"))
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", script_name):
            script_name = "script.py"
        script_content = str(params.get("script_content", ""))
        timeout = min(int(params.get("timeout_seconds", 120) or 120), 300)
        input_data = (
            params.get("input_data")
            if isinstance(params.get("input_data"), dict)
            else {}
        )
        requirements = params.get("requirements") or []
        arguments = params.get("arguments") or []
        if not isinstance(arguments, list):
            arguments = []

        if not script_content.strip():
            return {"error": "No script content provided"}
        if not getattr(settings, "CUSTOM_TOOL_DOCKER_ENABLED", False):
            return {
                "error": "Docker execution is not enabled; write_and_run_script requires Docker"
            }
        if requirements:
            # The container has no network, so `pip install` cannot succeed;
            # it used to be chained in front of the script with `&&`, which
            # meant asking for a package guaranteed the script never ran.
            return {
                "error": (
                    "requirements cannot be installed: the script runs in a "
                    "container with no network. Use only the Python standard "
                    "library, or run it without requirements."
                )
            }
        try:
            cts = CustomToolService()
            user = await _resolve_user(ctx)
            exec_result = await cts._execute_docker(
                config={
                    "image": "python:3.11-slim",
                    # The script arrives as a file and the input as stdin.
                    # Arguments are passed as argv, never spliced into the
                    # shell line: an apostrophe in the data used to end the
                    # quoted string it had been pasted into.
                    "command": [
                        "bash",
                        "-c",
                        'cat > /workspace/input.json; exec python "$0" "$@"',
                        f"/workspace/{script_name}",
                        *[str(arg) for arg in arguments[:10]],
                    ],
                    "input_mode": "both",
                    "input_file_path": f"/workspace/{script_name}",
                    "timeout_seconds": timeout,
                    "memory_limit": "512m",
                    "network_enabled": False,
                },
                inputs={
                    "stdin": json.dumps(input_data, default=str),
                    "input_file_content": script_content,
                },
                user=user,
            )
            history = state.get("code_execution_history")
            if not isinstance(history, list):
                history = []
            history.append(
                {
                    "tool": "write_and_run_script",
                    "success": True,
                    "script_name": script_name,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            state["code_execution_history"] = history[-50:]
            return {"success": True, "data": exec_result}
        except Exception as exc:
            return {"error": f"Script execution failed: {exc}"}

    async def _write_file(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state, job=ctx.job
        )
        if not ws:
            return {"error": "No active coding workspace"}
        path = str(params.get("path", "")).strip()
        content = str(params.get("content", ""))
        if not path:
            return {"error": "path is required"}
        err = executor.workspace_manager.write_file(
            ws,
            path,
            content,
            create_dirs=params.get("create_dirs", True),
        )
        if err:
            return {"error": err}
        modified = state.get("coding_modified_files")
        if not isinstance(modified, list):
            modified = []
        if path not in modified:
            modified.append(path)
        state["coding_modified_files"] = modified[-200:]
        return {
            "success": True,
            "data": {"path": path, "bytes_written": len(content.encode("utf-8"))},
        }

    async def _create_workspace_checkpoint(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state, job=ctx.job
        )
        if not ws:
            return {"error": "No active coding workspace"}
        checkpoint, error = executor.workspace_manager.create_checkpoint(
            ws,
            label=str(params.get("label") or "").strip(),
            kind="manual",
        )
        if error:
            return {"error": error}
        state["coding_last_checkpoint_id"] = str(
            (checkpoint or {}).get("checkpoint_id") or ""
        )
        return {"success": True, "data": checkpoint}

    async def _restore_workspace_checkpoint(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state, job=ctx.job
        )
        if not ws:
            return {"error": "No active coding workspace"}
        checkpoint_id = str(params.get("checkpoint_id") or "").strip()
        if not checkpoint_id:
            return {"error": "checkpoint_id is required"}
        result, error = executor.workspace_manager.restore_checkpoint(
            ws,
            checkpoint_id,
            preserve_current=bool(params.get("preserve_current", True)),
        )
        if error:
            return {"error": error}
        status = (result or {}).get("status") or {}
        state["coding_modified_files"] = list(
            dict.fromkeys(
                [
                    *list(status.get("modified") or []),
                    *list(status.get("added") or []),
                    *list(status.get("deleted") or []),
                ]
            )
        )[:200]
        state["coding_last_restored_checkpoint_id"] = checkpoint_id
        return {"success": True, "data": result}

    async def _hydrate_candidate_snapshot(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state, job=ctx.job
        )
        if not ws:
            return {"error": "No active coding workspace"}
        config = ctx.job.config if isinstance(ctx.job.config, dict) else {}
        handoff = (
            config.get("swarm_handoff")
            if isinstance(config.get("swarm_handoff"), dict)
            else {}
        )
        configured_manifest = (
            config.get("candidate_snapshot")
            if isinstance(config.get("candidate_snapshot"), dict)
            else handoff.get("candidate_snapshot")
            if isinstance(handoff.get("candidate_snapshot"), dict)
            else None
        )
        configured_manifests = (
            config.get("candidate_snapshots")
            if isinstance(config.get("candidate_snapshots"), list)
            else []
        )
        requested_snapshot_id = str(params.get("snapshot_id") or "").strip()
        manifest = configured_manifest
        if requested_snapshot_id:
            if (
                isinstance(configured_manifest, dict)
                and str(configured_manifest.get("snapshot_id") or "")
                == requested_snapshot_id
            ):
                manifest = configured_manifest
            else:
                manifest = next(
                    (
                        item
                        for item in configured_manifests
                        if isinstance(item, dict)
                        and str(item.get("snapshot_id") or "") == requested_snapshot_id
                    ),
                    None,
                )
        elif not isinstance(manifest, dict) and len(configured_manifests) == 1:
            manifest = (
                configured_manifests[0]
                if isinstance(configured_manifests[0], dict)
                else None
            )
        if not isinstance(manifest, dict):
            return {
                "error": (
                    "No matching system-provided candidate snapshot is available; "
                    "supply snapshot_id when multiple candidates exist"
                )
            }
        result, error = await executor.workspace_manager.hydrate_candidate_snapshot(
            ws,
            manifest,
        )
        if error:
            return {"error": error}
        state["coding_hydrated_candidate_snapshot_id"] = str(
            manifest.get("snapshot_id") or ""
        )
        state["coding_modified_files"] = list(
            dict.fromkeys(
                [
                    *list((result or {}).get("hydrated_files") or []),
                    *list((result or {}).get("deleted_files") or []),
                ]
            )
        )[:200]
        return {"success": True, "data": result}

    async def _persist_durable_workspace_checkpoint(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        from app.services.agent_coding_durable_checkpoint_service import (
            agent_coding_durable_checkpoint_service,
        )

        try:
            manifest = await agent_coding_durable_checkpoint_service.persist(
                executor,
                ctx.job,
                state,
                label=str(params.get("label") or "").strip(),
                reason="agent_requested",
                db=ctx.db,
            )
        except Exception as exc:
            return {"error": f"Failed to persist durable checkpoint: {exc}"}
        if not isinstance(manifest, dict):
            return {"error": "Durable checkpoint was not created"}
        return {
            "success": True,
            "data": {
                "checkpoint_id": str(manifest.get("checkpoint_id") or ""),
                "session_id": str(manifest.get("session_id") or ""),
                "workspace_state_digest": str(
                    manifest.get("workspace_state_digest") or ""
                ),
                "persistence_complete": bool(
                    manifest.get("persistence_complete", False)
                ),
                "changes_summary": manifest.get("changes_summary") or {},
            },
        }

    async def _restore_durable_workspace_checkpoint(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        checkpoint_id = str(params.get("checkpoint_id") or "").strip()
        if not checkpoint_id:
            return {"error": "checkpoint_id is required"}
        from app.services.agent_coding_durable_checkpoint_service import (
            agent_coding_durable_checkpoint_service,
        )

        try:
            result = await agent_coding_durable_checkpoint_service.restore(
                executor,
                ctx.job,
                state,
                checkpoint_id=checkpoint_id,
            )
        except Exception as exc:
            return {"error": f"Failed to restore durable checkpoint: {exc}"}
        return {"success": True, "data": result}

    async def _apply_patch(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state, job=ctx.job
        )
        if not ws:
            return {"error": "No active coding workspace"}
        diff_text = str(params.get("diff", "")).strip()
        dry_run = bool(params.get("dry_run", False))
        if not diff_text:
            return {"error": "diff is required"}
        try:
            from app.services.code_patch_apply_service import CodePatchApplyService

            svc = CodePatchApplyService()
            file_diffs = svc.parse(diff_text)
            if not file_diffs:
                # A diff that parses to no file changes is a malformed diff,
                # not an applied patch. Reported as an error so that repeating
                # it escalates, and so a contract requiring `patch_applied`
                # cannot be satisfied by one.
                return {
                    "error": (
                        "The diff parsed to no file changes, so nothing was "
                        "applied. A unified diff needs a file header and a "
                        "hunk header with line numbers:\n"
                        "  --- a/path/to/file\n"
                        "  +++ b/path/to/file\n"
                        "  @@ -12,7 +12,7 @@\n"
                        "then context lines, and ' -' / ' +' for the change. "
                        "A bare '@@' with no line numbers parses to nothing."
                    )
                }
            applied_files = []
            errors = []
            for file_diff in file_diffs:
                file_path = file_diff.path
                target = executor.workspace_manager.safe_resolve(ws, file_path)
                if not target or not target.is_file():
                    errors.append(f"File not found: {file_path}")
                    continue
                original = target.read_text(encoding="utf-8", errors="replace")
                new_text, _debug = svc.apply_to_text(original, file_diff)
                if not dry_run:
                    target.write_text(new_text, encoding="utf-8")
                applied_files.append(file_path)
            if not dry_run:
                modified = state.get("coding_modified_files")
                if not isinstance(modified, list):
                    modified = []
                for file_path in applied_files:
                    if file_path not in modified:
                        modified.append(file_path)
                state["coding_modified_files"] = modified[-200:]
            if not applied_files:
                # Every hunk failed. Reporting success here was the worst of
                # the possible answers: a coding loop whose contract requires
                # `patch_applied` was satisfied by a patch that changed
                # nothing, so the run believed it had fixed the code while the
                # tests went on failing for the original reason.
                return {
                    "error": (
                        "No file was changed by this patch. "
                        + (
                            "; ".join(str(e) for e in errors[:5])
                            if errors
                            else "Every hunk failed to apply -- the context "
                            "lines probably do not match the file as it "
                            "stands. Read the file first and quote it exactly."
                        )
                    ),
                    "data": {"applied_files": [], "errors": errors},
                }
            return {
                "success": True,
                "data": {
                    "applied_files": applied_files,
                    "errors": errors,
                    "dry_run": dry_run,
                    "files_count": len(applied_files),
                },
                "findings": [
                    {
                        "type": "patch_applied",
                        "applied_files": applied_files[:50],
                        "files_count": len(applied_files),
                        "dry_run": dry_run,
                        "errors": errors[:10],
                    }
                ],
            }
        except Exception as exc:
            return {"error": f"Patch failed: {exc}"}

    async def _run_repo_tests(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        """Run the repository's tests and report what actually happened.

        Distinct from `run_command` on purpose. A gate needs to know whether
        the tests *ran*, and an exit code cannot say: a harness that failed to
        start exits non-zero exactly like a failing test, and those call for
        opposite responses.
        """
        import asyncio
        import os

        from app.core.config import settings as app_settings
        from app.core.feature_flags import get_flag
        from app.services.coding_test_gate import (
            DEFAULT_TEST_COMMANDS,
            read_test_output,
        )

        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state, job=ctx.job
        )
        if ws is None:
            return {"error": "No active coding workspace"}

        enabled = await get_flag("unsafe_code_execution_enabled")
        if not enabled and not bool(
            getattr(app_settings, "ENABLE_UNSAFE_CODE_EXECUTION", False)
        ):
            return {
                "error": (
                    "Running tests requires unsafe_code_execution_enabled; "
                    "the suite runs real processes in the workspace."
                )
            }

        command = str(params.get("command") or "").strip()
        inferred_from = ""
        if not command:
            # Pick by the marker file present, rather than guessing one
            # ecosystem. A wrong default reports "no tests ran", which is at
            # least honest, but naming the marker makes it fixable.
            for marker, candidate in DEFAULT_TEST_COMMANDS:
                if os.path.exists(os.path.join(str(ws.base_path), marker)):
                    command, inferred_from = candidate, marker
                    break
        if not command:
            return {
                "error": (
                    "No test command given and no marker file recognised "
                    "(pytest.ini, pyproject.toml, package.json, go.mod, "
                    "Cargo.toml). Pass `command` explicitly."
                )
            }

        timeout = min(int(params.get("timeout_seconds", 300) or 300), 900)
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_shell(
                    command,
                    cwd=str(ws.base_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=10,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            return {
                "success": False,
                "data": {
                    "ran": False,
                    "green": False,
                    "note": f"Test run exceeded {timeout}s and was abandoned",
                },
            }
        except Exception as exc:  # noqa: BLE001 - the command itself is user input
            return {"error": f"Could not run the tests: {exc}"}

        outcome = read_test_output(
            stdout_bytes.decode("utf-8", errors="replace")[:20000],
            stderr_bytes.decode("utf-8", errors="replace")[:20000],
            proc.returncode,
        )
        evidence = outcome.as_evidence()
        evidence["command"] = command
        if inferred_from:
            evidence["command_inferred_from"] = inferred_from

        return {
            # `success` is whether the tool worked, not whether the tests
            # passed: a red suite is a successful measurement of a broken
            # tree, and conflating them makes a gate impossible to write.
            "success": True,
            "data": evidence,
            "findings": [
                {
                    "type": "test_result",
                    "title": (
                        f"{outcome.passed} passed, {outcome.failed} failed"
                        if outcome.ran
                        else "Tests did not run"
                    ),
                    **evidence,
                }
            ],
        }

    async def _propose_code_patch(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        """Record the workspace's changes as a reviewable proposal.

        Deliberately does not touch a repository or a remote. The proposal is
        the artefact a person reads; opening anything against a remote stays
        outside what a run can do on its own.
        """
        import asyncio

        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state, job=ctx.job
        )
        if ws is None:
            return {"error": "No active coding workspace"}

        title = str(params.get("title") or "").strip()
        if not title:
            return {"error": "title is required"}

        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    "git",
                    "diff",
                    cwd=str(ws.base_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=10,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=60
            )
            diff = stdout_bytes.decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            return {"error": f"Could not read the workspace diff: {exc}"}

        if proc.returncode not in (0, None):
            # Outside a repository git exits 129 with nothing on stdout, which
            # used to read as "no changes". A workspace built from KB
            # documents is given a baseline commit when it is made, so this is
            # now one whose baseline could not be written.
            detail = (stderr_bytes or b"").decode("utf-8", errors="replace").strip()
            return {
                "error": (
                    f"git diff failed (exit {proc.returncode}), so the "
                    "workspace's changes could not be read: "
                    f"{detail.splitlines()[0] if detail else 'no output'}"
                )
            }

        # Bare `git diff` shows tracked files only. A file the run created is
        # a change all the same, and get_status already counts it.
        diff += _new_file_diffs(ws, executor.workspace_manager, diff)

        if not diff.strip():
            # A proposal with no diff is the shape of a run that believes it
            # changed something and did not.
            return {
                "error": (
                    "The workspace has no uncommitted changes, so there is "
                    "nothing to propose."
                )
            }

        files = _diff_files(diff)
        stored_diff = diff[:200000]
        proposal = {
            "title": title,
            "rationale": str(params.get("rationale") or ""),
            "diff": stored_diff,
            "files": files,
            "lines_added": sum(
                1
                for line in diff.splitlines()
                if line.startswith("+") and not line.startswith("+++")
            ),
            "lines_removed": sum(
                1
                for line in diff.splitlines()
                if line.startswith("-") and not line.startswith("---")
            ),
            "workspace_id": getattr(ws, "workspace_id", None),
        }
        if len(diff) > len(stored_diff):
            proposal["diff_truncated_from"] = len(diff)
        state["code_patch_proposal"] = proposal

        # The proposal is meant for a person, and people read /code-patches.
        # Kept only in the run state, it reached no review surface at all.
        stored = await _store_patch_proposal(ctx, proposal)
        if isinstance(stored, dict) and stored.get("error"):
            return stored
        data = {k: v for k, v in proposal.items() if k != "diff"}
        if stored is not None:
            data["proposal_id"] = str(stored)

        return {
            "success": True,
            "data": data,
            "findings": [
                {
                    "type": "code_patch_proposal",
                    "title": title,
                    "files": files,
                    "lines_added": proposal["lines_added"],
                    "lines_removed": proposal["lines_removed"],
                }
            ],
        }

    async def _run_command(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import asyncio
        import os
        from datetime import datetime

        from app.core.config import settings as app_settings
        from app.core.feature_flags import get_flag

        state = ctx.state if isinstance(ctx.state, dict) else {}
        ws = executor.workspace_manager.get_or_default(
            params.get("workspace_id"), state, job=ctx.job
        )
        if not ws:
            return {"error": "No active coding workspace"}
        command = str(params.get("command", "")).strip()
        if not command:
            return {"error": "command is required"}
        from app.services.agent_job_creation_service import agent_job_creation_service

        unsafe_commands = agent_job_creation_service.find_unsafe_commands([command])
        if unsafe_commands:
            return {
                "success": False,
                "error": "Command rejected by coding harness safety policy",
                "data": {"blocked_commands": unsafe_commands},
            }
        enabled = await get_flag("unsafe_code_execution_enabled")
        if enabled is None:
            enabled = bool(getattr(app_settings, "ENABLE_UNSAFE_CODE_EXECUTION", False))
        if not enabled:
            return {
                "error": "Shell execution requires unsafe_code_execution_enabled feature flag"
            }
        timeout = min(int(params.get("timeout_seconds", 30) or 30), 120)
        extra_env = params.get("env") if isinstance(params.get("env"), dict) else {}
        env = {**os.environ, **extra_env, "HOME": str(ws.base_path)}
        max_output = int(
            getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_STDOUT_CHARS", 20000) or 20000
        )
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    "/bin/sh",
                    "-lc",
                    command,
                    cwd=str(ws.base_path),
                    env=env,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=5,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            stdout_str = stdout_bytes.decode("utf-8", errors="replace")[:max_output]
            stderr_str = stderr_bytes.decode("utf-8", errors="replace")[:max_output]
            history = state.get("coding_command_history")
            if not isinstance(history, list):
                history = []
            history.append(
                {
                    "command": command[:200],
                    "exit_code": proc.returncode,
                    "stdout_preview": stdout_str[:200],
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            state["coding_command_history"] = history[-50:]
            command_succeeded = proc.returncode == 0
            result = {
                "success": command_succeeded,
                "data": {
                    "exit_code": proc.returncode,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "command": command[:200],
                },
            }
            if not command_succeeded:
                result["error"] = f"Command exited with status {proc.returncode}"
            elif bool((ctx.job.config or {}).get("coding_harness_may_mutate")):
                workspace_status = executor.workspace_manager.get_status(ws)
                if int(workspace_status.get("changes_count") or 0) > 0:
                    try:
                        from app.services.agent_coding_durable_checkpoint_service import (
                            agent_coding_durable_checkpoint_service,
                        )

                        durable_checkpoint = (
                            await agent_coding_durable_checkpoint_service.persist(
                                executor,
                                ctx.job,
                                state,
                                label=f"Verified by {command[:80]}",
                                reason="successful_verification",
                                db=ctx.db,
                            )
                        )
                        if isinstance(durable_checkpoint, dict):
                            result["data"]["durable_checkpoint_id"] = str(
                                durable_checkpoint.get("checkpoint_id") or ""
                            )
                    except Exception as checkpoint_exc:
                        result["data"]["durable_checkpoint_error"] = str(
                            checkpoint_exc
                        )[:500]
            # The declared evidence has to actually be emitted: a contract
            # asking for command_result plans this tool, and without a finding
            # the tool runs, succeeds, and leaves the contract exactly as
            # unsatisfied as before.
            if isinstance(result, dict) and not result.get("error"):
                payload = (
                    result.get("data") if isinstance(result.get("data"), dict) else {}
                )
                result.setdefault(
                    "findings",
                    [
                        {
                            "type": "command_result",
                            "command": command[:200],
                            "exit_code": payload.get("exit_code"),
                            "stdout": str(payload.get("stdout") or "")[:2000],
                            "stderr": str(payload.get("stderr") or "")[:2000],
                        }
                    ],
                )
            return result
        except asyncio.TimeoutError:
            return {"error": f"Command timed out after {timeout}s"}
        except Exception as exc:
            return {"error": f"Command failed: {exc}"}

    async def _compile_c_snippet(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services import agent_compiler_sandbox

        return await agent_compiler_sandbox.compile_c_snippet(
            code=str(params.get("code") or ""),
            flags=str(params.get("flags") or "-O2"),
            emit=str(params.get("emit") or "asm"),
            label=str(params.get("label") or ""),
        )

    async def _retract_finding(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.models.agent_retraction import RetractionKind
        from app.services import agent_retract_tool, agent_retraction_service

        ref = str(params.get("ref") or "").strip()
        reason = str(params.get("reason") or "").strip()
        cited = params.get("contradicted_by")
        if isinstance(cited, str):
            try:
                cited = json.loads(cited)
            except json.JSONDecodeError:
                cited = [c.strip() for c in cited.split(",") if c.strip()]
        cited = [str(c) for c in (cited or [])]

        problem = agent_retract_tool.check(ref, reason, cited, ctx.state)
        if problem:
            return {"error": problem}

        # The finding has to exist, and be the caller's. Only the shape of
        # the ref was checked, so a job that does not exist, an index past
        # the end and another user's job were all "retracted" successfully.
        job_part, _, index_part = ref.partition("#")
        from uuid import UUID as _UUID

        from app.models.agent_job import AgentJob as _AgentJob

        owner_id = getattr(ctx.job, "user_id", None) or ctx.user_id
        try:
            target = (
                await ctx.db.execute(
                    select(_AgentJob).where(
                        _AgentJob.id == _UUID(job_part.strip()),
                        _AgentJob.user_id == owner_id,
                    )
                )
            ).scalar_one_or_none()
            index = int(index_part.strip())
        except (ValueError, TypeError, AttributeError):
            return {"error": f"{ref} does not name a finding (expected job_id#index)"}
        findings = (
            (target.results or {}).get("findings")
            if target is not None and isinstance(target.results, dict)
            else None
        )
        if not isinstance(findings, list) or not 0 <= index < len(findings):
            return {"error": f"No finding {ref} was found among your jobs"}

        row = await agent_retraction_service.retract(
            ctx.db,
            user_id=ctx.user_id or getattr(ctx.job, "user_id", None),
            kind=RetractionKind.FINDING,
            ref=ref,
            reason=reason,
            source="; ".join(cited)[:200],
            # The context has a job, not a job_id: this was always NULL.
            source_job_id=getattr(ctx.job, "id", None),
        )
        return {
            "success": True,
            "data": {
                "retracted": ref,
                "retraction_id": str(getattr(row, "id", "")),
                "contradicted_by": cited,
            },
            "note": (
                "That finding will no longer be recalled by later runs. It is "
                "withdrawn, not deleted: the record keeps the reason so a "
                "reader can tell why."
            ),
        }

    async def _verify_run_bundle(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services import agent_evidence_bundle as bundle

        job_id = getattr(getattr(ctx, "job", None), "id", None)
        if not job_id:
            return {"error": "No job in context; there is no bundle to verify"}

        # An earlier run's bundle, when asked for. recall_prior_findings hands
        # back the job that produced a number; without this, a run could reuse
        # that number but not check the evidence under it, which is trust
        # rather than verification -- in a project whose history includes a
        # prediction cited from a tool result that had failed.
        requested = str(params.get("job_id") or "").strip()
        other_job = None
        if requested and requested != str(job_id):
            from uuid import UUID as _UUID

            from app.models.agent_job import AgentJob as _AgentJob

            try:
                requested_uuid = _UUID(requested)
            except (ValueError, AttributeError, TypeError):
                return {
                    "error": (
                        f"job_id {requested!r} is not a job id. Use the "
                        "`recalled_from_job` value that recall_prior_findings "
                        "returns with each finding."
                    )
                }
            found = await ctx.db.execute(
                select(_AgentJob).where(
                    _AgentJob.id == requested_uuid,
                    # Scoped to the owner. A bundle holds whatever a run
                    # measured, and jobs belong to users.
                    _AgentJob.user_id == ctx.job.user_id,
                )
            )
            other_job = found.scalar_one_or_none()
            if other_job is None:
                return {
                    "error": (
                        f"No job {requested} belonging to this user. A bundle "
                        "can only be verified by the owner of the run that "
                        "wrote it."
                    )
                }
            job_id = other_job.id

        integrity = bundle.verify_integrity(str(job_id))
        if not integrity["entries"]:
            return {
                "error": (
                    f"Job {job_id} recorded no evidence, so there is nothing "
                    "to verify."
                    if other_job is not None
                    else "This run has recorded no evidence yet, so there is "
                    "nothing to verify. Run the measurements first."
                )
            }

        # Before any replay: the summary describes what the run recorded.
        summary = bundle.summarize(str(job_id))
        replay: Dict[str, Any] = {}
        if bool(params.get("replay", False)):
            import dataclasses

            replay_ctx = dataclasses.replace(
                ctx, extra={**(ctx.extra or {}), "replaying_bundle": True}
            )

            async def execute(tool: str, tool_params: Dict[str, Any]) -> Any:
                handled, result = await executor.tool_registry.try_execute(
                    tool, tool_params, replay_ctx
                )
                if not handled:
                    raise RuntimeError(f"Tool '{tool}' is unavailable for replay")
                return result

            replay = await bundle.replay_bundle(str(job_id), execute)

        verdict = replay.get("verdict") if replay else "not replayed"
        replay_summary = _replay_finding_summary(replay)
        warnings = ""
        if replay_summary and (
            replay_summary["error_count"] or replay_summary["program_issue_count"]
        ):
            warnings = (
                f", replay errors: {replay_summary['error_count']}, "
                f"program issues: {replay_summary['program_issue_count']}"
            )
        finding = {
            "type": "bundle_verified",
            "id": bundle.verification_context_id(str(job_id), integrity, replay),
            "title": (
                f"Evidence bundle {job_id}: {summary['entries']} calls recorded, "
                f"integrity {'intact' if integrity['intact'] else 'BROKEN'}, "
                f"replay {verdict}{warnings}"
            ),
            "verified": bool(integrity["intact"]),
            "intact": integrity["intact"],
            "replay_verdict": verdict,
            "verified_job_id": str(job_id),
            "replay_summary": replay_summary,
            "entries": summary["entries"],
        }
        findings = [finding]
        if bundle.reproduction_established(finding):
            findings.append(
                {
                    **deepcopy(finding),
                    "type": "bundle_reproduced",
                    "id": finding["id"] + ":reproduced",
                    "title": (
                        f"Reproduced bundle {job_id}: "
                        f"{replay_summary['reproduced']} compared results matched{warnings}"
                    ),
                }
            )
        return {
            "success": True,
            "verified_job_id": str(job_id),
            "verified_own_run": other_job is None,
            # Where the bundle actually is. A host path in a gitignored .env
            # sent two days of bundles into the container's own filesystem, to
            # be destroyed on the next recreate, while this tool read the same
            # wrong path and reported success every time. The location is the
            # one fact that would have made that visible.
            "bundle_root": str(bundle.BUNDLE_ROOT),
            "data": {
                "bundle": summary,
                "integrity": integrity,
                "replay": replay,
                "note": (
                    "Integrity shows the artifacts are the ones this run "
                    "produced. Only a replay shows they can be produced again, "
                    "and it judges nothing that reports wall clock."
                ),
            },
            "findings": findings,
        }

    async def _record_prediction(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services import agent_calibration_service as calibration

        job = getattr(ctx, "job", None)
        # ctx.user_id is not populated in autonomous runs; the owner is the
        # job's user, which is how the other write tools resolve it.
        owner_id = getattr(job, "user_id", None) or ctx.user_id
        tags = params.get("methodology_tags")

        # What evidence actually exists in this run right now. A prediction
        # that cites a measurement it never obtained is the worst failure this
        # store can suffer: a run predicted from "llvm-mca reported 11.8 cycles
        # per iteration" while its only mca call had failed, and the real
        # answer -- 59.05 -- arrived three iterations later. The error column
        # caught the consequence and could not see the cause.
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
        required = params.get("derived_from")
        required = (
            [str(r).strip() for r in required if str(r).strip()]
            if isinstance(required, list)
            else []
        )
        # Required, not optional. Left optional, the guard never fired: a run
        # that had just fabricated an llvm-mca result simply did not mention
        # what it derived from, and nothing asked. A prediction with no
        # measurement behind it is legitimate, but it has to say so.
        if not required:
            return {
                "error": (
                    "derived_from is required: list the finding types this "
                    "number comes from, e.g. ['cycle_model_measurement']. "
                    f"Findings available in this run: "
                    f"{', '.join(available) or 'none'}. If the prediction is a "
                    "judgement with no measurement behind it, pass ['none'] "
                    "and say so in the methodology."
                )
            }
        declared_guess = required == ["none"]
        if not declared_guess:
            from app.services import agent_evidence_citation

            resolved, missing = agent_evidence_citation.resolve_all(required, available)
            if missing:
                return {
                    "error": agent_evidence_citation.explain_unresolved(
                        missing, available
                    )
                }
            # Store the resolved type names rather than the prose the caller
            # wrote, so the record says which evidence it rests on.
            required = resolved
        try:
            # A savepoint, not the caller's transaction: a rejected insert
            # would otherwise poison the session the whole run shares.
            async with ctx.db.begin_nested():
                prediction = await calibration.record_prediction(
                    ctx.db,
                    subject=str(params.get("subject") or ""),
                    metric=str(params.get("metric") or ""),
                    predicted_value=float(params.get("predicted_value") or 0.0),
                    methodology=str(params.get("methodology") or ""),
                    prediction_basis=str(params.get("prediction_basis") or ""),
                    # Record what evidence was on hand when the claim was
                    # made, so a later reader can tell a derived prediction
                    # from a guess without taking the methodology text at its
                    # word.
                    methodology_tags=(
                        ([str(t) for t in tags] if isinstance(tags, list) else [])
                        + [f"evidence:{name}" for name in available]
                        + (["declared:no-measurement"] if declared_guess else [])
                    )
                    or None,
                    job_id=getattr(job, "id", None),
                    user_id=owner_id,
                )
            await ctx.db.commit()
        except calibration.CalibrationError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Could not record the prediction: {str(exc)[:200]}"}

        return {
            "success": True,
            "data": {
                "prediction_id": str(prediction.id),
                "subject": prediction.subject,
                "metric": prediction.metric,
                "predicted_value": prediction.predicted_value,
                "note": (
                    "Recorded before the outcome is known. Settle it with "
                    "record_measurement once the referee has run."
                ),
            },
            "findings": [
                {
                    "type": "prediction_recorded",
                    "title": (
                        f"Predicted {prediction.metric}={prediction.predicted_value} "
                        f"for {prediction.subject}"
                    ),
                    "prediction_id": str(prediction.id),
                }
            ],
        }

    async def _record_measurement(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from uuid import UUID as _PredUUID

        from app.services import agent_calibration_service as calibration

        raw_id = str(params.get("prediction_id") or "").strip()
        try:
            prediction_id = _PredUUID(raw_id)
        except (ValueError, AttributeError, TypeError):
            return {
                "error": (
                    f"prediction_id should be a UUID, got {raw_id!r}; it is the id "
                    "record_prediction returned."
                )
            }
        try:
            async with ctx.db.begin_nested():
                settled = await calibration.record_measurement(
                    ctx.db,
                    prediction_id=prediction_id,
                    measured_value=float(params.get("measured_value") or 0.0),
                    measurement_source=str(params.get("measurement_source") or ""),
                    notes=str(params.get("notes") or ""),
                )
            await ctx.db.commit()
        except calibration.CalibrationError as exc:
            return {"error": str(exc)}
        except Exception as exc:
            return {"error": f"Could not record the measurement: {str(exc)[:200]}"}

        return {
            "success": True,
            "data": {
                "prediction_id": str(settled.id),
                "predicted_value": settled.predicted_value,
                "measured_value": settled.measured_value,
                "error_absolute": settled.error_absolute,
                "relative_error": settled.error_relative,
                "measurement_source": settled.measurement_source,
            },
            "findings": [
                {
                    "type": "prediction_settled",
                    "title": (
                        f"{settled.subject}: predicted {settled.predicted_value}, "
                        f"measured {settled.measured_value} "
                        f"({settled.measurement_source})"
                    ),
                    "relative_error": settled.error_relative,
                }
            ],
        }

    async def _analyze_snippet_cycles(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services import agent_compiler_sandbox

        return await agent_compiler_sandbox.analyze_snippet_cycles(
            code=str(params.get("code") or ""),
            asm=str(params.get("asm") or ""),
            cpu=str(params.get("cpu") or ""),
            flags=str(params.get("flags") or "-O3"),
            target=str(
                params.get("target") or agent_compiler_sandbox.DEFAULT_ANALYSIS_TARGET
            ),
            iterations=params.get("iterations", 100),
            label=str(params.get("label") or ""),
        )

    async def _benchmark_c_snippet(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services import agent_compiler_sandbox

        # `repeat` is forwarded only when the caller actually chose one. It
        # used to be restated as `or 3` here, a second copy of a default that
        # also lives on benchmark_c_snippet -- so raising the sandbox default
        # to 5 changed nothing for agents, which reach the tool exclusively
        # through this wrapper. Measured: a swarm launched after the change
        # still took three trials, one of which stalled at 230 ms.
        kwargs: Dict[str, Any] = {
            "code": str(params.get("code") or ""),
            # No "-O2" default here any more: it is wrong for Rust, which
            # rejects the flag outright. The toolchain supplies its own.
            "flags": str(params.get("flags") or ""),
            "label": str(params.get("label") or ""),
            "language": str(params.get("language") or ""),
        }
        if params.get("repeat"):
            kwargs["repeat"] = int(params["repeat"])
        return await agent_compiler_sandbox.benchmark_c_snippet(**kwargs)

    async def _check_implementation(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        """Establish that the code about to be timed computes the right answer."""
        from app.services import agent_implementation_check as impl

        def program(value: Any) -> Optional[impl.OracleProgram]:
            if not isinstance(value, dict):
                return None
            return impl.OracleProgram(
                code=str(value.get("code") or ""),
                language=str(value.get("language") or ""),
            )

        generator = params.get("generator")
        outcome = await impl.check_implementation(
            code=str(params.get("code") or ""),
            cases=params.get("cases") or [],
            flags=str(params.get("flags") or ""),
            language=str(params.get("language") or ""),
            tolerance=float(params.get("tolerance") or impl.DEFAULT_TOLERANCE),
            reference=program(params.get("reference")),
            generator=program(generator),
            generated_cases=bounded_int(
                generator.get("count") if isinstance(generator, dict) else None,
                impl.DEFAULT_GENERATED_CASES,
                1,
                impl.MAX_GENERATED_CASES,
            ),
        )
        evidence = outcome.as_evidence()

        # A call that supplied nothing to check against is a mistake in the
        # call, not a fact about the code, and it is reported as an error so
        # that repeating it escalates. That matters: a check reporting
        # `verified: false` is a SUCCESSFUL tool call, so the repeat-failure
        # diagnosis never saw it, and one run made this identical mistake
        # three times in a row -- three iterations of its budget spent on a
        # correction nothing was pressing it to make.
        #
        # A check whose cases ran and failed stays a success with
        # verified=false: that is a real result about the implementation, and
        # turning it into an error would hide the thing the gate exists to
        # report.
        # The oracle refusals are the same kind: the reference, the
        # generator or their pairing is the call's to fix.
        if outcome.reason in (
            "no_cases",
            "bad_language",
            "bad_flags",
            "oracle_incomplete",
            "reference_is_candidate",
            "reference_compile_error",
            "generator_compile_error",
        ):
            return {"error": outcome.note, "data": evidence}

        return {
            "success": True,
            "data": evidence,
            # Recorded whether or not it passed. A failed check is a finding a
            # later stage needs to see: it is the difference between "this
            # algorithm is slow" and "this implementation is wrong".
            "findings": [
                {
                    "type": "implementation_verified",
                    "subject": str(params.get("label") or "implementation"),
                    **evidence,
                }
            ],
        }

    async def _compare_to_claim(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        """Score a measurement against the paper's number, or refuse to.

        The verdict is `incomparable` rather than an error whenever the
        comparison does not hold -- including when the implementation was never
        checked for correctness. That is not a technicality: a benchmark of
        code nobody verified is an accurate timing of unknown work, and scoring
        it against a paper's claim launders it into a reproduction result.
        Returning a verdict with the blocker named tells the run what to fix;
        an error would just look like the tool being broken.
        """
        from app.services import agent_claim_comparison as claims

        state = ctx.state if isinstance(ctx.state, dict) else {}
        findings = (
            state.get("findings") if isinstance(state.get("findings"), list) else []
        )
        verifications = [
            f
            for f in findings
            if isinstance(f, dict)
            and str(f.get("type") or "") == "implementation_verified"
        ]
        verified = any(f.get("verified") is True for f in verifications)

        comparison = claims.compare(
            claimed_value=_as_float(params.get("claimed_value")),
            measured_value=_as_float(params.get("measured_value")),
            claimed_unit=params.get("claimed_unit"),
            measured_unit=params.get("measured_unit"),
            measurement_source=params.get("measurement_source"),
            claimed_conditions=params.get("claimed_conditions"),
            measured_conditions=params.get("measured_conditions"),
            tolerance=_as_float(params.get("tolerance")),
        )

        # A number the machine was too busy to take cannot settle a claim
        # either, and the benchmark already reported how busy it was. The most
        # recent measurement is the one being scored.
        benchmarks = [
            f
            for f in findings
            if isinstance(f, dict)
            and str(f.get("type") or "") == "benchmark_measurement"
        ]
        concerns = claims.measurement_concerns(benchmarks[-1]) if benchmarks else []
        for concern in concerns:
            comparison.blockers.append(concern)
        if concerns:
            comparison.verdict = claims.VERDICT_INCOMPARABLE
            comparison.summary = (
                "Not comparable: the measurement itself is not trustworthy. "
                + concerns[0]
            )

        if not verified:
            reason = (
                "no implementation_verified finding in this run"
                if not verifications
                else "the correctness check on this implementation did not pass"
            )
            comparison.blockers.insert(
                0,
                (
                    f"The measured code was never established to compute the "
                    f"right answer ({reason}): a timing of unverified code is "
                    "accurate for work nobody checked. Run check_implementation "
                    "against the paper's worked examples first."
                ),
            )
            comparison.verdict = claims.VERDICT_INCOMPARABLE
            comparison.summary = (
                "Not comparable: the implementation's correctness was never "
                "established, so this measurement cannot settle the paper's claim."
            )

        evidence = comparison.as_evidence()
        return {
            "success": True,
            "data": evidence,
            "findings": [
                {
                    "type": "reproduction_verdict",
                    "subject": str(params.get("subject") or ""),
                    "metric": str(params.get("metric") or ""),
                    "measurement_source": str(params.get("measurement_source") or ""),
                    **evidence,
                }
            ],
        }

    async def _create_custom_tool(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        """Create a reusable tool owned by this user.

        Mirrors the validation on POST /user-tools: docker_container stays
        behind CUSTOM_TOOL_DOCKER_ENABLED, and workflow_runner is reserved for
        workflow synthesis, which fills in the workflow id it points at.
        """
        from app.models.workflow import UserTool
        from app.services.custom_tool_types import reject_custom_tool_type

        name = str(params.get("name") or "").strip()
        if not name:
            return {"error": "name is required"}
        tool_type = str(params.get("tool_type") or "").strip().lower()

        # An agent may not create a workflow_runner: that type points at a
        # workflow id which workflow synthesis fills in.
        rejection = reject_custom_tool_type(tool_type, include_workflow_runner=False)
        if rejection:
            return {"error": rejection}

        config = params.get("config")
        if not isinstance(config, dict) or not config:
            return {"error": "config is required and must be an object"}
        schema = params.get("parameters_schema")
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}

        # ctx.user_id is not populated in autonomous runs; the owner is the
        # job's user, which is how the other write tools resolve it.
        owner_id = getattr(getattr(ctx, "job", None), "user_id", None) or ctx.user_id
        if owner_id is None:
            return {"error": "Cannot determine the owning user for the new tool"}

        existing = (
            await ctx.db.execute(
                select(UserTool).where(
                    UserTool.user_id == owner_id, UserTool.name == name
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            return {
                "error": (
                    f"A tool named {name!r} already exists. Choose another "
                    "name, or call it with run_custom_tool."
                )
            }

        tool = UserTool(
            user_id=owner_id,
            name=name,
            description=str(params.get("description") or "").strip() or None,
            tool_type=tool_type,
            parameters_schema=schema,
            config=config,
            is_enabled=True,
        )
        # A savepoint, not the caller's transaction: a rejected insert here
        # would otherwise poison the session the whole run shares, and one bad
        # tool definition would end the job rather than the action.
        try:
            async with ctx.db.begin_nested():
                ctx.db.add(tool)
            await ctx.db.commit()
        except Exception as exc:
            return {"error": f"Could not create the tool: {str(exc)[:200]}"}

        return {
            "success": True,
            "data": {
                "tool_id": str(tool.id),
                "name": tool.name,
                "tool_type": tool.tool_type,
            },
            "findings": [
                {
                    "type": "tool_created",
                    "title": f"Created custom tool {tool.name!r} ({tool.tool_type})",
                    "tool_id": str(tool.id),
                }
            ],
        }

    def _tool_owner(ctx: AgentToolExecutionContext) -> Any:
        """Autonomous runs carry the user on the job, not on the context."""
        return getattr(getattr(ctx, "job", None), "user_id", None) or ctx.user_id

    async def _run_custom_tool_autonomous(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        # AgentService is the chat-mode surface and is not reachable from the
        # autonomous executor; go to the same service it uses.
        from sqlalchemy import func

        from app.models.user import User
        from app.models.workflow import UserTool
        from app.services.custom_tool_service import CustomToolService

        owner_id = _tool_owner(ctx)
        if owner_id is None:
            return {"error": "Cannot determine the owning user for this tool"}
        tool_name = str(params.get("tool_name") or "").strip()
        if not tool_name:
            return {"error": "tool_name is required"}

        tool = (
            await ctx.db.execute(
                select(UserTool).where(
                    UserTool.user_id == owner_id,
                    func.lower(UserTool.name) == tool_name.lower(),
                )
            )
        ).scalar_one_or_none()
        if tool is None:
            return {"error": f"No custom tool named {tool_name!r} for this user"}
        if not tool.is_enabled:
            return {"error": f"Custom tool {tool_name!r} is disabled"}

        user = (
            await ctx.db.execute(select(User).where(User.id == owner_id))
        ).scalar_one_or_none()

        inputs = params.get("inputs")
        if not isinstance(inputs, dict):
            inputs = {}
        try:
            output = await CustomToolService().execute_tool(
                tool=tool,
                inputs=inputs,
                user=user,
                db=ctx.db,
                workdir=await _sandbox_workdir(ctx),
            )
        except Exception as exc:
            return {"error": f"Custom tool {tool_name!r} failed: {str(exc)[:300]}"}

        return {
            "success": True,
            "data": {"tool_name": tool.name, "output": output},
            "findings": [
                {
                    "type": "custom_tool_result",
                    "title": f"{tool.name}: {str(output)[:180]}",
                }
            ],
        }

    async def _list_custom_tools_autonomous(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.models.workflow import UserTool

        owner_id = _tool_owner(ctx)
        if owner_id is None:
            return {"error": "Cannot determine the owning user for this tool"}
        tools = (
            (await ctx.db.execute(select(UserTool).where(UserTool.user_id == owner_id)))
            .scalars()
            .all()
        )
        return {
            "success": True,
            "data": {
                "count": len(tools),
                "tools": [
                    {
                        "name": t.name,
                        "tool_type": t.tool_type,
                        "description": t.description,
                        "enabled": bool(t.is_enabled),
                        "parameters_schema": t.parameters_schema or {},
                    }
                    for t in tools
                ],
            },
        }

    async def _calibration_report(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        from app.services import agent_calibration_service as calibration

        try:
            report = await calibration.calibration_report(
                ctx.db,
                metric=str(params.get("metric") or "") or None,
                subject=str(params.get("subject") or "") or None,
                limit=min(int(params.get("limit", 50) or 50), 200),
            )
        except Exception as exc:
            return {
                "error": f"Could not read the calibration history: {str(exc)[:200]}"
            }

        return {"success": True, "data": report}

    return FunctionToolProvider(
        name="autonomous_workspace_mutation_tools",
        modes={"autonomous"},
        handlers={
            "calibration_report": _calibration_report,
            "execute_python": _execute_python,
            "create_custom_tool": _create_custom_tool,
            "run_custom_tool": _run_custom_tool_autonomous,
            "list_custom_tools": _list_custom_tools_autonomous,
            "compile_c_snippet": _compile_c_snippet,
            "analyze_snippet_cycles": _analyze_snippet_cycles,
            "retract_finding": _retract_finding,
            "verify_run_bundle": _verify_run_bundle,
            "record_prediction": _record_prediction,
            "record_measurement": _record_measurement,
            "benchmark_c_snippet": _benchmark_c_snippet,
            "check_implementation": _check_implementation,
            "compare_to_claim": _compare_to_claim,
            "execute_data_pipeline": _execute_data_pipeline,
            "write_and_run_script": _write_and_run_script,
            "write_file": _write_file,
            "apply_patch": _apply_patch,
            "run_command": _run_command,
            "run_repo_tests": _run_repo_tests,
            "propose_code_patch": _propose_code_patch,
            "create_workspace_checkpoint": _create_workspace_checkpoint,
            "restore_workspace_checkpoint": _restore_workspace_checkpoint,
            "hydrate_candidate_snapshot": _hydrate_candidate_snapshot,
            "persist_durable_workspace_checkpoint": (
                _persist_durable_workspace_checkpoint
            ),
            "restore_durable_workspace_checkpoint": (
                _restore_durable_workspace_checkpoint
            ),
        },
    )
