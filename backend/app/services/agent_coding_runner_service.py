"""Deterministic runner services extracted from AutonomousAgentExecutor."""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import re
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import UUID

from loguru import logger
from sqlalchemy import desc, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app.models.agent_job import AgentJob, AgentJobStatus, ChainTriggerCondition
from app.models.agent_definition import AgentDefinition
from app.models.agent_tool_prior import AgentToolPrior
from app.models.user import User
from app.models.memory import UserPreferences
from app.services.ai_hub_dataset_preset_service import ai_hub_dataset_preset_service
from app.services.ai_hub_eval_service import ai_hub_eval_service
from app.services.project_profile_service import (
    build_project_profile,
    format_project_profile_for_prompt,
    infer_project_profile_from_paths,
)
from app.services.research_opportunity_service import (
    collect_research_opportunity_linked_ids,
    compute_research_opportunity_evidence_revision,
    compute_research_portfolio_config_revision,
    list_normalized_research_opportunities,
    merge_operator_fields,
    normalize_research_opportunity,
    summarize_research_opportunity_autonomy_states,
    summarize_research_opportunity_stages,
)
from app.services.autonomy_service import (
    build_domain_profile_compat_policy,
    current_domain_profile_policy_snapshot,
    resolve_domain_profile_automation_contract,
)


class AgentCodingRunnerService:
    async def run_code_patch_proposer(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Code Agent (MVP): generate a unified diff proposal against a git code source.

        Expects:
          - job.config.source_id (UUID of a git DocumentSource; legacy target_source_id accepted)
          - optional job.config.search_query, file_paths[], max_files, max_chars_per_file

        Produces:
          - job.results.code_patch (summary + diff + metadata)
          - CodePatchProposal row + artifact reference in job.output_artifacts
        """
        from uuid import UUID as _UUID
        from app.models.document import Document, DocumentSource
        from app.models.code_patch_proposal import CodePatchProposal

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "code_patch_proposer", "result": details})

        source_id_raw = None
        if isinstance(job.config, dict):
            source_id_raw = (job.config or {}).get("source_id") or (job.config or {}).get("target_source_id")
        if not source_id_raw:
            inherited = (job.config or {}).get("inherited_data") if isinstance(job.config, dict) else None
            if isinstance(inherited, dict):
                parent_results = inherited.get("parent_results") if isinstance(inherited.get("parent_results"), dict) else None
                if parent_results and isinstance(parent_results.get("repo_ingest"), dict):
                    source_id_raw = parent_results["repo_ingest"].get("source_id")
        if not source_id_raw:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing job.config.source_id"
            await db.commit()
            return {"status": "failed", "error": job.error}
        try:
            source_uuid = _UUID(str(source_id_raw))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid source_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        source = await db.get(DocumentSource, source_uuid)
        if not source:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Target source not found"
            await db.commit()
            return {"status": "failed", "error": job.error}

        source_type = str(getattr(source, "source_type", "") or "").strip().lower()
        cfg = job.config if isinstance(job.config, dict) else {}
        failure_symptom = str(cfg.get("failure_symptom") or "").strip()
        error_output = str(cfg.get("error_output") or "").strip()
        scope = str(cfg.get("scope") or "auto").strip().lower() or "auto"
        emit_execution_plan = bool(cfg.get("emit_execution_plan", True))
        create_workspace_from_source = bool(cfg.get("create_workspace_from_source", True))
        max_verification_commands = max(1, min(int(cfg.get("max_verification_commands") or 3), 6))

        _emit(5, "planning", f"Preparing code patch proposal for source: {source.name}")
        await db.commit()

        inherited = cfg.get("inherited_data") if isinstance(cfg, dict) else None
        parent_results = inherited.get("parent_results") if isinstance(inherited, dict) else None

        search_query = str(cfg.get("search_query") or "").strip()
        file_paths = cfg.get("file_paths")
        if not isinstance(file_paths, list):
            file_paths = []
        file_paths = [str(p).strip() for p in file_paths if str(p).strip()]

        max_files = int(cfg.get("max_files") or 6)
        max_files = max(1, min(max_files, 20))
        max_chars_per_file = int(cfg.get("max_chars_per_file") or 8000)
        max_chars_per_file = max(1000, min(max_chars_per_file, 20000))

        workspace_meta: Dict[str, Any] = {
            "created": False,
            "workspace_id": None,
            "source_type": source_type,
            "file_count": 0,
            "base_path": None,
            "error": None,
        }
        if source_type in {"github", "gitlab"} and create_workspace_from_source:
            try:
                ws = await executor.workspace_manager.create_from_source(str(source.id), db)
                workspace_meta = {
                    "created": True,
                    "workspace_id": str(ws.workspace_id),
                    "source_type": source_type,
                    "file_count": len(ws.original_hashes or {}),
                    "base_path": str(ws.base_path),
                    "error": None,
                }
            except Exception as exc:
                workspace_meta["error"] = str(exc)

        # If we are refining after an ExperimentRunner step, default to the previously-touched files.
        if not file_paths and isinstance(parent_results, dict):
            prev_patch = parent_results.get("code_patch") if isinstance(parent_results.get("code_patch"), dict) else None
            prev_touched = prev_patch.get("files_touched") if isinstance(prev_patch, dict) and isinstance(prev_patch.get("files_touched"), list) else []
            file_paths = [str(p).strip() for p in prev_touched if str(p).strip()][:max_files]

        # Collect candidate documents (code files)
        docs: list[Document] = []
        if file_paths:
            # Match by title or source_identifier
            for p in file_paths[:max_files]:
                res = await db.execute(
                    select(Document)
                    .where(
                        Document.source_id == source.id,
                        or_(
                            Document.title == p,
                            Document.source_identifier == p,
                            Document.file_path == p,
                        ),
                    )
                    .limit(1)
                )
                d = res.scalar_one_or_none()
                if d:
                    docs.append(d)
        else:
            # Use search if provided; otherwise recent files
            if search_query:
                try:
                    results, _total, _took = await executor.search_service.search(
                        query=search_query,
                        mode="smart",
                        page=1,
                        page_size=max_files,
                        source_id=str(source.id),
                        db=db,
                    )
                    ids = [r.get("id") for r in (results or []) if isinstance(r, dict) and r.get("id")]
                    for doc_id in ids[:max_files]:
                        try:
                            d = await db.get(Document, _UUID(str(doc_id)))
                        except Exception:
                            d = None
                        if d and d.source_id == source.id:
                            docs.append(d)
                except Exception:
                    docs = []

            if not docs:
                res = await db.execute(
                    select(Document)
                    .where(Document.source_id == source.id)
                    .order_by(Document.updated_at.desc())
                    .limit(max_files)
                )
                docs = list(res.scalars().all())

        if not docs:
            job.status = AgentJobStatus.FAILED.value
            job.error = "No code documents found for the target source"
            await db.commit()
            return {"status": "failed", "error": job.error}

        # Build prompt context
        _emit(20, "collecting", f"Loaded {len(docs)} candidate files")
        await db.commit()

        file_blocks: list[str] = []
        file_meta: list[dict] = []
        for d in docs[:max_files]:
            content = (d.content or "")[:max_chars_per_file]
            file_id = str(d.id)
            path = d.title or d.source_identifier or d.file_path or file_id
            file_meta.append({"document_id": file_id, "path": path, "truncated": len(d.content or "") > len(content)})
            file_blocks.append(f"### FILE: {path}\n```text\n{content}\n```\n")

        inferred_project_profile: Dict[str, Any] = {}
        try:
            if workspace_meta.get("created"):
                ws_obj = executor.workspace_manager.get(str(workspace_meta.get("workspace_id")))
                if ws_obj:
                    inferred_project_profile = infer_project_profile_from_paths(list((ws_obj.original_hashes or {}).keys()))
            if not inferred_project_profile:
                inferred_project_profile = infer_project_profile_from_paths([row.get("path") for row in file_meta if isinstance(row, dict)])
        except Exception:
            inferred_project_profile = {}

        verification_commands: List[str] = []
        explicit_commands = cfg.get("commands") if isinstance(cfg.get("commands"), list) else []
        for raw in explicit_commands:
            cmd = str(raw or "").strip()
            if cmd and cmd not in verification_commands:
                verification_commands.append(cmd)
            if len(verification_commands) >= max_verification_commands:
                break
        if not verification_commands and bool(cfg.get("auto_commands_from_project_profile", True)):
            verification_commands = executor._select_verification_commands_from_profile(
                inferred_project_profile,
                max_commands=max_verification_commands,
            )
        bootstrap_and_fallback = executor._get_bootstrap_and_fallback_commands_from_profile(
            inferred_project_profile,
            primary_commands=verification_commands,
            max_install=max_verification_commands,
            max_fallback=max_verification_commands,
        )
        verification_plan = {
            "commands": verification_commands,
            "bootstrap_commands": bootstrap_and_fallback.get("install") or [],
            "fallback_commands": bootstrap_and_fallback.get("fallback") or [],
            "auto_inferred": not bool(explicit_commands) and bool(verification_commands),
        }
        execution_plan_steps = [
            {
                "step_id": "triage_context",
                "title": "Triage failure context",
                "status": "done",
                "objective": "Ground the reported symptom, scope, and likely files.",
            },
            {
                "step_id": "draft_patch",
                "title": "Draft minimal patch",
                "status": "in_progress",
                "objective": "Propose the smallest safe change against the candidate files.",
            },
        ]
        if verification_commands:
            execution_plan_steps.append(
                {
                    "step_id": "verify_patch",
                    "title": "Verify candidate patch",
                    "status": "pending",
                    "objective": "Run bounded verification commands and capture failures.",
                    "commands": verification_commands,
                }
            )
        if error_output:
            execution_plan_steps.append(
                {
                    "step_id": "refine_from_failure",
                    "title": "Refine from observed failure output",
                    "status": "pending",
                    "objective": "Use captured stack traces or logs to refine the patch proposal.",
                }
            )

        _emit(40, "drafting", "Generating patch proposal with LLM")
        await db.commit()

        user_settings = await executor._load_user_settings(job.user_id, db)
        refinement_context: list[str] = []
        previous_diff_excerpt: str | None = None
        if isinstance(parent_results, dict):
            prev_patch = parent_results.get("code_patch") if isinstance(parent_results.get("code_patch"), dict) else None
            prev_proposal_id = str(prev_patch.get("proposal_id") or "").strip() if isinstance(prev_patch, dict) else ""
            if prev_proposal_id:
                try:
                    prev_proposal = await db.get(CodePatchProposal, _UUID(prev_proposal_id))
                except Exception:
                    prev_proposal = None
                if prev_proposal and prev_proposal.user_id == job.user_id:
                    prev_summary = str(prev_proposal.summary or "").strip()
                    if prev_summary:
                        refinement_context.append(f"Previous patch summary:\n{prev_summary}")
                    prev_diff = str(prev_proposal.diff_unified or "").strip()
                    if prev_diff:
                        previous_diff_excerpt = "\n".join(prev_diff.splitlines()[:160])[:6000]

            exp = parent_results.get("experiment_run") if isinstance(parent_results.get("experiment_run"), dict) else None
            if isinstance(exp, dict):
                runs = exp.get("runs") if isinstance(exp.get("runs"), list) else []
                failures = [r for r in runs if isinstance(r, dict) and not bool(r.get("ok"))]
                if failures:
                    lines: list[str] = []
                    for r in failures[:4]:
                        cmd = str(r.get("command") or "")[:200]
                        code = r.get("exit_code")
                        stderr = str(r.get("stderr") or "")[:1200]
                        stdout = str(r.get("stdout") or "")[:800]
                        lines.append(f"- cmd: {cmd}\n  exit_code: {code}\n  stderr:\n{stderr}\n  stdout:\n{stdout}")
                    refinement_context.append("Experiment failures (most recent):\n" + "\n".join(lines))

            patch_apply = parent_results.get("code_patch_apply") if isinstance(parent_results.get("code_patch_apply"), dict) else None
            if isinstance(patch_apply, dict):
                errs = patch_apply.get("errors") if isinstance(patch_apply.get("errors"), list) else []
                if errs:
                    lines = []
                    for e in errs[:6]:
                        if not isinstance(e, dict):
                            continue
                        path = str(e.get("path") or "")
                        err = str(e.get("error") or e.get("message") or "")[:400]
                        if path:
                            lines.append(f"- {path}: {err}")
                        else:
                            lines.append(f"- {err}")
                    if lines:
                        refinement_context.append("Patch apply errors (most recent):\n" + "\n".join(lines))

        refinement_block = ""
        if refinement_context or previous_diff_excerpt:
            parts: list[str] = []
            parts.extend(refinement_context)
            if previous_diff_excerpt:
                parts.append(
                    "Previous diff excerpt (for reference; produce a complete diff against FILES below):\n" + previous_diff_excerpt
                )
            refinement_block = "\n\nREFINEMENT CONTEXT:\n" + "\n\n".join(parts) + "\n"

        prompt = (
            "You are a senior software engineer acting as a Code Agent.\n"
            "Task: produce a minimal, safe patch as a unified diff.\n\n"
            "Rules:\n"
            "- Output MUST be valid JSON only.\n"
            "- JSON keys: title, summary, diff_unified, files_touched, risks, tests_to_run.\n"
            "- diff_unified MUST be a standard unified diff starting with ---/+++ lines.\n"
            "- Only change files shown below; do not invent new file contents.\n"
            "- Keep patch small and focused.\n\n"
            f"GOAL:\n{(job.goal or '').strip()}\n\n"
            f"FAILURE SYMPTOM:\n{failure_symptom or 'Not provided'}\n\n"
            f"SCOPE:\n{scope}\n\n"
            f"ERROR OUTPUT:\n{error_output[:4000] if error_output else 'Not provided'}\n\n"
            f"VERIFICATION PLAN:\n{json.dumps(verification_plan, ensure_ascii=False)}\n\n"
            f"EXECUTION PLAN:\n{json.dumps(execution_plan_steps, ensure_ascii=False)}\n\n"
            f"{refinement_block}\n"
            f"FILES:\n{''.join(file_blocks)}\n"
        )

        response = await executor.llm_service.generate_response(
            query=prompt,
            context=None,
            temperature=0.2,
            max_tokens=2000,
            user_settings=user_settings,
            task_type="code_agent",
            user_id=job.user_id,
            db=db,
            routing=executor._llm_routing_from_job_config(job.config),
        )

        try:
            payload = json.loads(response)
        except Exception:
            payload = {"title": "Code Patch Proposal", "summary": response[:800], "diff_unified": "", "files_touched": [], "risks": [], "tests_to_run": []}

        title = str(payload.get("title") or "Code Patch Proposal")[:500]
        summary = str(payload.get("summary") or "").strip() or None
        diff_unified = str(payload.get("diff_unified") or "").strip()
        files_touched = payload.get("files_touched") if isinstance(payload.get("files_touched"), list) else []
        risks = payload.get("risks") if isinstance(payload.get("risks"), list) else []
        tests_to_run = payload.get("tests_to_run") if isinstance(payload.get("tests_to_run"), list) else []

        if not diff_unified or "---" not in diff_unified or "+++" not in diff_unified:
            job.status = AgentJobStatus.FAILED.value
            job.error = "LLM did not produce a valid unified diff"
            await db.commit()
            return {"status": "failed", "error": job.error, "raw": response[:2000]}

        proposal = CodePatchProposal(
            user_id=job.user_id,
            job_id=job.id,
            source_id=source.id,
            title=title,
            summary=summary,
            diff_unified=diff_unified,
            metadata={
                "goal": (job.goal or "").strip(),
                "source_id": str(source.id),
                "target_source_name": source.name,
                "scope": scope,
                "failure_symptom": failure_symptom,
                "error_output_excerpt": error_output[:2000] if error_output else "",
                "workspace": workspace_meta,
                "inferred_project_profile": inferred_project_profile,
                "verification_plan": verification_plan,
                "execution_plan": execution_plan_steps,
                "files_context": file_meta,
                "files_touched": files_touched,
                "risks": risks,
                "tests_to_run": tests_to_run,
            },
            status="proposed",
        )
        db.add(proposal)
        await db.commit()
        await db.refresh(proposal)

        job.results = dict(parent_results) if isinstance(parent_results, dict) else {}
        prev_cp = job.results.get("code_patch") if isinstance(job.results.get("code_patch"), dict) else None
        if isinstance(prev_cp, dict):
            existing = job.results.get("code_patches")
            if not isinstance(existing, list):
                existing = []
            existing.append(prev_cp)
            job.results["code_patches"] = existing[-5:]
        job.results["code_patch"] = {
            "proposal_id": str(proposal.id),
            "title": title,
            "summary": summary,
            "scope": scope,
            "failure_symptom": failure_symptom,
            "files_context": file_meta,
            "files_touched": files_touched,
            "risks": risks,
            "tests_to_run": tests_to_run,
        }
        job.results["code_patch_execution"] = {
            "mode": "repo_bug_triage_patch_proposal" if (cfg.get("launch_mode") == "quick_start_repo_bug_triage") else "code_patch_proposal",
            "source_id": str(source.id),
            "source_name": str(source.name or ""),
            "source_type": source_type,
            "scope": scope,
            "failure_symptom": failure_symptom,
            "error_output": error_output[:4000] if error_output else "",
            "workspace": workspace_meta,
            "inferred_project_profile": inferred_project_profile,
            "verification_plan": verification_plan,
            "execution_plan": execution_plan_steps if emit_execution_plan else [],
            "proposal_strategy": str(cfg.get("proposal_strategy") or "").strip(),
            "recovery": executor._build_code_patch_execution_recovery(
                job=job,
                experiment_run=(
                    parent_results.get("experiment_run")
                    if isinstance(parent_results, dict) and isinstance(parent_results.get("experiment_run"), dict)
                    else None
                ),
            ),
        }
        if job.output_artifacts is None:
            job.output_artifacts = []
        job.output_artifacts.append({"type": "code_patch_proposal", "id": str(proposal.id), "title": title})

        _emit(100, "completed", f"Patch proposal ready: {title}")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()

        if progress_callback:
            try:
                await progress_callback(
                    {
                        "job_id": str(job.id),
                        "progress": job.progress,
                        "phase": job.current_phase,
                        "status": job.status,
                        "iteration": job.iteration,
                        "phase_details": job.phase_details,
                        "error": job.error,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
            except Exception:
                pass

        return {"status": "completed", "results": job.results}

    async def run_code_patch_apply_to_kb(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: apply the latest CodePatchProposal (unified diff) to KnowledgeDB code documents.

        This is intentionally gated to avoid accidental writes.

        Expects:
          - inherited_data.parent_results.code_patch.proposal_id OR config.proposal_id
          - optional config.apply_patch_to_kb (bool, default False)
          - optional config.dry_run (bool, default True)
          - optional config.proposal_strategy: 'best_passing' | 'latest' | 'explicit' (default 'latest')
          - optional config.enabled_key: string (default 'apply_patch_to_kb')
          - optional config.require_experiments_ok (bool, default True)
          - optional config.fail_on_block (bool, default False)
          - optional config.require_dry_run_first (bool, default False)

        Produces:
          - job.results.code_patch_kb_apply
        """
        from hashlib import sha256 as _sha256
        from uuid import UUID as _UUID

        from sqlalchemy import and_ as _and, or_ as _or

        from app.models.code_patch_proposal import CodePatchProposal
        from app.models.document import Document
        from app.services.code_patch_apply_service import code_patch_apply_service, UnifiedDiffApplyError
        from app.services.document_service import DocumentService

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "code_patch_apply_to_kb", "result": details})

        cfg = job.config if isinstance(job.config, dict) else {}
        inherited = (cfg or {}).get("inherited_data") if isinstance(cfg, dict) else None
        parent_results = inherited.get("parent_results") if isinstance(inherited, dict) else None
        base_results = dict(parent_results) if isinstance(parent_results, dict) else {}

        enabled_key = str(cfg.get("enabled_key") or "apply_patch_to_kb").strip() or "apply_patch_to_kb"
        enabled = bool(cfg.get(enabled_key, False))
        dry_run = bool(cfg.get("dry_run", True))
        require_experiments_ok = bool(cfg.get("require_experiments_ok", True))
        fail_on_block = bool(cfg.get("fail_on_block", False))
        require_dry_run_first = bool(cfg.get("require_dry_run_first", False))

        if not enabled:
            job.results = dict(base_results)
            job.results["code_patch_kb_apply"] = {
                "enabled": False,
                "ok": None,
                "dry_run": dry_run,
                "did_apply": False,
                "proposal_strategy": str(cfg.get("proposal_strategy") or "latest").strip().lower() or "latest",
                "enabled_key": enabled_key,
                "note": "Skipped (apply_patch_to_kb=false).",
            }
            _emit(100, "completed", "Skipped (apply_patch_to_kb=false)")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        exp_runs = base_results.get("experiment_runs") if isinstance(base_results.get("experiment_runs"), list) else []
        exp_cur = base_results.get("experiment_run") if isinstance(base_results.get("experiment_run"), dict) else None
        exp_all = [r for r in exp_runs if isinstance(r, dict)]
        if isinstance(exp_cur, dict):
            exp_all.append(exp_cur)

        # If we are going to write, require a previous KB dry-run to have succeeded (in-chain).
        prev_kb_apply = base_results.get("code_patch_kb_apply") if isinstance(base_results.get("code_patch_kb_apply"), dict) else None
        if (not dry_run) and require_dry_run_first:
            prev_ok = bool(prev_kb_apply.get("ok")) if isinstance(prev_kb_apply, dict) else False
            prev_dry = bool(prev_kb_apply.get("dry_run")) if isinstance(prev_kb_apply, dict) else False
            if not (prev_ok and prev_dry):
                job.results = dict(base_results)
                job.results["code_patch_kb_apply"] = {
                    "enabled": True,
                    "ok": False,
                    "dry_run": dry_run,
                    "did_apply": False,
                    "blocked": True,
                    "blocked_reason": "Missing/failed prior dry-run (require_dry_run_first=true).",
                }
                _emit(100, "completed" if not fail_on_block else "failed", "Blocked (missing/failed dry-run)")
                job.status = AgentJobStatus.COMPLETED.value if not fail_on_block else AgentJobStatus.FAILED.value
                if fail_on_block:
                    job.error = "Blocked from applying patch to KB"
                job.completed_at = datetime.utcnow()
                await db.commit()
                return {"status": job.status, "results": job.results, "error": job.error}

        proposal_strategy = str(cfg.get("proposal_strategy") or "latest").strip().lower()
        if proposal_strategy not in {"best_passing", "latest", "explicit"}:
            proposal_strategy = "latest"

        job.results = dict(base_results)
        proposal_id = str(cfg.get("proposal_id") or "").strip()
        if proposal_strategy == "explicit" and not proposal_id:
            proposal_id = str(cfg.get("proposal_id") or "").strip()

        if proposal_strategy == "best_passing" and not proposal_id:
            for r in reversed(exp_all):
                if r.get("ok") is True:
                    pid = str(r.get("proposal_id") or "").strip()
                    if pid:
                        proposal_id = pid
                        break

        if proposal_strategy == "latest" and not proposal_id:
            # Prefer the latest proposal_id in inherited code_patch results.
            code_patch = base_results.get("code_patch") if isinstance(base_results.get("code_patch"), dict) else None
            proposal_id = str((code_patch or {}).get("proposal_id") or "").strip()
        if not proposal_id:
            # Fallback: last history entry.
            hist = base_results.get("code_patches") if isinstance(base_results.get("code_patches"), list) else []
            for p in reversed(hist):
                if isinstance(p, dict) and str(p.get("proposal_id") or "").strip():
                    proposal_id = str(p.get("proposal_id") or "").strip()
                    break

        if not proposal_id:
            job.results = dict(base_results)
            job.results["code_patch_kb_apply"] = {
                "enabled": True,
                "ok": False,
                "dry_run": dry_run,
                "did_apply": False,
                "proposal_strategy": proposal_strategy,
                "enabled_key": enabled_key,
                "error": "No proposal_id found in config or inherited results.",
            }
            _emit(100, "failed", "Missing proposal_id")
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing proposal_id"
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "failed", "results": job.results, "error": job.error}

        if require_experiments_ok:
            last_for_proposal = None
            for r in reversed(exp_all):
                if str(r.get("proposal_id") or "").strip() == proposal_id:
                    last_for_proposal = r
                    break
            ok_val = last_for_proposal.get("ok") if isinstance(last_for_proposal, dict) else None
            if ok_val is not True:
                job.results = dict(base_results)
                job.results["code_patch_kb_apply"] = {
                    "enabled": True,
                    "ok": False,
                    "dry_run": dry_run,
                    "did_apply": False,
                    "blocked": True,
                    "proposal_id": proposal_id,
                    "proposal_strategy": proposal_strategy,
                    "enabled_key": enabled_key,
                    "blocked_reason": "No passing experiment run found for selected proposal (require_experiments_ok=true).",
                }
                _emit(100, "completed" if not fail_on_block else "failed", "Blocked (experiments not passing)")
                job.status = AgentJobStatus.COMPLETED.value if not fail_on_block else AgentJobStatus.FAILED.value
                if fail_on_block:
                    job.error = "Blocked from applying patch to KB"
                job.completed_at = datetime.utcnow()
                await db.commit()
                return {"status": job.status, "results": job.results, "error": job.error}

        try:
            proposal_uuid = _UUID(proposal_id)
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid proposal_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        proposal = await db.get(CodePatchProposal, proposal_uuid)
        if not proposal or proposal.user_id != job.user_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Code patch proposal not found"
            await db.commit()
            return {"status": "failed", "error": job.error}
        if not proposal.source_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Proposal missing source_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        # Human-in-the-loop gate: do not allow autonomous jobs to directly write patches to the KB
        # unless explicitly enabled at deployment level.
        from app.core.config import settings as _settings
        if (not dry_run) and (not bool(getattr(_settings, "AGENT_KB_PATCH_APPLY_ENABLED", False))):
            try:
                from app.models.patch_pr import PatchPR

                pr = PatchPR(
                    user_id=job.user_id,
                    source_id=proposal.source_id,
                    title=f"PatchPR: {proposal.title}"[:500],
                    description=(proposal.summary or None),
                    status="draft",
                    selected_proposal_id=proposal.id,
                    proposal_ids=[str(proposal.id)],
                    approvals=[],
                    checks={
                        "created_by": "autonomous_agent_executor",
                        "job_id": str(job.id),
                        "note": "Direct KB apply blocked; use PatchPR merge after review.",
                    },
                )
                db.add(pr)
                await db.commit()
                await db.refresh(pr)

                job.results = dict(base_results)
                job.results["code_patch_kb_apply"] = {
                    "enabled": True,
                    "ok": False,
                    "dry_run": dry_run,
                    "did_apply": False,
                    "blocked": True,
                    "proposal_id": str(proposal.id),
                    "proposal_strategy": proposal_strategy,
                    "enabled_key": enabled_key,
                    "blocked_reason": "Direct KB patch apply is disabled (AGENT_KB_PATCH_APPLY_ENABLED=false).",
                    "patch_pr_id": str(pr.id),
                }
                _emit(100, "completed" if not fail_on_block else "failed", "Blocked (requires PatchPR review/merge)")
                job.status = AgentJobStatus.COMPLETED.value if not fail_on_block else AgentJobStatus.FAILED.value
                if fail_on_block:
                    job.error = "Blocked from applying patch to KB"
                job.completed_at = datetime.utcnow()
                await db.commit()
                return {"status": job.status, "results": job.results, "error": job.error}
            except Exception:
                job.results = dict(base_results)
                job.results["code_patch_kb_apply"] = {
                    "enabled": True,
                    "ok": False,
                    "dry_run": dry_run,
                    "did_apply": False,
                    "blocked": True,
                    "proposal_id": str(proposal.id),
                    "proposal_strategy": proposal_strategy,
                    "enabled_key": enabled_key,
                    "blocked_reason": "Direct KB patch apply is disabled (AGENT_KB_PATCH_APPLY_ENABLED=false).",
                }
                _emit(100, "completed" if not fail_on_block else "failed", "Blocked (requires PatchPR review/merge)")
                job.status = AgentJobStatus.COMPLETED.value if not fail_on_block else AgentJobStatus.FAILED.value
                if fail_on_block:
                    job.error = "Blocked from applying patch to KB"
                job.completed_at = datetime.utcnow()
                await db.commit()
                return {"status": job.status, "results": job.results, "error": job.error}

        _emit(10, "parsing", "Parsing unified diff")
        await db.commit()

        try:
            file_diffs = code_patch_apply_service.parse(proposal.diff_unified or "")
        except UnifiedDiffApplyError as exc:
            job.results = dict(base_results)
            job.results["code_patch_kb_apply"] = {
                "enabled": True,
                "applied": False,
                "dry_run": dry_run,
                "proposal_id": str(proposal.id),
                "error": f"Invalid diff: {exc}",
            }
            _emit(100, "failed", "Invalid diff")
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid diff"
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "failed", "results": job.results, "error": job.error}

        if not file_diffs:
            job.results = dict(base_results)
            job.results["code_patch_kb_apply"] = {
                "enabled": True,
                "applied": False,
                "dry_run": dry_run,
                "proposal_id": str(proposal.id),
                "error": "No file diffs found",
            }
            _emit(100, "failed", "No file diffs found")
            job.status = AgentJobStatus.FAILED.value
            job.error = "No file diffs found"
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "failed", "results": job.results, "error": job.error}

        service = DocumentService()
        applied: list[dict] = []
        errors: list[dict] = []

        _emit(30, "applying", f"Applying patch to {len(file_diffs)} file(s){' (dry-run)' if dry_run else ''}")
        await db.commit()

        for idx, fd in enumerate(file_diffs[:200]):
            path = (fd.path or "").strip()
            if not path:
                continue

            res = await db.execute(
                select(Document)
                .where(
                    _and(
                        Document.source_id == proposal.source_id,
                        _or(
                            Document.file_path == path,
                            Document.source_identifier == path,
                            Document.title == path,
                        ),
                    )
                )
                .limit(1)
            )
            doc = res.scalar_one_or_none()
            if not doc:
                errors.append({"path": path, "error": "Document not found"})
                continue

            try:
                new_text, debug = code_patch_apply_service.apply_to_text(doc.content or "", fd)
            except UnifiedDiffApplyError as exc:
                errors.append({"path": path, "error": str(exc)})
                continue

            if not dry_run:
                doc.content = new_text
                doc.content_hash = _sha256(new_text.encode("utf-8")).hexdigest()
                doc.is_processed = False
                doc.processing_error = None

                try:
                    await service.reprocess_document(doc.id, db, user_id=job.user_id)
                except Exception:
                    pass

            applied.append({"path": path, "document_id": str(doc.id), "debug": debug})
            _emit(30 + int(50 * (idx + 1) / max(1, min(len(file_diffs), 200))), "applying", f"Patched: {path}")
            await db.commit()

        ok = len(errors) == 0
        if not dry_run:
            proposal.proposal_metadata = proposal.proposal_metadata if isinstance(proposal.proposal_metadata, dict) else {}
            proposal.proposal_metadata["apply_results"] = {"applied": applied, "errors": errors, "dry_run": False}
            proposal.status = "applied" if ok else "proposed"

        job.results = dict(base_results)
        job.results["code_patch_kb_apply"] = {
            "enabled": True,
            "ok": ok,
            "dry_run": dry_run,
            "did_apply": ok and (not dry_run),
            "proposal_id": str(proposal.id),
            "proposal_strategy": proposal_strategy,
            "enabled_key": enabled_key,
            "source_id": str(proposal.source_id),
            "applied_files": applied,
            "errors": errors,
        }

        _emit(100, "completed" if ok else "failed", "KB patch apply complete")
        job.status = AgentJobStatus.COMPLETED.value if ok else AgentJobStatus.FAILED.value
        if not ok:
            job.error = "KB patch apply failed"
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": job.status, "results": job.results, "error": job.error}

    async def run_coding_backlog_orchestrator(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        from app.models.coding_backlog import CodingBacklogItem
        from app.models.code_patch_proposal import CodePatchProposal
        from app.models.document import DocumentSource
        from app.services.agent_job_templates import (
            REPO_BUG_TRIAGE_REPAIR_TEMPLATE_ID,
            get_builtin_agent_job_template,
        )
        from app.tasks.agent_job_tasks import execute_agent_job_task

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "coding_backlog_orchestrator", "result": details})

        def _normalize_ids(values: Any) -> list[str]:
            if not isinstance(values, list):
                return []
            return [str(v).strip() for v in values if str(v).strip()]

        def _continuation_chain_config(item_id: UUID, previous_child_kind: str) -> dict[str, Any]:
            return {
                "trigger_condition": "on_any_end",
                "inherit_results": True,
                "inherit_config": False,
                "child_jobs": [
                    {
                        "name": "Coding Backlog — Continue",
                        "job_type": "analysis",
                        "goal": "Continue backlog orchestration after a repair/apply child completes.",
                        "config": {
                            "deterministic_runner": "coding_backlog_orchestrator",
                            "coding_backlog_item_id": str(item_id),
                            "coding_backlog_previous_child_kind": str(previous_child_kind or "").strip().lower() or "repair",
                        },
                        "max_iterations": 1,
                        "max_tool_calls": 0,
                        "max_llm_calls": 0,
                        "max_runtime_minutes": 10,
                    }
                ],
            }

        def _attach_terminal_continuation(chain_config: Optional[dict], item_id: UUID, previous_child_kind: str) -> Optional[dict]:
            if not isinstance(chain_config, dict):
                return None
            updated = deepcopy(chain_config)
            cursor = updated
            while isinstance(cursor.get("child_jobs"), list) and cursor.get("child_jobs"):
                child = cursor["child_jobs"][-1]
                if not isinstance(child, dict):
                    break
                if not isinstance(child.get("chain_config"), dict):
                    child["chain_config"] = _continuation_chain_config(item_id, previous_child_kind)
                    return updated
                cursor = child["chain_config"]
            return updated

        def _policy(item: CodingBacklogItem) -> dict[str, Any]:
            raw = item.policy if isinstance(item.policy, dict) else {}
            blocked = raw.get("blocked_path_prefixes") if isinstance(raw.get("blocked_path_prefixes"), list) else []
            return {
                "max_files_touched": max(0, int(raw.get("max_files_touched", 3) or 3)),
                "blocked_path_prefixes": [str(v).strip() for v in blocked if str(v).strip()],
                "max_auto_retries": max(0, int(raw.get("max_auto_retries", 1) or 1)),
                "require_experiments_ok": bool(raw.get("require_experiments_ok", True)),
                "confidence_threshold": max(0.0, min(float(raw.get("confidence_threshold", 0.55) or 0.55), 1.0)),
            }

        def _derive_scope(file_paths: list[str], fallback_scope: str) -> str:
            if fallback_scope and fallback_scope != "auto":
                return fallback_scope
            joined = " ".join(file_paths).lower()
            if "frontend/" in joined or "/src/" in joined:
                return "frontend"
            if "worker" in joined:
                return "worker"
            if "backend/" in joined:
                return "backend"
            return "auto"

        def _normalize_slice(raw: Any, index: int) -> dict[str, Any]:
            src = raw if isinstance(raw, dict) else {}
            file_paths = _normalize_ids(src.get("file_paths"))
            commands = _normalize_ids(src.get("commands"))
            title = str(src.get("title") or "").strip() or (
                f"Target {os.path.basename(file_paths[0])}" if file_paths else "Triage reported failure"
            )
            scope = str(src.get("scope") or "").strip().lower() or _derive_scope(file_paths, str(item.scope or "auto").strip().lower() or "auto")
            return {
                "slice_id": str(src.get("slice_id") or f"slice_{index + 1}"),
                "title": title[:160],
                "status": str(src.get("status") or "pending").strip().lower() or "pending",
                "scope": scope,
                "file_paths": file_paths,
                "commands": commands,
                "search_query": str(src.get("search_query") or "").strip()[:500],
                "goal": str(src.get("goal") or "").strip()[:2000],
                "retry_count": max(0, int(src.get("retry_count", 0) or 0)),
                "selected_proposal_id": str(src.get("selected_proposal_id") or "").strip() or None,
                "promotion_decision": str(src.get("promotion_decision") or "").strip() or None,
                "blocked_reason": str(src.get("blocked_reason") or "").strip() or None,
                "child_job_id": str(src.get("child_job_id") or "").strip() or None,
                "apply_job_id": str(src.get("apply_job_id") or "").strip() or None,
                "proposal_confidence": float(src.get("proposal_confidence", 0.0) or 0.0),
                "files_touched": _normalize_ids(src.get("files_touched")),
                "started_at": src.get("started_at"),
                "completed_at": src.get("completed_at"),
                "status_reason": str(src.get("status_reason") or "").strip() or None,
                "timeline": src.get("timeline") if isinstance(src.get("timeline"), list) else [],
                "job_lineage": src.get("job_lineage") if isinstance(src.get("job_lineage"), dict) else {
                    "repair_job_ids": [],
                    "apply_job_ids": [],
                    "patch_pr_ids": [],
                    "proposal_ids": [],
                    "retry_from_job_ids": [],
                },
                "artifact_history": src.get("artifact_history") if isinstance(src.get("artifact_history"), list) else [],
                "manual_promotion_history": src.get("manual_promotion_history") if isinstance(src.get("manual_promotion_history"), list) else [],
            }

        def _normalize_decomposition(item: CodingBacklogItem) -> dict[str, Any]:
            raw = item.decomposition if isinstance(item.decomposition, dict) else {}
            planned_raw = raw.get("planned_slices") if isinstance(raw.get("planned_slices"), list) else raw.get("slices_planned") if isinstance(raw.get("slices_planned"), list) else []
            planned = [_normalize_slice(entry, idx) for idx, entry in enumerate(planned_raw)]
            completed_slices = [str(v).strip() for v in (raw.get("completed_slices") if isinstance(raw.get("completed_slices"), list) else []) if str(v).strip()]
            failed_slices = [str(v).strip() for v in (raw.get("failed_slices") if isinstance(raw.get("failed_slices"), list) else []) if str(v).strip()]
            promotion_decisions = [entry for entry in (raw.get("promotion_decisions") if isinstance(raw.get("promotion_decisions"), list) else []) if isinstance(entry, dict)]
            backlog_timeline = [entry for entry in (raw.get("backlog_timeline") if isinstance(raw.get("backlog_timeline"), list) else []) if isinstance(entry, dict)]
            active_slice_id = str(raw.get("active_slice_id") or "").strip() or None
            total = len(planned)
            auto_applied = sum(1 for entry in planned if str(entry.get("promotion_decision") or "").strip().lower() == "auto_applied")
            proposal_only = sum(1 for entry in planned if str(entry.get("promotion_decision") or "").strip().lower() in {"proposal_only", "patch_pr"})
            pending = sum(1 for entry in planned if str(entry.get("status") or "").strip().lower() in {"pending", "repairing", "retrying", "applying"})
            return {
                "strategy": str(raw.get("strategy") or "portfolio_goal").strip() or "portfolio_goal",
                "planned_slices": planned,
                "active_slice_id": active_slice_id,
                "completed_slices": completed_slices,
                "failed_slices": failed_slices,
                "promotion_decisions": promotion_decisions,
                "backlog_timeline": backlog_timeline,
                "lineage_summary": {
                    "repair_job_count": sum(len((entry.get("job_lineage") or {}).get("repair_job_ids") or []) for entry in planned),
                    "apply_job_count": sum(len((entry.get("job_lineage") or {}).get("apply_job_ids") or []) for entry in planned),
                    "patch_pr_count": sum(len((entry.get("job_lineage") or {}).get("patch_pr_ids") or []) for entry in planned),
                    "proposal_count": sum(len((entry.get("job_lineage") or {}).get("proposal_ids") or []) for entry in planned),
                    "operator_action_count": sum(len(entry.get("manual_promotion_history") or []) for entry in planned),
                },
                "portfolio_progress": {
                    "total_slices": total,
                    "pending_slices": pending,
                    "completed_slices": len(completed_slices),
                    "failed_slices": len(failed_slices),
                    "auto_applied_slices": auto_applied,
                    "proposal_only_slices": proposal_only,
                },
            }

        def _save_decomposition(dec: dict[str, Any]) -> None:
            item.decomposition = dec

        def _timeline_entry(*, actor: str, action: str, previous_status: Optional[str] = None, new_status: Optional[str] = None, note: Optional[str] = None, related_job_id: Optional[str] = None, related_proposal_id: Optional[str] = None, related_patch_pr_id: Optional[str] = None, metadata: Optional[dict[str, Any]] = None) -> dict[str, Any]:
            row = {
                "at": datetime.utcnow().isoformat(),
                "actor": actor,
                "action": action,
                "previous_status": previous_status,
                "new_status": new_status,
            }
            if note:
                row["note"] = note
            if related_job_id:
                row["job_id"] = related_job_id
            if related_proposal_id:
                row["proposal_id"] = related_proposal_id
            if related_patch_pr_id:
                row["patch_pr_id"] = related_patch_pr_id
            if metadata:
                row["metadata"] = metadata
            return row

        def _append_backlog_timeline(dec: dict[str, Any], entry: dict[str, Any]) -> None:
            rows = dec.get("backlog_timeline") if isinstance(dec.get("backlog_timeline"), list) else []
            rows.append(entry)
            dec["backlog_timeline"] = rows[-100:]

        def _append_slice_timeline(slice_state: dict[str, Any], entry: dict[str, Any]) -> None:
            rows = slice_state.get("timeline") if isinstance(slice_state.get("timeline"), list) else []
            rows.append(entry)
            slice_state["timeline"] = rows[-60:]

        def _append_lineage_id(slice_state: dict[str, Any], lineage_key: str, value: Optional[str]) -> None:
            lineage = slice_state.get("job_lineage") if isinstance(slice_state.get("job_lineage"), dict) else {}
            existing = lineage.get(lineage_key) if isinstance(lineage.get(lineage_key), list) else []
            lineage[lineage_key] = _append_unique(existing, value)
            slice_state["job_lineage"] = lineage

        def _append_artifact_history(slice_state: dict[str, Any], artifact_type: str, artifact_id: Optional[str], label: Optional[str] = None) -> None:
            if not artifact_id:
                return
            rows = slice_state.get("artifact_history") if isinstance(slice_state.get("artifact_history"), list) else []
            rows.append({"at": datetime.utcnow().isoformat(), "artifact_type": artifact_type, "artifact_id": artifact_id, "label": label or artifact_type})
            slice_state["artifact_history"] = rows[-40:]

        def _current_portfolio_progress() -> dict[str, Any]:
            return _normalize_decomposition(item).get("portfolio_progress") or {}

        def _find_slice(dec: dict[str, Any], slice_id: Optional[str]) -> Optional[dict[str, Any]]:
            sid = str(slice_id or "").strip()
            if not sid:
                return None
            for entry in dec.get("planned_slices") or []:
                if str(entry.get("slice_id") or "").strip() == sid:
                    return entry
            return None

        def _append_unique(values: list[str], value: Optional[str]) -> list[str]:
            next_values = [str(v).strip() for v in values if str(v).strip()]
            current = str(value or "").strip()
            if current and current not in next_values:
                next_values.append(current)
            return next_values

        def _upsert_promotion_decision(dec: dict[str, Any], entry: dict[str, Any]) -> None:
            slice_id = str(entry.get("slice_id") or "").strip()
            decisions = dec.get("promotion_decisions") if isinstance(dec.get("promotion_decisions"), list) else []
            kept = [row for row in decisions if str((row or {}).get("slice_id") or "").strip() != slice_id]
            kept.append(entry)
            dec["promotion_decisions"] = kept[-12:]

        def _next_pending_slice(dec: dict[str, Any]) -> Optional[dict[str, Any]]:
            for entry in dec.get("planned_slices") or []:
                if str(entry.get("status") or "").strip().lower() == "pending":
                    return entry
            return None

        def _build_slice_plan(item: CodingBacklogItem) -> list[dict[str, Any]]:
            raw_paths = _normalize_ids(item.file_paths)
            raw_commands = _normalize_ids(item.commands)
            if raw_paths:
                max_slices = min(3, max(1, len(raw_paths)))
                chunk_size = max(1, math.ceil(len(raw_paths) / max_slices))
                groups = [raw_paths[idx: idx + chunk_size] for idx in range(0, len(raw_paths), chunk_size)]
            else:
                groups = [[]]
            slices: list[dict[str, Any]] = []
            default_scope = str(item.scope or "auto").strip().lower() or "auto"
            symptom = str(item.failure_symptom or "").strip() or str(item.portfolio_goal or "").strip()[:4000]
            for idx, group in enumerate(groups[:3]):
                scope = _derive_scope(group, default_scope)
                search_terms = " ".join(part for part in [("" if scope == "auto" else scope), symptom, " ".join(group[:2])] if part).strip()[:500]
                title = (
                    f"Target {os.path.basename(group[0])}" + (f" +{len(group) - 1}" if len(group) > 1 else "")
                    if group
                    else "Triage reported failure"
                )
                slices.append(
                    {
                        "slice_id": f"slice_{idx + 1}",
                        "title": title,
                        "status": "pending",
                        "scope": scope,
                        "file_paths": group,
                        "commands": raw_commands,
                        "search_query": search_terms,
                        "goal": symptom[:2000],
                        "retry_count": 0,
                        "selected_proposal_id": None,
                        "promotion_decision": None,
                        "blocked_reason": None,
                        "child_job_id": None,
                        "apply_job_id": None,
                        "proposal_confidence": 0.0,
                        "files_touched": [],
                        "started_at": None,
                        "completed_at": None,
                        "status_reason": None,
                    }
                )
            return slices

        def _build_promotion_evaluation(
            *,
            code_patch: dict[str, Any],
            latest_results: dict[str, Any],
            pol: dict[str, Any],
        ) -> dict[str, Any]:
            touched = _normalize_ids(code_patch.get("files_touched"))
            risks = _normalize_ids(code_patch.get("risks"))
            experiment_run = latest_results.get("experiment_run") if isinstance(latest_results.get("experiment_run"), dict) else {}
            experiment_ok = experiment_run.get("ok")
            confidence = 0.45
            if experiment_ok is True:
                confidence += 0.25
            if len(touched) <= 2:
                confidence += 0.1
            if not risks:
                confidence += 0.1
            elif len(risks) >= 3:
                confidence -= 0.15
            confidence = max(0.0, min(1.0, confidence))
            blocked_prefixes = [str(v).strip() for v in pol.get("blocked_path_prefixes", []) if str(v).strip()]
            blocked_by_path = any(any(path.startswith(prefix) for prefix in blocked_prefixes) for path in touched) if blocked_prefixes else False
            blocked_by_count = bool(pol.get("max_files_touched")) and len(touched) > int(pol.get("max_files_touched") or 0)
            blocked_by_experiments = bool(pol.get("require_experiments_ok", True)) and experiment_ok is not True
            blocked_by_confidence = confidence < float(pol.get("confidence_threshold", 0.55) or 0.55)
            blocked_reason = None
            if blocked_by_path:
                blocked_reason = "blocked_path_prefix"
            elif blocked_by_count:
                blocked_reason = "max_files_touched_exceeded"
            elif blocked_by_experiments:
                blocked_reason = "experiments_not_verified"
            elif blocked_by_confidence:
                blocked_reason = "confidence_below_threshold"
            return {
                "eligible": not any([blocked_by_path, blocked_by_count, blocked_by_experiments, blocked_by_confidence]),
                "blocked_reason": blocked_reason,
                "files_touched": touched,
                "files_touched_count": len(touched),
                "proposal_confidence": confidence,
                "risk_count": len(risks),
                "experiment_ok": experiment_ok,
                "policy": deepcopy(pol),
            }

        async def _load_child_jobs(item: CodingBacklogItem) -> list[AgentJob]:
            ids = _normalize_ids(item.child_job_ids)
            if not ids:
                return []
            child_rows: list[AgentJob] = []
            for raw in ids:
                try:
                    child_uuid = UUID(raw)
                except Exception:
                    continue
                child = await db.get(AgentJob, child_uuid)
                if child and child.user_id == item.user_id:
                    child_rows.append(child)
            return child_rows

        cfg = job.config if isinstance(job.config, dict) else {}
        item_id_raw = str(cfg.get("coding_backlog_item_id") or "").strip()
        if not item_id_raw:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.coding_backlog_item_id"
            await db.commit()
            return {"status": "failed", "error": job.error}
        try:
            item_uuid = UUID(item_id_raw)
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid coding_backlog_item_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        item = await db.get(CodingBacklogItem, item_uuid)
        if not item or str(item.user_id) != str(job.user_id):
            job.status = AgentJobStatus.FAILED.value
            job.error = "Coding backlog item not found"
            await db.commit()
            return {"status": "failed", "error": job.error}

        _emit(5, "loading", f"Loaded backlog item {item.title}")
        await db.commit()

        inherited = cfg.get("inherited_data") if isinstance(cfg.get("inherited_data"), dict) else {}
        inherited_parent_results = inherited.get("parent_results") if isinstance(inherited.get("parent_results"), dict) else None
        inherited_child_kind = str(cfg.get("coding_backlog_previous_child_kind") or "").strip().lower()

        if str(item.status or "").lower() in {"paused", "cancelled"}:
            item.latest_summary = {
                "status": str(item.status or ""),
                "note": "Backlog item is paused or cancelled; no new child job was spawned.",
            }
            job.results = {"coding_backlog": item.latest_summary}
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        child_jobs = await _load_child_jobs(item)
        last_child = child_jobs[-1] if child_jobs else None
        pol = _policy(item)
        dec = _normalize_decomposition(item)
        if not dec.get("planned_slices"):
            dec["planned_slices"] = _build_slice_plan(item)
            dec = _normalize_decomposition(item)
            _save_decomposition(dec)

        async def _spawn_repair_child(slice_state: dict[str, Any], *, retry_from: Optional[AgentJob] = None) -> AgentJob:
            source = await db.get(DocumentSource, item.source_id) if item.source_id else None
            template = get_builtin_agent_job_template(REPO_BUG_TRIAGE_REPAIR_TEMPLATE_ID)
            if not template:
                raise RuntimeError("Repo bug triage template unavailable")
            symptom = str(item.failure_symptom or "").strip() or str(slice_state.get("goal") or "").strip() or str(item.portfolio_goal or "").strip()[:4000]
            scope = str(slice_state.get("scope") or item.scope or "auto").strip().lower() or "auto"
            slice_paths = _normalize_ids(slice_state.get("file_paths"))
            slice_commands = _normalize_ids(slice_state.get("commands"))
            search_query = str(slice_state.get("search_query") or "").strip() or " ".join(
                part for part in [("" if scope == "auto" else scope), symptom, " ".join(slice_paths[:2])] if part
            ).strip()[:500]
            merged_config = dict(template.default_config or {})
            merged_config.update(
                {
                    "source_id": str(item.source_id),
                    "launch_mode": "quick_start_repo_bug_triage",
                    "failure_symptom": symptom,
                    "scope": scope,
                    "search_query": search_query,
                    "quick_start": {
                        "profile": "repo_bug_triage",
                        "version": "v2",
                        "source_name": str(getattr(source, "name", "") or ""),
                        "source_type": str(getattr(source, "source_type", "") or "").strip().lower(),
                        "scope": scope,
                        "autonomy_mode": "patch_proposal",
                        "execution_depth": "workspace_planned",
                    },
                    "coding_backlog_item_id": str(item.id),
                    "coding_backlog_child_kind": "repair",
                    "coding_backlog_goal_type": "portfolio_goal",
                    "coding_backlog_slice_id": str(slice_state.get("slice_id") or ""),
                    "coding_backlog_slice_title": str(slice_state.get("title") or ""),
                }
            )
            if slice_paths:
                merged_config["file_paths"] = slice_paths
            elif isinstance(item.file_paths, list):
                merged_config["file_paths"] = _normalize_ids(item.file_paths)
            if slice_commands:
                merged_config["commands"] = slice_commands
            elif isinstance(item.commands, list):
                merged_config["commands"] = _normalize_ids(item.commands)
            if str(item.error_output or "").strip():
                merged_config["error_output"] = str(item.error_output).strip()
            if retry_from and isinstance(retry_from.results, dict):
                recovery = (
                    ((retry_from.results.get("code_patch_execution") or {}).get("recovery"))
                    if isinstance(retry_from.results.get("code_patch_execution"), dict)
                    else None
                )
                if isinstance(recovery, dict):
                    merged_config["coding_recovery"] = {
                        "strategy": "refined_retry",
                        "retry_reason": str(recovery.get("retry_reason") or "").strip() or None,
                        "resume_hint": str(recovery.get("resume_hint") or "").strip() or None,
                        "last_failed_commands": _normalize_ids(recovery.get("last_failed_commands")),
                        "suggested_operator_actions": _normalize_ids(recovery.get("suggested_operator_actions")),
                    }
                    latest_failed_output = str(recovery.get("latest_failed_output") or "").strip()
                    if latest_failed_output:
                        merged_config["error_output"] = latest_failed_output[:4000]
                    merged_config["relaunch_from_job_id"] = str(retry_from.id)
            chain_config = _attach_terminal_continuation(template.default_chain_config, item.id, "repair")
            repair_job = AgentJob(
                name=f"{str(item.title or '').strip()[:120]} — {str(slice_state.get('title') or 'Repair')[:80]}",
                description="Repair slice launched from coding backlog orchestrator.",
                job_type=template.job_type,
                goal=str(slice_state.get("goal") or item.portfolio_goal or "").strip()[:8000] or template.default_goal,
                config=merged_config,
                user_id=item.user_id,
                status=AgentJobStatus.PENDING.value,
                parent_job_id=job.id,
                root_job_id=job.root_job_id or job.id,
                chain_depth=(job.chain_depth or 0) + 1,
                chain_config=chain_config,
                max_iterations=template.default_max_iterations,
                max_tool_calls=template.default_max_tool_calls,
                max_llm_calls=template.default_max_llm_calls,
                max_runtime_minutes=template.default_max_runtime_minutes,
            )
            db.add(repair_job)
            await db.flush()
            item.child_job_ids = _normalize_ids(item.child_job_ids) + [str(repair_job.id)]
            item.current_job_id = repair_job.id
            slice_state["status"] = "retrying" if retry_from else "repairing"
            slice_state["retry_count"] = max(0, int(slice_state.get("retry_count", 0) or 0)) + (1 if retry_from else 0)
            slice_state["child_job_id"] = str(repair_job.id)
            slice_state["status_reason"] = "refined_retry" if retry_from else "planned_repair"
            slice_state["started_at"] = slice_state.get("started_at") or datetime.utcnow().isoformat()
            _append_slice_timeline(
                slice_state,
                _timeline_entry(
                    actor="system",
                    action="repair_job_started",
                    previous_status="retrying" if retry_from else "pending",
                    new_status=slice_state["status"],
                    related_job_id=str(repair_job.id),
                    metadata={"retry_from_job_id": str(retry_from.id) if retry_from else None},
                ),
            )
            _append_lineage_id(slice_state, "repair_job_ids", str(repair_job.id))
            if retry_from:
                _append_lineage_id(slice_state, "retry_from_job_ids", str(retry_from.id))
            dec["active_slice_id"] = str(slice_state.get("slice_id") or "")
            _append_backlog_timeline(
                dec,
                _timeline_entry(
                    actor="system",
                    action="repair_job_started",
                    previous_status=str(item.status or "").strip() or None,
                    new_status="running",
                    related_job_id=str(repair_job.id),
                    metadata={"slice_id": str(slice_state.get("slice_id") or "")},
                ),
            )
            _save_decomposition(dec)
            item.latest_summary = {
                "status": "repair_started",
                "current_child_job_id": str(repair_job.id),
                "retry_from_job_id": str(retry_from.id) if retry_from else None,
                "active_slice_id": str(slice_state.get("slice_id") or ""),
                "active_slice_title": str(slice_state.get("title") or ""),
                "portfolio_progress": _current_portfolio_progress(),
            }
            execute_agent_job_task.delay(str(repair_job.id), str(item.user_id))
            return repair_job

        async def _spawn_apply_child(*, parent_results: Dict[str, Any], proposal_id: str, slice_state: dict[str, Any], promotion_eval: dict[str, Any]) -> AgentJob:
            apply_job = AgentJob(
                name=f"{str(item.title or '').strip()[:120]} — Apply {str(slice_state.get('title') or 'Patch')[:74]}",
                description="Auto-apply repair outcome from coding backlog orchestrator.",
                job_type="analysis",
                goal="Apply the selected code patch proposal to the knowledge base.",
                config={
                    "deterministic_runner": "code_patch_apply_to_kb",
                    "proposal_id": proposal_id,
                    "proposal_strategy": "explicit",
                    "apply_patch_to_kb": True,
                    "dry_run": False,
                    "require_experiments_ok": True,
                    "require_dry_run_first": False,
                    "fail_on_block": True,
                    "coding_backlog_item_id": str(item.id),
                    "coding_backlog_child_kind": "apply",
                    "coding_backlog_slice_id": str(slice_state.get("slice_id") or ""),
                    "inherited_data": {"parent_results": parent_results},
                },
                user_id=item.user_id,
                status=AgentJobStatus.PENDING.value,
                parent_job_id=job.id,
                root_job_id=job.root_job_id or job.id,
                chain_depth=(job.chain_depth or 0) + 1,
                chain_config=_continuation_chain_config(item.id, "apply"),
                max_iterations=1,
                max_tool_calls=0,
                max_llm_calls=0,
                max_runtime_minutes=15,
            )
            db.add(apply_job)
            await db.flush()
            item.child_job_ids = _normalize_ids(item.child_job_ids) + [str(apply_job.id)]
            item.current_job_id = apply_job.id
            item.latest_apply_job_id = apply_job.id
            slice_state["status"] = "applying"
            slice_state["apply_job_id"] = str(apply_job.id)
            slice_state["promotion_decision"] = "auto_applied"
            slice_state["selected_proposal_id"] = proposal_id
            slice_state["proposal_confidence"] = float(promotion_eval.get("proposal_confidence", 0.0) or 0.0)
            slice_state["files_touched"] = _normalize_ids(promotion_eval.get("files_touched"))
            _append_slice_timeline(
                slice_state,
                _timeline_entry(
                    actor="system",
                    action="auto_apply_started",
                    previous_status="auto_applied",
                    new_status="applying",
                    related_job_id=str(apply_job.id),
                    related_proposal_id=proposal_id,
                ),
            )
            _append_lineage_id(slice_state, "apply_job_ids", str(apply_job.id))
            _append_lineage_id(slice_state, "proposal_ids", proposal_id)
            _append_artifact_history(slice_state, "proposal", proposal_id, "Selected proposal")
            dec["active_slice_id"] = str(slice_state.get("slice_id") or "")
            decision_row = {
                "slice_id": str(slice_state.get("slice_id") or ""),
                "title": str(slice_state.get("title") or ""),
                "decision": "auto_applied",
                "proposal_id": proposal_id,
                "job_id": str(slice_state.get("child_job_id") or ""),
                "apply_job_id": str(apply_job.id),
                "blocked_reason": None,
                "proposal_confidence": float(promotion_eval.get("proposal_confidence", 0.0) or 0.0),
            }
            _upsert_promotion_decision(dec, decision_row)
            _save_decomposition(dec)
            item.latest_summary = {
                "status": "apply_started",
                "current_child_job_id": str(apply_job.id),
                "selected_proposal_id": proposal_id,
                "promotion_decision": "auto_applied",
                "active_slice_id": str(slice_state.get("slice_id") or ""),
                "active_slice_title": str(slice_state.get("title") or ""),
                "promotion_evaluation": decision_row,
                "portfolio_progress": _current_portfolio_progress(),
            }
            execute_agent_job_task.delay(str(apply_job.id), str(item.user_id))
            return apply_job

        if last_child is None:
            next_slice = _next_pending_slice(dec)
            if not next_slice:
                item.latest_summary = {
                    "status": "completed",
                    "portfolio_progress": dec.get("portfolio_progress") or {},
                    "note": "No actionable slices were planned for this backlog item.",
                }
                item.status = "completed"
                item.completed_at = datetime.utcnow()
                job.results = {"coding_backlog": item.latest_summary}
                job.status = AgentJobStatus.COMPLETED.value
                job.completed_at = datetime.utcnow()
                await db.commit()
                return {"status": "completed", "results": job.results}
            _emit(25, "planning", f"Launching first repair slice: {str(next_slice.get('title') or '')}")
            await _spawn_repair_child(next_slice)
            item.status = "running"
            job.results = {"coding_backlog": item.latest_summary}
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        child_kind = str(((last_child.config or {}).get("coding_backlog_child_kind") or "repair")).strip().lower()
        last_status = str(last_child.status or "").strip().lower()
        latest_results = inherited_parent_results if isinstance(inherited_parent_results, dict) and inherited_child_kind in {"repair", "apply"} else (last_child.results if isinstance(last_child.results, dict) else {})
        effective_child_kind = inherited_child_kind or child_kind
        effective_status = last_status
        if inherited_child_kind == "repair" and isinstance(latest_results, dict):
            experiment_run = latest_results.get("experiment_run") if isinstance(latest_results.get("experiment_run"), dict) else {}
            if experiment_run.get("ok") is False:
                effective_status = AgentJobStatus.FAILED.value
            elif experiment_run.get("ok") is True:
                effective_status = AgentJobStatus.COMPLETED.value
        elif inherited_child_kind == "apply" and isinstance(latest_results, dict):
            kb_apply = latest_results.get("code_patch_kb_apply") if isinstance(latest_results.get("code_patch_kb_apply"), dict) else {}
            if kb_apply.get("ok") is False:
                effective_status = AgentJobStatus.FAILED.value
            elif kb_apply.get("ok") is True:
                effective_status = AgentJobStatus.COMPLETED.value
        item.current_job_id = last_child.id
        active_slice = _find_slice(dec, dec.get("active_slice_id")) or _find_slice(dec, (last_child.config or {}).get("coding_backlog_slice_id"))

        if last_status in {AgentJobStatus.PENDING.value, AgentJobStatus.RUNNING.value, AgentJobStatus.PAUSED.value}:
            item.latest_summary = {
                "status": "waiting_on_child",
                "current_child_job_id": str(last_child.id),
                "child_kind": child_kind,
                "child_status": last_status,
                "active_slice_id": str((active_slice or {}).get("slice_id") or ""),
                "active_slice_title": str((active_slice or {}).get("title") or ""),
                "portfolio_progress": dec.get("portfolio_progress") or {},
            }
            item.status = "running"
            job.results = {"coding_backlog": item.latest_summary}
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        if effective_child_kind == "apply":
            apply_ok = bool(((latest_results or {}).get("code_patch_kb_apply") or {}).get("ok"))
            if active_slice:
                active_slice["completed_at"] = datetime.utcnow().isoformat()
                if apply_ok:
                    active_slice["status"] = "auto_applied"
                    active_slice["status_reason"] = "auto_apply_complete"
                    _append_slice_timeline(
                        active_slice,
                        _timeline_entry(
                            actor="system",
                            action="auto_apply_completed",
                            previous_status="applying",
                            new_status="auto_applied",
                            related_job_id=str(last_child.id),
                            related_proposal_id=str(active_slice.get("selected_proposal_id") or "") or None,
                        ),
                    )
                    dec["completed_slices"] = _append_unique(dec.get("completed_slices") or [], active_slice.get("slice_id"))
                    dec["active_slice_id"] = None
                    _append_backlog_timeline(
                        dec,
                        _timeline_entry(
                            actor="system",
                            action="auto_apply_completed",
                            previous_status="running",
                            new_status="running",
                            related_job_id=str(last_child.id),
                            metadata={"slice_id": str(active_slice.get("slice_id") or "")},
                        ),
                    )
                    _save_decomposition(dec)
                    dec = _normalize_decomposition(item)
                    next_slice = _next_pending_slice(dec)
                    if next_slice:
                        _emit(88, "planning", f"Auto-apply complete; launching next slice {str(next_slice.get('title') or '')}")
                        await _spawn_repair_child(next_slice)
                        item.status = "running"
                        job.results = {"coding_backlog": item.latest_summary}
                        job.status = AgentJobStatus.COMPLETED.value
                        job.completed_at = datetime.utcnow()
                        await db.commit()
                        return {"status": "completed", "results": job.results}
            item.latest_summary = {
                "status": "completed" if apply_ok else "failed",
                "current_child_job_id": str(last_child.id),
                "promotion_decision": "auto_applied",
                "active_slice_id": str((active_slice or {}).get("slice_id") or ""),
                "active_slice_title": str((active_slice or {}).get("title") or ""),
                "portfolio_progress": _current_portfolio_progress(),
            }
            item.status = "completed" if apply_ok else "failed"
            item.completed_at = datetime.utcnow()
            job.results = {"coding_backlog": item.latest_summary}
            job.status = AgentJobStatus.COMPLETED.value if apply_ok else AgentJobStatus.FAILED.value
            if not apply_ok:
                job.error = "Auto-apply child failed"
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": job.status, "results": job.results, "error": job.error}

        retry_count = int((active_slice or {}).get("retry_count", 0) or 0)

        if effective_status == AgentJobStatus.FAILED.value and active_slice and retry_count < int(pol.get("max_auto_retries", 1) or 1):
            _emit(55, "recovery", "Repair failed; launching refined retry")
            active_slice["status"] = "retrying"
            active_slice["status_reason"] = "repair_failed"
            _save_decomposition(dec)
            await _spawn_repair_child(active_slice, retry_from=last_child)
            item.status = "running"
            job.results = {"coding_backlog": item.latest_summary}
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}
        if effective_status == AgentJobStatus.FAILED.value:
            if active_slice:
                active_slice["status"] = "failed"
                active_slice["status_reason"] = "repair_failed_after_retries"
                active_slice["completed_at"] = datetime.utcnow().isoformat()
                active_slice["awaiting_operator_action"] = True
                active_slice["allowed_slice_actions"] = ["relaunch_slice", "skip_slice"]
                active_slice["recommended_next_action"] = "relaunch_slice"
                _append_slice_timeline(
                    active_slice,
                    _timeline_entry(
                        actor="system",
                        action="repair_failed",
                        previous_status="retrying" if retry_count else "repairing",
                        new_status="failed",
                        related_job_id=str(last_child.id),
                    ),
                )
                dec["failed_slices"] = _append_unique(dec.get("failed_slices") or [], active_slice.get("slice_id"))
                dec["active_slice_id"] = None
                _append_backlog_timeline(
                    dec,
                    _timeline_entry(
                        actor="system",
                        action="repair_failed",
                        previous_status="running",
                        new_status="awaiting_operator",
                        related_job_id=str(last_child.id),
                        metadata={"slice_id": str(active_slice.get("slice_id") or "")},
                    ),
                )
                _save_decomposition(dec)
            item.latest_summary = {
                "status": "failed",
                "current_child_job_id": str(last_child.id),
                "promotion_decision": "proposal_only",
                "note": "Repair failed after automatic retries; operator review required.",
                "active_slice_id": str((active_slice or {}).get("slice_id") or ""),
                "active_slice_title": str((active_slice or {}).get("title") or ""),
                "portfolio_progress": _current_portfolio_progress(),
                "waiting_on_operator_action": bool(active_slice),
                "allowed_slice_actions": (active_slice or {}).get("allowed_slice_actions") or [],
                "recommended_next_action": (active_slice or {}).get("recommended_next_action"),
            }
            item.status = "awaiting_operator"
            item.completed_at = None
            job.results = {"coding_backlog": item.latest_summary}
            job.status = AgentJobStatus.FAILED.value
            job.error = "Repair child failed"
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "failed", "results": job.results, "error": job.error}

        code_patch = (latest_results or {}).get("code_patch") if isinstance((latest_results or {}).get("code_patch"), dict) else {}
        proposal_id = str(code_patch.get("proposal_id") or "").strip()
        if not proposal_id:
            item.latest_summary = {
                "status": "failed",
                "current_child_job_id": str(last_child.id),
                "note": "Completed repair child without a proposal.",
            }
            item.status = "failed"
            item.completed_at = datetime.utcnow()
            job.results = {"coding_backlog": item.latest_summary}
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing proposal from repair child"
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "failed", "results": job.results, "error": job.error}

        try:
            item.latest_proposal_id = UUID(proposal_id)
        except Exception:
            item.latest_proposal_id = None

        promotion_eval = _build_promotion_evaluation(
            code_patch=code_patch,
            latest_results=latest_results if isinstance(latest_results, dict) else {},
            pol=pol,
        )
        decision = "auto_applied" if (item.auto_apply_enabled and not item.require_patch_pr and promotion_eval.get("eligible")) else ("patch_pr" if item.require_patch_pr else "proposal_only")
        if active_slice:
            active_slice["selected_proposal_id"] = proposal_id
            active_slice["proposal_confidence"] = float(promotion_eval.get("proposal_confidence", 0.0) or 0.0)
            active_slice["files_touched"] = _normalize_ids(promotion_eval.get("files_touched"))
            active_slice["promotion_decision"] = decision
            active_slice["blocked_reason"] = str(promotion_eval.get("blocked_reason") or "").strip() or None
            _append_lineage_id(active_slice, "proposal_ids", proposal_id)
            _append_artifact_history(active_slice, "proposal", proposal_id, "Selected proposal")
        decision_row = {
            "slice_id": str((active_slice or {}).get("slice_id") or ""),
            "title": str((active_slice or {}).get("title") or ""),
            "decision": decision,
            "proposal_id": proposal_id,
            "job_id": str(last_child.id),
            "blocked_reason": str(promotion_eval.get("blocked_reason") or "").strip() or None,
            "proposal_confidence": float(promotion_eval.get("proposal_confidence", 0.0) or 0.0),
            "files_touched_count": int(promotion_eval.get("files_touched_count", 0) or 0),
        }
        _upsert_promotion_decision(dec, decision_row)
        _save_decomposition(dec)

        if decision == "auto_applied":
            _emit(80, "promotion", "Launching auto-apply child")
            await _spawn_apply_child(
                parent_results=latest_results if isinstance(latest_results, dict) else {},
                proposal_id=proposal_id,
                slice_state=active_slice or {},
                promotion_eval=promotion_eval,
            )
            item.status = "running"
            job.results = {"coding_backlog": item.latest_summary}
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        if active_slice:
            active_slice["status"] = decision
            active_slice["status_reason"] = "promotion_policy_blocked" if promotion_eval.get("blocked_reason") else decision
            active_slice["completed_at"] = datetime.utcnow().isoformat()
            active_slice["awaiting_operator_action"] = True
            active_slice["allowed_slice_actions"] = ["apply_override", "create_patch_pr", "keep_proposal_only", "relaunch_slice", "skip_slice"]
            active_slice["recommended_next_action"] = (
                "create_patch_pr"
                if item.require_patch_pr or str(promotion_eval.get("blocked_reason") or "").strip().lower() in {"blocked_path_prefix", "max_files_touched_exceeded"}
                else "apply_override"
                if str(promotion_eval.get("blocked_reason") or "").strip().lower() == "confidence_below_threshold"
                else "keep_proposal_only"
            )
            _append_slice_timeline(
                active_slice,
                _timeline_entry(
                    actor="system",
                    action="promotion_waiting_on_operator",
                    previous_status="repairing",
                    new_status=decision,
                    related_job_id=str(last_child.id),
                    related_proposal_id=proposal_id,
                    metadata={"blocked_reason": str(promotion_eval.get("blocked_reason") or "").strip() or None},
                ),
            )
            dec["active_slice_id"] = None
            _append_backlog_timeline(
                dec,
                _timeline_entry(
                    actor="system",
                    action="awaiting_operator",
                    previous_status="running",
                    new_status="awaiting_operator",
                    related_job_id=str(last_child.id),
                    related_proposal_id=proposal_id,
                    metadata={"slice_id": str(active_slice.get("slice_id") or ""), "decision": decision},
                ),
            )
            _save_decomposition(dec)
            dec = _normalize_decomposition(item)
        item.latest_summary = {
            "status": "awaiting_operator",
            "current_child_job_id": str(last_child.id),
            "selected_proposal_id": proposal_id,
            "promotion_decision": decision,
            "blocked_reason": str(promotion_eval.get("blocked_reason") or "").strip() or ("require_patch_pr" if item.require_patch_pr else None),
            "active_slice_id": str((active_slice or {}).get("slice_id") or ""),
            "active_slice_title": str((active_slice or {}).get("title") or ""),
            "promotion_evaluation": {
                **promotion_eval,
                "proposal_id": proposal_id,
                "decision": decision,
            },
            "portfolio_progress": dec.get("portfolio_progress") or {},
            "waiting_on_operator_action": bool(active_slice),
            "allowed_slice_actions": (active_slice or {}).get("allowed_slice_actions") or [],
            "recommended_next_action": (active_slice or {}).get("recommended_next_action"),
        }
        item.status = "awaiting_operator"
        item.completed_at = None
        job.results = {"coding_backlog": item.latest_summary}
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": "completed", "results": job.results}
