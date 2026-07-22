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
from app.services.llm_service import LLMService
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


class AgentExperimentRunnerService:
    async def run_experiment_plan_generate(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: create an ExperimentPlan from a ResearchNote (Hypothesis section).

        Expects config:
          - research_note_id (UUID) OR note_id
          - optional prefer_section: 'hypothesis'|'full_note' (default 'hypothesis')
          - optional max_note_chars (default 12000)
        """
        from uuid import UUID as _UUID

        from app.models.experiment import ExperimentPlan
        from app.models.research_note import ResearchNote

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "experiment_plan_generate", "result": details})

        def _extract_hypothesis_section(markdown: str) -> Optional[str]:
            if not markdown:
                return None
            lines = markdown.splitlines()
            heading_re = re.compile(r"^(#{1,6})\s+(Hypothesis|Hypotheses)\s*$", re.IGNORECASE)
            start_idx = None
            start_level = None
            for i, line in enumerate(lines):
                m = heading_re.match(line.strip())
                if m:
                    start_idx = i + 1
                    start_level = len(m.group(1))
                    break
            if start_idx is None:
                return None
            next_heading_re = re.compile(r"^(#{1,6})\s+.+\s*$")
            out: list[str] = []
            for j in range(start_idx, len(lines)):
                m2 = next_heading_re.match(lines[j].strip())
                if m2 and len(m2.group(1)) <= (start_level or 6):
                    break
                out.append(lines[j])
            text = "\n".join(out).strip()
            return text or None

        cfg = job.config if isinstance(job.config, dict) else {}
        note_id_raw = cfg.get("research_note_id") or cfg.get("note_id")
        if not note_id_raw:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.research_note_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        try:
            note_id = _UUID(str(note_id_raw))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid research_note_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        note = await db.get(ResearchNote, note_id)
        if not note or note.user_id != job.user_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Research note not found"
            await db.commit()
            return {"status": "failed", "error": job.error}

        prefer_section = str(cfg.get("prefer_section") or "hypothesis").strip().lower()
        if prefer_section not in {"hypothesis", "full_note"}:
            prefer_section = "hypothesis"

        max_note_chars = int(cfg.get("max_note_chars") or 12000)
        max_note_chars = max(500, min(max_note_chars, 60000))

        content = (note.content_markdown or "").strip()
        if prefer_section == "hypothesis":
            hypothesis_text = _extract_hypothesis_section(content) or content
        else:
            hypothesis_text = content
        hypothesis_text = (hypothesis_text or "").strip()
        if len(hypothesis_text) > max_note_chars:
            hypothesis_text = hypothesis_text[:max_note_chars]

        prompt = "\n\n".join(
            [
                "You are an AI research engineer. Create a runnable experiment plan from the hypothesis.",
                "Return ONLY valid JSON. No markdown, no commentary.",
                "JSON schema (high level): {"
                '\"hypothesis\": string, \"problem_statement\": string, \"success_criteria\": [string],'
                '\"datasets\": [{\"name\": string, \"source\": string, \"split\": string|null, \"notes\": string|null}],'
                '\"metrics\": [{\"name\": string, \"definition\": string, \"direction\": \"higher_better\"|\"lower_better\"}],'
                '\"baselines\": [{\"name\": string, \"details\": string}],'
                '\"method\": {\"summary\": string, \"key_components\": [string]},'
                '\"experiments\": [{\"name\": string, \"purpose\": string, \"variables\": [string], \"expected_outcome\": string}],'
                '\"ablations\": [{\"name\": string, \"remove_or_change\": string, \"expected_effect\": string}] | [],'
                '\"evaluation_protocol\": string,'
                '\"compute_budget\": {\"hardware\": string|null, \"time_estimate\": string|null, \"notes\": string|null},'
                '\"timeline\": [{\"week\": string, \"deliverable\": string}] | [],'
                '\"risks\": [{\"risk\": string, \"mitigation\": string}] | [],'
                '\"repro_checklist\": [string] | []'
                "}",
                f"Note title: {note.title}",
                "Hypothesis section:",
                hypothesis_text,
                "Rules:",
                "- Keep it concrete: include at least 3 experiments and 2 metrics.",
                "- Ensure the JSON is parseable.",
            ]
        )

        _emit(10, "planning", "Generating experiment plan JSON")
        llm = LLMService()
        raw = await llm.generate_response(
            query=prompt,
            max_tokens=1500,
            temperature=0.2,
            task_type="workflow_synthesis",
            user_id=job.user_id,
            db=db,
        )

        try:
            parsed = json.loads(raw) if isinstance(raw, str) else dict(raw)
            if not isinstance(parsed, dict):
                raise ValueError("Plan must be an object")
        except Exception:
            m = re.search(r"\{.*\}", str(raw), flags=re.DOTALL)
            if not m:
                job.status = AgentJobStatus.FAILED.value
                job.error = "Model did not return valid JSON"
                await db.commit()
                return {"status": "failed", "error": job.error}
            parsed = json.loads(m.group(0))

        plan = ExperimentPlan(
            user_id=job.user_id,
            research_note_id=note.id,
            title=f"Experiment Plan: {note.title}",
            hypothesis_text=hypothesis_text if prefer_section == "hypothesis" else None,
            plan=parsed,
            generator="llm",
            generator_details={"generated_at": datetime.utcnow().isoformat(), "via": "agent_job"},
        )
        db.add(plan)
        await db.commit()
        await db.refresh(plan)

        cfg["experiment_plan_id"] = str(plan.id)
        cfg["research_note_id"] = str(note.id)
        job.config = cfg

        job.results = job.results or {}
        job.results["experiment_plan"] = {
            "experiment_plan_id": str(plan.id),
            "research_note_id": str(note.id),
            "title": plan.title,
        }
        _emit(100, "completed", "Experiment plan created")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": "completed", "results": job.results}

    async def run_experiment_loop_seed(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: seed a configurable experiment loop by writing a nested chain_config onto this job.

        Expects config:
          - research_note_id (UUID)
          - source_id (UUID of git DocumentSource)
          - commands (list[str]) baseline commands
          - optional max_runs (int, default 3, max 20)
          - optional command_variants, use_llm_decider (used by experiment_decide_next)
        """
        cfg = job.config if isinstance(job.config, dict) else {}

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "experiment_loop_seed", "result": details})

        research_note_id = str(cfg.get("research_note_id") or cfg.get("note_id") or "").strip()
        source_id = str(cfg.get("source_id") or cfg.get("target_source_id") or "").strip()
        if not research_note_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.research_note_id"
            await db.commit()
            return {"status": "failed", "error": job.error}
        if not source_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.source_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        max_runs = int(cfg.get("max_runs") or cfg.get("max_experiment_runs") or 3)
        max_runs = max(1, min(max_runs, 20))

        inherit_results = bool(cfg.get("inherit_results", True))
        inherit_config = bool(cfg.get("inherit_config", True))
        append_to_note = bool(cfg.get("append_to_note", True))

        prefix = str(job.name or "Experiment Loop").strip()[:160]

        def _mk_child(name: str, runner: str, goal: str, config_extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            payload: Dict[str, Any] = {
                "name": f"{prefix} - {name}"[:200],
                "job_type": "analysis",
                "goal": goal,
                "config": {"deterministic_runner": runner},
            }
            if config_extra:
                payload["config"].update(config_extra)
            return payload

        nodes: list[tuple[Dict[str, Any], Optional[str]]] = []
        nodes.append(
            (
                _mk_child(
                    "Generate Plan",
                    "experiment_plan_generate",
                    f"Generate an experiment plan from research note {research_note_id}",
                ),
                "on_complete",
            )
        )

        for i in range(max_runs):
            human_i = i + 1
            nodes.append(
                (
                    _mk_child(
                        f"Decide Next ({human_i})",
                        "experiment_decide_next",
                        f"Decide next commands for research note {research_note_id}",
                    ),
                    "on_complete",
                )
            )
            nodes.append(
                (
                    _mk_child(
                        f"Run ({human_i})",
                        "experiment_runner",
                        f"Run experiment commands for research note {research_note_id}",
                    ),
                    "on_any_end",
                )
            )
            nodes.append(
                (
                    _mk_child(
                        f"Persist ({human_i})",
                        "experiment_persist_results",
                        f"Persist experiment results for research note {research_note_id}",
                        {"append_to_note": append_to_note},
                    ),
                    "on_complete" if i < max_runs - 1 else None,
                )
            )

        child: Optional[Dict[str, Any]] = None
        for node, trig in reversed(nodes):
            if child is not None and trig:
                node["chain_config"] = {
                    "trigger_condition": trig,
                    "inherit_results": inherit_results,
                    "inherit_config": inherit_config,
                    "child_jobs": [child],
                }
            child = node

        if child is None:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Failed to seed loop (no steps)"
            await db.commit()
            return {"status": "failed", "error": job.error}

        job.chain_config = {
            "trigger_condition": "on_complete",
            "inherit_results": inherit_results,
            "inherit_config": inherit_config,
            "child_jobs": [child],
        }

        job.results = job.results or {}
        job.results["experiment_loop_seed"] = {
            "max_runs": max_runs,
            "total_child_jobs": len(nodes),
            "append_to_note": append_to_note,
        }
        _emit(100, "completed", f"Seeded experiment loop ({max_runs} runs)")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": "completed", "results": job.results}

    async def run_experiment_decide_next(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: decide next experiment command variant + run name.

        Expects config:
          - commands (list[str]) baseline (optional)
          - command_variants: list[list[str]] OR list[{name, commands}] (optional)
          - use_llm_decider: bool (optional)

        Produces/updates config:
          - commands (list[str]) for next experiment_runner step
          - run_name (string)
          - experiment_iteration (int)
        """
        cfg = job.config if isinstance(job.config, dict) else {}

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "experiment_decide_next", "result": details})

        iteration = int(cfg.get("experiment_iteration") or 0)
        variants = cfg.get("command_variants") if isinstance(cfg.get("command_variants"), list) else []

        # If variants provided, pick by iteration index.
        chosen_commands: list[str] = []
        chosen_name: str | None = None
        if variants and iteration < len(variants):
            v = variants[iteration]
            if isinstance(v, dict):
                chosen_name = str(v.get("name") or "").strip() or None
                cmds = v.get("commands")
                if isinstance(cmds, list):
                    chosen_commands = [str(x).strip() for x in cmds if str(x).strip()]
            elif isinstance(v, list):
                chosen_commands = [str(x).strip() for x in v if str(x).strip()]

        # Optional LLM decider if enabled and no variant picked.
        use_llm = bool(cfg.get("use_llm_decider")) and not chosen_commands
        if use_llm:
            inherited = cfg.get("inherited_data") if isinstance(cfg.get("inherited_data"), dict) else {}
            parent_results = inherited.get("parent_results") if isinstance(inherited.get("parent_results"), dict) else {}
            plan = parent_results.get("experiment_plan") if isinstance(parent_results.get("experiment_plan"), dict) else {}
            last_run = parent_results.get("experiment_run") if isinstance(parent_results.get("experiment_run"), dict) else {}
            prompt = "\n\n".join(
                [
                    "You are an AI research engineer. Propose the next experiment command(s) to run.",
                    "Return ONLY JSON: {\"run_name\": string, \"commands\": [string], \"rationale\": string}.",
                    "Constraints: up to 3 commands, no destructive commands.",
                    f"Iteration: {iteration}",
                    f"Experiment plan summary: {json.dumps(plan, ensure_ascii=False)[:3000]}",
                    f"Last run results: {json.dumps(last_run, ensure_ascii=False)[:3000]}",
                ]
            )
            llm = LLMService()
            raw = await llm.generate_response(
                query=prompt,
                max_tokens=600,
                temperature=0.2,
                task_type="workflow_synthesis",
                user_id=job.user_id,
                db=db,
            )
            try:
                payload = json.loads(raw)
            except Exception:
                m = re.search(r"\{.*\}", str(raw), flags=re.DOTALL)
                payload = json.loads(m.group(0)) if m else {}
            if isinstance(payload, dict):
                rn = str(payload.get("run_name") or "").strip()
                cmds = payload.get("commands")
                if isinstance(cmds, list):
                    chosen_commands = [str(x).strip() for x in cmds if str(x).strip()]
                if rn:
                    chosen_name = rn

        # Fallback: keep existing commands from config
        if not chosen_commands:
            base_cmds = cfg.get("commands") if isinstance(cfg.get("commands"), list) else []
            chosen_commands = [str(x).strip() for x in base_cmds if str(x).strip()]

        chosen_commands = chosen_commands[:6]
        if not chosen_name:
            chosen_name = "Baseline" if iteration == 0 else f"Ablation {iteration}"

        cfg["commands"] = chosen_commands
        cfg["run_name"] = chosen_name
        cfg["experiment_iteration"] = iteration + 1
        job.config = cfg

        job.results = job.results or {}
        job.results["experiment_next"] = {"iteration": iteration, "run_name": chosen_name, "commands": chosen_commands}

        _emit(100, "completed", f"Next run: {chosen_name} ({len(chosen_commands)} cmd)")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": "completed", "results": job.results}

    async def run_experiment_persist_results(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: persist the last experiment_runner output into an ExperimentRun row and optionally append to the note.

        Expects:
          - inherited_data.parent_results.experiment_run
          - experiment_plan_id in inherited results or config
          - optional append_to_note (default True)
        """
        from uuid import UUID as _UUID

        from app.models.experiment import ExperimentPlan, ExperimentRun
        from app.models.research_note import ResearchNote

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "experiment_persist_results", "result": details})

        cfg = job.config if isinstance(job.config, dict) else {}
        inherited = cfg.get("inherited_data") if isinstance(cfg.get("inherited_data"), dict) else {}
        parent_results = inherited.get("parent_results") if isinstance(inherited.get("parent_results"), dict) else {}

        exp = parent_results.get("experiment_run") if isinstance(parent_results.get("experiment_run"), dict) else None
        if not exp:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing parent_results.experiment_run"
            await db.commit()
            return {"status": "failed", "error": job.error}

        plan_id_raw = (
            (parent_results.get("experiment_plan") or {}).get("experiment_plan_id")
            if isinstance(parent_results.get("experiment_plan"), dict)
            else None
        )
        plan_id_raw = plan_id_raw or cfg.get("experiment_plan_id")
        if not plan_id_raw:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing experiment_plan_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        try:
            plan_id = _UUID(str(plan_id_raw))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid experiment_plan_id"
            await db.commit()
            return {"status": "failed", "error": job.error}

        plan = await db.get(ExperimentPlan, plan_id)
        if not plan or plan.user_id != job.user_id:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Experiment plan not found"
            await db.commit()
            return {"status": "failed", "error": job.error}

        run_name = str(cfg.get("run_name") or "").strip() or str((parent_results.get("experiment_next") or {}).get("run_name") or "").strip()
        if not run_name:
            run_name = "Run"

        ok = exp.get("ok")
        status = "completed" if ok is True else ("cancelled" if ok is None else "failed")

        run = ExperimentRun(
            user_id=job.user_id,
            experiment_plan_id=plan.id,
            name=run_name,
            status=status,
            progress=100,
            started_at=datetime.utcnow(),
            completed_at=datetime.utcnow(),
            results=exp,
            summary=(str(exp.get("note") or "Experiment run")[:20000]),
        )
        db.add(run)
        await db.commit()
        await db.refresh(run)

        job.results = job.results or {}
        job.results["experiment_run_record"] = {"experiment_run_id": str(run.id), "status": status, "name": run_name}

        if bool(cfg.get("append_to_note", True)):
            note = await db.get(ResearchNote, plan.research_note_id)
            if note and note.user_id == job.user_id:
                marker = f"<!-- experiment_run:{run.id} -->"
                existing = note.content_markdown or ""
                if marker not in existing:
                    lines: list[str] = [
                        "## Experiment Results",
                        marker,
                        "",
                        f"Run: **{run.name}**",
                        f"Status: {run.status}",
                        f"Updated: {datetime.utcnow().isoformat()}",
                        "",
                    ]
                    cmds = exp.get("commands") if isinstance(exp.get("commands"), list) else []
                    if cmds:
                        lines.append("Commands:")
                        for c in cmds[:10]:
                            lines.append(f"- `{str(c)[:240]}`")
                        lines.append("")
                    rr = exp.get("runs") if isinstance(exp.get("runs"), list) else []
                    if rr:
                        lines.append("Results (first 10):")
                        for r2 in rr[:10]:
                            cmd = str(r2.get("command") or "")[:200]
                            exit_code = r2.get("exit_code")
                            ok2 = r2.get("ok")
                            dur = r2.get("duration_ms")
                            line = f"- `{cmd}`"
                            if isinstance(ok2, bool):
                                line += f" · ok={str(ok2).lower()}"
                            if exit_code is not None:
                                line += f" · exit={exit_code}"
                            if dur is not None:
                                line += f" · {dur}ms"
                            lines.append(line)
                        lines.append("")
                    note.content_markdown = existing.rstrip() + "\n\n" + "\n".join(lines).rstrip() + "\n"
                    await db.commit()

        # Stop criteria for chained loops:
        # - stop_on_ok: stop if the experiment succeeded
        # - stop_metric_*: stop if a parsed metric plateaus over a window
        stop_reason: str | None = None
        if bool(cfg.get("stop_on_ok")) and status == "completed":
            stop_reason = "stop_on_ok"

        metric_regex = str(cfg.get("stop_metric_regex") or "").strip()
        if not stop_reason and metric_regex:
            direction = str(cfg.get("stop_metric_direction") or "higher_better").strip().lower()
            if direction not in {"higher_better", "lower_better"}:
                direction = "higher_better"
            window = int(cfg.get("stop_metric_window") or 3)
            window = max(2, min(window, 10))
            min_improvement = float(cfg.get("stop_metric_min_improvement") or 0.0)

            def _extract_metric(exp_run: dict) -> float | None:
                rr = exp_run.get("runs") if isinstance(exp_run.get("runs"), list) else []
                if not rr:
                    return None
                try:
                    rx = re.compile(metric_regex)
                except Exception:
                    return None
                for r2 in rr:
                    if not isinstance(r2, dict):
                        continue
                    text = "\n".join([str(r2.get("stdout") or ""), str(r2.get("stderr") or "")])
                    m = rx.search(text)
                    if not m:
                        continue
                    val_raw = None
                    if "value" in getattr(m, "groupdict", lambda: {})():
                        val_raw = m.group("value")
                    else:
                        try:
                            val_raw = m.group(1)
                        except Exception:
                            val_raw = m.group(0)
                    try:
                        return float(str(val_raw).strip())
                    except Exception:
                        return None
                return None

            # Build a metric history from prior runs (if present) + current.
            prior_runs = parent_results.get("experiment_runs") if isinstance(parent_results.get("experiment_runs"), list) else []
            hist: list[float] = []
            for pr in prior_runs[-10:]:
                if isinstance(pr, dict):
                    v = _extract_metric(pr)
                    if v is not None:
                        hist.append(v)
            v_cur = _extract_metric(exp)
            if v_cur is not None:
                hist.append(v_cur)

            if len(hist) >= window:
                recent = hist[-window:]
                first, last = recent[0], recent[-1]
                improvement = (last - first) if direction == "higher_better" else (first - last)
                if improvement < min_improvement:
                    stop_reason = f"stop_metric_plateau:{direction}:Δ{improvement}"

        if stop_reason:
            job.chain_triggered = True
            job.results["experiment_loop_stop"] = {"reason": stop_reason, "at_run_id": str(run.id)}
            # Best-effort append an explicit stop marker for human visibility.
            if bool(cfg.get("append_to_note", True)):
                note = await db.get(ResearchNote, plan.research_note_id)
                if note and note.user_id == job.user_id:
                    marker2 = f"<!-- experiment_loop_stop:{run.id} -->"
                    existing2 = note.content_markdown or ""
                    if marker2 not in existing2:
                        note.content_markdown = (
                            existing2.rstrip()
                            + "\n\n"
                            + "\n".join(
                                [
                                    "### Experiment Loop Stop",
                                    marker2,
                                    "",
                                    f"Reason: `{stop_reason}`",
                                    f"At run: `{run.id}`",
                                    f"Updated: {datetime.utcnow().isoformat()}",
                                    "",
                                ]
                            ).rstrip()
                            + "\n"
                        )
                        await db.commit()

        _emit(100, "completed", f"Persisted run {run.id}")
        job.status = AgentJobStatus.COMPLETED.value
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": "completed", "results": job.results}

    async def run_experiment_runner(
        self,
        executor: Any,
        *,
        job: AgentJob,
        db: AsyncSession,
        progress_callback: Optional[callable],
    ) -> Dict[str, Any]:
        """
        Deterministic runner: run a small command-based experiment against a git DocumentSource.

        This is explicitly gated by the existing "unsafe code execution" feature flag/settings.

        Expects:
          - job.config.source_id (UUID of DocumentSource; legacy target_source_id accepted)
          - optional job.config.commands: list[str] (shell commands)
          - optional job.config.latex_project_id: UUID (append a Results subsection)
        """
        import asyncio as _asyncio
        import os as _os
        import subprocess as _subprocess
        import tempfile as _tempfile
        from pathlib import Path as _Path
        from uuid import UUID as _UUID

        from app.core.config import settings as app_settings
        from app.core.feature_flags import get_flag as get_feature_flag, get_str as get_feature_str
        from app.models.document import Document, DocumentSource
        from app.models.code_patch_proposal import CodePatchProposal
        from app.models.experiment import ExperimentRun
        from app.models.domain_research_profile import DomainResearchProfile
        from app.models.research_portfolio import ResearchPortfolio
        from app.models.latex_project import LatexProject
        from app.services.code_patch_apply_service import code_patch_apply_service, UnifiedDiffApplyError
        from app.services.scientific_validation_service import get_scientific_validation_runtime_limits

        def _emit(progress: int, phase: str, details: str):
            job.progress = max(0, min(100, int(progress)))
            job.current_phase = phase
            job.phase_details = details
            job.last_activity_at = datetime.utcnow()
            job.add_log_entry({"phase": phase, "action": "experiment_runner", "result": details})

        def _safe_relpath(p: str) -> str:
            p = (p or "").replace("\\", "/").strip()
            p = p.lstrip("/")
            while p.startswith("./"):
                p = p[2:]
            parts = [x for x in p.split("/") if x not in {"", ".", ".."}]
            safe = "/".join(parts)
            return safe[:240]

        async def _linked_run() -> Optional[ExperimentRun]:
            run_id_raw = str(cfg.get("experiment_run_id") or "").strip()
            if not run_id_raw:
                return None
            try:
                run_uuid = _UUID(run_id_raw)
            except Exception:
                return None
            run = await db.get(ExperimentRun, run_uuid)
            if run is None or run.user_id != job.user_id:
                return None
            return run

        async def _update_linked_run(
            *,
            status: Optional[str] = None,
            progress: Optional[int] = None,
            results: Optional[dict[str, Any]] = None,
            summary: Optional[str] = None,
            started: bool = False,
            completed: bool = False,
        ) -> None:
            run = await _linked_run()
            if run is None:
                return
            if status:
                run.status = status
            if progress is not None:
                run.progress = max(0, min(100, int(progress)))
            if results is not None:
                run.results = results
            if summary is not None:
                run.summary = str(summary)[:20000]
            if started and run.started_at is None:
                run.started_at = datetime.utcnow()
            if completed:
                run.completed_at = datetime.utcnow()
            await db.flush()

        def _insert_before_end_document(source: str, addition: str) -> str:
            marker = "\\end{document}"
            s = (source or "")
            idx = s.rfind(marker)
            if idx == -1:
                return (s.rstrip() + "\n\n" + addition.strip() + "\n").lstrip("\n")
            before = s[:idx].rstrip()
            after = s[idx:]
            return f"{before}\n\n{addition.strip()}\n\n{after}"

        cfg = job.config if isinstance(job.config, dict) else {}
        scientific_validation = (
            cfg.get("scientific_validation")
            if isinstance(cfg.get("scientific_validation"), dict)
            else {}
        )
        enabled_raw = cfg.get("enabled")
        if enabled_raw is None:
            enabled = bool(cfg.get("enable_experiments", True))
        else:
            enabled = bool(enabled_raw)

        if not enabled:
            inherited = (cfg or {}).get("inherited_data") if isinstance(cfg, dict) else None
            parent_results = inherited.get("parent_results") if isinstance(inherited, dict) else None
            job.results = dict(parent_results) if isinstance(parent_results, dict) else {}
            job.results["experiment_run"] = {
                "enabled": False,
                "ran": False,
                "commands": [],
                "note": "Skipped (enable_experiments=false).",
            }
            await _update_linked_run(
                status="blocked" if scientific_validation else "completed",
                progress=100,
                results=job.results.get("experiment_run"),
                summary="Skipped (experiments disabled)",
                completed=True,
            )
            _emit(100, "completed", "Skipped (experiments disabled)")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        source_id_raw = cfg.get("source_id") or cfg.get("target_source_id")
        latex_project_id_raw = cfg.get("latex_project_id")

        # Default commands from inherited code patch results.
        commands = cfg.get("commands") if isinstance(cfg.get("commands"), list) else None
        if not commands:
            inherited = (cfg or {}).get("inherited_data") if isinstance(cfg, dict) else None
            parent_results = inherited.get("parent_results") if isinstance(inherited, dict) else None
            code_patch = parent_results.get("code_patch") if isinstance(parent_results, dict) else None
            tests_to_run = code_patch.get("tests_to_run") if isinstance(code_patch, dict) and isinstance(code_patch.get("tests_to_run"), list) else []
            commands = [str(x) for x in tests_to_run if str(x).strip()]

        commands = [str(c).strip() for c in (commands or []) if str(c).strip()]
        commands = commands[:6]

        if not source_id_raw:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Missing config.source_id"
            await _update_linked_run(status="failed", progress=100, summary=job.error, completed=True)
            await db.commit()
            return {"status": "failed", "error": job.error}

        try:
            source_uuid = _UUID(str(source_id_raw))
        except Exception:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Invalid source_id"
            await _update_linked_run(status="failed", progress=100, summary=job.error, completed=True)
            await db.commit()
            return {"status": "failed", "error": job.error}

        source = await db.get(DocumentSource, source_uuid)
        if not source:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Source not found"
            await _update_linked_run(status="failed", progress=100, summary=job.error, completed=True)
            await db.commit()
            return {"status": "failed", "error": job.error}

        inherited = (cfg or {}).get("inherited_data") if isinstance(cfg, dict) else None
        parent_results = inherited.get("parent_results") if isinstance(inherited, dict) else None
        base_results = dict(parent_results) if isinstance(parent_results, dict) else {}

        enabled_override = await get_feature_flag("unsafe_code_execution_enabled")
        enabled_effective = bool(enabled_override) if enabled_override is not None else bool(getattr(app_settings, "ENABLE_UNSAFE_CODE_EXECUTION", False))
        if scientific_validation:
            await _update_linked_run(status="provisioning", progress=5, started=True)

        # Optional fallback: infer verification commands from source/project profile.
        auto_from_profile = bool(cfg.get("auto_commands_from_project_profile", True))
        auto_bootstrap_retry = bool(cfg.get("auto_bootstrap_retry", True))
        inferred_profile: Dict[str, Any] = {}
        if not commands and auto_from_profile:
            try:
                inferred_profile = await build_project_profile(
                    job,
                    db,
                    source_id=str(source.id),
                    max_files=int(cfg.get("auto_commands_profile_max_files") or 300),
                )
            except Exception:
                inferred_profile = {}
            commands = executor._select_verification_commands_from_profile(inferred_profile, max_commands=3)
        profile_retry = executor._get_bootstrap_and_fallback_commands_from_profile(
            inferred_profile,
            primary_commands=commands,
            max_install=int(cfg.get("auto_bootstrap_install_max_commands") or 3),
            max_fallback=int(cfg.get("auto_bootstrap_fallback_max_commands") or 3),
        )
        bootstrap_commands = (
            cfg.get("bootstrap_commands") if isinstance(cfg.get("bootstrap_commands"), list) else profile_retry.get("install")
        )
        fallback_commands = (
            cfg.get("fallback_commands") if isinstance(cfg.get("fallback_commands"), list) else profile_retry.get("fallback")
        )
        bootstrap_commands = [str(cmd).strip() for cmd in (bootstrap_commands or []) if str(cmd).strip()][:6]
        fallback_commands = [str(cmd).strip() for cmd in (fallback_commands or []) if str(cmd).strip()][:6]

        if not commands:
            job.results = dict(base_results)
            job.results["experiment_run"] = {
                "source_id": str(source.id),
                "source_name": source.name,
                "enabled": enabled_effective,
                "ran": False,
                "commands": [],
                "note": "No commands provided (and no inherited tests_to_run).",
            }
            if inferred_profile:
                job.results["experiment_run"]["inferred_project_profile"] = inferred_profile
            await _update_linked_run(
                status="blocked" if scientific_validation else "completed",
                progress=100,
                results=job.results.get("experiment_run"),
                summary="No commands provided (and no inherited tests_to_run).",
                started=True,
                completed=True,
            )
            _emit(100, "completed", "No experiment commands to run")
            job.status = AgentJobStatus.COMPLETED.value
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "completed", "results": job.results}

        backend_override = await get_feature_str("unsafe_code_exec_backend")
        backend_effective = str(
            cfg.get("unsafe_code_exec_backend")
            or cfg.get("execution_backend")
            or backend_override
            or getattr(app_settings, "UNSAFE_CODE_EXEC_BACKEND", "subprocess")
            or "subprocess"
        ).strip().lower()
        if backend_effective not in {"subprocess", "docker"}:
            backend_effective = "subprocess"
        image_override = await get_feature_str("unsafe_code_exec_docker_image")
        image_effective = str(
            cfg.get("unsafe_code_exec_docker_image")
            or cfg.get("docker_image")
            or image_override
            or getattr(app_settings, "UNSAFE_CODE_EXEC_DOCKER_IMAGE", "python:3.11-slim")
            or "python:3.11-slim"
        ).strip()

        timeout_seconds = int(cfg.get("timeout_seconds") or getattr(app_settings, "UNSAFE_CODE_EXEC_TIMEOUT_SECONDS", 10))
        timeout_seconds = max(2, min(timeout_seconds, 1800))
        stdout_cap = int(getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_STDOUT_CHARS", 20000))
        stderr_cap = int(getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_STDERR_CHARS", 20000))
        runtime_limits = get_scientific_validation_runtime_limits()

        if scientific_validation:
            profile_snapshot = (
                scientific_validation.get("profile_snapshot")
                if isinstance(scientific_validation.get("profile_snapshot"), dict)
                else {}
            )
            recipe_snapshot = (
                scientific_validation.get("recipe_snapshot")
                if isinstance(scientific_validation.get("recipe_snapshot"), dict)
                else {}
            )
            capability_check = (
                scientific_validation.get("capability_check")
                if isinstance(scientific_validation.get("capability_check"), dict)
                else {}
            )
            policy_block_reason = ""
            expected_backend = str(profile_snapshot.get("backend") or cfg.get("unsafe_code_exec_backend") or "").strip().lower()
            expected_image = str(profile_snapshot.get("docker_image") or cfg.get("unsafe_code_exec_docker_image") or "").strip()
            expected_commands = recipe_snapshot.get("commands") if isinstance(recipe_snapshot.get("commands"), list) else []
            expected_benchmark_family = str(recipe_snapshot.get("benchmark_family") or "").strip()
            allowed_benchmark_families = set(
                str(item).strip() for item in (profile_snapshot.get("allowed_benchmark_families") or []) if str(item).strip()
            )
            allowed_perf_collectors = set(
                str(item).strip() for item in (profile_snapshot.get("allowed_perf_collectors") or []) if str(item).strip()
            )
            recipe_perf_collectors = set(
                str(item).strip() for item in (recipe_snapshot.get("allowed_perf_collectors") or []) if str(item).strip()
            )

            if backend_effective not in {"docker", "subprocess"}:
                policy_block_reason = "unsupported_backend"
            elif expected_backend and backend_effective != expected_backend:
                policy_block_reason = "recipe_profile_mismatch"
            elif backend_effective == "docker" and image_effective not in set(runtime_limits["allowed_docker_images"]):
                policy_block_reason = "disallowed_image"
            elif expected_image and backend_effective == "docker" and image_effective != expected_image:
                policy_block_reason = "recipe_profile_mismatch"
            elif expected_benchmark_family and expected_benchmark_family not in allowed_benchmark_families:
                policy_block_reason = "unsupported_benchmark_family"
            elif recipe_perf_collectors and not recipe_perf_collectors.issubset(allowed_perf_collectors):
                policy_block_reason = "recipe_profile_mismatch"
            elif capability_check and not bool(capability_check.get("ok")):
                policy_block_reason = "missing_capability"
            elif expected_commands and commands != expected_commands:
                policy_block_reason = "recipe_profile_mismatch"
            elif timeout_seconds > int(runtime_limits["max_timeout_seconds"]):
                policy_block_reason = "policy_limit_exceeded"

            if policy_block_reason:
                job.results = dict(base_results)
                job.results["experiment_run"] = {
                    "source_id": str(source.id),
                    "source_name": source.name,
                    "enabled": enabled_effective,
                    "ran": False,
                    "commands": commands,
                    "note": f"Scientific validation blocked: {policy_block_reason}.",
                }
                job.results["scientific_validation"] = {
                    "validation_kind": "scientific_validation",
                    "recipe_family": str(scientific_validation.get("recipe_family") or ""),
                    "recipe_id": str(scientific_validation.get("recipe_id") or ""),
                    "recipe_version": int(scientific_validation.get("recipe_version") or 1),
                    "sandbox_profile_id": str(scientific_validation.get("sandbox_profile_id") or ""),
                    "blocked_reason_code": policy_block_reason,
                    "capability_check": capability_check,
                    "profile_snapshot": profile_snapshot,
                    "recipe_snapshot": recipe_snapshot,
                    "status": "blocked",
                }
                await _update_linked_run(
                    status="blocked",
                    progress=100,
                    results=job.results.get("experiment_run"),
                    summary=f"Scientific validation blocked: {policy_block_reason}",
                    started=True,
                    completed=True,
                )
                _emit(100, "completed", f"Scientific validation blocked: {policy_block_reason}")
                job.status = AgentJobStatus.COMPLETED.value
                job.completed_at = datetime.utcnow()
                await db.commit()
                return {"status": "completed", "results": job.results}

        _emit(10, "loading", f"Loading files for source {source.name}")
        await db.commit()

        res = await db.execute(select(Document).where(Document.source_id == source.id))
        docs = list(res.scalars().all())
        if not docs:
            job.status = AgentJobStatus.FAILED.value
            job.error = "Source has no documents"
            await _update_linked_run(status="failed", progress=100, summary=job.error, completed=True)
            await db.commit()
            return {"status": "failed", "error": job.error}

        docs_by_path: dict[str, Document] = {}
        for d in docs[:2000]:
            p = _safe_relpath(d.file_path or d.source_identifier or d.title or "")
            if p and p not in docs_by_path:
                docs_by_path[p] = d

        files_list: list[dict] = []
        for d in docs[:400]:
            path = _safe_relpath(d.file_path or d.source_identifier or d.title or "")
            if not path:
                continue
            content = (d.content or "")
            if len(content) > 50000:
                content = content[:50000]
            files_list.append({"path": path, "content": content})
            if len(files_list) >= 120:
                break

        patch_apply: dict = {"proposal_id": None, "applied": [], "errors": []}
        code_patch = base_results.get("code_patch") if isinstance(base_results.get("code_patch"), dict) else None
        proposal_id = str((cfg or {}).get("code_patch_proposal_id") or (code_patch or {}).get("proposal_id") or "").strip()
        if proposal_id:
            patch_apply["proposal_id"] = proposal_id
            try:
                proposal_uuid = _UUID(proposal_id)
            except Exception:
                proposal_uuid = None
            proposal = await db.get(CodePatchProposal, proposal_uuid) if proposal_uuid else None
            if proposal and proposal.user_id == job.user_id:
                try:
                    file_diffs = code_patch_apply_service.parse(proposal.diff_unified or "")
                except UnifiedDiffApplyError as exc:
                    patch_apply["errors"].append({"error": f"Invalid diff: {exc}"})
                    file_diffs = []

                if file_diffs:
                    files_by_path: dict[str, int] = {}
                    for idx, ff in enumerate(files_list):
                        p = _safe_relpath(str(ff.get("path") or ""))
                        if p and p not in files_by_path:
                            files_by_path[p] = idx

                    for fd in file_diffs:
                        p = _safe_relpath(fd.path or "")
                        if not p or p in files_by_path:
                            continue
                        d = docs_by_path.get(p)
                        if not d:
                            patch_apply["errors"].append({"path": p, "error": "Document not found for patch path"})
                            continue
                        content = (d.content or "")
                        if len(content) > 50000:
                            content = content[:50000]
                        files_by_path[p] = len(files_list)
                        files_list.append({"path": p, "content": content})

                    for fd in file_diffs:
                        p = _safe_relpath(fd.path or "")
                        if not p:
                            continue
                        idx = files_by_path.get(p)
                        if idx is None:
                            patch_apply["errors"].append({"path": p, "error": "Missing file content for patch"})
                            continue
                        try:
                            new_text, debug = code_patch_apply_service.apply_to_text(str(files_list[idx].get("content") or ""), fd)
                        except UnifiedDiffApplyError as exc:
                            patch_apply["errors"].append({"path": p, "error": str(exc)})
                            continue
                        files_list[idx]["content"] = new_text
                        patch_apply["applied"].append({"path": p, "debug": debug})

        if proposal_id and patch_apply.get("errors"):
            job.results = dict(base_results)
            job.results["code_patch_apply"] = patch_apply
            await _update_linked_run(status="failed", progress=100, results=patch_apply, summary="Patch apply failed", started=True, completed=True)
            _emit(100, "failed", "Failed to apply patch before experiments")
            job.status = AgentJobStatus.FAILED.value
            job.error = "Patch apply failed"
            job.completed_at = datetime.utcnow()
            await db.commit()
            return {"status": "failed", "results": job.results, "error": job.error}

        behavior: Dict[str, Any] = {"enabled": enabled_effective, "backend": backend_effective, "ran": False}
        if inferred_profile:
            behavior["inferred_project_profile"] = inferred_profile
        if bootstrap_commands:
            behavior["bootstrap_commands"] = bootstrap_commands
        if fallback_commands:
            behavior["fallback_commands"] = fallback_commands
        runs: list[dict] = []

        if not enabled_effective:
            behavior["ran"] = False
            behavior["skipped_reason"] = "Server disabled unsafe code execution (unsafe_code_execution_enabled=false)"
        else:
            _emit(40, "running", f"Running {len(commands)} command(s) (unsafe)")
            await db.commit()
            with _tempfile.TemporaryDirectory(prefix="exp_runner_") as tmp:
                tmp_path = _Path(tmp)
                for f in files_list:
                    p = _safe_relpath(str(f.get("path") or ""))
                    if not p:
                        continue
                    out = tmp_path / p
                    out.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        out.write_text(str(f.get("content") or ""), encoding="utf-8")
                    except Exception:
                        continue

                env = dict(_os.environ or {})
                env.setdefault("PYTHONNOUSERSITE", "1")
                env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
                env.setdefault("PYTHONHASHSEED", "0")
                env.setdefault("HOME", tmp)

                def _limit_resources():
                    try:
                        import resource

                        cpu = int(max(1, min(timeout_seconds + 1, 7200)))
                        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
                        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
                        resource.setrlimit(resource.RLIMIT_FSIZE, (20 * 1024 * 1024, 20 * 1024 * 1024))
                        resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
                        mem_mb = int(cfg.get("unsafe_code_exec_max_memory_mb") or getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_MEMORY_MB", 512))
                        mem = max(128, min(mem_mb, 8192)) * 1024 * 1024
                        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
                    except Exception:
                        return

                async def _run_command_batch(batch: List[str], *, phase: str, progress_start: int, progress_end: int) -> List[dict]:
                    batch_runs: List[dict] = []
                    if not batch:
                        return batch_runs
                    for i, cmd in enumerate(batch):
                        start = datetime.utcnow()
                        rec = {
                            "command": cmd,
                            "phase": phase,
                            "ok": False,
                            "exit_code": None,
                            "stdout": "",
                            "stderr": "",
                            "duration_ms": 0,
                        }
                        try:
                            if backend_effective == "docker":
                                mem_mb = int(cfg.get("unsafe_code_exec_max_memory_mb") or getattr(app_settings, "UNSAFE_CODE_EXEC_MAX_MEMORY_MB", 512))
                                cpus = float(cfg.get("unsafe_code_exec_docker_cpus") or getattr(app_settings, "UNSAFE_CODE_EXEC_DOCKER_CPUS", 1.0) or 1.0)
                                pids = int(cfg.get("unsafe_code_exec_docker_pids_limit") or getattr(app_settings, "UNSAFE_CODE_EXEC_DOCKER_PIDS_LIMIT", 128))
                                command = [
                                    "docker",
                                    "run",
                                    "--rm",
                                    "--network",
                                    "none",
                                    "--cap-drop",
                                    "ALL",
                                    "--security-opt",
                                    "no-new-privileges",
                                    "--pids-limit",
                                    str(max(32, min(pids, 1024))),
                                    "--memory",
                                    f"{max(128, min(mem_mb, 8192))}m",
                                    "--cpus",
                                    str(max(0.25, min(cpus, 8.0))),
                                    "--user",
                                    "65534:65534",
                                    "-v",
                                    f"{tmp}:/work:rw",
                                    "-w",
                                    "/work",
                                    image_effective,
                                    "/bin/sh",
                                    "-lc",
                                    cmd,
                                ]
                                run_kwargs = {
                                    "cwd": str(tmp_path),
                                    "env": env,
                                    "capture_output": True,
                                    "text": True,
                                    "timeout": float(timeout_seconds),
                                }
                            else:
                                command = ["/bin/sh", "-lc", cmd]
                                run_kwargs = {
                                    "cwd": str(tmp_path),
                                    "env": env,
                                    "capture_output": True,
                                    "text": True,
                                    "timeout": float(timeout_seconds),
                                    "preexec_fn": _limit_resources if _os.name == "posix" else None,
                                }
                            completed = await _asyncio.wait_for(
                                _asyncio.to_thread(
                                    lambda: _subprocess.run(
                                        command,
                                        **run_kwargs,
                                    )
                                ),
                                timeout=float(timeout_seconds + 2),
                            )
                            rec["exit_code"] = int(completed.returncode)
                            rec["stdout"] = (completed.stdout or "")[:stdout_cap]
                            rec["stderr"] = (completed.stderr or "")[:stderr_cap]
                            rec["ok"] = completed.returncode == 0
                        except _subprocess.TimeoutExpired as e:
                            rec["stderr"] = (str(getattr(e, "stderr", "") or "") or "Timed out")[:stderr_cap]
                        except Exception as e:
                            rec["stderr"] = str(e)[:stderr_cap]
                        finally:
                            rec["duration_ms"] = int((datetime.utcnow() - start).total_seconds() * 1000)
                            runs.append(rec)
                            batch_runs.append(rec)
                        behavior["ran"] = True
                        next_progress = progress_start + int((progress_end - progress_start) * (i + 1) / max(1, len(batch)))
                        if scientific_validation:
                            await _update_linked_run(status="running", progress=next_progress, started=True)
                        _emit(next_progress, "running", f"[{phase}] Ran: {cmd}")
                        await db.commit()
                    return batch_runs

                primary_runs = await _run_command_batch(commands, phase="primary", progress_start=40, progress_end=72)
                bootstrap_used = False
                fallback_used = False
                retry_runs: List[dict] = []

                latest_primary_failure = next((r for r in reversed(primary_runs) if not bool(r.get("ok"))), None)
                if (
                    auto_bootstrap_retry
                    and latest_primary_failure is not None
                    and bootstrap_commands
                    and executor._should_bootstrap_after_verification_failure(latest_primary_failure)
                ):
                    bootstrap_used = True
                    behavior["bootstrap_attempted"] = True
                    _emit(74, "bootstrapping", "Primary verification failed; attempting environment bootstrap")
                    await db.commit()
                    bootstrap_runs = await _run_command_batch(bootstrap_commands, phase="bootstrap", progress_start=74, progress_end=84)
                    behavior["bootstrap_ok"] = bool(bootstrap_runs) and all(bool(r.get("ok")) for r in bootstrap_runs)
                    if behavior["bootstrap_ok"]:
                        retry_runs = await _run_command_batch(commands, phase="retry_primary", progress_start=84, progress_end=92)

                effective_runs = retry_runs if retry_runs else primary_runs
                latest_effective_failure = next((r for r in reversed(effective_runs) if not bool(r.get("ok"))), None)
                if latest_effective_failure is not None and fallback_commands:
                    fallback_used = True
                    behavior["fallback_attempted"] = True
                    _emit(93, "running", "Retrying verification with fallback commands")
                    await db.commit()
                    fallback_runs = await _run_command_batch(fallback_commands, phase="fallback", progress_start=93, progress_end=98)
                    behavior["fallback_ok"] = bool(fallback_runs) and all(bool(r.get("ok")) for r in fallback_runs)

                behavior["bootstrap_used"] = bootstrap_used
                behavior["fallback_used"] = fallback_used

        ok: bool | None
        if not enabled_effective:
            ok = None
        else:
            verification_runs = [
                r for r in runs
                if str(r.get("phase") or "") in {"primary", "retry_primary", "fallback"}
            ]
            if verification_runs:
                latest_phase = str(verification_runs[-1].get("phase") or "")
                latest_phase_runs = [
                    r for r in verification_runs
                    if str(r.get("phase") or "") == latest_phase
                ]
                ok = all(bool(r.get("ok")) for r in latest_phase_runs)
            else:
                ok = False
        behavior["ok"] = ok

        # Optional: write results back into LaTeX project.
        latex_updated = False
        if latex_project_id_raw:
            try:
                proj_uuid = _UUID(str(latex_project_id_raw))
            except Exception:
                proj_uuid = None
            if proj_uuid:
                proj = await db.get(LatexProject, proj_uuid)
                if proj and proj.user_id == job.user_id:
                    lines = ["\\section{Results}", f"\\subsection{{Experiment Runner ({source.name})}}"]
                    lines.append("\\begin{itemize}")
                    for r in runs[:10]:
                        cmd = str(r.get("command") or "")
                        status = "OK" if r.get("ok") else "FAIL"
                        lines.append(f"\\item \\texttt{{{cmd.replace('{', '').replace('}', '')[:120]}}}: {status}")
                    lines.append("\\end{itemize}")
                    if not enabled_effective:
                        lines.append("\\noindent \\textbf{Note:} Execution was skipped because unsafe code execution is disabled on the server.")
                    proj.tex_source = _insert_before_end_document(proj.tex_source or "", "\n".join(lines))
                    await db.commit()
                    latex_updated = True

        job.results = dict(base_results)
        prev_er = job.results.get("experiment_run") if isinstance(job.results.get("experiment_run"), dict) else None
        if isinstance(prev_er, dict):
            existing = job.results.get("experiment_runs")
            if not isinstance(existing, list):
                existing = []
            existing.append(prev_er)
            job.results["experiment_runs"] = existing[-5:]

        phase_summary = executor._summarize_experiment_run_phases(runs)
        job.results["code_patch_apply"] = patch_apply
        job.results["experiment_run"] = {
            "source_id": str(source.id),
            "source_name": source.name,
            "enabled": enabled_effective,
            "backend": backend_effective,
            "commands": commands,
            "verification_commands": commands,
            "bootstrap_commands": bootstrap_commands,
            "fallback_commands": fallback_commands,
            "runs": runs,
            "ok": ok,
            "final_phase": phase_summary.get("final_phase"),
            "phases": phase_summary.get("phases"),
            "verification_phases": phase_summary.get("verification_phases"),
            "failed_commands": phase_summary.get("failed_commands"),
            "proposal_id": patch_apply.get("proposal_id"),
            "latex_project_id": str(latex_project_id_raw) if latex_project_id_raw else None,
            "latex_updated": latex_updated,
            "inferred_project_profile": inferred_profile if inferred_profile else None,
            "bootstrap_attempted": bool(behavior.get("bootstrap_attempted")),
            "bootstrap_ok": behavior.get("bootstrap_ok"),
            "bootstrap_used": bool(behavior.get("bootstrap_used")),
            "fallback_attempted": bool(behavior.get("fallback_attempted")),
            "fallback_ok": behavior.get("fallback_ok"),
            "fallback_used": bool(behavior.get("fallback_used")),
        }
        if scientific_validation:
            job.results["scientific_validation"] = {
                "validation_kind": "scientific_validation",
                "recipe_family": str(scientific_validation.get("recipe_family") or ""),
                "recipe_id": str(scientific_validation.get("recipe_id") or ""),
                "recipe_version": int(scientific_validation.get("recipe_version") or 1),
                "sandbox_profile_id": str(scientific_validation.get("sandbox_profile_id") or ""),
                "domain_research_profile_id": str(scientific_validation.get("domain_research_profile_id") or "").strip() or None,
                "research_portfolio_id": str(scientific_validation.get("research_portfolio_id") or "").strip() or None,
                "hypothesis_id": str(scientific_validation.get("hypothesis_id") or "").strip() or None,
                "originating_job_id": str(scientific_validation.get("originating_job_id") or "").strip() or None,
                "decision_summary": str(scientific_validation.get("decision_summary") or "")[:2000] or None,
                "baseline_comparison": scientific_validation.get("baseline_comparison") if isinstance(scientific_validation.get("baseline_comparison"), dict) else {},
                "artifact_collection_rules": scientific_validation.get("artifact_collection_rules") if isinstance(scientific_validation.get("artifact_collection_rules"), list) else [],
                "budget_limit": scientific_validation.get("budget_limit"),
                "runtime_limit_minutes": scientific_validation.get("runtime_limit_minutes"),
                "blocked_reason_code": str(scientific_validation.get("blocked_reason_code") or "").strip() or None,
                "capability_check": scientific_validation.get("capability_check") if isinstance(scientific_validation.get("capability_check"), dict) else {},
                "profile_snapshot": scientific_validation.get("profile_snapshot") if isinstance(scientific_validation.get("profile_snapshot"), dict) else {},
                "recipe_snapshot": scientific_validation.get("recipe_snapshot") if isinstance(scientific_validation.get("recipe_snapshot"), dict) else {},
                "status": "succeeded" if ok else ("blocked" if not enabled_effective else "failed"),
            }
        code_patch_execution = (
            job.results.get("code_patch_execution")
            if isinstance(job.results.get("code_patch_execution"), dict)
            else None
        )
        if isinstance(code_patch_execution, dict):
            existing_recovery = (
                code_patch_execution.get("recovery")
                if isinstance(code_patch_execution.get("recovery"), dict)
                else None
            )
            code_patch_execution["recovery"] = executor._build_code_patch_execution_recovery(
                job=job,
                experiment_run=job.results.get("experiment_run") if isinstance(job.results.get("experiment_run"), dict) else None,
                existing_recovery=existing_recovery,
            )
            job.results["code_patch_execution"] = code_patch_execution

        scientific_run_status = "succeeded" if ok else ("blocked" if not enabled_effective else "failed")
        if scientific_validation:
            await _update_linked_run(
                status=scientific_run_status,
                progress=100,
                results=job.results.get("experiment_run") if isinstance(job.results.get("experiment_run"), dict) else {},
                summary=(
                    str((job.results.get("scientific_validation") or {}).get("decision_summary") or "")
                    or ("Scientific validation succeeded" if ok else "Scientific validation failed")
                ),
                started=True,
                completed=True,
            )
            await executor._update_scientific_validation_summary_links(
                db=db,
                profile_id=str(scientific_validation.get("domain_research_profile_id") or "").strip() or None,
                portfolio_id=str(scientific_validation.get("research_portfolio_id") or "").strip() or None,
                run_id=str(cfg.get("experiment_run_id") or "").strip() or None,
                run_record={
                    "run_id": str(cfg.get("experiment_run_id") or "").strip() or None,
                    "status": scientific_run_status,
                    "recipe_family": str(scientific_validation.get("recipe_family") or ""),
                    "sandbox_profile_id": str(scientific_validation.get("sandbox_profile_id") or ""),
                    "hypothesis_id": str(scientific_validation.get("hypothesis_id") or "").strip() or None,
                    "profile_id": str(scientific_validation.get("domain_research_profile_id") or "").strip() or None,
                    "portfolio_id": str(scientific_validation.get("research_portfolio_id") or "").strip() or None,
                    "job_id": str(job.id),
                },
            )

        if not enabled_effective:
            _emit(100, "completed", "Experiment run skipped (unsafe execution disabled)")
            job.status = AgentJobStatus.COMPLETED.value
        else:
            _emit(100, "completed" if ok else "failed", "Experiment run complete")
            job.status = AgentJobStatus.COMPLETED.value if ok else AgentJobStatus.FAILED.value
            if not ok:
                job.error = "Experiment run failed"
        job.completed_at = datetime.utcnow()
        await db.commit()
        return {"status": job.status, "results": job.results, "error": job.error}
