"""
Experiment Orchestrator endpoints.

Creates experiment plans from Research Notes (Hypothesis section) and tracks runs/results over time.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.experiment import ExperimentPlan, ExperimentRun
from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.research_note import ResearchNote
from app.models.user import User
from uuid import UUID

from app.schemas.experiment import (
    ExperimentPlanGenerateRequest,
    ExperimentPlanListResponse,
    ExperimentPlanResponse,
    ExperimentPlanUpdateRequest,
    ExperimentRunCreateRequest,
    ExperimentRunListResponse,
    ExperimentRunResponse,
    ExperimentRunStartRequest,
    ExperimentRunStartResponse,
    ExperimentRunSyncResponse,
    ExperimentRunUpdateRequest,
)
from app.services.llm_service import LLMService
from app.schemas.research_note import ResearchNoteResponse
from app.tasks.agent_job_tasks import execute_agent_job_task

router = APIRouter()


def _extract_hypothesis_section(markdown: str) -> Optional[str]:
    """
    Extract a hypothesis section from a markdown note, if present.

    Looks for headings like:
    - '# Hypothesis', '## Hypothesis', '## Hypotheses'
    and returns content until the next heading of the same/higher level.
    """
    if not markdown:
        return None
    lines = markdown.splitlines()
    # Find heading line index
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
    # Capture until next heading with level <= start_level
    out: list[str] = []
    next_heading_re = re.compile(r"^(#{1,6})\s+.+\s*$")
    for j in range(start_idx, len(lines)):
        m2 = next_heading_re.match(lines[j].strip())
        if m2 and len(m2.group(1)) <= (start_level or 6):
            break
        out.append(lines[j])
    text = "\n".join(out).strip()
    return text or None


def _build_experiment_plan_prompt(note_title: str, hypothesis_text: str, include: Dict[str, bool]) -> str:
    """
    Prompt for structured experiment plan generation.

    Output is strict JSON (no markdown) that front-end can render and users can edit.
    """
    sections = []
    sections.append("You are an AI research engineer. Create a runnable experiment plan from the hypothesis.")
    sections.append("Return ONLY valid JSON. No markdown, no commentary.")
    sections.append(
        "JSON schema (high level): {\n"
        '  "hypothesis": string,\n'
        '  "problem_statement": string,\n'
        '  "success_criteria": [string],\n'
        '  "datasets": [{"name": string, "source": string, "split": string|null, "notes": string|null}],\n'
        '  "metrics": [{"name": string, "definition": string, "direction": "higher_better"|"lower_better"}],\n'
        '  "baselines": [{"name": string, "details": string}],\n'
        '  "method": {"summary": string, "key_components": [string]},\n'
        '  "experiments": [{"name": string, "purpose": string, "variables": [string], "expected_outcome": string}],\n'
        '  "ablations": [{"name": string, "remove_or_change": string, "expected_effect": string}] | [],\n'
        '  "evaluation_protocol": string,\n'
        '  "compute_budget": {"hardware": string|null, "time_estimate": string|null, "notes": string|null},\n'
        '  "timeline": [{"week": string, "deliverable": string}] | [],\n'
        '  "risks": [{"risk": string, "mitigation": string}] | [],\n'
        '  "repro_checklist": [string] | []\n'
        "}"
    )
    sections.append(f"Note title: {note_title}")
    sections.append("Hypothesis section (may be short):")
    sections.append(hypothesis_text)

    # Constraints
    sections.append("Rules:")
    sections.append("- Keep it concrete: include at least 3 experiments and 2 metrics.")
    sections.append("- Prefer minimal feasible baselines.")
    if not include.get("ablations", True):
        sections.append('- Set "ablations" to an empty array [].')
    if not include.get("timeline", True):
        sections.append('- Set "timeline" to an empty array [].')
    if not include.get("risks", True):
        sections.append('- Set "risks" to an empty array [].')
    if not include.get("repro_checklist", True):
        sections.append('- Set "repro_checklist" to an empty array [].')
    sections.append("- Ensure the JSON is parseable.")

    return "\n\n".join(sections).strip()


@router.post("/plans/generate", response_model=ExperimentPlanResponse, status_code=status.HTTP_201_CREATED)
async def generate_experiment_plan(
    request: ExperimentPlanGenerateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    note = await db.get(ResearchNote, request.note_id)
    if not note or note.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Research note not found")

    content = (note.content_markdown or "").strip()
    if request.prefer_section == "hypothesis":
        hypothesis_text = _extract_hypothesis_section(content) or content
    else:
        hypothesis_text = content

    hypothesis_text = (hypothesis_text or "").strip()
    if request.max_note_chars and len(hypothesis_text) > int(request.max_note_chars):
        hypothesis_text = hypothesis_text[: int(request.max_note_chars)]

    include = {
        "ablations": bool(request.include_ablations),
        "timeline": bool(request.include_timeline),
        "risks": bool(request.include_risks),
        "repro_checklist": bool(request.include_repro_checklist),
    }

    llm = LLMService()
    prompt = _build_experiment_plan_prompt(note.title, hypothesis_text, include)

    try:
        raw = await llm.generate_response(
            query=prompt,
            max_tokens=1500,
            temperature=0.2,
            task_type="workflow_synthesis",
            user_id=current_user.id,
            db=db,
        )
    except Exception as exc:
        logger.warning(f"Experiment plan generation failed: {exc}")
        raise HTTPException(status_code=500, detail="Experiment plan generation failed")

    parsed: Dict[str, Any]
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else dict(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Plan must be an object")
    except Exception:
        # Try to salvage JSON from code fences or extra text
        try:
            m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if not m:
                raise ValueError("No JSON object found")
            parsed = json.loads(m.group(0))
        except Exception:
            raise HTTPException(status_code=422, detail="Model did not return valid JSON")

    plan = ExperimentPlan(
        user_id=current_user.id,
        research_note_id=note.id,
        title=f"Experiment Plan: {note.title}",
        hypothesis_text=hypothesis_text if request.prefer_section == "hypothesis" else None,
        plan=parsed,
        generator="llm",
        generator_details={"generated_at": datetime.utcnow().isoformat()},
    )
    db.add(plan)
    await db.commit()
    await db.refresh(plan)
    return ExperimentPlanResponse.model_validate(plan)


@router.get("/notes/{note_id}/plans", response_model=ExperimentPlanListResponse)
async def list_experiment_plans_for_note(
    note_id: UUID,
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    note = await db.get(ResearchNote, note_id)
    if not note or note.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Research note not found")

    stmt = (
        select(ExperimentPlan)
        .where(and_(ExperimentPlan.user_id == current_user.id, ExperimentPlan.research_note_id == note_id))
        .order_by(ExperimentPlan.created_at.desc())
        .limit(limit)
    )
    res = await db.execute(stmt)
    plans = list(res.scalars().all())
    return ExperimentPlanListResponse(plans=[ExperimentPlanResponse.model_validate(p) for p in plans])


@router.get("/plans/{plan_id}", response_model=ExperimentPlanResponse)
async def get_experiment_plan(
    plan_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    plan = await db.get(ExperimentPlan, plan_id)
    if not plan or plan.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Experiment plan not found")
    return ExperimentPlanResponse.model_validate(plan)


@router.patch("/plans/{plan_id}", response_model=ExperimentPlanResponse)
async def update_experiment_plan(
    plan_id: UUID,
    updates: ExperimentPlanUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    plan = await db.get(ExperimentPlan, plan_id)
    if not plan or plan.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Experiment plan not found")

    data = updates.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(plan, k, v)

    await db.commit()
    await db.refresh(plan)
    return ExperimentPlanResponse.model_validate(plan)


@router.post("/plans/{plan_id}/runs", response_model=ExperimentRunResponse, status_code=status.HTTP_201_CREATED)
async def create_experiment_run(
    plan_id: UUID,
    request: ExperimentRunCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    plan = await db.get(ExperimentPlan, plan_id)
    if not plan or plan.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Experiment plan not found")

    run = ExperimentRun(
        user_id=current_user.id,
        experiment_plan_id=plan.id,
        name=request.name,
        config=request.config,
        summary=request.summary,
        status="planned",
        progress=0,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)
    return ExperimentRunResponse.model_validate(run)


@router.get("/plans/{plan_id}/runs", response_model=ExperimentRunListResponse)
async def list_experiment_runs(
    plan_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    plan = await db.get(ExperimentPlan, plan_id)
    if not plan or plan.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Experiment plan not found")

    stmt = (
        select(ExperimentRun)
        .where(and_(ExperimentRun.user_id == current_user.id, ExperimentRun.experiment_plan_id == plan.id))
        .order_by(ExperimentRun.created_at.desc())
    )
    res = await db.execute(stmt)
    runs = list(res.scalars().all())
    return ExperimentRunListResponse(runs=[ExperimentRunResponse.model_validate(r) for r in runs])


@router.patch("/runs/{run_id}", response_model=ExperimentRunResponse)
async def update_experiment_run(
    run_id: UUID,
    updates: ExperimentRunUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    run = await db.get(ExperimentRun, run_id)
    if not run or run.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Experiment run not found")

    data = updates.model_dump(exclude_unset=True)

    # Auto-set timestamps for status transitions if not provided
    next_status = data.get("status")
    if next_status and next_status != run.status:
        if next_status == "running" and data.get("started_at") is None:
            data["started_at"] = datetime.utcnow()
        if next_status in {"completed", "failed", "cancelled"} and data.get("completed_at") is None:
            data["completed_at"] = datetime.utcnow()

    for k, v in data.items():
        setattr(run, k, v)

    await db.commit()
    await db.refresh(run)
    return ExperimentRunResponse.model_validate(run)


@router.post("/runs/{run_id}/start", response_model=ExperimentRunStartResponse, status_code=status.HTTP_201_CREATED)
async def start_experiment_run(
    run_id: UUID,
    request: ExperimentRunStartRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Start an ExperimentRun by creating an AgentJob that uses the deterministic `experiment_runner`.

    This runs shell commands against a git DocumentSource (explicitly gated by server unsafe execution settings).
    """
    run = await db.get(ExperimentRun, run_id)
    if not run or run.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Experiment run not found")

    plan = await db.get(ExperimentPlan, run.experiment_plan_id)
    if not plan or plan.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Experiment plan not found")

    # If run already linked, don't create a duplicate agent job.
    if run.agent_job_id:
        raise HTTPException(status_code=400, detail="Run already started (agent job exists)")

    commands = [str(c).strip() for c in (request.commands or []) if str(c).strip()]
    commands = commands[:6]

    job = AgentJob(
        name=f"Experiment Run: {run.name}",
        description=f"Experiment runner for plan '{plan.title}'",
        job_type="analysis",
        goal="Run experiment commands/tests and record results.",
        config={
            "deterministic_runner": "experiment_runner",
            "source_id": str(request.source_id),
            "commands": commands,
            "latex_project_id": str(request.latex_project_id) if request.latex_project_id else "",
            "timeout_seconds": int(request.timeout_seconds),
            "experiment_run_id": str(run.id),
            "experiment_plan_id": str(plan.id),
        },
        user_id=current_user.id,
        status=AgentJobStatus.PENDING.value,
        max_iterations=1,
        max_tool_calls=0,
        max_llm_calls=0,
        max_runtime_minutes=10,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    run.agent_job_id = job.id
    run.status = "running"
    run.progress = 0
    run.started_at = run.started_at or datetime.utcnow()
    await db.commit()
    await db.refresh(run)

    if request.start_immediately:
        execute_agent_job_task.delay(str(job.id), str(current_user.id))

    return ExperimentRunStartResponse(run=ExperimentRunResponse.model_validate(run), agent_job_id=job.id)


@router.post("/runs/{run_id}/sync", response_model=ExperimentRunSyncResponse)
async def sync_experiment_run_from_job(
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Sync ExperimentRun fields from the linked AgentJob (status/progress/results).
    """
    run = await db.get(ExperimentRun, run_id)
    if not run or run.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Experiment run not found")

    if not run.agent_job_id:
        raise HTTPException(status_code=400, detail="Run has no linked agent job")

    job = await db.get(AgentJob, run.agent_job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Linked agent job not found")

    # Map job status to run status (best-effort)
    job_status = str(job.status or "").lower()
    if job_status == "completed":
        run.status = "completed"
        run.completed_at = run.completed_at or (job.completed_at or datetime.utcnow())
        run.progress = 100
    elif job_status == "failed":
        run.status = "failed"
        run.completed_at = run.completed_at or (job.completed_at or datetime.utcnow())
        run.progress = int(job.progress or 0)
    elif job_status in {"cancelled", "canceled"}:
        run.status = "cancelled"
        run.completed_at = run.completed_at or (job.completed_at or datetime.utcnow())
        run.progress = int(job.progress or 0)
    elif job_status in {"running", "pending", "paused"}:
        run.status = "running"
        run.progress = int(job.progress or 0)

    # Pull results from job.results.experiment_run if present
    jr = job.results if isinstance(job.results, dict) else {}
    exp_run = jr.get("experiment_run") if isinstance(jr.get("experiment_run"), dict) else None
    if exp_run:
        run.results = exp_run
        if not run.summary:
            note = exp_run.get("note") or exp_run.get("summary")
            if note:
                run.summary = str(note)[:20000]

    await db.commit()
    await db.refresh(run)
    return ExperimentRunSyncResponse(run=ExperimentRunResponse.model_validate(run))


@router.post("/runs/{run_id}/append-to-note", response_model=ResearchNoteResponse)
async def append_experiment_run_to_note(
    run_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Append a concise experiment results section to the originating research note.

    Idempotent: if this run was already appended (by marker), it is a no-op.
    """
    run = await db.get(ExperimentRun, run_id)
    if not run or run.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Experiment run not found")

    plan = await db.get(ExperimentPlan, run.experiment_plan_id)
    if not plan or plan.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Experiment plan not found")

    note = await db.get(ResearchNote, plan.research_note_id)
    if not note or note.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Research note not found")

    marker = f"<!-- experiment_run:{run.id} -->"
    existing = (note.content_markdown or "")
    if marker in existing:
        return ResearchNoteResponse.model_validate(note)

    results = run.results if isinstance(run.results, dict) else {}
    commands = results.get("commands") if isinstance(results.get("commands"), list) else []
    runs = results.get("runs") if isinstance(results.get("runs"), list) else []
    ok = results.get("ok")

    status_line = f"Status: {run.status}"
    if isinstance(ok, bool):
        status_line += f" · ok={str(ok).lower()}"

    header = "## Experiment Results"
    block: list[str] = [
        header,
        marker,
        "",
        f"Run: **{run.name}**",
        status_line,
        f"Agent job: {str(run.agent_job_id) if run.agent_job_id else '-'}",
        f"Updated: {datetime.utcnow().isoformat()}",
        "",
    ]

    if commands:
        block.append("Commands:")
        for c in commands[:10]:
            block.append(f"- `{str(c)[:240]}`")
        block.append("")

    if runs:
        block.append("Results (first 10):")
        for r in runs[:10]:
            cmd = str(r.get("command") or "")[:200]
            exit_code = r.get("exit_code")
            ok2 = r.get("ok")
            dur = r.get("duration_ms")
            stderr = str(r.get("stderr") or "").strip()
            stderr_1 = stderr.splitlines()[0].strip()[:200] if stderr else ""

            line = f"- `{cmd}`"
            if isinstance(ok2, bool):
                line += f" · ok={str(ok2).lower()}"
            if exit_code is not None:
                line += f" · exit={exit_code}"
            if dur is not None:
                line += f" · {dur}ms"
            if stderr_1 and not ok2:
                line += f" · stderr: {stderr_1}"
            block.append(line)
        block.append("")

    note.content_markdown = existing.rstrip() + "\n\n" + "\n".join(block).rstrip() + "\n"
    await db.commit()
    await db.refresh(note)
    return ResearchNoteResponse.model_validate(note)
