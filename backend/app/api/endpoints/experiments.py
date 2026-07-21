"""
Experiment Orchestrator endpoints.

Creates experiment plans from Research Notes (Hypothesis section) and tracks runs/results over time.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.agent_jobs import _perform_job_action
from app.services.agent_job_scheduler_state import (
    extract_scheduler_state as _extract_scheduler_state,
)
from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.experiment import ExperimentPlan, ExperimentRun
from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.research_note import ResearchNote
from app.models.synthesis_job import SynthesisJob, SynthesisJobStatus, SynthesisJobType
from app.models.user import User
from uuid import UUID

from app.schemas.agent_job import AgentJobActionRequest
from app.schemas.experiment import (
    ExperimentRunActionRequest,
    ExperimentRunActionResponse,
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
from app.schemas.benchmark import BenchmarkSuiteListResponse, BenchmarkSuiteResponse
from app.services.llm_service import LLMService
from app.services.autonomy_event_service import record_autonomy_decision_event
from app.services.experiment_outcome_service import reconcile_experiment_run_outcome_to_originating_opportunity
from app.services.operator_interventions import derive_operator_interventions_with_outcomes
from app.services.benchmark_harness_service import get_benchmark_suite, list_benchmark_suites
from app.services.scientific_validation_service import build_scientific_validation_recipe, get_scientific_sandbox_profile
from app.services.synthesis_service import synthesis_service
from app.schemas.research_note import ResearchNoteResponse
from app.tasks.agent_job_tasks import execute_agent_job_task
from app.tasks.synthesis_tasks import execute_synthesis_task

router = APIRouter()


def _text(value: Any) -> str:
    return str(value or "").strip()


def _plan_benchmark_metadata(plan: ExperimentPlan) -> Dict[str, Any]:
    details = plan.generator_details if isinstance(plan.generator_details, dict) else {}
    plan_body = plan.plan if isinstance(plan.plan, dict) else {}
    provenance = plan_body.get("provenance") if isinstance(plan_body.get("provenance"), dict) else {}
    return {
        "benchmark_family": str(
            details.get("benchmark_family")
            or plan_body.get("benchmark_family")
            or provenance.get("benchmark_family")
            or ""
        ).strip() or None,
        "benchmark_suite_id": str(
            details.get("benchmark_suite_id")
            or plan_body.get("benchmark_suite_id")
            or provenance.get("benchmark_suite_id")
            or ""
        ).strip() or None,
        "benchmark_case_ids": [
            str(item).strip()
            for item in (
                details.get("benchmark_case_ids")
                if isinstance(details.get("benchmark_case_ids"), list)
                else (plan_body.get("benchmark_case_ids") if isinstance(plan_body.get("benchmark_case_ids"), list) else provenance.get("benchmark_case_ids"))
            ) or []
            if str(item).strip()
        ],
        "benchmark_baseline_id": str(
            details.get("benchmark_baseline_id")
            or plan_body.get("benchmark_baseline_id")
            or provenance.get("benchmark_baseline_id")
            or ""
        ).strip() or None,
    }


def _plan_to_response(plan: ExperimentPlan) -> ExperimentPlanResponse:
    benchmark = _plan_benchmark_metadata(plan)
    return ExperimentPlanResponse.model_validate(
        {
            **plan.__dict__,
            **benchmark,
        }
    )


def _scientific_validation_payload(run: ExperimentRun) -> Dict[str, Any]:
    config = run.config if isinstance(run.config, dict) else {}
    value = config.get("scientific_validation")
    return deepcopy(value) if isinstance(value, dict) else {}


def _set_scientific_validation_payload(run: ExperimentRun, payload: Dict[str, Any]) -> Dict[str, Any]:
    config = deepcopy(run.config) if isinstance(run.config, dict) else {}
    if payload:
        config["scientific_validation"] = payload
    else:
        config.pop("scientific_validation", None)
    run.config = config
    return config


def _is_scientific_validation_run(run: ExperimentRun) -> bool:
    return bool(_scientific_validation_payload(run))


def _run_operator_actions(run: ExperimentRun) -> list[Dict[str, Any]]:
    scientific_validation = _scientific_validation_payload(run)
    value = scientific_validation.get("operator_actions")
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _append_run_operator_action(
    run: ExperimentRun,
    *,
    action: str,
    current_user: User,
    note: str | None = None,
    previous_status: str | None = None,
    new_status: str | None = None,
    linked_job_id: UUID | str | None = None,
    linked_job_action: str | None = None,
    outcome_status: str | None = None,
    outcome_reason: str | None = None,
    parent_run_id: UUID | str | None = None,
    child_run_id: UUID | str | None = None,
) -> None:
    scientific_validation = _scientific_validation_payload(run)
    operator_actions = _run_operator_actions(run)
    operator_actions.append(
        {
            "action": str(action or "").strip(),
            "actor_user_id": str(current_user.id),
            "at": datetime.utcnow().isoformat(),
            "note": str(note or "").strip() or None,
            "previous_status": str(previous_status or run.status or "").strip() or None,
            "new_status": str(new_status or run.status or "").strip() or None,
            "linked_job_id": str(linked_job_id or "").strip() or None,
            "linked_job_action": str(linked_job_action or "").strip() or None,
            "outcome_status": str(outcome_status or "").strip() or None,
            "outcome_reason": str(outcome_reason or "").strip() or None,
            "parent_run_id": str(parent_run_id or "").strip() or None,
            "child_run_id": str(child_run_id or "").strip() or None,
        }
    )
    scientific_validation["operator_actions"] = operator_actions[-50:]
    _set_scientific_validation_payload(run, scientific_validation)


async def _start_experiment_run_internal(
    *,
    run: ExperimentRun,
    plan: ExperimentPlan,
    current_user: User,
    db: AsyncSession,
    source_id: UUID | str,
    commands: list[str],
    timeout_seconds: int,
    latex_project_id: UUID | str | None = None,
    start_immediately: bool = True,
) -> tuple[ExperimentRun, AgentJob]:
    if run.agent_job_id:
        raise HTTPException(status_code=400, detail="Run already started (agent job exists)")

    normalized_commands = _run_start_commands(run, commands)
    if not normalized_commands:
        raise HTTPException(status_code=400, detail="Run is missing executable commands")

    run_config = deepcopy(run.config) if isinstance(run.config, dict) else {}
    scientific_validation = (
        run_config.get("scientific_validation")
        if isinstance(run_config.get("scientific_validation"), dict)
        else None
    )
    job = AgentJob(
        name=f"Experiment Run: {run.name}",
        description=f"Experiment runner for plan '{plan.title}'",
        job_type="analysis",
        goal="Run experiment commands/tests and record results.",
        config={
            **run_config,
            "deterministic_runner": "experiment_runner",
            "source_id": str(source_id),
            "commands": normalized_commands,
            "latex_project_id": str(latex_project_id) if latex_project_id else "",
            "timeout_seconds": int(timeout_seconds),
            "experiment_run_id": str(run.id),
            "experiment_plan_id": str(plan.id),
            "scientific_validation": scientific_validation,
        },
        user_id=current_user.id,
        status=AgentJobStatus.PENDING.value,
        max_iterations=1,
        max_tool_calls=0,
        max_llm_calls=0,
        max_runtime_minutes=10,
    )
    db.add(job)
    await db.flush()

    run.agent_job_id = job.id
    run.status = "queued" if isinstance(scientific_validation, dict) else "running"
    run.progress = 0
    run.started_at = run.started_at or datetime.utcnow()
    run.completed_at = None

    if start_immediately:
        execute_agent_job_task.delay(str(job.id), str(current_user.id))

    return run, job


async def _sync_experiment_run_from_job_internal(
    *,
    run: ExperimentRun,
    job: AgentJob,
) -> ExperimentRun:
    job_status = str(job.status or "").lower()
    scientific_validation = _scientific_validation_payload(run)
    completed_status = "succeeded" if scientific_validation else "completed"
    if job_status == "completed":
        run.status = completed_status
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
    elif job_status == "paused":
        run.status = "paused"
        run.progress = int(job.progress or 0)
    elif job_status in {"running", "pending"}:
        if run.status not in {"queued", "provisioning"}:
            run.status = "running"
        run.progress = int(job.progress or 0)

    jr = job.results if isinstance(job.results, dict) else {}
    exp_run = jr.get("experiment_run") if isinstance(jr.get("experiment_run"), dict) else None
    if exp_run:
        run.results = {
            **{key: value for key, value in jr.items() if key != "experiment_run"},
            **exp_run,
        }
        note = exp_run.get("note") or exp_run.get("summary")
        if note and not run.summary:
            run.summary = str(note)[:20000]
    return run


def _spawn_child_experiment_run(
    *,
    source_run: ExperimentRun,
    current_user: User,
    action: str,
    note: str | None,
) -> ExperimentRun:
    config = deepcopy(source_run.config) if isinstance(source_run.config, dict) else {}
    scientific_validation = (
        config.get("scientific_validation")
        if isinstance(config.get("scientific_validation"), dict)
        else {}
    )
    scientific_validation["blocked_reason_code"] = None
    scientific_validation["operator_actions"] = [
        {
            "action": str(action or "").strip(),
            "actor_user_id": str(current_user.id),
            "at": datetime.utcnow().isoformat(),
            "note": str(note or "").strip() or None,
            "previous_status": str(source_run.status or "").strip() or None,
            "new_status": "planned",
            "linked_job_id": None,
            "linked_job_action": None,
            "outcome_status": "spawned",
            "outcome_reason": f"Created from run {source_run.id}",
            "parent_run_id": str(source_run.id),
            "child_run_id": None,
        }
    ]
    config["scientific_validation"] = scientific_validation
    return ExperimentRun(
        user_id=source_run.user_id,
        experiment_plan_id=source_run.experiment_plan_id,
        parent_run_id=source_run.id,
        name=f"{source_run.name} · {str(action).capitalize()} {int(source_run.retry_count or 0) + 1}",
        status="planned",
        progress=0,
        retry_count=int(source_run.retry_count or 0) + 1,
        config=config,
        summary=source_run.summary,
    )


def _run_to_response(run: ExperimentRun) -> ExperimentRunResponse:
    """Project a typed experiment-run payload from stored results."""
    now = datetime.utcnow()
    results = run.results if isinstance(run.results, dict) else {}
    experiment_run = results if results else None
    config = run.config if isinstance(run.config, dict) else {}
    scientific_validation = _scientific_validation_payload(run)
    execution_handoff = _run_execution_handoff(run)
    compiler_artifacts = _run_compiler_artifacts(run)
    measurement_summary = _run_measurement_summary(run)
    perf_counters = _run_perf_counters(run)
    artifact_inventory = (
        compiler_artifacts.get("artifact_inventory")
        if isinstance(compiler_artifacts.get("artifact_inventory"), list)
        else (
            measurement_summary.get("artifact_inventory")
            if isinstance(measurement_summary.get("artifact_inventory"), list)
            else []
        )
    )
    execution_strategy = results.get("execution_strategy") if isinstance(results.get("execution_strategy"), dict) else {}
    operator_interventions_raw = (
        execution_strategy.get("operator_interventions")
        if isinstance(execution_strategy.get("operator_interventions"), list)
        else []
    )
    operator_interventions = derive_operator_interventions_with_outcomes(
        [row for row in operator_interventions_raw if isinstance(row, dict)],
        current_status=run.status,
        completed_at=getattr(run, "completed_at", None),
    )
    return ExperimentRunResponse.model_validate(
        {
            **run.__dict__,
            "created_at": getattr(run, "created_at", None) or now,
            "updated_at": getattr(run, "updated_at", None) or getattr(run, "created_at", None) or now,
            "results": results or None,
            "parent_run_id": run.parent_run_id,
            "latest_child_run_id": run.latest_child_run_id,
            "validation_kind": str(scientific_validation.get("validation_kind") or "").strip() or None,
            "sandbox_profile_id": str(scientific_validation.get("sandbox_profile_id") or "").strip() or None,
            "recipe_family": str(scientific_validation.get("recipe_family") or "").strip() or None,
            "recipe_id": str(scientific_validation.get("recipe_id") or "").strip() or None,
            "recipe_version": int(scientific_validation.get("recipe_version") or 0) or None,
            "domain_research_profile_id": str(scientific_validation.get("domain_research_profile_id") or "").strip() or None,
            "research_portfolio_id": str(scientific_validation.get("research_portfolio_id") or "").strip() or None,
            "hypothesis_id": str(scientific_validation.get("hypothesis_id") or "").strip() or None,
            "originating_job_id": str(scientific_validation.get("originating_job_id") or "").strip() or None,
            "blocked_reason_code": str(scientific_validation.get("blocked_reason_code") or scientific_validation.get("blocked_reason") or "").strip() or None,
            "capability_check": scientific_validation.get("capability_check") if isinstance(scientific_validation.get("capability_check"), dict) else None,
            "profile_snapshot": scientific_validation.get("profile_snapshot") if isinstance(scientific_validation.get("profile_snapshot"), dict) else None,
            "recipe_snapshot": scientific_validation.get("recipe_snapshot") if isinstance(scientific_validation.get("recipe_snapshot"), dict) else None,
            "benchmark_family": str(scientific_validation.get("benchmark_family") or execution_handoff.get("benchmark_family") or "").strip() or None,
            "benchmark_suite_id": str(scientific_validation.get("benchmark_suite_id") or execution_handoff.get("benchmark_suite_id") or "").strip() or None,
            "benchmark_case_ids": [
                str(item).strip()
                for item in (
                    scientific_validation.get("benchmark_case_ids")
                    if isinstance(scientific_validation.get("benchmark_case_ids"), list)
                    else (execution_handoff.get("benchmark_case_ids") if isinstance(execution_handoff.get("benchmark_case_ids"), list) else [])
                )
                if str(item).strip()
            ],
            "benchmark_baseline_id": str(scientific_validation.get("benchmark_baseline_id") or execution_handoff.get("benchmark_baseline_id") or "").strip() or None,
            "measurement_summary": measurement_summary or None,
            "compiler_artifacts": compiler_artifacts or None,
            "perf_counters": perf_counters or None,
            "artifact_inventory": artifact_inventory,
            "repeat_count": _run_repeat_count(run),
            "experiment_run": experiment_run,
            "operator_interventions": operator_interventions or None,
            "operator_actions": _run_operator_actions(run) or None,
            "retry_count": int(run.retry_count or 0),
        }
    )


def _plan_execution_handoff(plan: ExperimentPlan) -> Dict[str, Any]:
    details = plan.generator_details if isinstance(plan.generator_details, dict) else {}
    plan_body = plan.plan if isinstance(plan.plan, dict) else {}
    benchmark = _plan_benchmark_metadata(plan)
    selected_hypothesis_ids = details.get("selected_hypothesis_ids")
    if not isinstance(selected_hypothesis_ids, list):
        selected_hypothesis_ids = plan_body.get("selected_hypothesis_ids") if isinstance(plan_body.get("selected_hypothesis_ids"), list) else []
    supporting_sources = details.get("supporting_sources")
    if not isinstance(supporting_sources, list):
        supporting_sources = plan_body.get("supporting_sources") if isinstance(plan_body.get("supporting_sources"), list) else []
    provenance = plan_body.get("provenance") if isinstance(plan_body.get("provenance"), dict) else {}
    source_paper_ids = details.get("source_paper_ids")
    if not isinstance(source_paper_ids, list):
        source_paper_ids = provenance.get("source_paper_ids") if isinstance(provenance.get("source_paper_ids"), list) else []
    source_document_ids = details.get("source_document_ids")
    if not isinstance(source_document_ids, list):
        source_document_ids = provenance.get("source_document_ids") if isinstance(provenance.get("source_document_ids"), list) else []
    source_run_ids = details.get("source_run_ids")
    if not isinstance(source_run_ids, list):
        source_run_ids = provenance.get("source_run_ids") if isinstance(provenance.get("source_run_ids"), list) else []
    autonomous_origin = details.get("autonomous_origin") if isinstance(details.get("autonomous_origin"), dict) else {}
    if not autonomous_origin and isinstance(provenance.get("autonomous_origin"), dict):
        autonomous_origin = provenance.get("autonomous_origin")
    if not autonomous_origin:
        source_kind = "profile" if _text(details.get("profile_id")) else ("portfolio" if _text(details.get("portfolio_id")) else "")
        source_id = _text(details.get("profile_id") or details.get("portfolio_id"))
        opportunity_id = _text(details.get("opportunity_id"))
        if source_kind and source_id and opportunity_id:
            autonomous_origin = {
                "source_kind": source_kind,
                "source_id": source_id,
                "opportunity_id": opportunity_id,
                "evidence_revision_at_launch": _text(details.get("evidence_revision_at_launch")) or None,
            }

    return {
        "execution_handoff_version": 1,
        "plan_scope": str(details.get("plan_mode") or plan_body.get("plan_scope") or "").strip() or None,
        "selected_hypothesis_ids": [str(item) for item in selected_hypothesis_ids if str(item).strip()],
        "supporting_sources": [dict(item) for item in supporting_sources if isinstance(item, dict)],
        "source_paper_ids": [str(item) for item in source_paper_ids if str(item).strip()],
        "source_document_ids": [str(item) for item in source_document_ids if str(item).strip()],
        "source_run_ids": [str(item) for item in source_run_ids if str(item).strip()],
        "primary_run_id": str(details.get("primary_run_id") or "").strip() or None,
        "comparison_run_id": str(details.get("comparison_run_id") or "").strip() or None,
        "regression_type": str(details.get("regression_type") or "").strip() or None,
        "explanation_mode": bool(details.get("explanation_mode")),
        "benchmark_family": benchmark["benchmark_family"],
        "benchmark_suite_id": benchmark["benchmark_suite_id"],
        "benchmark_case_ids": benchmark["benchmark_case_ids"],
        "benchmark_baseline_id": benchmark["benchmark_baseline_id"],
        "autonomous_origin": dict(autonomous_origin) if isinstance(autonomous_origin, dict) and autonomous_origin else None,
    }


def _merge_run_config_with_plan_handoff(plan: ExperimentPlan, run_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    config = deepcopy(run_config) if isinstance(run_config, dict) else {}
    handoff = _plan_execution_handoff(plan)
    if any(
        handoff.get(key)
        for key in (
            "plan_scope",
            "selected_hypothesis_ids",
            "supporting_sources",
            "source_paper_ids",
            "source_document_ids",
            "source_run_ids",
            "primary_run_id",
            "comparison_run_id",
            "regression_type",
            "explanation_mode",
            "benchmark_family",
            "benchmark_suite_id",
            "benchmark_case_ids",
            "benchmark_baseline_id",
            "autonomous_origin",
        )
    ):
        config["execution_handoff"] = handoff
    return config


def _default_run_name_from_plan(plan: ExperimentPlan) -> str:
    details = plan.generator_details if isinstance(plan.generator_details, dict) else {}
    plan_mode = str(details.get("plan_mode") or "").strip()
    selected_ids = details.get("selected_hypothesis_ids") if isinstance(details.get("selected_hypothesis_ids"), list) else []
    if plan_mode == "single_hypothesis":
        if selected_ids:
            return f"{plan.title} · {selected_ids[0]}"
        return f"{plan.title} · hypothesis"
    if plan_mode == "aggregate_note":
        return f"{plan.title} · aggregate"
    if plan_mode == "compiler_regression_followup":
        return f"{plan.title} · regression follow-up"
    return f"{plan.title} · run"


def _default_run_summary_from_plan(plan: ExperimentPlan) -> Optional[str]:
    plan_body = plan.plan if isinstance(plan.plan, dict) else {}
    objective = str(plan_body.get("objective") or "").strip()
    hypothesis = str(plan_body.get("hypothesis") or "").strip()
    handoff = _plan_execution_handoff(plan)
    selected_ids = handoff.get("selected_hypothesis_ids") if isinstance(handoff.get("selected_hypothesis_ids"), list) else []
    source_titles = [
        str(item.get("title") or item.get("id") or "").strip()
        for item in (handoff.get("supporting_sources") if isinstance(handoff.get("supporting_sources"), list) else [])
        if isinstance(item, dict)
    ]
    parts: list[str] = []
    if objective:
        parts.append(objective)
    if hypothesis:
        parts.append(hypothesis)
    if selected_ids:
        parts.append(f"Hypotheses: {', '.join(selected_ids[:5])}")
    if source_titles:
        parts.append(f"Sources: {', '.join([title for title in source_titles[:3] if title])}")
    summary = " | ".join(part for part in parts if part).strip()
    return summary[:20000] if summary else None


def _benchmark_queries_from_context(benchmark_context: Optional[Dict[str, Any]]) -> list[str]:
    if not isinstance(benchmark_context, dict):
        return []
    queries: list[str] = []
    for case in benchmark_context.get("selected_cases") if isinstance(benchmark_context.get("selected_cases"), list) else []:
        if not isinstance(case, dict):
            continue
        query = str(case.get("benchmark_query") or case.get("name") or "").strip()
        if query and query not in queries:
            queries.append(query)
    return queries[:8]


def _benchmark_commands_from_context(benchmark_context: Optional[Dict[str, Any]]) -> list[str]:
    if not isinstance(benchmark_context, dict):
        return []
    commands: list[str] = []
    for case in benchmark_context.get("selected_cases") if isinstance(benchmark_context.get("selected_cases"), list) else []:
        if not isinstance(case, dict):
            continue
        for raw in (case.get("compile_command_template"), case.get("run_command_template")):
            text = str(raw or "").strip()
            if text and text not in commands:
                commands.append(text)
    return commands[:6]


def _benchmark_context_from_suite_payload(
    suite: Optional[Dict[str, Any]],
    *,
    requested_case_ids: Optional[list[str]] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(suite, dict):
        return None
    suite_cases = suite.get("cases") if isinstance(suite.get("cases"), list) else []
    requested = [str(item).strip() for item in (requested_case_ids or []) if str(item).strip()]
    selected_cases = [
        dict(case)
        for case in suite_cases
        if isinstance(case, dict) and (not requested or str(case.get("id") or "").strip() in requested)
    ]
    if requested and not selected_cases:
        raise HTTPException(status_code=400, detail="No requested benchmark cases matched the selected suite")
    if not selected_cases:
        selected_cases = [dict(case) for case in suite_cases[: min(3, len(suite_cases))] if isinstance(case, dict)]
    baselines = suite.get("baselines") if isinstance(suite.get("baselines"), list) else []
    default_baseline_id = str((suite.get("metadata") or {}).get("default_baseline_id") or "").strip() if isinstance(suite.get("metadata"), dict) else ""
    selected_baseline = next(
        (
            dict(item)
            for item in baselines
            if isinstance(item, dict) and str(item.get("id") or "").strip() == default_baseline_id
        ),
        dict(baselines[0]) if baselines and isinstance(baselines[0], dict) else None,
    )
    return {
        "suite": dict(suite),
        "selected_cases": selected_cases,
        "selected_case_ids": [str(case.get("id") or "").strip() for case in selected_cases if str(case.get("id") or "").strip()],
        "baseline": selected_baseline,
        "benchmark_queries": _benchmark_queries_from_context({"selected_cases": selected_cases}),
        "default_commands": _benchmark_commands_from_context({"selected_cases": selected_cases}),
    }


def _benchmark_observability_from_context(benchmark_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    selected_cases = (
        benchmark_context.get("selected_cases")
        if isinstance(benchmark_context, dict) and isinstance(benchmark_context.get("selected_cases"), list)
        else []
    )
    artifact_inventory: list[str] = []
    metric_names: list[str] = []
    pass_signals: list[str] = []
    capture_ir = False
    capture_asm = False
    capture_remarks = False
    capture_compile_logs = False
    capture_perf_stat = False
    repeat_count = 1

    for case in selected_cases:
        if not isinstance(case, dict):
            continue
        for item in case.get("expected_artifacts") if isinstance(case.get("expected_artifacts"), list) else []:
            text = str(item).strip()
            if text and text not in artifact_inventory:
                artifact_inventory.append(text)
        for metric in case.get("metrics") if isinstance(case.get("metrics"), list) else []:
            if not isinstance(metric, dict):
                continue
            name = str(metric.get("name") or "").strip()
            if name and name not in metric_names:
                metric_names.append(name)
        metadata = case.get("metadata") if isinstance(case.get("metadata"), dict) else {}
        observability = metadata.get("observability") if isinstance(metadata.get("observability"), dict) else {}
        capture_ir = capture_ir or bool(observability.get("capture_ir"))
        capture_asm = capture_asm or bool(observability.get("capture_asm"))
        capture_remarks = capture_remarks or bool(observability.get("capture_remarks"))
        capture_compile_logs = capture_compile_logs or bool(observability.get("capture_compile_logs"))
        capture_perf_stat = capture_perf_stat or bool(observability.get("capture_perf_stat"))
        try:
            repeat_count = max(repeat_count, int(observability.get("repeat_count") or 1))
        except Exception:
            repeat_count = max(repeat_count, 1)
        for signal in observability.get("pass_signals") if isinstance(observability.get("pass_signals"), list) else []:
            text = str(signal).strip()
            if text and text not in pass_signals:
                pass_signals.append(text)

    if capture_compile_logs and "compiler_logs" not in artifact_inventory:
        artifact_inventory.append("compiler_logs")
    if capture_remarks and "compiler_remarks" not in artifact_inventory:
        artifact_inventory.append("compiler_remarks")
    if (capture_ir or capture_asm) and "ir_or_codegen_artifacts" not in artifact_inventory:
        artifact_inventory.append("ir_or_codegen_artifacts")
    if capture_perf_stat and "perf_counter_summary" not in artifact_inventory:
        artifact_inventory.append("perf_counter_summary")

    return {
        "capture_ir": capture_ir,
        "capture_asm": capture_asm,
        "capture_remarks": capture_remarks,
        "capture_compile_logs": capture_compile_logs,
        "capture_perf_stat": capture_perf_stat,
        "repeat_count": repeat_count,
        "artifact_inventory": artifact_inventory[:16],
        "metric_names": metric_names[:16],
        "pass_signals": pass_signals[:16],
    }


def _run_measurement_summary(run: ExperimentRun) -> Dict[str, Any]:
    results = run.results if isinstance(run.results, dict) else {}
    measurement_summary = (
        deepcopy(results.get("measurement_summary"))
        if isinstance(results.get("measurement_summary"), dict)
        else (
            deepcopy(_scientific_validation_payload(run).get("measurement_summary"))
            if isinstance(_scientific_validation_payload(run).get("measurement_summary"), dict)
            else {}
        )
    )
    if not isinstance(measurement_summary, dict):
        measurement_summary = {}
    compiler_artifacts = _run_compiler_artifacts(run)
    artifact_inventory = compiler_artifacts.get("artifact_inventory") if isinstance(compiler_artifacts.get("artifact_inventory"), list) else []
    if artifact_inventory and not isinstance(measurement_summary.get("artifact_inventory"), list):
        measurement_summary["artifact_inventory"] = artifact_inventory
    perf_counters = _run_perf_counters(run)
    if perf_counters and not isinstance(measurement_summary.get("perf_counters"), dict):
        measurement_summary["perf_counters"] = perf_counters
    repeat_count = _run_repeat_count(run)
    if repeat_count and not measurement_summary.get("repeat_count"):
        measurement_summary["repeat_count"] = repeat_count
    return measurement_summary


def _run_perf_counters(run: ExperimentRun) -> Dict[str, Any]:
    results = run.results if isinstance(run.results, dict) else {}
    direct = results.get("perf_counters")
    if isinstance(direct, dict) and direct:
        return deepcopy(direct)
    for container in (
        results.get("measurement_summary"),
        _scientific_validation_payload(run).get("measurement_summary"),
    ):
        if isinstance(container, dict):
            counters = container.get("perf_counters")
            if isinstance(counters, dict) and counters:
                return deepcopy(counters)
    return {}


def _run_compiler_artifacts(run: ExperimentRun) -> Dict[str, Any]:
    results = run.results if isinstance(run.results, dict) else {}
    base = deepcopy(results.get("compiler_artifacts")) if isinstance(results.get("compiler_artifacts"), dict) else {}
    scientific_validation = _scientific_validation_payload(run)
    observability = (
        scientific_validation.get("compiler_observability")
        if isinstance(scientific_validation.get("compiler_observability"), dict)
        else {}
    )
    artifact_inventory = []
    for item in (
        base.get("artifact_inventory")
        if isinstance(base.get("artifact_inventory"), list)
        else (observability.get("artifact_inventory") if isinstance(observability.get("artifact_inventory"), list) else [])
    ):
        text = str(item).strip()
        if text and text not in artifact_inventory:
            artifact_inventory.append(text)
    payload = {
        "ir_paths": [str(item).strip() for item in (base.get("ir_paths") if isinstance(base.get("ir_paths"), list) else []) if str(item).strip()][:8],
        "asm_paths": [str(item).strip() for item in (base.get("asm_paths") if isinstance(base.get("asm_paths"), list) else []) if str(item).strip()][:8],
        "remark_paths": [str(item).strip() for item in (base.get("remark_paths") if isinstance(base.get("remark_paths"), list) else []) if str(item).strip()][:8],
        "log_paths": [str(item).strip() for item in (base.get("log_paths") if isinstance(base.get("log_paths"), list) else []) if str(item).strip()][:8],
        "diff_summary": str(base.get("diff_summary") or observability.get("diff_summary") or "").strip() or None,
        "pass_signals": [str(item).strip() for item in (base.get("pass_signals") if isinstance(base.get("pass_signals"), list) else (observability.get("pass_signals") if isinstance(observability.get("pass_signals"), list) else [])) if str(item).strip()][:12],
        "artifact_inventory": artifact_inventory[:16],
        "capture_ir": bool(base.get("ir_paths")) or bool(observability.get("capture_ir")),
        "capture_asm": bool(base.get("asm_paths")) or bool(observability.get("capture_asm")),
        "capture_remarks": bool(base.get("remark_paths")) or bool(observability.get("capture_remarks")),
        "capture_compile_logs": bool(base.get("log_paths")) or bool(observability.get("capture_compile_logs")),
        "capture_perf_stat": bool(_run_perf_counters(run)) or bool(observability.get("capture_perf_stat")),
    }
    if any(
        payload.get(key)
        for key in (
            "ir_paths",
            "asm_paths",
            "remark_paths",
            "log_paths",
            "diff_summary",
            "pass_signals",
            "artifact_inventory",
            "capture_ir",
            "capture_asm",
            "capture_remarks",
            "capture_compile_logs",
            "capture_perf_stat",
        )
    ):
        return payload
    return {}


def _run_repeat_count(run: ExperimentRun) -> Optional[int]:
    measurement_summary = (
        (run.results if isinstance(run.results, dict) else {}).get("measurement_summary")
        if isinstance((run.results if isinstance(run.results, dict) else {}).get("measurement_summary"), dict)
        else (
            _scientific_validation_payload(run).get("measurement_summary")
            if isinstance(_scientific_validation_payload(run).get("measurement_summary"), dict)
            else {}
        )
    )
    if isinstance(measurement_summary, dict):
        try:
            value = int(measurement_summary.get("repeat_count") or 0)
            if value > 0:
                return value
        except Exception:
            pass
    observability = (
        _scientific_validation_payload(run).get("compiler_observability")
        if isinstance(_scientific_validation_payload(run).get("compiler_observability"), dict)
        else {}
    )
    try:
        value = int(observability.get("repeat_count") or 0)
        return value or None
    except Exception:
        return None


async def _build_benchmark_context_for_request(
    *,
    db: AsyncSession,
    benchmark_suite_id: Optional[str],
    benchmark_case_ids: Optional[list[str]],
) -> Optional[Dict[str, Any]]:
    suite_id = str(benchmark_suite_id or "").strip()
    if not suite_id:
        return None
    suite = await get_benchmark_suite(db, suite_id)
    if not isinstance(suite, dict):
        raise HTTPException(status_code=404, detail="Benchmark suite not found")
    return _benchmark_context_from_suite_payload(suite, requested_case_ids=benchmark_case_ids)


async def _build_scientific_validation_for_run(
    *,
    db: AsyncSession,
    plan: ExperimentPlan,
    run_config: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    handoff = _plan_execution_handoff(plan)
    benchmark_suite_id = str(handoff.get("benchmark_suite_id") or "").strip()
    if not benchmark_suite_id:
        return None
    benchmark_context = await _build_benchmark_context_for_request(
        db=db,
        benchmark_suite_id=benchmark_suite_id,
        benchmark_case_ids=(handoff.get("benchmark_case_ids") if isinstance(handoff.get("benchmark_case_ids"), list) else []),
    )
    if not isinstance(benchmark_context, dict):
        return None
    suite = benchmark_context["suite"]
    baseline = benchmark_context.get("baseline") if isinstance(benchmark_context.get("baseline"), dict) else {}
    supporting_sources = _experiment_supporting_sources(handoff)
    compiler_observability = _benchmark_observability_from_context(benchmark_context)
    objective = str((plan.plan or {}).get("objective") or plan.title or "").strip()
    hypothesis_title = str((plan.plan or {}).get("hypothesis") or plan.title or "").strip()[:240] or plan.title
    hypothesis_text = str(plan.hypothesis_text or objective or "").strip()[:2000] or hypothesis_title
    verification_commands = _benchmark_commands_from_context(benchmark_context)
    recipe = build_scientific_validation_recipe(
        track_type=str(suite.get("track_type") or "compiler"),
        objective=objective or "Validate benchmark-backed compiler hypothesis",
        hypothesis_title=hypothesis_title,
        hypothesis_text=hypothesis_text,
        benchmark_queries=benchmark_context.get("benchmark_queries"),
        verification_commands=verification_commands,
        supporting_sources=supporting_sources,
        supporting_evidence=[
            f"Benchmark suite: {suite.get('name')}",
            *[str(case.get("name") or "").strip() for case in benchmark_context.get("selected_cases", []) if isinstance(case, dict)],
        ],
    )
    profile = await get_scientific_sandbox_profile(
        db,
        str((baseline or {}).get("sandbox_profile_id") or "").strip() or None,
        track_type=str(suite.get("track_type") or "compiler"),
    )
    source_id = str(run_config.get("source_id") or "").strip() or None
    return {
        "validation_kind": "scientific_validation",
        "sandbox_profile_id": str((profile or {}).get("id") or "").strip() or None,
        "recipe_family": str(recipe.get("recipe_family") or "").strip() or None,
        "recipe_id": str(recipe.get("recipe_id") or "").strip() or None,
        "recipe_version": int(recipe.get("recipe_version") or 1),
        "benchmark_family": str(suite.get("benchmark_family") or recipe.get("benchmark_family") or "").strip() or None,
        "benchmark_suite_id": benchmark_suite_id,
        "benchmark_case_ids": benchmark_context.get("selected_case_ids", []),
        "benchmark_baseline_id": str((baseline or {}).get("id") or "").strip() or None,
        "benchmark_queries": benchmark_context.get("benchmark_queries", []),
        "artifact_collection_rules": recipe.get("artifact_collection_rules") if isinstance(recipe.get("artifact_collection_rules"), list) else [],
        "compiler_observability": {
            **(
                recipe.get("compiler_observability_defaults")
                if isinstance(recipe.get("compiler_observability_defaults"), dict)
                else {}
            ),
            **compiler_observability,
        },
        "baseline_comparison": {
            **(recipe.get("baseline_comparison") if isinstance(recipe.get("baseline_comparison"), dict) else {}),
            "benchmark_baseline_id": str((baseline or {}).get("id") or "").strip() or None,
            "baseline_measurements": dict((baseline or {}).get("measurements") or {}) if isinstance((baseline or {}).get("measurements"), dict) else {},
        },
        "measurement_summary": {
            "status": "pending",
            "benchmark_suite_id": benchmark_suite_id,
            "benchmark_case_ids": benchmark_context.get("selected_case_ids", []),
            "benchmark_baseline_id": str((baseline or {}).get("id") or "").strip() or None,
            "baseline_measurements": dict((baseline or {}).get("measurements") or {}) if isinstance((baseline or {}).get("measurements"), dict) else {},
            "artifact_inventory": compiler_observability.get("artifact_inventory", []),
            "repeat_count": compiler_observability.get("repeat_count", 1),
            "perf_counters": {},
        },
        "commands": verification_commands,
        "decision_summary": str(recipe.get("decision_summary") or "").strip() or None,
        "profile_snapshot": profile or {},
        "recipe_snapshot": recipe,
        "capability_check": {
            "ok": bool(source_id),
            "required": ["repo_reconstruction"],
            "satisfied": ["repo_reconstruction"] if source_id else [],
            "missing": [] if source_id else ["repo_reconstruction"],
        },
    }


def _run_execution_handoff(run: ExperimentRun) -> Dict[str, Any]:
    config = run.config if isinstance(run.config, dict) else {}
    handoff = config.get("execution_handoff")
    return deepcopy(handoff) if isinstance(handoff, dict) else {}


def _run_post_run_actions(run: ExperimentRun) -> Dict[str, Any]:
    config = run.config if isinstance(run.config, dict) else {}
    value = config.get("post_run_actions")
    return deepcopy(value) if isinstance(value, dict) else {}


def _set_run_post_run_actions(run: ExperimentRun, payload: Dict[str, Any]) -> Dict[str, Any]:
    config = deepcopy(run.config) if isinstance(run.config, dict) else {}
    if payload:
        config["post_run_actions"] = payload
    else:
        config.pop("post_run_actions", None)
    run.config = config
    return config


def _run_start_commands(run: ExperimentRun, request_commands: list[str]) -> list[str]:
    requested = [str(item).strip() for item in request_commands if str(item).strip()]
    if requested:
        return requested[:6]
    config = run.config if isinstance(run.config, dict) else {}
    fallback = config.get("commands") if isinstance(config.get("commands"), list) else []
    return [str(item).strip() for item in fallback if str(item).strip()][:6]


def _is_terminal_experiment_run_status(status_value: str | None) -> bool:
    return str(status_value or "").strip().lower() in {"succeeded", "completed", "failed", "blocked", "cancelled"}


def _experiment_supporting_sources(handoff: Dict[str, Any]) -> list[Dict[str, Any]]:
    raw = handoff.get("supporting_sources") if isinstance(handoff.get("supporting_sources"), list) else []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _experiment_selected_hypothesis_ids(handoff: Dict[str, Any]) -> list[str]:
    raw = handoff.get("selected_hypothesis_ids") if isinstance(handoff.get("selected_hypothesis_ids"), list) else []
    return [str(item).strip() for item in raw if str(item).strip()]


def _result_highlights(run: ExperimentRun) -> list[str]:
    results = run.results if isinstance(run.results, dict) else {}
    highlights: list[str] = []
    measurement_summary = _run_measurement_summary(run)
    compiler_artifacts = _run_compiler_artifacts(run)
    final_phase = str(results.get("final_phase") or "").strip()
    if final_phase:
        highlights.append(f"Final phase: {final_phase}")
    if results.get("bootstrap_attempted"):
        highlights.append(f"Bootstrap: {'ok' if results.get('bootstrap_ok') is True else 'attempted'}")
    if results.get("fallback_attempted"):
        highlights.append(f"Fallback: {'ok' if results.get('fallback_ok') is True else 'attempted'}")
    if isinstance(results.get("ok"), bool):
        highlights.append(f"ok={str(bool(results.get('ok'))).lower()}")
    execution_strategy = results.get("execution_strategy") if isinstance(results.get("execution_strategy"), dict) else {}
    execution_graph = execution_strategy.get("execution_graph") if isinstance(execution_strategy.get("execution_graph"), dict) else {}
    graph_health = execution_graph.get("graph_health") if isinstance(execution_graph.get("graph_health"), dict) else {}
    reasons = graph_health.get("reasons") if isinstance(graph_health.get("reasons"), list) else []
    first_reason = next((str(item).strip() for item in reasons if str(item).strip()), "")
    if first_reason:
        highlights.append(f"Recovery reason: {first_reason}")
    compile_time_ms = measurement_summary.get("compile_time_ms")
    runtime_ms = measurement_summary.get("runtime_ms")
    binary_size_bytes = measurement_summary.get("binary_size_bytes")
    artifact_diff_score = measurement_summary.get("artifact_diff_score")
    comparison = str(measurement_summary.get("comparison") or "").strip()
    if compile_time_ms is not None:
        highlights.append(f"Compile time: {compile_time_ms}ms")
    if runtime_ms is not None:
        highlights.append(f"Runtime: {runtime_ms}ms")
    if binary_size_bytes is not None:
        highlights.append(f"Binary size: {binary_size_bytes}B")
    if artifact_diff_score is not None:
        highlights.append(f"Artifact diff score: {artifact_diff_score}")
    if comparison:
        highlights.append(f"Baseline comparison: {comparison}")
    diff_summary = str(compiler_artifacts.get("diff_summary") or "").strip()
    if diff_summary:
        highlights.append(f"Artifact diff: {diff_summary[:80]}")
    pass_signals = compiler_artifacts.get("pass_signals") if isinstance(compiler_artifacts.get("pass_signals"), list) else []
    if pass_signals:
        highlights.append(f"Pass signals: {', '.join([str(item) for item in pass_signals[:2] if str(item).strip()])}")
    perf_counters = _run_perf_counters(run)
    if perf_counters:
        first_key = next(iter(perf_counters.keys()))
        highlights.append(f"{first_key}: {perf_counters[first_key]}")
    return highlights[:8]


def _run_evidence_item(run: ExperimentRun, plan: ExperimentPlan, *, appended_at: str) -> Dict[str, Any]:
    results = run.results if isinstance(run.results, dict) else {}
    handoff = _run_execution_handoff(run)
    verification_commands = (
        results.get("verification_commands") if isinstance(results.get("verification_commands"), list) else []
    )
    failed_commands = results.get("failed_commands") if isinstance(results.get("failed_commands"), list) else []
    compiler_artifacts = _run_compiler_artifacts(run)
    measurement_summary = _run_measurement_summary(run)
    return {
        "run_id": str(run.id),
        "experiment_plan_id": str(plan.id),
        "plan_scope": str(handoff.get("plan_scope") or "").strip() or None,
        "status": str(run.status or "").strip() or None,
        "summary": str(run.summary or results.get("summary") or results.get("note") or "").strip() or None,
        "appended_at": appended_at,
        "selected_hypothesis_ids": _experiment_selected_hypothesis_ids(handoff),
        "supporting_sources": _experiment_supporting_sources(handoff),
        "source_paper_ids": [
            str(item).strip()
            for item in (handoff.get("source_paper_ids") if isinstance(handoff.get("source_paper_ids"), list) else [])
            if str(item).strip()
        ],
        "source_document_ids": [
            str(item).strip()
            for item in (handoff.get("source_document_ids") if isinstance(handoff.get("source_document_ids"), list) else [])
            if str(item).strip()
        ],
        "verification_commands": [str(item)[:240] for item in verification_commands[:10] if str(item).strip()],
        "failed_commands": [str(item)[:240] for item in failed_commands[:10] if str(item).strip()],
        "result_highlights": _result_highlights(run),
        "measurement_summary": measurement_summary or None,
        "compiler_artifacts": compiler_artifacts or None,
        "perf_counters": _run_perf_counters(run) or None,
        "artifact_diff_summary": str(compiler_artifacts.get("diff_summary") or "").strip() or None,
        "artifact_inventory": (
            compiler_artifacts.get("artifact_inventory")
            if isinstance(compiler_artifacts.get("artifact_inventory"), list)
            else []
        ),
        "repeat_count": _run_repeat_count(run),
        "autonomous_origin": (
            dict(handoff.get("autonomous_origin"))
            if isinstance(handoff.get("autonomous_origin"), dict)
            else None
        ),
    }


def _append_run_evidence_to_structured_payload(
    structured_payload: Optional[Dict[str, Any]],
    *,
    run: ExperimentRun,
    plan: ExperimentPlan,
    appended_at: str,
) -> Optional[Dict[str, Any]]:
    payload = deepcopy(structured_payload) if isinstance(structured_payload, dict) else None
    if not isinstance(payload, dict):
        return None
    hypotheses = payload.get("hypotheses")
    if not isinstance(hypotheses, list):
        return payload

    selected_ids = set(_experiment_selected_hypothesis_ids(_run_execution_handoff(run)))
    if not selected_ids:
        return payload

    evidence_item = _run_evidence_item(run, plan, appended_at=appended_at)
    updated_any = False
    updated_hypotheses: list[Dict[str, Any]] = []
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, dict):
            updated_hypotheses.append(hypothesis)
            continue
        hypothesis_id = str(hypothesis.get("id") or "").strip()
        if hypothesis_id not in selected_ids:
            updated_hypotheses.append(hypothesis)
            continue

        existing_evidence = (
            [dict(item) for item in hypothesis.get("experiment_evidence") if isinstance(item, dict)]
            if isinstance(hypothesis.get("experiment_evidence"), list)
            else []
        )
        existing_evidence = [item for item in existing_evidence if str(item.get("run_id") or "").strip() != str(run.id)]
        existing_evidence.append(evidence_item)
        hypothesis_copy = dict(hypothesis)
        hypothesis_copy["experiment_evidence"] = existing_evidence
        updated_hypotheses.append(hypothesis_copy)
        updated_any = True

    if not updated_any:
        return payload

    payload["hypotheses"] = updated_hypotheses
    payload["last_appended_run_id"] = str(run.id)
    payload["last_appended_at"] = appended_at
    return payload


def _run_evidence_already_recorded(structured_payload: Optional[Dict[str, Any]], *, run: ExperimentRun) -> bool:
    payload = structured_payload if isinstance(structured_payload, dict) else None
    if not isinstance(payload, dict):
        return False
    hypotheses = payload.get("hypotheses")
    if not isinstance(hypotheses, list):
        return False

    selected_ids = set(_experiment_selected_hypothesis_ids(_run_execution_handoff(run)))
    if not selected_ids:
        return False

    for hypothesis in hypotheses:
        if not isinstance(hypothesis, dict):
            continue
        if str(hypothesis.get("id") or "").strip() not in selected_ids:
            continue
        evidence_rows = hypothesis.get("experiment_evidence") if isinstance(hypothesis.get("experiment_evidence"), list) else []
        if any(str(item.get("run_id") or "").strip() == str(run.id) for item in evidence_rows if isinstance(item, dict)):
            return True
    return False


def _build_experiment_run_note_block(run: ExperimentRun, *, marker: str) -> list[str]:
    """Build the markdown block appended to a research note for one experiment run."""
    results = run.results if isinstance(run.results, dict) else {}
    commands = results.get("commands") if isinstance(results.get("commands"), list) else []
    verification_commands = (
        results.get("verification_commands") if isinstance(results.get("verification_commands"), list) else []
    )
    bootstrap_commands = (
        results.get("bootstrap_commands") if isinstance(results.get("bootstrap_commands"), list) else []
    )
    fallback_commands = (
        results.get("fallback_commands") if isinstance(results.get("fallback_commands"), list) else []
    )
    failed_commands = (
        results.get("failed_commands") if isinstance(results.get("failed_commands"), list) else []
    )
    run_summary = str(run.summary or results.get("summary") or results.get("note") or "").strip()
    phases = results.get("phases") if isinstance(results.get("phases"), list) else []
    final_phase = str(results.get("final_phase") or "").strip()
    source_id = str(results.get("source_id") or "").strip()
    source_name = str(results.get("source_name") or "").strip()
    inferred_project_profile = (
        results.get("inferred_project_profile") if isinstance(results.get("inferred_project_profile"), dict) else {}
    )
    execution_strategy = (
        results.get("execution_strategy") if isinstance(results.get("execution_strategy"), dict) else {}
    )
    execution_graph = (
        execution_strategy.get("execution_graph")
        if isinstance(execution_strategy.get("execution_graph"), dict)
        else {}
    )
    graph_health = execution_graph.get("graph_health") if isinstance(execution_graph.get("graph_health"), dict) else {}
    recovery_reasons = (
        graph_health.get("reasons")
        if isinstance(graph_health.get("reasons"), list)
        else []
    )
    recommended_actions = (
        execution_graph.get("recommended_actions")
        if isinstance(execution_graph.get("recommended_actions"), list)
        else []
    )
    operator_interventions = derive_operator_interventions_with_outcomes(
        (
            execution_strategy.get("operator_interventions")
            if isinstance(execution_strategy.get("operator_interventions"), list)
            else []
        ),
        current_status=run.status,
        completed_at=getattr(run, "completed_at", None),
    )
    detected_stack = (
        inferred_project_profile.get("detected_stack")
        if isinstance(inferred_project_profile.get("detected_stack"), list)
        else []
    )
    bootstrap_attempted = bool(results.get("bootstrap_attempted"))
    bootstrap_ok = results.get("bootstrap_ok")
    fallback_attempted = bool(results.get("fallback_attempted"))
    fallback_ok = results.get("fallback_ok")
    runs = results.get("runs") if isinstance(results.get("runs"), list) else []
    ok = results.get("ok")
    recovery_open = bool(fallback_attempted and failed_commands and fallback_ok is not True)
    first_recovery_reason = next((str(item).strip() for item in recovery_reasons if str(item).strip()), "")
    first_recommended_action = next((str(item).strip() for item in recommended_actions if str(item).strip()), "")
    latest_operator_intervention = next(
        (item for item in reversed(operator_interventions) if isinstance(item, dict)),
        {},
    )
    latest_operator_action = str(latest_operator_intervention.get("action") or "").strip().replace("_", " ")
    latest_operator_status_before = str(latest_operator_intervention.get("job_status_before") or "").strip()
    latest_operator_status_after = str(latest_operator_intervention.get("job_status_after") or "").strip()
    latest_operator_note = str(latest_operator_intervention.get("note") or "").strip()
    latest_operator_outcome = str(latest_operator_intervention.get("outcome_status") or "").strip().replace("_", " ")
    latest_operator_outcome_reason = str(latest_operator_intervention.get("outcome_reason") or "").strip()
    execution_handoff = _run_execution_handoff(run)
    selected_hypothesis_ids = _experiment_selected_hypothesis_ids(execution_handoff)
    supporting_sources = _experiment_supporting_sources(execution_handoff)
    source_titles = [
        str(item.get("title") or item.get("id") or "").strip()
        for item in supporting_sources
        if str(item.get("title") or item.get("id") or "").strip()
    ]
    source_paper_ids = [
        str(item).strip()
        for item in (execution_handoff.get("source_paper_ids") if isinstance(execution_handoff.get("source_paper_ids"), list) else [])
        if str(item).strip()
    ]
    plan_scope = str(execution_handoff.get("plan_scope") or "").strip()
    measurement_summary = _run_measurement_summary(run)
    compiler_artifacts = _run_compiler_artifacts(run)
    perf_counters = _run_perf_counters(run)

    status_line = f"Status: {run.status}"
    if isinstance(ok, bool):
        status_line += f" · ok={str(ok).lower()}"

    block: list[str] = [
        "## Experiment Results",
        marker,
        "",
        f"Run: **{run.name}**",
        status_line,
        f"Agent job: {str(run.agent_job_id) if run.agent_job_id else '-'}",
        f"Updated: {datetime.utcnow().isoformat()}",
        "",
    ]

    summary_bits: list[str] = []
    if source_name:
        summary_bits.append(f"Source: {source_name}")
    if source_id:
        summary_bits.append(f"Source ID: `{source_id}`")
    if detected_stack:
        summary_bits.append(
            "Detected stack: " + ", ".join(str(item)[:60] for item in detected_stack[:8] if str(item).strip())
        )
    if final_phase:
        summary_bits.append(f"Final phase: `{final_phase}`")
    if bootstrap_attempted:
        summary_bits.append(f"Bootstrap: {'ok' if bootstrap_ok is True else 'attempted'}")
    if fallback_attempted:
        summary_bits.append(f"Fallback: {'ok' if fallback_ok is True else 'attempted'}")
    if recovery_open:
        summary_bits.append("Recovery: open")
    if phases:
        summary_bits.append(
            "Phases: " + " -> ".join(str(p)[:80] for p in phases[:8] if str(p).strip())
        )
    if summary_bits:
        block.append("Execution summary:")
        for line in summary_bits:
            block.append(f"- {line}")
        block.append("")

    if plan_scope or selected_hypothesis_ids or source_titles or source_paper_ids:
        block.append("Hypothesis scope:")
        if plan_scope:
            block.append(f"- Plan scope: {plan_scope}")
        if selected_hypothesis_ids:
            block.append(f"- Selected hypotheses: {', '.join(selected_hypothesis_ids[:8])}")
        if source_titles:
            block.append(f"- Supporting sources: {', '.join(source_titles[:5])}")
        if source_paper_ids:
            block.append(f"- Source papers: {', '.join(source_paper_ids[:8])}")
        block.append("")

    if latest_operator_action:
        latest_operator_line = latest_operator_action
        if latest_operator_status_before or latest_operator_status_after:
            latest_operator_line += f" ({latest_operator_status_before or '?'} -> {latest_operator_status_after or '?'})"
        block.append("Operator intervention:")
        block.append(f"- Latest: {latest_operator_line[:240]}")
        if latest_operator_outcome:
            block.append(f"- Outcome: {latest_operator_outcome[:240]}")
        if latest_operator_outcome_reason:
            block.append(f"- Outcome reason: {latest_operator_outcome_reason[:240]}")
        if latest_operator_note:
            block.append(f"- Note: {latest_operator_note[:240]}")
        block.append("")

    if run_summary:
        block.append("Summary:")
        block.append(run_summary[:2000])
        block.append("")

    if measurement_summary:
        block.append("Benchmark measurements:")
        compile_time_ms = measurement_summary.get("compile_time_ms")
        runtime_ms = measurement_summary.get("runtime_ms")
        binary_size_bytes = measurement_summary.get("binary_size_bytes")
        artifact_diff_score = measurement_summary.get("artifact_diff_score")
        comparison = str(measurement_summary.get("comparison") or "").strip()
        if compile_time_ms is not None:
            block.append(f"- Compile time: {compile_time_ms} ms")
        if runtime_ms is not None:
            block.append(f"- Runtime: {runtime_ms} ms")
        if binary_size_bytes is not None:
            block.append(f"- Binary size: {binary_size_bytes} bytes")
        if artifact_diff_score is not None:
            block.append(f"- Artifact diff score: {artifact_diff_score}")
        if comparison:
            block.append(f"- Baseline comparison: {comparison}")
        if measurement_summary.get("repeat_count") is not None:
            block.append(f"- Repeats: {measurement_summary.get('repeat_count')}")
        block.append("")

    if compiler_artifacts or perf_counters:
        block.append("Compiler observability:")
        artifact_inventory = (
            compiler_artifacts.get("artifact_inventory")
            if isinstance(compiler_artifacts.get("artifact_inventory"), list)
            else []
        )
        if artifact_inventory:
            block.append(f"- Captured artifacts: {', '.join([str(item) for item in artifact_inventory[:8] if str(item).strip()])}")
        diff_summary = str(compiler_artifacts.get("diff_summary") or "").strip()
        if diff_summary:
            block.append(f"- Artifact diff: {diff_summary[:240]}")
        pass_signals = compiler_artifacts.get("pass_signals") if isinstance(compiler_artifacts.get("pass_signals"), list) else []
        if pass_signals:
            block.append(f"- Pass signals: {', '.join([str(item) for item in pass_signals[:6] if str(item).strip()])}")
        for label, key in (
            ("IR paths", "ir_paths"),
            ("ASM paths", "asm_paths"),
            ("Remark paths", "remark_paths"),
            ("Log paths", "log_paths"),
        ):
            paths = compiler_artifacts.get(key) if isinstance(compiler_artifacts.get(key), list) else []
            if paths:
                block.append(f"- {label}: {', '.join([str(item) for item in paths[:4] if str(item).strip()])}")
        if perf_counters:
            block.append(
                "- Perf counters: "
                + " · ".join([f"{key}={value}" for key, value in list(perf_counters.items())[:4]])
            )
        block.append("")

    if recovery_open and (first_recovery_reason or first_recommended_action):
        block.append("Recovery guidance:")
        if first_recovery_reason:
            block.append(f"- Reason: {first_recovery_reason[:240]}")
        if first_recommended_action:
            block.append(f"- Next: {first_recommended_action[:240]}")
        block.append("")

    if verification_commands:
        block.append("Verification commands:")
        for c in verification_commands[:10]:
            block.append(f"- `{str(c)[:240]}`")
        block.append("")

    if bootstrap_commands:
        block.append("Bootstrap commands:")
        for c in bootstrap_commands[:10]:
            block.append(f"- `{str(c)[:240]}`")
        block.append("")

    if fallback_commands:
        block.append("Fallback verification commands:")
        for c in fallback_commands[:10]:
            block.append(f"- `{str(c)[:240]}`")
        block.append("")

    if failed_commands:
        block.append("Failed commands:")
        for c in failed_commands[:10]:
            block.append(f"- `{str(c)[:240]}`")
        block.append("")

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

    return block


async def _append_experiment_run_to_note_internal(
    *,
    run: ExperimentRun,
    plan: ExperimentPlan,
    note: ResearchNote,
    appended_at: str | None = None,
) -> tuple[ResearchNote, bool]:
    marker = f"<!-- experiment_run:{run.id} -->"
    existing = note.content_markdown or ""
    appended_timestamp = str(appended_at or datetime.utcnow().isoformat()).strip() or datetime.utcnow().isoformat()
    had_existing_evidence = _run_evidence_already_recorded(
        note.structured_payload if isinstance(note.structured_payload, dict) else None,
        run=run,
    )
    note.structured_payload = _append_run_evidence_to_structured_payload(
        note.structured_payload if isinstance(note.structured_payload, dict) else None,
        run=run,
        plan=plan,
        appended_at=appended_timestamp,
    )
    if marker not in existing:
        block = _build_experiment_run_note_block(run, marker=marker)
        note.content_markdown = existing.rstrip() + "\n\n" + "\n".join(block).rstrip() + "\n"

    post_run_actions = _run_post_run_actions(run)
    if post_run_actions:
        post_run_actions["append_status"] = "completed"
        post_run_actions["appended_at"] = appended_timestamp
        post_run_actions.pop("append_error", None)
        _set_run_post_run_actions(run, post_run_actions)
    return note, not had_existing_evidence


async def _queue_pending_hypothesis_reevaluation_draft(
    *,
    note: ResearchNote,
    run: ExperimentRun,
    db: AsyncSession,
    current_user: User,
) -> None:
    payload = note.structured_payload if isinstance(note.structured_payload, dict) else {}
    if str(payload.get("artifact_type") or "").strip() != SynthesisJobType.HYPOTHESIS_REEVALUATION.value:
        return
    if not isinstance(payload.get("hypotheses"), list) or not payload.get("hypotheses"):
        return

    pending_job_id = str(payload.get("pending_reevaluation_job_id") or "").strip()
    active_pending_job: SynthesisJob | None = None
    if pending_job_id:
        try:
            active_pending_job = await db.get(SynthesisJob, UUID(pending_job_id))
        except (TypeError, ValueError):
            active_pending_job = None
        if active_pending_job and (
            active_pending_job.user_id != current_user.id
            or active_pending_job.job_type != SynthesisJobType.HYPOTHESIS_REEVALUATION.value
        ):
            active_pending_job = None

    if active_pending_job and active_pending_job.status not in {
        SynthesisJobStatus.COMPLETED.value,
        SynthesisJobStatus.FAILED.value,
        SynthesisJobStatus.CANCELLED.value,
    }:
        source_run_ids = payload.get("pending_reevaluation_source_run_ids") if isinstance(payload.get("pending_reevaluation_source_run_ids"), list) else []
        deduped = [str(item).strip() for item in source_run_ids if str(item).strip()]
        run_id = str(run.id)
        if run_id not in deduped:
            deduped.append(run_id)
        payload["pending_reevaluation_source_run_ids"] = deduped
        note.structured_payload = payload
        return

    job = await synthesis_service.create_job(
        db=db,
        user_id=current_user.id,
        job_type=SynthesisJobType.HYPOTHESIS_REEVALUATION.value,
        title=f"Hypothesis Re-evaluation · {note.title}"[:500],
        document_ids=[],
        paper_ids=[],
        research_note_id=note.id,
        description="Auto-queued after new experiment evidence was appended to the note.",
        output_format="markdown",
        output_style="technical",
    )
    execute_synthesis_task.delay(str(job.id), str(current_user.id))

    payload["pending_reevaluation_job_id"] = str(job.id)
    payload["pending_reevaluation_created_at"] = datetime.utcnow().isoformat()
    payload["pending_reevaluation_reason"] = "new_experiment_evidence"
    payload["pending_reevaluation_source_run_ids"] = [str(run.id)]
    note.structured_payload = payload


async def _maybe_auto_append_experiment_run_to_note(
    *,
    run: ExperimentRun,
    plan: ExperimentPlan,
    db: AsyncSession,
    current_user: User,
) -> None:
    if not _is_terminal_experiment_run_status(run.status):
        return

    post_run_actions = _run_post_run_actions(run)
    if not post_run_actions or not bool(post_run_actions.get("auto_append_to_note")):
        return

    append_status = str(post_run_actions.get("append_status") or "").strip().lower()
    if append_status == "completed":
        return

    target_note_id = str(post_run_actions.get("target_note_id") or plan.research_note_id or "").strip()
    appended_at = datetime.utcnow().isoformat()
    try:
        if not target_note_id:
            raise ValueError("Missing target_note_id for auto-append")
        note_uuid = UUID(target_note_id)
        note = await db.get(ResearchNote, note_uuid)
        if not note or note.user_id != current_user.id:
            raise ValueError("Target research note not found")
        note, new_evidence_added = await _append_experiment_run_to_note_internal(
            run=run,
            plan=plan,
            note=note,
            appended_at=appended_at,
        )
        if new_evidence_added:
            await _queue_pending_hypothesis_reevaluation_draft(
                note=note,
                run=run,
                db=db,
                current_user=current_user,
            )
    except Exception as exc:
        logger.warning("Auto-append failed for run {} and note {}: {}", run.id, target_note_id, exc)
        post_run_actions["append_status"] = "failed"
        post_run_actions["append_error"] = str(exc)[:1000]
        _set_run_post_run_actions(run, post_run_actions)


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


def _structured_hypotheses(note: ResearchNote) -> list[dict[str, Any]]:
    payload = note.structured_payload if isinstance(note.structured_payload, dict) else {}
    value = payload.get("hypotheses")
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _is_reevaluated_hypothesis_note(note: ResearchNote) -> bool:
    payload = note.structured_payload if isinstance(note.structured_payload, dict) else {}
    return str(payload.get("artifact_type") or "").strip() == "hypothesis_reevaluation"


def _is_compiler_regression_explanation_note(note: ResearchNote) -> bool:
    payload = note.structured_payload if isinstance(note.structured_payload, dict) else {}
    return str(payload.get("artifact_type") or "").strip() == "compiler_regression_explanation"


def _top_ranked_hypothesis_id(note: ResearchNote) -> Optional[str]:
    hypotheses = _structured_hypotheses(note)
    if not hypotheses:
        return None
    sorted_hypotheses = sorted(
        hypotheses,
        key=lambda item: (
            float(item.get("rank") or 9999),
            -float(item.get("overall_score") or 0),
        ),
    )
    for item in sorted_hypotheses:
        hypothesis_id = str(item.get("id") or "").strip()
        if hypothesis_id:
            return hypothesis_id
    return None


def _build_structured_experiment_context(
    note: ResearchNote,
    *,
    plan_mode: str,
    hypothesis_id: Optional[str],
    max_note_chars: int,
    benchmark_context: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    payload = note.structured_payload if isinstance(note.structured_payload, dict) else {}
    if _is_compiler_regression_explanation_note(note):
        source_run_ids = payload.get("source_run_ids") if isinstance(payload.get("source_run_ids"), list) else []
        source_document_ids = payload.get("source_document_ids") if isinstance(payload.get("source_document_ids"), list) else []
        source_paper_ids = payload.get("source_paper_ids") if isinstance(payload.get("source_paper_ids"), list) else []
        likely_causes = payload.get("likely_causes") if isinstance(payload.get("likely_causes"), list) else []
        recommended_next_steps = payload.get("recommended_next_steps") if isinstance(payload.get("recommended_next_steps"), list) else []
        supporting_signals = payload.get("supporting_signals") if isinstance(payload.get("supporting_signals"), list) else []
        confounders = payload.get("confounders") if isinstance(payload.get("confounders"), list) else []
        metric_deltas = payload.get("metric_deltas") if isinstance(payload.get("metric_deltas"), list) else []
        artifact_deltas = payload.get("artifact_deltas") if isinstance(payload.get("artifact_deltas"), list) else []
        summary = str(payload.get("summary") or "").strip()
        regression_type = str(payload.get("regression_type") or "").strip() or "mixed"
        primary_run_id = str(payload.get("primary_run_id") or "").strip()
        comparison_run_id = str(payload.get("comparison_run_id") or "").strip()

        lines: list[str] = []
        if summary:
            lines.append(f"Explanation summary: {summary}")
        lines.append("Plan mode: compiler_regression_followup")
        lines.append(f"Regression type: {regression_type}")
        if primary_run_id or comparison_run_id:
            lines.append(
                "Compared runs: "
                + " vs ".join([item for item in [primary_run_id, comparison_run_id] if item])
            )
        if source_run_ids:
            lines.append(f"Source run ids: {', '.join(str(item) for item in source_run_ids[:12])}")
        if likely_causes:
            lines.append("Likely causes:")
            for item in likely_causes[:5]:
                if isinstance(item, dict):
                    title = str(item.get("title") or "Cause").strip()
                    confidence = str(item.get("confidence") or "").strip()
                    reason = str(item.get("reason") or "").strip()
                    lines.append(
                        f"- {title}"
                        + (f" [{confidence}]" if confidence else "")
                        + (f": {reason}" if reason else "")
                    )
                else:
                    lines.append(f"- {str(item).strip()}")
        if recommended_next_steps:
            lines.append("Recommended next steps:")
            for item in recommended_next_steps[:6]:
                text = str(item).strip()
                if text:
                    lines.append(f"- {text}")
        if supporting_signals:
            lines.append("Supporting signals:")
            for item in supporting_signals[:6]:
                text = str(item).strip()
                if text:
                    lines.append(f"- {text}")
        if confounders:
            lines.append("Confounders:")
            for item in confounders[:6]:
                text = str(item).strip()
                if text:
                    lines.append(f"- {text}")
        if metric_deltas:
            lines.append("Metric deltas:")
            for item in metric_deltas[:6]:
                if isinstance(item, dict):
                    metric = str(item.get("metric") or "metric").strip()
                    primary = item.get("primary")
                    comparison = item.get("comparison")
                    interpretation = str(item.get("interpretation") or "").strip()
                    lines.append(
                        f"- {metric}: {comparison} -> {primary}"
                        + (f" ({interpretation})" if interpretation else "")
                    )
        if artifact_deltas:
            lines.append("Artifact deltas:")
            for item in artifact_deltas[:6]:
                if isinstance(item, dict):
                    kind = str(item.get("kind") or "artifact").strip()
                    summary_text = str(item.get("summary") or "").strip()
                    lines.append(f"- {kind}: {summary_text}")

        if isinstance(benchmark_context, dict):
            suite = benchmark_context.get("suite") if isinstance(benchmark_context.get("suite"), dict) else {}
            baseline = benchmark_context.get("baseline") if isinstance(benchmark_context.get("baseline"), dict) else {}
            lines.append(
                f"Benchmark suite: {str(suite.get('name') or suite.get('id') or payload.get('benchmark_suite_id') or '').strip()} "
                f"({str(suite.get('benchmark_family') or payload.get('benchmark_family') or '').strip() or 'compiler_regression'})"
            )
            case_names = [
                str(item.get("name") or item.get("id") or "").strip()
                for item in benchmark_context.get("selected_cases") if isinstance(item, dict)
            ] if isinstance(benchmark_context.get("selected_cases"), list) else []
            if case_names:
                lines.append(f"Benchmark cases: {', '.join([name for name in case_names if name][:8])}")
            if baseline:
                lines.append(f"Benchmark baseline: {str(baseline.get('name') or baseline.get('id') or '').strip()}")
        elif payload.get("benchmark_suite_id"):
            lines.append(
                f"Benchmark suite: {str(payload.get('benchmark_suite_id')).strip()} "
                f"({str(payload.get('benchmark_family') or '').strip() or 'compiler_regression'})"
            )
            case_ids = payload.get("benchmark_case_ids") if isinstance(payload.get("benchmark_case_ids"), list) else []
            if case_ids:
                lines.append(f"Benchmark cases: {', '.join(str(item) for item in case_ids[:8])}")
            if payload.get("benchmark_baseline_id"):
                lines.append(f"Benchmark baseline: {str(payload.get('benchmark_baseline_id')).strip()}")

        hypothesis_text = "\n".join(lines).strip()
        if max_note_chars and len(hypothesis_text) > int(max_note_chars):
            hypothesis_text = hypothesis_text[: int(max_note_chars)]

        return {
            "hypothesis_text": hypothesis_text,
            "selected_hypotheses": [],
            "selected_hypothesis_ids": [],
            "source_paper_ids": [str(item) for item in source_paper_ids],
            "source_document_ids": [str(item) for item in source_document_ids],
            "supporting_sources": [],
            "benchmark_context": benchmark_context or None,
            "source_run_ids": [str(item) for item in source_run_ids],
            "primary_run_id": primary_run_id or None,
            "comparison_run_id": comparison_run_id or None,
            "regression_type": regression_type,
            "likely_causes": [dict(item) for item in likely_causes if isinstance(item, dict)],
            "recommended_next_steps": [str(item).strip() for item in recommended_next_steps if str(item).strip()],
            "supporting_signals": [str(item).strip() for item in supporting_signals if str(item).strip()],
            "confounders": [str(item).strip() for item in confounders if str(item).strip()],
        }

    hypotheses = _structured_hypotheses(note)
    if not hypotheses:
        return None

    selected: list[dict[str, Any]]
    if plan_mode == "single_hypothesis":
        target = next((item for item in hypotheses if str(item.get("id") or "").strip() == str(hypothesis_id or "").strip()), None)
        if not target:
            raise HTTPException(status_code=400, detail="Unknown hypothesis_id for this research note")
        selected = [target]
    else:
        selected = sorted(
            hypotheses,
            key=lambda item: (
                float(item.get("rank") or 9999),
                -float(item.get("overall_score") or 0),
            ),
        )[:3]

    lines: list[str] = []
    summary = str(payload.get("summary") or "").strip()
    if summary:
        lines.append(f"Note summary: {summary}")

    lines.append(f"Plan mode: {plan_mode}")
    if plan_mode == "aggregate_note":
        lines.append("Use the selected ranked hypotheses as one coordinated program of work.")
    else:
        lines.append("Focus on validating or falsifying the selected hypothesis only.")

    lines.append("Selected hypotheses:")
    for hypothesis in selected:
        title = str(hypothesis.get("title") or hypothesis.get("id") or "Hypothesis").strip()
        lines.append(f"- [{str(hypothesis.get('id') or '').strip() or title}] {title}")
        claim = str(hypothesis.get("claim") or "").strip()
        rationale = str(hypothesis.get("rationale") or "").strip()
        next_step = str(hypothesis.get("recommended_next_step") or "").strip()
        if claim:
            lines.append(f"  Claim: {claim}")
        if rationale:
            lines.append(f"  Rationale: {rationale}")
        if next_step:
            lines.append(f"  Suggested next step: {next_step}")
        sources = hypothesis.get("supporting_sources") if isinstance(hypothesis.get("supporting_sources"), list) else []
        if sources:
            source_text = ", ".join(
                str(src.get("title") or src.get("id") or "source")
                for src in sources[:5]
                if isinstance(src, dict)
            )
            if source_text:
                lines.append(f"  Supporting sources: {source_text}")

    source_paper_ids = payload.get("source_paper_ids") if isinstance(payload.get("source_paper_ids"), list) else []
    source_document_ids = payload.get("source_document_ids") if isinstance(payload.get("source_document_ids"), list) else []
    if source_paper_ids:
        lines.append(f"Source paper ids: {', '.join(str(item) for item in source_paper_ids[:20])}")
    if source_document_ids:
        lines.append(f"Source document ids: {', '.join(str(item) for item in source_document_ids[:20])}")
    if isinstance(benchmark_context, dict):
        suite = benchmark_context.get("suite") if isinstance(benchmark_context.get("suite"), dict) else {}
        baseline = benchmark_context.get("baseline") if isinstance(benchmark_context.get("baseline"), dict) else {}
        lines.append(
            f"Benchmark suite: {str(suite.get('name') or suite.get('id') or '').strip()} "
            f"({str(suite.get('benchmark_family') or '').strip() or 'compiler_regression'})"
        )
        case_names = [
            str(item.get("name") or item.get("id") or "").strip()
            for item in benchmark_context.get("selected_cases") if isinstance(item, dict)
        ] if isinstance(benchmark_context.get("selected_cases"), list) else []
        if case_names:
            lines.append(f"Benchmark cases: {', '.join([name for name in case_names if name][:8])}")
        if baseline:
            lines.append(f"Benchmark baseline: {str(baseline.get('name') or baseline.get('id') or '').strip()}")

    hypothesis_text = "\n".join(lines).strip()
    if max_note_chars and len(hypothesis_text) > int(max_note_chars):
        hypothesis_text = hypothesis_text[: int(max_note_chars)]

    supporting_sources: list[dict[str, Any]] = []
    for hypothesis in selected:
        sources = hypothesis.get("supporting_sources") if isinstance(hypothesis.get("supporting_sources"), list) else []
        for source in sources:
            if not isinstance(source, dict):
                continue
            key = str(source.get("id") or source.get("title") or "").strip()
            if not key or any(str(existing.get("id") or existing.get("title") or "").strip() == key for existing in supporting_sources):
                continue
            supporting_sources.append(dict(source))

    return {
        "hypothesis_text": hypothesis_text,
        "selected_hypotheses": selected,
        "selected_hypothesis_ids": [str(item.get("id") or "").strip() for item in selected if str(item.get("id") or "").strip()],
        "source_paper_ids": [str(item) for item in source_paper_ids],
        "source_document_ids": [str(item) for item in source_document_ids],
        "supporting_sources": supporting_sources,
        "benchmark_context": benchmark_context or None,
    }


def _build_experiment_plan_prompt(
    note_title: str,
    hypothesis_text: str,
    include: Dict[str, bool],
    *,
    plan_mode: str = "aggregate_note",
    selected_hypothesis_ids: Optional[list[str]] = None,
    source_paper_ids: Optional[list[str]] = None,
    source_document_ids: Optional[list[str]] = None,
    supporting_sources: Optional[list[dict[str, Any]]] = None,
    benchmark_context: Optional[dict[str, Any]] = None,
) -> str:
    """
    Prompt for structured experiment plan generation.

    Output is strict JSON (no markdown) that front-end can render and users can edit.
    """
    sections = []
    sections.append("You are an AI research engineer. Create a runnable experiment plan from the hypothesis.")
    sections.append("Return ONLY valid JSON. No markdown, no commentary.")
    sections.append(
        "JSON schema (high level): {\n"
        '  "objective": string,\n'
        '  "hypothesis": string,\n'
        '  "hypotheses": [{"id": string, "title": string, "claim": string}] | [],\n'
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
        '  "repro_checklist": [string] | [],\n'
        '  "plan_scope": "aggregate_note"|"single_hypothesis"|"compiler_regression_followup",\n'
        '  "selected_hypothesis_ids": [string],\n'
        '  "supporting_sources": [{"id": string|null, "title": string|null}] | [],\n'
        '  "provenance": {"source_paper_ids": [string], "source_document_ids": [string], "benchmark_suite_id": string|null, "benchmark_case_ids": [string], "benchmark_baseline_id": string|null, "benchmark_family": string|null},\n'
        '  "benchmark_family": string|null,\n'
        '  "benchmark_suite_id": string|null,\n'
        '  "benchmark_case_ids": [string],\n'
        '  "benchmark_baseline_id": string|null,\n'
        '  "benchmark_measurements": {"focus_metrics": [string], "artifact_expectations": [string]} | null\n'
        "}"
    )
    sections.append(f"Note title: {note_title}")
    sections.append("Hypothesis section (may be short):")
    sections.append(hypothesis_text)
    sections.append(f"Requested plan scope: {plan_mode}")
    if selected_hypothesis_ids:
        sections.append(f"Selected hypothesis ids: {', '.join(selected_hypothesis_ids)}")
    if supporting_sources:
        sections.append(f"Supporting sources: {json.dumps(supporting_sources)}")
    if source_paper_ids or source_document_ids:
        sections.append(
            "Provenance: "
            + json.dumps(
                {
                    "source_paper_ids": source_paper_ids or [],
                    "source_document_ids": source_document_ids or [],
                }
            )
        )
    if benchmark_context:
        sections.append("Benchmark harness context: " + json.dumps(benchmark_context))

    # Constraints
    sections.append("Rules:")
    sections.append("- Keep it concrete: include at least 3 experiments and 2 metrics.")
    sections.append("- Prefer minimal feasible baselines.")
    sections.append("- Always return plan_scope, selected_hypothesis_ids, supporting_sources, and provenance.")
    sections.append("- If benchmark context is provided, carry benchmark_family, benchmark_suite_id, benchmark_case_ids, benchmark_baseline_id, and benchmark_measurements into the plan.")
    sections.append("- In aggregate_note mode, coordinate the top hypotheses into a staged evaluation plan.")
    sections.append("- In single_hypothesis mode, optimize the plan for validating or falsifying the chosen hypothesis.")
    sections.append("- In compiler_regression_followup mode, optimize the plan to isolate or falsify the likely regression causes from the explanation and make the compared benchmark runs directly actionable.")
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


@router.get("/benchmark-suites", response_model=BenchmarkSuiteListResponse)
async def list_benchmark_harness_suites(
    track_type: Optional[str] = Query(default="compiler"),
    include_disabled: bool = Query(default=False),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    del current_user
    items = [
        BenchmarkSuiteResponse.model_validate(item)
        for item in await list_benchmark_suites(db, track_type=track_type, include_disabled=include_disabled)
        if isinstance(item, dict)
    ]
    return BenchmarkSuiteListResponse(items=items, total=len(items))


@router.get("/benchmark-suites/{suite_id}", response_model=BenchmarkSuiteResponse)
async def get_benchmark_harness_suite(
    suite_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    del current_user
    suite = await get_benchmark_suite(db, suite_id)
    if not isinstance(suite, dict):
        raise HTTPException(status_code=404, detail="Benchmark suite not found")
    return BenchmarkSuiteResponse.model_validate(suite)


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
    is_reevaluated_note = _is_reevaluated_hypothesis_note(note)
    is_compiler_explanation_note = _is_compiler_regression_explanation_note(note)
    auto_selected_hypothesis_id = _top_ranked_hypothesis_id(note) if is_reevaluated_note else None
    note_payload = note.structured_payload if isinstance(note.structured_payload, dict) else {}
    benchmark_context = await _build_benchmark_context_for_request(
        db=db,
        benchmark_suite_id=request.benchmark_suite_id or (
            str(note_payload.get("benchmark_suite_id") or "").strip() if is_compiler_explanation_note else None
        ),
        benchmark_case_ids=request.benchmark_case_ids or (
            [str(item) for item in note_payload.get("benchmark_case_ids")] if is_compiler_explanation_note and isinstance(note_payload.get("benchmark_case_ids"), list) else []
        ),
    )
    plan_mode = request.plan_mode or (
        "compiler_regression_followup"
        if is_compiler_explanation_note
        else ("single_hypothesis" if auto_selected_hypothesis_id else "aggregate_note")
    )
    effective_hypothesis_id = str(request.hypothesis_id or "").strip() or (auto_selected_hypothesis_id if plan_mode == "single_hypothesis" else "")
    if plan_mode == "single_hypothesis" and not effective_hypothesis_id:
        raise HTTPException(status_code=400, detail="single_hypothesis planning requires hypothesis_id")
    structured_context = _build_structured_experiment_context(
        note,
        plan_mode=plan_mode,
        hypothesis_id=effective_hypothesis_id,
        max_note_chars=request.max_note_chars,
        benchmark_context=benchmark_context,
    )
    if effective_hypothesis_id and structured_context is None:
        raise HTTPException(status_code=400, detail="This research note does not contain structured hypotheses")

    if structured_context is not None:
        hypothesis_text = structured_context["hypothesis_text"]
    else:
        if request.prefer_section == "hypothesis":
            hypothesis_text = _extract_hypothesis_section(content) or content
        else:
            hypothesis_text = content

        hypothesis_text = (hypothesis_text or "").strip()
        if request.max_note_chars and len(hypothesis_text) > int(request.max_note_chars):
            hypothesis_text = hypothesis_text[: int(request.max_note_chars)]
        if isinstance(benchmark_context, dict):
            suite = benchmark_context.get("suite") if isinstance(benchmark_context.get("suite"), dict) else {}
            case_names = [
                str(item.get("name") or item.get("id") or "").strip()
                for item in (benchmark_context.get("selected_cases") if isinstance(benchmark_context.get("selected_cases"), list) else [])
                if isinstance(item, dict)
            ]
            benchmark_lines = [
                f"Benchmark suite: {str(suite.get('name') or suite.get('id') or '').strip()}",
                f"Benchmark family: {str(suite.get('benchmark_family') or '').strip()}",
            ]
            if case_names:
                benchmark_lines.append(f"Benchmark cases: {', '.join([name for name in case_names if name][:8])}")
            hypothesis_text = "\n".join([hypothesis_text, "", *benchmark_lines]).strip()

    include = {
        "ablations": bool(request.include_ablations),
        "timeline": bool(request.include_timeline),
        "risks": bool(request.include_risks),
        "repro_checklist": bool(request.include_repro_checklist),
    }

    llm = LLMService()
    prompt = _build_experiment_plan_prompt(
        note.title,
        hypothesis_text,
        include,
        plan_mode=plan_mode,
        selected_hypothesis_ids=(structured_context or {}).get("selected_hypothesis_ids"),
        source_paper_ids=(structured_context or {}).get("source_paper_ids"),
        source_document_ids=(structured_context or {}).get("source_document_ids"),
        supporting_sources=(structured_context or {}).get("supporting_sources"),
        benchmark_context=benchmark_context,
    )

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

    parsed["plan_scope"] = plan_mode
    parsed["selected_hypothesis_ids"] = (structured_context or {}).get("selected_hypothesis_ids", [])
    parsed["supporting_sources"] = (structured_context or {}).get("supporting_sources", [])
    parsed["provenance"] = {
        "source_paper_ids": (structured_context or {}).get("source_paper_ids", []),
        "source_document_ids": (structured_context or {}).get("source_document_ids", []),
        "benchmark_suite_id": str((benchmark_context or {}).get("suite", {}).get("id") or "").strip() or None,
        "benchmark_case_ids": (benchmark_context or {}).get("selected_case_ids", []),
        "benchmark_baseline_id": str((benchmark_context or {}).get("baseline", {}).get("id") or "").strip() or None,
        "benchmark_family": str((benchmark_context or {}).get("suite", {}).get("benchmark_family") or "").strip() or None,
    }
    parsed["benchmark_family"] = str((benchmark_context or {}).get("suite", {}).get("benchmark_family") or parsed.get("benchmark_family") or "").strip() or None
    parsed["benchmark_suite_id"] = str((benchmark_context or {}).get("suite", {}).get("id") or parsed.get("benchmark_suite_id") or "").strip() or None
    parsed["benchmark_case_ids"] = (benchmark_context or {}).get("selected_case_ids", [])
    parsed["benchmark_baseline_id"] = str((benchmark_context or {}).get("baseline", {}).get("id") or parsed.get("benchmark_baseline_id") or "").strip() or None
    parsed["benchmark_measurements"] = parsed.get("benchmark_measurements") if isinstance(parsed.get("benchmark_measurements"), dict) else {
        "focus_metrics": [
            str(metric.get("name") or "").strip()
            for case in ((benchmark_context or {}).get("selected_cases") if isinstance((benchmark_context or {}).get("selected_cases"), list) else [])
            for metric in (case.get("metrics") if isinstance(case, dict) and isinstance(case.get("metrics"), list) else [])
            if isinstance(metric, dict) and str(metric.get("name") or "").strip()
        ][:8],
        "artifact_expectations": [
            str(item).strip()
            for case in ((benchmark_context or {}).get("selected_cases") if isinstance((benchmark_context or {}).get("selected_cases"), list) else [])
            for item in (case.get("expected_artifacts") if isinstance(case, dict) and isinstance(case.get("expected_artifacts"), list) else [])
            if str(item).strip()
        ][:8],
    }

    plan = ExperimentPlan(
        user_id=current_user.id,
        research_note_id=note.id,
        title=(
            f"Experiment Plan: {note.title}"
            if plan_mode == "aggregate_note"
            else (
                f"Experiment Plan: {note.title} · Regression Follow-up"
                if plan_mode == "compiler_regression_followup"
                else f"Experiment Plan: {note.title} · {((structured_context or {}).get('selected_hypotheses') or [{}])[0].get('title') or effective_hypothesis_id or 'Hypothesis'}"
            )
        ),
        hypothesis_text=hypothesis_text if request.prefer_section == "hypothesis" else None,
        plan=parsed,
        generator="llm",
        generator_details={
            "generated_at": datetime.utcnow().isoformat(),
            "plan_mode": plan_mode,
            "hypothesis_id": effective_hypothesis_id or None,
            "selected_hypothesis_ids": (structured_context or {}).get("selected_hypothesis_ids", []),
            "source_paper_ids": (structured_context or {}).get("source_paper_ids", []),
            "source_document_ids": (structured_context or {}).get("source_document_ids", []),
            "supporting_sources": (structured_context or {}).get("supporting_sources", []),
            "benchmark_family": str((benchmark_context or {}).get("suite", {}).get("benchmark_family") or "").strip() or None,
            "benchmark_suite_id": str((benchmark_context or {}).get("suite", {}).get("id") or "").strip() or None,
            "benchmark_case_ids": (benchmark_context or {}).get("selected_case_ids", []),
            "benchmark_baseline_id": str((benchmark_context or {}).get("baseline", {}).get("id") or "").strip() or None,
            "benchmark_suite_name": str((benchmark_context or {}).get("suite", {}).get("name") or "").strip() or None,
            "benchmark_case_names": [
                str(case.get("name") or "").strip()
                for case in ((benchmark_context or {}).get("selected_cases") if isinstance((benchmark_context or {}).get("selected_cases"), list) else [])
                if isinstance(case, dict) and str(case.get("name") or "").strip()
            ],
            "benchmark_default_commands": (benchmark_context or {}).get("default_commands", []),
            "reevaluation_mode": bool(is_reevaluated_note),
            "reevaluation_source_job_id": (
                str((((note.structured_payload or {}).get("scoring_policy") or {}).get("source_job_id") or "")).strip() or None
                if isinstance(note.structured_payload, dict)
                else None
            ),
            "explanation_mode": bool(is_compiler_explanation_note),
            "source_run_ids": (structured_context or {}).get("source_run_ids", []),
            "primary_run_id": (structured_context or {}).get("primary_run_id"),
            "comparison_run_id": (structured_context or {}).get("comparison_run_id"),
            "regression_type": (structured_context or {}).get("regression_type"),
            "likely_causes": (structured_context or {}).get("likely_causes", []),
            "recommended_next_steps": (structured_context or {}).get("recommended_next_steps", []),
        },
    )
    db.add(plan)
    await db.commit()
    await db.refresh(plan)
    return _plan_to_response(plan)


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
    return ExperimentPlanListResponse(plans=[_plan_to_response(p) for p in plans])


@router.get("/plans/{plan_id}", response_model=ExperimentPlanResponse)
async def get_experiment_plan(
    plan_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    plan = await db.get(ExperimentPlan, plan_id)
    if not plan or plan.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Experiment plan not found")
    return _plan_to_response(plan)


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
    return _plan_to_response(plan)


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

    run_name = str(request.name or "").strip() or _default_run_name_from_plan(plan)
    run_summary = request.summary if isinstance(request.summary, str) and request.summary.strip() else _default_run_summary_from_plan(plan)
    run_config = _merge_run_config_with_plan_handoff(plan, request.config)
    scientific_validation = await _build_scientific_validation_for_run(
        db=db,
        plan=plan,
        run_config=run_config,
    )
    if scientific_validation:
        run_config["scientific_validation"] = scientific_validation
        if not run_summary:
            run_summary = str(scientific_validation.get("decision_summary") or "").strip() or run_summary

    run = ExperimentRun(
        user_id=current_user.id,
        experiment_plan_id=plan.id,
        name=run_name,
        config=run_config or None,
        summary=run_summary,
        status="planned",
        progress=0,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)
    return _run_to_response(run)


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
    return ExperimentRunListResponse(runs=[_run_to_response(r) for r in runs])


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
    plan = await db.get(ExperimentPlan, run.experiment_plan_id)
    if not plan or plan.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Experiment plan not found")

    # Auto-set timestamps for status transitions if not provided
    next_status = data.get("status")
    if next_status and next_status != run.status:
        if next_status == "running" and data.get("started_at") is None:
            data["started_at"] = datetime.utcnow()
        if next_status in {"succeeded", "completed", "failed", "blocked", "cancelled"} and data.get("completed_at") is None:
            data["completed_at"] = datetime.utcnow()

    for k, v in data.items():
        setattr(run, k, v)

    await reconcile_experiment_run_outcome_to_originating_opportunity(
        db,
        run=run,
        plan=plan,
        recorded_at=getattr(run, "completed_at", None) or datetime.utcnow(),
    )
    await db.commit()
    await db.refresh(run)
    return _run_to_response(run)


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

    run, job = await _start_experiment_run_internal(
        run=run,
        plan=plan,
        current_user=current_user,
        db=db,
        source_id=request.source_id,
        commands=_run_start_commands(run, request.commands or []),
        timeout_seconds=int(request.timeout_seconds),
        latex_project_id=request.latex_project_id,
        start_immediately=bool(request.start_immediately),
    )
    await db.commit()
    await db.refresh(job)
    await db.refresh(run)
    return ExperimentRunStartResponse(run=_run_to_response(run), agent_job_id=job.id)


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

    plan = await db.get(ExperimentPlan, run.experiment_plan_id)
    if not plan or plan.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Experiment plan not found")

    job = await db.get(AgentJob, run.agent_job_id)
    if not job or job.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Linked agent job not found")

    await _sync_experiment_run_from_job_internal(run=run, job=job)
    await _maybe_auto_append_experiment_run_to_note(
        run=run,
        plan=plan,
        db=db,
        current_user=current_user,
    )
    await reconcile_experiment_run_outcome_to_originating_opportunity(
        db,
        run=run,
        plan=plan,
        recorded_at=getattr(run, "completed_at", None) or datetime.utcnow(),
    )
    await db.commit()
    await db.refresh(run)
    return ExperimentRunSyncResponse(run=_run_to_response(run))


@router.post("/runs/{run_id}/action", response_model=ExperimentRunActionResponse)
async def act_on_experiment_run(
    run_id: UUID,
    request: ExperimentRunActionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    run = await db.get(ExperimentRun, run_id)
    if not run or run.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Experiment run not found")

    plan = await db.get(ExperimentPlan, run.experiment_plan_id)
    if not plan or plan.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Experiment plan not found")

    action = str(request.action or "").strip().lower()
    note = str(request.note or "").strip() or None
    previous_status = str(run.status or "")
    scientific_validation = _scientific_validation_payload(run)
    blocked_reason_code = str(scientific_validation.get("blocked_reason_code") or scientific_validation.get("blocked_reason") or "").strip() or None
    is_scientific = bool(scientific_validation)
    agent_job_id: UUID | None = None

    if action == "start":
        if not is_scientific:
            raise HTTPException(status_code=400, detail="Run controls start is only supported for scientific validation runs")
        source_id = str((run.config or {}).get("source_id") or "").strip()
        commands = (run.config or {}).get("commands") if isinstance((run.config or {}).get("commands"), list) else []
        timeout_seconds = int((run.config or {}).get("timeout_seconds") or 60)
        if not source_id:
            raise HTTPException(status_code=400, detail="Scientific validation run is missing source_id")
        run, job = await _start_experiment_run_internal(
            run=run,
            plan=plan,
            current_user=current_user,
            db=db,
            source_id=source_id,
            commands=commands,
            timeout_seconds=timeout_seconds,
            latex_project_id=(run.config or {}).get("latex_project_id"),
            start_immediately=bool(request.start_immediately),
        )
        agent_job_id = job.id
        _append_run_operator_action(
            run,
            action="start",
            current_user=current_user,
            note=note,
            previous_status="planned",
            new_status=run.status,
            linked_job_id=job.id,
            outcome_status="applied",
            outcome_reason="Scientific validation run queued",
        )
    elif action == "sync":
        if not run.agent_job_id:
            raise HTTPException(status_code=400, detail="Run has no linked agent job")
        job = await db.get(AgentJob, run.agent_job_id)
        if not job or job.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Linked agent job not found")
        previous_status = run.status
        await _sync_experiment_run_from_job_internal(run=run, job=job)
        _append_run_operator_action(
            run,
            action="sync",
            current_user=current_user,
            note=note,
            previous_status=previous_status,
            new_status=run.status,
            linked_job_id=job.id,
            outcome_status="applied",
            outcome_reason="Run synchronized from linked agent job",
        )
    elif action in {"pause", "resume", "cancel"}:
        if not is_scientific:
            raise HTTPException(status_code=400, detail="Run controls are only supported for scientific validation runs")
        if not run.agent_job_id:
            raise HTTPException(status_code=400, detail="Run has no linked agent job")
        job = await db.get(AgentJob, run.agent_job_id)
        if not job or job.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Linked agent job not found")
        previous_status = run.status
        updated_job = await _perform_job_action(
            job,
            AgentJobActionRequest(action=action, checkpoint_note=note),
            db=db,
            current_user=current_user,
        )
        agent_job_id = updated_job.id
        if action == "cancel":
            run.status = "cancelled"
            run.completed_at = run.completed_at or datetime.utcnow()
        elif action == "pause":
            run.status = "paused"
        elif action == "resume":
            run.status = "queued"
            run.completed_at = None
        _append_run_operator_action(
            run,
            action=action,
            current_user=current_user,
            note=note,
            previous_status=previous_status,
            new_status=run.status,
            linked_job_id=updated_job.id,
            linked_job_action=action,
            outcome_status="applied",
            outcome_reason=f"Linked agent job {action} applied",
        )
    elif action in {"retry", "requeue"}:
        if not is_scientific:
            raise HTTPException(status_code=400, detail="Run controls are only supported for scientific validation runs")
        if action == "retry" and run.status not in {"succeeded", "completed", "failed", "blocked", "cancelled"}:
            raise HTTPException(status_code=400, detail="Retry requires a terminal scientific validation run")
        if action == "requeue" and run.status not in {"planned", "blocked"}:
            raise HTTPException(status_code=400, detail="Requeue is only allowed for planned or blocked scientific validation runs")
        linked_job = await db.get(AgentJob, run.agent_job_id) if run.agent_job_id else None
        if linked_job is not None and str(linked_job.user_id) != str(current_user.id):
            linked_job = None
        source_scheduler_state = _extract_scheduler_state(linked_job) if linked_job is not None else None
        child_run = _spawn_child_experiment_run(
            source_run=run,
            current_user=current_user,
            action=action,
            note=note,
        )
        db.add(child_run)
        await db.flush()
        run.latest_child_run_id = child_run.id
        child_actions = _run_operator_actions(child_run)
        if child_actions:
            child_actions[0]["child_run_id"] = str(child_run.id)
            scientific_validation = _scientific_validation_payload(child_run)
            scientific_validation["operator_actions"] = child_actions
            _set_scientific_validation_payload(child_run, scientific_validation)
        _append_run_operator_action(
            run,
            action=action,
            current_user=current_user,
            note=note,
            previous_status=run.status,
            new_status=run.status,
            outcome_status="applied",
            outcome_reason=f"Spawned child run {child_run.id}",
            child_run_id=child_run.id,
        )
        if request.start_immediately:
            source_id = str((child_run.config or {}).get("source_id") or "").strip()
            commands = (child_run.config or {}).get("commands") if isinstance((child_run.config or {}).get("commands"), list) else []
            timeout_seconds = int((child_run.config or {}).get("timeout_seconds") or 60)
            if not source_id:
                raise HTTPException(status_code=400, detail="Scientific validation child run is missing source_id")
            child_run, job = await _start_experiment_run_internal(
                run=child_run,
                plan=plan,
                current_user=current_user,
                db=db,
                source_id=source_id,
                commands=commands,
                timeout_seconds=timeout_seconds,
                latex_project_id=(child_run.config or {}).get("latex_project_id"),
                start_immediately=True,
            )
            agent_job_id = job.id
            child_actions = _run_operator_actions(child_run)
            if child_actions:
                child_actions.append(
                    {
                        "action": "start",
                        "actor_user_id": str(current_user.id),
                        "at": datetime.utcnow().isoformat(),
                        "note": note,
                        "previous_status": "planned",
                        "new_status": child_run.status,
                        "linked_job_id": str(job.id),
                        "linked_job_action": None,
                        "outcome_status": "applied",
                        "outcome_reason": "Child scientific validation run queued",
                        "parent_run_id": str(run.id),
                        "child_run_id": str(child_run.id),
                    }
                )
                scientific_validation = _scientific_validation_payload(child_run)
                scientific_validation["operator_actions"] = child_actions[-50:]
                _set_scientific_validation_payload(child_run, scientific_validation)
        run = child_run
    else:
        raise HTTPException(status_code=400, detail="Unsupported experiment run action")

    await _maybe_auto_append_experiment_run_to_note(
        run=run,
        plan=plan,
        db=db,
        current_user=current_user,
    )
    await reconcile_experiment_run_outcome_to_originating_opportunity(
        db,
        run=run,
        plan=plan,
        recorded_at=getattr(run, "completed_at", None) or datetime.utcnow(),
    )
    await record_autonomy_decision_event(
        db,
        user_id=current_user.id,
        event_type="validation_requeued" if action in {"retry", "requeue"} else "validation_operator_action",
        event_time=datetime.utcnow(),
        source_kind="validation_run",
        source_id=str(run.id),
        source_label=str(run.name or "Experiment run").strip(),
        decision_type="validation_requeued" if action in {"retry", "requeue"} else action,
        reason_code=blocked_reason_code or action,
        status=str(run.status or "").strip() or None,
        severity="high" if str(run.status or "").strip().lower() == "blocked" else "medium",
        actor_mode="operator",
        summary=f"{str(run.name or 'Experiment run').strip()}: {action.replace('_', ' ')}",
        operator_note=note,
        reason_label=("Validation requeued" if action in {"retry", "requeue"} else str(blocked_reason_code or action).replace("_", " ").strip().capitalize()),
        scheduler_state=source_scheduler_state,
        before_state={"status": previous_status},
        after_state={"status": str(run.status or "").strip()},
        deep_link={"target_tab": "jobs", "job_id": str(run.agent_job_id) if run.agent_job_id else None, "params": {"job": str(run.agent_job_id)} if run.agent_job_id else {}, "label": "Open Validation Job"},
        metadata={"experiment_plan_id": str(plan.id), "linked_job_id": str(agent_job_id) if agent_job_id else None},
    )
    await db.commit()
    await db.refresh(run)
    return ExperimentRunActionResponse(run=_run_to_response(run), agent_job_id=agent_job_id)


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

    note, new_evidence_added = await _append_experiment_run_to_note_internal(
        run=run,
        plan=plan,
        note=note,
        appended_at=datetime.utcnow().isoformat(),
    )
    if new_evidence_added:
        await _queue_pending_hypothesis_reevaluation_draft(
            note=note,
            run=run,
            db=db,
            current_user=current_user,
        )
    await reconcile_experiment_run_outcome_to_originating_opportunity(
        db,
        run=run,
        plan=plan,
        recorded_at=getattr(run, "completed_at", None) or datetime.utcnow(),
    )
    await db.commit()
    await db.refresh(run)
    await db.refresh(note)
    return ResearchNoteResponse.model_validate(note, from_attributes=True)
