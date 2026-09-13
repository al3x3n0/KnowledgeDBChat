"""Rebuild coding quick starts and recovery context from persisted job state."""

from typing import Any, Optional, TypeVar

from app.schemas.agent_job import (
    AgentJobQuickStartBugTriageSwarmRequest,
    AgentJobQuickStartBuildBreakSwarmRequest,
    AgentJobQuickStartFrontendRegressionSwarmRequest,
    AgentJobQuickStartRepoBugTriageRequest,
)
from app.services.agent_scope_service import normalize_scope_config

CodingSwarmRequest = (
    AgentJobQuickStartBugTriageSwarmRequest
    | AgentJobQuickStartBuildBreakSwarmRequest
    | AgentJobQuickStartFrontendRegressionSwarmRequest
)
CodingSwarmRequestT = TypeVar(
    "CodingSwarmRequestT",
    AgentJobQuickStartBugTriageSwarmRequest,
    AgentJobQuickStartBuildBreakSwarmRequest,
    AgentJobQuickStartFrontendRegressionSwarmRequest,
)


def _job_config(job: Any) -> dict[str, Any]:
    config = getattr(job, "config", None)
    return normalize_scope_config(config if isinstance(config, dict) else {}) or {}


def _launch_mode(config: dict[str, Any]) -> str:
    return str(config.get("launch_mode") or "").strip().lower()


def _source_id(config: dict[str, Any]) -> str:
    return str(config.get("source_id") or "").strip()


def _relaunch_overrides(
    job: Any,
    config: dict[str, Any],
    *,
    reserved: set[str],
) -> dict[str, Any]:
    overrides = {key: value for key, value in config.items() if key not in reserved}
    overrides["relaunch_from_job_id"] = str(getattr(job, "id", "") or "").strip()
    return overrides


def build_repo_bug_triage_relaunch_request(
    job: Any,
    *,
    retry_strategy: str = "clean_relaunch",
) -> Optional[AgentJobQuickStartRepoBugTriageRequest]:
    """Rebuild repository bug-triage state, optionally with retry recovery."""
    config = _job_config(job)
    if _launch_mode(config) != "quick_start_repo_bug_triage":
        return None

    source_id = _source_id(config)
    goal = str(getattr(job, "goal", "") or "").strip() or None
    failure_symptom = str(config.get("failure_symptom") or "").strip() or None
    if not source_id or (not goal and not failure_symptom):
        return None

    overrides = _relaunch_overrides(
        job,
        config,
        reserved={
            "source_id",
            "target_source_id",
            "launch_mode",
            "quick_start",
            "failure_symptom",
            "scope",
            "search_query",
            "commands",
            "file_paths",
            "error_output",
        },
    )
    request_error_output = (
        str(config.get("error_output")).strip()
        if config.get("error_output") is not None
        else None
    )
    coding_recovery = extract_repo_bug_triage_coding_recovery(job)
    if retry_strategy and retry_strategy != "clean_relaunch":
        overrides["coding_recovery"] = {
            "strategy": retry_strategy,
            "retry_reason": str(coding_recovery.get("retry_reason") or "").strip()
            or None,
            "resume_hint": str(coding_recovery.get("resume_hint") or "").strip()
            or None,
            "last_failed_commands": [
                str(command).strip()
                for command in (coding_recovery.get("last_failed_commands") or [])
                if str(command).strip()
            ][:6],
            "suggested_operator_actions": [
                str(action).strip()
                for action in (coding_recovery.get("suggested_operator_actions") or [])
                if str(action).strip()
            ][:6],
        }
        latest_failed_output = str(
            coding_recovery.get("latest_failed_output") or ""
        ).strip()
        if latest_failed_output:
            overrides["error_output"] = latest_failed_output[:4000]
            request_error_output = latest_failed_output[:4000]

    try:
        return AgentJobQuickStartRepoBugTriageRequest(
            name=str(getattr(job, "name", "") or "").strip() or None,
            goal=goal,
            failure_symptom=failure_symptom,
            source_id=source_id,
            scope=str(config.get("scope") or "auto").strip().lower() or "auto",
            search_query=(
                str(config.get("search_query")).strip()
                if config.get("search_query") is not None
                else None
            ),
            file_paths=(
                config.get("file_paths")
                if isinstance(config.get("file_paths"), list)
                else None
            ),
            commands=(
                config.get("commands")
                if isinstance(config.get("commands"), list)
                else None
            ),
            error_output=request_error_output,
            start_immediately=True,
            config_overrides=overrides or None,
        )
    except Exception:
        return None


def build_bug_triage_swarm_relaunch_request(
    job: Any,
) -> Optional[AgentJobQuickStartBugTriageSwarmRequest]:
    relaunch = build_coding_swarm_relaunch_request(
        job,
        launch_mode="quick_start_bug_triage_swarm",
        request_cls=AgentJobQuickStartBugTriageSwarmRequest,
    )
    return (
        relaunch
        if isinstance(relaunch, AgentJobQuickStartBugTriageSwarmRequest)
        else None
    )


def build_build_break_swarm_relaunch_request(
    job: Any,
) -> Optional[AgentJobQuickStartBuildBreakSwarmRequest]:
    relaunch = build_coding_swarm_relaunch_request(
        job,
        launch_mode="quick_start_build_break_swarm",
        request_cls=AgentJobQuickStartBuildBreakSwarmRequest,
    )
    return (
        relaunch
        if isinstance(relaunch, AgentJobQuickStartBuildBreakSwarmRequest)
        else None
    )


def build_frontend_regression_swarm_relaunch_request(
    job: Any,
) -> Optional[AgentJobQuickStartFrontendRegressionSwarmRequest]:
    relaunch = build_coding_swarm_relaunch_request(
        job,
        launch_mode="quick_start_frontend_regression_swarm",
        request_cls=AgentJobQuickStartFrontendRegressionSwarmRequest,
    )
    return (
        relaunch
        if isinstance(relaunch, AgentJobQuickStartFrontendRegressionSwarmRequest)
        else None
    )


def build_coding_swarm_relaunch_request(
    job: Any,
    *,
    launch_mode: str,
    request_cls: type[CodingSwarmRequestT],
) -> Optional[CodingSwarmRequestT]:
    """Rebuild a typed coding-swarm quick-start request."""
    config = _job_config(job)
    if _launch_mode(config) != launch_mode:
        return None

    source_id = _source_id(config)
    goal = str(getattr(job, "goal", "") or "").strip() or None
    failure_symptom = str(config.get("failure_symptom") or "").strip() or None
    if not source_id or (not goal and not failure_symptom):
        return None

    overrides = _relaunch_overrides(
        job,
        config,
        reserved={
            "source_id",
            "target_source_id",
            "launch_mode",
            "quick_start",
            "failure_symptom",
            "scope",
            "search_query",
            "commands",
            "file_paths",
            "error_output",
            "plan_then_act_enabled",
            "plan_max_steps",
            "subgoal_decomposition_enabled",
            "swarm_child_jobs_enabled",
            "swarm_max_agents",
            "swarm_roles",
            "swarm_inherit_results",
            "swarm_inherit_config",
            "swarm_fan_in_enabled",
            "swarm_fan_in_name",
            "swarm_fan_in_trigger_condition",
            "coding_swarm_enabled",
            "coding_swarm_profile",
            "coding_swarm_preset_key",
            "coding_swarm_auto_promote_best_slice",
            "coding_swarm_auto_launch_repair_chain",
            "coding_swarm_confidence_threshold",
            "coding_swarm_tiebreaker_threshold",
            "coding_swarm_repair_chain_name",
            "create_workspace_from_source",
            "emit_execution_plan",
            "auto_commands_from_project_profile",
            "max_verification_commands",
            "apply_patch_to_kb",
            "apply_patch_to_kb_confirm",
            "enable_memory",
        },
    )
    quick_start = (
        config.get("quick_start") if isinstance(config.get("quick_start"), dict) else {}
    )

    try:
        return request_cls(
            name=str(getattr(job, "name", "") or "").strip() or None,
            goal=goal,
            failure_symptom=failure_symptom,
            source_id=source_id,
            scope=str(config.get("scope") or quick_start.get("scope") or "auto")
            .strip()
            .lower()
            or "auto",
            search_query=(
                str(config.get("search_query")).strip()
                if config.get("search_query") is not None
                else None
            ),
            file_paths=(
                config.get("file_paths")
                if isinstance(config.get("file_paths"), list)
                else None
            ),
            commands=(
                config.get("commands")
                if isinstance(config.get("commands"), list)
                else None
            ),
            error_output=(
                str(config.get("error_output")).strip()
                if config.get("error_output") is not None
                else None
            ),
            max_agents=int(
                quick_start.get("max_agents") or config.get("swarm_max_agents") or 4
            ),
            profile_id=quick_start.get("profile_id"),
            start_immediately=True,
            config_overrides=overrides or None,
        )
    except Exception:
        return None


def extract_repo_bug_triage_coding_recovery(job: Any) -> dict[str, Any]:
    """Interpret persisted coding execution results as operator recovery state."""
    results = getattr(job, "results", None)
    results = results if isinstance(results, dict) else {}
    code_execution = (
        results.get("code_patch_execution")
        if isinstance(results.get("code_patch_execution"), dict)
        else {}
    )
    recovery = (
        code_execution.get("recovery")
        if isinstance(code_execution.get("recovery"), dict)
        else {}
    )
    experiment_run = (
        results.get("experiment_run")
        if isinstance(results.get("experiment_run"), dict)
        else {}
    )
    execution_strategy = (
        results.get("execution_strategy")
        if isinstance(results.get("execution_strategy"), dict)
        else {}
    )
    execution_graph = (
        execution_strategy.get("execution_graph")
        if isinstance(execution_strategy.get("execution_graph"), dict)
        else {}
    )
    graph_health = (
        execution_graph.get("graph_health")
        if isinstance(execution_graph.get("graph_health"), dict)
        else {}
    )
    graph_reasons = (
        graph_health.get("reasons")
        if isinstance(graph_health.get("reasons"), list)
        else []
    )
    failed_commands = [
        str(command).strip()
        for command in (
            experiment_run.get("failed_commands")
            if isinstance(experiment_run.get("failed_commands"), list)
            else []
        )
        if str(command).strip()
    ]

    latest_failed_output = ""
    runs = (
        experiment_run.get("runs")
        if isinstance(experiment_run.get("runs"), list)
        else []
    )
    for row in reversed(runs):
        if not isinstance(row, dict) or bool(row.get("ok")):
            continue
        latest_failed_output = str(row.get("stderr") or row.get("stdout") or "").strip()
        if latest_failed_output:
            break

    retry_reason = (
        str(recovery.get("retry_reason") or "").strip()
        or (str(graph_reasons[0]).strip() if graph_reasons else "")
        or ("Verification failed and needs a refined retry." if failed_commands else "")
    )
    can_resume = bool(recovery.get("can_resume_verification"))
    if not can_resume and str(getattr(job, "status", "") or "").lower() == "paused":
        final_phase = str(experiment_run.get("final_phase") or "").strip().lower()
        can_resume = final_phase in {"primary", "retry_primary", "fallback"} or bool(
            failed_commands
        )

    state = str(recovery.get("recovery_state") or "").strip().lower()
    if not state:
        if failed_commands and experiment_run.get("ok") is False:
            state = "verification_failed"
        elif can_resume:
            state = "needs_operator_retry"
        else:
            state = "relaunch_available"

    suggested_actions = [
        str(action).strip()
        for action in (
            recovery.get("suggested_operator_actions")
            if isinstance(recovery.get("suggested_operator_actions"), list)
            else []
        )
        if str(action).strip()
    ]
    if not suggested_actions:
        if failed_commands:
            suggested_actions.append("retry_with_refined_plan")
        if can_resume:
            suggested_actions.append("resume_verification")
        suggested_actions.append("relaunch_clean_run")

    return {
        "recovery_state": state,
        "last_failed_commands": failed_commands,
        "retry_reason": retry_reason,
        "resume_hint": str(recovery.get("resume_hint") or "").strip()
        or ("Resume verification from the paused job state." if can_resume else ""),
        "suggested_operator_actions": suggested_actions,
        "can_retry_with_refined_plan": bool(
            recovery.get(
                "can_retry_with_refined_plan",
                bool(
                    failed_commands or state in {"verification_failed", "plan_stalled"}
                ),
            )
        ),
        "can_resume_verification": can_resume,
        "latest_failed_output": (
            latest_failed_output[:4000] if latest_failed_output else ""
        ),
    }
