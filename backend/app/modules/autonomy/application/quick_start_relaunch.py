"""Rebuild quick-start requests from persisted autonomous-job state."""

from typing import Any, Optional

from app.modules.autonomy.application.quick_start_builders import (
    coerce_bool,
    normalize_swarm_roles,
)
from app.schemas.agent_job import (
    AgentJobQuickStartClaudeBackendRequest,
    AgentJobQuickStartDomainResearchRequest,
    AgentJobQuickStartRoleWorkflowRequest,
)
from app.services.agent_scope_service import normalize_scope_config


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
) -> Optional[dict[str, Any]]:
    overrides = {key: value for key, value in config.items() if key not in reserved}
    overrides["relaunch_from_job_id"] = str(getattr(job, "id", "") or "").strip()
    return overrides or None


def build_claude_backend_relaunch_request(
    job: Any,
) -> Optional[AgentJobQuickStartClaudeBackendRequest]:
    """Rebuild a Claude-backend quick-start request from a stored job."""
    config = _job_config(job)
    if _launch_mode(config) != "quick_start_claude_backend":
        return None

    source_id = _source_id(config)
    goal = str(getattr(job, "goal", "") or "").strip()
    if not source_id or not goal:
        return None

    search_query = config.get("search_query")
    commands = (
        config.get("commands") if isinstance(config.get("commands"), list) else None
    )
    file_paths = (
        config.get("file_paths") if isinstance(config.get("file_paths"), list) else None
    )
    overrides = _relaunch_overrides(
        job,
        config,
        reserved={
            "source_id",
            "target_source_id",
            "launch_mode",
            "quick_start",
            "search_query",
            "commands",
            "file_paths",
        },
    )

    try:
        return AgentJobQuickStartClaudeBackendRequest(
            name=str(getattr(job, "name", "") or "").strip() or None,
            goal=goal,
            source_id=source_id,
            search_query=(
                str(search_query).strip() if search_query is not None else None
            ),
            file_paths=file_paths,
            commands=commands,
            start_immediately=True,
            config_overrides=overrides,
        )
    except Exception:
        return None


def build_domain_research_relaunch_request(
    job: Any,
) -> Optional[AgentJobQuickStartDomainResearchRequest]:
    """Rebuild a domain-research quick-start request from a stored job."""
    config = _job_config(job)
    if _launch_mode(config) != "quick_start_domain_research":
        return None

    domain = str(config.get("domain") or "").strip()
    objective = str(config.get("objective") or "").strip()
    if not domain or not objective:
        return None

    overrides = _relaunch_overrides(
        job,
        config,
        reserved={
            "launch_mode",
            "deterministic_runner",
            "domain_research_mode",
            "domain",
            "objective",
            "customer_context",
            "source_scope",
            "track_type",
            "research_mode",
            "monitor_queries",
            "repo_source_ids",
            "benchmark_queries",
            "sandbox_profile_id",
            "report_format",
            "scoring_policy",
            "selection_policy",
            "persist_artifacts",
            "persist_target",
            "auto_launch_follow_up",
            "validation_policy",
            "follow_up_type",
            "prefer_sources",
            "max_documents",
            "max_papers",
            "profile_id",
            "auto_create_experiment_plans",
            "confidence_threshold",
            "search_query",
            "quick_start",
        },
    )

    try:
        return AgentJobQuickStartDomainResearchRequest(
            name=str(getattr(job, "name", "") or "").strip() or None,
            domain=domain,
            objective=objective,
            customer_context=str(config.get("customer_context") or "").strip() or None,
            source_scope=str(config.get("source_scope") or "kb_plus_arxiv")
            .strip()
            .lower()
            or "kb_plus_arxiv",
            track_type=str(config.get("track_type") or "generic").strip().lower()
            or "generic",
            research_mode=str(config.get("research_mode") or "literature_to_hypothesis")
            .strip()
            .lower()
            or "literature_to_hypothesis",
            monitor_queries=(
                config.get("monitor_queries")
                if isinstance(config.get("monitor_queries"), list)
                else None
            ),
            repo_source_ids=(
                config.get("repo_source_ids")
                if isinstance(config.get("repo_source_ids"), list)
                else None
            ),
            benchmark_queries=(
                config.get("benchmark_queries")
                if isinstance(config.get("benchmark_queries"), list)
                else None
            ),
            sandbox_profile_id=str(config.get("sandbox_profile_id") or "").strip()
            or None,
            report_format=str(config.get("report_format") or "brief_and_report")
            .strip()
            .lower()
            or "brief_and_report",
            scoring_policy=(
                config.get("scoring_policy")
                if isinstance(config.get("scoring_policy"), dict)
                else None
            ),
            selection_policy=(
                config.get("selection_policy")
                if isinstance(config.get("selection_policy"), dict)
                else None
            ),
            persist_artifacts=coerce_bool(
                config.get("persist_artifacts"), default=True
            ),
            auto_launch_follow_up=coerce_bool(
                config.get("auto_launch_follow_up"), default=True
            ),
            auto_create_experiment_plans=coerce_bool(
                config.get("auto_create_experiment_plans"), default=True
            ),
            automation_profile=str(config.get("automation_profile") or "balanced")
            .strip()
            .lower()
            or "balanced",
            automation_policy=(
                config.get("automation_policy")
                if isinstance(config.get("automation_policy"), dict)
                else None
            ),
            validation_policy=(
                config.get("validation_policy")
                if isinstance(config.get("validation_policy"), dict)
                else None
            ),
            max_documents=int(config.get("max_documents") or 10),
            max_papers=int(config.get("max_papers") or 8),
            profile_id=config.get("profile_id"),
            confidence_threshold=float(config.get("confidence_threshold") or 0.7),
            start_immediately=True,
            config_overrides=overrides,
        )
    except Exception:
        return None


def build_role_workflow_relaunch_request(
    job: Any,
) -> Optional[AgentJobQuickStartRoleWorkflowRequest]:
    """Rebuild a role-workflow quick-start request from a stored job."""
    config = _job_config(job)
    if _launch_mode(config) != "quick_start_role_workflow":
        return None

    goal = str(getattr(job, "goal", "") or "").strip()
    if not goal:
        return None

    roles = normalize_swarm_roles(config.get("swarm_roles"), max_roles=12)
    quick_start = (
        config.get("quick_start") if isinstance(config.get("quick_start"), dict) else {}
    )
    if not roles:
        roles = normalize_swarm_roles(quick_start.get("roles"), max_roles=12)

    try:
        max_agents = int(config.get("swarm_max_agents", 0) or 0)
    except Exception:
        max_agents = 0
    if max_agents <= 0:
        max_agents = len(roles) if roles else 4
    max_agents = max(1, min(max_agents, 12))

    memory = config.get("memory") if isinstance(config.get("memory"), dict) else {}
    memory_profile = (
        str(quick_start.get("memory_profile") or memory.get("profile") or "balanced")
        .strip()
        .lower()
    )
    if memory_profile not in {
        "off",
        "minimal",
        "balanced",
        "evidence",
        "synthesis",
    }:
        memory_profile = "balanced"

    approval = (
        config.get("approval_checkpoints")
        if isinstance(config.get("approval_checkpoints"), dict)
        else {}
    )
    approval_mode = str(quick_start.get("approval_mode") or "").strip().lower()
    if approval_mode not in {"high_impact", "none"}:
        approval_mode = (
            "high_impact"
            if coerce_bool(approval.get("enabled"), default=False)
            else "none"
        )

    execution_mode = (
        str(quick_start.get("execution_mode") or config.get("execution_mode") or "")
        .strip()
        .lower()
    )
    execution_mode = execution_mode.replace("-", "_").replace(" ", "_")
    if execution_mode in {"plan_then_act", "plan_execute", "planner_executor"}:
        execution_mode = "plan_and_execute"
    if execution_mode not in {"plan_and_execute", "adaptive"}:
        execution_mode = "plan_and_execute"

    extract_on_statuses = (
        memory.get("extract_on_statuses")
        if isinstance(memory.get("extract_on_statuses"), list)
        else []
    )
    extract_on_statuses = [
        str(value).strip().lower()
        for value in extract_on_statuses
        if str(value).strip()
    ]
    extract_memory_on_failure = (
        "failed" in set(extract_on_statuses)
        if extract_on_statuses
        else coerce_bool(memory.get("extract_on_failure"), default=True)
    )
    failed_types = (
        memory.get("failed_extraction_types")
        if isinstance(memory.get("failed_extraction_types"), list)
        else None
    )
    completed_types = (
        memory.get("completed_extraction_types")
        if isinstance(memory.get("completed_extraction_types"), list)
        else None
    )
    overrides = _relaunch_overrides(
        job,
        config,
        reserved={
            "launch_mode",
            "quick_start",
            "plan_then_act_enabled",
            "plan_max_steps",
            "execution_mode",
            "subgoal_decomposition_enabled",
            "swarm_child_jobs_enabled",
            "swarm_max_agents",
            "swarm_roles",
            "swarm_inherit_results",
            "swarm_inherit_config",
            "swarm_fan_in_enabled",
            "swarm_fan_in_name",
            "swarm_fan_in_trigger_condition",
            "approval_checkpoints",
            "memory",
            "enable_memory",
        },
    )

    try:
        return AgentJobQuickStartRoleWorkflowRequest(
            name=str(getattr(job, "name", "") or "").strip() or None,
            goal=goal,
            roles=roles[:12] or None,
            max_agents=max_agents,
            memory_profile=memory_profile,
            approval_mode=approval_mode,
            execution_mode=execution_mode,
            extract_memory_on_failure=extract_memory_on_failure,
            memory_failed_types=failed_types,
            memory_completed_types=completed_types,
            start_immediately=True,
            config_overrides=overrides,
        )
    except Exception:
        return None
