"""Pure request-to-configuration builders for autonomous-job quick starts."""

import re
from typing import Any

from app.schemas.agent_job import (
    AgentJobQuickStartClaudeBackendRequest,
    AgentJobQuickStartDomainResearchRequest,
    AgentJobQuickStartRepoBugTriageRequest,
    AgentJobQuickStartRoleWorkflowRequest,
)
from app.services.agent_scope_service import normalize_scope_config
from app.services.autonomy_service import resolve_domain_profile_automation_contract


def build_claude_backend_config(
    request: AgentJobQuickStartClaudeBackendRequest,
    *,
    source_name: str,
    source_type: str,
) -> dict:
    """Build normalized configuration for a backend coding quick start."""
    config: dict = {
        "source_id": str(request.source_id),
        "launch_mode": "quick_start_claude_backend",
        "quick_start": {
            "profile": "claude_backend",
            "version": "v1",
            "source_name": str(source_name or "").strip(),
            "source_type": str(source_type or "").strip().lower(),
        },
    }
    if request.search_query is not None:
        config["search_query"] = str(request.search_query)
    if isinstance(request.file_paths, list):
        config["file_paths"] = [
            str(path).strip() for path in request.file_paths if str(path).strip()
        ]
    if isinstance(request.commands, list):
        config["commands"] = [
            str(command).strip() for command in request.commands if str(command).strip()
        ]
    if isinstance(request.config_overrides, dict):
        config.update(normalize_scope_config(request.config_overrides) or {})
    return normalize_scope_config(config)


def build_domain_research_goal(
    request: AgentJobQuickStartDomainResearchRequest,
) -> str:
    """Build the canonical goal text for a domain-research quick start."""
    domain = str(request.domain or "").strip()
    objective = str(request.objective or "").strip()
    context = str(request.customer_context or "").strip()
    goal = f"Research the domain '{domain}'. Objective: {objective}"
    if context:
        goal += f"\nContext: {context}"
    return goal


def build_domain_research_config(
    request: AgentJobQuickStartDomainResearchRequest,
) -> dict:
    """Build normalized configuration for a domain-research quick start."""
    source_scope = str(request.source_scope or "kb_plus_arxiv").strip().lower()
    track_type = str(request.track_type or "generic").strip().lower()
    domain = str(request.domain or "").strip()
    objective = str(request.objective or "").strip()
    monitor_queries = [
        str(query).strip()
        for query in (request.monitor_queries or [])
        if str(query).strip()
    ]
    if not monitor_queries:
        monitor_queries = [f"{domain} {objective}".strip()[:240]]
    benchmark_queries = [
        str(query).strip()
        for query in (request.benchmark_queries or [])
        if str(query).strip()
    ][:16]
    repo_source_ids = [
        str(source_id).strip()
        for source_id in (request.repo_source_ids or [])
        if str(source_id).strip()
    ][:24]

    prefer_sources = ["documents", "arxiv"]
    if source_scope == "kb_only":
        prefer_sources = ["documents"]
    elif source_scope == "arxiv_only":
        prefer_sources = ["arxiv"]
    elif source_scope == "kb_plus_arxiv_plus_repo":
        prefer_sources = ["documents", "arxiv", "repo"]

    explicit_legacy_updates: dict[str, Any] = {}
    if "validation_policy" in request.model_fields_set:
        explicit_legacy_updates["validation_policy"] = request.validation_policy
    if "auto_launch_follow_up" in request.model_fields_set:
        explicit_legacy_updates["auto_launch_follow_up"] = request.auto_launch_follow_up
    if "auto_create_experiment_plans" in request.model_fields_set:
        explicit_legacy_updates[
            "auto_create_experiment_plans"
        ] = request.auto_create_experiment_plans
    if "confidence_threshold" in request.model_fields_set:
        explicit_legacy_updates["confidence_threshold"] = request.confidence_threshold

    automation_profile, automation_policy = resolve_domain_profile_automation_contract(
        automation_profile=request.automation_profile,
        automation_policy=request.automation_policy,
        current_snapshot={"validation_policy": request.validation_policy}
        if isinstance(request.validation_policy, dict)
        else None,
        explicit_updates=explicit_legacy_updates or None,
    )
    confidence_threshold = float(
        automation_policy.get("confidence_threshold")
        or request.confidence_threshold
        or 0.7
    )
    auto_create_experiment_plans = bool(
        automation_policy.get(
            "auto_create_experiment_plans", request.auto_create_experiment_plans
        )
    )
    auto_launch_follow_up = bool(
        automation_policy.get("auto_launch_follow_up", request.auto_launch_follow_up)
    )

    config: dict[str, Any] = {
        "launch_mode": "quick_start_domain_research",
        "deterministic_runner": "domain_research_orchestrator",
        "domain_research_mode": True,
        "profile_id": str(request.profile_id) if request.profile_id else None,
        "domain": domain,
        "objective": objective,
        "customer_context": str(request.customer_context or "").strip(),
        "source_scope": source_scope,
        "track_type": track_type,
        "research_mode": str(request.research_mode or "literature_to_hypothesis")
        .strip()
        .lower(),
        "monitor_queries": monitor_queries[:12],
        "repo_source_ids": repo_source_ids or None,
        "benchmark_queries": benchmark_queries or None,
        "sandbox_profile_id": str(request.sandbox_profile_id or "").strip() or None,
        "report_format": str(request.report_format or "brief_and_report")
        .strip()
        .lower(),
        "scoring_policy": request.scoring_policy
        if isinstance(request.scoring_policy, dict)
        else None,
        "selection_policy": request.selection_policy
        if isinstance(request.selection_policy, dict)
        else None,
        "persist_artifacts": bool(request.persist_artifacts),
        "persist_target": "research_notes",
        "automation_profile": automation_profile,
        "automation_policy": automation_policy,
        "auto_launch_follow_up": auto_launch_follow_up,
        "auto_create_experiment_plans": auto_create_experiment_plans,
        "follow_up_type": "deep_dive_chain",
        "prefer_sources": prefer_sources,
        "max_documents": int(request.max_documents or 10),
        "max_papers": int(request.max_papers or 8),
        "confidence_threshold": confidence_threshold,
        "search_query": f"{domain} {objective}".strip()[:500],
        "quick_start": {
            "profile": "domain_research",
            "version": "v2",
            "source_scope": source_scope,
            "track_type": track_type,
            "sandbox_profile_id": str(request.sandbox_profile_id or "").strip() or None,
            "research_mode": str(request.research_mode or "literature_to_hypothesis")
            .strip()
            .lower(),
            "persist_target": "research_notes",
            "report_format": str(request.report_format or "brief_and_report")
            .strip()
            .lower(),
            "automation_profile": automation_profile,
            "auto_launch_follow_up": auto_launch_follow_up,
            "auto_create_experiment_plans": auto_create_experiment_plans,
            "profile_id": str(request.profile_id) if request.profile_id else None,
        },
    }
    if "validation_policy" in request.model_fields_set:
        config["validation_policy"] = automation_policy
    if isinstance(request.config_overrides, dict):
        config.update(normalize_scope_config(request.config_overrides) or {})
    return normalize_scope_config(config)


def build_repo_bug_triage_goal(
    request: AgentJobQuickStartRepoBugTriageRequest,
) -> str:
    """Build the canonical goal text for repository bug triage."""
    symptom = str(request.failure_symptom or "").strip()
    goal = str(request.goal or "").strip()
    scope = str(request.scope or "auto").strip().lower()
    if symptom and goal:
        return (
            f"Triage and repair the reported {scope} bug. Symptom: {symptom}\n"
            f"Desired outcome: {goal}"
        )
    if symptom:
        return f"Triage and repair the reported {scope} bug. Symptom: {symptom}"
    return goal


def build_repo_bug_triage_config(
    request: AgentJobQuickStartRepoBugTriageRequest,
    *,
    source_name: str,
    source_type: str,
) -> dict:
    """Build normalized configuration for repository bug triage."""
    scope = str(request.scope or "auto").strip().lower()
    symptom = str(request.failure_symptom or "").strip()
    search_query = str(request.search_query or "").strip()
    if not search_query:
        scope_hint = "" if scope == "auto" else scope.replace("_", " ")
        search_query = " ".join(part for part in [scope_hint, symptom] if part).strip()[
            :500
        ]

    config: dict = {
        "source_id": str(request.source_id),
        "launch_mode": "quick_start_repo_bug_triage",
        "failure_symptom": symptom,
        "scope": scope or "auto",
        "quick_start": {
            "profile": "repo_bug_triage",
            "version": "v2",
            "source_name": str(source_name or "").strip(),
            "source_type": str(source_type or "").strip().lower(),
            "scope": scope or "auto",
            "autonomy_mode": "patch_proposal",
            "execution_depth": "workspace_planned",
        },
    }
    if search_query:
        config["search_query"] = search_query
    if request.error_output is not None:
        config["error_output"] = str(request.error_output)
    if isinstance(request.file_paths, list):
        config["file_paths"] = [
            str(path).strip() for path in request.file_paths if str(path).strip()
        ]
    if isinstance(request.commands, list):
        config["commands"] = [
            str(command).strip() for command in request.commands if str(command).strip()
        ]
    if isinstance(request.config_overrides, dict):
        config.update(normalize_scope_config(request.config_overrides) or {})
    return normalize_scope_config(config)


def coerce_bool(value: Any, default: bool = False) -> bool:
    """Coerce common boolean representations used by stored quick-start state."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    return default


def normalize_swarm_roles(roles: Any, *, max_roles: int = 12) -> list[str]:
    """Normalize and bound role identifiers for role-based workflows."""
    if not isinstance(roles, list):
        return []
    normalized: list[str] = []
    for raw in roles:
        role = str(raw or "").strip().lower()
        if not role:
            continue
        role = role.replace("-", "_").replace(" ", "_")
        if not re.match(r"^[a-z0-9_:\-]{2,120}$", role):
            continue
        if role not in normalized:
            normalized.append(role)
        if len(normalized) >= max(1, min(max_roles, 12)):
            break
    return normalized


def _build_role_memory_config(memory_profile: str) -> tuple[dict[str, Any], bool]:
    profile = str(memory_profile or "balanced").strip().lower()
    if profile not in {"off", "minimal", "balanced", "evidence", "synthesis"}:
        profile = "balanced"
    presets: dict[str, dict[str, Any]] = {
        "minimal": {
            "max_memories": 6,
            "memory_types": ["finding", "insight", "pattern"],
            "include_chat_memory": False,
        },
        "balanced": {
            "max_memories": 10,
            "memory_types": ["finding", "insight", "pattern", "lesson"],
            "include_chat_memory": True,
        },
        "evidence": {
            "max_memories": 14,
            "memory_types": ["finding", "insight", "fact", "context"],
            "include_chat_memory": True,
        },
        "synthesis": {
            "max_memories": 12,
            "memory_types": ["pattern", "lesson", "insight", "summary"],
            "include_chat_memory": True,
        },
    }
    role_profiles: dict[str, dict[str, Any]] = {
        "researcher": {
            "max_memories": 14 if profile in {"evidence", "synthesis"} else 12,
            "memory_types": ["finding", "insight", "fact", "context"],
            "include_chat_memory": True,
        },
        "critic": {
            "max_memories": 10 if profile == "minimal" else 12,
            "memory_types": ["pattern", "lesson", "finding", "insight"],
            "include_chat_memory": False,
        },
        "synthesizer": {
            "max_memories": 12,
            "memory_types": ["pattern", "lesson", "insight", "summary"],
            "include_chat_memory": True,
        },
        "verifier": {
            "max_memories": 10,
            "memory_types": ["finding", "fact", "context", "insight"],
            "include_chat_memory": True,
        },
    }
    if profile == "off":
        return (
            {
                "profile": "off",
                "enabled": False,
                "max_memories": 0,
                "memory_types": [],
                "include_chat_memory": False,
                "role_profiles": {},
            },
            False,
        )
    preset = dict(presets.get(profile, presets["balanced"]))
    preset["profile"] = profile
    preset["enabled"] = True
    preset["role_profiles"] = role_profiles
    return preset, True


def _build_role_approval_config(approval_mode: str) -> dict[str, Any]:
    mode = str(approval_mode or "high_impact").strip().lower()
    if mode == "none":
        return {"enabled": False}
    return {
        "enabled": True,
        "tools": [
            "create_document_from_text",
            "update_document",
            "delete_document",
            "ingest_url",
            "fetch_url_content",
            "run_python_code",
        ],
        "once_per_checkpoint": True,
        "message_prefix": "Role workflow checkpoint: review high-impact action",
    }


def build_role_workflow_config(
    request: AgentJobQuickStartRoleWorkflowRequest,
) -> dict:
    """Build normalized configuration for a role-based workflow."""
    roles = normalize_swarm_roles(request.roles or [])
    if not roles:
        roles = ["researcher_documents", "researcher_arxiv", "analyst", "synthesizer"]
    max_agents = max(1, min(int(request.max_agents or len(roles) or 4), 12))
    roles = roles[:max_agents]
    memory, enable_memory = _build_role_memory_config(
        str(request.memory_profile or "balanced")
    )
    approvals = _build_role_approval_config(str(request.approval_mode or "high_impact"))
    execution_mode = str(request.execution_mode or "plan_and_execute").strip().lower()
    if execution_mode not in {"plan_and_execute", "adaptive"}:
        execution_mode = "plan_and_execute"
    extract_on_failure = bool(
        request.extract_memory_on_failure
        if request.extract_memory_on_failure is not None
        else True
    )
    memory["extract_on_statuses"] = ["completed"] + (
        ["failed"] if extract_on_failure else []
    )
    memory["extract_on_failure"] = extract_on_failure
    if isinstance(request.memory_failed_types, list) and request.memory_failed_types:
        memory["failed_extraction_types"] = [
            str(value).strip().lower()
            for value in request.memory_failed_types
            if str(value).strip()
        ][:12]
    if (
        isinstance(request.memory_completed_types, list)
        and request.memory_completed_types
    ):
        memory["completed_extraction_types"] = [
            str(value).strip().lower()
            for value in request.memory_completed_types
            if str(value).strip()
        ][:12]

    config: dict[str, Any] = {
        "launch_mode": "quick_start_role_workflow",
        "quick_start": {
            "profile": "role_workflow",
            "version": "v1",
            "memory_profile": str(memory.get("profile") or "balanced"),
            "approval_mode": str(request.approval_mode or "high_impact"),
            "execution_mode": execution_mode,
            "extract_memory_on_failure": extract_on_failure,
            "roles": roles[:12],
            "max_agents": max_agents,
        },
        "plan_then_act_enabled": True,
        "plan_max_steps": 7,
        "execution_mode": execution_mode,
        "subgoal_decomposition_enabled": True,
        "swarm_child_jobs_enabled": True,
        "swarm_max_agents": max_agents,
        "swarm_roles": roles[:12],
        "swarm_inherit_results": True,
        "swarm_inherit_config": True,
        "swarm_fan_in_enabled": True,
        "swarm_fan_in_name": "Role Workflow Fan-In",
        "swarm_fan_in_trigger_condition": "on_any_end",
        "approval_checkpoints": approvals,
        "memory": memory,
        "enable_memory": bool(enable_memory),
    }
    if isinstance(request.config_overrides, dict):
        config.update(normalize_scope_config(request.config_overrides) or {})
    return normalize_scope_config(config)
