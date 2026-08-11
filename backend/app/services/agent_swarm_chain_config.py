"""Swarm chain configuration.

Plans the swarm child jobs a parent job should chain into, and normalises the
swarm settings that drive it. Extracted from AutonomousAgentExecutor: no
session, no LLM and no logging, so it is unit-tested directly. Mutating `job`
and `state` is the caller's contract, kept as-is by the extraction; the
step-event appender is injected so this module does not duplicate it.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Dict, List, Optional

from app.models.agent_job import AgentJob, ChainTriggerCondition


def get_swarm_config(job: AgentJob) -> Dict[str, Any]:
    """Get normalized config for swarm child-agent generation."""
    cfg = job.config if isinstance(job.config, dict) else {}

    def _as_int(key: str, default: int, lo: int, hi: int) -> int:
        try:
            val = int(cfg.get(key, default))
        except Exception:
            val = default
        return max(lo, min(val, hi))

    def _as_float(key: str, default: float, lo: float, hi: float) -> float:
        try:
            val = float(cfg.get(key, default))
        except Exception:
            val = default
        return max(lo, min(val, hi))

    roles = cfg.get("swarm_roles")
    if isinstance(roles, str):
        roles = [x.strip() for x in roles.split(",") if x.strip()]
    if not isinstance(roles, list):
        roles = []

    trigger = (
        str(
            cfg.get("swarm_trigger_condition", ChainTriggerCondition.ON_COMPLETE.value)
            or ChainTriggerCondition.ON_COMPLETE.value
        )
        .strip()
        .lower()
    )
    if trigger not in {
        ChainTriggerCondition.ON_COMPLETE.value,
        ChainTriggerCondition.ON_ANY_END.value,
        ChainTriggerCondition.ON_PROGRESS.value,
        ChainTriggerCondition.ON_FINDINGS.value,
        ChainTriggerCondition.ON_FAIL.value,
    }:
        trigger = ChainTriggerCondition.ON_COMPLETE.value

    return {
        "enabled": bool(cfg.get("swarm_child_jobs_enabled", False)),
        "max_agents": _as_int("swarm_max_agents", 4, 1, 12),
        "roles": roles,
        "inherit_results": bool(cfg.get("swarm_inherit_results", True)),
        "inherit_config": bool(cfg.get("swarm_inherit_config", False)),
        "trigger_condition": trigger,
        "max_iterations_ratio": _as_float(
            "swarm_child_max_iterations_ratio", 0.45, 0.1, 1.0
        ),
        "max_tool_calls_ratio": _as_float(
            "swarm_child_max_tool_calls_ratio", 0.45, 0.1, 1.0
        ),
        "max_llm_calls_ratio": _as_float(
            "swarm_child_max_llm_calls_ratio", 0.45, 0.1, 1.0
        ),
        "max_runtime_ratio": _as_float("swarm_child_max_runtime_ratio", 0.5, 0.1, 1.0),
        "min_iterations": _as_int("swarm_child_min_iterations", 6, 1, 100),
        "min_tool_calls": _as_int("swarm_child_min_tool_calls", 8, 1, 200),
        "min_llm_calls": _as_int("swarm_child_min_llm_calls", 6, 1, 200),
        "min_runtime_minutes": _as_int("swarm_child_min_runtime_minutes", 10, 1, 240),
        "goal_prefix": str(cfg.get("swarm_goal_prefix", "Swarm role")).strip()[:80],
        "fan_in_enabled": bool(cfg.get("swarm_fan_in_enabled", True)),
        "fan_in_name": str(cfg.get("swarm_fan_in_name", "Swarm Synthesis")).strip()[
            :120
        ],
        "fan_in_job_type": str(
            cfg.get("swarm_fan_in_job_type", "synthesis") or "synthesis"
        )
        .strip()
        .lower(),
        "fan_in_trigger_condition": str(
            cfg.get(
                "swarm_fan_in_trigger_condition",
                ChainTriggerCondition.ON_ANY_END.value,
            )
            or ChainTriggerCondition.ON_ANY_END.value
        )
        .strip()
        .lower(),
    }


def ensure_swarm_chain_config(
    job: AgentJob,
    state: Dict[str, Any],
    *,
    append_step_event: Callable[..., None],
) -> None:
    """Create a swarm of specialized child jobs when enabled and no chain exists yet."""
    swarm_cfg = get_swarm_config(job)
    cfg = job.config if isinstance(job.config, dict) else {}
    if not bool(swarm_cfg.get("enabled", False)):
        return
    if bool(state.get("swarm_chain_configured", False)):
        return

    chain = job.chain_config if isinstance(job.chain_config, dict) else {}
    existing_children = chain.get("child_jobs")
    if isinstance(existing_children, list) and existing_children:
        state["swarm_chain_configured"] = True
        state["swarm_child_jobs_count"] = len(existing_children)
        chain_data_existing = (
            chain.get("chain_data") if isinstance(chain.get("chain_data"), dict) else {}
        )
        state["swarm_fan_in_enabled"] = bool(
            chain_data_existing.get("swarm_fan_in_enabled", False)
        )
        state["swarm_fan_in_group_id"] = str(
            chain_data_existing.get("swarm_fan_in_group_id") or ""
        )
        append_step_event(
            state,
            {
                "type": "swarm_chain_reused",
                "iteration": int(job.iteration or 0),
                "child_jobs_count": len(existing_children),
                "fan_in_enabled": bool(state.get("swarm_fan_in_enabled", False)),
                "fan_in_group_id": str(state.get("swarm_fan_in_group_id") or ""),
            },
        )
        return

    coding_swarm_enabled = bool(
        cfg.get("coding_swarm_enabled")
        or str(cfg.get("launch_mode") or "").strip().lower()
        == "quick_start_bug_triage_swarm"
        or str(
            (cfg.get("quick_start") or {}).get("profile")
            if isinstance(cfg.get("quick_start"), dict)
            else ""
        )
        .strip()
        .lower()
        == "bug_triage_swarm"
    )
    if coding_swarm_enabled:
        from app.services.agent_coding_harness_service import (
            agent_coding_harness_service,
        )
        from app.services.agent_coding_workspace_session_service import (
            agent_coding_workspace_session_service,
        )

        role_templates = agent_coding_harness_service.get_role_catalog()
        role_template_aliases = agent_coding_harness_service.role_aliases()
        default_roles: List[Any] = [
            "reproducer",
            "root_cause",
            "patcher",
            "verifier",
        ]
        fallback_role_key = "reproducer"
    else:
        role_templates = {
            "researcher": {
                "name": "Researcher",
                "job_type": "research",
                "objective": "Gather high-signal evidence from papers and internal knowledge sources.",
                "agent_role": "researcher",
                "config": {
                    "prefer_sources": ["documents", "arxiv"],
                    "max_documents": 10,
                    "max_papers": 8,
                },
            },
            "researcher_documents": {
                "name": "Knowledge Researcher",
                "job_type": "research",
                "objective": "Focus on internal documents and existing knowledge-base evidence.",
                "agent_role": "researcher_documents",
                "config": {
                    "prefer_sources": ["documents"],
                    "max_documents": 14,
                    "max_papers": 2,
                },
            },
            "researcher_arxiv": {
                "name": "Literature Researcher",
                "job_type": "research",
                "objective": "Focus on external paper discovery and validation.",
                "agent_role": "researcher_arxiv",
                "config": {
                    "prefer_sources": ["arxiv"],
                    "max_documents": 4,
                    "max_papers": 12,
                },
            },
            "analyst": {
                "name": "Analyst",
                "job_type": "analysis",
                "objective": "Compare sources, identify gaps/contradictions, and stress-test assumptions.",
                "agent_role": "critic",
                "config": {"prefer_sources": ["documents", "arxiv"]},
            },
            "critic": {
                "name": "Critic",
                "job_type": "analysis",
                "objective": "Challenge assumptions and identify evidence gaps before synthesis.",
                "agent_role": "critic",
                "config": {"prefer_sources": ["documents", "arxiv"]},
            },
            "synthesizer": {
                "name": "Synthesizer",
                "job_type": "synthesis",
                "objective": "Produce concise synthesis with traceable evidence and clear next actions.",
                "agent_role": "synthesizer",
                "config": {"prefer_sources": ["documents"]},
            },
            "monitor": {
                "name": "Monitor",
                "job_type": "monitor",
                "objective": "Track updates and ingest newly relevant sources for the topic.",
                "agent_role": "verifier",
                "config": {"prefer_sources": ["arxiv", "documents"]},
            },
            "verifier": {
                "name": "Verifier",
                "job_type": "analysis",
                "objective": "Verify evidence quality, consistency, and confidence before final decisions.",
                "agent_role": "verifier",
                "config": {"prefer_sources": ["documents", "arxiv"]},
            },
            "knowledge_expander": {
                "name": "Knowledge Expander",
                "job_type": "knowledge_expansion",
                "objective": "Find adjacent concepts and add structured knowledge links.",
                "agent_role": "researcher",
                "config": {"prefer_sources": ["documents", "arxiv"]},
            },
        }
        role_template_aliases = {
            "research": "researcher",
            "researcher_docs": "researcher_documents",
            "document_researcher": "researcher_documents",
            "docs_researcher": "researcher_documents",
            "knowledge_researcher": "researcher_documents",
            "literature_researcher": "researcher_arxiv",
            "paper_researcher": "researcher_arxiv",
            "arxiv_researcher": "researcher_arxiv",
            "reviewer": "critic",
            "validator": "verifier",
            "qa": "verifier",
            "checker": "verifier",
            "writer": "synthesizer",
            "aggregator": "synthesizer",
            "synth": "synthesizer",
        }
        default_roles = [
            "researcher_documents",
            "researcher_arxiv",
            "analyst",
        ]
        fallback_role_key = "researcher"
    roles_raw = swarm_cfg.get("roles")
    if not isinstance(roles_raw, list) or not roles_raw:
        roles_raw = default_roles

    max_agents = int(swarm_cfg.get("max_agents", 4) or 4)
    max_agents = max(1, min(max_agents, 12))
    parent_goal = str(job.goal or "").strip()[:1600]
    fan_in_enabled = bool(swarm_cfg.get("fan_in_enabled", True))
    fan_in_trigger = (
        str(
            swarm_cfg.get(
                "fan_in_trigger_condition", ChainTriggerCondition.ON_ANY_END.value
            )
            or ChainTriggerCondition.ON_ANY_END.value
        )
        .strip()
        .lower()
    )
    if fan_in_trigger not in {
        ChainTriggerCondition.ON_COMPLETE.value,
        ChainTriggerCondition.ON_ANY_END.value,
        ChainTriggerCondition.ON_PROGRESS.value,
        ChainTriggerCondition.ON_FINDINGS.value,
        ChainTriggerCondition.ON_FAIL.value,
    }:
        fan_in_trigger = ChainTriggerCondition.ON_ANY_END.value

    child_max_iterations = max(
        int(swarm_cfg.get("min_iterations", 6) or 6),
        int(
            (job.max_iterations or 20)
            * float(swarm_cfg.get("max_iterations_ratio", 0.45) or 0.45)
        ),
    )
    child_max_tool_calls = max(
        int(swarm_cfg.get("min_tool_calls", 8) or 8),
        int(
            (job.max_tool_calls or 50)
            * float(swarm_cfg.get("max_tool_calls_ratio", 0.45) or 0.45)
        ),
    )
    child_max_llm_calls = max(
        int(swarm_cfg.get("min_llm_calls", 6) or 6),
        int(
            (job.max_llm_calls or 30)
            * float(swarm_cfg.get("max_llm_calls_ratio", 0.45) or 0.45)
        ),
    )
    child_max_runtime = max(
        int(swarm_cfg.get("min_runtime_minutes", 10) or 10),
        int(
            (job.max_runtime_minutes or 60)
            * float(swarm_cfg.get("max_runtime_ratio", 0.5) or 0.5)
        ),
    )

    allowed_job_types = {
        "research",
        "monitor",
        "analysis",
        "synthesis",
        "knowledge_expansion",
        "custom",
        "data_analysis",
    }
    fan_in_job_type = (
        str(swarm_cfg.get("fan_in_job_type", "synthesis") or "synthesis")
        .strip()
        .lower()
    )
    if fan_in_job_type not in allowed_job_types:
        fan_in_job_type = "synthesis"
    child_jobs: List[Dict[str, Any]] = []
    role_names: List[str] = []

    for idx, raw in enumerate(roles_raw, start=1):
        if len(child_jobs) >= max_agents:
            break

        role_tag = ""
        role_name = ""
        role_objective = ""
        role_job_type = ""
        role_cfg: Dict[str, Any] = {}
        role_template_key = "researcher"

        if isinstance(raw, dict):
            role_key = (
                str(
                    raw.get("role")
                    or raw.get("type")
                    or raw.get("name")
                    or "researcher"
                )
                .strip()
                .lower()
            )
            role_key = role_key.replace("-", "_").replace(" ", "_")
            role_key = re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]+", "_", role_key)).strip(
                "_"
            )
            role_template_key = (
                role_key
                if role_key in role_templates
                else role_template_aliases.get(role_key, fallback_role_key)
            )
            tpl = role_templates.get(
                role_template_key, role_templates[fallback_role_key]
            )
            role_name = str(raw.get("name") or tpl.get("name") or "Researcher").strip()
            role_objective = str(
                raw.get("objective") or tpl.get("objective") or ""
            ).strip()
            role_job_type = (
                str(raw.get("job_type") or tpl.get("job_type") or job.job_type)
                .strip()
                .lower()
            )
            role_agent_role = (
                str(raw.get("agent_role") or tpl.get("agent_role") or role_template_key)
                .strip()
                .lower()
            )
            role_cfg = dict(
                tpl.get("config") if isinstance(tpl.get("config"), dict) else {}
            )
            if isinstance(raw.get("config"), dict):
                role_cfg.update(raw.get("config") or {})
        else:
            role_token = str(raw or "").strip()
            if not role_token:
                continue
            role_key = role_token.lower().replace("-", "_").replace(" ", "_")
            if ":" in role_key:
                role_key, role_tag = [p.strip() for p in role_key.split(":", 1)]
            role_key = re.sub(r"_+", "_", re.sub(r"[^a-z0-9_]+", "_", role_key)).strip(
                "_"
            )
            role_template_key = (
                role_key
                if role_key in role_templates
                else role_template_aliases.get(role_key, fallback_role_key)
            )
            tpl = role_templates.get(
                role_template_key, role_templates[fallback_role_key]
            )
            role_name = str(tpl.get("name") or "Researcher").strip()
            role_objective = str(tpl.get("objective") or "").strip()
            role_job_type = str(tpl.get("job_type") or job.job_type).strip().lower()
            role_agent_role = (
                str(tpl.get("agent_role") or role_template_key).strip().lower()
            )
            role_cfg = dict(
                tpl.get("config") if isinstance(tpl.get("config"), dict) else {}
            )
            if role_tag:
                role_name = f"{role_name} ({role_tag[:40]})"
                role_objective = f"{role_objective} Focus tag: {role_tag[:120]}."

        if role_job_type not in allowed_job_types:
            role_job_type = str(job.job_type or "research")

        role_name = role_name[:120] if role_name else f"Role {idx}"
        role_names.append(role_name)
        goal_prefix = str(
            swarm_cfg.get("goal_prefix", "Swarm role") or "Swarm role"
        ).strip()[:80]
        role_goal = (
            f"{goal_prefix}: {role_name}\n"
            f"Objective: {role_objective}\n"
            f"Parent goal: {parent_goal}\n\n"
            "Deliver concise, evidence-backed findings specific to this role, then provide actionable next steps."
        )
        child_jobs.append(
            {
                "name": f"Swarm Agent {idx}: {role_name[:80]}",
                "description": "Auto-generated swarm child agent from parent autonomous job.",
                "job_type": role_job_type,
                "goal": role_goal[:2200],
                "config": {
                    **role_cfg,
                    **(
                        agent_coding_workspace_session_service.child_session_config(
                            job,
                            role=role_template_key,
                            role_index=idx,
                        )
                        if coding_swarm_enabled
                        else {}
                    ),
                    "origin": "swarm_child_agent",
                    "swarm_role": role_name[:120],
                    "swarm_role_key": role_template_key[:80],
                    "agent_role": role_agent_role[:80],
                    "swarm_role_index": idx,
                    "swarm_parent_job_id": str(job.id),
                    "swarm_root_goal": parent_goal[:800],
                    "auto_subgoal_child_jobs_enabled": False,
                    "swarm_child_jobs_enabled": False,
                },
                "max_iterations": child_max_iterations,
                "max_tool_calls": child_max_tool_calls,
                "max_llm_calls": child_max_llm_calls,
                "max_runtime_minutes": child_max_runtime,
            }
        )

    if not child_jobs:
        return

    fan_in_group_id = hashlib.sha256(
        f"swarm_fan_in:{job.id}:{max_agents}".encode("utf-8")
    ).hexdigest()[:16]
    fan_in_template: Optional[Dict[str, Any]] = None
    if fan_in_enabled:
        coding_swarm_profile = (
            str(cfg.get("coding_swarm_profile") or "").strip().lower()
        )
        if coding_swarm_enabled and not coding_swarm_profile:
            coding_swarm_profile = "bug_triage"
        fan_in_name = str(
            swarm_cfg.get("fan_in_name", "Swarm Synthesis") or "Swarm Synthesis"
        ).strip()[:120]
        fan_in_goal = (
            f"{fan_in_name}: Merge outputs from {len(child_jobs)} swarm agents.\n"
            f"Parent goal: {parent_goal}\n\n"
            "Use inherited swarm sibling results to produce: key findings, conflicts, confidence levels, "
            "and a consolidated recommendation with cited evidence."
        )
        fan_in_template = {
            "name": f"{fan_in_name}: Consolidated Output",
            "description": "Auto-generated fan-in aggregator for swarm child agents.",
            "job_type": fan_in_job_type,
            "goal": fan_in_goal[:2400],
            "config": {
                "origin": "swarm_fan_in_aggregator",
                "deterministic_runner": "swarm_fan_in_aggregate",
                "swarm_fan_in_group_id": fan_in_group_id,
                "swarm_parent_job_id": str(job.id),
                "swarm_role_count": len(child_jobs),
                "coding_swarm_enabled": coding_swarm_enabled,
                "coding_swarm_profile": coding_swarm_profile or None,
                "coding_harness_enabled": bool(
                    cfg.get("coding_harness_enabled", False)
                ),
                "coding_harness_version": str(
                    (
                        cfg.get("coding_harness")
                        if isinstance(cfg.get("coding_harness"), dict)
                        else {}
                    ).get("version")
                    or cfg.get("coding_harness_version")
                    or ""
                ).strip()
                or None,
                "coding_workspace_session_id": str(
                    cfg.get("coding_workspace_session_id") or ""
                ).strip()
                or None,
                "swarm_child_jobs_enabled": False,
                "auto_subgoal_child_jobs_enabled": False,
            },
            "max_iterations": child_max_iterations,
            "max_tool_calls": child_max_tool_calls,
            "max_llm_calls": child_max_llm_calls,
            "max_runtime_minutes": child_max_runtime,
        }
        for child in child_jobs:
            fan_in_child = {
                **fan_in_template,
                "config": dict(
                    fan_in_template.get("config")
                    if isinstance(fan_in_template.get("config"), dict)
                    else {}
                ),
            }
            child["chain_config"] = {
                "trigger_condition": fan_in_trigger,
                "inherit_results": True,
                "inherit_config": False,
                "chain_data": {
                    "source": "swarm_fan_in",
                    "swarm_fan_in_wait_for_all_siblings": True,
                    "swarm_fan_in_expected_siblings": len(child_jobs),
                    "swarm_fan_in_group_id": fan_in_group_id,
                },
                "child_jobs": [fan_in_child],
            }

    merged = dict(chain)
    merged.setdefault(
        "trigger_condition",
        str(
            swarm_cfg.get("trigger_condition")
            or ChainTriggerCondition.ON_COMPLETE.value
        ),
    )
    merged.setdefault("inherit_results", bool(swarm_cfg.get("inherit_results", True)))
    merged.setdefault("inherit_config", bool(swarm_cfg.get("inherit_config", False)))
    merged.setdefault("chain_data", {})
    if not isinstance(merged.get("chain_data"), dict):
        merged["chain_data"] = {}
    merged["chain_data"].update(
        {
            "source": "swarm_child_jobs",
            "generated_at_iteration": int(job.iteration or 0),
            "swarm_roles": role_names[:max_agents],
            "swarm_max_agents": max_agents,
            "swarm_fan_in_enabled": fan_in_enabled,
            "swarm_fan_in_group_id": fan_in_group_id if fan_in_enabled else "",
        }
    )
    merged["child_jobs"] = child_jobs
    job.chain_config = merged
    state["swarm_chain_configured"] = True
    state["swarm_child_jobs_count"] = len(child_jobs)
    state["swarm_roles_assigned"] = role_names[:max_agents]
    state["swarm_fan_in_enabled"] = fan_in_enabled
    state["swarm_fan_in_group_id"] = fan_in_group_id if fan_in_enabled else ""
    job.add_log_entry(
        {
            "phase": "swarm_chain_configured",
            "child_jobs_count": len(child_jobs),
            "roles": role_names[:max_agents],
            "trigger_condition": merged.get("trigger_condition"),
        }
    )
    append_step_event(
        state,
        {
            "type": "swarm_roles_configured",
            "iteration": int(job.iteration or 0),
            "child_jobs_count": len(child_jobs),
            "roles": role_names[:max_agents],
            "fan_in_enabled": fan_in_enabled,
            "fan_in_group_id": state.get("swarm_fan_in_group_id"),
            "trigger_condition": str(merged.get("trigger_condition") or ""),
        },
    )
