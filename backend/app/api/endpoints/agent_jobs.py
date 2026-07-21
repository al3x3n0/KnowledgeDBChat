"""
API endpoints for autonomous agent jobs.

Provides CRUD operations and control actions for autonomous agent jobs.
"""

import asyncio
import csv
from collections import Counter
from copy import deepcopy
import io
import json
import math
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from loguru import logger
from sqlalchemy import select, func, and_, or_, cast, String, literal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload, aliased
import redis.asyncio as redis

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.agent_job import (
    AgentJob,
    AgentJobStatus,
    AgentJobTemplate,
    AgentJobCheckpoint,
    AgentJobChainDefinition,
)
from app.models.agent_definition import AgentDefinition
from app.models.coding_backlog import CodingBacklogItem
from app.models.coding_swarm_profile import CodingSwarmProfile
from app.models.domain_research_profile import DomainResearchProfile
from app.models.experiment import ExperimentRun
from app.models.memory import ConversationMemory
from app.models.autonomy_decision_event import AutonomyDecisionEvent
from app.models.autonomy_decision_trace_view import AutonomyDecisionTraceView
from app.models.research_inbox import ResearchInboxItem
from app.models.research_portfolio import ResearchPortfolio
from app.models.user import User
from app.schemas.agent_job import (
    AgentCheckpointQueueBulkActionRequest,
    AgentCheckpointQueueBulkActionResponse,
    AgentCheckpointQueueBulkActionResultResponse,
    AgentCheckpointQueueActionResponse,
    CollaborationSummaryResponse,
    AgentDecisionTraceDeepLinkResponse,
    AgentDecisionTraceEventResponse,
    AgentDecisionTraceAnalyticsBucketResponse,
    AgentDecisionTraceAnalyticsResponse,
    AgentDecisionTraceAnalyticsTrendPointResponse,
    AgentDecisionTraceResponse,
    AgentDecisionTraceActionRequest,
    AgentDecisionTraceActionResponse,
    AgentDecisionTraceViewCreate,
    AgentDecisionTraceViewListResponse,
    AgentDecisionTraceViewResponse,
    AgentDecisionTraceViewUpdate,
    AgentCheckpointQueueFollowUpActionRequest,
    AgentCheckpointQueueFollowUpActionResponse,
    AgentCheckpointQueueBulkFollowUpActionRequest,
    AgentCheckpointQueueBulkFollowUpActionResponse,
    AgentCheckpointQueueBulkFollowUpActionResultResponse,
    AgentCheckpointQueueItemResponse,
    AgentCheckpointQueueResponse,
    AgentJobCreate,
    AgentJobQuickStartBugTriageSwarmRequest,
    AgentJobQuickStartBuildBreakSwarmRequest,
    AgentJobFromTemplate,
    AgentJobQuickStartClaudeBackendRequest,
    AgentJobQuickStartDomainResearchRequest,
    AgentJobPromoteDomainResearchRequest,
    AgentJobPromoteDomainResearchResponse,
    AgentJobQuickStartFrontendRegressionSwarmRequest,
    AgentJobQuickStartRepoBugTriageRequest,
    AgentJobQuickStartRoleWorkflowRequest,
    AgentJobUpdate,
    AgentJobResponse,
    AgentJobListResponse,
    AgentJobDetailResponse,
    AgentJobRelaunchLineageNode,
    AgentJobRelaunchLineageResponse,
    AgentJobTemplateResponse,
    AgentJobTemplateListResponse,
    AgentJobActionRequest,
    AgentJobStatsResponse,
    AgentJobSwarmAnalyticsResponse,
    AgentJobSwarmAnalyticsPresetRowResponse,
    AgentJobSwarmOutcomeAnalyticsResponse,
    AgentJobSwarmOutcomeCaseResponse,
    AgentJobSwarmOutcomePresetRowResponse,
    AgentJobCheckpointResponse,
    # Chain schemas
    AgentJobChainDefinitionCreate,
    AgentJobChainDefinitionUpdate,
    AgentJobChainDefinitionResponse,
    AgentJobChainDefinitionListResponse,
    AgentJobFromChainCreate,
    AgentJobChainStatusResponse,
    AgentJobSaveAsChainRequest,
    AgentJobFeedbackCreate,
    AgentJobFeedbackResponse,
    AgentJobFeedbackListResponse,
    AgentJobExtractedMemoryResponse,
    AgentJobMemoryResponse,
    AgentJobMemoryListResponse,
    AgentJobMemoryDeleteResponse,
    AgentJobMemoryExtractResponse,
    AgentJobMemoryStatsResponse,
    AgentJobMemorySearchItemResponse,
    AgentJobMemorySearchResponse,
    AgentJobMemoryGraphResponse,
)
from app.schemas.domain_research_profile import DomainResearchProfileCreate
from app.schemas.research_portfolio import ResearchPortfolioCreate
from app.tasks.agent_job_tasks import execute_agent_job_task, generate_job_summary
from app.models.ai_hub_recommendation_feedback import AIHubRecommendationFeedback
from app.core.feature_flags import get_str as get_feature_str
from app.schemas.customer_profile import CustomerProfile
from app.schemas.ai_hub_recommendation_feedback import (
    AIHubRecommendationFeedbackCreate,
    AIHubRecommendationFeedbackResponse,
    AIHubRecommendationFeedbackListResponse,
)
from app.services.agent_job_scheduler_state import (
    extract_scheduler_state,
    queue_reason_label,
)
from app.services.agent_job_queue_helpers import (
    extract_approval_checkpoint,
    extract_launch_mode,
    parse_optional_datetime,
    queue_age_minutes,
    queue_customer_for_job,
    queue_evidence_summary_for_job,
)
from app.services.operator_interventions import derive_operator_interventions_with_outcomes
from app.services.research_inbox_follow_up_service import (
    project_follow_up_relaunch_to_originating_opportunity,
    sync_follow_up_outcome_for_job,
)
from app.services.research_monitor_profile_service import research_monitor_profile_service
from app.services.research_opportunity_service import (
    classify_portfolio_operator_review,
    compute_research_portfolio_config_revision,
    collect_research_opportunity_linked_ids,
    list_normalized_research_opportunities,
    summarize_portfolio_operator_reviews,
    summarize_research_opportunity_autonomy_states,
    summarize_research_opportunity_stages,
)
from app.services.autonomy_service import (
    build_autonomy_summary,
    build_domain_profile_compat_policy,
    build_monitor_policy_compat_fields,
    current_domain_profile_policy_snapshot,
    resolve_domain_profile_automation_contract,
)
from app.services.autonomy_event_service import (
    apply_decision_trace_escalation,
    compute_decision_trace_escalation,
    event_to_trace_payload,
    maybe_emit_escalation_transition_notification,
    maybe_reopen_event_notification,
    record_autonomy_decision_event,
)
from app.services.collaboration_service import list_collaboration_user_ids
from app.services.collaboration_service import build_collaboration_summary
from app.services.scientific_validation_service import (
    get_scientific_sandbox_profile,
    normalize_portfolio_automation_profile,
    resolve_portfolio_automation_policy,
)
from app.services.agent_job_templates import (
    CLAUDE_CODE_BACKEND_TEMPLATE_ID,
    DOMAIN_RESEARCH_TEMPLATE_ID,
    REPO_BUG_TRIAGE_REPAIR_TEMPLATE_ID,
    get_builtin_agent_job_template,
    list_builtin_agent_job_templates,
)
from app.services.agent_job_chain_templates import (
    ARXIV_REPO_CODE_PATCH_CHAIN_ID,
    CUSTOMER_RESEARCH_SCOUT_DEEP_DIVE_CHAIN_ID,
    get_builtin_agent_job_chain_definition,
    list_builtin_agent_job_chain_definitions,
)
from app.services.autonomous_agent_executor import AutonomousAgentExecutor

# NOTE: api/routes.py mounts this router under `/agent-jobs`.
# Do not set a prefix here, otherwise routes become `/agent-jobs/agent-jobs/...`.
router = APIRouter()

QUEUE_BULK_ACTIONS: dict[str, set[str]] = {
    "approval_checkpoint": {"approve", "reject", "skip"},
    "job_recovery": {"restart", "resume", "cancel"},
}

FOLLOW_UP_AUTONOMY_MANUAL_ONLY = "manual_only"
FOLLOW_UP_AUTONOMY_AUTO_LAUNCH_SAFE = "auto_launch_safe"
FOLLOW_UP_AUTONOMY_QUEUE_FOR_APPROVAL = "queue_for_approval"

FOLLOW_UP_RECOMMENDATION_DEEP_DIVE_CHAIN = "deep_dive_chain"
FOLLOW_UP_RECOMMENDATION_SINGLE_RESEARCH_JOB = "single_research_job"
FOLLOW_UP_RECOMMENDATION_REPO_PATCH_CHAIN = "repo_patch_chain"


def _queue_priority_fields(
    *,
    item_type: str,
    reason_code: Optional[str],
    created_at: Optional[datetime],
    next_run_at: Optional[datetime],
    backoff_until: Optional[datetime],
    stale: bool,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    reference = now or datetime.utcnow()
    age_minutes = _queue_age_minutes(created_at, now=reference)
    priority_score = 0.0
    sla_bucket = "normal"
    escalation_level = "normal"
    is_overdue = False
    is_stale = bool(stale)
    normalized_reason = str(reason_code or "").strip().lower()

    if item_type == "approval_checkpoint":
        priority_score = 100 + min(age_minutes, 720) / 8
        if age_minutes >= 240:
            sla_bucket = "overdue"
            escalation_level = "high"
            is_overdue = True
        elif age_minutes >= 60:
            sla_bucket = "at_risk"
            escalation_level = "medium"
        else:
            sla_bucket = "normal"
            escalation_level = "normal"
    elif item_type == "job_recovery":
        priority_score = 80 + min(age_minutes, 720) / 12
        if normalized_reason == "execution_failure":
            priority_score += 16
        elif normalized_reason == "stalled_run":
            priority_score += 20
        elif normalized_reason in {"scheduler_backoff", "scheduled_recovery"}:
            priority_score += 10
        if backoff_until and backoff_until <= reference:
            priority_score += 14
            is_overdue = True
        if next_run_at and next_run_at <= reference:
            priority_score += 8
            is_overdue = True
        if is_stale:
            priority_score += 18
            is_overdue = True
        if is_overdue or age_minutes >= 180:
            sla_bucket = "overdue"
            escalation_level = "high"
        elif age_minutes >= 45:
            sla_bucket = "at_risk"
            escalation_level = "medium"
        else:
            sla_bucket = "normal"
            escalation_level = "normal"
    elif item_type == "policy_review":
        priority_score = 90 + min(age_minutes, 720) / 10
        if age_minutes >= 180:
            priority_score += 10
            sla_bucket = "overdue"
            escalation_level = "high"
            is_overdue = True
        elif age_minutes >= 30:
            sla_bucket = "at_risk"
            escalation_level = "medium"
        else:
            sla_bucket = "normal"
            escalation_level = "normal"
    elif item_type == "budget_review":
        priority_score = 74 + min(age_minutes, 720) / 14
        if age_minutes >= 240:
            priority_score += 8
            sla_bucket = "overdue"
            escalation_level = "high"
            is_overdue = True
        elif age_minutes >= 60:
            sla_bucket = "at_risk"
            escalation_level = "medium"
        else:
            sla_bucket = "normal"
            escalation_level = "normal"
    else:
        priority_score = 60 + min(age_minutes, 720) / 18
        if age_minutes >= 1440:
            priority_score += 12
            sla_bucket = "at_risk"
            escalation_level = "medium"
        else:
            sla_bucket = "normal"
            escalation_level = "normal"

    return {
        "priority_score": round(float(priority_score), 2),
        "age_minutes": age_minutes,
        "sla_bucket": sla_bucket,
        "escalation_level": escalation_level,
        "is_overdue": is_overdue,
        "is_stale": is_stale,
    }


def _is_source_owned_by_user(source: "DocumentSource", user: User) -> bool:
    cfg = source.config or {}
    if not isinstance(cfg, dict):
        return False
    requested_by = cfg.get("requested_by") or cfg.get("requestedBy")
    requested_by_user_id = cfg.get("requested_by_user_id") or cfg.get("requestedByUserId")
    return requested_by in {user.username, str(user.id)} or requested_by_user_id == str(user.id)


def _build_quick_start_claude_backend_config(
    request: AgentJobQuickStartClaudeBackendRequest,
    *,
    source_name: str,
    source_type: str,
) -> dict:
    """Build normalized config payload for Claude backend quick start."""
    merged_config: dict = {
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
        merged_config["search_query"] = str(request.search_query)
    if isinstance(request.file_paths, list):
        merged_config["file_paths"] = [str(p).strip() for p in request.file_paths if str(p).strip()]
    if isinstance(request.commands, list):
        merged_config["commands"] = [str(c).strip() for c in request.commands if str(c).strip()]
    if isinstance(request.config_overrides, dict):
        merged_config.update(_normalize_scope_config(request.config_overrides) or {})
    return _normalize_scope_config(merged_config)


def _build_domain_research_goal(request: AgentJobQuickStartDomainResearchRequest) -> str:
    domain = str(request.domain or "").strip()
    objective = str(request.objective or "").strip()
    customer_context = str(request.customer_context or "").strip()
    goal = f"Research the domain '{domain}'. Objective: {objective}"
    if customer_context:
        goal += f"\nContext: {customer_context}"
    return goal


def _build_quick_start_domain_research_config(
    request: AgentJobQuickStartDomainResearchRequest,
) -> dict:
    """Build normalized config payload for domain research quick start."""
    source_scope = str(request.source_scope or "kb_plus_arxiv").strip().lower()
    track_type = str(request.track_type or "generic").strip().lower()
    domain = str(request.domain or "").strip()
    objective = str(request.objective or "").strip()
    monitor_queries = [str(q).strip() for q in (request.monitor_queries or []) if str(q).strip()]
    if not monitor_queries:
        monitor_queries = [f"{domain} {objective}".strip()[:240]]
    benchmark_queries = [str(q).strip() for q in (request.benchmark_queries or []) if str(q).strip()][:16]
    repo_source_ids = [str(source_id).strip() for source_id in (request.repo_source_ids or []) if str(source_id).strip()][:24]

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
        explicit_legacy_updates["auto_create_experiment_plans"] = request.auto_create_experiment_plans
    if "confidence_threshold" in request.model_fields_set:
        explicit_legacy_updates["confidence_threshold"] = request.confidence_threshold

    automation_profile, normalized_validation_policy = resolve_domain_profile_automation_contract(
        automation_profile=request.automation_profile,
        automation_policy=request.automation_policy,
        current_snapshot={"validation_policy": request.validation_policy} if isinstance(request.validation_policy, dict) else None,
        explicit_updates=explicit_legacy_updates or None,
    )
    confidence_threshold = float(normalized_validation_policy.get("confidence_threshold") or request.confidence_threshold or 0.7)
    auto_create_experiment_plans = bool(normalized_validation_policy.get("auto_create_experiment_plans", request.auto_create_experiment_plans))
    auto_launch_follow_up = bool(normalized_validation_policy.get("auto_launch_follow_up", request.auto_launch_follow_up))

    merged_config: dict[str, Any] = {
        "launch_mode": "quick_start_domain_research",
        "deterministic_runner": "domain_research_orchestrator",
        "domain_research_mode": True,
        "profile_id": str(request.profile_id) if request.profile_id else None,
        "domain": domain,
        "objective": objective,
        "customer_context": str(request.customer_context or "").strip(),
        "source_scope": source_scope,
        "track_type": track_type,
        "research_mode": str(request.research_mode or "literature_to_hypothesis").strip().lower(),
        "monitor_queries": monitor_queries[:12],
        "repo_source_ids": repo_source_ids or None,
        "benchmark_queries": benchmark_queries or None,
        "sandbox_profile_id": str(request.sandbox_profile_id or "").strip() or None,
        "report_format": str(request.report_format or "brief_and_report").strip().lower(),
        "scoring_policy": request.scoring_policy if isinstance(request.scoring_policy, dict) else None,
        "selection_policy": request.selection_policy if isinstance(request.selection_policy, dict) else None,
        "persist_artifacts": bool(request.persist_artifacts),
        "persist_target": "research_notes",
        "automation_profile": automation_profile,
        "automation_policy": normalized_validation_policy,
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
            "research_mode": str(request.research_mode or "literature_to_hypothesis").strip().lower(),
            "persist_target": "research_notes",
            "report_format": str(request.report_format or "brief_and_report").strip().lower(),
            "automation_profile": automation_profile,
            "auto_launch_follow_up": auto_launch_follow_up,
            "auto_create_experiment_plans": auto_create_experiment_plans,
            "profile_id": str(request.profile_id) if request.profile_id else None,
        },
    }
    if "validation_policy" in request.model_fields_set:
        merged_config["validation_policy"] = normalized_validation_policy
    if isinstance(request.config_overrides, dict):
        merged_config.update(_normalize_scope_config(request.config_overrides) or {})
    return _normalize_scope_config(merged_config)


def _extract_domain_research_promotion(job: AgentJob) -> dict[str, Any]:
    cfg = job.config if isinstance(job.config, dict) else {}
    quick_start = cfg.get("quick_start") if isinstance(cfg.get("quick_start"), dict) else {}
    results = job.results if isinstance(job.results, dict) else {}

    promotion = cfg.get("promotion") if isinstance(cfg.get("promotion"), dict) else {}
    if not promotion and isinstance(quick_start.get("promotion"), dict):
        promotion = quick_start.get("promotion") or {}
    if not promotion and isinstance(results.get("promotion"), dict):
        promotion = results.get("promotion") or {}
    return dict(promotion) if isinstance(promotion, dict) else {}


def _build_domain_research_promotion_seed(job: AgentJob) -> dict[str, Any]:
    cfg = job.config if isinstance(job.config, dict) else {}
    if _extract_launch_mode(cfg) != "quick_start_domain_research":
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Job is not a domain research quick start")

    domain = str(cfg.get("domain") or "").strip()
    objective = str(cfg.get("objective") or "").strip()
    if not domain or not objective:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Job is missing normalized domain research config")

    automation_profile, automation_policy = resolve_domain_profile_automation_contract(
        automation_profile=cfg.get("automation_profile"),
        automation_policy=cfg.get("automation_policy"),
        current_snapshot={"validation_policy": cfg.get("validation_policy")} if isinstance(cfg.get("validation_policy"), dict) else None,
    )

    monitor_queries = [str(q).strip() for q in (cfg.get("monitor_queries") or []) if str(q).strip()][:12]
    benchmark_queries = [str(q).strip() for q in (cfg.get("benchmark_queries") or []) if str(q).strip()][:16]
    repo_source_ids = [str(v).strip() for v in (cfg.get("repo_source_ids") or []) if str(v).strip()][:24]
    title = str(job.name or "").strip()[:200] or f"{domain[:120]} Monitor"

    return {
        "profile": {
            "title": title,
            "domain": domain,
            "objective": objective,
            "customer_context": str(cfg.get("customer_context") or "").strip() or None,
            "source_scope": str(cfg.get("source_scope") or "kb_plus_arxiv_plus_repo").strip(),
            "track_type": str(cfg.get("track_type") or "compiler").strip(),
            "research_mode": str(cfg.get("research_mode") or "literature_to_hypothesis").strip(),
            "monitor_queries": monitor_queries or [f"{domain} {objective}".strip()[:240]],
            "repo_source_ids": repo_source_ids or None,
            "benchmark_queries": benchmark_queries or None,
            "report_format": str(cfg.get("report_format") or "brief_and_report").strip(),
            "scoring_policy": cfg.get("scoring_policy") if isinstance(cfg.get("scoring_policy"), dict) else None,
            "selection_policy": cfg.get("selection_policy") if isinstance(cfg.get("selection_policy"), dict) else None,
            "automation_profile": automation_profile,
            "automation_policy": automation_policy,
            "sandbox_profile_id": str(cfg.get("sandbox_profile_id") or "").strip() or None,
            "interval_minutes": int(cfg.get("interval_minutes") or 1440),
            "persist_artifacts": bool(cfg.get("persist_artifacts", True)),
            "auto_launch_follow_up": bool(automation_policy.get("auto_launch_follow_up", cfg.get("auto_launch_follow_up", True))),
            "auto_create_experiment_plans": bool(automation_policy.get("auto_create_experiment_plans", cfg.get("auto_create_experiment_plans", True))),
            "confidence_threshold": float(automation_policy.get("confidence_threshold", cfg.get("confidence_threshold") or 0.7)),
            "max_documents": int(cfg.get("max_documents") or 10),
            "max_papers": int(cfg.get("max_papers") or 8),
            "start_immediately": False,
        },
        "portfolio": {
            "title": f"{domain[:160]} Fleet",
            "objective": objective,
            "sandbox_profile_id": str(cfg.get("sandbox_profile_id") or "").strip() or None,
            "automation_profile": normalize_portfolio_automation_profile(cfg.get("automation_profile"), default="balanced"),
            "automation_policy": resolve_portfolio_automation_policy(cfg.get("automation_profile"), cfg.get("automation_policy")),
            "start_immediately": False,
        },
    }


async def _validate_domain_research_sandbox_profile(
    db: AsyncSession,
    *,
    sandbox_profile_id: Optional[str],
    track_type: Optional[str],
) -> None:
    requested = str(sandbox_profile_id or "").strip()
    if not requested:
        return
    profile = await get_scientific_sandbox_profile(
        db,
        requested,
        track_type=str(track_type or "").strip() or None,
        include_disabled=False,
    )
    if not isinstance(profile, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown or disabled sandbox profile")


def _build_repo_bug_triage_goal(request: AgentJobQuickStartRepoBugTriageRequest) -> str:
    symptom = str(request.failure_symptom or "").strip()
    goal = str(request.goal or "").strip()
    scope = str(request.scope or "auto").strip().lower()

    if symptom and goal:
        return f"Triage and repair the reported {scope} bug. Symptom: {symptom}\nDesired outcome: {goal}"
    if symptom:
        return f"Triage and repair the reported {scope} bug. Symptom: {symptom}"
    return goal


def _build_bug_triage_swarm_goal(request: AgentJobQuickStartBugTriageSwarmRequest) -> str:
    symptom = str(request.failure_symptom or "").strip()
    goal = str(request.goal or "").strip()
    scope = str(request.scope or "auto").strip().lower()

    if symptom and goal:
        return (
            f"Run a coding bug triage swarm for the reported {scope} bug.\n"
            f"Symptom: {symptom}\n"
            f"Desired outcome: {goal}"
        )
    if symptom:
        return f"Run a coding bug triage swarm for the reported {scope} bug. Symptom: {symptom}"
    return goal


_CODING_SWARM_PRESET_DEFINITIONS: dict[str, dict[str, Any]] = {
    "bug_triage_swarm": {
        "launch_mode": "quick_start_bug_triage_swarm",
        "coding_profile": "bug_triage",
        "quick_start_profile": "bug_triage_swarm",
        "display_name": "Bug Triage Swarm",
        "goal_prefix": "Run a coding bug triage swarm",
        "default_scope": "auto",
        "default_search_suffix": "bug symptom",
        "roles": ["reproducer", "root_cause", "patcher", "verifier"],
        "fan_in_name": "Bug Triage Swarm Fan-In",
        "confidence_threshold": 0.70,
        "tiebreaker_threshold": 0.50,
    },
    "build_break_swarm": {
        "launch_mode": "quick_start_build_break_swarm",
        "coding_profile": "build_break",
        "quick_start_profile": "build_break_swarm",
        "display_name": "Build Break Swarm",
        "goal_prefix": "Run a coding swarm for the reported build break",
        "default_scope": "backend",
        "default_search_suffix": "build break compile failure",
        "roles": ["reproducer", "root_cause", "patcher", "verifier"],
        "fan_in_name": "Build Break Swarm Fan-In",
        "confidence_threshold": 0.72,
        "tiebreaker_threshold": 0.52,
    },
    "frontend_regression_swarm": {
        "launch_mode": "quick_start_frontend_regression_swarm",
        "coding_profile": "frontend_regression",
        "quick_start_profile": "frontend_regression_swarm",
        "display_name": "Frontend Regression Swarm",
        "goal_prefix": "Run a coding swarm for the reported frontend regression",
        "default_scope": "frontend",
        "default_search_suffix": "frontend regression ui failure",
        "roles": ["reproducer", "root_cause", "patcher", "verifier"],
        "fan_in_name": "Frontend Regression Swarm Fan-In",
        "confidence_threshold": 0.70,
        "tiebreaker_threshold": 0.50,
    },
}


def _get_coding_swarm_preset_definition(preset_key: str) -> dict[str, Any]:
    preset = _CODING_SWARM_PRESET_DEFINITIONS.get(str(preset_key or "").strip().lower())
    if not isinstance(preset, dict):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unknown coding swarm preset")
    return preset


def _build_coding_swarm_goal(
    request: AgentJobQuickStartBugTriageSwarmRequest | AgentJobQuickStartBuildBreakSwarmRequest | AgentJobQuickStartFrontendRegressionSwarmRequest,
    *,
    preset_key: str,
) -> str:
    preset = _get_coding_swarm_preset_definition(preset_key)
    symptom = str(request.failure_symptom or "").strip()
    goal = str(request.goal or "").strip()
    scope = str(request.scope or preset.get("default_scope") or "auto").strip().lower()
    goal_prefix = str(preset.get("goal_prefix") or "Run a coding swarm").strip()

    if symptom and goal:
        return f"{goal_prefix} for the reported {scope} issue.\nSymptom: {symptom}\nDesired outcome: {goal}"
    if symptom:
        return f"{goal_prefix} for the reported {scope} issue. Symptom: {symptom}"
    return goal


def _merge_coding_swarm_request_with_profile(
    request: AgentJobQuickStartBugTriageSwarmRequest | AgentJobQuickStartBuildBreakSwarmRequest | AgentJobQuickStartFrontendRegressionSwarmRequest,
    *,
    profile: Optional[CodingSwarmProfile],
    preset_key: str,
) -> AgentJobQuickStartBugTriageSwarmRequest | AgentJobQuickStartBuildBreakSwarmRequest | AgentJobQuickStartFrontendRegressionSwarmRequest:
    if profile is None:
        return request

    preset = _get_coding_swarm_preset_definition(preset_key)
    profile_preset_key = str(getattr(profile, "preset_key", "") or "").strip().lower()
    if profile_preset_key and profile_preset_key != preset_key:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Selected coding swarm profile is for preset '{profile_preset_key}', not '{preset_key}'",
        )
    if str(getattr(profile, "status", "active") or "active").strip().lower() not in {"active", "enabled"}:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Coding swarm profile is not active")

    payload = request.model_dump(exclude_none=False)
    if str(payload.get("source_id") or "") != str(profile.source_id):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Coding swarm profile source does not match request source")

    if not str(payload.get("scope") or "").strip():
        payload["scope"] = str(profile.scope_default or preset.get("default_scope") or "auto").strip().lower()
    if not str(payload.get("search_query") or "").strip() and str(profile.saved_search_query or "").strip():
        payload["search_query"] = str(profile.saved_search_query).strip()
    if not isinstance(payload.get("commands"), list) or not payload.get("commands"):
        payload["commands"] = list(profile.default_commands or []) or None
    if not isinstance(payload.get("file_paths"), list) or not payload.get("file_paths"):
        payload["file_paths"] = list(profile.default_file_paths or []) or None
    if not int(payload.get("max_agents") or 0):
        payload["max_agents"] = int(profile.max_agents or 4)
    payload["profile_id"] = profile.id
    return request.__class__(**payload)


def _normalize_uuid_str_list(values: object, limit: int = 100) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        try:
            value = str(UUID(str(raw))).strip()
        except Exception:
            continue
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= limit:
            break
    return out


def _normalize_coding_swarm_profile_visibility(value: object) -> str:
    return "shared" if str(value or "private").strip().lower() == "shared" else "private"


def _is_coding_swarm_profile_visible_to_user(profile: CodingSwarmProfile, user: User) -> bool:
    if user.is_admin() or str(profile.user_id) == str(user.id):
        return True
    if _normalize_coding_swarm_profile_visibility(getattr(profile, "visibility", "private")) != "shared":
        return False
    return str(user.id) in _normalize_uuid_str_list(getattr(profile, "shared_with_user_ids", None), 200)


def _build_swarm_collaboration_payload(
    *,
    owner_user_id: UUID | str,
    visibility: str = "private",
    shared_with_user_ids: Optional[list[str]] = None,
    assigned_user_id: Optional[str] = None,
    assigned_by_user_id: Optional[str] = None,
    assigned_at: Optional[str] = None,
    review_note: Optional[str] = None,
) -> dict[str, Any]:
    normalized_shared = _normalize_uuid_str_list(shared_with_user_ids or [], 200)
    if assigned_user_id:
        normalized_shared = _normalize_uuid_str_list([*normalized_shared, assigned_user_id], 200)
    return {
        "owner_user_id": str(owner_user_id),
        "shared_review": _normalize_coding_swarm_profile_visibility(visibility) == "shared" or bool(normalized_shared),
        "shared_with_user_ids": normalized_shared,
        "assigned_user_id": str(assigned_user_id).strip() if assigned_user_id else None,
        "assigned_by_user_id": str(assigned_by_user_id).strip() if assigned_by_user_id else None,
        "assigned_at": str(assigned_at).strip() if assigned_at else None,
        "review_note": str(review_note or "").strip() or None,
    }


def _extract_swarm_collaboration(job: AgentJob) -> dict[str, Any]:
    results = job.results if isinstance(job.results, dict) else {}
    raw = results.get("swarm_collaboration") if isinstance(results.get("swarm_collaboration"), dict) else {}
    return _build_swarm_collaboration_payload(
        owner_user_id=raw.get("owner_user_id") or getattr(job, "user_id", None) or uuid.uuid4(),
        visibility="shared" if bool(raw.get("shared_review")) or _normalize_uuid_str_list(raw.get("shared_with_user_ids"), 200) else "private",
        shared_with_user_ids=_normalize_uuid_str_list(raw.get("shared_with_user_ids"), 200),
        assigned_user_id=str(raw.get("assigned_user_id") or "").strip() or None,
        assigned_by_user_id=str(raw.get("assigned_by_user_id") or "").strip() or None,
        assigned_at=str(raw.get("assigned_at") or "").strip() or None,
        review_note=str(raw.get("review_note") or "").strip() or None,
    )


def _is_job_visible_to_user(job: AgentJob, user: User) -> bool:
    if user.is_admin() or str(job.user_id) == str(user.id):
        return True
    cfg = job.config if isinstance(job.config, dict) else {}
    if not _infer_coding_swarm_preset_key(job) and str(_extract_launch_mode(cfg) or "").strip() != "bug_triage_swarm_repair_handoff":
        return False
    collaboration = _extract_swarm_collaboration(job)
    if str(collaboration.get("assigned_user_id") or "").strip() == str(user.id):
        return True
    return str(user.id) in _normalize_uuid_str_list(collaboration.get("shared_with_user_ids"), 200)


def _normalize_coding_backlog_visibility(value: object) -> str:
    return "shared" if str(value or "private").strip().lower() == "shared" else "private"


def _is_backlog_item_visible_to_user(item: CodingBacklogItem, user: User) -> bool:
    if user.is_admin() or str(item.user_id) == str(user.id):
        return True
    if str(getattr(item, "assigned_user_id", "") or "").strip() == str(user.id):
        return True
    if _normalize_coding_backlog_visibility(getattr(item, "visibility", "private")) != "shared":
        return False
    return str(user.id) in _normalize_uuid_str_list(getattr(item, "shared_with_user_ids", None), 200)


def _extract_backlog_route_mode(item: Optional[CodingBacklogItem]) -> Optional[str]:
    if item is None:
        return None
    lineage = item.lineage if isinstance(item.lineage, dict) else {}
    mode = str(lineage.get("originating_swarm_route_mode") or "").strip().lower()
    if mode in {"auto", "manual"}:
        return mode
    return None


def _store_swarm_collaboration(job: AgentJob, collaboration: dict[str, Any]) -> None:
    results_payload = dict(job.results) if isinstance(job.results, dict) else {}
    results_payload["swarm_collaboration"] = collaboration
    job.results = results_payload


async def _resolve_coding_swarm_profile(
    db: AsyncSession,
    *,
    current_user: User,
    source_id: UUID,
    profile_id: Optional[UUID],
    preset_key: str,
) -> Optional[CodingSwarmProfile]:
    if profile_id:
        profile = await db.get(CodingSwarmProfile, profile_id)
        if not profile or not _is_coding_swarm_profile_visible_to_user(profile, current_user):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Coding swarm profile not found")
        if str(profile.source_id) != str(source_id):
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Coding swarm profile source does not match request source")
        return profile

    row = (
        await db.execute(
            select(CodingSwarmProfile)
            .where(
                CodingSwarmProfile.user_id == current_user.id,
                CodingSwarmProfile.source_id == source_id,
                CodingSwarmProfile.preset_key == preset_key,
                CodingSwarmProfile.is_default.is_(True),
            )
            .order_by(desc(CodingSwarmProfile.updated_at))
            .limit(1)
        )
    ).scalars().first()
    return row


def _build_quick_start_coding_swarm_config(
    request: AgentJobQuickStartBugTriageSwarmRequest | AgentJobQuickStartBuildBreakSwarmRequest | AgentJobQuickStartFrontendRegressionSwarmRequest,
    *,
    source_name: str,
    source_type: str,
    preset_key: str,
) -> dict:
    preset = _get_coding_swarm_preset_definition(preset_key)
    scope = str(request.scope or preset.get("default_scope") or "auto").strip().lower()
    symptom = str(request.failure_symptom or "").strip()
    search_query = str(request.search_query or "").strip()
    if not search_query:
        scope_hint = "" if scope == "auto" else scope.replace("_", " ")
        default_search_suffix = str(preset.get("default_search_suffix") or "").strip()
        search_query = " ".join(part for part in [scope_hint, symptom, default_search_suffix] if part).strip()[:500]

    max_agents = max(1, min(int(request.max_agents or 4), 4))
    swarm_roles = list(preset.get("roles") or ["reproducer", "root_cause", "patcher", "verifier"])[:max_agents]

    merged_config: dict[str, Any] = {
        "source_id": str(request.source_id),
        "launch_mode": str(preset.get("launch_mode") or "").strip(),
        "failure_symptom": symptom,
        "scope": scope or str(preset.get("default_scope") or "auto"),
        "quick_start": {
            "profile": str(preset.get("quick_start_profile") or preset_key).strip(),
            "version": "v1",
            "source_name": str(source_name or "").strip(),
            "source_type": str(source_type or "").strip().lower(),
            "scope": scope or str(preset.get("default_scope") or "auto"),
            "autonomy_mode": "max_autonomy",
            "entry_point": "dedicated_quick_start",
            "max_agents": max_agents,
            "roles": swarm_roles,
            "preset_key": preset_key,
            "profile_id": str(request.profile_id) if getattr(request, "profile_id", None) else None,
        },
        "plan_then_act_enabled": True,
        "plan_max_steps": 6,
        "subgoal_decomposition_enabled": False,
        "swarm_child_jobs_enabled": True,
        "swarm_max_agents": max_agents,
        "swarm_roles": swarm_roles,
        "swarm_inherit_results": True,
        "swarm_inherit_config": True,
        "swarm_fan_in_enabled": True,
        "swarm_fan_in_name": str(preset.get("fan_in_name") or "Coding Swarm Fan-In").strip(),
        "swarm_fan_in_trigger_condition": "on_any_end",
        "coding_swarm_enabled": True,
        "coding_swarm_profile": str(preset.get("coding_profile") or preset_key).strip(),
        "coding_swarm_preset_key": preset_key,
        "coding_swarm_auto_promote_best_slice": True,
        "coding_swarm_auto_launch_repair_chain": True,
        "coding_swarm_confidence_threshold": float(preset.get("confidence_threshold") or 0.70),
        "coding_swarm_tiebreaker_threshold": float(preset.get("tiebreaker_threshold") or 0.50),
        "coding_swarm_repair_chain_name": "repo_bug_triage_repair",
        "create_workspace_from_source": True,
        "emit_execution_plan": True,
        "auto_commands_from_project_profile": True,
        "max_verification_commands": 3,
        "apply_patch_to_kb": False,
        "apply_patch_to_kb_confirm": False,
        "enable_memory": False,
    }
    if search_query:
        merged_config["search_query"] = search_query
    if request.error_output is not None:
        merged_config["error_output"] = str(request.error_output)
    if isinstance(request.file_paths, list):
        merged_config["file_paths"] = [str(p).strip() for p in request.file_paths if str(p).strip()]
    if isinstance(request.commands, list):
        merged_config["commands"] = [str(c).strip() for c in request.commands if str(c).strip()]
    if isinstance(request.config_overrides, dict):
        merged_config.update(_normalize_scope_config(request.config_overrides) or {})
    return _normalize_scope_config(merged_config)


def _build_quick_start_repo_bug_triage_config(
    request: AgentJobQuickStartRepoBugTriageRequest,
    *,
    source_name: str,
    source_type: str,
) -> dict:
    """Build normalized config payload for repo bug triage quick start."""
    scope = str(request.scope or "auto").strip().lower()
    symptom = str(request.failure_symptom or "").strip()
    search_query = str(request.search_query or "").strip()
    if not search_query:
        scope_hint = "" if scope == "auto" else scope.replace("_", " ")
        search_query = " ".join(part for part in [scope_hint, symptom] if part).strip()[:500]

    merged_config: dict = {
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
        merged_config["search_query"] = search_query
    if request.error_output is not None:
        merged_config["error_output"] = str(request.error_output)
    if isinstance(request.file_paths, list):
        merged_config["file_paths"] = [str(p).strip() for p in request.file_paths if str(p).strip()]
    if isinstance(request.commands, list):
        merged_config["commands"] = [str(c).strip() for c in request.commands if str(c).strip()]
    if isinstance(request.config_overrides, dict):
        merged_config.update(_normalize_scope_config(request.config_overrides) or {})
    return _normalize_scope_config(merged_config)


def _build_quick_start_bug_triage_swarm_config(
    request: AgentJobQuickStartBugTriageSwarmRequest,
    *,
    source_name: str,
    source_type: str,
) -> dict:
    return _build_quick_start_coding_swarm_config(
        request,
        source_name=source_name,
        source_type=source_type,
        preset_key="bug_triage_swarm",
    )


def _coerce_bool(value: Any, default: bool = False) -> bool:
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


def _extract_enable_memory_from_config(config: Optional[dict], default: bool = True) -> bool:
    if not isinstance(config, dict):
        return bool(default)
    if "enable_memory" in config:
        return _coerce_bool(config.get("enable_memory"), default=default)
    memory = config.get("memory")
    if isinstance(memory, dict) and "enabled" in memory:
        return _coerce_bool(memory.get("enabled"), default=default)
    return bool(default)


def _normalize_swarm_roles(roles: Any, *, max_roles: int = 12) -> list[str]:
    if not isinstance(roles, list):
        return []
    out: list[str] = []
    for raw in roles:
        role = str(raw or "").strip().lower()
        if not role:
            continue
        role = role.replace("-", "_").replace(" ", "_")
        if not re.match(r"^[a-z0-9_:\-]{2,120}$", role):
            continue
        if role not in out:
            out.append(role)
        if len(out) >= max(1, min(max_roles, 12)):
            break
    return out


def _build_role_workflow_memory_config(
    memory_profile: str,
) -> tuple[dict[str, Any], bool]:
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


def _build_role_workflow_approval_config(approval_mode: str) -> dict[str, Any]:
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


def _build_quick_start_role_workflow_config(
    request: AgentJobQuickStartRoleWorkflowRequest,
) -> dict:
    roles = _normalize_swarm_roles(request.roles or [])
    if not roles:
        roles = ["researcher_documents", "researcher_arxiv", "analyst", "synthesizer"]
    max_agents = int(request.max_agents or len(roles) or 4)
    max_agents = max(1, min(max_agents, 12))
    roles = roles[:max_agents]

    memory_cfg, enable_memory = _build_role_workflow_memory_config(str(request.memory_profile or "balanced"))
    approval_cfg = _build_role_workflow_approval_config(str(request.approval_mode or "high_impact"))
    execution_mode = str(request.execution_mode or "plan_and_execute").strip().lower()
    if execution_mode not in {"plan_and_execute", "adaptive"}:
        execution_mode = "plan_and_execute"
    extract_on_failure = bool(request.extract_memory_on_failure if request.extract_memory_on_failure is not None else True)
    extract_statuses = ["completed"] + (["failed"] if extract_on_failure else [])
    memory_cfg["extract_on_statuses"] = extract_statuses
    memory_cfg["extract_on_failure"] = extract_on_failure
    if isinstance(request.memory_failed_types, list) and request.memory_failed_types:
        memory_cfg["failed_extraction_types"] = [str(x).strip().lower() for x in request.memory_failed_types if str(x).strip()][:12]
    if isinstance(request.memory_completed_types, list) and request.memory_completed_types:
        memory_cfg["completed_extraction_types"] = [str(x).strip().lower() for x in request.memory_completed_types if str(x).strip()][:12]

    merged_config: dict[str, Any] = {
        "launch_mode": "quick_start_role_workflow",
        "quick_start": {
            "profile": "role_workflow",
            "version": "v1",
            "memory_profile": str(memory_cfg.get("profile") or "balanced"),
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
        "approval_checkpoints": approval_cfg,
        "memory": memory_cfg,
        "enable_memory": bool(enable_memory),
    }
    if isinstance(request.config_overrides, dict):
        merged_config.update(_normalize_scope_config(request.config_overrides) or {})
    return _normalize_scope_config(merged_config)


def _extract_source_id_from_config(config: Optional[dict]) -> str:
    normalized = _normalize_scope_config(config if isinstance(config, dict) else None) or {}
    return str(normalized.get("source_id") or "").strip()


def _extract_relaunch_parent_job_id(config: Optional[dict]) -> Optional[UUID]:
    if not isinstance(config, dict):
        return None
    raw = str(config.get("relaunch_from_job_id") or "").strip()
    if not raw:
        return None
    try:
        return UUID(raw)
    except Exception:
        return None


def _build_relaunch_children_counts(config_rows: list[tuple[UUID, Optional[dict]]]) -> dict[UUID, int]:
    counts: dict[UUID, int] = {}
    for _job_id, cfg in config_rows:
        parent_id = _extract_relaunch_parent_job_id(cfg if isinstance(cfg, dict) else None)
        if not parent_id:
            continue
        counts[parent_id] = int(counts.get(parent_id, 0) or 0) + 1
    return counts


def _json_relaunch_parent_expr(model=AgentJob):
    try:
        return model.config["relaunch_from_job_id"].as_string()
    except Exception:
        return model.config["relaunch_from_job_id"].astext


def _json_launch_mode_expr(model=AgentJob):
    try:
        return model.config["launch_mode"].as_string()
    except Exception:
        return model.config["launch_mode"].astext


async def _build_relaunch_children_counts_for_user(
    db: AsyncSession,
    *,
    user_id: UUID,
) -> dict[UUID, int]:
    parent_expr = _json_relaunch_parent_expr().label("parent_id")
    rows = await db.execute(
        select(parent_expr, func.count().label("children_count"))
        .where(
            and_(
                AgentJob.user_id == user_id,
                parent_expr.is_not(None),
                parent_expr != "",
            )
        )
        .group_by(parent_expr)
    )
    counts: dict[UUID, int] = {}
    for parent_raw, child_count in rows.all():
        text = str(parent_raw or "").strip()
        if not text:
            continue
        try:
            parent_id = UUID(text)
        except Exception:
            continue
        counts[parent_id] = int(child_count or 0)
    return counts


def _to_relaunch_lineage_node(job: AgentJob) -> AgentJobRelaunchLineageNode:
    cfg = job.config if isinstance(job.config, dict) else {}
    return AgentJobRelaunchLineageNode(
        id=job.id,
        name=job.name,
        status=job.status,
        created_at=job.created_at,
        launch_mode=_extract_launch_mode(cfg) or None,
    )


def _build_relaunch_lineage(
    job: AgentJob,
    jobs_by_id: dict[UUID, AgentJob],
    *,
    max_ancestors: int = 100,
    max_descendants: int = 500,
) -> AgentJobRelaunchLineageResponse:
    max_ancestors = max(1, min(int(max_ancestors or 0), 300))
    max_descendants = max(1, min(int(max_descendants or 0), 2000))

    parent_by_child: dict[UUID, UUID] = {}
    children_by_parent: dict[UUID, list[AgentJob]] = {}

    for item in jobs_by_id.values():
        cfg = item.config if isinstance(item.config, dict) else {}
        parent_id = _extract_relaunch_parent_job_id(cfg)
        if not parent_id or parent_id not in jobs_by_id:
            continue
        parent_by_child[item.id] = parent_id
        children_by_parent.setdefault(parent_id, []).append(item)

    ancestors: list[AgentJobRelaunchLineageNode] = []
    seen: set[UUID] = {job.id}
    cur_id = parent_by_child.get(job.id)
    while cur_id and cur_id in jobs_by_id and cur_id not in seen and len(ancestors) < max_ancestors:
        cur = jobs_by_id[cur_id]
        ancestors.append(_to_relaunch_lineage_node(cur))
        seen.add(cur_id)
        cur_id = parent_by_child.get(cur_id)
    ancestors_truncated = bool(cur_id and cur_id in jobs_by_id and cur_id not in seen)

    descendants: list[AgentJobRelaunchLineageNode] = []
    queue: list[UUID] = [job.id]
    seen_desc: set[UUID] = {job.id}
    while queue and len(descendants) < max_descendants:
        parent_id = queue.pop(0)
        children = children_by_parent.get(parent_id, [])
        children_sorted = sorted(
            children,
            key=lambda x: x.created_at or datetime.min,
        )
        for child in children_sorted:
            if child.id in seen_desc:
                continue
            seen_desc.add(child.id)
            descendants.append(_to_relaunch_lineage_node(child))
            queue.append(child.id)
    descendants_truncated = bool(queue)

    root = ancestors[-1].id if ancestors else job.id
    latest_child_job_id = None
    if descendants:
        latest_node = max(
            descendants,
            key=lambda n: n.created_at or datetime.min,
        )
        latest_child_job_id = latest_node.id
    parent_job_id = ancestors[0].id if ancestors else None

    return AgentJobRelaunchLineageResponse(
        job_id=job.id,
        root_job_id=root,
        parent_job_id=parent_job_id,
        latest_child_job_id=latest_child_job_id,
        ancestors_truncated=ancestors_truncated,
        descendants_truncated=descendants_truncated,
        ancestors=ancestors,
        descendants=descendants,
    )


def _build_quick_start_relaunch_request(job: AgentJob) -> Optional[AgentJobQuickStartClaudeBackendRequest]:
    cfg = _normalize_scope_config(job.config if isinstance(job.config, dict) else {}) or {}
    if _extract_launch_mode(cfg) != "quick_start_claude_backend":
        return None

    source_id = _extract_source_id_from_config(cfg)
    goal = str(getattr(job, "goal", "") or "").strip()
    if not source_id or not goal:
        return None

    search_query = cfg.get("search_query")
    commands = cfg.get("commands") if isinstance(cfg.get("commands"), list) else None
    file_paths = cfg.get("file_paths") if isinstance(cfg.get("file_paths"), list) else None

    reserved = {
        "source_id",
        "target_source_id",
        "launch_mode",
        "quick_start",
        "search_query",
        "commands",
        "file_paths",
    }
    overrides = {k: v for k, v in cfg.items() if k not in reserved}
    overrides["relaunch_from_job_id"] = str(getattr(job, "id", "") or "").strip()

    try:
        return AgentJobQuickStartClaudeBackendRequest(
            name=str(getattr(job, "name", "") or "").strip() or None,
            goal=goal,
            source_id=source_id,
            search_query=(str(search_query).strip() if search_query is not None else None),
            file_paths=file_paths,
            commands=commands,
            start_immediately=True,
            config_overrides=overrides or None,
        )
    except Exception:
        return None


def _build_quick_start_domain_research_relaunch_request(
    job: AgentJob,
) -> Optional[AgentJobQuickStartDomainResearchRequest]:
    cfg = _normalize_scope_config(job.config if isinstance(job.config, dict) else {}) or {}
    if _extract_launch_mode(cfg) != "quick_start_domain_research":
        return None

    domain = str(cfg.get("domain") or "").strip()
    objective = str(cfg.get("objective") or "").strip()
    if not domain or not objective:
        return None

    reserved = {
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
    }
    overrides = {k: v for k, v in cfg.items() if k not in reserved}
    overrides["relaunch_from_job_id"] = str(getattr(job, "id", "") or "").strip()

    try:
        return AgentJobQuickStartDomainResearchRequest(
            name=str(getattr(job, "name", "") or "").strip() or None,
            domain=domain,
            objective=objective,
            customer_context=str(cfg.get("customer_context") or "").strip() or None,
            source_scope=str(cfg.get("source_scope") or "kb_plus_arxiv").strip().lower() or "kb_plus_arxiv",
            track_type=str(cfg.get("track_type") or "generic").strip().lower() or "generic",
            research_mode=str(cfg.get("research_mode") or "literature_to_hypothesis").strip().lower() or "literature_to_hypothesis",
            monitor_queries=cfg.get("monitor_queries") if isinstance(cfg.get("monitor_queries"), list) else None,
            repo_source_ids=cfg.get("repo_source_ids") if isinstance(cfg.get("repo_source_ids"), list) else None,
            benchmark_queries=cfg.get("benchmark_queries") if isinstance(cfg.get("benchmark_queries"), list) else None,
            sandbox_profile_id=str(cfg.get("sandbox_profile_id") or "").strip() or None,
            report_format=str(cfg.get("report_format") or "brief_and_report").strip().lower() or "brief_and_report",
            scoring_policy=cfg.get("scoring_policy") if isinstance(cfg.get("scoring_policy"), dict) else None,
            selection_policy=cfg.get("selection_policy") if isinstance(cfg.get("selection_policy"), dict) else None,
            persist_artifacts=_coerce_bool(cfg.get("persist_artifacts"), default=True),
            auto_launch_follow_up=_coerce_bool(cfg.get("auto_launch_follow_up"), default=True),
            auto_create_experiment_plans=_coerce_bool(cfg.get("auto_create_experiment_plans"), default=True),
            automation_profile=str(cfg.get("automation_profile") or "balanced").strip().lower() or "balanced",
            automation_policy=cfg.get("automation_policy") if isinstance(cfg.get("automation_policy"), dict) else None,
            validation_policy=cfg.get("validation_policy") if isinstance(cfg.get("validation_policy"), dict) else None,
            max_documents=int(cfg.get("max_documents") or 10),
            max_papers=int(cfg.get("max_papers") or 8),
            profile_id=cfg.get("profile_id"),
            confidence_threshold=float(cfg.get("confidence_threshold") or 0.7),
            start_immediately=True,
            config_overrides=overrides or None,
        )
    except Exception:
        return None


def _build_quick_start_repo_bug_triage_relaunch_request(
    job: AgentJob,
    *,
    retry_strategy: str = "clean_relaunch",
) -> Optional[AgentJobQuickStartRepoBugTriageRequest]:
    cfg = _normalize_scope_config(job.config if isinstance(job.config, dict) else {}) or {}
    if _extract_launch_mode(cfg) != "quick_start_repo_bug_triage":
        return None

    source_id = _extract_source_id_from_config(cfg)
    goal = str(getattr(job, "goal", "") or "").strip() or None
    failure_symptom = str(cfg.get("failure_symptom") or "").strip() or None
    if not source_id or (not goal and not failure_symptom):
        return None

    reserved = {
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
    }
    overrides = {k: v for k, v in cfg.items() if k not in reserved}
    overrides["relaunch_from_job_id"] = str(getattr(job, "id", "") or "").strip()
    request_error_output = (str(cfg.get("error_output")).strip() if cfg.get("error_output") is not None else None)
    coding_recovery = _extract_repo_bug_triage_coding_recovery(job)
    if retry_strategy and retry_strategy != "clean_relaunch":
        overrides["coding_recovery"] = {
            "strategy": retry_strategy,
            "retry_reason": str(coding_recovery.get("retry_reason") or "").strip() or None,
            "resume_hint": str(coding_recovery.get("resume_hint") or "").strip() or None,
            "last_failed_commands": [
                str(cmd).strip()
                for cmd in (coding_recovery.get("last_failed_commands") or [])
                if str(cmd).strip()
            ][:6],
            "suggested_operator_actions": [
                str(action).strip()
                for action in (coding_recovery.get("suggested_operator_actions") or [])
                if str(action).strip()
            ][:6],
        }
        latest_failed_output = str(coding_recovery.get("latest_failed_output") or "").strip()
        if latest_failed_output:
            overrides["error_output"] = latest_failed_output[:4000]
            request_error_output = latest_failed_output[:4000]

    try:
        return AgentJobQuickStartRepoBugTriageRequest(
            name=str(getattr(job, "name", "") or "").strip() or None,
            goal=goal,
            failure_symptom=failure_symptom,
            source_id=source_id,
            scope=str(cfg.get("scope") or "auto").strip().lower() or "auto",
            search_query=(str(cfg.get("search_query")).strip() if cfg.get("search_query") is not None else None),
            file_paths=cfg.get("file_paths") if isinstance(cfg.get("file_paths"), list) else None,
            commands=cfg.get("commands") if isinstance(cfg.get("commands"), list) else None,
            error_output=request_error_output,
            start_immediately=True,
            config_overrides=overrides or None,
        )
    except Exception:
        return None


def _build_quick_start_bug_triage_swarm_relaunch_request(
    job: AgentJob,
) -> Optional[AgentJobQuickStartBugTriageSwarmRequest]:
    relaunch = _build_quick_start_coding_swarm_relaunch_request(
        job,
        launch_mode="quick_start_bug_triage_swarm",
        request_cls=AgentJobQuickStartBugTriageSwarmRequest,
    )
    return relaunch if isinstance(relaunch, AgentJobQuickStartBugTriageSwarmRequest) else None


def _build_quick_start_build_break_swarm_relaunch_request(
    job: AgentJob,
) -> Optional[AgentJobQuickStartBuildBreakSwarmRequest]:
    relaunch = _build_quick_start_coding_swarm_relaunch_request(
        job,
        launch_mode="quick_start_build_break_swarm",
        request_cls=AgentJobQuickStartBuildBreakSwarmRequest,
    )
    return relaunch if isinstance(relaunch, AgentJobQuickStartBuildBreakSwarmRequest) else None


def _build_quick_start_frontend_regression_swarm_relaunch_request(
    job: AgentJob,
) -> Optional[AgentJobQuickStartFrontendRegressionSwarmRequest]:
    relaunch = _build_quick_start_coding_swarm_relaunch_request(
        job,
        launch_mode="quick_start_frontend_regression_swarm",
        request_cls=AgentJobQuickStartFrontendRegressionSwarmRequest,
    )
    return relaunch if isinstance(relaunch, AgentJobQuickStartFrontendRegressionSwarmRequest) else None


def _build_quick_start_coding_swarm_relaunch_request(
    job: AgentJob,
    *,
    launch_mode: str,
    request_cls: type[AgentJobQuickStartBugTriageSwarmRequest | AgentJobQuickStartBuildBreakSwarmRequest | AgentJobQuickStartFrontendRegressionSwarmRequest],
) -> Optional[AgentJobQuickStartBugTriageSwarmRequest | AgentJobQuickStartBuildBreakSwarmRequest | AgentJobQuickStartFrontendRegressionSwarmRequest]:
    cfg = _normalize_scope_config(job.config if isinstance(job.config, dict) else {}) or {}
    if _extract_launch_mode(cfg) != launch_mode:
        return None

    source_id = _extract_source_id_from_config(cfg)
    goal = str(getattr(job, "goal", "") or "").strip() or None
    failure_symptom = str(cfg.get("failure_symptom") or "").strip() or None
    if not source_id or (not goal and not failure_symptom):
        return None

    reserved = {
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
    }
    overrides = {k: v for k, v in cfg.items() if k not in reserved}
    overrides["relaunch_from_job_id"] = str(getattr(job, "id", "") or "").strip()

    quick_start = cfg.get("quick_start") if isinstance(cfg.get("quick_start"), dict) else {}
    try:
        return request_cls(
            name=str(getattr(job, "name", "") or "").strip() or None,
            goal=goal,
            failure_symptom=failure_symptom,
            source_id=source_id,
            scope=str(cfg.get("scope") or quick_start.get("scope") or "auto").strip().lower() or "auto",
            search_query=(str(cfg.get("search_query")).strip() if cfg.get("search_query") is not None else None),
            file_paths=cfg.get("file_paths") if isinstance(cfg.get("file_paths"), list) else None,
            commands=cfg.get("commands") if isinstance(cfg.get("commands"), list) else None,
            error_output=(str(cfg.get("error_output")).strip() if cfg.get("error_output") is not None else None),
            max_agents=int(quick_start.get("max_agents") or cfg.get("swarm_max_agents") or 4),
            profile_id=quick_start.get("profile_id"),
            start_immediately=True,
            config_overrides=overrides or None,
        )
    except Exception:
        return None


def _extract_repo_bug_triage_coding_recovery(job: AgentJob) -> dict[str, Any]:
    results = getattr(job, "results", None)
    results = results if isinstance(results, dict) else {}
    code_exec = results.get("code_patch_execution") if isinstance(results.get("code_patch_execution"), dict) else {}
    recovery = code_exec.get("recovery") if isinstance(code_exec.get("recovery"), dict) else {}
    experiment_run = results.get("experiment_run") if isinstance(results.get("experiment_run"), dict) else {}
    execution_strategy = results.get("execution_strategy") if isinstance(results.get("execution_strategy"), dict) else {}
    execution_graph = execution_strategy.get("execution_graph") if isinstance(execution_strategy.get("execution_graph"), dict) else {}
    graph_health = execution_graph.get("graph_health") if isinstance(execution_graph.get("graph_health"), dict) else {}
    graph_reasons = graph_health.get("reasons") if isinstance(graph_health.get("reasons"), list) else []
    failed_commands = [
        str(cmd).strip()
        for cmd in (experiment_run.get("failed_commands") if isinstance(experiment_run.get("failed_commands"), list) else [])
        if str(cmd).strip()
    ]
    latest_failed_output = ""
    runs = experiment_run.get("runs") if isinstance(experiment_run.get("runs"), list) else []
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
    if not can_resume and str(getattr(job, "status", "") or "").lower() == AgentJobStatus.PAUSED.value:
        final_phase = str(experiment_run.get("final_phase") or "").strip().lower()
        can_resume = final_phase in {"primary", "retry_primary", "fallback"} or bool(failed_commands)
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
        for action in (recovery.get("suggested_operator_actions") if isinstance(recovery.get("suggested_operator_actions"), list) else [])
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
        "resume_hint": str(recovery.get("resume_hint") or "").strip() or (
            "Resume verification from the paused job state." if can_resume else ""
        ),
        "suggested_operator_actions": suggested_actions,
        "can_retry_with_refined_plan": bool(recovery.get("can_retry_with_refined_plan", bool(failed_commands or state in {"verification_failed", "plan_stalled"}))),
        "can_resume_verification": can_resume,
        "latest_failed_output": latest_failed_output[:4000] if latest_failed_output else "",
    }


def _build_quick_start_role_workflow_relaunch_request(
    job: AgentJob,
) -> Optional[AgentJobQuickStartRoleWorkflowRequest]:
    cfg = _normalize_scope_config(job.config if isinstance(job.config, dict) else {}) or {}
    if _extract_launch_mode(cfg) != "quick_start_role_workflow":
        return None

    goal = str(getattr(job, "goal", "") or "").strip()
    if not goal:
        return None

    roles = _normalize_swarm_roles(cfg.get("swarm_roles"), max_roles=12)
    if not roles:
        quick_start = cfg.get("quick_start") if isinstance(cfg.get("quick_start"), dict) else {}
        roles = _normalize_swarm_roles(quick_start.get("roles"), max_roles=12)

    try:
        max_agents = int(cfg.get("swarm_max_agents", 0) or 0)
    except Exception:
        max_agents = 0
    if max_agents <= 0:
        max_agents = len(roles) if roles else 4
    max_agents = max(1, min(max_agents, 12))

    quick_start = cfg.get("quick_start") if isinstance(cfg.get("quick_start"), dict) else {}
    memory = cfg.get("memory") if isinstance(cfg.get("memory"), dict) else {}
    memory_profile = str(
        quick_start.get("memory_profile")
        or memory.get("profile")
        or "balanced"
    ).strip().lower()
    if memory_profile not in {"off", "minimal", "balanced", "evidence", "synthesis"}:
        memory_profile = "balanced"

    approval = cfg.get("approval_checkpoints") if isinstance(cfg.get("approval_checkpoints"), dict) else {}
    approval_mode = str(quick_start.get("approval_mode") or "").strip().lower()
    if approval_mode not in {"high_impact", "none"}:
        approval_mode = "high_impact" if _coerce_bool(approval.get("enabled"), default=False) else "none"
    execution_mode = str(quick_start.get("execution_mode") or cfg.get("execution_mode") or "").strip().lower()
    execution_mode = execution_mode.replace("-", "_").replace(" ", "_")
    if execution_mode in {"plan_then_act", "plan_execute", "planner_executor"}:
        execution_mode = "plan_and_execute"
    if execution_mode not in {"plan_and_execute", "adaptive"}:
        execution_mode = "plan_and_execute"
    extract_on_statuses = memory.get("extract_on_statuses") if isinstance(memory.get("extract_on_statuses"), list) else []
    extract_on_statuses = [str(x).strip().lower() for x in extract_on_statuses if str(x).strip()]
    extract_memory_on_failure = (
        "failed" in set(extract_on_statuses)
        if extract_on_statuses
        else _coerce_bool(memory.get("extract_on_failure"), default=True)
    )
    failed_types = memory.get("failed_extraction_types") if isinstance(memory.get("failed_extraction_types"), list) else None
    completed_types = memory.get("completed_extraction_types") if isinstance(memory.get("completed_extraction_types"), list) else None

    reserved = {
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
    }
    overrides = {k: v for k, v in cfg.items() if k not in reserved}
    overrides["relaunch_from_job_id"] = str(getattr(job, "id", "") or "").strip()

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
            config_overrides=overrides or None,
        )
    except Exception:
        return None


def _is_none_launch_mode(mode: str) -> bool:
    value = str(mode or "").strip().lower()
    return value in {"", "__none__", "none", "manual"}


def _matches_launch_mode_filter(config: Optional[dict], launch_mode_filter: str) -> bool:
    needle = str(launch_mode_filter or "").strip().lower()
    if not needle:
        return True
    mode = _extract_launch_mode(config)
    if needle in {"__none__", "none", "manual"}:
        return _is_none_launch_mode(mode)
    return mode == needle


def _build_launch_mode_counts(configs: list[Optional[dict]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cfg in configs:
        mode = _extract_launch_mode(cfg if isinstance(cfg, dict) else None)
        if _is_none_launch_mode(mode):
            continue
        counts[mode] = int(counts.get(mode, 0) or 0) + 1
    return counts


def _build_launch_mode_stats(configs: list[Optional[dict]]) -> tuple[dict[str, int], int]:
    counts = _build_launch_mode_counts(configs)
    none_count = 0
    for cfg in configs:
        mode = _extract_launch_mode(cfg if isinstance(cfg, dict) else None)
        if _is_none_launch_mode(mode):
            none_count += 1
    return counts, none_count


def _append_launch_log_if_present(job: AgentJob) -> bool:
    cfg = job.config if isinstance(job.config, dict) else {}
    launch_mode = _extract_launch_mode(cfg)
    if not launch_mode:
        return False
    quick_start = cfg.get("quick_start") if isinstance(cfg.get("quick_start"), dict) else {}
    coding_recovery = cfg.get("coding_recovery") if isinstance(cfg.get("coding_recovery"), dict) else {}
    relaunch_from_job_id = str(cfg.get("relaunch_from_job_id") or "").strip() or None
    source_id = _extract_source_id_from_config(cfg) or None
    search_query = str(cfg.get("search_query") or "").strip() or None
    commands = cfg.get("commands") if isinstance(cfg.get("commands"), list) else []
    file_paths = cfg.get("file_paths") if isinstance(cfg.get("file_paths"), list) else []
    job.add_log_entry(
        {
            "phase": "launch",
            "action": "job_launch",
            "result": {
                "launch_mode": launch_mode,
                "quick_start_profile": str(quick_start.get("profile") or "").strip() or None,
                "quick_start_version": str(quick_start.get("version") or "").strip() or None,
                "source_name": str(quick_start.get("source_name") or "").strip() or None,
                "source_type": str(quick_start.get("source_type") or "").strip() or None,
                "source_id": source_id,
                "search_query": search_query,
                "commands_count": len(commands),
                "file_paths_count": len(file_paths),
                "relaunch_from_job_id": relaunch_from_job_id,
                "coding_recovery_strategy": str(coding_recovery.get("strategy") or "").strip() or None,
            },
        }
    )
    return True


def _find_unsafe_commands(commands: Optional[list[str]]) -> list[str]:
    if not isinstance(commands, list):
        return []
    blocked_patterns = [
        r"\brm\s+-rf\b",
        r"\bsudo\b",
        r"\bmkfs\b",
        r"\bdd\s+if=",
        r"\bshutdown\b",
        r"\breboot\b",
        r"\bhalt\b",
        r"\bpoweroff\b",
        r"\bchown\b",
        r"\bchmod\s+777\b",
    ]
    compiled = [re.compile(pat, re.IGNORECASE) for pat in blocked_patterns]
    blocked: list[str] = []
    for raw in commands:
        cmd = str(raw or "").strip()
        if not cmd:
            continue
        if any(rx.search(cmd) for rx in compiled):
            blocked.append(cmd)
    return blocked[:6]


def _score_template_recommendation(
    template: AgentJobTemplateResponse,
    *,
    category: Optional[str],
    recommend_goal: Optional[str],
    recommend_scope: Optional[str],
) -> tuple[int, list[str]]:
    """Lightweight ranking so relevant templates appear first."""
    score = 0
    reasons: list[str] = []

    name = str(template.name or "").strip().lower()
    display_name = str(template.display_name or "").strip().lower()
    tpl_category = str(template.category or "").strip().lower()
    cfg = template.default_config if isinstance(template.default_config, dict) else {}
    runner = str(cfg.get("deterministic_runner") or "").strip().lower()

    if category and tpl_category and tpl_category == str(category).strip().lower():
        score += 10
        reasons.append("matches_category")

    goal_text = str(recommend_goal or "").strip().lower()
    scope_text = str(recommend_scope or "").strip().lower()
    context = f"{goal_text} {scope_text}".strip()

    backend_signals = ["backend", "api", "server", "fastapi", "flask", "django", "pytest"]
    code_signals = ["code", "patch", "refactor", "test", "bug", "fix", "implementation"]
    latex_signals = ["latex", "paper", "citation", "bibtex"]

    backend_context = any(sig in context for sig in backend_signals)
    code_context = any(sig in context for sig in code_signals)
    latex_context = any(sig in context for sig in latex_signals)

    if backend_context and name == "claude_code_backend":
        score += 80
        reasons.append("backend_loop_specialized")
    if code_context and tpl_category == "code":
        score += 20
        reasons.append("code_category_fit")
    if backend_context and runner in {"code_patch_proposer", "experiment_runner"}:
        score += 15
        reasons.append("backend_code_runner_fit")
    if backend_context and ("backend" in display_name or "backend" in str(template.default_goal or "").lower()):
        score += 10
        reasons.append("backend_goal_fit")
    if latex_context and tpl_category == "latex":
        score += 30
        reasons.append("latex_category_fit")

    if not category and not context and template.is_system:
        # Keep built-in system templates slightly above custom by default.
        score += 2
        reasons.append("system_default")

    return score, reasons[:4]


def _extract_swarm_summary(
    job: AgentJob,
    *,
    current_user_id: Optional[str] = None,
    user_lookup: Optional[dict[str, User]] = None,
) -> Optional[dict]:
    """Build a compact swarm/fan-in summary for API consumers."""
    results = job.results if isinstance(job.results, dict) else {}
    execution_strategy = results.get("execution_strategy") if isinstance(results.get("execution_strategy"), dict) else {}
    swarm_exec = execution_strategy.get("swarm") if isinstance(execution_strategy.get("swarm"), dict) else {}
    fan_in = results.get("swarm_fan_in") if isinstance(results.get("swarm_fan_in"), dict) else {}

    cfg = job.config if isinstance(job.config, dict) else {}
    enabled = bool(cfg.get("swarm_child_jobs_enabled", False) or swarm_exec.get("enabled", False))
    configured = bool(swarm_exec.get("configured", False) or swarm_exec)
    fan_in_enabled = bool(swarm_exec.get("fan_in_enabled", False))

    expected_siblings = int(fan_in.get("expected_siblings", 0) or 0)
    received_siblings = int(fan_in.get("received_siblings", 0) or 0)
    terminal_siblings = int(fan_in.get("terminal_siblings", 0) or 0)
    if expected_siblings <= 0:
        expected_siblings = int(swarm_exec.get("child_jobs_count", 0) or 0)
    if received_siblings <= 0 and expected_siblings > 0:
        received_siblings = expected_siblings
    if terminal_siblings <= 0 and received_siblings > 0:
        terminal_siblings = received_siblings

    roles = []
    raw_roles = fan_in.get("roles")
    if isinstance(raw_roles, list) and raw_roles:
        roles = [str(r).strip() for r in raw_roles if str(r).strip()][:20]
    elif isinstance(swarm_exec.get("roles_assigned"), list):
        roles = [str(r).strip() for r in swarm_exec.get("roles_assigned", []) if str(r).strip()][:20]

    confidence = fan_in.get("confidence") if isinstance(fan_in.get("confidence"), dict) else {}
    consensus_rows = fan_in.get("consensus_findings") if isinstance(fan_in.get("consensus_findings"), list) else []
    consensus_findings = [
        str(row.get("finding") or "").strip()[:280]
        for row in consensus_rows
        if isinstance(row, dict) and str(row.get("finding") or "").strip()
    ][:10]
    conflicts = fan_in.get("conflicts") if isinstance(fan_in.get("conflicts"), list) else []
    action_plan = fan_in.get("action_plan") if isinstance(fan_in.get("action_plan"), list) else []
    collaboration = _extract_swarm_collaboration(job)

    if not any([enabled, configured, fan_in, swarm_exec]):
        return None

    return {
        "enabled": enabled,
        "configured": configured,
        "fan_in_enabled": fan_in_enabled,
        "fan_in_group_id": str(fan_in.get("fan_in_group_id") or swarm_exec.get("fan_in_group_id") or "").strip(),
        "roles": roles,
        "role_count": len(roles),
        "expected_siblings": expected_siblings,
        "received_siblings": received_siblings,
        "terminal_siblings": terminal_siblings,
        "consensus_count": len(consensus_rows),
        "consensus_findings": consensus_findings,
        "conflict_count": len(conflicts),
        "conflicts": conflicts[:10],
        "action_plan": action_plan[:10],
        "confidence": confidence,
        "winning_slice_id": str(fan_in.get("winning_slice_id") or "").strip() or None,
        "winning_role": str(fan_in.get("winning_role") or "").strip() or None,
        "promotion_reason": str(fan_in.get("promotion_reason") or "").strip() or None,
        "review_state": str(fan_in.get("review_state") or "").strip() or None,
        "review_reason": str(fan_in.get("review_reason") or "").strip() or None,
        "review_required": bool(fan_in.get("review_required", False)),
        "tie_breaker_attempted": bool(fan_in.get("tie_breaker_attempted", False)),
        "tie_breaker_job_id": str(fan_in.get("tie_breaker_job_id") or "").strip() or None,
        "tie_breaker_source_job_id": str(fan_in.get("tie_breaker_source_job_id") or "").strip() or None,
        "file_converged": bool(fan_in.get("file_converged", False)),
        "file_convergence_support": int(fan_in.get("file_convergence_support", 0) or 0),
        "top_file_cluster": fan_in.get("top_file_cluster") if isinstance(fan_in.get("top_file_cluster"), dict) else None,
        "command_converged": bool(fan_in.get("command_converged", False)),
        "command_convergence_support": int(fan_in.get("command_convergence_support", 0) or 0),
        "top_command_cluster": fan_in.get("top_command_cluster") if isinstance(fan_in.get("top_command_cluster"), dict) else None,
        "repair_chain_job_id": str(fan_in.get("repair_chain_job_id") or "").strip() or None,
        "candidate_paths": fan_in.get("candidate_paths")[:10] if isinstance(fan_in.get("candidate_paths"), list) else [],
        "recommended_commands": fan_in.get("recommended_commands")[:10] if isinstance(fan_in.get("recommended_commands"), list) else [],
        "owner_user_id": str(collaboration.get("owner_user_id") or job.user_id),
        "shared_review": bool(collaboration.get("shared_review")),
        "shared_with_user_ids": collaboration.get("shared_with_user_ids") or [],
        "assigned_user_id": collaboration.get("assigned_user_id"),
        "assigned_at": collaboration.get("assigned_at"),
        "assigned_by_user_id": collaboration.get("assigned_by_user_id"),
        "review_note": collaboration.get("review_note"),
        "collaboration_summary": build_collaboration_summary(
            owner_user_id=str(collaboration.get("owner_user_id") or job.user_id),
            visibility="shared" if collaboration.get("shared_with_user_ids") else "private",
            shared_with_user_ids=list(collaboration.get("shared_with_user_ids") or []),
            assigned_user_id=str(collaboration.get("assigned_user_id") or "").strip() or None,
            assigned_by_user_id=str(collaboration.get("assigned_by_user_id") or "").strip() or None,
            assigned_at=str(collaboration.get("assigned_at") or "").strip() or None,
            note=str(collaboration.get("review_note") or "").strip() or None,
            current_user_id=current_user_id,
            user_lookup=user_lookup,
        ),
    }


_CODING_SWARM_ANALYTICS_PRESETS: dict[str, dict[str, str]] = {
    "bug_triage_swarm": {
        "launch_mode": "quick_start_bug_triage_swarm",
        "label": "Bug Triage Swarm",
    },
    "build_break_swarm": {
        "launch_mode": "quick_start_build_break_swarm",
        "label": "Build Break Swarm",
    },
    "frontend_regression_swarm": {
        "launch_mode": "quick_start_frontend_regression_swarm",
        "label": "Frontend Regression Swarm",
    },
}


def _infer_coding_swarm_preset_key(job: AgentJob) -> str:
    cfg = job.config if isinstance(job.config, dict) else {}
    quick_start = cfg.get("quick_start") if isinstance(cfg.get("quick_start"), dict) else {}
    preset_key = str(quick_start.get("preset_key") or cfg.get("coding_swarm_preset_key") or "").strip().lower()
    if preset_key in _CODING_SWARM_ANALYTICS_PRESETS:
        return preset_key
    launch_mode = _extract_launch_mode(cfg)
    for key, meta in _CODING_SWARM_ANALYTICS_PRESETS.items():
        if str(meta.get("launch_mode") or "") == launch_mode:
            return key
    return ""


def _swarm_confidence_bucket(overall: float) -> str:
    if overall >= 0.70:
        return "high"
    if overall >= 0.50:
        return "medium"
    return "low"


def _iso_or_none(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if isinstance(value, datetime) else None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _normalize_datetime(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            return None
    return None


def _datetime_sort_key(value: Optional[datetime]) -> float:
    if not isinstance(value, datetime):
        return 0.0
    try:
        return value.timestamp()
    except Exception:
        return 0.0


def _extract_code_patch_execution(job: AgentJob) -> dict[str, Any]:
    results = job.results if isinstance(job.results, dict) else {}
    return results.get("code_patch_execution") if isinstance(results.get("code_patch_execution"), dict) else {}


def _derive_repair_verification_status(job: AgentJob) -> tuple[Optional[str], Optional[str]]:
    code_exec = _extract_code_patch_execution(job)
    recovery = code_exec.get("recovery") if isinstance(code_exec.get("recovery"), dict) else {}
    recovery_state = str(recovery.get("recovery_state") or "").strip().lower()
    if recovery_state in {"verified", "verification_succeeded", "verified_fix"}:
        return "succeeded", str(recovery.get("retry_reason") or "Verification succeeded.").strip()
    if recovery_state in {"verification_failed", "failed", "verification_error"}:
        return "failed", str(recovery.get("retry_reason") or "Verification failed.").strip()

    execution_log = job.execution_log if isinstance(job.execution_log, list) else []
    verify_events = [
        entry for entry in execution_log
        if isinstance(entry, dict) and entry.get("verify_success") is not None
    ]
    if verify_events:
        latest = verify_events[-1]
        if bool(latest.get("verify_success")):
            return "succeeded", "Verification succeeded."
        return "failed", "Verification failed."

    results = job.results if isinstance(job.results, dict) else {}
    experiment_run = results.get("experiment_run") if isinstance(results.get("experiment_run"), dict) else {}
    runs = experiment_run.get("runs") if isinstance(experiment_run.get("runs"), list) else []
    if runs:
        normalized_runs = [row for row in runs if isinstance(row, dict)]
        if normalized_runs and all(bool(row.get("ok")) for row in normalized_runs):
            return "succeeded", "Experiment verification runs succeeded."
        if any(not bool(row.get("ok")) for row in normalized_runs):
            return "failed", "Experiment verification runs failed."

    if code_exec:
        if str(job.status or "").strip().lower() in {
            AgentJobStatus.PENDING.value,
            AgentJobStatus.RUNNING.value,
            AgentJobStatus.PAUSED.value,
        }:
            return "pending", "Verification is still in progress."
        return "incomplete", "Repair completed without explicit verification evidence."
    return None, None


def _derive_swarm_terminal_outcome(
    *,
    review_state: str,
    repair_job: Optional[AgentJob],
    verification_status: Optional[str],
    backlog_item: Optional[CodingBacklogItem],
) -> tuple[str, str]:
    if verification_status == "succeeded":
        return "verified_fix", "Repair verification succeeded."
    if backlog_item is not None and repair_job is None:
        return "backlog_routed", "Swarm findings were routed to backlog."
    if repair_job is not None:
        repair_status = str(repair_job.status or "").strip().lower()
        if verification_status == "failed" or repair_status in {AgentJobStatus.FAILED.value, AgentJobStatus.CANCELLED.value}:
            return "repair_failed", "Repair chain failed or verification failed."
        return "stalled_after_handoff", "Repair handoff exists without a verified fix."
    if review_state in {"needs_review", "insufficient_swarm_consensus", "consensus_failed", "tie_break_running", "manual_promotion"}:
        return "needs_review", "Swarm outcome still requires operator review."
    return "needs_review", "No downstream action recorded."


def _derive_swarm_outcome_case(
    swarm_job: AgentJob,
    *,
    repair_jobs_by_id: dict[str, AgentJob],
    backlog_by_swarm_job_id: dict[str, list[CodingBacklogItem]],
    current_user_id: Optional[str] = None,
    user_lookup: Optional[dict[str, User]] = None,
) -> AgentJobSwarmOutcomeCaseResponse:
    cfg = swarm_job.config if isinstance(swarm_job.config, dict) else {}
    quick_start = cfg.get("quick_start") if isinstance(cfg.get("quick_start"), dict) else {}
    swarm_summary = _extract_swarm_summary(swarm_job) or {}
    collaboration = _extract_swarm_collaboration(swarm_job)
    preset_key = _infer_coding_swarm_preset_key(swarm_job)
    launch_mode = _extract_launch_mode(cfg)
    source_id = str(cfg.get("source_id") or "").strip() or None
    source_label = str(quick_start.get("source_name") or source_id or "").strip() or None
    review_state = str(swarm_summary.get("review_state") or "").strip().lower()
    review_reason = str(swarm_summary.get("review_reason") or swarm_summary.get("promotion_reason") or "").strip() or None
    confidence_overall = _safe_float(((swarm_summary.get("confidence") or {}) if isinstance(swarm_summary.get("confidence"), dict) else {}).get("overall"))
    repair_job_id = str(swarm_summary.get("repair_chain_job_id") or "").strip()
    repair_job = repair_jobs_by_id.get(repair_job_id) if repair_job_id else None
    verification_status, verification_reason = _derive_repair_verification_status(repair_job) if repair_job is not None else (None, None)
    backlog_items = backlog_by_swarm_job_id.get(str(swarm_job.id), [])
    backlog_item = backlog_items[0] if backlog_items else None

    promotion_mode = "none"
    if review_state == "auto_promoted":
        promotion_mode = "auto"
    elif review_state == "manual_promotion" or (repair_job is not None and str(swarm_summary.get("promotion_reason") or "").strip().lower().startswith("manually promoted")):
        promotion_mode = "manual"

    repair_handoff_at = repair_job.created_at if repair_job is not None else None
    backlog_routed_at = backlog_item.created_at if backlog_item is not None else None
    latest_downstream_at = None
    for candidate in [repair_job.completed_at if repair_job is not None else None, repair_job.last_activity_at if repair_job is not None else None, backlog_item.updated_at if backlog_item is not None else None]:
        if isinstance(candidate, datetime) and (latest_downstream_at is None or candidate > latest_downstream_at):
            latest_downstream_at = candidate

    handoff_latency_minutes = None
    if isinstance(swarm_job.completed_at, datetime) and isinstance(repair_handoff_at, datetime):
        handoff_latency_minutes = max(0.0, round((repair_handoff_at - swarm_job.completed_at).total_seconds() / 60.0, 2))
    elif isinstance(swarm_job.created_at, datetime) and isinstance(repair_handoff_at, datetime):
        handoff_latency_minutes = max(0.0, round((repair_handoff_at - swarm_job.created_at).total_seconds() / 60.0, 2))

    terminal_outcome, terminal_reason = _derive_swarm_terminal_outcome(
        review_state=review_state,
        repair_job=repair_job,
        verification_status=verification_status,
        backlog_item=backlog_item,
    )

    return AgentJobSwarmOutcomeCaseResponse(
        swarm_job_id=str(swarm_job.id),
        swarm_job_name=str(swarm_job.name or "").strip() or None,
        preset_key=preset_key,
        launch_mode=launch_mode,
        source_id=source_id,
        source_label=source_label,
        swarm_status=str(swarm_job.status or "").strip() or None,
        swarm_completed_at=swarm_job.completed_at or swarm_job.last_activity_at or swarm_job.created_at,
        review_state=review_state or None,
        review_reason=review_reason,
        owner_user_id=str(collaboration.get("owner_user_id") or swarm_job.user_id),
        assigned_user_id=str(collaboration.get("assigned_user_id") or "").strip() or None,
        assigned_at=datetime.fromisoformat(str(collaboration.get("assigned_at"))) if str(collaboration.get("assigned_at") or "").strip() else None,
        assigned_by_user_id=str(collaboration.get("assigned_by_user_id") or "").strip() or None,
        review_note=str(collaboration.get("review_note") or "").strip() or None,
        collaboration_summary=CollaborationSummaryResponse.model_validate(
            build_collaboration_summary(
                owner_user_id=str(collaboration.get("owner_user_id") or swarm_job.user_id),
                visibility="shared" if collaboration.get("shared_with_user_ids") else "private",
                shared_with_user_ids=list(collaboration.get("shared_with_user_ids") or []),
                assigned_user_id=str(collaboration.get("assigned_user_id") or "").strip() or None,
                assigned_by_user_id=str(collaboration.get("assigned_by_user_id") or "").strip() or None,
                assigned_at=str(collaboration.get("assigned_at") or "").strip() or None,
                note=str(collaboration.get("review_note") or "").strip() or None,
                current_user_id=current_user_id,
                user_lookup=user_lookup,
            )
        ),
        promotion_mode=promotion_mode,
        confidence_overall=round(confidence_overall, 4) if confidence_overall is not None else None,
        tie_breaker_attempted=bool(swarm_summary.get("tie_breaker_attempted") or swarm_summary.get("tie_breaker_job_id")),
        repair_job_id=str(repair_job.id) if repair_job is not None else None,
        repair_job_name=str(repair_job.name or "").strip() if repair_job is not None and str(repair_job.name or "").strip() else None,
        repair_status=str(repair_job.status or "").strip() if repair_job is not None and str(repair_job.status or "").strip() else None,
        repair_handoff_at=repair_handoff_at,
        verification_status=verification_status,
        verification_reason=verification_reason,
        backlog_item_id=str(backlog_item.id) if backlog_item is not None else None,
        backlog_title=str(backlog_item.title or "").strip() if backlog_item is not None and str(backlog_item.title or "").strip() else None,
        backlog_status=str(backlog_item.status or "").strip() if backlog_item is not None and str(backlog_item.status or "").strip() else None,
        backlog_route_mode=_extract_backlog_route_mode(backlog_item),
        backlog_routed_at=backlog_routed_at,
        latest_downstream_at=latest_downstream_at,
        handoff_latency_minutes=handoff_latency_minutes,
        terminal_outcome=terminal_outcome,
        terminal_reason=terminal_reason,
    )


def _extract_goal_contract_summary(job: AgentJob) -> Optional[dict]:
    """Build compact goal-contract status for quick UI rendering."""
    results = job.results if isinstance(job.results, dict) else {}
    contract = results.get("goal_contract") if isinstance(results.get("goal_contract"), dict) else {}
    if not contract:
        return None

    enabled = bool(contract.get("enabled", False))
    if not enabled and not contract:
        return None
    missing = contract.get("missing") if isinstance(contract.get("missing"), list) else []
    contract_cfg = contract.get("contract") if isinstance(contract.get("contract"), dict) else {}
    metrics = contract.get("metrics") if isinstance(contract.get("metrics"), dict) else {}
    return {
        "enabled": enabled,
        "satisfied": bool(contract.get("satisfied", True)),
        "missing_count": len(missing),
        "missing": [str(x)[:120] for x in missing[:10]],
        "strict_completion": bool(contract_cfg.get("strict_completion", False)),
        "satisfied_iteration": int(contract.get("satisfied_iteration", 0) or 0),
        "metrics": metrics,
    }


# Scheduler-state helpers moved to app.services.agent_job_scheduler_state;
# queue field helpers moved to app.services.agent_job_queue_helpers.
# Aliased here (private names) for backward compatibility with existing
# call sites and importers.
_extract_scheduler_state = extract_scheduler_state
_queue_reason_label = queue_reason_label
_extract_approval_checkpoint = extract_approval_checkpoint
_extract_launch_mode = extract_launch_mode
_parse_optional_datetime = parse_optional_datetime
_queue_age_minutes = queue_age_minutes
_queue_customer_for_job = queue_customer_for_job
_queue_evidence_summary_for_job = queue_evidence_summary_for_job


def _portfolio_summary_payload(portfolio: ResearchPortfolio) -> dict[str, Any]:
    automation_profile = normalize_portfolio_automation_profile(getattr(portfolio, "automation_profile", None), default="balanced")
    effective_policy = resolve_portfolio_automation_policy(automation_profile, portfolio.automation_policy)
    opportunities = list_normalized_research_opportunities(portfolio.opportunities)
    summary = build_autonomy_summary(
        raw_summary=portfolio.latest_summary if isinstance(portfolio.latest_summary, dict) else {},
        opportunities=opportunities,
        automation_profile=automation_profile,
        effective_policy=effective_policy,
        sandbox_profile_id=portfolio.sandbox_profile_id,
        config_revision_key="portfolio_config_revision",
    )
    return {
        "automation_profile": automation_profile,
        "effective_policy": effective_policy,
        "opportunities": opportunities,
        "summary": summary,
    }


def _profile_summary_payload(profile: DomainResearchProfile) -> dict[str, Any]:
    automation_profile, effective_policy = resolve_domain_profile_automation_contract(
        automation_profile=getattr(profile, "automation_profile", None),
        automation_policy=getattr(profile, "automation_policy", None),
        current_snapshot=current_domain_profile_policy_snapshot(profile),
    )
    opportunities = list_normalized_research_opportunities(
        (profile.latest_summary or {}).get("opportunities") if isinstance((profile.latest_summary or {}).get("opportunities"), list) else (profile.latest_summary or {}).get("idea_candidates")
    )
    summary = build_autonomy_summary(
        raw_summary=profile.latest_summary if isinstance(profile.latest_summary, dict) else {},
        opportunities=opportunities,
        automation_profile=automation_profile,
        effective_policy=effective_policy,
        sandbox_profile_id=profile.sandbox_profile_id,
        config_revision_key="profile_config_revision",
    )
    return {
        "automation_profile": automation_profile,
        "effective_policy": effective_policy,
        "opportunities": opportunities,
        "summary": summary,
    }


def _clean_queue_text_list(value: Any, *, limit: int = 8) -> Optional[list[str]]:
    if not isinstance(value, list):
        return None
    out: list[str] = []
    for row in value:
        text = str(row or "").strip()
        if not text or text in out:
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return out or None


def _build_operator_queue_context(
    *,
    objective: Optional[str],
    domain: Optional[str] = None,
    track_type: Optional[str] = None,
    source_scope: Optional[str] = None,
    repo_source_ids: Any = None,
    benchmark_queries: Any = None,
    sandbox_profile_id: Optional[str] = None,
    automation_profile: Optional[str] = None,
    effective_policy: Optional[dict[str, Any]] = None,
    confidence: Any = None,
    readiness: Any = None,
    linked_note_ids: Any = None,
    linked_experiment_plan_ids: Any = None,
    linked_validation_run_ids: Any = None,
    child_job_ids: Any = None,
) -> dict[str, Any]:
    normalized_confidence: Optional[float] = None
    try:
        normalized_confidence = round(float(confidence), 4) if confidence is not None else None
    except Exception:
        normalized_confidence = None
    normalized_readiness: Optional[float] = None
    try:
        normalized_readiness = round(float(readiness), 4) if readiness is not None else None
    except Exception:
        normalized_readiness = None
    return {
        "domain": str(domain or "").strip() or None,
        "objective": str(objective or "").strip() or None,
        "track_type": str(track_type or "").strip() or None,
        "source_scope": str(source_scope or "").strip() or None,
        "repo_source_ids": _clean_queue_text_list(repo_source_ids),
        "benchmark_queries": _clean_queue_text_list(benchmark_queries),
        "sandbox_profile_id": str(sandbox_profile_id or "").strip() or None,
        "automation_profile": str(automation_profile or "").strip() or None,
        "effective_policy": dict(effective_policy) if isinstance(effective_policy, dict) else None,
        "confidence": normalized_confidence,
        "readiness": normalized_readiness,
        "linked_note_ids": _clean_queue_text_list(linked_note_ids),
        "linked_experiment_plan_ids": _clean_queue_text_list(linked_experiment_plan_ids),
        "linked_validation_run_ids": _clean_queue_text_list(linked_validation_run_ids),
        "child_job_ids": _clean_queue_text_list(child_job_ids),
    }


def _follow_up_opportunity_reason_label(
    *,
    opportunity_id: str,
    profile: Optional[DomainResearchProfile] = None,
    portfolio: Optional[ResearchPortfolio] = None,
) -> str:
    rows: list[dict[str, Any]] = []
    if profile is not None:
        rows = _profile_summary_payload(profile)["opportunities"]
    elif portfolio is not None:
        rows = _portfolio_summary_payload(portfolio)["opportunities"]
    row = next(
        (candidate for candidate in rows if str(candidate.get("opportunity_id") or "").strip() == str(opportunity_id or "").strip()),
        None,
    )
    if isinstance(row, dict):
        label_source = str(row.get("canonical_key") or row.get("title") or opportunity_id).strip()
        if label_source:
            normalized = label_source.replace("-", " ").replace("_", " ").strip()
            return normalized[:1].upper() + normalized[1:].lower() if normalized else _queue_reason_label(opportunity_id)
    return _queue_reason_label(opportunity_id)


async def _resolve_portfolio_parent_job_for_queue(
    *,
    db: AsyncSession,
    portfolio: ResearchPortfolio,
) -> AgentJob:
    parent_job_id = portfolio.latest_run_job_id or portfolio.active_job_id
    if not parent_job_id:
        raise HTTPException(status_code=400, detail="Portfolio must run at least once before launching downstream actions")
    parent_job = await db.get(AgentJob, parent_job_id)
    if parent_job is None or parent_job.user_id != portfolio.user_id:
        raise HTTPException(status_code=400, detail="Latest portfolio run is unavailable")
    return parent_job


async def _resolve_profile_parent_job_for_queue(
    *,
    db: AsyncSession,
    profile: DomainResearchProfile,
) -> AgentJob:
    parent_job_id = profile.latest_run_job_id or profile.active_job_id
    if not parent_job_id:
        raise HTTPException(status_code=400, detail="Profile must run at least once before launching downstream actions")
    parent_job = await db.get(AgentJob, parent_job_id)
    if parent_job is None or parent_job.user_id != profile.user_id:
        raise HTTPException(status_code=400, detail="Latest profile run is unavailable")
    return parent_job


async def _sync_portfolio_queue_state(
    *,
    portfolio: ResearchPortfolio,
    opportunities: list[dict[str, Any]],
) -> None:
    normalized = list_normalized_research_opportunities(opportunities)
    linked_ids = collect_research_opportunity_linked_ids(normalized)
    automation_profile = normalize_portfolio_automation_profile(getattr(portfolio, "automation_profile", None), default="balanced")
    effective_policy = resolve_portfolio_automation_policy(automation_profile, portfolio.automation_policy)
    summary = build_autonomy_summary(
        raw_summary=portfolio.latest_summary if isinstance(portfolio.latest_summary, dict) else {},
        opportunities=normalized,
        automation_profile=automation_profile,
        effective_policy=effective_policy,
        sandbox_profile_id=portfolio.sandbox_profile_id,
        config_revision_key="portfolio_config_revision",
    )
    portfolio.opportunities = normalized
    portfolio.latest_summary = summary
    portfolio.latest_note_ids = list(
        dict.fromkeys(
            [
                *([str(v) for v in (portfolio.latest_note_ids or []) if str(v).strip()]),
                *linked_ids["note_ids"],
            ]
        )
    )[:30]
    portfolio.latest_experiment_plan_ids = linked_ids["plan_ids"][:30]
    portfolio.latest_validation_run_ids = linked_ids["run_ids"][:30]
    portfolio.child_job_ids = linked_ids["child_job_ids"][:50]


async def _sync_profile_queue_state(
    *,
    profile: DomainResearchProfile,
    opportunities: list[dict[str, Any]],
) -> None:
    normalized = list_normalized_research_opportunities(opportunities)
    linked_ids = collect_research_opportunity_linked_ids(normalized)
    payload = _profile_summary_payload(profile)
    summary = payload["summary"]
    summary["opportunities"] = normalized
    profile.latest_summary = summary
    profile.latest_note_ids = list(
        dict.fromkeys([*([str(v) for v in (profile.latest_note_ids or []) if str(v).strip()]), *linked_ids["note_ids"]])
    )[:30]
    profile.latest_experiment_plan_ids = linked_ids["plan_ids"][:30]
    profile.latest_validation_run_ids = linked_ids["run_ids"][:30]


def _normalize_follow_up_autonomy_mode(raw: Any) -> str:
    mode = str(raw or "").strip().lower()
    if mode in {
        FOLLOW_UP_AUTONOMY_MANUAL_ONLY,
        FOLLOW_UP_AUTONOMY_AUTO_LAUNCH_SAFE,
        FOLLOW_UP_AUTONOMY_QUEUE_FOR_APPROVAL,
    }:
        return mode
    return FOLLOW_UP_AUTONOMY_MANUAL_ONLY


def _get_follow_up_policy_from_job(job: Optional[AgentJob]) -> dict[str, Any]:
    config = job.config if isinstance(getattr(job, "config", None), dict) else {}
    automation = research_monitor_profile_service.resolve_monitor_automation_config(config)
    compat = automation["follow_up_autonomy"]
    return {
        "mode": _normalize_follow_up_autonomy_mode(compat.get("mode")),
        "allowed_recommendations": [
            str(value).strip()
            for value in (compat.get("allowed_recommendations") or [])
            if str(value).strip()
        ] or [
            FOLLOW_UP_RECOMMENDATION_DEEP_DIVE_CHAIN,
            FOLLOW_UP_RECOMMENDATION_SINGLE_RESEARCH_JOB,
        ],
        "automation_profile": automation["automation_profile"],
        "automation_policy": automation["automation_policy"],
        "effective_policy": automation["effective_policy"],
    }


def _get_autonomy_budget_from_job(job: Optional[AgentJob]) -> dict[str, int]:
    config = job.config if isinstance(getattr(job, "config", None), dict) else {}
    return research_monitor_profile_service._normalize_budget_config(config.get("autonomy_budget"))


def _decision_trace_reason_label(reason_code: Optional[str]) -> Optional[str]:
    text = str(reason_code or "").strip().lower()
    if not text:
        return None
    return _queue_reason_label(text)


def _decision_trace_parse_time(raw: Any, *, fallback: Optional[datetime] = None) -> Optional[datetime]:
    if isinstance(raw, datetime):
        return raw
    text = str(raw or "").strip()
    if not text:
        return fallback
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return fallback


def _decision_trace_event_id(
    source_kind: str,
    source_id: Optional[str],
    decision_type: str,
    event_time: datetime,
    suffix: Optional[str] = None,
) -> str:
    suffix_text = str(suffix or "").strip()
    timestamp = event_time.isoformat()
    raw = "|".join(
        [
            str(source_kind or "").strip(),
            str(source_id or "").strip(),
            str(decision_type or "").strip(),
            timestamp,
            suffix_text,
        ]
    )
    return uuid.uuid5(uuid.NAMESPACE_URL, raw).hex


def _build_decision_trace_event(
    *,
    event_type: str,
    event_time: datetime,
    source_kind: str,
    source_id: Optional[str],
    source_label: Optional[str],
    decision_type: str,
    summary: str,
    customer: Optional[str] = None,
    reason_code: Optional[str] = None,
    reason_label: Optional[str] = None,
    scheduler_state: Optional[dict[str, Any]] = None,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    actor_mode: Optional[str] = None,
    operator_note: Optional[str] = None,
    before_state: Optional[dict[str, Any]] = None,
    after_state: Optional[dict[str, Any]] = None,
    deep_link: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
    is_derived: bool = False,
    record_origin: Optional[str] = None,
    suffix: Optional[str] = None,
    operator_context: Optional[dict[str, Any]] = None,
) -> AgentDecisionTraceEventResponse:
    normalized_operator_context = (
        _build_operator_queue_context(
            objective=(operator_context or {}).get("objective"),
            domain=(operator_context or {}).get("domain"),
            track_type=(operator_context or {}).get("track_type"),
            source_scope=(operator_context or {}).get("source_scope"),
            repo_source_ids=(operator_context or {}).get("repo_source_ids"),
            benchmark_queries=(operator_context or {}).get("benchmark_queries"),
            sandbox_profile_id=(operator_context or {}).get("sandbox_profile_id"),
            automation_profile=(operator_context or {}).get("automation_profile"),
            effective_policy=(operator_context or {}).get("effective_policy"),
            confidence=(operator_context or {}).get("confidence"),
            readiness=(operator_context or {}).get("readiness"),
            linked_note_ids=(operator_context or {}).get("linked_note_ids"),
            linked_experiment_plan_ids=(operator_context or {}).get("linked_experiment_plan_ids"),
            linked_validation_run_ids=(operator_context or {}).get("linked_validation_run_ids"),
            child_job_ids=(operator_context or {}).get("child_job_ids"),
        )
        if isinstance(operator_context, dict)
        else {}
    )
    return AgentDecisionTraceEventResponse(
        event_id=_decision_trace_event_id(source_kind, source_id, decision_type, event_time, suffix=suffix),
        event_type=str(event_type or "").strip() or str(decision_type or "").strip() or "event",
        event_time=event_time,
        source_kind=str(source_kind or "").strip() or "unknown",
        source_id=str(source_id or "").strip() or None,
        source_label=str(source_label or "").strip() or None,
        customer=str(customer or "").strip() or None,
        decision_type=str(decision_type or "").strip() or "event",
        reason_code=str(reason_code or "").strip() or None,
        reason_label=str(reason_label or "").strip() or None,
        scheduler_state=deepcopy(scheduler_state) if isinstance(scheduler_state, dict) else None,
        status=str(status or "").strip() or None,
        severity=str(severity or "").strip() or None,
        actor_mode=str(actor_mode or "").strip() or None,
        summary=str(summary or "").strip() or "Autonomy event",
        operator_note=str(operator_note or "").strip() or None,
        before_state=before_state or None,
        after_state=after_state or None,
        deep_link=AgentDecisionTraceDeepLinkResponse.model_validate(deep_link) if isinstance(deep_link, dict) else None,
        metadata=metadata or None,
        is_derived=bool(is_derived),
        record_origin=str(record_origin or "").strip() or ("derived" if is_derived else "persisted"),
        domain=normalized_operator_context.get("domain"),
        objective=normalized_operator_context.get("objective"),
        track_type=normalized_operator_context.get("track_type"),
        source_scope=normalized_operator_context.get("source_scope"),
        repo_source_ids=normalized_operator_context.get("repo_source_ids"),
        benchmark_queries=normalized_operator_context.get("benchmark_queries"),
        sandbox_profile_id=normalized_operator_context.get("sandbox_profile_id"),
        automation_profile=normalized_operator_context.get("automation_profile"),
        effective_policy=normalized_operator_context.get("effective_policy"),
        confidence=normalized_operator_context.get("confidence"),
        readiness=normalized_operator_context.get("readiness"),
        linked_note_ids=normalized_operator_context.get("linked_note_ids"),
        linked_experiment_plan_ids=normalized_operator_context.get("linked_experiment_plan_ids"),
        linked_validation_run_ids=normalized_operator_context.get("linked_validation_run_ids"),
        child_job_ids=normalized_operator_context.get("child_job_ids"),
    )


TRACE_TRIAGE_ACTIONS = {
    "acknowledge",
    "start_investigation",
    "resolve",
    "reopen",
    "toggle_pin",
    "assign",
    "unassign",
    "set_due_at",
    "clear_due_at",
    "approve_launch",
    "reject_launch",
    "relaunch_follow_up",
}
TRACE_TRIAGE_STATUSES = {"new", "acknowledged", "investigating", "resolved"}


def _trace_event_follow_up_target(event: AutonomyDecisionEvent) -> tuple[str, str, str]:
    source_kind = str(event.source_kind or "").strip().lower()
    if source_kind not in {"domain_profile", "portfolio"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Decision trace event does not support follow-up approval actions",
        )

    event_kind = str(event.event_type or event.decision_type or "").strip().lower()
    if event_kind not in {"follow_up_queued", "follow_up_queued_for_approval"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Decision trace event is not a pending follow-up approval",
        )

    source_id = str(event.source_id or "").strip()
    if not source_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Decision trace event is missing its follow-up owner identifier",
        )

    metadata = event.event_metadata if isinstance(event.event_metadata, dict) else {}
    deep_link = event.deep_link if isinstance(event.deep_link, dict) else {}
    deep_link_params = deep_link.get("params") if isinstance(deep_link.get("params"), dict) else {}
    opportunity_id = str(
        metadata.get("profile_opportunity_id")
        or metadata.get("portfolio_opportunity_id")
        or metadata.get("opportunity_id")
        or deep_link_params.get("opportunityId")
        or ""
    ).strip()
    if not opportunity_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Decision trace event is missing its follow-up opportunity identifier",
        )

    return source_kind, source_id, opportunity_id


def _trace_event_follow_up_relaunch_job_id(event: AutonomyDecisionEvent) -> str:
    event_kind = str(event.event_type or event.decision_type or "").strip().lower()
    after_state = event.after_state if isinstance(event.after_state, dict) else {}
    outcome_status = str(
        after_state.get("follow_up_outcome_status")
        or event.status
        or event_kind
        or ""
    ).strip().lower()
    if event_kind not in {"follow_up_failed", "follow_up_cancelled"} and outcome_status not in {"failed", "cancelled"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Decision trace event is not a relaunchable follow-up outcome",
        )
    job_id = str(
        after_state.get("follow_up_last_job_id")
        or (
            (event.event_metadata if isinstance(event.event_metadata, dict) else {}).get("follow_up_last_job_id")
        )
        or ""
    ).strip()
    if not job_id:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Decision trace event is missing its follow-up job identifier",
        )
    return job_id


def _normalize_trace_view_filters(filters: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(filters, dict):
        return {}
    normalized: dict[str, Any] = {}
    allowed_text_keys = {
        "source_kind",
        "decision_type",
        "customer",
        "status",
        "severity",
        "actor_mode",
        "triage_status",
        "assigned_to_user_id",
        "escalation_state",
        "date_range",
    }
    for key in allowed_text_keys:
        value = str(filters.get(key) or "").strip()
        if value:
            normalized[key] = value
    for key in {"pinned", "actionable_only", "unassigned_only"}:
        if key in filters:
            normalized[key] = bool(filters.get(key))
    return normalized


async def _list_trace_visible_user_ids(
    db: AsyncSession,
    *,
    current_user: User,
) -> set[UUID]:
    return await list_collaboration_user_ids(db, current_user=current_user)


async def _load_persisted_trace_event_for_user(
    db: AsyncSession,
    *,
    event_id: UUID,
    current_user: User,
) -> Optional[AutonomyDecisionEvent]:
    visible_user_ids = await _list_trace_visible_user_ids(db, current_user=current_user)
    if current_user.id not in visible_user_ids:
        visible_user_ids.add(current_user.id)
    result = await db.execute(
        select(AutonomyDecisionEvent).where(
            AutonomyDecisionEvent.id == event_id,
            AutonomyDecisionEvent.user_id.in_(visible_user_ids),
        )
    )
    return result.scalars().first()


async def _validate_trace_assignee(
    db: AsyncSession,
    *,
    current_user: User,
    assigned_to_user_id: Optional[UUID],
) -> Optional[UUID]:
    if assigned_to_user_id is None:
        return None
    visible_user_ids = await _list_trace_visible_user_ids(db, current_user=current_user)
    if assigned_to_user_id not in visible_user_ids and not current_user.is_admin():
        return None
    user = (
        await db.execute(
            select(User).where(
                User.id == assigned_to_user_id,
                User.is_active.is_(True),
            )
        )
    ).scalars().first()
    return user.id if user is not None else None


async def _build_collaboration_user_lookup(
    db: AsyncSession,
    *,
    current_user: User,
) -> dict[str, User]:
    visible_user_ids = await list_collaboration_user_ids(db, current_user=current_user)
    if current_user.id not in visible_user_ids:
        visible_user_ids.add(current_user.id)
    rows = list((await db.execute(select(User).where(User.id.in_(visible_user_ids)))).scalars().all())
    return {str(row.id): row for row in rows}


def _build_decision_trace_from_queue_items(items: list[AgentCheckpointQueueItemResponse]) -> list[AgentDecisionTraceEventResponse]:
    events: list[AgentDecisionTraceEventResponse] = []
    mapping = {
        "follow_up_recommendation": "follow_up_queued",
        "policy_review": "policy_guardrail_triggered",
        "budget_review": "budget_clamped",
        "job_recovery": "job_recovery_queued",
        "approval_checkpoint": "approval_required",
    }
    for item in items:
        event_time = item.created_at or item.backoff_until or item.next_run_at
        if event_time is None:
            continue
        source_kind = (
            "portfolio"
            if item.portfolio_id
            else "domain_profile"
            if item.domain_research_profile_id
            else "queue"
        )
        source_id = (
            str(item.portfolio_id)
            if item.portfolio_id
            else str(item.domain_research_profile_id)
            if item.domain_research_profile_id
            else str(item.job_id or item.inbox_item_id or item.queue_key)
        )
        source_label = item.portfolio_title or item.domain_research_profile_title or item.job_name or item.title
        params: dict[str, str] = {}
        target_tab = "queue"
        if item.portfolio_id:
            target_tab = "fleet"
            params["fleetId"] = str(item.portfolio_id)
            if item.portfolio_opportunity_id:
                params["opportunityId"] = str(item.portfolio_opportunity_id)
        elif item.domain_research_profile_id:
            target_tab = "domain"
            params["profileId"] = str(item.domain_research_profile_id)
            if item.profile_opportunity_id:
                params["opportunityId"] = str(item.profile_opportunity_id)
        elif item.customer:
            params["health_customer"] = str(item.customer)
        if item.job_id:
            params["job"] = str(item.job_id)
        events.append(
            _build_decision_trace_event(
                event_type=mapping.get(item.item_type, "queue_item_open"),
                event_time=event_time,
                source_kind=source_kind,
                source_id=source_id,
                source_label=source_label,
                customer=item.customer,
                decision_type=mapping.get(item.item_type, "queue_item_open"),
                reason_code=item.reason_code,
                reason_label=item.reason_label,
                scheduler_state=item.scheduler_state,
                status=item.status,
                severity=item.escalation_level,
                actor_mode="autonomous",
                summary=item.summary or item.title,
                operator_note=item.follow_up_operator_note,
                deep_link={
                    "target_tab": target_tab,
                    "job_id": item.job_id,
                    "params": params,
                    "label": f"Open {source_label or 'source'}",
                },
                metadata={
                    "queue_key": item.queue_key,
                    "item_type": item.item_type,
                    "reason_label": item.reason_label,
                    "scheduler_state": item.scheduler_state,
                    "profile_opportunity_id": item.profile_opportunity_id,
                    "portfolio_opportunity_id": item.portfolio_opportunity_id,
                },
                suffix=item.queue_key,
                operator_context=_build_operator_queue_context(
                    objective=item.objective,
                    domain=item.domain,
                    track_type=item.track_type,
                    source_scope=item.source_scope,
                    repo_source_ids=item.repo_source_ids,
                    benchmark_queries=item.benchmark_queries,
                    sandbox_profile_id=item.sandbox_profile_id,
                    automation_profile=item.automation_profile,
                    effective_policy=item.effective_policy,
                    confidence=item.confidence,
                    readiness=item.readiness,
                    linked_note_ids=item.linked_note_ids,
                    linked_experiment_plan_ids=item.linked_experiment_plan_ids,
                    linked_validation_run_ids=item.linked_validation_run_ids,
                    child_job_ids=item.child_job_ids,
                ),
            )
        )
    return events


def _decorate_trace_event_payload(
    payload: dict[str, Any],
    *,
    user_lookup: dict[str, User],
    current_user_id: UUID,
) -> dict[str, Any]:
    owner_user_id = str(payload.get("owner_user_id") or "").strip()
    assignee_user_id = str(payload.get("assigned_to_user_id") or "").strip()
    owner = user_lookup.get(owner_user_id)
    assignee = user_lookup.get(assignee_user_id)
    payload["owner_label"] = (
        str(owner.full_name or owner.username or owner.email or owner.id).strip()
        if owner is not None
        else None
    )
    payload["assignee_label"] = (
        str(assignee.full_name or assignee.username or assignee.email or assignee.id).strip()
        if assignee is not None
        else None
    )
    payload["is_owned_by_current_user"] = owner_user_id == str(current_user_id)
    payload["is_assigned_to_current_user"] = assignee_user_id == str(current_user_id)
    return payload


def _build_decision_trace_from_job(job: AgentJob) -> list[AgentDecisionTraceEventResponse]:
    events: list[AgentDecisionTraceEventResponse] = []
    execution_strategy = job.results.get("execution_strategy") if isinstance(getattr(job, "results", None), dict) and isinstance(job.results.get("execution_strategy"), dict) else {}
    job_reference_time = job.last_activity_at or job.completed_at or job.started_at or job.created_at
    operator_interventions = derive_operator_interventions_with_outcomes(
        execution_strategy.get("operator_interventions") if isinstance(execution_strategy.get("operator_interventions"), list) else [],
        current_status=job.status,
        completed_at=getattr(job, "completed_at", None),
    )
    customer = _queue_customer_for_job(job)
    for index, row in enumerate(operator_interventions):
        if not isinstance(row, dict):
            continue
        event_time = _decision_trace_parse_time(row.get("at"), fallback=job_reference_time)
        if event_time is None:
            continue
        action = str(row.get("action") or "operator_intervention").strip().lower()
        events.append(
            _build_decision_trace_event(
                event_type="job_operator_action",
                event_time=event_time,
                source_kind="job",
                source_id=str(job.id) if job.id else None,
                source_label=job.name,
                customer=customer,
                decision_type=action or "operator_intervention",
                reason_code=str(row.get("outcome_status") or "").strip() or None,
                reason_label=None,
                status=str(row.get("job_status_after") or row.get("job_status_before") or job.status or "").strip() or None,
                severity="medium",
                actor_mode="operator",
                summary=f"{job.name}: {action.replace('_', ' ')}",
                operator_note=str(row.get("note") or "").strip() or None,
                before_state={"job_status": row.get("job_status_before")} if row.get("job_status_before") else None,
                after_state={"job_status": row.get("job_status_after")} if row.get("job_status_after") else None,
                deep_link={
                    "target_tab": "jobs",
                    "job_id": job.id,
                    "params": {"job": str(job.id)},
                    "label": "Open Job",
                },
                metadata={
                    "outcome_status": row.get("outcome_status"),
                    "outcome_reason": row.get("outcome_reason"),
                    "metadata": row.get("metadata"),
                },
                suffix=str(index),
            )
        )
    scheduler_state = execution_strategy.get("scheduler_state") if isinstance(execution_strategy.get("scheduler_state"), dict) else {}
    queue_reason = str(scheduler_state.get("queue_reason") or "").strip().lower()
    scheduled_at = _decision_trace_parse_time(
        scheduler_state.get("last_dispatched_at") or scheduler_state.get("last_scheduled_at") or scheduler_state.get("backoff_until"),
        fallback=job_reference_time,
    )
    if queue_reason and scheduled_at is not None:
        reason_label = _decision_trace_reason_label(queue_reason)
        events.append(
            _build_decision_trace_event(
                event_type="job_recovery_queued",
                event_time=scheduled_at,
                source_kind="job",
                source_id=str(job.id) if job.id else None,
                source_label=job.name,
                customer=customer,
                decision_type="job_recovery_queued",
                reason_code=queue_reason,
                reason_label=reason_label,
                scheduler_state=scheduler_state,
                status=job.status,
                severity="high" if queue_reason in {"execution_failure", "stalled_run"} else "medium",
                actor_mode="autonomous",
                summary=f"{job.name}: queued for scheduler recovery",
                deep_link={
                    "target_tab": "queue",
                    "job_id": job.id,
                    "params": {"tab": "queue", "job": str(job.id)},
                    "label": "Open Checkpoint Queue",
                },
                metadata={
                    "scheduler_state": scheduler_state,
                    "reason_label": reason_label,
                },
                suffix="scheduler",
            )
        )
    return events


def _build_decision_trace_from_opportunities(
    *,
    source_kind: str,
    source_id: str,
    source_label: str,
    customer: Optional[str],
    opportunities: list[dict[str, Any]],
    deep_link_params: dict[str, str],
    domain: Optional[str] = None,
    objective: Optional[str] = None,
    track_type: Optional[str] = None,
    source_scope: Optional[str] = None,
    repo_source_ids: Any = None,
    benchmark_queries: Any = None,
    sandbox_profile_id: Optional[str] = None,
    automation_profile: Optional[str] = None,
    effective_policy: Optional[dict[str, Any]] = None,
) -> list[AgentDecisionTraceEventResponse]:
    events: list[AgentDecisionTraceEventResponse] = []
    for row in opportunities:
        event_time = _decision_trace_parse_time(
            row.get("follow_up_outcome_recorded_at")
            or row.get("follow_up_launched_at")
            or row.get("follow_up_reviewed_at")
            or row.get("last_material_change_at")
            or row.get("last_evaluated_at")
            or row.get("updated_at")
        )
        if event_time is None:
            continue
        autonomy_state = str(row.get("autonomy_state") or "").strip().lower()
        review_status = str(row.get("follow_up_review_status") or "").strip().lower()
        follow_up_outcome_status = str(row.get("follow_up_outcome_status") or "").strip().lower()
        reason_code = (
            str(
                row.get("last_decision_reason_code")
                or row.get("last_blocked_reason_code")
                or row.get("last_skip_reason_code")
                or ""
            ).strip()
            or None
        )
        decision_type = str(row.get("last_decision_type") or "").strip().lower()
        event_type = decision_type or "opportunity_updated"
        status = autonomy_state or str(row.get("stage") or "").strip().lower() or None
        if follow_up_outcome_status == "completed":
            event_type = "follow_up_completed"
            decision_type = "follow_up_completed"
            status = "completed"
        elif follow_up_outcome_status == "failed":
            event_type = "follow_up_failed"
            decision_type = "follow_up_failed"
            status = "failed"
        elif follow_up_outcome_status == "cancelled":
            event_type = "follow_up_cancelled"
            decision_type = "follow_up_cancelled"
            status = "cancelled"
        elif review_status == "approved_launch":
            event_type = "follow_up_approved"
            decision_type = "follow_up_approved"
        elif review_status == "rejected":
            event_type = "follow_up_rejected"
            decision_type = "follow_up_rejected"
        elif autonomy_state == "cooldown":
            event_type = "opportunity_cooldown"
            decision_type = "opportunity_cooldown"
        elif autonomy_state == "blocked_structural":
            event_type = "opportunity_blocked"
            decision_type = "opportunity_blocked"
        elif autonomy_state == "completed_waiting_change":
            event_type = "opportunity_completed_waiting_change"
            decision_type = "opportunity_completed_waiting_change"
        elif review_status == "pending_approval":
            event_type = "follow_up_queued"
            decision_type = "follow_up_queued"
        elif autonomy_state == "active":
            event_type = "follow_up_launched"
            decision_type = "follow_up_launched"
        title = str(row.get("title") or row.get("canonical_key") or row.get("opportunity_id") or "Opportunity").strip()
        summary = f"{source_label}: {title} is {event_type.replace('_', ' ')}"
        if follow_up_outcome_status and str(row.get("follow_up_outcome_summary") or "").strip():
            summary = f"{summary} - {str(row.get('follow_up_outcome_summary') or '').strip()}"
        event_actor_mode = "operator" if review_status in {"approved_launch", "rejected"} else "autonomous"
        deep_link_params_with_target = dict(deep_link_params)
        opportunity_id = str(row.get("opportunity_id") or "").strip()
        if opportunity_id:
            deep_link_params_with_target["opportunityId"] = opportunity_id
        if source_kind == "domain_profile" and source_id:
            deep_link_params_with_target.setdefault("profileId", source_id)
        if source_kind == "portfolio" and source_id:
            deep_link_params_with_target.setdefault("fleetId", source_id)
        events.append(
            _build_decision_trace_event(
                event_type=event_type,
                event_time=event_time,
                source_kind=source_kind,
                source_id=source_id,
                source_label=source_label,
                customer=customer,
                decision_type=decision_type or event_type,
                reason_code=reason_code,
                reason_label=_decision_trace_reason_label(reason_code),
                status=status,
                severity="high" if autonomy_state == "blocked_structural" else "medium" if autonomy_state == "cooldown" else "normal",
                actor_mode=event_actor_mode,
                summary=summary,
                operator_note=str(row.get("follow_up_review_note") or row.get("operator_note") or "").strip() or None,
                before_state=None,
                after_state={
                    "autonomy_state": autonomy_state or None,
                    "stage": row.get("stage"),
                    "review_status": review_status or None,
                    "follow_up_outcome_status": follow_up_outcome_status or None,
                    "follow_up_last_job_id": row.get("follow_up_last_job_id"),
                },
                deep_link={
                    "target_tab": deep_link_params.get("tab") or source_kind,
                    "params": deep_link_params_with_target,
                    "label": f"Open {source_label}",
                },
                metadata={
                    "opportunity_id": row.get("opportunity_id"),
                    "canonical_key": row.get("canonical_key"),
                    "evidence_revision": row.get("evidence_revision"),
                    "follow_up_outcome_summary": row.get("follow_up_outcome_summary"),
                },
                suffix=str(row.get("opportunity_id") or row.get("canonical_key") or ""),
                operator_context=_build_operator_queue_context(
                    objective=str(row.get("objective") or objective or "").strip() or None,
                    domain=str(row.get("domain") or domain or "").strip() or None,
                    track_type=str(row.get("track_type") or track_type or "").strip() or None,
                    source_scope=str(row.get("source_scope") or source_scope or "").strip() or None,
                    repo_source_ids=row.get("repo_source_ids") or repo_source_ids,
                    benchmark_queries=row.get("benchmark_queries") or benchmark_queries,
                    sandbox_profile_id=str(row.get("sandbox_profile_id") or sandbox_profile_id or "").strip() or None,
                    automation_profile=str(row.get("automation_profile") or automation_profile or "").strip() or None,
                    effective_policy=(
                        row.get("effective_policy")
                        if isinstance(row.get("effective_policy"), dict)
                        else effective_policy
                    ),
                    confidence=row.get("confidence"),
                    readiness=row.get("readiness"),
                    linked_note_ids=row.get("linked_note_ids"),
                    linked_experiment_plan_ids=row.get("linked_experiment_plan_ids"),
                    linked_validation_run_ids=row.get("linked_validation_run_ids"),
                    child_job_ids=row.get("child_job_ids"),
                ),
            )
        )
    return events


def _build_decision_trace_from_monitor_snapshot(snapshot: dict[str, Any]) -> list[AgentDecisionTraceEventResponse]:
    events: list[AgentDecisionTraceEventResponse] = []
    for row in snapshot.get("monitors") or []:
        if not isinstance(row, dict):
            continue
        source_id = str(row.get("monitor_job_id") or row.get("customer") or row.get("monitor_name") or "").strip() or None
        source_label = str(row.get("monitor_name") or row.get("customer") or "Research monitor").strip()
        customer = str(row.get("customer") or "").strip() or None
        for entry in row.get("recent_policy_history") or []:
            if not isinstance(entry, dict):
                continue
            event_time = _decision_trace_parse_time(entry.get("at"))
            if event_time is None:
                continue
            change_source = str(entry.get("change_source") or "").strip().lower()
            decision_type = "policy_rollback" if "rollback" in change_source else "policy_updated"
            events.append(
                _build_decision_trace_event(
                    event_type=decision_type,
                    event_time=event_time,
                    source_kind="monitor",
                    source_id=source_id,
                    source_label=source_label,
                    customer=customer,
                    decision_type=decision_type,
                    reason_code=str(entry.get("change_source") or "").strip() or None,
                    reason_label=_decision_trace_reason_label(str(entry.get("change_source") or "").strip() or None),
                    status=str(row.get("policy_guardrail_state") or row.get("policy_guardrail_status") or "").strip() or None,
                    severity="medium",
                    actor_mode="operator",
                    summary=f"{source_label}: {decision_type.replace('_', ' ')}",
                    operator_note=str(entry.get("change_reason") or "").strip() or None,
                    before_state={"effective_policy": entry.get("previous_effective_policy")},
                    after_state={"effective_policy": entry.get("next_effective_policy")},
                    deep_link={
                        "target_tab": "health",
                        "params": {"tab": "health", "health_customer": customer or ""},
                        "label": "Open Autonomy Health",
                    },
                    metadata={"history_entry_id": entry.get("id")},
                    suffix=str(entry.get("id") or ""),
                )
            )
        if str(row.get("policy_guardrail_status") or "").strip().lower() == "active":
            event_time = _decision_trace_parse_time(
                row.get("latest_policy_changed_at"),
                fallback=_decision_trace_parse_time(snapshot.get("generated_at")),
            )
            if event_time is not None:
                events.append(
                    _build_decision_trace_event(
                        event_type="policy_guardrail_triggered",
                        event_time=event_time,
                        source_kind="monitor",
                        source_id=source_id,
                        source_label=source_label,
                        customer=customer,
                        decision_type="policy_guardrail_triggered",
                        reason_code=str(row.get("policy_guardrail_action") or "").strip() or None,
                        reason_label=_decision_trace_reason_label(str(row.get("policy_guardrail_action") or "").strip() or None),
                        status=str(row.get("policy_guardrail_state") or row.get("policy_guardrail_status") or "").strip() or None,
                        severity="high",
                        actor_mode="autonomous",
                        summary=f"{source_label}: policy guardrail triggered",
                        before_state={"effective_policy": row.get("effective_policy")},
                        after_state={"target_policy": row.get("policy_guardrail_target_policy")},
                        deep_link={
                            "target_tab": "health",
                            "params": {"tab": "health", "health_customer": customer or ""},
                            "label": "Open Autonomy Health",
                        },
                        metadata={"reasons": row.get("policy_guardrail_reasons")},
                        suffix="guardrail",
                    )
                )
        if str(row.get("budget_clamp_state") or "").strip().lower() not in {"", "normal", "none"}:
            event_time = _decision_trace_parse_time(
                row.get("latest_budget_changed_at"),
                fallback=_decision_trace_parse_time(snapshot.get("generated_at")),
            )
            if event_time is not None:
                events.append(
                    _build_decision_trace_event(
                        event_type="budget_clamped",
                        event_time=event_time,
                        source_kind="monitor",
                        source_id=source_id,
                        source_label=source_label,
                        customer=customer,
                        decision_type="budget_clamped",
                        reason_code=str((row.get("budget_clamp_reasons") or [None])[0] or "").strip() or None,
                        reason_label=_decision_trace_reason_label(str((row.get("budget_clamp_reasons") or [None])[0] or "").strip() or None),
                        status=str(row.get("budget_clamp_state") or "").strip() or None,
                        severity="high",
                        actor_mode="autonomous",
                        summary=f"{source_label}: budget clamp active",
                        after_state={"budget_clamp_state": row.get("budget_clamp_state")},
                        deep_link={
                            "target_tab": "health",
                            "params": {"tab": "health", "health_customer": customer or ""},
                            "label": "Open Autonomy Health",
                        },
                        metadata={"budget_clamp_reasons": row.get("budget_clamp_reasons")},
                        suffix="budget",
                    )
                )
    for customer_row in snapshot.get("customers") or []:
        if not isinstance(customer_row, dict):
            continue
        customer = str(customer_row.get("customer") or "").strip() or None
        for entry in customer_row.get("recent_rebalance_history") or []:
            if not isinstance(entry, dict):
                continue
            event_time = _decision_trace_parse_time(entry.get("at"))
            if event_time is None:
                continue
            events.append(
                _build_decision_trace_event(
                    event_type="customer_rebalanced",
                    event_time=event_time,
                    source_kind="monitor",
                    source_id=customer or str(entry.get("id") or ""),
                    source_label=customer or "Customer portfolio",
                    customer=customer,
                    decision_type="customer_rebalanced",
                    reason_code=str(entry.get("change_source") or "").strip() or None,
                    reason_label=_decision_trace_reason_label(str(entry.get("change_source") or "").strip() or None),
                    status=str(entry.get("evaluation_status") or entry.get("evaluation_state") or "").strip() or None,
                    severity="medium",
                    actor_mode="operator",
                    summary=f"{customer or 'Customer'}: customer rebalance applied",
                    operator_note=str(entry.get("change_reason") or "").strip() or None,
                    before_state={"before_capacity": entry.get("before_capacity")},
                    after_state={"after_capacity": entry.get("after_capacity")},
                    deep_link={
                        "target_tab": "health",
                        "params": {"tab": "health", "health_customer": customer or ""},
                        "label": "Open Autonomy Health",
                    },
                    metadata={"history_entry_id": entry.get("id")},
                    suffix=str(entry.get("id") or ""),
                )
            )
    return events


def _build_decision_trace_from_validation_runs(runs: list[ExperimentRun]) -> list[AgentDecisionTraceEventResponse]:
    events: list[AgentDecisionTraceEventResponse] = []
    for run in runs:
        config = run.config if isinstance(run.config, dict) else {}
        scientific_validation = config.get("scientific_validation") if isinstance(config.get("scientific_validation"), dict) else {}
        execution_handoff = config.get("execution_handoff") if isinstance(config.get("execution_handoff"), dict) else {}
        autonomous_origin = execution_handoff.get("autonomous_origin") if isinstance(execution_handoff.get("autonomous_origin"), dict) else {}
        profile_snapshot = scientific_validation.get("profile_snapshot") if isinstance(scientific_validation.get("profile_snapshot"), dict) else {}
        blocked_reason_code = str(
            scientific_validation.get("blocked_reason_code") or scientific_validation.get("blocked_reason") or ""
        ).strip()
        operator_actions = scientific_validation.get("operator_actions") if isinstance(scientific_validation.get("operator_actions"), list) else []
        hypothesis_id = str(scientific_validation.get("hypothesis_id") or autonomous_origin.get("opportunity_id") or "").strip() or None
        profile_id = str(scientific_validation.get("domain_research_profile_id") or "").strip() or None
        portfolio_id = str(scientific_validation.get("research_portfolio_id") or "").strip() or None
        deep_link: dict[str, Any] = {
            "target_tab": "jobs",
            "job_id": run.agent_job_id,
            "params": {"job": str(run.agent_job_id)} if run.agent_job_id else {},
            "label": "Open Validation Job",
        }
        if profile_id:
            deep_link = {
                "target_tab": "domain",
                "job_id": run.agent_job_id,
                "params": {
                    "tab": "domain",
                    "profileId": profile_id,
                    **({"opportunityId": hypothesis_id} if hypothesis_id else {}),
                    **({"job": str(run.agent_job_id)} if run.agent_job_id else {}),
                },
                "label": "Open Domain",
            }
        elif portfolio_id:
            deep_link = {
                "target_tab": "fleet",
                "job_id": run.agent_job_id,
                "params": {
                    "tab": "fleet",
                    "fleetId": portfolio_id,
                    **({"opportunityId": hypothesis_id} if hypothesis_id else {}),
                    **({"job": str(run.agent_job_id)} if run.agent_job_id else {}),
                },
                "label": "Open Fleet",
            }
        operator_context = _build_operator_queue_context(
            objective=str(scientific_validation.get("decision_summary") or run.summary or "").strip() or None,
            domain=str(profile_snapshot.get("domain") or "").strip() or None,
            track_type=str(scientific_validation.get("track_type") or "").strip() or None,
            source_scope=str(scientific_validation.get("source_scope") or "").strip() or None,
            repo_source_ids=scientific_validation.get("repo_source_ids"),
            benchmark_queries=scientific_validation.get("benchmark_queries"),
            sandbox_profile_id=str(scientific_validation.get("sandbox_profile_id") or "").strip() or None,
            automation_profile=str(scientific_validation.get("automation_profile") or "").strip() or None,
            effective_policy=scientific_validation.get("effective_policy") if isinstance(scientific_validation.get("effective_policy"), dict) else None,
            confidence=scientific_validation.get("confidence"),
            readiness=scientific_validation.get("readiness"),
            linked_experiment_plan_ids=[str(run.experiment_plan_id)] if run.experiment_plan_id else None,
            linked_validation_run_ids=[str(run.id)] if run.id else None,
            child_job_ids=[str(run.agent_job_id)] if run.agent_job_id else None,
        )
        if blocked_reason_code:
            event_time = run.updated_at or run.created_at
            events.append(
                _build_decision_trace_event(
                    event_type="validation_blocked",
                    event_time=event_time,
                    source_kind="validation_run",
                    source_id=str(run.id) if run.id else None,
                    source_label=run.name,
                    decision_type="validation_blocked",
                    reason_code=blocked_reason_code,
                    reason_label=_decision_trace_reason_label(blocked_reason_code),
                    status=run.status,
                    severity="high",
                    actor_mode="autonomous",
                    summary=f"{run.name}: validation blocked",
                    deep_link=deep_link,
                    metadata={
                        "experiment_plan_id": str(run.experiment_plan_id) if run.experiment_plan_id else None,
                        "opportunity_id": hypothesis_id,
                    },
                    suffix="blocked",
                    operator_context=operator_context,
                )
            )
        for index, action in enumerate(operator_actions):
            if not isinstance(action, dict):
                continue
            event_time = _decision_trace_parse_time(action.get("at"), fallback=run.updated_at or run.created_at)
            if event_time is None:
                continue
            action_name = str(action.get("action") or "operator_action").strip().lower()
            event_type = "validation_requeued" if action_name in {"requeue", "retry", "restart"} else "validation_operator_action"
            events.append(
                _build_decision_trace_event(
                    event_type=event_type,
                    event_time=event_time,
                    source_kind="validation_run",
                    source_id=str(run.id) if run.id else None,
                    source_label=run.name,
                    decision_type=event_type,
                    reason_code=str(action.get("outcome_status") or action_name or "").strip() or None,
                    reason_label=_decision_trace_reason_label(str(action.get("outcome_status") or action_name or "").strip() or None),
                    status=str(action.get("new_status") or run.status or "").strip() or None,
                    severity="medium",
                    actor_mode="operator",
                    summary=f"{run.name}: {action_name.replace('_', ' ')}",
                    operator_note=str(action.get("note") or "").strip() or None,
                    before_state={"status": action.get("previous_status")} if action.get("previous_status") else None,
                    after_state={"status": action.get("new_status")} if action.get("new_status") else None,
                    deep_link=deep_link,
                    metadata={
                        "linked_job_id": action.get("linked_job_id"),
                        "experiment_plan_id": str(run.experiment_plan_id) if run.experiment_plan_id else None,
                        "opportunity_id": hypothesis_id,
                    },
                    suffix=str(index),
                    operator_context=operator_context,
                )
            )
    return events


async def _record_job_operator_event(
    *,
    db: AsyncSession,
    job: AgentJob,
    current_user: User,
    action: str,
    note: Optional[str],
    previous_status: Optional[str],
    next_status: Optional[str],
    scheduler_state: Optional[dict[str, Any]] = None,
    metadata: Optional[dict[str, Any]] = None,
    summary: Optional[str] = None,
) -> None:
    await record_autonomy_decision_event(
        db,
        user_id=current_user.id,
        event_type="job_operator_action",
        event_time=datetime.utcnow(),
        source_kind="job",
        source_id=str(job.id) if job.id else None,
        source_label=str(job.name or "Agent job").strip() or "Agent job",
        customer=_queue_customer_for_job(job),
        decision_type=str(action or "").strip().lower() or "operator_intervention",
        reason_code=(str((metadata or {}).get("reason_code") or "").strip() or None),
        status=str(next_status or job.status or "").strip() or None,
        severity="medium",
        actor_mode="operator",
        summary=summary or f"{str(job.name or 'Agent job').strip()}: {str(action or 'operator action').replace('_', ' ')}",
        operator_note=note,
        reason_label=_decision_trace_reason_label((metadata or {}).get("reason_code")),
        scheduler_state=scheduler_state if isinstance(scheduler_state, dict) else None,
        before_state={"job_status": previous_status} if previous_status else None,
        after_state={"job_status": next_status} if next_status else None,
        deep_link={
            "target_tab": "jobs",
            "job_id": str(job.id) if job.id else None,
            "params": {"job": str(job.id)} if job.id else {},
            "label": "Open Job",
        },
        metadata=metadata or None,
    )

def _tokenize_learning_text(text: str) -> list[str]:
    raw = re.findall(r"[a-zA-Z0-9_\-]+", (text or "").lower())
    stop = {
        "the","and","for","with","from","that","this","into","over","under","when","where","what","which","while",
        "your","you","are","our","their","they","them","then","than","also","only","just","more","most","less",
        "use","using","used","make","made","help","helps","via","can","could","should","would","may","might","will",
        "data","dataset","datasets","model","models","train","training","eval","evaluate","evaluation","assistant",
        "job","jobs","paper","papers","doc","docs","document","documents","research","monitor",
    }
    out: list[str] = []
    for token in raw:
        token = token.strip("_-")
        if len(token) < 3 or token in stop:
            continue
        out.append(token)
    return out


async def _load_follow_up_learning_profile(
    *,
    db: AsyncSession,
    user_id: UUID,
    customer: Optional[str],
) -> dict[str, Any]:
    try:
        from app.services.research_monitor_profile_service import research_monitor_profile_service

        profile = await research_monitor_profile_service.get_profile(
            db=db,
            user_id=user_id,
            customer=customer,
        )
    except Exception:
        profile = None
    if profile is None:
        return {
            "token_scores": {},
            "phrase_scores": {},
            "recommendation_scores": {},
            "source_type_scores": {},
            "outcome_counters": {},
        }
    return {
        "token_scores": profile.token_scores if isinstance(getattr(profile, "token_scores", None), dict) else {},
        "phrase_scores": profile.phrase_scores if isinstance(getattr(profile, "phrase_scores", None), dict) else {},
        "recommendation_scores": (
            profile.recommendation_scores if isinstance(getattr(profile, "recommendation_scores", None), dict) else {}
        ),
        "source_type_scores": (
            profile.source_type_scores if isinstance(getattr(profile, "source_type_scores", None), dict) else {}
        ),
        "outcome_counters": profile.outcome_counters if isinstance(getattr(profile, "outcome_counters", None), dict) else {},
    }


def _score_follow_up_action_for_item(
    item: ResearchInboxItem,
    action_row: AgentCheckpointQueueActionResponse,
    *,
    learning_profile: Optional[dict[str, Any]] = None,
) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []
    recommendation_key = str(action_row.recommendation_key or "").strip()
    item_type = str(item.item_type or "").strip().lower()
    title = str(item.title or "").strip()
    summary = str(item.summary or "").strip()
    text = f"{title} {summary}".strip()
    tokens = _tokenize_learning_text(text)
    phrases = [f"{tokens[idx]} {tokens[idx + 1]}" for idx in range(len(tokens) - 1)]

    token_scores = learning_profile.get("token_scores") if isinstance(learning_profile, dict) else {}
    phrase_scores = learning_profile.get("phrase_scores") if isinstance(learning_profile, dict) else {}
    recommendation_scores = learning_profile.get("recommendation_scores") if isinstance(learning_profile, dict) else {}
    source_type_scores = learning_profile.get("source_type_scores") if isinstance(learning_profile, dict) else {}

    if recommendation_key and recommendation_key in recommendation_scores:
        delta = int(recommendation_scores.get(recommendation_key) or 0)
        score += delta * 10
        reasons.append(f"learned_recommendation:{delta}")

    if item_type and item_type in source_type_scores:
        delta = int(source_type_scores.get(item_type) or 0)
        score += delta * 6
        reasons.append(f"source_type:{item_type}:{delta}")

    token_delta = sum(int(token_scores.get(token) or 0) for token in tokens[:10]) if isinstance(token_scores, dict) else 0
    if token_delta:
        score += token_delta
        reasons.append("token_bias")

    phrase_delta = sum(int(phrase_scores.get(phrase) or 0) for phrase in phrases[:6]) if isinstance(phrase_scores, dict) else 0
    if phrase_delta:
        score += phrase_delta * 2
        reasons.append("phrase_bias")

    if recommendation_key == FOLLOW_UP_RECOMMENDATION_DEEP_DIVE_CHAIN:
        score += 24
        reasons.append("deep_dive_default")
        if item_type == "arxiv":
            score += 4
            reasons.append("paper_deep_dive_fit")
    elif recommendation_key == FOLLOW_UP_RECOMMENDATION_SINGLE_RESEARCH_JOB:
        score += 18
        reasons.append("single_job_default")
        if item_type == "document":
            score += 3
            reasons.append("document_single_job_fit")
    elif recommendation_key == FOLLOW_UP_RECOMMENDATION_REPO_PATCH_CHAIN:
        score += 8
        reasons.append("repo_patch_specialized")
        meta = item.item_metadata if isinstance(item.item_metadata, dict) else {}
        repos = meta.get("repos") if isinstance(meta.get("repos"), list) else []
        if repos:
            score += 14
            reasons.append("repos_present")
        if item_type == "arxiv":
            score += 5
            reasons.append("paper_repo_fit")

    if str(action_row.autonomy_eligibility or "").strip().lower() == "auto_launchable":
        score += 5
        reasons.append("safe_autonomy_eligible")

    return int(score), reasons[:5]


def _customer_profile_key(customer: Optional[str]) -> str:
    return str(customer or "").strip().lower()


async def _launch_follow_up_action(
    action_row: AgentCheckpointQueueActionResponse,
    *,
    db: AsyncSession,
    current_user: User,
) -> AgentJobResponse:
    if action_row.chain_create_payload:
        request = AgentJobFromChainCreate.model_validate(action_row.chain_create_payload)
        return await create_job_from_chain(request, db, current_user)
    if action_row.job_create_payload:
        request = AgentJobCreate.model_validate(action_row.job_create_payload)
        return await create_agent_job(request, db, current_user)
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="Recommendation could not be translated into a bounded launch payload",
    )


async def _apply_follow_up_policy_on_accept(
    *,
    item: ResearchInboxItem,
    current_user: User,
    db: AsyncSession,
) -> None:
    if item.follow_up_launch_status == "launched":
        return

    source_job = None
    if item.job_id:
        source_job = await db.get(AgentJob, item.job_id)
        if source_job and source_job.user_id != current_user.id:
            source_job = None

    policy = _get_follow_up_policy_from_job(source_job)
    mode = policy["mode"]
    allowed_recommendations = set(policy["allowed_recommendations"])
    budget_snapshot = await research_monitor_profile_service.build_monitor_budget_snapshot(
        db=db,
        user_id=current_user.id,
        monitor_job=source_job,
    ) if source_job else {
        "autonomy_budget": research_monitor_profile_service._normalize_budget_config(None),
        "budget_usage": research_monitor_profile_service._empty_budget_usage(),
        "budget_remaining": research_monitor_profile_service._empty_budget_usage(),
        "budget_throttle_state": "normal",
        "budget_throttle_reasons": [],
    }
    customer_budget_snapshot = await research_monitor_profile_service.build_customer_budget_snapshot(
        db=db,
        user_id=current_user.id,
        customer=(str(item.customer or "").strip() or None),
    )
    learning_profile = await _load_follow_up_learning_profile(
        db=db,
        user_id=current_user.id,
        customer=(str(item.customer or "").strip() or None),
    )
    actions = _build_follow_up_actions_for_inbox_item(item, learning_profile=learning_profile)
    preferred_action = next((action for action in actions if action.recommended), actions[0] if actions else None)
    recommended_action = preferred_action
    if mode != FOLLOW_UP_AUTONOMY_MANUAL_ONLY:
        for action in actions:
            recommendation_key = str(action.recommendation_key or "").strip()
            eligibility = str(action.autonomy_eligibility or "").strip().lower() or "manual_only"
            if eligibility == "auto_launchable" and recommendation_key in allowed_recommendations:
                recommended_action = action
                break

    item.follow_up_policy_mode = mode
    item.follow_up_job_id = None
    item.follow_up_chain_definition_id = None
    item.follow_up_launched_at = None
    item.follow_up_operator_decision = None
    item.follow_up_operator_note = None
    item.follow_up_operator_acted_at = None
    item.follow_up_operator_user_id = None
    item.follow_up_outcome_status = None
    item.follow_up_outcome_recorded_at = None
    item.follow_up_outcome_summary = None
    item.follow_up_budget_decision = None
    item.follow_up_budget_reason = None
    item.follow_up_budget_throttle_state = str(budget_snapshot.get("budget_throttle_state") or "normal")
    item.follow_up_customer_budget_decision = None
    item.follow_up_customer_budget_reason = None
    item.follow_up_customer_budget_throttle_state = str(customer_budget_snapshot.get("customer_budget_throttle_state") or "normal")

    if recommended_action is None:
        item.follow_up_decision = "manual"
        item.follow_up_launch_status = "blocked"
        item.follow_up_block_reason = "No supported follow-up recommendation is available for this inbox item."
        item.follow_up_recommendation_key = None
        return

    recommendation_key = str(recommended_action.recommendation_key or "").strip() or None
    eligibility = str(recommended_action.autonomy_eligibility or "").strip().lower() or "manual_only"
    item.follow_up_recommendation_key = recommendation_key

    if mode == FOLLOW_UP_AUTONOMY_MANUAL_ONLY:
        item.follow_up_decision = "manual"
        item.follow_up_launch_status = "blocked"
        item.follow_up_block_reason = "Monitor policy is set to manual follow-up launches."
        return

    if eligibility != "auto_launchable":
        item.follow_up_decision = "manual"
        item.follow_up_launch_status = "blocked"
        item.follow_up_block_reason = "Recommended follow-up is outside the safe auto-launch allowlist."
        return

    if recommendation_key and recommendation_key not in allowed_recommendations:
        item.follow_up_decision = "manual"
        item.follow_up_launch_status = "blocked"
        item.follow_up_block_reason = "Recommendation is not allowlisted by this monitor policy."
        return

    throttle_state = str(budget_snapshot.get("budget_throttle_state") or "normal")
    throttle_reasons = [str(reason).strip() for reason in (budget_snapshot.get("budget_throttle_reasons") or []) if str(reason).strip()]
    customer_throttle_state = str(customer_budget_snapshot.get("customer_budget_throttle_state") or "normal")
    customer_throttle_reasons = [
        str(reason).strip()
        for reason in (customer_budget_snapshot.get("customer_budget_throttle_reasons") or [])
        if str(reason).strip()
    ]
    effective_mode = mode
    if mode == FOLLOW_UP_AUTONOMY_AUTO_LAUNCH_SAFE and throttle_state == "auto_launch_throttled":
        effective_mode = FOLLOW_UP_AUTONOMY_QUEUE_FOR_APPROVAL
        item.follow_up_budget_decision = "downgraded_to_queue"
    elif throttle_state == "manual_only_clamped":
        effective_mode = FOLLOW_UP_AUTONOMY_MANUAL_ONLY
        item.follow_up_budget_decision = "clamped_to_manual"
    if mode == FOLLOW_UP_AUTONOMY_AUTO_LAUNCH_SAFE and customer_throttle_state == "auto_launch_throttled":
        effective_mode = FOLLOW_UP_AUTONOMY_QUEUE_FOR_APPROVAL
        item.follow_up_customer_budget_decision = "downgraded_to_queue"
    elif customer_throttle_state == "manual_only_clamped":
        effective_mode = FOLLOW_UP_AUTONOMY_MANUAL_ONLY
        item.follow_up_customer_budget_decision = "clamped_to_manual"
    if effective_mode != mode:
        if item.follow_up_budget_decision:
            item.follow_up_budget_reason = "; ".join(throttle_reasons[:3]) or "Monitor autonomy budget is currently exhausted."
            item.follow_up_budget_throttle_state = throttle_state
        if item.follow_up_customer_budget_decision:
            item.follow_up_customer_budget_reason = "; ".join(customer_throttle_reasons[:3]) or "Customer autonomy budget is currently exhausted."
            item.follow_up_customer_budget_throttle_state = customer_throttle_state
        item.follow_up_block_reason = item.follow_up_customer_budget_reason or item.follow_up_budget_reason

    if effective_mode == FOLLOW_UP_AUTONOMY_QUEUE_FOR_APPROVAL:
        item.follow_up_decision = "queued_for_approval"
        item.follow_up_launch_status = "pending_approval"
        item.follow_up_block_reason = item.follow_up_customer_budget_reason or item.follow_up_budget_reason or "Safe follow-up is prepared and waiting for operator approval."
        return

    if effective_mode == FOLLOW_UP_AUTONOMY_MANUAL_ONLY:
        item.follow_up_decision = "manual"
        item.follow_up_launch_status = "blocked"
        item.follow_up_block_reason = item.follow_up_customer_budget_reason or item.follow_up_budget_reason or "Autonomy budgets currently clamp follow-ups to manual mode."
        return

    try:
        launched = await _launch_follow_up_action(
            recommended_action,
            db=db,
            current_user=current_user,
        )
        item.follow_up_decision = "auto_launched"
        item.follow_up_launch_status = "launched"
        item.follow_up_job_id = launched.id
        if recommended_action.chain_create_payload:
            item.follow_up_chain_definition_id = AgentJobFromChainCreate.model_validate(
                recommended_action.chain_create_payload
            ).chain_definition_id
        else:
            item.follow_up_chain_definition_id = None
        item.follow_up_launched_at = datetime.utcnow()
        item.follow_up_block_reason = None
    except HTTPException as exc:
        item.follow_up_decision = "launch_failed"
        item.follow_up_launch_status = "failed"
        item.follow_up_block_reason = str(exc.detail)
    except Exception as exc:
        logger.warning(f"Failed to auto-launch follow-up for inbox item {item.id}: {exc}")
        item.follow_up_decision = "launch_failed"
        item.follow_up_launch_status = "failed"
        item.follow_up_block_reason = str(exc)[:500]


async def _perform_follow_up_queue_action(
    *,
    item: Optional[ResearchInboxItem] = None,
    portfolio: Optional[ResearchPortfolio] = None,
    portfolio_opportunity_id: Optional[str] = None,
    profile: Optional[DomainResearchProfile] = None,
    profile_opportunity_id: Optional[str] = None,
    action: str,
    operator_note: Optional[str],
    db: AsyncSession,
    current_user: User,
) -> AgentCheckpointQueueFollowUpActionResponse:
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in {"approve_launch", "reject_launch"}:
        raise HTTPException(status_code=400, detail="Unknown follow-up queue action")
    if sum(1 for value in (item is not None, portfolio is not None, profile is not None) if value) > 1:
        raise HTTPException(status_code=400, detail="Queue action target is ambiguous")
    if item is None and portfolio is None and profile is None:
        raise HTTPException(status_code=400, detail="Queue action target is required")
    if portfolio is not None:
        opportunity_id = str(portfolio_opportunity_id or "").strip()
        if not opportunity_id:
            raise HTTPException(status_code=400, detail="portfolio_opportunity_id is required")
        payload = _portfolio_summary_payload(portfolio)
        opportunities = payload["opportunities"]
        effective_policy = payload["effective_policy"]
        opportunity_index = next(
            (idx for idx, row in enumerate(opportunities) if str(row.get("opportunity_id") or "").strip() == opportunity_id),
            -1,
        )
        if opportunity_index < 0:
            raise HTTPException(status_code=404, detail="Portfolio opportunity not found")
        opportunity = dict(opportunities[opportunity_index])
        review = classify_portfolio_operator_review(opportunity, effective_policy=effective_policy)
        if not review or review.get("review_type") != "follow_up_recommendation":
            raise HTTPException(status_code=400, detail="Portfolio opportunity is not currently waiting for follow-up approval")
        acted_at = datetime.utcnow()
        opportunity["follow_up_reviewed_at"] = acted_at.isoformat()
        opportunity["follow_up_reviewed_by_user_id"] = str(current_user.id)
        opportunity["follow_up_review_note"] = (operator_note or "").strip() or None
        opportunity["updated_at"] = acted_at.isoformat()
        opportunity["decision_source"] = "operator"
        opportunity["operator_note"] = opportunity["follow_up_review_note"]
        opportunity["follow_up_review_evidence_revision"] = str(opportunity.get("evidence_revision") or review.get("evidence_revision") or "").strip() or None

        if normalized_action == "reject_launch":
            opportunity["follow_up_review_status"] = "rejected"
            opportunity["last_decision_type"] = "follow_up_rejected"
            opportunity["last_decision_reason_code"] = "operator_rejected_follow_up"
            opportunities[opportunity_index] = opportunity
            await _sync_portfolio_queue_state(portfolio=portfolio, opportunities=opportunities)
            return AgentCheckpointQueueFollowUpActionResponse(
                portfolio_id=portfolio.id,
                portfolio_opportunity_id=opportunity_id,
                follow_up_launch_status="rejected",
                follow_up_operator_decision="rejected",
                detail=opportunity["follow_up_review_note"] or "Operator rejected the queued follow-up launch.",
            )

        if opportunity.get("child_job_ids"):
            raise HTTPException(status_code=400, detail="Follow-up already launched for this opportunity")
        parent_job = await _resolve_portfolio_parent_job_for_queue(db=db, portfolio=portfolio)
        executor = AutonomousAgentExecutor()
        child_job = await executor._create_domain_research_follow_up_job(
            db=db,
            job=parent_job,
            domain=str(opportunity.get("title") or portfolio.title),
            objective=portfolio.objective,
            customer_context="research_portfolio",
            track_type=str(opportunity.get("track_type") or "generic"),
            source_scope="kb_plus_arxiv_plus_repo" if opportunity.get("source_repo_ids") else "kb_plus_arxiv",
            top_idea=opportunity,
            docs=[],
            repo_documents=[],
            papers=[],
            repo_source_ids=[str(v) for v in (opportunity.get("source_repo_ids") or []) if str(v).strip()],
            benchmark_queries=[],
            automation_profile=portfolio.automation_profile,
            automation_policy=effective_policy,
            sandbox_profile_id=portfolio.sandbox_profile_id,
        )
        if child_job is None:
            raise HTTPException(status_code=400, detail="Failed to launch follow-up job")
        opportunity["child_job_ids"] = list(
            dict.fromkeys([*([str(v) for v in (opportunity.get("child_job_ids") or []) if str(v).strip()]), str(child_job.id)])
        )[:8]
        opportunity["decision_state"] = "accepted"
        opportunity["stage"] = "validating"
        opportunity["follow_up_review_status"] = "approved_launch"
        opportunity["last_decision_type"] = "follow_up_approved_launch"
        opportunity["last_decision_reason_code"] = "operator_approved_follow_up"
        opportunities[opportunity_index] = opportunity
        await _sync_portfolio_queue_state(portfolio=portfolio, opportunities=opportunities)
        execute_agent_job_task.delay(str(child_job.id), str(portfolio.user_id))
        return AgentCheckpointQueueFollowUpActionResponse(
            portfolio_id=portfolio.id,
            portfolio_opportunity_id=opportunity_id,
            follow_up_launch_status="launched",
            follow_up_operator_decision="approved_launch",
            follow_up_job_id=child_job.id,
            detail="Follow-up launched from queue approval",
        )

    if profile is not None:
        opportunity_id = str(profile_opportunity_id or "").strip()
        if not opportunity_id:
            raise HTTPException(status_code=400, detail="profile_opportunity_id is required")
        payload = _profile_summary_payload(profile)
        opportunities = payload["opportunities"]
        effective_policy = payload["effective_policy"]
        opportunity_index = next(
            (idx for idx, row in enumerate(opportunities) if str(row.get("opportunity_id") or "").strip() == opportunity_id),
            -1,
        )
        if opportunity_index < 0:
            raise HTTPException(status_code=404, detail="Profile opportunity not found")
        opportunity = dict(opportunities[opportunity_index])
        review = classify_portfolio_operator_review(opportunity, effective_policy=effective_policy)
        if not review or review.get("review_type") != "follow_up_recommendation":
            raise HTTPException(status_code=400, detail="Profile opportunity is not currently waiting for follow-up approval")
        acted_at = datetime.utcnow()
        opportunity["follow_up_reviewed_at"] = acted_at.isoformat()
        opportunity["follow_up_reviewed_by_user_id"] = str(current_user.id)
        opportunity["follow_up_review_note"] = (operator_note or "").strip() or None
        opportunity["updated_at"] = acted_at.isoformat()
        opportunity["decision_source"] = "operator"
        opportunity["operator_note"] = opportunity["follow_up_review_note"]
        opportunity["follow_up_review_evidence_revision"] = str(opportunity.get("evidence_revision") or review.get("evidence_revision") or "").strip() or None

        if normalized_action == "reject_launch":
            opportunity["follow_up_review_status"] = "rejected"
            opportunity["last_decision_type"] = "follow_up_rejected"
            opportunity["last_decision_reason_code"] = "operator_rejected_follow_up"
            opportunities[opportunity_index] = opportunity
            await _sync_profile_queue_state(profile=profile, opportunities=opportunities)
            return AgentCheckpointQueueFollowUpActionResponse(
                domain_research_profile_id=profile.id,
                profile_opportunity_id=opportunity_id,
                follow_up_launch_status="rejected",
                follow_up_operator_decision="rejected",
                detail=opportunity["follow_up_review_note"] or "Operator rejected the queued follow-up launch.",
            )

        if opportunity.get("child_job_ids"):
            raise HTTPException(status_code=400, detail="Follow-up already launched for this opportunity")
        parent_job = await _resolve_profile_parent_job_for_queue(db=db, profile=profile)
        executor = AutonomousAgentExecutor()
        child_job = await executor._create_domain_research_follow_up_job(
            db=db,
            job=parent_job,
            domain=profile.domain,
            objective=profile.objective,
            customer_context=str(profile.customer_context or ""),
            track_type=str(profile.track_type or "generic"),
            source_scope=str(profile.source_scope or "kb_plus_arxiv"),
            top_idea=opportunity,
            docs=[],
            repo_documents=[],
            papers=[],
            repo_source_ids=[str(v) for v in (profile.repo_source_ids or []) if str(v).strip()],
            benchmark_queries=[str(v) for v in (profile.benchmark_queries or []) if str(v).strip()],
            automation_profile=profile.automation_profile,
            automation_policy=effective_policy,
            sandbox_profile_id=profile.sandbox_profile_id,
            profile_id=str(profile.id),
        )
        if child_job is None:
            raise HTTPException(status_code=400, detail="Failed to launch follow-up job")
        opportunity["child_job_ids"] = list(
            dict.fromkeys([*([str(v) for v in (opportunity.get("child_job_ids") or []) if str(v).strip()]), str(child_job.id)])
        )[:8]
        opportunity["decision_state"] = "accepted"
        opportunity["stage"] = "validating"
        opportunity["follow_up_review_status"] = "approved_launch"
        opportunity["last_decision_type"] = "follow_up_approved_launch"
        opportunity["last_decision_reason_code"] = "operator_approved_follow_up"
        opportunities[opportunity_index] = opportunity
        await _sync_profile_queue_state(profile=profile, opportunities=opportunities)
        execute_agent_job_task.delay(str(child_job.id), str(profile.user_id))
        return AgentCheckpointQueueFollowUpActionResponse(
            domain_research_profile_id=profile.id,
            profile_opportunity_id=opportunity_id,
            follow_up_launch_status="launched",
            follow_up_operator_decision="approved_launch",
            follow_up_job_id=child_job.id,
            detail="Follow-up launched from queue approval",
        )

    assert item is not None
    if str(item.status or "").strip().lower() != "accepted":
        raise HTTPException(status_code=400, detail="Only accepted inbox items support follow-up queue actions")
    if str(item.follow_up_launch_status or "").strip().lower() != "pending_approval":
        raise HTTPException(status_code=400, detail="Follow-up is not currently waiting for approval")
    if str(item.follow_up_operator_decision or "").strip():
        raise HTTPException(status_code=400, detail="Follow-up already has an operator decision")

    acted_at = datetime.utcnow()
    item.follow_up_operator_acted_at = acted_at
    item.follow_up_operator_user_id = current_user.id
    item.follow_up_operator_note = (operator_note or "").strip() or None

    if normalized_action == "reject_launch":
        item.follow_up_operator_decision = "rejected"
        item.follow_up_decision = "rejected"
        item.follow_up_launch_status = "rejected"
        item.follow_up_block_reason = item.follow_up_operator_note or "Operator rejected the queued follow-up launch."
        return AgentCheckpointQueueFollowUpActionResponse(
            inbox_item_id=item.id,
            follow_up_launch_status=item.follow_up_launch_status,
            follow_up_operator_decision=item.follow_up_operator_decision,
            detail=item.follow_up_block_reason,
        )

    learning_profile = await _load_follow_up_learning_profile(
        db=db,
        user_id=current_user.id,
        customer=(str(item.customer or "").strip() or None),
    )
    actions = _build_follow_up_actions_for_inbox_item(item, learning_profile=learning_profile)
    action_row = next(
        (
            row for row in actions
            if str(row.recommendation_key or "").strip() == str(item.follow_up_recommendation_key or "").strip()
        ),
        None,
    )
    if action_row is None:
        raise HTTPException(status_code=422, detail="Queued recommendation can no longer be resolved")
    if str(action_row.autonomy_eligibility or "").strip().lower() != "auto_launchable":
        raise HTTPException(status_code=422, detail="Queued recommendation is not safe to approve-launch")

    launched = await _launch_follow_up_action(action_row, db=db, current_user=current_user)
    item.follow_up_operator_decision = "approved_launch"
    item.follow_up_decision = "approved_and_launched"
    item.follow_up_launch_status = "launched"
    item.follow_up_job_id = launched.id
    if action_row.chain_create_payload:
        item.follow_up_chain_definition_id = AgentJobFromChainCreate.model_validate(
            action_row.chain_create_payload
        ).chain_definition_id
    else:
        item.follow_up_chain_definition_id = None
    item.follow_up_launched_at = acted_at
    item.follow_up_block_reason = None
    item.follow_up_outcome_status = None
    item.follow_up_outcome_recorded_at = None
    item.follow_up_outcome_summary = None
    return AgentCheckpointQueueFollowUpActionResponse(
        inbox_item_id=item.id,
        follow_up_launch_status=item.follow_up_launch_status,
        follow_up_operator_decision=item.follow_up_operator_decision,
        follow_up_job_id=item.follow_up_job_id,
        follow_up_chain_definition_id=item.follow_up_chain_definition_id,
        detail="Follow-up launched from queue approval",
    )


async def _relaunch_follow_up_inbox_item(
    *,
    item: ResearchInboxItem,
    operator_note: Optional[str],
    db: AsyncSession,
    current_user: User,
) -> AgentCheckpointQueueFollowUpActionResponse:
    if str(item.status or "").strip().lower() != "accepted":
        raise HTTPException(status_code=400, detail="Only accepted inbox items can relaunch a follow-up")

    outcome_status = str(item.follow_up_outcome_status or "").strip().lower()
    launch_status = str(item.follow_up_launch_status or "").strip().lower()
    if outcome_status not in {"failed", "cancelled"} or launch_status != "launched":
        raise HTTPException(status_code=400, detail="Only failed or cancelled launched follow-ups can be relaunched")

    learning_profile = await _load_follow_up_learning_profile(
        db=db,
        user_id=current_user.id,
        customer=(str(item.customer or "").strip() or None),
    )
    actions = _build_follow_up_actions_for_inbox_item(item, learning_profile=learning_profile)
    action_row = next(
        (
            row for row in actions
            if str(row.recommendation_key or "").strip() == str(item.follow_up_recommendation_key or "").strip()
        ),
        None,
    )
    if action_row is None:
        raise HTTPException(status_code=422, detail="Stored follow-up recommendation can no longer be resolved")
    if str(action_row.autonomy_eligibility or "").strip().lower() != "auto_launchable":
        raise HTTPException(status_code=422, detail="Stored follow-up recommendation is no longer safe to relaunch")

    launched = await _launch_follow_up_action(action_row, db=db, current_user=current_user)
    item.follow_up_decision = "relaunched"
    item.follow_up_launch_status = "launched"
    item.follow_up_job_id = launched.id
    item.follow_up_chain_definition_id = None
    if action_row.chain_create_payload:
        item.follow_up_chain_definition_id = AgentJobFromChainCreate.model_validate(
            action_row.chain_create_payload
        ).chain_definition_id
    launched_at = datetime.utcnow()
    item.follow_up_launched_at = launched_at
    item.follow_up_block_reason = (operator_note or "").strip() or None
    item.follow_up_budget_decision = None
    item.follow_up_budget_reason = None
    item.follow_up_budget_throttle_state = None
    item.follow_up_customer_budget_decision = None
    item.follow_up_customer_budget_reason = None
    item.follow_up_customer_budget_throttle_state = None
    item.follow_up_outcome_status = None
    item.follow_up_outcome_recorded_at = None
    item.follow_up_outcome_summary = None
    await project_follow_up_relaunch_to_originating_opportunity(
        db=db,
        job=launched,
        launched_at=launched_at,
    )
    return AgentCheckpointQueueFollowUpActionResponse(
        inbox_item_id=item.id,
        follow_up_launch_status=item.follow_up_launch_status,
        follow_up_operator_decision=item.follow_up_operator_decision,
        follow_up_job_id=item.follow_up_job_id,
        follow_up_chain_definition_id=item.follow_up_chain_definition_id,
        detail="Follow-up relaunched",
    )


async def _record_follow_up_queue_decision_event(
    *,
    db: AsyncSession,
    current_user: User,
    action: str,
    operator_note: Optional[str],
    source_kind: str,
    source_id: str,
    source_label: str,
    customer: Optional[str],
    reason_code: Optional[str],
    reason_label: Optional[str],
    scheduler_state: Optional[dict[str, Any]],
    follow_up_launch_status: Optional[str],
    follow_up_operator_decision: Optional[str],
    deep_link: dict[str, Any],
    metadata: dict[str, Any],
    after_state: dict[str, Any],
) -> None:
    normalized_action = str(action or "").strip().lower()
    normalized_scheduler_state = (
        {
            key: value
            for key, value in (scheduler_state or {}).items()
            if value not in (None, "", 0)
        }
        if isinstance(scheduler_state, dict)
        else None
    )
    await record_autonomy_decision_event(
        db,
        user_id=current_user.id,
        event_type="follow_up_approved" if normalized_action == "approve_launch" else "follow_up_rejected",
        event_time=datetime.utcnow(),
        source_kind=source_kind,
        source_id=source_id,
        source_label=source_label,
        customer=customer,
        decision_type="follow_up_approved" if normalized_action == "approve_launch" else "follow_up_rejected",
        reason_code=reason_code,
        status=str(follow_up_launch_status or "").strip() or None,
        severity="medium",
        actor_mode="operator",
        summary=f"{source_label}: {'approved' if normalized_action == 'approve_launch' else 'rejected'} queued follow-up",
        operator_note=operator_note,
        reason_label=reason_label,
        scheduler_state=normalized_scheduler_state,
        after_state=after_state,
        deep_link=deep_link,
        metadata=metadata,
    )


    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _build_follow_up_actions_for_inbox_item(
    item: ResearchInboxItem,
    *,
    learning_profile: Optional[dict[str, Any]] = None,
) -> list[AgentCheckpointQueueActionResponse]:
    """Return bounded follow-up launch recommendations for accepted inbox items."""
    title = str(item.title or "").strip() or "Selected research signal"
    summary = str(item.summary or "").strip()
    customer = str(item.customer or "").strip()
    customer_hint = f"Customer: {customer}" if customer else ""
    inbox_item_payload = {
        "id": str(item.id),
        "item_type": str(item.item_type or "").strip(),
        "item_key": str(item.item_key or "").strip(),
        "title": title,
        "url": str(item.url or "").strip() or None,
        "summary": summary or None,
        "customer": customer or None,
    }
    top_documents = []
    top_papers = []
    if str(item.item_type or "").strip().lower() == "arxiv":
        top_papers.append(
            {
                "id": str(item.item_key),
                "title": title,
                "url": str(item.url or "").strip() or None,
                "score": None,
                "source": "inbox",
            }
        )
    else:
        top_documents.append(
            {
                "id": str(item.item_key),
                "title": title,
                "url": str(item.url or "").strip() or None,
                "score": None,
                "source": "inbox",
            }
        )

    inherited_data = {
        "parent_results": {
            "summary": f"Seeded from accepted Research Inbox item: {title}",
            "research_bundle": {
                "top_documents": top_documents,
                "top_papers": top_papers,
                "insights": [],
                "next_steps": [],
                "artifacts": [],
            },
            "inbox_items": [inbox_item_payload],
        },
        "parent_findings": [
            {
                "type": "paper" if top_papers else "document",
                "title": title,
                "id": str(item.item_key),
                "url": str(item.url or "").strip() or None,
                "snippet": summary or None,
            }
        ],
    }

    actions = [
        AgentCheckpointQueueActionResponse(
            kind="launch_chain",
            label="Launch Deep Dive",
            description="Start the recommended scout-to-deep-dive chain with this accepted signal preloaded.",
            recommended=True,
            launch_label="Deep Dive Chain",
            recommendation_key=FOLLOW_UP_RECOMMENDATION_DEEP_DIVE_CHAIN,
            autonomy_eligibility="auto_launchable",
            chain_create_payload={
                "chain_definition_id": str(CUSTOMER_RESEARCH_SCOUT_DEEP_DIVE_CHAIN_ID),
                "name_prefix": f"Inbox Research - {datetime.utcnow().strftime('%Y-%m-%d')}",
                "variables": {"goal": f"Deep-dive on {title} and propose concrete next steps."},
                "config_overrides": {
                    "customer_context": customer_hint,
                    "prefer_sources": ["documents", "arxiv"],
                    "max_documents": 12,
                    "max_papers": 8,
                    "persist_artifacts": False,
                    "reading_list_name": "Customer Research",
                    "inherited_data": inherited_data,
                },
                "start_immediately": True,
            },
        ),
        AgentCheckpointQueueActionResponse(
            kind="launch_job",
            label="Launch Single Research Job",
            description="Create one bounded research job instead of a full chain.",
            launch_label="Single Research Job",
            recommendation_key=FOLLOW_UP_RECOMMENDATION_SINGLE_RESEARCH_JOB,
            autonomy_eligibility="auto_launchable",
            job_create_payload={
                "name": f"Inbox Research - {datetime.utcnow().strftime('%Y-%m-%d')}",
                "job_type": "research",
                "goal": f"Deep-dive on {title} and propose concrete next steps.",
                "config": {
                    "customer_context": customer_hint,
                    "prefer_sources": ["documents", "arxiv"],
                    "max_documents": 12,
                    "max_papers": 8,
                    "persist_artifacts": False,
                    "reading_list_name": "Customer Research",
                    "inherited_data": inherited_data,
                },
                "start_immediately": True,
            },
        ),
    ]

    metadata = item.item_metadata if isinstance(item.item_metadata, dict) else {}
    repos = metadata.get("repos") if isinstance(metadata.get("repos"), list) else []
    if str(item.item_type or "").strip().lower() == "arxiv" and repos:
        actions.append(
            AgentCheckpointQueueActionResponse(
                kind="launch_chain",
                label="Launch Repo -> Patch Chain",
                description="Use extracted repository links to move from paper to repo ingest and a patch proposal.",
                launch_label="Repo -> Patch Chain",
                recommendation_key=FOLLOW_UP_RECOMMENDATION_REPO_PATCH_CHAIN,
                autonomy_eligibility="manual_only",
                chain_create_payload={
                    "chain_definition_id": str(ARXIV_REPO_CODE_PATCH_CHAIN_ID),
                    "name_prefix": f"Paper Repo Patch - {datetime.utcnow().strftime('%Y-%m-%d')}",
                    "variables": {"goal": f"Implement the most relevant change suggested by {title}"},
                    "config_overrides": {
                        "inbox_item_id": str(item.id),
                        "customer_context": customer_hint,
                    },
                    "start_immediately": True,
                },
            )
        )
    for action in actions:
        score, reasons = _score_follow_up_action_for_item(
            item,
            action,
            learning_profile=learning_profile,
        )
        action.recommendation_score = score
        action.recommendation_reasons = reasons
        action.recommended = False
    actions.sort(
        key=lambda action: (
            1 if str(action.autonomy_eligibility or "").strip().lower() == "manual_only" else 0,
            -int(action.recommendation_score or 0),
            0 if str(action.recommendation_key or "") == FOLLOW_UP_RECOMMENDATION_DEEP_DIVE_CHAIN else 1,
        )
    )
    if actions:
        actions[0].recommended = True
    return actions


def _build_checkpoint_queue_items(
    jobs: list[AgentJob],
    inbox_items: list[ResearchInboxItem],
    portfolios: Optional[list[ResearchPortfolio]] = None,
    profiles: Optional[list[DomainResearchProfile]] = None,
    *,
    learning_profiles: Optional[dict[str, dict[str, Any]]] = None,
    monitor_health_rows: Optional[list[dict[str, Any]]] = None,
) -> list[AgentCheckpointQueueItemResponse]:
    """Project approvals, recoveries, and accepted-inbox follow-ups into one queue."""
    items: list[AgentCheckpointQueueItemResponse] = []
    now = datetime.utcnow()

    for job in jobs:
        checkpoint = _extract_approval_checkpoint(job)
        scheduler_state = _extract_scheduler_state(job)
        customer = _queue_customer_for_job(job)
        job_response = _job_to_response(job)
        if checkpoint:
            created_at = job.last_activity_at or job.completed_at or job.started_at or job.created_at
            urgency = _queue_priority_fields(
                item_type="approval_checkpoint",
                reason_code="approval_required",
                created_at=created_at,
                next_run_at=job.next_run_at,
                backoff_until=None,
                stale=False,
                now=now,
            )
            action_rows = [
                AgentCheckpointQueueActionResponse(kind="job_action", label="Approve", action="approve", recommended=True),
                AgentCheckpointQueueActionResponse(kind="job_action", label="Edit + Approve", action="edit"),
                AgentCheckpointQueueActionResponse(kind="job_action", label="Reject", action="reject"),
                AgentCheckpointQueueActionResponse(kind="job_action", label="Skip Step", action="skip"),
            ]
            items.append(
                AgentCheckpointQueueItemResponse(
                    queue_key=f"approval:{job.id}",
                    item_type="approval_checkpoint",
                    priority=100,
                    title=job.name,
                    summary=str(checkpoint.get("message") or job.goal or "").strip()[:320] or None,
                    evidence_summary=_queue_evidence_summary_for_job(job),
                    status=job.status,
                    customer=customer,
                    job_name=job.name,
                    job_type=str(job.job_type or "").strip() or None,
                    reason_code="approval_required",
                    reason_label=_queue_reason_label("approval_required"),
                    recommended_action="approve",
                    priority_score=urgency["priority_score"],
                    age_minutes=urgency["age_minutes"],
                    sla_bucket=urgency["sla_bucket"],
                    escalation_level=urgency["escalation_level"],
                    is_overdue=urgency["is_overdue"],
                    is_stale=urgency["is_stale"],
                    next_run_at=job.next_run_at,
                    backoff_until=None,
                    action_count=len(action_rows),
                    created_at=created_at,
                    job_id=job.id,
                    job=job_response,
                    checkpoint=checkpoint,
                    scheduler_state=scheduler_state,
                    actions=action_rows,
                )
            )
            continue

        is_recurring = str(job.schedule_type or "").strip().lower() in {"recurring", "continuous"}
        failed_or_paused = str(job.status or "").strip().lower() in {
            AgentJobStatus.FAILED.value,
            AgentJobStatus.PAUSED.value,
        }
        stale_running = (
            str(job.status or "").strip().lower() == AgentJobStatus.RUNNING.value
            and job.last_activity_at is not None
            and (now - job.last_activity_at) > timedelta(minutes=30)
        )
        if is_recurring and (failed_or_paused or stale_running):
            reason = (
                str((scheduler_state or {}).get("queue_reason") or "").strip()
                or ("stalled_run" if stale_running else "scheduled_recovery")
            )
            created_at = job.last_activity_at or job.completed_at or job.started_at or job.created_at
            backoff_until = _parse_optional_datetime((scheduler_state or {}).get("backoff_until"))
            urgency = _queue_priority_fields(
                item_type="job_recovery",
                reason_code=reason,
                created_at=created_at,
                next_run_at=job.next_run_at,
                backoff_until=backoff_until,
                stale=stale_running,
                now=now,
            )
            launch_mode = _extract_launch_mode(job.config if isinstance(job.config, dict) else None)
            is_repo_bug_triage = launch_mode == "quick_start_repo_bug_triage"
            action_rows = [
                AgentCheckpointQueueActionResponse(
                    kind="job_action",
                    label="Retry with refined plan" if is_repo_bug_triage else "Restart",
                    action="restart",
                    recommended=True,
                ),
                AgentCheckpointQueueActionResponse(
                    kind="job_action",
                    label="Resume verification" if is_repo_bug_triage else "Resume",
                    action="resume",
                ),
                AgentCheckpointQueueActionResponse(kind="job_action", label="Cancel", action="cancel"),
            ]
            items.append(
                AgentCheckpointQueueItemResponse(
                    queue_key=f"recovery:{job.id}",
                    item_type="job_recovery",
                    priority=80,
                    title=job.name,
                    summary=(job.error or job.phase_details or f"Recurring job requires operator recovery ({reason}).")[:320],
                    evidence_summary=_queue_evidence_summary_for_job(job),
                    status=job.status,
                    customer=customer,
                    job_name=job.name,
                    job_type=str(job.job_type or "").strip() or None,
                    reason_code=reason,
                    reason_label=_queue_reason_label(reason),
                    recommended_action=(
                        "restart" if reason in {"execution_failure", "stalled_run", "scheduled_recovery"} else "resume"
                    ),
                    priority_score=urgency["priority_score"],
                    age_minutes=urgency["age_minutes"],
                    sla_bucket=urgency["sla_bucket"],
                    escalation_level=urgency["escalation_level"],
                    is_overdue=urgency["is_overdue"],
                    is_stale=urgency["is_stale"],
                    next_run_at=job.next_run_at,
                    backoff_until=backoff_until,
                    action_count=len(action_rows),
                    created_at=created_at,
                    job_id=job.id,
                    job=job_response,
                    scheduler_state={**(scheduler_state or {}), "queue_reason": reason},
                    actions=action_rows,
                )
            )

    for monitor in monitor_health_rows or []:
        monitor_job_id = monitor.get("monitor_job_id")
        if not monitor_job_id:
            continue
        job = next((candidate for candidate in jobs if candidate.id == monitor_job_id), None)
        if job is None:
            continue
        customer = str(monitor.get("customer") or "").strip() or _queue_customer_for_job(job)
        created_at = monitor.get("latest_policy_changed_at") or job.last_activity_at or job.completed_at or job.started_at or job.created_at
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except Exception:
                created_at = job.last_activity_at or job.completed_at or job.started_at or job.created_at
        guardrail_status = str(monitor.get("policy_guardrail_status") or "").strip().lower()
        guardrail_action = str(monitor.get("policy_guardrail_action") or "").strip().lower()
        guardrail_target_policy = monitor.get("policy_guardrail_target_policy")
        guardrail_compat_fields = build_monitor_policy_compat_fields(
            automation_profile=monitor.get("automation_profile") or monitor.get("autonomy_mode"),
            automation_policy=monitor.get("automation_policy"),
            effective_policy=monitor.get("effective_policy"),
            default_allowed=list(research_monitor_profile_service.SAFE_AUTONOMY_RECOMMENDATIONS),
            target_policy=guardrail_target_policy,
        )
        guardrail_policy = guardrail_compat_fields.get("policy_guardrail_follow_up_autonomy")
        history_entry_id = str(monitor.get("policy_guardrail_target_history_entry_id") or "").strip() or None
        if guardrail_status == "active":
            urgency = _queue_priority_fields(
                item_type="policy_review",
                reason_code="policy_guardrail",
                created_at=created_at,
                next_run_at=None,
                backoff_until=None,
                stale=False,
                now=now,
            )
            actions = [
                AgentCheckpointQueueActionResponse(
                    kind="policy_action",
                    label="Apply safeguard",
                    action="apply_guardrail",
                    recommended=True,
                    description="Apply the recommended rollback or downgrade for this degrading monitor policy.",
                    policy_rollback_payload=({"history_entry_id": history_entry_id} if guardrail_action == "rollback" and history_entry_id else None),
                    policy_update_payload=(
                        {
                            "automation_profile": str(monitor.get("automation_profile") or monitor.get("autonomy_mode") or "balanced").strip() or "balanced",
                            "automation_policy": (
                                {
                                    "follow_up_review_mode": str((guardrail_target_policy or {}).get("follow_up_review_mode") or "").strip() or None,
                                    "allowed_recommendations": list((guardrail_target_policy or {}).get("allowed_recommendations") or []),
                                }
                                if isinstance(guardrail_target_policy, dict)
                                else None
                            ),
                            "mode": str((guardrail_policy or {}).get("mode") or "").strip() or None,
                            "allowed_recommendations": list((guardrail_policy or {}).get("allowed_recommendations") or []),
                            "change_source": "policy_guardrail",
                        }
                        if guardrail_action == "downgrade" and isinstance(guardrail_policy, dict)
                        else None
                    ),
                ),
                AgentCheckpointQueueActionResponse(
                    kind="policy_action",
                    label="Compare Before/After",
                    action="compare_before_after",
                    description="Open the latest policy comparison for this degrading rollout.",
                    policy_rollback_payload=({"history_entry_id": history_entry_id} if history_entry_id else None),
                ),
            ]
            items.append(
                AgentCheckpointQueueItemResponse(
                    queue_key=f"policy_review:{monitor_job_id}:{history_entry_id or 'current'}",
                    item_type="policy_review",
                    priority=90,
                    title=str(monitor.get("monitor_name") or job.name or "Monitor policy review").strip(),
                    summary=(
                        f"Latest policy evaluation is degrading. Suggested safeguard: "
                        f"{'roll back to the previous policy' if guardrail_action == 'rollback' else 'downgrade autonomy mode'}."
                    ),
                    evidence_summary=" · ".join(
                        [str(reason).strip() for reason in (monitor.get("policy_guardrail_reasons") or []) if str(reason).strip()]
                    )[:320] or None,
                    status=str(job.status or "").strip() or None,
                    customer=customer or None,
                    job_name=str(job.name or "").strip() or None,
                    job_type=str(job.job_type or "").strip() or None,
                    reason_code="policy_guardrail",
                    reason_label="Policy safeguard review",
                    recommended_action="apply_guardrail",
                    priority_score=urgency["priority_score"],
                    age_minutes=urgency["age_minutes"],
                    sla_bucket=urgency["sla_bucket"],
                    escalation_level=urgency["escalation_level"],
                    is_overdue=urgency["is_overdue"],
                    is_stale=urgency["is_stale"],
                    action_count=len(actions),
                    created_at=created_at,
                    job_id=job.id,
                    job=_job_to_response(job),
                    policy_guardrail_status=guardrail_status,
                    policy_guardrail_action=guardrail_action or None,
                    policy_guardrail_target_history_entry_id=history_entry_id,
                    policy_guardrail_reasons=list(monitor.get("policy_guardrail_reasons") or []),
                    policy_guardrail_target_policy=(guardrail_target_policy if isinstance(guardrail_target_policy, dict) else None),
                    policy_guardrail_follow_up_autonomy=(guardrail_policy if isinstance(guardrail_policy, dict) else None),
                    actions=actions,
                )
            )
        budget_state = str(monitor.get("budget_throttle_state") or "").strip().lower()
        if budget_state != "normal":
            budget_reasons = [str(reason).strip() for reason in (monitor.get("budget_throttle_reasons") or []) if str(reason).strip()]
            budget_created_at = created_at
            budget_urgency = _queue_priority_fields(
                item_type="budget_review",
                reason_code="budget_throttle",
                created_at=budget_created_at,
                next_run_at=None,
                backoff_until=None,
                stale=False,
                now=now,
            )
            items.append(
                AgentCheckpointQueueItemResponse(
                    queue_key=f"budget_review:{monitor_job_id}:{budget_state}",
                    item_type="budget_review",
                    priority=70,
                    title=str(monitor.get("monitor_name") or job.name or "Monitor budget review").strip(),
                    summary=(
                        f"Autonomy is temporarily throttled to {budget_state.replace('_', ' ')} for this monitor."
                    ),
                    evidence_summary=" · ".join(budget_reasons[:3])[:320] or None,
                    status=str(job.status or "").strip() or None,
                    customer=customer or None,
                    job_name=str(job.name or "").strip() or None,
                    job_type=str(job.job_type or "").strip() or None,
                    reason_code="budget_throttle",
                    reason_label="Autonomy budget review",
                    recommended_action="open_monitor",
                    priority_score=budget_urgency["priority_score"],
                    age_minutes=budget_urgency["age_minutes"],
                    sla_bucket=budget_urgency["sla_bucket"],
                    escalation_level=budget_urgency["escalation_level"],
                    is_overdue=budget_urgency["is_overdue"],
                    is_stale=budget_urgency["is_stale"],
                    action_count=1,
                    created_at=budget_created_at,
                    job_id=job.id,
                    job=_job_to_response(job),
                    budget_throttle_state=budget_state,
                    budget_reason="; ".join(budget_reasons[:3]) or None,
                    actions=[
                        AgentCheckpointQueueActionResponse(
                            kind="policy_action",
                            label="Open Monitor",
                            action="open_monitor",
                            description="Open this monitor in Autonomy Health to inspect budget pressure and adjust limits.",
                        )
                    ],
                )
            )

    for item in inbox_items:
        launch_status = str(item.follow_up_launch_status or "").strip().lower()
        operator_decision = str(item.follow_up_operator_decision or "").strip().lower()
        if launch_status == "launched" or operator_decision == "rejected" or launch_status == "rejected":
            continue
        actions = _build_follow_up_actions_for_inbox_item(
            item,
            learning_profile=(learning_profiles or {}).get(_customer_profile_key(item.customer)),
        )
        created_at = item.updated_at or item.discovered_at
        follow_up_decision = str(item.follow_up_decision or "").strip() or None
        follow_up_policy_mode = str(item.follow_up_policy_mode or "").strip() or None
        follow_up_launch_status = str(item.follow_up_launch_status or "").strip() or None
        follow_up_block_reason = str(item.follow_up_block_reason or "").strip() or None
        follow_up_reason_code = "accepted_inbox_item"
        follow_up_reason_label = _queue_reason_label("accepted_inbox_item")
        if follow_up_decision == "queued_for_approval":
            follow_up_reason_code = "follow_up_launch_approval"
            follow_up_reason_label = "Follow-up launch approval"
        elif follow_up_launch_status == "blocked":
            follow_up_reason_code = "follow_up_blocked"
            follow_up_reason_label = "Follow-up blocked by policy"
        elif follow_up_launch_status == "failed":
            follow_up_reason_code = "follow_up_launch_failed"
            follow_up_reason_label = "Follow-up launch failed"
        urgency = _queue_priority_fields(
            item_type="follow_up_recommendation",
            reason_code=follow_up_reason_code,
            created_at=created_at,
            next_run_at=None,
            backoff_until=None,
            stale=False,
            now=now,
        )
        if follow_up_launch_status == "pending_approval":
            actions = [
                AgentCheckpointQueueActionResponse(
                    kind="follow_up_action",
                    label="Approve & Launch",
                    action="approve_launch",
                    description="Approve this bounded safe follow-up and launch it immediately.",
                    recommended=True,
                    recommendation_key=(str(item.follow_up_recommendation_key or "").strip() or None),
                    follow_up_action_payload={"inbox_item_id": str(item.id)},
                ),
                AgentCheckpointQueueActionResponse(
                    kind="follow_up_action",
                    label="Reject Launch",
                    action="reject_launch",
                    description="Reject this queued safe follow-up without creating a downstream job.",
                    recommendation_key=(str(item.follow_up_recommendation_key or "").strip() or None),
                    follow_up_action_payload={"inbox_item_id": str(item.id)},
                ),
            ]
        items.append(
            AgentCheckpointQueueItemResponse(
                queue_key=f"followup:{item.id}",
                item_type="follow_up_recommendation",
                priority=60,
                title=item.title,
                summary=(
                    follow_up_block_reason
                    or item.summary
                    or f"Accepted {item.item_type} signal ready for follow-up."
                )[:320],
                evidence_summary=(item.summary or item.url or item.title)[:320],
                status=item.status,
                customer=(str(item.customer or "").strip() or None),
                job_name=None,
                job_type="research",
                reason_code=follow_up_reason_code,
                reason_label=follow_up_reason_label,
                recommended_action=next((action.action or action.launch_label or action.label for action in actions if action.recommended), None) or (actions[0].launch_label if actions else None),
                priority_score=urgency["priority_score"],
                age_minutes=urgency["age_minutes"],
                sla_bucket=urgency["sla_bucket"],
                escalation_level=urgency["escalation_level"],
                is_overdue=urgency["is_overdue"],
                is_stale=urgency["is_stale"],
                next_run_at=None,
                backoff_until=None,
                action_count=len(actions),
                created_at=created_at,
                inbox_item_id=item.id,
                inbox_item={
                    "id": str(item.id),
                    "item_type": item.item_type,
                    "item_key": item.item_key,
                    "title": item.title,
                    "summary": item.summary,
                    "url": item.url,
                    "customer": item.customer,
                },
                follow_up_decision=follow_up_decision,
                follow_up_policy_mode=follow_up_policy_mode,
                follow_up_launch_status=follow_up_launch_status,
                follow_up_block_reason=follow_up_block_reason,
                follow_up_budget_decision=(str(item.follow_up_budget_decision or "").strip() or None),
                follow_up_budget_reason=(str(item.follow_up_budget_reason or "").strip() or None),
                follow_up_budget_throttle_state=(str(item.follow_up_budget_throttle_state or "").strip() or None),
                follow_up_customer_budget_decision=(str(item.follow_up_customer_budget_decision or "").strip() or None),
                follow_up_customer_budget_reason=(str(item.follow_up_customer_budget_reason or "").strip() or None),
                follow_up_customer_budget_throttle_state=(str(item.follow_up_customer_budget_throttle_state or "").strip() or None),
                follow_up_recommendation_key=(str(item.follow_up_recommendation_key or "").strip() or None),
                follow_up_job_id=item.follow_up_job_id,
                follow_up_chain_definition_id=item.follow_up_chain_definition_id,
                follow_up_operator_decision=(str(item.follow_up_operator_decision or "").strip() or None),
                follow_up_operator_note=(str(item.follow_up_operator_note or "").strip() or None),
                follow_up_operator_acted_at=item.follow_up_operator_acted_at,
                follow_up_operator_user_id=item.follow_up_operator_user_id,
                budget_throttle_state=(str(item.follow_up_budget_throttle_state or "").strip() or None),
                budget_reason=(str(item.follow_up_budget_reason or "").strip() or None),
                customer_budget_throttle_state=(str(item.follow_up_customer_budget_throttle_state or "").strip() or None),
                customer_budget_reason=(str(item.follow_up_customer_budget_reason or "").strip() or None),
                actions=actions,
            )
        )

    for portfolio in portfolios or []:
        payload = _portfolio_summary_payload(portfolio)
        opportunities = payload["opportunities"]
        effective_policy = payload["effective_policy"]
        for opportunity in opportunities:
            review = classify_portfolio_operator_review(opportunity, effective_policy=effective_policy)
            if not review:
                continue
            created_at = None
            if str(opportunity.get("follow_up_reviewed_at") or "").strip():
                created_at = _parse_optional_datetime(opportunity.get("follow_up_reviewed_at"))
            if created_at is None:
                created_at = _parse_optional_datetime(opportunity.get("updated_at")) or portfolio.updated_at or portfolio.created_at
            review_type = str(review.get("review_type") or "").strip()
            reason_code = str(review.get("reason_code") or "").strip()
            urgency = _queue_priority_fields(
                item_type=review_type,
                reason_code=reason_code,
                created_at=created_at,
                next_run_at=None,
                backoff_until=None,
                stale=False,
                now=now,
            )
            queue_key = (
                f"portfolio:{review_type}:{portfolio.id}:"
                f"{str(opportunity.get('opportunity_id') or '').strip()}:"
                f"{str(review.get('evidence_revision') or '').strip()}:"
                f"{str((payload['summary'] or {}).get('portfolio_config_revision') or '').strip() or 'current'}"
            )
            common = dict(
                queue_key=queue_key,
                item_type=review_type,
                priority=90 if review_type == "policy_review" else 70 if review_type == "budget_review" else 60,
                title=str(opportunity.get("title") or portfolio.title or "Research fleet review").strip(),
                summary=(
                    str(opportunity.get("follow_up_review_note") or "").strip()
                    or str(opportunity.get("operator_note") or "").strip()
                    or (
                        "Queued follow-up is ready for approval."
                        if review_type == "follow_up_recommendation"
                        else "Open the fleet card to review this blocked opportunity."
                    )
                )[:320],
                evidence_summary=" · ".join(
                    [str(row).strip() for row in (opportunity.get("supporting_evidence") or []) if str(row).strip()]
                )[:320] or str(opportunity.get("hypothesis") or "").strip()[:320] or None,
                status=str(portfolio.status or "").strip() or None,
                customer=None,
                job_name=str(portfolio.title or "").strip() or None,
                job_type="research",
                reason_code=reason_code,
                reason_label=str(review.get("reason_label") or _queue_reason_label(reason_code)),
                priority_score=urgency["priority_score"],
                age_minutes=urgency["age_minutes"],
                sla_bucket=urgency["sla_bucket"],
                escalation_level=urgency["escalation_level"],
                is_overdue=urgency["is_overdue"],
                is_stale=urgency["is_stale"],
                created_at=created_at,
                job_id=portfolio.active_job_id or portfolio.latest_run_job_id,
                portfolio_id=portfolio.id,
                portfolio_title=str(portfolio.title or "").strip() or None,
                portfolio_opportunity_id=str(opportunity.get("opportunity_id") or "").strip() or None,
                portfolio_opportunity_key=str(opportunity.get("canonical_key") or "").strip() or None,
                follow_up_operator_note=str(opportunity.get("follow_up_review_note") or "").strip() or None,
                **_build_operator_queue_context(
                    objective=str(portfolio.objective or "").strip() or None,
                    domain=None,
                    track_type=str(opportunity.get("track_type") or "generic").strip() or None,
                    source_scope="kb_plus_arxiv_plus_repo" if _clean_queue_text_list(opportunity.get("source_repo_ids")) else "kb_plus_arxiv",
                    repo_source_ids=opportunity.get("source_repo_ids"),
                    benchmark_queries=None,
                    sandbox_profile_id=str(portfolio.sandbox_profile_id or "").strip() or None,
                    automation_profile=payload["automation_profile"],
                    effective_policy=effective_policy,
                    confidence=opportunity.get("confidence"),
                    readiness=opportunity.get("readiness"),
                    linked_note_ids=portfolio.latest_note_ids,
                    linked_experiment_plan_ids=opportunity.get("linked_experiment_plan_ids") or portfolio.latest_experiment_plan_ids,
                    linked_validation_run_ids=opportunity.get("linked_validation_run_ids") or portfolio.latest_validation_run_ids,
                    child_job_ids=opportunity.get("child_job_ids") or portfolio.child_job_ids,
                ),
            )
            if review_type == "follow_up_recommendation":
                actions = [
                    AgentCheckpointQueueActionResponse(
                        kind="follow_up_action",
                        label="Approve & Launch",
                        action="approve_launch",
                        description="Approve this bounded fleet follow-up and launch it immediately.",
                        recommended=True,
                        recommendation_key="portfolio_follow_up",
                        follow_up_action_payload={
                            "portfolio_id": str(portfolio.id),
                            "portfolio_opportunity_id": str(opportunity.get("opportunity_id") or "").strip(),
                        },
                    ),
                    AgentCheckpointQueueActionResponse(
                        kind="follow_up_action",
                        label="Reject Launch",
                        action="reject_launch",
                        description="Reject this queued fleet follow-up for the current evidence revision.",
                        recommendation_key="portfolio_follow_up",
                        follow_up_action_payload={
                            "portfolio_id": str(portfolio.id),
                            "portfolio_opportunity_id": str(opportunity.get("opportunity_id") or "").strip(),
                        },
                    ),
                ]
                items.append(
                    AgentCheckpointQueueItemResponse(
                        **common,
                        recommended_action="approve_launch",
                        action_count=len(actions),
                        follow_up_launch_status="pending_approval",
                        follow_up_policy_mode=str(effective_policy.get("follow_up_review_mode") or "").strip() or None,
                        follow_up_operator_decision=str(opportunity.get("follow_up_review_status") or "").strip() or None,
                        actions=actions,
                    )
                )
                continue

            actions = [
                AgentCheckpointQueueActionResponse(
                    kind="policy_action",
                    label="Open Fleet",
                    action="open_fleet",
                    description="Open this research fleet and inspect the targeted opportunity.",
                )
            ]
            items.append(
                AgentCheckpointQueueItemResponse(
                    **common,
                    recommended_action="open_fleet",
                    action_count=1,
                    budget_reason=(reason_code if review_type == "budget_review" else None),
                    actions=actions,
                )
            )

    for profile in profiles or []:
        payload = _profile_summary_payload(profile)
        opportunities = payload["opportunities"]
        effective_policy = payload["effective_policy"]
        for opportunity in opportunities:
            review = classify_portfolio_operator_review(opportunity, effective_policy=effective_policy)
            if not review:
                continue
            created_at = None
            if str(opportunity.get("follow_up_reviewed_at") or "").strip():
                created_at = _parse_optional_datetime(opportunity.get("follow_up_reviewed_at"))
            if created_at is None:
                created_at = _parse_optional_datetime(opportunity.get("updated_at")) or profile.updated_at or profile.created_at
            review_type = str(review.get("review_type") or "").strip()
            reason_code = str(review.get("reason_code") or "").strip()
            urgency = _queue_priority_fields(
                item_type=review_type,
                reason_code=reason_code,
                created_at=created_at,
                next_run_at=None,
                backoff_until=None,
                stale=False,
                now=now,
            )
            queue_key = (
                f"profile:{review_type}:{profile.id}:"
                f"{str(opportunity.get('opportunity_id') or '').strip()}:"
                f"{str(review.get('evidence_revision') or '').strip()}:"
                f"{str((payload['summary'] or {}).get('profile_config_revision') or '').strip() or 'current'}"
            )
            common = dict(
                queue_key=queue_key,
                item_type=review_type,
                priority=90 if review_type == "policy_review" else 70 if review_type == "budget_review" else 60,
                title=str(opportunity.get("title") or profile.title or "Domain profile review").strip(),
                summary=(
                    str(opportunity.get("follow_up_review_note") or "").strip()
                    or str(opportunity.get("operator_note") or "").strip()
                    or (
                        "Queued follow-up is ready for approval."
                        if review_type == "follow_up_recommendation"
                        else "Open the domain profile card to review this blocked opportunity."
                    )
                )[:320],
                evidence_summary=" · ".join(
                    [str(row).strip() for row in (opportunity.get("supporting_evidence") or []) if str(row).strip()]
                )[:320] or str(opportunity.get("hypothesis") or "").strip()[:320] or None,
                status=str(profile.status or "").strip() or None,
                customer=None,
                job_name=str(profile.title or "").strip() or None,
                job_type="research",
                reason_code=reason_code,
                reason_label=str(review.get("reason_label") or _queue_reason_label(reason_code)),
                priority_score=urgency["priority_score"],
                age_minutes=urgency["age_minutes"],
                sla_bucket=urgency["sla_bucket"],
                escalation_level=urgency["escalation_level"],
                is_overdue=urgency["is_overdue"],
                is_stale=urgency["is_stale"],
                created_at=created_at,
                job_id=profile.active_job_id or profile.latest_run_job_id,
                domain_research_profile_id=profile.id,
                domain_research_profile_title=str(profile.title or "").strip() or None,
                profile_opportunity_id=str(opportunity.get("opportunity_id") or "").strip() or None,
                profile_opportunity_key=str(opportunity.get("canonical_key") or "").strip() or None,
                follow_up_operator_note=str(opportunity.get("follow_up_review_note") or "").strip() or None,
                **_build_operator_queue_context(
                    objective=str(profile.objective or "").strip() or None,
                    domain=str(profile.domain or "").strip() or None,
                    track_type=str(profile.track_type or opportunity.get("track_type") or "generic").strip() or None,
                    source_scope=str(profile.source_scope or "kb_plus_arxiv").strip() or None,
                    repo_source_ids=profile.repo_source_ids or opportunity.get("source_repo_ids"),
                    benchmark_queries=profile.benchmark_queries,
                    sandbox_profile_id=str(profile.sandbox_profile_id or "").strip() or None,
                    automation_profile=payload["automation_profile"],
                    effective_policy=effective_policy,
                    confidence=opportunity.get("confidence"),
                    readiness=opportunity.get("readiness"),
                    linked_note_ids=profile.latest_note_ids,
                    linked_experiment_plan_ids=opportunity.get("linked_experiment_plan_ids") or profile.latest_experiment_plan_ids,
                    linked_validation_run_ids=opportunity.get("linked_validation_run_ids") or profile.latest_validation_run_ids,
                    child_job_ids=opportunity.get("child_job_ids"),
                ),
            )
            if review_type == "follow_up_recommendation":
                actions = [
                    AgentCheckpointQueueActionResponse(
                        kind="follow_up_action",
                        label="Approve & Launch",
                        action="approve_launch",
                        description="Approve this bounded profile follow-up and launch it immediately.",
                        recommended=True,
                        recommendation_key="profile_follow_up",
                        follow_up_action_payload={
                            "domain_research_profile_id": str(profile.id),
                            "profile_opportunity_id": str(opportunity.get("opportunity_id") or "").strip(),
                        },
                    ),
                    AgentCheckpointQueueActionResponse(
                        kind="follow_up_action",
                        label="Reject Launch",
                        action="reject_launch",
                        description="Reject this queued profile follow-up for the current evidence revision.",
                        recommendation_key="profile_follow_up",
                        follow_up_action_payload={
                            "domain_research_profile_id": str(profile.id),
                            "profile_opportunity_id": str(opportunity.get("opportunity_id") or "").strip(),
                        },
                    ),
                ]
                items.append(
                    AgentCheckpointQueueItemResponse(
                        **common,
                        recommended_action="approve_launch",
                        action_count=len(actions),
                        follow_up_launch_status="pending_approval",
                        follow_up_policy_mode=str(effective_policy.get("follow_up_review_mode") or "").strip() or None,
                        follow_up_operator_decision=str(opportunity.get("follow_up_review_status") or "").strip() or None,
                        actions=actions,
                    )
                )
                continue

            items.append(
                AgentCheckpointQueueItemResponse(
                    **common,
                    recommended_action="open_fleet",
                    action_count=1,
                    budget_reason=(reason_code if review_type == "budget_review" else None),
                    actions=[
                        AgentCheckpointQueueActionResponse(
                            kind="policy_action",
                            label="Open Profile",
                            action="open_fleet",
                            description="Open this domain profile and inspect the targeted opportunity.",
                        )
                    ],
                )
            )

    items.sort(
        key=lambda row: (
            -int(row.priority or 0),
            -float(row.priority_score or 0),
            -(row.created_at.timestamp() if row.created_at else 0.0),
        )
    )
    return items


def _approval_payload_from_results(results: Optional[dict]) -> tuple[dict, dict, Optional[dict]]:
    payload = results if isinstance(results, dict) else {}
    execution = payload.get("execution_strategy") if isinstance(payload.get("execution_strategy"), dict) else {}
    approval = execution.get("approval_checkpoints") if isinstance(execution.get("approval_checkpoints"), dict) else {}
    pending = (
        approval.get("pending")
        if isinstance(approval.get("pending"), dict)
        else (payload.get("approval_checkpoint") if isinstance(payload.get("approval_checkpoint"), dict) else None)
    )
    return payload, approval, pending


def _normalize_checkpoint_action_patch(patch: Any) -> dict[str, Any]:
    if not isinstance(patch, dict):
        return {}

    out: dict[str, Any] = {}
    if "tool" in patch:
        tool = str(patch.get("tool") or "").strip()
        if not tool:
            raise ValueError("checkpoint_action_patch.tool cannot be empty")
        if not re.match(r"^[a-zA-Z0-9_:\\-]{2,80}$", tool):
            raise ValueError("checkpoint_action_patch.tool is invalid")
        out["tool"] = tool

    if "purpose" in patch:
        out["purpose"] = str(patch.get("purpose") or "").strip()[:220]

    if "params" in patch:
        params = patch.get("params")
        if params is None:
            out["params"] = {}
        elif isinstance(params, dict):
            out["params"] = _normalize_scope_keys_deep(params)
        else:
            raise ValueError("checkpoint_action_patch.params must be an object")

    return out


def _apply_checkpoint_action_patch(pending_checkpoint: dict, patch: dict[str, Any]) -> dict[str, Any]:
    action_payload = (
        pending_checkpoint.get("action")
        if isinstance(pending_checkpoint.get("action"), dict)
        else {}
    )
    merged = dict(action_payload)
    if "tool" in patch:
        merged["tool"] = patch["tool"]
    if "purpose" in patch:
        merged["purpose"] = patch["purpose"]
    if "params" in patch:
        merged["params"] = patch["params"]

    pending_checkpoint["action"] = merged
    pending_checkpoint["updated_at"] = datetime.utcnow().isoformat()
    return merged


def _append_approval_event(
    approval: dict,
    pending_checkpoint: dict,
    *,
    method: str,
    user_id: UUID,
    note: Optional[str] = None,
    edited_action: Optional[dict[str, Any]] = None,
) -> None:
    event_key = "rejections" if method == "reject_action" else "approvals"
    events = approval.get(event_key) if isinstance(approval.get(event_key), list) else []
    event = {
        "at": datetime.utcnow().isoformat(),
        "approved_by": str(user_id),
        "method": method,
        "checkpoint": {
            "iteration": int(pending_checkpoint.get("iteration", 0) or 0),
            "action_tool": str(((pending_checkpoint.get("action") or {}).get("tool") or "")).strip(),
            "plan_step_id": str(pending_checkpoint.get("plan_step_id") or "").strip() or None,
            "plan_step_index": int(pending_checkpoint.get("plan_step_index", -1) or -1),
        },
    }
    if note:
        event["note"] = str(note)[:1000]
    if isinstance(edited_action, dict) and edited_action:
        event["edited_action"] = edited_action
    events.append(event)
    approval[event_key] = events[-50:]


def _set_current_plan_step_status(
    state: Optional[dict],
    *,
    status: str,
    advance_next: bool = False,
) -> dict[str, Any]:
    payload = state if isinstance(state, dict) else {}
    plan = payload.get("execution_plan") if isinstance(payload.get("execution_plan"), list) else []
    if not plan:
        return {"step_id": "", "plan_step_index": -1}

    idx = int(payload.get("plan_step_index", 0) or 0)
    idx = max(0, min(idx, len(plan) - 1))
    step = plan[idx] if isinstance(plan[idx], dict) else {}

    step_id = str(step.get("step_id") or f"step_{idx + 1}").strip()
    if isinstance(step, dict):
        step["status"] = str(status).strip()[:40] or "pending"
        step["updated_at"] = datetime.utcnow().isoformat()
        plan[idx] = step

    if advance_next:
        next_idx = min(len(plan) - 1, idx + 1)
        payload["plan_step_index"] = next_idx
        if next_idx != idx and isinstance(plan[next_idx], dict) and str(plan[next_idx].get("status") or "") != "done":
            plan[next_idx]["status"] = "in_progress"
            plan[next_idx]["updated_at"] = datetime.utcnow().isoformat()

    payload["execution_plan"] = plan
    return {"step_id": step_id, "plan_step_index": idx}


def _append_step_event(state: Optional[dict], event: dict[str, Any], *, max_events: int = 500) -> None:
    payload = state if isinstance(state, dict) else {}
    if not isinstance(event, dict):
        return
    rows = payload.get("step_events") if isinstance(payload.get("step_events"), list) else []
    row = dict(event)
    row.setdefault("at", datetime.utcnow().isoformat())
    rows.append(row)
    payload["step_events"] = rows[-max(20, min(int(max_events or 500), 5000)):]


def _append_operator_intervention(
    results_payload: dict,
    *,
    action: str,
    actor_user_id: UUID | str,
    note: Optional[str] = None,
    job_status_before: Optional[str] = None,
    job_status_after: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    max_events: int = 200,
) -> dict[str, Any]:
    execution = (
        results_payload.get("execution_strategy")
        if isinstance(results_payload.get("execution_strategy"), dict)
        else {}
    )
    rows = (
        execution.get("operator_interventions")
        if isinstance(execution.get("operator_interventions"), list)
        else []
    )
    row: dict[str, Any] = {
        "action": str(action or "").strip()[:80] or "unknown",
        "actor_user_id": str(actor_user_id or "").strip() or None,
        "at": datetime.utcnow().isoformat(),
    }
    if note:
        row["note"] = str(note).strip()[:1000]
    if job_status_before:
        row["job_status_before"] = str(job_status_before).strip()[:40]
    if job_status_after:
        row["job_status_after"] = str(job_status_after).strip()[:40]
    if isinstance(metadata, dict) and metadata:
        row["metadata"] = metadata
    rows.append(row)
    execution["operator_interventions"] = rows[-max(20, min(int(max_events or 200), 1000)):]
    results_payload["execution_strategy"] = execution
    return row


def _sync_execution_strategy_state(
    results_payload: dict,
    *,
    approval_payload: Optional[dict] = None,
    state: Optional[dict] = None,
) -> dict:
    execution = (
        results_payload.get("execution_strategy")
        if isinstance(results_payload.get("execution_strategy"), dict)
        else {}
    )
    if isinstance(approval_payload, dict):
        execution["approval_checkpoints"] = approval_payload
    if isinstance(state, dict):
        execution["step_events"] = (
            state.get("step_events")
            if isinstance(state.get("step_events"), list)
            else []
        )[-300:]
    results_payload["execution_strategy"] = execution
    return execution


async def _load_latest_job_checkpoint(
    job_id: UUID,
    db: AsyncSession,
) -> Optional[AgentJobCheckpoint]:
    result = await db.execute(
        select(AgentJobCheckpoint)
        .where(AgentJobCheckpoint.job_id == job_id)
        .order_by(AgentJobCheckpoint.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


def _extract_executive_digest(job: AgentJob) -> Optional[dict]:
    """Extract deterministic executive digest payload when present."""
    results = job.results if isinstance(job.results, dict) else {}
    digest = results.get("executive_digest") if isinstance(results.get("executive_digest"), dict) else None
    return digest


def _sanitize_tool_names(values: Optional[list[str]], *, limit: int = 12) -> list[str]:
    out: list[str] = []
    if not isinstance(values, list):
        return out
    for raw in values:
        tool = str(raw or "").strip()
        if not tool:
            continue
        if not re.match(r"^[a-zA-Z0-9_:\\-]{2,80}$", tool):
            continue
        if tool not in out:
            out.append(tool)
        if len(out) >= max(1, min(limit, 40)):
            break
    return out


def _normalize_scope_config(config: Optional[dict]) -> Optional[dict]:
    """
    Normalize legacy scope keys in job/chain config to canonical `source_id`.

    Backward compatibility:
    - Accepts incoming `target_source_id`
    - Promotes to `source_id` when missing
    - Removes `target_source_id` in stored/returned configs
    """
    if not isinstance(config, dict):
        return config

    out = dict(config)
    source_id = str(out.get("source_id") or "").strip()
    target_source_id = str(out.get("target_source_id") or "").strip()
    if not source_id and target_source_id:
        out["source_id"] = target_source_id
    out.pop("target_source_id", None)
    return out


def _normalize_scope_keys_deep(value: Any) -> Any:
    """
    Recursively normalize legacy scope keys inside nested payloads.

    This keeps API contracts canonical (`source_id`) even when old keys
    (`target_source_id`) appear in nested chain/template structures.
    """
    if isinstance(value, dict):
        normalized = _normalize_scope_config(value) or {}
        out: dict[str, Any] = {}
        for k, v in normalized.items():
            out[k] = _normalize_scope_keys_deep(v)
        return out
    if isinstance(value, list):
        return [_normalize_scope_keys_deep(v) for v in value]
    return value


def _merge_chain_step_config(default_settings: Optional[dict], step_config: Optional[dict]) -> dict:
    """
    Merge chain defaults with step config while keeping inherited root scope stable.

    Defaults provide the canonical top-level `source_id` for the chain unless it is
    absent. Nested step config still overrides or augments nested config branches.
    """

    def _merge(base: Any, override: Any, *, preserve_source_id: bool) -> Any:
        if isinstance(base, dict) and isinstance(override, dict):
            merged = deepcopy(base)
            for key, value in override.items():
                if key == "source_id" and preserve_source_id and str(merged.get("source_id") or "").strip():
                    continue
                existing = merged.get(key)
                merged[key] = _merge(existing, value, preserve_source_id=False)
            return merged
        return deepcopy(override)

    base = _normalize_scope_keys_deep(default_settings) or {}
    override = _normalize_scope_keys_deep(step_config) or {}
    return _merge(base, override, preserve_source_id=True)


def _memory_to_feedback_response(memory: ConversationMemory) -> AgentJobFeedbackResponse:
    context = memory.context if isinstance(memory.context, dict) else {}
    preferred = context.get("preferred_tools") if isinstance(context.get("preferred_tools"), list) else []
    discouraged = context.get("discouraged_tools") if isinstance(context.get("discouraged_tools"), list) else []
    try:
        rating = int(context.get("rating", 0) or 0)
    except Exception:
        rating = 0
    rating = max(1, min(5, rating)) if rating else 3
    return AgentJobFeedbackResponse(
        id=memory.id,
        job_id=memory.job_id,
        rating=rating,
        feedback=str(context.get("feedback_text") or memory.content or "").strip() or None,
        target_type=str(context.get("target_type") or "job"),
        target_id=str(context.get("target_id") or "").strip() or None,
        scope=str(context.get("scope") or "user"),
        preferred_tools=[str(x) for x in preferred[:20]],
        discouraged_tools=[str(x) for x in discouraged[:20]],
        checkpoint=str(context.get("checkpoint") or "").strip() or None,
        created_at=memory.created_at,
    )


def _job_to_response(
    job: AgentJob,
    *,
    relaunch_children_count: int = 0,
    current_user_id: Optional[str] = None,
    user_lookup: Optional[dict[str, User]] = None,
) -> AgentJobResponse:
    """Convert AgentJob model to response schema."""
    cfg = job.config if isinstance(job.config, dict) else {}
    results = job.results if isinstance(job.results, dict) else {}
    now = datetime.utcnow()
    experiment_run = results.get("experiment_run") if isinstance(results.get("experiment_run"), dict) else None
    experiment_runs_raw = results.get("experiment_runs") if isinstance(results.get("experiment_runs"), list) else []
    experiment_runs = [row for row in experiment_runs_raw if isinstance(row, dict)]
    execution_strategy = results.get("execution_strategy") if isinstance(results.get("execution_strategy"), dict) else {}
    scheduler_state = _extract_scheduler_state(job)
    operator_interventions_raw = (
        execution_strategy.get("operator_interventions")
        if isinstance(execution_strategy.get("operator_interventions"), list)
        else []
    )
    effective_intervention_status = job.status
    if operator_interventions_raw:
        last_intervention = next(
            (
                row
                for row in reversed(operator_interventions_raw)
                if isinstance(row, dict)
            ),
            None,
        )
        if isinstance(last_intervention, dict):
            action = str(last_intervention.get("action") or "").strip().lower()
            derived_status = str(last_intervention.get("job_status_after") or "").strip()
            if derived_status and action in {"pause", "reject"}:
                effective_intervention_status = derived_status
    operator_interventions = derive_operator_interventions_with_outcomes(
        [row for row in operator_interventions_raw if isinstance(row, dict)],
        current_status=effective_intervention_status,
        completed_at=job.completed_at,
        status_values={
            "completed": AgentJobStatus.COMPLETED.value,
            "failed": AgentJobStatus.FAILED.value,
            "cancelled": AgentJobStatus.CANCELLED.value,
            "pending": AgentJobStatus.PENDING.value,
            "running": AgentJobStatus.RUNNING.value,
            "paused": AgentJobStatus.PAUSED.value,
        },
    )
    relaunch_from_job_id = _extract_relaunch_parent_job_id(cfg)
    promotion = _extract_domain_research_promotion(job)
    promoted_profile_id_raw = str(
        promotion.get("domain_research_profile_id")
        or promotion.get("promoted_domain_research_profile_id")
        or ""
    ).strip()
    promoted_portfolio_id_raw = str(
        promotion.get("research_portfolio_id")
        or promotion.get("promoted_research_portfolio_id")
        or ""
    ).strip()
    return AgentJobResponse(
        id=job.id or uuid.uuid4(),
        name=job.name,
        description=job.description,
        job_type=job.job_type,
        goal=job.goal,
        goal_criteria=job.goal_criteria,
        config=_normalize_scope_config(job.config),
        launch_mode=_extract_launch_mode(cfg) or None,
        relaunch_from_job_id=relaunch_from_job_id,
        relaunch_children_count=max(0, int(relaunch_children_count or 0)),
        promotion_status=str(promotion.get("status") or "").strip() or None,
        promoted_domain_research_profile_id=(
            UUID(promoted_profile_id_raw) if re.fullmatch(r"[0-9a-fA-F-]{36}", promoted_profile_id_raw) else None
        ),
        promoted_research_portfolio_id=(
            UUID(promoted_portfolio_id_raw) if re.fullmatch(r"[0-9a-fA-F-]{36}", promoted_portfolio_id_raw) else None
        ),
        agent_definition_id=job.agent_definition_id,
        agent_definition_name=job.agent_definition.name if job.agent_definition else None,
        user_id=job.user_id,
        status=job.status,
        progress=int(job.progress or 0),
        current_phase=job.current_phase,
        phase_details=job.phase_details,
        iteration=int(job.iteration or 0),
        max_iterations=int(job.max_iterations or 0),
        max_tool_calls=int(job.max_tool_calls or 0),
        max_llm_calls=int(job.max_llm_calls or 0),
        max_runtime_minutes=int(job.max_runtime_minutes or 0),
        tool_calls_used=int(job.tool_calls_used or 0),
        llm_calls_used=int(job.llm_calls_used or 0),
        tokens_used=int(job.tokens_used or 0),
        error=job.error,
        error_count=int(job.error_count or 0),
        schedule_type=job.schedule_type,
        schedule_cron=job.schedule_cron,
        next_run_at=job.next_run_at,
        scheduler_state=scheduler_state,
        results=results or None,
        experiment_run=experiment_run,
        experiment_runs=experiment_runs or None,
        operator_interventions=operator_interventions or None,
        output_artifacts=job.output_artifacts,
        created_at=job.created_at or now,
        started_at=job.started_at,
        completed_at=job.completed_at,
        last_activity_at=job.last_activity_at,
        celery_task_id=job.celery_task_id,
        # Chain fields
        parent_job_id=job.parent_job_id,
        root_job_id=job.root_job_id,
        chain_depth=int(job.chain_depth or 0),
        chain_triggered=bool(job.chain_triggered),
        chain_config=_normalize_scope_keys_deep(job.chain_config),
        swarm_summary=_extract_swarm_summary(job, current_user_id=current_user_id, user_lookup=user_lookup),
        goal_contract_summary=_extract_goal_contract_summary(job),
        approval_checkpoint=_extract_approval_checkpoint(job),
        executive_digest=_extract_executive_digest(job),
    )


def _chain_definition_to_response(chain: AgentJobChainDefinition | object) -> AgentJobChainDefinitionResponse:
    """Convert chain definition model/object to response with normalized scope keys."""
    raw_steps = getattr(chain, "chain_steps", None)
    chain_steps: list[dict] = []
    if isinstance(raw_steps, list):
        for step in raw_steps:
            if not isinstance(step, dict):
                continue
            item = dict(step)
            if isinstance(item.get("config"), dict):
                item["config"] = _normalize_scope_keys_deep(item.get("config"))
            chain_steps.append(item)

    return AgentJobChainDefinitionResponse(
        id=getattr(chain, "id"),
        name=getattr(chain, "name"),
        display_name=getattr(chain, "display_name"),
        description=getattr(chain, "description", None),
        chain_steps=chain_steps,
        default_settings=_normalize_scope_keys_deep(getattr(chain, "default_settings", None)),
        owner_user_id=getattr(chain, "owner_user_id", None),
        is_system=bool(getattr(chain, "is_system", False)),
        is_active=bool(getattr(chain, "is_active", True)),
        created_at=getattr(chain, "created_at"),
        updated_at=getattr(chain, "updated_at"),
    )


@router.post("", response_model=AgentJobResponse, status_code=status.HTTP_201_CREATED)
async def create_agent_job(
    job_create: AgentJobCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Create a new autonomous agent job.

    Creates a background job that will work autonomously toward the specified goal.
    """
    # Validate agent definition if specified
    if job_create.agent_definition_id:
        result = await db.execute(
            select(AgentDefinition).where(AgentDefinition.id == job_create.agent_definition_id)
        )
        agent_def = result.scalar_one_or_none()
        if not agent_def:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent definition not found",
            )

    # Validate parent job if specified (for manual chaining)
    if job_create.parent_job_id:
        parent_result = await db.execute(
            select(AgentJob).where(
                and_(
                    AgentJob.id == job_create.parent_job_id,
                    AgentJob.user_id == current_user.id,
                )
            )
        )
        parent_job = parent_result.scalar_one_or_none()
        if not parent_job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Parent job not found",
            )

    # Create the job
    config_payload = _normalize_scope_config(job_create.config)
    job = AgentJob(
        name=job_create.name,
        description=job_create.description,
        job_type=job_create.job_type,
        goal=job_create.goal,
        goal_criteria=job_create.goal_criteria,
        config=config_payload,
        agent_definition_id=job_create.agent_definition_id,
        user_id=current_user.id,
        status=AgentJobStatus.PENDING.value,
        enable_memory=_extract_enable_memory_from_config(config_payload, default=True),
        max_iterations=job_create.max_iterations or 100,
        max_tool_calls=job_create.max_tool_calls or 500,
        max_llm_calls=job_create.max_llm_calls or 200,
        max_runtime_minutes=job_create.max_runtime_minutes or 60,
        schedule_type=job_create.schedule_type,
        schedule_cron=job_create.schedule_cron,
        # Chain fields
        chain_config=_normalize_scope_keys_deep(job_create.chain_config),
        parent_job_id=job_create.parent_job_id,
        chain_depth=parent_job.chain_depth + 1 if job_create.parent_job_id and parent_job else 0,
        root_job_id=parent_job.root_job_id or parent_job.id if job_create.parent_job_id and parent_job else None,
    )

    # Set next_run_at for scheduled jobs
    if job_create.schedule_type and job_create.schedule_cron:
        try:
            from croniter import croniter
            cron = croniter(job_create.schedule_cron, datetime.utcnow())
            job.next_run_at = cron.get_next(datetime)
        except Exception as e:
            logger.warning(f"Invalid cron expression: {e}")
    elif job_create.schedule_type == "continuous" and not job.next_run_at:
        # Continuous jobs use a simple interval (handled by scheduler task).
        job.next_run_at = datetime.utcnow()

    db.add(job)
    await db.commit()
    await db.refresh(job)
    if _append_launch_log_if_present(job):
        await db.commit()

    logger.info(f"Created agent job {job.id} for user {current_user.id}")

    # Start immediately if requested (including scheduled jobs).
    # For scheduled jobs, we also advance `next_run_at` to avoid an immediate duplicate run by the scheduler.
    if job_create.start_immediately:
        execute_agent_job_task.delay(str(job.id), str(current_user.id))
        logger.info(f"Queued agent job {job.id} for immediate execution")

        if job.schedule_type == "continuous":
            try:
                interval = int(((job.config or {}).get("interval_minutes") or 30))
            except Exception:
                interval = 30
            interval = max(1, min(interval, 24 * 60))
            job.next_run_at = datetime.utcnow() + timedelta(minutes=interval)
            await db.commit()
        elif job.schedule_type == "once":
            job.next_run_at = None
            await db.commit()

    return _job_to_response(job)


@router.post("/from-template", response_model=AgentJobResponse, status_code=status.HTTP_201_CREATED)
async def create_job_from_template(
    request: AgentJobFromTemplate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Create a new agent job from a template.

    Uses the template's default configuration with optional overrides.
    """
    # Load template (builtin first, then DB)
    builtin = get_builtin_agent_job_template(request.template_id)
    template = None
    if builtin is None:
        result = await db.execute(
            select(AgentJobTemplate).where(
                and_(
                    AgentJobTemplate.id == request.template_id,
                    AgentJobTemplate.is_active == True,
                )
            )
        )
        template = result.scalar_one_or_none()

    if not builtin and not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job template not found or not active",
        )

    # Merge config
    base_config = builtin.default_config if builtin else (template.default_config or {})
    config = _normalize_scope_config(dict(base_config) if base_config else {})
    if request.config:
        config.update(_normalize_scope_config(request.config) or {})
    config = _normalize_scope_config(config)

    template_chain_config = (
        request.chain_config
        if request.chain_config
        else (builtin.default_chain_config if builtin else getattr(template, "default_chain_config", None))
    )

    # Create job from template
    job = AgentJob(
        name=request.name,
        description=(builtin.description if builtin else template.description),
        job_type=(builtin.job_type if builtin else template.job_type),
        goal=request.goal or (builtin.default_goal if builtin else template.default_goal),
        config=config,
        agent_definition_id=(builtin.agent_definition_id if builtin else template.agent_definition_id),
        user_id=current_user.id,
        status=AgentJobStatus.PENDING.value,
        enable_memory=_extract_enable_memory_from_config(config, default=True),
        max_iterations=(builtin.default_max_iterations if builtin else template.default_max_iterations),
        max_tool_calls=(builtin.default_max_tool_calls if builtin else template.default_max_tool_calls),
        max_llm_calls=(builtin.default_max_llm_calls if builtin else template.default_max_llm_calls),
        max_runtime_minutes=(builtin.default_max_runtime_minutes if builtin else template.default_max_runtime_minutes),
        chain_config=_normalize_scope_keys_deep(template_chain_config),
    )

    db.add(job)
    await db.commit()
    await db.refresh(job)
    if _append_launch_log_if_present(job):
        await db.commit()

    template_name = builtin.name if builtin else str(getattr(template, "name", "") or "")
    logger.info(f"Created agent job {job.id} from template {template_name}")

    # Start immediately if requested
    if request.start_immediately:
        execute_agent_job_task.delay(str(job.id), str(current_user.id))

    return _job_to_response(job)


@router.post("/quick-start/claude-backend", response_model=AgentJobResponse, status_code=status.HTTP_201_CREATED)
async def quick_start_claude_backend_job(
    request: AgentJobQuickStartClaudeBackendRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Quick-start helper for a Claude-style backend coding loop.

    Creates a job from the built-in `claude_code_backend` template with
    required source scope + goal and optional overrides.
    """
    from app.models.document import Document, DocumentSource

    source = await db.get(DocumentSource, request.source_id)
    if not source:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document source not found")
    source_type = str(getattr(source, "source_type", "") or "").strip().lower()
    if source_type not in {"github", "gitlab"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Quick start requires a github/gitlab document source",
        )
    if not current_user.is_admin() and not _is_source_owned_by_user(source, current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized for this source")

    doc_count_result = await db.execute(
        select(func.count()).where(Document.source_id == source.id)
    )
    doc_count = int(doc_count_result.scalar() or 0)
    if doc_count <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Source has no documents; ingest/sync the repository first",
        )

    unsafe_commands = _find_unsafe_commands(request.commands)
    if unsafe_commands:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Quick start rejected potentially destructive command(s)",
                "blocked_commands": unsafe_commands,
            },
        )

    merged_config = _build_quick_start_claude_backend_config(
        request,
        source_name=str(getattr(source, "name", "") or ""),
        source_type=source_type,
    )

    job_name = str(request.name or "").strip()
    if not job_name:
        job_name = f"Claude Backend Loop - {datetime.utcnow().strftime('%Y-%m-%d')}"

    template_request = AgentJobFromTemplate(
        template_id=CLAUDE_CODE_BACKEND_TEMPLATE_ID,
        name=job_name,
        goal=request.goal,
        config=merged_config,
        start_immediately=bool(request.start_immediately),
    )
    return await create_job_from_template(template_request, db, current_user)


@router.post("/quick-start/domain-research", response_model=AgentJobResponse, status_code=status.HTTP_201_CREATED)
async def quick_start_domain_research_job(
    request: AgentJobQuickStartDomainResearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Quick-start helper for domain research with Research Note persistence."""
    merged_config = _build_quick_start_domain_research_config(request)

    job_name = str(request.name or "").strip()
    if not job_name:
        job_name = f"Domain Research - {str(request.domain or '').strip()[:80]}"

    template_request = AgentJobFromTemplate(
        template_id=DOMAIN_RESEARCH_TEMPLATE_ID,
        name=job_name,
        goal=_build_domain_research_goal(request),
        config=merged_config,
        start_immediately=bool(request.start_immediately),
    )
    return await create_job_from_template(template_request, db, current_user)


@router.post("/quick-start/repo-bug-triage", response_model=AgentJobResponse, status_code=status.HTTP_201_CREATED)
async def quick_start_repo_bug_triage_job(
    request: AgentJobQuickStartRepoBugTriageRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Quick-start helper for a repo-wide symptom-driven bug triage + repair loop.

    Creates a job from the built-in `repo_bug_triage_repair` template with
    required source scope plus failure context and optional overrides.
    """
    from app.models.document import Document, DocumentSource

    source = await db.get(DocumentSource, request.source_id)
    if not source:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document source not found")
    source_type = str(getattr(source, "source_type", "") or "").strip().lower()
    if source_type not in {"github", "gitlab"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Quick start requires a github/gitlab document source",
        )
    if not current_user.is_admin() and not _is_source_owned_by_user(source, current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized for this source")

    doc_count_result = await db.execute(
        select(func.count()).where(Document.source_id == source.id)
    )
    doc_count = int(doc_count_result.scalar() or 0)
    if doc_count <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Source has no documents; ingest/sync the repository first",
        )

    unsafe_commands = _find_unsafe_commands(request.commands)
    if unsafe_commands:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Quick start rejected potentially destructive command(s)",
                "blocked_commands": unsafe_commands,
            },
        )

    merged_config = _build_quick_start_repo_bug_triage_config(
        request,
        source_name=str(getattr(source, "name", "") or ""),
        source_type=source_type,
    )

    job_name = str(request.name or "").strip()
    if not job_name:
        job_name = f"Repo Bug Triage - {datetime.utcnow().strftime('%Y-%m-%d')}"

    template_request = AgentJobFromTemplate(
        template_id=REPO_BUG_TRIAGE_REPAIR_TEMPLATE_ID,
        name=job_name,
        goal=_build_repo_bug_triage_goal(request),
        config=merged_config,
        start_immediately=bool(request.start_immediately),
    )
    return await create_job_from_template(template_request, db, current_user)


@router.post("/quick-start/bug-triage-swarm", response_model=AgentJobResponse, status_code=status.HTTP_201_CREATED)
async def quick_start_bug_triage_swarm_job(
    request: AgentJobQuickStartBugTriageSwarmRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Quick-start helper for a coding-focused bug triage swarm."""
    return await _create_quick_start_coding_swarm_job(
        request=request,
        db=db,
        current_user=current_user,
        preset_key="bug_triage_swarm",
    )


@router.post("/quick-start/build-break-swarm", response_model=AgentJobResponse, status_code=status.HTTP_201_CREATED)
async def quick_start_build_break_swarm_job(
    request: AgentJobQuickStartBuildBreakSwarmRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    return await _create_quick_start_coding_swarm_job(
        request=request,
        db=db,
        current_user=current_user,
        preset_key="build_break_swarm",
    )


@router.post("/quick-start/frontend-regression-swarm", response_model=AgentJobResponse, status_code=status.HTTP_201_CREATED)
async def quick_start_frontend_regression_swarm_job(
    request: AgentJobQuickStartFrontendRegressionSwarmRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    return await _create_quick_start_coding_swarm_job(
        request=request,
        db=db,
        current_user=current_user,
        preset_key="frontend_regression_swarm",
    )


async def _create_quick_start_coding_swarm_job(
    *,
    request: AgentJobQuickStartBugTriageSwarmRequest | AgentJobQuickStartBuildBreakSwarmRequest | AgentJobQuickStartFrontendRegressionSwarmRequest,
    db: AsyncSession,
    current_user: User,
    preset_key: str,
):
    from app.models.document import Document, DocumentSource

    preset = _get_coding_swarm_preset_definition(preset_key)
    source = await db.get(DocumentSource, request.source_id)
    if not source:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document source not found")
    source_type = str(getattr(source, "source_type", "") or "").strip().lower()
    if source_type not in {"github", "gitlab"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Quick start requires a github/gitlab document source",
        )
    profile = await _resolve_coding_swarm_profile(
        db,
        current_user=current_user,
        source_id=request.source_id,
        profile_id=getattr(request, "profile_id", None),
        preset_key=preset_key,
    )
    profile_grants_access = bool(profile is not None and _is_coding_swarm_profile_visible_to_user(profile, current_user))
    if not current_user.is_admin() and not _is_source_owned_by_user(source, current_user) and not profile_grants_access:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized for this source")

    doc_count_result = await db.execute(
        select(func.count()).where(Document.source_id == source.id)
    )
    doc_count = int(doc_count_result.scalar() or 0)
    if doc_count <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Source has no documents; ingest/sync the repository first",
        )

    request = _merge_coding_swarm_request_with_profile(request, profile=profile, preset_key=preset_key)

    unsafe_commands = _find_unsafe_commands(request.commands)
    if unsafe_commands:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Quick start rejected potentially destructive command(s)",
                "blocked_commands": unsafe_commands,
            },
        )

    merged_config = _build_quick_start_coding_swarm_config(
        request,
        source_name=str(getattr(source, "name", "") or ""),
        source_type=source_type,
        preset_key=preset_key,
    )

    job_name = str(request.name or "").strip()
    if not job_name:
        job_name = f"{str(preset.get('display_name') or 'Coding Swarm').strip()} - {datetime.utcnow().strftime('%Y-%m-%d')}"

    job = AgentJob(
        name=job_name,
        description=f"{str(preset.get('display_name') or 'Coding Swarm').strip()} with automatic fan-in and repair-chain handoff.",
        job_type="analysis",
        goal=_build_coding_swarm_goal(request, preset_key=preset_key),
        config=merged_config,
        user_id=current_user.id,
        status=AgentJobStatus.PENDING.value,
        enable_memory=_extract_enable_memory_from_config(merged_config, default=False),
        max_iterations=90,
        max_tool_calls=360,
        max_llm_calls=140,
        max_runtime_minutes=120,
    )
    _store_swarm_collaboration(
        job,
        _build_swarm_collaboration_payload(
            owner_user_id=current_user.id,
            visibility=_normalize_coding_swarm_profile_visibility(getattr(profile, "visibility", "private") if profile else "private"),
            shared_with_user_ids=_normalize_uuid_str_list(getattr(profile, "shared_with_user_ids", None), 200) if profile else [],
        ),
    )

    db.add(job)
    if profile is not None:
        profile.latest_job_id = job.id
        db.add(profile)
    await db.commit()
    await db.refresh(job)
    if _append_launch_log_if_present(job):
        await db.commit()

    if request.start_immediately:
        execute_agent_job_task.delay(str(job.id), str(current_user.id))

    return _job_to_response(job)


@router.post("/quick-start/role-workflow", response_model=AgentJobResponse, status_code=status.HTTP_201_CREATED)
async def quick_start_role_workflow_job(
    request: AgentJobQuickStartRoleWorkflowRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Quick-start helper for launching a role-based multi-agent workflow."""
    merged_config = _build_quick_start_role_workflow_config(request)

    job_name = str(request.name or "").strip()
    if not job_name:
        job_name = f"Role Workflow - {datetime.utcnow().strftime('%Y-%m-%d')}"

    job = AgentJob(
        name=job_name,
        description="Role-based multi-agent plan-and-execute workflow (quick start).",
        job_type="research",
        goal=str(request.goal or "").strip(),
        config=merged_config,
        user_id=current_user.id,
        status=AgentJobStatus.PENDING.value,
        enable_memory=_extract_enable_memory_from_config(merged_config, default=True),
        max_iterations=120,
        max_tool_calls=700,
        max_llm_calls=260,
        max_runtime_minutes=120,
    )

    db.add(job)
    await db.commit()
    await db.refresh(job)
    if _append_launch_log_if_present(job):
        await db.commit()

    if request.start_immediately:
        execute_agent_job_task.delay(str(job.id), str(current_user.id))

    return _job_to_response(job)


@router.get("", response_model=AgentJobListResponse)
async def list_agent_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    job_type: Optional[str] = Query(None, description="Filter by job type"),
    launch_mode: Optional[str] = Query(
        None,
        description="Filter by launch mode (e.g. quick_start_claude_backend, quick_start_role_workflow)",
    ),
    relaunch_from_job_id: Optional[str] = Query(None, description="Filter jobs relaunched from a specific parent job id"),
    has_relaunch_children: Optional[bool] = Query(None, description="Filter jobs by whether they have relaunch descendants"),
    swarm_only: bool = Query(False, description="Only return jobs with swarm summary data"),
    swarm_min_consensus: int = Query(0, ge=0, le=100, description="Minimum swarm consensus findings"),
    visibility_scope: str = Query("mine", description="Visibility scope: mine|shared|all"),
    sort_by: str = Query(
        "created_desc",
        description="Sort mode: created_desc|created_asc|swarm_confidence_desc|swarm_consensus_desc|swarm_conflicts_desc",
    ),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List agent jobs with optional shared coding swarm visibility."""
    visibility_scope = str(visibility_scope or "mine").strip().lower() or "mine"
    query = select(AgentJob)
    if visibility_scope == "mine":
        query = query.where(AgentJob.user_id == current_user.id)
    elif visibility_scope == "shared":
        query = query.where(AgentJob.user_id != current_user.id)

    if status:
        query = query.where(AgentJob.status == status)
    if job_type:
        query = query.where(AgentJob.job_type == job_type)
    launch_mode_filter = str(launch_mode or "").strip().lower()
    launch_mode_expr = func.lower(func.trim(func.coalesce(_json_launch_mode_expr(), "")))
    if launch_mode_filter:
        if launch_mode_filter in {"__none__", "none", "manual"}:
            query = query.where(launch_mode_expr.in_(["", "__none__", "none", "manual"]))
        else:
            query = query.where(launch_mode_expr == launch_mode_filter)
    relaunch_from_filter_raw = str(relaunch_from_job_id or "").strip()
    relaunch_from_filter_uuid: Optional[UUID] = None
    if relaunch_from_filter_raw:
        try:
            relaunch_from_filter_uuid = UUID(relaunch_from_filter_raw)
        except Exception:
            raise HTTPException(
                status_code=422,
                detail="Invalid relaunch_from_job_id",
            )
    relaunch_parent_expr = _json_relaunch_parent_expr()
    if relaunch_from_filter_uuid is not None:
        query = query.where(relaunch_parent_expr == str(relaunch_from_filter_uuid))
    if has_relaunch_children is not None and visibility_scope == "mine":
        child = aliased(AgentJob)
        child_parent_expr = _json_relaunch_parent_expr(child)
        has_children_expr = (
            select(literal(1))
            .where(
                and_(
                    child.user_id == current_user.id,
                    child_parent_expr == cast(AgentJob.id, String),
                )
            )
            .exists()
        )
        query = query.where(has_children_expr if has_relaunch_children else ~has_children_expr)
    relaunch_children_counts = (
        await _build_relaunch_children_counts_for_user(db, user_id=current_user.id)
        if visibility_scope == "mine"
        else {}
    )

    sort_mode = str(sort_by or "created_desc").strip().lower()
    allowed_sort_modes = {
        "created_desc",
        "created_asc",
        "swarm_confidence_desc",
        "swarm_consensus_desc",
        "swarm_conflicts_desc",
    }
    if sort_mode not in allowed_sort_modes:
        sort_mode = "created_desc"

    requires_swarm_projection = (
        bool(swarm_only)
        or int(swarm_min_consensus or 0) > 0
        or sort_mode.startswith("swarm_")
        or sort_mode == "created_asc"
    )

    if not requires_swarm_projection:
        page_query = query.options(selectinload(AgentJob.agent_definition))
        page_query = page_query.order_by(AgentJob.created_at.desc())
        result = await db.execute(page_query)
        jobs_all = result.scalars().all()
        if visibility_scope != "mine":
            jobs_all = [job for job in jobs_all if _is_job_visible_to_user(job, current_user)]
        total = len(jobs_all)
        offset = (page - 1) * page_size
        jobs = jobs_all[offset : offset + page_size]
    else:
        all_query = query.options(selectinload(AgentJob.agent_definition))
        all_query = all_query.order_by(AgentJob.created_at.desc())
        all_result = await db.execute(all_query)
        jobs_all = all_result.scalars().all()
        if visibility_scope != "mine":
            jobs_all = [job for job in jobs_all if _is_job_visible_to_user(job, current_user)]

        rows = []
        for job in jobs_all:
            swarm_summary = _extract_swarm_summary(job)
            if swarm_only and not swarm_summary:
                continue
            if int(swarm_min_consensus or 0) > 0:
                consensus_count = int((swarm_summary or {}).get("consensus_count", 0) or 0)
                if consensus_count < int(swarm_min_consensus or 0):
                    continue
            rows.append((job, swarm_summary))

        def _created_ts(job: AgentJob) -> float:
            try:
                return float(job.created_at.timestamp()) if job.created_at else 0.0
            except Exception:
                return 0.0

        if sort_mode == "created_asc":
            rows.sort(key=lambda x: _created_ts(x[0]))
        elif sort_mode == "swarm_confidence_desc":
            rows.sort(
                key=lambda x: (
                    float((((x[1] or {}).get("confidence") or {}).get("overall") or 0.0)),
                    int((x[1] or {}).get("consensus_count", 0) or 0),
                    _created_ts(x[0]),
                ),
                reverse=True,
            )
        elif sort_mode == "swarm_consensus_desc":
            rows.sort(
                key=lambda x: (
                    int((x[1] or {}).get("consensus_count", 0) or 0),
                    float((((x[1] or {}).get("confidence") or {}).get("overall") or 0.0)),
                    _created_ts(x[0]),
                ),
                reverse=True,
            )
        elif sort_mode == "swarm_conflicts_desc":
            rows.sort(
                key=lambda x: (
                    int((x[1] or {}).get("conflict_count", 0) or 0),
                    int((x[1] or {}).get("consensus_count", 0) or 0),
                    _created_ts(x[0]),
                ),
                reverse=True,
            )

        total = len(rows)
        offset = (page - 1) * page_size
        jobs = [j for j, _ in rows[offset : offset + page_size]]

    collaboration_user_lookup = await _build_collaboration_user_lookup(db, current_user=current_user)

    return AgentJobListResponse(
        jobs=[
            _job_to_response(
                job,
                relaunch_children_count=int(relaunch_children_counts.get(job.id, 0) or 0),
                current_user_id=str(current_user.id),
                user_lookup=collaboration_user_lookup,
            )
            for job in jobs
        ],
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )


@router.get("/stats", response_model=AgentJobStatsResponse)
async def get_job_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get statistics for the current user's agent jobs.
    """
    base_query = select(AgentJob).where(AgentJob.user_id == current_user.id)

    # Count by status
    status_counts = {}
    for s in AgentJobStatus:
        count_result = await db.execute(
            select(func.count()).where(
                and_(AgentJob.user_id == current_user.id, AgentJob.status == s.value)
            )
        )
        status_counts[s.value] = count_result.scalar()

    # Total counts
    total_result = await db.execute(
        select(
            func.sum(AgentJob.iteration),
            func.sum(AgentJob.tool_calls_used),
            func.sum(AgentJob.llm_calls_used),
        ).where(AgentJob.user_id == current_user.id)
    )
    totals = total_result.one()

    # Launch mode breakdown
    launch_rows = await db.execute(
        select(AgentJob.config).where(AgentJob.user_id == current_user.id)
    )
    launch_configs = [row[0] for row in launch_rows.all()]
    launch_mode_counts, launch_mode_none_count = _build_launch_mode_stats(launch_configs)

    # Average completion time
    completed_jobs = await db.execute(
        select(AgentJob).where(
            and_(
                AgentJob.user_id == current_user.id,
                AgentJob.status == AgentJobStatus.COMPLETED.value,
                AgentJob.started_at.isnot(None),
                AgentJob.completed_at.isnot(None),
            )
        )
    )
    completed = completed_jobs.scalars().all()

    avg_time = None
    if completed:
        durations = [
            (job.completed_at - job.started_at).total_seconds() / 60
            for job in completed
        ]
        avg_time = sum(durations) / len(durations)

    # Success rate
    total_finished = status_counts.get("completed", 0) + status_counts.get("failed", 0)
    success_rate = None
    if total_finished > 0:
        success_rate = status_counts.get("completed", 0) / total_finished

    return AgentJobStatsResponse(
        total_jobs=sum(status_counts.values()),
        running_jobs=status_counts.get("running", 0),
        pending_jobs=status_counts.get("pending", 0),
        completed_jobs=status_counts.get("completed", 0),
        failed_jobs=status_counts.get("failed", 0),
        total_iterations=totals[0] or 0,
        total_tool_calls=totals[1] or 0,
        total_llm_calls=totals[2] or 0,
        avg_completion_time_minutes=avg_time,
        success_rate=success_rate,
        launch_mode_counts=launch_mode_counts,
        launch_mode_none_count=launch_mode_none_count,
    )


@router.get("/swarm-analytics", response_model=AgentJobSwarmAnalyticsResponse)
async def get_swarm_analytics(
    source_id: Optional[UUID] = Query(None),
    preset_key: Optional[str] = Query(None),
    visibility_scope: str = Query("mine"),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    allowed_presets = set(_CODING_SWARM_ANALYTICS_PRESETS.keys())
    normalized_preset = str(preset_key or "").strip().lower() or None
    if normalized_preset and normalized_preset not in allowed_presets:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Unknown coding swarm preset")

    visibility_scope = str(visibility_scope or "mine").strip().lower() or "mine"
    job_rows = (await db.execute(select(AgentJob))).scalars().all()
    if visibility_scope == "mine":
        job_rows = [job for job in job_rows if str(job.user_id) == str(current_user.id)]
    elif visibility_scope == "shared":
        job_rows = [job for job in job_rows if str(job.user_id) != str(current_user.id) and _is_job_visible_to_user(job, current_user)]
    else:
        job_rows = [job for job in job_rows if _is_job_visible_to_user(job, current_user)]

    backlog_rows = [item for item in (await db.execute(select(CodingBacklogItem))).scalars().all() if _is_backlog_item_visible_to_user(item, current_user)]

    preset_accumulators: dict[str, dict[str, Any]] = {
        key: {
            "preset_key": key,
            "launch_mode": meta["launch_mode"],
            "label": meta["label"],
            "total_runs": 0,
            "confidence_values": [],
            "high_confidence_runs": 0,
            "medium_confidence_runs": 0,
            "low_confidence_runs": 0,
            "auto_promoted_runs": 0,
            "review_needed_runs": 0,
            "tie_breaker_runs": 0,
            "manual_promotion_runs": 0,
            "repair_handoff_runs": 0,
            "backlog_handoff_runs": 0,
            "auto_backlog_handoff_runs": 0,
            "manual_backlog_handoff_runs": 0,
            "backlog_auto_suppressed_runs": 0,
        }
        for key, meta in _CODING_SWARM_ANALYTICS_PRESETS.items()
    }

    for job in job_rows:
        cfg = job.config if isinstance(job.config, dict) else {}
        launch_mode = _extract_launch_mode(cfg)
        if launch_mode not in {meta["launch_mode"] for meta in _CODING_SWARM_ANALYTICS_PRESETS.values()}:
            continue
        if source_id and str(cfg.get("source_id") or "") != str(source_id):
            continue
        if date_from and job.created_at and job.created_at < date_from:
            continue
        if date_to and job.created_at and job.created_at > date_to:
            continue
        row_preset = _infer_coding_swarm_preset_key(job)
        if not row_preset or (normalized_preset and row_preset != normalized_preset):
            continue
        acc = preset_accumulators[row_preset]
        acc["total_runs"] += 1
        swarm_summary = _extract_swarm_summary(job) or {}
        confidence = swarm_summary.get("confidence") if isinstance(swarm_summary.get("confidence"), dict) else {}
        overall = float(confidence.get("overall") or 0.0)
        acc["confidence_values"].append(overall)
        bucket = _swarm_confidence_bucket(overall)
        acc[f"{bucket}_confidence_runs"] += 1
        review_state = str(swarm_summary.get("review_state") or "").strip().lower()
        if review_state == "auto_promoted":
            acc["auto_promoted_runs"] += 1
        if bool(swarm_summary.get("review_required")) or review_state in {"needs_review", "insufficient_swarm_consensus", "consensus_failed"}:
            acc["review_needed_runs"] += 1
        if bool(swarm_summary.get("tie_breaker_attempted")) or str(swarm_summary.get("tie_breaker_job_id") or "").strip():
            acc["tie_breaker_runs"] += 1
        if review_state == "manual_promotion":
            acc["manual_promotion_runs"] += 1
        if str(swarm_summary.get("repair_chain_job_id") or "").strip():
            acc["repair_handoff_runs"] += 1
        if str(swarm_summary.get("backlog_auto_route_suppressed_reason") or "").strip():
            acc["backlog_auto_suppressed_runs"] += 1

    for item in backlog_rows:
        if date_from and item.created_at and item.created_at < date_from:
            continue
        if date_to and item.created_at and item.created_at > date_to:
            continue
        if source_id and str(getattr(item, "source_id", "") or "") != str(source_id):
            continue
        lineage = item.lineage if isinstance(item.lineage, dict) else {}
        row_preset = str(lineage.get("originating_swarm_preset") or "").strip().lower()
        if row_preset not in preset_accumulators:
            continue
        if normalized_preset and row_preset != normalized_preset:
            continue
        preset_accumulators[row_preset]["backlog_handoff_runs"] += 1
        route_mode = _extract_backlog_route_mode(item)
        if route_mode == "auto":
            preset_accumulators[row_preset]["auto_backlog_handoff_runs"] += 1
        else:
            preset_accumulators[row_preset]["manual_backlog_handoff_runs"] += 1

    preset_rows: list[AgentJobSwarmAnalyticsPresetRowResponse] = []
    for key in _CODING_SWARM_ANALYTICS_PRESETS.keys():
        acc = preset_accumulators[key]
        total_runs = int(acc["total_runs"] or 0)
        confidence_values = [float(v) for v in acc.pop("confidence_values", []) if isinstance(v, (int, float))]
        avg_confidence = (sum(confidence_values) / len(confidence_values)) if confidence_values else None
        promotion_rate = (float(acc["repair_handoff_runs"]) / total_runs) if total_runs > 0 else None
        review_rate = (float(acc["review_needed_runs"]) / total_runs) if total_runs > 0 else None
        tie_breaker_rate = (float(acc["tie_breaker_runs"]) / total_runs) if total_runs > 0 else None
        preset_rows.append(
            AgentJobSwarmAnalyticsPresetRowResponse(
                preset_key=key,
                launch_mode=str(acc["launch_mode"]),
                label=str(acc["label"]),
                total_runs=total_runs,
                avg_confidence=round(avg_confidence, 4) if avg_confidence is not None else None,
                high_confidence_runs=int(acc["high_confidence_runs"]),
                medium_confidence_runs=int(acc["medium_confidence_runs"]),
                low_confidence_runs=int(acc["low_confidence_runs"]),
                auto_promoted_runs=int(acc["auto_promoted_runs"]),
                review_needed_runs=int(acc["review_needed_runs"]),
                tie_breaker_runs=int(acc["tie_breaker_runs"]),
                manual_promotion_runs=int(acc["manual_promotion_runs"]),
                repair_handoff_runs=int(acc["repair_handoff_runs"]),
                backlog_handoff_runs=int(acc["backlog_handoff_runs"]),
                auto_backlog_handoff_runs=int(acc["auto_backlog_handoff_runs"]),
                manual_backlog_handoff_runs=int(acc["manual_backlog_handoff_runs"]),
                backlog_auto_suppressed_runs=int(acc["backlog_auto_suppressed_runs"]),
                promotion_rate=round(promotion_rate, 4) if promotion_rate is not None else None,
                review_rate=round(review_rate, 4) if review_rate is not None else None,
                tie_breaker_rate=round(tie_breaker_rate, 4) if tie_breaker_rate is not None else None,
            )
        )

    filtered_rows = [row for row in preset_rows if (not normalized_preset or row.preset_key == normalized_preset)]
    totals = {
        "total_runs": sum(row.total_runs for row in filtered_rows),
        "auto_promoted_runs": sum(row.auto_promoted_runs for row in filtered_rows),
        "review_needed_runs": sum(row.review_needed_runs for row in filtered_rows),
        "tie_breaker_runs": sum(row.tie_breaker_runs for row in filtered_rows),
        "repair_handoff_runs": sum(row.repair_handoff_runs for row in filtered_rows),
        "backlog_handoff_runs": sum(row.backlog_handoff_runs for row in filtered_rows),
        "auto_backlog_handoff_runs": sum(row.auto_backlog_handoff_runs for row in filtered_rows),
        "manual_backlog_handoff_runs": sum(row.manual_backlog_handoff_runs for row in filtered_rows),
        "backlog_auto_suppressed_runs": sum(row.backlog_auto_suppressed_runs for row in filtered_rows),
    }
    confidence_pool = [row.avg_confidence for row in filtered_rows if row.avg_confidence is not None]
    totals["avg_confidence"] = round(sum(confidence_pool) / len(confidence_pool), 4) if confidence_pool else None

    return AgentJobSwarmAnalyticsResponse(
        preset_rows=filtered_rows,
        totals=totals,
        filters={
            "source_id": str(source_id) if source_id else None,
            "preset_key": normalized_preset,
            "visibility_scope": visibility_scope,
            "date_from": date_from.isoformat() if date_from else None,
            "date_to": date_to.isoformat() if date_to else None,
        },
    )


@router.get("/swarm-outcomes", response_model=AgentJobSwarmOutcomeAnalyticsResponse)
async def get_swarm_outcomes(
    source_id: Optional[UUID] = Query(None),
    preset_key: Optional[str] = Query(None),
    terminal_outcome: Optional[str] = Query(None),
    promotion_mode: Optional[str] = Query(None),
    visibility_scope: str = Query("mine"),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    allowed_presets = set(_CODING_SWARM_ANALYTICS_PRESETS.keys())
    normalized_preset = str(preset_key or "").strip().lower() or None
    if normalized_preset and normalized_preset not in allowed_presets:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Unknown coding swarm preset")

    normalized_terminal_outcome = str(terminal_outcome or "").strip().lower() or None
    normalized_promotion_mode = str(promotion_mode or "").strip().lower() or None

    visibility_scope = str(visibility_scope or "mine").strip().lower() or "mine"
    job_rows = (await db.execute(select(AgentJob))).scalars().all()
    if visibility_scope == "mine":
        job_rows = [job for job in job_rows if str(job.user_id) == str(current_user.id)]
    elif visibility_scope == "shared":
        job_rows = [job for job in job_rows if str(job.user_id) != str(current_user.id) and _is_job_visible_to_user(job, current_user)]
    else:
        job_rows = [job for job in job_rows if _is_job_visible_to_user(job, current_user)]
    backlog_rows = [item for item in (await db.execute(select(CodingBacklogItem))).scalars().all() if _is_backlog_item_visible_to_user(item, current_user)]

    repair_jobs_by_id: dict[str, AgentJob] = {str(job.id): job for job in job_rows}
    swarm_jobs_by_id: dict[str, AgentJob] = {str(job.id): job for job in job_rows}
    backlog_by_swarm_job_id: dict[str, list[CodingBacklogItem]] = {}
    for item in backlog_rows:
        lineage = item.lineage if isinstance(item.lineage, dict) else {}
        swarm_job_id = str(lineage.get("originating_swarm_job_id") or "").strip()
        if not swarm_job_id:
            continue
        backlog_by_swarm_job_id.setdefault(swarm_job_id, []).append(item)
    collaboration_user_lookup = await _build_collaboration_user_lookup(db, current_user=current_user)

    cases: list[AgentJobSwarmOutcomeCaseResponse] = []
    for job in job_rows:
        cfg = job.config if isinstance(job.config, dict) else {}
        launch_mode = _extract_launch_mode(cfg)
        if launch_mode not in {meta["launch_mode"] for meta in _CODING_SWARM_ANALYTICS_PRESETS.values()}:
            continue
        if source_id and str(cfg.get("source_id") or "") != str(source_id):
            continue
        event_at = job.completed_at or job.last_activity_at or job.created_at
        if date_from and event_at and event_at < date_from:
            continue
        if date_to and event_at and event_at > date_to:
            continue
        row_preset = _infer_coding_swarm_preset_key(job)
        if not row_preset or (normalized_preset and row_preset != normalized_preset):
            continue
        case = _derive_swarm_outcome_case(
            job,
            repair_jobs_by_id=repair_jobs_by_id,
            backlog_by_swarm_job_id=backlog_by_swarm_job_id,
            current_user_id=str(current_user.id),
            user_lookup=collaboration_user_lookup,
        )
        if normalized_terminal_outcome and case.terminal_outcome != normalized_terminal_outcome:
            continue
        if normalized_promotion_mode and case.promotion_mode != normalized_promotion_mode:
            continue
        cases.append(case)

    cases.sort(
        key=lambda item: _datetime_sort_key(
            item.latest_downstream_at or item.repair_handoff_at or item.backlog_routed_at or item.swarm_completed_at
        ),
        reverse=True,
    )

    preset_accumulators: dict[str, dict[str, Any]] = {
        key: {
            "preset_key": key,
            "launch_mode": meta["launch_mode"],
            "label": meta["label"],
            "total_swarm_roots": 0,
            "auto_promoted_runs": 0,
            "manual_promoted_runs": 0,
            "tie_breaker_runs": 0,
            "repair_handoff_runs": 0,
            "verified_fix_runs": 0,
            "repair_failed_runs": 0,
            "backlog_routed_runs": 0,
            "auto_backlog_routed_runs": 0,
            "manual_backlog_routed_runs": 0,
            "backlog_auto_suppressed_runs": 0,
            "needs_review_runs": 0,
            "stalled_after_handoff_runs": 0,
            "confidence_values": [],
            "handoff_minutes": [],
        }
        for key, meta in _CODING_SWARM_ANALYTICS_PRESETS.items()
    }

    for case in cases:
        acc = preset_accumulators[case.preset_key]
        acc["total_swarm_roots"] += 1
        if case.promotion_mode == "auto":
            acc["auto_promoted_runs"] += 1
        elif case.promotion_mode == "manual":
            acc["manual_promoted_runs"] += 1
        if case.tie_breaker_attempted:
            acc["tie_breaker_runs"] += 1
        if case.repair_job_id:
            acc["repair_handoff_runs"] += 1
        if case.terminal_outcome == "verified_fix":
            acc["verified_fix_runs"] += 1
        elif case.terminal_outcome == "repair_failed":
            acc["repair_failed_runs"] += 1
        elif case.terminal_outcome == "backlog_routed":
            acc["backlog_routed_runs"] += 1
            if case.backlog_route_mode == "auto":
                acc["auto_backlog_routed_runs"] += 1
            else:
                acc["manual_backlog_routed_runs"] += 1
        elif case.terminal_outcome == "needs_review":
            acc["needs_review_runs"] += 1
        elif case.terminal_outcome == "stalled_after_handoff":
            acc["stalled_after_handoff_runs"] += 1
        source_job = swarm_jobs_by_id.get(case.swarm_job_id)
        source_summary = _extract_swarm_summary(source_job) if source_job is not None else {}
        if str((source_summary or {}).get("backlog_auto_route_suppressed_reason") or "").strip():
            acc["backlog_auto_suppressed_runs"] += 1
        if case.confidence_overall is not None:
            acc["confidence_values"].append(float(case.confidence_overall))
        if case.handoff_latency_minutes is not None:
            acc["handoff_minutes"].append(float(case.handoff_latency_minutes))

    preset_rows: list[AgentJobSwarmOutcomePresetRowResponse] = []
    for key in _CODING_SWARM_ANALYTICS_PRESETS.keys():
        acc = preset_accumulators[key]
        confidence_values = [float(v) for v in acc.pop("confidence_values", []) if isinstance(v, (int, float))]
        handoff_minutes = [float(v) for v in acc.pop("handoff_minutes", []) if isinstance(v, (int, float))]
        avg_confidence = (sum(confidence_values) / len(confidence_values)) if confidence_values else None
        avg_handoff_minutes = (sum(handoff_minutes) / len(handoff_minutes)) if handoff_minutes else None
        preset_rows.append(
            AgentJobSwarmOutcomePresetRowResponse(
                preset_key=key,
                launch_mode=str(acc["launch_mode"]),
                label=str(acc["label"]),
                total_swarm_roots=int(acc["total_swarm_roots"]),
                auto_promoted_runs=int(acc["auto_promoted_runs"]),
                manual_promoted_runs=int(acc["manual_promoted_runs"]),
                tie_breaker_runs=int(acc["tie_breaker_runs"]),
                repair_handoff_runs=int(acc["repair_handoff_runs"]),
                verified_fix_runs=int(acc["verified_fix_runs"]),
                repair_failed_runs=int(acc["repair_failed_runs"]),
                backlog_routed_runs=int(acc["backlog_routed_runs"]),
                auto_backlog_routed_runs=int(acc["auto_backlog_routed_runs"]),
                manual_backlog_routed_runs=int(acc["manual_backlog_routed_runs"]),
                backlog_auto_suppressed_runs=int(acc["backlog_auto_suppressed_runs"]),
                needs_review_runs=int(acc["needs_review_runs"]),
                stalled_after_handoff_runs=int(acc["stalled_after_handoff_runs"]),
                avg_confidence=round(avg_confidence, 4) if avg_confidence is not None else None,
                avg_handoff_minutes=round(avg_handoff_minutes, 2) if avg_handoff_minutes is not None else None,
            )
        )

    filtered_rows = [row for row in preset_rows if (not normalized_preset or row.preset_key == normalized_preset)]
    totals = {
        "total_swarm_roots": sum(row.total_swarm_roots for row in filtered_rows),
        "auto_promoted_runs": sum(row.auto_promoted_runs for row in filtered_rows),
        "manual_promoted_runs": sum(row.manual_promoted_runs for row in filtered_rows),
        "tie_breaker_runs": sum(row.tie_breaker_runs for row in filtered_rows),
        "repair_handoff_runs": sum(row.repair_handoff_runs for row in filtered_rows),
        "verified_fix_runs": sum(row.verified_fix_runs for row in filtered_rows),
        "repair_failed_runs": sum(row.repair_failed_runs for row in filtered_rows),
        "backlog_routed_runs": sum(row.backlog_routed_runs for row in filtered_rows),
        "auto_backlog_routed_runs": sum(row.auto_backlog_routed_runs for row in filtered_rows),
        "manual_backlog_routed_runs": sum(row.manual_backlog_routed_runs for row in filtered_rows),
        "backlog_auto_suppressed_runs": sum(row.backlog_auto_suppressed_runs for row in filtered_rows),
        "needs_review_runs": sum(row.needs_review_runs for row in filtered_rows),
        "stalled_after_handoff_runs": sum(row.stalled_after_handoff_runs for row in filtered_rows),
    }
    confidence_pool = [row.avg_confidence for row in filtered_rows if row.avg_confidence is not None]
    handoff_pool = [row.avg_handoff_minutes for row in filtered_rows if row.avg_handoff_minutes is not None]
    totals["avg_confidence"] = round(sum(confidence_pool) / len(confidence_pool), 4) if confidence_pool else None
    totals["avg_handoff_minutes"] = round(sum(handoff_pool) / len(handoff_pool), 2) if handoff_pool else None

    return AgentJobSwarmOutcomeAnalyticsResponse(
        preset_rows=filtered_rows,
        cases=cases[:200],
        totals=totals,
        filters={
            "source_id": str(source_id) if source_id else None,
            "preset_key": normalized_preset,
            "terminal_outcome": normalized_terminal_outcome,
            "promotion_mode": normalized_promotion_mode,
            "visibility_scope": visibility_scope,
            "date_from": _iso_or_none(date_from),
            "date_to": _iso_or_none(date_to),
        },
    )


@router.get("/checkpoint-queue", response_model=AgentCheckpointQueueResponse)
async def get_checkpoint_queue(
    item_type: Optional[str] = Query(None, description="Filter by queue item type"),
    status: Optional[str] = Query(None, description="Filter by queue item/job status"),
    customer: Optional[str] = Query(None, description="Filter by customer"),
    job_type: Optional[str] = Query(None, description="Filter by job type"),
    sla_bucket: Optional[str] = Query(None, description="Filter by SLA bucket"),
    escalation_level: Optional[str] = Query(None, description="Filter by escalation level"),
    overdue_only: bool = Query(False, description="Only include overdue queue items"),
    sort_by: str = Query("priority_score_desc", description="Sort mode: priority_score_desc|sla_desc|age_desc|priority_desc|created_desc|created_asc"),
    limit: int = Query(100, ge=1, le=300, description="Maximum queue items to return"),
    offset: int = Query(0, ge=0, description="Queue offset"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Build an operator-first queue from existing jobs and accepted inbox items.

    This keeps approvals, recurring job recoveries, and accepted-signal follow-up
    recommendations in one review surface without requiring a new persistence model.
    """
    jobs_result = await db.execute(
        select(AgentJob)
        .options(selectinload(AgentJob.agent_definition))
        .where(AgentJob.user_id == current_user.id)
        .order_by(AgentJob.created_at.desc())
    )
    jobs = list(jobs_result.scalars().all())

    inbox_result = await db.execute(
        select(ResearchInboxItem)
        .where(
            and_(
                ResearchInboxItem.user_id == current_user.id,
                ResearchInboxItem.status == "accepted",
            )
        )
        .order_by(ResearchInboxItem.updated_at.desc(), ResearchInboxItem.discovered_at.desc())
    )
    inbox_items = list(inbox_result.scalars().all())

    portfolios_result = await db.execute(
        select(ResearchPortfolio)
        .where(ResearchPortfolio.user_id == current_user.id)
        .order_by(ResearchPortfolio.updated_at.desc())
    )
    portfolios = list(portfolios_result.scalars().all())
    profiles_result = await db.execute(
        select(DomainResearchProfile)
        .where(DomainResearchProfile.user_id == current_user.id)
        .order_by(DomainResearchProfile.updated_at.desc())
    )
    profiles = list(profiles_result.scalars().all())

    learning_profiles: dict[str, dict[str, Any]] = {}
    customers = sorted(
        {
            str(item.customer or "").strip()
            for item in inbox_items
        }
    )
    for customer_key in customers:
        learning_profiles[_customer_profile_key(customer_key)] = await _load_follow_up_learning_profile(
            db=db,
            user_id=current_user.id,
            customer=customer_key or None,
        )

    items_all = _build_checkpoint_queue_items(
        jobs,
        inbox_items,
        portfolios,
        profiles,
        learning_profiles=learning_profiles,
        monitor_health_rows=research_monitor_profile_service.build_effectiveness_snapshot(
            items=inbox_items,
            jobs_by_id={job.id: job for job in jobs if job.id is not None},
        ).get("monitors", []),
    )

    by_type = Counter(str(row.item_type or "").strip() or "unknown" for row in items_all)
    by_status = Counter(str(row.status or "").strip() or "unknown" for row in items_all)
    by_customer = Counter(str(row.customer or "").strip() or "Unassigned" for row in items_all)
    by_sla_bucket = Counter(str(row.sla_bucket or "").strip() or "unknown" for row in items_all)
    by_escalation_level = Counter(str(row.escalation_level or "").strip() or "unknown" for row in items_all)

    item_type_filter = str(item_type or "").strip().lower()
    status_filter = str(status or "").strip().lower()
    customer_filter = str(customer or "").strip().lower()
    job_type_filter = str(job_type or "").strip().lower()
    sla_bucket_filter = str(sla_bucket or "").strip().lower()
    escalation_level_filter = str(escalation_level or "").strip().lower()

    items_filtered = []
    for row in items_all:
        if item_type_filter and str(row.item_type or "").strip().lower() != item_type_filter:
            continue
        if status_filter and str(row.status or "").strip().lower() != status_filter:
            continue
        if customer_filter and str(row.customer or "").strip().lower() != customer_filter:
            continue
        if job_type_filter and str(row.job_type or "").strip().lower() != job_type_filter:
            continue
        if sla_bucket_filter and str(row.sla_bucket or "").strip().lower() != sla_bucket_filter:
            continue
        if escalation_level_filter and str(row.escalation_level or "").strip().lower() != escalation_level_filter:
            continue
        if overdue_only and not bool(row.is_overdue):
            continue
        items_filtered.append(row)

    sort_mode = str(sort_by or "priority_desc").strip().lower()
    if sort_mode == "created_asc":
        items_filtered.sort(key=lambda row: (row.created_at.timestamp() if row.created_at else 0.0, -int(row.priority or 0)))
    elif sort_mode == "created_desc":
        items_filtered.sort(key=lambda row: (row.created_at.timestamp() if row.created_at else 0.0, int(row.priority or 0)), reverse=True)
    elif sort_mode == "age_desc":
        items_filtered.sort(key=lambda row: (int(row.age_minutes or 0), float(row.priority_score or 0)), reverse=True)
    elif sort_mode == "sla_desc":
        sla_rank = {"overdue": 3, "at_risk": 2, "normal": 1}
        esc_rank = {"high": 3, "medium": 2, "normal": 1}
        items_filtered.sort(
            key=lambda row: (
                sla_rank.get(str(row.sla_bucket or ""), 0),
                esc_rank.get(str(row.escalation_level or ""), 0),
                bool(row.is_overdue),
                float(row.priority_score or 0),
                int(row.age_minutes or 0),
            ),
            reverse=True,
        )
    elif sort_mode == "priority_score_desc":
        items_filtered.sort(
            key=lambda row: (
                float(row.priority_score or 0),
                bool(row.is_overdue),
                int(row.age_minutes or 0),
                row.created_at.timestamp() if row.created_at else 0.0,
            ),
            reverse=True,
        )
    else:
        items_filtered.sort(
            key=lambda row: (
                int(row.priority or 0),
                float(row.priority_score or 0),
                row.created_at.timestamp() if row.created_at else 0.0,
            ),
            reverse=True,
        )

    total = len(items_filtered)
    items = items_filtered[offset : offset + limit]
    return AgentCheckpointQueueResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        approvals=sum(1 for row in items_filtered if row.item_type == "approval_checkpoint"),
        recoveries=sum(1 for row in items_filtered if row.item_type == "job_recovery"),
        follow_ups=sum(1 for row in items_filtered if row.item_type == "follow_up_recommendation"),
        policy_reviews=sum(1 for row in items_filtered if row.item_type == "policy_review"),
        budget_reviews=sum(1 for row in items_filtered if row.item_type == "budget_review"),
        by_type=dict(by_type),
        by_status=dict(by_status),
        by_customer=dict(by_customer),
        by_sla_bucket=dict(by_sla_bucket),
        by_escalation_level=dict(by_escalation_level),
    )


@router.get("/decision-trace", response_model=AgentDecisionTraceResponse)
async def get_decision_trace(
    source_kind: Optional[str] = Query(None, description="Filter by event source kind"),
    decision_type: Optional[str] = Query(None, description="Filter by normalized decision type"),
    customer: Optional[str] = Query(None, description="Filter by customer"),
    status: Optional[str] = Query(None, description="Filter by derived event status"),
    severity: Optional[str] = Query(None, description="Filter by event severity"),
    actor_mode: Optional[str] = Query(None, description="Filter by actor mode: operator|autonomous"),
    triage_status: Optional[str] = Query(None, description="Filter by operator triage status"),
    assigned_to_user_id: Optional[UUID] = Query(None, description="Filter by assignee"),
    unassigned_only: bool = Query(False, description="Only include unassigned persisted events"),
    escalation_state: Optional[str] = Query(None, description="Filter by escalation state"),
    pinned: Optional[bool] = Query(None, description="Filter by pinned state for persisted events"),
    actionable_only: bool = Query(False, description="Only include persisted actionable events"),
    start_at: Optional[datetime] = Query(None, description="Only include events at or after this time"),
    end_at: Optional[datetime] = Query(None, description="Only include events at or before this time"),
    limit: int = Query(100, ge=1, le=300, description="Maximum decision trace items to return"),
    offset: int = Query(0, ge=0, description="Decision trace offset"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    source_kind_filter = str(source_kind or "").strip().lower()
    decision_type_filter = str(decision_type or "").strip().lower()
    customer_filter = str(customer or "").strip().lower()
    status_filter = str(status or "").strip().lower()
    severity_filter = str(severity or "").strip().lower()
    actor_mode_filter = str(actor_mode or "").strip().lower()
    triage_status_filter = str(triage_status or "").strip().lower()
    escalation_state_filter = str(escalation_state or "").strip().lower()

    visible_user_ids = await _list_trace_visible_user_ids(db, current_user=current_user)
    if current_user.id not in visible_user_ids:
        visible_user_ids.add(current_user.id)
    visible_users = list(
        (
            await db.execute(
                select(User).where(User.id.in_(visible_user_ids))
            )
        ).scalars().all()
    )
    user_lookup = {str(user.id): user for user in visible_users}

    query = select(AutonomyDecisionEvent).where(AutonomyDecisionEvent.user_id.in_(visible_user_ids))
    if source_kind_filter:
        query = query.where(func.lower(AutonomyDecisionEvent.source_kind) == source_kind_filter)
    if decision_type_filter:
        query = query.where(func.lower(AutonomyDecisionEvent.decision_type) == decision_type_filter)
    if customer_filter:
        query = query.where(func.lower(func.coalesce(AutonomyDecisionEvent.customer, "")) == customer_filter)
    if status_filter:
        query = query.where(func.lower(func.coalesce(AutonomyDecisionEvent.status, "")) == status_filter)
    if severity_filter:
        query = query.where(func.lower(func.coalesce(AutonomyDecisionEvent.severity, "")) == severity_filter)
    if actor_mode_filter:
        query = query.where(func.lower(func.coalesce(AutonomyDecisionEvent.actor_mode, "")) == actor_mode_filter)
    if triage_status_filter:
        query = query.where(func.lower(func.coalesce(AutonomyDecisionEvent.triage_status, "")) == triage_status_filter)
    if assigned_to_user_id is not None:
        query = query.where(AutonomyDecisionEvent.assigned_to_user_id == assigned_to_user_id)
    if unassigned_only:
        query = query.where(AutonomyDecisionEvent.assigned_to_user_id.is_(None))
    if escalation_state_filter:
        query = query.where(func.lower(func.coalesce(AutonomyDecisionEvent.escalation_state, "")) == escalation_state_filter)
    if pinned is not None:
        query = query.where(AutonomyDecisionEvent.pinned == bool(pinned))
    if start_at is not None:
        query = query.where(AutonomyDecisionEvent.event_time >= start_at)
    if end_at is not None:
        query = query.where(AutonomyDecisionEvent.event_time <= end_at)
    query = query.order_by(AutonomyDecisionEvent.event_time.desc(), AutonomyDecisionEvent.created_at.desc())
    persisted_rows = list((await db.execute(query)).scalars().all())
    escalation_mutated = False
    for row in persisted_rows:
        previous_escalation_state = str(row.escalation_state or "none").strip().lower() or "none"
        apply_decision_trace_escalation(row)
        if str(row.escalation_state or "none").strip().lower() != previous_escalation_state:
            escalation_mutated = True
            await maybe_emit_escalation_transition_notification(db, row, previous_state=previous_escalation_state)
    if escalation_mutated:
        await db.commit()

    events: list[AgentDecisionTraceEventResponse] = [
        AgentDecisionTraceEventResponse.model_validate(
            _decorate_trace_event_payload(
                event_to_trace_payload(row),
                user_lookup=user_lookup,
                current_user_id=current_user.id,
            )
        )
        for row in persisted_rows
    ]

    persisted_source_kinds = {
        str(row.source_kind or "").strip().lower()
        for row in persisted_rows
        if str(row.source_kind or "").strip()
    }
    need_derived_fallback = (not actionable_only) and (not persisted_rows or not source_kind_filter)
    if source_kind_filter and source_kind_filter not in persisted_source_kinds:
        need_derived_fallback = not actionable_only

    if need_derived_fallback:
        jobs_result = await db.execute(
            select(AgentJob)
            .options(selectinload(AgentJob.agent_definition))
            .where(AgentJob.user_id == current_user.id)
            .order_by(
                AgentJob.last_activity_at.desc(),
                AgentJob.completed_at.desc(),
                AgentJob.started_at.desc(),
                AgentJob.created_at.desc(),
            )
        )
        jobs = list(jobs_result.scalars().all())

        inbox_result = await db.execute(
            select(ResearchInboxItem)
            .where(ResearchInboxItem.user_id == current_user.id)
            .order_by(ResearchInboxItem.updated_at.desc(), ResearchInboxItem.discovered_at.desc())
        )
        inbox_items = list(inbox_result.scalars().all())

        portfolios_result = await db.execute(
            select(ResearchPortfolio)
            .where(ResearchPortfolio.user_id == current_user.id)
            .order_by(ResearchPortfolio.updated_at.desc())
        )
        portfolios = list(portfolios_result.scalars().all())

        profiles_result = await db.execute(
            select(DomainResearchProfile)
            .where(DomainResearchProfile.user_id == current_user.id)
            .order_by(DomainResearchProfile.updated_at.desc())
        )
        profiles = list(profiles_result.scalars().all())

        runs_result = await db.execute(
            select(ExperimentRun)
            .where(ExperimentRun.user_id == current_user.id)
            .order_by(ExperimentRun.updated_at.desc(), ExperimentRun.created_at.desc())
        )
        validation_runs = list(runs_result.scalars().all())

        learning_profiles: dict[str, dict[str, Any]] = {}
        customers = sorted({str(item.customer or "").strip() for item in inbox_items if str(item.customer or "").strip()})
        for customer_key in customers:
            learning_profiles[_customer_profile_key(customer_key)] = await _load_follow_up_learning_profile(
                db=db,
                user_id=current_user.id,
                customer=customer_key or None,
            )

        monitor_snapshot = research_monitor_profile_service.build_effectiveness_snapshot(
            items=inbox_items,
            jobs_by_id={job.id: job for job in jobs if job.id is not None},
        )
        queue_items = _build_checkpoint_queue_items(
            jobs,
            [item for item in inbox_items if str(item.status or "").strip().lower() == "accepted"],
            portfolios,
            profiles,
            learning_profiles=learning_profiles,
            monitor_health_rows=monitor_snapshot.get("monitors", []),
        )

        derived_events: list[AgentDecisionTraceEventResponse] = []
        derived_events.extend(_build_decision_trace_from_queue_items(queue_items))
        for job in jobs:
            derived_events.extend(_build_decision_trace_from_job(job))
        for portfolio in portfolios:
            payload = _portfolio_summary_payload(portfolio)
            derived_events.extend(
                _build_decision_trace_from_opportunities(
                    source_kind="portfolio",
                    source_id=str(portfolio.id),
                    source_label=str(portfolio.title or "Research fleet").strip(),
                    customer=None,
                    opportunities=payload["opportunities"],
                    deep_link_params={"tab": "fleet", "fleetId": str(portfolio.id)},
                    objective=str(portfolio.objective or "").strip() or None,
                    sandbox_profile_id=str(portfolio.sandbox_profile_id or "").strip() or None,
                    automation_profile=str(portfolio.automation_profile or "").strip() or None,
                    effective_policy=payload["summary"].get("effective_policy") if isinstance(payload["summary"], dict) else None,
                )
            )
        for profile in profiles:
            payload = _profile_summary_payload(profile)
            derived_events.extend(
                _build_decision_trace_from_opportunities(
                    source_kind="domain_profile",
                    source_id=str(profile.id),
                    source_label=str(profile.title or "Domain profile").strip(),
                    customer=str(profile.customer_context or "").strip() or None,
                    opportunities=payload["opportunities"],
                    deep_link_params={"tab": "domain"},
                    domain=str(profile.domain or "").strip() or None,
                    objective=str(profile.objective or "").strip() or None,
                    track_type=str(profile.track_type or "").strip() or None,
                    source_scope=str(profile.source_scope or "").strip() or None,
                    repo_source_ids=profile.repo_source_ids,
                    benchmark_queries=profile.benchmark_queries,
                    sandbox_profile_id=str(profile.sandbox_profile_id or "").strip() or None,
                    automation_profile=str(profile.automation_profile or "").strip() or None,
                    effective_policy=payload["summary"].get("effective_policy") if isinstance(payload["summary"], dict) else None,
                )
            )
        derived_events.extend(_build_decision_trace_from_monitor_snapshot(monitor_snapshot))
        derived_events.extend(_build_decision_trace_from_validation_runs(validation_runs))

        for item in derived_events:
            normalized_kind = str(item.source_kind or "").strip().lower()
            if source_kind_filter:
                if normalized_kind != source_kind_filter or normalized_kind in persisted_source_kinds:
                    continue
            elif normalized_kind in persisted_source_kinds:
                continue
            events.append(
                item.model_copy(
                    update={
                        "is_derived": True,
                        "record_origin": "derived_fallback",
                    }
                )
            )

    filtered_items: list[AgentDecisionTraceEventResponse] = []
    for item in events:
        if source_kind_filter and str(item.source_kind or "").strip().lower() != source_kind_filter:
            continue
        if decision_type_filter and str(item.decision_type or "").strip().lower() != decision_type_filter:
            continue
        if customer_filter and str(item.customer or "").strip().lower() != customer_filter:
            continue
        if status_filter and str(item.status or "").strip().lower() != status_filter:
            continue
        if severity_filter and str(item.severity or "").strip().lower() != severity_filter:
            continue
        if actor_mode_filter and str(item.actor_mode or "").strip().lower() != actor_mode_filter:
            continue
        if triage_status_filter and str(item.triage_status or "").strip().lower() != triage_status_filter:
            continue
        if assigned_to_user_id is not None and str(item.assigned_to_user_id or "").strip() != str(assigned_to_user_id):
            continue
        if unassigned_only and item.assigned_to_user_id:
            continue
        if escalation_state_filter and str(item.escalation_state or "").strip().lower() != escalation_state_filter:
            continue
        if pinned is not None and bool(item.pinned) != bool(pinned):
            continue
        if actionable_only and bool(item.is_derived):
            continue
        if start_at is not None and item.event_time < start_at:
            continue
        if end_at is not None and item.event_time > end_at:
            continue
        filtered_items.append(item)

    filtered_items.sort(
        key=lambda row: (
            row.event_time.timestamp() if row.event_time else 0.0,
            str(row.event_id or ""),
        ),
        reverse=True,
    )
    total = len(filtered_items)
    by_source_kind = Counter(str(item.source_kind or "").strip() or "unknown" for item in filtered_items)
    by_decision_type = Counter(str(item.decision_type or "").strip() or "unknown" for item in filtered_items)
    by_status = Counter(str(item.status or "").strip() or "unknown" for item in filtered_items)
    by_customer = Counter(str(item.customer or "").strip() or "Unassigned" for item in filtered_items)
    by_severity = Counter(str(item.severity or "").strip() or "unknown" for item in filtered_items)
    by_actor_mode = Counter(str(item.actor_mode or "").strip() or "unknown" for item in filtered_items)
    by_triage_status = Counter(str(item.triage_status or "").strip() or "unknown" for item in filtered_items)
    by_assignee = Counter(str(item.assigned_to_user_id or "").strip() or "unassigned" for item in filtered_items)
    by_escalation_state = Counter(str(item.escalation_state or "").strip() or "none" for item in filtered_items)
    overdue_count = sum(
        1
        for item in filtered_items
        if item.due_at
        and str(item.triage_status or "").strip().lower() != "resolved"
        and item.due_at <= datetime.utcnow()
    )
    items = filtered_items[offset : offset + limit]

    return AgentDecisionTraceResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        by_source_kind=dict(by_source_kind),
        by_decision_type=dict(by_decision_type),
        by_status=dict(by_status),
        by_customer=dict(by_customer),
        by_severity=dict(by_severity),
        by_actor_mode=dict(by_actor_mode),
        by_triage_status=dict(by_triage_status),
        by_assignee=dict(by_assignee),
        by_escalation_state=dict(by_escalation_state),
        overdue_count=overdue_count,
        has_more=offset + len(items) < total,
    )


def _trace_analytics_bucket_rows(counter: Counter[str], *, limit: int = 5) -> list[AgentDecisionTraceAnalyticsBucketResponse]:
    return [
        AgentDecisionTraceAnalyticsBucketResponse(value=value, count=count)
        for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


async def _load_full_decision_trace_events(
    *,
    source_kind: Optional[str],
    decision_type: Optional[str],
    customer: Optional[str],
    status: Optional[str],
    severity: Optional[str],
    actor_mode: Optional[str],
    triage_status: Optional[str],
    assigned_to_user_id: Optional[UUID],
    unassigned_only: bool,
    escalation_state: Optional[str],
    pinned: Optional[bool],
    actionable_only: bool,
    start_at: Optional[datetime],
    end_at: Optional[datetime],
    db: AsyncSession,
    current_user: User,
    page_size: int = 300,
) -> list[AgentDecisionTraceEventResponse]:
    offset = 0
    collected: list[AgentDecisionTraceEventResponse] = []
    while True:
        page = await get_decision_trace(
            source_kind=source_kind,
            decision_type=decision_type,
            customer=customer,
            status=status,
            severity=severity,
            actor_mode=actor_mode,
            triage_status=triage_status,
            assigned_to_user_id=assigned_to_user_id,
            unassigned_only=unassigned_only,
            escalation_state=escalation_state,
            pinned=pinned,
            actionable_only=actionable_only,
            start_at=start_at,
            end_at=end_at,
            limit=page_size,
            offset=offset,
            db=db,
            current_user=current_user,
        )
        collected.extend(page.items)
        if not page.has_more or not page.items:
            break
        offset += page.limit
    return collected


@router.get("/decision-trace/export")
async def export_decision_trace(
    format: str = Query("json", description="Export format: json or csv"),
    source_kind: Optional[str] = Query(None, description="Filter by event source kind"),
    decision_type: Optional[str] = Query(None, description="Filter by normalized decision type"),
    customer: Optional[str] = Query(None, description="Filter by customer"),
    status: Optional[str] = Query(None, description="Filter by derived event status"),
    severity: Optional[str] = Query(None, description="Filter by event severity"),
    actor_mode: Optional[str] = Query(None, description="Filter by actor mode: operator|autonomous"),
    triage_status: Optional[str] = Query(None, description="Filter by operator triage status"),
    assigned_to_user_id: Optional[UUID] = Query(None, description="Filter by assignee"),
    unassigned_only: bool = Query(False, description="Only include unassigned persisted events"),
    escalation_state: Optional[str] = Query(None, description="Filter by escalation state"),
    pinned: Optional[bool] = Query(None, description="Filter by pinned state for persisted events"),
    actionable_only: bool = Query(False, description="Only include persisted actionable events"),
    start_at: Optional[datetime] = Query(None, description="Only include events at or after this time"),
    end_at: Optional[datetime] = Query(None, description="Only include events at or before this time"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    export_format = str(format or "json").strip().lower()
    if export_format not in {"json", "csv"}:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Unsupported export format")

    items = await _load_full_decision_trace_events(
        source_kind=source_kind,
        decision_type=decision_type,
        customer=customer,
        status=status,
        severity=severity,
        actor_mode=actor_mode,
        triage_status=triage_status,
        assigned_to_user_id=assigned_to_user_id,
        unassigned_only=unassigned_only,
        escalation_state=escalation_state,
        pinned=pinned,
        actionable_only=actionable_only,
        start_at=start_at,
        end_at=end_at,
        db=db,
        current_user=current_user,
    )

    if export_format == "json":
        by_source_kind = Counter(str(item.source_kind or "").strip() or "unknown" for item in items)
        by_decision_type = Counter(str(item.decision_type or "").strip() or "unknown" for item in items)
        by_status = Counter(str(item.status or "").strip() or "unknown" for item in items)
        by_customer = Counter(str(item.customer or "").strip() or "Unassigned" for item in items)
        by_severity = Counter(str(item.severity or "").strip() or "unknown" for item in items)
        by_actor_mode = Counter(str(item.actor_mode or "").strip() or "unknown" for item in items)
        by_triage_status = Counter(str(item.triage_status or "").strip() or "unknown" for item in items)
        by_assignee = Counter(str(item.assigned_to_user_id or "").strip() or "unassigned" for item in items)
        by_escalation_state = Counter(str(item.escalation_state or "").strip() or "none" for item in items)
        overdue_count = sum(
            1
            for item in items
            if item.due_at
            and str(item.triage_status or "").strip().lower() != "resolved"
            and item.due_at <= datetime.utcnow()
        )
        payload = AgentDecisionTraceResponse(
            items=items,
            total=len(items),
            limit=len(items),
            offset=0,
            by_source_kind=dict(by_source_kind),
            by_decision_type=dict(by_decision_type),
            by_status=dict(by_status),
            by_customer=dict(by_customer),
            by_severity=dict(by_severity),
            by_actor_mode=dict(by_actor_mode),
            by_triage_status=dict(by_triage_status),
            by_assignee=dict(by_assignee),
            by_escalation_state=dict(by_escalation_state),
            overdue_count=overdue_count,
            has_more=False,
        )
        return Response(
            content=json.dumps(payload.model_dump(mode="json"), ensure_ascii=False),
            media_type="application/json",
        )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"decision_trace_export_{timestamp}.csv"
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "event_id",
            "event_time",
            "event_type",
            "source_kind",
            "source_id",
            "source_label",
            "customer",
            "decision_type",
            "reason_code",
            "reason_label",
            "status",
            "severity",
            "actor_mode",
            "summary",
            "operator_note",
            "triage_status",
            "pinned",
            "is_derived",
            "record_origin",
            "scheduler_state",
            "metadata",
            "before_state",
            "after_state",
            "deep_link",
            "team_bucket",
            "due_at",
            "escalation_state",
            "escalation_reason",
            "escalated_at",
        ],
    )
    writer.writeheader()
    for item in items:
        row = item.model_dump(mode="json")
        writer.writerow(
            {
                "event_id": row.get("event_id"),
                "event_time": row.get("event_time"),
                "event_type": row.get("event_type"),
                "source_kind": row.get("source_kind"),
                "source_id": row.get("source_id"),
                "source_label": row.get("source_label"),
                "customer": row.get("customer"),
                "decision_type": row.get("decision_type"),
                "reason_code": row.get("reason_code"),
                "reason_label": row.get("reason_label"),
                "status": row.get("status"),
                "severity": row.get("severity"),
                "actor_mode": row.get("actor_mode"),
                "summary": row.get("summary"),
                "operator_note": row.get("operator_note"),
                "triage_status": row.get("triage_status"),
                "pinned": row.get("pinned"),
                "is_derived": row.get("is_derived"),
                "record_origin": row.get("record_origin"),
                "scheduler_state": json.dumps(row.get("scheduler_state"), ensure_ascii=False),
                "metadata": json.dumps(row.get("metadata"), ensure_ascii=False),
                "before_state": json.dumps(row.get("before_state"), ensure_ascii=False),
                "after_state": json.dumps(row.get("after_state"), ensure_ascii=False),
                "deep_link": json.dumps(row.get("deep_link"), ensure_ascii=False),
                "team_bucket": row.get("team_bucket"),
                "due_at": row.get("due_at"),
                "escalation_state": row.get("escalation_state"),
                "escalation_reason": row.get("escalation_reason"),
                "escalated_at": row.get("escalated_at"),
            }
        )

    return Response(
        content=buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/decision-trace/analytics", response_model=AgentDecisionTraceAnalyticsResponse)
async def get_decision_trace_analytics(
    source_kind: Optional[str] = Query(None, description="Filter by event source kind"),
    decision_type: Optional[str] = Query(None, description="Filter by normalized decision type"),
    customer: Optional[str] = Query(None, description="Filter by customer"),
    status: Optional[str] = Query(None, description="Filter by derived event status"),
    severity: Optional[str] = Query(None, description="Filter by event severity"),
    actor_mode: Optional[str] = Query(None, description="Filter by actor mode: operator|autonomous"),
    triage_status: Optional[str] = Query(None, description="Filter by operator triage status"),
    assigned_to_user_id: Optional[UUID] = Query(None, description="Filter by assignee"),
    unassigned_only: bool = Query(False, description="Only include unassigned persisted events"),
    escalation_state: Optional[str] = Query(None, description="Filter by escalation state"),
    pinned: Optional[bool] = Query(None, description="Filter by pinned state for persisted events"),
    actionable_only: bool = Query(False, description="Only include persisted actionable events"),
    start_at: Optional[datetime] = Query(None, description="Only include events at or after this time"),
    end_at: Optional[datetime] = Query(None, description="Only include events at or before this time"),
    days: int = Query(7, ge=1, le=30, description="Trend window in days"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    source_kind_filter = str(source_kind or "").strip().lower()
    decision_type_filter = str(decision_type or "").strip().lower()
    customer_filter = str(customer or "").strip().lower()
    status_filter = str(status or "").strip().lower()
    severity_filter = str(severity or "").strip().lower()
    actor_mode_filter = str(actor_mode or "").strip().lower()
    triage_status_filter = str(triage_status or "").strip().lower()
    escalation_state_filter = str(escalation_state or "").strip().lower()

    visible_user_ids = await _list_trace_visible_user_ids(db, current_user=current_user)
    if current_user.id not in visible_user_ids:
        visible_user_ids.add(current_user.id)

    query = select(AutonomyDecisionEvent).where(AutonomyDecisionEvent.user_id.in_(visible_user_ids))
    if source_kind_filter:
        query = query.where(func.lower(AutonomyDecisionEvent.source_kind) == source_kind_filter)
    if decision_type_filter:
        query = query.where(func.lower(AutonomyDecisionEvent.decision_type) == decision_type_filter)
    if customer_filter:
        query = query.where(func.lower(func.coalesce(AutonomyDecisionEvent.customer, "")) == customer_filter)
    if status_filter:
        query = query.where(func.lower(func.coalesce(AutonomyDecisionEvent.status, "")) == status_filter)
    if severity_filter:
        query = query.where(func.lower(func.coalesce(AutonomyDecisionEvent.severity, "")) == severity_filter)
    if actor_mode_filter:
        query = query.where(func.lower(func.coalesce(AutonomyDecisionEvent.actor_mode, "")) == actor_mode_filter)
    if triage_status_filter:
        query = query.where(func.lower(func.coalesce(AutonomyDecisionEvent.triage_status, "")) == triage_status_filter)
    if assigned_to_user_id is not None:
        query = query.where(AutonomyDecisionEvent.assigned_to_user_id == assigned_to_user_id)
    if unassigned_only:
        query = query.where(AutonomyDecisionEvent.assigned_to_user_id.is_(None))
    if escalation_state_filter:
        query = query.where(func.lower(func.coalesce(AutonomyDecisionEvent.escalation_state, "")) == escalation_state_filter)
    if pinned is not None:
        query = query.where(AutonomyDecisionEvent.pinned == bool(pinned))
    if start_at is not None:
        query = query.where(AutonomyDecisionEvent.event_time >= start_at)
    if end_at is not None:
        query = query.where(AutonomyDecisionEvent.event_time <= end_at)
    query = query.order_by(AutonomyDecisionEvent.event_time.desc(), AutonomyDecisionEvent.created_at.desc())

    rows = list((await db.execute(query)).scalars().all())

    by_source_kind = Counter()
    by_triage_status = Counter()
    by_decision_type = Counter()
    by_reason_label = Counter()
    by_queue_reason = Counter()
    daily_counts: Counter[str] = Counter()

    for row in rows:
        payload = event_to_trace_payload(row)
        by_source_kind[str(payload["source_kind"] or "").strip() or "unknown"] += 1
        by_triage_status[str(payload["triage_status"] or "").strip() or "unknown"] += 1
        by_decision_type[str(payload["decision_type"] or "").strip() or "unknown"] += 1
        by_reason_label[str(payload["reason_label"] or "").strip() or "unknown"] += 1
        scheduler_state = payload.get("scheduler_state")
        queue_reason = (
            str((scheduler_state or {}).get("queue_reason") or "").strip()
            if isinstance(scheduler_state, dict)
            else ""
        )
        by_queue_reason[queue_reason or "unknown"] += 1
        event_day = (
            payload["event_time"].date().isoformat()
            if isinstance(payload.get("event_time"), datetime)
            else datetime.utcnow().date().isoformat()
        )
        daily_counts[event_day] += 1

    today = datetime.utcnow().date()
    daily_trend = [
        AgentDecisionTraceAnalyticsTrendPointResponse(
            day=(today - timedelta(days=days - 1 - index)).isoformat(),
            count=int(daily_counts.get((today - timedelta(days=days - 1 - index)).isoformat(), 0) or 0),
        )
        for index in range(days)
    ]

    return AgentDecisionTraceAnalyticsResponse(
        window_days=days,
        total=len(rows),
        by_source_kind=dict(sorted(by_source_kind.items(), key=lambda item: (-item[1], item[0]))),
        by_triage_status=dict(sorted(by_triage_status.items(), key=lambda item: (-item[1], item[0]))),
        top_decision_types=_trace_analytics_bucket_rows(by_decision_type),
        top_reason_labels=_trace_analytics_bucket_rows(by_reason_label),
        top_queue_reasons=_trace_analytics_bucket_rows(by_queue_reason),
        daily_trend=daily_trend,
    )


@router.post("/decision-trace/{event_id}/action", response_model=AgentDecisionTraceActionResponse)
async def act_on_decision_trace_event(
    event_id: UUID,
    request: AgentDecisionTraceActionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    action = str(request.action or "").strip().lower()
    if action not in TRACE_TRIAGE_ACTIONS:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Unsupported decision trace action")

    event = await _load_persisted_trace_event_for_user(db, event_id=event_id, current_user=current_user)
    if event is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Decision trace event not found")
    if bool(event.is_derived):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Derived fallback events are read-only")

    now = datetime.utcnow()
    note = str(request.note or "").strip() or None
    current_status = str(event.triage_status or "new").strip().lower() or "new"
    previous_escalation_state = str(compute_decision_trace_escalation(event)[0] or "none").strip().lower() or "none"

    if action in {"approve_launch", "reject_launch"}:
        source_kind, source_id, opportunity_id = _trace_event_follow_up_target(event)
        try:
            owner_id = UUID(source_id)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Decision trace event has an invalid follow-up owner identifier",
            ) from exc
        response: AgentCheckpointQueueFollowUpActionResponse
        if source_kind == "domain_profile":
            profile_result = await db.execute(
                select(DomainResearchProfile).where(
                    and_(
                        DomainResearchProfile.id == owner_id,
                        DomainResearchProfile.user_id == current_user.id,
                    )
                )
            )
            profile = profile_result.scalar_one_or_none()
            if profile is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Domain research profile not found")
            response = await _perform_follow_up_queue_action(
                profile=profile,
                profile_opportunity_id=opportunity_id,
                action=action,
                operator_note=note,
                db=db,
                current_user=current_user,
            )
        else:
            portfolio_result = await db.execute(
                select(ResearchPortfolio).where(
                    and_(
                        ResearchPortfolio.id == owner_id,
                        ResearchPortfolio.user_id == current_user.id,
                    )
                )
            )
            portfolio = portfolio_result.scalar_one_or_none()
            if portfolio is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research portfolio not found")
            response = await _perform_follow_up_queue_action(
                portfolio=portfolio,
                portfolio_opportunity_id=opportunity_id,
                action=action,
                operator_note=note,
                db=db,
                current_user=current_user,
            )

        follow_up_event_type = "follow_up_approved" if action == "approve_launch" else "follow_up_rejected"
        follow_up_reason_code = "operator_approved_follow_up" if action == "approve_launch" else "operator_rejected_follow_up"
        prior_after_state = event.after_state if isinstance(event.after_state, dict) else {}
        next_after_state = dict(prior_after_state)
        next_after_state.update(
            {
                "opportunity_id": opportunity_id,
                "follow_up_launch_status": response.follow_up_launch_status,
                "follow_up_operator_decision": response.follow_up_operator_decision,
            }
        )
        if response.follow_up_job_id:
            next_after_state["follow_up_job_id"] = str(response.follow_up_job_id)

        event.event_type = follow_up_event_type
        event.decision_type = follow_up_event_type
        event.reason_code = follow_up_reason_code
        event.status = str(response.follow_up_launch_status or "").strip() or None
        event.actor_mode = "operator"
        event.summary = (
            f"{str(event.source_label or 'Autonomy source').strip()}: "
            f"{'approved' if action == 'approve_launch' else 'rejected'} queued follow-up"
        )
        event.before_state = prior_after_state or event.before_state
        event.after_state = next_after_state
        event.operator_note = note or event.operator_note
        event.acknowledged_at = event.acknowledged_at or now
        event.acknowledged_by_user_id = event.acknowledged_by_user_id or current_user.id
        event.triage_status = "resolved"
        event.resolved_at = now
        event.resolved_by_user_id = current_user.id
        event.resolution_note = note or event.resolution_note
    elif action == "relaunch_follow_up":
        prior_after_state = event.after_state if isinstance(event.after_state, dict) else {}
        follow_up_job_id = _trace_event_follow_up_relaunch_job_id(event)
        try:
            follow_up_job_uuid = UUID(follow_up_job_id)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Decision trace event has an invalid follow-up job identifier",
            ) from exc
        inbox_result = await db.execute(
            select(ResearchInboxItem).where(
                and_(
                    ResearchInboxItem.user_id == current_user.id,
                    ResearchInboxItem.follow_up_job_id == follow_up_job_uuid,
                )
            )
        )
        inbox_item = inbox_result.scalar_one_or_none()
        if inbox_item is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Decision trace event could not resolve a relaunchable inbox follow-up",
            )
        response = await _relaunch_follow_up_inbox_item(
            item=inbox_item,
            operator_note=note,
            db=db,
            current_user=current_user,
        )
        next_after_state = dict(prior_after_state)
        next_after_state.update(
            {
                "follow_up_launch_status": response.follow_up_launch_status,
                "follow_up_outcome_status": None,
                "follow_up_last_job_id": str(response.follow_up_job_id or "") or None,
            }
        )
        event.event_type = "follow_up_launched"
        event.decision_type = "follow_up_launched"
        event.reason_code = "operator_relaunched_follow_up"
        event.status = "active"
        event.actor_mode = "operator"
        event.summary = (
            f"{str(event.source_label or 'Autonomy source').strip()}: "
            "relaunched terminal follow-up"
        )
        event.before_state = prior_after_state or event.before_state
        event.after_state = next_after_state
        event.operator_note = note or event.operator_note
        event.acknowledged_at = event.acknowledged_at or now
        event.acknowledged_by_user_id = event.acknowledged_by_user_id or current_user.id
        event.triage_status = "resolved"
        event.resolved_at = now
        event.resolved_by_user_id = current_user.id
        event.resolution_note = note or event.resolution_note
    elif action == "acknowledge":
        event.triage_status = "acknowledged"
        event.acknowledged_at = now
        event.acknowledged_by_user_id = current_user.id
    elif action == "start_investigation":
        event.triage_status = "investigating"
        event.acknowledged_at = event.acknowledged_at or now
        event.acknowledged_by_user_id = event.acknowledged_by_user_id or current_user.id
    elif action == "resolve":
        event.triage_status = "resolved"
        event.acknowledged_at = event.acknowledged_at or now
        event.acknowledged_by_user_id = event.acknowledged_by_user_id or current_user.id
        event.resolved_at = now
        event.resolved_by_user_id = current_user.id
        event.resolution_note = note or event.resolution_note
    elif action == "reopen":
        event.triage_status = "new"
        event.resolved_at = None
        event.resolved_by_user_id = None
        event.resolution_note = None
        if note:
            event.operator_note = note
        if current_status == "resolved":
            await maybe_reopen_event_notification(db, event)
    elif action == "toggle_pin":
        event.pinned = not bool(event.pinned)
    elif action == "assign":
        assignee_id = await _validate_trace_assignee(db, current_user=current_user, assigned_to_user_id=request.assigned_to_user_id)
        if assignee_id is None:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Valid assignee is required")
        event.assigned_to_user_id = assignee_id
        event.assigned_at = now
        event.assigned_by_user_id = current_user.id
    elif action == "unassign":
        event.assigned_to_user_id = None
        event.assigned_at = None
        event.assigned_by_user_id = None
    elif action == "set_due_at":
        if request.due_at is None:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Due date is required")
        event.due_at = request.due_at
    elif action == "clear_due_at":
        event.due_at = None

    event.last_viewed_at = now
    event.updated_at = now
    if note and action not in {"resolve", "reopen"}:
        event.operator_note = note
    apply_decision_trace_escalation(event, now=now)

    await db.commit()
    await maybe_emit_escalation_transition_notification(db, event, previous_state=previous_escalation_state)
    await db.commit()
    await db.refresh(event)
    visible_user_ids = await _list_trace_visible_user_ids(db, current_user=current_user)
    if current_user.id not in visible_user_ids:
        visible_user_ids.add(current_user.id)
    visible_users = list((await db.execute(select(User).where(User.id.in_(visible_user_ids)))).scalars().all())
    user_lookup = {str(user.id): user for user in visible_users}
    return AgentDecisionTraceActionResponse(
        event=AgentDecisionTraceEventResponse.model_validate(
            _decorate_trace_event_payload(
                event_to_trace_payload(event),
                user_lookup=user_lookup,
                current_user_id=current_user.id,
            )
        )
    )


@router.get("/decision-trace/views", response_model=AgentDecisionTraceViewListResponse)
async def list_decision_trace_views(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    rows = list(
        (
            await db.execute(
                select(AutonomyDecisionTraceView)
                .where(AutonomyDecisionTraceView.user_id == current_user.id)
                .order_by(AutonomyDecisionTraceView.is_default.desc(), AutonomyDecisionTraceView.updated_at.desc())
            )
        ).scalars().all()
    )
    return AgentDecisionTraceViewListResponse(
        items=[AgentDecisionTraceViewResponse.model_validate(row) for row in rows],
        total=len(rows),
    )


@router.post("/decision-trace/views", response_model=AgentDecisionTraceViewResponse, status_code=status.HTTP_201_CREATED)
async def create_decision_trace_view(
    request: AgentDecisionTraceViewCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    name = str(request.name or "").strip()
    if not name:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Decision trace view name is required")
    if request.is_default:
        await db.execute(
            AutonomyDecisionTraceView.__table__.update()
            .where(AutonomyDecisionTraceView.user_id == current_user.id)
            .values(is_default=False)
        )
    row = AutonomyDecisionTraceView(
        user_id=current_user.id,
        name=name,
        filters=_normalize_trace_view_filters(request.filters),
        is_default=bool(request.is_default),
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return AgentDecisionTraceViewResponse.model_validate(row)


@router.patch("/decision-trace/views/{view_id}", response_model=AgentDecisionTraceViewResponse)
async def update_decision_trace_view(
    view_id: UUID,
    request: AgentDecisionTraceViewUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    row = (
        await db.execute(
            select(AutonomyDecisionTraceView).where(
                AutonomyDecisionTraceView.id == view_id,
                AutonomyDecisionTraceView.user_id == current_user.id,
            )
        )
    ).scalars().first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Decision trace view not found")

    if request.name is not None:
        next_name = str(request.name or "").strip()
        if not next_name:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Decision trace view name is required")
        row.name = next_name
    if request.filters is not None:
        row.filters = _normalize_trace_view_filters(request.filters)
    if request.is_default is not None:
        if bool(request.is_default):
            await db.execute(
                AutonomyDecisionTraceView.__table__.update()
                .where(
                    AutonomyDecisionTraceView.user_id == current_user.id,
                    AutonomyDecisionTraceView.id != row.id,
                )
                .values(is_default=False)
            )
        row.is_default = bool(request.is_default)
    row.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(row)
    return AgentDecisionTraceViewResponse.model_validate(row)


@router.delete("/decision-trace/views/{view_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_decision_trace_view(
    view_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    row = (
        await db.execute(
            select(AutonomyDecisionTraceView).where(
                AutonomyDecisionTraceView.id == view_id,
                AutonomyDecisionTraceView.user_id == current_user.id,
            )
        )
    ).scalars().first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Decision trace view not found")
    await db.delete(row)
    await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/templates", response_model=AgentJobTemplateListResponse)
async def list_job_templates(
    category: Optional[str] = Query(None, description="Filter by category"),
    recommend_goal: Optional[str] = Query(None, description="Optional goal text used for relevance ranking"),
    recommend_scope: Optional[str] = Query(None, description="Optional scope hint (e.g. backend/frontend) for ranking"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    List available job templates.

    Returns system templates and user's own templates.
    """
    query = select(AgentJobTemplate).where(
        and_(
            AgentJobTemplate.is_active == True,
            or_(
                AgentJobTemplate.is_system == True,
                AgentJobTemplate.owner_user_id == current_user.id,
            )
        )
    )

    if category:
        query = query.where(AgentJobTemplate.category == category)

    query = query.order_by(AgentJobTemplate.is_system.desc(), AgentJobTemplate.name)

    result = await db.execute(query)
    templates = result.scalars().all()

    builtin = list_builtin_agent_job_templates(category)

    response_templates: list[AgentJobTemplateResponse] = []
    for t in templates:
        model = AgentJobTemplateResponse.model_validate(t)
        response_templates.append(
            model.model_copy(
                update={
                    "default_config": _normalize_scope_keys_deep(model.default_config),
                    "default_chain_config": _normalize_scope_keys_deep(model.default_chain_config),
                }
            )
        )
    response_templates.extend(
        [
            AgentJobTemplateResponse(
                id=t.id,
                name=t.name,
                display_name=t.display_name,
                description=t.description,
                category=t.category,
                job_type=t.job_type,
                default_goal=t.default_goal,
                default_config=_normalize_scope_keys_deep(t.default_config),
                default_chain_config=_normalize_scope_keys_deep(t.default_chain_config),
                agent_definition_id=t.agent_definition_id,
                default_max_iterations=t.default_max_iterations,
                default_max_tool_calls=t.default_max_tool_calls,
                default_max_llm_calls=t.default_max_llm_calls,
                default_max_runtime_minutes=t.default_max_runtime_minutes,
                is_system=t.is_system,
                is_active=t.is_active,
                owner_user_id=t.owner_user_id,
                created_at=t.created_at,
                updated_at=t.updated_at,
            )
            for t in builtin
        ]
    )

    ranked: list[tuple[int, AgentJobTemplateResponse]] = []
    for tpl in response_templates:
        rec_score, rec_reasons = _score_template_recommendation(
            tpl,
            category=category,
            recommend_goal=recommend_goal,
            recommend_scope=recommend_scope,
        )
        ranked.append(
            (
                rec_score,
                tpl.model_copy(
                    update={
                        "recommended": rec_score > 0,
                        "recommendation_score": rec_score,
                        "recommendation_reasons": rec_reasons,
                    }
                ),
            )
        )

    ranked.sort(
        key=lambda row: (
            -int(row[0]),
            0 if bool(row[1].is_system) else 1,
            str(row[1].name or "").lower(),
        )
    )
    ordered_templates = [row[1] for row in ranked]

    return AgentJobTemplateListResponse(
        templates=ordered_templates,
        total=len(ordered_templates),
    )


# ============================================================================
# Chain Definition Endpoints
#
# IMPORTANT: Keep these static routes above `/{job_id}`. FastAPI matches routes
# in declaration order, and `/{job_id}` would otherwise capture "/chains".
# ============================================================================

@router.get("/chains", response_model=AgentJobChainDefinitionListResponse)
async def list_chain_definitions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    List available job chain definitions.

    Returns system chains and user's own chains.
    """
    query = select(AgentJobChainDefinition).where(
        and_(
            AgentJobChainDefinition.is_active == True,
            or_(
                AgentJobChainDefinition.is_system == True,
                AgentJobChainDefinition.owner_user_id == current_user.id,
            )
        )
    )
    query = query.order_by(AgentJobChainDefinition.is_system.desc(), AgentJobChainDefinition.name)

    result = await db.execute(query)
    chains = result.scalars().all()
    builtin = list_builtin_agent_job_chain_definitions()

    return AgentJobChainDefinitionListResponse(
        chains=[_chain_definition_to_response(c) for c in chains]
        + [_chain_definition_to_response(c) for c in builtin],
        total=len(chains) + len(builtin),
    )


@router.post("/chains", response_model=AgentJobChainDefinitionResponse, status_code=status.HTTP_201_CREATED)
async def create_chain_definition(
    chain_create: AgentJobChainDefinitionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Create a new job chain definition.

    Chain definitions can be used to create multi-step job sequences.
    """
    # Check for duplicate name
    existing = await db.execute(
        select(AgentJobChainDefinition).where(AgentJobChainDefinition.name == chain_create.name)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chain definition with this name already exists",
        )

    # Convert chain_steps to list of dicts
    chain_steps = [step.model_dump() for step in chain_create.chain_steps]
    for step in chain_steps:
        if isinstance(step, dict) and isinstance(step.get("config"), dict):
            step["config"] = _normalize_scope_keys_deep(step.get("config"))

    chain = AgentJobChainDefinition(
        name=chain_create.name,
        display_name=chain_create.display_name,
        description=chain_create.description,
        chain_steps=chain_steps,
        default_settings=_normalize_scope_keys_deep(chain_create.default_settings),
        owner_user_id=current_user.id,
        is_system=False,
        is_active=True,
    )

    db.add(chain)
    await db.commit()
    await db.refresh(chain)

    logger.info(f"Created chain definition {chain.id} for user {current_user.id}")

    return _chain_definition_to_response(chain)


@router.get("/chains/{chain_id}", response_model=AgentJobChainDefinitionResponse)
async def get_chain_definition(
    chain_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get a specific chain definition.
    """
    builtin = get_builtin_agent_job_chain_definition(chain_id)
    chain = None
    if builtin is None:
        result = await db.execute(
            select(AgentJobChainDefinition).where(
                and_(
                    AgentJobChainDefinition.id == chain_id,
                    or_(
                        AgentJobChainDefinition.is_system == True,
                        AgentJobChainDefinition.owner_user_id == current_user.id,
                    )
                )
            )
        )
        chain = result.scalar_one_or_none()
    else:
        chain = builtin

    if not chain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chain definition not found",
        )

    return _chain_definition_to_response(chain)


@router.patch("/chains/{chain_id}", response_model=AgentJobChainDefinitionResponse)
async def update_chain_definition(
    chain_id: UUID,
    chain_update: AgentJobChainDefinitionUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Update a chain definition.

    Only the owner can update non-system chains.
    """
    result = await db.execute(
        select(AgentJobChainDefinition).where(
            and_(
                AgentJobChainDefinition.id == chain_id,
                AgentJobChainDefinition.owner_user_id == current_user.id,
                AgentJobChainDefinition.is_system == False,
            )
        )
    )
    chain = result.scalar_one_or_none()

    if not chain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chain definition not found or not editable",
        )

    # Apply updates
    update_data = chain_update.model_dump(exclude_unset=True)
    if "chain_steps" in update_data and update_data["chain_steps"]:
        step_payloads = [step.model_dump() for step in chain_update.chain_steps]
        for step in step_payloads:
            if isinstance(step, dict) and isinstance(step.get("config"), dict):
                step["config"] = _normalize_scope_keys_deep(step.get("config"))
        update_data["chain_steps"] = step_payloads
    if "default_settings" in update_data:
        update_data["default_settings"] = _normalize_scope_keys_deep(update_data.get("default_settings"))

    for field, value in update_data.items():
        setattr(chain, field, value)

    chain.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(chain)

    return _chain_definition_to_response(chain)


@router.delete("/chains/{chain_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chain_definition(
    chain_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Delete a chain definition.

    Only the owner can delete non-system chains.
    """
    result = await db.execute(
        select(AgentJobChainDefinition).where(
            and_(
                AgentJobChainDefinition.id == chain_id,
                AgentJobChainDefinition.owner_user_id == current_user.id,
                AgentJobChainDefinition.is_system == False,
            )
        )
    )
    chain = result.scalar_one_or_none()

    if not chain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chain definition not found or not deletable",
        )

    await db.delete(chain)
    await db.commit()


@router.post("/from-chain", response_model=AgentJobResponse, status_code=status.HTTP_201_CREATED)
async def create_job_from_chain(
    request: AgentJobFromChainCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Create and start a job chain from a chain definition.

    Creates the first job in the chain. Subsequent jobs will be created
    automatically as each job completes based on trigger conditions.
    """
    # Load chain definition (builtin first, then DB)
    builtin = get_builtin_agent_job_chain_definition(request.chain_definition_id)
    chain = None
    if builtin is None:
        result = await db.execute(
            select(AgentJobChainDefinition).where(
                and_(
                    AgentJobChainDefinition.id == request.chain_definition_id,
                    AgentJobChainDefinition.is_active == True,
                    or_(
                        AgentJobChainDefinition.is_system == True,
                        AgentJobChainDefinition.owner_user_id == current_user.id,
                    )
                )
            )
        )
        chain = result.scalar_one_or_none()
    else:
        chain = builtin

    if not chain:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chain definition not found or not active",
        )

    if not chain.chain_steps:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chain definition has no steps",
        )


@router.get("/{job_id}", response_model=AgentJobDetailResponse)
async def get_agent_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get details of a specific agent job.

    Includes full execution log.
    """
    result = await db.execute(
        select(AgentJob)
        .options(selectinload(AgentJob.agent_definition))
        .where(AgentJob.id == job_id)
    )
    job = result.scalar_one_or_none()

    if not job or not _is_job_visible_to_user(job, current_user):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    relaunch_children_counts = await _build_relaunch_children_counts_for_user(
        db,
        user_id=current_user.id,
    )
    collaboration_user_lookup = await _build_collaboration_user_lookup(db, current_user=current_user)
    response = _job_to_response(
        job,
        relaunch_children_count=int(relaunch_children_counts.get(job.id, 0) or 0),
        current_user_id=str(current_user.id),
        user_lookup=collaboration_user_lookup,
    )
    return AgentJobDetailResponse(
        **response.model_dump(),
        execution_log=job.execution_log,
    )


@router.post("/{job_id}/promote-domain-research", response_model=AgentJobPromoteDomainResearchResponse)
async def promote_domain_research_job(
    job_id: UUID,
    payload: AgentJobPromoteDomainResearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    from app.api.endpoints.domain_research_profiles import _profile_response
    from app.api.endpoints.research_portfolios import _portfolio_response

    result = await db.execute(
        select(AgentJob)
        .options(selectinload(AgentJob.agent_definition))
        .where(AgentJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if not job or not _is_job_visible_to_user(job, current_user):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent job not found")

    cfg = job.config if isinstance(job.config, dict) else {}
    if _extract_launch_mode(cfg) != "quick_start_domain_research":
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Only domain research quick-start jobs can be promoted")
    if str(job.status or "").strip().lower() != AgentJobStatus.COMPLETED.value:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Only completed domain research quick-start jobs can be promoted")
    if str(cfg.get("profile_id") or "").strip():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Job is already linked to a saved domain research profile")

    existing_promotion = _extract_domain_research_promotion(job)
    if str(existing_promotion.get("domain_research_profile_id") or existing_promotion.get("promoted_domain_research_profile_id") or "").strip():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Job has already been promoted")

    promotion_seed = _build_domain_research_promotion_seed(job)
    profile_data = dict(promotion_seed["profile"])
    profile_data.update(payload.profile.model_dump(exclude_none=True))
    profile_data["start_immediately"] = False
    profile_request = DomainResearchProfileCreate.model_validate(profile_data)
    await _validate_domain_research_sandbox_profile(
        db,
        sandbox_profile_id=profile_request.sandbox_profile_id,
        track_type=profile_request.track_type,
    )
    profile_automation_profile, profile_automation_policy = resolve_domain_profile_automation_contract(
        automation_profile=profile_request.automation_profile,
        automation_policy=profile_request.automation_policy,
        explicit_updates=profile_request.model_dump(exclude_none=True),
    )

    if payload.target_mode == "profile_with_portfolio" and not payload.portfolio_id and payload.portfolio is None:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Provide an existing portfolio_id or portfolio payload when attaching to a fleet")

    profile = DomainResearchProfile(
        user_id=current_user.id,
        title=profile_request.title,
        domain=profile_request.domain,
        objective=profile_request.objective,
        customer_context=profile_request.customer_context,
        status="draft",
        source_scope=profile_request.source_scope,
        track_type=profile_request.track_type,
        research_mode=profile_request.research_mode,
        monitor_queries=profile_request.monitor_queries,
        repo_source_ids=[str(v) for v in (profile_request.repo_source_ids or [])] or None,
        benchmark_queries=profile_request.benchmark_queries,
        report_format=profile_request.report_format,
        scoring_policy=profile_request.scoring_policy,
        selection_policy=profile_request.selection_policy,
        validation_policy=build_domain_profile_compat_policy(profile_automation_policy),
        automation_profile=profile_automation_profile,
        automation_policy=profile_automation_policy,
        sandbox_profile_id=profile_request.sandbox_profile_id,
        interval_minutes=profile_request.interval_minutes,
        persist_artifacts=profile_request.persist_artifacts,
        auto_launch_follow_up=bool(profile_automation_policy.get("auto_launch_follow_up", profile_request.auto_launch_follow_up)),
        auto_create_experiment_plans=bool(profile_automation_policy.get("auto_create_experiment_plans", profile_request.auto_create_experiment_plans)),
        confidence_threshold=float(profile_automation_policy.get("confidence_threshold", profile_request.confidence_threshold)),
        max_documents=profile_request.max_documents,
        max_papers=profile_request.max_papers,
    )
    db.add(profile)
    await db.flush()

    queued_job_ids: list[str] = []
    if payload.start_profile_now:
        profile_job_config = _build_quick_start_domain_research_config(
            AgentJobQuickStartDomainResearchRequest.model_validate(
                {
                    **profile_data,
                    "start_immediately": False,
                    "profile_id": profile.id,
                }
            )
        )
        profile_job_config["profile_id"] = str(profile.id)
        profile_job_config["monitor_mode"] = "profile"
        profile_job_config["interval_minutes"] = int(profile.interval_minutes or 1440)
        profile_job = AgentJob(
            user_id=current_user.id,
            name=f"Domain Monitor — {profile.title}",
            goal=_build_domain_research_goal(
                AgentJobQuickStartDomainResearchRequest.model_validate(
                    {
                        **profile_data,
                        "start_immediately": False,
                        "profile_id": profile.id,
                    }
                )
            ),
            job_type="research",
            status=AgentJobStatus.PENDING.value,
            progress=0,
            schedule_type="continuous",
            schedule_cron=None,
            next_run_at=datetime.utcnow() + timedelta(minutes=int(profile.interval_minutes or 1440)),
            config=profile_job_config,
            max_iterations=6,
            max_tool_calls=20,
            max_llm_calls=12,
            max_runtime_minutes=45,
        )
        db.add(profile_job)
        await db.flush()
        profile.latest_run_job_id = profile_job.id
        profile.active_job_id = profile_job.id
        profile.status = "running"
        profile.started_at = profile.started_at or datetime.utcnow()
        profile.paused_at = None
        queued_job_ids.append(str(profile_job.id))

    portfolio: ResearchPortfolio | None = None
    if payload.target_mode == "profile_with_portfolio":
        if payload.portfolio_id:
            portfolio = await db.get(ResearchPortfolio, payload.portfolio_id)
            if portfolio is None or portfolio.user_id != current_user.id:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Research portfolio not found")
            linked_profile_ids = [str(v).strip() for v in (portfolio.linked_profile_ids or []) if str(v).strip()]
            if str(profile.id) not in linked_profile_ids:
                linked_profile_ids.append(str(profile.id))
            portfolio.linked_profile_ids = linked_profile_ids[:24]
        else:
            portfolio_seed = dict(promotion_seed["portfolio"])
            portfolio_seed.update((payload.portfolio.model_dump(exclude_none=True) if payload.portfolio else {}))
            portfolio_seed["linked_profile_ids"] = [profile.id]
            portfolio_seed["start_immediately"] = False
            portfolio_request = ResearchPortfolioCreate.model_validate(portfolio_seed)
            await _validate_domain_research_sandbox_profile(
                db,
                sandbox_profile_id=portfolio_request.sandbox_profile_id,
                track_type=profile.track_type,
            )
            portfolio = ResearchPortfolio(
                user_id=current_user.id,
                title=portfolio_request.title,
                objective=portfolio_request.objective,
                status="draft",
                linked_profile_ids=[str(profile.id)],
                automation_profile=normalize_portfolio_automation_profile(portfolio_request.automation_profile, default="balanced"),
                automation_policy=resolve_portfolio_automation_policy(portfolio_request.automation_profile, portfolio_request.automation_policy),
                sandbox_profile_id=portfolio_request.sandbox_profile_id,
                opportunities=[],
                child_job_ids=[],
            )
            db.add(portfolio)
            await db.flush()

        if payload.run_portfolio_now and portfolio is not None:
            portfolio_job = AgentJob(
                user_id=current_user.id,
                name=f"Research Fleet — {portfolio.title}",
                goal=portfolio.objective,
                job_type="research",
                status=AgentJobStatus.PENDING.value,
                progress=0,
                schedule_type="once",
                schedule_cron=None,
                next_run_at=None,
                config={
                    "launch_mode": "research_fleet_portfolio",
                    "deterministic_runner": "research_fleet_orchestrator",
                    "research_portfolio_id": str(portfolio.id),
                    "linked_profile_ids": list(portfolio.linked_profile_ids or []),
                    "automation_profile": normalize_portfolio_automation_profile(portfolio.automation_profile, default="balanced"),
                    "automation_policy": resolve_portfolio_automation_policy(portfolio.automation_profile, portfolio.automation_policy),
                    "sandbox_profile_id": str(portfolio.sandbox_profile_id or "").strip() or None,
                    "interval_minutes": 1440,
                },
                max_iterations=6,
                max_tool_calls=24,
                max_llm_calls=16,
                max_runtime_minutes=45,
            )
            db.add(portfolio_job)
            await db.flush()
            portfolio.latest_run_job_id = portfolio_job.id
            queued_job_ids.append(str(portfolio_job.id))

    promotion_status = "promoted_to_profile_and_portfolio" if portfolio is not None else "promoted_to_profile"
    promotion_metadata = {
        "status": promotion_status,
        "promoted_at": datetime.utcnow().isoformat(),
        "source_job_id": str(job.id),
        "domain_research_profile_id": str(profile.id),
        "research_portfolio_id": str(portfolio.id) if portfolio is not None else None,
        "target_mode": payload.target_mode,
        "start_profile_now": bool(payload.start_profile_now),
        "run_portfolio_now": bool(payload.run_portfolio_now),
    }
    cfg = dict(cfg)
    cfg["promotion"] = promotion_metadata
    quick_start = dict(cfg.get("quick_start")) if isinstance(cfg.get("quick_start"), dict) else {}
    quick_start["promotion"] = promotion_metadata
    cfg["quick_start"] = quick_start
    job.config = cfg

    await db.commit()
    await db.refresh(job)
    await db.refresh(profile)
    if portfolio is not None:
        await db.refresh(portfolio)
    refreshed_promotion = _extract_domain_research_promotion(job)
    refreshed_promotion_status = str(refreshed_promotion.get("status") or promotion_status).strip() or promotion_status
    refreshed_profile_id_raw = str(
        refreshed_promotion.get("domain_research_profile_id")
        or refreshed_promotion.get("promoted_domain_research_profile_id")
        or profile.id
    ).strip()
    refreshed_portfolio_id_raw = str(
        refreshed_promotion.get("research_portfolio_id")
        or refreshed_promotion.get("promoted_research_portfolio_id")
        or (portfolio.id if portfolio is not None else "")
    ).strip()

    for queued_job_id in queued_job_ids:
        execute_agent_job_task.delay(queued_job_id, str(current_user.id))

    return AgentJobPromoteDomainResearchResponse(
        source_job_id=job.id,
        promotion_status=refreshed_promotion_status,
        domain_research_profile_id=UUID(refreshed_profile_id_raw) if re.fullmatch(r"[0-9a-fA-F-]{36}", refreshed_profile_id_raw) else profile.id,
        research_portfolio_id=(
            UUID(refreshed_portfolio_id_raw)
            if re.fullmatch(r"[0-9a-fA-F-]{36}", refreshed_portfolio_id_raw)
            else (portfolio.id if portfolio is not None else None)
        ),
        profile=await _profile_response(profile, db),
        portfolio=(await _portfolio_response(portfolio, db) if portfolio is not None else None),
        source_job=_job_to_response(job, current_user_id=str(current_user.id)),
    )


@router.get("/{job_id}/relaunch-lineage", response_model=AgentJobRelaunchLineageResponse)
async def get_agent_job_relaunch_lineage(
    job_id: UUID,
    ancestor_limit: int = Query(100, ge=1, le=300, description="Max ancestor nodes to include"),
    descendant_limit: int = Query(500, ge=1, le=2000, description="Max descendant nodes to include"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get relaunch lineage for a given job.

    Traverses `config.relaunch_from_job_id` links within current user's jobs.
    """
    result = await db.execute(
        select(AgentJob).where(and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id))
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    jobs_result = await db.execute(
        select(AgentJob).where(AgentJob.user_id == current_user.id)
    )
    jobs = jobs_result.scalars().all()
    jobs_by_id: dict[UUID, AgentJob] = {j.id: j for j in jobs}
    return _build_relaunch_lineage(
        job,
        jobs_by_id,
        max_ancestors=ancestor_limit,
        max_descendants=descendant_limit,
    )


@router.get("/{job_id}/ai-hub/recommendation-feedback", response_model=AIHubRecommendationFeedbackListResponse)
async def list_ai_hub_recommendation_feedback(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List feedback entries for this AI Scientist job."""
    job_result = await db.execute(
        select(AgentJob).where(
            and_(
                AgentJob.id == job_id,
                AgentJob.user_id == current_user.id,
            )
        )
    )
    job = job_result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Agent job not found")

    res = await db.execute(
        select(AIHubRecommendationFeedback)
        .where(
            and_(
                AIHubRecommendationFeedback.agent_job_id == job_id,
                AIHubRecommendationFeedback.user_id == current_user.id,
            )
        )
        .order_by(AIHubRecommendationFeedback.created_at.desc())
    )
    items = res.scalars().all()
    return AIHubRecommendationFeedbackListResponse(
        items=[AIHubRecommendationFeedbackResponse.model_validate(x) for x in items],
        total=len(items),
    )


@router.post("/{job_id}/ai-hub/recommendation-feedback", response_model=AIHubRecommendationFeedbackResponse)
async def create_ai_hub_recommendation_feedback(
    job_id: UUID,
    payload: AIHubRecommendationFeedbackCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create an accept/reject feedback entry for an AI Scientist recommendation."""
    job_result = await db.execute(
        select(AgentJob).where(
            and_(
                AgentJob.id == job_id,
                AgentJob.user_id == current_user.id,
            )
        )
    )
    job = job_result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Agent job not found")

    customer_profile_name = None
    customer_profile_id = None
    customer_keywords = None
    raw_profile = await get_feature_str("ai_hub_customer_profile")
    if raw_profile:
        try:
            cp = CustomerProfile.model_validate(json.loads(raw_profile))
            customer_profile_id = cp.id
            customer_profile_name = cp.name
            customer_keywords = cp.keywords
        except Exception:
            pass

    item_id = payload.item_id.strip()
    if not item_id:
        raise HTTPException(status_code=400, detail="item_id required")

    # De-dupe: if user already provided feedback for this exact tuple in this job, update the latest entry.
    existing = await db.execute(
        select(AIHubRecommendationFeedback)
        .where(
            and_(
                AIHubRecommendationFeedback.agent_job_id == job_id,
                AIHubRecommendationFeedback.user_id == current_user.id,
                AIHubRecommendationFeedback.workflow == payload.workflow,
                AIHubRecommendationFeedback.item_type == payload.item_type,
                AIHubRecommendationFeedback.item_id == item_id,
            )
        )
        .order_by(AIHubRecommendationFeedback.created_at.desc())
        .limit(1)
    )
    row = existing.scalar_one_or_none()

    if row:
        row.decision = payload.decision
        row.reason = payload.reason
        row.customer_profile_id = customer_profile_id
        row.customer_profile_name = customer_profile_name
        row.customer_keywords = customer_keywords
        await db.commit()
        await db.refresh(row)
        return AIHubRecommendationFeedbackResponse.model_validate(row)

    fb = AIHubRecommendationFeedback(
        user_id=current_user.id,
        agent_job_id=job_id,
        customer_profile_id=customer_profile_id,
        customer_profile_name=customer_profile_name,
        customer_keywords=customer_keywords,
        workflow=payload.workflow,
        item_type=payload.item_type,
        item_id=item_id,
        decision=payload.decision,
        reason=payload.reason,
    )
    db.add(fb)
    await db.commit()
    await db.refresh(fb)
    return AIHubRecommendationFeedbackResponse.model_validate(fb)


@router.patch("/{job_id}", response_model=AgentJobResponse)
async def update_agent_job(
    job_id: UUID,
    job_update: AgentJobUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Update an agent job.

    Only pending or paused jobs can be updated.
    """
    result = await db.execute(
        select(AgentJob)
        .options(selectinload(AgentJob.agent_definition))
        .where(and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id))
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    if job.status not in [AgentJobStatus.PENDING.value, AgentJobStatus.PAUSED.value]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot update job in status: {job.status}",
        )

    # Apply updates
    update_data = job_update.model_dump(exclude_unset=True)
    if "config" in update_data:
        update_data["config"] = _normalize_scope_config(update_data.get("config"))
    for field, value in update_data.items():
        setattr(job, field, value)

    await db.commit()
    await db.refresh(job)

    return _job_to_response(job)


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Delete an agent job.

    Running jobs must be cancelled first.
    """
    result = await db.execute(
        select(AgentJob).where(
            and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id)
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    if job.status == AgentJobStatus.RUNNING.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete running job. Cancel it first.",
        )

    # Delete checkpoints
    await db.execute(
        select(AgentJobCheckpoint).where(AgentJobCheckpoint.job_id == job_id)
    )

    await db.delete(job)
    await db.commit()


async def _perform_job_action(
    job: AgentJob,
    request: AgentJobActionRequest,
    *,
    db: AsyncSession,
    current_user: User,
) -> AgentJob:
    action = request.action.lower()
    checkpoint_note = str(request.checkpoint_note or "").strip() or None
    action_payload = request.action_payload if isinstance(request.action_payload, dict) else {}
    collaboration_actions = {
        "launch_tie_breaker",
        "promote_swarm_candidate",
        "assign_swarm_review",
        "clear_swarm_assignment",
        "update_swarm_review_note",
    }
    if action in collaboration_actions and not _is_job_visible_to_user(job, current_user):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent job not found")

    async def _resolve_checkpoint_context(
        require_pending: bool = True,
    ) -> tuple[dict, dict, dict, AgentJobCheckpoint | None]:
        results_payload, approval_payload, pending_checkpoint = _approval_payload_from_results(job.results)
        if require_pending and not isinstance(pending_checkpoint, dict):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No pending approval checkpoint for this job",
            )
        checkpoint_row = await _load_latest_job_checkpoint(job.id, db)
        return results_payload, approval_payload, pending_checkpoint or {}, checkpoint_row

    if action == "pause":
        if job.status != AgentJobStatus.RUNNING.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Can only pause running jobs",
            )
        results_payload = job.results if isinstance(job.results, dict) else {}
        _append_operator_intervention(
            results_payload,
            action="pause",
            actor_user_id=current_user.id,
            note=checkpoint_note,
            job_status_before=job.status,
            job_status_after=AgentJobStatus.PAUSED.value,
        )
        job.results = results_payload
        job.status = AgentJobStatus.PAUSED.value
        job.add_log_entry({"phase": "paused", "reason": "user_request"})

    elif action == "resume":
        if job.status != AgentJobStatus.PAUSED.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Can only resume paused jobs",
            )
        results_payload, approval_payload, pending_checkpoint, checkpoint_row = await _resolve_checkpoint_context(
            require_pending=False
        )
        state = dict(checkpoint_row.state) if checkpoint_row and isinstance(checkpoint_row.state, dict) else {}
        if pending_checkpoint:
            approval_payload["pending"] = None
            _append_step_event(
                state,
                {
                    "type": "checkpoint_approved",
                    "method": "resume_action",
                    "iteration": int(pending_checkpoint.get("iteration", 0) or 0),
                    "plan_step_id": str(pending_checkpoint.get("plan_step_id") or "").strip() or None,
                    "plan_step_index": int(pending_checkpoint.get("plan_step_index", -1) or -1),
                    "tool": str(((pending_checkpoint.get("action") or {}).get("tool") or "")).strip() or None,
                    "note": checkpoint_note,
                    "actor_user_id": str(current_user.id),
                },
            )
            _append_approval_event(
                approval_payload,
                pending_checkpoint,
                method="resume_action",
                user_id=current_user.id,
                note=checkpoint_note,
            )
            job.add_log_entry(
                {
                    "phase": "approval_checkpoint_approved",
                    "reason": "resume_action",
                    "action_tool": str(((pending_checkpoint.get("action") or {}).get("tool") or "")).strip(),
                }
            )
            if checkpoint_row:
                state["approval_checkpoint_pending"] = None
                _set_current_plan_step_status(state, status="in_progress", advance_next=False)
                checkpoint_row.state = state
                db.add(checkpoint_row)
            _sync_execution_strategy_state(
                results_payload,
                approval_payload=approval_payload,
                state=state,
            )
            results_payload["approval_checkpoint"] = None
        _append_operator_intervention(
            results_payload,
            action="resume",
            actor_user_id=current_user.id,
            note=checkpoint_note,
            job_status_before=job.status,
            job_status_after=AgentJobStatus.PENDING.value,
            metadata={
                "approval_checkpoint_pending": bool(pending_checkpoint),
            },
        )
        job.results = results_payload
        job.status = AgentJobStatus.PENDING.value
        job.add_log_entry({"phase": "resumed", "reason": "user_request"})
        # Queue for execution
        execute_agent_job_task.delay(str(job.id), str(current_user.id))

    elif action in {"approve", "edit", "skip", "reject"}:
        if job.status != AgentJobStatus.PAUSED.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Approval checkpoint actions require paused status",
            )
        results_payload, approval_payload, pending_checkpoint, checkpoint_row = await _resolve_checkpoint_context(
            require_pending=True
        )
        state = dict(checkpoint_row.state) if checkpoint_row and isinstance(checkpoint_row.state, dict) else {}

        action_patch: dict[str, Any] = {}
        if request.checkpoint_action_patch is not None:
            try:
                action_patch = _normalize_checkpoint_action_patch(request.checkpoint_action_patch)
            except ValueError as e:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

        if action == "edit":
            if not action_patch:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="edit action requires checkpoint_action_patch",
                )
            edited_action = _apply_checkpoint_action_patch(pending_checkpoint, action_patch)
            if isinstance(state, dict):
                state["approval_override_action"] = edited_action
                state["approval_checkpoint_pending"] = None
                _set_current_plan_step_status(state, status="in_progress", advance_next=False)
                _append_step_event(
                    state,
                    {
                        "type": "checkpoint_approved",
                        "method": "edit_action",
                        "iteration": int(pending_checkpoint.get("iteration", 0) or 0),
                        "plan_step_id": str(pending_checkpoint.get("plan_step_id") or "").strip() or None,
                        "plan_step_index": int(pending_checkpoint.get("plan_step_index", -1) or -1),
                        "tool": str((edited_action.get("tool") or "")).strip() or None,
                        "note": checkpoint_note,
                        "actor_user_id": str(current_user.id),
                    },
                )
            _append_approval_event(
                approval_payload,
                pending_checkpoint,
                method="edit_action",
                user_id=current_user.id,
                note=checkpoint_note,
                edited_action=edited_action,
            )
            approval_payload["pending"] = None
            _sync_execution_strategy_state(
                results_payload,
                approval_payload=approval_payload,
                state=state,
            )
            results_payload["approval_checkpoint"] = None
            _append_operator_intervention(
                results_payload,
                action="edit",
                actor_user_id=current_user.id,
                note=checkpoint_note,
                job_status_before=job.status,
                job_status_after=AgentJobStatus.PENDING.value,
                metadata={
                    "plan_step_id": str(pending_checkpoint.get("plan_step_id") or "").strip() or None,
                    "plan_step_index": int(pending_checkpoint.get("plan_step_index", -1) or -1),
                    "tool": str((edited_action.get("tool") or "")).strip() or None,
                },
            )
            job.results = results_payload
            job.status = AgentJobStatus.PENDING.value
            job.current_phase = "approval_edited"
            job.phase_details = "Checkpoint action edited and approved"
            job.add_log_entry(
                {
                    "phase": "approval_checkpoint_edited",
                    "reason": "edit_action",
                    "action_tool": str((edited_action.get("tool") or "")).strip(),
                }
            )
            if checkpoint_row and isinstance(state, dict):
                checkpoint_row.state = state
                db.add(checkpoint_row)
            execute_agent_job_task.delay(str(job.id), str(current_user.id))

        elif action == "approve":
            edited_action = {}
            if action_patch:
                edited_action = _apply_checkpoint_action_patch(pending_checkpoint, action_patch)
                if isinstance(state, dict):
                    state["approval_override_action"] = edited_action
            if isinstance(state, dict):
                state["approval_checkpoint_pending"] = None
                _set_current_plan_step_status(state, status="in_progress", advance_next=False)
                _append_step_event(
                    state,
                    {
                        "type": "checkpoint_approved",
                        "method": "approve_action",
                        "iteration": int(pending_checkpoint.get("iteration", 0) or 0),
                        "plan_step_id": str(pending_checkpoint.get("plan_step_id") or "").strip() or None,
                        "plan_step_index": int(pending_checkpoint.get("plan_step_index", -1) or -1),
                        "tool": str((((pending_checkpoint.get("action") or {}).get("tool")) or "")).strip() or None,
                        "note": checkpoint_note,
                        "actor_user_id": str(current_user.id),
                    },
                )
            _append_approval_event(
                approval_payload,
                pending_checkpoint,
                method="approve_action",
                user_id=current_user.id,
                note=checkpoint_note,
                edited_action=edited_action if edited_action else None,
            )
            approval_payload["pending"] = None
            _sync_execution_strategy_state(
                results_payload,
                approval_payload=approval_payload,
                state=state,
            )
            results_payload["approval_checkpoint"] = None
            _append_operator_intervention(
                results_payload,
                action="approve",
                actor_user_id=current_user.id,
                note=checkpoint_note,
                job_status_before=job.status,
                job_status_after=AgentJobStatus.PENDING.value,
                metadata={
                    "plan_step_id": str(pending_checkpoint.get("plan_step_id") or "").strip() or None,
                    "plan_step_index": int(pending_checkpoint.get("plan_step_index", -1) or -1),
                    "tool": str(((pending_checkpoint.get("action") or {}).get("tool") or "")).strip() or None,
                    "edited_action": bool(edited_action),
                },
            )
            job.results = results_payload
            job.status = AgentJobStatus.PENDING.value
            job.current_phase = "approval_approved"
            job.phase_details = "Checkpoint approved"
            job.add_log_entry(
                {
                    "phase": "approval_checkpoint_approved",
                    "reason": "approve_action",
                    "action_tool": str(((pending_checkpoint.get("action") or {}).get("tool") or "")).strip(),
                }
            )
            if checkpoint_row and isinstance(state, dict):
                checkpoint_row.state = state
                db.add(checkpoint_row)
            execute_agent_job_task.delay(str(job.id), str(current_user.id))

        elif action == "skip":
            if isinstance(state, dict):
                state["approval_checkpoint_pending"] = None
                step_meta = _set_current_plan_step_status(state, status="skipped", advance_next=True)
                _append_step_event(
                    state,
                    {
                        "type": "step_skipped",
                        "method": "skip_action",
                        "iteration": int(pending_checkpoint.get("iteration", 0) or 0),
                        "plan_step_id": str(step_meta.get("step_id") or "") or None,
                        "plan_step_index": int(step_meta.get("plan_step_index", -1) or -1),
                        "tool": str((((pending_checkpoint.get("action") or {}).get("tool")) or "")).strip() or None,
                        "note": checkpoint_note,
                        "actor_user_id": str(current_user.id),
                    },
                )
            else:
                step_meta = {"step_id": "", "plan_step_index": -1}
            _append_approval_event(
                approval_payload,
                pending_checkpoint,
                method="skip_action",
                user_id=current_user.id,
                note=checkpoint_note,
            )
            approval_payload["pending"] = None
            _sync_execution_strategy_state(
                results_payload,
                approval_payload=approval_payload,
                state=state,
            )
            results_payload["approval_checkpoint"] = None
            _append_operator_intervention(
                results_payload,
                action="skip",
                actor_user_id=current_user.id,
                note=checkpoint_note,
                job_status_before=job.status,
                job_status_after=AgentJobStatus.PENDING.value,
                metadata={
                    "step_id": str(step_meta.get("step_id") or ""),
                    "plan_step_index": int(step_meta.get("plan_step_index", -1) or -1),
                },
            )
            job.results = results_payload
            job.status = AgentJobStatus.PENDING.value
            job.current_phase = "approval_skipped"
            job.phase_details = "Skipped current plan step and resumed"
            job.add_log_entry(
                {
                    "phase": "approval_checkpoint_skipped",
                    "reason": "skip_action",
                    "step_id": str(step_meta.get("step_id") or ""),
                    "plan_step_index": int(step_meta.get("plan_step_index", -1) or -1),
                }
            )
            if checkpoint_row and isinstance(state, dict):
                checkpoint_row.state = state
                db.add(checkpoint_row)
            execute_agent_job_task.delay(str(job.id), str(current_user.id))

        else:  # reject
            if isinstance(state, dict):
                state["approval_checkpoint_pending"] = None
                step_meta = _set_current_plan_step_status(state, status="failed", advance_next=False)
                _append_step_event(
                    state,
                    {
                        "type": "checkpoint_rejected",
                        "method": "reject_action",
                        "iteration": int(pending_checkpoint.get("iteration", 0) or 0),
                        "plan_step_id": str(step_meta.get("step_id") or "") or None,
                        "plan_step_index": int(step_meta.get("plan_step_index", -1) or -1),
                        "tool": str((((pending_checkpoint.get("action") or {}).get("tool")) or "")).strip() or None,
                        "note": checkpoint_note,
                        "actor_user_id": str(current_user.id),
                    },
                )
            else:
                step_meta = {"step_id": "", "plan_step_index": -1}
            _append_approval_event(
                approval_payload,
                pending_checkpoint,
                method="reject_action",
                user_id=current_user.id,
                note=checkpoint_note,
            )
            approval_payload["pending"] = None
            _sync_execution_strategy_state(
                results_payload,
                approval_payload=approval_payload,
                state=state,
            )
            results_payload["approval_checkpoint"] = None
            _append_operator_intervention(
                results_payload,
                action="reject",
                actor_user_id=current_user.id,
                note=checkpoint_note,
                job_status_before=job.status,
                job_status_after=AgentJobStatus.PAUSED.value,
                metadata={
                    "step_id": str(step_meta.get("step_id") or ""),
                    "plan_step_index": int(step_meta.get("plan_step_index", -1) or -1),
                },
            )
            job.results = results_payload
            job.status = AgentJobStatus.PAUSED.value
            job.current_phase = "approval_rejected"
            job.phase_details = (
                str(checkpoint_note or "Checkpoint rejected. Edit, approve, skip, or resume when ready.")[:280]
            )
            job.add_log_entry(
                {
                    "phase": "approval_checkpoint_rejected",
                    "reason": "reject_action",
                    "step_id": str(step_meta.get("step_id") or ""),
                    "plan_step_index": int(step_meta.get("plan_step_index", -1) or -1),
                    "note": str(checkpoint_note or "")[:300] or None,
                }
            )
            if checkpoint_row and isinstance(state, dict):
                checkpoint_row.state = state
                db.add(checkpoint_row)

    elif action == "cancel":
        if job.status not in [
            AgentJobStatus.PENDING.value,
            AgentJobStatus.RUNNING.value,
            AgentJobStatus.PAUSED.value,
        ]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel job in status: {job.status}",
            )
        results_payload = job.results if isinstance(job.results, dict) else {}
        _append_operator_intervention(
            results_payload,
            action="cancel",
            actor_user_id=current_user.id,
            note=checkpoint_note,
            job_status_before=job.status,
            job_status_after=AgentJobStatus.CANCELLED.value,
        )
        job.results = results_payload
        job.status = AgentJobStatus.CANCELLED.value
        job.completed_at = datetime.utcnow()
        job.add_log_entry({"phase": "cancelled", "reason": "user_request"})
        await sync_follow_up_outcome_for_job(db, job)

    elif action == "restart":
        if job.status not in [
            AgentJobStatus.COMPLETED.value,
            AgentJobStatus.FAILED.value,
            AgentJobStatus.CANCELLED.value,
        ]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Can only restart completed, failed, or cancelled jobs",
            )
        launch_mode = _extract_launch_mode(job.config if isinstance(job.config, dict) else None)
        if launch_mode == "quick_start_repo_bug_triage":
            retry_request = _build_quick_start_repo_bug_triage_relaunch_request(
                job,
                retry_strategy="refined_retry",
            )
            if retry_request is not None:
                new_job = await quick_start_repo_bug_triage_job(retry_request, db, current_user)
                results_payload = job.results if isinstance(job.results, dict) else {}
                recovery = _extract_repo_bug_triage_coding_recovery(job)
                _append_operator_intervention(
                    results_payload,
                    action="restart",
                    actor_user_id=current_user.id,
                    note=checkpoint_note,
                    job_status_before=job.status,
                    job_status_after=job.status,
                    metadata={
                        "new_job_id": str(new_job.id),
                        "launch_mode": "quick_start_repo_bug_triage",
                        "recovery_strategy": "refined_retry",
                        "retry_reason": str(recovery.get("retry_reason") or "").strip() or None,
                    },
                )
                job.results = results_payload
                job.add_log_entry(
                    {
                        "phase": "restart_requested",
                        "reason": "user_request",
                        "result": {
                            "new_job_id": str(new_job.id),
                            "launch_mode": "quick_start_repo_bug_triage",
                            "recovery_strategy": "refined_retry",
                        },
                    }
                )
                await db.commit()
                return new_job
        # Reset job state
        previous_status = job.status
        job.status = AgentJobStatus.PENDING.value
        job.progress = 0
        job.iteration = 0
        job.tool_calls_used = 0
        job.llm_calls_used = 0
        job.tokens_used = 0
        job.error = None
        job.error_count = 0
        job.started_at = None
        job.completed_at = None
        job.current_phase = None
        job.phase_details = None
        job.execution_log = []
        results_payload = {}
        _append_operator_intervention(
            results_payload,
            action="restart",
            actor_user_id=current_user.id,
            note=checkpoint_note,
            job_status_before=previous_status,
            job_status_after=AgentJobStatus.PENDING.value,
        )
        job.results = results_payload
        job.output_artifacts = None
        job.add_log_entry({"phase": "restarted", "reason": "user_request"})
        # Queue for execution
        execute_agent_job_task.delay(str(job.id), str(current_user.id))

    elif action == "relaunch":
        if job.status not in [
            AgentJobStatus.COMPLETED.value,
            AgentJobStatus.FAILED.value,
            AgentJobStatus.CANCELLED.value,
        ]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Can only relaunch completed, failed, or cancelled jobs",
            )
        launch_mode = _extract_launch_mode(job.config if isinstance(job.config, dict) else None)
        new_job = None
        relaunch_mode = ""
        if launch_mode == "quick_start_claude_backend":
            relaunch_request = _build_quick_start_relaunch_request(job)
            if relaunch_request is not None:
                new_job = await quick_start_claude_backend_job(relaunch_request, db, current_user)
                relaunch_mode = "quick_start_claude_backend"
        elif launch_mode == "quick_start_domain_research":
            relaunch_request = _build_quick_start_domain_research_relaunch_request(job)
            if relaunch_request is not None:
                new_job = await quick_start_domain_research_job(relaunch_request, db, current_user)
                relaunch_mode = "quick_start_domain_research"
        elif launch_mode == "quick_start_repo_bug_triage":
            relaunch_request = _build_quick_start_repo_bug_triage_relaunch_request(
                job,
                retry_strategy="clean_relaunch",
            )
            if relaunch_request is not None:
                new_job = await quick_start_repo_bug_triage_job(relaunch_request, db, current_user)
                relaunch_mode = "quick_start_repo_bug_triage"
        elif launch_mode == "quick_start_bug_triage_swarm":
            relaunch_request = _build_quick_start_bug_triage_swarm_relaunch_request(job)
            if relaunch_request is not None:
                new_job = await quick_start_bug_triage_swarm_job(relaunch_request, db, current_user)
                relaunch_mode = "quick_start_bug_triage_swarm"
        elif launch_mode == "quick_start_build_break_swarm":
            relaunch_request = _build_quick_start_build_break_swarm_relaunch_request(job)
            if relaunch_request is not None:
                new_job = await quick_start_build_break_swarm_job(relaunch_request, db, current_user)
                relaunch_mode = "quick_start_build_break_swarm"
        elif launch_mode == "quick_start_frontend_regression_swarm":
            relaunch_request = _build_quick_start_frontend_regression_swarm_relaunch_request(job)
            if relaunch_request is not None:
                new_job = await quick_start_frontend_regression_swarm_job(relaunch_request, db, current_user)
                relaunch_mode = "quick_start_frontend_regression_swarm"
        elif launch_mode == "quick_start_role_workflow":
            relaunch_request = _build_quick_start_role_workflow_relaunch_request(job)
            if relaunch_request is not None:
                new_job = await quick_start_role_workflow_job(relaunch_request, db, current_user)
                relaunch_mode = "quick_start_role_workflow"

        if new_job is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "Relaunch is only supported for quick-start Claude backend, "
                    "domain research, repo bug triage, coding swarm quick starts, or role-workflow jobs with valid launch configuration"
                ),
            )

        job.add_log_entry(
            {
                "phase": "relaunch_requested",
                "reason": "user_request",
                "result": {
                    "new_job_id": str(new_job.id),
                    "launch_mode": relaunch_mode,
                },
            }
        )
        results_payload = job.results if isinstance(job.results, dict) else {}
        _append_operator_intervention(
            results_payload,
            action="relaunch",
            actor_user_id=current_user.id,
            note=checkpoint_note,
            job_status_before=job.status,
            job_status_after=job.status,
            metadata={
                "new_job_id": str(new_job.id),
                "launch_mode": relaunch_mode,
                "recovery_strategy": "clean_relaunch" if relaunch_mode == "quick_start_repo_bug_triage" else None,
            },
        )
        job.results = results_payload
        await db.commit()
        return new_job

    elif action == "generate_summary":
        if job.status != AgentJobStatus.COMPLETED.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Can only generate summary for completed jobs",
            )
        generate_job_summary.delay(str(job.id))

    elif action == "assign_swarm_review":
        if not _infer_coding_swarm_preset_key(job):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Assignment is only available on coding swarm jobs")
        assigned_user_id = str(action_payload.get("assigned_user_id") or current_user.id).strip()
        try:
            assigned_user = await db.get(User, UUID(assigned_user_id))
        except Exception:
            assigned_user = None
        if assigned_user is None or not bool(getattr(assigned_user, "is_active", False)):
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Assigned user not found")
        collaboration = _extract_swarm_collaboration(job)
        collaboration = _build_swarm_collaboration_payload(
            owner_user_id=collaboration.get("owner_user_id") or job.user_id,
            visibility="shared",
            shared_with_user_ids=[*list(collaboration.get("shared_with_user_ids") or []), assigned_user_id],
            assigned_user_id=assigned_user_id,
            assigned_by_user_id=str(current_user.id),
            assigned_at=datetime.utcnow().isoformat(),
            review_note=str(collaboration.get("review_note") or "").strip() or None,
        )
        _store_swarm_collaboration(job, collaboration)
        job.add_log_entry({"phase": "swarm_review_assigned", "result": {"assigned_user_id": assigned_user_id}})

    elif action == "clear_swarm_assignment":
        if not _infer_coding_swarm_preset_key(job):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Assignment is only available on coding swarm jobs")
        collaboration = _extract_swarm_collaboration(job)
        collaboration = _build_swarm_collaboration_payload(
            owner_user_id=collaboration.get("owner_user_id") or job.user_id,
            visibility="shared" if bool(collaboration.get("shared_review")) else "private",
            shared_with_user_ids=list(collaboration.get("shared_with_user_ids") or []),
            review_note=str(collaboration.get("review_note") or "").strip() or None,
        )
        _store_swarm_collaboration(job, collaboration)
        job.add_log_entry({"phase": "swarm_assignment_cleared", "reason": "user_request"})

    elif action == "update_swarm_review_note":
        if not _infer_coding_swarm_preset_key(job):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Review notes are only available on coding swarm jobs")
        collaboration = _extract_swarm_collaboration(job)
        collaboration = _build_swarm_collaboration_payload(
            owner_user_id=collaboration.get("owner_user_id") or job.user_id,
            visibility="shared" if bool(collaboration.get("shared_review")) else "private",
            shared_with_user_ids=list(collaboration.get("shared_with_user_ids") or []),
            assigned_user_id=str(collaboration.get("assigned_user_id") or "").strip() or None,
            assigned_by_user_id=str(collaboration.get("assigned_by_user_id") or "").strip() or None,
            assigned_at=str(collaboration.get("assigned_at") or "").strip() or None,
            review_note=str(action_payload.get("review_note") or "").strip() or None,
        )
        _store_swarm_collaboration(job, collaboration)
        job.add_log_entry({"phase": "swarm_review_note_updated", "reason": "user_request"})

    elif action == "launch_tie_breaker":
        if job.status not in [
            AgentJobStatus.COMPLETED.value,
            AgentJobStatus.FAILED.value,
            AgentJobStatus.CANCELLED.value,
            AgentJobStatus.PAUSED.value,
        ]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Can only launch a tie-breaker from completed, failed, cancelled, or paused fan-in jobs",
            )
        results_payload = job.results if isinstance(job.results, dict) else {}
        fan_in = results_payload.get("swarm_fan_in") if isinstance(results_payload.get("swarm_fan_in"), dict) else {}
        cfg = job.config if isinstance(job.config, dict) else {}
        inherited = cfg.get("inherited_data") if isinstance(cfg.get("inherited_data"), dict) else {}
        swarm_payload = inherited.get("swarm") if isinstance(inherited.get("swarm"), dict) else {}
        if not fan_in or not swarm_payload or not bool(cfg.get("coding_swarm_enabled") or fan_in):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Bug triage swarm tie-breaker is only available on coding swarm fan-in jobs with inherited sibling data",
            )
        executor = AutonomousAgentExecutor()
        new_job = await executor._launch_bug_triage_swarm_tie_breaker_job(
            fan_in_job=job,
            db=db,
            merged=fan_in,
            swarm_payload=swarm_payload,
        )
        if new_job is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to launch verifier tie-breaker",
            )
        fan_in["review_state"] = "tie_break_running"
        fan_in["review_required"] = False
        fan_in["review_reason"] = str(fan_in.get("review_reason") or "Verifier tie-breaker running.")
        fan_in["tie_breaker_job_id"] = str(new_job.id)
        fan_in["tie_breaker_attempted"] = True
        results_payload["swarm_fan_in"] = fan_in
        _append_operator_intervention(
            results_payload,
            action="launch_tie_breaker",
            actor_user_id=current_user.id,
            note=checkpoint_note,
            job_status_before=job.status,
            job_status_after=job.status,
            metadata={"new_job_id": str(new_job.id)},
        )
        job.results = results_payload
        job.add_log_entry(
            {
                "phase": "tie_breaker_requested",
                "reason": "user_request",
                "result": {"new_job_id": str(new_job.id)},
            }
        )
        await db.commit()
        execute_agent_job_task.delay(str(new_job.id), str(current_user.id))
        return new_job

    elif action == "promote_swarm_candidate":
        if job.status not in [
            AgentJobStatus.COMPLETED.value,
            AgentJobStatus.FAILED.value,
            AgentJobStatus.CANCELLED.value,
            AgentJobStatus.PAUSED.value,
        ]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Can only promote a swarm candidate from completed, failed, cancelled, or paused fan-in jobs",
            )
        results_payload = job.results if isinstance(job.results, dict) else {}
        fan_in = results_payload.get("swarm_fan_in") if isinstance(results_payload.get("swarm_fan_in"), dict) else {}
        cfg = job.config if isinstance(job.config, dict) else {}
        if not fan_in or not bool(cfg.get("coding_swarm_enabled") or fan_in):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Manual promotion is only available on bug triage swarm fan-in jobs",
            )
        candidate_rows = fan_in.get("candidate_paths") if isinstance(fan_in.get("candidate_paths"), list) else []
        candidate_job_id = str(action_payload.get("candidate_job_id") or "").strip()
        if not candidate_job_id and candidate_rows:
            try:
                candidate_index = int(action_payload.get("candidate_index", 0) or 0)
            except Exception:
                candidate_index = 0
            if 0 <= candidate_index < len(candidate_rows) and isinstance(candidate_rows[candidate_index], dict):
                candidate_job_id = str(candidate_rows[candidate_index].get("job_id") or "").strip()
        executor = AutonomousAgentExecutor()
        new_job = await executor._launch_bug_triage_swarm_repair_job(
            fan_in_job=job,
            db=db,
            merged=fan_in,
            candidate_job_id=candidate_job_id,
            manual_promotion=True,
        )
        if new_job is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Failed to launch repair chain from the selected swarm candidate",
            )
        fan_in["repair_chain_job_id"] = str(new_job.id)
        fan_in["review_state"] = "manual_promotion"
        fan_in["review_required"] = False
        fan_in["promotion_reason"] = (
            f"Manually promoted swarm candidate {candidate_job_id[:8]} into the repair chain."
            if candidate_job_id
            else "Manually promoted the leading swarm candidate into the repair chain."
        )
        results_payload["swarm_fan_in"] = fan_in
        _append_operator_intervention(
            results_payload,
            action="promote_swarm_candidate",
            actor_user_id=current_user.id,
            note=checkpoint_note,
            job_status_before=job.status,
            job_status_after=job.status,
            metadata={
                "new_job_id": str(new_job.id),
                "candidate_job_id": candidate_job_id or None,
            },
        )
        job.results = results_payload
        job.add_log_entry(
            {
                "phase": "swarm_candidate_promoted",
                "reason": "user_request",
                "result": {
                    "new_job_id": str(new_job.id),
                    "candidate_job_id": candidate_job_id or None,
                },
            }
        )
        await db.commit()
        execute_agent_job_task.delay(str(new_job.id), str(current_user.id))
        return new_job

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Unknown action: "
                f"{action}. Valid actions: pause, resume, cancel, restart, relaunch, "
                "generate_summary, approve, reject, edit, skip, launch_tie_breaker, "
                "promote_swarm_candidate, assign_swarm_review, clear_swarm_assignment, "
                "update_swarm_review_note"
            ),
        )
    return job


@router.post("/{job_id}/action", response_model=AgentJobResponse)
async def job_action(
    job_id: UUID,
    request: AgentJobActionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Perform an action on an agent job.

    Actions: pause, resume, cancel, restart, relaunch, generate_summary,
    approve, reject, edit, skip, launch_tie_breaker, promote_swarm_candidate,
    assign_swarm_review, clear_swarm_assignment, update_swarm_review_note
    """
    result = await db.execute(
        select(AgentJob)
        .options(selectinload(AgentJob.agent_definition))
        .where(AgentJob.id == job_id)
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )
    owner_only_actions = {"pause", "resume", "cancel", "restart", "relaunch", "generate_summary", "approve", "reject", "edit", "skip"}
    if request.action.lower() in owner_only_actions:
        if not (current_user.is_admin() or str(job.user_id) == str(current_user.id)):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent job not found")
    elif not _is_job_visible_to_user(job, current_user):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent job not found")

    previous_status = str(job.status or "")
    job_or_new_job = await _perform_job_action(
        job,
        request,
        db=db,
        current_user=current_user,
    )
    await _record_job_operator_event(
        db=db,
        job=job,
        current_user=current_user,
        action=request.action,
        note=request.checkpoint_note,
        previous_status=previous_status,
        next_status=str(job.status or ""),
        scheduler_state=_extract_scheduler_state(job),
        metadata={
            "returned_job_id": str(job_or_new_job.id) if getattr(job_or_new_job, "id", None) else None,
            "spawned_new_job": str(getattr(job_or_new_job, "id", None) or "") != str(job.id),
        },
        summary=f"{str(job.name or 'Agent job').strip()}: {str(request.action or '').strip().replace('_', ' ')}",
    )
    await db.commit()
    await db.refresh(job_or_new_job)

    return _job_to_response(job_or_new_job)


@router.post(
    "/checkpoint-queue/follow-up-action",
    response_model=AgentCheckpointQueueFollowUpActionResponse,
)
async def checkpoint_queue_follow_up_action(
    request: AgentCheckpointQueueFollowUpActionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Approve or reject a queued follow-up recommendation."""
    if request.inbox_item_id is not None:
        item_result = await db.execute(
            select(ResearchInboxItem).where(
                and_(
                    ResearchInboxItem.id == request.inbox_item_id,
                    ResearchInboxItem.user_id == current_user.id,
                )
            )
        )
        item = item_result.scalar_one_or_none()
        if item is None:
            raise HTTPException(status_code=404, detail="Inbox item not found")
        source_job = await db.get(AgentJob, item.job_id) if item.job_id else None
        if source_job is not None and str(source_job.user_id) != str(current_user.id):
            source_job = None
        source_scheduler_state = _extract_scheduler_state(source_job) if source_job is not None else None

        response = await _perform_follow_up_queue_action(
            item=item,
            action=request.action,
            operator_note=request.operator_note,
            db=db,
            current_user=current_user,
        )
        await _record_follow_up_queue_decision_event(
            db=db,
            current_user=current_user,
            action=request.action,
            operator_note=request.operator_note,
            source_kind="queue",
            source_id=str(item.id),
            source_label=str(item.title or "Research inbox item").strip(),
            customer=str(item.customer or "").strip() or None,
            reason_code=str(item.follow_up_recommendation_key or item.follow_up_block_reason or "").strip() or None,
            reason_label=_queue_reason_label(str(item.follow_up_recommendation_key or item.follow_up_block_reason or "")),
            scheduler_state=source_scheduler_state,
            follow_up_launch_status=response.follow_up_launch_status,
            follow_up_operator_decision=response.follow_up_operator_decision,
            deep_link={
                "target_tab": "queue",
                "params": {"tab": "queue"},
                "label": "Open Checkpoint Queue",
            },
            metadata={"inbox_item_id": str(item.id)},
            after_state={
                "follow_up_launch_status": response.follow_up_launch_status,
                "follow_up_operator_decision": response.follow_up_operator_decision,
            },
        )
        await db.commit()
        return response

    if request.domain_research_profile_id is not None:
        profile_result = await db.execute(
            select(DomainResearchProfile).where(
                and_(
                    DomainResearchProfile.id == request.domain_research_profile_id,
                    DomainResearchProfile.user_id == current_user.id,
                )
            )
        )
        profile = profile_result.scalar_one_or_none()
        if profile is None:
            raise HTTPException(status_code=404, detail="Domain research profile not found")
        parent_job = await _resolve_profile_parent_job_for_queue(db=db, profile=profile)
        source_scheduler_state = _extract_scheduler_state(parent_job)

        response = await _perform_follow_up_queue_action(
            profile=profile,
            profile_opportunity_id=request.profile_opportunity_id,
            action=request.action,
            operator_note=request.operator_note,
            db=db,
            current_user=current_user,
        )
        reason_label = _follow_up_opportunity_reason_label(profile=profile, opportunity_id=str(request.profile_opportunity_id or "").strip())
        await _record_follow_up_queue_decision_event(
            db=db,
            current_user=current_user,
            action=request.action,
            operator_note=request.operator_note,
            source_kind="domain_profile",
            source_id=str(profile.id),
            source_label=str(profile.title or "Domain profile").strip(),
            customer=str(profile.customer_context or "").strip() or None,
            reason_code=str(request.profile_opportunity_id or "").strip() or None,
            reason_label=reason_label,
            scheduler_state=source_scheduler_state,
            follow_up_launch_status=response.follow_up_launch_status,
            follow_up_operator_decision=response.follow_up_operator_decision,
            deep_link={
                "target_tab": "domain",
                "params": {"tab": "domain"},
                "label": "Open Domain Profiles",
            },
            metadata={"profile_opportunity_id": str(request.profile_opportunity_id or "")},
            after_state={
                "profile_opportunity_id": str(request.profile_opportunity_id or ""),
                "follow_up_launch_status": response.follow_up_launch_status,
                "follow_up_operator_decision": response.follow_up_operator_decision,
            },
        )
        await db.commit()
        return response

    portfolio_result = await db.execute(
        select(ResearchPortfolio).where(
            and_(
                ResearchPortfolio.id == request.portfolio_id,
                ResearchPortfolio.user_id == current_user.id,
            )
        )
    )
    portfolio = portfolio_result.scalar_one_or_none()
    if portfolio is None:
        raise HTTPException(status_code=404, detail="Research portfolio not found")
    parent_job = await _resolve_portfolio_parent_job_for_queue(db=db, portfolio=portfolio)
    source_scheduler_state = _extract_scheduler_state(parent_job)

    response = await _perform_follow_up_queue_action(
        portfolio=portfolio,
        portfolio_opportunity_id=request.portfolio_opportunity_id,
        action=request.action,
        operator_note=request.operator_note,
        db=db,
        current_user=current_user,
    )
    reason_label = _follow_up_opportunity_reason_label(portfolio=portfolio, opportunity_id=str(request.portfolio_opportunity_id or "").strip())
    await _record_follow_up_queue_decision_event(
        db=db,
        current_user=current_user,
        action=request.action,
        operator_note=request.operator_note,
        source_kind="portfolio",
        source_id=str(portfolio.id),
        source_label=str(portfolio.title or "Research fleet").strip(),
        customer=None,
        reason_code=str(request.portfolio_opportunity_id or "").strip() or None,
        reason_label=reason_label,
        scheduler_state=source_scheduler_state,
        follow_up_launch_status=response.follow_up_launch_status,
        follow_up_operator_decision=response.follow_up_operator_decision,
        deep_link={
            "target_tab": "fleet",
            "params": {"tab": "fleet", "fleetId": str(portfolio.id)},
            "label": "Open Research Fleet",
        },
        metadata={"portfolio_opportunity_id": str(request.portfolio_opportunity_id or "")},
        after_state={
            "portfolio_opportunity_id": str(request.portfolio_opportunity_id or ""),
            "follow_up_launch_status": response.follow_up_launch_status,
            "follow_up_operator_decision": response.follow_up_operator_decision,
        },
    )
    await db.commit()
    return response


@router.post(
    "/checkpoint-queue/follow-up-bulk-action",
    response_model=AgentCheckpointQueueBulkFollowUpActionResponse,
)
async def checkpoint_queue_bulk_follow_up_action(
    request: AgentCheckpointQueueBulkFollowUpActionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Apply a homogeneous bulk follow-up action to one profile or portfolio."""
    action = str(request.action or "").strip().lower()
    if action not in {"approve_launch", "reject_launch"}:
        raise HTTPException(status_code=400, detail="Unknown follow-up queue action")

    results: list[AgentCheckpointQueueBulkFollowUpActionResultResponse] = []
    if request.domain_research_profile_id is not None:
        profile_result = await db.execute(
            select(DomainResearchProfile).where(
                and_(
                    DomainResearchProfile.id == request.domain_research_profile_id,
                    DomainResearchProfile.user_id == current_user.id,
                )
            )
        )
        profile = profile_result.scalar_one_or_none()
        if profile is None:
            raise HTTPException(status_code=404, detail="Domain research profile not found")
        parent_job = await _resolve_profile_parent_job_for_queue(db=db, profile=profile)
        source_scheduler_state = _extract_scheduler_state(parent_job)
        requested_ids = list(dict.fromkeys(request.profile_opportunity_ids))
        for opportunity_id in requested_ids:
            try:
                response = await _perform_follow_up_queue_action(
                    profile=profile,
                    profile_opportunity_id=opportunity_id,
                    action=action,
                    operator_note=request.operator_note,
                    db=db,
                    current_user=current_user,
                )
                await _record_follow_up_queue_decision_event(
                    db=db,
                    current_user=current_user,
                    action=action,
                    operator_note=request.operator_note,
                    source_kind="domain_profile",
                    source_id=str(profile.id),
                    source_label=str(profile.title or "Domain profile").strip(),
                    customer=str(profile.customer_context or "").strip() or None,
                    reason_code=opportunity_id,
                    reason_label=_follow_up_opportunity_reason_label(profile=profile, opportunity_id=opportunity_id),
                    scheduler_state=source_scheduler_state,
                    follow_up_launch_status=response.follow_up_launch_status,
                    follow_up_operator_decision=response.follow_up_operator_decision,
                    deep_link={"target_tab": "domain", "params": {"tab": "domain"}, "label": "Open Domain Profiles"},
                    metadata={"profile_opportunity_id": opportunity_id},
                    after_state={
                        "profile_opportunity_id": opportunity_id,
                        "follow_up_launch_status": response.follow_up_launch_status,
                        "follow_up_operator_decision": response.follow_up_operator_decision,
                    },
                )
                results.append(
                    AgentCheckpointQueueBulkFollowUpActionResultResponse(
                        domain_research_profile_id=profile.id,
                        profile_opportunity_id=opportunity_id,
                        ok=True,
                        follow_up_launch_status=response.follow_up_launch_status,
                        follow_up_operator_decision=response.follow_up_operator_decision,
                        follow_up_job_id=response.follow_up_job_id,
                        detail=response.detail,
                    )
                )
            except HTTPException as exc:
                results.append(
                    AgentCheckpointQueueBulkFollowUpActionResultResponse(
                        domain_research_profile_id=profile.id,
                        profile_opportunity_id=opportunity_id,
                        ok=False,
                        error=str(exc.detail),
                    )
                )
        await db.commit()
        applied = sum(1 for row in results if row.ok)
        return AgentCheckpointQueueBulkFollowUpActionResponse(
            requested_count=len(requested_ids),
            applied=applied,
            failed=len(results) - applied,
            results=results,
        )

    portfolio_result = await db.execute(
        select(ResearchPortfolio).where(
            and_(
                ResearchPortfolio.id == request.portfolio_id,
                ResearchPortfolio.user_id == current_user.id,
            )
        )
    )
    portfolio = portfolio_result.scalar_one_or_none()
    if portfolio is None:
        raise HTTPException(status_code=404, detail="Research portfolio not found")
    parent_job = await _resolve_portfolio_parent_job_for_queue(db=db, portfolio=portfolio)
    source_scheduler_state = _extract_scheduler_state(parent_job)
    requested_ids = list(dict.fromkeys(request.portfolio_opportunity_ids))
    for opportunity_id in requested_ids:
        try:
            response = await _perform_follow_up_queue_action(
                portfolio=portfolio,
                portfolio_opportunity_id=opportunity_id,
                action=action,
                operator_note=request.operator_note,
                db=db,
                current_user=current_user,
            )
            await _record_follow_up_queue_decision_event(
                db=db,
                current_user=current_user,
                action=action,
                operator_note=request.operator_note,
                source_kind="portfolio",
                source_id=str(portfolio.id),
                source_label=str(portfolio.title or "Research fleet").strip(),
                customer=None,
                reason_code=opportunity_id,
                reason_label=_follow_up_opportunity_reason_label(portfolio=portfolio, opportunity_id=opportunity_id),
                scheduler_state=source_scheduler_state,
                follow_up_launch_status=response.follow_up_launch_status,
                follow_up_operator_decision=response.follow_up_operator_decision,
                deep_link={"target_tab": "fleet", "params": {"tab": "fleet", "fleetId": str(portfolio.id)}, "label": "Open Research Fleet"},
                metadata={"portfolio_opportunity_id": opportunity_id},
                after_state={
                    "portfolio_opportunity_id": opportunity_id,
                    "follow_up_launch_status": response.follow_up_launch_status,
                    "follow_up_operator_decision": response.follow_up_operator_decision,
                },
            )
            results.append(
                AgentCheckpointQueueBulkFollowUpActionResultResponse(
                    portfolio_id=portfolio.id,
                    portfolio_opportunity_id=opportunity_id,
                    ok=True,
                    follow_up_launch_status=response.follow_up_launch_status,
                    follow_up_operator_decision=response.follow_up_operator_decision,
                    follow_up_job_id=response.follow_up_job_id,
                    detail=response.detail,
                )
            )
        except HTTPException as exc:
            results.append(
                AgentCheckpointQueueBulkFollowUpActionResultResponse(
                    portfolio_id=portfolio.id,
                    portfolio_opportunity_id=opportunity_id,
                    ok=False,
                    error=str(exc.detail),
                )
            )
    await db.commit()
    applied = sum(1 for row in results if row.ok)
    return AgentCheckpointQueueBulkFollowUpActionResponse(
        requested_count=len(requested_ids),
        applied=applied,
        failed=len(results) - applied,
        results=results,
    )


def _validate_bulk_queue_action(item_type: str, action: str) -> None:
    normalized_item_type = str(item_type or "").strip().lower()
    normalized_action = str(action or "").strip().lower()
    allowed = QUEUE_BULK_ACTIONS.get(normalized_item_type)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Bulk actions are only supported for approval_checkpoint and job_recovery items",
        )
    if normalized_action not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Action {normalized_action} is not allowed for queue item type {normalized_item_type}",
        )


def _job_matches_bulk_queue_item_type(job: AgentJob, item_type: str) -> tuple[bool, Optional[str]]:
    normalized_item_type = str(item_type or "").strip().lower()
    if normalized_item_type == "approval_checkpoint":
        _, _, pending_checkpoint = _approval_payload_from_results(job.results)
        if job.status != AgentJobStatus.PAUSED.value or not isinstance(pending_checkpoint, dict):
            return False, "Job is not currently paused on an approval checkpoint"
        return True, None

    if normalized_item_type == "job_recovery":
        scheduler_state = (
            ((job.results or {}).get("execution_strategy") or {}).get("scheduler_state")
            if isinstance(job.results, dict)
            else None
        )
        queue_reason = (
            str((scheduler_state or {}).get("queue_reason") or "").strip().lower()
            if isinstance(scheduler_state, dict)
            else ""
        )
        if queue_reason not in {"execution_failure", "stalled_run", "scheduler_backoff"}:
            return False, "Job is not currently represented as a recovery queue item"
        return True, None

    return False, "Unsupported queue item type"


@router.post("/checkpoint-queue/bulk-action", response_model=AgentCheckpointQueueBulkActionResponse)
async def checkpoint_queue_bulk_action(
    request: AgentCheckpointQueueBulkActionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Apply a safe homogeneous bulk action to queue-backed jobs."""
    item_type = str(request.item_type or "").strip().lower()
    action = str(request.action or "").strip().lower()
    _validate_bulk_queue_action(item_type, action)

    requested_ids = list(dict.fromkeys(request.job_ids))
    jobs_result = await db.execute(
        select(AgentJob)
        .options(selectinload(AgentJob.agent_definition))
        .where(and_(AgentJob.user_id == current_user.id, AgentJob.id.in_(requested_ids)))
    )
    jobs = list(jobs_result.scalars().all())
    job_by_id = {job.id: job for job in jobs}

    results: list[AgentCheckpointQueueBulkActionResultResponse] = []
    for job_id in requested_ids:
        job = job_by_id.get(job_id)
        if job is None:
            results.append(
                AgentCheckpointQueueBulkActionResultResponse(
                    job_id=job_id,
                    ok=False,
                    error="Agent job not found",
                )
            )
            continue

        matches, mismatch_reason = _job_matches_bulk_queue_item_type(job, item_type)
        if not matches:
            results.append(
                AgentCheckpointQueueBulkActionResultResponse(
                    job_id=job_id,
                    ok=False,
                    status=str(job.status or ""),
                    queue_key=f"{item_type}:{job.id}",
                    error=mismatch_reason or "Job does not match selected queue item type",
                )
            )
            continue

        try:
            async with db.begin_nested():
                previous_status = str(job.status or "")
                updated_job = await _perform_job_action(
                    job,
                    AgentJobActionRequest(
                        action=action,
                        checkpoint_note=request.checkpoint_note,
                    ),
                    db=db,
                    current_user=current_user,
                )
                await db.flush()
                await _record_job_operator_event(
                    db=db,
                    job=job,
                    current_user=current_user,
                    action=action,
                    note=request.checkpoint_note,
                    previous_status=previous_status,
                    next_status=str(updated_job.status or ""),
                    scheduler_state=_extract_scheduler_state(job),
                    metadata={"queue_item_type": item_type, "bulk_action": True},
                    summary=f"{str(job.name or 'Agent job').strip()}: bulk {action.replace('_', ' ')}",
                )
                results.append(
                    AgentCheckpointQueueBulkActionResultResponse(
                        job_id=job_id,
                        ok=True,
                        status=str(updated_job.status or ""),
                        queue_key=f"{item_type}:{updated_job.id}",
                    )
                )
        except HTTPException as exc:
            results.append(
                AgentCheckpointQueueBulkActionResultResponse(
                    job_id=job_id,
                    ok=False,
                    status=str(job.status or ""),
                    queue_key=f"{item_type}:{job.id}",
                    error=str(exc.detail),
                )
            )

    await db.commit()
    applied = sum(1 for row in results if row.ok)
    failed = len(results) - applied
    return AgentCheckpointQueueBulkActionResponse(
        requested_count=len(requested_ids),
        applied=applied,
        failed=failed,
        results=results,
    )


@router.get("/{job_id}/log")
async def get_job_log(
    job_id: UUID,
    limit: int = Query(50, ge=1, le=500, description="Number of log entries"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get execution log for an agent job.

    Returns paginated log entries.
    """
    result = await db.execute(
        select(AgentJob).where(
            and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id)
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    log = job.execution_log or []
    total = len(log)

    # Apply pagination
    paginated_log = log[offset:offset + limit]

    return {
        "entries": paginated_log,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + limit) < total,
    }


@router.get("/{job_id}/step-events")
async def get_job_step_events(
    job_id: UUID,
    limit: int = Query(100, ge=1, le=500, description="Number of step events"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get per-step execution/approval events for an agent job."""
    result = await db.execute(
        select(AgentJob).where(
            and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id)
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    results_payload = job.results if isinstance(job.results, dict) else {}
    execution = (
        results_payload.get("execution_strategy")
        if isinstance(results_payload.get("execution_strategy"), dict)
        else {}
    )
    result_rows = execution.get("step_events") if isinstance(execution.get("step_events"), list) else []

    checkpoint_row = await _load_latest_job_checkpoint(job.id, db)
    checkpoint_state = checkpoint_row.state if checkpoint_row and isinstance(checkpoint_row.state, dict) else {}
    checkpoint_rows = checkpoint_state.get("step_events") if isinstance(checkpoint_state.get("step_events"), list) else []

    # Prefer the richer source; checkpoint may be newer while paused, results may be richer after completion.
    if checkpoint_rows and len(checkpoint_rows) >= len(result_rows):
        source = "checkpoint_state"
        rows = checkpoint_rows
    else:
        source = "results_execution_strategy"
        rows = result_rows
    rows = [r for r in rows if isinstance(r, dict)]
    total = len(rows)
    paginated = rows[offset:offset + limit]

    return {
        "items": paginated,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + limit) < total,
        "source": source,
    }


@router.get("/{job_id}/export")
async def export_job_results(
    job_id: UUID,
    format: str = Query("docx", description="Export format: docx, pdf, or pptx"),
    style: str = Query("professional", description="Visual style: professional, technical, or casual"),
    include_log: bool = Query(False, description="Include execution log in export"),
    include_metadata: bool = Query(True, description="Include job metadata in export"),
    enhance: bool = Query(False, description="Use LLM to generate executive summary and insights"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Export agent job results to a document or presentation.

    Supported formats:
    - docx: Microsoft Word document
    - pdf: PDF document
    - pptx: PowerPoint presentation

    When enhance=true, uses LLM to generate:
    - Executive summary
    - Key insights
    - Recommendations

    Returns the file as a downloadable attachment.
    """
    from app.services.job_results_exporter import JobResultsExporter

    # Validate format
    if format not in ["docx", "pdf", "pptx"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format: {format}. Use docx, pdf, or pptx.",
        )

    # Validate style
    if style not in ["professional", "technical", "casual"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported style: {style}. Use professional, technical, or casual.",
        )

    # Get job
    result = await db.execute(
        select(AgentJob).where(
            and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id)
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    # Generate export
    try:
        exporter = JobResultsExporter(style=style)
        if enhance:
            # Apply per-user LLM settings to enhanced export generation.
            user_settings = None
            try:
                from app.models.memory import UserPreferences
                from app.services.llm_service import UserLLMSettings
                prefs_res = await db.execute(select(UserPreferences).where(UserPreferences.user_id == current_user.id))
                prefs = prefs_res.scalar_one_or_none()
                user_settings = UserLLMSettings.from_preferences(prefs) if prefs else None
            except Exception:
                user_settings = None
            # Use async LLM-enhanced export
            file_bytes = await exporter.export_enhanced(
                job=job,
                format=format,
                include_log=include_log,
                include_metadata=include_metadata,
                user_id=current_user.id,
                user_settings=user_settings,
            )
        else:
            # Use standard export
            file_bytes = exporter.export(
                job=job,
                format=format,
                include_log=include_log,
                include_metadata=include_metadata,
            )
    except Exception as e:
        logger.error(f"Failed to export job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Export failed: {str(e)}",
        )

    # Determine content type and filename
    content_types = {
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "pdf": "application/pdf",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    }

    # Sanitize job name for filename
    safe_name = "".join(c for c in job.name if c.isalnum() or c in (' ', '-', '_')).strip()
    safe_name = safe_name[:50] or "agent_job"
    filename = f"{safe_name}_report.{format}"

    logger.info(f"Exported job {job_id} as {format} for user {current_user.id}")

    return Response(
        content=file_bytes,
        media_type=content_types[format],
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )


@router.get("/{job_id}/checkpoints", response_model=list[AgentJobCheckpointResponse])
async def get_job_checkpoints(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get checkpoints for an agent job.

    Useful for debugging and understanding job progress.
    """
    # Verify job belongs to user
    job_result = await db.execute(
        select(AgentJob).where(
            and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id)
        )
    )
    job = job_result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    result = await db.execute(
        select(AgentJobCheckpoint)
        .where(AgentJobCheckpoint.job_id == job_id)
        .order_by(AgentJobCheckpoint.created_at.desc())
    )
    checkpoints = result.scalars().all()

    return [AgentJobCheckpointResponse.model_validate(cp) for cp in checkpoints]


@router.websocket("/{job_id}/progress")
async def agent_job_progress_websocket(
    websocket: WebSocket,
    job_id: str,
    token: str = Query(...),
):
    """
    WebSocket endpoint for real-time agent job progress updates.

    Subscribes to Redis pub/sub for progress updates from the Celery worker.
    """
    from app.core.config import settings
    from app.api.endpoints.auth import get_user_from_token
    from app.core.database import async_session_factory
    from app.utils.websocket_manager import websocket_manager

    # Authenticate
    try:
        user = await get_user_from_token(token)
        if not user:
            await websocket.close(code=4001, reason="Invalid token")
            return
    except Exception:
        await websocket.close(code=4001, reason="Authentication failed")
        return

    # Verify job ownership
    async with async_session_factory() as db:
        job_result = await db.execute(
            select(AgentJob).where(
                and_(AgentJob.id == UUID(job_id), AgentJob.user_id == user.id)
            )
        )
        job = job_result.scalar_one_or_none()

        if not job:
            await websocket.accept()
            await websocket.send_json({"type": "error", "error": "Job not found"})
            await websocket.close(code=4004, reason="Job not found")
            return

    # Accept connection
    await websocket.accept()

    # Send initial state
    await websocket.send_json({
        "type": "connected",
        "job_id": job_id,
        "status": job.status,
        "progress": job.progress,
    })

    # Connect to Redis pub/sub
    redis_client = None
    pubsub = None

    try:
        redis_client = redis.from_url(settings.REDIS_URL)
        pubsub = redis_client.pubsub()
        channel = f"agent_job:{job_id}:progress"

        await pubsub.subscribe(channel)
        logger.info(f"WebSocket subscribed to {channel}")

        # Listen for messages
        while True:
            try:
                # Check for WebSocket messages (ping/close)
                try:
                    msg = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=0.1
                    )
                    if msg == "ping":
                        await websocket.send_text("pong")
                except asyncio.TimeoutError:
                    pass

                # Check for Redis messages
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0
                )

                if message and message["type"] == "message":
                    data = json.loads(message["data"])
                    await websocket.send_json(data)

                    # Close on completion
                    if data.get("status") in ["completed", "failed", "cancelled"]:
                        logger.info(f"Job {job_id} finished, closing WebSocket")
                        break

            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for job {job_id}")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket loop: {e}")
                break

    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except:
            pass

    finally:
        # Cleanup
        if pubsub:
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.close()
            except:
                pass
        if redis_client:
            try:
                await redis_client.close()
            except:
                pass
        try:
            await websocket.close()
        except:
            pass


    # Get first step configuration
    job_config = chain.create_job_config_for_step(
        step_index=0,
        variables=request.variables,
        parent_results=None,
    )

    if not job_config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create job configuration from chain",
        )

    # Merge with config overrides
    default_settings = chain.default_settings or {}
    if request.config_overrides:
        default_settings.update(request.config_overrides)
    default_settings = _normalize_scope_keys_deep(default_settings) or {}

    # Build chain config for next step trigger
    chain_config = None
    if len(chain.chain_steps) > 1:
        next_step = chain.chain_steps[1]
        chain_config = {
            "trigger_condition": chain.chain_steps[0].get("trigger_condition", "on_complete"),
            "inherit_results": default_settings.get("inherit_results", True),
            "chain_definition_id": str(chain.id),
            "current_step_index": 0,
            "total_steps": len(chain.chain_steps),
            "variables": request.variables,
            "child_jobs": [{
                "name": f"{request.name_prefix} - {next_step.get('step_name', 'Step 2')}",
                "job_type": next_step.get("job_type", "custom"),
                "goal": _substitute_variables(next_step.get("goal_template", ""), request.variables),
                "config": _merge_chain_step_config(default_settings, next_step.get("config", {}) or {}),
                "chain_config": _build_chain_config_for_step(chain, 1, request.variables, default_settings) if len(chain.chain_steps) > 2 else None,
            }],
        }
        if chain.chain_steps[0].get("trigger_thresholds"):
            chain_config.update(chain.chain_steps[0]["trigger_thresholds"])

    # Create the first job
    job = AgentJob(
        name=f"{request.name_prefix} - {job_config.get('name', 'Step 1')}",
        description=f"Chain: {chain.display_name}",
        job_type=job_config.get("job_type", "custom"),
        goal=job_config.get("goal", ""),
        config=_merge_chain_step_config(default_settings, job_config.get("config", {}) or {}),
        user_id=current_user.id,
        status=AgentJobStatus.PENDING.value,
        max_iterations=default_settings.get("max_iterations", 100),
        max_tool_calls=default_settings.get("max_tool_calls", 500),
        max_llm_calls=default_settings.get("max_llm_calls", 200),
        max_runtime_minutes=default_settings.get("max_runtime_minutes", 60),
        chain_config=_normalize_scope_keys_deep(chain_config),
        chain_depth=0,
    )

    db.add(job)
    await db.commit()
    await db.refresh(job)

    logger.info(f"Created chain job {job.id} from chain {chain.name} for user {current_user.id}")

    # Start immediately if requested
    if request.start_immediately:
        execute_agent_job_task.delay(str(job.id), str(current_user.id))

    return _job_to_response(job)


@router.get("/{job_id}/chain-status", response_model=AgentJobChainStatusResponse)
async def get_chain_status(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Get the status of an entire job chain.

    Works with any job in the chain - finds the root and returns full chain status.
    """
    # Get the job
    result = await db.execute(
        select(AgentJob)
        .options(selectinload(AgentJob.agent_definition))
        .where(and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id))
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    # Find root job
    root_job_id = job.root_job_id or job.id

    # Get all jobs in chain
    chain_result = await db.execute(
        select(AgentJob)
        .options(selectinload(AgentJob.agent_definition))
        .where(
            and_(
                AgentJob.user_id == current_user.id,
                or_(
                    AgentJob.id == root_job_id,
                    AgentJob.root_job_id == root_job_id,
                )
            )
        )
        .order_by(AgentJob.chain_depth, AgentJob.created_at)
    )
    chain_jobs = chain_result.scalars().all()

    # Calculate chain status
    total_steps = len(chain_jobs)
    completed_steps = len([j for j in chain_jobs if j.status == AgentJobStatus.COMPLETED.value])
    failed_jobs = [j for j in chain_jobs if j.status == AgentJobStatus.FAILED.value]
    running_jobs = [j for j in chain_jobs if j.status == AgentJobStatus.RUNNING.value]

    # Determine current step
    current_step = 0
    for i, j in enumerate(chain_jobs):
        if j.status in [AgentJobStatus.RUNNING.value, AgentJobStatus.PENDING.value]:
            current_step = i
            break
        elif j.status == AgentJobStatus.COMPLETED.value:
            current_step = i + 1

    # Calculate overall progress
    overall_progress = 0
    if total_steps > 0:
        for j in chain_jobs:
            overall_progress += j.progress
        overall_progress = overall_progress // total_steps

    # Determine chain status
    if failed_jobs:
        chain_status = "failed"
    elif completed_steps == total_steps:
        chain_status = "completed"
    elif running_jobs:
        chain_status = "running"
    elif completed_steps > 0:
        chain_status = "partially_completed"
    else:
        chain_status = "pending"

    # Get chain definition ID if available
    chain_definition_id = None
    if chain_jobs and chain_jobs[0].chain_config:
        chain_definition_id = chain_jobs[0].chain_config.get("chain_definition_id")

    return AgentJobChainStatusResponse(
        root_job_id=root_job_id,
        chain_definition_id=UUID(chain_definition_id) if chain_definition_id else None,
        total_steps=total_steps,
        completed_steps=completed_steps,
        current_step=current_step,
        overall_progress=overall_progress,
        status=chain_status,
        jobs=[_job_to_response(j) for j in chain_jobs],
    )


@router.post("/{job_id}/save-as-chain", response_model=AgentJobChainDefinitionResponse, status_code=status.HTTP_201_CREATED)
async def save_job_as_chain_definition(
    job_id: UUID,
    request: AgentJobSaveAsChainRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Save a completed (or in-progress) job chain as a reusable chain definition ("playbook").

    This primarily uses `chain_config.child_jobs` to reconstruct a linear chain, falling back
    to the persisted job hierarchy when needed.
    """
    result = await db.execute(
        select(AgentJob)
        .options(selectinload(AgentJob.agent_definition))
        .where(and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id))
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent job not found")

    # Prefer the root job if available
    root_job_id = job.root_job_id or job.id
    root_job = await db.get(AgentJob, root_job_id)
    if root_job is None:
        root_job = job
    is_recovery_playbook, recovery_reason_label, recovery_summary = _build_recovery_playbook_metadata(root_job)

    def _safe_step_name(name: str) -> str:
        s = (name or "").strip()
        return s[:200] if s else "Step"

    def _step_from_payload(payload: dict, chain_cfg: Optional[dict]) -> dict:
        trig = (chain_cfg or {}).get("trigger_condition") or "on_complete"
        thresholds = None
        if isinstance(chain_cfg, dict) and isinstance(chain_cfg.get("progress_threshold"), int):
            thresholds = {"progress_threshold": int(chain_cfg["progress_threshold"])}
        if isinstance(chain_cfg, dict) and isinstance(chain_cfg.get("findings_threshold"), int):
            thresholds = {**(thresholds or {}), "findings_threshold": int(chain_cfg["findings_threshold"])}

        step = {
            "step_name": _safe_step_name(str(payload.get("name") or "Step")),
            "template_id": None,
            "job_type": str(payload.get("job_type") or "custom"),
            "goal_template": str(payload.get("goal") or ""),
            "config": payload.get("config") if isinstance(payload.get("config"), dict) else None,
            "trigger_condition": str(trig),
            "trigger_thresholds": thresholds,
        }
        return step

    # Build chain steps (linear) from chain_config child_jobs nesting
    steps: list[dict] = []
    cur_payload: dict = {
        "name": root_job.name,
        "job_type": root_job.job_type,
        "goal": root_job.goal,
        "config": root_job.config,
    }
    cur_chain_cfg: Optional[dict] = root_job.chain_config if isinstance(root_job.chain_config, dict) else None
    seen = set()

    while True:
        steps.append(_step_from_payload(cur_payload, cur_chain_cfg))
        if len(steps) >= 25:
            break

        child_jobs = (cur_chain_cfg or {}).get("child_jobs") if isinstance(cur_chain_cfg, dict) else None
        if not isinstance(child_jobs, list) or not child_jobs:
            break

        child0 = child_jobs[0]
        if not isinstance(child0, dict):
            break
        key = json.dumps(child0, sort_keys=True, default=str)[:2000]
        if key in seen:
            break
        seen.add(key)

        cur_payload = child0
        cur_chain_cfg = child0.get("chain_config") if isinstance(child0.get("chain_config"), dict) else None

    # If we only captured one step and the job tree exists in DB, attempt to extend using persisted hierarchy.
    if len(steps) <= 1:
        chain_result = await db.execute(
            select(AgentJob)
            .where(
                and_(
                    AgentJob.user_id == current_user.id,
                    or_(
                        AgentJob.id == root_job_id,
                        AgentJob.root_job_id == root_job_id,
                    ),
                )
            )
            .order_by(AgentJob.chain_depth, AgentJob.created_at)
        )
        chain_jobs = list(chain_result.scalars().all())
        if chain_jobs:
            by_parent: dict[UUID, list[AgentJob]] = {}
            for j in chain_jobs:
                if j.parent_job_id:
                    by_parent.setdefault(j.parent_job_id, []).append(j)
            for kids in by_parent.values():
                kids.sort(key=lambda x: (x.created_at or datetime.utcnow(), str(x.id)))

            linear: list[AgentJob] = []
            cur = chain_jobs[0]
            linear.append(cur)
            while len(linear) < 25:
                kids = by_parent.get(cur.id) or []
                if not kids:
                    break
                cur = kids[0]
                linear.append(cur)

            steps = []
            for j in linear:
                cfg = j.chain_config if isinstance(j.chain_config, dict) else None
                steps.append(
                    _step_from_payload(
                        {"name": j.name, "job_type": j.job_type, "goal": j.goal, "config": j.config},
                        cfg,
                    )
                )

    # Ensure last step doesn't imply a trigger if no child exists (cosmetic)
    if steps:
        steps[-1]["trigger_condition"] = "on_complete"
        steps[-1]["trigger_thresholds"] = None

    now = datetime.utcnow()

    def _slugify(s: str) -> str:
        s2 = re.sub(r"[^a-z0-9_]+", "_", (s or "").strip().lower())
        s2 = re.sub(r"_+", "_", s2).strip("_")
        return s2[:40] or "job"

    requested_name = (request.name or "").strip() if request.name else ""
    if requested_name:
        existing = await db.execute(select(AgentJobChainDefinition).where(AgentJobChainDefinition.name == requested_name))
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Chain definition name already exists")
        name = requested_name
    else:
        base = _slugify(root_job.name)
        prefix = "playbook_recovery" if is_recovery_playbook else "playbook"
        name = f"{prefix}_{base}_{now.strftime('%Y%m%d_%H%M%S')}"
        name = name[:100]
        # Ensure uniqueness
        for _ in range(5):
            existing = await db.execute(select(AgentJobChainDefinition).where(AgentJobChainDefinition.name == name))
            if not existing.scalar_one_or_none():
                break
            name = (name[:90] + "_" + uuid.uuid4().hex[:8])[:100]

    display_name = (request.display_name or "").strip() if request.display_name else ""
    if not display_name:
        display_name = f"{root_job.name} ({'Recovery Playbook' if is_recovery_playbook else 'Playbook'})"

    description = (request.description or "").strip() if request.description else ""
    if not description:
        description = f"Saved from job {root_job_id} on {now.isoformat()}."
        if is_recovery_playbook:
            description = f"{description} Saved as a recovery playbook."
            if recovery_reason_label:
                description = f"{description} Recovery reason: {recovery_reason_label}."
            if recovery_summary:
                description = f"{description} {recovery_summary}"

    chain = AgentJobChainDefinition(
        name=name,
        display_name=display_name,
        description=description,
        chain_steps=steps,
        default_settings={
            "inherit_results": True,
            "inherit_config": True,
            "max_iterations": int(getattr(root_job, "max_iterations", 100) or 100),
            "max_tool_calls": int(getattr(root_job, "max_tool_calls", 500) or 500),
            "max_llm_calls": int(getattr(root_job, "max_llm_calls", 200) or 200),
            "max_runtime_minutes": int(getattr(root_job, "max_runtime_minutes", 60) or 60),
        },
        owner_user_id=current_user.id,
        is_system=False,
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    db.add(chain)
    await db.commit()
    await db.refresh(chain)

    return _chain_definition_to_response(chain)


def _substitute_variables(template: str, variables: dict) -> str:
    """Substitute {variable} placeholders in a template string."""
    result = template
    for key, value in variables.items():
        result = result.replace(f"{{{key}}}", value)
    return result


def _build_chain_config_for_step(
    chain: AgentJobChainDefinition,
    step_index: int,
    variables: dict,
    default_settings: dict,
) -> Optional[dict]:
    """Build chain config for a specific step in the chain."""
    if step_index >= len(chain.chain_steps):
        return None

    step = chain.chain_steps[step_index]

    config = {
        "trigger_condition": step.get("trigger_condition", "on_complete"),
        "inherit_results": default_settings.get("inherit_results", True),
        "chain_definition_id": str(chain.id),
        "current_step_index": step_index,
        "total_steps": len(chain.chain_steps),
        "variables": variables,
    }

    if step.get("trigger_thresholds"):
        config.update(step["trigger_thresholds"])

    # Add next step as child job
    if step_index + 1 < len(chain.chain_steps):
        next_step = chain.chain_steps[step_index + 1]
        config["child_jobs"] = [{
            "name": next_step.get("step_name", f"Step {step_index + 2}"),
            "job_type": next_step.get("job_type", "custom"),
            "goal": _substitute_variables(next_step.get("goal_template", ""), variables),
            "config": _merge_chain_step_config(default_settings, next_step.get("config", {}) or {}),
            "chain_config": _build_chain_config_for_step(chain, step_index + 1, variables, default_settings),
        }]

    return _normalize_scope_keys_deep(config)


def _is_recovery_playbook_candidate(job: AgentJob) -> bool:
    """Return True when a saved chain should be branded as a recovery playbook."""
    scheduler_state = _extract_scheduler_state(job) or {}
    queue_reason = str(scheduler_state.get("queue_reason") or "").strip().lower()
    if queue_reason in {"execution_failure", "stalled_run", "scheduled_recovery", "scheduler_backoff"}:
        return True
    status = str(getattr(job, "status", "") or "").strip().lower()
    if status in {AgentJobStatus.FAILED.value, AgentJobStatus.CANCELLED.value}:
        return bool(str(getattr(job, "error", "") or "").strip() or str(getattr(job, "phase_details", "") or "").strip() or queue_reason)
    if status == AgentJobStatus.PAUSED.value:
        return bool(queue_reason or str(getattr(job, "phase_details", "") or "").strip())
    return False


def _build_recovery_playbook_metadata(job: AgentJob) -> tuple[bool, Optional[str], Optional[str]]:
    """Build a compact recovery context for playbook naming and description defaults."""
    scheduler_state = _extract_scheduler_state(job) or {}
    queue_reason = str(scheduler_state.get("queue_reason") or "").strip().lower()
    is_recovery = _is_recovery_playbook_candidate(job)
    if not is_recovery:
        return False, None, None

    reason_label = _queue_reason_label(queue_reason) if queue_reason else None
    fragments: list[str] = []
    if reason_label:
        fragments.append(f"Recovery reason: {reason_label}.")
    status = str(getattr(job, "status", "") or "").strip().lower()
    if status:
        fragments.append(f"Current status: {status}.")
    error = str(getattr(job, "error", "") or "").strip()
    if error:
        fragments.append(f"Error: {error[:240]}.")
    phase_details = str(getattr(job, "phase_details", "") or "").strip()
    if phase_details and phase_details != error:
        fragments.append(f"Details: {phase_details[:240]}.")
    summary = " ".join(fragments).strip() or None
    return True, reason_label, summary


# ============================================================================
# Job Memory Endpoints
# ============================================================================

def _to_int(value: Any, default: int = 0) -> int:
    """Best-effort int coercion for resilient response serialization."""
    try:
        return int(value)
    except Exception:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float coercion for resilient response serialization."""
    try:
        parsed = float(value)
        if not math.isfinite(parsed):
            return default
        return parsed
    except Exception:
        return default


def _to_string_list(value: Any) -> list[str]:
    """Normalize any list-like payload into a clean list of non-empty strings."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        token = str(item or "").strip()
        if token:
            out.append(token)
    return out


def _to_string(value: Any, default: str = "") -> str:
    """Normalize scalar-like payloads into trimmed strings."""
    text = str(value or "").strip()
    return text if text else default


def _build_extract_job_memories_response(
    *,
    job_id: UUID,
    memories: list[Any],
    extraction_stats: Optional[dict] = None,
) -> AgentJobMemoryExtractResponse:
    """Serialize manual extraction response with dedup-aware extraction stats."""
    stats = extraction_stats if isinstance(extraction_stats, dict) else {}
    return AgentJobMemoryExtractResponse(
        job_id=str(job_id),
        memories_created=len(memories),
        parsed_count=_to_int(stats.get("parsed_count", 0), 0),
        candidate_count=_to_int(stats.get("candidate_count", 0), 0),
        skipped_duplicates=_to_int(stats.get("skipped_duplicates", 0), 0),
        is_relaunch_chain=bool(stats.get("is_relaunch_chain", False)),
        relaunch_root_job_id=str(stats.get("relaunch_root_job_id") or "").strip() or None,
        memories=[
            AgentJobExtractedMemoryResponse(
                id=str(m.id),
                type=_to_string(getattr(m, "memory_type", None), "unknown"),
                content=_to_string(getattr(m, "content", None), ""),
                importance_score=_to_float(getattr(m, "importance_score", 0.0), 0.0),
                tags=_to_string_list(getattr(m, "tags", None)),
            )
            for m in memories
        ],
    )


def _build_job_memory_response(
    *,
    job_id: UUID | str,
    memory: Any,
) -> AgentJobMemoryResponse:
    """Serialize one ConversationMemory-like object for job memory endpoints."""
    return AgentJobMemoryResponse(
        id=str(memory.id),
        job_id=str(job_id),
        type=_to_string(getattr(memory, "memory_type", None), "unknown"),
        content=_to_string(getattr(memory, "content", None), ""),
        importance_score=_to_float(getattr(memory, "importance_score", 0.0), 0.0),
        tags=_to_string_list(getattr(memory, "tags", None)),
        context=memory.context if isinstance(memory.context, dict) else None,
        access_count=_to_int(getattr(memory, "access_count", 0), 0),
        created_at=memory.created_at.isoformat() if getattr(memory, "created_at", None) else None,
    )


def _build_job_memories_list_response(
    *,
    job_id: UUID,
    memories: list[Any],
) -> AgentJobMemoryListResponse:
    """Serialize list response for one job's memories."""
    return AgentJobMemoryListResponse(
        job_id=str(job_id),
        memories=[
            _build_job_memory_response(job_id=job_id, memory=m)
            for m in memories
        ],
        total=len(memories),
    )


def _build_memory_search_response(
    *,
    query: str,
    memories: list[Any],
) -> AgentJobMemorySearchResponse:
    """Serialize search response payload for memory search endpoint."""
    return AgentJobMemorySearchResponse(
        query=query,
        memories=[
            AgentJobMemorySearchItemResponse(
                id=str(m.id),
                type=_to_string(getattr(m, "memory_type", None), "unknown"),
                content=_to_string(getattr(m, "content", None), ""),
                importance_score=_to_float(getattr(m, "importance_score", 0.0), 0.0),
                tags=_to_string_list(getattr(m, "tags", None)),
                job_id=str(m.job_id) if getattr(m, "job_id", None) else None,
                access_count=_to_int(getattr(m, "access_count", 0), 0),
                created_at=m.created_at.isoformat() if getattr(m, "created_at", None) else None,
            )
            for m in memories
        ],
        total=len(memories),
    )


def _build_memory_stats_response(
    *,
    stats: Optional[dict[str, Any]],
) -> AgentJobMemoryStatsResponse:
    """Normalize and serialize aggregate memory stats payload."""
    payload = stats if isinstance(stats, dict) else {}

    by_type: dict[str, int] = {}
    by_type_raw = payload.get("by_type")
    if isinstance(by_type_raw, dict):
        for key, value in by_type_raw.items():
            token = str(key or "").strip()
            if not token:
                continue
            try:
                by_type[token] = int(value or 0)
            except Exception:
                by_type[token] = 0

    most_accessed_rows: list[dict[str, Any]] = []
    most_accessed_raw = payload.get("most_accessed")
    if isinstance(most_accessed_raw, list):
        for item in most_accessed_raw:
            if not isinstance(item, dict):
                continue
            most_accessed_rows.append(
                {
                    "id": str(item.get("id") or ""),
                    "type": str(item.get("type") or ""),
                    "content": str(item.get("content") or ""),
                    "access_count": _to_int(item.get("access_count"), 0),
                }
            )

    most_important_rows: list[dict[str, Any]] = []
    most_important_raw = payload.get("most_important")
    if isinstance(most_important_raw, list):
        for item in most_important_raw:
            if not isinstance(item, dict):
                continue
            most_important_rows.append(
                {
                    "id": str(item.get("id") or ""),
                    "type": str(item.get("type") or ""),
                    "content": str(item.get("content") or ""),
                    "importance": _to_float(item.get("importance"), 0.0),
                }
            )

    return AgentJobMemoryStatsResponse(
        total_memories=_to_int(payload.get("total_memories", 0), 0),
        by_type=by_type,
        job_sourced=_to_int(payload.get("job_sourced", 0), 0),
        chat_sourced=_to_int(payload.get("chat_sourced", 0), 0),
        manual=_to_int(payload.get("manual", 0), 0),
        most_accessed=most_accessed_rows,
        most_important=most_important_rows,
    )


def _build_memory_graph_response(
    *,
    graph: Optional[dict[str, Any]],
    job_id: Optional[UUID | str] = None,
) -> AgentJobMemoryGraphResponse:
    """Normalize and serialize task-memory graph payload."""
    payload = graph if isinstance(graph, dict) else {}

    nodes: list[dict[str, Any]] = []
    nodes_raw = payload.get("nodes")
    if isinstance(nodes_raw, list):
        for node in nodes_raw:
            if not isinstance(node, dict):
                continue
            tags_list = _to_string_list(node.get("tags"))
            nodes.append(
                {
                    "id": str(node.get("id") or ""),
                    "type": str(node.get("type") or ""),
                    "content": str(node.get("content") or ""),
                    "importance_score": _to_float(node.get("importance_score"), 0.0),
                    "tags": tags_list,
                    "job_id": str(node.get("job_id") or "").strip() or None,
                    "created_at": str(node.get("created_at") or "").strip() or None,
                    "project_scope": str(node.get("project_scope") or "").strip() or None,
                    "execution_outcome": str(node.get("execution_outcome") or "").strip() or None,
                    "strategy_signal": str(node.get("strategy_signal") or "").strip() or None,
                    "access_count": _to_int(node.get("access_count"), 0),
                }
            )

    edges: list[dict[str, Any]] = []
    edges_raw = payload.get("edges")
    if isinstance(edges_raw, list):
        for edge in edges_raw:
            if not isinstance(edge, dict):
                continue
            reasons_list = _to_string_list(edge.get("reasons"))
            edges.append(
                {
                    "source": str(edge.get("source") or ""),
                    "target": str(edge.get("target") or ""),
                    "weight": _to_float(edge.get("weight"), 0.0),
                    "reasons": reasons_list,
                }
            )

    stats_raw = payload.get("stats")
    stats_out: dict[str, Any] = {}
    if isinstance(stats_raw, dict):
        for key, value in stats_raw.items():
            token = str(key or "").strip()
            if token:
                stats_out[token] = value

    normalized_job_id = str(job_id).strip() if job_id is not None else ""
    if not normalized_job_id:
        normalized_job_id = str(payload.get("job_id") or "").strip()

    return AgentJobMemoryGraphResponse(
        nodes=nodes,
        edges=edges,
        stats=stats_out,
        job_id=normalized_job_id or None,
    )


@router.get("/{job_id}/memories", response_model=AgentJobMemoryListResponse)
async def get_job_memories(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AgentJobMemoryListResponse:
    """
    Get all memories created from a specific job.

    Returns memories extracted from the job's results.
    """
    from app.services.agent_job_memory_service import agent_job_memory_service

    # Verify job belongs to user
    result = await db.execute(
        select(AgentJob).where(
            and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id)
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    memories = await agent_job_memory_service.get_job_memories(
        job_id=job_id,
        user_id=str(current_user.id),
        db=db,
    )

    return _build_job_memories_list_response(
        job_id=job_id,
        memories=memories,
    )


@router.post("/{job_id}/memories/extract", response_model=AgentJobMemoryExtractResponse)
async def extract_job_memories(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AgentJobMemoryExtractResponse:
    """
    Manually trigger memory extraction from a completed job.

    Useful for re-extracting memories or extracting from older jobs.
    """
    from app.services.agent_job_memory_service import agent_job_memory_service

    # Verify job belongs to user
    result = await db.execute(
        select(AgentJob).where(
            and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id)
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    if job.status not in ["completed", "failed"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only extract memories from completed or failed jobs",
        )

    try:
        extraction_stats: dict = {}
        memories = await agent_job_memory_service.extract_memories_from_job(
            job=job,
            user_id=str(current_user.id),
            db=db,
            extraction_reason="manual_extract",
            force_extract=True,
            stats_out=extraction_stats,
        )

        logger.info(f"Manually extracted {len(memories)} memories from job {job_id}")

        return _build_extract_job_memories_response(
            job_id=job_id,
            memories=memories,
            extraction_stats=extraction_stats,
        )
    except Exception as e:
        logger.error(f"Failed to extract memories from job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory extraction failed: {str(e)}",
        )


@router.post("/{job_id}/memories", response_model=AgentJobMemoryResponse)
async def create_job_memory(
    job_id: UUID,
    memory_type: str = Query(..., description="Memory type: finding, insight, pattern, or lesson"),
    content: str = Query(..., description="Memory content"),
    importance: float = Query(0.5, ge=0.0, le=1.0, description="Importance score"),
    tags: Optional[str] = Query(None, description="Comma-separated tags"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AgentJobMemoryResponse:
    """
    Manually create a memory from a job.

    Allows users to add custom memories derived from job insights.
    """
    from app.services.agent_job_memory_service import agent_job_memory_service

    # Validate memory type
    if memory_type not in ["finding", "insight", "pattern", "lesson", "fact", "context"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid memory type. Use: finding, insight, pattern, lesson, fact, or context",
        )

    # Verify job belongs to user
    result = await db.execute(
        select(AgentJob).where(
            and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id)
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    # Parse tags
    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    try:
        memory = await agent_job_memory_service.create_memory_from_job(
            job=job,
            memory_type=memory_type,
            content=content,
            user_id=str(current_user.id),
            db=db,
            importance=importance,
            tags=tag_list,
        )

        return _build_job_memory_response(
            job_id=job_id,
            memory=memory,
        )
    except Exception as e:
        logger.error(f"Failed to create memory for job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory creation failed: {str(e)}",
        )


@router.delete("/{job_id}/memories", response_model=AgentJobMemoryDeleteResponse)
async def delete_job_memories(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AgentJobMemoryDeleteResponse:
    """
    Delete all memories created from a job.

    Performs a soft delete (memories marked inactive, not removed).
    """
    from app.services.agent_job_memory_service import agent_job_memory_service

    # Verify job belongs to user
    result = await db.execute(
        select(AgentJob).where(
            and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id)
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent job not found",
        )

    deleted_count = await agent_job_memory_service.delete_job_memories(
        job_id=job_id,
        user_id=str(current_user.id),
        db=db,
    )

    return AgentJobMemoryDeleteResponse(
        job_id=str(job_id),
        deleted_count=int(deleted_count or 0),
    )


@router.post("/{job_id}/feedback", response_model=AgentJobFeedbackResponse)
async def create_agent_job_feedback(
    job_id: UUID,
    payload: AgentJobFeedbackCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """
    Store human feedback for a job/checkpoint and convert it into learning memory.

    Feedback is persisted as a `lesson` memory tagged with `human_feedback`, then
    used by the autonomous executor to bias future tool choices and prompts.
    """
    from app.services.agent_job_memory_service import agent_job_memory_service

    job_res = await db.execute(
        select(AgentJob).where(
            and_(
                AgentJob.id == job_id,
                AgentJob.user_id == current_user.id,
            )
        )
    )
    job = job_res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Agent job not found")

    scope = str(payload.scope or "user").strip().lower()
    if scope not in {"user", "customer", "team"}:
        raise HTTPException(status_code=400, detail="scope must be one of: user, customer, team")

    preferred_tools = _sanitize_tool_names(payload.preferred_tools)
    discouraged_tools = _sanitize_tool_names(payload.discouraged_tools)
    if preferred_tools and discouraged_tools:
        overlap = [t for t in preferred_tools if t in set(discouraged_tools)]
        if overlap:
            discouraged_tools = [t for t in discouraged_tools if t not in set(overlap)]

    target_type = str(payload.target_type or "job").strip().lower()
    if target_type not in {"job", "checkpoint", "finding", "action", "tool"}:
        raise HTTPException(status_code=400, detail="target_type must be job, checkpoint, finding, action, or tool")

    team_key = str(payload.team_key or "").strip()
    if scope == "team" and not team_key:
        raise HTTPException(status_code=400, detail="team_key is required when scope=team")

    scope_marker = f"user:{current_user.id}"
    if scope == "customer":
        customer = str((job.config or {}).get("customer") or "").strip()
        if not customer:
            raw_profile = await get_feature_str("ai_hub_customer_profile")
            if raw_profile:
                try:
                    cp = CustomerProfile.model_validate(json.loads(raw_profile))
                    customer = str(cp.id or cp.name or "").strip()
                except Exception:
                    customer = ""
        if not customer:
            raise HTTPException(status_code=400, detail="customer scope requires job.config.customer or ai_hub_customer_profile")
        scope_marker = f"customer:{customer[:120]}"
    elif scope == "team":
        scope_marker = f"team:{team_key[:120]}"

    rating = max(1, min(int(payload.rating), 5))
    sentiment = "positive" if rating >= 4 else ("negative" if rating <= 2 else "neutral")
    feedback_text = str(payload.feedback or "").strip()
    target_id = str(payload.target_id or "").strip()
    checkpoint = str(payload.checkpoint or "").strip()

    content = feedback_text or f"User rated {target_type} as {rating}/5."
    importance = min(1.0, max(0.35, 0.55 + (abs(rating - 3) * 0.1)))

    tags = [
        "human_feedback",
        "feedback",
        f"feedback:{sentiment}",
        f"rating:{rating}",
        f"job_type:{job.job_type}",
        f"target:{target_type}",
        f"scope:{scope}",
        scope_marker,
    ]
    tags.extend([f"prefer_tool:{t}" for t in preferred_tools])
    tags.extend([f"avoid_tool:{t}" for t in discouraged_tools])
    tags = list(dict.fromkeys([t for t in tags if str(t).strip()]))

    context = {
        "feedback_type": "human",
        "rating": rating,
        "feedback_text": feedback_text,
        "target_type": target_type,
        "target_id": target_id or None,
        "checkpoint": checkpoint or None,
        "scope": scope,
        "scope_marker": scope_marker,
        "preferred_tools": preferred_tools,
        "discouraged_tools": discouraged_tools,
        "job_id": str(job.id),
        "job_name": job.name,
        "job_type": job.job_type,
        "job_status": job.status,
        "recorded_at": datetime.utcnow().isoformat(),
    }

    memory = ConversationMemory(
        user_id=current_user.id,
        job_id=job.id,
        memory_type="lesson",
        content=content,
        importance_score=importance,
        tags=tags,
        context=context,
    )
    db.add(memory)
    job.add_log_entry(
        {
            "phase": "human_feedback_recorded",
            "rating": rating,
            "target_type": target_type,
            "scope": scope,
            "preferred_tools": preferred_tools[:8],
            "discouraged_tools": discouraged_tools[:8],
        }
    )
    await db.commit()
    await db.refresh(memory)

    try:
        await agent_job_memory_service.link_memories_into_task_graph(
            new_memories=[memory],
            user_id=str(current_user.id),
            db=db,
        )
        await db.refresh(memory)
    except Exception as graph_exc:
        logger.warning(f"Failed linking feedback memory {memory.id} into task graph: {graph_exc}")

    return _memory_to_feedback_response(memory)


@router.get("/{job_id}/feedback", response_model=AgentJobFeedbackListResponse)
async def list_agent_job_feedback(
    job_id: UUID,
    limit: int = Query(50, ge=1, le=200, description="Max feedback entries"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List human feedback entries captured for a specific job."""
    job_res = await db.execute(
        select(AgentJob).where(
            and_(
                AgentJob.id == job_id,
                AgentJob.user_id == current_user.id,
            )
        )
    )
    job = job_res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Agent job not found")

    res = await db.execute(
        select(ConversationMemory)
        .where(
            and_(
                ConversationMemory.user_id == current_user.id,
                ConversationMemory.job_id == job_id,
                ConversationMemory.is_active == True,
                ConversationMemory.memory_type == "lesson",
            )
        )
        .order_by(ConversationMemory.created_at.desc())
        .limit(max(20, limit * 3))
    )
    memories = list(res.scalars().all())
    rows = []
    for memory in memories:
        tags = [str(t).strip().lower() for t in (memory.tags if isinstance(memory.tags, list) else []) if str(t).strip()]
        context = memory.context if isinstance(memory.context, dict) else {}
        is_feedback = ("human_feedback" in tags) or str(context.get("feedback_type") or "").strip().lower() == "human"
        if is_feedback:
            rows.append(_memory_to_feedback_response(memory))
        if len(rows) >= limit:
            break
    return AgentJobFeedbackListResponse(items=rows, total=len(rows))


@router.get("/memory/feedback", response_model=AgentJobFeedbackListResponse)
async def list_learning_feedback(
    scope: Optional[str] = Query(None, description="Optional scope filter: user|customer|team"),
    limit: int = Query(100, ge=1, le=300, description="Max feedback entries"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List recent human feedback memories used by the learning loop."""
    scope_filter = str(scope or "").strip().lower()
    if scope_filter and scope_filter not in {"user", "customer", "team"}:
        raise HTTPException(status_code=400, detail="scope must be user, customer, or team")

    res = await db.execute(
        select(ConversationMemory)
        .where(
            and_(
                ConversationMemory.user_id == current_user.id,
                ConversationMemory.is_active == True,
                ConversationMemory.memory_type == "lesson",
            )
        )
        .order_by(ConversationMemory.created_at.desc())
        .limit(max(50, limit * 3))
    )
    memories = list(res.scalars().all())
    items = []
    for memory in memories:
        tags = [str(t).strip().lower() for t in (memory.tags if isinstance(memory.tags, list) else []) if str(t).strip()]
        context = memory.context if isinstance(memory.context, dict) else {}
        if "human_feedback" not in tags and str(context.get("feedback_type") or "").strip().lower() != "human":
            continue
        if scope_filter and str(context.get("scope") or "").strip().lower() != scope_filter:
            continue
        items.append(_memory_to_feedback_response(memory))
        if len(items) >= limit:
            break
    return AgentJobFeedbackListResponse(items=items, total=len(items))


@router.get("/memory/graph", response_model=AgentJobMemoryGraphResponse)
async def get_task_memory_graph(
    limit: int = Query(120, ge=20, le=300, description="Max memories to include as graph nodes"),
    min_link_score: float = Query(1.0, ge=0.2, le=10.0, description="Minimum edge score"),
    max_edges: int = Query(800, ge=50, le=3000, description="Maximum graph edges"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AgentJobMemoryGraphResponse:
    """Return reusable cross-job task memory graph (lessons, failed paths, successful strategies)."""
    from app.services.agent_job_memory_service import agent_job_memory_service

    graph = await agent_job_memory_service.get_task_memory_graph(
        user_id=str(current_user.id),
        db=db,
        limit=limit,
        min_link_score=min_link_score,
        max_edges=max_edges,
    )
    return _build_memory_graph_response(graph=graph)


@router.get("/{job_id}/memories/graph", response_model=AgentJobMemoryGraphResponse)
async def get_job_memory_graph(
    job_id: UUID,
    neighbor_depth: int = Query(1, ge=1, le=2, description="Neighborhood depth around this job's memory nodes"),
    limit: int = Query(180, ge=20, le=300, description="Max nodes to scan"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AgentJobMemoryGraphResponse:
    """Return task-memory subgraph centered on memories created by a specific job."""
    from app.services.agent_job_memory_service import agent_job_memory_service

    job_res = await db.execute(
        select(AgentJob).where(
            and_(
                AgentJob.id == job_id,
                AgentJob.user_id == current_user.id,
            )
        )
    )
    job = job_res.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Agent job not found")

    graph = await agent_job_memory_service.get_task_memory_graph(
        user_id=str(current_user.id),
        db=db,
        limit=limit,
        min_link_score=1.0,
        max_edges=1200,
    )
    nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []
    job_node_ids = {str(n.get("id")) for n in nodes if str(n.get("job_id") or "") == str(job_id)}
    if not job_node_ids:
        return _build_memory_graph_response(
            graph={"nodes": [], "edges": [], "stats": {"memory_count": 0, "edge_count": 0}},
            job_id=job_id,
        )

    selected = set(job_node_ids)
    hops = max(1, min(int(neighbor_depth or 1), 2))
    for _ in range(hops):
        expanded = set(selected)
        for e in edges:
            src = str(e.get("source") or "")
            dst = str(e.get("target") or "")
            if src in selected or dst in selected:
                expanded.add(src)
                expanded.add(dst)
        selected = expanded

    sub_nodes = [n for n in nodes if str(n.get("id")) in selected]
    sub_edges = [e for e in edges if str(e.get("source")) in selected and str(e.get("target")) in selected]
    return _build_memory_graph_response(
        graph={
            "nodes": sub_nodes,
            "edges": sub_edges,
            "stats": {
                "memory_count": len(sub_nodes),
                "edge_count": len(sub_edges),
                "job_memory_count": len(job_node_ids),
                "neighbor_depth": hops,
            },
        },
        job_id=job_id,
    )


@router.get("/memory/stats", response_model=AgentJobMemoryStatsResponse)
async def get_memory_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AgentJobMemoryStatsResponse:
    """
    Get memory statistics for the current user.

    Includes breakdown by type, source, and most accessed/important memories.
    """
    from app.services.agent_job_memory_service import agent_job_memory_service

    stats = await agent_job_memory_service.get_memory_stats_for_user(
        user_id=str(current_user.id),
        db=db,
    )

    return _build_memory_stats_response(stats=stats)


@router.get("/memory/search", response_model=AgentJobMemorySearchResponse)
async def search_memories(
    query: str = Query(..., description="Search query"),
    memory_types: Optional[str] = Query(None, description="Comma-separated memory types to filter"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
) -> AgentJobMemorySearchResponse:
    """
    Search memories relevant to a query.

    Returns memories ranked by relevance to the query.
    """
    from app.models.memory import ConversationMemory
    from sqlalchemy import desc

    # Build query
    type_list = [t.strip() for t in memory_types.split(",")] if memory_types else None

    base_query = (
        select(ConversationMemory)
        .where(
            and_(
                ConversationMemory.user_id == current_user.id,
                ConversationMemory.is_active == True,
            )
        )
    )

    if type_list:
        base_query = base_query.where(ConversationMemory.memory_type.in_(type_list))

    # Simple text search (for now, could be enhanced with vector search)
    base_query = base_query.where(
        ConversationMemory.content.ilike(f"%{query}%")
    )

    base_query = base_query.order_by(desc(ConversationMemory.importance_score)).limit(limit)

    result = await db.execute(base_query)
    memories = list(result.scalars().all())

    return _build_memory_search_response(
        query=query,
        memories=memories,
    )
