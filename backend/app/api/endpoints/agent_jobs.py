"""
API endpoints for autonomous agent jobs.

Provides CRUD operations and control actions for autonomous agent jobs.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob, AgentJobCheckpoint, AgentJobStatus
from app.models.autonomy_decision_event import AutonomyDecisionEvent
from app.models.coding_backlog import CodingBacklogItem
from app.models.domain_research_profile import DomainResearchProfile
from app.models.experiment import ExperimentRun
from app.models.research_inbox import ResearchInboxItem
from app.models.research_portfolio import ResearchPortfolio
from app.models.user import User
from app.modules.autonomy.api import ai_hub_feedback as ai_hub_feedback_routes
from app.modules.autonomy.api import chain_definitions as chain_definition_routes
from app.modules.autonomy.api import job_feedback as job_feedback_routes
from app.modules.autonomy.api import job_memories as job_memory_routes
from app.modules.autonomy.api import relaunch_lineage as relaunch_lineage_routes
from app.modules.autonomy.api.chain_execution import build_chain_execution_api
from app.modules.autonomy.api.checkpoint_follow_up_actions import (
    build_checkpoint_follow_up_action_api,
)
from app.modules.autonomy.api.checkpoint_job_actions import (
    build_checkpoint_job_action_api,
)
from app.modules.autonomy.api.checkpoint_queue import build_checkpoint_queue_api
from app.modules.autonomy.api.decision_trace_actions import (
    build_decision_trace_action_api,
)
from app.modules.autonomy.api.decision_trace_queries import (
    build_decision_trace_query_api,
)
from app.modules.autonomy.api.decision_trace_reporting import (
    build_decision_trace_reporting_api,
)
from app.modules.autonomy.api.decision_trace_views import build_decision_trace_view_api
from app.modules.autonomy.api.domain_research_promotion import (
    build_domain_research_promotion_api,
)
from app.modules.autonomy.api.follow_up_queue_actions import (
    build_follow_up_queue_action_api,
)
from app.modules.autonomy.api.job_actions import build_job_action_api
from app.modules.autonomy.api.job_checkpoints import build_job_checkpoint_api
from app.modules.autonomy.api.job_crud import (
    build_job_creation_api,
    build_job_record_api,
)
from app.modules.autonomy.api.job_exports import build_job_export_api
from app.modules.autonomy.api.job_logs import build_job_log_api
from app.modules.autonomy.api.job_progress import build_job_progress_api
from app.modules.autonomy.api.job_queries import build_job_query_api
from app.modules.autonomy.api.job_step_events import build_job_step_event_api
from app.modules.autonomy.api.job_templates import build_job_template_api
from app.modules.autonomy.api.quick_starts import (
    QuickStartBuilders,
    build_quick_start_api,
)
from app.modules.autonomy.api.swarm_analytics import build_swarm_analytics_api
from app.modules.autonomy.api.swarm_outcomes import build_swarm_outcomes_api
from app.modules.autonomy.application import (
    checkpoint_queue_composer,
    checkpoint_queue_priority,
    coding_swarm_relaunch,
    decision_trace_events,
    decision_trace_follow_up_targets,
    decision_trace_jobs,
    decision_trace_loader,
    decision_trace_monitors,
    decision_trace_opportunities,
    decision_trace_queue,
    decision_trace_validation,
    domain_research_promotion_seed,
    feedback_presenters,
    follow_up_inbox_relaunch,
    follow_up_learning_profiles,
    follow_up_policy,
    follow_up_queue_dispatcher,
    follow_up_queue_events,
    follow_up_queue_inbox,
    follow_up_recommendations,
    job_action_checkpoints,
    job_action_interventions,
    job_action_plan_state,
    job_operator_events,
    job_presenters,
    memory_presenters,
    operator_queue_context,
    portfolio_queue_state,
    quick_start_builders,
    quick_start_relaunch,
    relaunch_lineage,
    repair_verification,
    swarm_outcome_cases,
    swarm_summaries,
)
from app.modules.autonomy.application.job_action_contracts import (
    JobActionDependencies,
    JobActionError,
)
from app.modules.autonomy.application.job_action_state_machine import (
    perform_job_action as perform_job_action_state_machine,
)
from app.modules.autonomy.application.relaunch_dispatcher import (
    QuickStartRelaunchDispatcher,
    RelaunchRoute,
)
from app.modules.autonomy.application.template_recommendations import (
    score_template_recommendation,
)
from app.schemas.agent_job import (  # Chain schemas
    AgentCheckpointQueueActionResponse,
    AgentCheckpointQueueFollowUpActionResponse,
    AgentCheckpointQueueItemResponse,
    AgentDecisionTraceEventResponse,
    AgentJobActionRequest,
    AgentJobCreate,
    AgentJobFromChainCreate,
    AgentJobQuickStartBugTriageSwarmRequest,
    AgentJobQuickStartBuildBreakSwarmRequest,
    AgentJobQuickStartFrontendRegressionSwarmRequest,
    AgentJobResponse,
    AgentJobSwarmOutcomeCaseResponse,
)
from app.services import research_inbox_follow_up_service
from app.services.agent_chain_definition_service import agent_chain_definition_service
from app.services.agent_chain_launch_service import agent_chain_launch_service
from app.services.agent_coding_swarm_launch_service import (
    AgentCodingSwarmLaunchError,
    agent_coding_swarm_launch_service,
)
from app.services.agent_job_creation_service import agent_job_creation_service
from app.services.agent_job_queue_helpers import (
    extract_approval_checkpoint,
    extract_launch_mode,
    parse_optional_datetime,
    queue_age_minutes,
    queue_customer_for_job,
    queue_evidence_summary_for_job,
)
from app.services.agent_job_scheduler_state import (
    extract_scheduler_state,
    queue_reason_label,
)
from app.services.agent_scope_service import (
    normalize_scope_config as _normalize_scope_config,
)
from app.services.agent_scope_service import (
    normalize_scope_keys_deep as _normalize_scope_keys_deep,
)
from app.services.agent_swarm_collaboration_service import (
    build_collaboration_payload as _build_swarm_collaboration_payload,
)
from app.services.agent_swarm_collaboration_service import (
    extract_swarm_collaboration as _extract_swarm_collaboration,
)
from app.services.agent_swarm_collaboration_service import (
    is_profile_visible_to_user,
    normalize_profile_visibility,
)
from app.services.agent_swarm_collaboration_service import (
    normalize_uuid_str_list as _normalize_uuid_str_list,
)
from app.services.agent_swarm_collaboration_service import (
    store_swarm_collaboration as _store_swarm_collaboration,
)
from app.services.autonomy_event_service import record_autonomy_decision_event
from app.services.autonomy_service import (
    build_autonomy_summary,
    build_monitor_policy_compat_fields,
    current_domain_profile_policy_snapshot,
    resolve_domain_profile_automation_contract,
)
from app.services.collaboration_service import list_collaboration_user_ids
from app.services.research_monitor_profile_service import (
    research_monitor_profile_service,
)
from app.services.research_opportunity_service import (
    classify_portfolio_operator_review,
    collect_research_opportunity_linked_ids,
    list_normalized_research_opportunities,
)
from app.services.scientific_validation_service import (
    get_scientific_sandbox_profile,
    normalize_portfolio_automation_profile,
    resolve_portfolio_automation_policy,
)
from app.tasks.agent_job_tasks import execute_agent_job_task, generate_job_summary

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

FOLLOW_UP_RECOMMENDATION_DEEP_DIVE_CHAIN = (
    follow_up_recommendations.FOLLOW_UP_RECOMMENDATION_DEEP_DIVE_CHAIN
)
FOLLOW_UP_RECOMMENDATION_SINGLE_RESEARCH_JOB = (
    follow_up_recommendations.FOLLOW_UP_RECOMMENDATION_SINGLE_RESEARCH_JOB
)
FOLLOW_UP_RECOMMENDATION_REPO_PATCH_CHAIN = (
    follow_up_recommendations.FOLLOW_UP_RECOMMENDATION_REPO_PATCH_CHAIN
)


_queue_priority_fields = checkpoint_queue_priority.queue_priority_fields


_is_source_owned_by_user = agent_coding_swarm_launch_service.is_source_owned_by_user
_build_quick_start_claude_backend_config = (
    quick_start_builders.build_claude_backend_config
)
_build_domain_research_goal = quick_start_builders.build_domain_research_goal
_build_quick_start_domain_research_config = (
    quick_start_builders.build_domain_research_config
)


def _extract_domain_research_promotion(job: AgentJob) -> dict[str, Any]:
    cfg = job.config if isinstance(job.config, dict) else {}
    quick_start = (
        cfg.get("quick_start") if isinstance(cfg.get("quick_start"), dict) else {}
    )
    results = job.results if isinstance(job.results, dict) else {}

    promotion = cfg.get("promotion") if isinstance(cfg.get("promotion"), dict) else {}
    if not promotion and isinstance(quick_start.get("promotion"), dict):
        promotion = quick_start.get("promotion") or {}
    if not promotion and isinstance(results.get("promotion"), dict):
        promotion = results.get("promotion") or {}
    return dict(promotion) if isinstance(promotion, dict) else {}


def _build_domain_research_promotion_seed(job: AgentJob) -> dict[str, Any]:
    try:
        return domain_research_promotion_seed.build_domain_research_promotion_seed(
            job,
            deps=domain_research_promotion_seed.DomainResearchPromotionSeedDependencies(
                extract_launch_mode=_extract_launch_mode,
                resolve_domain_automation_contract=(
                    resolve_domain_profile_automation_contract
                ),
                normalize_portfolio_automation_profile=(
                    normalize_portfolio_automation_profile
                ),
                resolve_portfolio_automation_policy=(
                    resolve_portfolio_automation_policy
                ),
            ),
        )
    except domain_research_promotion_seed.DomainResearchPromotionSeedError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(error),
        ) from error


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
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unknown or disabled sandbox profile",
        )


_build_repo_bug_triage_goal = quick_start_builders.build_repo_bug_triage_goal


_is_coding_swarm_profile_visible_to_user = is_profile_visible_to_user
_normalize_coding_swarm_profile_visibility = normalize_profile_visibility


def _is_job_visible_to_user(job: AgentJob, user: User) -> bool:
    if user.is_admin() or str(job.user_id) == str(user.id):
        return True
    cfg = job.config if isinstance(job.config, dict) else {}
    if (
        not _infer_coding_swarm_preset_key(job)
        and str(_extract_launch_mode(cfg) or "").strip()
        != "bug_triage_swarm_repair_handoff"
    ):
        return False
    collaboration = _extract_swarm_collaboration(job)
    if str(collaboration.get("assigned_user_id") or "").strip() == str(user.id):
        return True
    return str(user.id) in _normalize_uuid_str_list(
        collaboration.get("shared_with_user_ids"), 200
    )


def _normalize_coding_backlog_visibility(value: object) -> str:
    return (
        "shared" if str(value or "private").strip().lower() == "shared" else "private"
    )


def _is_backlog_item_visible_to_user(item: CodingBacklogItem, user: User) -> bool:
    if user.is_admin() or str(item.user_id) == str(user.id):
        return True
    if str(getattr(item, "assigned_user_id", "") or "").strip() == str(user.id):
        return True
    if (
        _normalize_coding_backlog_visibility(getattr(item, "visibility", "private"))
        != "shared"
    ):
        return False
    return str(user.id) in _normalize_uuid_str_list(
        getattr(item, "shared_with_user_ids", None), 200
    )


def _extract_backlog_route_mode(item: Optional[CodingBacklogItem]) -> Optional[str]:
    if item is None:
        return None
    lineage = item.lineage if isinstance(item.lineage, dict) else {}
    mode = str(lineage.get("originating_swarm_route_mode") or "").strip().lower()
    if mode in {"auto", "manual"}:
        return mode
    return None


def _build_quick_start_coding_swarm_config(
    request: AgentJobQuickStartBugTriageSwarmRequest
    | AgentJobQuickStartBuildBreakSwarmRequest
    | AgentJobQuickStartFrontendRegressionSwarmRequest,
    *,
    source_name: str,
    source_type: str,
    preset_key: str,
) -> dict:
    try:
        return agent_coding_swarm_launch_service.build_config(
            request,
            source_name=source_name,
            source_type=source_type,
            preset_key=preset_key,
        )
    except AgentCodingSwarmLaunchError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=error.detail,
        ) from error


_build_quick_start_repo_bug_triage_config = (
    quick_start_builders.build_repo_bug_triage_config
)


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


_coerce_bool = quick_start_builders.coerce_bool
_normalize_swarm_roles = quick_start_builders.normalize_swarm_roles
_build_quick_start_role_workflow_config = (
    quick_start_builders.build_role_workflow_config
)


def _extract_source_id_from_config(config: Optional[dict]) -> str:
    normalized = (
        _normalize_scope_config(config if isinstance(config, dict) else None) or {}
    )
    return str(normalized.get("source_id") or "").strip()


_extract_relaunch_parent_job_id = relaunch_lineage.extract_parent_job_id
_build_relaunch_children_counts = relaunch_lineage.build_children_counts


def _json_relaunch_parent_expr(model=AgentJob):
    try:
        return model.config["relaunch_from_job_id"].as_string()
    except Exception:
        return model.config["relaunch_from_job_id"].astext


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


_build_relaunch_lineage = relaunch_lineage.build_lineage


_build_quick_start_relaunch_request = (
    quick_start_relaunch.build_claude_backend_relaunch_request
)


_build_quick_start_domain_research_relaunch_request = (
    quick_start_relaunch.build_domain_research_relaunch_request
)


_build_quick_start_repo_bug_triage_relaunch_request = (
    coding_swarm_relaunch.build_repo_bug_triage_relaunch_request
)


_build_quick_start_bug_triage_swarm_relaunch_request = (
    coding_swarm_relaunch.build_bug_triage_swarm_relaunch_request
)


_build_quick_start_build_break_swarm_relaunch_request = (
    coding_swarm_relaunch.build_build_break_swarm_relaunch_request
)


_build_quick_start_frontend_regression_swarm_relaunch_request = (
    coding_swarm_relaunch.build_frontend_regression_swarm_relaunch_request
)


_build_quick_start_coding_swarm_relaunch_request = (
    coding_swarm_relaunch.build_coding_swarm_relaunch_request
)


_extract_repo_bug_triage_coding_recovery = (
    coding_swarm_relaunch.extract_repo_bug_triage_coding_recovery
)


_build_quick_start_role_workflow_relaunch_request = (
    quick_start_relaunch.build_role_workflow_relaunch_request
)


def _is_none_launch_mode(mode: str) -> bool:
    value = str(mode or "").strip().lower()
    return value in {"", "__none__", "none", "manual"}


def _matches_launch_mode_filter(
    config: Optional[dict], launch_mode_filter: str
) -> bool:
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


def _build_launch_mode_stats(
    configs: list[Optional[dict]],
) -> tuple[dict[str, int], int]:
    counts = _build_launch_mode_counts(configs)
    none_count = 0
    for cfg in configs:
        mode = _extract_launch_mode(cfg if isinstance(cfg, dict) else None)
        if _is_none_launch_mode(mode):
            none_count += 1
    return counts, none_count


_append_launch_log_if_present = agent_job_creation_service.append_launch_log_if_present
_find_unsafe_commands = agent_job_creation_service.find_unsafe_commands
_score_template_recommendation = score_template_recommendation


_extract_swarm_summary = swarm_summaries.extract_swarm_summary


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
    quick_start = (
        cfg.get("quick_start") if isinstance(cfg.get("quick_start"), dict) else {}
    )
    preset_key = (
        str(quick_start.get("preset_key") or cfg.get("coding_swarm_preset_key") or "")
        .strip()
        .lower()
    )
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


def _datetime_sort_key(value: Optional[datetime]) -> float:
    if not isinstance(value, datetime):
        return 0.0
    try:
        return value.timestamp()
    except Exception:
        return 0.0


def _derive_repair_verification_status(
    job: AgentJob,
) -> tuple[Optional[str], Optional[str]]:
    return repair_verification.derive_repair_verification_status(job)


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
        if verification_status == "failed" or repair_status in {
            AgentJobStatus.FAILED.value,
            AgentJobStatus.CANCELLED.value,
        }:
            return "repair_failed", "Repair chain failed or verification failed."
        return "stalled_after_handoff", "Repair handoff exists without a verified fix."
    if review_state in {
        "needs_review",
        "insufficient_swarm_consensus",
        "consensus_failed",
        "tie_break_running",
        "manual_promotion",
    }:
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
    return swarm_outcome_cases.derive_swarm_outcome_case(
        swarm_job,
        repair_jobs_by_id=repair_jobs_by_id,
        backlog_by_swarm_job_id=backlog_by_swarm_job_id,
        current_user_id=current_user_id,
        user_lookup=user_lookup,
        deps=swarm_outcome_cases.SwarmOutcomeCaseDependencies(
            extract_swarm_summary=_extract_swarm_summary,
            extract_collaboration=_extract_swarm_collaboration,
            infer_preset_key=_infer_coding_swarm_preset_key,
            extract_launch_mode=_extract_launch_mode,
            safe_float=_safe_float,
            derive_verification_status=_derive_repair_verification_status,
            derive_terminal_outcome=_derive_swarm_terminal_outcome,
            extract_backlog_route_mode=_extract_backlog_route_mode,
        ),
    )


def _extract_goal_contract_summary(job: AgentJob) -> Optional[dict]:
    """Build compact goal-contract status for quick UI rendering."""
    results = job.results if isinstance(job.results, dict) else {}
    contract = (
        results.get("goal_contract")
        if isinstance(results.get("goal_contract"), dict)
        else {}
    )
    if not contract:
        return None

    enabled = bool(contract.get("enabled", False))
    if not enabled and not contract:
        return None
    missing = (
        contract.get("missing") if isinstance(contract.get("missing"), list) else []
    )
    contract_cfg = (
        contract.get("contract") if isinstance(contract.get("contract"), dict) else {}
    )
    metrics = (
        contract.get("metrics") if isinstance(contract.get("metrics"), dict) else {}
    )
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
    automation_profile = normalize_portfolio_automation_profile(
        getattr(portfolio, "automation_profile", None), default="balanced"
    )
    effective_policy = resolve_portfolio_automation_policy(
        automation_profile, portfolio.automation_policy
    )
    opportunities = list_normalized_research_opportunities(portfolio.opportunities)
    summary = build_autonomy_summary(
        raw_summary=portfolio.latest_summary
        if isinstance(portfolio.latest_summary, dict)
        else {},
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
        (profile.latest_summary or {}).get("opportunities")
        if isinstance((profile.latest_summary or {}).get("opportunities"), list)
        else (profile.latest_summary or {}).get("idea_candidates")
    )
    summary = build_autonomy_summary(
        raw_summary=profile.latest_summary
        if isinstance(profile.latest_summary, dict)
        else {},
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


_clean_queue_text_list = operator_queue_context.clean_text_list
_build_operator_queue_context = operator_queue_context.build_operator_queue_context


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
        (
            candidate
            for candidate in rows
            if str(candidate.get("opportunity_id") or "").strip()
            == str(opportunity_id or "").strip()
        ),
        None,
    )
    if isinstance(row, dict):
        label_source = str(
            row.get("canonical_key") or row.get("title") or opportunity_id
        ).strip()
        if label_source:
            normalized = label_source.replace("-", " ").replace("_", " ").strip()
            return (
                normalized[:1].upper() + normalized[1:].lower()
                if normalized
                else _queue_reason_label(opportunity_id)
            )
    return _queue_reason_label(opportunity_id)


async def _resolve_portfolio_parent_job_for_queue(
    *,
    db: AsyncSession,
    portfolio: ResearchPortfolio,
) -> AgentJob:
    parent_job_id = portfolio.latest_run_job_id or portfolio.active_job_id
    if not parent_job_id:
        raise HTTPException(
            status_code=400,
            detail="Portfolio must run at least once before launching downstream actions",
        )
    parent_job = await db.get(AgentJob, parent_job_id)
    if parent_job is None or parent_job.user_id != portfolio.user_id:
        raise HTTPException(
            status_code=400, detail="Latest portfolio run is unavailable"
        )
    return parent_job


async def _resolve_profile_parent_job_for_queue(
    *,
    db: AsyncSession,
    profile: DomainResearchProfile,
) -> AgentJob:
    parent_job_id = profile.latest_run_job_id or profile.active_job_id
    if not parent_job_id:
        raise HTTPException(
            status_code=400,
            detail="Profile must run at least once before launching downstream actions",
        )
    parent_job = await db.get(AgentJob, parent_job_id)
    if parent_job is None or parent_job.user_id != profile.user_id:
        raise HTTPException(status_code=400, detail="Latest profile run is unavailable")
    return parent_job


_sync_portfolio_queue_state = portfolio_queue_state.sync_portfolio_queue_state


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
        dict.fromkeys(
            [
                *([str(v) for v in (profile.latest_note_ids or []) if str(v).strip()]),
                *linked_ids["note_ids"],
            ]
        )
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
    automation = research_monitor_profile_service.resolve_monitor_automation_config(
        config
    )
    compat = automation["follow_up_autonomy"]
    return {
        "mode": _normalize_follow_up_autonomy_mode(compat.get("mode")),
        "allowed_recommendations": [
            str(value).strip()
            for value in (compat.get("allowed_recommendations") or [])
            if str(value).strip()
        ]
        or [
            FOLLOW_UP_RECOMMENDATION_DEEP_DIVE_CHAIN,
            FOLLOW_UP_RECOMMENDATION_SINGLE_RESEARCH_JOB,
        ],
        "automation_profile": automation["automation_profile"],
        "automation_policy": automation["automation_policy"],
        "effective_policy": automation["effective_policy"],
    }


def _decision_trace_reason_label(reason_code: Optional[str]) -> Optional[str]:
    text = str(reason_code or "").strip().lower()
    if not text:
        return None
    return _queue_reason_label(text)


def _decision_trace_parse_time(
    raw: Any, *, fallback: Optional[datetime] = None
) -> Optional[datetime]:
    if isinstance(raw, datetime):
        return raw
    text = str(raw or "").strip()
    if not text:
        return fallback
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return fallback


_decision_trace_event_id = decision_trace_events.decision_trace_event_id


def _build_decision_trace_event(**kwargs) -> AgentDecisionTraceEventResponse:
    return decision_trace_events.build_decision_trace_event(
        **kwargs,
        deps=decision_trace_events.DecisionTraceEventDependencies(
            build_operator_context=_build_operator_queue_context,
        ),
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
    try:
        return decision_trace_follow_up_targets.resolve_follow_up_target(event)
    except decision_trace_follow_up_targets.DecisionTraceFollowUpTargetError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=error.detail,
        ) from error


def _trace_event_follow_up_relaunch_job_id(event: AutonomyDecisionEvent) -> str:
    event_kind = str(event.event_type or event.decision_type or "").strip().lower()
    after_state = event.after_state if isinstance(event.after_state, dict) else {}
    outcome_status = (
        str(
            after_state.get("follow_up_outcome_status")
            or event.status
            or event_kind
            or ""
        )
        .strip()
        .lower()
    )
    if event_kind not in {
        "follow_up_failed",
        "follow_up_cancelled",
    } and outcome_status not in {"failed", "cancelled"}:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Decision trace event is not a relaunchable follow-up outcome",
        )
    job_id = str(
        after_state.get("follow_up_last_job_id")
        or (
            (
                event.event_metadata if isinstance(event.event_metadata, dict) else {}
            ).get("follow_up_last_job_id")
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
        (
            await db.execute(
                select(User).where(
                    User.id == assigned_to_user_id,
                    User.is_active.is_(True),
                )
            )
        )
        .scalars()
        .first()
    )
    return user.id if user is not None else None


async def _build_collaboration_user_lookup(
    db: AsyncSession,
    *,
    current_user: User,
) -> dict[str, User]:
    visible_user_ids = await list_collaboration_user_ids(db, current_user=current_user)
    if current_user.id not in visible_user_ids:
        visible_user_ids.add(current_user.id)
    rows = list(
        (await db.execute(select(User).where(User.id.in_(visible_user_ids))))
        .scalars()
        .all()
    )
    return {str(row.id): row for row in rows}


def _build_decision_trace_from_queue_items(
    items: list[AgentCheckpointQueueItemResponse],
) -> list[AgentDecisionTraceEventResponse]:
    return decision_trace_queue.build_queue_decision_trace(
        items,
        deps=decision_trace_queue.QueueDecisionTraceDependencies(
            build_event=_build_decision_trace_event,
            build_operator_context=_build_operator_queue_context,
        ),
    )


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
        str(
            assignee.full_name or assignee.username or assignee.email or assignee.id
        ).strip()
        if assignee is not None
        else None
    )
    payload["is_owned_by_current_user"] = owner_user_id == str(current_user_id)
    payload["is_assigned_to_current_user"] = assignee_user_id == str(current_user_id)
    return payload


def _build_decision_trace_from_job(
    job: AgentJob,
) -> list[AgentDecisionTraceEventResponse]:
    return decision_trace_jobs.build_job_decision_trace(
        job,
        deps=decision_trace_jobs.JobDecisionTraceDependencies(
            parse_time=_decision_trace_parse_time,
            build_event=_build_decision_trace_event,
            queue_customer_for_job=_queue_customer_for_job,
            reason_label=_decision_trace_reason_label,
        ),
    )


_build_decision_trace_from_opportunities = (
    decision_trace_opportunities.bind_opportunity_decision_trace(
        deps=decision_trace_opportunities.OpportunityDecisionTraceDependencies(
            parse_time=_decision_trace_parse_time,
            reason_label=_decision_trace_reason_label,
            build_event=_build_decision_trace_event,
            build_operator_context=_build_operator_queue_context,
        )
    )
)


def _build_decision_trace_from_monitor_snapshot(
    snapshot: dict[str, Any]
) -> list[AgentDecisionTraceEventResponse]:
    return decision_trace_monitors.build_monitor_decision_trace(
        snapshot,
        deps=decision_trace_monitors.MonitorDecisionTraceDependencies(
            parse_time=_decision_trace_parse_time,
            reason_label=_decision_trace_reason_label,
            build_event=_build_decision_trace_event,
        ),
    )


def _build_decision_trace_from_validation_runs(
    runs: list[ExperimentRun],
) -> list[AgentDecisionTraceEventResponse]:
    return decision_trace_validation.build_validation_decision_trace(
        runs,
        deps=decision_trace_validation.ValidationDecisionTraceDependencies(
            parse_time=_decision_trace_parse_time,
            reason_label=_decision_trace_reason_label,
            build_event=_build_decision_trace_event,
            build_operator_context=_build_operator_queue_context,
        ),
    )


async def _record_job_operator_event(**kwargs) -> None:
    await job_operator_events.record_job_operator_event(
        **kwargs,
        deps=job_operator_events.JobOperatorEventDependencies(
            record_event=record_autonomy_decision_event,
            queue_customer_for_job=_queue_customer_for_job,
            reason_label=_decision_trace_reason_label,
        ),
    )


_tokenize_learning_text = follow_up_recommendations.tokenize_learning_text


_load_follow_up_learning_profile = (
    follow_up_learning_profiles.load_follow_up_learning_profile
)


def _score_follow_up_action_for_item(
    item: ResearchInboxItem,
    action_row: AgentCheckpointQueueActionResponse,
    *,
    learning_profile: Optional[dict[str, Any]] = None,
) -> tuple[int, list[str]]:
    return follow_up_recommendations.score_follow_up_action(
        item,
        action_row,
        learning_profile=learning_profile,
    )


def _customer_profile_key(customer: Optional[str]) -> str:
    return str(customer or "").strip().lower()


async def _launch_follow_up_action(
    action_row: AgentCheckpointQueueActionResponse,
    *,
    db: AsyncSession,
    current_user: User,
) -> AgentJobResponse:
    if action_row.chain_create_payload:
        request = AgentJobFromChainCreate.model_validate(
            action_row.chain_create_payload
        )
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
    return await follow_up_policy.apply_follow_up_policy_on_accept(
        item=item,
        current_user=current_user,
        db=db,
        deps=follow_up_policy.FollowUpPolicyDependencies(
            get_policy_from_job=_get_follow_up_policy_from_job,
            budget_service=research_monitor_profile_service,
            load_learning_profile=_load_follow_up_learning_profile,
            build_follow_up_actions=_build_follow_up_actions_for_inbox_item,
            launch_follow_up_action=_launch_follow_up_action,
        ),
    )


def _build_follow_up_queue_dispatcher_dependencies():
    return follow_up_queue_dispatcher.FollowUpQueueDispatcherDependencies(
        load_learning_profile=_load_follow_up_learning_profile,
        build_follow_up_actions=_build_follow_up_actions_for_inbox_item,
        launch_follow_up_action=_launch_follow_up_action,
        build_portfolio_summary=_portfolio_summary_payload,
        build_profile_summary=_profile_summary_payload,
        classify_operator_review=classify_portfolio_operator_review,
        sync_portfolio_queue_state=_sync_portfolio_queue_state,
        sync_profile_queue_state=_sync_profile_queue_state,
        resolve_portfolio_parent_job=_resolve_portfolio_parent_job_for_queue,
        resolve_profile_parent_job=_resolve_profile_parent_job_for_queue,
        execute_agent_job_task=execute_agent_job_task,
    )


follow_up_queue_action_api = build_follow_up_queue_action_api(
    dependencies_factory=_build_follow_up_queue_dispatcher_dependencies,
)
_perform_follow_up_queue_action = (
    follow_up_queue_action_api.perform_follow_up_queue_action
)


async def _relaunch_follow_up_inbox_item(
    *,
    item: ResearchInboxItem,
    operator_note: Optional[str],
    db: AsyncSession,
    current_user: User,
) -> AgentCheckpointQueueFollowUpActionResponse:
    try:
        return await follow_up_inbox_relaunch.relaunch_inbox_follow_up(
            item=item,
            operator_note=operator_note,
            db=db,
            current_user=current_user,
            deps=follow_up_inbox_relaunch.InboxFollowUpRelaunchDependencies(
                load_learning_profile=_load_follow_up_learning_profile,
                build_follow_up_actions=_build_follow_up_actions_for_inbox_item,
                launch_follow_up_action=_launch_follow_up_action,
                project_relaunch_to_originating_opportunity=(
                    research_inbox_follow_up_service.project_follow_up_relaunch_to_originating_opportunity
                ),
            ),
        )
    except follow_up_queue_inbox.FollowUpQueueActionError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=error.detail,
        ) from error


async def _record_follow_up_queue_decision_event(**kwargs) -> None:
    current_user = kwargs.pop("current_user")
    kwargs.pop("follow_up_operator_decision", None)
    await follow_up_queue_events.record_follow_up_queue_decision(
        **kwargs,
        user_id=current_user.id,
        deps=follow_up_queue_events.FollowUpQueueEventDependencies(
            record_event=record_autonomy_decision_event,
        ),
    )


def _build_follow_up_actions_for_inbox_item(
    item: ResearchInboxItem,
    *,
    learning_profile: Optional[dict[str, Any]] = None,
) -> list[AgentCheckpointQueueActionResponse]:
    return follow_up_recommendations.build_follow_up_actions(
        item,
        learning_profile=learning_profile,
    )


_approval_payload_from_results = job_action_checkpoints.approval_payload_from_results


_normalize_checkpoint_action_patch = (
    job_action_checkpoints.normalize_checkpoint_action_patch
)


_apply_checkpoint_action_patch = job_action_checkpoints.apply_checkpoint_action_patch


_append_approval_event = job_action_checkpoints.append_approval_event


_set_current_plan_step_status = job_action_plan_state.set_current_plan_step_status


_append_step_event = job_action_checkpoints.append_step_event


_append_operator_intervention = job_action_interventions.append_operator_intervention


_sync_execution_strategy_state = job_action_checkpoints.sync_execution_strategy_state


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
    digest = (
        results.get("executive_digest")
        if isinstance(results.get("executive_digest"), dict)
        else None
    )
    return digest


_sanitize_tool_names = feedback_presenters.sanitize_tool_names
_memory_to_feedback_response = feedback_presenters.memory_to_feedback_response


def _job_to_response(
    job: AgentJob,
    *,
    relaunch_children_count: int = 0,
    current_user_id: Optional[str] = None,
    user_lookup: Optional[dict[str, User]] = None,
) -> AgentJobResponse:
    return job_presenters.present_job(
        job,
        relaunch_children_count=relaunch_children_count,
        current_user_id=current_user_id,
        user_lookup=user_lookup,
        deps=job_presenters.JobPresenterDependencies(
            extract_launch_mode=_extract_launch_mode,
            extract_promotion=_extract_domain_research_promotion,
            extract_swarm_summary=_extract_swarm_summary,
            extract_goal_contract_summary=_extract_goal_contract_summary,
            extract_approval_checkpoint=_extract_approval_checkpoint,
            extract_executive_digest=_extract_executive_digest,
        ),
    )


_build_checkpoint_queue_items = (
    checkpoint_queue_composer.bind_checkpoint_queue_composer(
        dependencies_factory=lambda: (
            checkpoint_queue_composer.CheckpointQueueCompositionDependencies(
                extract_approval_checkpoint=_extract_approval_checkpoint,
                extract_scheduler_state=_extract_scheduler_state,
                queue_customer_for_job=_queue_customer_for_job,
                present_job=_job_to_response,
                queue_priority_fields=_queue_priority_fields,
                queue_evidence_summary_for_job=_queue_evidence_summary_for_job,
                queue_reason_label=_queue_reason_label,
                parse_optional_datetime=_parse_optional_datetime,
                extract_launch_mode=_extract_launch_mode,
                build_policy_compat_fields=build_monitor_policy_compat_fields,
                safe_autonomy_recommendations=tuple(
                    research_monitor_profile_service.SAFE_AUTONOMY_RECOMMENDATIONS
                ),
                build_follow_up_actions=_build_follow_up_actions_for_inbox_item,
                customer_profile_key=_customer_profile_key,
                build_portfolio_summary=_portfolio_summary_payload,
                build_profile_summary=_profile_summary_payload,
                classify_operator_review=classify_portfolio_operator_review,
                build_operator_context=_build_operator_queue_context,
                clean_text_list=_clean_queue_text_list,
            )
        )
    )
)


_chain_definition_to_response = agent_chain_definition_service.to_response

router.include_router(ai_hub_feedback_routes.router)
router.include_router(chain_definition_routes.router)
router.include_router(job_feedback_routes.router)
router.include_router(job_memory_routes.router)
router.include_router(relaunch_lineage_routes.router)
job_creation_api = build_job_creation_api(
    job_serializer=_job_to_response,
    execute_job_task=execute_agent_job_task,
    router=router,
)
create_agent_job = job_creation_api.create_agent_job
create_job_from_template = job_creation_api.create_job_from_template
get_job_memories = job_memory_routes.get_job_memories
extract_job_memories = job_memory_routes.extract_job_memories
create_job_memory = job_memory_routes.create_job_memory
delete_job_memories = job_memory_routes.delete_job_memories
get_task_memory_graph = job_memory_routes.get_task_memory_graph
get_job_memory_graph = job_memory_routes.get_job_memory_graph
get_memory_stats = job_memory_routes.get_memory_stats
search_memories = job_memory_routes.search_memories
create_agent_job_feedback = job_feedback_routes.create_agent_job_feedback
list_agent_job_feedback = job_feedback_routes.list_agent_job_feedback
list_learning_feedback = job_feedback_routes.list_learning_feedback
list_ai_hub_recommendation_feedback = (
    ai_hub_feedback_routes.list_ai_hub_recommendation_feedback
)
create_ai_hub_recommendation_feedback = (
    ai_hub_feedback_routes.create_ai_hub_recommendation_feedback
)
get_agent_job_relaunch_lineage = relaunch_lineage_routes.get_agent_job_relaunch_lineage
chain_execution_api = build_chain_execution_api(
    job_serializer=_job_to_response,
    execute_job_task=execute_agent_job_task,
)
router.include_router(chain_execution_api.router)

# Temporary compatibility exports for direct callers while the aggregate
# endpoint module is decomposed.
list_chain_definitions = chain_definition_routes.list_chain_definitions
create_chain_definition = chain_definition_routes.create_chain_definition
get_chain_definition = chain_definition_routes.get_chain_definition
update_chain_definition = chain_definition_routes.update_chain_definition
delete_chain_definition = chain_definition_routes.delete_chain_definition
create_job_from_chain = chain_execution_api.create_job_from_chain
get_chain_status = chain_execution_api.get_chain_status
save_job_as_chain_definition = chain_execution_api.save_job_as_chain_definition


quick_start_api = build_quick_start_api(
    builders=QuickStartBuilders(
        claude_backend_config=_build_quick_start_claude_backend_config,
        domain_research_config=_build_quick_start_domain_research_config,
        domain_research_goal=_build_domain_research_goal,
        repo_bug_triage_config=_build_quick_start_repo_bug_triage_config,
        repo_bug_triage_goal=_build_repo_bug_triage_goal,
        role_workflow_config=_build_quick_start_role_workflow_config,
    ),
    create_job_from_template=create_job_from_template,
    job_serializer=_job_to_response,
    execute_job_task=execute_agent_job_task,
)
router.include_router(quick_start_api.router)

# Temporary compatibility exports while quick-start builders and callers move
# into the autonomy module.
quick_start_claude_backend_job = quick_start_api.quick_start_claude_backend_job
quick_start_domain_research_job = quick_start_api.quick_start_domain_research_job
quick_start_repo_bug_triage_job = quick_start_api.quick_start_repo_bug_triage_job
quick_start_bug_triage_swarm_job = quick_start_api.quick_start_bug_triage_swarm_job
quick_start_build_break_swarm_job = quick_start_api.quick_start_build_break_swarm_job
quick_start_frontend_regression_swarm_job = (
    quick_start_api.quick_start_frontend_regression_swarm_job
)
_create_quick_start_coding_swarm_job = (
    quick_start_api.create_quick_start_coding_swarm_job
)
quick_start_role_workflow_job = quick_start_api.quick_start_role_workflow_job

_quick_start_relaunch_dispatcher = QuickStartRelaunchDispatcher(
    routes={
        "quick_start_claude_backend": RelaunchRoute(
            builder=_build_quick_start_relaunch_request,
            launcher=quick_start_claude_backend_job,
        ),
        "quick_start_domain_research": RelaunchRoute(
            builder=_build_quick_start_domain_research_relaunch_request,
            launcher=quick_start_domain_research_job,
        ),
        "quick_start_repo_bug_triage": RelaunchRoute(
            builder=_build_quick_start_repo_bug_triage_relaunch_request,
            launcher=quick_start_repo_bug_triage_job,
            builder_kwargs={"retry_strategy": "clean_relaunch"},
        ),
        "quick_start_bug_triage_swarm": RelaunchRoute(
            builder=_build_quick_start_bug_triage_swarm_relaunch_request,
            launcher=quick_start_bug_triage_swarm_job,
        ),
        "quick_start_build_break_swarm": RelaunchRoute(
            builder=_build_quick_start_build_break_swarm_relaunch_request,
            launcher=quick_start_build_break_swarm_job,
        ),
        "quick_start_frontend_regression_swarm": RelaunchRoute(
            builder=_build_quick_start_frontend_regression_swarm_relaunch_request,
            launcher=quick_start_frontend_regression_swarm_job,
        ),
        "quick_start_role_workflow": RelaunchRoute(
            builder=_build_quick_start_role_workflow_relaunch_request,
            launcher=quick_start_role_workflow_job,
        ),
    },
    refined_repo_route=RelaunchRoute(
        builder=_build_quick_start_repo_bug_triage_relaunch_request,
        launcher=quick_start_repo_bug_triage_job,
        builder_kwargs={"retry_strategy": "refined_retry"},
    ),
    recovery_extractor=_extract_repo_bug_triage_coding_recovery,
)


job_query_api = build_job_query_api(
    router=router,
    job_serializer=_job_to_response,
    is_job_visible=_is_job_visible_to_user,
    extract_swarm_summary=_extract_swarm_summary,
    load_relaunch_children_counts=_build_relaunch_children_counts_for_user,
    load_collaboration_user_lookup=_build_collaboration_user_lookup,
    build_launch_mode_stats=_build_launch_mode_stats,
)
list_agent_jobs = job_query_api.list_agent_jobs
get_job_stats = job_query_api.get_job_stats


swarm_analytics_api = build_swarm_analytics_api(
    router=router,
    presets=_CODING_SWARM_ANALYTICS_PRESETS,
    is_job_visible=_is_job_visible_to_user,
    is_backlog_visible=_is_backlog_item_visible_to_user,
    extract_launch_mode=_extract_launch_mode,
    infer_preset_key=_infer_coding_swarm_preset_key,
    extract_swarm_summary=_extract_swarm_summary,
    confidence_bucket=_swarm_confidence_bucket,
    backlog_route_mode=_extract_backlog_route_mode,
)
get_swarm_analytics = swarm_analytics_api.get_swarm_analytics


swarm_outcomes_api = build_swarm_outcomes_api(
    router=router,
    presets=_CODING_SWARM_ANALYTICS_PRESETS,
    is_job_visible=_is_job_visible_to_user,
    is_backlog_visible=_is_backlog_item_visible_to_user,
    extract_launch_mode=_extract_launch_mode,
    infer_preset_key=_infer_coding_swarm_preset_key,
    derive_outcome_case=_derive_swarm_outcome_case,
    extract_swarm_summary=_extract_swarm_summary,
    datetime_sort_key=_datetime_sort_key,
    iso_or_none=_iso_or_none,
    load_collaboration_user_lookup=_build_collaboration_user_lookup,
)
get_swarm_outcomes = swarm_outcomes_api.get_swarm_outcomes


checkpoint_queue_api = build_checkpoint_queue_api(
    router=router,
    build_queue_items=_build_checkpoint_queue_items,
    customer_profile_key=_customer_profile_key,
    load_learning_profile=_load_follow_up_learning_profile,
    build_monitor_snapshot=(
        research_monitor_profile_service.build_effectiveness_snapshot
    ),
)
get_checkpoint_queue = checkpoint_queue_api.get_checkpoint_queue


async def _load_derived_decision_trace_events(
    *,
    db: AsyncSession,
    current_user: User,
) -> list[AgentDecisionTraceEventResponse]:
    return await decision_trace_loader.load_derived_decision_trace_events(
        db=db,
        current_user=current_user,
        deps=decision_trace_loader.DecisionTraceLoaderDependencies(
            customer_profile_key=_customer_profile_key,
            load_learning_profile=_load_follow_up_learning_profile,
            build_monitor_snapshot=(
                research_monitor_profile_service.build_effectiveness_snapshot
            ),
            build_queue_items=_build_checkpoint_queue_items,
            build_queue_events=_build_decision_trace_from_queue_items,
            build_job_events=_build_decision_trace_from_job,
            portfolio_summary=_portfolio_summary_payload,
            profile_summary=_profile_summary_payload,
            build_opportunity_events=_build_decision_trace_from_opportunities,
            build_monitor_events=_build_decision_trace_from_monitor_snapshot,
            build_validation_events=_build_decision_trace_from_validation_runs,
        ),
    )


decision_trace_query_api = build_decision_trace_query_api(
    router=router,
    list_visible_user_ids=_list_trace_visible_user_ids,
    decorate_trace_event_payload=_decorate_trace_event_payload,
    load_derived_events=_load_derived_decision_trace_events,
)
get_decision_trace = decision_trace_query_api.get_decision_trace


decision_trace_reporting_api = build_decision_trace_reporting_api(
    router=router,
    get_decision_trace=get_decision_trace,
    list_visible_user_ids=_list_trace_visible_user_ids,
)
export_decision_trace = decision_trace_reporting_api.export_decision_trace
get_decision_trace_analytics = decision_trace_reporting_api.get_decision_trace_analytics
_load_full_decision_trace_events = (
    decision_trace_reporting_api.load_full_decision_trace_events
)


async def _perform_follow_up_queue_action_for_trace(**kwargs):
    """Resolve the legacy action hook at call time for extension compatibility."""

    return await _perform_follow_up_queue_action(**kwargs)


async def _relaunch_follow_up_inbox_item_for_trace(**kwargs):
    """Resolve the legacy relaunch hook at call time for extension compatibility."""

    return await _relaunch_follow_up_inbox_item(**kwargs)


decision_trace_action_api = build_decision_trace_action_api(
    router=router,
    allowed_actions=TRACE_TRIAGE_ACTIONS,
    load_persisted_event=_load_persisted_trace_event_for_user,
    resolve_follow_up_target=_trace_event_follow_up_target,
    perform_follow_up_queue_action=_perform_follow_up_queue_action_for_trace,
    resolve_follow_up_job_id=_trace_event_follow_up_relaunch_job_id,
    relaunch_follow_up_inbox_item=_relaunch_follow_up_inbox_item_for_trace,
    validate_assignee=_validate_trace_assignee,
    list_visible_user_ids=_list_trace_visible_user_ids,
    decorate_trace_event_payload=_decorate_trace_event_payload,
)
act_on_decision_trace_event = decision_trace_action_api.act_on_decision_trace_event


decision_trace_view_api = build_decision_trace_view_api(
    router=router,
    normalize_filters=_normalize_trace_view_filters,
)
list_decision_trace_views = decision_trace_view_api.list_decision_trace_views
create_decision_trace_view = decision_trace_view_api.create_decision_trace_view
update_decision_trace_view = decision_trace_view_api.update_decision_trace_view
delete_decision_trace_view = decision_trace_view_api.delete_decision_trace_view


job_template_api = build_job_template_api(
    router=router,
    normalize_scope_keys=_normalize_scope_keys_deep,
)
list_job_templates = job_template_api.list_job_templates


# ============================================================================
# Chain Definition Endpoints
#
# IMPORTANT: Keep these static routes above `/{job_id}`. FastAPI matches routes
# in declaration order, and `/{job_id}` would otherwise capture "/chains".
# ============================================================================


job_record_api = build_job_record_api(
    job_serializer=_job_to_response,
    is_job_visible=_is_job_visible_to_user,
    normalize_scope_config=_normalize_scope_config,
    load_relaunch_children_counts=_build_relaunch_children_counts_for_user,
    load_collaboration_user_lookup=_build_collaboration_user_lookup,
)
router.include_router(job_record_api.router)
get_agent_job = job_record_api.get_agent_job
update_agent_job = job_record_api.update_agent_job
delete_agent_job = job_record_api.delete_agent_job


async def _present_promoted_domain_profile(profile, db):
    from app.api.endpoints.domain_research_profiles import _profile_response

    return await _profile_response(profile, db)


async def _present_promoted_research_portfolio(portfolio, db):
    from app.api.endpoints.research_portfolios import _portfolio_response

    return await _portfolio_response(portfolio, db)


domain_research_promotion_api = build_domain_research_promotion_api(
    router=router,
    is_job_visible=_is_job_visible_to_user,
    extract_launch_mode=_extract_launch_mode,
    extract_promotion=_extract_domain_research_promotion,
    build_promotion_seed=_build_domain_research_promotion_seed,
    validate_sandbox_profile=_validate_domain_research_sandbox_profile,
    build_domain_config=_build_quick_start_domain_research_config,
    build_domain_goal=_build_domain_research_goal,
    present_profile=_present_promoted_domain_profile,
    present_portfolio=_present_promoted_research_portfolio,
    present_job=_job_to_response,
    execute_job_task=execute_agent_job_task,
)
promote_domain_research_job = domain_research_promotion_api.promote_domain_research_job


def _build_job_action_dependencies() -> JobActionDependencies:
    """Resolve extension hooks at call time while keeping dependencies explicit."""

    return JobActionDependencies(
        is_job_visible=_is_job_visible_to_user,
        approval_payload_from_results=_approval_payload_from_results,
        load_latest_checkpoint=_load_latest_job_checkpoint,
        append_operator_intervention=_append_operator_intervention,
        append_step_event=_append_step_event,
        normalize_checkpoint_action_patch=_normalize_checkpoint_action_patch,
        apply_checkpoint_action_patch=_apply_checkpoint_action_patch,
        set_current_plan_step_status=_set_current_plan_step_status,
        append_approval_event=_append_approval_event,
        sync_execution_strategy_state=_sync_execution_strategy_state,
        quick_start_relaunch_dispatcher=_quick_start_relaunch_dispatcher,
        infer_coding_swarm_preset_key=_infer_coding_swarm_preset_key,
        extract_swarm_collaboration=_extract_swarm_collaboration,
        build_swarm_collaboration_payload=_build_swarm_collaboration_payload,
        store_swarm_collaboration=_store_swarm_collaboration,
        execute_agent_job_task=execute_agent_job_task,
        generate_job_summary=generate_job_summary,
    )


_JOB_ACTION_DEPENDENCIES = _build_job_action_dependencies()


async def _perform_job_action(
    job: AgentJob,
    request: AgentJobActionRequest,
    *,
    db: AsyncSession,
    current_user: User,
) -> AgentJob:
    """Compatibility wrapper around the autonomy application state machine."""

    try:
        return await perform_job_action_state_machine(
            job,
            request,
            deps=_build_job_action_dependencies(),
            db=db,
            current_user=current_user,
        )
    except JobActionError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.detail,
        ) from exc


async def _perform_job_action_for_api(*args, **kwargs):
    """Resolve the legacy action state machine at call time."""

    return await _perform_job_action(*args, **kwargs)


async def _record_job_operator_event_for_api(**kwargs):
    """Resolve the legacy operator-event hook at call time."""

    return await _record_job_operator_event(**kwargs)


job_action_api = build_job_action_api(
    router=router,
    is_job_visible=_is_job_visible_to_user,
    perform_job_action=_perform_job_action_for_api,
    record_operator_event=_record_job_operator_event_for_api,
    extract_scheduler_state=_extract_scheduler_state,
    present_job=_job_to_response,
)
job_action = job_action_api.job_action


async def _perform_follow_up_queue_action_for_api(**kwargs):
    """Resolve the legacy follow-up action hook at call time."""

    return await _perform_follow_up_queue_action(**kwargs)


async def _record_follow_up_queue_decision_event_for_api(**kwargs):
    """Resolve the legacy decision-event hook at call time."""

    return await _record_follow_up_queue_decision_event(**kwargs)


checkpoint_follow_up_action_api = build_checkpoint_follow_up_action_api(
    router=router,
    perform_follow_up_action=_perform_follow_up_queue_action_for_api,
    record_decision_event=_record_follow_up_queue_decision_event_for_api,
    resolve_profile_parent_job=_resolve_profile_parent_job_for_queue,
    resolve_portfolio_parent_job=_resolve_portfolio_parent_job_for_queue,
    extract_scheduler_state=_extract_scheduler_state,
    follow_up_reason_label=_follow_up_opportunity_reason_label,
    queue_reason_label=_queue_reason_label,
)
checkpoint_queue_follow_up_action = (
    checkpoint_follow_up_action_api.checkpoint_queue_follow_up_action
)
checkpoint_queue_bulk_follow_up_action = (
    checkpoint_follow_up_action_api.checkpoint_queue_bulk_follow_up_action
)


checkpoint_job_action_api = build_checkpoint_job_action_api(
    router=router,
    allowed_actions=QUEUE_BULK_ACTIONS,
    extract_approval_payload=_approval_payload_from_results,
    perform_job_action=_perform_job_action_for_api,
    record_operator_event=_record_job_operator_event_for_api,
    extract_scheduler_state=_extract_scheduler_state,
)
checkpoint_queue_bulk_action = checkpoint_job_action_api.checkpoint_queue_bulk_action
_validate_bulk_queue_action = checkpoint_job_action_api.validate_bulk_queue_action
_job_matches_bulk_queue_item_type = (
    checkpoint_job_action_api.job_matches_bulk_queue_item_type
)


job_log_api = build_job_log_api()
router.include_router(job_log_api.router)
get_job_log = job_log_api.get_job_log


job_step_event_api = build_job_step_event_api(
    load_latest_checkpoint=_load_latest_job_checkpoint,
)
router.include_router(job_step_event_api.router)
get_job_step_events = job_step_event_api.get_job_step_events


job_export_api = build_job_export_api()
router.include_router(job_export_api.router)
export_job_results = job_export_api.export_job_results
export_job_transcript = job_export_api.export_job_transcript


job_checkpoint_api = build_job_checkpoint_api()
router.include_router(job_checkpoint_api.router)
get_job_checkpoints = job_checkpoint_api.get_job_checkpoints


job_progress_api = build_job_progress_api()
router.include_router(job_progress_api.router)
agent_job_progress_websocket = job_progress_api.agent_job_progress_websocket


_build_chain_config_for_step = agent_chain_launch_service.build_chain_config_for_step


# ============================================================================
# Job Memory Endpoints
# ============================================================================

_to_int = memory_presenters.to_int
_to_float = memory_presenters.to_float
_to_string_list = memory_presenters.to_string_list
_to_string = memory_presenters.to_string
_build_extract_job_memories_response = (
    memory_presenters.build_extract_job_memories_response
)
_build_job_memory_response = memory_presenters.build_job_memory_response
_build_job_memories_list_response = memory_presenters.build_job_memories_list_response
_build_memory_search_response = memory_presenters.build_memory_search_response
_build_memory_stats_response = memory_presenters.build_memory_stats_response
_build_memory_graph_response = memory_presenters.build_memory_graph_response
