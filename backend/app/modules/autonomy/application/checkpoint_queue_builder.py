"""The operator queue, composed: jobs, inbox items, portfolios and profiles
turned into the rows a person answers.

This lived in ``api/endpoints/agent_jobs.py`` as a private composer bound to
about twenty endpoint-private helpers, so the monitoring task that alerts on
queue rows imported it from the endpoint -- the last upward import the
layering guard allowed (``tests/test_layering.py``). The endpoint, the control
plane and the monitoring task now all read it from here. ``present_job`` is the
job presenter with this application's extractors bound, which the composer
needs for every job row.
"""

from __future__ import annotations

from typing import Any, Optional

from app.models.agent_job import AgentJob
from app.models.domain_research_profile import DomainResearchProfile
from app.models.research_portfolio import ResearchPortfolio
from app.models.user import User
from app.modules.autonomy.application import (
    checkpoint_queue_composer,
    checkpoint_queue_priority,
    follow_up_recommendations,
    job_presenters,
    operator_queue_context,
    swarm_summaries,
)
from app.schemas.agent_job import AgentJobResponse
from app.services.agent_job_queue_helpers import (
    extract_approval_checkpoint,
    extract_launch_mode,
    parse_optional_datetime,
    queue_customer_for_job,
    queue_evidence_summary_for_job,
)
from app.services.agent_job_scheduler_state import (
    extract_scheduler_state,
    queue_reason_label,
)
from app.services.autonomy_service import (
    build_autonomy_summary,
    build_monitor_policy_compat_fields,
    current_domain_profile_policy_snapshot,
    resolve_domain_profile_automation_contract,
)
from app.services.research_monitor_profile_service import (
    research_monitor_profile_service,
)
from app.services.research_opportunity_service import (
    classify_portfolio_operator_review,
    list_normalized_research_opportunities,
)
from app.services.scientific_validation_service import (
    normalize_portfolio_automation_profile,
    resolve_portfolio_automation_policy,
)


def customer_profile_key(customer: Optional[str]) -> str:
    return str(customer or "").strip().lower()


def portfolio_summary_payload(portfolio: ResearchPortfolio) -> dict[str, Any]:
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


def profile_summary_payload(profile: DomainResearchProfile) -> dict[str, Any]:
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


def extract_domain_research_promotion(job: AgentJob) -> dict[str, Any]:
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


def extract_goal_contract_summary(job: AgentJob) -> Optional[dict]:
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


def extract_executive_digest(job: AgentJob) -> Optional[dict]:
    """Extract deterministic executive digest payload when present."""
    results = job.results if isinstance(job.results, dict) else {}
    digest = (
        results.get("executive_digest")
        if isinstance(results.get("executive_digest"), dict)
        else None
    )
    return digest


def present_job(
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
            extract_launch_mode=extract_launch_mode,
            extract_promotion=extract_domain_research_promotion,
            extract_swarm_summary=swarm_summaries.extract_swarm_summary,
            extract_goal_contract_summary=extract_goal_contract_summary,
            extract_approval_checkpoint=extract_approval_checkpoint,
            extract_executive_digest=extract_executive_digest,
        ),
    )


build_checkpoint_queue_items = checkpoint_queue_composer.bind_checkpoint_queue_composer(
    dependencies_factory=lambda: (
        checkpoint_queue_composer.CheckpointQueueCompositionDependencies(
            extract_approval_checkpoint=extract_approval_checkpoint,
            extract_scheduler_state=extract_scheduler_state,
            queue_customer_for_job=queue_customer_for_job,
            present_job=present_job,
            queue_priority_fields=checkpoint_queue_priority.queue_priority_fields,
            queue_evidence_summary_for_job=queue_evidence_summary_for_job,
            queue_reason_label=queue_reason_label,
            parse_optional_datetime=parse_optional_datetime,
            extract_launch_mode=extract_launch_mode,
            build_policy_compat_fields=build_monitor_policy_compat_fields,
            safe_autonomy_recommendations=tuple(
                research_monitor_profile_service.SAFE_AUTONOMY_RECOMMENDATIONS
            ),
            build_follow_up_actions=follow_up_recommendations.build_follow_up_actions,
            customer_profile_key=customer_profile_key,
            build_portfolio_summary=portfolio_summary_payload,
            build_profile_summary=profile_summary_payload,
            classify_operator_review=classify_portfolio_operator_review,
            build_operator_context=operator_queue_context.build_operator_queue_context,
            clean_text_list=operator_queue_context.clean_text_list,
        )
    )
)


__all__ = [
    "build_checkpoint_queue_items",
    "customer_profile_key",
    "extract_domain_research_promotion",
    "extract_executive_digest",
    "extract_goal_contract_summary",
    "portfolio_summary_payload",
    "present_job",
    "profile_summary_payload",
]
