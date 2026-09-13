"""Load and aggregate derived autonomy decision-trace events."""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.agent_job import AgentJob
from app.models.domain_research_profile import DomainResearchProfile
from app.models.experiment import ExperimentRun
from app.models.research_inbox import ResearchInboxItem
from app.models.research_portfolio import ResearchPortfolio
from app.models.user import User
from app.schemas.agent_job import AgentDecisionTraceEventResponse


@dataclass(frozen=True)
class DecisionTraceLoaderDependencies:
    customer_profile_key: Callable[[str | None], str]
    load_learning_profile: Callable[..., Awaitable[dict[str, Any]]]
    build_monitor_snapshot: Callable[..., dict[str, Any]]
    build_queue_items: Callable[..., list[Any]]
    build_queue_events: Callable[[list[Any]], list[AgentDecisionTraceEventResponse]]
    build_job_events: Callable[[AgentJob], list[AgentDecisionTraceEventResponse]]
    portfolio_summary: Callable[[ResearchPortfolio], dict[str, Any]]
    profile_summary: Callable[[DomainResearchProfile], dict[str, Any]]
    build_opportunity_events: Callable[..., list[AgentDecisionTraceEventResponse]]
    build_monitor_events: Callable[[dict], list[AgentDecisionTraceEventResponse]]
    build_validation_events: Callable[
        [list[Any]], list[AgentDecisionTraceEventResponse]
    ]


async def load_derived_decision_trace_events(
    *,
    db: AsyncSession,
    current_user: User,
    deps: DecisionTraceLoaderDependencies,
) -> list[AgentDecisionTraceEventResponse]:
    """Load visible source state and derive a unified decision-event stream."""
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
        .order_by(
            ResearchInboxItem.updated_at.desc(),
            ResearchInboxItem.discovered_at.desc(),
        )
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

    learning_profiles = await _load_learning_profiles(
        db=db,
        current_user=current_user,
        inbox_items=inbox_items,
        deps=deps,
    )
    monitor_snapshot = deps.build_monitor_snapshot(
        items=inbox_items,
        jobs_by_id={job.id: job for job in jobs if job.id is not None},
    )
    queue_items = deps.build_queue_items(
        jobs,
        [
            item
            for item in inbox_items
            if str(item.status or "").strip().lower() == "accepted"
        ],
        portfolios,
        profiles,
        learning_profiles=learning_profiles,
        monitor_health_rows=monitor_snapshot.get("monitors", []),
    )

    derived_events = deps.build_queue_events(queue_items)
    for job in jobs:
        derived_events.extend(deps.build_job_events(job))
    for portfolio in portfolios:
        derived_events.extend(_build_portfolio_events(portfolio, deps=deps))
    for profile in profiles:
        derived_events.extend(_build_profile_events(profile, deps=deps))
    derived_events.extend(deps.build_monitor_events(monitor_snapshot))
    derived_events.extend(deps.build_validation_events(validation_runs))
    return derived_events


async def _load_learning_profiles(
    *,
    db: AsyncSession,
    current_user: User,
    inbox_items: list[ResearchInboxItem],
    deps: DecisionTraceLoaderDependencies,
) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    customers = sorted(
        {
            str(item.customer or "").strip()
            for item in inbox_items
            if str(item.customer or "").strip()
        }
    )
    for customer in customers:
        profiles[
            deps.customer_profile_key(customer)
        ] = await deps.load_learning_profile(
            db=db,
            user_id=current_user.id,
            customer=customer or None,
        )
    return profiles


def _build_portfolio_events(
    portfolio: ResearchPortfolio,
    *,
    deps: DecisionTraceLoaderDependencies,
) -> list[AgentDecisionTraceEventResponse]:
    payload = deps.portfolio_summary(portfolio)
    summary = payload["summary"]
    return deps.build_opportunity_events(
        source_kind="portfolio",
        source_id=str(portfolio.id),
        source_label=str(portfolio.title or "Research fleet").strip(),
        customer=None,
        opportunities=payload["opportunities"],
        deep_link_params={"tab": "fleet", "fleetId": str(portfolio.id)},
        objective=str(portfolio.objective or "").strip() or None,
        sandbox_profile_id=str(portfolio.sandbox_profile_id or "").strip() or None,
        automation_profile=str(portfolio.automation_profile or "").strip() or None,
        effective_policy=(
            summary.get("effective_policy") if isinstance(summary, dict) else None
        ),
    )


def _build_profile_events(
    profile: DomainResearchProfile,
    *,
    deps: DecisionTraceLoaderDependencies,
) -> list[AgentDecisionTraceEventResponse]:
    payload = deps.profile_summary(profile)
    summary = payload["summary"]
    return deps.build_opportunity_events(
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
        effective_policy=(
            summary.get("effective_policy") if isinstance(summary, dict) else None
        ),
    )
