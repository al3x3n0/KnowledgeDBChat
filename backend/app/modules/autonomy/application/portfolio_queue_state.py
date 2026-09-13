"""Synchronize portfolio opportunity state after queue decisions."""

from dataclasses import dataclass
from typing import Any, Callable

from app.models.research_portfolio import ResearchPortfolio
from app.services.autonomy_service import build_autonomy_summary
from app.services.research_opportunity_service import (
    collect_research_opportunity_linked_ids,
    list_normalized_research_opportunities,
)
from app.services.scientific_validation_service import (
    normalize_portfolio_automation_profile,
    resolve_portfolio_automation_policy,
)


@dataclass(frozen=True)
class PortfolioQueueStateDependencies:
    normalize_opportunities: Callable[..., list[dict[str, Any]]]
    collect_linked_ids: Callable[..., dict[str, list[str]]]
    normalize_automation_profile: Callable[..., str]
    resolve_automation_policy: Callable[..., dict[str, Any]]
    build_summary: Callable[..., dict[str, Any]]


DEFAULT_DEPENDENCIES = PortfolioQueueStateDependencies(
    normalize_opportunities=list_normalized_research_opportunities,
    collect_linked_ids=collect_research_opportunity_linked_ids,
    normalize_automation_profile=normalize_portfolio_automation_profile,
    resolve_automation_policy=resolve_portfolio_automation_policy,
    build_summary=build_autonomy_summary,
)


async def sync_portfolio_queue_state(
    *,
    portfolio: ResearchPortfolio,
    opportunities: list[dict[str, Any]],
    deps: PortfolioQueueStateDependencies = DEFAULT_DEPENDENCIES,
) -> None:
    """Project normalized opportunities and their linked IDs onto a portfolio."""
    normalized = deps.normalize_opportunities(opportunities)
    linked_ids = deps.collect_linked_ids(normalized)
    automation_profile = deps.normalize_automation_profile(
        getattr(portfolio, "automation_profile", None),
        default="balanced",
    )
    effective_policy = deps.resolve_automation_policy(
        automation_profile,
        portfolio.automation_policy,
    )
    summary = deps.build_summary(
        raw_summary=(
            portfolio.latest_summary
            if isinstance(portfolio.latest_summary, dict)
            else {}
        ),
        opportunities=normalized,
        automation_profile=automation_profile,
        effective_policy=effective_policy,
        sandbox_profile_id=portfolio.sandbox_profile_id,
        config_revision_key="portfolio_config_revision",
    )

    portfolio.opportunities = normalized
    portfolio.latest_summary = summary
    portfolio.latest_note_ids = _merge_existing_ids(
        portfolio.latest_note_ids,
        linked_ids["note_ids"],
        limit=30,
    )
    portfolio.latest_experiment_plan_ids = linked_ids["plan_ids"][:30]
    portfolio.latest_validation_run_ids = linked_ids["run_ids"][:30]
    portfolio.child_job_ids = linked_ids["child_job_ids"][:50]


def _merge_existing_ids(existing: Any, linked: list[str], *, limit: int) -> list[str]:
    existing_ids = [str(value) for value in (existing or []) if str(value).strip()]
    return list(dict.fromkeys([*existing_ids, *linked]))[:limit]
