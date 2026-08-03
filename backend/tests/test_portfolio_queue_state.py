"""Tests for portfolio state synchronization after queue decisions."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.modules.autonomy.application.portfolio_queue_state import (
    PortfolioQueueStateDependencies,
    sync_portfolio_queue_state,
)


def _portfolio(**overrides):
    values = {
        "automation_profile": "guarded",
        "automation_policy": {"follow_up_review_mode": "approval_required"},
        "latest_summary": {"status": "completed"},
        "sandbox_profile_id": "sandbox-1",
        "opportunities": [],
        "latest_note_ids": ["note-existing", "note-existing"],
        "latest_experiment_plan_ids": [],
        "latest_validation_run_ids": [],
        "child_job_ids": [],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_sync_portfolio_queue_state_projects_normalized_summary_and_ids():
    source_opportunities = [{"opportunity_id": "raw"}]
    normalized = [{"opportunity_id": "normalized"}]
    linked_ids = {
        "note_ids": ["note-linked"],
        "plan_ids": ["plan-1"],
        "run_ids": ["run-1"],
        "child_job_ids": ["job-1"],
    }
    summary = {"opportunities": normalized, "status": "completed"}
    normalize_opportunities = Mock(return_value=normalized)
    collect_linked_ids = Mock(return_value=linked_ids)
    normalize_profile = Mock(return_value="guarded")
    resolve_policy = Mock(return_value={"mode": "guarded"})
    build_summary = Mock(return_value=summary)
    portfolio = _portfolio()

    await sync_portfolio_queue_state(
        portfolio=portfolio,
        opportunities=source_opportunities,
        deps=PortfolioQueueStateDependencies(
            normalize_opportunities=normalize_opportunities,
            collect_linked_ids=collect_linked_ids,
            normalize_automation_profile=normalize_profile,
            resolve_automation_policy=resolve_policy,
            build_summary=build_summary,
        ),
    )

    normalize_opportunities.assert_called_once_with(source_opportunities)
    collect_linked_ids.assert_called_once_with(normalized)
    normalize_profile.assert_called_once_with("guarded", default="balanced")
    resolve_policy.assert_called_once_with(
        "guarded",
        portfolio.automation_policy,
    )
    build_summary.assert_called_once_with(
        raw_summary={"status": "completed"},
        opportunities=normalized,
        automation_profile="guarded",
        effective_policy={"mode": "guarded"},
        sandbox_profile_id="sandbox-1",
        config_revision_key="portfolio_config_revision",
    )
    assert portfolio.opportunities is normalized
    assert portfolio.latest_summary is summary
    assert portfolio.latest_note_ids == ["note-existing", "note-linked"]
    assert portfolio.latest_experiment_plan_ids == ["plan-1"]
    assert portfolio.latest_validation_run_ids == ["run-1"]
    assert portfolio.child_job_ids == ["job-1"]


@pytest.mark.asyncio
async def test_sync_portfolio_queue_state_applies_linked_id_limits():
    linked_ids = {
        "note_ids": [f"note-{index}" for index in range(40)],
        "plan_ids": [f"plan-{index}" for index in range(40)],
        "run_ids": [f"run-{index}" for index in range(40)],
        "child_job_ids": [f"job-{index}" for index in range(60)],
    }
    dependencies = PortfolioQueueStateDependencies(
        normalize_opportunities=lambda rows: rows,
        collect_linked_ids=lambda _rows: linked_ids,
        normalize_automation_profile=lambda *_args, **_kwargs: "balanced",
        resolve_automation_policy=lambda *_args: {},
        build_summary=lambda **_kwargs: {},
    )
    portfolio = _portfolio(latest_note_ids=[])

    await sync_portfolio_queue_state(
        portfolio=portfolio,
        opportunities=[],
        deps=dependencies,
    )

    assert len(portfolio.latest_note_ids) == 30
    assert len(portfolio.latest_experiment_plan_ids) == 30
    assert len(portfolio.latest_validation_run_ids) == 30
    assert len(portfolio.child_job_ids) == 50
