from uuid import UUID

from app.schemas.research_portfolio import ResearchPortfolioCreate
from app.services.autonomy_service import (
    build_autonomy_summary,
    resolve_monitor_automation_contract,
)


def test_research_portfolio_create_normalizes_defaults():
    payload = ResearchPortfolioCreate(
        title=" Retrieval Fleet ",
        objective=" Rank ideas ",
        linked_profile_ids="11111111-1111-1111-1111-111111111111,22222222-2222-2222-2222-222222222222",
        automation_policy={},
    )

    assert payload.title == "Retrieval Fleet"
    assert payload.objective == "Rank ideas"
    assert payload.automation_profile == "balanced"
    assert payload.linked_profile_ids == [
        UUID("11111111-1111-1111-1111-111111111111"),
        UUID("22222222-2222-2222-2222-222222222222"),
    ]
    assert payload.automation_policy == {
        "max_auto_follow_up_launches": 2,
        "confidence_threshold": 0.72,
        "experiment_readiness_threshold": 0.8,
        "duplicate_window_items": 60,
        "auto_create_experiment_plans": True,
        "auto_launch_follow_up": True,
        "auto_execute_validation_runs": False,
        "max_concurrent_validation_runs": 1,
        "max_validation_runtime_minutes": 20,
        "max_validation_budget_per_run": 25.0,
        "follow_up_review_mode": "queue_for_approval",
        "validation_backoff_policy": {
            "max_consecutive_failures": 2,
            "cooldown_minutes": 180,
        },
        "auto_launch_experiment_runs": False,
    }


def test_research_portfolio_create_normalizes_max_autonomy_profile_defaults():
    payload = ResearchPortfolioCreate(
        title=" Scientific Fleet ",
        objective=" Convert discoveries into validated plans ",
        linked_profile_ids=["11111111-1111-1111-1111-111111111111"],
        automation_profile="max_autonomy",
        automation_policy={},
    )

    assert payload.automation_profile == "max_autonomy"
    assert payload.automation_policy == {
        "max_auto_follow_up_launches": 4,
        "confidence_threshold": 0.68,
        "experiment_readiness_threshold": 0.72,
        "duplicate_window_items": 120,
        "auto_create_experiment_plans": True,
        "auto_launch_follow_up": True,
        "auto_execute_validation_runs": True,
        "max_concurrent_validation_runs": 2,
        "max_validation_runtime_minutes": 30,
        "max_validation_budget_per_run": 50.0,
        "follow_up_review_mode": "auto_launch_safe",
        "validation_backoff_policy": {
            "max_consecutive_failures": 2,
            "cooldown_minutes": 180,
        },
        "auto_launch_experiment_runs": True,
    }


def test_build_autonomy_summary_populates_canonical_fields():
    summary = build_autonomy_summary(
        raw_summary={},
        opportunities=[
            {
                "opportunity_id": "opp-1",
                "canonical_key": "cache_miss_hotspot",
                "title": "Cache miss hotspot",
                "hypothesis": "Improve locality",
                "stage": "planned",
                "decision_state": "accepted",
                "autonomy_state": "eligible",
                "linked_experiment_plan_ids": ["plan-1"],
            }
        ],
        automation_profile="max_autonomy",
        effective_policy={
            "follow_up_review_mode": "queue_for_approval",
            "confidence_threshold": 0.68,
        },
        sandbox_profile_id="scientific-generic-sandbox",
        config_revision_key="portfolio_config_revision",
    )

    assert summary["autonomy_mode"] == "max_autonomy"
    assert summary["effective_policy"]["follow_up_review_mode"] == "queue_for_approval"
    assert summary["stage_counts"]["planned"] == 1
    assert summary["autonomy_state_counts"]["eligible"] == 1
    assert summary["portfolio_config_revision"]


def test_resolve_monitor_automation_contract_keeps_compatibility_mirror():
    payload = resolve_monitor_automation_contract(
        {
            "automation_profile": "balanced",
            "automation_policy": {
                "follow_up_review_mode": "queue_for_approval",
                "allowed_recommendations": ["single_research_job"],
            },
        },
        default_allowed=["deep_dive_chain", "single_research_job"],
    )

    assert payload["automation_profile"] == "balanced"
    assert payload["effective_policy"]["follow_up_review_mode"] == "queue_for_approval"
    assert payload["follow_up_autonomy"]["mode"] == "queue_for_approval"
    assert payload["follow_up_autonomy"]["allowed_recommendations"] == [
        "single_research_job"
    ]
