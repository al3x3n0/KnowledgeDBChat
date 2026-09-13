"""Tests for domain-research profile and portfolio promotion seeds."""

from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.endpoints.agent_jobs import _build_domain_research_promotion_seed
from app.models.agent_job import AgentJob
from app.modules.autonomy.application.domain_research_promotion_seed import (
    DomainResearchPromotionSeedDependencies,
    DomainResearchPromotionSeedError,
    build_domain_research_promotion_seed,
)
from app.services.agent_job_queue_helpers import extract_launch_mode
from app.services.autonomy_service import resolve_domain_profile_automation_contract
from app.services.scientific_validation_service import (
    normalize_portfolio_automation_profile,
    resolve_portfolio_automation_policy,
)

DEPENDENCIES = DomainResearchPromotionSeedDependencies(
    extract_launch_mode=extract_launch_mode,
    resolve_domain_automation_contract=resolve_domain_profile_automation_contract,
    normalize_portfolio_automation_profile=normalize_portfolio_automation_profile,
    resolve_portfolio_automation_policy=resolve_portfolio_automation_policy,
)


def _job(config):
    return AgentJob(
        name="Compiler Research " + "x" * 240,
        goal="Research compiler optimization",
        job_type="research",
        user_id=uuid4(),
        config=config,
    )


def test_promotion_seed_requires_domain_research_launch_mode():
    with pytest.raises(DomainResearchPromotionSeedError) as exc_info:
        build_domain_research_promotion_seed(
            _job({"domain": "Compilers", "objective": "Find optimization gaps"}),
            deps=DEPENDENCIES,
        )

    assert str(exc_info.value) == "Job is not a domain research quick start"


def test_legacy_promotion_seed_wrapper_preserves_http_error_contract():
    with pytest.raises(HTTPException) as exc_info:
        _build_domain_research_promotion_seed(
            _job({"domain": "Compilers", "objective": "Find optimization gaps"})
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "Job is not a domain research quick start"


def test_promotion_seed_requires_domain_and_objective():
    with pytest.raises(DomainResearchPromotionSeedError) as exc_info:
        build_domain_research_promotion_seed(
            _job({"launch_mode": "quick_start_domain_research", "domain": "LLVM"}),
            deps=DEPENDENCIES,
        )

    assert str(exc_info.value) == "Job is missing normalized domain research config"


def test_promotion_seed_builds_bounded_profile_and_portfolio_payloads():
    seed = build_domain_research_promotion_seed(
        _job(
            {
                "launch_mode": "quick_start_domain_research",
                "domain": "Compiler Microarchitecture",
                "objective": "Find measurable scheduling improvements",
                "automation_profile": "balanced",
                "monitor_queries": ["", *[f"query-{index}" for index in range(15)]],
                "benchmark_queries": [f"bench-{index}" for index in range(20)],
                "repo_source_ids": [f"repo-{index}" for index in range(30)],
                "sandbox_profile_id": "compiler-sandbox",
                "interval_minutes": 60,
                "max_documents": 14,
                "max_papers": 9,
            }
        ),
        deps=DEPENDENCIES,
    )

    profile = seed["profile"]
    portfolio = seed["portfolio"]
    assert len(profile["title"]) == 200
    assert len(profile["monitor_queries"]) == 12
    assert len(profile["benchmark_queries"]) == 16
    assert len(profile["repo_source_ids"]) == 24
    assert profile["track_type"] == "compiler"
    assert profile["research_mode"] == "literature_to_hypothesis"
    assert profile["sandbox_profile_id"] == "compiler-sandbox"
    assert profile["interval_minutes"] == 60
    assert profile["max_documents"] == 14
    assert profile["max_papers"] == 9
    assert profile["start_immediately"] is False
    assert portfolio["title"] == "Compiler Microarchitecture Fleet"
    assert portfolio["objective"] == "Find measurable scheduling improvements"
    assert portfolio["sandbox_profile_id"] == "compiler-sandbox"
    assert portfolio["start_immediately"] is False
