from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.agent_chain_orchestration_service import (
    AgentChainOrchestrationService,
)
from app.services.agent_follow_up_job_service import AgentFollowUpJobService
from app.services.agent_scientific_validation_service import (
    AgentScientificValidationService,
)


def _make_job(config=None) -> AgentJob:
    return AgentJob(
        name="Orchestration Test",
        goal="Research compiler regressions",
        job_type="research",
        user_id=uuid4(),
        status=AgentJobStatus.RUNNING.value,
        config=config or {},
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=30,
    )


def test_scientific_validation_context_key_is_stable():
    service = AgentScientificValidationService()

    key = service.scientific_validation_context_key(
        object(),
        profile_id="profile-1",
        portfolio_id="portfolio-1",
        hypothesis_id="hypothesis-1",
    )

    assert key == "profile-1|portfolio-1|hypothesis-1"


@pytest.mark.asyncio
async def test_follow_up_job_service_inherits_policy_and_sandbox_context(db_session):
    service = AgentFollowUpJobService()
    parent_job = _make_job(
        {
            "automation_profile": "balanced",
            "automation_policy": {"follow_up_review_mode": "queue_for_approval"},
            "sandbox_profile_id": "scientific-compiler-sandbox",
        }
    )
    db_session.add(parent_job)
    await db_session.flush()

    child = await service.create_domain_research_follow_up_job(
        object(),
        db=db_session,
        job=parent_job,
        domain="Compiler optimization",
        objective="Validate the strongest compiler hypothesis",
        customer_context="bounded compiler research",
        track_type="compiler",
        source_scope="kb_plus_arxiv_plus_repo",
        top_idea={
            "title": "Vectorization regression",
            "hypothesis": "A pass ordering issue regressed vectorization",
        },
        docs=[],
        repo_documents=[],
        papers=[],
        repo_source_ids=["repo-source-1"],
        benchmark_queries=["compile time regression"],
        automation_profile="balanced",
        automation_policy={
            "follow_up_review_mode": "queue_for_approval",
            "auto_execute_validation_runs": False,
        },
        sandbox_profile_id="scientific-compiler-sandbox",
        profile_id="profile-compiler-1",
    )

    assert child is not None
    assert child.config["automation_profile"] == "balanced"
    assert (
        child.config["automation_policy"]["follow_up_review_mode"]
        == "queue_for_approval"
    )
    assert child.config["sandbox_profile_id"] == "scientific-compiler-sandbox"
    assert child.config["profile_id"] == "profile-compiler-1"
    assert (
        child.config["validation_policy"]["follow_up_review_mode"]
        == "queue_for_approval"
    )


@pytest.mark.asyncio
async def test_chain_service_builds_swarm_sibling_payload_excluding_aggregators(
    db_session,
):
    service = AgentChainOrchestrationService()
    parent = _make_job()
    db_session.add(parent)
    await db_session.flush()

    child_a = AgentJob(
        name="Child A",
        goal="A",
        job_type="analysis",
        user_id=parent.user_id,
        status=AgentJobStatus.COMPLETED.value,
        parent_job_id=parent.id,
        progress=100,
        config={"swarm_role": "researcher"},
        results={"summary": "done"},
    )
    child_b = AgentJob(
        name="Aggregator",
        goal="B",
        job_type="synthesis",
        user_id=parent.user_id,
        status=AgentJobStatus.COMPLETED.value,
        parent_job_id=parent.id,
        progress=100,
        config={"origin": "swarm_fan_in_aggregator", "swarm_role": "aggregator"},
        results={"summary": "aggregate"},
    )
    db_session.add_all([child_a, child_b])
    await db_session.flush()

    payload = await service.build_swarm_sibling_payload(object(), parent, db_session)

    assert payload["expected_siblings"] == 1
    assert payload["terminal_siblings"] == 1
    assert len(payload["sibling_jobs"]) == 1
    assert payload["sibling_jobs"][0]["role"] == "researcher"


@pytest.mark.asyncio
async def test_chain_service_trigger_progress_chain_uses_executor_hook():
    service = AgentChainOrchestrationService()
    job = _make_job()
    job.should_trigger_chain = lambda event, value: event == "progress"
    executor = SimpleNamespace(
        _trigger_chained_jobs=AsyncMock(return_value=["child-1"])
    )

    triggered = await service.trigger_progress_chain(
        executor,
        job,
        progress=42,
        findings_count=7,
        db=AsyncMock(),
    )

    assert triggered == ["child-1"]
    executor._trigger_chained_jobs.assert_awaited_once()
