import asyncio
from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api.endpoints import domain_research_profiles, research_portfolios
from app.core.database import get_db
from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.domain_research_profile import DomainResearchProfile
from app.models.experiment import ExperimentPlan, ExperimentRun
from app.models.research_inbox import ResearchInboxItem
from app.models.research_note import ResearchNote
from app.models.research_portfolio import ResearchPortfolio
from app.models.synthesis_job import SynthesisJob


@pytest.fixture
def scientific_validation_summary_client(db_session, test_user):
    app = FastAPI()
    app.include_router(domain_research_profiles.router, prefix="/api/v1/domain-research-profiles")
    app.include_router(research_portfolios.router, prefix="/api/v1/research-portfolios")

    def override_get_db():
        return db_session

    async def override_current_user():
        return test_user

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[domain_research_profiles.get_current_active_user] = override_current_user
    app.dependency_overrides[research_portfolios.get_current_active_user] = override_current_user

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_domain_profile_list_embeds_latest_validation_runs(
    scientific_validation_summary_client,
    db_session,
    test_user,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Validation Note",
        content_markdown="## Hypothesis\nValidation summary should be embedded.",
        tags=["scientific-validation"],
    )
    plan = ExperimentPlan(
        user_id=test_user.id,
        research_note_id=note.id,
        title="Validation Plan",
        hypothesis_text="Validation summary should be embedded.",
        plan={"experiments": []},
    )
    run = ExperimentRun(
        user_id=test_user.id,
        experiment_plan_id=plan.id,
        agent_job_id=None,
        name="Compiler Validation Run",
        status="blocked",
        progress=100,
        config={
            "scientific_validation": {
                "validation_kind": "scientific_validation",
                "sandbox_profile_id": "scientific-compiler-sandbox",
                "recipe_family": "compiler_validation",
                "recipe_id": "compiler_validation_v1",
                "blocked_reason_code": "disallowed_image",
                "hypothesis_id": "idea_1",
                "profile_snapshot": {
                    "id": "scientific-compiler-sandbox",
                    "name": "Compiler Validation Sandbox",
                },
            }
        },
    )
    profile = DomainResearchProfile(
        user_id=test_user.id,
        title="Compiler Frontier",
        domain="Compiler",
        objective="Track compiler opportunities",
        status="running",
        automation_profile="max_autonomy",
        automation_policy={"follow_up_review_mode": "queue_for_approval"},
        source_scope="kb_plus_arxiv_plus_repo",
        track_type="compiler",
        research_mode="literature_to_hypothesis",
        report_format="brief_and_report",
        interval_minutes=1440,
        persist_artifacts=True,
        auto_launch_follow_up=True,
        auto_create_experiment_plans=True,
        confidence_threshold=0.7,
        max_documents=10,
        max_papers=8,
        latest_validation_run_ids=[str(run.id)],
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        plan.research_note_id = note.id
        db_session.add(plan)
        await db_session.flush()
        run.experiment_plan_id = plan.id
        db_session.add(run)
        await db_session.flush()
        profile.latest_validation_run_ids = [str(run.id)]
        db_session.add(profile)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = scientific_validation_summary_client.get("/api/v1/domain-research-profiles")

    assert response.status_code == 200
    payload = response.json()["items"][0]
    assert payload["automation_profile"] == "max_autonomy"
    assert payload["effective_policy"]["follow_up_review_mode"] == "queue_for_approval"
    assert payload["latest_summary"]["autonomy_mode"] == "max_autonomy"
    assert payload["latest_summary"]["autonomy_state_counts"]["eligible"] == 0
    assert payload["latest_summary"]["queued_operator_reviews_count"] == 0
    assert payload["latest_summary"]["profile_config_revision"]
    assert payload["latest_validation_run_ids"] == [str(run.id)]
    assert len(payload["latest_validation_runs"]) == 1
    summary = payload["latest_validation_runs"][0]
    assert summary["id"] == str(run.id)
    assert summary["status"] == "blocked"
    assert summary["recipe_id"] == "compiler_validation_v1"
    assert summary["blocked_reason_code"] == "disallowed_image"
    assert summary["sandbox_profile_name"] == "Compiler Validation Sandbox"


def test_domain_profile_validation_summary_embeds_compiler_artifact_handoff(
    scientific_validation_summary_client,
    db_session,
    test_user,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Validation Note",
        content_markdown="## Hypothesis\nValidation summary should include compiler artifact lineage.",
        tags=["scientific-validation"],
    )
    plan = ExperimentPlan(
        user_id=test_user.id,
        research_note_id=note.id,
        title="Validation Plan",
        hypothesis_text="Validation summary should include compiler artifact lineage.",
        plan={"experiments": []},
    )
    run = ExperimentRun(
        user_id=test_user.id,
        experiment_plan_id=plan.id,
        agent_job_id=None,
        name="Compiler Validation Run",
        status="completed",
        progress=100,
        config={
            "scientific_validation": {
                "validation_kind": "scientific_validation",
                "sandbox_profile_id": "scientific-compiler-sandbox",
                "recipe_family": "compiler_validation",
                "recipe_id": "compiler_validation_v1",
                "benchmark_family": "compiler_regression",
                "benchmark_suite_id": "compiler-llvm-regression-core",
                "benchmark_case_ids": ["case-instcombine-sroa"],
                "source_run_ids": [],
                "comparison_run_id": "00000000-0000-0000-0000-000000000123",
                "hypothesis_id": "idea_1",
                "track_type": "compiler",
                "profile_snapshot": {
                    "id": "scientific-compiler-sandbox",
                    "name": "Compiler Validation Sandbox",
                },
            }
        },
    )
    explanation_job = SynthesisJob(
        user_id=test_user.id,
        job_type="compiler_regression_explanation",
        title="Compiler explanation",
        document_ids=[],
        paper_ids=[],
        options={
            "experiment_run_ids": [str(run.id), "00000000-0000-0000-0000-000000000123"],
            "primary_run_id": str(run.id),
            "comparison_run_id": "00000000-0000-0000-0000-000000000123",
        },
        output_format="markdown",
        output_style="technical",
        status="completed",
        progress=100,
        result_metadata={
            "source_run_ids": [str(run.id), "00000000-0000-0000-0000-000000000123"],
            "primary_run_id": str(run.id),
            "comparison_run_id": "00000000-0000-0000-0000-000000000123",
        },
    )
    explanation_note = ResearchNote(
        user_id=test_user.id,
        title="Compiler explanation note",
        content_markdown="## Explanation",
        source_synthesis_job_id=explanation_job.id,
        structured_payload={
            "artifact_type": "compiler_regression_explanation",
            "source_run_ids": [],
            "primary_run_id": None,
            "comparison_run_id": "00000000-0000-0000-0000-000000000123",
        },
    )
    profile = DomainResearchProfile(
        user_id=test_user.id,
        title="Compiler Frontier",
        domain="Compiler",
        objective="Track compiler opportunities",
        status="running",
        automation_profile="balanced",
        automation_policy={"follow_up_review_mode": "queue_for_approval"},
        source_scope="kb_plus_arxiv_plus_repo",
        track_type="compiler",
        research_mode="literature_to_hypothesis",
        report_format="brief_and_report",
        interval_minutes=1440,
        persist_artifacts=True,
        auto_launch_follow_up=True,
        auto_create_experiment_plans=True,
        confidence_threshold=0.7,
        max_documents=10,
        max_papers=8,
        latest_validation_run_ids=[str(run.id)],
        repo_source_ids=["repo-source-1"],
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        plan.research_note_id = note.id
        db_session.add(plan)
        await db_session.flush()
        run.experiment_plan_id = plan.id
        db_session.add(run)
        await db_session.flush()
        run.config = {
            **(run.config or {}),
            "scientific_validation": {
                **((run.config or {}).get("scientific_validation") or {}),
                "source_run_ids": [str(run.id), "00000000-0000-0000-0000-000000000123"],
                "primary_run_id": str(run.id),
            },
        }
        explanation_job.options["experiment_run_ids"] = [str(run.id), "00000000-0000-0000-0000-000000000123"]
        db_session.add(explanation_job)
        await db_session.flush()
        explanation_note.structured_payload = {
            **(explanation_note.structured_payload or {}),
            "source_run_ids": [str(run.id), "00000000-0000-0000-0000-000000000123"],
            "primary_run_id": str(run.id),
        }
        explanation_note.source_synthesis_job_id = explanation_job.id
        db_session.add(explanation_note)
        await db_session.flush()
        db_session.add(profile)
        await db_session.flush()
        run.config = {
            **(run.config or {}),
            "scientific_validation": {
                **((run.config or {}).get("scientific_validation") or {}),
                "domain_research_profile_id": str(profile.id),
            },
        }
        profile.latest_validation_run_ids = [str(run.id)]
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = scientific_validation_summary_client.get("/api/v1/domain-research-profiles")

    assert response.status_code == 200
    summary = response.json()["items"][0]["latest_validation_runs"][0]
    assert summary["track_type"] == "compiler"
    assert summary["domain_research_profile_id"] == str(profile.id)
    assert summary["benchmark_family"] == "compiler_regression"
    assert summary["benchmark_suite_id"] == "compiler-llvm-regression-core"
    assert summary["compiler_artifact_summary"]["explanation_note_id"] == str(explanation_note.id)
    assert summary["compiler_artifact_summary"]["explanation_synthesis_job_id"] == str(explanation_job.id)
    assert summary["compiler_artifact_summary"]["source_run_ids"] == [str(run.id), "00000000-0000-0000-0000-000000000123"]
    assert "create_patch_proposal" in summary["compiler_artifact_summary"]["available_actions"]


def test_domain_profile_validation_summary_embeds_proposal_lineage_and_patch_draft_readiness(
    scientific_validation_summary_client,
    db_session,
    test_user,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Validation Note",
        content_markdown="## Hypothesis\nValidation summary should include proposal lineage.",
        tags=["scientific-validation"],
    )
    plan = ExperimentPlan(
        user_id=test_user.id,
        research_note_id=note.id,
        title="Validation Plan",
        hypothesis_text="Validation summary should include proposal lineage.",
        plan={"experiments": []},
    )
    run = ExperimentRun(
        user_id=test_user.id,
        experiment_plan_id=plan.id,
        agent_job_id=None,
        name="Compiler Validation Run",
        status="completed",
        progress=100,
        config={
            "scientific_validation": {
                "validation_kind": "scientific_validation",
                "sandbox_profile_id": "scientific-compiler-sandbox",
                "recipe_family": "compiler_validation",
                "recipe_id": "compiler_validation_v1",
                "benchmark_family": "compiler_regression",
                "benchmark_suite_id": "compiler-llvm-regression-core",
                "benchmark_case_ids": ["case-instcombine-sroa"],
                "source_run_ids": [],
                "comparison_run_id": "00000000-0000-0000-0000-000000000123",
                "hypothesis_id": "idea_2",
                "track_type": "compiler",
                "profile_snapshot": {
                    "id": "scientific-compiler-sandbox",
                    "name": "Compiler Validation Sandbox",
                },
            }
        },
    )
    explanation_job = SynthesisJob(
        user_id=test_user.id,
        job_type="compiler_regression_explanation",
        title="Compiler explanation",
        document_ids=[],
        paper_ids=[],
        options={
            "experiment_run_ids": [str(run.id), "00000000-0000-0000-0000-000000000123"],
            "primary_run_id": str(run.id),
            "comparison_run_id": "00000000-0000-0000-0000-000000000123",
        },
        output_format="markdown",
        output_style="technical",
        status="completed",
        progress=100,
        result_metadata={
            "source_run_ids": [str(run.id), "00000000-0000-0000-0000-000000000123"],
            "primary_run_id": str(run.id),
            "comparison_run_id": "00000000-0000-0000-0000-000000000123",
        },
    )
    explanation_note = ResearchNote(
        user_id=test_user.id,
        title="Compiler explanation note",
        content_markdown="## Explanation",
        source_synthesis_job_id=explanation_job.id,
        structured_payload={
            "artifact_type": "compiler_regression_explanation",
            "source_run_ids": [],
            "primary_run_id": None,
            "comparison_run_id": "00000000-0000-0000-0000-000000000123",
        },
    )
    proposal_job = SynthesisJob(
        user_id=test_user.id,
        job_type="compiler_patch_proposal",
        title="Compiler proposal",
        document_ids=[],
        paper_ids=[],
        research_note_id=explanation_note.id,
        output_format="markdown",
        output_style="technical",
        status="completed",
        progress=100,
    )
    proposal_note = ResearchNote(
        user_id=test_user.id,
        title="Compiler proposal note",
        content_markdown="## Proposal",
        source_synthesis_job_id=proposal_job.id,
        structured_payload={
            "artifact_type": "compiler_patch_proposal",
            "source_explanation_note_id": None,
        },
    )
    profile = DomainResearchProfile(
        user_id=test_user.id,
        title="Compiler Frontier",
        domain="Compiler",
        objective="Track compiler opportunities",
        status="running",
        automation_profile="balanced",
        automation_policy={"follow_up_review_mode": "queue_for_approval"},
        source_scope="kb_plus_arxiv_plus_repo",
        track_type="compiler",
        research_mode="literature_to_hypothesis",
        report_format="brief_and_report",
        interval_minutes=1440,
        persist_artifacts=True,
        auto_launch_follow_up=True,
        auto_create_experiment_plans=True,
        confidence_threshold=0.7,
        max_documents=10,
        max_papers=8,
        latest_validation_run_ids=[str(run.id)],
        repo_source_ids=["repo-source-1"],
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        plan.research_note_id = note.id
        db_session.add(plan)
        await db_session.flush()
        run.experiment_plan_id = plan.id
        db_session.add(run)
        await db_session.flush()
        run.config = {
            **(run.config or {}),
            "scientific_validation": {
                **((run.config or {}).get("scientific_validation") or {}),
                "source_run_ids": [str(run.id), "00000000-0000-0000-0000-000000000123"],
                "primary_run_id": str(run.id),
            },
        }
        db_session.add(explanation_job)
        await db_session.flush()
        explanation_note.structured_payload = {
            **(explanation_note.structured_payload or {}),
            "source_run_ids": [str(run.id), "00000000-0000-0000-0000-000000000123"],
            "primary_run_id": str(run.id),
        }
        explanation_note.source_synthesis_job_id = explanation_job.id
        db_session.add(explanation_note)
        await db_session.flush()
        proposal_job.research_note_id = explanation_note.id
        db_session.add(proposal_job)
        await db_session.flush()
        proposal_note.structured_payload = {
            **(proposal_note.structured_payload or {}),
            "source_explanation_note_id": str(explanation_note.id),
        }
        proposal_note.source_synthesis_job_id = proposal_job.id
        db_session.add(proposal_note)
        await db_session.flush()
        db_session.add(profile)
        await db_session.flush()
        run.config = {
            **(run.config or {}),
            "scientific_validation": {
                **((run.config or {}).get("scientific_validation") or {}),
                "domain_research_profile_id": str(profile.id),
            },
        }
        profile.latest_validation_run_ids = [str(run.id)]
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = scientific_validation_summary_client.get("/api/v1/domain-research-profiles")

    assert response.status_code == 200
    summary = response.json()["items"][0]["latest_validation_runs"][0]
    artifact_summary = summary["compiler_artifact_summary"]
    assert artifact_summary["explanation_note_id"] == str(explanation_note.id)
    assert artifact_summary["proposal_note_id"] == str(proposal_note.id)
    assert artifact_summary["proposal_synthesis_job_id"] == str(proposal_job.id)
    assert artifact_summary["source_explanation_note_id"] == str(explanation_note.id)
    assert "create_patch_draft" in artifact_summary["available_actions"]


def test_research_portfolio_list_embeds_latest_validation_runs(
    scientific_validation_summary_client,
    db_session,
    test_user,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Portfolio Validation Note",
        content_markdown="## Hypothesis\nPortfolio validation summary should be embedded.",
        tags=["scientific-validation"],
    )
    plan = ExperimentPlan(
        user_id=test_user.id,
        research_note_id=note.id,
        title="Portfolio Validation Plan",
        hypothesis_text="Portfolio validation summary should be embedded.",
        plan={"experiments": []},
    )
    run = ExperimentRun(
        user_id=test_user.id,
        experiment_plan_id=plan.id,
        name="Microarchitecture Validation Run",
        status="running",
        progress=55,
        config={
            "scientific_validation": {
                "validation_kind": "scientific_validation",
                "sandbox_profile_id": "scientific-microarchitecture-sandbox",
                "recipe_family": "microarchitecture_validation",
                "recipe_id": "microarchitecture_validation_v1",
                "hypothesis_id": "idea_7",
                "profile_snapshot": {
                    "id": "scientific-microarchitecture-sandbox",
                    "name": "Microarchitecture Validation Sandbox",
                },
            }
        },
    )
    portfolio = ResearchPortfolio(
        user_id=test_user.id,
        title="Scientific Fleet",
        objective="Track and validate scientific opportunities",
        status="running",
        linked_profile_ids=[],
        automation_policy={},
        sandbox_profile_id="scientific-generic-sandbox",
        opportunities=[],
        latest_summary={},
        latest_note_ids=[],
        latest_experiment_plan_ids=[],
        latest_validation_run_ids=[str(run.id)],
        child_job_ids=[],
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        plan.research_note_id = note.id
        db_session.add(plan)
        await db_session.flush()
        run.experiment_plan_id = plan.id
        db_session.add(run)
        await db_session.flush()
        portfolio.latest_validation_run_ids = [str(run.id)]
        db_session.add(portfolio)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = scientific_validation_summary_client.get("/api/v1/research-portfolios")

    assert response.status_code == 200
    payload = response.json()["items"][0]
    assert payload["automation_profile"] == "balanced"
    assert payload["effective_policy"]["duplicate_window_items"] == 60
    assert payload["effective_policy"]["follow_up_review_mode"] == "queue_for_approval"
    assert payload["latest_summary"]["autonomy_state_counts"]["eligible"] == 0
    assert payload["latest_summary"]["queued_operator_reviews_count"] == 0
    assert payload["latest_summary"]["portfolio_config_revision"]
    assert payload["latest_validation_run_ids"] == [str(run.id)]
    assert len(payload["latest_validation_runs"]) == 1
    summary = payload["latest_validation_runs"][0]
    assert summary["id"] == str(run.id)
    assert summary["status"] == "running"
    assert summary["progress"] == 55
    assert summary["recipe_family"] == "microarchitecture_validation"
    assert summary["sandbox_profile_name"] == "Microarchitecture Validation Sandbox"


def test_domain_profile_opportunity_create_plan_links_plan(
    scientific_validation_summary_client,
    db_session,
    test_user,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Opportunity Note",
        content_markdown="## Hypothesis\nCreate a plan from this opportunity.",
    )
    profile = DomainResearchProfile(
        user_id=test_user.id,
        title="Compiler Frontier",
        domain="Compiler",
        objective="Track compiler opportunities",
        status="running",
        source_scope="kb_plus_arxiv_plus_repo",
        track_type="compiler",
        research_mode="literature_to_hypothesis",
        report_format="brief_and_report",
        interval_minutes=1440,
        persist_artifacts=True,
        auto_launch_follow_up=True,
        auto_create_experiment_plans=True,
        confidence_threshold=0.7,
        max_documents=10,
        max_papers=8,
        latest_note_ids=[],
        latest_summary={
            "opportunities": [
                {
                    "opportunity_id": "opp_compile",
                    "canonical_key": "compiler_hotspot",
                    "title": "Compiler hotspot",
                    "hypothesis": "Scheduler bottleneck",
                    "decision_state": "pending_review",
                    "stage": "discovered",
                    "source_note_ids": [],
                }
            ]
        },
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        profile.latest_note_ids = [str(note.id)]
        db_session.add(profile)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = scientific_validation_summary_client.post(
        f"/api/v1/domain-research-profiles/{profile.id}/opportunities/opp_compile/action",
        json={"action": "create_plan"},
    )

    assert response.status_code == 200
    payload = response.json()
    opp = payload["opportunities"][0]
    assert opp["decision_state"] == "accepted"
    assert opp["stage"] == "planned"
    assert len(opp["linked_experiment_plan_ids"]) == 1

    second = scientific_validation_summary_client.post(
        f"/api/v1/domain-research-profiles/{profile.id}/opportunities/opp_compile/action",
        json={"action": "create_plan"},
    )
    assert second.status_code == 200
    second_opp = second.json()["opportunities"][0]
    assert second_opp["linked_experiment_plan_ids"] == opp["linked_experiment_plan_ids"]


def test_research_portfolio_opportunity_suppress_requires_note(
    scientific_validation_summary_client,
    db_session,
    test_user,
):
    portfolio = ResearchPortfolio(
        user_id=test_user.id,
        title="Scientific Fleet",
        objective="Track and validate scientific opportunities",
        status="running",
        linked_profile_ids=[],
        automation_policy={},
        sandbox_profile_id="scientific-generic-sandbox",
        opportunities=[
            {
                "opportunity_id": "opp_portfolio",
                "canonical_key": "compiler_hotspot",
                "title": "Compiler hotspot",
                "hypothesis": "Scheduler bottleneck",
                "decision_state": "pending_review",
                "stage": "discovered",
            }
        ],
        latest_summary={},
        latest_note_ids=[],
        latest_experiment_plan_ids=[],
        latest_validation_run_ids=[],
        child_job_ids=[],
    )

    async def _seed():
        db_session.add(portfolio)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    bad = scientific_validation_summary_client.post(
        f"/api/v1/research-portfolios/{portfolio.id}/opportunities/opp_portfolio/action",
        json={"action": "suppress"},
    )
    assert bad.status_code == 400

    good = scientific_validation_summary_client.post(
        f"/api/v1/research-portfolios/{portfolio.id}/opportunities/opp_portfolio/action",
        json={"action": "suppress", "operator_note": "Duplicate signal"},
    )
    assert good.status_code == 200
    opp = good.json()["opportunities"][0]
    assert opp["decision_state"] == "suppressed"
    assert opp["stage"] == "suppressed"
    assert opp["operator_note"] == "Duplicate signal"


def test_research_portfolio_opportunity_create_plan_implies_acceptance(
    scientific_validation_summary_client,
    db_session,
    test_user,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Portfolio Opportunity Note",
        content_markdown="## Hypothesis\nCreate a portfolio plan from this opportunity.",
    )
    portfolio = ResearchPortfolio(
        user_id=test_user.id,
        title="Scientific Fleet",
        objective="Track and validate scientific opportunities",
        status="running",
        linked_profile_ids=[],
        automation_policy={},
        sandbox_profile_id="scientific-generic-sandbox",
        opportunities=[
            {
                "opportunity_id": "opp_portfolio_plan",
                "canonical_key": "portfolio_hotspot",
                "title": "Portfolio hotspot",
                "hypothesis": "Benchmarking bottleneck",
                "decision_state": "pending_review",
                "stage": "discovered",
                "source_note_ids": [],
            }
        ],
        latest_summary={},
        latest_note_ids=[],
        latest_experiment_plan_ids=[],
        latest_validation_run_ids=[],
        child_job_ids=[],
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        portfolio.latest_note_ids = [str(note.id)]
        db_session.add(portfolio)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = scientific_validation_summary_client.post(
        f"/api/v1/research-portfolios/{portfolio.id}/opportunities/opp_portfolio_plan/action",
        json={"action": "create_plan"},
    )

    assert response.status_code == 200
    payload = response.json()
    opp = payload["opportunities"][0]
    assert opp["decision_state"] == "accepted"
    assert opp["stage"] == "planned"
    assert len(opp["linked_experiment_plan_ids"]) == 1

    second = scientific_validation_summary_client.post(
        f"/api/v1/research-portfolios/{portfolio.id}/opportunities/opp_portfolio_plan/action",
        json={"action": "create_plan"},
    )
    assert second.status_code == 200
    second_opp = second.json()["opportunities"][0]
    assert second_opp["linked_experiment_plan_ids"] == opp["linked_experiment_plan_ids"]


def test_domain_profile_opportunity_launch_validation_is_idempotent(
    scientific_validation_summary_client,
    db_session,
    test_user,
    monkeypatch,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Validation Opportunity Note",
        content_markdown="## Hypothesis\nLaunch validation from this opportunity.",
    )
    parent_job = AgentJob(
        user_id=test_user.id,
        name="Profile Parent Job",
        goal="Track compiler opportunities",
        job_type="research",
        status=AgentJobStatus.COMPLETED.value,
        progress=100,
        config={},
    )
    profile = DomainResearchProfile(
        user_id=test_user.id,
        title="Compiler Frontier",
        domain="Compiler",
        objective="Track compiler opportunities",
        status="running",
        source_scope="kb_plus_arxiv_plus_repo",
        track_type="compiler",
        research_mode="literature_to_hypothesis",
        report_format="brief_and_report",
        interval_minutes=1440,
        persist_artifacts=True,
        auto_launch_follow_up=True,
        auto_create_experiment_plans=True,
        confidence_threshold=0.7,
        max_documents=10,
        max_papers=8,
        latest_note_ids=[],
        latest_summary={
            "opportunities": [
                {
                    "opportunity_id": "opp_validate",
                    "canonical_key": "compiler_validation",
                    "title": "Compiler validation",
                    "hypothesis": "Launch a bounded validation run",
                    "decision_state": "pending_review",
                    "stage": "discovered",
                    "source_note_ids": [],
                }
            ]
        },
    )
    created_runs: list[str] = []

    async def _seed():
        db_session.add(note)
        db_session.add(parent_job)
        await db_session.flush()
        profile.latest_note_ids = [str(note.id)]
        profile.latest_run_job_id = parent_job.id
        db_session.add(profile)
        await db_session.commit()

    async def _fake_create_validation_run(self, **kwargs):
        created_runs.append(str(kwargs.get("hypothesis_id") or ""))
        return {"run_id": "run-profile-1", "status": "queued", "job_id": "job-profile-1"}

    monkeypatch.setattr(
        "app.services.autonomous_agent_executor.AutonomousAgentExecutor._create_scientific_validation_run",
        _fake_create_validation_run,
    )

    asyncio.get_event_loop().run_until_complete(_seed())

    first = scientific_validation_summary_client.post(
        f"/api/v1/domain-research-profiles/{profile.id}/opportunities/opp_validate/action",
        json={"action": "launch_validation"},
    )
    assert first.status_code == 200
    first_opp = first.json()["opportunities"][0]
    assert first_opp["decision_state"] == "accepted"
    assert first_opp["stage"] == "validating"
    assert first_opp["linked_validation_run_ids"] == ["run-profile-1"]
    assert len(first_opp["linked_experiment_plan_ids"]) == 1

    second = scientific_validation_summary_client.post(
        f"/api/v1/domain-research-profiles/{profile.id}/opportunities/opp_validate/action",
        json={"action": "launch_validation"},
    )
    assert second.status_code == 200
    second_opp = second.json()["opportunities"][0]
    assert second_opp["linked_validation_run_ids"] == ["run-profile-1"]
    assert second_opp["linked_experiment_plan_ids"] == first_opp["linked_experiment_plan_ids"]
    assert created_runs == ["opp_validate"]

    async def _count_plans():
        plans = list((await db_session.execute(select(ExperimentPlan))).scalars().all())
        return len(plans)

    assert asyncio.get_event_loop().run_until_complete(_count_plans()) == 1


def test_research_portfolio_opportunity_launch_validation_is_idempotent(
    scientific_validation_summary_client,
    db_session,
    test_user,
    monkeypatch,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Portfolio Validation Opportunity Note",
        content_markdown="## Hypothesis\nLaunch portfolio validation from this opportunity.",
    )
    parent_job = AgentJob(
        user_id=test_user.id,
        name="Portfolio Parent Job",
        goal="Track scientific opportunities",
        job_type="research",
        status=AgentJobStatus.COMPLETED.value,
        progress=100,
        config={},
    )
    portfolio = ResearchPortfolio(
        user_id=test_user.id,
        title="Scientific Fleet",
        objective="Track and validate scientific opportunities",
        status="running",
        linked_profile_ids=[],
        automation_policy={},
        sandbox_profile_id="scientific-generic-sandbox",
        opportunities=[
            {
                "opportunity_id": "opp_portfolio_validate",
                "canonical_key": "portfolio_validation",
                "title": "Portfolio validation",
                "hypothesis": "Launch a bounded validation run",
                "decision_state": "pending_review",
                "stage": "discovered",
                "source_note_ids": [],
            }
        ],
        latest_summary={},
        latest_note_ids=[],
        latest_experiment_plan_ids=[],
        latest_validation_run_ids=[],
        child_job_ids=[],
    )
    created_runs: list[str] = []

    async def _seed():
        db_session.add(note)
        db_session.add(parent_job)
        await db_session.flush()
        portfolio.latest_note_ids = [str(note.id)]
        portfolio.latest_run_job_id = parent_job.id
        db_session.add(portfolio)
        await db_session.commit()

    async def _fake_create_validation_run(self, **kwargs):
        created_runs.append(str(kwargs.get("hypothesis_id") or ""))
        return {"run_id": "run-portfolio-1", "status": "queued", "job_id": "job-portfolio-1"}

    monkeypatch.setattr(
        "app.services.autonomous_agent_executor.AutonomousAgentExecutor._create_scientific_validation_run",
        _fake_create_validation_run,
    )

    asyncio.get_event_loop().run_until_complete(_seed())

    first = scientific_validation_summary_client.post(
        f"/api/v1/research-portfolios/{portfolio.id}/opportunities/opp_portfolio_validate/action",
        json={"action": "launch_validation"},
    )
    assert first.status_code == 200
    first_opp = first.json()["opportunities"][0]
    assert first_opp["decision_state"] == "accepted"
    assert first_opp["stage"] == "validating"
    assert first_opp["linked_validation_run_ids"] == ["run-portfolio-1"]
    assert len(first_opp["linked_experiment_plan_ids"]) == 1

    second = scientific_validation_summary_client.post(
        f"/api/v1/research-portfolios/{portfolio.id}/opportunities/opp_portfolio_validate/action",
        json={"action": "launch_validation"},
    )
    assert second.status_code == 200
    second_opp = second.json()["opportunities"][0]
    assert second_opp["linked_validation_run_ids"] == ["run-portfolio-1"]
    assert second_opp["linked_experiment_plan_ids"] == first_opp["linked_experiment_plan_ids"]
    assert created_runs == ["opp_portfolio_validate"]


def test_domain_profile_opportunity_actions_emit_explicit_reason_labels(
    scientific_validation_summary_client,
    db_session,
    test_user,
    monkeypatch,
):
    parent_job = AgentJob(
        user_id=test_user.id,
        name="Profile Parent Job",
        goal="Track compiler opportunities",
        job_type="research",
        status=AgentJobStatus.COMPLETED.value,
        progress=100,
        config={},
        results={
            "execution_strategy": {
                "scheduler_state": {
                    "queue_reason": "execution_failure",
                    "last_scheduled_at": "2026-03-16T09:00:00Z",
                    "last_dispatched_at": "2026-03-16T09:05:00Z",
                }
            }
        },
    )
    profile = DomainResearchProfile(
        user_id=test_user.id,
        title="Compiler Frontier",
        domain="Compiler",
        objective="Track compiler opportunities",
        status="running",
        source_scope="kb_plus_arxiv_plus_repo",
        track_type="compiler",
        research_mode="literature_to_hypothesis",
        report_format="brief_and_report",
        interval_minutes=1440,
        persist_artifacts=True,
        auto_launch_follow_up=True,
        auto_create_experiment_plans=True,
        confidence_threshold=0.7,
        max_documents=10,
        max_papers=8,
        latest_note_ids=[],
        latest_run_job_id=parent_job.id,
        latest_summary={
            "opportunities": [
                {
                    "opportunity_id": "opp_validate_label",
                    "canonical_key": "compiler_validation",
                    "title": "Compiler validation",
                    "hypothesis": "Reuse an existing validation run",
                    "decision_state": "pending_review",
                    "stage": "discovered",
                    "linked_validation_run_ids": ["run-existing"],
                },
                {
                    "opportunity_id": "opp_follow_up_label",
                    "canonical_key": "compiler_follow_up",
                    "title": "Compiler follow-up",
                    "hypothesis": "Reuse an existing follow-up job",
                    "decision_state": "pending_review",
                    "stage": "discovered",
                    "child_job_ids": ["child-existing"],
                },
                {
                    "opportunity_id": "opp_accept_label",
                    "canonical_key": "compiler_accept",
                    "title": "Compiler accepted idea",
                    "hypothesis": "Accept this opportunity",
                    "decision_state": "pending_review",
                    "stage": "discovered",
                },
            ]
        },
    )
    captured_events: list[dict[str, object]] = []

    async def _seed():
        db_session.add(parent_job)
        await db_session.flush()
        profile.latest_run_job_id = parent_job.id
        db_session.add(profile)
        await db_session.commit()

    async def _capture_event(*args, **kwargs):
        captured_events.append(
            {
                "event_type": kwargs.get("event_type"),
                "reason_label": kwargs.get("reason_label"),
                "decision_type": kwargs.get("decision_type"),
                "scheduler_state": kwargs.get("scheduler_state"),
            }
        )
        return SimpleNamespace(id="event-1")

    monkeypatch.setattr(
        "app.api.endpoints.domain_research_profiles.record_autonomy_decision_event",
        _capture_event,
    )

    asyncio.get_event_loop().run_until_complete(_seed())

    validation_response = scientific_validation_summary_client.post(
        f"/api/v1/domain-research-profiles/{profile.id}/opportunities/opp_validate_label/action",
        json={"action": "launch_validation"},
    )
    follow_up_response = scientific_validation_summary_client.post(
        f"/api/v1/domain-research-profiles/{profile.id}/opportunities/opp_follow_up_label/action",
        json={"action": "launch_follow_up"},
    )
    accept_response = scientific_validation_summary_client.post(
        f"/api/v1/domain-research-profiles/{profile.id}/opportunities/opp_accept_label/action",
        json={"action": "accept"},
    )

    assert validation_response.status_code == 200
    assert follow_up_response.status_code == 200
    assert accept_response.status_code == 200
    assert captured_events == [
        {
            "event_type": "validation_requeued",
            "reason_label": "Validation requeued",
            "decision_type": "validation_requeued",
            "scheduler_state": {
                "last_run_status": None,
                "failure_streak": 0,
                "queue_reason": "execution_failure",
                "last_scheduled_at": "2026-03-16T09:00:00Z",
                "last_dispatched_at": "2026-03-16T09:05:00Z",
                "current_run_started_at": None,
                "last_successful_run_at": None,
                "last_completed_run_at": None,
                "last_failure_at": None,
                "backoff_until": None,
                "backoff_seconds": 0,
            },
        },
        {
            "event_type": "follow_up_launched",
            "reason_label": "Follow-up launched",
            "decision_type": "follow_up_launched",
            "scheduler_state": {
                "last_run_status": None,
                "failure_streak": 0,
                "queue_reason": "execution_failure",
                "last_scheduled_at": "2026-03-16T09:00:00Z",
                "last_dispatched_at": "2026-03-16T09:05:00Z",
                "current_run_started_at": None,
                "last_successful_run_at": None,
                "last_completed_run_at": None,
                "last_failure_at": None,
                "backoff_until": None,
                "backoff_seconds": 0,
            },
        },
        {
            "event_type": "opportunity_accepted",
            "reason_label": "Opportunity accepted",
            "decision_type": "opportunity_accepted",
            "scheduler_state": {
                "last_run_status": None,
                "failure_streak": 0,
                "queue_reason": "execution_failure",
                "last_scheduled_at": "2026-03-16T09:00:00Z",
                "last_dispatched_at": "2026-03-16T09:05:00Z",
                "current_run_started_at": None,
                "last_successful_run_at": None,
                "last_completed_run_at": None,
                "last_failure_at": None,
                "backoff_until": None,
                "backoff_seconds": 0,
            },
        },
    ]


def test_research_portfolio_opportunities_emit_scheduler_state_in_trace_events(
    scientific_validation_summary_client,
    db_session,
    test_user,
    monkeypatch,
):
    parent_job = AgentJob(
        user_id=test_user.id,
        name="Portfolio Parent Job",
        goal="Track scientific opportunities",
        job_type="research",
        status=AgentJobStatus.COMPLETED.value,
        progress=100,
        config={},
        results={
            "execution_strategy": {
                "scheduler_state": {
                    "queue_reason": "scheduled_recovery",
                    "last_scheduled_at": "2026-03-16T10:00:00Z",
                    "last_dispatched_at": "2026-03-16T10:05:00Z",
                }
            }
        },
    )
    portfolio = ResearchPortfolio(
        user_id=test_user.id,
        title="Scientific Fleet",
        objective="Track and validate scientific opportunities",
        status="running",
        linked_profile_ids=[],
        automation_policy={},
        sandbox_profile_id="scientific-generic-sandbox",
        opportunities=[
            {
                "opportunity_id": "opp_portfolio_validate",
                "canonical_key": "portfolio_validation",
                "title": "Portfolio validation",
                "hypothesis": "Reuse an existing validation run",
                "decision_state": "pending_review",
                "stage": "discovered",
                "linked_validation_run_ids": ["run-existing"],
            },
            {
                "opportunity_id": "opp_portfolio_follow_up",
                "canonical_key": "portfolio_follow_up",
                "title": "Portfolio follow-up",
                "hypothesis": "Reuse an existing follow-up job",
                "decision_state": "pending_review",
                "stage": "discovered",
                "child_job_ids": ["child-existing"],
            },
        ],
        latest_summary={},
        latest_note_ids=[],
        latest_experiment_plan_ids=[],
        latest_validation_run_ids=[],
        child_job_ids=[],
        latest_run_job_id=parent_job.id,
    )
    captured_events: list[dict[str, object]] = []

    async def _seed():
        db_session.add(parent_job)
        await db_session.flush()
        portfolio.latest_run_job_id = parent_job.id
        db_session.add(portfolio)
        await db_session.commit()

    async def _capture_event(*args, **kwargs):
        captured_events.append(
            {
                "event_type": kwargs.get("event_type"),
                "reason_label": kwargs.get("reason_label"),
                "decision_type": kwargs.get("decision_type"),
                "scheduler_state": kwargs.get("scheduler_state"),
            }
        )
        return SimpleNamespace(id="event-2")

    monkeypatch.setattr(research_portfolios, "record_autonomy_decision_event", _capture_event)

    asyncio.get_event_loop().run_until_complete(_seed())

    validation_response = scientific_validation_summary_client.post(
        f"/api/v1/research-portfolios/{portfolio.id}/opportunities/opp_portfolio_validate/action",
        json={"action": "launch_validation"},
    )
    follow_up_response = scientific_validation_summary_client.post(
        f"/api/v1/research-portfolios/{portfolio.id}/opportunities/opp_portfolio_follow_up/action",
        json={"action": "launch_follow_up"},
    )

    assert validation_response.status_code == 200
    assert follow_up_response.status_code == 200
    assert captured_events == [
        {
            "event_type": "validation_requeued",
            "reason_label": "Validation requeued",
            "decision_type": "validation_requeued",
            "scheduler_state": {
                "last_run_status": None,
                "failure_streak": 0,
                "queue_reason": "scheduled_recovery",
                "last_scheduled_at": "2026-03-16T10:00:00Z",
                "last_dispatched_at": "2026-03-16T10:05:00Z",
                "current_run_started_at": None,
                "last_successful_run_at": None,
                "last_completed_run_at": None,
                "last_failure_at": None,
                "backoff_until": None,
                "backoff_seconds": 0,
            },
        },
        {
            "event_type": "follow_up_launched",
            "reason_label": "Follow-up launched",
            "decision_type": "follow_up_launched",
            "scheduler_state": {
                "last_run_status": None,
                "failure_streak": 0,
                "queue_reason": "scheduled_recovery",
                "last_scheduled_at": "2026-03-16T10:00:00Z",
                "last_dispatched_at": "2026-03-16T10:05:00Z",
                "current_run_started_at": None,
                "last_successful_run_at": None,
                "last_completed_run_at": None,
                "last_failure_at": None,
                "backoff_until": None,
                "backoff_seconds": 0,
            },
        },
    ]


def test_domain_profile_validation_requeue_omits_malformed_scheduler_state(
    scientific_validation_summary_client,
    db_session,
    test_user,
    monkeypatch,
):
    parent_job = AgentJob(
        user_id=test_user.id,
        name="Domain Parent Job",
        goal="Track compiler opportunities",
        job_type="monitor",
        status="running",
        results={"execution_strategy": {"scheduler_state": "bad-payload"}},
    )
    profile = DomainResearchProfile(
        user_id=test_user.id,
        title="Compiler Frontier",
        domain="Compiler",
        objective="Track compiler opportunities",
        status="running",
        source_scope="kb_plus_arxiv_plus_repo",
        track_type="compiler",
        research_mode="literature_to_hypothesis",
        report_format="brief_and_report",
        interval_minutes=1440,
        persist_artifacts=True,
        auto_launch_follow_up=True,
        auto_create_experiment_plans=True,
        confidence_threshold=0.7,
        max_documents=10,
        max_papers=8,
        latest_run_job_id=parent_job.id,
        latest_summary={
            "opportunities": [
                {
                    "opportunity_id": "opp_validate",
                    "canonical_key": "compiler_validation",
                    "title": "Compiler validation",
                    "hypothesis": "Reuse an existing validation run",
                    "decision_state": "pending_review",
                    "stage": "discovered",
                    "linked_validation_run_ids": ["run-existing"],
                }
            ]
        },
    )
    captured: list[dict[str, object]] = []

    async def _seed():
        db_session.add(parent_job)
        await db_session.flush()
        profile.latest_run_job_id = parent_job.id
        db_session.add(profile)
        await db_session.commit()

    async def _capture_event(*args, **kwargs):
        captured.append({"event_type": kwargs.get("event_type"), "scheduler_state": kwargs.get("scheduler_state")})
        return SimpleNamespace(id="event-3")

    monkeypatch.setattr(domain_research_profiles, "record_autonomy_decision_event", _capture_event)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = scientific_validation_summary_client.post(
        f"/api/v1/domain-research-profiles/{profile.id}/opportunities/opp_validate/action",
        json={"action": "launch_validation"},
    )

    assert response.status_code == 200
    assert captured == [{"event_type": "validation_requeued", "scheduler_state": None}]


def test_domain_profile_opportunity_launch_follow_up_is_idempotent(
    scientific_validation_summary_client,
    db_session,
    test_user,
    monkeypatch,
):
    parent_job = AgentJob(
        user_id=test_user.id,
        name="Profile Follow-up Parent Job",
        goal="Track compiler opportunities",
        job_type="research",
        status=AgentJobStatus.COMPLETED.value,
        progress=100,
        config={},
    )
    profile = DomainResearchProfile(
        user_id=test_user.id,
        title="Compiler Frontier",
        domain="Compiler",
        objective="Track compiler opportunities",
        status="running",
        source_scope="kb_plus_arxiv_plus_repo",
        track_type="compiler",
        research_mode="literature_to_hypothesis",
        report_format="brief_and_report",
        interval_minutes=1440,
        persist_artifacts=True,
        auto_launch_follow_up=True,
        auto_create_experiment_plans=True,
        confidence_threshold=0.7,
        max_documents=10,
        max_papers=8,
        latest_summary={
            "opportunities": [
                {
                    "opportunity_id": "opp_follow_up",
                    "canonical_key": "compiler_follow_up",
                    "title": "Compiler follow-up",
                    "hypothesis": "Launch a deeper research pass",
                    "decision_state": "pending_review",
                    "stage": "discovered",
                }
            ]
        },
    )
    created_jobs: list[str] = []
    queued_jobs: list[tuple[str, str]] = []

    async def _seed():
        db_session.add(parent_job)
        await db_session.flush()
        profile.latest_run_job_id = parent_job.id
        db_session.add(profile)
        await db_session.commit()

    async def _fake_create_follow_up(self, **kwargs):
        created_jobs.append(str(kwargs.get("top_idea", {}).get("opportunity_id") or ""))
        return SimpleNamespace(id="child-profile-1")

    monkeypatch.setattr(
        "app.services.autonomous_agent_executor.AutonomousAgentExecutor._create_domain_research_follow_up_job",
        _fake_create_follow_up,
    )
    monkeypatch.setattr(
        "app.api.endpoints.domain_research_profiles.execute_agent_job_task.delay",
        lambda job_id, user_id: queued_jobs.append((job_id, user_id)),
    )

    asyncio.get_event_loop().run_until_complete(_seed())

    first = scientific_validation_summary_client.post(
        f"/api/v1/domain-research-profiles/{profile.id}/opportunities/opp_follow_up/action",
        json={"action": "launch_follow_up"},
    )
    assert first.status_code == 200
    first_opp = first.json()["opportunities"][0]
    assert first_opp["decision_state"] == "accepted"
    assert first_opp["stage"] == "validating"
    assert first_opp["child_job_ids"] == ["child-profile-1"]
    assert queued_jobs == [("child-profile-1", str(test_user.id))]

    second = scientific_validation_summary_client.post(
        f"/api/v1/domain-research-profiles/{profile.id}/opportunities/opp_follow_up/action",
        json={"action": "launch_follow_up"},
    )
    assert second.status_code == 200
    second_opp = second.json()["opportunities"][0]
    assert second_opp["child_job_ids"] == ["child-profile-1"]
    assert created_jobs == ["opp_follow_up"]


def test_research_portfolio_opportunity_launch_follow_up_is_idempotent(
    scientific_validation_summary_client,
    db_session,
    test_user,
    monkeypatch,
):
    parent_job = AgentJob(
        user_id=test_user.id,
        name="Portfolio Follow-up Parent Job",
        goal="Track scientific opportunities",
        job_type="research",
        status=AgentJobStatus.COMPLETED.value,
        progress=100,
        config={},
    )
    portfolio = ResearchPortfolio(
        user_id=test_user.id,
        title="Scientific Fleet",
        objective="Track and validate scientific opportunities",
        status="running",
        linked_profile_ids=[],
        automation_policy={},
        sandbox_profile_id="scientific-generic-sandbox",
        opportunities=[
            {
                "opportunity_id": "opp_portfolio_follow_up",
                "canonical_key": "portfolio_follow_up",
                "title": "Portfolio follow-up",
                "hypothesis": "Launch a deeper portfolio research pass",
                "decision_state": "pending_review",
                "stage": "discovered",
            }
        ],
        latest_summary={},
        latest_note_ids=[],
        latest_experiment_plan_ids=[],
        latest_validation_run_ids=[],
        child_job_ids=[],
    )
    created_jobs: list[str] = []
    queued_jobs: list[tuple[str, str]] = []

    async def _seed():
        db_session.add(parent_job)
        await db_session.flush()
        portfolio.latest_run_job_id = parent_job.id
        db_session.add(portfolio)
        await db_session.commit()

    async def _fake_create_follow_up(self, **kwargs):
        created_jobs.append(str(kwargs.get("top_idea", {}).get("opportunity_id") or ""))
        return SimpleNamespace(id="child-portfolio-1")

    monkeypatch.setattr(
        "app.services.autonomous_agent_executor.AutonomousAgentExecutor._create_domain_research_follow_up_job",
        _fake_create_follow_up,
    )
    monkeypatch.setattr(
        "app.api.endpoints.research_portfolios.execute_agent_job_task.delay",
        lambda job_id, user_id: queued_jobs.append((job_id, user_id)),
    )

    asyncio.get_event_loop().run_until_complete(_seed())

    first = scientific_validation_summary_client.post(
        f"/api/v1/research-portfolios/{portfolio.id}/opportunities/opp_portfolio_follow_up/action",
        json={"action": "launch_follow_up"},
    )
    assert first.status_code == 200
    first_opp = first.json()["opportunities"][0]
    assert first_opp["decision_state"] == "accepted"
    assert first_opp["stage"] == "validating"
    assert first_opp["child_job_ids"] == ["child-portfolio-1"]
    assert queued_jobs == [("child-portfolio-1", str(test_user.id))]

    second = scientific_validation_summary_client.post(
        f"/api/v1/research-portfolios/{portfolio.id}/opportunities/opp_portfolio_follow_up/action",
        json={"action": "launch_follow_up"},
    )
    assert second.status_code == 200
    second_opp = second.json()["opportunities"][0]
    assert second_opp["child_job_ids"] == ["child-portfolio-1"]
    assert created_jobs == ["opp_portfolio_follow_up"]


def test_domain_profile_opportunity_relaunch_follow_up_reuses_inbox_flow(
    scientific_validation_summary_client,
    db_session,
    test_user,
    monkeypatch,
):
    parent_job = AgentJob(
        user_id=test_user.id,
        name="Profile Relaunch Parent Job",
        goal="Track compiler opportunities",
        job_type="research",
        status=AgentJobStatus.COMPLETED.value,
        progress=100,
        config={},
    )
    profile = DomainResearchProfile(
        user_id=test_user.id,
        title="Compiler Frontier",
        domain="Compiler",
        objective="Track compiler opportunities",
        status="running",
        source_scope="kb_plus_arxiv_plus_repo",
        track_type="compiler",
        research_mode="literature_to_hypothesis",
        report_format="brief_and_report",
        interval_minutes=1440,
        persist_artifacts=True,
        auto_launch_follow_up=True,
        auto_create_experiment_plans=True,
        confidence_threshold=0.7,
        max_documents=10,
        max_papers=8,
        latest_summary={
            "opportunities": [
                {
                    "opportunity_id": "opp_relaunch",
                    "canonical_key": "compiler_follow_up_relaunch",
                    "title": "Compiler follow-up relaunch",
                    "decision_state": "accepted",
                    "stage": "accepted",
                    "follow_up_outcome_status": "failed",
                    "follow_up_last_job_id": None,
                }
            ]
        },
    )
    inbox_item = ResearchInboxItem(
        user_id=test_user.id,
        item_type="document",
        item_key="domain-relaunch-item",
        title="Compiler relaunch source",
        status="accepted",
        follow_up_launch_status="launched",
        follow_up_outcome_status="failed",
    )

    async def _seed():
        db_session.add(parent_job)
        await db_session.flush()
        profile.latest_run_job_id = parent_job.id
        db_session.add(profile)
        await db_session.flush()
        inbox_item.follow_up_job_id = parent_job.id
        profile.latest_summary["opportunities"][0]["follow_up_last_job_id"] = str(parent_job.id)
        db_session.add(inbox_item)
        await db_session.commit()

    captured: dict[str, object] = {}

    async def _fake_relaunch(*, item, operator_note, db, current_user):
        captured["item_id"] = str(item.id)
        captured["operator_note"] = operator_note
        item.follow_up_job_id = UUID("00000000-0000-0000-0000-000000000111")
        row = profile.latest_summary["opportunities"][0]
        row["follow_up_outcome_status"] = None
        row["follow_up_last_job_id"] = str(item.follow_up_job_id)
        row["follow_up_launched_at"] = "2026-03-27T12:00:00Z"
        row["stage"] = "accepted"
        row["last_decision_type"] = "follow_up_launched"
        row["last_decision_reason_code"] = "follow_up_relaunched"
        return SimpleNamespace(follow_up_job_id=item.follow_up_job_id)

    monkeypatch.setattr(domain_research_profiles, "_relaunch_follow_up_inbox_item", _fake_relaunch)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = scientific_validation_summary_client.post(
        f"/api/v1/domain-research-profiles/{profile.id}/opportunities/opp_relaunch/action",
        json={"action": "relaunch_follow_up", "operator_note": "Retry now"},
    )

    assert response.status_code == 200
    payload_opp = response.json()["opportunities"][0]
    assert captured["item_id"] == str(inbox_item.id)
    assert captured["operator_note"] == "Retry now"
    assert payload_opp["follow_up_outcome_status"] is None
    assert payload_opp["follow_up_last_job_id"] == "00000000-0000-0000-0000-000000000111"
    assert payload_opp["last_decision_reason_code"] == "follow_up_relaunched"


def test_research_portfolio_opportunity_relaunch_follow_up_reuses_inbox_flow(
    scientific_validation_summary_client,
    db_session,
    test_user,
    monkeypatch,
):
    parent_job = AgentJob(
        user_id=test_user.id,
        name="Portfolio Relaunch Parent Job",
        goal="Track scientific opportunities",
        job_type="research",
        status=AgentJobStatus.COMPLETED.value,
        progress=100,
        config={},
    )
    portfolio = ResearchPortfolio(
        user_id=test_user.id,
        title="Scientific Fleet",
        objective="Track and validate scientific opportunities",
        status="running",
        linked_profile_ids=[],
        automation_policy={},
        sandbox_profile_id="scientific-generic-sandbox",
        opportunities=[
            {
                "opportunity_id": "opp_portfolio_relaunch",
                "canonical_key": "portfolio_follow_up_relaunch",
                "title": "Portfolio follow-up relaunch",
                "decision_state": "accepted",
                "stage": "accepted",
                "follow_up_outcome_status": "cancelled",
                "follow_up_last_job_id": None,
            }
        ],
        latest_summary={},
        latest_note_ids=[],
        latest_experiment_plan_ids=[],
        latest_validation_run_ids=[],
        child_job_ids=[],
    )
    inbox_item = ResearchInboxItem(
        user_id=test_user.id,
        item_type="document",
        item_key="portfolio-relaunch-item",
        title="Portfolio relaunch source",
        status="accepted",
        follow_up_launch_status="launched",
        follow_up_outcome_status="cancelled",
    )

    async def _seed():
        db_session.add(parent_job)
        await db_session.flush()
        portfolio.latest_run_job_id = parent_job.id
        db_session.add(portfolio)
        await db_session.flush()
        inbox_item.follow_up_job_id = parent_job.id
        portfolio.opportunities[0]["follow_up_last_job_id"] = str(parent_job.id)
        db_session.add(inbox_item)
        await db_session.commit()

    captured: dict[str, object] = {}

    async def _fake_relaunch(*, item, operator_note, db, current_user):
        captured["item_id"] = str(item.id)
        captured["operator_note"] = operator_note
        item.follow_up_job_id = UUID("00000000-0000-0000-0000-000000000222")
        row = portfolio.opportunities[0]
        row["follow_up_outcome_status"] = None
        row["follow_up_last_job_id"] = str(item.follow_up_job_id)
        row["follow_up_launched_at"] = "2026-03-27T12:00:00Z"
        row["stage"] = "accepted"
        row["last_decision_type"] = "follow_up_launched"
        row["last_decision_reason_code"] = "follow_up_relaunched"
        return SimpleNamespace(follow_up_job_id=item.follow_up_job_id)

    monkeypatch.setattr(research_portfolios, "_relaunch_follow_up_inbox_item", _fake_relaunch)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = scientific_validation_summary_client.post(
        f"/api/v1/research-portfolios/{portfolio.id}/opportunities/opp_portfolio_relaunch/action",
        json={"action": "relaunch_follow_up", "operator_note": "Retry fleet follow-up"},
    )

    assert response.status_code == 200
    payload_opp = response.json()["opportunities"][0]
    assert captured["item_id"] == str(inbox_item.id)
    assert captured["operator_note"] == "Retry fleet follow-up"
    assert payload_opp["follow_up_outcome_status"] is None
    assert payload_opp["follow_up_last_job_id"] == "00000000-0000-0000-0000-000000000222"
    assert payload_opp["last_decision_reason_code"] == "follow_up_relaunched"


def test_domain_profile_opportunity_reopen_preserves_linked_artifacts(
    scientific_validation_summary_client,
    db_session,
    test_user,
):
    profile = DomainResearchProfile(
        user_id=test_user.id,
        title="Compiler Frontier",
        domain="Compiler",
        objective="Track compiler opportunities",
        status="running",
        source_scope="kb_plus_arxiv_plus_repo",
        track_type="compiler",
        research_mode="literature_to_hypothesis",
        report_format="brief_and_report",
        interval_minutes=1440,
        persist_artifacts=True,
        auto_launch_follow_up=True,
        auto_create_experiment_plans=True,
        confidence_threshold=0.7,
        max_documents=10,
        max_papers=8,
        latest_experiment_plan_ids=["plan-profile-1"],
        latest_summary={
            "opportunities": [
                {
                    "opportunity_id": "opp_profile_reopen",
                    "canonical_key": "profile_reopen",
                    "title": "Profile reopen",
                    "hypothesis": "Preserve linked artifacts",
                    "decision_state": "suppressed",
                    "decision_source": "operator",
                    "operator_note": "Need more evidence",
                    "stage": "suppressed",
                    "linked_experiment_plan_ids": ["plan-profile-1"],
                }
            ]
        },
    )

    async def _seed():
        db_session.add(profile)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = scientific_validation_summary_client.post(
        f"/api/v1/domain-research-profiles/{profile.id}/opportunities/opp_profile_reopen/action",
        json={"action": "reopen"},
    )
    assert response.status_code == 200
    opp = response.json()["opportunities"][0]
    assert opp["decision_state"] == "pending_review"
    assert opp["linked_experiment_plan_ids"] == ["plan-profile-1"]
    assert opp["stage"] == "planned"


def test_research_portfolio_opportunity_reopen_preserves_linked_artifacts(
    scientific_validation_summary_client,
    db_session,
    test_user,
):
    portfolio = ResearchPortfolio(
        user_id=test_user.id,
        title="Scientific Fleet",
        objective="Track and validate scientific opportunities",
        status="running",
        linked_profile_ids=[],
        automation_policy={},
        sandbox_profile_id="scientific-generic-sandbox",
        opportunities=[
            {
                "opportunity_id": "opp_reopen",
                "canonical_key": "portfolio_reopen",
                "title": "Portfolio reopen",
                "hypothesis": "Preserve linked artifacts",
                "decision_state": "suppressed",
                "decision_source": "operator",
                "operator_note": "Not ready yet",
                "stage": "suppressed",
                "linked_experiment_plan_ids": ["plan-portfolio-1"],
            }
        ],
        latest_summary={},
        latest_note_ids=[],
        latest_experiment_plan_ids=["plan-portfolio-1"],
        latest_validation_run_ids=[],
        child_job_ids=[],
    )

    async def _seed():
        db_session.add(portfolio)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = scientific_validation_summary_client.post(
        f"/api/v1/research-portfolios/{portfolio.id}/opportunities/opp_reopen/action",
        json={"action": "reopen"},
    )
    assert response.status_code == 200
    opp = response.json()["opportunities"][0]
    assert opp["decision_state"] == "pending_review"
    assert opp["linked_experiment_plan_ids"] == ["plan-portfolio-1"]
    assert opp["stage"] == "planned"


def test_profile_response_exposes_stage_counts_from_normalized_opportunities(
    scientific_validation_summary_client,
    db_session,
    test_user,
):
    profile = DomainResearchProfile(
        user_id=test_user.id,
        title="Compiler Frontier",
        domain="Compiler",
        objective="Track compiler opportunities",
        status="running",
        source_scope="kb_plus_arxiv_plus_repo",
        track_type="compiler",
        research_mode="literature_to_hypothesis",
        report_format="brief_and_report",
        interval_minutes=1440,
        persist_artifacts=True,
        auto_launch_follow_up=True,
        auto_create_experiment_plans=True,
        confidence_threshold=0.7,
        max_documents=10,
        max_papers=8,
        latest_summary={
            "opportunities": [
                {
                    "opportunity_id": "opp_a",
                    "canonical_key": "a",
                    "title": "Opportunity A",
                    "decision_state": "accepted",
                },
                {
                    "opportunity_id": "opp_b",
                    "canonical_key": "b",
                    "title": "Opportunity B",
                    "decision_state": "suppressed",
                },
            ]
        },
    )

    async def _seed():
        db_session.add(profile)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = scientific_validation_summary_client.get("/api/v1/domain-research-profiles")
    assert response.status_code == 200
    payload = response.json()["items"][0]
    assert payload["latest_summary"]["stage_counts"]["accepted"] == 1
    assert payload["latest_summary"]["stage_counts"]["suppressed"] == 1
