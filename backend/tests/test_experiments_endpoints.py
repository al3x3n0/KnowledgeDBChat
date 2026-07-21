import asyncio
import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.endpoints import experiments
from app.core.database import get_db
from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.domain_research_profile import DomainResearchProfile
from app.models.experiment import ExperimentPlan, ExperimentRun
from app.models.research_note import ResearchNote
from app.models.research_portfolio import ResearchPortfolio
from app.models.synthesis_job import SynthesisJob


@pytest.fixture
def experiment_client(db_session, test_user):
    app = FastAPI()
    app.include_router(experiments.router, prefix="/api/v1/experiments")

    def override_get_db():
        return db_session

    async def override_current_user():
        return test_user

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[experiments.get_current_active_user] = override_current_user

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_list_experiment_runs_exposes_typed_experiment_run_payload(
    experiment_client,
    db_session,
    test_user,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Bootstrap Validation",
        content_markdown="## Hypothesis\nThe bootstrap retry should recover missing toolchains.",
        tags=["agents"],
    )
    plan = ExperimentPlan(
        user_id=test_user.id,
        research_note_id=note.id,
        title="Bootstrap Validation Plan",
        hypothesis_text="Bootstrap retry should recover missing toolchains.",
        plan={"experiments": []},
    )
    run = ExperimentRun(
        user_id=test_user.id,
        experiment_plan_id=plan.id,
        name="Retry Run",
        status="completed",
        progress=100,
        config={
            "scientific_validation": {
                "validation_kind": "scientific_validation",
                "sandbox_profile_id": "scientific-compiler-sandbox",
                "recipe_family": "compiler_validation",
                "recipe_id": "compiler_validation_v1",
                "recipe_version": 1,
                "blocked_reason_code": None,
                "capability_check": {
                    "ok": True,
                    "required": ["repo_reconstruction"],
                    "satisfied": ["repo_reconstruction"],
                    "missing": [],
                },
                "profile_snapshot": {
                    "id": "scientific-compiler-sandbox",
                    "track_type": "compiler",
                },
                "recipe_snapshot": {
                    "recipe_id": "compiler_validation_v1",
                    "commands": ["CI=true npm --prefix frontend test -- --watchAll=false"],
                },
            }
        },
        results={
            "source_id": "repo-1",
            "final_phase": "retry_primary",
            "bootstrap_attempted": True,
            "bootstrap_ok": True,
            "verification_commands": ["CI=true npm --prefix frontend test -- --watchAll=false"],
            "execution_strategy": {
                "operator_interventions": [
                    {
                        "action": "restart",
                        "actor_user_id": "user-1",
                        "job_status_before": "failed",
                        "job_status_after": "pending",
                        "note": "Retry after fallback failure",
                    }
                ]
            },
        },
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        plan.research_note_id = note.id
        db_session.add(plan)
        await db_session.flush()
        run.experiment_plan_id = plan.id
        db_session.add(run)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = experiment_client.get(f"/api/v1/experiments/plans/{plan.id}/runs")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["runs"]) == 1
    typed = payload["runs"][0]["experiment_run"]
    assert typed["source_id"] == "repo-1"
    assert typed["final_phase"] == "retry_primary"
    assert typed["bootstrap_ok"] is True
    assert payload["runs"][0]["validation_kind"] == "scientific_validation"
    assert payload["runs"][0]["sandbox_profile_id"] == "scientific-compiler-sandbox"
    assert payload["runs"][0]["recipe_id"] == "compiler_validation_v1"
    assert payload["runs"][0]["recipe_version"] == 1
    assert payload["runs"][0]["capability_check"]["ok"] is True
    assert payload["runs"][0]["profile_snapshot"]["id"] == "scientific-compiler-sandbox"
    assert payload["runs"][0]["recipe_snapshot"]["recipe_id"] == "compiler_validation_v1"
    interventions = payload["runs"][0]["operator_interventions"]
    assert len(interventions) == 1
    assert interventions[0]["action"] == "restart"
    assert interventions[0]["job_status_before"] == "failed"
    assert interventions[0]["job_status_after"] == "pending"


def test_sync_experiment_run_from_job_projects_typed_payload(
    experiment_client,
    db_session,
    test_user,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Sync Validation",
        content_markdown="## Hypothesis\nSync should project typed experiment payload.",
        tags=["agents"],
    )
    plan = ExperimentPlan(
        user_id=test_user.id,
        research_note_id=note.id,
        title="Sync Validation Plan",
        hypothesis_text="Sync should project typed experiment payload.",
        plan={"experiments": []},
    )
    job = AgentJob(
        name="Experiment Runner Job",
        goal="Run experiment verification",
        job_type="analysis",
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
        progress=100,
        iteration=1,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=10,
        results={
            "experiment_run": {
                "source_id": "repo-2",
                "final_phase": "fallback",
                "fallback_attempted": True,
                "fallback_ok": False,
                "failed_commands": ["python -m pytest -q backend/tests"],
            },
            "execution_strategy": {
                "operator_interventions": [
                    {
                        "action": "pause",
                        "actor_user_id": "user-1",
                        "job_status_before": "running",
                        "job_status_after": "paused",
                        "note": "Inspect failing fallback output",
                    }
                ]
            },
        },
    )
    run = ExperimentRun(
        user_id=test_user.id,
        experiment_plan_id=plan.id,
        agent_job_id=job.id,
        name="Sync Run",
        status="running",
        progress=50,
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        plan.research_note_id = note.id
        db_session.add(plan)
        await db_session.flush()
        db_session.add(job)
        await db_session.flush()
        run.experiment_plan_id = plan.id
        run.agent_job_id = job.id
        db_session.add(run)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = experiment_client.post(f"/api/v1/experiments/runs/{run.id}/sync")

    assert response.status_code == 200
    payload = response.json()["run"]
    assert payload["status"] == "completed"
    assert payload["experiment_run"]["source_id"] == "repo-2"
    assert payload["experiment_run"]["final_phase"] == "fallback"
    assert payload["experiment_run"]["fallback_attempted"] is True
    assert payload["operator_interventions"] is not None
    assert len(payload["operator_interventions"]) == 1
    assert payload["operator_interventions"][0]["action"] == "pause"
    assert payload["operator_interventions"][0]["job_status_before"] == "running"
    assert payload["operator_interventions"][0]["job_status_after"] == "paused"


def test_sync_experiment_run_reconciles_domain_opportunity_outcome(
    experiment_client,
    db_session,
    test_user,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Compiler Validation",
        content_markdown="## Hypothesis\nValidation should close the loop.",
        tags=["agents"],
    )
    plan = ExperimentPlan(
        user_id=test_user.id,
        research_note_id=note.id,
        title="Compiler Validation Plan",
        hypothesis_text="Completed validation should mark the opportunity completed.",
        plan={"experiments": []},
    )
    job = AgentJob(
        name="Scientific Validation Job",
        goal="Run scientific validation",
        job_type="analysis",
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
        progress=100,
        results={"experiment_run": {"summary": "Benchmark regression reproduced and explained."}},
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
                    "opportunity_id": "opp-compiler-1",
                    "canonical_key": "compiler_regression",
                    "title": "Compiler regression",
                    "hypothesis": "Regression exists",
                    "decision_state": "accepted",
                    "stage": "validating",
                    "linked_experiment_plan_ids": [],
                    "linked_validation_run_ids": [],
                    "source_note_ids": [],
                }
            ]
        },
        latest_note_ids=[],
        latest_experiment_plan_ids=[],
        latest_validation_run_ids=[],
    )
    run = ExperimentRun(
        user_id=test_user.id,
        experiment_plan_id=plan.id,
        agent_job_id=job.id,
        name="Compiler Validation Run",
        status="running",
        progress=50,
        config={},
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        plan.research_note_id = note.id
        db_session.add(plan)
        db_session.add(job)
        db_session.add(profile)
        await db_session.flush()
        run.config = {
            "scientific_validation": {
                "domain_research_profile_id": str(profile.id),
                "hypothesis_id": "opp-compiler-1",
            },
            "execution_handoff": {
                "autonomous_origin": {
                    "source_kind": "profile",
                    "source_id": str(profile.id),
                    "opportunity_id": "opp-compiler-1",
                }
            },
            "post_run_actions": {
                "auto_append_to_note": True,
                "target_note_id": str(note.id),
                "append_status": "pending",
            },
        }
        run.experiment_plan_id = plan.id
        run.agent_job_id = job.id
        db_session.add(run)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = experiment_client.post(f"/api/v1/experiments/runs/{run.id}/sync")

    assert response.status_code == 200

    async def _assert():
        refreshed = await db_session.get(DomainResearchProfile, profile.id)
        row = refreshed.latest_summary["opportunities"][0]
        assert row["stage"] == "completed"
        assert row["autonomy_state"] == "completed_waiting_change"
        assert row["last_decision_reason_code"] == "completed_current_evidence"
        assert row["latest_validation_status"] == "completed"
        assert row["latest_validation_job_id"] == str(job.id)
        assert row["follow_up_outcome_status"] == "completed"
        assert row["follow_up_last_job_id"] == str(job.id)
        assert row["source_note_ids"] == [str(note.id)]
        assert str(note.id) in (refreshed.latest_note_ids or [])
        assert str(plan.id) in (refreshed.latest_experiment_plan_ids or [])
        assert str(run.id) in (refreshed.latest_validation_run_ids or [])

    asyncio.get_event_loop().run_until_complete(_assert())


def test_update_experiment_run_reconciles_blocked_portfolio_opportunity(
    experiment_client,
    db_session,
    test_user,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Fleet Validation",
        content_markdown="## Hypothesis\nBlocked validation should project to the opportunity.",
        tags=["agents"],
    )
    plan = ExperimentPlan(
        user_id=test_user.id,
        research_note_id=note.id,
        title="Fleet Validation Plan",
        hypothesis_text="Blocked validation should mark the fleet opportunity blocked.",
        plan={"experiments": []},
    )
    portfolio = ResearchPortfolio(
        user_id=test_user.id,
        title="Scientific Fleet",
        objective="Track and validate opportunities",
        status="running",
        linked_profile_ids=[],
        automation_policy={},
        opportunities=[
            {
                "opportunity_id": "opp-fleet-1",
                "canonical_key": "blocked_validation",
                "title": "Blocked validation",
                "hypothesis": "Sandbox policy blocks execution",
                "decision_state": "accepted",
                "stage": "validating",
                "linked_experiment_plan_ids": [],
                "linked_validation_run_ids": [],
            }
        ],
        latest_summary={},
        latest_note_ids=[],
        latest_experiment_plan_ids=[],
        latest_validation_run_ids=[],
        child_job_ids=[],
    )
    run = ExperimentRun(
        user_id=test_user.id,
        experiment_plan_id=plan.id,
        name="Blocked Fleet Run",
        status="planned",
        progress=0,
        config={},
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        plan.research_note_id = note.id
        db_session.add(plan)
        db_session.add(portfolio)
        await db_session.flush()
        run.config = {
            "scientific_validation": {
                "research_portfolio_id": str(portfolio.id),
                "hypothesis_id": "opp-fleet-1",
                "blocked_reason_code": "sandbox_policy_rejected",
            },
            "execution_handoff": {
                "autonomous_origin": {
                    "source_kind": "portfolio",
                    "source_id": str(portfolio.id),
                    "opportunity_id": "opp-fleet-1",
                }
            },
        }
        run.experiment_plan_id = plan.id
        db_session.add(run)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = experiment_client.patch(
        f"/api/v1/experiments/runs/{run.id}",
        json={"status": "blocked"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "blocked"

    async def _assert():
        refreshed = await db_session.get(ResearchPortfolio, portfolio.id)
        row = refreshed.opportunities[0]
        assert row["stage"] == "blocked"
        assert row["autonomy_state"] == "blocked_structural"
        assert row["latest_validation_status"] == "blocked"
        assert row["latest_validation_blocked_reason_code"] == "sandbox_policy_rejected"
        assert row["follow_up_outcome_status"] == "blocked"
        assert row["last_decision_reason_code"] == "sandbox_policy_rejected"
        assert row["last_blocked_reason_code"] == "sandbox_policy_rejected"
        assert str(note.id) in (refreshed.latest_note_ids or [])
        assert str(plan.id) in (refreshed.latest_experiment_plan_ids or [])
        assert str(run.id) in (refreshed.latest_validation_run_ids or [])

    asyncio.get_event_loop().run_until_complete(_assert())


def test_sync_experiment_run_auto_appends_to_target_note_when_configured(
    experiment_client,
    db_session,
    test_user,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Auto Append Validation",
        content_markdown="## Hypothesis\nAuto append run evidence on completion.",
        structured_payload={
            "research_mode": "paper_to_hypothesis",
            "hypotheses": [
                {
                    "id": "hyp-1",
                    "rank": 1,
                    "title": "Auto append hypothesis",
                    "claim": "Completed runs should land on the note automatically.",
                    "rationale": "Direct launch should close the loop.",
                    "novelty_score": 0.7,
                    "evidence_score": 0.5,
                    "testability_score": 0.8,
                    "overall_score": 0.68,
                    "recommended_next_step": "Run the validation loop.",
                }
            ],
        },
    )
    plan = ExperimentPlan(
        user_id=test_user.id,
        research_note_id=note.id,
        title="Auto Append Plan",
        hypothesis_text="Auto append test.",
        plan={"experiments": []},
    )
    job = AgentJob(
        name="Auto Append Job",
        goal="Run auto append validation",
        job_type="analysis",
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
        progress=100,
        iteration=1,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=10,
        results={
            "experiment_run": {
                "source_id": "repo-auto",
                "final_phase": "retry_primary",
                "verification_commands": ["python -m pytest -q backend/tests/test_experiments_endpoints.py"],
            },
            "measurement_summary": {
                "compile_time_ms": 1204,
                "comparison": "improvement",
                "perf_counters": {"instructions": 950},
            },
            "compiler_artifacts": {
                "ir_paths": ["/tmp/instcombine_sroa.ll"],
                "diff_summary": "SROA fired earlier than baseline.",
                "pass_signals": ["instcombine", "sroa"],
            },
        },
    )
    run = ExperimentRun(
        user_id=test_user.id,
        experiment_plan_id=plan.id,
        agent_job_id=job.id,
        name="Auto Append Run",
        status="running",
        progress=50,
        config={
            "execution_handoff": {
                "plan_scope": "single_hypothesis",
                "selected_hypothesis_ids": ["hyp-1"],
                "source_paper_ids": ["paper-1"],
            },
            "post_run_actions": {
                "auto_append_to_note": True,
                "target_note_id": str(note.id),
                "append_status": "pending",
            },
        },
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        plan.research_note_id = note.id
        db_session.add(plan)
        await db_session.flush()
        db_session.add(job)
        await db_session.flush()
        run.experiment_plan_id = plan.id
        run.agent_job_id = job.id
        db_session.add(run)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = experiment_client.post(f"/api/v1/experiments/runs/{run.id}/sync")

    assert response.status_code == 200
    payload = response.json()["run"]
    assert payload["status"] == "completed"
    assert payload["config"]["post_run_actions"]["append_status"] == "completed"
    assert payload["config"]["post_run_actions"]["appended_at"]

    async def _verify():
        refreshed_note = await db_session.get(ResearchNote, note.id)
        assert refreshed_note is not None
        assert f"<!-- experiment_run:{run.id} -->" in (refreshed_note.content_markdown or "")
        hypotheses = refreshed_note.structured_payload["hypotheses"]
        assert len(hypotheses[0]["experiment_evidence"]) == 1
        assert hypotheses[0]["experiment_evidence"][0]["run_id"] == str(run.id)
        assert hypotheses[0]["experiment_evidence"][0]["artifact_diff_summary"] == "SROA fired earlier than baseline."
        assert hypotheses[0]["experiment_evidence"][0]["perf_counters"]["instructions"] == 950

    asyncio.get_event_loop().run_until_complete(_verify())

    second_response = experiment_client.post(f"/api/v1/experiments/runs/{run.id}/sync")
    assert second_response.status_code == 200

    async def _verify_idempotent():
        refreshed_note = await db_session.get(ResearchNote, note.id)
        assert refreshed_note is not None
        assert (refreshed_note.content_markdown or "").count(f"<!-- experiment_run:{run.id} -->") == 1
        hypotheses = refreshed_note.structured_payload["hypotheses"]
        assert len(hypotheses[0]["experiment_evidence"]) == 1

    asyncio.get_event_loop().run_until_complete(_verify_idempotent())


def test_sync_experiment_run_queues_pending_reevaluation_draft_for_reevaluated_note(
    experiment_client,
    db_session,
    test_user,
    monkeypatch,
):
    queued: list[tuple[str, str]] = []

    def _fake_delay(job_id: str, user_id: str):
        queued.append((job_id, user_id))

    monkeypatch.setattr(experiments.execute_synthesis_task, "delay", _fake_delay)

    note = ResearchNote(
        user_id=test_user.id,
        title="Queued Draft Validation",
        content_markdown="## Hypothesis\nQueue a reevaluation draft after evidence append.",
        structured_payload={
            "artifact_type": "hypothesis_reevaluation",
            "research_mode": "paper_to_hypothesis",
            "summary": "Reevaluated hypothesis note.",
            "hypotheses": [
                {
                    "id": "hyp-1",
                    "rank": 1,
                    "title": "Queued draft hypothesis",
                    "claim": "Evidence append should trigger draft reevaluation.",
                    "rationale": "The note should stay review-gated but up to date.",
                    "novelty_score": 0.7,
                    "evidence_score": 0.5,
                    "testability_score": 0.8,
                    "overall_score": 0.68,
                    "recommended_next_step": "Run the experiment.",
                }
            ],
        },
    )
    plan = ExperimentPlan(
        user_id=test_user.id,
        research_note_id=note.id,
        title="Queued Draft Plan",
        hypothesis_text="Queue draft test.",
        plan={"experiments": []},
    )
    job = AgentJob(
        name="Queued Draft Job",
        goal="Run queued draft validation",
        job_type="analysis",
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
        progress=100,
        iteration=1,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=10,
        results={
            "experiment_run": {
                "source_id": "repo-auto",
                "final_phase": "retry_primary",
                "verification_commands": ["python -m pytest -q backend/tests/test_experiments_endpoints.py"],
            }
        },
    )
    run = ExperimentRun(
        user_id=test_user.id,
        experiment_plan_id=plan.id,
        agent_job_id=job.id,
        name="Queued Draft Run",
        status="running",
        progress=50,
        config={
            "execution_handoff": {
                "plan_scope": "single_hypothesis",
                "selected_hypothesis_ids": ["hyp-1"],
            },
            "post_run_actions": {
                "auto_append_to_note": True,
                "target_note_id": str(note.id),
                "append_status": "pending",
            },
        },
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        plan.research_note_id = note.id
        db_session.add(plan)
        await db_session.flush()
        db_session.add(job)
        await db_session.flush()
        run.experiment_plan_id = plan.id
        run.agent_job_id = job.id
        db_session.add(run)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = experiment_client.post(f"/api/v1/experiments/runs/{run.id}/sync")

    assert response.status_code == 200
    assert queued

    async def _verify():
        refreshed_note = await db_session.get(ResearchNote, note.id)
        assert refreshed_note is not None
        pending_job_id = refreshed_note.structured_payload["pending_reevaluation_job_id"]
        assert pending_job_id
        assert refreshed_note.structured_payload["pending_reevaluation_reason"] == "new_experiment_evidence"
        assert refreshed_note.structured_payload["pending_reevaluation_source_run_ids"] == [str(run.id)]
        queued_job = await db_session.get(SynthesisJob, pending_job_id)
        assert queued_job is not None
        assert queued_job.job_type == "hypothesis_reevaluation"
        assert str(queued_job.research_note_id) == str(note.id)

    asyncio.get_event_loop().run_until_complete(_verify())

    second_response = experiment_client.post(f"/api/v1/experiments/runs/{run.id}/sync")
    assert second_response.status_code == 200
    assert len(queued) == 1


def test_sync_experiment_run_records_auto_append_failure_without_overwriting_run_status(
    experiment_client,
    db_session,
    test_user,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Auto Append Failure",
        content_markdown="## Hypothesis\nAppend failure should not change terminal run status.",
    )
    plan = ExperimentPlan(
        user_id=test_user.id,
        research_note_id=note.id,
        title="Failure Plan",
        hypothesis_text="Failure path.",
        plan={"experiments": []},
    )
    job = AgentJob(
        name="Auto Append Failure Job",
        goal="Run append failure validation",
        job_type="analysis",
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
        progress=100,
        iteration=1,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=10,
        results={"experiment_run": {"final_phase": "complete"}},
    )
    run = ExperimentRun(
        user_id=test_user.id,
        experiment_plan_id=plan.id,
        agent_job_id=job.id,
        name="Failure Run",
        status="running",
        progress=20,
        config={
            "post_run_actions": {
                "auto_append_to_note": True,
                "target_note_id": "00000000-0000-0000-0000-000000000099",
                "append_status": "pending",
            }
        },
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        plan.research_note_id = note.id
        db_session.add(plan)
        await db_session.flush()
        db_session.add(job)
        await db_session.flush()
        run.experiment_plan_id = plan.id
        run.agent_job_id = job.id
        db_session.add(run)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = experiment_client.post(f"/api/v1/experiments/runs/{run.id}/sync")

    assert response.status_code == 200
    payload = response.json()["run"]
    assert payload["status"] == "completed"
    assert payload["config"]["post_run_actions"]["append_status"] == "failed"
    assert "not found" in payload["config"]["post_run_actions"]["append_error"].lower()

    manual_append_response = experiment_client.post(f"/api/v1/experiments/runs/{run.id}/append-to-note")
    assert manual_append_response.status_code == 200

    async def _verify():
        refreshed_run = await db_session.get(ExperimentRun, run.id)
        refreshed_note = await db_session.get(ResearchNote, note.id)
        assert refreshed_run is not None
        assert refreshed_note is not None
        assert refreshed_run.config["post_run_actions"]["append_status"] == "completed"
        assert f"<!-- experiment_run:{run.id} -->" in (refreshed_note.content_markdown or "")

    asyncio.get_event_loop().run_until_complete(_verify())


def test_append_experiment_run_to_note_includes_bootstrap_summary(
    experiment_client,
    db_session,
    test_user,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Append Validation",
        content_markdown="## Hypothesis\nPersist experiment execution details.",
        tags=["agents"],
        structured_payload={
            "research_mode": "paper_to_hypothesis",
            "hypotheses": [
                {
                    "id": "hyp-1",
                    "rank": 1,
                    "title": "Bootstrap recovery",
                    "claim": "Bootstrap should recover missing envs.",
                    "rationale": "Most failures come from missing envs.",
                    "novelty_score": 0.7,
                    "evidence_score": 0.8,
                    "testability_score": 0.9,
                    "overall_score": 0.82,
                    "recommended_next_step": "Run backend suite.",
                }
            ],
        },
    )
    plan = ExperimentPlan(
        user_id=test_user.id,
        research_note_id=note.id,
        title="Append Validation Plan",
        hypothesis_text="Persist experiment execution details.",
        plan={"experiments": []},
    )
    run = ExperimentRun(
        user_id=test_user.id,
        experiment_plan_id=plan.id,
        name="Append Run",
        status="completed",
        progress=100,
        config={
            "execution_handoff": {
                "plan_scope": "single_hypothesis",
                "selected_hypothesis_ids": ["hyp-1"],
                "supporting_sources": [{"id": "paper-1", "title": "Bootstrap paper"}],
                "source_paper_ids": ["paper-1"],
                "source_document_ids": ["doc-1"],
            }
        },
        summary="Bootstrap recovered the environment, but fallback still failed on backend verification.",
        results={
            "ok": True,
            "source_id": "repo-3",
            "source_name": "Knowledge Repo",
            "final_phase": "retry_primary",
            "phases": ["primary", "bootstrap", "retry_primary"],
            "bootstrap_attempted": True,
            "bootstrap_ok": True,
            "fallback_attempted": True,
            "fallback_ok": False,
            "inferred_project_profile": {"detected_stack": ["node", "python"]},
            "verification_commands": ["CI=true npm --prefix frontend test -- --watchAll=false"],
            "bootstrap_commands": ["npm --prefix frontend install"],
            "fallback_commands": ["python3 -m pytest -q backend/tests"],
            "failed_commands": ["npm test"],
            "execution_strategy": {
                "operator_interventions": [
                    {
                        "action": "restart",
                        "job_status_before": "failed",
                        "job_status_after": "pending",
                        "note": "Retry after fallback failure",
                    }
                ],
                "execution_graph": {
                    "graph_health": {
                        "reasons": ["fallback verification still failing"],
                    },
                    "recommended_actions": ["Inspect failing fallback output"],
                }
            },
        },
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        plan.research_note_id = note.id
        db_session.add(plan)
        await db_session.flush()
        run.experiment_plan_id = plan.id
        db_session.add(run)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = experiment_client.post(f"/api/v1/experiments/runs/{run.id}/append-to-note")

    assert response.status_code == 200
    content = response.json()["content_markdown"]
    assert "Execution summary:" in content
    assert "Source: Knowledge Repo" in content
    assert "Source ID: `repo-3`" in content
    assert "Detected stack: node, python" in content
    assert "Summary:" in content
    assert "Bootstrap recovered the environment" in content
    assert "Hypothesis scope:" in content
    assert "Plan scope: single_hypothesis" in content
    assert "Selected hypotheses: hyp-1" in content
    assert "Supporting sources: Bootstrap paper" in content
    assert "Source papers: paper-1" in content
    assert "Final phase: `retry_primary`" in content
    assert "Bootstrap: ok" in content
    assert "Fallback: attempted" in content
    assert "Recovery: open" in content
    assert "Operator intervention:" in content
    assert "Latest: restart (failed -> pending)" in content
    assert "Outcome: resolved" in content
    assert "Outcome reason: Job completed after intervention" in content
    assert "Note: Retry after fallback failure" in content
    assert "Recovery guidance:" in content
    assert "Reason: fallback verification still failing" in content
    assert "Next: Inspect failing fallback output" in content
    assert "Verification commands:" in content
    assert "Bootstrap commands:" in content
    assert "Fallback verification commands:" in content
    assert "Failed commands:" in content
    hypotheses = response.json()["structured_payload"]["hypotheses"]
    assert len(hypotheses) == 1
    evidence = hypotheses[0]["experiment_evidence"]
    assert len(evidence) == 1
    assert evidence[0]["run_id"] == str(run.id)
    assert evidence[0]["plan_scope"] == "single_hypothesis"
    assert evidence[0]["source_paper_ids"] == ["paper-1"]
    assert evidence[0]["source_document_ids"] == ["doc-1"]
    assert evidence[0]["supporting_sources"][0]["title"] == "Bootstrap paper"

    second_response = experiment_client.post(f"/api/v1/experiments/runs/{run.id}/append-to-note")
    assert second_response.status_code == 200
    second_content = second_response.json()["content_markdown"]
    assert second_content.count(f"<!-- experiment_run:{run.id} -->") == 1
    second_hypotheses = second_response.json()["structured_payload"]["hypotheses"]
    assert len(second_hypotheses[0]["experiment_evidence"]) == 1


def test_append_aggregate_experiment_run_updates_each_selected_hypothesis(
    experiment_client,
    db_session,
    test_user,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Aggregate Append Validation",
        content_markdown="## Hypotheses\nAggregate experiment evidence should be linked.",
        structured_payload={
            "research_mode": "paper_to_hypothesis",
            "hypotheses": [
                {
                    "id": "hyp-1",
                    "rank": 1,
                    "title": "First hypothesis",
                    "claim": "Claim one.",
                    "rationale": "Rationale one.",
                    "novelty_score": 0.7,
                    "evidence_score": 0.8,
                    "testability_score": 0.8,
                    "overall_score": 0.79,
                    "recommended_next_step": "Test one.",
                },
                {
                    "id": "hyp-2",
                    "rank": 2,
                    "title": "Second hypothesis",
                    "claim": "Claim two.",
                    "rationale": "Rationale two.",
                    "novelty_score": 0.6,
                    "evidence_score": 0.7,
                    "testability_score": 0.9,
                    "overall_score": 0.75,
                    "recommended_next_step": "Test two.",
                },
            ],
        },
    )
    plan = ExperimentPlan(
        user_id=test_user.id,
        research_note_id=note.id,
        title="Aggregate Validation Plan",
        hypothesis_text="Aggregate validation should enrich all selected hypotheses.",
        plan={"experiments": []},
    )
    run = ExperimentRun(
        user_id=test_user.id,
        experiment_plan_id=plan.id,
        name="Aggregate Append Run",
        status="completed",
        progress=100,
        config={
            "execution_handoff": {
                "plan_scope": "aggregate_note",
                "selected_hypothesis_ids": ["hyp-1", "hyp-2"],
                "supporting_sources": [{"id": "paper-2", "title": "Aggregate paper"}],
                "source_paper_ids": ["paper-2"],
                "source_document_ids": ["doc-2"],
            }
        },
        summary="Aggregate run completed with shared evidence.",
        results={
            "ok": True,
            "final_phase": "aggregate_eval",
            "verification_commands": ["python -m pytest -q backend/tests/test_experiments_endpoints.py"],
        },
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        plan.research_note_id = note.id
        db_session.add(plan)
        await db_session.flush()
        run.experiment_plan_id = plan.id
        db_session.add(run)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = experiment_client.post(f"/api/v1/experiments/runs/{run.id}/append-to-note")

    assert response.status_code == 200
    hypotheses = response.json()["structured_payload"]["hypotheses"]
    for item in hypotheses:
        assert len(item["experiment_evidence"]) == 1
        assert item["experiment_evidence"][0]["run_id"] == str(run.id)
        assert item["experiment_evidence"][0]["plan_scope"] == "aggregate_note"


def test_experiment_run_action_pause_proxies_to_linked_job(
    experiment_client,
    db_session,
    test_user,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Pause Validation",
        content_markdown="## Hypothesis\nScientific validation should support pause.",
        tags=["agents"],
    )
    plan = ExperimentPlan(
        user_id=test_user.id,
        research_note_id=note.id,
        title="Pause Validation Plan",
        hypothesis_text="Scientific validation should support pause.",
        plan={"experiments": []},
    )
    job = AgentJob(
        name="Scientific Validation Job",
        goal="Execute validation",
        job_type="analysis",
        user_id=test_user.id,
        status=AgentJobStatus.RUNNING.value,
        progress=25,
        iteration=1,
        max_iterations=1,
        max_tool_calls=0,
        max_llm_calls=0,
        max_runtime_minutes=10,
    )
    run = ExperimentRun(
        user_id=test_user.id,
        experiment_plan_id=plan.id,
        agent_job_id=job.id,
        name="Pauseable Run",
        status="running",
        progress=25,
        config={
            "scientific_validation": {
                "validation_kind": "scientific_validation",
                "sandbox_profile_id": "scientific-compiler-sandbox",
                "recipe_family": "compiler_validation",
                "operator_actions": [],
            }
        },
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        plan.research_note_id = note.id
        db_session.add(plan)
        await db_session.flush()
        db_session.add(job)
        await db_session.flush()
        run.experiment_plan_id = plan.id
        run.agent_job_id = job.id
        db_session.add(run)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = experiment_client.post(
        f"/api/v1/experiments/runs/{run.id}/action",
        json={"action": "pause", "note": "Pause this validation"},
    )

    assert response.status_code == 200
    payload = response.json()["run"]
    assert payload["status"] == "paused"
    assert payload["agent_job_id"] == str(job.id)
    assert payload["operator_actions"] is not None
    assert payload["operator_actions"][-1]["action"] == "pause"
    assert payload["operator_actions"][-1]["linked_job_action"] == "pause"
    assert payload["operator_actions"][-1]["outcome_status"] == "applied"


def test_experiment_run_action_retry_creates_child_run_lineage(
    experiment_client,
    db_session,
    test_user,
    monkeypatch,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Retry Validation",
        content_markdown="## Hypothesis\nScientific validation retries should create child runs.",
        tags=["agents"],
    )
    job = AgentJob(
        user_id=test_user.id,
        name="Validation Job",
        goal="Run scientific validation",
        job_type="research",
        status=AgentJobStatus.FAILED.value,
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
    plan = ExperimentPlan(
        user_id=test_user.id,
        research_note_id=note.id,
        title="Retry Validation Plan",
        hypothesis_text="Scientific validation retries should create child runs.",
        plan={"experiments": []},
    )
    run = ExperimentRun(
        user_id=test_user.id,
        experiment_plan_id=plan.id,
        agent_job_id=job.id,
        name="Failed Validation Run",
        status="failed",
        progress=100,
        retry_count=0,
        config={
            "source_id": "repo-9",
            "commands": ["python -m pytest -q backend/tests"],
            "timeout_seconds": 90,
            "scientific_validation": {
                "validation_kind": "scientific_validation",
                "sandbox_profile_id": "scientific-compiler-sandbox",
                "recipe_family": "compiler_validation",
                "recipe_id": "compiler_validation_v1",
                "blocked_reason_code": "missing_capability",
            },
        },
    )
    captured = {}

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        plan.research_note_id = note.id
        db_session.add(plan)
        await db_session.flush()
        db_session.add(job)
        await db_session.flush()
        run.experiment_plan_id = plan.id
        run.agent_job_id = job.id
        db_session.add(run)
        await db_session.commit()

    async def _fake_record(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(experiments, "record_autonomy_decision_event", _fake_record)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = experiment_client.post(
        f"/api/v1/experiments/runs/{run.id}/action",
        json={"action": "retry", "note": "Retry after updating the sandbox profile", "start_immediately": False},
    )

    assert response.status_code == 200
    payload = response.json()["run"]
    assert payload["parent_run_id"] == str(run.id)
    assert payload["retry_count"] == 1
    assert payload["status"] == "planned"
    assert payload["agent_job_id"] is None
    assert payload["operator_actions"] is not None
    assert payload["operator_actions"][0]["action"] == "retry"
    assert payload["operator_actions"][0]["outcome_status"] == "spawned"
    assert captured["reason_label"] == "Validation requeued"
    assert captured["scheduler_state"] == {
        "queue_reason": "execution_failure",
        "last_scheduled_at": "2026-03-16T09:00:00Z",
        "last_dispatched_at": "2026-03-16T09:05:00Z",
    }

    async def _verify():
        parent = await db_session.get(ExperimentRun, run.id)
        child = await db_session.get(ExperimentRun, payload["id"])
        assert parent is not None
        assert child is not None
        assert str(parent.latest_child_run_id) == payload["id"]
        assert child.parent_run_id == parent.id
        assert child.retry_count == 1

    asyncio.get_event_loop().run_until_complete(_verify())


def test_experiment_run_action_retry_omits_malformed_scheduler_state(
    experiment_client,
    db_session,
    test_user,
    monkeypatch,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Retry Validation",
        content_markdown="## Hypothesis\nScientific validation retries should drop malformed scheduler state.",
        tags=["agents"],
    )
    job = AgentJob(
        user_id=test_user.id,
        name="Validation Job",
        goal="Run scientific validation",
        job_type="research",
        status=AgentJobStatus.FAILED.value,
        results={"execution_strategy": {"scheduler_state": "bad-payload"}},
    )
    plan = ExperimentPlan(
        user_id=test_user.id,
        research_note_id=note.id,
        title="Retry Validation Plan",
        hypothesis_text="Scientific validation retries should drop malformed scheduler state.",
        plan={"experiments": []},
    )
    run = ExperimentRun(
        user_id=test_user.id,
        experiment_plan_id=plan.id,
        agent_job_id=job.id,
        name="Failed Validation Run",
        status="failed",
        progress=100,
        retry_count=0,
        config={
            "source_id": "repo-9",
            "commands": ["python -m pytest -q backend/tests"],
            "timeout_seconds": 90,
            "scientific_validation": {
                "validation_kind": "scientific_validation",
                "sandbox_profile_id": "scientific-compiler-sandbox",
                "recipe_family": "compiler_validation",
                "recipe_id": "compiler_validation_v1",
                "blocked_reason_code": "missing_capability",
            },
        },
    )
    captured = {}

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        plan.research_note_id = note.id
        db_session.add(plan)
        await db_session.flush()
        db_session.add(job)
        await db_session.flush()
        run.experiment_plan_id = plan.id
        run.agent_job_id = job.id
        db_session.add(run)
        await db_session.commit()

    async def _fake_record(*args, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(experiments, "record_autonomy_decision_event", _fake_record)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = experiment_client.post(
        f"/api/v1/experiments/runs/{run.id}/action",
        json={"action": "retry", "note": "Retry after updating the sandbox profile", "start_immediately": False},
    )

    assert response.status_code == 200
    assert captured["reason_label"] == "Validation requeued"
    assert captured["scheduler_state"] is None


def test_generate_experiment_plan_uses_structured_hypotheses_for_aggregate_mode(
    experiment_client,
    db_session,
    test_user,
    monkeypatch,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Compiler Hypotheses",
        content_markdown="## Hypothesis\nlegacy fallback",
        structured_payload={
            "research_mode": "paper_to_hypothesis",
            "summary": "Cross-paper compiler synthesis.",
            "source_paper_ids": ["paper-1", "paper-2"],
            "source_document_ids": ["doc-1", "doc-2"],
            "hypotheses": [
                {
                    "id": "hyp-1",
                    "rank": 1,
                    "title": "Layout plus scheduling",
                    "claim": "If layout transforms are paired with schedule-aware pressure control, locality gains survive register pressure.",
                    "rationale": "Two papers optimize adjacent bottlenecks.",
                    "overall_score": 0.9,
                    "supporting_sources": [{"id": "paper-1", "title": "Compiler layouts"}],
                    "recommended_next_step": "Implement on stencil kernels.",
                },
                {
                    "id": "hyp-2",
                    "rank": 2,
                    "title": "Adaptive prefetch threshold",
                    "claim": "If prefetch distance adapts to phase behavior, stalls decline on irregular kernels.",
                    "rationale": "Static thresholds underfit phase changes.",
                    "overall_score": 0.8,
                    "supporting_sources": [{"id": "paper-2", "title": "Adaptive prefetching"}],
                    "recommended_next_step": "Evaluate on irregular kernels.",
                },
            ],
        },
        tags=["hypotheses"],
    )

    prompts: list[str] = []

    async def _fake_generate_response(self, query=None, **kwargs):
        prompts.append(str(query or ""))
        return json.dumps(
            {
                "objective": "Validate top compiler hypotheses.",
                "hypothesis": "Aggregate hypothesis program",
                "hypotheses": [
                    {"id": "hyp-1", "title": "Layout plus scheduling", "claim": "..."},
                    {"id": "hyp-2", "title": "Adaptive prefetch threshold", "claim": "..."},
                ],
                "problem_statement": "Need to validate paper-derived ideas.",
                "success_criteria": ["runtime improves"],
                "datasets": [{"name": "PolyBench", "source": "suite", "split": None, "notes": None}],
                "metrics": [{"name": "runtime", "definition": "lower is better", "direction": "lower_better"}],
                "baselines": [{"name": "baseline", "details": "current compiler"}],
                "method": {"summary": "Run staged evaluation", "key_components": ["layout", "scheduling"]},
                "experiments": [{"name": "E1", "purpose": "test", "variables": ["layout"], "expected_outcome": "improves"}],
                "ablations": [],
                "evaluation_protocol": "Compare against baseline.",
                "compute_budget": {"hardware": None, "time_estimate": None, "notes": None},
                "timeline": [],
                "risks": [],
                "repro_checklist": [],
            }
        )

    monkeypatch.setattr(experiments.LLMService, "generate_response", _fake_generate_response)

    async def _seed():
        db_session.add(note)
        await db_session.commit()
        await db_session.refresh(note)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = experiment_client.post(
        "/api/v1/experiments/plans/generate",
        json={"note_id": str(note.id), "plan_mode": "aggregate_note"},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["generator_details"]["plan_mode"] == "aggregate_note"
    assert payload["generator_details"]["selected_hypothesis_ids"] == ["hyp-1", "hyp-2"]
    assert payload["generator_details"]["source_paper_ids"] == ["paper-1", "paper-2"]
    assert payload["plan"]["plan_scope"] == "aggregate_note"
    assert payload["plan"]["selected_hypothesis_ids"] == ["hyp-1", "hyp-2"]
    assert payload["plan"]["provenance"]["source_document_ids"] == ["doc-1", "doc-2"]
    assert "Selected hypotheses:" in prompts[0]
    assert "hyp-1" in prompts[0]
    assert "hyp-2" in prompts[0]


def test_list_benchmark_suites_returns_builtin_compiler_harnesses(
    experiment_client,
):
    response = experiment_client.get("/api/v1/experiments/benchmark-suites")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] >= 1
    assert any(item["benchmark_family"] == "compiler_regression" for item in payload["items"])
    first_suite = payload["items"][0]
    assert isinstance(first_suite["cases"], list)
    assert isinstance(first_suite["baselines"], list)


def test_generate_experiment_plan_carries_benchmark_suite_metadata(
    experiment_client,
    db_session,
    test_user,
    monkeypatch,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Benchmark-backed compiler note",
        content_markdown="## Hypothesis\nUse the compiler harness.",
        structured_payload={
            "hypotheses": [
                {
                    "id": "hyp-1",
                    "rank": 1,
                    "title": "Compiler regression hypothesis",
                    "claim": "Compile-time and codegen deltas should be measurable.",
                    "rationale": "The benchmark harness provides stable comparisons.",
                    "overall_score": 0.95,
                }
            ]
        },
    )

    async def _fake_generate_response(self, query=None, **kwargs):
        return json.dumps(
            {
                "objective": "Validate the hypothesis against a compiler benchmark suite.",
                "hypothesis": "Benchmark-backed validation",
                "hypotheses": [{"id": "hyp-1", "title": "Compiler regression hypothesis", "claim": "Compile-time and codegen deltas should be measurable."}],
                "problem_statement": "Need benchmark-backed evidence.",
                "success_criteria": ["compile_time improves"],
                "datasets": [{"name": "LLVM Regression Core", "source": "benchmark_suite", "split": None, "notes": None}],
                "metrics": [{"name": "compile_time_ms", "definition": "lower is better", "direction": "lower_better"}],
                "baselines": [{"name": "LLVM main baseline", "details": "clang-18"}],
                "method": {"summary": "Use benchmark harness", "key_components": ["suite", "baseline"]},
                "experiments": [{"name": "Harness run", "purpose": "measure", "variables": ["suite"], "expected_outcome": "observable deltas"}],
                "ablations": [],
                "evaluation_protocol": "Compare against harness baseline.",
                "compute_budget": {"hardware": None, "time_estimate": None, "notes": None},
                "timeline": [],
                "risks": [],
                "repro_checklist": [],
            }
        )

    monkeypatch.setattr(experiments.LLMService, "generate_response", _fake_generate_response)

    async def _seed():
        db_session.add(note)
        await db_session.commit()
        await db_session.refresh(note)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = experiment_client.post(
        "/api/v1/experiments/plans/generate",
        json={
            "note_id": str(note.id),
            "plan_mode": "single_hypothesis",
            "hypothesis_id": "hyp-1",
            "benchmark_suite_id": "compiler-llvm-regression-core",
            "benchmark_case_ids": ["case-instcombine-sroa"],
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["benchmark_suite_id"] == "compiler-llvm-regression-core"
    assert payload["benchmark_case_ids"] == ["case-instcombine-sroa"]
    assert payload["generator_details"]["benchmark_family"] == "compiler_regression"
    assert payload["plan"]["provenance"]["benchmark_suite_id"] == "compiler-llvm-regression-core"
    assert payload["plan"]["benchmark_case_ids"] == ["case-instcombine-sroa"]


def test_generate_experiment_plan_validates_single_hypothesis_selection(
    experiment_client,
    db_session,
    test_user,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Structured Note",
        content_markdown="## Hypothesis\nfallback",
        structured_payload={
            "hypotheses": [
                {
                    "id": "hyp-1",
                    "rank": 1,
                    "title": "Only hypothesis",
                    "claim": "Claim",
                    "rationale": "Why",
                    "overall_score": 0.9,
                }
            ]
        },
    )

    async def _seed():
        db_session.add(note)
        await db_session.commit()
        await db_session.refresh(note)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = experiment_client.post(
        "/api/v1/experiments/plans/generate",
        json={"note_id": str(note.id), "plan_mode": "single_hypothesis", "hypothesis_id": "missing"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unknown hypothesis_id for this research note"


def test_generate_experiment_plan_defaults_reevaluated_note_to_top_ranked_hypothesis(
    experiment_client,
    db_session,
    test_user,
    monkeypatch,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Reevaluated Note",
        content_markdown="## Hypothesis\nfallback",
        structured_payload={
            "artifact_type": "hypothesis_reevaluation",
            "scoring_policy": {"source_job_id": "syn-reeval-1"},
            "source_paper_ids": ["paper-1"],
            "source_document_ids": ["doc-1"],
            "hypotheses": [
                {
                    "id": "hyp-2",
                    "rank": 2,
                    "title": "Lower ranked",
                    "claim": "Claim two",
                    "rationale": "Why two",
                    "overall_score": 0.7,
                },
                {
                    "id": "hyp-1",
                    "rank": 1,
                    "title": "Top ranked",
                    "claim": "Claim one",
                    "rationale": "Why one",
                    "overall_score": 0.9,
                    "recommended_next_step": "Run the strongest benchmark first.",
                },
            ],
        },
    )

    async def _fake_generate_response(self, query=None, **kwargs):
        return json.dumps(
            {
                "objective": "Validate the reevaluated top hypothesis.",
                "hypothesis": "Top ranked hypothesis only",
                "hypotheses": [{"id": "hyp-1", "title": "Top ranked", "claim": "Claim one"}],
                "problem_statement": "Need to test the strongest reevaluated idea.",
                "success_criteria": ["runtime improves"],
                "datasets": [],
                "metrics": [{"name": "runtime", "definition": "lower is better", "direction": "lower_better"}],
                "baselines": [{"name": "baseline", "details": "current system"}],
                "method": {"summary": "Target top idea", "key_components": ["top"]},
                "experiments": [{"name": "E1", "purpose": "test top", "variables": ["top"], "expected_outcome": "improves"}],
                "ablations": [],
                "evaluation_protocol": "Compare against baseline.",
                "compute_budget": {"hardware": None, "time_estimate": None, "notes": None},
                "timeline": [],
                "risks": [],
                "repro_checklist": [],
            }
        )

    monkeypatch.setattr(experiments.LLMService, "generate_response", _fake_generate_response)

    async def _seed():
        db_session.add(note)
        await db_session.commit()
        await db_session.refresh(note)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = experiment_client.post(
        "/api/v1/experiments/plans/generate",
        json={"note_id": str(note.id)},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["generator_details"]["plan_mode"] == "single_hypothesis"
    assert payload["generator_details"]["hypothesis_id"] == "hyp-1"
    assert payload["generator_details"]["selected_hypothesis_ids"] == ["hyp-1"]
    assert payload["generator_details"]["reevaluation_mode"] is True
    assert payload["generator_details"]["reevaluation_source_job_id"] == "syn-reeval-1"
    assert payload["plan"]["plan_scope"] == "single_hypothesis"
    assert payload["plan"]["selected_hypothesis_ids"] == ["hyp-1"]


def test_generate_experiment_plan_defaults_explanation_note_to_followup_mode(
    experiment_client,
    db_session,
    test_user,
    monkeypatch,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Compiler Regression Explanation",
        content_markdown="## Explanation\nRegression summary.",
        structured_payload={
            "artifact_type": "compiler_regression_explanation",
            "summary": "Compile time regressed after vectorization remarks disappeared.",
            "regression_type": "compile_time",
            "source_run_ids": ["run-new", "run-old"],
            "primary_run_id": "run-new",
            "comparison_run_id": "run-old",
            "likely_causes": [
                {
                    "title": "Vectorizer not firing",
                    "confidence": "medium",
                    "reason": "loop-vectorize remarks disappeared",
                }
            ],
            "recommended_next_steps": ["Diff pass remarks across both builds"],
            "source_document_ids": ["doc-1"],
            "benchmark_family": "compiler_regression",
            "benchmark_suite_id": "compiler-llvm-regression-core",
            "benchmark_case_ids": ["case-instcombine-sroa"],
            "benchmark_baseline_id": "baseline-llvm-main",
        },
    )
    prompts: list[str] = []

    async def _fake_generate_response(self, query=None, **kwargs):
        prompts.append(str(query or ""))
        return json.dumps(
            {
                "objective": "Isolate why vectorization remarks disappeared on the compared benchmark case.",
                "hypothesis": "The regression is caused by a missed vectorization decision.",
                "hypotheses": [],
                "problem_statement": "Need a targeted regression follow-up plan.",
                "success_criteria": ["compile_time_ms returns to baseline"],
                "datasets": [{"name": "LLVM Regression Core", "source": "benchmark_suite", "split": None, "notes": None}],
                "metrics": [
                    {"name": "compile_time_ms", "definition": "lower is better", "direction": "lower_better"},
                    {"name": "artifact_diff_score", "definition": "lower is better", "direction": "lower_better"},
                ],
                "baselines": [{"name": "LLVM main baseline", "details": "baseline-llvm-main"}],
                "method": {"summary": "Compare remarks and codegen outputs across the regressed case", "key_components": ["remarks diff", "IR diff"]},
                "experiments": [
                    {"name": "Remark diff", "purpose": "diff pass remarks", "variables": ["build"], "expected_outcome": "identify missing vectorization trigger"},
                    {"name": "IR capture", "purpose": "compare IR outputs", "variables": ["pipeline"], "expected_outcome": "find lowered divergence"},
                    {"name": "Case replay", "purpose": "re-run benchmark case", "variables": ["baseline", "candidate"], "expected_outcome": "confirm regression scope"},
                ],
                "ablations": [],
                "evaluation_protocol": "Compare the regressed run against the prior compatible run and baseline.",
                "compute_budget": {"hardware": None, "time_estimate": None, "notes": None},
                "timeline": [],
                "risks": [],
                "repro_checklist": [],
            }
        )

    monkeypatch.setattr(experiments.LLMService, "generate_response", _fake_generate_response)

    async def _seed():
        db_session.add(note)
        await db_session.commit()
        await db_session.refresh(note)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = experiment_client.post(
        "/api/v1/experiments/plans/generate",
        json={"note_id": str(note.id)},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["generator_details"]["plan_mode"] == "compiler_regression_followup"
    assert payload["generator_details"]["explanation_mode"] is True
    assert payload["generator_details"]["source_run_ids"] == ["run-new", "run-old"]
    assert payload["generator_details"]["primary_run_id"] == "run-new"
    assert payload["generator_details"]["comparison_run_id"] == "run-old"
    assert payload["generator_details"]["regression_type"] == "compile_time"
    assert payload["benchmark_suite_id"] == "compiler-llvm-regression-core"
    assert payload["benchmark_case_ids"] == ["case-instcombine-sroa"]
    assert payload["benchmark_baseline_id"] == "baseline-llvm-main"
    assert payload["plan"]["plan_scope"] == "compiler_regression_followup"
    assert payload["plan"]["provenance"]["benchmark_suite_id"] == "compiler-llvm-regression-core"
    assert "Likely causes:" in prompts[0]
    assert "Requested plan scope: compiler_regression_followup" in prompts[0]
    assert "Diff pass remarks across both builds" in prompts[0]


def test_create_experiment_run_seeds_execution_handoff_from_plan(
    experiment_client,
    db_session,
    test_user,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Handoff Note",
        content_markdown="## Hypothesis\nRun seeded metadata.",
        tags=["hypotheses"],
    )
    plan = ExperimentPlan(
        user_id=test_user.id,
        research_note_id=note.id,
        title="Compiler Plan",
        hypothesis_text="Seed metadata",
        plan={
            "objective": "Validate compiler hypothesis",
            "hypothesis": "If layout and scheduling are combined, runtime improves.",
            "plan_scope": "single_hypothesis",
            "selected_hypothesis_ids": ["hyp-1"],
            "supporting_sources": [{"id": "paper-1", "title": "Compiler layouts"}],
            "provenance": {"source_paper_ids": ["paper-1"], "source_document_ids": ["doc-1"]},
        },
        generator_details={
            "plan_mode": "single_hypothesis",
            "selected_hypothesis_ids": ["hyp-1"],
            "source_paper_ids": ["paper-1"],
            "source_document_ids": ["doc-1"],
            "supporting_sources": [{"id": "paper-1", "title": "Compiler layouts"}],
        },
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        plan.research_note_id = note.id
        db_session.add(plan)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = experiment_client.post(
        f"/api/v1/experiments/plans/{plan.id}/runs",
        json={"name": "Focused validation run"},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["config"]["execution_handoff"]["plan_scope"] == "single_hypothesis"
    assert payload["config"]["execution_handoff"]["selected_hypothesis_ids"] == ["hyp-1"]
    assert payload["config"]["execution_handoff"]["source_paper_ids"] == ["paper-1"]
    assert payload["summary"]


def test_create_experiment_run_seeds_scientific_validation_for_benchmark_plan(
    experiment_client,
    db_session,
    test_user,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Benchmark plan note",
        content_markdown="## Hypothesis\nUse benchmark suite.",
    )
    plan = ExperimentPlan(
        user_id=test_user.id,
        research_note_id=note.id,
        title="Benchmark-backed Plan",
        hypothesis_text="Use benchmark-backed validation",
        plan={
            "objective": "Validate compiler benchmark hypothesis",
            "benchmark_family": "compiler_regression",
            "benchmark_suite_id": "compiler-llvm-regression-core",
            "benchmark_case_ids": ["case-instcombine-sroa"],
            "benchmark_baseline_id": "baseline-llvm-main",
            "provenance": {
                "benchmark_family": "compiler_regression",
                "benchmark_suite_id": "compiler-llvm-regression-core",
                "benchmark_case_ids": ["case-instcombine-sroa"],
                "benchmark_baseline_id": "baseline-llvm-main",
            },
        },
        generator_details={
            "plan_mode": "single_hypothesis",
            "benchmark_family": "compiler_regression",
            "benchmark_suite_id": "compiler-llvm-regression-core",
            "benchmark_case_ids": ["case-instcombine-sroa"],
            "benchmark_baseline_id": "baseline-llvm-main",
        },
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        plan.research_note_id = note.id
        db_session.add(plan)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = experiment_client.post(
        f"/api/v1/experiments/plans/{plan.id}/runs",
        json={
            "name": "Benchmark-backed run",
            "config": {
                "source_id": "00000000-0000-0000-0000-000000000001",
                "commands": ["clang -O3 -S -emit-llvm fixtures/llvm/instcombine_sroa.c -o /tmp/instcombine_sroa.ll"],
            },
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["validation_kind"] == "scientific_validation"
    assert payload["benchmark_family"] == "compiler_regression"
    assert payload["benchmark_suite_id"] == "compiler-llvm-regression-core"
    assert payload["benchmark_case_ids"] == ["case-instcombine-sroa"]
    assert payload["benchmark_baseline_id"] == "baseline-llvm-main"
    assert payload["config"]["scientific_validation"]["recipe_family"] == "compiler_validation"
    assert payload["config"]["scientific_validation"]["compiler_observability"]["capture_ir"] is True
    assert payload["config"]["scientific_validation"]["compiler_observability"]["capture_remarks"] is True
    assert payload["measurement_summary"]["artifact_inventory"]
    assert payload["artifact_inventory"]
    assert payload["repeat_count"] == 2


def test_sync_experiment_run_projects_compiler_observability_into_response(
    experiment_client,
    db_session,
    test_user,
):
    note = ResearchNote(
        user_id=test_user.id,
        title="Compiler Observability",
        content_markdown="## Hypothesis\nProject compiler artifacts into run responses.",
    )
    plan = ExperimentPlan(
        user_id=test_user.id,
        research_note_id=note.id,
        title="Compiler Observability Plan",
        hypothesis_text="Need compiler observability",
        plan={"experiments": []},
    )
    job = AgentJob(
        name="Compiler Observability Job",
        goal="Sync compiler observability output",
        job_type="analysis",
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
        progress=100,
        iteration=1,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=10,
        results={
            "experiment_run": {
                "source_id": "repo-obs",
                "final_phase": "verification",
                "verification_commands": ["clang -O3 -S fixture.c -o /tmp/fixture.s"],
            },
            "measurement_summary": {
                "compile_time_ms": 1180,
                "artifact_diff_score": 0.14,
                "comparison": "improvement",
                "perf_counters": {"instructions": 1024, "branch_misses": 12},
            },
            "compiler_artifacts": {
                "asm_paths": ["/tmp/fixture.s"],
                "remark_paths": ["/tmp/fixture.opt.yaml"],
                "log_paths": ["/tmp/compile.log"],
                "diff_summary": "Vectorized inner loop with reduced spill pressure.",
                "pass_signals": ["loop-vectorize", "regalloc"],
            },
        },
    )
    run = ExperimentRun(
        user_id=test_user.id,
        experiment_plan_id=plan.id,
        agent_job_id=job.id,
        name="Compiler Observability Run",
        status="running",
        progress=50,
        config={
            "scientific_validation": {
                "validation_kind": "scientific_validation",
                "benchmark_family": "compiler_regression",
                "benchmark_suite_id": "compiler-llvm-regression-core",
                "benchmark_case_ids": ["case-loop-vectorize-reduction"],
                "benchmark_baseline_id": "baseline-llvm-main",
                "measurement_summary": {
                    "status": "pending",
                    "artifact_inventory": ["compiler_logs", "compiler_remarks", "ir_or_codegen_artifacts"],
                    "repeat_count": 3,
                },
                "compiler_observability": {
                    "capture_asm": True,
                    "capture_remarks": True,
                    "capture_compile_logs": True,
                    "artifact_inventory": ["compiler_logs", "compiler_remarks", "ir_or_codegen_artifacts"],
                    "repeat_count": 3,
                },
            }
        },
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        plan.research_note_id = note.id
        db_session.add(plan)
        await db_session.flush()
        db_session.add(job)
        await db_session.flush()
        run.experiment_plan_id = plan.id
        run.agent_job_id = job.id
        db_session.add(run)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = experiment_client.post(f"/api/v1/experiments/runs/{run.id}/sync")

    assert response.status_code == 200
    payload = response.json()["run"]
    assert payload["compiler_artifacts"]["capture_asm"] is True
    assert payload["compiler_artifacts"]["diff_summary"] == "Vectorized inner loop with reduced spill pressure."
    assert payload["artifact_inventory"] == ["compiler_logs", "compiler_remarks", "ir_or_codegen_artifacts"]
    assert payload["perf_counters"]["instructions"] == 1024
    assert payload["measurement_summary"]["artifact_diff_score"] == 0.14
    assert payload["repeat_count"] == 3


def test_start_experiment_run_forwards_execution_handoff_to_agent_job(
    experiment_client,
    db_session,
    test_user,
    monkeypatch,
):
    from app.tasks.agent_job_tasks import execute_agent_job_task

    queued: list[tuple[str, str]] = []

    def _fake_delay(job_id: str, user_id: str):
        queued.append((job_id, user_id))

    monkeypatch.setattr(execute_agent_job_task, "delay", _fake_delay)

    note = ResearchNote(
        user_id=test_user.id,
        title="Start Handoff",
        content_markdown="## Hypothesis\nForward handoff.",
        tags=["hypotheses"],
    )
    plan = ExperimentPlan(
        user_id=test_user.id,
        research_note_id=note.id,
        title="Aggregate Plan",
        hypothesis_text="Forward metadata",
        plan={"objective": "Validate aggregate hypotheses"},
        generator_details={"plan_mode": "aggregate_note"},
    )
    run = ExperimentRun(
        user_id=test_user.id,
        experiment_plan_id=plan.id,
        name="Aggregate run",
        status="planned",
        progress=0,
        config={
            "source_id": "repo-1",
            "commands": ["python -m pytest -q"],
            "timeout_seconds": 60,
            "execution_handoff": {
                "execution_handoff_version": 1,
                "plan_scope": "aggregate_note",
                "selected_hypothesis_ids": ["hyp-1", "hyp-2"],
                "source_paper_ids": ["paper-1"],
                "source_document_ids": ["doc-1"],
                "supporting_sources": [{"id": "paper-1", "title": "Compiler layouts"}],
            },
        },
    )

    async def _seed():
        db_session.add(note)
        await db_session.flush()
        plan.research_note_id = note.id
        db_session.add(plan)
        await db_session.flush()
        run.experiment_plan_id = plan.id
        db_session.add(run)
        await db_session.commit()
        await db_session.refresh(run)

    asyncio.get_event_loop().run_until_complete(_seed())

    response = experiment_client.post(
        f"/api/v1/experiments/runs/{run.id}/start",
        json={"source_id": "00000000-0000-0000-0000-000000000001", "commands": ["python -m pytest -q"], "timeout_seconds": 60},
    )

    assert response.status_code == 200
    assert queued

    async def _verify():
        started_run = await db_session.get(ExperimentRun, run.id)
        assert started_run is not None
        job = await db_session.get(AgentJob, started_run.agent_job_id)
        assert job is not None
        assert job.config["execution_handoff"]["plan_scope"] == "aggregate_note"
        assert job.config["execution_handoff"]["selected_hypothesis_ids"] == ["hyp-1", "hyp-2"]

    asyncio.get_event_loop().run_until_complete(_verify())
