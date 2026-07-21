from uuid import uuid4

from app.api.endpoints.experiments import _build_experiment_run_note_block, _run_to_response
from app.models.experiment import ExperimentRun


def test_run_to_response_projects_typed_experiment_run_payload():
    run = ExperimentRun(
        id=uuid4(),
        user_id=uuid4(),
        experiment_plan_id=uuid4(),
        name="Bootstrap Retry Run",
        status="completed",
        progress=100,
        results={
            "source_id": "repo-1",
            "ok": True,
            "final_phase": "retry_primary",
            "bootstrap_attempted": True,
            "bootstrap_ok": True,
            "verification_commands": ["CI=true npm --prefix frontend test -- --watchAll=false"],
        },
    )

    response = _run_to_response(run)

    assert response.results is not None
    assert response.experiment_run is not None
    assert response.experiment_run.source_id == "repo-1"
    assert response.experiment_run.final_phase == "retry_primary"
    assert response.experiment_run.bootstrap_ok is True
    assert response.experiment_run.verification_commands == [
        "CI=true npm --prefix frontend test -- --watchAll=false"
    ]


def test_run_to_response_projects_typed_operator_interventions():
    run = ExperimentRun(
        id=uuid4(),
        user_id=uuid4(),
        experiment_plan_id=uuid4(),
        name="Intervention Projection",
        status="running",
        progress=40,
        results={
            "execution_strategy": {
                "operator_interventions": [
                    {
                        "action": "restart",
                        "actor_user_id": "user-1",
                        "at": "2026-03-10T01:00:00Z",
                        "job_status_before": "failed",
                        "job_status_after": "pending",
                        "note": "Retry after fallback failure",
                    }
                ]
            }
        },
    )

    response = _run_to_response(run)

    assert response.operator_interventions is not None
    assert len(response.operator_interventions) == 1
    assert response.operator_interventions[0].action == "restart"
    assert response.operator_interventions[0].actor_user_id == "user-1"
    assert response.operator_interventions[0].job_status_before == "failed"
    assert response.operator_interventions[0].job_status_after == "pending"
    assert response.operator_interventions[0].note == "Retry after fallback failure"
    assert response.operator_interventions[0].outcome_status == "applied"
    assert response.operator_interventions[0].outcome_reason == "Job resumed after intervention"


def test_run_to_response_ignores_invalid_operator_intervention_shapes():
    run = ExperimentRun(
        id=uuid4(),
        user_id=uuid4(),
        experiment_plan_id=uuid4(),
        name="Intervention Projection Invalid",
        status="running",
        progress=20,
        results={
            "execution_strategy": {
                "operator_interventions": [
                    "bad-payload",
                    {"action": "resume", "job_status_before": "paused", "job_status_after": "running"},
                ]
            }
        },
    )

    response = _run_to_response(run)

    assert response.operator_interventions is not None
    assert len(response.operator_interventions) == 1
    assert response.operator_interventions[0].action == "resume"
    assert response.operator_interventions[0].job_status_before == "paused"
    assert response.operator_interventions[0].job_status_after == "running"
    assert response.operator_interventions[0].outcome_status == "pending"
    assert response.operator_interventions[0].outcome_reason == "Awaiting job outcome"


def test_build_experiment_run_note_block_includes_summary_and_bootstrap_context():
    run = ExperimentRun(
        id=uuid4(),
        user_id=uuid4(),
        experiment_plan_id=uuid4(),
        name="Append Run",
        status="completed",
        progress=100,
        summary="Recovered environment but backend verification still failed.",
        results={
            "source_id": "repo-9",
            "source_name": "Repo Nine",
            "final_phase": "fallback",
            "phases": ["primary", "bootstrap", "fallback"],
            "bootstrap_attempted": True,
            "bootstrap_ok": True,
            "fallback_attempted": True,
            "fallback_ok": False,
            "inferred_project_profile": {"detected_stack": ["node", "python"]},
            "verification_commands": ["npm test"],
            "bootstrap_commands": ["npm install"],
            "fallback_commands": ["python3 -m pytest -q"],
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

    block = _build_experiment_run_note_block(run, marker="<!-- experiment_run:test -->")
    text = "\n".join(block)

    assert "## Experiment Results" in text
    assert "Source: Repo Nine" in text
    assert "Source ID: `repo-9`" in text
    assert "Detected stack: node, python" in text
    assert "Final phase: `fallback`" in text
    assert "Bootstrap: ok" in text
    assert "Fallback: attempted" in text
    assert "Recovery: open" in text
    assert "Operator intervention:" in text
    assert "Latest: restart (failed -> pending)" in text
    assert "Outcome: resolved" in text
    assert "Outcome reason: Job completed after intervention" in text
    assert "Note: Retry after fallback failure" in text
    assert "Summary:" in text
    assert "Recovered environment but backend verification still failed." in text
    assert "Recovery guidance:" in text
    assert "Reason: fallback verification still failing" in text
    assert "Next: Inspect failing fallback output" in text
