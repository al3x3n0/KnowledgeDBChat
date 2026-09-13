from pathlib import Path

import pytest

from app.services.autonomous_rnd_eval_service import (
    AutonomousRnDEvalHarness,
    AutonomousRnDEvalSuite,
    AutonomousRnDEvalTask,
    EvalDefinitionError,
)

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _task(*, trials: int = 3) -> AutonomousRnDEvalTask:
    return AutonomousRnDEvalTask(
        id="reproduce",
        name="Reproduce",
        category="experiment",
        prompt="Reproduce the result.",
        trials=trials,
        pass_threshold=1.0,
        graders=[
            {"type": "status_equals", "expected": "completed"},
            {"type": "traceable_claims", "min_evidence_per_claim": 1},
            {
                "type": "experiment_execution",
                "min_repeat_count": 3,
                "require_commands_ok": True,
            },
        ],
    )


def _passing_outcome():
    return {
        "status": "completed",
        "claims": [{"id": "claim-1", "evidence_ids": ["evidence-1"]}],
        "evidence": [{"id": "evidence-1", "kind": "benchmark_output"}],
        "experiment": {"repeat_count": 3, "all_commands_ok": True},
    }


def test_loads_compiler_research_suite():
    suite = AutonomousRnDEvalHarness().load_suite(
        BACKEND_ROOT / "evals/autonomous_rnd/compiler_research_v1.json"
    )

    assert suite.id == "compiler_research_v1"
    assert len(suite.tasks) == 3
    assert all(task.trials == 3 for task in suite.tasks)


def test_rejects_duplicate_tasks_and_malformed_graders():
    raw_task = {
        "id": "duplicate",
        "graders": [{"type": "status_equals"}],
    }
    with pytest.raises(EvalDefinitionError, match="duplicate task ids"):
        AutonomousRnDEvalSuite.from_dict({"id": "suite", "tasks": [raw_task, raw_task]})

    with pytest.raises(EvalDefinitionError, match="graders must be JSON objects"):
        AutonomousRnDEvalTask.from_dict({"id": "bad", "graders": ["not-a-grader"]})


def test_grades_traceable_experiment_outcome():
    report = AutonomousRnDEvalHarness().grade_trial(_task(), _passing_outcome())

    assert report["passed"] is True
    assert report["score"] == 1.0


def test_missing_evidence_fails_required_traceability():
    outcome = _passing_outcome()
    outcome["evidence"] = []

    report = AutonomousRnDEvalHarness().grade_trial(_task(), outcome)

    assert report["passed"] is False
    assert report["required_passed"] is False
    assert report["score"] < 1.0


def test_unverified_or_rejected_evidence_cannot_support_verified_claims():
    for verification_status in ("unverified", "rejected"):
        outcome = _passing_outcome()
        outcome["evidence"][0]["verification_status"] = verification_status

        report = AutonomousRnDEvalHarness().grade_trial(_task(), outcome)

        assert report["passed"] is False
        assert report["required_passed"] is False


def test_verified_external_evidence_can_support_traceable_claim():
    outcome = _passing_outcome()
    outcome["evidence"][0]["verification_status"] = "verified"

    report = AutonomousRnDEvalHarness().grade_trial(_task(), outcome)

    assert report["passed"] is True


def test_verification_plan_coverage_requires_task_for_unresolved_evidence():
    task = AutonomousRnDEvalTask(
        id="coverage",
        name="Coverage",
        category="safety",
        prompt="Plan verification.",
        trials=1,
        pass_threshold=1.0,
        graders=[{"type": "verification_plan_coverage"}],
    )
    outcome = {
        "evidence": [
            {
                "id": "external-agent:request-1",
                "kind": "external_agent_response",
                "verification_status": "unverified",
            }
        ],
        "verification_plan": {"tasks": []},
    }

    report = AutonomousRnDEvalHarness().grade_trial(task, outcome)

    assert report["passed"] is False
    outcome["verification_plan"]["tasks"] = [
        {"evidence_id": "external-agent:request-1"}
    ]
    assert AutonomousRnDEvalHarness().grade_trial(task, outcome)["passed"] is True


def test_aggregates_capability_and_reliability_separately():
    task = _task(trials=2)
    suite = AutonomousRnDEvalSuite(
        id="suite",
        name="Suite",
        version=1,
        seed=100,
        tasks=[task],
    )
    failed = _passing_outcome()
    failed["experiment"] = {"repeat_count": 1, "all_commands_ok": True}

    report = AutonomousRnDEvalHarness().grade_suite_outcomes(
        suite, {task.id: [_passing_outcome(), failed]}
    )

    assert report["pass_at_k"] == 1.0
    assert report["pass_pow_k"] == 0.0
    assert report["tasks"][0]["pass_rate"] == 0.5


def test_missing_trials_cannot_pass_reliability_gate():
    task = _task(trials=3)
    suite = AutonomousRnDEvalSuite(
        id="suite",
        name="Suite",
        version=1,
        seed=100,
        tasks=[task],
    )

    report = AutonomousRnDEvalHarness().grade_suite_outcomes(
        suite, {task.id: [_passing_outcome()]}
    )

    assert report["pass_at_k"] == 1.0
    assert report["pass_pow_k"] == 0.0
    assert report["tasks"][0]["expected_trial_count"] == 3


@pytest.mark.asyncio
async def test_run_suite_uses_deterministic_seeds_and_captures_failures():
    task = _task(trials=2)
    suite = AutonomousRnDEvalSuite(
        id="suite",
        name="Suite",
        version=1,
        seed=700,
        tasks=[task],
    )
    observed = []

    async def executor(received_task, trial_index, seed):
        observed.append((received_task.id, trial_index, seed))
        if trial_index == 1:
            raise RuntimeError("model unavailable")
        return _passing_outcome()

    report = await AutonomousRnDEvalHarness().run_suite(suite, executor)

    assert observed == [("reproduce", 0, 700), ("reproduce", 1, 701)]
    assert report["tasks"][0]["passed_count"] == 1
    failed_outcome = report["tasks"][0]["trials"][1]["outcome"]
    assert failed_outcome["status"] == "error"
    assert "RuntimeError: model unavailable" in failed_outcome["error"]
