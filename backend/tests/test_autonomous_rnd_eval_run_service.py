import asyncio
from uuid import uuid4

from app.models.autonomous_rnd_eval_run import AutonomousRndEvalRun
from app.services.autonomous_rnd_eval_run_service import (
    EvalRunError,
    autonomous_rnd_eval_run_service,
)


def _report(*, tasks, mean_score=0.5, pass_at_k=1.0, pass_pow_k=0.5):
    return {
        "suite_id": "compiler_research_v1",
        "suite_name": "Compiler research",
        "suite_version": 1,
        "task_count": len(tasks),
        "trial_count": sum(int(task.get("trial_count") or 0) for task in tasks),
        "mean_score": mean_score,
        "pass_at_k": pass_at_k,
        "pass_pow_k": pass_pow_k,
        "tasks": tasks,
    }


def _task(task_id, *, pass_pow_k, pass_at_k=True, mean_score=1.0, trial_count=3):
    return {
        "task_id": task_id,
        "task_name": task_id,
        "trial_count": trial_count,
        "passed_count": trial_count if pass_pow_k else 1,
        "pass_at_k": pass_at_k,
        "pass_pow_k": pass_pow_k,
        "mean_score": mean_score,
    }


def _run(report, **overrides):
    values = {
        "id": uuid4(),
        "user_id": uuid4(),
        "suite_id": report["suite_id"],
        "suite_name": report["suite_name"],
        "suite_version": report["suite_version"],
        "source": "grade_jobs",
        "is_baseline": False,
        "task_count": report["task_count"],
        "trial_count": report["trial_count"],
        "mean_score": report["mean_score"],
        "pass_at_k": report["pass_at_k"],
        "pass_pow_k": report["pass_pow_k"],
        "report": report,
    }
    values.update(overrides)
    return AutonomousRndEvalRun(**values)


def test_compare_flags_task_that_lost_all_trial_reliability():
    baseline = _run(
        _report(
            tasks=[_task("reproduce", pass_pow_k=True), _task("null", pass_pow_k=True)],
            pass_pow_k=1.0,
        )
    )
    candidate = _run(
        _report(
            tasks=[
                _task("reproduce", pass_pow_k=False, mean_score=0.4),
                _task("null", pass_pow_k=True),
            ],
            pass_pow_k=0.5,
        )
    )

    comparison = autonomous_rnd_eval_run_service.compare(
        baseline=baseline, candidate=candidate
    )

    assert comparison["has_regression"] is True
    assert comparison["regressed_task_ids"] == ["reproduce"]
    assert comparison["improved_task_ids"] == []
    assert comparison["metrics"]["pass_pow_k"]["delta"] == -0.5
    statuses = {task["task_id"]: task["status"] for task in comparison["tasks"]}
    assert statuses == {"reproduce": "regressed", "null": "unchanged"}


def test_compare_reports_improvement_without_regression():
    baseline = _run(
        _report(tasks=[_task("reproduce", pass_pow_k=False)], pass_pow_k=0.0)
    )
    candidate = _run(
        _report(tasks=[_task("reproduce", pass_pow_k=True)], pass_pow_k=1.0)
    )

    comparison = autonomous_rnd_eval_run_service.compare(
        baseline=baseline, candidate=candidate
    )

    assert comparison["has_regression"] is False
    assert comparison["improved_task_ids"] == ["reproduce"]
    assert comparison["metrics"]["pass_pow_k"]["delta"] == 1.0


def test_compare_tracks_added_and_removed_tasks_across_suite_versions():
    baseline = _run(_report(tasks=[_task("reproduce", pass_pow_k=True)]))
    candidate_report = _report(tasks=[_task("reconcile", pass_pow_k=True)])
    candidate_report["suite_version"] = 2
    candidate = _run(candidate_report, suite_version=2)

    comparison = autonomous_rnd_eval_run_service.compare(
        baseline=baseline, candidate=candidate
    )

    statuses = {task["task_id"]: task["status"] for task in comparison["tasks"]}
    assert statuses == {"reproduce": "removed", "reconcile": "added"}
    assert comparison["suite_version_changed"] is True
    # Added and removed tasks are not reliability regressions on their own.
    assert comparison["regressed_task_ids"] == []


def test_compare_rejects_runs_from_different_suites():
    baseline = _run(_report(tasks=[_task("reproduce", pass_pow_k=True)]))
    other_report = _report(tasks=[_task("reproduce", pass_pow_k=True)])
    other_report["suite_id"] = "retrieval_quality_v1"
    candidate = _run(other_report, suite_id="retrieval_quality_v1")

    try:
        autonomous_rnd_eval_run_service.compare(baseline=baseline, candidate=candidate)
    except EvalRunError as exc:
        assert "different evaluation suites" in str(exc)
    else:  # pragma: no cover - guards against a silent contract change
        raise AssertionError("Expected EvalRunError for cross-suite comparison")


def test_records_run_and_promotes_single_baseline_per_suite(db_session, test_user):
    async def _exercise():
        first = await autonomous_rnd_eval_run_service.record_run(
            db_session,
            user_id=test_user.id,
            report=_report(tasks=[_task("reproduce", pass_pow_k=True)]),
            task_bindings={"reproduce": ["job-1", "job-2"]},
            label="  nightly  ",
        )
        second = await autonomous_rnd_eval_run_service.record_run(
            db_session,
            user_id=test_user.id,
            report=_report(tasks=[_task("reproduce", pass_pow_k=False)]),
        )
        await db_session.commit()

        assert first.label == "nightly"
        assert first.task_bindings == {"reproduce": ["job-1", "job-2"]}
        assert first.is_baseline is False

        await autonomous_rnd_eval_run_service.set_baseline(
            db_session, user_id=test_user.id, run_id=first.id
        )
        await db_session.commit()
        baseline = await autonomous_rnd_eval_run_service.get_baseline(
            db_session, user_id=test_user.id, suite_id="compiler_research_v1"
        )
        assert baseline.id == first.id

        # Promoting the second run must demote the first, leaving one anchor.
        await autonomous_rnd_eval_run_service.set_baseline(
            db_session, user_id=test_user.id, run_id=second.id
        )
        await db_session.commit()
        baseline = await autonomous_rnd_eval_run_service.get_baseline(
            db_session, user_id=test_user.id, suite_id="compiler_research_v1"
        )
        assert baseline.id == second.id

        runs = await autonomous_rnd_eval_run_service.list_runs(
            db_session, user_id=test_user.id, suite_id="compiler_research_v1"
        )
        assert len(runs) == 2
        assert sum(1 for run in runs if run.is_baseline) == 1

    asyncio.get_event_loop().run_until_complete(_exercise())


def test_record_run_rejects_report_without_suite_id(db_session, test_user):
    async def _exercise():
        report = _report(tasks=[_task("reproduce", pass_pow_k=True)])
        report["suite_id"] = ""
        try:
            await autonomous_rnd_eval_run_service.record_run(
                db_session, user_id=test_user.id, report=report
            )
        except EvalRunError as exc:
            assert "suite_id" in str(exc)
        else:  # pragma: no cover - guards against a silent contract change
            raise AssertionError("Expected EvalRunError for missing suite_id")

    asyncio.get_event_loop().run_until_complete(_exercise())
