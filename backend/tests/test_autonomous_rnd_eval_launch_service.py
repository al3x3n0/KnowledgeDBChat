import asyncio

import pytest

from app.core.config import settings
from app.models.agent_job import AgentJobStatus
from app.models.autonomous_rnd_eval_launch import (
    EVAL_LAUNCH_STATUS_COMPLETED,
    EVAL_LAUNCH_STATUS_RUNNING,
)
from app.services.autonomous_rnd_eval_launch_service import (
    EvalLaunchError,
    autonomous_rnd_eval_launch_service,
)
from app.services.autonomous_rnd_eval_service import autonomous_rnd_eval_harness


@pytest.fixture
def launch_enabled(monkeypatch):
    monkeypatch.setattr(settings, "AUTONOMOUS_RND_EVAL_LAUNCH_ENABLED", True)
    monkeypatch.setattr(settings, "AUTONOMOUS_RND_EVAL_MAX_TRIAL_JOBS", 30)
    return settings


def _suite():
    return autonomous_rnd_eval_harness.load_builtin_suite("compiler_research_v1")


def _passing_results(index):
    return {
        "evaluation_outcome": {
            "claims": [{"id": f"claim-{index}", "evidence_ids": [f"evidence-{index}"]}],
            "evidence": [{"id": f"evidence-{index}", "kind": "benchmark_output"}],
            "experiment": {"repeat_count": 3, "all_commands_ok": True},
        },
        "execution_strategy": {
            "step_events": [
                {"type": "step_completed", "tool": "run_experiment", "iteration": 2}
            ]
        },
    }


def test_launch_is_disabled_by_default(db_session, test_user, monkeypatch):
    monkeypatch.setattr(settings, "AUTONOMOUS_RND_EVAL_LAUNCH_ENABLED", False)

    async def _exercise():
        with pytest.raises(EvalLaunchError) as excinfo:
            await autonomous_rnd_eval_launch_service.launch(
                db_session, user_id=test_user.id, suite=_suite()
            )
        assert excinfo.value.status_code == 403

    asyncio.get_event_loop().run_until_complete(_exercise())


def test_launch_refuses_to_exceed_the_trial_job_cap(
    db_session, test_user, launch_enabled, monkeypatch
):
    monkeypatch.setattr(settings, "AUTONOMOUS_RND_EVAL_MAX_TRIAL_JOBS", 4)

    async def _exercise():
        with pytest.raises(EvalLaunchError) as excinfo:
            await autonomous_rnd_eval_launch_service.launch(
                db_session,
                user_id=test_user.id,
                suite=_suite(),
                trials_override=3,
            )
        assert "above the AUTONOMOUS_RND_EVAL_MAX_TRIAL_JOBS cap" in str(excinfo.value)

    asyncio.get_event_loop().run_until_complete(_exercise())


def test_launch_creates_one_job_per_task_trial_and_binds_them(
    db_session, test_user, launch_enabled
):
    async def _exercise():
        suite = _suite()
        launch, jobs = await autonomous_rnd_eval_launch_service.launch(
            db_session,
            user_id=test_user.id,
            suite=suite,
            trials_override=2,
            label="nightly",
        )
        await db_session.commit()

        assert len(jobs) == len(suite.tasks) * 2
        assert launch.job_count == len(jobs)
        assert launch.status == EVAL_LAUNCH_STATUS_RUNNING
        assert set(launch.task_bindings) == {task.id for task in suite.tasks}
        for task in suite.tasks:
            assert len(launch.task_bindings[task.id]) == 2

        binding = jobs[0].config["autonomous_rnd_eval"]
        assert binding["launch_id"] == str(launch.id)
        assert binding["suite_id"] == suite.id
        assert binding["trial_index"] == 0

    asyncio.get_event_loop().run_until_complete(_exercise())


def test_finalize_waits_until_every_trial_settles(
    db_session, test_user, launch_enabled
):
    async def _exercise():
        launch, jobs = await autonomous_rnd_eval_launch_service.launch(
            db_session, user_id=test_user.id, suite=_suite(), trials_override=1
        )
        await db_session.commit()

        jobs[0].status = AgentJobStatus.COMPLETED.value
        await db_session.commit()

        progress = await autonomous_rnd_eval_launch_service.progress(
            db_session, launch=launch
        )
        assert progress["is_ready"] is False
        assert progress["settled_count"] == 1

        run = await autonomous_rnd_eval_launch_service.finalize(
            db_session, launch=launch
        )
        assert run is None
        assert launch.status == EVAL_LAUNCH_STATUS_RUNNING

    asyncio.get_event_loop().run_until_complete(_exercise())


def test_finalize_grades_settled_trials_into_a_persisted_run(
    db_session, test_user, launch_enabled
):
    async def _exercise():
        suite = _suite()
        launch, jobs = await autonomous_rnd_eval_launch_service.launch(
            db_session, user_id=test_user.id, suite=suite, trials_override=1
        )
        await db_session.commit()

        for index, job in enumerate(jobs):
            job.status = AgentJobStatus.COMPLETED.value
            job.results = _passing_results(index)
            job.output_artifacts = [
                {"type": "compiler_logs"},
                {"type": "ir_or_codegen_artifacts"},
                {"type": "benchmark_output"},
            ]
        await db_session.commit()

        run = await autonomous_rnd_eval_launch_service.finalize(
            db_session, launch=launch
        )
        await db_session.commit()

        assert run is not None
        assert launch.status == EVAL_LAUNCH_STATUS_COMPLETED
        assert launch.run_id == run.id
        assert launch.completed_at is not None
        assert run.source == "launch"
        assert run.suite_id == suite.id
        assert run.trial_count == len(jobs)
        assert run.task_bindings == launch.task_bindings

        # Finalizing again must not create a second run for the same launch.
        repeat = await autonomous_rnd_eval_launch_service.finalize(
            db_session, launch=launch
        )
        assert repeat.id == run.id

    asyncio.get_event_loop().run_until_complete(_exercise())


def test_finalize_counts_a_deleted_trial_job_as_a_failed_trial(
    db_session, test_user, launch_enabled
):
    async def _exercise():
        launch, jobs = await autonomous_rnd_eval_launch_service.launch(
            db_session, user_id=test_user.id, suite=_suite(), trials_override=1
        )
        await db_session.commit()

        for job in jobs[1:]:
            job.status = AgentJobStatus.COMPLETED.value
        await db_session.delete(jobs[0])
        await db_session.commit()

        progress = await autonomous_rnd_eval_launch_service.progress(
            db_session, launch=launch
        )
        assert progress["missing_count"] == 1
        assert progress["is_ready"] is True

        run = await autonomous_rnd_eval_launch_service.finalize(
            db_session, launch=launch
        )
        await db_session.commit()

        assert run is not None
        # The missing trial is graded as an error outcome, so its task cannot
        # report all-trial reliability.
        assert run.pass_pow_k < 1.0
        assert run.trial_count == launch.job_count

    asyncio.get_event_loop().run_until_complete(_exercise())
