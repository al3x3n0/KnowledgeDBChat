"""A run is scored from all its jobs, and an eval waits for all of them.

The eval harness graded the trial job alone. A trial that proposes a plan
does its work in the stages; a stage short of its contract is retried by a
new job. Grading the first job graded the planning, or the attempt that was
replaced, and the launch finalised while the work was still running.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.core.config import settings
from app.models.agent_job import AgentJob, AgentJobStatus
from app.services import agent_run_scorecard, agent_run_waiting
from app.services.autonomous_rnd_eval_launch_service import (
    autonomous_rnd_eval_launch_service,
)
from app.services.autonomous_rnd_eval_service import (
    AutonomousRnDEvalSuite,
    autonomous_rnd_eval_harness,
)

pytestmark = pytest.mark.unit

T0 = datetime(2026, 10, 29, 12, 0, tzinfo=timezone.utc)


def _job(
    *,
    minutes=0,
    status="completed",
    phase=None,
    config=None,
    results=None,
    log=None,
    root=None,
):
    return SimpleNamespace(
        id=uuid4(),
        root_job_id=root,
        parent_job_id=root,
        name="job",
        status=status,
        current_phase=phase,
        config=config or {},
        results=results or {},
        execution_log=log or [],
        chain_config={},
        chain_triggered=False,
        created_at=T0 + timedelta(minutes=minutes),
        started_at=T0 + timedelta(minutes=minutes),
        completed_at=(
            T0 + timedelta(minutes=minutes + 5) if status == "completed" else None
        ),
    )


def _contract(satisfied, required=("benchmark_measurement",), missing=()):
    return {
        "enabled": True,
        "satisfied": satisfied,
        "missing": list(missing),
        "contract": {"required_finding_types": list(required)},
    }


def _step(tool, success=True):
    return {"phase": "iteration_complete", "action": tool, "success": success}


def _measurement(subject):
    return {"type": "benchmark_measurement", "subject": subject, "title": subject}


class TestFinalJobs:
    def test_an_attempt_a_retry_names_is_not_final(self):
        first = _job(status="paused", phase="stage_retried_automatically")
        retry = _job(
            minutes=10, status="running", config={"auto_retry_of": str(first.id)}
        )
        assert agent_run_waiting.final_jobs([first, retry]) == [retry]

    def test_a_restarted_stage_keeps_only_its_newest_job(self):
        old = _job(config={"pipeline_stage": "bench"})
        new = _job(minutes=20, config={"pipeline_stage": "bench"})
        other = _job(minutes=5, config={"pipeline_stage": "report"})
        assert agent_run_waiting.final_jobs([old, other, new]) == [other, new]

    def test_a_paused_attempt_does_not_hide_its_running_retry(self):
        """Counted, the paused attempt read as waiting on a person."""
        first = _job(status="paused", phase="stage_retried_automatically")
        retry = _job(
            minutes=10, status="running", config={"auto_retry_of": str(first.id)}
        )
        assert agent_run_waiting.waiting_on([first, retry]) == "job"

    def test_a_job_still_writing_up_is_not_over(self):
        """Graded at 'completed', a trial's conclusion was minutes away."""
        job = _job()
        job.execution_lease_expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=2
        )
        assert agent_run_waiting.waiting_on([job]) == "job"
        job.execution_lease_expires_at = None
        assert agent_run_waiting.waiting_on([job]) is None

    def test_a_lapsed_lease_does_not_hold_a_run_open(self):
        job = _job()
        job.execution_lease_expires_at = datetime.now(timezone.utc) - timedelta(
            minutes=1
        )
        assert agent_run_waiting.waiting_on([job]) is None


class TestScore:
    def test_the_stages_decide_not_the_job_that_planned_them(self):
        planner = _job(
            results={
                "conclusion": {"answer": "Proposed a plan", "generated_by": "llm"}
            },
            log=[_step("propose_pipeline")],
        )
        stage = _job(
            minutes=10,
            root=planner.id,
            config={"pipeline_stage": "bench"},
            results={
                "goal_contract": _contract(True),
                "findings": [_measurement("fastmod"), _measurement("plain %")],
                "conclusion": {"answer": "2.3x faster", "generated_by": "llm"},
            },
            log=[_step("search_documents"), _step("benchmark_c_snippet")],
        )
        card = agent_run_scorecard.score([planner, stage], now=T0 + timedelta(hours=1))

        assert card["settled"] is True
        assert card["contract_met"] is True
        assert card["benchmark"]["baselined"] is True
        assert card["benchmark"]["paired"] is False
        assert card["conclusion"]["answered"] is True
        assert card["conclusion"]["job_id"] == str(stage.id)
        assert card["iterations"] == 3
        assert card["iterations_to_first_evidence"] == 3
        assert card["first_evidence_tool"] == "benchmark_c_snippet"
        assert card["wall_seconds"] == 15 * 60

    def test_a_retried_stage_is_judged_by_its_retry_and_its_iterations_count(self):
        first = _job(
            status="paused",
            phase="stage_retried_automatically",
            results={
                "goal_contract": _contract(False, missing=["benchmark_measurement"])
            },
            log=[_step("search_documents")] * 4,
        )
        retry = _job(
            minutes=30,
            config={"auto_retry_of": str(first.id)},
            results={
                "goal_contract": _contract(True),
                "findings": [_measurement("only one")],
            },
            log=[_step("benchmark_c_snippet")],
        )
        card = agent_run_scorecard.score([first, retry], now=T0 + timedelta(hours=1))

        assert card["superseded_attempts"] == 1
        assert card["contract_met"] is True
        assert card["iterations"] == 5
        assert card["iterations_to_first_evidence"] == 5
        assert card["benchmark"]["baselined"] is False, "one subject has no baseline"

    def test_asking_for_nothing_is_not_meeting_a_contract(self):
        card = agent_run_scorecard.score([_job()])
        assert card["contract_met"] is None
        assert card["benchmark"]["baselined"] is None, "nothing was timed"

    def test_a_listed_conclusion_is_not_an_answer(self):
        card = agent_run_scorecard.score(
            [
                _job(
                    results={
                        "conclusion": {
                            "answer": None,
                            "generated_by": "findings_only",
                        }
                    }
                )
            ]
        )
        assert card["conclusion"] == {
            "answered": False,
            "generated_by": "findings_only",
            "confidence": None,
            "job_id": card["conclusion"]["job_id"],
        }

    def test_a_paired_comparison_says_whether_it_claimed_a_difference(self):
        job = _job(
            results={
                "findings": [
                    _measurement("a"),
                    _measurement("b"),
                    {
                        "type": "benchmark_comparison",
                        "subject": "a vs b",
                        "verdict": "indistinguishable",
                    },
                ]
            }
        )
        card = agent_run_scorecard.score([job])
        assert card["benchmark"]["paired"] is True
        assert card["benchmark"]["claimed_difference"] is False

    def test_an_inherited_finding_is_counted_where_it_was_made(self):
        made = _measurement("fastmod")
        parent = _job(results={"findings": [made]})
        child = _job(
            minutes=10,
            root=parent.id,
            results={"findings": [{**made, "inherited": True}]},
        )
        card = agent_run_scorecard.score([parent, child])
        assert card["benchmark"]["measurements"] == 1

    def test_a_run_waiting_on_a_plans_approval_is_not_over(self):
        card = agent_run_scorecard.score(
            [_job(status="paused", phase="awaiting_approval")]
        )
        assert card["settled"] is False
        assert card["waiting_on"] == "approval"

    def test_a_blocked_run_is_an_outcome(self):
        card = agent_run_scorecard.score(
            [_job(status="paused", phase="blocked_needs_input")]
        )
        assert card["settled"] is True


class TestSummary:
    def test_rates_count_only_trials_the_question_applies_to(self):
        timed = {
            "jobs": 1,
            "contract_met": True,
            "benchmark": {"baselined": True},
            "conclusion": {"answered": True},
            "iterations": 4,
            "iterations_to_first_evidence": 2,
            "wall_seconds": 100.0,
        }
        untimed = {
            "jobs": 1,
            "contract_met": False,
            "benchmark": {"baselined": None},
            "conclusion": {"answered": False},
            "iterations": 10,
            "iterations_to_first_evidence": None,
            "wall_seconds": 300.0,
        }
        summary = agent_run_scorecard.summarize([timed, untimed])

        assert summary["trials"] == 2
        assert summary["contract_met"] == {"rate": 0.5, "n": 2}
        assert summary["baselined"] == {"rate": 1.0, "n": 1}
        assert summary["iterations_to_first_evidence"] == {"median": 2.0, "n": 1}
        assert summary["never_produced_evidence"] == 1
        assert summary["wall_seconds"]["median"] == 200.0


def test_the_report_carries_run_metrics_and_graders_read_the_scorecard():
    suite = AutonomousRnDEvalSuite.from_dict(
        {
            "id": "s",
            "tasks": [
                {
                    "id": "t",
                    "trials": 1,
                    "job_type": "research",
                    "config": {"goal_contract": {"required_finding_types": ["x"]}},
                    "graders": [
                        {
                            "id": "met",
                            "type": "field_equals",
                            "path": "run.contract_met",
                            "expected": True,
                        }
                    ],
                }
            ],
        }
    )
    assert suite.tasks[0].job_type == "research"
    assert suite.tasks[0].config == {"goal_contract": {"required_finding_types": ["x"]}}

    card = agent_run_scorecard.score([_job(results={"goal_contract": _contract(True)})])
    report = autonomous_rnd_eval_harness.grade_suite_outcomes(
        suite, {"t": [{"status": "completed", "run": card}]}
    )
    assert report["pass_pow_k"] == 1.0
    assert report["run_metrics"]["contract_met"] == {"rate": 1.0, "n": 1}
    assert report["tasks"][0]["run_metrics"]["trials"] == 1


def test_the_rigour_suite_loads_and_names_only_scorecard_paths():
    suite = autonomous_rnd_eval_harness.load_builtin_suite("agent_rigour_v1")
    assert len(suite.tasks) >= 3
    for task in suite.tasks:
        assert task.config and task.config.get("goal_contract"), task.id
        for grader in task.graders:
            assert grader["path"].startswith("run."), (task.id, grader["id"])


@pytest.fixture
def launch_enabled(monkeypatch):
    monkeypatch.setattr(settings, "AUTONOMOUS_RND_EVAL_LAUNCH_ENABLED", True)
    monkeypatch.setattr(settings, "AUTONOMOUS_RND_EVAL_MAX_TRIAL_JOBS", 30)


def test_a_launch_waits_for_the_plan_its_trial_proposed(
    db_session, test_user, launch_enabled
):
    async def _exercise():
        # finalize() reloads the suite by id, so this is the built-in one.
        suite = autonomous_rnd_eval_harness.load_builtin_suite("agent_rigour_v1")
        launch, jobs = await autonomous_rnd_eval_launch_service.launch(
            db_session, user_id=test_user.id, suite=suite, trials_override=1
        )
        await db_session.commit()
        task = suite.tasks[0]
        (trial_id,) = launch.task_bindings[task.id]
        trial = next(job for job in jobs if str(job.id) == trial_id)
        assert trial.job_type == task.job_type
        assert trial.config["goal_contract"] == task.config["goal_contract"]

        for job in jobs:
            job.status = AgentJobStatus.COMPLETED.value
        stage = AgentJob(
            user_id=test_user.id,
            name="bench",
            goal="benchmark it",
            job_type="research",
            status=AgentJobStatus.RUNNING.value,
            root_job_id=trial.id,
            parent_job_id=trial.id,
            config={"pipeline_stage": "bench"},
        )
        db_session.add(stage)
        await db_session.commit()

        progress = await autonomous_rnd_eval_launch_service.progress(
            db_session, launch=launch
        )
        assert progress["is_ready"] is False, "the trial ended, its plan has not"
        assert progress["settled_count"] == len(jobs) - 1

        stage.status = AgentJobStatus.COMPLETED.value
        stage.results = {"goal_contract": _contract(True)}
        await db_session.commit()

        run = await autonomous_rnd_eval_launch_service.finalize(
            db_session, launch=launch
        )
        assert run is not None
        report = next(t for t in run.report["tasks"] if t["task_id"] == task.id)
        (trial_report,) = report["trials"]
        card = trial_report["outcome"]["run"]
        assert card["jobs"] == 2
        assert card["contract_met"] is True, "the stage met it; the trial had none"
        assert run.report["run_metrics"]["trials"] == len(jobs)

    asyncio.get_event_loop().run_until_complete(_exercise())
