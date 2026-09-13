"""Unit tests for quick-start relaunch dispatch orchestration."""

from types import SimpleNamespace

import pytest

from app.api.endpoints import agent_jobs
from app.modules.autonomy.application.relaunch_dispatcher import (
    QuickStartRelaunchDispatcher,
    RelaunchOutcome,
    RelaunchRoute,
)
from app.schemas.agent_job import AgentJobActionRequest


def _dispatcher(*, calls):
    def build(job, **kwargs):
        calls.append(("build", job.id, kwargs))
        return {"job_id": job.id, **kwargs}

    async def launch(request, db, current_user):
        calls.append(("launch", request, db, current_user.id))
        return SimpleNamespace(id="new-job")

    def recovery(job):
        calls.append(("recovery", job.id))
        return {"retry_reason": "verification failed"}

    route = RelaunchRoute(builder=build, launcher=launch)
    return QuickStartRelaunchDispatcher(
        routes={"quick_start_claude_backend": route},
        refined_repo_route=RelaunchRoute(
            builder=build,
            launcher=launch,
            builder_kwargs={"retry_strategy": "refined_retry"},
        ),
        recovery_extractor=recovery,
    )


@pytest.mark.asyncio
async def test_relaunch_dispatches_matching_builder_and_launcher():
    calls = []
    dispatcher = _dispatcher(calls=calls)
    job = SimpleNamespace(
        id="old-job",
        config={"launch_mode": " QUICK_START_CLAUDE_BACKEND "},
    )
    user = SimpleNamespace(id="user-1")

    outcome = await dispatcher.relaunch(job, db="db", current_user=user)

    assert outcome is not None
    assert outcome.job.id == "new-job"
    assert outcome.launch_mode == "quick_start_claude_backend"
    assert outcome.recovery_strategy is None
    assert calls == [
        ("build", "old-job", {}),
        ("launch", {"job_id": "old-job"}, "db", "user-1"),
    ]


@pytest.mark.asyncio
async def test_relaunch_returns_none_for_unknown_or_invalid_route():
    calls = []
    dispatcher = _dispatcher(calls=calls)
    unknown_job = SimpleNamespace(id="unknown", config={"launch_mode": "manual"})

    assert (
        await dispatcher.relaunch(
            unknown_job,
            db="db",
            current_user=SimpleNamespace(id="user-1"),
        )
        is None
    )
    assert calls == []


@pytest.mark.asyncio
async def test_refined_repo_retry_preserves_strategy_and_recovery():
    calls = []
    dispatcher = _dispatcher(calls=calls)
    job = SimpleNamespace(
        id="repo-job",
        config={"launch_mode": "quick_start_repo_bug_triage"},
    )
    user = SimpleNamespace(id="user-2")

    outcome = await dispatcher.refined_repo_retry(
        job,
        db="db",
        current_user=user,
    )

    assert outcome is not None
    assert outcome.launch_mode == "quick_start_repo_bug_triage"
    assert outcome.recovery_strategy == "refined_retry"
    assert outcome.recovery == {"retry_reason": "verification failed"}
    assert calls == [
        ("build", "repo-job", {"retry_strategy": "refined_retry"}),
        (
            "launch",
            {"job_id": "repo-job", "retry_strategy": "refined_retry"},
            "db",
            "user-2",
        ),
        ("recovery", "repo-job"),
    ]


class _ActionDb:
    def __init__(self):
        self.commits = 0

    async def commit(self):
        self.commits += 1


def _action_job():
    job = SimpleNamespace(
        id="old-job",
        name="Old job",
        status="completed",
        user_id="user-1",
        config={"launch_mode": "quick_start_repo_bug_triage"},
        results={},
        execution_log=[],
    )

    def add_log_entry(entry):
        job.execution_log.append(entry)

    job.add_log_entry = add_log_entry
    return job


@pytest.mark.asyncio
async def test_job_relaunch_action_applies_dispatcher_outcome(monkeypatch):
    new_job = SimpleNamespace(id="new-job")

    class _Dispatcher:
        async def relaunch(self, job, *, db, current_user):
            return RelaunchOutcome(
                job=new_job,
                launch_mode="quick_start_repo_bug_triage",
                recovery_strategy="clean_relaunch",
            )

    monkeypatch.setattr(
        agent_jobs,
        "_quick_start_relaunch_dispatcher",
        _Dispatcher(),
    )
    job = _action_job()
    db = _ActionDb()

    result = await agent_jobs._perform_job_action(
        job,
        AgentJobActionRequest(action="relaunch", checkpoint_note="Try cleanly"),
        db=db,
        current_user=SimpleNamespace(id="user-1"),
    )

    assert result is new_job
    assert db.commits == 1
    assert job.execution_log[-1]["phase"] == "relaunch_requested"
    intervention = job.results["execution_strategy"]["operator_interventions"][-1]
    assert intervention["action"] == "relaunch"
    assert intervention["metadata"]["recovery_strategy"] == "clean_relaunch"


@pytest.mark.asyncio
async def test_job_restart_action_applies_refined_retry_outcome(monkeypatch):
    new_job = SimpleNamespace(id="retry-job")

    class _Dispatcher:
        async def refined_repo_retry(self, job, *, db, current_user):
            return RelaunchOutcome(
                job=new_job,
                launch_mode="quick_start_repo_bug_triage",
                recovery_strategy="refined_retry",
                recovery={"retry_reason": "verification failed"},
            )

    monkeypatch.setattr(
        agent_jobs,
        "_quick_start_relaunch_dispatcher",
        _Dispatcher(),
    )
    job = _action_job()
    db = _ActionDb()

    result = await agent_jobs._perform_job_action(
        job,
        AgentJobActionRequest(action="restart", checkpoint_note="Refine the plan"),
        db=db,
        current_user=SimpleNamespace(id="user-1"),
    )

    assert result is new_job
    assert db.commits == 1
    assert job.execution_log[-1]["phase"] == "restart_requested"
    intervention = job.results["execution_strategy"]["operator_interventions"][-1]
    assert intervention["action"] == "restart"
    assert intervention["metadata"]["recovery_strategy"] == "refined_retry"
    assert intervention["metadata"]["retry_reason"] == "verification failed"
