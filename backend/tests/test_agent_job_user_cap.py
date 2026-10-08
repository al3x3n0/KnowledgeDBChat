"""One user's running agent jobs are capped where a worker claims them.

A running job holds a worker process for hours and nothing bounded one user's
share of them. The cap leaves further jobs pending; these tests are about the
ways "left pending" could go wrong -- a job that waits for ever, a job failed
for waiting, a job redelivered for ever after it ended.
"""

from __future__ import annotations

import asyncio
import types
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.core.config import settings
from app.models.agent_job import AgentJob, AgentJobStatus
from app.services import agent_job_user_cap
from app.tasks import agent_job_tasks
from tests.conftest import TestSessionLocal


def _run(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def _job(user_id, *, status=AgentJobStatus.PENDING.value, held=False) -> AgentJob:
    job = AgentJob(
        id=uuid4(),
        name="Capped job",
        goal="Wait its turn",
        job_type="research",
        user_id=user_id,
        status=status,
        config={},
        results={},
    )
    if held:
        job.status = AgentJobStatus.RUNNING.value
        job.execution_lease_owner = "worker-1"
        job.execution_lease_token = str(uuid4())
        job.execution_lease_expires_at = datetime.utcnow() + timedelta(minutes=5)
    return job


def _seed(db, *jobs):
    async def go():
        db.add_all(jobs)
        await db.commit()

    _run(go())


class _Executor:
    ran: list = []

    async def execute_job(self, *, job_id, db, progress_callback=None):
        _Executor.ran.append(job_id)
        row = (
            await db.execute(select(AgentJob).where(AgentJob.id == job_id))
        ).scalar_one()
        row.status = AgentJobStatus.COMPLETED.value
        await db.commit()
        return {"status": AgentJobStatus.COMPLETED.value, "progress": 100}


async def _noop(*_args, **_kwargs):
    return None


@pytest.fixture
def task(monkeypatch):
    monkeypatch.setattr(
        agent_job_tasks, "create_celery_session", lambda: TestSessionLocal
    )
    monkeypatch.setattr(agent_job_tasks, "AutonomousAgentExecutor", _Executor)
    monkeypatch.setattr(agent_job_tasks, "_publish_job_progress", _noop)
    monkeypatch.setattr(agent_job_tasks, "sync_follow_up_outcome_for_job", _noop)
    monkeypatch.setattr(
        agent_job_tasks,
        "current_task",
        types.SimpleNamespace(request=types.SimpleNamespace(id="task-1")),
    )
    monkeypatch.setattr(settings, "AGENT_JOBS_MAX_RUNNING_PER_USER", 2)
    monkeypatch.setattr(settings, "AGENT_JOBS_USER_CAP_RETRY_SECONDS", 60)
    _Executor.ran = []

    def execute(job):
        return _run(
            agent_job_tasks._execute_agent_job_async(str(job.id), str(job.user_id))
        )

    return execute


def _state(job):
    return agent_job_tasks._scheduler_state(job)


class TestAUserAtTheCap:
    def test_the_next_job_is_left_pending_and_not_run(self, db_session, task):
        user = uuid4()
        job = _job(user)
        _seed(db_session, _job(user, held=True), _job(user, held=True), job)

        outcome = task(job)
        _run(db_session.refresh(job))

        assert outcome["status"] == agent_job_tasks.USER_CAP_DEFERRED
        assert _Executor.ran == []
        assert job.status == AgentJobStatus.PENDING.value
        assert job.execution_lease_expires_at is None
        assert job.error is None

    def test_it_says_why_it_is_waiting(self, db_session, task):
        user = uuid4()
        job = _job(user)
        _seed(db_session, _job(user, held=True), _job(user, held=True), job)

        task(job)
        _run(db_session.refresh(job))

        state = _state(job)
        assert state["queue_reason"] == agent_job_user_cap.QUEUE_REASON
        assert state["user_running_jobs"] == 2
        assert state["user_job_cap"] == 2

    def test_waiting_adds_nothing_to_the_log(self, db_session, task):
        # It is tried every minute; a log entry per attempt would grow the row
        # for as long as the user's other jobs run.
        user = uuid4()
        job = _job(user)
        _seed(db_session, _job(user, held=True), _job(user, held=True), job)

        task(job)
        task(job)
        _run(db_session.refresh(job))

        assert not job.execution_log

    def test_a_job_marked_running_goes_back_to_pending(self, db_session, task):
        # Left "running" with no worker, the sweep would read its silence as a
        # dead worker and eventually fail it.
        user = uuid4()
        job = _job(user, status=AgentJobStatus.RUNNING.value)
        job.celery_task_id = "an-earlier-delivery"
        _seed(db_session, _job(user, held=True), _job(user, held=True), job)

        task(job)
        _run(db_session.refresh(job))

        assert job.status == AgentJobStatus.PENDING.value
        assert job.celery_task_id is None


class TestWhoIsNotHeld:
    def test_a_user_under_the_cap(self, db_session, task):
        user = uuid4()
        job = _job(user)
        _seed(db_session, _job(user, held=True), job)

        task(job)

        assert _Executor.ran == [job.id]

    def test_somebody_elses_jobs_do_not_count(self, db_session, task):
        other = uuid4()
        job = _job(uuid4())
        _seed(db_session, _job(other, held=True), _job(other, held=True), job)

        task(job)

        assert _Executor.ran == [job.id]

    def test_a_job_whose_worker_died_does_not_count(self, db_session, task):
        # Still "running" in the status column, but nobody is running it.
        user = uuid4()
        dead = _job(user, held=True)
        dead.execution_lease_expires_at = datetime.utcnow() - timedelta(minutes=5)
        job = _job(user)
        _seed(db_session, dead, _job(user, held=True), job)

        task(job)

        assert _Executor.ran == [job.id]

    def test_no_cap_when_it_is_zero(self, db_session, task, monkeypatch):
        monkeypatch.setattr(settings, "AGENT_JOBS_MAX_RUNNING_PER_USER", 0)
        user = uuid4()
        job = _job(user)
        _seed(db_session, _job(user, held=True), _job(user, held=True), job)

        task(job)

        assert _Executor.ran == [job.id]

    def test_running_clears_the_reason_it_waited(self, db_session, task):
        user = uuid4()
        blocker = _job(user, held=True)
        job = _job(user)
        _seed(db_session, blocker, _job(user, held=True), job)
        task(job)

        blocker.execution_lease_expires_at = None
        _run(db_session.commit())
        task(job)
        _run(db_session.refresh(job))

        assert _Executor.ran == [job.id]
        state = _state(job)
        assert state.get("queue_reason") != agent_job_user_cap.QUEUE_REASON
        assert "deferred_until" not in state


class TestWhatMustNotComeBackForEver:
    """Holding a job means sending it again; these must simply end."""

    @pytest.mark.parametrize(
        "status",
        [
            AgentJobStatus.COMPLETED.value,
            AgentJobStatus.FAILED.value,
            AgentJobStatus.PAUSED.value,
        ],
    )
    def test_a_job_that_is_not_waiting_to_run(self, db_session, task, status):
        user = uuid4()
        job = _job(user, status=status)
        _seed(db_session, _job(user, held=True), _job(user, held=True), job)

        outcome = task(job)

        assert outcome["status"] == "lease_conflict"

    def test_a_second_delivery_of_a_job_a_worker_holds(self, db_session, task):
        user = uuid4()
        job = _job(user, held=True)
        _seed(db_session, _job(user, held=True), _job(user, held=True), job)

        outcome = task(job)

        assert outcome["status"] == "lease_conflict"


class TestRedelivery:
    def _held_outcome(self, monkeypatch, job_id):
        async def held(*_args, **_kwargs):
            return {
                "status": agent_job_tasks.USER_CAP_DEFERRED,
                "job_id": job_id,
                "retry_in_seconds": 60,
            }

        monkeypatch.setattr(agent_job_tasks, "_execute_agent_job_async", held)

    def test_a_held_job_is_sent_again_later(self, monkeypatch):
        job_id, user_id = str(uuid4()), str(uuid4())
        self._held_outcome(monkeypatch, job_id)
        sent = []
        monkeypatch.setattr(
            agent_job_tasks.execute_agent_job_task,
            "apply_async",
            lambda **kwargs: sent.append(kwargs),
        )

        agent_job_tasks.execute_agent_job_task.run(job_id, user_id)

        assert len(sent) == 1
        assert sent[0]["args"] == [job_id, user_id]
        assert 60 <= sent[0]["countdown"] <= 75

    def test_a_broker_that_refuses_does_not_fail_the_job(self, monkeypatch):
        # The job has not started; the sweep queues it once its wait lapses.
        job_id, user_id = str(uuid4()), str(uuid4())
        self._held_outcome(monkeypatch, job_id)

        def refuse(**_kwargs):
            raise ConnectionError("broker is away")

        monkeypatch.setattr(
            agent_job_tasks.execute_agent_job_task, "apply_async", refuse
        )

        agent_job_tasks.execute_agent_job_task.run(job_id, user_id)


class TestTheSweepAndAWaitingJob:
    def _waiting(self, db_session, *, deferred_minutes_ago):
        job = _job(uuid4())
        job.created_at = datetime.utcnow() - timedelta(minutes=180)
        then = datetime.utcnow() - timedelta(minutes=deferred_minutes_ago)
        agent_job_tasks._write_scheduler_state(
            job,
            agent_job_user_cap.waiting_state(now=then, running=2, cap=2, retry_in=60),
        )
        _seed(db_session, job)
        return job

    def _sweep(self, monkeypatch):
        monkeypatch.setattr(
            agent_job_tasks, "create_celery_session", lambda: TestSessionLocal
        )
        sent = []
        monkeypatch.setattr(
            agent_job_tasks.execute_agent_job_task,
            "delay",
            lambda *args, **kwargs: sent.append(args),
        )
        agent_job_tasks.check_stalled_agent_jobs(undispatched_minutes=60)
        return sent

    def test_a_job_waiting_its_turn_is_not_one_nobody_picked_up(
        self, db_session, monkeypatch
    ):
        # Three hours old with no task id is what "never dispatched" looks
        # like, and three such requeues fail the job.
        job = self._waiting(db_session, deferred_minutes_ago=1)

        sent = self._sweep(monkeypatch)
        _run(db_session.refresh(job))

        assert sent == []
        assert agent_job_tasks.count_undispatched_requeues(job) == 0

    def test_a_wait_whose_redelivery_never_came_is_queued_again(
        self, db_session, monkeypatch
    ):
        job = self._waiting(db_session, deferred_minutes_ago=30)

        sent = self._sweep(monkeypatch)

        assert sent and str(job.id) in sent[0]
