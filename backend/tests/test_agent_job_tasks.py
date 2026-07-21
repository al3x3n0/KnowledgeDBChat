from __future__ import annotations

import asyncio
import sys
import types
from datetime import datetime, timedelta
from uuid import uuid4

from tests.conftest import TestSessionLocal
from sqlalchemy import select

from app.models.agent_job import AgentJob, AgentJobCheckpoint, AgentJobStatus
from app.tasks import agent_job_tasks


def _run(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def _patch_celery_session_factory(monkeypatch):
    monkeypatch.setattr(agent_job_tasks, "create_celery_session", lambda: TestSessionLocal)


def _make_job(*, status: str, schedule_type: str | None = None) -> AgentJob:
    return AgentJob(
        id=uuid4(),
        name="Task Lifecycle Job",
        goal="Exercise lifecycle task logic",
        job_type="monitor",
        user_id=uuid4(),
        status=status,
        schedule_type=schedule_type,
        config={"interval_minutes": 15},
        results={},
        progress=13,
        current_phase="running",
        phase_details="Working",
        started_at=datetime(2026, 3, 16, 10, 0, 0),
        completed_at=datetime(2026, 3, 16, 10, 1, 0),
        last_activity_at=datetime(2026, 3, 16, 10, 2, 0),
        celery_task_id="celery-123",
    )


def test_process_scheduled_agent_jobs_resets_and_requeues_due_once_job(db_session, monkeypatch):
    _patch_celery_session_factory(monkeypatch)

    job = _make_job(status=AgentJobStatus.COMPLETED.value, schedule_type="once")
    job.next_run_at = datetime.utcnow() - timedelta(minutes=5)
    job.error = "previous error"
    job.progress = 57

    _run(_seed_job(db_session, job))

    queued_jobs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        agent_job_tasks.execute_agent_job_task,
        "delay",
        lambda job_id, user_id: queued_jobs.append((job_id, user_id)),
    )

    agent_job_tasks.process_scheduled_agent_jobs()
    _run(db_session.refresh(job))

    assert queued_jobs == [(str(job.id), str(job.user_id))]
    assert job.status == AgentJobStatus.PENDING.value
    assert job.progress == 0
    assert job.current_phase is None
    assert job.phase_details is None
    assert job.started_at is None
    assert job.completed_at is None
    assert job.error is None
    assert job.celery_task_id is None
    assert job.last_activity_at is not None
    assert job.next_run_at is None


def test_process_scheduled_agent_jobs_resets_and_requeues_due_recurring_job(db_session, monkeypatch):
    _patch_celery_session_factory(monkeypatch)

    job = _make_job(status=AgentJobStatus.COMPLETED.value, schedule_type="recurring")
    job.schedule_cron = "0 * * * *"
    job.next_run_at = datetime.utcnow() - timedelta(minutes=5)

    _run(_seed_job(db_session, job))

    fixed_next_run = datetime(2026, 3, 16, 11, 0, 0)
    croniter_mod = types.ModuleType("croniter")

    class _FakeCronIter:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_next(self, _typ):
            return fixed_next_run

    croniter_mod.croniter = _FakeCronIter
    monkeypatch.setitem(sys.modules, "croniter", croniter_mod)

    queued_jobs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        agent_job_tasks.execute_agent_job_task,
        "delay",
        lambda job_id, user_id: queued_jobs.append((job_id, user_id)),
    )

    agent_job_tasks.process_scheduled_agent_jobs()
    _run(db_session.refresh(job))

    assert queued_jobs == [(str(job.id), str(job.user_id))]
    assert job.status == AgentJobStatus.PENDING.value
    assert job.next_run_at == fixed_next_run


def test_resume_paused_agent_jobs_requeues_when_under_limits(db_session, monkeypatch):
    _patch_celery_session_factory(monkeypatch)

    job = _make_job(status=AgentJobStatus.PAUSED.value)
    job.last_activity_at = datetime.utcnow() - timedelta(minutes=10)
    job.max_iterations = 20
    job.max_tool_calls = 20
    job.max_llm_calls = 20
    job.iteration = 3
    job.tool_calls_used = 2
    job.llm_calls_used = 1
    job.celery_task_id = "paused-celery"
    job.error = "stale pause"

    _run(_seed_job(db_session, job))

    queued_jobs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        agent_job_tasks.execute_agent_job_task,
        "delay",
        lambda job_id, user_id: queued_jobs.append((job_id, user_id)),
    )

    agent_job_tasks.resume_paused_agent_jobs()
    _run(db_session.refresh(job))

    assert queued_jobs == [(str(job.id), str(job.user_id))]
    assert job.status == AgentJobStatus.PENDING.value
    assert job.error is None
    assert job.celery_task_id is None
    assert job.last_activity_at is not None


def test_resume_paused_agent_jobs_skips_resource_limited_jobs(db_session, monkeypatch):
    _patch_celery_session_factory(monkeypatch)

    job = _make_job(status=AgentJobStatus.PAUSED.value)
    job.last_activity_at = datetime.utcnow() - timedelta(minutes=10)
    job.max_iterations = 4
    job.iteration = 4
    job.celery_task_id = "paused-celery"
    job.error = "stale pause"

    _run(_seed_job(db_session, job))

    queued_jobs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        agent_job_tasks.execute_agent_job_task,
        "delay",
        lambda job_id, user_id: queued_jobs.append((job_id, user_id)),
    )

    agent_job_tasks.resume_paused_agent_jobs()
    _run(db_session.refresh(job))

    assert queued_jobs == []
    assert job.status == AgentJobStatus.PAUSED.value
    assert job.error == "stale pause"
    assert job.celery_task_id == "paused-celery"


def test_cleanup_old_agent_jobs_deletes_checkpoints_and_terminal_jobs(db_session, monkeypatch):
    _patch_celery_session_factory(monkeypatch)

    old_job = _make_job(status=AgentJobStatus.COMPLETED.value)
    old_job.created_at = datetime.utcnow() - timedelta(days=45)
    checkpoint = AgentJobCheckpoint(
        id=uuid4(),
        job_id=old_job.id,
        iteration=7,
        phase="cleanup",
        state={"step": "done"},
        context={"note": "old checkpoint"},
    )

    _run(_seed_job(db_session, old_job, checkpoint))

    agent_job_tasks.cleanup_old_agent_jobs(days=30)

    async def _verify():
        job_row = await db_session.execute(select(AgentJob).where(AgentJob.id == old_job.id))
        checkpoint_row = await db_session.execute(
            select(AgentJobCheckpoint).where(AgentJobCheckpoint.job_id == old_job.id)
        )
        return job_row.scalar_one_or_none(), checkpoint_row.scalars().all()

    job_row, checkpoint_rows = _run(_verify())

    assert job_row is None
    assert checkpoint_rows == []


def test_check_stalled_agent_jobs_marks_job_failed_and_reports_progress(db_session, monkeypatch):
    _patch_celery_session_factory(monkeypatch)

    job = _make_job(status=AgentJobStatus.RUNNING.value, schedule_type="continuous")
    job.last_activity_at = datetime.utcnow() - timedelta(minutes=45)
    job.progress = 21
    job.results = {}

    _run(_seed_job(db_session, job))

    follow_up_calls = []

    async def _fake_sync_follow_up_outcome_for_job(db, job_obj):
        follow_up_calls.append((db, job_obj.id))

    progress_calls = []

    async def _fake_publish_job_progress(**kwargs):
        progress_calls.append(kwargs)

    monkeypatch.setattr(agent_job_tasks, "sync_follow_up_outcome_for_job", _fake_sync_follow_up_outcome_for_job)
    monkeypatch.setattr(agent_job_tasks, "_publish_job_progress", _fake_publish_job_progress)

    agent_job_tasks.check_stalled_agent_jobs(timeout_minutes=30)
    _run(db_session.refresh(job))

    assert follow_up_calls
    assert follow_up_calls[0][0] is not None
    assert follow_up_calls[0][1] == job.id
    assert len(progress_calls) == 1
    assert progress_calls[0]["phase"] == "stalled"
    assert progress_calls[0]["status"] == "failed"
    assert job.status == AgentJobStatus.FAILED.value
    assert job.error == "Job stalled - no activity for 30 minutes"
    assert job.completed_at is not None
    assert job.celery_task_id is None
    state = (((job.results or {}).get("execution_strategy") or {}).get("scheduler_state") or {})
    assert state["queue_reason"] == "stalled_run"
    assert state["last_run_status"] == AgentJobStatus.FAILED.value


def test_execute_agent_job_async_records_success_and_scheduler_state(db_session, monkeypatch):
    _patch_celery_session_factory(monkeypatch)

    job = _make_job(status=AgentJobStatus.PENDING.value, schedule_type="continuous")
    job.results = {
        "execution_strategy": {
            "scheduler_state": {
                "failure_streak": 3,
                "queue_reason": "old_failure",
                "backoff_seconds": 180,
                "last_run_status": AgentJobStatus.FAILED.value,
            }
        }
    }
    _run(_seed_job(db_session, job))

    task_id = "task-success-123"
    monkeypatch.setattr(
        agent_job_tasks,
        "current_task",
        types.SimpleNamespace(request=types.SimpleNamespace(id=task_id)),
    )

    progress_calls = []
    follow_up_calls = []

    async def _fake_publish_job_progress(**kwargs):
        progress_calls.append(kwargs)

    async def _fake_sync_follow_up_outcome_for_job(db, job_obj):
        follow_up_calls.append((db, job_obj.id))

    class _FakeExecutor:
        async def execute_job(self, *, job_id, db, progress_callback=None):
            result = await db.execute(select(AgentJob).where(AgentJob.id == job_id))
            job_row = result.scalar_one()
            job_row.status = AgentJobStatus.COMPLETED.value
            job_row.progress = 100
            job_row.current_phase = "done"
            job_row.phase_details = "Finished successfully"
            job_row.error = None
            job_row.completed_at = datetime(2026, 3, 16, 10, 3, 0)
            job_row.last_activity_at = datetime(2026, 3, 16, 10, 3, 0)
            await db.commit()
            if progress_callback is not None:
                await progress_callback(
                    {
                        "progress": 100,
                        "phase": "done",
                        "iteration": 4,
                        "phase_details": "Finished successfully",
                    }
                )
            return {"status": AgentJobStatus.COMPLETED.value, "progress": 100, "iterations": 4}

    monkeypatch.setattr(agent_job_tasks, "_publish_job_progress", _fake_publish_job_progress)
    monkeypatch.setattr(agent_job_tasks, "sync_follow_up_outcome_for_job", _fake_sync_follow_up_outcome_for_job)
    monkeypatch.setattr(agent_job_tasks, "AutonomousAgentExecutor", lambda: _FakeExecutor())

    _run(agent_job_tasks._execute_agent_job_async(str(job.id), str(job.user_id)))
    _run(db_session.refresh(job))

    assert job.status == AgentJobStatus.COMPLETED.value
    assert job.progress == 100
    assert job.celery_task_id == task_id
    assert job.completed_at is not None
    state = (((job.results or {}).get("execution_strategy") or {}).get("scheduler_state") or {})
    assert state["last_run_status"] == AgentJobStatus.COMPLETED.value
    assert state["failure_streak"] == 0
    assert state["queue_reason"] is None
    assert state["backoff_seconds"] == 0
    assert state["backoff_until"] is None
    assert state["current_run_started_at"] is None
    assert state["last_successful_run_at"] is not None
    assert state["last_completed_run_at"] is not None
    assert len(progress_calls) >= 2
    assert progress_calls[0]["phase"] == "starting"
    assert progress_calls[-1]["phase"] == "completed"
    assert progress_calls[-1]["status"] == AgentJobStatus.COMPLETED.value
    assert len(follow_up_calls) == 1
    assert follow_up_calls[0][0] is not None
    assert follow_up_calls[0][1] == job.id


def test_execute_agent_job_async_records_failure_outcome(db_session, monkeypatch):
    _patch_celery_session_factory(monkeypatch)

    job = _make_job(status=AgentJobStatus.PENDING.value, schedule_type="continuous")
    job.config = {"interval_minutes": 10}
    _run(_seed_job(db_session, job))

    task_id = "task-failure-123"
    monkeypatch.setattr(
        agent_job_tasks,
        "current_task",
        types.SimpleNamespace(request=types.SimpleNamespace(id=task_id)),
    )

    progress_calls = []
    follow_up_calls = []

    async def _fake_publish_job_progress(**kwargs):
        progress_calls.append(kwargs)

    async def _fake_sync_follow_up_outcome_for_job(db, job_obj):
        follow_up_calls.append((db, job_obj.id))

    class _FailingExecutor:
        async def execute_job(self, *, job_id, db, progress_callback=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(agent_job_tasks, "_publish_job_progress", _fake_publish_job_progress)
    monkeypatch.setattr(agent_job_tasks, "sync_follow_up_outcome_for_job", _fake_sync_follow_up_outcome_for_job)
    monkeypatch.setattr(agent_job_tasks, "AutonomousAgentExecutor", lambda: _FailingExecutor())

    _run(agent_job_tasks._execute_agent_job_async(str(job.id), str(job.user_id)))
    _run(db_session.refresh(job))

    assert job.status == AgentJobStatus.FAILED.value
    assert job.error == "boom"
    assert job.completed_at is not None
    assert job.celery_task_id == task_id
    assert job.next_run_at is not None
    state = (((job.results or {}).get("execution_strategy") or {}).get("scheduler_state") or {})
    assert state["last_run_status"] == AgentJobStatus.FAILED.value
    assert state["failure_streak"] == 1
    assert state["queue_reason"] == "execution_failure"
    assert state["backoff_seconds"] == 600
    assert state["backoff_until"] is not None
    assert state["current_run_started_at"] is None
    assert state["last_failure_at"] is not None
    assert len(progress_calls) >= 2
    assert progress_calls[0]["phase"] == "starting"
    assert progress_calls[-1]["phase"] == "error"
    assert progress_calls[-1]["status"] == "failed"
    assert progress_calls[-1]["error"] == "boom"
    assert len(follow_up_calls) == 1
    assert follow_up_calls[0][0] is not None
    assert follow_up_calls[0][1] == job.id


def test_execute_agent_job_task_persists_failure_when_loop_bootstrap_breaks(db_session, monkeypatch):
    _patch_celery_session_factory(monkeypatch)

    job = _make_job(status=AgentJobStatus.PENDING.value, schedule_type="continuous")
    _run(_seed_job(db_session, job))

    follow_up_calls = []

    async def _fake_sync_follow_up_outcome_for_job(db, job_obj):
        follow_up_calls.append((db, job_obj.id))

    monkeypatch.setattr(agent_job_tasks, "sync_follow_up_outcome_for_job", _fake_sync_follow_up_outcome_for_job)

    real_asyncio_run = asyncio.run
    call_count = {"value": 0}

    def _fake_asyncio_run(coro):
        call_count["value"] += 1
        if call_count["value"] == 1:
            raise RuntimeError("loop broke")
        return real_asyncio_run(coro)

    monkeypatch.setattr(agent_job_tasks.asyncio, "run", _fake_asyncio_run)

    task = types.SimpleNamespace(
        request=types.SimpleNamespace(retries=2),
        max_retries=2,
        retry=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("retry should not be called")),
    )

    agent_job_tasks.execute_agent_job_task(task, str(job.id), str(job.user_id))
    _run(db_session.refresh(job))

    assert call_count["value"] == 2
    assert job.status == AgentJobStatus.FAILED.value
    assert job.error == "Task error: loop broke"
    assert job.completed_at is not None
    state = (((job.results or {}).get("execution_strategy") or {}).get("scheduler_state") or {})
    assert state["last_run_status"] == AgentJobStatus.FAILED.value
    assert state["queue_reason"] == "task_failure"
    assert state["failure_streak"] == 1
    assert state["current_run_started_at"] is None
    assert len(follow_up_calls) == 1
    assert follow_up_calls[0][0] is not None
    assert follow_up_calls[0][1] == job.id


async def _seed_job(db_session, job, checkpoint=None):
    db_session.add(job)
    if checkpoint is not None:
        db_session.add(checkpoint)
    await db_session.commit()
    await db_session.refresh(job)
    if checkpoint is not None:
        await db_session.refresh(checkpoint)
