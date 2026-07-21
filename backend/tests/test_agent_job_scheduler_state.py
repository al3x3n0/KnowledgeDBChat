from datetime import datetime
from uuid import uuid4

from app.models.agent_job import AgentJob, AgentJobStatus
from app.tasks.agent_job_tasks import _mark_scheduler_dispatched, _record_scheduler_outcome


def test_scheduler_state_tracks_dispatch_and_completion():
    job = AgentJob(
        id=uuid4(),
        name="Recurring Job",
        goal="Monitor for changes",
        job_type="monitor",
        user_id=uuid4(),
        status=AgentJobStatus.PENDING.value,
        schedule_type="continuous",
        config={"interval_minutes": 15},
        results={},
    )
    now = datetime(2026, 3, 16, 10, 0, 0)

    _mark_scheduler_dispatched(job, dispatched_at=now)
    _record_scheduler_outcome(job, outcome=AgentJobStatus.COMPLETED.value, happened_at=now)

    state = (((job.results or {}).get("execution_strategy") or {}).get("scheduler_state") or {})
    assert state["last_run_status"] == AgentJobStatus.COMPLETED.value
    assert state["failure_streak"] == 0
    assert state["last_successful_run_at"] == now.isoformat()
    assert state["backoff_until"] is None


def test_scheduler_state_applies_backoff_for_recurring_failures():
    job = AgentJob(
        id=uuid4(),
        name="Recurring Job",
        goal="Monitor for changes",
        job_type="monitor",
        user_id=uuid4(),
        status=AgentJobStatus.FAILED.value,
        schedule_type="continuous",
        config={"interval_minutes": 10},
        results={},
    )
    now = datetime(2026, 3, 16, 10, 0, 0)

    _record_scheduler_outcome(
        job,
        outcome=AgentJobStatus.FAILED.value,
        happened_at=now,
        queue_reason="execution_failure",
    )

    state = (((job.results or {}).get("execution_strategy") or {}).get("scheduler_state") or {})
    assert state["last_run_status"] == AgentJobStatus.FAILED.value
    assert state["failure_streak"] == 1
    assert state["queue_reason"] == "execution_failure"
    assert state["backoff_seconds"] == 600
    assert job.next_run_at is not None
