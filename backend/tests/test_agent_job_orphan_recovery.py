"""Recovering a job whose worker died, instead of failing it.

A week-long plan is many jobs, and any of them can lose its worker to a
restart or a crash. Checkpoints exist to make that survivable, but the stalled
sweep marked every quiet running job failed, so the checkpoint was never used
and the work was thrown away.
"""

from datetime import datetime, timedelta
from uuid import uuid4

from app.models.agent_job import AgentJob, AgentJobStatus
from app.tasks.agent_job_tasks import (
    MAX_ORPHAN_RECOVERIES,
    ORPHAN_RECOVERY_PHASE,
    count_orphan_recoveries,
    is_orphaned,
)


def _job(**kwargs) -> AgentJob:
    base = dict(
        id=uuid4(),
        name="Long run",
        goal="Measure something",
        job_type="research",
        user_id=uuid4(),
        status=AgentJobStatus.RUNNING.value,
        iteration=4,
        max_iterations=100,
        tool_calls_used=10,
        max_tool_calls=100,
        llm_calls_used=10,
        max_llm_calls=100,
        max_runtime_minutes=60,
    )
    base.update(kwargs)
    return AgentJob(**base)


def test_a_job_whose_lease_expired_has_no_worker():
    now = datetime.utcnow()
    job = _job(execution_lease_expires_at=now - timedelta(minutes=5))

    assert is_orphaned(job, now) is True


def test_a_job_holding_a_live_lease_is_still_owned():
    """Requeueing this one would run the same job in two workers."""
    now = datetime.utcnow()
    job = _job(execution_lease_expires_at=now + timedelta(minutes=5))

    assert is_orphaned(job, now) is False


def test_a_job_that_never_took_a_lease_counts_as_orphaned():
    assert is_orphaned(_job(execution_lease_expires_at=None), datetime.utcnow()) is True


def test_an_aware_lease_timestamp_is_compared_correctly():
    """The column is timezone-aware; comparing it to a naive now would raise."""
    from datetime import timezone

    now = datetime.utcnow()
    job = _job(
        execution_lease_expires_at=(now - timedelta(minutes=1)).replace(
            tzinfo=timezone.utc
        )
    )

    assert is_orphaned(job, now) is True


def test_recoveries_are_counted_from_the_execution_log():
    job = _job()
    assert count_orphan_recoveries(job) == 0

    job.add_log_entry({"phase": ORPHAN_RECOVERY_PHASE, "recovery_attempt": 1})
    job.add_log_entry({"phase": "thinking"})
    job.add_log_entry({"phase": ORPHAN_RECOVERY_PHASE, "recovery_attempt": 2})

    assert count_orphan_recoveries(job) == 2


def test_recovery_is_bounded_so_a_crash_loop_still_fails():
    job = _job()
    for attempt in range(MAX_ORPHAN_RECOVERIES):
        job.add_log_entry(
            {"phase": ORPHAN_RECOVERY_PHASE, "recovery_attempt": attempt + 1}
        )

    assert count_orphan_recoveries(job) >= MAX_ORPHAN_RECOVERIES


def test_an_exhausted_job_is_not_worth_recovering():
    """Requeueing a job with no budget left would just fail again."""
    job = _job(iteration=100, max_iterations=100)
    limited, reason = job.is_resource_limited()

    assert limited is True
    assert reason == "max_iterations"
