"""A worker that has lost its lease must not write the job it no longer owns.

Measured: a worker frozen for two hours woke, found its lease gone, and wrote
status='failed' (and its workspace artifacts) over a job that a newer
execution had resumed 13 seconds earlier. That execution completed; the row
still said failed, with the stale worker's error.
"""

from uuid import uuid4

import pytest

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.agent_execution_lease_service import ExecutionLeaseLostError
from app.services.autonomous_agent_executor import AutonomousAgentExecutor


@pytest.mark.asyncio
async def test_a_lost_lease_leaves_the_job_row_untouched(db_session, monkeypatch):
    job = AgentJob(
        name="owned elsewhere",
        goal="g",
        job_type="analysis",
        user_id=uuid4(),
        status=AgentJobStatus.RUNNING.value,
        config={},
        max_iterations=4,
    )
    db_session.add(job)
    await db_session.commit()
    job_id = job.id

    executor = AutonomousAgentExecutor()
    chain_events = []

    async def loses_lease(*args, **kwargs):
        raise ExecutionLeaseLostError(
            f"Execution lease lost for job {job_id} at fence 1"
        )

    async def record_chain(job, event, db):
        chain_events.append(event)

    monkeypatch.setattr(executor, "_run_autonomous_loop", loses_lease)
    monkeypatch.setattr(executor, "_trigger_chained_jobs", record_chain)

    with pytest.raises(ExecutionLeaseLostError):
        await executor.execute_job(job_id, db_session)

    await db_session.rollback()
    fresh = await db_session.get(AgentJob, job_id)
    await db_session.refresh(fresh)
    assert fresh.status == AgentJobStatus.RUNNING.value
    assert not fresh.error
    assert chain_events == []  # no "fail" event fired on someone else's job


@pytest.mark.asyncio
async def test_any_other_failure_is_still_recorded(db_session, monkeypatch):
    job = AgentJob(
        name="really fails",
        goal="g",
        job_type="analysis",
        user_id=uuid4(),
        status=AgentJobStatus.RUNNING.value,
        config={},
        max_iterations=4,
    )
    db_session.add(job)
    await db_session.commit()
    executor = AutonomousAgentExecutor()

    async def breaks(*args, **kwargs):
        raise RuntimeError("tool exploded")

    async def no_chain(job, event, db):
        return None

    monkeypatch.setattr(executor, "_run_autonomous_loop", breaks)
    monkeypatch.setattr(executor, "_trigger_chained_jobs", no_chain)
    result = await executor.execute_job(job.id, db_session)
    assert result["status"] == "failed"
    fresh = await db_session.get(AgentJob, job.id)
    assert (
        fresh.status == AgentJobStatus.FAILED.value and fresh.error == "tool exploded"
    )
