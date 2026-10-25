"""Reloading job control state must not rearm already-spent call budgets."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy import update

from app.models.agent_job import AgentJob
from app.services.autonomous_agent_executor import (
    AutonomousAgentExecutor,
    _AutonomousRuntimeAdapter,
)

pytestmark = pytest.mark.unit


def _job(user_id=None):
    return AgentJob(
        id=uuid4(),
        user_id=user_id or uuid4(),
        name="Refresh budget",
        goal="Read evidence",
        job_type="analysis",
        status="running",
        config={},
        iteration=0,
        error_count=0,
        execution_log=[],
        llm_calls_used=0,
        tool_calls_used=0,
        max_llm_calls=5,
        max_tool_calls=5,
        max_iterations=10,
        max_runtime_minutes=5,
    )


def _adapter(job, db):
    return _AutonomousRuntimeAdapter(
        executor=AutonomousAgentExecutor(),
        job=job,
        db=db,
        agent_def=None,
        user_settings=None,
        state={"findings": [], "observations": []},
        start_time=datetime.utcnow(),
        max_runtime=timedelta(minutes=5),
        progress_callback=None,
    )


@pytest.mark.parametrize("counter", ["llm_calls_used", "tool_calls_used"])
async def test_refresh_cannot_rearm_a_partly_spent_budget(counter):
    job = _job()

    async def refresh(row):
        setattr(row, counter, 0)

    db = AsyncMock()
    db.refresh.side_effect = refresh
    adapter = _adapter(job, db)
    rounds = 0
    while await adapter.can_continue():
        setattr(job, counter, getattr(job, counter) + 1)
        rounds += 1
        assert rounds < 10, "Refresh rearmed the spent call allowance"

    assert rounds == 5
    assert getattr(job, counter) == 5


@pytest.mark.parametrize("control", ["higher_usage", "lower_limit", "cancelled"])
@pytest.mark.parametrize(
    "counter,limit",
    [("llm_calls_used", "max_llm_calls"), ("tool_calls_used", "max_tool_calls")],
)
async def test_refreshed_control_changes_still_stop_the_loop(control, counter, limit):
    job = _job()
    setattr(job, counter, 2)

    async def refresh(row):
        setattr(row, counter, 5 if control == "higher_usage" else 0)
        if control == "lower_limit":
            setattr(row, limit, 1)
        if control == "cancelled":
            row.status = "cancelled"

    db = AsyncMock()
    db.refresh.side_effect = refresh

    assert not await _adapter(job, db).can_continue()
    assert getattr(job, counter) == (5 if control == "higher_usage" else 2)
    if control == "cancelled":
        assert job.status == "cancelled"


@pytest.mark.parametrize("persisted", [0, 4, 5])
async def test_real_database_refresh_keeps_local_usage_and_honors_higher_usage(
    db_session, test_user, persisted
):
    job = _job(test_user.id)
    db_session.add(job)
    await db_session.commit()
    job.llm_calls_used = 2
    job.tool_calls_used = 3
    # Simulate a separate controller updating the persisted counters without
    # rewriting the worker's uncommitted view of the job.
    with db_session.no_autoflush:
        await db_session.execute(
            update(AgentJob)
            .where(AgentJob.id == job.id)
            .values(llm_calls_used=persisted, tool_calls_used=persisted)
            .execution_options(synchronize_session=False)
        )
        allowed = await _adapter(job, db_session).can_continue()

    assert allowed is (persisted < 5)
    assert job.llm_calls_used == max(2, persisted)
    assert job.tool_calls_used == max(3, persisted)


async def test_real_refresh_preserves_cancellation_when_usage_is_committed(
    db_session, test_user
):
    job = _job(test_user.id)
    db_session.add(job)
    await db_session.commit()
    job.llm_calls_used, job.tool_calls_used = 2, 3
    with db_session.no_autoflush:
        await db_session.execute(
            update(AgentJob)
            .where(AgentJob.id == job.id)
            .values(status="cancelled")
            .execution_options(synchronize_session=False)
        )
        assert not await _adapter(job, db_session).can_continue()
    await db_session.commit()
    await db_session.refresh(job)

    assert job.status == "cancelled"
    assert (job.llm_calls_used, job.tool_calls_used) == (2, 3)
