"""Tests for fenced autonomous job execution leases."""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import update

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.agent_execution_lease_service import (
    AgentExecutionLeaseService,
    ExecutionLeaseLostError,
)
from app.tasks import agent_job_tasks
from tests.conftest import TestSessionLocal


async def _create_job(db_session) -> AgentJob:
    job = AgentJob(
        name="Lease test",
        goal="Execute once",
        job_type="coding",
        user_id=uuid4(),
        status=AgentJobStatus.PENDING.value,
        config={},
        max_iterations=20,
        max_tool_calls=20,
        max_llm_calls=20,
        max_runtime_minutes=30,
    )
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)
    return job


@pytest.mark.asyncio
async def test_only_one_worker_can_claim_active_job(db_session):
    service = AgentExecutionLeaseService()
    job = await _create_job(db_session)
    first = await service.acquire(
        db=db_session,
        job_id=job.id,
        owner_id="worker-a",
        ttl_seconds=120,
    )
    second = await service.acquire(
        db=db_session,
        job_id=job.id,
        owner_id="worker-b",
        ttl_seconds=120,
    )

    assert first is not None
    assert first.fence == 1
    assert second is None
    await service.assert_owned(db=db_session, lease=first)


@pytest.mark.asyncio
async def test_expired_lease_increments_fence_and_rejects_stale_worker(db_session):
    service = AgentExecutionLeaseService()
    job = await _create_job(db_session)
    first = await service.acquire(
        db=db_session,
        job_id=job.id,
        owner_id="worker-a",
        ttl_seconds=30,
    )
    await db_session.execute(
        update(AgentJob)
        .where(AgentJob.id == job.id)
        .values(execution_lease_expires_at=datetime.utcnow() - timedelta(seconds=1))
    )
    await db_session.commit()

    second = await service.acquire(
        db=db_session,
        job_id=job.id,
        owner_id="worker-b",
        ttl_seconds=120,
    )

    assert second is not None
    assert second.fence == first.fence + 1
    with pytest.raises(ExecutionLeaseLostError):
        await service.assert_owned(db=db_session, lease=first)
    assert await service.release(db=db_session, lease=first) is False
    assert await service.release(db=db_session, lease=second) is True


@pytest.mark.asyncio
async def test_expired_owner_cannot_renew_lease(db_session):
    service = AgentExecutionLeaseService()
    job = await _create_job(db_session)
    acquired_at = datetime.utcnow()
    lease = await service.acquire(
        db=db_session,
        job_id=job.id,
        owner_id="worker-a",
        ttl_seconds=30,
        now=acquired_at,
    )

    renewed = await service.renew(
        db=db_session,
        lease=lease,
        ttl_seconds=30,
        now=acquired_at + timedelta(seconds=31),
    )

    assert renewed is None


@pytest.mark.asyncio
async def test_duplicate_celery_delivery_exits_before_executor(db_session, monkeypatch):
    job = await _create_job(db_session)
    service = AgentExecutionLeaseService()
    first = await service.acquire(
        db=db_session,
        job_id=job.id,
        owner_id="worker-a",
        ttl_seconds=120,
    )
    monkeypatch.setattr(
        agent_job_tasks,
        "create_celery_session",
        lambda: TestSessionLocal,
    )

    class _UnexpectedExecutor:
        def __init__(self):
            raise AssertionError("duplicate delivery must not initialize executor")

    monkeypatch.setattr(
        agent_job_tasks,
        "AutonomousAgentExecutor",
        _UnexpectedExecutor,
    )

    result = await agent_job_tasks._execute_agent_job_async(
        str(job.id),
        str(job.user_id),
        lease_owner_id="worker-b",
    )

    assert result["status"] == "lease_conflict"
    assert await service.release(db=db_session, lease=first) is True


def test_lease_predicates_are_evaluated_by_the_database():
    """Comparing lease times in Python crashed the takeover path.

    Postgres returns execution_lease_expires_at timezone-aware while the
    service uses a naive utcnow(), so SQLAlchemy's default in-memory
    evaluation of the WHERE clause raised "can't compare offset-naive and
    offset-aware datetimes". It only evaluated when a previous lease existed,
    which is why it stayed hidden until a job whose worker had died was
    claimed by another worker -- the path crash recovery depends on.
    """
    import inspect

    from app.services import agent_execution_lease_service as module

    for name in ("acquire", "renew"):
        source = inspect.getsource(getattr(module.AgentExecutionLeaseService, name))
        assert "synchronize_session=False" in source, name
