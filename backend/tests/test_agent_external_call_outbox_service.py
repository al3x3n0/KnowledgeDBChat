"""Tests for transactional external-call enqueue and claimed delivery."""

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest
from sqlalchemy import update

from app.models.agent_external_call_outbox import AgentExternalCallOutbox
from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.workflow import UserTool
from app.services.agent_external_call_outbox_service import (
    AgentExternalCallOutboxService,
)
from app.services.agent_external_response_correlation_service import (
    AgentExternalResponseCorrelationService,
)
from app.services.autonomous_agent_executor import AutonomousAgentExecutor
from app.services.external_agent_gateway_service import ExternalAgentGatewayError
from app.services.agent_runtime_state_service import initialize_runtime_state
from app.tasks import agent_external_call_outbox_tasks
from app.tasks.agent_job_tasks import execute_agent_job_task
from tests.conftest import TestSessionLocal


async def _seed_connection(db_session, test_user):
    tool = UserTool(
        user_id=test_user.id,
        name=f"CompOps Outbox {uuid4()}",
        description="Transactional outbox test",
        tool_type="external_agent",
        parameters_schema={},
        config={
            "provider_type": "compops",
            "endpoint_url": "https://compops.example.test",
            "capabilities": ["compops.runs.submit"],
            "auth_type": "none",
            "timeout_seconds": 10,
        },
        is_enabled=True,
    )
    job = AgentJob(
        name="Outbox agent",
        goal="Submit one external run",
        job_type="analysis",
        user_id=test_user.id,
        status=AgentJobStatus.RUNNING.value,
        config={},
        max_iterations=20,
        max_tool_calls=20,
        max_llm_calls=20,
        max_runtime_minutes=30,
    )
    db_session.add_all([tool, job])
    await db_session.commit()
    await db_session.refresh(tool)
    await db_session.refresh(job)
    return tool, job


@pytest.mark.asyncio
async def test_enqueue_is_transactional_and_idempotent(db_session, test_user):
    service = AgentExternalCallOutboxService()
    tool, job = await _seed_connection(db_session, test_user)
    tool_id = tool.id
    job_id = job.id
    user_id = test_user.id
    first, created = await service.enqueue(
        db=db_session,
        user_id=user_id,
        tool_id=tool_id,
        job_id=job_id,
        capability="compops.runs.submit",
        payload={"operator": "vectorize"},
        idempotency_key="idem-transaction-1",
    )
    first_id = first.id
    await db_session.rollback()
    async with TestSessionLocal() as verification_db:
        missing = await verification_db.get(AgentExternalCallOutbox, first_id)
    assert missing is None

    committed, created = await service.enqueue(
        db=db_session,
        user_id=user_id,
        tool_id=tool_id,
        job_id=job_id,
        capability="compops.runs.submit",
        payload={"operator": "vectorize"},
        idempotency_key="idem-transaction-1",
    )
    await db_session.commit()
    duplicate, duplicate_created = await service.enqueue(
        db=db_session,
        user_id=user_id,
        tool_id=tool_id,
        job_id=job_id,
        capability="compops.runs.submit",
        payload={"operator": "vectorize"},
        idempotency_key="idem-transaction-1",
    )

    assert created is True
    assert duplicate_created is False
    assert duplicate.id == committed.id


@pytest.mark.asyncio
async def test_expired_claim_is_recovered_with_new_token(db_session, test_user):
    service = AgentExternalCallOutboxService()
    tool, job = await _seed_connection(db_session, test_user)
    row, _ = await service.enqueue(
        db=db_session,
        user_id=test_user.id,
        tool_id=tool.id,
        job_id=job.id,
        capability="compops.runs.submit",
        payload={"operator": "unroll"},
        idempotency_key="idem-claim-1",
    )
    await db_session.commit()
    first = await service.claim_next(db=db_session, owner_id="worker-a")
    assert first.id == row.id
    first_token = first.claim_token
    assert await service.claim_next(db=db_session, owner_id="worker-b") is None

    await db_session.execute(
        update(AgentExternalCallOutbox)
        .where(AgentExternalCallOutbox.id == row.id)
        .values(claim_expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    )
    await db_session.commit()
    recovered = await service.claim_next(db=db_session, owner_id="worker-b")

    assert recovered.id == row.id
    assert recovered.claim_token != first_token
    assert recovered.attempts == 2


@pytest.mark.asyncio
async def test_claimed_delivery_acknowledges_gateway_result(
    db_session, test_user, monkeypatch
):
    service = AgentExternalCallOutboxService()
    tool, job = await _seed_connection(db_session, test_user)
    row, _ = await service.enqueue(
        db=db_session,
        user_id=test_user.id,
        tool_id=tool.id,
        job_id=job.id,
        capability="compops.runs.submit",
        payload={"operator": "schedule"},
        idempotency_key="idem-deliver-1",
    )
    await db_session.commit()
    claimed = await service.claim_next(db=db_session, owner_id="worker-a")
    calls = []

    async def _invoke(**kwargs):
        calls.append(kwargs)
        return {
            "output": {"run_id": "run-1"},
            "provenance": {"request_id": kwargs["request_id"]},
        }

    monkeypatch.setattr(
        "app.services.agent_external_call_outbox_service."
        "external_agent_gateway_service.invoke",
        _invoke,
    )
    result = await service.deliver_claimed(db=db_session, row=claimed)
    await db_session.refresh(row)

    assert result["status"] == "succeeded"
    assert row.status == "succeeded"
    assert row.delivered_at is not None
    assert calls[0]["request_id"] == row.request_id
    assert row.response["output"]["run_id"] == "run-1"


@pytest.mark.asyncio
async def test_delivery_retries_then_dead_letters(db_session, test_user, monkeypatch):
    service = AgentExternalCallOutboxService()
    tool, job = await _seed_connection(db_session, test_user)
    row, _ = await service.enqueue(
        db=db_session,
        user_id=test_user.id,
        tool_id=tool.id,
        job_id=job.id,
        capability="compops.runs.submit",
        payload={"operator": "tile"},
        idempotency_key="idem-retry-1",
        max_attempts=2,
    )
    await db_session.commit()

    async def _fail(**_kwargs):
        raise ExternalAgentGatewayError("temporary outage")

    monkeypatch.setattr(
        "app.services.agent_external_call_outbox_service."
        "external_agent_gateway_service.invoke",
        _fail,
    )
    first = await service.claim_next(db=db_session, owner_id="worker-a")
    first_result = await service.deliver_claimed(db=db_session, row=first)
    assert first_result["status"] == "retry"

    await db_session.execute(
        update(AgentExternalCallOutbox)
        .where(AgentExternalCallOutbox.id == row.id)
        .values(next_attempt_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    )
    await db_session.commit()
    second = await service.claim_next(db=db_session, owner_id="worker-b")
    second_result = await service.deliver_claimed(db=db_session, row=second)

    assert second_result["status"] == "dead_letter"
    await db_session.refresh(row)
    assert row.status == "dead_letter"
    assert row.attempts == 2


@pytest.mark.asyncio
async def test_autonomous_tool_commits_journal_and_outbox_together(
    db_session, test_user
):
    tool, job = await _seed_connection(db_session, test_user)
    executor = AutonomousAgentExecutor()
    state = initialize_runtime_state()
    state["execution_plan"] = [
        {
            "step_id": "submit_compiler_run",
            "title": "Submit compiler experiment",
            "status": "in_progress",
        }
    ]
    job.iteration = 1
    action = {
        "tool": "enqueue_external_agent_call",
        "params": {
            "tool_id": str(tool.id),
            "capability": "compops.runs.submit",
            "payload": {"operator": "fuse"},
        },
    }

    result = await executor.action_service.act(
        executor,
        job,
        action,
        state,
        db_session,
    )

    assert result["success"] is True
    outbox_id = result["data"]["outbox_id"]
    row = await db_session.get(AgentExternalCallOutbox, UUID(outbox_id))
    assert row is not None
    assert row.idempotency_key == action["_idempotency_key"]
    assert row.correlation["plan_step_id"] == "submit_compiler_run"
    assert result["deferred_external"] is True
    assert job.status == AgentJobStatus.PAUSED.value
    assert job.current_phase == "awaiting_external"
    assert state["execution_plan"][0]["status"] == "waiting_external"
    assert state["external_calls_pending"][outbox_id]["capability"] == (
        "compops.runs.submit"
    )
    assert state["execution_journal"][-1]["event_type"] == "tool_result"
    checkpoint = await executor.checkpoint_service.load_latest_checkpoint(
        job_id=job.id,
        db=db_session,
    )
    assert checkpoint.context["reason"] == "tool_result"
    status_result = await executor.action_service.act(
        executor,
        job,
        {
            "tool": "get_external_call_status",
            "params": {"outbox_id": outbox_id},
        },
        state,
        db_session,
    )
    assert status_result["success"] is True
    assert status_result["data"]["status"] == "pending"


@pytest.mark.asyncio
async def test_outbox_worker_delivers_committed_rows(
    db_session, test_user, monkeypatch
):
    service = AgentExternalCallOutboxService()
    tool, job = await _seed_connection(db_session, test_user)
    row, _ = await service.enqueue(
        db=db_session,
        user_id=test_user.id,
        tool_id=tool.id,
        job_id=job.id,
        capability="compops.runs.submit",
        payload={"operator": "pipeline"},
        idempotency_key="idem-worker-1",
    )
    await db_session.commit()

    async def _invoke(**kwargs):
        return {
            "output": {"run_id": "run-worker"},
            "provenance": {"request_id": kwargs["request_id"]},
        }

    monkeypatch.setattr(
        "app.services.agent_external_call_outbox_service."
        "external_agent_gateway_service.invoke",
        _invoke,
    )
    monkeypatch.setattr(
        agent_external_call_outbox_tasks,
        "create_celery_session",
        lambda: TestSessionLocal,
    )

    summary = (
        await agent_external_call_outbox_tasks._async_deliver_external_call_outbox(
            owner_id="outbox-worker",
            batch_size=10,
        )
    )
    await db_session.refresh(row)

    assert summary["claimed"] == 1
    assert summary["succeeded"] == 1
    assert row.status == "succeeded"


@pytest.mark.asyncio
async def test_delivered_response_completes_waiting_step_and_resumes_once(
    db_session, test_user, monkeypatch
):
    delivery_service = AgentExternalCallOutboxService()
    correlation_service = AgentExternalResponseCorrelationService()
    tool, job = await _seed_connection(db_session, test_user)
    executor = AutonomousAgentExecutor()
    state = initialize_runtime_state(
        {
            "execution_mode": "plan_and_execute",
            "execution_plan": [
                {
                    "step_id": "submit_run",
                    "title": "Submit compiler run",
                    "status": "in_progress",
                },
                {
                    "step_id": "analyze_run",
                    "title": "Analyze compiler run",
                    "status": "pending",
                },
            ],
        }
    )
    job.iteration = 3
    action_result = await executor.action_service.act(
        executor,
        job,
        {
            "tool": "enqueue_external_agent_call",
            "params": {
                "tool_id": str(tool.id),
                "capability": "compops.runs.submit",
                "payload": {"operator": "vectorize"},
            },
        },
        state,
        db_session,
    )
    outbox_id = UUID(action_result["data"]["outbox_id"])

    async def _invoke(**kwargs):
        return {
            "output": {"run_id": "compiler-run-42", "speedup": 1.7},
            "provenance": {"request_id": kwargs["request_id"]},
        }

    queued = []

    def _delay(job_id, user_id):
        queued.append((job_id, user_id))

    monkeypatch.setattr(
        "app.services.agent_external_call_outbox_service."
        "external_agent_gateway_service.invoke",
        _invoke,
    )
    monkeypatch.setattr(
        execute_agent_job_task,
        "delay",
        _delay,
    )
    claimed = await delivery_service.claim_next(
        db=db_session,
        owner_id="delivery-worker",
    )
    assert claimed.id == outbox_id
    delivered = await delivery_service.deliver_claimed(
        db=db_session,
        row=claimed,
    )
    assert delivered["status"] == "succeeded"

    response_claim = await correlation_service.claim_next(
        db=db_session,
        owner_id="response-worker",
    )
    assert response_claim.id == outbox_id
    correlated = await correlation_service.correlate_and_dispatch(
        db=db_session,
        row=response_claim,
    )
    await db_session.refresh(job)
    row = await db_session.get(AgentExternalCallOutbox, outbox_id)
    checkpoint = await executor.checkpoint_service.load_latest_checkpoint(
        job_id=job.id,
        db=db_session,
    )

    assert correlated["status"] == "resume_enqueued"
    assert job.status == AgentJobStatus.PENDING.value
    assert row.correlated_at is not None
    assert row.resume_enqueued_at is not None
    assert queued == [(str(job.id), str(job.user_id))]
    assert checkpoint.context["reason"] == "external_call_response"
    assert str(outbox_id) in checkpoint.state["external_call_results"]
    assert str(outbox_id) not in checkpoint.state["external_calls_pending"]
    assert checkpoint.state["execution_plan"][0]["status"] == "done"
    assert checkpoint.state["execution_plan"][1]["status"] == "in_progress"
    assert checkpoint.state["plan_step_index"] == 1
    assert (
        await correlation_service.claim_next(
            db=db_session,
            owner_id="response-worker-2",
        )
        is None
    )
    assert len(queued) == 1
