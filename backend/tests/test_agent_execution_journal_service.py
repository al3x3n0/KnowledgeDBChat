"""Tests for durable autonomous execution journaling and recovery."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.agent_checkpoint_service import AgentCheckpointService
from app.services.agent_execution_journal_service import AgentExecutionJournalService
from app.services.agent_runtime_state_service import initialize_runtime_state


class _CheckpointRecorder:
    def __init__(self):
        self.calls = []

    async def save_checkpoint(self, **kwargs):
        self.calls.append(kwargs)


def _job(**config):
    return SimpleNamespace(
        id=uuid4(),
        iteration=7,
        current_phase="acting",
        config=config,
        results={},
    )


@pytest.mark.asyncio
async def test_journal_persists_hash_chained_intent_and_result():
    service = AgentExecutionJournalService()
    recorder = _CheckpointRecorder()
    executor = SimpleNamespace(checkpoint_service=recorder)
    job = _job()
    state = {}
    action = {
        "tool": "run_command",
        "purpose": "Verify patch",
        "params": {"command": "pytest -q"},
    }

    intent = await service.begin_tool_call(
        executor=executor,
        job=job,
        state=state,
        action=action,
        db=object(),
    )
    result = {"success": True, "artifacts": [{"type": "log", "path": "test.log"}]}
    completed = await service.complete_tool_call(
        executor=executor,
        job=job,
        state=state,
        intent=intent,
        result=result,
        db=object(),
    )

    assert [row["event_type"] for row in state["execution_journal"]] == [
        "tool_intent",
        "tool_result",
    ]
    assert completed["previous_hash"] == intent["entry_hash"]
    assert state["execution_journal_pending"] is None
    assert result["_journal_invocation_id"] == intent["invocation_id"]
    assert [call["reason"] for call in recorder.calls] == [
        "tool_intent",
        "tool_result",
    ]
    assert job.results["execution_journal"]["cursor"]["sequence"] == 2


@pytest.mark.asyncio
async def test_journal_redacts_sensitive_params_and_disables_direct_retry():
    service = AgentExecutionJournalService()
    executor = SimpleNamespace(checkpoint_service=_CheckpointRecorder())
    job = _job()
    state = {}

    intent = await service.begin_tool_call(
        executor=executor,
        job=job,
        state=state,
        action={
            "tool": "external_call",
            "params": {"api_key": "secret-value", "payload": {"token": "hidden"}},
        },
        db=object(),
    )
    reconciliation = service.reconcile_interrupted(job=job, state=state)

    assert intent["action"]["params"]["api_key"] == "[REDACTED]"
    assert intent["action"]["params"]["payload"]["token"] == "[REDACTED]"
    assert reconciliation["retryable_from_journal"] is False
    assert reconciliation["requires_action_edit"] is True
    assert state["execution_journal_pending"] is None
    assert state["approval_checkpoint_pending"] == reconciliation
    assert job.results["approval_checkpoint"] == reconciliation
    assert (
        job.results["execution_strategy"]["approval_checkpoints"]["pending"]
        == reconciliation
    )


@pytest.mark.asyncio
async def test_explicit_retry_preserves_stable_idempotency_key():
    service = AgentExecutionJournalService()
    executor = SimpleNamespace(checkpoint_service=_CheckpointRecorder())
    job = _job()
    state = {}
    action = {"tool": "external_call", "params": {"payload": "safe"}}

    first = await service.begin_tool_call(
        executor=executor,
        job=job,
        state=state,
        action=action,
        db=object(),
    )
    reconciliation = service.reconcile_interrupted(job=job, state=state)
    retry_action = dict(reconciliation["action"])
    job.iteration += 1
    second = await service.begin_tool_call(
        executor=executor,
        job=job,
        state=state,
        action=retry_action,
        db=object(),
    )

    assert first["idempotency_key"] == second["idempotency_key"]
    assert retry_action["_idempotency_key"] == first["idempotency_key"]


@pytest.mark.asyncio
async def test_completed_call_is_recovered_once_after_bookkeeping_crash():
    service = AgentExecutionJournalService()
    executor = SimpleNamespace(checkpoint_service=_CheckpointRecorder())
    job = _job()
    state = {"actions_taken": []}
    intent = await service.begin_tool_call(
        executor=executor,
        job=job,
        state=state,
        action={"tool": "write_file", "params": {"path": "src/a.py"}},
        db=object(),
    )
    await service.complete_tool_call(
        executor=executor,
        job=job,
        state=state,
        intent=intent,
        result={"success": True},
        db=object(),
    )

    assert service.recover_completed_action(state=state) is True
    assert service.recover_completed_action(state=state) is False
    recovered = state["actions_taken"][0]
    assert recovered["journal_recovered"] is True
    assert recovered["action"]["tool"] == "write_file"


def test_interrupted_call_becomes_hash_chained_reconciliation():
    service = AgentExecutionJournalService()
    job = _job()
    state = {
        "execution_journal": [],
        "execution_journal_cursor": {},
        "execution_journal_pending": {
            "invocation_id": "inv-1",
            "intent_event_id": "event-1",
            "action": {"tool": "deploy", "params": {"target": "staging"}},
            "retryable_from_journal": True,
            "contains_redactions": False,
        },
    }

    reconciliation = service.reconcile_interrupted(job=job, state=state)

    assert reconciliation["checkpoint_type"] == "execution_reconciliation"
    assert reconciliation["action"]["tool"] == "deploy"
    assert state["execution_journal"][-1]["event_type"] == "tool_interrupted"
    assert state["execution_reconciliation_pending"] == reconciliation


@pytest.mark.asyncio
async def test_unresolved_intent_survives_database_restart_boundary(db_session):
    job = AgentJob(
        name="Journal restart",
        goal="Safely resume",
        job_type="coding",
        user_id=uuid4(),
        status=AgentJobStatus.RUNNING.value,
        config={},
        max_iterations=20,
        max_tool_calls=20,
        max_llm_calls=20,
        max_runtime_minutes=30,
    )
    job.current_phase = "acting"
    job.iteration = 4
    db_session.add(job)
    await db_session.commit()

    checkpoint_service = AgentCheckpointService()
    first_service = AgentExecutionJournalService()
    await first_service.begin_tool_call(
        executor=SimpleNamespace(checkpoint_service=checkpoint_service),
        job=job,
        state=initialize_runtime_state(),
        action={"tool": "write_file", "params": {"path": "src/restart.py"}},
        db=db_session,
    )

    checkpoint = await checkpoint_service.load_latest_checkpoint(
        job_id=job.id, db=db_session
    )
    restored_state = initialize_runtime_state(checkpoint.state)
    fresh_service = AgentExecutionJournalService()
    reconciliation = fresh_service.reconcile_interrupted(job=job, state=restored_state)

    assert checkpoint.context["reason"] == "tool_intent"
    assert reconciliation["checkpoint_type"] == "execution_reconciliation"
    assert reconciliation["action"]["params"]["path"] == "src/restart.py"
