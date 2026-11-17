"""Suspend, poll and resume the real journal/worker boundaries without Kubernetes."""

import asyncio
import json
import time
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core.config import settings
from app.models.agent_job import AgentJob
from app.services import agent_job_user_cap, broker_queue, kubernetes_sandbox
from app.services import sandbox_continuation_service as poller
from app.services import sandbox_continuation_state as control
from app.services.agent_checkpoint_service import AgentCheckpointService
from app.services.agent_runtime_state_service import initialize_runtime_state
from app.services.autonomous_agent_executor import (
    AutonomousAgentExecutor,
    _AutonomousRuntimeAdapter,
)
from app.services.sandbox_invocation import SandboxStepPending, current, invocation
from app.tasks import agent_job_tasks
from tests.conftest import TEST_ON_POSTGRES, TestSessionLocal
from tests.test_kubernetes_sandbox import IMAGE, FakeAPI

pytestmark = pytest.mark.unit


@pytest.fixture
async def suspended(monkeypatch, tmp_path, db_session, test_user):
    monkeypatch.setattr(settings, "KUBERNETES_SANDBOX_DEFER_ENABLED", True)
    monkeypatch.setattr(settings, "SANDBOX_EXECUTION_BACKEND", "kubernetes")
    config = kubernetes_sandbox.Configuration(
        "experiments", "data", tmp_path / "records"
    )
    api = FakeAPI(config)
    api.complete = False
    job = AgentJob(
        id=uuid4(),
        user_id=test_user.id,
        name="continuation",
        goal="measure once",
        job_type="research",
        status="running",
        config={},
        results={},
        iteration=1,
        max_iterations=3,
        tool_calls_used=0,
        max_tool_calls=10,
        llm_calls_used=1,
        max_llm_calls=10,
        max_runtime_minutes=60,
        started_at=datetime.utcnow(),
    )
    db_session.add(job)
    await db_session.commit()
    executor = AutonomousAgentExecutor()
    calls = []

    async def execute(_executor, _job, _action, _state, _db):
        directory = tmp_path / f"worker-{len(calls)}"
        directory.mkdir()
        (directory / "input.txt").write_text("same experiment")
        calls.append(directory)
        result = await kubernetes_sandbox.run(
            "simulate",
            str(directory),
            image=IMAGE,
            timeout_seconds=60,
            memory="2048m",
            cpus="2",
            config=config,
            api=api,
        )
        assert (directory / "result.txt").read_text() == "measured result"
        return {
            "success": result[0] == 0,
            "findings": [{"type": "measurement", "title": "measured"}],
        }

    monkeypatch.setattr(executor.action_service, "_act_unjournaled", execute)
    state = initialize_runtime_state()
    action = {"tool": "compile_c_snippet", "params": {"code": "int main(){}"}}
    with pytest.raises(SandboxStepPending):
        await executor.action_service.act(
            executor, job, action, state, db_session, allow_sandbox_defer=True
        )
    assert current.get() is None
    return SimpleNamespace(
        job=job,
        state=state,
        action=action,
        executor=executor,
        api=api,
        config=config,
        calls=calls,
        db=db_session,
    )


async def test_suspension_keeps_intent_and_does_not_charge_a_second_iteration(
    suspended,
):
    s = suspended
    checkpoint = await AgentCheckpointService().load_latest_checkpoint(
        job_id=s.job.id, db=s.db
    )
    assert checkpoint.context["reason"] == control.WAITING_PHASE
    assert s.job.status == "paused"
    assert s.job.current_phase == control.WAITING_PHASE
    assert (
        s.job.iteration == 1
        and s.job.llm_calls_used == 1
        and s.job.tool_calls_used == 0
    )
    assert s.state["execution_journal_pending"]
    assert [entry["event_type"] for entry in s.state["execution_journal"]] == [
        "tool_intent"
    ]
    assert control.continuation(s.job, checkpoint.state) is not None
    assert s.api.posts == 1


async def test_running_experiment_is_not_requeued(suspended, monkeypatch):
    s = suspended
    sent = []
    monkeypatch.setattr(
        agent_job_tasks.execute_agent_job_task, "delay", lambda *args: sent.append(args)
    )
    result = await poller.poll_waiting(s.db, api=s.api, config=s.config)
    assert result["queued"] == 0 and result["errors"] == 0
    assert s.job.status == "paused" and sent == []


async def test_live_leased_candidates_cannot_starve_another_users_ready_step(
    suspended, monkeypatch
):
    from app.models.user import User

    s = suspended
    now = datetime.utcnow()
    other_user = User(
        id=uuid4(),
        username="other-poller-user",
        email="other-poller@example.com",
        hashed_password="test-only",
    )
    s.db.add(other_user)
    await s.db.flush()
    live = AgentJob(
        id=uuid4(),
        user_id=other_user.id,
        name="starting worker",
        goal="wait",
        job_type="research",
        status="pending",
        results=deepcopy(s.job.results),
        last_activity_at=now - timedelta(days=1),
        execution_lease_expires_at=now + timedelta(minutes=5),
    )
    s.db.add(live)
    await s.db.commit()
    s.api.complete = True
    sent = []
    monkeypatch.setattr(
        agent_job_tasks.execute_agent_job_task, "delay", lambda *args: sent.append(args)
    )
    first = await poller.poll_waiting(
        s.db, api=s.api, config=s.config, limit=1, now=now
    )
    assert first["queued"] == 0
    second = await poller.poll_waiting(
        s.db, api=s.api, config=s.config, limit=1, now=now
    )
    assert second["queued"] == 1
    assert sent == [(str(s.job.id), str(s.job.user_id))]
    await s.db.refresh(live)
    assert live.status == "pending" and live.last_activity_at.replace(
        tzinfo=None
    ) == now - timedelta(days=1)


async def test_ready_continuation_clears_its_obsolete_user_cap_wait(
    suspended, monkeypatch
):
    s = suspended
    s.api.complete = True
    now = datetime.utcnow()
    strategy = {
        "retained": "metadata",
        "scheduler_state": dict(
            agent_job_user_cap.waiting_state(now=now, running=2, cap=1, retry_in=60),
            failure_streak=2,
        ),
    }
    s.job.results = dict(s.job.results, execution_strategy=strategy)
    await s.db.commit()
    monkeypatch.setattr(
        agent_job_tasks.execute_agent_job_task, "delay", lambda *args: None
    )
    result = await poller.poll_waiting(s.db, api=s.api, config=s.config, now=now)
    assert result["queued"] == 1
    scheduler = s.job.results["execution_strategy"]["scheduler_state"]
    assert scheduler["queue_reason"] is None and "deferred_until" not in scheduler
    assert scheduler["failure_streak"] == 2
    assert s.job.results["execution_strategy"]["retained"] == "metadata"
    assert control.waiting(s.job) is not None


async def test_ready_result_resumes_exact_action_without_another_model_decision(
    suspended, monkeypatch
):
    s = suspended
    s.api.complete = True
    sent = []
    monkeypatch.setattr(
        agent_job_tasks.execute_agent_job_task, "delay", lambda *args: sent.append(args)
    )
    assert (await poller.poll_waiting(s.db, api=s.api, config=s.config))["queued"] == 1
    assert sent == [(str(s.job.id), str(s.job.user_id))]
    assert (await poller.poll_waiting(s.db, api=s.api, config=s.config))["queued"] == 0
    s.job.status = "running"
    await s.db.commit()
    checkpoint = await s.executor.checkpoint_service.load_latest_checkpoint(
        job_id=s.job.id, db=s.db
    )
    restored = initialize_runtime_state(json.loads(json.dumps(checkpoint.state)))
    monkeypatch.setattr(
        s.executor, "_get_counterfactual_config", lambda job: {"enabled": False}
    )
    monkeypatch.setattr(
        s.executor, "_get_execution_graph_config", lambda job: {"enabled": False}
    )
    monkeypatch.setattr(s.executor, "trigger_progress_chain", AsyncMock())
    adapter = _AutonomousRuntimeAdapter(
        executor=s.executor,
        job=s.job,
        agent_def=None,
        user_settings=None,
        state=restored,
        db=s.db,
        start_time=datetime.utcnow(),
        max_runtime=timedelta(minutes=60),
        progress_callback=None,
    )
    adapter.think_phase = AsyncMock(side_effect=AssertionError("must not think again"))
    adapter.observe_phase = AsyncMock(
        side_effect=AssertionError("must not observe again")
    )
    adapter.evaluate_phase = AsyncMock(
        return_value={"progress": 20, "should_stop": False}
    )
    assert await adapter.resume_sandbox_iteration() == {}
    assert s.job.iteration == 1 and s.job.llm_calls_used == 1
    assert s.job.tool_calls_used == 1
    assert len(restored["actions_taken"]) == 1
    assert len(restored["findings"]) == 1
    assert restored.get(control.STATE_KEY) is None
    assert control.waiting(s.job) is None
    assert s.api.posts == 1 and len(s.calls) == 2
    assert [e["event_type"] for e in restored["execution_journal"]] == [
        "tool_intent",
        "tool_result",
    ]


async def test_collected_result_replay_does_not_call_the_tool_twice(suspended):
    s = suspended
    s.api.complete = True
    s.job.status = "running"
    await s.db.commit()
    result = await s.executor.action_service.act(
        s.executor, s.job, s.action, s.state, s.db, allow_sandbox_defer=True
    )
    again = await s.executor.action_service.act(
        s.executor, s.job, s.action, s.state, s.db, allow_sandbox_defer=True
    )
    assert result == again
    assert len(s.calls) == 2 and s.api.posts == 1


async def test_a_validated_continuation_passes_its_step_reference_to_the_runtime(
    suspended, monkeypatch
):
    s = suspended
    reference = deepcopy(control.waiting(s.job)["reference"])
    s.api.complete = True
    s.job.status = "running"
    await s.db.commit()
    actual = s.executor.action_service._act_unjournaled
    observed = []

    async def act(*args):
        observed.append(current.get().resume_reference)
        return await actual(*args)

    monkeypatch.setattr(s.executor.action_service, "_act_unjournaled", act)
    await s.executor.action_service.act(
        s.executor, s.job, s.action, s.state, s.db, allow_sandbox_defer=True
    )
    assert observed == [reference]
    assert reference["step"] == 1 and reference["workspace"] == 1


async def test_owner_cancel_is_polled_and_never_requeues(suspended, monkeypatch):
    s = suspended
    s.job.status = "cancelled"
    await s.db.commit()
    assert await agent_job_user_cap.running_for_user(s.db, s.job.user_id) == 1
    sent = []
    monkeypatch.setattr(
        agent_job_tasks.execute_agent_job_task, "delay", lambda *args: sent.append(args)
    )
    result = await poller.poll_waiting(s.db, api=s.api, config=s.config)
    assert result["cancelled"] == 1 and result["errors"] == 0
    assert len(s.api.deletes) == 1 and sent == []
    assert control.waiting(s.job) is None and s.job.status == "cancelled"
    assert await agent_job_user_cap.running_for_user(s.db, s.job.user_id) == 0


async def test_lost_dispatch_is_retried_without_spending_celery_retries(
    suspended, monkeypatch
):
    s = suspended
    s.api.complete = True
    sent = []

    def broken(*args):
        sent.append(args)
        raise ConnectionError("broker unavailable")

    monkeypatch.setattr(agent_job_tasks.execute_agent_job_task, "delay", broken)
    now = datetime.utcnow()
    assert (await poller.poll_waiting(s.db, api=s.api, config=s.config, now=now))[
        "queued"
    ] == 1
    monkeypatch.setattr(
        agent_job_tasks.execute_agent_job_task, "delay", lambda *args: sent.append(args)
    )
    assert (
        await poller.poll_waiting(
            s.db, api=s.api, config=s.config, now=now + timedelta(minutes=3)
        )
    )["queued"] == 1
    assert len(sent) == 2 and s.job.status == "pending"
    assert s.api.posts == 1


async def test_queued_continuation_is_not_published_again_after_the_grace_window(
    suspended, monkeypatch
):
    s = suspended
    s.api.complete = True
    sent = []
    monkeypatch.setattr(
        agent_job_tasks.execute_agent_job_task, "delay", lambda *args: sent.append(args)
    )
    now = datetime.utcnow()
    assert (await poller.poll_waiting(s.db, api=s.api, config=s.config, now=now))[
        "queued"
    ] == 1
    lookups = []

    def presence(ids):
        lookups.append(ids)
        return {str(s.job.id): True}

    monkeypatch.setattr(
        broker_queue, "agent_job_message_presence", presence, raising=False
    )
    for minutes in (3, 6):
        result = await poller.poll_waiting(
            s.db, api=s.api, config=s.config, now=now + timedelta(minutes=minutes)
        )
        assert result["queued"] == 0 and result["errors"] == 0
    assert len(sent) == 1 and s.job.status == "pending"
    assert lookups == [{str(s.job.id)}, {str(s.job.id)}]


@pytest.mark.parametrize("visible", [False, None])
async def test_missing_or_unknown_broker_message_still_recovers(
    suspended, monkeypatch, visible
):
    s = suspended
    s.api.complete = True
    now = datetime.utcnow()
    sent = []
    monkeypatch.setattr(
        agent_job_tasks.execute_agent_job_task, "delay", lambda *args: sent.append(args)
    )
    assert (await poller.poll_waiting(s.db, api=s.api, config=s.config, now=now))[
        "queued"
    ] == 1
    monkeypatch.setattr(
        broker_queue,
        "agent_job_message_presence",
        lambda ids: {job_id: visible for job_id in ids},
        raising=False,
    )
    assert (
        await poller.poll_waiting(
            s.db, api=s.api, config=s.config, now=now + timedelta(minutes=3)
        )
    )["queued"] == 1
    assert len(sent) == 2


async def test_cancellation_during_broker_lookup_still_cleans_up_the_experiment(
    suspended, monkeypatch
):
    s = suspended
    now = datetime.utcnow()
    s.job.status = "pending"
    control.write_wait(
        s.job,
        dict(
            control.waiting(s.job),
            dispatched_at=(now - timedelta(minutes=3)).isoformat(),
        ),
    )
    await s.db.commit()
    actual = asyncio.to_thread

    async def looked_up(function, *args, **kwargs):
        if function is broker_queue.agent_job_message_presence:
            s.job.status = "cancelled"
            await s.db.commit()
            return {str(s.job.id): True}
        return await actual(function, *args, **kwargs)

    monkeypatch.setattr(poller.asyncio, "to_thread", looked_up)
    result = await poller.poll_waiting(s.db, api=s.api, config=s.config, now=now)
    assert result["cancelled"] == 1 and result["queued"] == 0
    assert control.waiting(s.job) is None and len(s.api.deletes) == 1


async def test_goal_deadline_cancels_the_experiment_and_queues_its_timeout(
    suspended, monkeypatch
):
    s = suspended
    wait = dict(control.waiting(s.job), deadline_at=time.time() - 1)
    control.write_wait(s.job, wait)
    await s.db.commit()
    sent = []
    monkeypatch.setattr(
        agent_job_tasks.execute_agent_job_task, "delay", lambda *args: sent.append(args)
    )
    result = await poller.poll_waiting(s.db, api=s.api, config=s.config)
    assert result["queued"] == 1 and len(s.api.deletes) == 1
    assert len(sent) == 1
    name = wait["reference"]["name"]
    record = json.loads((s.config.root / name / "record.json").read_text())
    assert record["status"] == "timed_out"


async def test_goal_timeout_waits_for_pods_to_stop_before_releasing_capacity(
    suspended, monkeypatch
):
    s = suspended
    s.api.delete_pending = True
    wait = dict(control.waiting(s.job), deadline_at=time.time() - 1)
    control.write_wait(s.job, wait)
    await s.db.commit()
    sent = []
    monkeypatch.setattr(
        agent_job_tasks.execute_agent_job_task, "delay", lambda *args: sent.append(args)
    )
    first = await poller.poll_waiting(s.db, api=s.api, config=s.config)
    assert first["queued"] == 0 and first["errors"] == 0
    assert s.job.status == "paused" and sent == []
    assert await agent_job_user_cap.running_for_user(s.db, s.job.user_id) == 1
    name = wait["reference"]["name"]
    path = s.config.root / name / "record.json"
    assert json.loads(path.read_text())["stop_reason"] == "timed_out"
    # A late success cannot change the timeout decision while deletion drains.
    s.api.complete = True
    assert (await poller.poll_waiting(s.db, api=s.api, config=s.config))["queued"] == 0
    s.job.status = "running"  # An operator or stale delivery resumes too early.
    await s.db.commit()
    with pytest.raises(SandboxStepPending):
        await s.executor.action_service.act(
            s.executor, s.job, s.action, s.state, s.db, allow_sandbox_defer=True
        )
    assert s.job.status == "paused" and s.api.posts == 1
    # Losing the poller after DELETE is also safe: the durable timeout survives
    # both a new poll and the Job disappearing in the meantime.
    s.api.jobs.clear()
    final = await poller.poll_waiting(s.db, api=s.api, config=s.config)
    assert final["queued"] == 1 and final["errors"] == 0 and len(sent) == 1
    assert json.loads(path.read_text())["status"] == "timed_out"


async def test_owner_cancellation_holds_cap_until_deletion_is_confirmed(suspended):
    s = suspended
    s.api.delete_pending = True
    s.job.status = "cancelled"
    await s.db.commit()
    assert (await poller.poll_waiting(s.db, api=s.api, config=s.config))[
        "cancelled"
    ] == 0
    assert control.waiting(s.job) is not None
    assert await agent_job_user_cap.running_for_user(s.db, s.job.user_id) == 1
    s.api.jobs.clear()
    assert (await poller.poll_waiting(s.db, api=s.api, config=s.config))[
        "cancelled"
    ] == 1
    assert await agent_job_user_cap.running_for_user(s.db, s.job.user_id) == 0


async def test_owner_cancellation_does_not_delete_a_replaced_experiment(suspended):
    s = suspended
    s.job.status = "cancelled"
    await s.db.commit()
    name = control.waiting(s.job)["reference"]["name"]
    s.api.jobs[name]["metadata"]["uid"] = "replacement"
    result = await poller.poll_waiting(s.db, api=s.api, config=s.config)
    assert result["errors"] == 1 and result["cancelled"] == 0
    assert s.api.deletes == [] and name in s.api.jobs
    assert "identity changed" in control.waiting(s.job)["last_error"]
    assert await agent_job_user_cap.running_for_user(s.db, s.job.user_id) == 1


async def test_waiting_experiments_count_toward_user_cap(suspended):
    s = suspended
    assert await agent_job_user_cap.running_for_user(s.db, s.job.user_id) == 1
    assert (
        await agent_job_user_cap.running_for_user(
            s.db, s.job.user_id, excluding=s.job.id
        )
        == 0
    )


async def test_collecting_an_already_counted_experiment_cannot_deadlock_on_a_lower_cap(
    suspended, monkeypatch
):
    s = suspended
    other = AgentJob(
        id=uuid4(),
        user_id=s.job.user_id,
        name="other experiment",
        goal="finish",
        job_type="research",
        status="paused",
        results={"sandbox_wait": {"key": "other"}},
    )
    s.db.add(other)
    s.job.status = "pending"
    await s.db.commit()
    monkeypatch.setattr(settings, "AGENT_JOBS_MAX_RUNNING_PER_USER", 1)
    assert await agent_job_user_cap.running_for_user(s.db, s.job.user_id) == 2
    assert await agent_job_tasks._hold_for_user_cap(s.db, s.job) is None
    assert s.job.status == "pending"
    assert await agent_job_user_cap.running_for_user(s.db, s.job.user_id) == 2


async def test_human_gate_and_changed_action_cannot_be_auto_resumed(suspended):
    s = suspended
    modified = deepcopy(s.state)
    modified["approval_checkpoint_pending"] = {"message": "operator review"}
    assert control.continuation(s.job, modified) is None
    with pytest.raises(RuntimeError, match="different action"):
        await s.executor.action_service.act(
            s.executor,
            s.job,
            {"tool": "compile_c_snippet", "params": {"code": "changed"}},
            s.state,
            s.db,
            allow_sandbox_defer=True,
        )
    assert s.api.posts == 1


async def test_task_returns_waiting_and_releases_lease_without_marking_failure(
    monkeypatch, db_session, test_user
):
    job = AgentJob(
        id=uuid4(),
        user_id=test_user.id,
        name="yield",
        goal="wait",
        job_type="research",
        status="pending",
        config={},
        results={},
    )
    db_session.add(job)
    await db_session.commit()

    class Executor:
        async def execute_job(self, *, job_id, db, progress_callback):
            row = await db.get(AgentJob, job_id)
            row.status = "paused"
            row.current_phase = control.WAITING_PHASE
            await db.commit()
            raise SandboxStepPending({"name": "experiment"})

    monkeypatch.setattr(settings, "AGENT_JOBS_MAX_RUNNING_PER_USER", 0)
    monkeypatch.setattr(
        agent_job_tasks, "create_celery_session", lambda: TestSessionLocal
    )
    monkeypatch.setattr(agent_job_tasks, "AutonomousAgentExecutor", Executor)
    monkeypatch.setattr(agent_job_tasks, "_publish_job_progress", AsyncMock())
    follow_up = AsyncMock()
    monkeypatch.setattr(agent_job_tasks, "sync_follow_up_outcome_for_job", follow_up)
    outcome = await agent_job_tasks._execute_agent_job_async(
        str(job.id), str(job.user_id)
    )
    await db_session.refresh(job)
    assert outcome["status"] == control.WAITING_PHASE
    assert job.status == "paused" and job.error is None
    assert job.execution_lease_owner is None and job.execution_lease_expires_at is None
    follow_up.assert_not_awaited()


def test_deadline_survives_timezone_reload_and_resets_only_for_a_new_run():
    started = datetime(2026, 1, 2, 3)
    job = SimpleNamespace(started_at=started, max_runtime_minutes=60)
    state = {}
    first = control.deadline(job, state)
    job.started_at = started.replace(tzinfo=timezone.utc)
    assert control.deadline(job, state) == first
    job.started_at += timedelta(days=1)
    assert control.deadline(job, state) == first + 86400


async def test_plain_invocation_keeps_synchronous_behavior():
    with invocation("job", "key") as context:
        assert context.defer is False


def test_new_runs_get_new_journal_keys_but_explicit_retries_keep_their_key():
    from app.services.agent_execution_journal_service import (
        AgentExecutionJournalService,
    )

    async def check():
        journal = AgentExecutionJournalService()
        executor = SimpleNamespace(
            checkpoint_service=SimpleNamespace(save_checkpoint=AsyncMock())
        )
        job = SimpleNamespace(
            id=uuid4(),
            config={},
            iteration=1,
            current_phase="acting",
            started_at=datetime(2026, 1, 1),
        )

        async def begin(action):
            return await journal.begin_tool_call(
                executor=executor, job=job, state={}, action=action, db=None
            )

        action = {"tool": "compile_c_snippet", "params": {"code": "same"}}
        first = await begin(deepcopy(action))
        job.started_at = job.started_at.replace(tzinfo=timezone.utc)
        assert (await begin(deepcopy(action)))["idempotency_key"] == first[
            "idempotency_key"
        ]
        job.started_at += timedelta(days=1)
        assert (await begin(deepcopy(action)))["idempotency_key"] != first[
            "idempotency_key"
        ]
        retry = dict(action, _idempotency_key=first["idempotency_key"])
        assert (await begin(retry))["idempotency_key"] == first["idempotency_key"]

    asyncio.run(check())


async def test_restart_and_delete_cannot_discard_an_active_experiment_reference(
    suspended,
):
    from fastapi import HTTPException

    from app.api.endpoints import agent_jobs
    from app.modules.autonomy.application.job_action_contracts import JobActionError
    from app.modules.autonomy.application.job_action_recovery import (
        perform_recovery_action,
    )

    s = suspended
    s.job.status = "cancelled"
    await s.db.commit()
    user = SimpleNamespace(id=s.job.user_id)
    with pytest.raises(JobActionError) as recovery:
        await perform_recovery_action(
            s.job, "restart", None, deps=None, db=s.db, current_user=user
        )
    assert recovery.value.status_code == 409
    with pytest.raises(HTTPException) as deletion:
        await agent_jobs.delete_agent_job(s.job.id, db=s.db, current_user=user)
    assert deletion.value.status_code == 409
    assert control.waiting(s.job) is not None


@pytest.mark.skipif(not TEST_ON_POSTGRES, reason="requires PostgreSQL row locks")
async def test_parallel_pollers_do_not_dispatch_the_same_ready_action(
    suspended, monkeypatch
):
    s = suspended
    s.api.complete = True
    entered, release = asyncio.Event(), asyncio.Event()
    actual = s.api.request

    async def paused_request(method, path, **kwargs):
        if method == "GET" and "/jobs/" in path:
            entered.set()
            await release.wait()
        return await actual(method, path, **kwargs)

    monkeypatch.setattr(s.api, "request", paused_request)
    sent = []
    monkeypatch.setattr(
        agent_job_tasks.execute_agent_job_task, "delay", lambda *args: sent.append(args)
    )
    async with TestSessionLocal() as first, TestSessionLocal() as second:
        task = asyncio.create_task(
            poller.poll_waiting(first, api=s.api, config=s.config)
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            other = await asyncio.wait_for(
                poller.poll_waiting(second, api=s.api, config=s.config), timeout=5
            )
            assert other["queued"] == 0
        finally:
            release.set()
        assert (await task)["queued"] == 1
    assert len(sent) == 1


@pytest.mark.parametrize(
    "tool,redacted,expected",
    [
        ("compile_c_snippet", False, True),
        ("compile_c_snippet", True, False),
        ("unreviewed_tool", False, False),
    ],
)
async def test_only_replay_safe_unredacted_primary_actions_can_defer(
    monkeypatch, tool, redacted, expected
):
    from app.services.agent_action_service import AgentActionService
    from app.services.agent_execution_journal_service import (
        agent_execution_journal_service,
    )

    monkeypatch.setattr(settings, "KUBERNETES_SANDBOX_DEFER_ENABLED", True)

    async def begin(**kwargs):
        kwargs["action"]["_idempotency_key"] = "intent"
        return {
            "idempotency_key": "intent",
            "retryable_from_journal": not redacted,
            "contains_redactions": redacted,
        }

    service = AgentActionService()

    async def act(*args):
        assert current.get().defer is expected
        return {"success": True}

    monkeypatch.setattr(agent_execution_journal_service, "begin_tool_call", begin)
    monkeypatch.setattr(
        agent_execution_journal_service, "complete_tool_call", AsyncMock()
    )
    monkeypatch.setattr(service, "_act_unjournaled", act)
    executor = SimpleNamespace(
        _apply_default_scope_to_action=lambda action, job: action
    )
    job = SimpleNamespace(
        id=uuid4(), results={}, started_at=datetime.utcnow(), max_runtime_minutes=60
    )
    await service.act(
        executor, job, {"tool": tool, "params": {}}, {}, None, allow_sandbox_defer=True
    )


async def test_generic_pause_sweep_does_not_resume_experiment_waits(
    suspended, monkeypatch
):
    s = suspended
    s.job.last_activity_at = datetime.utcnow() - timedelta(minutes=20)
    await s.db.commit()
    monkeypatch.setattr(
        agent_job_tasks, "create_celery_session", lambda: TestSessionLocal
    )
    sent, work = [], []
    monkeypatch.setattr(
        agent_job_tasks.execute_agent_job_task, "delay", lambda *args: sent.append(args)
    )
    # Capture the task's loop handoff, then run its real session/query logic
    # on this test's loop, which also works with asyncpg connections.
    with monkeypatch.context() as patch:
        patch.setattr(
            agent_job_tasks.asyncio, "run", lambda coroutine: work.append(coroutine)
        )
        agent_job_tasks.resume_paused_agent_jobs()
    await work.pop()
    await s.db.refresh(s.job)
    assert s.job.status == "paused" and sent == []


async def test_recurring_schedule_does_not_reset_a_pending_continuation(
    suspended, monkeypatch
):
    s = suspended
    original_start = control.run_epoch(s.job)
    s.job.status = "pending"
    s.job.schedule_type = "recurring"
    s.job.schedule_cron = "*/5 * * * *"
    s.job.next_run_at = datetime.utcnow() - timedelta(minutes=1)
    await s.db.commit()
    monkeypatch.setattr(
        agent_job_tasks, "create_celery_session", lambda: TestSessionLocal
    )
    sent, work = [], []
    monkeypatch.setattr(
        agent_job_tasks.execute_agent_job_task, "delay", lambda *args: sent.append(args)
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            agent_job_tasks.asyncio, "run", lambda coroutine: work.append(coroutine)
        )
        agent_job_tasks.process_scheduled_agent_jobs()
    await work.pop()
    await s.db.refresh(s.job)
    assert control.run_epoch(s.job) == original_start
    assert control.waiting(s.job) is not None and sent == []


@pytest.fixture
def maintenance(monkeypatch):
    monkeypatch.setattr(
        agent_job_tasks, "create_celery_session", lambda: TestSessionLocal
    )
    monkeypatch.setattr(agent_job_tasks, "_publish_job_progress", AsyncMock())
    monkeypatch.setattr(agent_job_tasks, "sync_follow_up_outcome_for_job", AsyncMock())
    monkeypatch.setattr(
        agent_job_tasks.broker_queue, "waiting_agent_job_ids", lambda: set()
    )
    sent = []
    monkeypatch.setattr(
        agent_job_tasks.execute_agent_job_task, "delay", lambda *args: sent.append(args)
    )

    async def run(task, **kwargs):
        work = []
        with monkeypatch.context() as patch:
            patch.setattr(
                agent_job_tasks.asyncio, "run", lambda coroutine: work.append(coroutine)
            )
            task(**kwargs)
        await work.pop()
        return sent

    return run


async def test_maintenance_does_not_fail_a_ready_continuation_as_never_started(
    suspended, maintenance
):
    s = suspended
    s.job.status = "pending"
    s.job.created_at = datetime.utcnow() - timedelta(days=2)
    s.job.celery_task_id = None
    s.job.execution_log = [
        {"phase": agent_job_tasks.UNDISPATCHED_REQUEUE_PHASE}
        for _ in range(agent_job_tasks.MAX_UNDISPATCHED_REQUEUES)
    ]
    await s.db.commit()
    sent = await maintenance(agent_job_tasks.check_stalled_agent_jobs)
    await s.db.refresh(s.job)
    assert s.job.status == "pending" and s.job.error is None and sent == []
    assert control.waiting(s.job) is not None
    assert agent_job_tasks.count_undispatched_requeues(s.job) == 3


async def test_maintenance_recovers_an_orphaned_continuation_at_iteration_limit(
    suspended, maintenance
):
    s = suspended
    s.job.status = "running"
    s.job.max_iterations = s.job.iteration
    s.job.last_activity_at = datetime.utcnow() - timedelta(minutes=45)
    s.job.execution_lease_owner = "lost-worker"
    s.job.execution_lease_expires_at = datetime.utcnow() - timedelta(minutes=5)
    await s.db.commit()
    sent = await maintenance(agent_job_tasks.check_stalled_agent_jobs)
    await s.db.refresh(s.job)
    assert s.job.status == "paused" and s.job.error is None and sent == []
    assert s.job.current_phase == control.WAITING_PHASE
    assert s.job.celery_task_id is None
    assert control.continuation(s.job, s.state) is not None
    assert agent_job_tasks.count_orphan_recoveries(s.job) == 1
    assert (await poller.poll_waiting(s.db, api=s.api, config=s.config))["queued"] == 0
    s.api.complete = True
    assert (await poller.poll_waiting(s.db, api=s.api, config=s.config))["queued"] == 1
    assert sent == [(str(s.job.id), str(s.job.user_id))]
    assert s.job.iteration == 1 and s.job.llm_calls_used == 1


async def test_maintenance_retains_a_cancelled_job_until_experiment_cleanup(
    suspended, maintenance
):
    from sqlalchemy import select

    from app.models.agent_job import AgentJobCheckpoint

    s = suspended
    job_id = s.job.id
    s.job.status = "cancelled"
    s.job.created_at = datetime.utcnow() - timedelta(days=45)
    await s.db.commit()
    await maintenance(agent_job_tasks.cleanup_old_agent_jobs, days=30)
    assert (
        await s.db.execute(select(AgentJob).where(AgentJob.id == job_id))
    ).scalar_one()
    checkpoints = select(AgentJobCheckpoint).where(AgentJobCheckpoint.job_id == job_id)
    assert (await s.db.execute(checkpoints)).scalars().all()
    assert (await poller.poll_waiting(s.db, api=s.api, config=s.config))[
        "cancelled"
    ] == 1
    await maintenance(agent_job_tasks.cleanup_old_agent_jobs, days=30)
    assert (
        await s.db.execute(select(AgentJob).where(AgentJob.id == job_id))
    ).scalar_one_or_none() is None
    assert (await s.db.execute(checkpoints)).scalars().all() == []


async def test_maintenance_retains_a_terminal_job_while_a_worker_still_holds_it(
    db_session, test_user, maintenance
):
    from sqlalchemy import select

    job = AgentJob(
        id=uuid4(),
        user_id=test_user.id,
        name="cancelled during handoff",
        goal="preserve the worker's cancellation handoff",
        job_type="research",
        status="cancelled",
        results={},
        created_at=datetime.utcnow() - timedelta(days=45),
        execution_lease_expires_at=datetime.utcnow() + timedelta(minutes=5),
    )
    db_session.add(job)
    await db_session.commit()
    query = select(AgentJob).where(AgentJob.id == job.id)
    await maintenance(agent_job_tasks.cleanup_old_agent_jobs, days=30)
    assert (await db_session.execute(query)).scalar_one().id == job.id
    job.execution_lease_expires_at = datetime.utcnow() - timedelta(minutes=1)
    await db_session.commit()
    await maintenance(agent_job_tasks.cleanup_old_agent_jobs, days=30)
    assert (await db_session.execute(query)).scalar_one_or_none() is None


async def test_maintenance_still_bounds_repeated_continuation_worker_crashes(
    suspended, maintenance
):
    s = suspended
    s.job.status = "running"
    s.job.last_activity_at = datetime.utcnow() - timedelta(minutes=45)
    s.job.execution_log = [
        {"phase": agent_job_tasks.ORPHAN_RECOVERY_PHASE}
        for _ in range(agent_job_tasks.MAX_ORPHAN_RECOVERIES)
    ]
    await s.db.commit()
    sent = await maintenance(agent_job_tasks.check_stalled_agent_jobs)
    await s.db.refresh(s.job)
    assert s.job.status == "failed" and sent == []
    assert "lost its worker" in s.job.error
    assert control.waiting(s.job) is not None
    assert (await poller.poll_waiting(s.db, api=s.api, config=s.config))[
        "cancelled"
    ] == 1


async def test_maintenance_dispatches_an_old_orphan_only_once(
    db_session, test_user, maintenance
):
    job = AgentJob(
        id=uuid4(),
        user_id=test_user.id,
        name="old orphan",
        goal="recover the saved run once",
        job_type="research",
        status="running",
        results={},
        created_at=datetime.utcnow() - timedelta(days=2),
        started_at=datetime.utcnow(),
        last_activity_at=datetime.utcnow() - timedelta(minutes=45),
        execution_lease_expires_at=datetime.utcnow() - timedelta(minutes=5),
    )
    db_session.add(job)
    await db_session.commit()
    sent = await maintenance(agent_job_tasks.check_stalled_agent_jobs)
    await db_session.refresh(job)
    assert sent == [(str(job.id), str(job.user_id))]
    assert agent_job_tasks.count_orphan_recoveries(job) == 1
    assert agent_job_tasks.count_undispatched_requeues(job) == 0
