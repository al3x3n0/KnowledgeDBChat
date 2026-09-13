"""A broker that cannot be reached must not lose a finished job.

The children are committed before anything is dispatched. Losing the broker
after that is a delivery problem, and the campaign machinery is built on the
property that a reboot costs only the downtime -- so it must not cost the
parent's result as well.
"""

from __future__ import annotations

from unittest.mock import patch
from uuid import uuid4

import pytest

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.agent_chain_orchestration_service import (
    AgentChainOrchestrationService,
)


class _Executor:
    """Only what trigger_chained_jobs reaches on this path."""

    def __init__(self):
        self.events = []

    def _append_job_result_step_event(self, _job, event):
        self.events.append(event)

    async def _evaluate_swarm_fan_in_gate(self, _job, _db):
        return {"enabled": False}

    async def _create_chained_job(self, parent_job, child_config, db):
        """Real rows, so the dispatch loop has real ids to fail on.

        Creation is guarded and covered elsewhere; what these tests are about
        is what happens after it succeeds.
        """
        child = AgentJob(
            name=str(child_config.get("name") or "child"),
            goal=str(child_config.get("goal") or "g"),
            job_type=str(child_config.get("job_type") or "research"),
            user_id=parent_job.user_id,
            status=AgentJobStatus.PENDING.value,
            config={},
        )
        db.add(child)
        await db.flush()
        return child


async def _parent_with_chain(db):
    """A real AgentJob configured to spawn two children on completion."""
    job = AgentJob(
        name="parent",
        goal="measure something",
        job_type="research",
        user_id=uuid4(),
        status=AgentJobStatus.RUNNING.value,
        config={},
        chain_config={
            "trigger_condition": "on_complete",
            "child_jobs": [
                {"name": "child one", "goal": "g1", "job_type": "research"},
                {"name": "child two", "goal": "g2", "job_type": "research"},
            ],
        },
    )
    db.add(job)
    await db.commit()
    return job


@pytest.mark.asyncio
async def test_a_broker_outage_does_not_raise_out_of_a_finished_parent(db_session):
    """The real method, not a copy of it. The children are committed before
    anything is dispatched, so a broker that cannot be reached must cost a
    delay and not the parent's result."""
    executor = _Executor()
    parent = await _parent_with_chain(db_session)

    with patch(
        "app.tasks.agent_job_tasks.execute_agent_job_task.delay",
        side_effect=ConnectionRefusedError("no broker"),
    ):
        created = await AgentChainOrchestrationService().trigger_chained_jobs(
            executor, parent, "complete", db_session
        )

    assert len(created) == 2, "both children were created and are reported"
    failures = [e for e in executor.events if e["type"] == "chain_dispatch_failed"]
    assert failures and sorted(failures[0]["job_ids"]) == sorted(created)


@pytest.mark.asyncio
async def test_one_bad_dispatch_does_not_stop_the_others(db_session):
    """Guarded per job rather than around the loop: a broker that refuses one
    message and accepts the next should deliver the next."""
    executor = _Executor()
    parent = await _parent_with_chain(db_session)
    calls = []

    def _delay(job_id, _user):
        calls.append(job_id)
        if len(calls) == 1:
            raise ConnectionRefusedError("transient")

    with patch(
        "app.tasks.agent_job_tasks.execute_agent_job_task.delay", side_effect=_delay
    ):
        created = await AgentChainOrchestrationService().trigger_chained_jobs(
            executor, parent, "complete", db_session
        )

    assert len(calls) == len(created) == 2, "the second was still attempted"
    failures = [e for e in executor.events if e["type"] == "chain_dispatch_failed"]
    assert len(failures[0]["job_ids"]) == 1


@pytest.mark.asyncio
async def test_a_working_broker_records_no_failure_event(db_session):
    executor = _Executor()
    parent = await _parent_with_chain(db_session)

    with patch("app.tasks.agent_job_tasks.execute_agent_job_task.delay"):
        await AgentChainOrchestrationService().trigger_chained_jobs(
            executor, parent, "complete", db_session
        )

    assert [e for e in executor.events if e["type"] == "chain_dispatch_failed"] == []
