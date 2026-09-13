"""Whether the last sibling to finish can start the fan-in.

It could not. The gate is evaluated during the asking job's own finalisation,
before its terminal status is visible to a fresh SELECT, so every sibling
counted N-1 and deferred -- including the last one. A deferral schedules no
retry, so nobody ever fired and the fan-in was unreachable for a swarm of any
size. That is why no job in this database has ever carried a swarm summary.

Measured on the first swarm ever run here: the verifier finished last and
recorded `swarm_fan_in_deferred terminal=1 expected=2`.
"""

import uuid

import pytest

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.agent_chain_orchestration_service import (
    AgentChainOrchestrationService,
)

pytestmark = pytest.mark.unit


def _child(parent_id, status, *, group="g1", expected=2):
    job = AgentJob(
        id=uuid.uuid4(),
        name="swarm child",
        goal="measure",
        job_type="analysis",
        status=status,
        iteration=1,
        max_iterations=4,
        config={"agent_role": "researcher"},
        results={},
        execution_log=[],
        chain_config={
            "trigger_condition": "on_any_end",
            "child_jobs": [{"name": "fan-in"}],
            "chain_data": {
                "swarm_fan_in_wait_for_all_siblings": True,
                "swarm_fan_in_group_id": group,
                "swarm_fan_in_expected_siblings": expected,
            },
        },
    )
    job.parent_job_id = parent_id
    job.user_id = uuid.uuid4()
    return job


@pytest.mark.asyncio
class TestTheLastSiblingCanFire:
    async def test_the_asker_counts_its_own_finished_status(
        self, db_session, monkeypatch
    ):
        """The exact failure. Both siblings are done, but the database still
        shows the asking one as running because its finalisation has not
        committed yet."""
        parent_id = uuid.uuid4()
        first = _child(parent_id, AgentJobStatus.COMPLETED.value)
        # The asker: finished in memory, still 'running' as far as a SELECT is
        # concerned.
        asker = _child(parent_id, AgentJobStatus.RUNNING.value)
        db_session.add_all([first, asker])
        await db_session.commit()
        asker.status = AgentJobStatus.COMPLETED.value

        svc = AgentChainOrchestrationService()
        gate = await svc.evaluate_swarm_fan_in_gate(svc, asker, db_session)

        assert (
            gate["ready"] is True
        ), "the last sibling to finish must be able to start the fan-in"
        assert gate["terminal_siblings"] == 2

    async def test_it_still_defers_when_a_sibling_really_is_running(self, db_session):
        """The gate must not become a rubber stamp: a swarm whose siblings are
        genuinely still working has nothing to merge."""
        parent_id = uuid.uuid4()
        busy = _child(parent_id, AgentJobStatus.RUNNING.value)
        asker = _child(parent_id, AgentJobStatus.RUNNING.value)
        db_session.add_all([busy, asker])
        await db_session.commit()
        asker.status = AgentJobStatus.COMPLETED.value

        svc = AgentChainOrchestrationService()
        gate = await svc.evaluate_swarm_fan_in_gate(svc, asker, db_session)

        assert gate["ready"] is False
        assert gate["terminal_siblings"] == 1

    async def test_a_failed_sibling_still_counts_as_finished(self, db_session):
        """`on_any_end` means what it says. A swarm where one role failed has
        less to merge, not nothing -- and waiting for it forever is how a
        single failure strands the whole swarm."""
        parent_id = uuid.uuid4()
        broken = _child(parent_id, AgentJobStatus.FAILED.value)
        asker = _child(parent_id, AgentJobStatus.RUNNING.value)
        db_session.add_all([broken, asker])
        await db_session.commit()
        asker.status = AgentJobStatus.COMPLETED.value

        svc = AgentChainOrchestrationService()
        gate = await svc.evaluate_swarm_fan_in_gate(svc, asker, db_session)

        assert gate["ready"] is True


@pytest.mark.asyncio
class TestOnlyOneFanInPerGroup:
    """Fixing the last-finisher race turned "nobody fires" into "everybody
    fires".

    Both siblings now reach `ready` at the same moment, and neither sees the
    other's fan-in yet. Measured on the first swarm where the fan-in worked at
    all: two `Swarm Synthesis` jobs for one group, both of which then failed
    with "Missing inherited swarm sibling data".
    """

    async def test_an_existing_fan_in_stands_the_second_sibling_down(self, db_session):
        parent_id = uuid.uuid4()
        first = _child(parent_id, AgentJobStatus.COMPLETED.value)
        asker = _child(parent_id, AgentJobStatus.COMPLETED.value)
        db_session.add_all([first, asker])
        await db_session.commit()

        # The fan-in the first sibling already created.
        fan_in = AgentJob(
            id=uuid.uuid4(),
            name="Swarm Synthesis: Consolidated Output",
            goal="merge",
            job_type="synthesis",
            status=AgentJobStatus.PENDING.value,
            iteration=0,
            max_iterations=4,
            config={
                "origin": "swarm_fan_in_aggregator",
                "swarm_fan_in_group_id": "g1",
            },
            results={},
            execution_log=[],
        )
        fan_in.parent_job_id = first.id
        fan_in.user_id = uuid.uuid4()
        db_session.add(fan_in)
        await db_session.commit()

        svc = AgentChainOrchestrationService()
        gate = await svc.evaluate_swarm_fan_in_gate(svc, asker, db_session)

        assert gate["already_exists"] is True, (
            "one swarm must produce one merge, however many siblings finish " "at once"
        )

    async def test_a_fan_in_from_another_group_does_not_count(self, db_session):
        """Two swarms running at once must not stand each other down."""
        parent_id = uuid.uuid4()
        first = _child(parent_id, AgentJobStatus.COMPLETED.value)
        asker = _child(parent_id, AgentJobStatus.COMPLETED.value)
        db_session.add_all([first, asker])
        await db_session.commit()

        stranger = AgentJob(
            id=uuid.uuid4(),
            name="Swarm Synthesis: other group",
            goal="merge",
            job_type="synthesis",
            status=AgentJobStatus.PENDING.value,
            iteration=0,
            max_iterations=4,
            config={
                "origin": "swarm_fan_in_aggregator",
                "swarm_fan_in_group_id": "SOME_OTHER_GROUP",
            },
            results={},
            execution_log=[],
        )
        stranger.parent_job_id = first.id
        stranger.user_id = uuid.uuid4()
        db_session.add(stranger)
        await db_session.commit()

        svc = AgentChainOrchestrationService()
        gate = await svc.evaluate_swarm_fan_in_gate(svc, asker, db_session)

        assert gate["already_exists"] is False
        assert gate["ready"] is True


@pytest.mark.asyncio
class TestTheAggregatorCanSeeItsSiblings:
    """The fan-in is created as a child of whichever role fires the chain, so
    "siblings" means the PEERS of that role -- the children of its parent.

    The payload builder read `parent_job.id` instead, looking for the firing
    role's own descendants. There are none but the aggregator itself, which it
    filters out, so it returned {} and the `swarm` payload was never attached.
    The aggregator then failed with "Missing inherited swarm sibling data"
    every time it ran. The gate beside it had the same question right.
    """

    async def test_peers_are_collected_not_descendants(self, db_session):
        parent_id = uuid.uuid4()
        researcher = _child(parent_id, AgentJobStatus.COMPLETED.value)
        researcher.config = {"agent_role": "researcher", "swarm_role": "Researcher"}
        researcher.results = {"findings": [{"type": "benchmark_measurement"}]}
        verifier = _child(parent_id, AgentJobStatus.COMPLETED.value)
        verifier.config = {"agent_role": "verifier", "swarm_role": "Verifier"}
        verifier.results = {"findings": [{"type": "benchmark_measurement"}]}
        db_session.add_all([researcher, verifier])
        await db_session.commit()

        svc = AgentChainOrchestrationService()
        payload = await svc.build_swarm_sibling_payload(svc, researcher, db_session)

        roles = sorted(
            str((row or {}).get("role") or "").lower()
            for row in (payload.get("sibling_jobs") or [])
        )
        assert roles == [
            "researcher",
            "verifier",
        ], "the firing role and its peer, which is what there is to merge"

    async def test_an_aggregator_is_not_its_own_sibling(self, db_session):
        parent_id = uuid.uuid4()
        researcher = _child(parent_id, AgentJobStatus.COMPLETED.value)
        researcher.config = {"agent_role": "researcher", "swarm_role": "Researcher"}
        db_session.add(researcher)
        await db_session.commit()

        aggregator = _child(parent_id, AgentJobStatus.PENDING.value)
        aggregator.config = {"origin": "swarm_fan_in_aggregator"}
        db_session.add(aggregator)
        await db_session.commit()

        svc = AgentChainOrchestrationService()
        payload = await svc.build_swarm_sibling_payload(svc, researcher, db_session)

        origins = [
            str((row or {}).get("role") or "")
            for row in (payload.get("sibling_jobs") or [])
        ]
        assert "swarm_fan_in_aggregator" not in origins


@pytest.mark.asyncio
class TestApprovalGatedSwarmsDoNotDeadlock:
    """A role paused at an approval gate has finished its work.

    Terminal is {completed, failed, cancelled}, and an approval gate sets a job
    to PAUSED. So gating a swarm on approval would have every role pause, none
    count toward the fan-in, and the merge never reach `ready` -- and approving
    would change nothing, because the gate still saw zero finished siblings.
    A gate nobody can open is worse than no gate.

    "Still working" and "done, waiting for you" are different states. It is the
    same distinction the pipeline run view draws between `waiting` and dead.
    """

    def _paused_for_approval(self, parent_id):
        job = _child(parent_id, AgentJobStatus.PAUSED.value)
        job.results = {
            "approval_checkpoint": {
                "checkpoint_type": "chain_gate",
                "message": "waiting for approval",
            }
        }
        return job

    async def test_a_role_waiting_for_approval_counts_as_finished(self, db_session):
        parent_id = uuid.uuid4()
        waiting = self._paused_for_approval(parent_id)
        asker = _child(parent_id, AgentJobStatus.COMPLETED.value)
        db_session.add_all([waiting, asker])
        await db_session.commit()

        svc = AgentChainOrchestrationService()
        gate = await svc.evaluate_swarm_fan_in_gate(svc, asker, db_session)

        assert (
            gate["ready"] is True
        ), "a role holding for a person is not a role still computing"
        assert gate["terminal_siblings"] == 2

    async def test_a_plain_paused_role_does_not_count(self, db_session):
        """Paused for any other reason -- a resource limit, an operator pause --
        means the work is genuinely unfinished, and merging would be merging
        half a swarm."""
        parent_id = uuid.uuid4()
        stopped = _child(parent_id, AgentJobStatus.PAUSED.value)
        stopped.results = {}
        asker = _child(parent_id, AgentJobStatus.COMPLETED.value)
        db_session.add_all([stopped, asker])
        await db_session.commit()

        svc = AgentChainOrchestrationService()
        gate = await svc.evaluate_swarm_fan_in_gate(svc, asker, db_session)

        assert gate["ready"] is False
        assert gate["terminal_siblings"] == 1
