"""A chain that waits for a person, and the ordinary one that does not.

Every trigger condition before this fired on its own -- on_complete, on_fail,
on_any_end, on_progress, on_findings -- so a pipeline stage that should stop
for review had no way to say so. The risk in adding one is not the new path
but the old one: mid-run approvals resume real jobs, and they must keep doing
exactly that.
"""

import pytest

from app.models.agent_job import AgentJob, AgentJobStatus, ChainTriggerCondition
from app.services.agent_runtime_finalizer import (
    CHAIN_GATE_CHECKPOINT,
    _hold_for_chain_approval,
)


def _job(**over):
    fields = {
        "status": AgentJobStatus.COMPLETED.value,
        "iteration": 2,
        "results": {},
        "chain_config": {
            "trigger_condition": "on_approval",
            "child_jobs": [{"name": "next", "config": {"pipeline_stage": "analyse"}}],
        },
    }
    fields.update(over)
    job = AgentJob(**fields)
    job.execution_log = []
    return job


class TestTheTriggerCondition:
    def test_an_approval_gated_chain_does_not_fire_on_completion(self):
        """Completing is what makes it ready to be approved."""
        assert _job().should_trigger_chain("complete") is False

    def test_it_fires_on_the_approval(self):
        assert _job().should_trigger_chain("approval") is True

    def test_an_ordinary_chain_still_fires_on_completion(self):
        job = _job(chain_config={"child_jobs": [{}]})
        assert job.should_trigger_chain("complete") is True

    def test_an_ordinary_chain_does_not_fire_on_an_approval(self):
        """The event is new; no existing condition should answer to it."""
        for condition in (
            ChainTriggerCondition.ON_COMPLETE,
            ChainTriggerCondition.ON_FAIL,
            ChainTriggerCondition.ON_ANY_END,
            ChainTriggerCondition.ON_PROGRESS,
            ChainTriggerCondition.ON_FINDINGS,
        ):
            job = _job(
                chain_config={
                    "trigger_condition": condition.value,
                    "child_jobs": [{}],
                }
            )
            assert job.should_trigger_chain("approval") is False, condition

    def test_a_chain_already_triggered_does_not_fire_again(self):
        job = _job()
        job.chain_triggered = True
        assert job.should_trigger_chain("approval") is False


class TestHoldingTheJob:
    def test_a_gated_job_pauses_instead_of_finishing(self):
        job = _job()
        _hold_for_chain_approval(job)
        assert job.status == AgentJobStatus.PAUSED.value
        assert job.current_phase == "awaiting_approval"

    def test_it_writes_the_payload_the_approval_queue_reads(self):
        """The queue item is built from results['approval_checkpoint'] alone,
        so this is what makes the wait visible to a person."""
        from app.services.agent_job_queue_helpers import extract_approval_checkpoint

        job = _job()
        _hold_for_chain_approval(job)
        assert extract_approval_checkpoint(job)
        assert job.results["approval_checkpoint"]["checkpoint_type"] == (
            CHAIN_GATE_CHECKPOINT
        )

    def test_the_message_names_what_is_waiting(self):
        job = _job()
        _hold_for_chain_approval(job)
        payload = job.results["approval_checkpoint"]
        assert "analyse" in payload["message"]
        assert payload["waiting_stages"] == ["analyse"]

    def test_an_ordinary_chain_is_left_alone(self):
        job = _job(chain_config={"child_jobs": [{}]})
        _hold_for_chain_approval(job)
        assert job.status == AgentJobStatus.COMPLETED.value
        assert "approval_checkpoint" not in job.results

    def test_a_job_that_failed_is_left_alone(self):
        job = _job(status=AgentJobStatus.FAILED.value)
        _hold_for_chain_approval(job)
        assert job.status == AgentJobStatus.FAILED.value

    def test_a_gate_with_nothing_after_it_holds_nothing(self):
        job = _job(chain_config={"trigger_condition": "on_approval", "child_jobs": []})
        _hold_for_chain_approval(job)
        assert job.status == AgentJobStatus.COMPLETED.value

    def test_a_chain_already_triggered_is_not_re_held(self):
        job = _job()
        job.chain_triggered = True
        _hold_for_chain_approval(job)
        assert job.status == AgentJobStatus.COMPLETED.value


class TestDecidingTheGate:
    """The approve path, which mid-run approvals share.

    The ordinary path re-queues the job so it can carry out the action that was
    approved. A gate has no action to carry out -- its job is done -- so it must
    start the chain instead, and must not re-run anything.
    """

    @pytest.mark.asyncio
    async def test_approving_starts_the_chain_and_does_not_rerun_the_job(
        self, monkeypatch
    ):
        import app.modules.autonomy.application.job_action_checkpoint_decisions as mod

        triggered = {}
        requeued = []

        class _FakeExecutor:
            async def _trigger_chained_jobs(self, job, event, db):
                triggered["event"] = event
                return ["child-1"]

        monkeypatch.setattr(
            "app.services.autonomous_agent_executor.AutonomousAgentExecutor",
            _FakeExecutor,
        )

        job = _job()
        _hold_for_chain_approval(job)
        decided = await mod._decide_chain_gate(
            job,
            "approve",
            job.results["approval_checkpoint"],
            "looks right",
            dict(job.results),
            {},
            {},
            None,
            deps=_deps(requeued),
            db=_FakeDb(),
            current_user=_FakeUser(),
        )

        assert triggered["event"] == "approval"
        assert decided.status == AgentJobStatus.COMPLETED.value
        assert requeued == [], "a gate must not re-run the job it gates"

    @pytest.mark.asyncio
    async def test_rejecting_stops_the_pipeline_without_starting_anything(self):
        import app.modules.autonomy.application.job_action_checkpoint_decisions as mod

        requeued = []
        job = _job()
        _hold_for_chain_approval(job)
        decided = await mod._decide_chain_gate(
            job,
            "reject",
            job.results["approval_checkpoint"],
            "not yet",
            dict(job.results),
            {},
            {},
            None,
            deps=_deps(requeued),
            db=_FakeDb(),
            current_user=_FakeUser(),
        )

        assert decided.chain_triggered is True, "nothing may start it later"
        assert decided.status == AgentJobStatus.COMPLETED.value
        assert requeued == []


class _FakeDb:
    def add(self, _row):
        pass

    async def commit(self):
        pass


class _FakeUser:
    id = "user-1"


def _deps(requeued):
    class _Task:
        def delay(self, *args):
            requeued.append(args)

    class _Deps:
        execute_agent_job_task = _Task()

        @staticmethod
        def append_approval_event(*args, **kwargs):
            pass

        @staticmethod
        def sync_execution_strategy_state(*args, **kwargs):
            pass

    return _Deps()
