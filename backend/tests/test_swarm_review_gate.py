"""Whether a swarm's merged verdict stops for a person.

The first swarm run whose roles both actually benchmarked produced a verdict
nobody saw: it completed, wrote `swarm_fan_in` into its results, and the only
thing that ever read it was a hand-written SQL query. A swarm costs several
agents and produces exactly one judgement, so completing silently wastes the
expensive part.

The risk in adding a gate is not the new pause but the release. This job has
already done its work -- the merge is in its results -- so approving must
never re-queue it, and must start whatever the chain was waiting for even
though the chain was never told the job completed.
"""

import pytest

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services import agent_swarm_review_gate as gate
from app.services.agent_runtime_finalizer import (
    SWARM_REVIEW_CHECKPOINT,
    _hold_for_chain_approval,
    _hold_for_swarm_review,
)

pytestmark = pytest.mark.unit


CLEAN = {
    "corroborated_count": 3,
    "contested_count": 0,
    "inconclusive_count": 0,
    "typed_agreement": 1.0,
}
CONTESTED = {"corroborated_count": 1, "contested_count": 1, "typed_agreement": 0.5}
UNRESOLVED = {
    "corroborated_count": 0,
    "contested_count": 0,
    "inconclusive_count": 1,
    "typed_agreement": None,
}


def _job(fan_in=None, **over):
    fields = {
        "status": AgentJobStatus.COMPLETED.value,
        "iteration": 3,
        "results": {"swarm_fan_in": dict(fan_in)} if fan_in is not None else {},
        "config": {},
    }
    fields.update(over)
    job = AgentJob(**fields)
    job.execution_log = []
    return job


class TestWhatWarrantsAPerson:
    def test_a_clean_corroboration_does_not_stop_anyone(self):
        """The case that must pass through. A gate met on every clean run is
        one people learn to approve without reading."""
        assert gate.decide(CLEAN).hold is False

    def test_a_disagreement_stops(self):
        assert gate.decide(CONTESTED).hold is True

    def test_an_unresolvable_measurement_stops(self):
        """Nearly as actionable as a disagreement: the fix is usually a rerun
        on a quiet machine."""
        assert gate.decide(UNRESOLVED).hold is True

    def test_a_missing_role_stops(self):
        """The merge is over a subset, and the absent role may be the one that
        disagreed."""
        decision = gate.decide(
            {
                **CLEAN,
                "conflicts": [
                    {
                        "type": "incomplete_swarm",
                        "description": "Only 1/2 sibling jobs reached a terminal state.",
                    }
                ],
            }
        )
        assert decision.hold is True
        assert "1/2" in decision.reasons[0]

    def test_a_swarm_that_cross_checked_nothing_stops(self):
        """No conflict is reported because there was no comparison to
        conflict, which is exactly why it needs saying: N opinions, zero
        corroboration."""
        decision = gate.decide(
            {"corroborated_count": 0, "contested_count": 0, "typed_agreement": None}
        )
        assert decision.hold is True

    def test_the_reasons_are_specific_enough_to_act_on(self):
        reasons = gate.decide(CONTESTED).reasons
        assert reasons and "disagreed" in reasons[0]


class TestThePolicy:
    def test_never_lets_a_disagreement_through(self):
        assert gate.decide(CONTESTED, gate.NEVER).hold is False

    def test_always_holds_a_clean_run_and_says_why(self):
        decision = gate.decide(CLEAN, gate.ALWAYS)
        assert decision.hold is True
        assert decision.reasons == ["policy is to review every swarm merge"]

    def test_an_unknown_policy_falls_back_rather_than_raising(self):
        """This runs in the finalizer, after the work is done. A typo in a job
        config must not be what loses it."""
        assert gate.normalize_policy("nonsense") == gate.ON_DISPUTE
        assert gate.decide(CONTESTED, "nonsense").hold is True

    def test_policy_names_are_forgiving_about_shape(self):
        assert gate.normalize_policy("ON-Dispute") == gate.ON_DISPUTE


class TestHoldingTheJob:
    def test_a_disputed_merge_pauses_instead_of_finishing(self):
        job = _job(CONTESTED)
        _hold_for_swarm_review(job)
        assert job.status == AgentJobStatus.PAUSED.value
        assert job.current_phase == "awaiting_approval"

    def test_a_clean_merge_is_left_alone(self):
        job = _job(CLEAN)
        _hold_for_swarm_review(job)
        assert job.status == AgentJobStatus.COMPLETED.value

    def test_a_job_that_is_not_a_swarm_is_left_alone(self):
        job = _job()
        _hold_for_swarm_review(job)
        assert job.status == AgentJobStatus.COMPLETED.value

    def test_a_job_that_did_not_complete_is_left_alone(self):
        """Nothing to review: there is no verdict yet."""
        job = _job(CONTESTED, status=AgentJobStatus.FAILED.value)
        _hold_for_swarm_review(job)
        assert job.status == AgentJobStatus.FAILED.value

    def test_it_writes_the_payload_the_approval_queue_reads(self):
        from app.services.agent_job_queue_helpers import extract_approval_checkpoint

        job = _job(CONTESTED)
        _hold_for_swarm_review(job)
        assert extract_approval_checkpoint(job)
        assert job.results["approval_checkpoint"]["checkpoint_type"] == (
            SWARM_REVIEW_CHECKPOINT
        )

    def test_the_message_says_what_to_look_at(self):
        job = _job(CONTESTED)
        _hold_for_swarm_review(job)
        assert "disagreed" in job.results["approval_checkpoint"]["message"]

    def test_a_per_job_override_wins_over_the_default(self):
        job = _job(CONTESTED, config={"swarm_review_gate": "never"})
        _hold_for_swarm_review(job)
        assert job.status == AgentJobStatus.COMPLETED.value

    def test_a_per_job_override_can_hold_a_clean_run(self):
        job = _job(CLEAN, config={"swarm_review_gate": "always"})
        _hold_for_swarm_review(job)
        assert job.status == AgentJobStatus.PAUSED.value


class TestItComposesWithTheChainGate:
    def test_one_run_does_not_hold_twice(self):
        """Both write results['approval_checkpoint']. Holding for the swarm
        and then again for the chain would mean approving the same finished
        work in two places -- and the second hold would overwrite the first
        payload, so the reader would never learn why it first stopped."""
        job = _job(
            CONTESTED,
            chain_config={
                "trigger_condition": "on_approval",
                "child_jobs": [{"name": "next"}],
            },
        )
        _hold_for_swarm_review(job)
        _hold_for_chain_approval(job)

        assert job.status == AgentJobStatus.PAUSED.value
        assert job.results["approval_checkpoint"]["checkpoint_type"] == (
            SWARM_REVIEW_CHECKPOINT
        ), "the swarm gate ran first and must be the reason on record"

    def test_a_clean_merge_still_reaches_the_chain_gate(self):
        """The swarm gate declining to hold must not consume the chain's."""
        from app.services.agent_runtime_finalizer import CHAIN_GATE_CHECKPOINT

        job = _job(
            CLEAN,
            chain_config={
                "trigger_condition": "on_approval",
                "child_jobs": [{"name": "next"}],
            },
        )
        _hold_for_swarm_review(job)
        _hold_for_chain_approval(job)

        assert job.status == AgentJobStatus.PAUSED.value
        assert job.results["approval_checkpoint"]["checkpoint_type"] == (
            CHAIN_GATE_CHECKPOINT
        )


class TestTheChainIsNotStranded:
    """The hold happens before the finalizer triggers the chain, so at release
    the chain has never been told this job finished. An ON_APPROVAL chain
    answers only the `approval` event and every other chain answers only
    `complete` -- and this gate fires on the verdict, so it can be holding
    either kind. Sending one event would leave the other's stages waiting
    forever behind a parent marked done.
    """

    def test_an_ordinary_chain_answers_complete_and_not_approval(self):
        job = _job(CONTESTED, chain_config={"child_jobs": [{"name": "next"}]})
        job.goal_contract = None
        assert job.should_trigger_chain("approval") is False
        assert job.should_trigger_chain("complete") is True

    def test_an_approval_chain_answers_approval_and_not_complete(self):
        job = _job(
            CONTESTED,
            chain_config={
                "trigger_condition": "on_approval",
                "child_jobs": [{"name": "next"}],
            },
        )
        assert job.should_trigger_chain("complete") is False
        assert job.should_trigger_chain("approval") is True

    def test_trying_approval_then_complete_covers_both(self):
        """What the release handler does. Whichever matches fires; the
        chain_triggered flag makes the second attempt a no-op."""
        for chain_config, expected in (
            ({"child_jobs": [{"name": "n"}]}, True),
            (
                {"trigger_condition": "on_approval", "child_jobs": [{"name": "n"}]},
                True,
            ),
        ):
            job = _job(CONTESTED, chain_config=chain_config)
            fired = job.should_trigger_chain("approval") or job.should_trigger_chain(
                "complete"
            )
            assert fired is expected, chain_config


class TestTheAggregatorIsReachedAtAll:
    """The swarm fan-in aggregator is a *deterministic runner*, and that path
    returns before `finalize_job` -- so the gate, which lived only in
    finalize_job, could never see the one job it exists for.

    Found by running a swarm rather than by reading: the merge came back
    `inconclusive=1`, the exact case `on_dispute` holds for, and the
    aggregator completed with no checkpoint written. These pin the seam so it
    cannot silently move back.
    """

    def test_the_gate_is_applied_on_the_deterministic_path(self):
        import inspect

        from app.services.autonomous_agent_executor import AutonomousAgentExecutor

        source = inspect.getsource(AutonomousAgentExecutor.execute_job)
        assert "hold_for_swarm_review(job)" in source, (
            "the deterministic-runner branch must apply the hold; it does not "
            "reach finalize_job"
        )

    def test_a_held_job_does_not_start_its_chain_on_that_path(self):
        """The chain starts on the person's decision. Starting it here would
        run the next stage on a verdict nobody has accepted."""
        import inspect

        from app.services.autonomous_agent_executor import AutonomousAgentExecutor

        source = inspect.getsource(AutonomousAgentExecutor.execute_job)
        held = source.index("hold_for_swarm_review(job)")
        guard = source.index("AgentJobStatus.PAUSED.value", held)
        trigger = source.index("_trigger_chained_jobs", held)
        assert (
            guard < trigger
        ), "the paused check must come before the chain is triggered"

    def test_the_public_name_is_what_the_executor_imports(self):
        from app.services import agent_runtime_finalizer as fin

        assert fin.hold_for_swarm_review is fin._hold_for_swarm_review


class TestThePolicyReachesTheJobThatReadsIt:
    """`config.swarm_review_gate` is set on the swarm a person launches, and
    read on the aggregator that holds the verdict. Those are different jobs,
    and chained children do not inherit config by default -- so the documented
    per-job override applied to nothing at all until the aggregator template
    carried it across.
    """

    def test_the_aggregator_template_carries_the_policy(self):
        import inspect

        from app.services import agent_swarm_chain_config

        source = inspect.getsource(agent_swarm_chain_config)
        assert (
            '"swarm_review_gate": cfg.get("swarm_review_gate")' in source
        ), "the fan-in aggregator config must carry the parent's gate policy"

    def test_an_aggregator_without_a_policy_uses_the_default(self):
        """None must mean "fall back", not "never hold"."""
        job = _job(CONTESTED, config={"swarm_review_gate": None})
        _hold_for_swarm_review(job)
        assert job.status == AgentJobStatus.PAUSED.value


class TestReleasingTheGate:
    """Approving a held verdict must complete the job, start whatever the
    chain was waiting for, and never re-queue the aggregator -- its work is
    already in its results, which is what the person just read.

    The chain part is the trap. The hold happens before the finalizer would
    have triggered anything, so at release the chain has never been told this
    job finished; an ON_APPROVAL chain answers only `approval` and every other
    chain only `complete`, and the gate fires on the verdict rather than on
    how the chain was configured.
    """

    @staticmethod
    def _deps(requeued):
        from app.modules.autonomy.application.job_action_contracts import (
            JobActionDependencies,
        )

        class _Task:
            @staticmethod
            def delay(*args, **kwargs):
                requeued.append(args)

        def _noop(*args, **kwargs):
            return None

        return JobActionDependencies(
            is_job_visible=lambda *a, **k: True,
            approval_payload_from_results=lambda *a, **k: {},
            load_latest_checkpoint=_noop,
            append_operator_intervention=_noop,
            append_step_event=_noop,
            normalize_checkpoint_action_patch=lambda patch: patch,
            apply_checkpoint_action_patch=_noop,
            set_current_plan_step_status=_noop,
            append_approval_event=lambda payload, state, event: None,
            sync_execution_strategy_state=_noop,
            quick_start_relaunch_dispatcher=None,
            infer_coding_swarm_preset_key=_noop,
            extract_swarm_collaboration=_noop,
            build_swarm_collaboration_payload=_noop,
            store_swarm_collaboration=_noop,
            execute_agent_job_task=_Task,
            generate_job_summary=_noop,
        )

    class _DB:
        def add(self, *_):
            return None

        async def commit(self):
            return None

    class _User:
        id = "11111111-1111-1111-1111-111111111111"

    async def _decide(self, action, chain_config, monkeypatch, triggered):
        from app.modules.autonomy.application import (
            job_action_checkpoint_decisions as decisions,
        )

        job = _job(CONTESTED, chain_config=chain_config)
        _hold_for_swarm_review(job)
        assert job.status == AgentJobStatus.PAUSED.value

        async def _fake_trigger(self_, j, event, db):
            triggered.append(event)
            # Mirror the real guard: a chain fires once.
            if j.chain_triggered or not j.should_trigger_chain(event):
                return []
            j.chain_triggered = True
            return ["child"]

        from app.services.autonomous_agent_executor import AutonomousAgentExecutor

        monkeypatch.setattr(
            AutonomousAgentExecutor, "_trigger_chained_jobs", _fake_trigger
        )

        requeued = []
        await decisions._decide_swarm_review(
            job,
            action,
            dict(job.results["approval_checkpoint"]),
            "looked at it",
            dict(job.results),
            {},
            {},
            None,
            deps=self._deps(requeued),
            db=self._DB(),
            current_user=self._User(),
        )
        return job, triggered, requeued

    async def test_approving_completes_without_re_running_the_job(self, monkeypatch):
        triggered = []
        job, _, requeued = await self._decide(
            "approve", {"child_jobs": [{"name": "next"}]}, monkeypatch, triggered
        )
        assert job.status == AgentJobStatus.COMPLETED.value
        assert requeued == [], "the merge is done; re-running it would redo it"

    async def test_an_ordinary_chain_is_started(self, monkeypatch):
        triggered = []
        job, triggered, _ = await self._decide(
            "approve", {"child_jobs": [{"name": "next"}]}, monkeypatch, triggered
        )
        assert job.chain_triggered is True, "an on_complete chain must not strand"
        assert "complete" in triggered

    async def test_an_approval_chain_is_started(self, monkeypatch):
        triggered = []
        job, triggered, _ = await self._decide(
            "approve",
            {"trigger_condition": "on_approval", "child_jobs": [{"name": "next"}]},
            monkeypatch,
            triggered,
        )
        assert job.chain_triggered is True
        assert "approval" in triggered

    async def test_rejecting_stops_the_pipeline(self, monkeypatch):
        triggered = []
        job, triggered, requeued = await self._decide(
            "reject", {"child_jobs": [{"name": "next"}]}, monkeypatch, triggered
        )
        assert job.status == AgentJobStatus.COMPLETED.value
        assert job.chain_triggered is True, "nothing may start it later"
        assert triggered == [], "a rejected verdict starts no stage"
        assert requeued == []

    async def test_the_checkpoint_is_cleared_so_it_cannot_be_decided_twice(
        self, monkeypatch
    ):
        triggered = []
        job, _, _ = await self._decide(
            "approve", {"child_jobs": [{"name": "next"}]}, monkeypatch, triggered
        )
        assert job.results.get("approval_checkpoint") in (None, {})
