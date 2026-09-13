"""Backward edges: a stage asking for an earlier one to be redone.

The forward path assumes each stage got what it needed. The stage that finds
out otherwise is downstream -- an implement stage discovering the specification
left the tie-breaking unstated. Before this the run simply ended with an unmet
contract: correct, and useless, because the thing needing redoing was upstream
and nothing could say so.

The refusals are the feature. An unbounded backward edge is a ping-pong, and a
request with no diagnosis asks for the same work to be done the same way.
"""

import pytest

from app.services import agent_stage_rerun as rerun

pytestmark = pytest.mark.unit

GOOD_REASON = (
    "The specification leaves the rejection threshold unstated, so the "
    "implementation cannot be written to match the paper."
)


def _config(targets=("specify",), budget=2):
    return {"may_revisit": list(targets), "revisit_budget": budget}


class TestItRefusesWhatWouldBecomeALoop:
    def test_a_stage_with_no_declared_edges(self):
        verdict = rerun.evaluate(
            stage="specify", reason=GOOD_REASON, config={}, results={}
        )

        assert not verdict.ok
        assert "declares no backward edges" in verdict.error
        # And says what to do instead, because an unmet contract IS a result.
        assert "is a real result" in verdict.error

    def test_an_undeclared_target(self):
        verdict = rerun.evaluate(
            stage="find", reason=GOOD_REASON, config=_config(), results={}
        )

        assert not verdict.ok
        assert "not a stage this one may send work back to" in verdict.error
        assert "specify" in verdict.error, "name what IS allowed"

    def test_a_spent_budget(self):
        results = {}
        for i in range(2):
            rerun.record(
                results,
                from_stage="implement",
                to_stage="specify",
                reason=GOOD_REASON,
                iteration=i,
            )

        verdict = rerun.evaluate(
            stage="specify",
            reason=GOOD_REASON,
            config=_config(budget=2),
            results=results,
        )

        assert not verdict.ok
        assert "spent its backward-edge budget" in verdict.error

    def test_a_request_with_no_diagnosis(self):
        """ "Try again" asks for the same work done the same way."""
        verdict = rerun.evaluate(
            stage="specify", reason="try again", config=_config(), results={}
        )

        assert not verdict.ok
        assert "Say what was wrong" in verdict.error


class TestAValidRequestIsAllowed:
    def test_a_declared_edge_with_a_reason_and_budget(self):
        verdict = rerun.evaluate(
            stage="specify", reason=GOOD_REASON, config=_config(), results={}
        )

        assert verdict.ok
        assert verdict.stage == "specify"
        assert verdict.reason == GOOD_REASON

    def test_the_budget_is_shared_across_stages_not_per_edge(self):
        """Two stages each going back once is a pipeline converging. The same
        two sending work back and forth is a loop, and only a shared count can
        tell them apart."""
        results = {}
        rerun.record(
            results,
            from_stage="measure",
            to_stage="implement",
            reason=GOOD_REASON,
            iteration=1,
        )

        verdict = rerun.evaluate(
            stage="specify",
            reason=GOOD_REASON,
            config=_config(budget=1),
            results=results,
        )

        assert not verdict.ok, "a different stage still spends the same budget"


class TestTheLedgerRecordsTheDetour:
    def test_a_hop_is_written_with_its_reason(self):
        results = {}
        rerun.record(
            results,
            from_stage="implement",
            to_stage="specify",
            reason=GOOD_REASON,
            iteration=4,
        )

        hop = results[rerun.REVISIT_LEDGER_KEY][0]
        assert hop["from"] == "implement" and hop["to"] == "specify"
        assert "tie" in hop["reason"] or "threshold" in hop["reason"]
        assert hop["iteration"] == 4

    def test_spent_counts_what_the_ledger_holds(self):
        results = {}
        assert rerun.spent(results) == 0
        rerun.record(
            results, from_stage="a", to_stage="b", reason=GOOD_REASON, iteration=1
        )
        assert rerun.spent(results) == 1

    def test_a_run_that_never_went_back_reads_as_zero(self):
        assert rerun.spent(None) == 0
        assert rerun.spent({"other": "keys"}) == 0


@pytest.mark.asyncio
class TestTheFinaliserSendsItBackInsteadOfForward:
    """Both directions would be wrong. The next stage would start on the output
    this one just called the problem, while the earlier stage redid it
    underneath -- two versions of the same evidence in one run, with nothing to
    say which the result came from.
    """

    @staticmethod
    def _job():
        import uuid as _uuid

        from app.models.agent_job import AgentJob, AgentJobStatus

        job = AgentJob(
            id=_uuid.uuid4(),
            name="p: implement",
            goal="implement it",
            job_type="coding",
            status=AgentJobStatus.COMPLETED.value,
            iteration=4,
            max_iterations=6,
            tool_calls_used=0,
            max_tool_calls=50,
            llm_calls_used=0,
            max_llm_calls=50,
            error_count=0,
            config={"pipeline_stage": "implement", "may_revisit": ["specify"]},
            results={},
            execution_log=[],
        )
        job.root_job_id = _uuid.uuid4()
        return job

    async def test_a_request_goes_back_and_the_chain_does_not_go_forward(
        self, monkeypatch, db_session
    ):
        from app.services import agent_runtime_finalizer as fin

        job = self._job()
        went_back = {}

        async def _restart(*, root_job_id, stage_id, executor, db, note):
            went_back["stage"] = stage_id
            went_back["note"] = note

            class _Child:
                id = "child-1"

            return _Child()

        monkeypatch.setattr(
            "app.services.agent_pipeline_restart.restart_from_stage", _restart
        )

        handled = await fin._send_the_work_back(
            object(),
            job,
            {"stage": "specify", "reason": GOOD_REASON, "iteration": 4},
            db_session,
        )

        assert handled is True
        assert went_back["stage"] == "specify"
        assert went_back["note"] == GOOD_REASON, "the correction must travel"
        # And the detour is on the record: a result reached after going back is
        # not the same as one reached first try.
        assert job.results[rerun.REVISIT_LEDGER_KEY][0]["to"] == "specify"

    async def test_a_refused_restart_lets_the_stage_finish(
        self, monkeypatch, db_session
    ):
        """A stage that did its work and cannot get the detour should still
        record what it found -- that is more than the run had before."""
        from app.services import agent_pipeline_restart as restart_mod
        from app.services import agent_runtime_finalizer as fin

        job = self._job()

        async def _refuse(**kwargs):
            raise restart_mod.PipelineRestartError("specify is still running")

        monkeypatch.setattr(restart_mod, "restart_from_stage", _refuse)

        handled = await fin._send_the_work_back(
            object(),
            job,
            {"stage": "specify", "reason": GOOD_REASON, "iteration": 4},
            db_session,
        )

        assert handled is False
        phases = [e.get("phase") for e in job.execution_log]
        assert "stage_rerun_refused" in phases, (
            "a run whose detour was refused must not look like one that never " "asked"
        )


@pytest.mark.asyncio
class TestAHandoffIsNotABackwardEdge:
    """They look equivalent and are not.

    Measured: a `mine` stage correctly judged its profile too coarse to mine --
    exactly the condition its backward edge was declared for -- and created a
    handoff job to re-profile instead. `request_stage_rerun` was in its 120
    tools and its goal text said to send the work back. It chose the handoff.

    The harm is not the extra job. A handoff carries no `pipeline_stage`, so
    nothing re-derives the stages after it: the re-profile would have run and
    produced evidence no contract could consume, while `mine` still lacked a
    candidate.
    """

    @staticmethod
    def _ctx(may_revisit):
        import uuid as _uuid

        from app.models.agent_job import AgentJob, AgentJobStatus

        job = AgentJob(
            id=_uuid.uuid4(),
            name="p: mine",
            goal="mine fusion candidates",
            job_type="research",
            status=AgentJobStatus.RUNNING.value,
            iteration=2,
            max_iterations=10,
            config={"pipeline_stage": "mine", "may_revisit": may_revisit},
            results={},
            execution_log=[],
        )
        job.chain_depth = 0

        class _Ctx:
            pass

        ctx = _Ctx()
        ctx.job = job
        ctx.state = {}
        ctx.db = None
        ctx.user_id = _uuid.uuid4()
        return ctx

    async def _handoff(self, ctx, goal="Re-profile the workload"):
        from app.services import agent_tool_dispatch as dispatch

        class _Executor:
            pass

        provider = dispatch.build_autonomous_output_state_provider(_Executor())
        handler = provider._handlers["create_handoff"]
        return await handler(
            {"goal": goal, "expected_outputs": ["a finer profile"]}, ctx
        )

    async def test_a_stage_with_an_edge_is_pointed_at_it(self):
        result = await self._handoff(self._ctx(["profile"]))

        assert "request_stage_rerun" in result["error"]
        assert "profile" in result["error"], "name the stage it may go back to"
        # And why, so the refusal teaches rather than blocks.
        assert "outside the pipeline" in result["error"]

    async def test_a_stage_with_no_edge_may_still_hand_off(self):
        """The control. Handoffs remain the right tool for genuinely new work,
        and most jobs are not pipeline stages with backward edges at all."""
        result = await self._handoff(self._ctx([]))

        assert "request_stage_rerun" not in str(result.get("error") or "")
