"""The fusion chain worked inside one job and failed across a pipeline.

`find_fusion_candidates` reads the profiler's hot blocks out of the run's own
`actions_taken`, on purpose: handing a large structure to the next tool through
the model is expensive, and a truncated copy mines a different program than the
one that was profiled. That reasoning holds inside a job and breaks across a
pipeline, where profile and mine are different jobs.

Measured: a `mine` stage inherited a real profile -- 411M instructions, hottest
function at 63.2% -- and got "No hot blocks to mine. Run profile_c_workload
first", which is advice to redo work an earlier stage had already done. It
paused for want of new findings, twice, across two attempts at the stage.

A finding is the unit that crosses a stage boundary: the action ledger a child
inherits keeps a tool name and a success flag, not a result. So the blocks ride
on the finding, and the miner reads them from there when its own history has
none.
"""

import pytest

pytestmark = pytest.mark.unit

BLOCKS = [
    {"function": "main'2", "executions": 8388608, "instructions": ["fmul", "fadd"]},
    {"function": "main'1", "executions": 65536, "instructions": ["ldr", "str"]},
]


class TestTheProfileFindingCarriesItsBlocks:
    def test_the_finding_declares_hot_blocks(self):
        import inspect

        from app.services import agent_profile_sandbox as prof

        source = inspect.getsource(prof)
        assert '"hot_blocks": blocks[:MAX_CARRIED_BLOCKS]' in source

    def test_the_carry_is_bounded(self):
        """A finding is serialised into every checkpoint of every stage that
        inherits it, so an unbounded tail costs every downstream iteration."""
        from app.services.agent_profile_sandbox import MAX_CARRIED_BLOCKS

        assert 1 <= MAX_CARRIED_BLOCKS <= 50


class TestTheMinerReadsAnInheritedProfile:
    """Against the real lookup, not a fixture."""

    @staticmethod
    def _look(state):
        from app.services.agent_tool_dispatch import hot_blocks_from_findings

        return hot_blocks_from_findings(state)

    def test_an_inherited_profile_is_found(self):
        state = {
            "findings": [
                {"type": "dynamic_profile", "hot_blocks": BLOCKS, "inherited": True}
            ]
        }

        assert self._look(state) == BLOCKS

    def test_its_own_profile_wins_over_an_inherited_one(self):
        """A stage that re-profiled should mine what it just took."""
        own = [{"function": "fresh", "executions": 10}]
        state = {
            "findings": [
                {"type": "dynamic_profile", "hot_blocks": BLOCKS, "inherited": True},
                {"type": "dynamic_profile", "hot_blocks": own},
            ]
        }

        assert self._look(state) == own

    def test_the_most_recent_of_its_own_wins(self):
        older = [{"function": "older", "executions": 1}]
        newer = [{"function": "newer", "executions": 2}]
        state = {
            "findings": [
                {"type": "dynamic_profile", "hot_blocks": older},
                {"type": "dynamic_profile", "hot_blocks": newer},
            ]
        }

        assert self._look(state) == newer

    @pytest.mark.parametrize(
        "state",
        [
            None,
            {},
            {"findings": "not a list"},
            {"findings": []},
            {"findings": [{"type": "benchmark_measurement", "hot_blocks": BLOCKS}]},
            {"findings": [{"type": "dynamic_profile"}]},
            {"findings": [{"type": "dynamic_profile", "hot_blocks": []}]},
        ],
    )
    def test_nothing_to_find_is_none_not_a_crash(self, state):
        """None is the honest answer, and the caller turns it into the message
        telling the run to profile first -- which is right when there really is
        no profile."""
        assert self._look(state) is None


class TestTheAdviceWasWrongNotJustUnhelpful:
    def test_the_miner_no_longer_only_reads_local_actions(self):
        """ "Run profile_c_workload first" told a stage to redo work its parent
        had already done -- the worst kind of error message, one that sends the
        reader somewhere useless with confidence."""
        import inspect

        from app.services import agent_tool_dispatch as dispatch

        source = inspect.getsource(dispatch)
        assert "hot_blocks_from_findings" in source
        assert "different jobs" in source
