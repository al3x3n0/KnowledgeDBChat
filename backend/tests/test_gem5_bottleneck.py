"""Reading what limited a simulated run.

The ranking here was wrong once, against real runs, and the tests are that
failure written down. It ranked a full issue queue first on a kernel where
idealising the L1 recovered 84% of the cycles and widening the issue queue
recovered 3.4% -- because it compared a share of *events* against a share of
*cycles*, and the queue fills precisely because the loads are missing.
"""

import pytest

from app.services import gem5_bottleneck as bn

# The strided read-modify-write kernel, at the numbers it actually produced.
MEMORY_BOUND = {
    "system.cpu.numCycles": 28052741.0,
    "simTicks": 14026370500.0,
    "system.cpu.ipc": 0.523553,
    "system.cpu.rename.IQFullEvents": 3057932.0,
    "system.cpu.rename.ROBFullEvents": 75.0,
    "system.cpu.rename.LQFullEvents": 0.0,
    "system.cpu.rename.SQFullEvents": 0.0,
    "system.cpu.rename.status::Blocked": 22563000.0,
    "system.cpu.dcache.demandMisses::total": 2097800.0,
    "system.cpu.dcache.demandMshrMisses::total": 524580.0,
    "system.cpu.dcache.demandAvgMissLatency::total": 86500.0,
    "system.cpu.lsq0.blockedByCache": 511632.0,
    "system.cpu.branchPred.condPredicted": 2000000.0,
    "system.cpu.branchPred.condIncorrect": 400.0,
    "system.cpu.fetchStats0.icacheStallCycles": 14000.0,
}


class TestTicksBecomeCycles:
    def test_the_ratio_is_derived_from_the_run(self):
        """A constant for 2 GHz would be silently wrong on any configuration
        that set a different clock."""
        assert bn.ticks_per_cycle(MEMORY_BOUND) == 500.0

    def test_a_run_with_no_cycles_has_no_ratio(self):
        assert bn.ticks_per_cycle({"simTicks": 1.0}) is None


class TestWhichStructureIsBinding:
    def test_the_dominant_structure_is_named_with_its_study(self):
        press = bn.backpressure(MEMORY_BOUND)

        assert press["dominant"] == "IQ"
        assert press["headroom_target"] == "issue_queue"

    def test_a_run_that_never_blocked_names_nothing(self):
        """Reporting a dominant structure from four zeroes would point every
        study at the issue queue by default."""
        press = bn.backpressure({"system.cpu.numCycles": 100.0})

        assert press["dominant"] is None
        assert press["headroom_target"] is None


class TestTheRankingThatWasWrong:
    def test_memory_outranks_the_queue_it_fills(self):
        """The measured truth on this kernel: L1 headroom 84%, issue queue
        3.4%. The previous ranking had them the other way round."""
        signals = bn.rank_signals(MEMORY_BOUND)

        assert signals[0]["headroom_target"] == "l1d_capacity"
        assert signals[1]["headroom_target"] == "issue_queue"

    def test_the_queue_signal_says_it_may_be_a_symptom(self):
        signals = bn.rank_signals(MEMORY_BOUND)
        queue = next(s for s in signals if s["headroom_target"] == "issue_queue")

        assert "downstream of memory latency" in queue["evidence"]

    def test_miss_cost_uses_mshr_misses_not_demand_misses(self):
        """A demand miss that merges into an outstanding request costs nothing
        extra; counting it inflates the figure by the degree of clustering."""
        ratio = bn.memory_cost_ratio(MEMORY_BOUND)
        expected = 524580.0 * (86500.0 / 500.0) / 28052741.0

        assert abs(ratio - expected) < 1e-9

    def test_a_run_that_touches_no_memory_has_no_memory_signal(self):
        compute = {
            "system.cpu.numCycles": 1000.0,
            "simTicks": 500000.0,
            "system.cpu.rename.IQFullEvents": 10.0,
            "system.cpu.rename.status::Blocked": 200.0,
        }

        assert bn.memory_cost_ratio(compute) is None
        assert [s["headroom_target"] for s in bn.rank_signals(compute)] == [
            "issue_queue"
        ]


class TestWhatToDoNext:
    def test_more_than_one_target_is_recommended(self):
        """Attribution is heuristic and has been wrong here. measure_headroom
        takes a list, so naming three costs one simulation each and cannot
        dead-end the study on a single bad rank."""
        study = bn.attribute(MEMORY_BOUND)["next_study"]

        assert "'l1d_capacity'" in study and "'issue_queue'" in study

    def test_a_run_with_no_signal_says_so_rather_than_inventing_one(self):
        study = bn.attribute({"system.cpu.numCycles": 10.0})["next_study"]

        assert "No signal stood out" in study

    def test_the_caveat_refuses_to_be_read_as_a_budget(self):
        """These overlap; a reader who sums them gets more than the run."""
        caveat = bn.attribute(MEMORY_BOUND)["caveat"]

        assert "sums are not" in caveat
        assert "between runs" in caveat


# PCE INT8 multi-head attention, at the numbers it actually produced. Rename
# barely blocks here; decode does. Scaling the backpressure signal by rename's
# own share ranked the load queue LAST, and idealising it recovered 4.68% --
# more than every other structure combined, and more than the three targets
# the ranking did recommend, which were worth 0.00%, 0.00% and 0.40%.
DECODE_BOUND = {
    "system.cpu.numCycles": 12867523.0,
    "simTicks": 6433761500.0,
    "system.cpu.ipc": 1.270732,
    "system.cpu.rename.IQFullEvents": 351136.0,
    "system.cpu.rename.ROBFullEvents": 9145.0,
    "system.cpu.rename.LQFullEvents": 9342457.0,
    "system.cpu.rename.SQFullEvents": 47124.0,
    "system.cpu.rename.status::Blocked": 74631.0,
    "system.cpu.decode.status::Blocked": 9658370.0,
    "system.cpu.dcache.demandMshrMisses::total": 1500.0,
    "system.cpu.dcache.demandAvgMissLatency::total": 90000.0,
    "system.cpu.fetchStats0.icacheStallCycles": 274078.0,
}


class TestBackpressureIsScaledByTheWorstBlockedStage:
    """Backpressure propagates backwards: a full structure blocks rename,
    which blocks decode, which blocks fetch. Which stage records the stall
    depends on the model, so rename's own share is not the measure."""

    def test_the_binding_structure_ranks_first(self):
        signals = bn.rank_signals(DECODE_BOUND)

        assert signals[0]["headroom_target"] == "load_queue"

    def test_it_is_scaled_by_decode_not_rename(self):
        signals = bn.rank_signals(DECODE_BOUND)
        queue = next(s for s in signals if s["headroom_target"] == "load_queue")

        assert queue["strength"] == pytest.approx(9658370.0 / 12867523.0, rel=1e-3)

    def test_the_recommendation_leads_with_it(self):
        assert "'load_queue'" in bn.attribute(DECODE_BOUND)["next_study"]
