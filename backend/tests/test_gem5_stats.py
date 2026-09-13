"""Reading a gem5 stats.txt.

The referee's whole job is to produce a number that settles a prediction, so
the failure that matters here is returning None because a statistic was spelled
differently in another gem5 version, or returning a figure whose provenance
cannot be checked.
"""

from app.services import gem5_stats

SAMPLE = """
---------- Begin Simulation Statistics ----------
simSeconds                                   0.000234                       # Seconds simulated
simTicks                                    234000000                       # Ticks simulated
simInsts                                       410000                       # Number of instructions simulated
system.cpu.numCycles                           468000                       # Number of cpu cycles simulated
system.cpu.dcache.overallHits::total            50123                       # number of hits
system.cpu.branchPred.condPredicted             12000                       # Number of conditional branches predicted
---------- End Simulation Statistics   ----------
"""


def test_scalar_statistics_are_parsed():
    stats = gem5_stats.parse(SAMPLE.splitlines())

    assert stats["simSeconds"] == 0.000234
    assert stats["system.cpu.numCycles"] == 468000
    assert stats["system.cpu.dcache.overallHits::total"] == 50123
    assert "---------- Begin Simulation Statistics ----------" not in stats


def test_the_summary_answers_how_long_it_took():
    summary = gem5_stats.summarize(gem5_stats.parse(SAMPLE.splitlines()))

    assert summary["cycles"] == 468000
    assert summary["instructions"] == 410000
    assert summary["ipc"] == 0.8761
    assert summary["sim_seconds"] == 0.000234


def test_the_summary_says_which_statistic_it_read():
    """A cycle count is only comparable to one from the same statistic."""
    summary = gem5_stats.summarize(gem5_stats.parse(SAMPLE.splitlines()))

    assert summary["cycles_stat"] == "system.cpu.numCycles"
    assert summary["instructions_stat"] == "simInsts"


def test_an_alternative_spelling_is_still_found():
    """Statistic names move between gem5 versions."""
    stats = gem5_stats.parse(
        ["system.switch_cpus.numCycles 900 # cycles", "simInsts 450 # insts"]
    )

    summary = gem5_stats.summarize(stats)

    assert summary["cycles"] == 900
    assert summary["cycles_stat"] == "system.switch_cpus.numCycles"
    assert summary["ipc"] == 0.5


def test_a_missing_cycle_count_yields_no_ipc_rather_than_a_guess():
    summary = gem5_stats.summarize(gem5_stats.parse(["simInsts 450 # insts"]))

    assert summary["cycles"] is None
    assert summary["ipc"] is None


def test_speedup_refuses_to_invent_a_result():
    assert gem5_stats.speedup(1000, 800) == 1.25
    assert gem5_stats.speedup(None, 800) is None
    assert gem5_stats.speedup(1000, 0) is None
    assert gem5_stats.speedup(1000, None) is None


def test_distribution_rows_do_not_break_the_parse():
    stats = gem5_stats.parse(
        [
            "system.cpu.dcache.demandMissLatency::samples 120 200 300 # dist",
            "system.cpu.numCycles 42 # cycles",
        ]
    )

    assert stats["system.cpu.dcache.demandMissLatency::samples"] == 120
    assert stats["system.cpu.numCycles"] == 42
