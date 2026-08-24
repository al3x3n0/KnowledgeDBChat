"""Counters sampled over time, which is what a hardware predictor reads.

Run totals cannot train or evaluate a predictor: "the program missed cache 4M
times" and "here is the miss rate every 200k instructions" are different data,
and only the second has a time axis.
"""

from __future__ import annotations

import pytest

from app.services import agent_gem5_sandbox as gem5
from app.services import gem5_stats

STATS = """
---------- Begin Simulation Statistics ----------
system.cpu.numCycles                    1000    # cycles
system.cpu.dcache.overallMisses          10     # misses
system.clk_domain.clock                   500   # constant
---------- End Simulation Statistics ----------
---------- Begin Simulation Statistics ----------
system.cpu.numCycles                    2000    # cycles
system.cpu.dcache.overallMisses         900     # misses
system.clk_domain.clock                   500   # constant
---------- End Simulation Statistics ----------
---------- Begin Simulation Statistics ----------
system.cpu.numCycles                    1500    # cycles
system.cpu.dcache.overallMisses          20     # misses
system.clk_domain.clock                   500   # constant
---------- End Simulation Statistics ----------
"""


def test_each_dump_is_one_interval():
    intervals = gem5_stats.parse_intervals(STATS.splitlines())

    assert len(intervals) == 3
    assert intervals[1]["system.cpu.numCycles"] == 2000


def test_a_run_that_never_sampled_is_a_trace_of_length_one():
    """Not an error -- a total, and the caller can see that from the length."""
    single = STATS.split("---------- Begin")[1]
    intervals = gem5_stats.parse_intervals(("---------- Begin" + single).splitlines())

    assert len(intervals) == 1


def test_constant_counters_are_dropped():
    """gem5 emits several hundred; clock periods and configured sizes are
    identical in every interval and cannot predict anything that changes."""
    intervals = gem5_stats.parse_intervals(STATS.splitlines())

    varying = gem5_stats.varying_counters(intervals)

    assert "system.clk_domain.clock" not in varying
    assert "system.cpu.dcache.overallMisses" in varying


def test_counters_are_ranked_by_relative_not_absolute_movement():
    """Misses swing 10 -> 900 -> 20; cycles only 1000 -> 2000. Ranking by raw
    spread would put cycles first purely for being measured in thousands."""
    intervals = gem5_stats.parse_intervals(STATS.splitlines())

    varying = gem5_stats.varying_counters(intervals)

    assert varying[0] == "system.cpu.dcache.overallMisses"


def test_a_single_interval_has_nothing_to_vary():
    assert gem5_stats.varying_counters([{"a": 1.0}]) == []


def test_series_are_aligned_by_interval():
    intervals = gem5_stats.parse_intervals(STATS.splitlines())

    series = gem5_stats.as_series(intervals, ["system.cpu.numCycles"])

    assert series["system.cpu.numCycles"] == [1000.0, 2000.0, 1500.0]


def test_a_counter_absent_from_an_interval_reads_as_zero():
    series = gem5_stats.as_series([{"a": 5.0}, {}], ["a"])

    assert series["a"] == [5.0, 0.0]


def test_the_sample_macro_is_the_verified_encoding():
    """m5 pseudo-ops on AArch64 are `0xff000110 | (func << 16)` and
    DUMP_RESET_STATS is func 0x42. util/m5 is absent from this image -- the
    gem5 build was stripped to 574 MB -- so the instruction is emitted
    directly. Verified against the image: four calls produced four stats
    sections with counts reset between them."""
    assert "0xff420110" in gem5.M5_SAMPLE_MACRO
    assert 0xFF000110 | (0x42 << 16) == 0xFF420110


@pytest.mark.asyncio
async def test_a_workload_that_never_samples_is_refused():
    """Without a single M5_SAMPLE() this returns one total and would be read
    as a trace. Refused before any simulation, so it costs nothing."""
    result = await gem5.sample_counters(code="int main(void){return 0;}")

    assert result["success"] is False
    assert "never calls M5_SAMPLE()" in result["error"]
    assert "one total rather than a trace" in result["error"]


@pytest.mark.asyncio
async def test_an_unknown_core_is_refused_with_the_list():
    result = await gem5.sample_counters(
        code="int main(void){ M5_SAMPLE(); return 0; }", cpu_type="M1Ultra"
    )

    assert result["success"] is False
    assert "Unknown cpu_type" in result["error"]
