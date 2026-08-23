"""One measurement is not a measurement.

Re-taking this project's per-instruction table produced latencies agreeing to
within 5% and throughputs disagreeing by 12-55%, from a harness whose controls
were passing. Both had come out of one run each originally and both looked
equally credible; only taking them again separated them.
"""

from __future__ import annotations

import pytest

from app.services import agent_measurement_replication as replication
from app.services import agent_measurement_validity as validity


def _result(metrics: dict, success: bool = True) -> dict:
    return {
        "success": success,
        "data": {"reported_metrics": {k: [v] for k, v in metrics.items()}},
        "findings": [{"type": "benchmark_measurement", "title": "a kernel"}],
    }


def test_a_number_that_holds_across_calls_is_reproduced():
    verdict = replication.judge(
        [{"cycles_per_op": 4.01}, {"cycles_per_op": 4.00}, {"cycles_per_op": 4.02}]
    )

    assert verdict["all_reproduced"] is True
    assert verdict["metrics"]["cycles_per_op"]["reproducible"] is True
    assert verdict["metrics"]["cycles_per_op"]["median"] == 4.01


def test_a_number_that_moves_between_calls_is_not():
    """The throughputs: 0.567, 0.878, 1.021 across three runs of one harness."""
    verdict = replication.judge(
        [
            {"recip_throughput": 0.567},
            {"recip_throughput": 0.878},
            {"recip_throughput": 1.021},
        ]
    )

    assert verdict["all_reproduced"] is False
    assert verdict["not_reproduced"] == ["recip_throughput"]
    assert verdict["metrics"]["recip_throughput"]["spread"] > 0.5


def test_latencies_and_throughputs_are_judged_separately():
    """The re-take's actual shape: the same runs carried one metric that held
    and one that did not, and reporting a single verdict for the call would
    have thrown away the good one or kept the bad one."""
    verdict = replication.judge(
        [
            {"latency": 4.01, "recip_throughput": 0.567},
            {"latency": 4.00, "recip_throughput": 0.878},
            {"latency": 4.02, "recip_throughput": 1.021},
        ]
    )

    assert verdict["reproduced"] == ["latency"]
    assert verdict["not_reproduced"] == ["recip_throughput"]
    assert verdict["all_reproduced"] is False


def test_a_metric_only_some_runs_reported_did_not_reproduce():
    verdict = replication.judge(
        [{"a": 1.0, "b": 2.0}, {"a": 1.0}, {"a": 1.0, "b": 2.0}]
    )

    assert verdict["metrics"]["b"]["reported_by_all_runs"] is False
    assert verdict["metrics"]["b"]["reproducible"] is False


def test_no_metrics_at_all_is_not_agreement():
    """A program that printed nothing gives nothing to reproduce, and calling
    that success is the same mistake as calling a skipped control a pass."""
    verdict = replication.judge([{}, {}, {}])

    assert verdict["any_metrics"] is False
    assert verdict["all_reproduced"] is False


def test_spread_needs_more_than_one_sample():
    assert replication.spread([4.0]) is None
    assert replication.spread([]) is None


@pytest.mark.asyncio
async def test_a_measurement_is_taken_several_times():
    calls = []

    async def _once():
        calls.append(1)
        return _result({"ns_per_op": 4.0 + len(calls) * 0.01})

    result = await replication.run_replicated(_once, "benchmark_c_snippet")

    assert len(calls) == 3
    assert result["data"]["replication"]["all_reproduced"] is True
    assert result["data"]["replication"]["calls"] == 3


@pytest.mark.asyncio
async def test_the_record_reaches_the_findings_not_only_the_result():
    """A contract reads findings; a reproduction record that stops at the tool
    result cannot be checked by one."""

    async def _once():
        return _result({"ns_per_op": 4.0})

    result = await replication.run_replicated(_once, "benchmark_c_snippet")

    assert result["findings"][0]["replication"]["all_reproduced"] is True


@pytest.mark.asyncio
async def test_the_median_run_is_returned_not_the_first():
    """If one call is disturbed, the answer returned should be the one the
    other two outvote."""
    values = [9.0, 4.0, 4.1]

    async def _once():
        return _result({"ns_per_op": values.pop(0)})

    result = await replication.run_replicated(_once, "benchmark_c_snippet")

    assert result["data"]["reported_metrics"]["ns_per_op"] == [4.1]


@pytest.mark.asyncio
async def test_a_failed_call_is_kept_rather_than_hidden():
    """Two successes and a failure is a different situation from three
    successes."""
    outcomes = [
        _result({"ns_per_op": 4.0}),
        {"success": False, "error": "oom"},
        _result({"ns_per_op": 4.0}),
    ]

    async def _once():
        return outcomes.pop(0)

    result = await replication.run_replicated(_once, "benchmark_c_snippet")

    assert result["data"]["replication"]["failed_calls"] == 1
    assert result["data"]["replication"]["runs"] == 2


@pytest.mark.asyncio
async def test_every_call_failing_still_returns_a_result():
    async def _once():
        return {"success": False, "error": "sandbox down"}

    result = await replication.run_replicated(_once, "benchmark_c_snippet")

    assert result["success"] is False
    assert result["data"]["replication"]["all_reproduced"] is False


def test_deterministic_tools_are_not_replicated():
    """Callgrind counts, llvm-mca and gem5 give the same answer every time;
    three calls would buy that answer at three times the cost."""
    assert replication.is_replicated("benchmark_c_snippet") is True
    for tool in ("profile_c_workload", "analyze_snippet_cycles", "simulate_c_workload"):
        assert replication.is_replicated(tool) is False


# --- the contract predicate ----------------------------------------------


def _state_with(record: dict) -> dict:
    return {
        "findings": [
            {
                "type": "benchmark_measurement",
                "title": "a kernel",
                "replication": record,
            }
        ]
    }


def test_a_contract_can_require_measurements_to_reproduce():
    state = _state_with(
        {
            "runs": 3,
            "band": 0.05,
            "all_reproduced": False,
            "not_reproduced": ["recip_throughput"],
            "metrics": {"recip_throughput": {"reproducible": False, "spread": 0.55}},
        }
    )

    result = validity.evaluate({"validity": {"measurements_reproduce": True}}, state)

    assert "validity:measurements_reproduce" in result["missing"]


def test_a_reproduced_measurement_satisfies_it():
    state = _state_with(
        {
            "runs": 3,
            "band": 0.05,
            "all_reproduced": True,
            "not_reproduced": [],
            "metrics": {},
        }
    )

    result = validity.evaluate({"validity": {"measurements_reproduce": True}}, state)

    assert result["missing"] == []


def test_the_remedy_says_not_to_quote_one_run():
    state = _state_with(
        {
            "runs": 3,
            "band": 0.05,
            "all_reproduced": False,
            "not_reproduced": ["recip_throughput"],
            "metrics": {},
        }
    )
    result = validity.evaluate({"validity": {"measurements_reproduce": True}}, state)

    lines = validity.explain(result["missing"], result["details"])

    assert lines and "recip_throughput" in lines[0]
    assert "instead of quoting one run" in lines[0]


def test_replication_is_described_for_the_prompt():
    described = validity.describe({"measurements_reproduce": True})

    assert any(
        "benchmark_c_snippet" in line and "3 times" in line for line in described
    )


def test_replication_measures_variance_not_bias():
    """The infinity defect reproduced perfectly, run after run, because it was
    wrong the same way every time. This is pinned so the module is never read
    as catching more than it does."""
    identical = [{"cycles_per_op": 3.339}] * 3

    verdict = replication.judge(identical)

    assert verdict["all_reproduced"] is True, (
        "a measurement that is wrong the same way every time reproduces; "
        "controls and value checks are what catch that, not this"
    )


def test_a_control_call_is_never_itself_replicated():
    """The controls already take a median over 31 rounds internally.
    Replicating them would multiply the cost of verifying the instrument by
    the cost of using it -- and the marker is explicit rather than a label
    match, because a renamed label would fail silently and expensively."""
    from app.services import agent_tool_controls as controls

    control = controls.controls_for("benchmark_c_snippet")[0]
    assert controls.is_control_call(dict(control.params)) is False
    assert controls.is_control_call({controls.CONTROL_MARKER: True}) is True
