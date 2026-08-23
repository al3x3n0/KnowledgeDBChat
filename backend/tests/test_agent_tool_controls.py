"""A contract that checks results still assumes the instrument worked.

These are the two failures that motivated the module, written as tests: a host
timing position rather than instructions, and a host the null control called
clean while a chain of known ratio 2.0 read 3.47.
"""

from __future__ import annotations

import pytest

from app.services import agent_measurement_validity as validity
from app.services import agent_tool_controls as controls


def _ok(ratio: float) -> dict:
    return {"success": True, "data": {"stdout": f"control_ratio={ratio:.6f}\n"}}


def _control(name: str) -> controls.Control:
    return next(
        c for c in controls.controls_for("benchmark_c_snippet") if c.name == name
    )


def test_a_correct_instrument_passes():
    assert controls.judge(_control("null_control"), _ok(1.0))["passed"] is True
    assert controls.judge(_control("scale_control"), _ok(1.9979))["passed"] is True


def test_a_host_timing_position_is_caught():
    """Two identical chains read 0.53 on a loaded machine, and every latency
    this project had measured was a ratio of exactly that kind."""
    verdict = controls.judge(_control("null_control"), _ok(0.53))

    assert verdict["passed"] is False
    assert verdict["actual"] == 0.53
    assert "timing position" in verdict["reason"]


def test_the_scale_control_catches_what_the_null_control_cannot():
    """The case that earned the second control: a disturbance hitting two
    identical chains cancels, so the null control read 1.0000 four times while
    the host could not measure a difference at all."""
    assert controls.judge(_control("null_control"), _ok(1.0))["passed"] is True
    assert controls.judge(_control("scale_control"), _ok(3.47))["passed"] is False


def test_a_control_that_could_not_run_is_a_failure_not_a_skip():
    """'The control did not run' and 'the control passed' must never resolve
    the same way."""
    failed = controls.judge(
        _control("null_control"), {"success": False, "error": "boom"}
    )
    silent = controls.judge(
        _control("null_control"), {"success": True, "data": {"stdout": "hello"}}
    )

    assert failed["passed"] is False
    assert silent["passed"] is False
    assert "nothing to check it against" in silent["reason"]


def test_the_failure_says_what_the_control_catches():
    """'A control failed' is not actionable; what it catches is."""
    reason = controls.judge(_control("scale_control"), _ok(3.47))["reason"]

    assert "3.47" in reason
    assert "cancels" in reason


# --- bracketing -----------------------------------------------------------


def _state(control_events, tool_uses):
    actions = [{"action": {"tool": t}} for t in tool_uses]
    return {"actions_taken": actions, "instrument_controls": control_events}


def _passing(name, at):
    return {
        "tool": "benchmark_c_snippet",
        "control": name,
        "passed": True,
        "at_action": at,
    }


def test_controls_on_both_sides_verify_the_instrument():
    state = _state(
        [
            _passing("null_control", 0),
            _passing("scale_control", 0),
            _passing("null_control", 9),
            _passing("scale_control", 9),
        ],
        ["benchmark_c_snippet"] * 5,
    )

    assert controls.bracket_status(state, "benchmark_c_snippet")["bracketed"] is True
    assert controls.unverified_instruments(state) == []


def test_a_control_that_only_precedes_is_not_enough():
    """A host can drift mid-run. A measurement run on this project was
    discarded because the control taken AFTER it read 2.2012, and its data
    showed fsqrt at 16.91 where two accepted runs read 10.01 and 10.43."""
    state = _state(
        [_passing("null_control", 0), _passing("scale_control", 0)],
        ["benchmark_c_snippet"] * 5,
    )

    status = controls.bracket_status(state, "benchmark_c_snippet")

    assert status["bracketed"] is False
    assert "after its last" in status["reason"]
    assert status["missing_after"] == ["null_control", "scale_control"]


def test_a_failed_control_disqualifies_the_window():
    state = _state(
        [
            _passing("null_control", 0),
            {
                "tool": "benchmark_c_snippet",
                "control": "scale_control",
                "passed": False,
                "at_action": 0,
                "reason": "scale_control read 3.4749",
            },
        ],
        ["benchmark_c_snippet"],
    )

    status = controls.bracket_status(state, "benchmark_c_snippet")

    assert status["bracketed"] is False
    assert "3.4749" in status["reason"]


def test_a_tool_never_used_needs_no_control():
    state = _state([], ["web_search", "record_finding"])

    assert controls.bracket_status(state, "benchmark_c_snippet")["bracketed"] is True


def test_an_uncontrolled_tool_is_not_held_to_this():
    assert controls.is_controlled("web_search") is False
    assert controls.bracket_status({}, "web_search")["controlled"] is False


# --- the contract predicate ----------------------------------------------


def test_a_contract_can_require_verified_instruments():
    state = _state(
        [_passing("null_control", 0), _passing("scale_control", 0)],
        ["benchmark_c_snippet"] * 3,
    )

    result = validity.evaluate({"validity": {"instruments_verified": True}}, state)

    assert "validity:instruments_verified" in result["missing"]
    assert (
        result["details"]["unverified_instruments"][0]["tool"] == "benchmark_c_snippet"
    )


def test_verified_instruments_satisfy_the_contract():
    state = _state(
        [
            _passing("null_control", 0),
            _passing("scale_control", 0),
            _passing("null_control", 4),
            _passing("scale_control", 4),
        ],
        ["benchmark_c_snippet"] * 3,
    )

    result = validity.evaluate({"validity": {"instruments_verified": True}}, state)

    assert result["missing"] == []


def test_the_remedy_names_the_tool_and_what_to_do():
    state = _state([_passing("null_control", 0)], ["benchmark_c_snippet"])
    result = validity.evaluate({"validity": {"instruments_verified": True}}, state)

    lines = validity.explain(result["missing"], result["details"])

    assert lines and "benchmark_c_snippet" in lines[0]
    assert "before your first measurement" in lines[0]


def test_the_requirement_is_described_for_the_prompt():
    described = validity.describe({"instruments_verified": True})

    assert any("null_control" in line and "scale_control" in line for line in described)


def test_the_control_programs_carry_the_statistic_that_works():
    """Pooling totals reads 0.53; the median of per-round ratios reads 1.0000.
    If that ever reverts, these controls stop being able to see anything."""
    for program in (controls.NULL_CONTROL_PROGRAM, controls.SCALE_CONTROL_PROGRAM):
        assert "qsort" in program, "the statistic must be a median, not a sum"
        assert 'asm("x9")' in program, "accumulators must be pinned, not clobbered"
        assert "ratios[31 / 2]" in program


# --- the runner and the two hooks ----------------------------------------


@pytest.mark.asyncio
async def test_running_controls_records_verdicts_on_the_state():
    calls = []

    async def _call(tool, params):
        calls.append(tool)
        return _ok(1.0 if "identical" in params.get("label", "") else 2.0)

    state = {"actions_taken": [{"action": {"tool": "benchmark_c_snippet"}}]}
    verdicts = await controls.run_controls(_call, "benchmark_c_snippet", state)

    assert [v["passed"] for v in verdicts] == [True, True]
    assert len(state["instrument_controls"]) == 2
    assert calls == ["benchmark_c_snippet", "benchmark_c_snippet"]


@pytest.mark.asyncio
async def test_a_control_that_raises_is_recorded_as_a_failure():
    """Swallowing it would leave no verdict, which reads as an absent control
    -- the same outcome reached less clearly."""

    async def _call(tool, params):
        raise RuntimeError("sandbox is down")

    state = {"actions_taken": []}
    verdicts = await controls.run_controls(_call, "benchmark_c_snippet", state)

    assert all(v["passed"] is False for v in verdicts)
    assert "sandbox is down" in verdicts[0]["reason"]


@pytest.mark.asyncio
async def test_controls_run_once_per_run_not_once_per_call():
    """The controls are themselves measurements; two around every call would
    cost more than the work being measured."""

    async def _call(tool, params):
        return _ok(1.0)

    state = {"actions_taken": []}
    assert controls.needs_pre_control(state, "benchmark_c_snippet") is True
    await controls.run_controls(_call, "benchmark_c_snippet", state)
    assert controls.needs_pre_control(state, "benchmark_c_snippet") is False


def test_a_post_control_is_needed_once_the_tool_is_used_again():
    state = _state([_passing("null_control", 0)], ["benchmark_c_snippet"] * 3)

    assert controls.needs_post_control(state, "benchmark_c_snippet") is True

    state["instrument_controls"].append(_passing("null_control", 5))
    assert controls.needs_post_control(state, "benchmark_c_snippet") is False


def test_no_post_control_is_needed_if_the_tool_was_never_used():
    assert (
        controls.needs_post_control({"actions_taken": []}, "benchmark_c_snippet")
        is False
    )
