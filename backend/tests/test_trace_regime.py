"""A number is only as sound as the trace it was taken on.

Every other validity predicate judges an output: was the instrument verified,
did the measurement reproduce, is the value physically possible. This one
judges the input, and it exists because a measurement can pass all of those and
still have been taken across two experiments spliced together.
"""

from __future__ import annotations

from app.services import agent_measurement_validity as validity
from app.services import agent_trace_regime as regime


def _sample(intervals=400, break_at=104):
    return {
        "series": {
            "a": [float(i) for i in range(intervals)],
            "b": [float(i % 7) for i in range(intervals)],
        },
        "regime_change": (
            {"at_interval": break_at, "ratio": 3.9} if break_at is not None else None
        ),
    }


def _finding(**overrides):
    finding = {
        "type": "predictability_ceiling",
        "title": "thread0 ipc: 0.106 bits beyond persistence",
        "analysed_from_interval": 0,
        "trace_intervals": 400,
        "trace_regime_change_at": 104,
        "analysed_across_regime_change": True,
    }
    finding.update(overrides)
    return finding


# --- the window ------------------------------------------------------------


def test_a_window_over_the_whole_trace_straddles_a_break_in_it():
    series, stamp = regime.window(_sample(), from_interval=0)

    assert len(series["a"]) == 400
    assert stamp["analysed_across_regime_change"] is True
    assert stamp["trace_regime_change_at"] == 104


def test_starting_at_the_break_does_not_straddle_it():
    """The remedy has to exist before the requirement does. A run told about a
    break at 104 measures from 104 and satisfies this."""
    series, stamp = regime.window(_sample(), from_interval=104)

    assert len(series["a"]) == 296
    assert series["a"][0] == 104.0
    assert stamp["analysed_from_interval"] == 104
    assert stamp["analysed_across_regime_change"] is False


def test_a_trace_with_no_break_never_straddles_one():
    _, stamp = regime.window(_sample(break_at=None))

    assert stamp["trace_regime_change_at"] is None
    assert stamp["analysed_across_regime_change"] is False


def test_a_window_past_the_end_is_clamped_rather_than_inverted():
    """A caller that asks to start beyond the trace gets an empty window, not a
    negative slice that silently wraps to the whole thing."""
    series, stamp = regime.window(_sample(intervals=50, break_at=None), 900)

    assert series["a"] == []
    assert stamp["analysed_from_interval"] == 50


def test_a_non_numeric_from_interval_is_treated_as_no_window():
    _, stamp = regime.window(_sample(), from_interval="the beginning")

    assert stamp["analysed_from_interval"] == 0


# --- the check -------------------------------------------------------------


def test_a_finding_measured_across_a_break_is_caught():
    offenders = regime.findings_across_regime_change({"findings": [_finding()]})

    assert len(offenders) == 1
    assert offenders[0]["regime_change_at"] == 104


def test_a_finding_measured_on_one_side_is_not():
    clean = _finding(analysed_from_interval=104, analysed_across_regime_change=False)

    assert regime.findings_across_regime_change({"findings": [clean]}) == []


def test_a_finding_that_never_touched_a_trace_is_not_judged():
    """A type absent from the trace-derived list cannot straddle anything, and
    flagging it would blame the wrong finding for the wrong reason."""
    unrelated = _finding(type="fusion_candidate")

    assert regime.findings_across_regime_change({"findings": [unrelated]}) == []


# --- the contract ----------------------------------------------------------


def test_a_contract_requiring_one_regime_holds_the_run_back():
    result = validity.evaluate(
        {"validity": {"traces_one_regime": True}}, {"findings": [_finding()]}
    )

    assert "validity:traces_one_regime" in result["missing"]
    assert result["details"]["across_regime_change"][0]["regime_change_at"] == 104


def test_a_contract_not_requiring_it_says_nothing():
    result = validity.evaluate(
        {"validity": {"records_method": True}}, {"findings": [_finding()]}
    )

    assert "validity:traces_one_regime" not in result["missing"]


def test_the_explanation_names_the_interval_to_re_run_from():
    """`validity:traces_one_regime` tells a model nothing it can act on. The
    interval the warning named, and the parameter that acts on it, do."""
    result = validity.evaluate(
        {"validity": {"traces_one_regime": True}}, {"findings": [_finding()]}
    )

    lines = validity.explain(result["missing"], result["details"])

    assert lines and "from_interval=104" in lines[0]


def test_the_requirement_is_described_for_the_prompt():
    """Validity requirements go into the stable thinking prompt because they
    change how the work is done, not only when it may stop."""
    described = " ".join(validity.describe({"traces_one_regime": True}))

    assert "from_interval" in described


def test_the_predicate_is_reachable_from_the_contract_vocabulary():
    """A predicate the docstring does not list is one nobody writes a contract
    against; three were dead on arrival for exactly that."""
    assert "traces_one_regime" in validity.__doc__


# --- why it matters --------------------------------------------------------


def test_a_spliced_trace_inflates_persistence_which_is_the_whole_point():
    """The claim this predicate rests on, checked rather than asserted.

    Two regimes spliced together are trivially predictable at the join: almost
    every interval is in the same regime as the one before it, so persistence
    absorbs the splice and reads as structure the workload does not have. Each
    regime alone is memoryless here, so the honest answer is near zero.
    """
    import random

    from app.services import agent_predictability as pred

    rng = random.Random(17)
    # Memoryless within each regime, so any persistence found is the splice.
    quiet = [rng.uniform(0, 10) for _ in range(120)]
    busy = [rng.uniform(100, 110) for _ in range(120)]
    noise = [rng.uniform(0, 1) for _ in range(240)]

    # Two bins, so the split lands on the regime boundary and the estimate is
    # purely "does the trace stay in the regime it was already in". At three
    # the within-regime spread muddies it to 0.44 -- still most of the effect,
    # but the point is clearest where the binning is not also being measured.
    across = pred.ceiling({"t": quiet + busy, "c": noise}, "t", bins=2)
    after = pred.ceiling({"t": busy, "c": noise[:120]}, "t", bins=2)

    assert across["persistence_information_bits"] > 0.8, "the splice reads as structure"
    assert after["persistence_information_bits"] < 0.1, "neither regime alone has any"
