"""Measuring a design's fitness before paying for the measurement.

The critic reasons about an artifact and can be wrong. This does arithmetic on
a cheap measurement of it and therefore refuses rather than advises. Every
check is a property of the trace the design will produce, not a rule about how
the workload was written -- which is why the traps it catches did not have to
be foreseen.
"""

from __future__ import annotations

from app.services import agent_experiment_preflight as pre


def _steady(n=400, work=20000.0):
    return [work + (i % 7) * 3 for i in range(n)]


def test_a_sound_design_passes():
    v = pre.judge(_steady(), timeout_seconds=1800)

    assert v["fit"] is True
    assert v["blocking"] == []


def test_a_trace_too_short_to_estimate_on_is_blocking():
    v = pre.judge(_steady(n=20))

    assert v["fit"] is False
    assert v["blocking"][0]["check"] == "intervals"
    assert "46" in v["blocking"][0]["summary"]


def test_a_run_that_cannot_finish_is_blocking_before_it_starts():
    """The projection is the point: half an hour of simulation should not be
    the thing that discovers a workload is too big."""
    v = pre.judge(_steady(n=400, work=2_000_000.0), timeout_seconds=1800)

    blocking = [c for c in v["blocking"] if c["check"] == "cost"]
    assert blocking, "an 8000-second run under a 1800-second timeout must block"
    assert v["projected_o3_seconds"] > 1800


def test_blocked_phases_are_caught_when_alternation_was_asked_for():
    """The live case. An agent asked for phases that alternate wrote a hundred
    of one then a hundred of the other, and the harness only found out after
    simulating it."""
    blocked = [20000.0] * 100 + [900000.0] * 100

    v = pre.judge(blocked, intends_alternating_phases=True)

    regime = [c for c in v["concerns"] if c["check"] == "regime"]
    assert regime and regime[0]["severity"] == "blocking"
    assert 90 < regime[0]["at_interval"] < 110
    assert "Interleave" in regime[0]["remedy"]


def test_a_regime_change_is_only_a_warning_when_alternation_was_not_asked_for():
    """A run that legitimately has a startup phase should be told to measure
    one side, not told its design is wrong."""
    blocked = [20000.0] * 100 + [900000.0] * 200

    v = pre.judge(blocked, intends_alternating_phases=False)

    regime = [c for c in v["concerns"] if c["check"] == "regime"]
    assert regime and regime[0]["severity"] == "serious"
    assert "from_interval" in regime[0]["remedy"]


def test_alternating_phases_are_not_mistaken_for_two_regimes():
    """A workload doing what it was asked to do must not be condemned for it:
    both halves of any split contain both phases, so their medians agree."""
    alternating = [20000.0 if i % 2 else 900000.0 for i in range(300)]

    v = pre.judge(alternating, intends_alternating_phases=True)

    assert [c for c in v["concerns"] if c["check"] == "regime"] == []


def test_constant_work_is_flagged_but_not_blocking():
    """It is exactly right for a primary under contention and exactly wrong for
    a solo predictability study, so this reports rather than refuses."""
    v = pre.judge([20000.0] * 400)

    flat = [c for c in v["concerns"] if c["check"] == "variation"]
    assert flat and flat[0]["severity"] == "serious"
    assert v["fit"] is True


def test_the_refusal_names_every_blocking_defect_and_its_remedy():
    v = pre.judge([20000.0] * 20)

    text = pre.refusal(v)

    assert "46" in text
    assert "M5_SAMPLE" in text
    assert "spends that time to learn this" in text


def test_the_minimum_counts_the_interval_the_shift_spends():
    """Same accounting as the estimator it is protecting: predicting t+1 from t
    costs one interval, and a preflight that lets a trace through one short is
    not protecting anything."""
    assert pre.minimum_intervals(3) == 46
