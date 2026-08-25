"""What a buildable predictor actually gets, against the ceiling it was sold on.

An information ceiling says what is available. These tests are about the other
half: whether a few hundred transistors reach it, measured the only way that
means anything -- on intervals the tables were not warmed on.
"""

from __future__ import annotations

import random

from app.services import agent_predictor_design as design


def _long(pattern, times=20):
    return list(pattern) * times


def _tap_driven(n=400, seed=1):
    """A tap at t that names the target at t+1 outright, persistence weak."""
    rng = random.Random(seed)
    tap = [float(rng.randint(0, 2)) for _ in range(n)]
    return {"t": [0.0] + tap[:-1], "tap": tap}


# --- the mechanism ---------------------------------------------------------


def test_a_saturating_counter_needs_convincing_before_it_moves():
    """The whole reason hysteresis exists: one surprise must not flip the
    prediction, or the counter chases noise instead of tracking the workload."""
    table = design.SaturatingTable(levels=3, hysteresis=2)
    for _ in range(10):
        table.update(("x",), 0)
    assert table.predict(("x",)) == 0

    table.update(("x",), 2)
    assert table.predict(("x",)) == 0, "one surprise may not move the prediction"

    table.update(("x",), 2)
    assert table.predict(("x",)) == 1, "a second must"


def test_state_cost_is_reported_in_bits_not_entries():
    """Three levels at hysteresis 2 is six states, which is three bits. A
    design compared on entry count rather than bits is not being costed."""
    assert design.SaturatingTable(3, 1).bits_per_entry() == 2
    assert design.SaturatingTable(3, 2).bits_per_entry() == 3


def test_a_shared_ordinal_counter_predicts_a_level_that_never_occurs():
    """Why the set needs both mechanisms. A single counter per cell is ordinal:
    to get from level 0 to level 2 it travels through level 1. On a workload
    that only ever visits the extremes it parks in between, predicting a level
    the workload never produces -- which looks from the outside exactly like
    the tap carrying no information. Per-level counters cannot do this: their
    leader is always a level that was actually observed.

    On the SMT trace this is not hypothetical. The shared counter converged to
    predicting the same as last interval and read as a dead tap; per-level
    counters on the same feature pair reached the ceiling exactly.
    """
    shared = design.SaturatingTable(levels=3, hysteresis=2)
    leveled = design.LevelCounters(levels=3)

    for i in range(40):
        outcome = 0 if i % 2 else 2
        shared.update(("cell",), outcome)
        leveled.update(("cell",), outcome)

    assert shared.predict(("cell",)) == 1, "parked between the two real levels"
    assert leveled.predict(("cell",)) in (0, 2), "only ever an observed level"


def test_per_level_counters_cost_more_and_the_cost_is_reported():
    """Three levels at two bits each is six bits per cell, against three for a
    shared counter. A design that reaches the ceiling by spending double is a
    different answer from one that reaches it for free."""
    assert design.LevelCounters(3).bits_per_entry() == 6
    assert design.SaturatingTable(3, 2).bits_per_entry() == 3


# --- the traps -------------------------------------------------------------


def test_a_determining_tap_is_found_and_reaches_its_ceiling():
    """The control. If this does not read 100% of headroom, nothing below
    means anything."""
    result = design.evaluate(_tap_driven(), "t", "tap")

    assert result["measured"] is True
    assert result["ceiling_accuracy"] == 1.0
    assert result["best_share_of_headroom"] == 1.0
    assert result["survives_null"] is True


def test_a_tap_carrying_nothing_does_not_survive_the_null():
    """A table indexed by a counter with no relationship to the target still
    fits the warm-up, and on a short scored segment that fit is worth
    something. The null is what stops it being reported as a design."""
    rng = random.Random(2)
    series = {
        "t": [float(rng.randint(0, 2)) for _ in range(400)],
        "tap": [float(rng.randint(0, 2)) for _ in range(400)],
    }

    result = design.evaluate(series, "t", "tap", trials=20)

    assert result["survives_null"] is False
    # Not asserting which sentence comes back. Against the stronger of the two
    # baselines this case now has no headroom at all, so the honest verdict is
    # "nothing to build" rather than "inside the null" -- a better answer, and
    # a test pinned to the wording would have called the improvement a
    # regression.
    assert result["best_gain_over_persistence"] <= result["null_p95_gain"]


def test_the_reference_is_fit_on_the_warmup_not_the_scored_segment():
    """Fit the majority-per-cell reference on the scored rows and it is fit to
    the answers, which turns the ceiling into a number nothing falls short of.
    Rows in cells the warm-up never saw are counted, not absorbed."""
    indices = [(0,)] * 60 + [(1,)] * 60
    outcomes = [0] * 60 + [1] * 60

    accuracy, unseen = design._oracle(indices, outcomes, split=60)

    assert unseen == 60, "every scored row is in a cell the warm-up never saw"
    assert accuracy == 0.0, "an unseen cell must not be scored as correct"


def test_a_warmup_too_small_to_fill_the_table_is_refused():
    """An untrained cell predicts its reset state, so the design would be
    scored on a table that was never built."""
    result = design.evaluate(_tap_driven(n=80), "t", "tap", split=0.4)

    assert result["measured"] is False
    assert "never built" in result["refusal"]


def test_a_scored_segment_too_short_to_separate_designs_is_refused():
    result = design.evaluate(_tap_driven(n=120), "t", "tap", split=0.85)

    assert result["measured"] is False
    assert "the same design" in result["refusal"]


# --- what the modes say ----------------------------------------------------


def test_holding_the_table_is_scored_alongside_letting_it_update():
    """Scoring only the updating mode reports a design that works as a design
    that does not: a counter that moves on every surprise chases noise, and on
    a trace where holding is worth more than the tap, that is the finding."""
    result = design.evaluate(_tap_driven(), "t", "tap")

    for row in result["designs"]:
        assert "gain_frozen" in row and "gain_online" in row
        assert row["best_mode"] in ("frozen", "online")
        assert row["gain_over_persistence"] == max(
            row["gain_frozen"], row["gain_online"]
        )


def test_beating_the_ceiling_is_reported_as_a_short_trace_not_a_result():
    """A design cannot beat its own ceiling. When it does, the ceiling was fit
    on the warm-up and the scored segment left room for luck -- which is a
    statement about the trace, not about the design."""
    best = {
        "design": "last value + tap",
        "state_bits": 27,
        "best_mode": "frozen",
        "gain_over_persistence": 0.055,
        "gain_frozen": 0.055,
        "gain_online": 0.055,
        "share_of_headroom": 1.1,
    }

    verdict = design._verdict(0.7, 0.75, best, null_p95=0.0, unseen=1, scored=200)

    assert "not a bound" in verdict


def test_the_tool_is_registered_everywhere():
    """A tool missing from any one of these is invisible to agents."""
    from app.agent_core import tool_catalog
    from app.services import agent_job_tool_policy, agent_tools

    source = "".join(
        open(f).read()
        for f in (
            tool_catalog.__file__,
            agent_job_tool_policy.__file__,
            agent_tools.__file__,
        )
    )
    assert source.count("evaluate_predictor_design") >= 3

    names = {t["name"] for t in agent_tools.AGENT_TOOLS}
    assert "evaluate_predictor_design" in names


# --- what "persistence" means ----------------------------------------------


def test_the_baseline_is_the_stronger_of_repeating_and_a_last_value_table():
    """These are not the same thing and this module conflated them. Repeating
    the last value is free. A table indexed by it is what the information
    measure means by persistence -- H(next) - H(next|last) is the best any use
    of the last value could do, and repeating is only one such use."""
    # Bins alternate, so repeating is never right and the table is nearly
    # always right. A real Godot trace looked exactly like this.
    alternating = [0.0 if i % 2 else 100.0 for i in range(400)]
    series = {"t": alternating, "tap": [float(i % 5) for i in range(400)]}

    result = design.evaluate(series, "t", "tap", trials=10)

    assert result["repeat_last_value_accuracy"] == 0.0
    assert result["last_value_table_accuracy"] > 0.9
    assert result["baseline"] == "last-value table"
    assert result["persistence_accuracy"] == result["last_value_table_accuracy"]


def test_a_table_that_learned_the_cycle_is_not_called_a_slower_repeat():
    """The defect this replaced: scored against repeating alone, a table that
    had learned an alternation read as gaining +99.5% over persistence and was
    then dismissed as 'a slower way of predicting the same as last interval'.
    Both halves of that sentence were wrong."""
    alternating = [0.0 if i % 2 else 100.0 for i in range(400)]
    series = {"t": alternating, "tap": [float(i % 5) for i in range(400)]}

    result = design.evaluate(series, "t", "tap", trials=10)

    assert result["headroom"] < 0.05, "the last value already explains it"
    assert result["best_gain_over_persistence"] < 0.05
    assert result["survives_null"] is False


def test_where_repeating_wins_it_stays_the_baseline():
    """On an autocorrelated target a freshly trained table is worse than the
    free thing, and the tap must still beat the free thing."""
    runs = _long([10.0] * 8 + [90.0] * 8, times=25)
    series = {"t": runs, "tap": [float(i % 5) for i in range(len(runs))]}

    result = design.evaluate(series, "t", "tap", trials=10)

    assert result["repeat_last_value_accuracy"] > result["last_value_table_accuracy"]
    assert result["baseline"] == "repeat last value"
