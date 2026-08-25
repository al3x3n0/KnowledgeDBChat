"""How much signal is in the counters, before anyone designs a predictor.

The number that matters is not how well a counter predicts the next interval.
It is how much it adds over predicting the same as last interval -- because
programs run in phases, almost everything is autocorrelated, and almost every
counter looks predictive until you ask what it contributes.
"""

from __future__ import annotations

from app.services import agent_predictability as pred


def _long(pattern, times=20):
    return list(pattern) * times


def test_entropy_of_a_constant_is_zero():
    assert pred.entropy([1, 1, 1, 1]) == 0.0


def test_entropy_of_a_fair_coin_is_one_bit():
    assert abs(pred.entropy([0, 1, 0, 1]) - 1.0) < 1e-9


def test_quantile_bins_not_equal_width():
    """Equal-width bins put almost every interval in one bucket for a counter
    with a long tail, which most hardware counters have."""
    values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1000]

    labels = pred.discretize(values, bins=2)

    assert len(set(labels)) == 2, "a long tail must not collapse to one bin"


def test_a_counter_that_only_repeats_the_target_adds_nothing():
    """The whole point. A counter perfectly correlated with the target's own
    last value is telling you the program has phases, which you knew.

    The runs have to be long for persistence to carry anything: with a period
    of four, knowing the previous value tells you nothing about the next --
    after a low you are equally likely to see low or high. Phases in real
    programs last many intervals, which is exactly why persistence is such a
    strong baseline there.
    """
    target = _long([10.0] * 8 + [90.0] * 8, times=6)
    echo = list(target)

    signal = pred.counter_signal(echo, target)

    assert signal["persistence_information"] > 0.3
    assert signal["information_beyond_persistence"] < 0.05


def test_a_counter_that_leads_the_target_adds_a_lot():
    """A counter whose value at t determines the target at t+1, in a way last
    value cannot supply."""
    lead = _long([0.0, 1.0, 0.0, 1.0])
    target = _long([5.0, 5.0, 99.0, 5.0])

    signal = pred.counter_signal(lead, target)

    assert signal["information_beyond_persistence"] > 0.1


def test_a_target_that_never_changes_has_nothing_to_predict():
    signal = pred.counter_signal(_long([1.0, 2.0]), _long([7.0, 7.0]))

    assert signal["target_entropy"] == 0.0
    assert "nothing to predict" in signal["note"]


def test_a_short_trace_is_refused_with_the_count_it_needs():
    """Below the sample threshold the estimate is mostly empty bins, which look
    like structure and are not. 'Could not measure' and 'measured, and there is
    nothing' are opposite findings."""
    series = {"target": [1.0, 2.0, 3.0], "other": [4.0, 5.0, 6.0]}

    result = pred.ceiling(series, "target")

    assert result["measured"] is False
    assert result["intervals_needed"] == pred.MIN_PER_CELL * 9
    assert "cannot support an estimate" in result["refusal"]


def test_an_absent_target_is_refused_with_what_is_available():
    result = pred.ceiling({"a": _long([1.0, 2.0])}, "system.cpu.numCycles")

    assert result["measured"] is False
    assert "no counter named" in result["refusal"]
    assert "Available" in result["refusal"]


def test_a_persistence_dominated_workload_says_do_not_build_one():
    """The honest negative result this study exists to be able to reach."""
    target = _long([10.0] * 8 + [90.0] * 8, times=6)
    series = {"target": target, "echo": list(target)}

    result = pred.ceiling(series, "target")

    assert result["measured"] is True
    assert "no single counter recovers" in result["verdict"] or (
        "Persistence already explains" in result["verdict"]
    )


def test_the_ceiling_is_stated_as_an_upper_bound():
    lead = _long([0.0, 1.0, 0.0, 1.0], times=20)
    target = _long([5.0, 5.0, 99.0, 5.0], times=20)
    series = {"target": target, "lead": lead}

    result = pred.ceiling(series, "target")

    assert result["measured"] is True
    assert "upper bound" in result["verdict"]
    assert result["counters"][0]["counter"] == "lead"


def test_counters_are_ranked_by_what_they_add_not_what_they_know():
    target = _long([10.0] * 8 + [90.0] * 8, times=6)
    # Fires on the interval before every switch, which last value cannot know.
    lead = _long([0.0] * 7 + [1.0] + [0.0] * 7 + [1.0], times=6)
    series = {"target": target, "echo": list(target), "lead": lead}

    result = pred.ceiling(series, "target")
    order = [c["counter"] for c in result["counters"]]

    assert order[0] == "lead", "an echo of the target adds nothing over persistence"


def test_every_counter_tool_is_registered_everywhere():
    """A tool missing from any one of these is invisible to agents, and three
    validity predicates were dead on arrival earlier today for exactly that."""
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
    for tool in (
        "sample_hardware_counters",
        "measure_predictability",
        "select_counter_taps",
    ):
        assert source.count(tool) >= 3, f"{tool} is not registered everywhere"

    names = (
        {t["name"] for t in agent_tools.AGENT_TOOLS}
        if hasattr(agent_tools, "AGENT_TOOLS")
        else set()
    )
    if names:
        assert "sample_hardware_counters" in names
        assert "measure_predictability" in names
        assert "select_counter_taps" in names


# --- the null --------------------------------------------------------------


def test_pure_noise_does_not_survive_the_null():
    """The check that made this module honest. At 65 intervals with 50
    counters, the best 'information beyond persistence' on data with no
    relationship at all reads about 0.31 bits -- which is more than a real
    kernel produced. Without the null, that was reported as a finding."""
    import random

    rng = random.Random(7)
    target = [rng.random() for _ in range(65)]
    series = {"t": target}
    for i in range(50):
        series[f"n{i}"] = [rng.random() for _ in range(65)]

    result = pred.ceiling(series, "t")

    assert result["measured"] is True
    assert result["survives_null"] is False
    assert result["best_counter_beyond_persistence_bits"] > 0.2, (
        "the estimator does report apparent signal on noise -- that is the bias "
        "this null exists to expose"
    )
    assert "INSIDE the null" in result["verdict"]


def test_the_null_is_over_the_maximum_across_counters():
    """Comparing fifty counters against a 95th percentile means two or three
    clear it by chance. The max-statistic null asks how large the best of
    fifty is when none is related."""
    import random

    rng = random.Random(3)
    target = [rng.random() for _ in range(60)]
    series = {
        "t": target,
        **{f"n{i}": [rng.random() for _ in range(60)] for i in range(20)},
    }

    result = pred.ceiling(series, "t")

    assert result["null"]["statistic"] == "maximum across counters"
    assert result["null"]["counters_tested"] == 20


def test_a_planted_relationship_does_survive_the_null():
    """The null must not reject everything, or it is not a discriminator."""
    target = _long([10.0] * 8 + [90.0] * 8, times=6)
    lead = _long([0.0] * 7 + [1.0] + [0.0] * 7 + [1.0], times=6)

    result = pred.ceiling({"target": target, "lead": lead}, "target")

    assert result["survives_null"] is True


def test_the_null_is_deterministic():
    """A null that moves between runs cannot be argued with."""
    import random

    rng = random.Random(11)
    target = [rng.random() for _ in range(60)]
    series = {"t": target, "a": [rng.random() for _ in range(60)]}

    first = pred.ceiling(series, "t")["null"]["null_p95"]
    second = pred.ceiling(series, "t")["null"]["null_p95"]

    assert first == second


# --- which taps, together --------------------------------------------------


def test_the_depth_limit_counts_the_interval_the_shift_spends():
    """Three taps at three bins need 5 * 3**4 = 405 usable pairs. A trace of
    405 intervals yields 404 of them, and rounding that in the tool's favour is
    how a depth limit stops limiting anything exactly at the boundary."""
    assert pred.max_taps_for(405) == 2
    assert pred.max_taps_for(406) == 3


def test_a_trace_too_short_for_even_one_tap_is_refused():
    """Same rule the single-counter estimate follows: 'could not measure' and
    'measured, and there is nothing' are opposite findings."""
    short = {
        "t": [float(i % 5) for i in range(40)],
        "c": [float(i % 3) for i in range(40)],
    }

    result = pred.select_taps(short, "t")

    assert result["measured"] is False
    assert "46" in result["refusal"], "the refusal must name the length it needs"


def test_two_real_taps_both_survive_their_own_null():
    """The target's next value is the sum of two independent drivers, so each
    adds over persistence and neither is redundant with the other."""
    import random

    rng = random.Random(11)
    a = [float(rng.randint(0, 1)) for _ in range(400)]
    b = [float(rng.randint(0, 1)) for _ in range(400)]
    target = [0.0] + [a[t] + b[t] for t in range(399)]
    series = {"t": target, "a": a, "b": b}
    series.update({f"noise{i}": [rng.random() for _ in range(400)] for i in range(8)})

    result = pred.select_taps(series, "t", trials=20)

    assert result["recommended_taps"] == 2
    assert set(result["taps"]) == {"a", "b"}
    assert all(step["survives_null"] for step in result["selection"])


def test_selection_bias_alone_does_not_recommend_a_tap():
    """Fifty counters and no relationship at all. Greedy selection reports a
    number here whatever the data says, which is the whole reason the null runs
    the selection rather than scoring a counter someone else picked."""
    import random

    rng = random.Random(3)
    series = {"t": [rng.random() for _ in range(200)]}
    series.update({f"c{i}": [rng.random() for _ in range(200)] for i in range(50)})

    result = pred.select_taps(series, "t", trials=20)

    assert result["recommended_taps"] == 0
    assert result["survives_null"] is False
    assert "worth a wire" in result["verdict"]


def test_a_tap_is_judged_on_what_it_added_not_on_the_running_total():
    """The defect this replaced. A second tap that is pure selection bias still
    lifts the cumulative total above a null taken at full depth, so scoring the
    total buys wires for noise. Each tap is bought with its own increment."""
    import random

    rng = random.Random(5)
    driver = [float(rng.randint(0, 2)) for _ in range(400)]
    # Corrupted, so entropy is left over for a spurious second tap to shave.
    # A driver that determined the target exactly would leave nothing to buy,
    # and the greedy would stop on its own -- which is not the trap here.
    target = [0.0] + [
        d if rng.random() > 0.3 else float(rng.randint(0, 2)) for d in driver[:-1]
    ]
    series = {"t": target, "driver": driver}
    series.update({f"c{i}": [rng.random() for _ in range(400)] for i in range(40)})

    result = pred.select_taps(series, "t", trials=20)

    assert result["selection"][0]["tap"] == "driver"
    assert result["recommended_taps"] == 1, "only the real driver may be bought"
    second = result["selection"][1]
    assert second["survives_null"] is False
    assert second["total_beyond_persistence"] > result["total_beyond_persistence"]
