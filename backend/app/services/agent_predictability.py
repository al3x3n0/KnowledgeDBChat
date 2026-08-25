"""How much signal is in the counters, before anyone designs a predictor.

A hardware predictor is worth silicon only if the counters it taps actually
carry information about the thing it predicts. That is a property of the
workload and the counter set, not of the predictor, and it can be measured
first -- which means a study can end cheaply and honestly instead of after
someone has designed three predictors that were never going to work.

This is the instruction-mix census applied to prediction: establish the
ceiling before spending anything on the design.

**The number that matters is not the raw one.** A counter can be highly
informative about the next interval and still be worthless, because the
*previous value of the target* already told you the same thing. Programs run in
phases: almost everything is autocorrelated, so almost every counter looks
predictive until you ask what it adds. A perceptron that does not beat
predict-same-as-last-interval is not worth a transistor, and published
ML-in-hardware results have died exactly there.

So the headline here is **information beyond persistence**: how much a counter
tells you about the next interval that the target's own last value did not.
Everything else is reported alongside so the gap is visible.

**Refusing to answer is a supported outcome.** Estimating a conditional entropy
over two discretised variables needs enough samples per cell; below that the
estimate is dominated by empty bins and reliably reports signal that is not
there. A short trace gets a refusal naming how many intervals it would need,
never a number -- the same rule the instrument controls follow, because "could
not measure" and "measured, and there is nothing" are opposite findings.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: Discretisation bins. Three -- low, middle, high -- on purpose: the sample
#: cost of estimating a conditional entropy grows with the square of this, and
#: a trace is tens of intervals, not millions.
DEFAULT_BINS = 3

#: Samples required per cell of the joint distribution. Below this the estimate
#: is mostly empty bins, which look like structure.
MIN_PER_CELL = 5


def _min_intervals(bins: int) -> int:
    """Intervals needed to condition on two discretised variables at once."""
    return MIN_PER_CELL * bins * bins


def discretize(values: Sequence[float], bins: int = DEFAULT_BINS) -> List[int]:
    """Quantile bins, so each level holds a similar number of samples.

    Equal-width bins would put almost every interval in one bucket for a
    counter with a long tail -- which most hardware counters have -- and an
    entropy computed over one occupied bin is zero for a reason that has
    nothing to do with the workload.
    """
    numbers = [float(v) for v in values]
    if not numbers:
        return []
    ordered = sorted(numbers)
    n = len(ordered)
    edges = [ordered[min(n - 1, int(round(i * n / bins)))] for i in range(1, bins)]

    labels = []
    for value in numbers:
        label = 0
        for edge in edges:
            if value > edge:
                label += 1
        labels.append(min(label, bins - 1))
    return labels


def entropy(labels: Sequence[int]) -> float:
    """Shannon entropy in bits."""
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = len(labels)
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c)


def conditional_entropy(target: Sequence[int], given: Sequence[Sequence[int]]) -> float:
    """H(target | given), where `given` is one or more label sequences."""
    if not target:
        return 0.0
    keys = list(zip(*given)) if given else [()] * len(target)
    groups: Dict[Tuple, List[int]] = {}
    for key, value in zip(keys, target):
        groups.setdefault(key, []).append(value)

    total = len(target)
    return sum((len(g) / total) * entropy(g) for g in groups.values())


def _shift(labels: Sequence[int]) -> Tuple[List[int], List[int]]:
    """(features at t, target at t+1) -- the only alignment that predicts."""
    return list(labels[:-1]), list(labels[1:])


def counter_signal(
    counter: Sequence[float],
    target: Sequence[float],
    bins: int = DEFAULT_BINS,
) -> Dict[str, Any]:
    """What one counter at interval t says about the target at t+1.

    Three numbers, and the third is the one that decides anything:

    * `information` -- what the counter alone tells you.
    * `persistence_information` -- what the target's own last value tells you.
      This is the baseline any hardware predictor must beat.
    * `information_beyond_persistence` -- what the counter adds on top of that.
      A counter scoring high on the first and zero here is telling you the
      program has phases, which you already knew.
    """
    counter_labels = discretize(counter, bins)
    target_labels = discretize(target, bins)
    feature, ahead = _shift(counter_labels)
    last_target, _ = _shift(target_labels)
    _, target_ahead = _shift(target_labels)

    base = entropy(target_ahead)
    if base <= 0:
        return {
            "information": 0.0,
            "persistence_information": 0.0,
            "information_beyond_persistence": 0.0,
            "target_entropy": 0.0,
            "note": (
                "the target never changes across this trace, so there is "
                "nothing to predict"
            ),
        }

    h_given_counter = conditional_entropy(target_ahead, [feature])
    h_given_last = conditional_entropy(target_ahead, [last_target])
    h_given_both = conditional_entropy(target_ahead, [last_target, feature])

    return {
        "information": round(base - h_given_counter, 4),
        "persistence_information": round(base - h_given_last, 4),
        "information_beyond_persistence": round(h_given_last - h_given_both, 4),
        "target_entropy": round(base, 4),
        "note": "",
    }


#: Permutations for the null. Enough to place an observed value against a 95th
#: percentile without making the check cost more than the study.
NULL_TRIALS = 100


def shuffle_null(
    counters: Mapping[str, Sequence[float]],
    target: Sequence[float],
    bins: int = DEFAULT_BINS,
    trials: int = NULL_TRIALS,
    seed: int = 12345,
) -> Dict[str, Any]:
    """What this statistic reports on data where the relationship is destroyed.

    Conditional mutual information is *positively biased* with small samples:
    conditioning on two discretised variables splits a short trace across
    bins**2 cells, and sparse cells manufacture apparent structure. The
    signature is a counter that scores near zero alone and high in
    combination, which is exactly what the first real trace produced.

    So the observed value is placed against a null built by permuting the
    counter -- same marginal distribution, same trace length, same bin counts,
    no relationship to the target. Whatever the estimator reports on that is
    bias, and an observation inside the null is not a finding.

    The null is over the MAXIMUM across all counters, not one counter at a
    time. A trace carries tens of counters, and comparing each against a 95th
    percentile means one in twenty clears it by chance: with fifty counters,
    two or three "findings" are guaranteed on data with no structure at all.
    The max-statistic null asks the right question -- how large is the best of
    fifty counters when none of them is related -- and answers it at the same
    trace length, bin count and marginals as the real thing.

    Deterministic: a null that moves between runs cannot be argued with.
    """
    import random

    series = {k: list(v) for k, v in counters.items()}
    rng = random.Random(seed)
    null = []
    for _ in range(max(1, trials)):
        best = 0.0
        for values in series.values():
            shuffled = values[:]
            rng.shuffle(shuffled)
            best = max(
                best,
                counter_signal(shuffled, target, bins)[
                    "information_beyond_persistence"
                ],
            )
        null.append(best)
    null.sort()
    index = min(len(null) - 1, int(0.95 * len(null)))
    return {
        "trials": len(null),
        "counters_tested": len(series),
        "null_median": round(null[len(null) // 2], 4),
        "null_p95": round(null[index], 4),
        "statistic": "maximum across counters",
    }


def ceiling(
    series: Mapping[str, Sequence[float]],
    target: str,
    bins: int = DEFAULT_BINS,
    top: int = 12,
) -> Dict[str, Any]:
    """The best any predictor could do from these counters, and from which.

    Returns a refusal rather than a number when the trace is too short to
    estimate on. That is the finding in that case: nothing was measured.
    """
    if target not in series:
        return {
            "measured": False,
            "refusal": (
                f"no counter named {target!r} in this trace. Available: "
                + ", ".join(sorted(series)[:12])
            ),
        }

    target_values = list(series[target])
    intervals = len(target_values)
    needed = _min_intervals(bins)
    if intervals < needed:
        return {
            "measured": False,
            "intervals": intervals,
            "intervals_needed": needed,
            "refusal": (
                f"{intervals} intervals cannot support an estimate over "
                f"{bins} bins: conditioning on two discretised variables needs "
                f"about {needed} to keep {MIN_PER_CELL} samples per cell. Below "
                "that the estimate is mostly empty bins, which look like "
                "structure and are not. Take a longer trace -- more "
                "M5_SAMPLE() calls -- or ask for fewer bins."
            ),
        }

    scored = []
    for name, values in series.items():
        if name == target or len(values) != intervals:
            continue
        signal = counter_signal(values, target_values, bins)
        scored.append({"counter": name, **signal})

    scored.sort(key=lambda row: row["information_beyond_persistence"], reverse=True)

    # Place the best counter against a null. A number the estimator would have
    # reported on unrelated data is not a finding, and this trace length is
    # where that bias lives.
    null: Dict[str, Any] = {}
    if scored:
        candidates = {
            row["counter"]: series[row["counter"]]
            for row in scored
            if row["counter"] in series
        }
        null = shuffle_null(candidates, target_values, bins)
        for row in scored:
            row["above_null_p95"] = bool(
                row["information_beyond_persistence"] > null["null_p95"]
            )
    persistence = scored[0]["persistence_information"] if scored else 0.0
    target_entropy = scored[0]["target_entropy"] if scored else 0.0
    best_beyond = scored[0]["information_beyond_persistence"] if scored else 0.0

    return {
        "measured": True,
        "target": target,
        "intervals": intervals,
        "bins": bins,
        "target_entropy_bits": target_entropy,
        "persistence_information_bits": round(persistence, 4),
        "best_counter_beyond_persistence_bits": round(best_beyond, 4),
        "counters": scored[:top],
        "null": null,
        "survives_null": bool(null and best_beyond > null.get("null_p95", 0.0)),
        "verdict": _verdict(
            target_entropy, persistence, best_beyond, null.get("null_p95")
        ),
    }


def _verdict(
    target_entropy: float,
    persistence: float,
    best_beyond: float,
    null_p95: Optional[float] = None,
) -> str:
    """What the numbers mean for whether to build anything."""
    if target_entropy <= 0.05:
        return (
            "The target barely varies across this trace, so there is nothing "
            "for a predictor to do. This is a property of the workload, not a "
            "negative result about predictors."
        )
    remaining = target_entropy - persistence
    if remaining <= 0.05:
        return (
            f"Persistence already explains essentially all of it "
            f"({persistence:.2f} of {target_entropy:.2f} bits). A "
            "last-value predictor is the answer here and no learned model can "
            "beat it by enough to pay for itself."
        )
    if best_beyond <= 0.05:
        return (
            f"{remaining:.2f} bits are left unexplained by persistence, and no "
            "single counter recovers any of them. Either the signal is not in "
            "this counter set, or it needs a combination -- which costs more "
            "taps and should only be tried with a reason."
        )
    if null_p95 is not None and best_beyond <= null_p95:
        return (
            f"{best_beyond:.2f} bits beyond persistence is INSIDE the null "
            f"({null_p95:.2f} at the 95th percentile on permuted data), so it "
            "is what this estimator reports on a relationship that does not "
            "exist. Conditional mutual information is positively biased at "
            "this trace length. Take a longer trace before believing it."
        )
    share = best_beyond / remaining if remaining else 0.0
    return (
        f"{best_beyond:.2f} bits available beyond persistence, "
        f"{share:.0%} of what persistence leaves. That is the ceiling for a "
        "predictor tapping one counter; it is an upper bound and no design "
        "reaches its ceiling."
    )


def describe() -> List[str]:
    return [
        "predictability is measured before any predictor is designed, and the "
        "number that matters is information beyond persistence -- what a "
        "counter adds over predicting the same as last interval",
        "a trace too short to estimate on returns a refusal naming how many "
        "intervals it needs, never a number",
        "which counters to tap together is decided by greedy selection whose "
        "null runs the same selection on permuted data, and each tap is bought "
        "with its own increment rather than the running total -- a second tap "
        "that lifts the cumulative number while sitting inside its own null is "
        "a counter that won an auction among fifty, not a design",
        "the depth of that selection is computed from the trace, not chosen: "
        "each further tap multiplies the cells the estimate needs by the bin "
        "count, and a trace of a few hundred intervals supports one or two",
    ]


# --- which taps, together --------------------------------------------------
#
# Each PMU tap costs wires and area, so the design question is not "is there
# signal" but "which four counters get most of it". That is a subset-selection
# problem, and two things make it dangerous in a way the single-counter version
# was not.
#
# Sample cost grows exponentially. Conditioning on the target's last value plus
# k counters at b bins needs b**(k+1) cells: at 3 bins, one counter is 9 cells,
# two is 27, three is 81. A 400-interval trace supports two taps and not three,
# and the depth limit is computed rather than chosen.
#
# And selection is itself a source of bias. Picking the best of fifty and then
# measuring it is the multiple-comparisons trap; picking the best of fifty,
# then the best of the remaining forty-nine, compounds it. So the null runs the
# WHOLE greedy selection on permuted data, which is the only way its answer
# includes the cost of having selected.


def max_taps_for(intervals: int, bins: int = DEFAULT_BINS) -> int:
    """How many counters this trace can support jointly, with persistence.

    Counted in *usable* samples, not intervals. Predicting t+1 from t spends
    one interval on the shift, so a 405-interval trace estimates over 404
    pairs -- and three taps at three bins need 405. Counting the interval the
    shift consumed is how a depth limit stops limiting anything exactly at the
    boundary where it was supposed to bite.
    """
    usable = max(0, intervals - 1)
    taps = 0
    while True:
        cells = bins ** (taps + 2)
        if MIN_PER_CELL * cells > usable:
            return taps
        taps += 1


def _joint_information(
    target_ahead: Sequence[int],
    last_target: Sequence[int],
    features: Sequence[Sequence[int]],
) -> float:
    """Information about the next interval from persistence plus these taps."""
    base = conditional_entropy(target_ahead, [last_target])
    return base - conditional_entropy(target_ahead, [last_target, *features])


def _greedy(
    labels: Mapping[str, List[int]],
    target_ahead: List[int],
    last_target: List[int],
    max_taps: int,
) -> List[Dict[str, Any]]:
    chosen: List[str] = []
    chosen_labels: List[List[int]] = []
    gained = 0.0
    steps: List[Dict[str, Any]] = []

    for _ in range(max_taps):
        best_name, best_total = None, gained
        for name, values in labels.items():
            if name in chosen:
                continue
            total = _joint_information(
                target_ahead, last_target, [*chosen_labels, values]
            )
            if total > best_total:
                best_name, best_total = name, total
        if best_name is None:
            break
        chosen.append(best_name)
        chosen_labels.append(labels[best_name])
        steps.append(
            {
                "tap": best_name,
                "taps": len(chosen),
                "total_beyond_persistence": round(best_total, 4),
                "added": round(best_total - gained, 4),
            }
        )
        gained = best_total
    return steps


def select_taps(
    series: Mapping[str, Sequence[float]],
    target: str,
    bins: int = DEFAULT_BINS,
    trials: int = 50,
    seed: int = 4242,
) -> Dict[str, Any]:
    """Which counters, together, carry the signal -- and whether they really do.

    Greedy forward selection over the counters, starting from persistence,
    stopping at the depth the trace can support. The null runs the same
    selection on permuted counters and takes the best it reaches, so the
    threshold already contains the advantage that selection itself confers.
    """
    import random

    if target not in series:
        return {
            "measured": False,
            "refusal": f"no counter named {target!r} in this trace",
        }

    target_values = list(series[target])
    intervals = len(target_values)
    depth = max_taps_for(intervals, bins)
    if depth < 1:
        return {
            "measured": False,
            "intervals": intervals,
            "refusal": (
                f"{intervals} intervals cannot support even one tap alongside "
                f"persistence at {bins} bins: that needs "
                f"{MIN_PER_CELL * bins ** 2 + 1} intervals, one of which the "
                "t-to-t+1 shift spends. A shorter trace can be asked about "
                "single counters, not combinations."
            ),
        }

    target_labels = discretize(target_values, bins)
    last_target, target_ahead = _shift(target_labels)
    labels = {
        name: _shift(discretize(list(values), bins))[0]
        for name, values in series.items()
        if name != target and len(values) == intervals
    }

    steps = _greedy(labels, target_ahead, last_target, depth)

    # A null per depth, and -- the number that decides anything -- a null per
    # INCREMENT. Whether the selected set carries signal and whether the second
    # tap earns its wires are different questions, and only the second one is a
    # design decision. At step d the greedy still has forty-odd counters to
    # choose from, so the best increment it can reach on permuted data is not
    # small, and an increment inside that is a counter that won an auction.
    rng = random.Random(seed)
    null_added: Dict[int, List[float]] = {d: [] for d in range(1, depth + 1)}
    for _ in range(max(1, trials)):
        shuffled = {}
        for name, values in labels.items():
            copy = values[:]
            rng.shuffle(copy)
            shuffled[name] = copy
        trial = _greedy(shuffled, target_ahead, last_target, depth)
        for d in range(1, depth + 1):
            null_added[d].append(trial[d - 1]["added"] if len(trial) >= d else 0.0)

    def _p95(values: List[float]) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]

    surviving = 0
    still_surviving = True
    for step in steps:
        threshold = _p95(null_added[step["taps"]])
        step["null_p95_added"] = round(threshold, 4)
        step["survives_null"] = bool(step["added"] > threshold)
        # Once a tap fails its own null the deeper ones were selected on top of
        # noise, so the recommendation stops here rather than skipping past it.
        if still_surviving and step["survives_null"]:
            surviving = step["taps"]
        else:
            still_surviving = False

    kept = steps[:surviving]
    observed = kept[-1]["total_beyond_persistence"] if kept else 0.0

    return {
        "measured": True,
        "target": target,
        "intervals": intervals,
        "bins": bins,
        "max_taps_supported": depth,
        "selection": steps,
        "recommended_taps": surviving,
        "taps": [step["tap"] for step in kept],
        "total_beyond_persistence": observed,
        "total_at_full_depth": (
            steps[-1]["total_beyond_persistence"] if steps else 0.0
        ),
        "survives_null": bool(surviving),
        "verdict": _tap_verdict(steps, surviving, depth),
        "note": (
            f"At {bins} bins a trace of {intervals} intervals supports "
            f"{depth} tap(s) alongside persistence; each further tap "
            f"multiplies the cells by {bins}. Every tap is placed against a "
            "null of the same greedy selection on permuted counters, so the "
            "threshold already contains the advantage selection confers -- and "
            "each tap is judged on what IT added, not on the running total, "
            "because that is what its wires are being bought with."
        ),
    }


def _tap_verdict(steps: List[Dict[str, Any]], surviving: int, depth: int) -> str:
    """What the selection means for how many taps to build."""
    if not steps:
        return (
            "No counter adds anything to persistence at all. The signal is not "
            "in this counter set, and no combination of them changes that."
        )
    if surviving == 0:
        best = steps[0]
        return (
            f"The best first tap ({best['tap']}) adds {best['added']:.3f} bits, "
            f"INSIDE the null of {best['null_p95_added']:.3f} for choosing the "
            "best of this many counters on permuted data. That is what greedy "
            "selection reports on no relationship at all. Nothing here is "
            "worth a wire."
        )
    kept = steps[:surviving]
    names = ", ".join(step["tap"] for step in kept)
    total = kept[-1]["total_beyond_persistence"]
    if surviving < len(steps):
        first_dead = steps[surviving]
        return (
            f"{surviving} tap(s) survive their own null: {names}, "
            f"{total:.3f} bits beyond persistence together. The next one "
            f"({first_dead['tap']}) adds {first_dead['added']:.3f} against a "
            f"null of {first_dead['null_p95_added']:.3f}, so it is selection "
            "bias and not signal -- build the survivors and stop."
        )
    if surviving == depth:
        return (
            f"All {surviving} tap(s) this trace can support survive their null "
            f"({names}, {total:.3f} bits beyond persistence). Whether a further "
            "tap would add anything is UNMEASURED, not answered: the trace is "
            "too short to condition on one more. A longer trace, not a "
            "conclusion."
        )
    return (
        f"{surviving} tap(s) survive their null: {names}, {total:.3f} bits "
        "beyond persistence."
    )
