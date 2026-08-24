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
    ]
