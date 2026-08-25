"""What a buildable predictor actually gets, against the ceiling it was sold on.

`agent_predictability` measures how much information the counters carry about
the next interval. That is an upper bound, and an upper bound is not a design:
it says what is available, not what a few hundred transistors can reach. This
module closes that loop by running the predictor the measurement indicated and
scoring it against its own ceiling on held-out intervals.

The point is to be able to end a study. If the cheap design captures most of
what the feature set supports, a learned model has nothing left to win and
nobody needs to generate a training corpus to discover that. If a real gap
opens, the gap is the argument for spending more -- and it is a measured gap
rather than an assumed one.

**Three traps, all of which have killed published results.**

*Splitting a time series at random leaks.* Adjacent intervals are nearly
identical, so a random split puts each test row's near-twin in training and
every predictor looks excellent. The split here is contiguous: the first part
of the trace warms the tables, the rest is scored.

*The reference must be fit on training data too.* The accuracy analogue of the
information ceiling is the best a table indexed by these features could do --
majority outcome per cell. Fit that on the test segment and it is fit to the
answers, which turns the ceiling into a number nothing can fall short of.

*Beating persistence is the bar, not beating chance.* A predictor that reads
90% correct against a target that repeats 88% of the time has done nothing.
Every number here is reported as a gain over predicting the same as last
interval, and placed against a null that shuffles the tap and reruns the whole
pipeline.

**Both update modes are scored, and neither is the honest default.** A table
that keeps updating is what most predictors do, and updating during the scored
segment is not leakage: the outcome of interval t is known long before t+2 is
predicted. But a table trained once and held is equally buildable -- it is the
same design with a longer time constant -- and a counter that moves on every
surprise chases noise. Scoring only the updating mode reports a design that
works as a design that does not, which is what this module did until a trace
where holding the table was worth more than the tap itself showed it.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from app.services.agent_predictability import (
    DEFAULT_BINS,
    MIN_PER_CELL,
    discretize,
)

#: Fraction of the trace that warms the tables. The rest is scored.
DEFAULT_SPLIT = 0.5

#: Scored intervals below which an accuracy difference is not worth reporting.
#: Two designs a few rows apart on a short segment are the same design.
MIN_SCORED = 30


class SaturatingTable:
    """A table of saturating counters, one per index -- a bimodal predictor.

    `hysteresis` is how many internal steps make up one predicted level, which
    is the difference between a counter that flips on a single surprise and one
    that has to be convinced. It is also what the entry costs: `levels *
    hysteresis` states, so 3 levels at hysteresis 2 is 6 states, 3 bits.
    """

    def __init__(self, levels: int, hysteresis: int = 2) -> None:
        self.levels = levels
        self.hysteresis = hysteresis
        self.states = levels * hysteresis
        # Reset state is the middle, which is what an untrained entry predicts.
        self._mid = self.states // 2
        self._table: Dict[Tuple[int, ...], int] = {}

    def bits_per_entry(self) -> int:
        bits = 1
        while (1 << bits) < self.states:
            bits += 1
        return bits

    def predict(self, index: Tuple[int, ...]) -> int:
        return self._table.get(index, self._mid) // self.hysteresis

    def _confident_state(self, level: int) -> int:
        """Where repeated agreement drives the counter for this outcome.

        A saturating counter earns its hysteresis by settling at the EXTREME of
        its state space, not at the edge of its prediction band. Settling on
        the edge means one contrary interval crosses the boundary and the
        prediction flips, which is the behaviour hysteresis exists to prevent
        -- and the shape this had until a test asked it to resist one surprise.
        The middle levels have no outer edge to hide behind and sit mid-band,
        which is a real property of sharing one counter across levels rather
        than an approximation.
        """
        if level <= 0:
            return 0
        if level >= self.levels - 1:
            return self.states - 1
        return level * self.hysteresis + (self.hysteresis - 1) // 2

    def update(self, index: Tuple[int, ...], outcome: int) -> None:
        state = self._table.get(index, self._mid)
        target = self._confident_state(outcome)
        if target > state:
            state += 1
        elif target < state:
            state -= 1
        self._table[index] = max(0, min(self.states - 1, state))

    def entries(self) -> int:
        return len(self._table)


class LevelCounters:
    """Per cell, one small counter per outcome level; predict the leader.

    The shared saturating counter above has to travel through the middle of its
    range to change its mind, so where a cell's majority is weak it never
    arrives and the table quietly degenerates into predicting the same as last
    interval. That is a property of the mechanism, not of the information --
    and telling those two apart is the entire point of scoring against a
    ceiling, so the mechanism that CAN reach the ceiling has to be in the set.

    This is the direct hardware analogue of the reference: the leader among a
    cell's counters is its majority outcome, tracked incrementally. It costs
    one counter per level instead of one per cell.
    """

    def __init__(self, levels: int, width: int = 4) -> None:
        self.levels = levels
        self.width = width
        self._table: Dict[Tuple[int, ...], List[int]] = {}

    def bits_per_entry(self) -> int:
        bits = 1
        while (1 << bits) < self.width:
            bits += 1
        return bits * self.levels

    def predict(self, index: Tuple[int, ...]) -> int:
        counters = self._table.get(index)
        if not counters:
            # An untrained cell has no leader. Predicting the middle level is
            # what a reset table does; it is not a claim about the workload.
            return self.levels // 2
        return max(range(self.levels), key=lambda level: counters[level])

    def update(self, index: Tuple[int, ...], outcome: int) -> None:
        counters = self._table.setdefault(index, [0] * self.levels)
        for level in range(self.levels):
            if level == outcome:
                counters[level] = min(self.width - 1, counters[level] + 1)
            else:
                counters[level] = max(0, counters[level] - 1)


def _run_levels(
    indices: Sequence[Tuple[int, ...]],
    outcomes: Sequence[int],
    split: int,
    levels: int,
    online: bool,
) -> float:
    table = LevelCounters(levels)
    for i in range(split):
        table.update(indices[i], outcomes[i])

    correct = 0
    for i in range(split, len(outcomes)):
        if table.predict(indices[i]) == outcomes[i]:
            correct += 1
        if online:
            table.update(indices[i], outcomes[i])
    scored = len(outcomes) - split
    return correct / scored if scored else 0.0


def _run(
    indices: Sequence[Tuple[int, ...]],
    outcomes: Sequence[int],
    split: int,
    levels: int,
    hysteresis: int,
    online: bool,
) -> float:
    """Accuracy over the scored segment, after warming on the first `split`."""
    table = SaturatingTable(levels, hysteresis)
    for i in range(split):
        table.update(indices[i], outcomes[i])

    correct = 0
    for i in range(split, len(outcomes)):
        if table.predict(indices[i]) == outcomes[i]:
            correct += 1
        if online:
            table.update(indices[i], outcomes[i])
    scored = len(outcomes) - split
    return correct / scored if scored else 0.0


def _oracle(
    indices: Sequence[Tuple[int, ...]],
    outcomes: Sequence[int],
    split: int,
) -> Tuple[float, int]:
    """Best a table on these features could do, fit on the warm-up only.

    The accuracy analogue of the information ceiling: pick the most common
    outcome per cell. Returns the accuracy and how many scored rows landed in a
    cell the warm-up never saw -- a cell with no training is the honest reason
    a ceiling is not reached, and it must be visible rather than absorbed.
    """
    counts: Dict[Tuple[int, ...], Dict[int, int]] = {}
    for i in range(split):
        counts.setdefault(indices[i], {})
        counts[indices[i]][outcomes[i]] = counts[indices[i]].get(outcomes[i], 0) + 1
    best = {
        index: max(hist.items(), key=lambda kv: kv[1])[0]
        for index, hist in counts.items()
    }

    correct = unseen = 0
    for i in range(split, len(outcomes)):
        if indices[i] not in best:
            unseen += 1
            continue
        if best[indices[i]] == outcomes[i]:
            correct += 1
    scored = len(outcomes) - split
    return (correct / scored if scored else 0.0), unseen


def _designs(
    last: Sequence[int],
    tap: Sequence[int],
    outcomes: Sequence[int],
    split: int,
    bins: int,
) -> List[Dict[str, Any]]:
    paired = [(a, b) for a, b in zip(last, tap)]
    solo = [(a,) for a in last]

    rows = []
    for name, indices, hysteresis, cells in (
        ("bimodal on last value", solo, 2, bins),
        ("last value + tap", paired, 1, bins * bins),
        ("last value + tap, with hysteresis", paired, 2, bins * bins),
    ):
        table = SaturatingTable(bins, hysteresis)
        rows.append(
            {
                "design": name,
                "cells": cells,
                "bits_per_cell": table.bits_per_entry(),
                "state_bits": cells * table.bits_per_entry(),
                "accuracy_frozen": round(
                    _run(indices, outcomes, split, bins, hysteresis, False), 4
                ),
                "accuracy_online": round(
                    _run(indices, outcomes, split, bins, hysteresis, True), 4
                ),
            }
        )

    levels = LevelCounters(bins)
    rows.append(
        {
            "design": "last value + tap, per-level counters",
            "cells": bins * bins,
            "bits_per_cell": levels.bits_per_entry(),
            "state_bits": bins * bins * levels.bits_per_entry(),
            "accuracy_frozen": round(
                _run_levels(paired, outcomes, split, bins, False), 4
            ),
            "accuracy_online": round(
                _run_levels(paired, outcomes, split, bins, True), 4
            ),
        }
    )
    return rows


def evaluate(
    series: Mapping[str, Sequence[float]],
    target: str,
    tap: str,
    bins: int = DEFAULT_BINS,
    split: float = DEFAULT_SPLIT,
    trials: int = 50,
    seed: int = 909,
) -> Dict[str, Any]:
    """Run the indicated predictor and score it against its own ceiling.

    Returns a refusal rather than a number when the trace cannot support the
    comparison -- the same rule the ceiling estimate follows, because "the
    design reached 60% of its ceiling" and "we could not tell" are opposite
    findings and only one of them justifies building anything.
    """
    for name in (target, tap):
        if name not in series:
            return {
                "measured": False,
                "refusal": (
                    f"no counter named {name!r} in this trace. Available: "
                    + ", ".join(sorted(series)[:12])
                ),
            }

    target_labels = discretize(list(series[target]), bins)
    tap_labels = discretize(list(series[tap]), bins)
    if len(target_labels) != len(tap_labels):
        return {
            "measured": False,
            "refusal": (
                f"{target} has {len(target_labels)} intervals and {tap} has "
                f"{len(tap_labels)}. These are not the same trace."
            ),
        }

    # Features at t, outcome at t+1 -- the only alignment that predicts.
    last, tap_at_t, outcomes = (
        target_labels[:-1],
        tap_labels[:-1],
        target_labels[1:],
    )
    pairs = len(outcomes)
    warm = int(pairs * split)
    scored = pairs - warm

    needed_warm = MIN_PER_CELL * bins * bins
    if warm < needed_warm:
        return {
            "measured": False,
            "intervals": len(target_labels),
            "refusal": (
                f"a warm-up of {warm} pairs cannot fill a {bins}x{bins} table: "
                f"that needs {needed_warm} to keep {MIN_PER_CELL} per cell. An "
                "untrained cell predicts its reset state, so the design would "
                "be scored on a table that was never built."
            ),
        }
    if scored < MIN_SCORED:
        return {
            "measured": False,
            "intervals": len(target_labels),
            "refusal": (
                f"{scored} scored intervals is too few to separate designs; "
                f"{MIN_SCORED} is the floor. Two designs a handful of rows "
                "apart on a segment this short are the same design."
            ),
        }

    persistence = (
        sum(
            1
            for lastval, outcome in zip(last[warm:], outcomes[warm:])
            if lastval == outcome
        )
        / scored
    )
    ceiling_accuracy, unseen = _oracle(
        [(a, b) for a, b in zip(last, tap_at_t)], outcomes, warm
    )
    designs = _designs(last, tap_at_t, outcomes, warm, bins)

    headroom = ceiling_accuracy - persistence
    for row in designs:
        frozen = row["accuracy_frozen"] - persistence
        online = row["accuracy_online"] - persistence
        row["gain_frozen"] = round(frozen, 4)
        row["gain_online"] = round(online, 4)
        # A table trained once and held is as buildable as one that keeps
        # updating -- it is a longer time constant, in the limit. Scoring only
        # the updating mode reports a working design as a failed one.
        row["best_mode"] = "frozen" if frozen >= online else "online"
        gain = max(frozen, online)
        row["gain_over_persistence"] = round(gain, 4)
        row["share_of_headroom"] = (
            round(gain / headroom, 3) if headroom > 1e-9 else None
        )

    best = max(designs, key=lambda row: row["gain_over_persistence"])

    # The null: shuffle the tap and rerun everything. A design whose gain is
    # inside this is reading a counter that carries nothing, and the table is
    # merely a slower way of predicting the same as last interval.
    rng = random.Random(seed)
    null_gains = []
    for _ in range(max(1, trials)):
        shuffled = tap_at_t[:]
        rng.shuffle(shuffled)
        rows = _designs(last, shuffled, outcomes, warm, bins)
        # Best design AND best mode, matching how the observed number is taken.
        null_gains.append(
            max(max(row["accuracy_frozen"], row["accuracy_online"]) for row in rows)
            - persistence
        )
    null_gains.sort()
    null_p95 = null_gains[min(len(null_gains) - 1, int(0.95 * len(null_gains)))]

    return {
        "measured": True,
        "target": target,
        "tap": tap,
        "bins": bins,
        "intervals": len(target_labels),
        "warmup_intervals": warm,
        "scored_intervals": scored,
        "persistence_accuracy": round(persistence, 4),
        "ceiling_accuracy": round(ceiling_accuracy, 4),
        "headroom": round(headroom, 4),
        "scored_rows_in_untrained_cells": unseen,
        "designs": designs,
        "best_design": best["design"],
        "best_gain_over_persistence": best["gain_over_persistence"],
        "best_share_of_headroom": best["share_of_headroom"],
        "null_p95_gain": round(null_p95, 4),
        "survives_null": bool(best["gain_over_persistence"] > null_p95),
        "ceiling_exceeded": bool(
            best["share_of_headroom"] is not None and best["share_of_headroom"] > 1.0
        ),
        "verdict": _verdict(
            persistence, ceiling_accuracy, best, null_p95, unseen, scored
        ),
    }


def _verdict(
    persistence: float,
    ceiling: float,
    best: Dict[str, Any],
    null_p95: float,
    unseen: int,
    scored: int,
) -> str:
    """What the numbers mean for whether to build this or something bigger."""
    headroom = ceiling - persistence
    gain = best["gain_over_persistence"]

    if headroom <= 0.005:
        return (
            f"Predicting the same as last interval is already {persistence:.1%} "
            f"correct, and the best a table on this feature set could do is "
            f"{ceiling:.1%}. There is nothing here to build: no predictor "
            "reading this counter can be meaningfully better than a wire."
        )
    if gain <= null_p95:
        return (
            f"The best design gains {gain:+.1%} over persistence, INSIDE the "
            f"null of {null_p95:+.1%} from shuffling the tap. The table is a "
            "slower way of predicting the same as last interval. Either the "
            "counter carries nothing at this granularity, or the scored "
            f"segment ({scored} intervals) is too short to show it."
        )

    share = best["share_of_headroom"]
    # A design cannot beat its own ceiling. If it does, the ceiling was fit on
    # the warm-up and the scored segment is short enough for luck to cover the
    # difference -- which makes the ceiling an estimate, not a bound, and says
    # so rather than rounding the number down to look tidy.
    if share is not None and share > 1.0:
        return (
            f"{best['design']} ({best['state_bits']} bits of state, "
            f"{best['best_mode']}) gains {gain:+.1%} over persistence, which "
            f"is {share:.0%} of the {headroom:.1%} its own ceiling allows. "
            "Exceeding the ceiling means the ceiling is not a bound here: it "
            f"was fit on the warm-up, and {scored} scored intervals leave "
            "enough room for luck to cover the gap. Read this as 'the cheap "
            "design gets everything this feature set has, and the trace is "
            "too short to say more precisely than that'."
        )
    note = (
        f" {unseen} of {scored} scored intervals landed in cells the warm-up "
        "never saw, which is part of why the ceiling is not reached."
        if unseen
        else ""
    )
    # Whether continuous update helps or hurts is a design parameter, not a
    # detail: a counter that moves on every surprise chases noise, and where
    # that costs more than the tap is worth, the time constant is the thing to
    # spend on rather than more state.
    chasing = best["gain_frozen"] - best["gain_online"]
    chase = (
        f" Holding the table after warm-up is worth {chasing:+.1%} against "
        "letting it update every interval, so the time constant matters more "
        "here than the table does."
        if chasing > 0.005
        else ""
    )

    if share is not None and share >= 0.7:
        return (
            f"{best['design']} ({best['state_bits']} bits of state, "
            f"{best['best_mode']}) gains {gain:+.1%} over persistence and "
            f"captures {share:.0%} of the {headroom:.1%} a table on these "
            "features supports. That is most of what the feature set has to "
            "give, so a learned model is competing for the remainder and the "
            f"study can end here rather than in a training corpus.{note}{chase}"
        )
    return (
        f"{best['design']} ({best['state_bits']} bits of state, "
        f"{best['best_mode']}) gains {gain:+.1%} over persistence but captures "
        f"only {share:.0%} of the {headroom:.1%} a table on these features "
        "supports. The gap is the argument for something with more state -- "
        f"and it is a measured gap, which is what makes it worth spending "
        f"on.{note}{chase}"
    )


def describe() -> List[str]:
    return [
        "an information ceiling says what is available, not what a design "
        "reaches; the cheap predictor is run and scored against its own "
        "ceiling before anyone generates a training corpus",
        "the split of a counter trace is contiguous, never random -- adjacent "
        "intervals are near-identical and a random split puts each scored "
        "row's twin in the warm-up, which makes every predictor look excellent",
        "every predictor number is a gain over predicting the same as last "
        "interval, placed against a null that shuffles the tap and reruns the "
        "whole pipeline",
    ]
