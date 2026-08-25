"""Measure a design's fitness before paying for the measurement.

The adversarial critic reads an artifact and reasons about it. This does not
read it at all: it runs the workload in the cheapest simulator model there is,
counts what comes out, and compares that against what the estimator waiting
downstream actually needs. Where the critic is general and fallible, this is
narrow and certain, and it can therefore refuse rather than advise.

**Why it catches traps nobody enumerated.** Every check here is a property of
the trace the design will produce -- how many intervals, how much they vary,
whether they split into regimes, what they will cost -- rather than a rule about
how the workload was written. The four defects this study lost time to were a
trace too short to estimate on, two regimes spliced together, a target so flat
that binning invented its structure, and phases blocked instead of interleaved.
All four are visible here as numbers, and none of them needed to be foreseen.

**Instruction counts, not cycles.** The cheap model has no timing, so its cycle
counts mean nothing. Instructions per interval are still exact, and they answer
the structural questions: a workload that does the same work every interval,
or does one kind of work and then another, says so in its instruction counts
whatever the timing model.

The consequence, stated rather than hidden: a design whose *cycles* vary while
its instruction counts do not will pass here and may still be flat. This
narrows the gap; it does not close it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from app.services.agent_gem5_sandbox import find_regime_change
from app.services.agent_predictability import (
    MIN_PER_CELL,
    DEFAULT_BINS,
    relative_spread,
)

#: What an out-of-order model sustains, near enough for an order-of-magnitude
#: projection. The sandbox docstring quotes the same figure; it is a planning
#: number, not a measurement, and the projection says so.
O3_INSTRUCTIONS_PER_SECOND = 100_000

#: Below this the instruction count per interval is constant for practical
#: purposes, and a target that tracks the work done will have nothing in it.
FLAT_INSTRUCTION_SPREAD = 0.02


def minimum_intervals(bins: int = DEFAULT_BINS) -> int:
    """What the ceiling estimator needs, plus the interval the shift spends."""
    return MIN_PER_CELL * bins * bins + 1


def judge(
    instructions_per_interval: Sequence[float],
    *,
    bins: int = DEFAULT_BINS,
    timeout_seconds: int = 1800,
    intends_alternating_phases: bool = False,
) -> Dict[str, Any]:
    """What the trace this design produces will and will not support.

    Pure: given the per-interval instruction counts, everything below is
    arithmetic. The expensive part -- obtaining them -- happens elsewhere, so
    this is testable without a simulator.
    """
    counts = [float(v) for v in instructions_per_interval]
    intervals = len(counts)
    total = sum(counts)
    projected = total / O3_INSTRUCTIONS_PER_SECOND if total else 0.0
    spread = relative_spread(counts) if counts else None
    needed = minimum_intervals(bins)

    concerns: List[Dict[str, Any]] = []

    if intervals < needed:
        concerns.append(
            {
                "check": "intervals",
                "severity": "blocking",
                "summary": (
                    f"{intervals} intervals; the ceiling estimate needs "
                    f"{needed} at {bins} bins"
                ),
                "remedy": (
                    "Call M5_SAMPLE() more often, or run more iterations of "
                    "the outer loop. Below this the estimate is mostly empty "
                    "bins, which look like structure and are not."
                ),
            }
        )

    if projected > timeout_seconds:
        concerns.append(
            {
                "check": "cost",
                "severity": "blocking",
                "summary": (
                    f"{total:,.0f} instructions is about "
                    f"{projected / 60:.0f} min of out-of-order simulation, "
                    f"past the {timeout_seconds // 60} min timeout"
                ),
                "remedy": (
                    "Shrink the work per interval, not the interval count: "
                    "fewer intervals cost the estimate its sample size."
                ),
            }
        )

    if spread is not None and spread < FLAT_INSTRUCTION_SPREAD:
        concerns.append(
            {
                "check": "variation",
                "severity": "serious",
                "summary": (
                    f"instructions per interval vary by {spread:.1%}; the work "
                    "done is effectively constant"
                ),
                "remedy": (
                    "That is fine for a primary under contention, where the "
                    "point is that its own work does not vary. It is not fine "
                    "for a solo predictability study: a target that tracks the "
                    "work will be flat, and three bins will split its jitter "
                    "into levels that are the quantiser rather than the "
                    "workload."
                ),
            }
        )

    regime = find_regime_change(counts) if intervals >= 60 else None
    if regime:
        blocked = bool(intends_alternating_phases)
        concerns.append(
            {
                "check": "regime",
                "severity": "blocking" if blocked else "serious",
                "summary": (
                    f"the work changes level {regime['ratio']}x at interval "
                    f"{regime['at_interval']}: this design produces two "
                    "regimes, not one"
                ),
                "remedy": (
                    "Interleave the phases so each interval sees both, rather "
                    "than running all of one and then all of the other."
                    if blocked
                    else "Measure one side with from_interval, or lengthen the "
                    "run until the opening regime is a negligible fraction."
                ),
                "at_interval": regime["at_interval"],
            }
        )

    blocking = [c for c in concerns if c["severity"] == "blocking"]
    return {
        "measured": True,
        "intervals": intervals,
        "instructions_total": total,
        "instruction_spread": (round(spread, 4) if spread is not None else None),
        "projected_o3_seconds": round(projected, 1),
        "regime_change": regime,
        "concerns": concerns,
        "blocking": blocking,
        "fit": not blocking,
    }


def refusal(verdict: Dict[str, Any]) -> str:
    """One message naming every blocking defect and its remedy."""
    lines = [
        "This design was measured in the cheap simulator model before "
        "committing to the expensive one, and it will not support the "
        "measurement it is for:"
    ]
    for concern in verdict.get("blocking") or []:
        lines.append(f"- {concern['summary']}. {concern['remedy']}")
    lines.append(
        f"Measured: {verdict.get('intervals')} intervals, "
        f"{verdict.get('instructions_total'):,.0f} instructions, "
        f"about {float(verdict.get('projected_o3_seconds') or 0) / 60:.0f} min "
        "of out-of-order simulation. Fix the design and call again; running it "
        "as it stands spends that time to learn this."
    )
    return " ".join(lines)


def describe() -> List[str]:
    return [
        "a workload is run in the cheapest simulator model first and its "
        "instruction counts are checked against what the estimator needs -- "
        "interval count, variation, regime structure and projected cost -- "
        "before the expensive run is started",
        "that check refuses rather than advises, because it is arithmetic on a "
        "measurement rather than a judgement about a design",
    ]
