"""Deciding whether a measurement reproduces a paper's claim.

A run that implements an algorithm from a paper and times it ends with two
numbers: what the paper said, and what the machine did. Turning that pair into
a verdict is where reproduction studies usually go wrong, and they go wrong in
both directions.

The first failure is calling a mismatch a refutation. A paper reporting 3.0x on
a Xeon at n=10^7 and a run measuring 2.1x on aarch64 at n=10^5 has not failed
to reproduce anything: those two numbers were never comparable, and reporting
"not reproduced" claims a finding the run did not earn. `incomparable` is a
first-class verdict here for that reason, and it is not a weaker form of
failure -- it is the honest answer whenever the comparison itself does not
hold, and it names which condition broke so the run can go and fix it.

The second failure is the flattering match. Absolute times move by an order of
magnitude across machines, so an absolute claim measured on different hardware
agrees or disagrees by luck. Only ratios -- speedup, reduction, cycles per
element -- survive the move, and only approximately. So the unit decides what a
comparison can mean at all, before any arithmetic happens.

What this module deliberately does not do is decide whether the implementation
was faithful to the paper. A verdict of `reproduced` says the number matched;
it says nothing about whether the thing measured was the paper's algorithm.
That is `agent_implementation_check`'s question, and keeping the two apart is
what stops "it was fast" from being read as "it was right and fast".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

#: How far from the claim still counts as reproducing it, as a fraction of the
#: claimed value. Papers report speedups to two significant figures and the
#: machine underneath is never the same one, so demanding better than this
#: measures the reporting convention rather than the algorithm. An author who
#: knows their claim is tighter than this can say so per comparison.
DEFAULT_TOLERANCE = 0.20

#: Units whose value is a ratio of two measurements taken the same way. These
#: survive a change of machine, approximately, because the machine appears in
#: both halves and largely cancels.
_RELATIVE_UNITS = {
    "x",
    "×",
    "speedup",
    "ratio",
    "factor",
    "fold",
    "percent",
    "%",
    "pct",
    "percentage",
    "reduction",
    "improvement",
    "cycles_per_element",
    "cycles/element",
    "cycles_per_iteration",
    "cycles/iteration",
    "instructions_per_cycle",
    "ipc",
    "bytes_per_cycle",
}

#: Units naming a duration, a rate or a count on one specific machine. Two of
#: these from two different machines are not the same quantity, however close
#: the numbers happen to land.
_ABSOLUTE_UNITS = {
    "s",
    "sec",
    "secs",
    "second",
    "seconds",
    "ms",
    "millisecond",
    "milliseconds",
    "us",
    "µs",
    "microsecond",
    "microseconds",
    "ns",
    "nanosecond",
    "nanoseconds",
    "cycle",
    "cycles",
    "instruction",
    "instructions",
    "gflops",
    "mflops",
    "flops",
    "gb/s",
    "mb/s",
    "ops/s",
    "throughput",
    "bytes",
    "kb",
    "mb",
    "gb",
}

#: Ratios expressed as a percentage rather than a multiplier, so a 40% speedup
#: is not read as a 40x one.
_PERCENT_UNITS = {"percent", "%", "pct", "percentage"}

VERDICT_REPRODUCED = "reproduced"
VERDICT_NOT_REPRODUCED = "not_reproduced"
VERDICT_INCOMPARABLE = "incomparable"


@dataclass
class ClaimComparison:
    """The verdict, and everything needed to argue with it."""

    verdict: str
    claimed_value: Optional[float] = None
    measured_value: Optional[float] = None
    #: measured / claimed, when both exist and share a unit. 1.0 is exact.
    ratio: Optional[float] = None
    #: |measured - claimed| / |claimed|, the number the tolerance is against.
    relative_error: Optional[float] = None
    tolerance: float = DEFAULT_TOLERANCE
    unit_kind: str = "unknown"
    #: Which conditions stopped the comparison, when the verdict is
    #: incomparable. Named individually so a run can fix one and retry.
    blockers: List[str] = field(default_factory=list)
    #: Things that do not block the comparison but change what it is worth.
    caveats: List[str] = field(default_factory=list)
    summary: str = ""

    @property
    def comparable(self) -> bool:
        return self.verdict != VERDICT_INCOMPARABLE

    def as_evidence(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "comparable": self.comparable,
            "claimed_value": self.claimed_value,
            "measured_value": self.measured_value,
            "ratio": self.ratio,
            "relative_error": self.relative_error,
            "tolerance": self.tolerance,
            "unit_kind": self.unit_kind,
            "blockers": self.blockers,
            "caveats": self.caveats,
            "summary": self.summary,
        }


def normalize_unit(unit: Optional[str]) -> str:
    """Reduce a unit to a comparable token.

    Papers write the same unit a dozen ways -- "1.8x", "1.8 X", "speedup of
    1.8", "×" -- and a comparison that treats those as different units refuses
    every real claim it is given.
    """
    text = str(unit or "").strip().lower()
    if not text:
        return ""
    text = text.replace("times", "x").replace(" ", "")
    text = re.sub(r"^(a|the)", "", text)
    # "1.8x" or "x1.8" written into the unit field: keep the unit, drop the
    # number, which belongs in the value.
    text = re.sub(r"[\d.]+", "", text).strip("_-/")
    return text or ""


def unit_kind(unit: Optional[str]) -> str:
    """Whether a unit denotes a ratio, an absolute quantity, or neither."""
    token = normalize_unit(unit)
    if not token:
        return "unknown"
    if token in _RELATIVE_UNITS:
        return "relative"
    if token in _ABSOLUTE_UNITS:
        return "absolute"
    # Compound ratios like "cycles_per_byte" are relative by construction.
    if "per" in token or "/" in token:
        return "relative"
    return "unknown"


def _as_percent_multiplier(value: float, unit: Optional[str]) -> float:
    """Read a percentage claim as the multiplier it means.

    A paper claiming "40% faster" and a run measuring "1.4x" agree, and a
    comparison that reads 40 against 1.4 reports a catastrophic failure that
    did not happen.
    """
    if normalize_unit(unit) in _PERCENT_UNITS:
        return 1.0 + (value / 100.0)
    return value


def compare(
    *,
    claimed_value: Optional[float],
    measured_value: Optional[float],
    claimed_unit: Optional[str] = None,
    measured_unit: Optional[str] = None,
    measurement_source: Optional[str] = None,
    claimed_conditions: Optional[Dict[str, Any]] = None,
    measured_conditions: Optional[Dict[str, Any]] = None,
    tolerance: Optional[float] = None,
) -> ClaimComparison:
    """Compare one measured number against one claimed number.

    Every path that cannot yield a defensible verdict returns `incomparable`
    with the reason named, rather than a number dressed up as a conclusion.
    """
    tol = DEFAULT_TOLERANCE if tolerance is None else float(tolerance)
    if tol <= 0:
        tol = DEFAULT_TOLERANCE

    claimed_conditions = claimed_conditions or {}
    measured_conditions = measured_conditions or {}
    blockers: List[str] = []
    caveats: List[str] = []

    kind = unit_kind(claimed_unit)

    # A claim with no number is not a claim a measurement can test. Papers make
    # plenty of these ("substantially faster") and the honest response is to
    # say the paper did not give a number, not to score against a guess.
    if claimed_value is None:
        blockers.append(
            "The paper's claim has no numeric value, so there is nothing to "
            "compare a measurement against"
        )
    if measured_value is None:
        blockers.append("No measured value was supplied")

    # A number without a referee is a recalled number, and this whole chain
    # exists to keep those out of verdicts.
    if not str(measurement_source or "").strip():
        blockers.append(
            "No measurement_source: a number whose origin is not stated cannot "
            "settle a claim, because nothing distinguishes it from a recollection"
        )

    claimed_token = normalize_unit(claimed_unit)
    measured_token = normalize_unit(measured_unit)
    if claimed_token and measured_token and claimed_token != measured_token:
        # Percent against multiplier is the one mismatch that is really a
        # notation difference, and it is converted rather than refused.
        percent_vs_ratio = {claimed_token, measured_token} <= (
            _PERCENT_UNITS | {"x", "speedup", "ratio", "factor", "fold"}
        )
        if not percent_vs_ratio:
            blockers.append(
                f"Units differ: the paper claims {claimed_token!r} and the run "
                f"measured {measured_token!r}; these are not the same quantity"
            )
    elif not claimed_token:
        caveats.append(
            "The claim carries no unit, so the comparison assumes both numbers "
            "denote the same quantity"
        )

    # Hardware. An absolute time on someone else's machine is not a target this
    # machine can hit or miss; a ratio mostly survives the move.
    claimed_hw = str(claimed_conditions.get("hardware") or "").strip().lower()
    measured_hw = str(measured_conditions.get("hardware") or "").strip().lower()
    if claimed_hw and measured_hw and claimed_hw != measured_hw:
        if kind == "absolute":
            blockers.append(
                f"An absolute {claimed_token or 'measurement'} claimed on "
                f"{claimed_hw!r} cannot be checked on {measured_hw!r}: the two "
                "numbers are properties of different machines"
            )
        else:
            caveats.append(
                f"Measured on {measured_hw!r} against a claim made on "
                f"{claimed_hw!r}; a ratio survives that move only approximately"
            )
    elif kind == "absolute" and not (claimed_hw and measured_hw):
        caveats.append(
            "An absolute quantity is being compared without both machines "
            "stated, so a match may be coincidence"
        )

    # Input size. The one condition that changes a ratio outright: an algorithm
    # whose advantage is asymptotic shows none of it at a size that fits in L1.
    claimed_n = claimed_conditions.get("input_size")
    measured_n = measured_conditions.get("input_size")
    if claimed_n is not None and measured_n is not None:
        if str(claimed_n).strip() != str(measured_n).strip():
            blockers.append(
                f"Input size differs: the claim is at {claimed_n} and the "
                "measurement at "
                f"{measured_n}; an algorithmic advantage is a function of size, "
                "so these do not test each other"
            )
    elif claimed_n is not None or measured_n is not None:
        caveats.append(
            "Input size is stated on only one side, so the comparison assumes "
            "they match"
        )

    if blockers:
        return ClaimComparison(
            verdict=VERDICT_INCOMPARABLE,
            claimed_value=claimed_value,
            measured_value=measured_value,
            tolerance=tol,
            unit_kind=kind,
            blockers=blockers,
            caveats=caveats,
            summary=(
                "The measurement and the claim are not comparable: "
                + blockers[0]
                + ("" if len(blockers) == 1 else f" (and {len(blockers) - 1} more)")
            ),
        )

    claimed = _as_percent_multiplier(float(claimed_value), claimed_unit)
    measured = _as_percent_multiplier(float(measured_value), measured_unit)

    if claimed == 0:
        # Relative error is undefined; refuse rather than divide.
        return ClaimComparison(
            verdict=VERDICT_INCOMPARABLE,
            claimed_value=claimed_value,
            measured_value=measured_value,
            tolerance=tol,
            unit_kind=kind,
            blockers=["The claimed value is zero, so relative error is undefined"],
            caveats=caveats,
            summary="The claimed value is zero; relative error is undefined.",
        )

    ratio = measured / claimed
    relative_error = abs(measured - claimed) / abs(claimed)
    reproduced = relative_error <= tol

    if reproduced:
        summary = (
            f"Measured {measured_value:g} against a claimed {claimed_value:g} "
            f"({relative_error:.1%} off, within the {tol:.0%} tolerance): the "
            "claim reproduces."
        )
    else:
        direction = "above" if measured > claimed else "below"
        summary = (
            f"Measured {measured_value:g} against a claimed {claimed_value:g} "
            f"-- {relative_error:.1%} {direction} the claim, outside the "
            f"{tol:.0%} tolerance."
        )

    return ClaimComparison(
        verdict=VERDICT_REPRODUCED if reproduced else VERDICT_NOT_REPRODUCED,
        claimed_value=claimed_value,
        measured_value=measured_value,
        ratio=ratio,
        relative_error=relative_error,
        tolerance=tol,
        unit_kind=kind,
        caveats=caveats,
        summary=summary,
    )


def describe() -> List[str]:
    """What makes a reproduction verdict honest, for the run that issues one."""
    return [
        "A measurement can fail to reproduce a claim, or fail to be comparable "
        "to it, and these are different findings. A 3.0x claimed on one "
        "machine at one input size and a 2.1x measured on another at a "
        "different size have not refuted each other -- they never tested each "
        "other, and reporting 'not reproduced' claims a result the run did not "
        "earn.",
        "State the conditions on both sides: hardware, input size, and what "
        "the baseline was. Without them the comparison cannot tell a real "
        "disagreement from a change of machine, and defaults to trusting the "
        "numbers.",
        "Absolute quantities -- milliseconds, cycles, GB/s -- do not travel "
        "between machines. Only ratios do, and only approximately. An absolute "
        "claim measured on different hardware agrees or disagrees by luck.",
        "Extract the paper's claimed number BEFORE implementing anything. A "
        "target written down after the measurement is known is not a target, "
        "and the tolerance will quietly grow to fit whatever was measured.",
        "A verdict of 'reproduced' says the number matched. It does not say "
        "the thing measured was the paper's algorithm; that is what the "
        "correctness check establishes, and without it a matching number is a "
        "coincidence between two unknown quantities.",
    ]
