"""Findings measured across a break in the trace they were measured on.

A counter trace that changes regime part way through is two experiments. The
intervals before the break describe a machine that does not recur -- a
co-runner that has not finished initialising, a cache still cold, a phase the
workload enters once and leaves -- and a number taken across it largely
measures the break.

This is not hypothetical and it is not small. On the SMT trace this project
ran, persistence about thread-0 IPC read 0.843 bits across the break and 0.405
after it: more than half of what was reported as predictable structure was the
trace announcing that solo intervals stay solo. The tap was unaffected, so the
error was invisible in the number everyone was watching.

`sample_hardware_counters` detects the break and says so in a warning. A
warning is a thing a run may read and proceed past, which is why this exists:
the tools stamp each finding with the window it analysed and the break it knew
about, and a contract that declares `traces_one_regime` will not let a run
conclude on a finding that straddles one.

**The remedy has to exist before the requirement does.** Each of the three
tools takes `from_interval`, so a run told about a break at 104 can measure
from 104 and satisfy this. A check whose only compliant answer is "do not
measure" is not a check, it is a wall -- this codebase has shipped one of those
before, in a contract asking for error bars that the only tool reporting them
could not express.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

#: Finding types derived from a counter trace, and therefore exposed to a break
#: in one. A type absent from here is not checked -- a finding that never
#: touched a trace cannot straddle anything.
TRACE_DERIVED_TYPES = (
    "predictability_ceiling",
    "counter_tap_selection",
    "predictor_design_result",
)


def window(data: Mapping[str, Any], from_interval: Any = 0) -> Tuple[Dict, Dict]:
    """The slice of a trace a caller asked for, and what it straddles.

    Returns the windowed series and the stamp that travels onto the finding.
    Stamping is what makes the choice checkable afterwards: the run that took
    the number knew both the break and the window it chose, and a check that
    re-derived either would be judging a different trace than the one the
    number came from.
    """
    raw = data.get("series") if isinstance(data.get("series"), dict) else {}
    series = {name: values for name, values in raw.items() if isinstance(values, list)}
    total = max((len(v) for v in series.values()), default=0)

    regime = (
        data.get("regime_change")
        if isinstance(data.get("regime_change"), dict)
        else None
    )
    at = regime.get("at_interval") if regime else None

    try:
        start = int(from_interval or 0)
    except (TypeError, ValueError):
        start = 0
    start = max(0, min(start, total))

    return (
        {name: list(values)[start:] for name, values in series.items()},
        {
            "analysed_from_interval": start,
            "trace_intervals": total,
            "trace_regime_change_at": at,
            # A break at the very start of the window was not crossed, and one
            # at or past the end is not in the data that was analysed.
            "analysed_across_regime_change": bool(
                at is not None and start < at < total
            ),
        },
    )


def _findings(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    raw = state.get("findings")
    return [f for f in raw if isinstance(f, dict)] if isinstance(raw, list) else []


def findings_across_regime_change(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Trace-derived findings whose analysed window straddles a known break.

    Reads the stamp the tool left rather than recomputing anything: the run
    that produced the finding knew both the break and the window it chose, and
    a check that re-derives either would be judging a different trace than the
    one the number came from.
    """
    offenders = []
    for finding in _findings(state):
        if str(finding.get("type") or "").strip() not in TRACE_DERIVED_TYPES:
            continue
        if not bool(finding.get("analysed_across_regime_change")):
            continue
        offenders.append(
            {
                "title": finding.get("title"),
                "type": finding.get("type"),
                "regime_change_at": finding.get("trace_regime_change_at"),
                "analysed_from_interval": finding.get("analysed_from_interval"),
                "trace_intervals": finding.get("trace_intervals"),
            }
        )
    return offenders


def explain(offenders: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for entry in offenders:
        at = entry.get("regime_change_at")
        lines.append(
            f"\"{entry.get('title')}\" was measured across a regime change at "
            f"interval {at}: the analysed window starts at "
            f"{entry.get('analysed_from_interval')} and runs to "
            f"{entry.get('trace_intervals')}, so it spans two different "
            "machines. The intervals before the break do not recur, and a "
            "number taken across it largely measures the break -- persistence "
            "on this project's SMT trace read 0.843 across one and 0.405 "
            f"after it. Re-run the measurement with from_interval={at} to "
            "study the steady side, or lengthen the run until the opening "
            "regime is a negligible fraction of it."
        )
    return lines


def describe() -> List[str]:
    return [
        "a counter trace that changes regime part way through is two "
        "experiments, and a predictability number taken across the break "
        "largely measures the break rather than the workload",
        "when sample_hardware_counters reports a regime change, pass "
        "from_interval to measure the steady side -- a finding that straddles "
        "the break will not satisfy this contract",
    ]
