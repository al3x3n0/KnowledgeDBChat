"""Take a nondeterministic measurement more than once, and say whether it held.

The evidence for this is direct. Re-taking this project's per-instruction
latencies produced, across independent runs of the same harness on a host whose
controls were passing:

* latencies agreeing to **within 5%** and landing on integers, and
* reciprocal throughputs disagreeing by **12% to 55%**, with only 1 of 9 inside
  the same band.

Both came out of one run each originally, and both looked equally credible.
Nothing in the numbers said which was which -- only taking them again did.

**This is not the same as `repeat`.** `benchmark_c_snippet` already runs several
trials inside one process and reports `all_ms` and `trial_spread`, which catches
scheduling noise during a single program. What varied here was *between*
separate calls: each run's internal spread looked fine while its answer moved by
half. A statistic computed inside one process cannot see the drift the process
sat in.

**What replication does not catch, and must not be read as catching.** It
measures variance, not bias. This project's worst defect -- chains that reached
infinity within a few iterations, so the harness timed exceptional-value
arithmetic instead of the instructions named -- reproduced *perfectly*, run
after run, because it was wrong the same way every time. Controls catch a broken
instrument, replication catches a noisy one, and neither catches measuring the
wrong thing. That third gap is still open and is not closed by this module.

Only tools whose answers genuinely move are replicated. Callgrind returns exact
dynamic instruction counts, llvm-mca is a static model, and gem5 is
deterministic; running any of them three times is three times the cost for the
same number, so they are absent from the registry on purpose.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

#: Fractional spread a metric may show across independent calls and still be
#: called reproducible. 0.05 is where the re-take separated cleanly: latencies
#: inside it, throughputs far outside.
DEFAULT_BAND = 0.05

#: How many independent calls. Three is the smallest number that has a median
#: and can outvote a single disturbed run -- which is exactly what happened when
#: a third measurement run drifted and the other two agreed.
DEFAULT_RUNS = 3


@dataclass(frozen=True)
class Replicated:
    """A tool whose answer is expected to move between identical calls."""

    tool: str
    runs: int
    band: float
    #: Where the numbers to compare live in the result. Metrics the program
    #: printed about itself, rather than the harness's own timings, because a
    #: harness that prints `ns_per_op=` has already done the arithmetic that
    #: makes its timing mean something.
    metrics_path: Sequence[str]
    why: str


REPLICATED: Dict[str, Replicated] = {
    "benchmark_c_snippet": Replicated(
        tool="benchmark_c_snippet",
        runs=DEFAULT_RUNS,
        band=DEFAULT_BAND,
        metrics_path=("data", "reported_metrics"),
        why=(
            "wall clock on a shared host. Its own trial spread is computed "
            "inside one process and cannot see the drift that process sat in: "
            "re-taking this project's throughputs moved them 12-55% while each "
            "run's internal spread looked fine."
        ),
    ),
}


def is_replicated(tool: str) -> bool:
    return str(tool or "") in REPLICATED


def replicated_tools() -> List[str]:
    return sorted(REPLICATED)


def spec_for(tool: str) -> Optional[Replicated]:
    return REPLICATED.get(str(tool or ""))


def _dig(result: Any, path: Sequence[str]) -> Any:
    node: Any = result
    for key in path:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def extract_metrics(result: Any, path: Sequence[str]) -> Dict[str, float]:
    """One number per metric from one call.

    A program that printed a metric several times in one run has already been
    summarised by the sandbox into a list; the median of that list is this
    call's answer, and the comparison across calls happens above it.
    """
    raw = _dig(result, path)
    if not isinstance(raw, Mapping):
        return {}
    out: Dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)):
            out[str(key)] = float(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            numbers = [float(v) for v in value if isinstance(v, (int, float))]
            if numbers:
                out[str(key)] = statistics.median(numbers)
    return out


def spread(values: Sequence[float]) -> Optional[float]:
    """Fractional spread of a metric across calls.

    Full range over the median rather than a standard deviation: with three
    samples a standard deviation is barely a statistic, and the question here
    is "did any run disagree", which the range answers directly.
    """
    numbers = [v for v in values if isinstance(v, (int, float))]
    if len(numbers) < 2:
        return None
    middle = statistics.median(numbers)
    if middle == 0:
        return None if max(numbers) == min(numbers) else float("inf")
    return abs(max(numbers) - min(numbers)) / abs(middle)


def judge(
    per_run_metrics: Sequence[Mapping[str, float]], band: float = DEFAULT_BAND
) -> Dict[str, Any]:
    """Which of this measurement's numbers held up across the calls."""
    runs = len(per_run_metrics)
    keys = sorted({k for run in per_run_metrics for k in run})

    metrics: Dict[str, Any] = {}
    for key in keys:
        values = [run[key] for run in per_run_metrics if key in run]
        observed = spread(values)
        # A metric only some runs reported did not reproduce: it was absent
        # from a run, which is a disagreement about whether it exists.
        complete = len(values) == runs
        reproducible = bool(complete and observed is not None and observed <= band)
        metrics[key] = {
            "values": [round(v, 6) for v in values],
            "median": round(statistics.median(values), 6) if values else None,
            "spread": None if observed is None else round(observed, 4),
            "reported_by_all_runs": complete,
            "reproducible": reproducible,
        }

    reproduced = [k for k, m in metrics.items() if m["reproducible"]]
    failed = [k for k, m in metrics.items() if not m["reproducible"]]

    return {
        "runs": runs,
        "band": band,
        "metrics": metrics,
        "reproduced": reproduced,
        "not_reproduced": failed,
        # No metrics at all is not agreement. A program that printed nothing
        # gives nothing to reproduce, and calling that success is the same
        # mistake as calling a skipped control a pass.
        "all_reproduced": bool(metrics) and not failed,
        "any_metrics": bool(metrics),
        # Without this a call that printed nothing reports all_reproduced
        # false with an empty not_reproduced list, which reads as "nothing
        # failed" -- seen in a live run and genuinely confusing.
        "note": (
            ""
            if metrics
            else (
                "the program printed no key=value metrics, so there was "
                "nothing to reproduce; this is not agreement"
            )
        ),
    }


async def run_replicated(
    call: Callable[[], Any],
    tool: str,
    *,
    runs: Optional[int] = None,
    band: Optional[float] = None,
) -> Dict[str, Any]:
    """Call `call()` several times and report what agreed.

    Returns the run whose metrics sit at the median, with a `replication` block
    attached. The median run rather than the first: if one call is disturbed,
    the answer returned should be the representative one, and with three calls
    the median is exactly the one the other two outvote.

    A failed call is kept rather than retried. Two successes and a failure is a
    different situation from three successes, and hiding the failure would make
    them look identical.
    """
    spec = spec_for(tool)
    total = int(runs or (spec.runs if spec else DEFAULT_RUNS))
    width = float(band if band is not None else (spec.band if spec else DEFAULT_BAND))
    path = spec.metrics_path if spec else ("data", "reported_metrics")

    results: List[Any] = []
    for _ in range(max(1, total)):
        results.append(await call())

    successful = [r for r in results if isinstance(r, Mapping) and r.get("success")]
    per_run = [extract_metrics(r, path) for r in successful]
    verdict = judge(per_run, band=width)
    verdict["calls"] = len(results)
    verdict["failed_calls"] = len(results) - len(successful)

    if not successful:
        chosen = results[0] if results else {"success": False, "error": "no calls made"}
        return _attach(chosen, verdict)

    chosen = successful[0]
    if verdict["metrics"]:
        # Order by the first metric every run reported, so "the median run" is
        # a real run and not a synthesis of several.
        shared = [k for k, m in verdict["metrics"].items() if m["reported_by_all_runs"]]
        if shared:
            key = shared[0]
            ordered = sorted(
                zip(per_run, successful), key=lambda pair: pair[0].get(key, 0.0)
            )
            chosen = ordered[len(ordered) // 2][1]

    return _attach(chosen, verdict)


def _attach(result: Any, verdict: Mapping[str, Any]) -> Any:
    if not isinstance(result, Mapping):
        return result
    enriched = dict(result)
    data = (
        dict(enriched.get("data") or {})
        if isinstance(enriched.get("data"), Mapping)
        else {}
    )
    data["replication"] = dict(verdict)
    enriched["data"] = data

    # Onto the findings too, not only the tool result: a contract reads
    # findings, and a reproduction record that stops at the result cannot be
    # checked by one.
    findings = enriched.get("findings")
    if isinstance(findings, Sequence) and not isinstance(findings, (str, bytes)):
        enriched["findings"] = [
            {**f, "replication": dict(verdict)} if isinstance(f, Mapping) else f
            for f in findings
        ]
    return enriched


# --- reading the run's own record -----------------------------------------


def unreproduced_measurements(state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Findings in this run whose numbers did not hold up across calls."""
    findings = state.get("findings") if isinstance(state, Mapping) else None
    if not isinstance(findings, Sequence):
        return []

    out: List[Dict[str, Any]] = []
    for finding in findings:
        if not isinstance(finding, Mapping):
            continue
        record = finding.get("replication")
        if not isinstance(record, Mapping):
            continue
        if record.get("all_reproduced"):
            continue
        out.append(
            {
                "title": str(finding.get("title") or finding.get("type") or "")[:120],
                "runs": record.get("runs"),
                "not_reproduced": list(record.get("not_reproduced") or [])[:5],
                "band": record.get("band"),
                "detail": {
                    k: v
                    for k, v in (record.get("metrics") or {}).items()
                    if not v.get("reproducible")
                },
            }
        )
    return out


def describe() -> List[str]:
    return [
        f"{spec.tool}: taken {spec.runs} times independently; a number whose "
        f"spread across them exceeds {spec.band:.0%} is not reported as a "
        f"result ({spec.why})"
        for spec in (REPLICATED[t] for t in replicated_tools())
    ]
