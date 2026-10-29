"""Five numbers about a finished run, read from what it recorded.

The weak spots worked on in October 2026 -- a stage that explores without
producing its evidence, a benchmark with no baseline, a conclusion that came
back empty -- were each found by reading one live run by hand. Whether a fix
helped is a question about many runs, so it needs numbers a report can
average and two reports can diff:

* **contract** -- did every stage that had a contract meet it;
* **benchmark** -- was a measurement taken beside a baseline, and did a
  paired comparison claim a difference;
* **conclusion** -- did the run end with an answer, or only with its findings
  listed (`generated_by` says which);
* **iterations to first evidence** -- how long the run explored before a
  tool that produces what it was asked for succeeded;
* **wall time**, from the run profile.

A run is the job and every job sharing its root, so a proposed plan's stages
and a stage's retries are scored with the job that started them. Retried
attempts are not judged (their retry is) but their iterations count, since
they were spent. Findings inherited from an earlier stage are counted once,
where they were made.

This reads the rows and never asks a model; it is the instrument, and it
must not need a second opinion.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from app.services import agent_evidence_map, agent_run_profile
from app.services.agent_loop_policy import finding_identity
from app.services.agent_pipeline_spec import _required_types
from app.services.agent_run_waiting import WAITING_ON_PERSON, final_jobs, waiting_on
from app.utils.datetimes import as_aware_utc

DECISIVE_VERDICTS = ("faster", "slower")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _results(job: Any) -> Dict[str, Any]:
    results = getattr(job, "results", None)
    return results if isinstance(results, dict) else {}


def _created(job: Any) -> Optional[datetime]:
    return as_aware_utc(getattr(job, "created_at", None))


def _in_order(run: Sequence[Any]) -> List[Any]:
    return sorted(run, key=lambda job: _created(job) or _EPOCH)


def _contract(final: Sequence[Any]) -> Dict[str, Any]:
    stages = []
    for job in final:
        block = _results(job).get("goal_contract")
        if not isinstance(block, Mapping) or not block.get("enabled"):
            continue
        stages.append(
            {
                "job_id": str(job.id),
                "satisfied": bool(block.get("satisfied")),
                "missing": [str(m) for m in (block.get("missing") or [])][:10],
                "required": _required_types(block.get("contract") or {}),
            }
        )
    met = sum(1 for stage in stages if stage["satisfied"])
    return {
        # None, not True, when nothing had a contract: "met" would be a pass
        # earned by asking for nothing.
        "met": (met == len(stages)) if stages else None,
        "stages": len(stages),
        "stages_met": met,
        "missing": sorted(
            {m for stage in stages if not stage["satisfied"] for m in stage["missing"]}
        ),
        "_required": sorted({t for stage in stages for t in stage["required"]}),
    }


def _findings(run: Sequence[Any]) -> List[Dict[str, Any]]:
    unique: Dict[Any, Dict[str, Any]] = {}
    for job in _in_order(run):
        raw = _results(job).get("findings")
        for finding in raw if isinstance(raw, list) else []:
            if isinstance(finding, dict) and finding.get("inherited") is not True:
                unique.setdefault(finding_identity(finding), finding)
    return list(unique.values())


def _benchmark(findings: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    measurements = [f for f in findings if f.get("type") == "benchmark_measurement"]
    comparisons = [f for f in findings if f.get("type") == "benchmark_comparison"]
    subjects = sorted({str(f.get("subject") or "") for f in measurements} - {""})
    verdicts = [str(f.get("verdict") or "") for f in comparisons]
    return {
        "measurements": len(measurements),
        "subjects": subjects,
        # None when nothing was timed: a run that benchmarked nothing neither
        # kept nor dropped a baseline.
        "baselined": (len(subjects) >= 2 or bool(comparisons))
        if measurements or comparisons
        else None,
        "paired": bool(comparisons),
        "verdicts": verdicts,
        "claimed_difference": (
            any(v in DECISIVE_VERDICTS for v in verdicts) if comparisons else None
        ),
    }


def _conclusion(final: Sequence[Any]) -> Dict[str, Any]:
    """The conclusion of the job that ended last: a plan's last stage, not
    the job that proposed it."""
    for job in sorted(
        final,
        key=lambda j: as_aware_utc(getattr(j, "completed_at", None))
        or _created(j)
        or _EPOCH,
        reverse=True,
    ):
        conclusion = _results(job).get("conclusion")
        if isinstance(conclusion, Mapping):
            return {
                "answered": bool(str(conclusion.get("answer") or "").strip()),
                "generated_by": conclusion.get("generated_by"),
                "confidence": conclusion.get("confidence"),
                "job_id": str(job.id),
            }
    return {"answered": False, "generated_by": None, "confidence": None}


def _iterations(run: Sequence[Any], wanted: Set[str]) -> Dict[str, Any]:
    """Iterations spent, and how many before evidence was first produced.

    Findings do not record the iteration that made them, so "first evidence"
    is the first successful call to a tool declared to produce a type the
    run was asked for (any evidence tool when nothing was asked for).
    """
    total = 0
    first: Optional[int] = None
    first_tool: Optional[str] = None
    for job in _in_order(run):
        for entry in getattr(job, "execution_log", None) or []:
            if (
                not isinstance(entry, dict)
                or entry.get("phase") != "iteration_complete"
            ):
                continue
            total += 1
            if first is not None or entry.get("success") is not True:
                continue
            evidence = agent_evidence_map.entry_for(str(entry.get("action") or ""))
            if evidence is None:
                continue
            if not wanted or wanted & set(evidence.produces):
                first, first_tool = total, evidence.tool
    return {
        "iterations": total,
        "iterations_to_first_evidence": first,
        "first_evidence_tool": first_tool,
    }


def score(run: Sequence[Any], *, now: Optional[datetime] = None) -> Dict[str, Any]:
    """The scorecard of one run: the job and every job sharing its root."""
    run = list(run)
    if not run:
        return {"jobs": 0, "settled": True}
    final = final_jobs(run)
    waiting = waiting_on(run)
    contract = _contract(final)
    wanted = set(contract.pop("_required"))
    root = run[0].root_job_id or run[0].id
    profile = agent_run_profile.build(run, root_id=root, now=now)
    return {
        "jobs": len(run),
        "superseded_attempts": len(run) - len(final),
        # Paused for a person is an outcome when nobody is watching; waiting
        # on a job or on a plan's approval is not over yet.
        "settled": waiting in (None, WAITING_ON_PERSON),
        "waiting_on": waiting,
        "contract": contract,
        "contract_met": contract["met"],
        "benchmark": _benchmark(_findings(run)),
        "conclusion": _conclusion(final),
        **_iterations(run, wanted),
        "wall_seconds": profile.get("wall_seconds"),
    }


#: The scorecard fields an eval report averages across trials.
RATE_FIELDS = (
    ("contract_met", ("contract_met",)),
    ("baselined", ("benchmark", "baselined")),
    ("conclusion_answered", ("conclusion", "answered")),
)
MEDIAN_FIELDS = (
    ("iterations_to_first_evidence", ("iterations_to_first_evidence",)),
    ("iterations", ("iterations",)),
    ("wall_seconds", ("wall_seconds",)),
)


def _at(card: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = card
    for part in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def summarize(cards: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Rates and medians over scorecards, each with how many trials it covers.

    A rate is over the trials where the question applies (a run that timed
    nothing is not counted for or against `baselined`), so each carries its
    own `n`. A trial that never reached its first evidence is counted in
    `never_produced_evidence` rather than given a number.
    """
    cards = [card for card in cards if isinstance(card, Mapping) and card.get("jobs")]
    out: Dict[str, Any] = {"trials": len(cards)}
    for name, path in RATE_FIELDS:
        values = [_at(card, path) for card in cards]
        known = [bool(v) for v in values if v is not None]
        out[name] = {
            "rate": round(sum(known) / len(known), 4) if known else None,
            "n": len(known),
        }
    for name, path in MEDIAN_FIELDS:
        values = [_at(card, path) for card in cards]
        numbers = [float(v) for v in values if isinstance(v, (int, float))]
        out[name] = {"median": _median(numbers), "n": len(numbers)}
    out["never_produced_evidence"] = sum(
        1 for card in cards if card.get("iterations_to_first_evidence") is None
    )
    return out


__all__ = ["DECISIVE_VERDICTS", "score", "summarize"]
