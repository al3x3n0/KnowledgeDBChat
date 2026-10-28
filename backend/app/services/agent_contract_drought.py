"""Telling a run that it has stopped making progress on its contract.

The first stage of the first plan a run proposed for itself spent all ten of
its iterations orienting -- reading findings, the spec, cloning a repository,
browsing it -- and ended with its contract unmet and no file written. It was
never stalled by the loop policy's measure: cloning and browsing record
findings, so the count of findings grew every round. What did not grow was
anything the contract asked for, and nothing said so.

The prompt already lists what the contract still owes. This adds the part a
list cannot: how long it has been since the run produced any of it, and which
tool would, so a run that is exploring learns that exploring is not working.

`agent_unproductive_cycle` catches the other half: a streak of calls to tools
that cannot produce evidence at all (progress reports, recalls). Cloning and
browsing do produce evidence -- `repo_workspace`, documents -- so that check
rightly lets them through; this one asks whether any of it is evidence the
contract wants.

Progress is the contract's missing list getting shorter, or the count of a
required finding type going up (``benchmark_measurement>=2`` reads the same at
one of two as at none). Anything else -- findings of other types, reports,
reads -- is not progress on the contract, which is the point.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

#: Iterations without contract progress before the run is told.
DROUGHT_ITERATIONS = 3

STATE_KEY = "contract_progress"


def _required_types(missing: List[str]) -> List[str]:
    """Finding or artifact types named by `<label>:<type>[>=N]` entries."""
    out: List[str] = []
    for item in missing:
        if item.startswith(("validity:", "result_key:")) or ":" not in item:
            continue
        name = item.split(":", 1)[1].split(">=", 1)[0].strip()
        if name and name not in out:
            out.append(name)
    return out


def _snapshot(contract_eval: Mapping[str, Any]) -> Dict[str, Any]:
    missing = sorted(str(m) for m in (contract_eval.get("missing") or []) if m)
    metrics = contract_eval.get("metrics")
    have = metrics.get("finding_types") if isinstance(metrics, Mapping) else None
    have = have if isinstance(have, Mapping) else {}
    counts = {t: int(have.get(t, 0) or 0) for t in _required_types(missing)}
    return {"missing": missing, "counts": counts}


def _progressed(before: Mapping[str, Any], now: Mapping[str, Any]) -> bool:
    if set(now["missing"]) < set(before.get("missing") or []):
        return True
    old = before.get("counts") if isinstance(before.get("counts"), Mapping) else {}
    return any(n > int(old.get(t, 0) or 0) for t, n in now["counts"].items())


def record(
    state: Dict[str, Any], contract_eval: Optional[Mapping[str, Any]], iteration: int
) -> None:
    """Note this iteration's contract state; remember when it last improved."""
    contract_eval = contract_eval if isinstance(contract_eval, Mapping) else {}
    if not contract_eval.get("enabled") or contract_eval.get("satisfied"):
        state.pop(STATE_KEY, None)
        return
    now = _snapshot(contract_eval)
    before = state.get(STATE_KEY)
    if not isinstance(before, dict) or _progressed(before, now):
        state[STATE_KEY] = {**now, "iteration": int(iteration or 0)}
    else:
        # Keep the iteration of the last improvement; refresh the rest so a
        # requirement that disappears and returns is judged against the latest.
        state[STATE_KEY] = {**now, "iteration": int(before.get("iteration") or 0)}


def iterations_without_progress(state: Mapping[str, Any], iteration: int) -> int:
    entry = state.get(STATE_KEY)
    if not isinstance(entry, Mapping):
        return 0
    return max(0, int(iteration or 0) - int(entry.get("iteration") or 0))


def format_drought(
    state: Mapping[str, Any], *, iteration: int, job_type: Optional[str]
) -> str:
    """The note for the prompt, or "" while the run is getting somewhere."""
    dry = iterations_without_progress(state, iteration)
    entry = state.get(STATE_KEY)
    if dry < DROUGHT_ITERATIONS or not isinstance(entry, Mapping):
        return ""
    missing = list(entry.get("missing") or [])
    if not missing:
        return ""

    from app.services import agent_evidence_map

    how = agent_evidence_map.describe_chain(
        _required_types(missing), job_type=job_type or "research"
    )
    text = (
        f"NO PROGRESS ON THE CONTRACT FOR {dry} ITERATIONS. None of the last "
        f"{dry} actions produced what it still requires "
        f"({', '.join(missing[:6])}). More reading, searching or browsing will "
        "not satisfy it: make your next action one that produces the missing "
        "evidence."
    )
    if how:
        text += " " + " ".join(how[:4])
    text += (
        " If something prevents that, record the obstacle as a finding and "
        "say so, rather than exploring further."
    )
    return text


__all__ = [
    "DROUGHT_ITERATIONS",
    "STATE_KEY",
    "format_drought",
    "iterations_without_progress",
    "record",
]
