"""Noticing that a run has asked the same question twice and got the answer.

`agent_failure_diagnosis` handles the same call failing the same way. This is
its counterpart, and the gap it fills was costing more: a run asked to build on
earlier work spent eight of nine iterations alternating two tools, each call
identical to one it had already made, each returning the same sixteen findings.
Nothing was broken. It simply had no way to notice it already knew.

The note is attached to the result rather than raised as an error, for the
reason the failure version documents: it travels in the history the model
reads, so seeing it costs no tool call.

WHY THIS DOES NOT SERVE A CACHED ANSWER. Repeating a measurement is sometimes
exactly right -- a second trial of a benchmark is a sample, not a duplicate,
and this project's whole calibration story rests on being able to take one.
Short-circuiting would silently turn a re-measurement into a copy of the first,
which is the kind of quiet substitution the rest of this codebase spends its
guards preventing. So the call runs, and the note says only what is true: you
asked this before, and here is when.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Mapping, Optional

#: Params that vary between otherwise identical calls and say nothing about
#: what was asked.
IGNORED_PARAMS = {"label", "reason", "purpose"}

#: How many identical calls before saying anything. The second is a
#: coincidence worth a gentle note; by the third the run is looping.
FIRST_NOTE_AT = 2


def _canonical_params(params: Any) -> str:
    if not isinstance(params, Mapping):
        return ""
    salient = {
        key: value for key, value in params.items() if str(key) not in IGNORED_PARAMS
    }
    try:
        return json.dumps(salient, sort_keys=True, default=str)
    except Exception:  # pragma: no cover - defensive
        return str(sorted(salient))


def signature(tool: str, params: Any) -> str:
    """Identify a call by what was asked, not by what came back."""
    digest = hashlib.sha256(_canonical_params(params).encode("utf-8")).hexdigest()[:16]
    return f"{str(tool or '').strip()}:{digest}"


def _succeeded(result: Any) -> bool:
    if not isinstance(result, Mapping):
        return False
    if result.get("error"):
        return False
    return bool(result.get("success", True))


def prior_successes(
    tool: str, params: Any, state: Optional[Dict[str, Any]]
) -> List[int]:
    """Iterations where this exact call already succeeded, oldest first."""
    if not isinstance(state, dict):
        return []
    actions = state.get("actions_taken")
    if not isinstance(actions, list):
        return []

    wanted = signature(tool, params)
    seen: List[int] = []
    for entry in actions:
        if not isinstance(entry, dict):
            continue
        action = entry.get("action")
        if not isinstance(action, dict):
            continue
        if signature(action.get("tool"), action.get("params")) != wanted:
            continue
        if not _succeeded(entry.get("result")):
            continue
        iteration = entry.get("iteration")
        seen.append(int(iteration) if isinstance(iteration, int) else -1)
    return seen


def analyze(
    action: Optional[Dict[str, Any]],
    result: Any,
    state: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """A note for a call the run has already made and already answered.

    None when this is the first time, when the call failed, or when the
    earlier ones failed -- a retry after a failure is progress, not a loop.
    """
    if not isinstance(action, dict) or not _succeeded(result):
        return None

    tool = str(action.get("tool") or "").strip()
    if not tool:
        return None

    earlier = prior_successes(tool, action.get("params"), state)
    if len(earlier) + 1 < FIRST_NOTE_AT:
        return None

    attempt = len(earlier) + 1
    where = ", ".join(str(i) for i in earlier if i >= 0)
    note = (
        f"This is call {attempt} of {tool} with the same arguments"
        + (f"; it already succeeded at iteration {where}" if where else "")
        + ". The answer is the same one. "
    )
    if attempt >= 3:
        note += (
            "Three identical calls is a loop: the next iteration is about to "
            "be spent re-reading something already in this run's history. "
            "Whatever comes next needs different arguments, a different tool, "
            "or the work the answer was for."
        )
    else:
        note += (
            "If it was asked again to check something changed, nothing did -- "
            "the arguments are identical. Use the answer already in the "
            "history rather than a third call for it."
        )
    return {"attempt": attempt, "earlier_iterations": earlier, "note": note}
