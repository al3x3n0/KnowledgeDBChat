"""Checkpoint and summary state for a suspended, journaled sandbox action."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

WAITING_PHASE = "waiting_sandbox"
STATE_KEY = "sandbox_continuation"
RESULTS_KEY = "sandbox_wait"


def run_epoch(job: Any) -> Optional[float]:
    started = getattr(job, "started_at", None)
    if started is None:
        return None
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return started.timestamp()


def waiting(job: Any) -> Optional[dict]:
    results = getattr(job, "results", None)
    results = results if isinstance(results, dict) else {}
    value = results.get(RESULTS_KEY)
    return value if isinstance(value, dict) and value.get("key") else None


def write_wait(job: Any, wait: Optional[dict]) -> None:
    results = dict(job.results) if isinstance(job.results, dict) else {}
    if wait is None:
        results.pop(RESULTS_KEY, None)
    else:
        results[RESULTS_KEY] = wait
    job.results = results


def continuation(job: Any, state: dict) -> Optional[dict]:
    """Only an exact, durably journaled continuation can resume automatically."""
    marker = state.get(STATE_KEY)
    summary = waiting(job)
    if not isinstance(marker, dict) or not summary:
        return None
    if state.get("sandbox_run_started_at") != run_epoch(job):
        return None
    if state.get("approval_checkpoint_pending") or state.get(
        "execution_reconciliation_pending"
    ):
        return None
    if (
        marker.get("key") != summary["key"]
        or marker.get("job_id") != str(job.id)
        or marker.get("iteration") != int(job.iteration or 0)
    ):
        return None
    intent = state.get("execution_journal_pending")
    if marker.get("phase") == "collected":
        completion = state.get("execution_journal_last_completion") or {}
        if intent or completion.get("invocation_id") != marker.get("invocation_id"):
            return None
        return marker if isinstance(marker.get("result"), dict) else None
    if (
        not isinstance(intent, dict)
        or intent.get("idempotency_key") != marker["key"]
        or intent.get("invocation_id") != marker.get("invocation_id")
        or not intent.get("retryable_from_journal")
        or intent.get("contains_redactions")
    ):
        return None
    return marker


def action_matches(marker: dict, action: dict) -> bool:
    saved = marker.get("action") or {}
    return (
        action.get("_idempotency_key") == marker.get("key")
        and action.get("tool") == saved.get("tool")
        and (action.get("params") or {}) == (saved.get("params") or {})
    )


def deadline(job: Any, state: dict) -> float:
    value = state.get("sandbox_deadline_at")
    started = getattr(job, "started_at", None)
    epoch = run_epoch(job)
    if isinstance(value, (float, int)) and state.get("sandbox_run_started_at") == epoch:
        return float(value)
    started = started or datetime.now(timezone.utc)
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    value = (
        started + timedelta(minutes=int(job.max_runtime_minutes or 60))
    ).timestamp()
    state["sandbox_deadline_at"] = value
    state["sandbox_run_started_at"] = epoch
    return value


def suspend(job: Any, state: dict, reference: dict) -> dict:
    pending = state["execution_journal_pending"]
    marker = {
        "version": 1,
        "job_id": str(job.id),
        "iteration": int(job.iteration or 0),
        "key": pending["idempotency_key"],
        "invocation_id": pending["invocation_id"],
        "action": deepcopy(pending["action"]),
        "intent": deepcopy(pending),
        "reference": dict(reference),
        "phase": "waiting",
        "deadline_at": deadline(job, state),
    }
    state[STATE_KEY] = marker
    return marker


def clear(job: Any, state: dict) -> None:
    state.pop(STATE_KEY, None)
    write_wait(job, None)
