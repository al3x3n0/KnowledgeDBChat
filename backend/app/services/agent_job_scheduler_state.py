"""Scheduler-state helpers for recurring agent jobs.

Extracted from ``api/endpoints/agent_jobs.py`` so that endpoints, Celery
tasks, and services can share this logic without importing endpoint
internals. Pure functions — no DB or request coupling.
"""

from __future__ import annotations

from typing import Optional

from app.models.agent_job import AgentJob


def extract_scheduler_state(job: Optional[AgentJob]) -> Optional[dict]:
    """Extract compact recurring-run scheduler metadata for queueing and UI."""
    if job is None:
        return None
    results = job.results if isinstance(job.results, dict) else {}
    execution = (
        results.get("execution_strategy")
        if isinstance(results.get("execution_strategy"), dict)
        else {}
    )
    raw = (
        execution.get("scheduler_state")
        if isinstance(execution.get("scheduler_state"), dict)
        else {}
    )
    if not raw:
        return None
    return {
        "last_run_status": str(raw.get("last_run_status") or "").strip() or None,
        "failure_streak": max(0, int(raw.get("failure_streak", 0) or 0)),
        "last_scheduled_at": raw.get("last_scheduled_at"),
        "last_dispatched_at": raw.get("last_dispatched_at"),
        "current_run_started_at": raw.get("current_run_started_at"),
        "last_successful_run_at": raw.get("last_successful_run_at"),
        "last_completed_run_at": raw.get("last_completed_run_at"),
        "last_failure_at": raw.get("last_failure_at"),
        "backoff_until": raw.get("backoff_until"),
        "backoff_seconds": max(0, int(raw.get("backoff_seconds", 0) or 0)),
        "queue_reason": str(raw.get("queue_reason") or "").strip() or None,
    }


def queue_reason_label(reason_code: str) -> str:
    """Human-readable label for a queue reason code."""
    code = str(reason_code or "").strip().lower()
    labels = {
        "approval_required": "Approval required",
        "execution_failure": "Execution failure",
        "stalled_run": "Stalled run",
        "scheduled_recovery": "Scheduled recovery",
        "accepted_inbox_item": "Accepted inbox signal",
        "follow_up_launch_approval": "Follow-up launch approval",
        "follow_up_blocked": "Follow-up blocked by policy",
        "follow_up_launch_failed": "Follow-up launch failed",
        "policy_guardrail": "Policy safeguard review",
        "budget_throttle": "Autonomy budget review",
    }
    # Fallback matches the sentence-case convention of the labels above
    # (e.g. "Execution failure", not "Execution Failure").
    return labels.get(
        code, code.replace("_", " ").strip().capitalize() or "Needs review"
    )
