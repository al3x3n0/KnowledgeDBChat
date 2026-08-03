"""Append bounded, structured operator interventions to job results."""

from datetime import datetime
from typing import Any
from uuid import UUID


def append_operator_intervention(
    results_payload: dict,
    *,
    action: str,
    actor_user_id: UUID | str,
    note: str | None = None,
    job_status_before: str | None = None,
    job_status_after: str | None = None,
    metadata: dict[str, Any] | None = None,
    max_events: int = 200,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Append one intervention and retain a bounded recent history."""
    execution = (
        results_payload.get("execution_strategy")
        if isinstance(results_payload.get("execution_strategy"), dict)
        else {}
    )
    rows = (
        execution.get("operator_interventions")
        if isinstance(execution.get("operator_interventions"), list)
        else []
    )
    row: dict[str, Any] = {
        "action": str(action or "").strip()[:80] or "unknown",
        "actor_user_id": str(actor_user_id or "").strip() or None,
        "at": (now or datetime.utcnow()).isoformat(),
    }
    if note:
        row["note"] = str(note).strip()[:1000]
    if job_status_before:
        row["job_status_before"] = str(job_status_before).strip()[:40]
    if job_status_after:
        row["job_status_after"] = str(job_status_after).strip()[:40]
    if isinstance(metadata, dict) and metadata:
        row["metadata"] = metadata

    rows.append(row)
    history_limit = max(20, min(int(max_events or 200), 1000))
    execution["operator_interventions"] = rows[-history_limit:]
    results_payload["execution_strategy"] = execution
    return row
