from __future__ import annotations

from datetime import datetime
from typing import Any, Optional


def derive_operator_interventions_with_outcomes(
    rows: list[dict[str, Any]] | None,
    *,
    current_status: Optional[str] = None,
    completed_at: Optional[datetime] = None,
    status_values: Optional[dict[str, str]] = None,
) -> list[dict[str, Any]]:
    """Project intervention rows with lightweight derived outcome state."""
    items = [dict(row) for row in (rows or []) if isinstance(row, dict)]
    if not items:
        return []

    status_values = status_values or {}
    completed = str(status_values.get("completed") or "completed").strip().lower()
    failed = str(status_values.get("failed") or "failed").strip().lower()
    cancelled = str(status_values.get("cancelled") or "cancelled").strip().lower()
    pending = str(status_values.get("pending") or "pending").strip().lower()
    running = str(status_values.get("running") or "running").strip().lower()
    paused = str(status_values.get("paused") or "paused").strip().lower()

    terminal_statuses = {completed, failed, cancelled}
    active_statuses = {pending, running, paused}
    current_status_norm = str(current_status or "").strip().lower() or None
    resolved_at_value = completed_at.isoformat() if isinstance(completed_at, datetime) else None

    for index, item in enumerate(items):
        next_item = items[index + 1] if index + 1 < len(items) else None
        if next_item:
            next_action = str(next_item.get("action") or "").strip().replace("_", " ") or "later action"
            item["outcome_status"] = "superseded"
            item["outcome_reason"] = f"Superseded by {next_action}"
            item["resolved_at"] = str(next_item.get("at") or "").strip() or None
            continue

        action = str(item.get("action") or "").strip().lower()
        if current_status_norm in terminal_statuses:
            if current_status_norm == completed:
                item["outcome_status"] = "resolved"
                item["outcome_reason"] = "Job completed after intervention"
            elif current_status_norm == failed:
                item["outcome_status"] = "unresolved"
                item["outcome_reason"] = "Job failed after intervention"
            else:
                item["outcome_status"] = "cancelled"
                item["outcome_reason"] = "Job was cancelled after intervention"
            item["resolved_at"] = resolved_at_value or str(item.get("at") or "").strip() or None
            continue

        if current_status_norm in active_statuses:
            if action in {"pause", "reject"} and current_status_norm == paused:
                item["outcome_status"] = "applied"
                item["outcome_reason"] = "Job remains paused after intervention"
            elif action in {"approve", "edit", "skip", "restart", "relaunch"} and current_status_norm in {
                pending,
                running,
            }:
                item["outcome_status"] = "applied"
                item["outcome_reason"] = "Job resumed after intervention"
            elif action == "cancel":
                item["outcome_status"] = "pending"
                item["outcome_reason"] = "Cancellation requested; awaiting terminal sync"
            else:
                item["outcome_status"] = "pending"
                item["outcome_reason"] = "Awaiting job outcome"
            item["resolved_at"] = None
            continue

        item["outcome_status"] = "unknown"
        item["outcome_reason"] = "Outcome could not be determined"
        item["resolved_at"] = None

    return items
