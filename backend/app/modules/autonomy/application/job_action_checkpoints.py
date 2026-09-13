"""Normalize and persist operator decisions at agent-job checkpoints."""

import re
from datetime import datetime
from typing import Any
from uuid import UUID

from app.services.agent_scope_service import normalize_scope_keys_deep


def approval_payload_from_results(
    results: dict | None,
) -> tuple[dict, dict, dict | None]:
    """Resolve the canonical approval payload and pending checkpoint."""
    payload = results if isinstance(results, dict) else {}
    execution = (
        payload.get("execution_strategy")
        if isinstance(payload.get("execution_strategy"), dict)
        else {}
    )
    approval = (
        execution.get("approval_checkpoints")
        if isinstance(execution.get("approval_checkpoints"), dict)
        else {}
    )
    pending = (
        approval.get("pending")
        if isinstance(approval.get("pending"), dict)
        else (
            payload.get("approval_checkpoint")
            if isinstance(payload.get("approval_checkpoint"), dict)
            else None
        )
    )
    return payload, approval, pending


def normalize_checkpoint_action_patch(patch: Any) -> dict[str, Any]:
    """Validate and normalize the editable checkpoint action fields."""
    if not isinstance(patch, dict):
        return {}

    out: dict[str, Any] = {}
    if "tool" in patch:
        tool = str(patch.get("tool") or "").strip()
        if not tool:
            raise ValueError("checkpoint_action_patch.tool cannot be empty")
        if not re.match(r"^[a-zA-Z0-9_:\\-]{2,80}$", tool):
            raise ValueError("checkpoint_action_patch.tool is invalid")
        out["tool"] = tool

    if "purpose" in patch:
        out["purpose"] = str(patch.get("purpose") or "").strip()[:220]

    if "params" in patch:
        params = patch.get("params")
        if params is None:
            out["params"] = {}
        elif isinstance(params, dict):
            out["params"] = normalize_scope_keys_deep(params)
        else:
            raise ValueError("checkpoint_action_patch.params must be an object")

    return out


def apply_checkpoint_action_patch(
    pending_checkpoint: dict,
    patch: dict[str, Any],
) -> dict[str, Any]:
    """Merge an operator patch into the pending action."""
    action_payload = (
        pending_checkpoint.get("action")
        if isinstance(pending_checkpoint.get("action"), dict)
        else {}
    )
    merged = dict(action_payload)
    if "tool" in patch:
        merged["tool"] = patch["tool"]
    if "purpose" in patch:
        merged["purpose"] = patch["purpose"]
    if "params" in patch:
        merged["params"] = patch["params"]
    if "tool" in patch or "params" in patch:
        merged.pop("_idempotency_key", None)

    pending_checkpoint["action"] = merged
    pending_checkpoint["updated_at"] = datetime.utcnow().isoformat()
    return merged


def append_approval_event(
    approval: dict,
    pending_checkpoint: dict,
    *,
    method: str,
    user_id: UUID,
    note: str | None = None,
    edited_action: dict[str, Any] | None = None,
) -> None:
    """Append one bounded approval or rejection audit event."""
    event_key = "rejections" if method == "reject_action" else "approvals"
    events = (
        approval.get(event_key) if isinstance(approval.get(event_key), list) else []
    )
    event = {
        "at": datetime.utcnow().isoformat(),
        "approved_by": str(user_id),
        "method": method,
        "checkpoint": {
            "iteration": int(pending_checkpoint.get("iteration", 0) or 0),
            "action_tool": str(
                ((pending_checkpoint.get("action") or {}).get("tool") or "")
            ).strip(),
            "plan_step_id": str(pending_checkpoint.get("plan_step_id") or "").strip()
            or None,
            "plan_step_index": int(pending_checkpoint.get("plan_step_index", -1) or -1),
        },
    }
    if note:
        event["note"] = str(note)[:1000]
    if isinstance(edited_action, dict) and edited_action:
        event["edited_action"] = edited_action
    events.append(event)
    approval[event_key] = events[-50:]


def append_step_event(
    state: dict | None,
    event: dict[str, Any],
    *,
    max_events: int = 500,
) -> None:
    """Append a bounded execution step event to checkpoint state."""
    payload = state if isinstance(state, dict) else {}
    if not isinstance(event, dict):
        return
    rows = (
        payload.get("step_events")
        if isinstance(payload.get("step_events"), list)
        else []
    )
    row = dict(event)
    row.setdefault("at", datetime.utcnow().isoformat())
    rows.append(row)
    payload["step_events"] = rows[-max(20, min(int(max_events or 500), 5000)) :]


def sync_execution_strategy_state(
    results_payload: dict,
    *,
    approval_payload: dict | None = None,
    state: dict | None = None,
) -> dict:
    """Synchronize approval and step-event state into job results."""
    execution = (
        results_payload.get("execution_strategy")
        if isinstance(results_payload.get("execution_strategy"), dict)
        else {}
    )
    if isinstance(approval_payload, dict):
        execution["approval_checkpoints"] = approval_payload
    if isinstance(state, dict):
        execution["step_events"] = (
            state.get("step_events")
            if isinstance(state.get("step_events"), list)
            else []
        )[-300:]
    results_payload["execution_strategy"] = execution
    return execution
