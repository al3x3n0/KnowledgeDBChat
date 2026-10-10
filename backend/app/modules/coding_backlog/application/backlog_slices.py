"""The state of one backlog slice, and of the item around it, as plain data.

Everything here works on the dictionaries an item keeps in its JSON columns
and touches neither the database nor HTTP: which actions a slice allows,
which one is recommended, what the item's summary says, who it is shared with.
These lived in ``api/endpoints/coding_backlog.py``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from app.models.coding_backlog import CodingBacklogItem
from app.services.coding_backlog_items import recompute_portfolio_progress
from app.services.collaboration_service import normalize_collaboration_visibility
from app.services.config_values import uuid_list


def normalize_str_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(v).strip() for v in values if str(v).strip()]


def normalize_collaboration(
    payload: Any, *, fallback_owner_user_id: Optional[str] = None
) -> dict[str, Any]:
    raw = payload if isinstance(payload, dict) else {}
    assigned_user_id = None
    assigned_by_user_id = None
    assigned_at = None
    for key in ("assigned_user_id", "assigned_by_user_id"):
        raw_value = str(raw.get(key) or "").strip()
        if not raw_value:
            continue
        try:
            normalized = str(UUID(raw_value))
        except Exception:
            normalized = None
        if key == "assigned_user_id":
            assigned_user_id = normalized
        else:
            assigned_by_user_id = normalized
    assigned_at_raw = raw.get("assigned_at")
    if isinstance(assigned_at_raw, datetime):
        assigned_at = assigned_at_raw.isoformat()
    elif str(assigned_at_raw or "").strip():
        assigned_at = str(assigned_at_raw).strip()
    owner_user_id = (
        str(raw.get("owner_user_id") or fallback_owner_user_id or "").strip() or None
    )
    visibility = normalize_collaboration_visibility(raw.get("visibility"))
    shared_with_user_ids = uuid_list(raw.get("shared_with_user_ids"), 200)
    if assigned_user_id and assigned_user_id not in shared_with_user_ids:
        shared_with_user_ids.append(assigned_user_id)
    return {
        "owner_user_id": owner_user_id,
        "visibility": visibility,
        "shared_with_user_ids": shared_with_user_ids,
        "assigned_user_id": assigned_user_id,
        "assigned_at": assigned_at,
        "assigned_by_user_id": assigned_by_user_id,
        "note": str(raw.get("note") or "").strip() or None,
    }


def append_manual_promotion_history(
    slice_state: dict[str, Any],
    *,
    action: str,
    operator_note: Optional[str] = None,
    proposal_id: Optional[str] = None,
    patch_pr_id: Optional[str] = None,
    apply_job_id: Optional[str] = None,
) -> None:
    rows = (
        slice_state.get("manual_promotion_history")
        if isinstance(slice_state.get("manual_promotion_history"), list)
        else []
    )
    rows.append(
        {
            "at": datetime.utcnow().isoformat(),
            "action": action,
            "operator_note": operator_note,
            "proposal_id": proposal_id,
            "patch_pr_id": patch_pr_id,
            "apply_job_id": apply_job_id,
        }
    )
    slice_state["manual_promotion_history"] = rows[-40:]


def allowed_actions_for_slice(slice_state: dict[str, Any]) -> list[str]:
    status = str(slice_state.get("status") or "").strip().lower()
    if status in {"proposal_only", "blocked", "patch_pr"} or bool(
        slice_state.get("awaiting_operator_action")
    ):
        return [
            "apply_override",
            "create_patch_pr",
            "keep_proposal_only",
            "relaunch_slice",
            "skip_slice",
        ]
    if status == "failed":
        return ["relaunch_slice", "skip_slice"]
    return []


def recommended_action_for_slice(slice_state: dict[str, Any]) -> Optional[str]:
    blocked_reason = str(slice_state.get("blocked_reason") or "").strip().lower()
    if blocked_reason in {
        "blocked_path_prefix",
        "max_files_touched_exceeded",
        "require_patch_pr",
    }:
        return "create_patch_pr"
    if blocked_reason == "confidence_below_threshold":
        return "apply_override"
    if str(slice_state.get("status") or "").strip().lower() == "failed":
        return "relaunch_slice"
    return "keep_proposal_only"


def refresh_waiting_metadata(
    decomposition: dict[str, Any], slice_state: Optional[dict[str, Any]]
) -> None:
    for row in decomposition.get("planned_slices") or []:
        row["allowed_slice_actions"] = (
            allowed_actions_for_slice(row) if row is slice_state else []
        )
    if slice_state is not None:
        slice_state["awaiting_operator_action"] = True
        slice_state["allowed_slice_actions"] = allowed_actions_for_slice(slice_state)
        slice_state["recommended_next_action"] = recommended_action_for_slice(
            slice_state
        )


def clear_waiting_metadata(slice_state: dict[str, Any]) -> None:
    slice_state["awaiting_operator_action"] = False
    slice_state["allowed_slice_actions"] = []
    slice_state["recommended_next_action"] = None


def set_latest_summary(
    item: CodingBacklogItem,
    decomposition: dict[str, Any],
    *,
    status_value: str,
    slice_state: Optional[dict[str, Any]] = None,
    note: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    summary = {
        "status": status_value,
        "portfolio_progress": recompute_portfolio_progress(decomposition),
        "waiting_on_operator_action": bool(
            slice_state and slice_state.get("awaiting_operator_action")
        ),
        "allowed_slice_actions": (
            slice_state.get("allowed_slice_actions") if slice_state else []
        )
        or [],
        "recommended_next_action": slice_state.get("recommended_next_action")
        if slice_state
        else None,
        "active_slice_id": slice_state.get("slice_id")
        if slice_state
        else decomposition.get("active_slice_id"),
        "active_slice_title": slice_state.get("title") if slice_state else None,
        "note": note,
    }
    if extra:
        summary.update(extra)
    item.latest_summary = summary


TERMINAL_CLOSURE_REASONS = {
    "fixed_through_backlog",
    "promoted_to_repair",
    "duplicate",
    "false_alarm",
    "outdated",
    "blocked_external",
}


def normalize_closure_reason(value: Any) -> Optional[str]:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in TERMINAL_CLOSURE_REASONS else None
