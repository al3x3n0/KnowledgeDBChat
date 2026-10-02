"""Edits to a coding backlog item's ``decomposition`` document.

The operator's endpoint and the backlog orchestrator both write the same JSON
-- a backlog timeline, per-slice timelines, job lineage, artifact history and
promotion decisions -- and each carried its own copy of these functions, the
orchestrator's nested inside its runner method. The caps here (100 backlog
events, 60 per slice, 40 artifacts, 12 decisions) are part of the document's
shape: with two copies, changing one would have made the length of a history
depend on who last wrote it.

Every function tolerates a missing or mistyped field, because the document is
a JSON column that older rows populated differently.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

BACKLOG_TIMELINE_LIMIT = 100
SLICE_TIMELINE_LIMIT = 60
ARTIFACT_HISTORY_LIMIT = 40
PROMOTION_DECISIONS_LIMIT = 12


def _rows(holder: dict[str, Any], key: str) -> list:
    value = holder.get(key)
    return value if isinstance(value, list) else []


def timeline_entry(
    *,
    actor: str,
    action: str,
    previous_status: Optional[str] = None,
    new_status: Optional[str] = None,
    note: Optional[str] = None,
    related_job_id: Optional[str] = None,
    related_proposal_id: Optional[str] = None,
    related_patch_pr_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    entry = {
        "at": datetime.utcnow().isoformat(),
        "actor": actor,
        "action": action,
        "previous_status": previous_status,
        "new_status": new_status,
    }
    if note:
        entry["note"] = note
    if related_job_id:
        entry["job_id"] = related_job_id
    if related_proposal_id:
        entry["proposal_id"] = related_proposal_id
    if related_patch_pr_id:
        entry["patch_pr_id"] = related_patch_pr_id
    if metadata:
        entry["metadata"] = metadata
    return entry


def append_backlog_timeline(
    decomposition: dict[str, Any], entry: dict[str, Any]
) -> None:
    rows = _rows(decomposition, "backlog_timeline")
    rows.append(entry)
    decomposition["backlog_timeline"] = rows[-BACKLOG_TIMELINE_LIMIT:]


def append_slice_timeline(slice_state: dict[str, Any], entry: dict[str, Any]) -> None:
    rows = _rows(slice_state, "timeline")
    rows.append(entry)
    slice_state["timeline"] = rows[-SLICE_TIMELINE_LIMIT:]


def append_unique(values: Any, value: Optional[str]) -> list[str]:
    out = (
        [str(v).strip() for v in values if str(v).strip()]
        if isinstance(values, list)
        else []
    )
    target = str(value or "").strip()
    if target and target not in out:
        out.append(target)
    return out


def append_lineage_id(
    slice_state: dict[str, Any], lineage_key: str, value: Optional[str]
) -> None:
    lineage = (
        slice_state.get("job_lineage")
        if isinstance(slice_state.get("job_lineage"), dict)
        else {}
    )
    lineage[lineage_key] = append_unique(lineage.get(lineage_key), value)
    slice_state["job_lineage"] = lineage


def append_artifact_history(
    slice_state: dict[str, Any],
    artifact_type: str,
    artifact_id: Optional[str],
    label: Optional[str] = None,
) -> None:
    if not artifact_id:
        return
    rows = _rows(slice_state, "artifact_history")
    rows.append(
        {
            "at": datetime.utcnow().isoformat(),
            "artifact_type": artifact_type,
            "artifact_id": artifact_id,
            "label": label or artifact_type,
        }
    )
    slice_state["artifact_history"] = rows[-ARTIFACT_HISTORY_LIMIT:]


def find_slice(
    decomposition: dict[str, Any], slice_id: Optional[str]
) -> Optional[dict[str, Any]]:
    target = str(slice_id or "").strip()
    if not target:
        return None
    for row in decomposition.get("planned_slices") or []:
        if str((row or {}).get("slice_id") or "").strip() == target:
            return row
    return None


def upsert_promotion_decision(
    decomposition: dict[str, Any], entry: dict[str, Any]
) -> None:
    """One decision per slice: a later one replaces the earlier."""
    slice_id = str(entry.get("slice_id") or "").strip()
    kept = [
        row
        for row in _rows(decomposition, "promotion_decisions")
        if str((row or {}).get("slice_id") or "").strip() != slice_id
    ]
    kept.append(entry)
    decomposition["promotion_decisions"] = kept[-PROMOTION_DECISIONS_LIMIT:]
