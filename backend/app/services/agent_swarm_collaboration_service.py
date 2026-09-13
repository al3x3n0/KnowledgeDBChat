"""Shared collaboration metadata helpers for coding swarm jobs and profiles."""

from __future__ import annotations

import uuid
from typing import Any, Optional
from uuid import UUID

from app.models.agent_job import AgentJob
from app.models.coding_swarm_profile import CodingSwarmProfile
from app.models.user import User


def normalize_uuid_str_list(
    values: object,
    limit: int = 100,
) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in values:
        try:
            value = str(UUID(str(raw))).strip()
        except (TypeError, ValueError):
            continue
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
        if len(normalized) >= limit:
            break
    return normalized


def normalize_profile_visibility(value: object) -> str:
    return (
        "shared" if str(value or "private").strip().lower() == "shared" else "private"
    )


def is_profile_visible_to_user(
    profile: CodingSwarmProfile,
    user: User,
) -> bool:
    if user.is_admin() or str(profile.user_id) == str(user.id):
        return True
    if normalize_profile_visibility(profile.visibility) != "shared":
        return False
    return str(user.id) in normalize_uuid_str_list(
        profile.shared_with_user_ids,
        200,
    )


def build_collaboration_payload(
    *,
    owner_user_id: UUID | str,
    visibility: str = "private",
    shared_with_user_ids: Optional[list[str]] = None,
    assigned_user_id: Optional[str] = None,
    assigned_by_user_id: Optional[str] = None,
    assigned_at: Optional[str] = None,
    review_note: Optional[str] = None,
) -> dict[str, Any]:
    normalized_shared = normalize_uuid_str_list(
        shared_with_user_ids or [],
        200,
    )
    if assigned_user_id:
        normalized_shared = normalize_uuid_str_list(
            [*normalized_shared, assigned_user_id],
            200,
        )
    return {
        "owner_user_id": str(owner_user_id),
        "shared_review": (
            normalize_profile_visibility(visibility) == "shared"
            or bool(normalized_shared)
        ),
        "shared_with_user_ids": normalized_shared,
        "assigned_user_id": (
            str(assigned_user_id).strip() if assigned_user_id else None
        ),
        "assigned_by_user_id": (
            str(assigned_by_user_id).strip() if assigned_by_user_id else None
        ),
        "assigned_at": str(assigned_at).strip() if assigned_at else None,
        "review_note": str(review_note or "").strip() or None,
    }


def extract_swarm_collaboration(job: AgentJob) -> dict[str, Any]:
    results = job.results if isinstance(job.results, dict) else {}
    raw = (
        results.get("swarm_collaboration")
        if isinstance(results.get("swarm_collaboration"), dict)
        else {}
    )
    return build_collaboration_payload(
        owner_user_id=(
            raw.get("owner_user_id") or getattr(job, "user_id", None) or uuid.uuid4()
        ),
        visibility=(
            "shared"
            if bool(raw.get("shared_review"))
            or normalize_uuid_str_list(
                raw.get("shared_with_user_ids"),
                200,
            )
            else "private"
        ),
        shared_with_user_ids=normalize_uuid_str_list(
            raw.get("shared_with_user_ids"),
            200,
        ),
        assigned_user_id=str(raw.get("assigned_user_id") or "").strip() or None,
        assigned_by_user_id=str(raw.get("assigned_by_user_id") or "").strip() or None,
        assigned_at=str(raw.get("assigned_at") or "").strip() or None,
        review_note=str(raw.get("review_note") or "").strip() or None,
    )


def store_swarm_collaboration(
    job: AgentJob,
    collaboration: dict[str, Any],
) -> None:
    results = dict(job.results) if isinstance(job.results, dict) else {}
    results["swarm_collaboration"] = collaboration
    job.results = results
