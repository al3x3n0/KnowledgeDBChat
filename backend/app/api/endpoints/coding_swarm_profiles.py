from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.coding_swarm_profile import CodingSwarmProfile
from app.models.document import DocumentSource
from app.models.user import User
from app.schemas.coding_swarm_profile import (
    CodingSwarmProfileCreate,
    CodingSwarmProfileListResponse,
    CodingSwarmProfileResponse,
    CodingSwarmProfileUpdate,
)
from app.services.collaboration_service import (
    build_collaboration_summary,
    list_collaboration_user_ids,
)

router = APIRouter()


def _normalize_str_list(values: object, limit: int) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value[:500])
        if len(out) >= limit:
            break
    return out


def _normalize_uuid_list(values: object, limit: int = 50) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        try:
            text = str(UUID(str(raw))).strip()
        except Exception:
            continue
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _normalize_visibility(value: object) -> str:
    normalized = str(value or "private").strip().lower()
    return "shared" if normalized == "shared" else "private"


def _is_profile_visible_to_user(profile: CodingSwarmProfile, user_id: UUID) -> bool:
    if str(profile.user_id) == str(user_id):
        return True
    if _normalize_visibility(getattr(profile, "visibility", "private")) != "shared":
        return False
    shared_with = _normalize_uuid_list(
        getattr(profile, "shared_with_user_ids", None), 200
    )
    return str(user_id) in shared_with


async def _build_profile_user_lookup(
    db: AsyncSession, *, current_user: User
) -> dict[str, User]:
    visible_user_ids = await list_collaboration_user_ids(db, current_user=current_user)
    rows = list(
        (await db.execute(select(User).where(User.id.in_(visible_user_ids))))
        .scalars()
        .all()
    )
    return {str(row.id): row for row in rows if row is not None}


def _profile_to_response(
    profile: CodingSwarmProfile,
    *,
    current_user: User | None = None,
    user_lookup: dict[str, User] | None = None,
) -> CodingSwarmProfileResponse:
    collaboration_summary = build_collaboration_summary(
        owner_user_id=str(getattr(profile, "user_id", "") or "") or None,
        visibility=_normalize_visibility(getattr(profile, "visibility", "private")),
        shared_with_user_ids=_normalize_uuid_list(
            getattr(profile, "shared_with_user_ids", None), 200
        ),
        current_user_id=str(getattr(current_user, "id", "") or "") or None,
        user_lookup=user_lookup,
    )
    return CodingSwarmProfileResponse.model_validate(
        {
            **profile.__dict__,
            "shared_with_user_ids": _normalize_uuid_list(
                getattr(profile, "shared_with_user_ids", None), 200
            ),
            "visibility": _normalize_visibility(
                getattr(profile, "visibility", "private")
            ),
            "collaboration_summary": collaboration_summary,
        }
    )


async def _get_profile_or_404(
    db: AsyncSession, profile_id: UUID, user_id: UUID
) -> CodingSwarmProfile:
    profile = await db.get(CodingSwarmProfile, profile_id)
    if not profile or profile.user_id != user_id:
        raise HTTPException(status_code=404, detail="Coding swarm profile not found")
    return profile


async def _get_visible_profile_or_404(
    db: AsyncSession, profile_id: UUID, user_id: UUID
) -> CodingSwarmProfile:
    profile = await db.get(CodingSwarmProfile, profile_id)
    if not profile or not _is_profile_visible_to_user(profile, user_id):
        raise HTTPException(status_code=404, detail="Coding swarm profile not found")
    return profile


async def _validate_source(
    db: AsyncSession, source_id: UUID, user_id: UUID
) -> DocumentSource:
    source = await db.get(DocumentSource, source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Document source not found")
    source_type = str(source.source_type or "").strip().lower()
    if source_type not in {"github", "gitlab"}:
        raise HTTPException(
            status_code=400,
            detail="Coding swarm profiles require a git-backed document source",
        )
    source_user_id = getattr(source, "user_id", None)
    if source_user_id is not None and str(source_user_id) != str(user_id):
        raise HTTPException(status_code=404, detail="Document source not found")
    return source


async def _clear_other_defaults(
    db: AsyncSession, *, user_id: UUID, source_id: UUID, keep_id: UUID | None = None
) -> None:
    rows = list(
        (
            await db.execute(
                select(CodingSwarmProfile).where(
                    CodingSwarmProfile.user_id == user_id,
                    CodingSwarmProfile.source_id == source_id,
                    CodingSwarmProfile.is_default.is_(True),
                )
            )
        )
        .scalars()
        .all()
    )
    for row in rows:
        if keep_id and str(row.id) == str(keep_id):
            continue
        row.is_default = False
        db.add(row)


@router.get("", response_model=CodingSwarmProfileListResponse)
async def list_coding_swarm_profiles(
    source_id: UUID | None = Query(None),
    preset_key: str | None = Query(None),
    visibility_scope: str | None = Query(None, description="mine|shared|all"),
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    visibility_scope = str(visibility_scope or "all").strip().lower() or "all"
    stmt = select(CodingSwarmProfile).where(
        or_(
            CodingSwarmProfile.user_id == current_user.id,
            CodingSwarmProfile.visibility == "shared",
        )
    )
    if source_id:
        stmt = stmt.where(CodingSwarmProfile.source_id == source_id)
    if str(preset_key or "").strip():
        stmt = stmt.where(CodingSwarmProfile.preset_key == str(preset_key).strip())
    stmt = (
        stmt.order_by(
            desc(CodingSwarmProfile.is_default), desc(CodingSwarmProfile.updated_at)
        )
        .offset(offset)
        .limit(limit)
    )
    rows = [
        row
        for row in list((await db.execute(stmt)).scalars().all())
        if _is_profile_visible_to_user(row, current_user.id)
    ]
    user_lookup = await _build_profile_user_lookup(db, current_user=current_user)
    if visibility_scope == "mine":
        rows = [row for row in rows if str(row.user_id) == str(current_user.id)]
    elif visibility_scope == "shared":
        rows = [row for row in rows if str(row.user_id) != str(current_user.id)]
    total = len(rows)
    rows = rows[:limit]
    return CodingSwarmProfileListResponse(
        items=[
            _profile_to_response(
                row, current_user=current_user, user_lookup=user_lookup
            )
            for row in rows
        ],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.post(
    "", response_model=CodingSwarmProfileResponse, status_code=status.HTTP_201_CREATED
)
async def create_coding_swarm_profile(
    payload: CodingSwarmProfileCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    await _validate_source(db, payload.source_id, current_user.id)
    if payload.is_default:
        await _clear_other_defaults(
            db, user_id=current_user.id, source_id=payload.source_id
        )
    profile = CodingSwarmProfile(
        user_id=current_user.id,
        source_id=payload.source_id,
        title=payload.title,
        description=payload.description,
        status="active",
        preset_key=str(payload.preset_key or "").strip(),
        scope_default=str(payload.scope_default or "auto").strip() or "auto",
        default_commands=_normalize_str_list(payload.default_commands, 8) or None,
        default_file_paths=_normalize_str_list(payload.default_file_paths, 16) or None,
        max_agents=max(1, min(int(payload.max_agents or 4), 4)),
        safe_command_policy=str(payload.safe_command_policy or "standard").strip()
        or "standard",
        saved_search_query=str(payload.saved_search_query or "").strip() or None,
        is_default=bool(payload.is_default),
        visibility=_normalize_visibility(payload.visibility),
        shared_with_user_ids=_normalize_uuid_list(payload.shared_with_user_ids, 200)
        or None,
        profile_metadata=payload.profile_metadata
        if isinstance(payload.profile_metadata, dict)
        else None,
    )
    db.add(profile)
    await db.commit()
    await db.refresh(profile)
    user_lookup = await _build_profile_user_lookup(db, current_user=current_user)
    return _profile_to_response(
        profile, current_user=current_user, user_lookup=user_lookup
    )


@router.patch("/{profile_id}", response_model=CodingSwarmProfileResponse)
async def update_coding_swarm_profile(
    profile_id: UUID,
    payload: CodingSwarmProfileUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    profile = await _get_profile_or_404(db, profile_id, current_user.id)
    if payload.title is not None:
        profile.title = payload.title
    if payload.description is not None:
        profile.description = payload.description
    if payload.preset_key is not None:
        profile.preset_key = str(payload.preset_key or "").strip()
    if payload.scope_default is not None:
        profile.scope_default = str(payload.scope_default or "auto").strip() or "auto"
    if payload.default_commands is not None:
        profile.default_commands = (
            _normalize_str_list(payload.default_commands, 8) or None
        )
    if payload.default_file_paths is not None:
        profile.default_file_paths = (
            _normalize_str_list(payload.default_file_paths, 16) or None
        )
    if payload.max_agents is not None:
        profile.max_agents = max(1, min(int(payload.max_agents or 4), 4))
    if payload.safe_command_policy is not None:
        profile.safe_command_policy = (
            str(payload.safe_command_policy or "standard").strip() or "standard"
        )
    if payload.saved_search_query is not None:
        profile.saved_search_query = (
            str(payload.saved_search_query or "").strip() or None
        )
    if payload.profile_metadata is not None:
        profile.profile_metadata = (
            payload.profile_metadata
            if isinstance(payload.profile_metadata, dict)
            else None
        )
    if payload.status is not None:
        profile.status = str(payload.status or "active").strip() or "active"
    if payload.visibility is not None:
        profile.visibility = _normalize_visibility(payload.visibility)
    if payload.shared_with_user_ids is not None:
        profile.shared_with_user_ids = (
            _normalize_uuid_list(payload.shared_with_user_ids, 200) or None
        )
    if payload.is_default is not None:
        profile.is_default = bool(payload.is_default)
        if profile.is_default:
            await _clear_other_defaults(
                db,
                user_id=current_user.id,
                source_id=profile.source_id,
                keep_id=profile.id,
            )
    db.add(profile)
    await db.commit()
    await db.refresh(profile)
    user_lookup = await _build_profile_user_lookup(db, current_user=current_user)
    return _profile_to_response(
        profile, current_user=current_user, user_lookup=user_lookup
    )


@router.delete("/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_coding_swarm_profile(
    profile_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    profile = await _get_profile_or_404(db, profile_id, current_user.id)
    await db.delete(profile)
    await db.commit()
