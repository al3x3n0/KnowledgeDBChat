from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.domain_research_profile import DomainResearchProfile
from app.models.research_portfolio import ResearchPortfolio
from app.models.scientific_sandbox_profile import ScientificSandboxProfile
from app.models.user import User
from app.schemas.scientific_sandbox_profile import (
    ScientificSandboxProfileCreate,
    ScientificSandboxProfileListResponse,
    ScientificSandboxProfileResponse,
    ScientificSandboxProfileUpdate,
)
from app.services.auth_service import require_admin
from app.services.scientific_validation_service import (
    get_scientific_sandbox_profile,
    list_scientific_sandbox_profiles,
    validate_scientific_sandbox_profile_payload,
)


router = APIRouter()


def _profile_response(payload: dict) -> ScientificSandboxProfileResponse:
    return ScientificSandboxProfileResponse.model_validate(payload)


@router.get("", response_model=ScientificSandboxProfileListResponse)
async def list_sandbox_profiles(
    include_disabled: bool = Query(False),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    del current_user
    items = [
        _profile_response(item)
        for item in await list_scientific_sandbox_profiles(db, include_disabled=include_disabled)
        if isinstance(item, dict)
    ]
    return ScientificSandboxProfileListResponse(items=items, total=len(items))


@router.get("/{profile_id}", response_model=ScientificSandboxProfileResponse)
async def get_sandbox_profile(
    profile_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    del current_user
    profile = await get_scientific_sandbox_profile(db, profile_id, include_disabled=True)
    if not isinstance(profile, dict):
        raise HTTPException(status_code=404, detail="Scientific sandbox profile not found")
    return _profile_response(profile)


@router.post("", response_model=ScientificSandboxProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_sandbox_profile(
    payload: ScientificSandboxProfileCreate,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    existing = await db.get(ScientificSandboxProfile, payload.id)
    if existing is not None:
        raise HTTPException(status_code=400, detail="Scientific sandbox profile id already exists")
    try:
        normalized = validate_scientific_sandbox_profile_payload(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if normalized["is_default"]:
        stmt = select(ScientificSandboxProfile).where(
            ScientificSandboxProfile.track_type == normalized["track_type"],
            ScientificSandboxProfile.is_default.is_(True),
        )
        for row in (await db.execute(stmt)).scalars().all():
            row.is_default = False

    profile = ScientificSandboxProfile(
        **normalized,
        created_by_user_id=current_user.id,
    )
    db.add(profile)
    await db.commit()
    await db.refresh(profile)
    return _profile_response(profile.to_dict())


@router.patch("/{profile_id}", response_model=ScientificSandboxProfileResponse)
async def update_sandbox_profile(
    profile_id: str,
    payload: ScientificSandboxProfileUpdate,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    del current_user
    profile = await db.get(ScientificSandboxProfile, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Scientific sandbox profile not found")
    base = profile.to_dict()
    merged = {
        **base,
        **payload.model_dump(exclude_unset=True),
        "system_managed": bool(profile.system_managed),
    }
    try:
        normalized = validate_scientific_sandbox_profile_payload(
            merged,
            allow_system_managed=bool(profile.system_managed),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if profile.system_managed:
        for key in ("name", "description", "enabled", "is_default"):
            if key in payload.model_dump(exclude_unset=True):
                setattr(profile, key, normalized[key])
    else:
        for key, value in normalized.items():
            setattr(profile, key, value)

    if normalized["is_default"]:
        stmt = select(ScientificSandboxProfile).where(
            ScientificSandboxProfile.track_type == normalized["track_type"],
            ScientificSandboxProfile.is_default.is_(True),
            ScientificSandboxProfile.id != profile.id,
        )
        for row in (await db.execute(stmt)).scalars().all():
            row.is_default = False

    await db.commit()
    await db.refresh(profile)
    return _profile_response(profile.to_dict())


@router.delete("/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_sandbox_profile(
    profile_id: str,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    del current_user
    profile = await db.get(ScientificSandboxProfile, profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Scientific sandbox profile not found")
    if profile.system_managed:
        raise HTTPException(status_code=400, detail="System-managed sandbox profiles cannot be deleted")

    domain_ref = (
        await db.execute(
            select(DomainResearchProfile.id).where(DomainResearchProfile.sandbox_profile_id == profile.id).limit(1)
        )
    ).scalars().first()
    portfolio_ref = (
        await db.execute(
            select(ResearchPortfolio.id).where(ResearchPortfolio.sandbox_profile_id == profile.id).limit(1)
        )
    ).scalars().first()
    if domain_ref or portfolio_ref:
        raise HTTPException(status_code=400, detail="Sandbox profile is still referenced by research configuration")

    await db.delete(profile)
    await db.commit()
    return None
