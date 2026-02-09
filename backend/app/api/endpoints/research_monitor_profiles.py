"""
Research monitor profiles endpoints.

These profiles are learned from Research Inbox triage and can be inspected/edited.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.research_monitor_profile import ResearchMonitorProfile
from app.models.user import User
from app.schemas.research_monitor_profile import ResearchMonitorProfileResponse, ResearchMonitorProfileUpdateRequest
from app.services.auth_service import get_current_user


router = APIRouter()


@router.get("", response_model=list[ResearchMonitorProfileResponse])
async def list_monitor_profiles(
    customer: Optional[str] = Query(None, description="Optional customer tag"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(ResearchMonitorProfile).where(ResearchMonitorProfile.user_id == current_user.id)
    if customer:
        stmt = stmt.where(ResearchMonitorProfile.customer == customer)
    stmt = stmt.order_by(desc(ResearchMonitorProfile.updated_at))
    res = await db.execute(stmt)
    profiles = list(res.scalars().all())
    return [ResearchMonitorProfileResponse.model_validate(p) for p in profiles]


@router.patch("/{profile_id}", response_model=ResearchMonitorProfileResponse)
async def update_monitor_profile(
    profile_id: str,
    payload: ResearchMonitorProfileUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        from uuid import UUID

        pid = UUID(profile_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid profile id")

    profile = await db.get(ResearchMonitorProfile, pid)
    if not profile or profile.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Profile not found")

    if payload.muted_tokens is not None:
        profile.muted_tokens = [str(x).strip().lower() for x in (payload.muted_tokens or []) if str(x).strip()]
    if payload.muted_patterns is not None:
        profile.muted_patterns = [str(x).strip() for x in (payload.muted_patterns or []) if str(x).strip()]
    if payload.notes is not None:
        profile.notes = (payload.notes or "").strip() or None

    await db.commit()
    await db.refresh(profile)
    return ResearchMonitorProfileResponse.model_validate(profile)


class ResearchMonitorProfileUpsertRequest(ResearchMonitorProfileUpdateRequest):
    customer: Optional[str] = None
    merge_lists: bool = True


@router.post("/upsert", response_model=ResearchMonitorProfileResponse)
async def upsert_monitor_profile(
    payload: ResearchMonitorProfileUpsertRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create or update a monitor profile for the current user (+ optional customer).

    This enables users to mute tokens/patterns even before they have enough triage history.
    """
    customer = (payload.customer or "").strip() or None
    try:
        stmt = select(ResearchMonitorProfile).where(ResearchMonitorProfile.user_id == current_user.id)
        if customer:
            stmt = stmt.where(ResearchMonitorProfile.customer == customer)
        else:
            stmt = stmt.where(ResearchMonitorProfile.customer.is_(None))
        res = await db.execute(stmt.limit(1))
        profile = res.scalar_one_or_none()

        if not profile:
            profile = ResearchMonitorProfile(
                user_id=current_user.id,
                customer=customer,
                token_scores={},
                muted_tokens=[],
                muted_patterns=[],
                notes=None,
            )
            db.add(profile)
            await db.commit()
            await db.refresh(profile)

        if payload.muted_tokens is not None:
            incoming = [str(x).strip().lower() for x in (payload.muted_tokens or []) if str(x).strip()]
            if payload.merge_lists:
                existing = profile.muted_tokens if isinstance(profile.muted_tokens, list) else []
                merged = list(dict.fromkeys([*(str(x).strip().lower() for x in existing if str(x).strip()), *incoming]))
                profile.muted_tokens = merged
            else:
                profile.muted_tokens = incoming
        if payload.muted_patterns is not None:
            incoming = [str(x).strip() for x in (payload.muted_patterns or []) if str(x).strip()]
            if payload.merge_lists:
                existing = profile.muted_patterns if isinstance(profile.muted_patterns, list) else []
                merged = list(dict.fromkeys([*(str(x).strip() for x in existing if str(x).strip()), *incoming]))
                profile.muted_patterns = merged
            else:
                profile.muted_patterns = incoming
        if payload.notes is not None:
            profile.notes = (payload.notes or "").strip() or None

        await db.commit()
        await db.refresh(profile)
        return ResearchMonitorProfileResponse.model_validate(profile)
    except Exception as exc:
        logger.error(f"Failed to upsert monitor profile: {exc}")
        raise HTTPException(status_code=500, detail="Failed to upsert profile")
