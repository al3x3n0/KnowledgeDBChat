"""
Saved searches and share links.
"""

import secrets as pysecrets
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, desc, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.saved_search import SavedSearch, SearchShare
from app.models.user import User
from app.schemas.searches import (
    SavedSearchCreateRequest,
    SavedSearchResponse,
    SearchShareCreateRequest,
    SearchShareResponse,
)
from app.services.auth_service import get_current_user

router = APIRouter()


@router.get("", response_model=List[SavedSearchResponse])
async def list_saved_searches(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List saved searches for the current user plus system-wide pre-built searches."""
    result = await db.execute(
        select(SavedSearch)
        .where(
            or_(
                SavedSearch.user_id == current_user.id,
                SavedSearch.user_id.is_(None)  # System-wide searches
            )
        )
        .order_by(
            SavedSearch.user_id.is_(None).desc(),  # System searches first
            desc(SavedSearch.updated_at)
        )
    )
    items = result.scalars().all()
    return [
        SavedSearchResponse(
            id=s.id,
            name=s.name,
            query=s.query,
            filters=s.filters,
            is_system=s.user_id is None,
            created_at=s.created_at,
            updated_at=s.updated_at,
        )
        for s in items
    ]


@router.post("", response_model=SavedSearchResponse)
async def create_saved_search(
    payload: SavedSearchCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    search = SavedSearch(
        user_id=current_user.id,
        name=payload.name.strip(),
        query=payload.query.strip(),
        filters=payload.filters,
    )
    db.add(search)
    await db.commit()
    await db.refresh(search)
    return SavedSearchResponse(
        id=search.id,
        name=search.name,
        query=search.query,
        filters=search.filters,
        is_system=False,
        created_at=search.created_at,
        updated_at=search.updated_at,
    )


@router.delete("/{search_id}")
async def delete_saved_search(
    search_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(SavedSearch).where(SavedSearch.id == search_id, SavedSearch.user_id == current_user.id)
    )
    row = result.scalar_one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Saved search not found")
    await db.delete(row)
    await db.commit()
    return {"success": True}


@router.post("/shares", response_model=SearchShareResponse)
async def create_search_share(
    payload: SearchShareCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    token = pysecrets.token_urlsafe(24)
    share = SearchShare(
        token=token,
        created_by=current_user.id,
        query=payload.query.strip(),
        filters=payload.filters,
    )
    db.add(share)
    await db.commit()
    await db.refresh(share)
    return SearchShareResponse(token=share.token, query=share.query, filters=share.filters, created_at=share.created_at)


@router.get("/shares/{token}", response_model=SearchShareResponse)
async def get_search_share(
    token: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _ = current_user  # authenticated access required
    result = await db.execute(select(SearchShare).where(SearchShare.token == token))
    share = result.scalar_one_or_none()
    if not share:
        raise HTTPException(status_code=404, detail="Share not found")
    return SearchShareResponse(token=share.token, query=share.query, filters=share.filters, created_at=share.created_at)

