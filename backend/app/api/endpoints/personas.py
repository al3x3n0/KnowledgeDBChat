"""
Persona management API endpoints.
"""

from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from loguru import logger

from app.core.database import get_db
from app.models.persona import Persona, PersonaEditRequest
from app.models.user import User
from app.schemas.persona import (
    PersonaCreate,
    PersonaResponse,
    PersonaUpdate,
    PersonaEditRequestCreate,
    PersonaEditRequestResponse,
)
from app.schemas.common import PaginatedResponse
from app.services.auth_service import get_current_user, require_admin
from app.utils.exceptions import ValidationError

router = APIRouter()


@router.get("/", response_model=PaginatedResponse[PersonaResponse])
async def list_personas(
    page: int = 1,
    page_size: int = 20,
    search: Optional[str] = None,
    platform_id: Optional[str] = None,
    name: Optional[str] = None,
    include_inactive: bool = False,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get paginated personas available in the workspace.
    """
    if page < 1:
        raise ValidationError("Page must be >= 1", field="page")
    if page_size < 1 or page_size > 100:
        raise ValidationError("Page size must be between 1 and 100", field="page_size")

    skip = (page - 1) * page_size

    query = select(Persona)
    if not include_inactive:
        query = query.where(Persona.is_active.is_(True))

    if platform_id:
        query = query.where(Persona.platform_id == platform_id)
    if name:
        query = query.where(Persona.name.ilike(name))
    if search:
        like = f"%{search}%"
        query = query.where(
            (Persona.name.ilike(like))
            | (Persona.platform_id.ilike(like))
            | (Persona.description.ilike(like))
        )

    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    result = await db.execute(
        query.order_by(Persona.updated_at.desc()).offset(skip).limit(page_size)
    )
    personas = result.scalars().all()

    items = [PersonaResponse.from_orm(p) for p in personas]
    return PaginatedResponse.create(items=items, total=total, page=page, page_size=page_size)


@router.get("/{persona_id}", response_model=PersonaResponse)
async def get_persona(
    persona_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get single persona by ID."""
    result = await db.execute(select(Persona).where(Persona.id == persona_id))
    persona = result.scalar_one_or_none()
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")
    return PersonaResponse.from_orm(persona)


@router.post("/", response_model=PersonaResponse, status_code=201)
async def create_persona(
    data: PersonaCreate,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Create a new persona."""
    persona = Persona(
        name=data.name.strip(),
        platform_id=data.platform_id.strip() if data.platform_id else None,
        description=data.description,
        avatar_url=data.avatar_url,
        extra_metadata=data.extra_metadata,
        user_id=data.user_id,
        is_active=data.is_active if data.is_active is not None else True,
        is_system=data.is_system if data.is_system is not None else False,
    )
    db.add(persona)

    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        logger.warning(f"Failed to create persona due to integrity error: {exc}")
        raise HTTPException(status_code=400, detail="Persona name or platform_id already exists")

    await db.refresh(persona)
    return PersonaResponse.from_orm(persona)


@router.put("/{persona_id}", response_model=PersonaResponse)
async def update_persona(
    persona_id: UUID,
    data: PersonaUpdate,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Update an existing persona."""
    result = await db.execute(select(Persona).where(Persona.id == persona_id))
    persona = result.scalar_one_or_none()
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")

    if data.name is not None:
        persona.name = data.name.strip()
    if data.platform_id is not None:
        persona.platform_id = data.platform_id.strip() if data.platform_id else None
    if data.description is not None:
        persona.description = data.description
    if data.avatar_url is not None:
        persona.avatar_url = data.avatar_url
    if data.extra_metadata is not None:
        persona.extra_metadata = data.extra_metadata
    if data.user_id is not None:
        persona.user_id = data.user_id
    if data.is_active is not None:
        persona.is_active = data.is_active
    if data.is_system is not None:
        persona.is_system = data.is_system

    try:
        await db.commit()
    except IntegrityError as exc:
        await db.rollback()
        logger.warning(f"Failed to update persona due to integrity error: {exc}")
        raise HTTPException(status_code=400, detail="Persona name or platform_id already exists")

    await db.refresh(persona)
    return PersonaResponse.from_orm(persona)


@router.delete("/{persona_id}", status_code=204)
async def delete_persona(
    persona_id: UUID,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """Delete a persona."""
    result = await db.execute(select(Persona).where(Persona.id == persona_id))
    persona = result.scalar_one_or_none()
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")

    await db.delete(persona)
    await db.commit()
    return None


@router.post("/{persona_id}/edit-request", response_model=PersonaEditRequestResponse, status_code=201)
async def request_persona_edit(
    persona_id: UUID,
    data: PersonaEditRequestCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Allow non-admin users to request persona updates for admin review."""
    result = await db.execute(select(Persona).where(Persona.id == persona_id))
    persona = result.scalar_one_or_none()
    if not persona:
        raise HTTPException(status_code=404, detail="Persona not found")

    message = data.message.strip()
    if len(message) < 5:
        raise ValidationError("Request message must include some details", field="message")

    request = PersonaEditRequest(
        persona_id=persona_id,
        requested_by=current_user.id,
        document_id=data.document_id,
        message=message,
        status="pending",
    )
    db.add(request)
    await db.commit()
    await db.refresh(request)

    return PersonaEditRequestResponse(
        id=request.id,
        persona_id=request.persona_id,
        requested_by=request.requested_by,
        requested_by_name=current_user.full_name or current_user.username,
        document_id=request.document_id,
        message=request.message,
        status=request.status,
        created_at=request.created_at,
        resolved_at=request.resolved_at,
    )
