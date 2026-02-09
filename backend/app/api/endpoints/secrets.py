"""
Secrets vault endpoints (per-user).
"""

from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.secret import UserSecret
from app.models.user import User
from app.schemas.secrets import SecretCreateRequest, SecretResponse, SecretRevealResponse
from app.services.auth_service import get_current_user
from app.services.secret_service import SecretService

router = APIRouter()


@router.get("", response_model=List[SecretResponse])
async def list_secrets(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(UserSecret).where(UserSecret.user_id == current_user.id).order_by(UserSecret.name)
    )
    items = result.scalars().all()
    return [
        SecretResponse(
            id=s.id,
            name=s.name,
            created_at=s.created_at,
            updated_at=s.updated_at,
        )
        for s in items
    ]


@router.post("", response_model=SecretRevealResponse)
async def upsert_secret(
    payload: SecretCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    svc = SecretService()
    name = payload.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Secret name is required")

    result = await db.execute(
        select(UserSecret).where(UserSecret.user_id == current_user.id, UserSecret.name == name)
    )
    secret = result.scalar_one_or_none()
    if secret:
        secret.encrypted_value = svc.encrypt(payload.value)
    else:
        secret = UserSecret(
            user_id=current_user.id,
            name=name,
            encrypted_value=svc.encrypt(payload.value),
        )
        db.add(secret)

    await db.commit()
    await db.refresh(secret)
    logger.info(f"Updated secret '{name}' for user {current_user.id}")

    return SecretRevealResponse(
        id=secret.id,
        name=secret.name,
        value=payload.value,
        created_at=secret.created_at,
        updated_at=secret.updated_at,
    )


@router.delete("/{secret_id}")
async def delete_secret(
    secret_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(UserSecret).where(UserSecret.id == secret_id, UserSecret.user_id == current_user.id)
    )
    secret = result.scalar_one_or_none()
    if not secret:
        raise HTTPException(status_code=404, detail="Secret not found")

    await db.delete(secret)
    await db.commit()
    return {"success": True}

