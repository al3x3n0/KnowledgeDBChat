"""
API Key management endpoints.

Allows users to create, list, and revoke API keys for external tool access.
"""

from typing import List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.database import get_db
from app.models.user import User
from app.services.auth_service import get_current_user, require_admin
from app.services.api_key_service import api_key_service
from app.schemas.api_key import (
    APIKeyCreate,
    APIKeyUpdate,
    APIKeyResponse,
    APIKeyCreateResponse,
    APIKeyListResponse,
    APIKeyUsageStats,
)

router = APIRouter()


@router.post("/", response_model=APIKeyCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_api_key(
    request: APIKeyCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new API key.

    The API key will be shown only once in the response. Store it securely!

    **Scopes** (optional, None = full access):
    - `read`: Read-only access to documents and search
    - `write`: Create and modify documents
    - `chat`: Access chat/agent functionality
    - `documents`: Full document management
    - `workflows`: Execute workflows
    - `admin`: Full administrative access
    """
    try:
        api_key, plain_key = await api_key_service.create_api_key(
            db=db,
            user_id=current_user.id,
            name=request.name,
            description=request.description,
            scopes=request.scopes,
            expires_in_days=request.expires_in_days,
            rate_limit_per_minute=request.rate_limit_per_minute,
            rate_limit_per_day=request.rate_limit_per_day,
        )

        return APIKeyCreateResponse(
            id=api_key.id,
            name=api_key.name,
            description=api_key.description,
            key_prefix=api_key.key_prefix,
            api_key=plain_key,
            scopes=api_key.scopes,
            rate_limit_per_minute=api_key.rate_limit_per_minute,
            rate_limit_per_day=api_key.rate_limit_per_day,
            expires_at=api_key.expires_at,
            created_at=api_key.created_at,
            message="Store this API key securely. It will not be shown again!",
        )

    except Exception as e:
        logger.error(f"Error creating API key: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create API key: {str(e)}",
        )


@router.get("/", response_model=APIKeyListResponse)
async def list_api_keys(
    include_revoked: bool = False,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List all API keys for the current user.

    By default, revoked keys are not included.
    """
    api_keys = await api_key_service.list_api_keys(
        db=db,
        user_id=current_user.id,
        include_revoked=include_revoked,
    )

    return APIKeyListResponse(
        api_keys=[APIKeyResponse.model_validate(key) for key in api_keys],
        total=len(api_keys),
    )


@router.get("/{key_id}", response_model=APIKeyResponse)
async def get_api_key(
    key_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get details of a specific API key."""
    api_key = await api_key_service.get_api_key(
        db=db,
        key_id=key_id,
        user_id=current_user.id,
    )

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    return APIKeyResponse.model_validate(api_key)


@router.patch("/{key_id}", response_model=APIKeyResponse)
async def update_api_key(
    key_id: UUID,
    request: APIKeyUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update an API key's metadata (name, description, scopes, rate limits)."""
    api_key = await api_key_service.update_api_key(
        db=db,
        key_id=key_id,
        user_id=current_user.id,
        name=request.name,
        description=request.description,
        scopes=request.scopes,
        rate_limit_per_minute=request.rate_limit_per_minute,
        rate_limit_per_day=request.rate_limit_per_day,
        is_active=request.is_active,
    )

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    return APIKeyResponse.model_validate(api_key)


@router.delete("/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_api_key(
    key_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Revoke an API key.

    This action cannot be undone. The key will immediately stop working.
    """
    success = await api_key_service.revoke_api_key(
        db=db,
        key_id=key_id,
        user_id=current_user.id,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )


@router.get("/{key_id}/usage", response_model=APIKeyUsageStats)
async def get_api_key_usage(
    key_id: UUID,
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get usage statistics for an API key."""
    # Verify ownership
    api_key = await api_key_service.get_api_key(
        db=db,
        key_id=key_id,
        user_id=current_user.id,
    )

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    stats = await api_key_service.get_usage_stats(
        db=db,
        key_id=key_id,
        days=days,
    )

    return APIKeyUsageStats(**stats)


# Admin endpoints


@router.get("/admin/all", response_model=APIKeyListResponse)
async def admin_list_all_api_keys(
    include_revoked: bool = False,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    [Admin] List all API keys in the system.
    """
    from sqlalchemy import select
    from app.models.api_key import APIKey

    query = select(APIKey)
    if not include_revoked:
        query = query.where(APIKey.revoked_at.is_(None))
    query = query.order_by(APIKey.created_at.desc())

    result = await db.execute(query)
    api_keys = list(result.scalars().all())

    return APIKeyListResponse(
        api_keys=[APIKeyResponse.model_validate(key) for key in api_keys],
        total=len(api_keys),
    )


@router.delete("/admin/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def admin_revoke_api_key(
    key_id: UUID,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    [Admin] Revoke any API key in the system.
    """
    success = await api_key_service.revoke_api_key(
        db=db,
        key_id=key_id,
        user_id=None,  # Admin can revoke any key
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )
