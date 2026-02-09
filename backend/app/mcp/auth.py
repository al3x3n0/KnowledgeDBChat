"""
MCP authentication and authorization.

Handles API key validation and user context for MCP requests.
"""

import hashlib
from datetime import datetime
from typing import Optional
from uuid import UUID

from fastapi import HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.models.api_key import APIKey, APIKeyUsageLog
from app.models.user import User


class MCPAuthContext:
    """Authentication context for MCP requests."""

    def __init__(
        self,
        user: User,
        api_key: APIKey,
        scopes: list[str],
    ):
        self.user = user
        self.api_key = api_key
        self.scopes = scopes or []

    @property
    def user_id(self) -> UUID:
        return self.user.id

    @property
    def is_admin(self) -> bool:
        return self.user.role == "admin"

    def has_scope(self, scope: str) -> bool:
        """Check if context has a specific scope."""
        return self.api_key.has_scope(scope)

    def require_scope(self, scope: str) -> None:
        """Require a specific scope, raise error if missing."""
        if not self.has_scope(scope):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"API key missing required scope: {scope}"
            )


def hash_api_key(key: str) -> str:
    """Hash an API key for storage/comparison."""
    return hashlib.sha256(key.encode()).hexdigest()


async def validate_api_key(
    api_key: str,
    db: AsyncSession,
    required_scope: Optional[str] = None,
) -> Optional[MCPAuthContext]:
    """
    Validate an API key and return auth context.

    Args:
        api_key: The API key string to validate
        db: Database session
        required_scope: Optional scope that must be present

    Returns:
        MCPAuthContext if valid, None otherwise
    """
    if not api_key:
        return None

    # Hash the key for lookup
    key_hash = hash_api_key(api_key)

    # Find the API key
    result = await db.execute(
        select(APIKey)
        .where(APIKey.key_hash == key_hash)
        .where(APIKey.is_active == True)
        .where(APIKey.revoked_at == None)
    )
    db_key = result.scalar_one_or_none()

    if not db_key:
        logger.warning(f"Invalid API key attempted: {api_key[:12]}...")
        return None

    # Check if key is valid (not expired, etc.)
    if not db_key.is_valid():
        logger.warning(f"Expired/invalid API key: {db_key.id}")
        return None

    # Check required scope
    if required_scope and not db_key.has_scope(required_scope):
        logger.warning(f"API key {db_key.id} missing scope: {required_scope}")
        return None

    # Load the user
    result = await db.execute(
        select(User).where(User.id == db_key.user_id)
    )
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        logger.warning(f"API key {db_key.id} belongs to inactive user")
        return None

    # Update last used
    db_key.last_used_at = datetime.utcnow()
    db_key.usage_count += 1
    await db.commit()

    logger.info(f"API key authenticated: {db_key.name} (user: {user.username})")

    return MCPAuthContext(
        user=user,
        api_key=db_key,
        scopes=db_key.scopes or [],
    )


async def log_api_usage(
    db: AsyncSession,
    api_key_id: UUID,
    endpoint: str,
    method: str,
    status_code: int,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    response_time_ms: Optional[int] = None,
) -> None:
    """Log API key usage for auditing."""
    try:
        log = APIKeyUsageLog(
            api_key_id=api_key_id,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            ip_address=ip_address,
            user_agent=user_agent,
            response_time_ms=response_time_ms,
        )
        db.add(log)
        await db.commit()
    except Exception as e:
        logger.warning(f"Failed to log API usage: {e}")


def extract_api_key(request: Request) -> Optional[str]:
    """Extract API key from request headers or query params."""
    # Try Authorization header first (Bearer token)
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]

    # Try X-API-Key header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return api_key

    # Try query parameter
    api_key = request.query_params.get("api_key")
    if api_key:
        return api_key

    return None
