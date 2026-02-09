"""
API Key management service.

Handles creation, validation, and revocation of API keys for external tool access.
"""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, and_
from loguru import logger

from app.models.api_key import APIKey, APIKeyUsageLog
from app.models.user import User


class APIKeyService:
    """Service for managing API keys."""

    # API key format: prefix_randomstring (e.g., "taic_abc123...")
    KEY_PREFIX = "taic"  # Team AI Cloud
    KEY_LENGTH = 48  # Total length of random part

    def generate_key(self) -> Tuple[str, str, str]:
        """
        Generate a new API key.

        Returns:
            Tuple of (full_key, key_prefix, key_hash)
            - full_key: The complete API key to give to the user (shown only once)
            - key_prefix: First 8 characters for identification
            - key_hash: SHA256 hash of the full key for storage
        """
        # Generate random bytes and convert to hex
        random_part = secrets.token_urlsafe(self.KEY_LENGTH)
        full_key = f"{self.KEY_PREFIX}_{random_part}"

        # Get prefix for identification (first 8 chars after the prefix_)
        key_prefix = full_key[:12]  # "taic_" + first 7 chars

        # Hash the full key for secure storage
        key_hash = hashlib.sha256(full_key.encode()).hexdigest()

        return full_key, key_prefix, key_hash

    def hash_key(self, key: str) -> str:
        """Hash an API key for comparison."""
        return hashlib.sha256(key.encode()).hexdigest()

    async def create_api_key(
        self,
        db: AsyncSession,
        user_id: UUID,
        name: str,
        description: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
        rate_limit_per_minute: int = 60,
        rate_limit_per_day: int = 10000,
    ) -> Tuple[APIKey, str]:
        """
        Create a new API key for a user.

        Args:
            db: Database session
            user_id: The user who owns this key
            name: Human-readable name for the key
            description: Optional description
            scopes: List of allowed scopes (None = full access)
            expires_in_days: Days until expiration (None = never)
            rate_limit_per_minute: Requests per minute limit
            rate_limit_per_day: Requests per day limit

        Returns:
            Tuple of (APIKey model, plain_text_key)
            The plain_text_key is only available at creation time!
        """
        # Generate the key
        full_key, key_prefix, key_hash = self.generate_key()

        # Calculate expiration
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        # Create the API key record
        api_key = APIKey(
            name=name,
            description=description,
            key_prefix=key_prefix,
            key_hash=key_hash,
            user_id=user_id,
            scopes=scopes,
            expires_at=expires_at,
            rate_limit_per_minute=rate_limit_per_minute,
            rate_limit_per_day=rate_limit_per_day,
            is_active=True,
        )

        db.add(api_key)
        await db.commit()
        await db.refresh(api_key)

        logger.info(f"Created API key '{name}' for user {user_id}")

        return api_key, full_key

    async def validate_api_key(
        self,
        db: AsyncSession,
        key: str,
        required_scope: Optional[str] = None,
    ) -> Optional[Tuple[APIKey, User]]:
        """
        Validate an API key and return the associated key and user.

        Args:
            db: Database session
            key: The full API key to validate
            required_scope: Optional scope that must be present

        Returns:
            Tuple of (APIKey, User) if valid, None otherwise
        """
        if not key or not key.startswith(f"{self.KEY_PREFIX}_"):
            return None

        # Hash the provided key
        key_hash = self.hash_key(key)

        # Find the API key
        result = await db.execute(
            select(APIKey, User)
            .join(User, APIKey.user_id == User.id)
            .where(APIKey.key_hash == key_hash)
        )
        row = result.first()

        if not row:
            return None

        api_key, user = row

        # Check if key is valid
        if not api_key.is_valid():
            logger.warning(f"Invalid API key attempted: {api_key.key_prefix}...")
            return None

        # Check if user is active
        if not user.is_active:
            logger.warning(f"API key for inactive user: {user.username}")
            return None

        # Check scope if required
        if required_scope and not api_key.has_scope(required_scope):
            logger.warning(f"API key lacks required scope: {required_scope}")
            return None

        return api_key, user

    async def update_usage(
        self,
        db: AsyncSession,
        api_key: APIKey,
        ip_address: Optional[str] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        status_code: Optional[int] = None,
        response_time_ms: Optional[int] = None,
        user_agent: Optional[str] = None,
    ):
        """Update API key usage statistics and optionally log the request."""
        now = datetime.utcnow()

        # Update the key's last used info
        api_key.last_used_at = now
        api_key.last_used_ip = ip_address
        api_key.usage_count = (api_key.usage_count or 0) + 1

        # Log the usage if endpoint info is provided
        if endpoint:
            usage_log = APIKeyUsageLog(
                api_key_id=api_key.id,
                endpoint=endpoint,
                method=method or "GET",
                status_code=status_code,
                ip_address=ip_address,
                user_agent=user_agent,
                response_time_ms=response_time_ms,
            )
            db.add(usage_log)

        await db.commit()

    async def list_api_keys(
        self,
        db: AsyncSession,
        user_id: UUID,
        include_revoked: bool = False,
    ) -> List[APIKey]:
        """List all API keys for a user."""
        query = select(APIKey).where(APIKey.user_id == user_id)

        if not include_revoked:
            query = query.where(APIKey.revoked_at.is_(None))

        query = query.order_by(APIKey.created_at.desc())
        result = await db.execute(query)

        return list(result.scalars().all())

    async def get_api_key(
        self,
        db: AsyncSession,
        key_id: UUID,
        user_id: Optional[UUID] = None,
    ) -> Optional[APIKey]:
        """Get a specific API key by ID."""
        query = select(APIKey).where(APIKey.id == key_id)

        if user_id:
            query = query.where(APIKey.user_id == user_id)

        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def revoke_api_key(
        self,
        db: AsyncSession,
        key_id: UUID,
        user_id: Optional[UUID] = None,
    ) -> bool:
        """
        Revoke an API key.

        Args:
            db: Database session
            key_id: The API key ID to revoke
            user_id: Optional user ID for ownership verification

        Returns:
            True if revoked, False if not found
        """
        api_key = await self.get_api_key(db, key_id, user_id)

        if not api_key:
            return False

        api_key.is_active = False
        api_key.revoked_at = datetime.utcnow()

        await db.commit()
        logger.info(f"Revoked API key: {api_key.key_prefix}... (name: {api_key.name})")

        return True

    async def update_api_key(
        self,
        db: AsyncSession,
        key_id: UUID,
        user_id: UUID,
        name: Optional[str] = None,
        description: Optional[str] = None,
        scopes: Optional[List[str]] = None,
        rate_limit_per_minute: Optional[int] = None,
        rate_limit_per_day: Optional[int] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[APIKey]:
        """Update an API key's metadata."""
        api_key = await self.get_api_key(db, key_id, user_id)

        if not api_key:
            return None

        if name is not None:
            api_key.name = name
        if description is not None:
            api_key.description = description
        if scopes is not None:
            api_key.scopes = scopes
        if rate_limit_per_minute is not None:
            api_key.rate_limit_per_minute = rate_limit_per_minute
        if rate_limit_per_day is not None:
            api_key.rate_limit_per_day = rate_limit_per_day
        if is_active is not None:
            api_key.is_active = is_active

        await db.commit()
        await db.refresh(api_key)

        return api_key

    async def get_usage_stats(
        self,
        db: AsyncSession,
        key_id: UUID,
        days: int = 30,
    ) -> dict:
        """Get usage statistics for an API key."""
        from sqlalchemy import func

        api_key = await self.get_api_key(db, key_id)
        if not api_key:
            return {}

        since = datetime.utcnow() - timedelta(days=days)

        # Count requests
        count_result = await db.execute(
            select(func.count(APIKeyUsageLog.id))
            .where(
                and_(
                    APIKeyUsageLog.api_key_id == key_id,
                    APIKeyUsageLog.timestamp >= since,
                )
            )
        )
        total_requests = count_result.scalar() or 0

        # Get unique endpoints
        endpoints_result = await db.execute(
            select(APIKeyUsageLog.endpoint, func.count(APIKeyUsageLog.id))
            .where(
                and_(
                    APIKeyUsageLog.api_key_id == key_id,
                    APIKeyUsageLog.timestamp >= since,
                )
            )
            .group_by(APIKeyUsageLog.endpoint)
            .order_by(func.count(APIKeyUsageLog.id).desc())
            .limit(10)
        )
        top_endpoints = [
            {"endpoint": row[0], "count": row[1]}
            for row in endpoints_result.all()
        ]

        return {
            "key_id": str(key_id),
            "key_name": api_key.name,
            "period_days": days,
            "total_requests": total_requests,
            "lifetime_requests": api_key.usage_count or 0,
            "last_used_at": api_key.last_used_at.isoformat() if api_key.last_used_at else None,
            "top_endpoints": top_endpoints,
        }


# Singleton instance
api_key_service = APIKeyService()
