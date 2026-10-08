"""
Authentication service for user management and JWT tokens.

Supports both JWT token authentication and API key authentication for external tools.
"""
import asyncio
import hashlib
from datetime import datetime
from typing import Optional
from uuid import UUID

import bcrypt
from fastapi import Depends, Header, HTTPException, Request, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from passlib.context import CryptContext
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.tokens import create_access_token, decode_access_token
from app.models.user import User
from app.services.ldap_service import LdapError, LdapUnavailable, ldap_service


class AuthService:
    """Service for authentication and authorization."""

    # API key header name
    API_KEY_HEADER = "X-API-Key"

    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.security = HTTPBearer(
            auto_error=False
        )  # Don't auto-error to allow API key fallback
        self.api_key_header = APIKeyHeader(name=self.API_KEY_HEADER, auto_error=False)

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        # Ensure password is bytes for bcrypt
        if isinstance(password, str):
            password_bytes = password.encode("utf-8")
        else:
            password_bytes = password

        # Bcrypt has a 72-byte limit - handle long passwords
        if len(password_bytes) > 72:
            logger.debug(
                f"Password exceeds 72 bytes ({len(password_bytes)}), pre-hashing with SHA256"
            )
            # Pre-hash with SHA256 to reduce to fixed 64 bytes
            password_hash = hashlib.sha256(password_bytes).hexdigest()
            password_bytes = password_hash.encode("utf-8")

        # Use bcrypt directly to avoid passlib validation issues
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        # Return as string (bcrypt returns bytes)
        return hashed.decode("utf-8")

    def get_password_hash(self, password: str) -> str:
        """Alias for hash_password for compatibility."""
        return self.hash_password(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        # Ensure password is bytes for bcrypt
        if isinstance(plain_password, str):
            password_bytes = plain_password.encode("utf-8")
        else:
            password_bytes = plain_password

        # Ensure hashed_password is bytes
        if isinstance(hashed_password, str):
            hashed_bytes = hashed_password.encode("utf-8")
        else:
            hashed_bytes = hashed_password

        # Try direct verification first
        if bcrypt.checkpw(password_bytes, hashed_bytes):
            return True

        # If that fails and password is > 72 bytes, try with pre-hash
        if len(password_bytes) > 72:
            password_hash = hashlib.sha256(password_bytes).hexdigest().encode("utf-8")
            return bcrypt.checkpw(password_hash, hashed_bytes)

        return False

    def create_access_token(self, user_id: UUID) -> str:
        """Create a JWT access token."""
        return create_access_token(user_id)

    async def get_user_by_username(
        self, username: str, db: AsyncSession
    ) -> Optional[User]:
        """Get user by username."""
        result = await db.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()

    async def get_user_by_email(self, email: str, db: AsyncSession) -> Optional[User]:
        """Get user by email."""
        result = await db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def get_user_by_id(self, user_id: str, db: AsyncSession) -> Optional[User]:
        """Get user by ID."""
        try:
            user_uuid = UUID(user_id)
            result = await db.execute(select(User).where(User.id == user_uuid))
            return result.scalar_one_or_none()
        except ValueError:
            return None

    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        db: AsyncSession,
        full_name: Optional[str] = None,
    ) -> User:
        """Create a new user."""
        # Check if username already exists
        existing_user = await self.get_user_by_username(username, db)
        if existing_user:
            raise ValueError("Username already exists")

        # Check if email already exists
        existing_email = await self.get_user_by_email(email, db)
        if existing_email:
            raise ValueError("Email already exists")

        # Create new user
        hashed_password = await asyncio.to_thread(self.hash_password, password)
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
            is_active=True,
            is_verified=False,
            role="user",
        )

        db.add(user)
        await db.commit()
        await db.refresh(user)

        logger.info(f"Created new user: {username}")
        return user

    async def authenticate_user(
        self, username: str, password: str, db: AsyncSession
    ) -> Optional[User]:
        """The user these credentials belong to, or None.

        Local accounts are checked against the password stored here. With LDAP
        enabled, LDAP-managed accounts and names unknown here are checked
        against the directory. The rules, each of which the first version got
        wrong:

        * A deactivated account does not sign in, whatever the directory
          says. (An LDAP login used to set ``is_active`` back to true.)
        * When the directory answers "no", that is the answer. The stored
          password is tried only when the directory could not be *reached*,
          and only if ``LDAP_LOCAL_FALLBACK_WHEN_UNAVAILABLE`` allows it --
          it used to be tried after every refusal, so an account disabled in
          the directory kept working with its old local password.
        * A directory login does not take over a local account that merely
          shares its username; see ``_may_link``.
        """
        user = await self.get_user_by_username(username, db)
        if user is not None and not user.is_active:
            logger.info(f"Login refused for deactivated user {username}")
            return None

        ldap_managed = (
            user is not None and getattr(user, "auth_provider", "local") == "ldap"
        )

        if user is not None and not ldap_managed:
            if await asyncio.to_thread(
                self.verify_password, password, user.hashed_password
            ):
                return user

        if not ldap_service.is_configured():
            return None

        try:
            ldap_user = await ldap_service.authenticate_async(username, password)
        except LdapUnavailable as exc:
            logger.warning(f"LDAP could not be asked about {username}: {exc}")
            if (
                ldap_managed
                and getattr(settings, "LDAP_LOCAL_FALLBACK_WHEN_UNAVAILABLE", False)
                and await asyncio.to_thread(
                    self.verify_password, password, user.hashed_password
                )
            ):
                logger.warning(
                    f"LDAP user {username} signed in with the stored password "
                    "while the directory was unreachable"
                )
                return user
            return None
        except LdapError as exc:
            logger.error(f"LDAP is misconfigured; {username} cannot sign in: {exc}")
            return None

        if ldap_user is None:
            return None

        existing = user
        if existing is None:
            existing = await self.get_user_by_username(ldap_user.username, db)
        if existing is None and ldap_user.email:
            existing = await self.get_user_by_email(ldap_user.email, db)

        if existing is not None:
            if not existing.is_active:
                logger.info(
                    f"LDAP login for {username} matches a deactivated account; refused"
                )
                return None
            if not self._may_link(existing, ldap_user):
                logger.warning(
                    f"LDAP login for {username} matches the local account "
                    f"'{existing.username}' but their emails differ; not linked. "
                    "Import the user as an admin, or set "
                    "LDAP_LINK_LOCAL_USERS_BY_USERNAME."
                )
                return None
        elif not getattr(settings, "LDAP_CREATE_USER_ON_LOGIN", True):
            return None

        try:
            return await self._upsert_user_from_ldap(db, ldap_user, existing=existing)
        except ValueError as exc:
            logger.warning(f"LDAP user {username} could not be provisioned: {exc}")
            return None

    @staticmethod
    def _may_link(existing: User, ldap_user) -> bool:
        """Whether a directory login may become this existing account.

        An account already LDAP-managed is the same person by construction.
        A local one is linked when the directory's email is its email: a
        shared username alone would let whoever is called "admin" in the
        directory take over the local "admin".
        """
        if getattr(existing, "auth_provider", "local") == "ldap":
            return True
        if getattr(settings, "LDAP_LINK_LOCAL_USERS_BY_USERNAME", False):
            return True
        ours = (existing.email or "").strip().lower()
        theirs = (getattr(ldap_user, "email", None) or "").strip().lower()
        return bool(ours) and ours == theirs

    async def _upsert_user_from_ldap(
        self,
        db: AsyncSession,
        ldap_user,
        existing: Optional[User] = None,
        *,
        default_role: str = "user",
    ) -> User:
        """Create or update the local record for a directory user.

        Never reactivates: ``is_active`` belongs to this application's admins.
        The role follows the group mapping when one is configured and is left
        alone when none is -- it used to be overwritten with "user" either
        way, demoting a local admin who signed in through LDAP.
        """
        from datetime import datetime

        sync = bool(getattr(settings, "LDAP_SYNC_ON_LOGIN", True))
        groups = list(getattr(ldap_user, "groups", None) or [])
        mapped_role = ldap_service.map_role(groups)

        user = existing
        if user is None and getattr(ldap_user, "email", None):
            # Avoid a duplicate for someone already known by their email.
            user = await self.get_user_by_email(ldap_user.email, db)

        if user is None:
            import secrets

            username = (ldap_user.username or "").strip()
            email = (ldap_user.email or "").strip()
            if not username:
                raise ValueError("LDAP entry has no username attribute")
            if not email:
                raise ValueError(
                    "LDAP user has no email and LDAP_DEFAULT_EMAIL_DOMAIN is not set"
                )

            user = User(
                username=username,
                email=email,
                full_name=getattr(ldap_user, "full_name", None),
                # Never used: an LDAP-managed account is checked against the
                # directory. Random so that it cannot be guessed either.
                hashed_password=await asyncio.to_thread(
                    self.hash_password, secrets.token_urlsafe(32)
                ),
                is_active=True,
                is_verified=True,
                role=mapped_role or default_role,
                auth_provider="ldap",
                auth_subject=getattr(ldap_user, "dn", None),
                auth_metadata={"groups": groups},
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
            return user

        if sync:
            new_email = (getattr(ldap_user, "email", None) or "").strip()
            if new_email and new_email.lower() != (user.email or "").lower():
                # Emails are unique: taking one another account holds would
                # fail the whole login at commit.
                holder = await self.get_user_by_email(new_email, db)
                if holder is None or holder.id == user.id:
                    user.email = new_email
                else:
                    logger.warning(
                        f"LDAP email {new_email} for {user.username} belongs to "
                        f"another account ({holder.username}); kept the old one"
                    )
            if getattr(ldap_user, "full_name", None):
                user.full_name = ldap_user.full_name

        user.auth_provider = "ldap"
        user.auth_subject = getattr(ldap_user, "dn", None)
        user.auth_metadata = {"groups": groups}
        if mapped_role is not None:
            user.role = mapped_role
        user.is_verified = True
        user.updated_at = datetime.utcnow()

        await db.commit()
        await db.refresh(user)
        return user

    async def get_current_user(
        self,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(
            HTTPBearer(auto_error=False)
        ),
        api_key: Optional[str] = Header(None, alias="X-API-Key"),
        db: AsyncSession = Depends(get_db),
        request: Request = None,
    ) -> User:
        """
        Get current user from JWT token or API key.

        Supports two authentication methods:
        1. JWT Bearer token: Authorization: Bearer <token>
        2. API Key: X-API-Key: <api_key>
        """
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

        # Try API key authentication first (if header is present)
        if api_key:
            user = await self._authenticate_with_api_key(api_key, db, request)
            if user:
                return user
            # If API key was provided but invalid, raise error
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

        # Try JWT token authentication
        if credentials:
            user_id = decode_access_token(credentials.credentials)
            if user_id is None:
                raise credentials_exception

            user = await self.get_user_by_id(user_id, db)
            if user is None:
                raise credentials_exception

            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User account is disabled",
                )

            return user

        # No authentication provided
        raise credentials_exception

    async def _authenticate_with_api_key(
        self,
        api_key: str,
        db: AsyncSession,
        request: Optional[Request] = None,
    ) -> Optional[User]:
        """Authenticate using an API key."""
        from app.services.api_key_service import api_key_service

        result = await api_key_service.validate_api_key(db, api_key)
        if not result:
            return None

        key_obj, user = result

        # Update usage statistics
        ip_address = None
        user_agent = None
        endpoint = None
        method = None

        if request:
            ip_address = request.client.host if request.client else None
            user_agent = request.headers.get("user-agent")
            endpoint = str(request.url.path)
            method = request.method

        await api_key_service.update_usage(
            db=db,
            api_key=key_obj,
            ip_address=ip_address,
            endpoint=endpoint,
            method=method,
            user_agent=user_agent,
        )

        logger.debug(
            f"API key authentication successful: {key_obj.key_prefix}... for user {user.username}"
        )
        return user

    async def require_admin(self, current_user: User) -> User:
        """Require admin privileges."""
        return ensure_admin(current_user)

    async def update_password(
        self, user: User, current_password: str, new_password: str, db: AsyncSession
    ) -> bool:
        """Update user password."""
        if not await asyncio.to_thread(
            self.verify_password, current_password, user.hashed_password
        ):
            return False

        user.hashed_password = await asyncio.to_thread(self.hash_password, new_password)
        user.updated_at = datetime.utcnow()

        await db.commit()
        logger.info(f"Password updated for user {user.username}")
        return True


# Global instance for dependency injection
auth_service = AuthService()


def is_admin(user: Optional[User]) -> bool:
    """Whether this user is an administrator. The one place that decides.

    This was answered in about a dozen ways: the model's method, a raw
    comparison of the role string in seven places, three private helpers and
    a second ``require_admin``. One of the variants named the method without
    calling it -- a bound method, always truthy -- and it guarded three
    pipeline routes that therefore let any signed-in user read or restart
    anyone's run. A check that exists in one form cannot be written wrongly
    in another; ``tests/test_one_admin_check.py`` refuses the other forms.

    ``None`` is not an administrator, so a caller holding an optional user
    needs no guard of its own.
    """
    return bool(user is not None and user.is_admin())


def ensure_admin(
    user: Optional[User], detail: str = "Admin privileges required"
) -> User:
    """``user``, or a 403. For a handler that already has the user in hand.

    Use ``Depends(require_admin)`` when the whole route is for administrators;
    this is for the route that is open to everyone and has one admin-only
    branch.
    """
    if not is_admin(user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)
    return user


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
    api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
    request: Request = None,
) -> User:
    """
    Dependency for getting current user.

    Supports both JWT Bearer token and API key authentication:
    - JWT: Authorization: Bearer <token>
    - API Key: X-API-Key: <api_key>
    """
    return await auth_service.get_current_user(credentials, api_key, db, request)


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    ),
    api_key: Optional[str] = Header(None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
    request: Request = None,
) -> Optional[User]:
    """
    Dependency for getting current user optionally (returns None if not authenticated).

    Useful for endpoints that work both authenticated and anonymously.
    """
    try:
        return await auth_service.get_current_user(credentials, api_key, db, request)
    except HTTPException:
        return None


async def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Dependency for requiring admin privileges."""
    return await auth_service.require_admin(current_user)


async def require_scope(scope: str):
    """
    Factory for creating a dependency that requires a specific API key scope.

    Usage:
        @router.get("/protected")
        async def protected_endpoint(user: User = Depends(require_scope("documents"))):
            ...
    """

    async def _require_scope(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(
            HTTPBearer(auto_error=False)
        ),
        api_key: Optional[str] = Header(None, alias="X-API-Key"),
        db: AsyncSession = Depends(get_db),
        request: Request = None,
    ) -> User:
        # If using API key, verify scope
        if api_key:
            from app.services.api_key_service import api_key_service

            result = await api_key_service.validate_api_key(
                db, api_key, required_scope=scope
            )
            if not result:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"API key lacks required scope: {scope}",
                )
            key_obj, user = result

            # Update usage
            if request:
                await api_key_service.update_usage(
                    db=db,
                    api_key=key_obj,
                    ip_address=request.client.host if request.client else None,
                    endpoint=str(request.url.path),
                    method=request.method,
                    user_agent=request.headers.get("user-agent"),
                )

            return user

        # JWT tokens have full access (scopes don't apply)
        return await auth_service.get_current_user(credentials, api_key, db, request)

    return _require_scope
