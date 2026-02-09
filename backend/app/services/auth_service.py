"""
Authentication service for user management and JWT tokens.

Supports both JWT token authentication and API key authentication for external tools.
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple
from uuid import UUID
import hashlib
import bcrypt
from fastapi import Depends, HTTPException, status, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from jose import JWTError, jwt
from passlib.context import CryptContext
from loguru import logger

from app.core.database import get_db
from app.core.config import settings
from app.models.user import User


class AuthService:
    """Service for authentication and authorization."""

    # API key header name
    API_KEY_HEADER = "X-API-Key"

    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.security = HTTPBearer(auto_error=False)  # Don't auto-error to allow API key fallback
        self.api_key_header = APIKeyHeader(name=self.API_KEY_HEADER, auto_error=False)
    
    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        # Ensure password is bytes for bcrypt
        if isinstance(password, str):
            password_bytes = password.encode('utf-8')
        else:
            password_bytes = password
        
        # Bcrypt has a 72-byte limit - handle long passwords
        if len(password_bytes) > 72:
            logger.debug(f"Password exceeds 72 bytes ({len(password_bytes)}), pre-hashing with SHA256")
            # Pre-hash with SHA256 to reduce to fixed 64 bytes
            password_hash = hashlib.sha256(password_bytes).hexdigest()
            password_bytes = password_hash.encode('utf-8')
        
        # Use bcrypt directly to avoid passlib validation issues
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password_bytes, salt)
        # Return as string (bcrypt returns bytes)
        return hashed.decode('utf-8')
    
    def get_password_hash(self, password: str) -> str:
        """Alias for hash_password for compatibility."""
        return self.hash_password(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        # Ensure password is bytes for bcrypt
        if isinstance(plain_password, str):
            password_bytes = plain_password.encode('utf-8')
        else:
            password_bytes = plain_password
        
        # Ensure hashed_password is bytes
        if isinstance(hashed_password, str):
            hashed_bytes = hashed_password.encode('utf-8')
        else:
            hashed_bytes = hashed_password
        
        # Try direct verification first
        if bcrypt.checkpw(password_bytes, hashed_bytes):
            return True
        
        # If that fails and password is > 72 bytes, try with pre-hash
        if len(password_bytes) > 72:
            password_hash = hashlib.sha256(password_bytes).hexdigest().encode('utf-8')
            return bcrypt.checkpw(password_hash, hashed_bytes)
        
        return False
    
    def create_access_token(self, user_id: UUID) -> str:
        """Create a JWT access token."""
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode = {
            "sub": str(user_id),
            "exp": expire,
            "type": "access"
        }
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt
    
    async def get_user_by_username(self, username: str, db: AsyncSession) -> Optional[User]:
        """Get user by username."""
        result = await db.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()
    
    async def get_user_by_email(self, email: str, db: AsyncSession) -> Optional[User]:
        """Get user by email."""
        result = await db.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()
    
    async def get_user_by_id(self, user_id: str, db: AsyncSession) -> Optional[User]:
        """Get user by ID."""
        try:
            user_uuid = UUID(user_id)
            result = await db.execute(
                select(User).where(User.id == user_uuid)
            )
            return result.scalar_one_or_none()
        except ValueError:
            return None
    
    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        db: AsyncSession,
        full_name: Optional[str] = None
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
        hashed_password = self.hash_password(password)
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
            is_active=True,
            is_verified=False,
            role="user"
        )
        
        db.add(user)
        await db.commit()
        await db.refresh(user)
        
        logger.info(f"Created new user: {username}")
        return user
    
    async def authenticate_user(
        self,
        username: str,
        password: str,
        db: AsyncSession
    ) -> Optional[User]:
        """Authenticate a user with username and password."""
        user = await self.get_user_by_username(username, db)
        
        if not user:
            return None
        
        if not user.is_active:
            return None
        
        if not self.verify_password(password, user.hashed_password):
            return None
        
        return user
    
    async def get_current_user(
        self,
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
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
            try:
                payload = jwt.decode(
                    credentials.credentials,
                    settings.SECRET_KEY,
                    algorithms=[settings.ALGORITHM]
                )
                user_id: str = payload.get("sub")
                if user_id is None:
                    raise credentials_exception
            except JWTError:
                raise credentials_exception

            user = await self.get_user_by_id(user_id, db)
            if user is None:
                raise credentials_exception

            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User account is disabled"
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

        logger.debug(f"API key authentication successful: {key_obj.key_prefix}... for user {user.username}")
        return user
    
    async def require_admin(
        self,
        current_user: User
    ) -> User:
        """Require admin privileges."""
        if not current_user.is_admin():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin privileges required"
            )
        return current_user
    
    async def update_password(
        self,
        user: User,
        current_password: str,
        new_password: str,
        db: AsyncSession
    ) -> bool:
        """Update user password."""
        if not self.verify_password(current_password, user.hashed_password):
            return False
        
        user.hashed_password = self.hash_password(new_password)
        user.updated_at = datetime.utcnow()
        
        await db.commit()
        logger.info(f"Password updated for user {user.username}")
        return True


# Global instance for dependency injection
auth_service = AuthService()


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
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
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
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


async def require_admin(
    current_user: User = Depends(get_current_user)
) -> User:
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
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
        api_key: Optional[str] = Header(None, alias="X-API-Key"),
        db: AsyncSession = Depends(get_db),
        request: Request = None,
    ) -> User:
        # If using API key, verify scope
        if api_key:
            from app.services.api_key_service import api_key_service

            result = await api_key_service.validate_api_key(db, api_key, required_scope=scope)
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
