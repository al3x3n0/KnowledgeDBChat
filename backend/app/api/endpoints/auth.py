"""
Authentication-related API endpoints.
"""

from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from jose import JWTError, jwt
from passlib.context import CryptContext
from loguru import logger

from app.core.database import get_db
from app.core.config import settings
from app.core.rate_limit import limiter, AUTH_LIMIT
from app.models.user import User
from app.services.auth_service import AuthService
from app.schemas.auth import (
    UserLogin,
    UserRegister,
    UserResponse,
    TokenResponse
)

router = APIRouter()
auth_service = AuthService()
security = HTTPBearer()


@router.post("/register", response_model=UserResponse)
@limiter.limit(AUTH_LIMIT)
async def register(
    request: Request,
    user_data: UserRegister,
    db: AsyncSession = Depends(get_db)
):
    """Register a new user."""
    try:
        user = await auth_service.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
            db=db
        )
        return UserResponse.from_orm(user)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")


@router.post("/login", response_model=TokenResponse)
@limiter.limit(AUTH_LIMIT)
async def login(
    request: Request,
    user_data: UserLogin,
    db: AsyncSession = Depends(get_db)
):
    """Login user and return access token."""
    try:
        user = await auth_service.authenticate_user(
            username=user_data.username,
            password=user_data.password,
            db=db
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password"
            )
        
        access_token = auth_service.create_access_token(user.id)
        
        # Update last login
        user.last_login = datetime.utcnow()
        user.login_count += 1
        await db.commit()
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user=UserResponse.from_orm(user)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(auth_service.get_current_user)
):
    """Get current user information."""
    return UserResponse.from_orm(current_user)


@router.post("/logout")
async def logout():
    """Logout user (client-side token removal)."""
    return {"message": "Successfully logged out"}


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
):
    """Refresh access token."""
    try:
        # Verify current token
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Get user
        user = await auth_service.get_user_by_id(user_id, db)
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        
        # Create new token
        new_token = auth_service.create_access_token(user.id)
        
        return TokenResponse(
            access_token=new_token,
            token_type="bearer",
            user=UserResponse.from_orm(user)
        )
    
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(status_code=500, detail="Token refresh failed")


