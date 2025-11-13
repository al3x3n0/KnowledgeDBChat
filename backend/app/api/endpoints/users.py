"""
User management API endpoints.
"""

from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from loguru import logger

from app.core.database import get_db
from app.models.user import User
from app.services.auth_service import get_current_user, require_admin, auth_service
from app.schemas.auth import UserResponse, PasswordChange
from app.schemas.common import PaginatedResponse
from app.utils.exceptions import ValidationError
from app.core.logging import log_error
from sqlalchemy import select, func

router = APIRouter()


@router.get("/", response_model=PaginatedResponse[UserResponse])
async def get_users(
    page: int = 1,
    page_size: int = 20,
    search: Optional[str] = None,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """
    Get paginated list of users (admin only).
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page (1-100)
        search: Optional search term for username or email
        current_user: Current authenticated admin user
        db: Database session
        
    Returns:
        Paginated response with users
    """
    try:
        # Validate pagination parameters
        if page < 1:
            raise ValidationError("Page must be >= 1", field="page")
        if page_size < 1 or page_size > 100:
            raise ValidationError("Page size must be between 1 and 100", field="page_size")
        
        # Calculate skip
        skip = (page - 1) * page_size
        
        # Build query
        base_query = select(User)
        
        # Apply search filter
        if search:
            search_term = f"%{search}%"
            base_query = base_query.where(
                (User.username.ilike(search_term)) | (User.email.ilike(search_term))
            )
        
        # Get total count
        count_query = select(func.count()).select_from(base_query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0
        
        # Get paginated results
        query = base_query.order_by(User.created_at.desc()).offset(skip).limit(page_size)
        result = await db.execute(query)
        users = result.scalars().all()
        
        # Convert to response models
        items = [UserResponse.from_orm(user) for user in users]
        
        # Return paginated response
        return PaginatedResponse.create(
            items=items,
            total=total,
            page=page,
            page_size=page_size
        )
    except ValidationError:
        raise
    except Exception as e:
        log_error(e, context={"page": page, "page_size": page_size})
        raise HTTPException(status_code=500, detail="Failed to retrieve users")


@router.put("/me/password")
async def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Change current user's password."""
    try:
        success = await auth_service.update_password(
            user=current_user,
            current_password=password_data.current_password,
            new_password=password_data.new_password,
            db=db
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Current password is incorrect")
        
        return {"message": "Password updated successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing password: {e}")
        raise HTTPException(status_code=500, detail="Failed to change password")
