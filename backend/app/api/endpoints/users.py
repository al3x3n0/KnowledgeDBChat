"""
User management API endpoints.
"""

from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from loguru import logger
from pydantic import BaseModel, Field

from app.core.database import get_db
from app.models.user import User
from app.models.memory import UserPreferences
from app.services.auth_service import get_current_user, require_admin, auth_service
from app.schemas.auth import UserResponse, PasswordChange
from app.schemas.memory import UserPreferencesResponse, UserPreferencesUpdate
from app.schemas.common import PaginatedResponse
from app.utils.exceptions import ValidationError
from app.core.logging import log_error
from app.services.llm_service import LLMService
from app.core.config import settings


# Supported task types for per-task model configuration
LLM_TASK_TYPES = [
    "chat",
    "title_generation",
    "summarization",
    "query_expansion",
    "memory_extraction",
    "workflow_synthesis",
    # Agent / jobs
    "code_agent",
    "research_engineer_scientist",
    "latex_reviewer_critic",
    # Knowledge graph / extraction
    "knowledge_extraction",
    # Presentation generation
    "presentation_outline",
    "presentation_diagram",
    "presentation_slide",
]


class LLMSettingsUpdate(BaseModel):
    """Schema for updating only LLM settings."""
    llm_provider: Optional[str] = Field(None, description="LLM provider: ollama, deepseek, openai, or custom URL")
    llm_model: Optional[str] = Field(None, description="Default model name (used for chat)")
    llm_api_url: Optional[str] = Field(None, description="Custom API URL override")
    llm_api_key: Optional[str] = Field(None, description="API key for external providers")
    llm_temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Temperature (0.0-2.0)")
    llm_max_tokens: Optional[int] = Field(None, ge=1, le=32000, description="Max response tokens")
    llm_task_models: Optional[dict] = Field(
        None,
        description="Per-task model overrides: {task_type: model_name}"
    )
    llm_task_providers: Optional[dict] = Field(
        None,
        description="Per-task provider overrides: {task_type: provider}"
    )


class LLMSettingsResponse(BaseModel):
    """Schema for LLM settings response."""
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_url: Optional[str] = None
    llm_api_key_set: bool = False  # Don't return the actual key, just whether it's set
    llm_temperature: Optional[float] = None
    llm_max_tokens: Optional[int] = None
    llm_task_models: Optional[dict] = None  # Per-task model overrides
    llm_task_providers: Optional[dict] = None  # Per-task provider overrides

    class Config:
        from_attributes = True


class LLMModelsResponse(BaseModel):
    provider: str
    models: List[str] = Field(default_factory=list)
    default_model: Optional[str] = None
    error: Optional[str] = None


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


@router.get("/me/preferences", response_model=UserPreferencesResponse)
async def get_my_preferences(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current user's preferences."""
    try:
        result = await db.execute(
            select(UserPreferences).where(UserPreferences.user_id == current_user.id)
        )
        preferences = result.scalar_one_or_none()

        if not preferences:
            # Create default preferences if they don't exist
            preferences = UserPreferences(user_id=current_user.id)
            db.add(preferences)
            await db.commit()
            await db.refresh(preferences)

        return preferences
    except Exception as e:
        logger.error(f"Error getting preferences: {e}")
        raise HTTPException(status_code=500, detail="Failed to get preferences")


@router.put("/me/preferences", response_model=UserPreferencesResponse)
async def update_my_preferences(
    updates: UserPreferencesUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update current user's preferences."""
    try:
        result = await db.execute(
            select(UserPreferences).where(UserPreferences.user_id == current_user.id)
        )
        preferences = result.scalar_one_or_none()

        if not preferences:
            # Create preferences if they don't exist
            preferences = UserPreferences(user_id=current_user.id)
            db.add(preferences)

        # Update only provided fields
        update_data = updates.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(preferences, field, value)

        await db.commit()
        await db.refresh(preferences)

        return preferences
    except Exception as e:
        logger.error(f"Error updating preferences: {e}")
        raise HTTPException(status_code=500, detail="Failed to update preferences")


@router.get("/me/llm-settings", response_model=LLMSettingsResponse)
async def get_my_llm_settings(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current user's LLM settings."""
    try:
        result = await db.execute(
            select(UserPreferences).where(UserPreferences.user_id == current_user.id)
        )
        preferences = result.scalar_one_or_none()

        if not preferences:
            # Return defaults (all None)
            return LLMSettingsResponse()

        return LLMSettingsResponse(
            llm_provider=preferences.llm_provider,
            llm_model=preferences.llm_model,
            llm_api_url=preferences.llm_api_url,
            llm_api_key_set=bool(preferences.llm_api_key),
            llm_temperature=preferences.llm_temperature,
            llm_max_tokens=preferences.llm_max_tokens,
            llm_task_models=preferences.llm_task_models,
            llm_task_providers=getattr(preferences, "llm_task_providers", None),
        )
    except Exception as e:
        logger.error(f"Error getting LLM settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to get LLM settings")


@router.put("/me/llm-settings", response_model=LLMSettingsResponse)
async def update_my_llm_settings(
    updates: LLMSettingsUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update current user's LLM settings."""
    try:
        result = await db.execute(
            select(UserPreferences).where(UserPreferences.user_id == current_user.id)
        )
        preferences = result.scalar_one_or_none()

        if not preferences:
            # Create preferences if they don't exist
            preferences = UserPreferences(user_id=current_user.id)
            db.add(preferences)

        # Update only provided fields
        update_data = updates.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(preferences, field, value)

        await db.commit()
        await db.refresh(preferences)

        return LLMSettingsResponse(
            llm_provider=preferences.llm_provider,
            llm_model=preferences.llm_model,
            llm_api_url=preferences.llm_api_url,
            llm_api_key_set=bool(preferences.llm_api_key),
            llm_temperature=preferences.llm_temperature,
            llm_max_tokens=preferences.llm_max_tokens,
            llm_task_models=preferences.llm_task_models,
            llm_task_providers=getattr(preferences, "llm_task_providers", None),
        )
    except Exception as e:
        logger.error(f"Error updating LLM settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to update LLM settings")


@router.delete("/me/llm-settings")
async def clear_my_llm_settings(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Clear current user's LLM settings (revert to system defaults)."""
    try:
        result = await db.execute(
            select(UserPreferences).where(UserPreferences.user_id == current_user.id)
        )
        preferences = result.scalar_one_or_none()

        if preferences:
            preferences.llm_provider = None
            preferences.llm_model = None
            preferences.llm_api_url = None
            preferences.llm_api_key = None
            preferences.llm_temperature = None
            preferences.llm_max_tokens = None
            preferences.llm_task_models = None
            preferences.llm_task_providers = None
            await db.commit()

        return {"message": "LLM settings cleared, using system defaults"}
    except Exception as e:
        logger.error(f"Error clearing LLM settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear LLM settings")


@router.get("/me/llm-models", response_model=LLMModelsResponse)
async def list_my_llm_models(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    provider: Optional[str] = Query(None, description="Provider to list models for (overrides saved preference)"),
):
    """
    List available models for the effective provider.

    Currently supports listing models from Ollama (`/api/tags`) when provider is `ollama`.
    """
    try:
        result = await db.execute(
            select(UserPreferences).where(UserPreferences.user_id == current_user.id)
        )
        preferences = result.scalar_one_or_none()

        effective_provider = (provider or getattr(preferences, "llm_provider", None) or settings.LLM_PROVIDER or "ollama").lower()

        if effective_provider != "ollama":
            return LLMModelsResponse(
                provider=effective_provider,
                models=[],
                default_model=None,
            )

        svc = LLMService()
        models = await svc.list_available_models()
        names = sorted({m.get("name") for m in models if isinstance(m, dict) and m.get("name")})
        return LLMModelsResponse(
            provider="ollama",
            models=names,
            default_model=svc.default_model,
        )
    except Exception as e:
        logger.error(f"Error listing user LLM models: {e}")
        return LLMModelsResponse(provider="unknown", models=[], default_model=None, error="Failed to list models")
