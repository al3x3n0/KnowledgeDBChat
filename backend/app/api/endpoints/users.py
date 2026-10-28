"""
User management API endpoints.
"""

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.logging import log_error
from app.models.memory import UserPreferences
from app.models.user import User
from app.schemas.auth import PasswordChange, UserResponse
from app.schemas.common import PaginatedResponse
from app.schemas.memory import UserPreferencesResponse, UserPreferencesUpdate
from app.services import llm_provider_catalog as provider_catalog
from app.services.auth_service import auth_service, get_current_user, require_admin
from app.services.collaboration_service import (
    list_collaboration_users as load_collaboration_users,
)
from app.services.llm_service import LLMService
from app.services.ui_preferences import normalize as normalize_ui_preferences
from app.utils.exceptions import ValidationError

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


class CustomProviderIn(BaseModel):
    """A saved provider as the page sends it. ``api_key`` is sent only to
    set or replace the key; leaving it out keeps the one already saved."""

    id: Optional[str] = Field(None, max_length=32)
    name: str = Field(..., max_length=60)
    kind: str = Field("openai_compatible", max_length=32)
    api_url: str = Field(..., max_length=500)
    default_model: Optional[str] = Field(None, max_length=100)
    models: List[str] = Field(default_factory=list, max_length=50)
    api_key: Optional[str] = Field(None, max_length=500)
    clear_api_key: bool = False


class CustomProviderOut(BaseModel):
    id: str
    name: str
    kind: str
    api_url: str
    default_model: Optional[str] = None
    models: List[str] = Field(default_factory=list)
    api_key_set: bool = False


class LLMSettingsUpdate(BaseModel):
    """Schema for updating only LLM settings."""

    llm_provider: Optional[str] = Field(
        None, description="LLM provider: ollama, deepseek, openai, or custom URL"
    )
    llm_model: Optional[str] = Field(
        None, description="Default model name (used for chat)"
    )
    llm_api_url: Optional[str] = Field(None, description="Custom API URL override")
    llm_api_key: Optional[str] = Field(
        None, description="API key for external providers"
    )
    llm_temperature: Optional[float] = Field(
        None, ge=0.0, le=2.0, description="Temperature (0.0-2.0)"
    )
    llm_max_tokens: Optional[int] = Field(
        None, ge=1, le=32000, description="Max response tokens"
    )
    llm_task_models: Optional[Dict[str, str]] = Field(
        None, description="Per-task model overrides: {task_type: model_name}"
    )
    llm_task_providers: Optional[Dict[str, str]] = Field(
        None, description="Per-task provider overrides: {task_type: provider}"
    )
    llm_providers: Optional[List[CustomProviderIn]] = Field(
        None,
        description="Saved custom providers, named elsewhere as custom:<id>",
        max_length=provider_catalog.MAX_CUSTOM_PROVIDERS,
    )


class LLMSettingsResponse(BaseModel):
    """Schema for LLM settings response."""

    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_api_url: Optional[str] = None
    llm_api_key_set: bool = False  # Don't return the actual key, just whether it's set
    llm_temperature: Optional[float] = None
    llm_max_tokens: Optional[int] = None
    llm_task_models: Optional[Dict[str, str]] = None  # Per-task model overrides
    llm_task_providers: Optional[Dict[str, str]] = None  # Per-task provider overrides
    llm_providers: List[CustomProviderOut] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class LLMModelsResponse(BaseModel):
    provider: str
    models: List[str] = Field(default_factory=list)
    default_model: Optional[str] = None
    error: Optional[str] = None


class BuiltinProviderOut(BaseModel):
    id: str
    label: str
    api_url: Optional[str] = None
    default_model: Optional[str] = None
    models: List[str] = Field(default_factory=list)
    system_configured: bool = False
    requires_api_key: bool = True


class ProviderTemplateOut(BaseModel):
    id: str
    label: str
    kind: str
    api_url: str
    requires_api_key: bool = False
    description: str = ""


class LLMProviderCatalogResponse(BaseModel):
    system_provider: str
    system_model: Optional[str] = None
    builtin: List[BuiltinProviderOut]
    templates: List[ProviderTemplateOut]
    kinds: List[str]


class ProviderTestRequest(BaseModel):
    kind: str = Field("openai_compatible", max_length=32)
    api_url: str = Field(..., max_length=500)
    api_key: Optional[str] = Field(None, max_length=500)
    # A saved provider whose key to use when none is sent; only if the URL
    # is still the saved one, so a saved key never goes somewhere new.
    id: Optional[str] = Field(None, max_length=32)


class ProviderTestResponse(BaseModel):
    ok: bool
    models: List[str] = Field(default_factory=list)
    error: Optional[str] = None


def _llm_settings_response(preferences: UserPreferences) -> LLMSettingsResponse:
    return LLMSettingsResponse(
        llm_provider=preferences.llm_provider,
        llm_model=preferences.llm_model,
        llm_api_url=preferences.llm_api_url,
        llm_api_key_set=bool(preferences.llm_api_key),
        llm_temperature=preferences.llm_temperature,
        llm_max_tokens=preferences.llm_max_tokens,
        llm_task_models=preferences.llm_task_models,
        llm_task_providers=getattr(preferences, "llm_task_providers", None),
        llm_providers=provider_catalog.public_view(
            getattr(preferences, "llm_providers", None)
        ),
    )


def _system_model(provider: str) -> Optional[str]:
    if provider == "ollama":
        return settings.DEFAULT_MODEL
    for p in provider_catalog.builtin_providers():
        if p["id"] == provider:
            return p["default_model"]
    return settings.DEFAULT_MODEL


async def _own_preferences(db: AsyncSession, user: User) -> Optional[UserPreferences]:
    result = await db.execute(
        select(UserPreferences).where(UserPreferences.user_id == user.id)
    )
    return result.scalar_one_or_none()


router = APIRouter()


@router.get("/collaboration", response_model=PaginatedResponse[UserResponse])
async def list_collaboration_users(
    page: int = 1,
    page_size: int = 20,
    search: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List active users for lightweight collaboration pickers."""
    try:
        if page < 1:
            raise ValidationError("Page must be >= 1", field="page")
        if page_size < 1 or page_size > 100:
            raise ValidationError(
                "Page size must be between 1 and 100", field="page_size"
            )

        skip = (page - 1) * page_size
        users = await load_collaboration_users(
            db, current_user=current_user, search=search
        )
        total = len(users)
        users = users[skip : skip + page_size]
        items = [UserResponse.from_orm(user) for user in users]
        return PaginatedResponse.create(
            items=items, total=total, page=page, page_size=page_size
        )
    except ValidationError:
        raise
    except Exception as e:
        log_error(e, context={"page": page, "page_size": page_size, "search": search})
        raise HTTPException(
            status_code=500, detail="Failed to retrieve collaboration users"
        )


@router.get("/", response_model=PaginatedResponse[UserResponse])
async def get_users(
    page: int = 1,
    page_size: int = 20,
    search: Optional[str] = None,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
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
            raise ValidationError(
                "Page size must be between 1 and 100", field="page_size"
            )

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
        query = (
            base_query.order_by(User.created_at.desc()).offset(skip).limit(page_size)
        )
        result = await db.execute(query)
        users = result.scalars().all()

        # Convert to response models
        items = [UserResponse.from_orm(user) for user in users]

        # Return paginated response
        return PaginatedResponse.create(
            items=items, total=total, page=page, page_size=page_size
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
    db: AsyncSession = Depends(get_db),
):
    """Change current user's password."""
    if getattr(current_user, "auth_provider", "local") == "ldap":
        # The password that signs this account in is the directory's. One set
        # here is never checked, except by the outage fallback
        # (LDAP_LOCAL_FALLBACK_WHEN_UNAVAILABLE), where it would outlive the
        # account being disabled in the directory.
        raise HTTPException(
            status_code=400,
            detail="This account's password is managed by the directory; "
            "change it there.",
        )
    try:
        success = await auth_service.update_password(
            user=current_user,
            current_password=password_data.current_password,
            new_password=password_data.new_password,
            db=db,
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
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
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
    db: AsyncSession = Depends(get_db),
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

        # `ui` is the one field whose value is a free-form document the
        # client composes, so it is normalized rather than stored as sent:
        # unknown keys dropped, lists capped, labels trimmed. Every other
        # field here is a typed scalar Pydantic has already bounded.
        if "ui" in update_data:
            update_data["ui"] = normalize_ui_preferences(update_data["ui"])
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
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
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

        return _llm_settings_response(preferences)
    except Exception as e:
        logger.error(f"Error getting LLM settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to get LLM settings")


@router.put("/me/llm-settings", response_model=LLMSettingsResponse)
async def update_my_llm_settings(
    updates: LLMSettingsUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
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
        if "llm_providers" in update_data:
            try:
                update_data["llm_providers"] = (
                    provider_catalog.normalize_custom_providers(
                        updates.llm_providers,
                        getattr(preferences, "llm_providers", None),
                    )
                    or None
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
        for field, value in update_data.items():
            setattr(preferences, field, value)

        # Every provider choice must name something, checked on the state
        # this update leaves: deleting a saved provider and moving off it
        # arrive in one request.
        task_providers = getattr(preferences, "llm_task_providers", None) or {}
        unknown = provider_catalog.unknown_references(
            [preferences.llm_provider, *task_providers.values()],
            getattr(preferences, "llm_providers", None),
        )
        if unknown:
            await db.rollback()
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unknown provider: "
                    + ", ".join(sorted(set(unknown)))
                    + ". Choose a built-in provider or a saved one."
                ),
            )

        await db.commit()
        await db.refresh(preferences)

        return _llm_settings_response(preferences)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating LLM settings: {e}")
        raise HTTPException(status_code=500, detail="Failed to update LLM settings")


@router.delete("/me/llm-settings")
async def clear_my_llm_settings(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
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
    provider: Optional[str] = Query(
        None, description="Provider to list models for (overrides saved preference)"
    ),
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

        effective_provider = (
            provider
            or getattr(preferences, "llm_provider", None)
            or settings.LLM_PROVIDER
            or "ollama"
        ).lower()

        if effective_provider.startswith(provider_catalog.CUSTOM_PREFIX):
            stored = getattr(preferences, "llm_providers", None)
            saved = provider_catalog.find_custom(stored, effective_provider)
            if saved is None:
                return LLMModelsResponse(
                    provider=effective_provider,
                    error="No saved provider by that name",
                )
            listed = await provider_catalog.list_models(
                str(saved.get("kind")),
                str(saved.get("api_url")),
                provider_catalog.decrypt_key(saved),
            )
            names = list(saved.get("models") or [])
            names += [m for m in listed["models"] if m not in names]
            return LLMModelsResponse(
                provider=effective_provider,
                models=names,
                default_model=saved.get("default_model"),
                error=listed["error"],
            )

        if effective_provider != "ollama":
            for p in provider_catalog.builtin_providers():
                if p["id"] == effective_provider:
                    return LLMModelsResponse(
                        provider=effective_provider,
                        models=p["models"],
                        default_model=p["default_model"],
                    )
            return LLMModelsResponse(
                provider=effective_provider,
                models=[],
                default_model=None,
            )

        svc = LLMService()
        models = await svc.list_available_models()
        names = sorted(
            {m.get("name") for m in models if isinstance(m, dict) and m.get("name")}
        )
        return LLMModelsResponse(
            provider="ollama",
            models=names,
            default_model=svc.default_model,
        )
    except Exception as e:
        logger.error(f"Error listing user LLM models: {e}")
        return LLMModelsResponse(
            provider="unknown",
            models=[],
            default_model=None,
            error="Failed to list models",
        )


@router.get("/me/llm-providers", response_model=LLMProviderCatalogResponse)
async def get_llm_provider_catalog(
    current_user: User = Depends(get_current_user),
):
    """What can be chosen: the built-in providers (and whether the
    deployment holds a key for each) and templates for a custom one."""
    system_provider = (settings.LLM_PROVIDER or "deepseek").lower()
    return LLMProviderCatalogResponse(
        system_provider=system_provider,
        system_model=_system_model(system_provider),
        builtin=provider_catalog.builtin_providers(),
        templates=provider_catalog.TEMPLATES,
        kinds=list(provider_catalog.CUSTOM_KINDS),
    )


@router.post("/me/llm-providers/test", response_model=ProviderTestResponse)
async def test_llm_provider(
    body: ProviderTestRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Ask an endpoint which models it serves, before or after saving it.

    Only model names come back, never the server's reply.
    """
    if body.kind not in provider_catalog.CUSTOM_KINDS:
        raise HTTPException(status_code=400, detail=f"Unknown kind {body.kind!r}")
    api_key = body.api_key
    if not api_key and body.id:
        preferences = await _own_preferences(db, current_user)
        saved = provider_catalog.find_custom(
            getattr(preferences, "llm_providers", None),
            provider_catalog.CUSTOM_PREFIX + body.id,
        )
        if saved and str(saved.get("api_url")) == body.api_url.rstrip("/"):
            api_key = provider_catalog.decrypt_key(saved)
    result = await provider_catalog.list_models(body.kind, body.api_url, api_key)
    return ProviderTestResponse(**result)
