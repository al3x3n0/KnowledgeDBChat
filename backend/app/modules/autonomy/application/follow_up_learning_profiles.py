"""Load and normalize learned follow-up recommendation preferences."""

from typing import Any, Awaitable, Callable
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

ProfileLoader = Callable[..., Awaitable[Any]]
PROFILE_FIELDS = (
    "token_scores",
    "phrase_scores",
    "recommendation_scores",
    "source_type_scores",
    "outcome_counters",
)


async def load_follow_up_learning_profile(
    *,
    db: AsyncSession,
    user_id: UUID,
    customer: str | None,
    profile_loader: ProfileLoader | None = None,
) -> dict[str, Any]:
    """Return a stable empty-or-normalized learning profile contract."""
    loader = profile_loader or _load_profile
    try:
        profile = await loader(
            db=db,
            user_id=user_id,
            customer=customer,
        )
    except Exception:
        profile = None
    if profile is None:
        return empty_follow_up_learning_profile()
    return {
        field: value if isinstance(value := getattr(profile, field, None), dict) else {}
        for field in PROFILE_FIELDS
    }


def empty_follow_up_learning_profile() -> dict[str, dict[str, Any]]:
    """Build a new empty profile so callers can mutate it safely."""
    return {field: {} for field in PROFILE_FIELDS}


async def _load_profile(
    *,
    db: AsyncSession,
    user_id: UUID,
    customer: str | None,
) -> Any:
    from app.services.research_monitor_profile_service import (
        research_monitor_profile_service,
    )

    return await research_monitor_profile_service.get_profile(
        db=db,
        user_id=user_id,
        customer=customer,
    )
