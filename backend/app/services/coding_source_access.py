"""Repository source permissions shared by the coding workflow."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.coding_swarm_profile import CodingSwarmProfile
from app.models.document import DocumentSource
from app.models.user import User
from app.services.agent_swarm_collaboration_service import is_profile_visible_to_user


def is_source_owned_by_user(source: DocumentSource, user: User) -> bool:
    config = source.config if isinstance(source.config, dict) else {}
    owner_id = config.get("requested_by_user_id") or config.get("requestedByUserId")
    if owner_id:
        return str(owner_id) == str(user.id)
    owner = config.get("requested_by") or config.get("requestedBy")
    return owner in {user.username, str(user.id)}


async def can_access_coding_source(
    db: AsyncSession,
    user: User,
    source_id: UUID,
    *,
    write: bool = False,
) -> bool:
    source = await db.get(DocumentSource, source_id)
    if source is None or not user.is_active:
        return False
    if user.is_admin() or is_source_owned_by_user(source, user):
        return True
    if write:
        return False
    profiles = await db.execute(
        select(CodingSwarmProfile, User)
        .join(User, User.id == CodingSwarmProfile.user_id)
        .where(
            CodingSwarmProfile.source_id == source_id,
            CodingSwarmProfile.status == "active",
            CodingSwarmProfile.visibility == "shared",
            User.is_active.is_(True),
        )
    )
    return any(
        is_profile_visible_to_user(profile, user)
        and (owner.is_admin() or is_source_owned_by_user(source, owner))
        for profile, owner in profiles
    )
