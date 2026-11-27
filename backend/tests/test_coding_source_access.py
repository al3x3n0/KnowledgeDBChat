from uuid import uuid4

import pytest

from app.models.coding_swarm_profile import CodingSwarmProfile
from app.models.document import DocumentSource
from app.services.auth_service import AuthService
from app.services.coding_source_access import can_access_coding_source


@pytest.mark.asyncio
async def test_only_the_source_owner_can_grant_repository_access(db_session, test_user):
    owner = await AuthService().create_user(
        username="repo-owner",
        email="repo-owner@example.com",
        password="testpassword123",
        full_name="Repository Owner",
        db=db_session,
    )
    source = DocumentSource(
        name=f"private-{uuid4()}",
        source_type="github",
        config={"requested_by_user_id": str(owner.id)},
    )
    db_session.add(source)
    await db_session.flush()
    profile = CodingSwarmProfile(
        user_id=test_user.id,
        source_id=source.id,
        title="Profile",
        preset_key="bug_triage_swarm",
        visibility="shared",
        shared_with_user_ids=[str(test_user.id)],
    )
    db_session.add(profile)
    await db_session.commit()
    assert not await can_access_coding_source(db_session, test_user, source.id)

    profile.user_id = owner.id
    await db_session.commit()
    assert await can_access_coding_source(db_session, test_user, source.id)
    assert not await can_access_coding_source(
        db_session, test_user, source.id, write=True
    )
    assert await can_access_coding_source(db_session, owner, source.id, write=True)
