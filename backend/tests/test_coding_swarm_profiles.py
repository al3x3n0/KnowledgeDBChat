from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from app.api.endpoints.coding_swarm_profiles import _profile_to_response


def test_profile_to_response_includes_collaboration_summary():
    owner_id = uuid4()
    shared_user_id = uuid4()
    profile = SimpleNamespace(
        id=uuid4(),
        user_id=owner_id,
        source_id=uuid4(),
        title="Bug Triage Default",
        description="Default repo profile",
        status="active",
        preset_key="bug_triage_swarm",
        scope_default="frontend",
        default_commands=["CI=true npm --prefix frontend test -- --watchAll=false"],
        default_file_paths=["frontend/src/pages/DocumentsPage.tsx"],
        max_agents=4,
        safe_command_policy="standard",
        saved_search_query="save regression",
        is_default=True,
        visibility="shared",
        shared_with_user_ids=[shared_user_id],
        latest_job_id=None,
        profile_metadata={},
        created_at=datetime(2026, 3, 10, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 10, tzinfo=timezone.utc),
    )
    current_user = SimpleNamespace(id=owner_id)
    user_lookup = {
        str(owner_id): SimpleNamespace(
            id=owner_id,
            full_name="Repo Owner",
            username="owner",
            email="owner@example.com",
        )
    }

    response = _profile_to_response(
        profile, current_user=current_user, user_lookup=user_lookup
    )

    assert response.visibility == "shared"
    assert [str(user_id) for user_id in response.shared_with_user_ids] == [
        str(shared_user_id)
    ]
    assert response.collaboration_summary is not None
    assert response.collaboration_summary.owner_user_id == owner_id
    assert response.collaboration_summary.owner_label == "Repo Owner"
    assert response.collaboration_summary.visibility_scope == "shared"
    assert response.collaboration_summary.is_owned_by_current_user is True
