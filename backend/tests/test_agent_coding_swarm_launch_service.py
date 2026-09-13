import hashlib
from uuid import uuid4

import pytest

from app.api.endpoints import agent_jobs
from app.models.agent_job import AgentJobStatus
from app.models.coding_swarm_profile import CodingSwarmProfile
from app.models.document import Document, DocumentSource
from app.schemas.agent_job import AgentJobQuickStartBugTriageSwarmRequest
from app.services.agent_coding_swarm_launch_service import (
    AgentCodingSwarmLaunchError,
    agent_coding_swarm_launch_service,
)


async def _create_repository_source(
    db_session,
    test_user,
    *,
    with_document: bool = True,
) -> DocumentSource:
    source = DocumentSource(
        name=f"Repository {uuid4().hex}",
        source_type="github",
        config={"requested_by_user_id": str(test_user.id)},
    )
    db_session.add(source)
    await db_session.flush()
    if with_document:
        content = "def failing_path():\n    return False\n"
        db_session.add(
            Document(
                title="backend/example.py",
                content=content,
                content_hash=hashlib.sha256(content.encode()).hexdigest(),
                source_id=source.id,
                source_identifier="backend/example.py",
            )
        )
    await db_session.commit()
    await db_session.refresh(source)
    return source


@pytest.mark.asyncio
async def test_launch_persists_coding_swarm_contract(
    db_session,
    test_user,
):
    source = await _create_repository_source(db_session, test_user)
    request = AgentJobQuickStartBugTriageSwarmRequest(
        source_id=source.id,
        failure_symptom="The repository tests fail in the parser.",
        start_immediately=False,
    )

    job = await agent_coding_swarm_launch_service.launch(
        request=request,
        db=db_session,
        current_user=test_user,
        preset_key="bug_triage_swarm",
    )

    assert job.status == AgentJobStatus.PENDING.value
    assert job.config["launch_mode"] == "quick_start_bug_triage_swarm"
    assert job.config["source_id"] == str(source.id)
    assert job.config["quick_start"]["preset_key"] == "bug_triage_swarm"
    assert job.config["swarm_child_jobs_enabled"] is True
    assert job.config["coding_harness"]["version"] == "v2"
    assert job.config["coding_harness"]["delegation"]["single_mutation_owner"] is True
    assert job.config["native_tool_loop"]["enabled"] is True
    assert job.config["verification_required_for_completion"] is True
    assert job.config["coding_workspace_session_id"] == f"coding-session-{job.id}"
    assert job.config["coding_workspace_session"]["job_id"] == str(job.id)
    assert job.enable_memory is False
    collaboration = job.results["swarm_collaboration"]
    assert collaboration["owner_user_id"] == str(test_user.id)
    assert collaboration["shared_review"] is False
    assert job.execution_log[0]["phase"] == "launch"


@pytest.mark.asyncio
async def test_launch_applies_profile_and_links_latest_job_after_flush(
    db_session,
    test_user,
):
    source = await _create_repository_source(db_session, test_user)
    shared_user_id = uuid4()
    profile = CodingSwarmProfile(
        user_id=test_user.id,
        source_id=source.id,
        title="Shared bug triage",
        status="active",
        preset_key="bug_triage_swarm",
        scope_default="backend",
        default_commands=["pytest -q backend/tests/test_parser.py"],
        default_file_paths=["backend/app/parser.py"],
        saved_search_query="parser regression",
        max_agents=4,
        safe_command_policy="standard",
        is_default=True,
        visibility="shared",
        shared_with_user_ids=[str(shared_user_id)],
    )
    db_session.add(profile)
    await db_session.commit()
    await db_session.refresh(profile)
    request = AgentJobQuickStartBugTriageSwarmRequest(
        source_id=source.id,
        profile_id=profile.id,
        failure_symptom="Parser regression",
        start_immediately=False,
    )

    job = await agent_coding_swarm_launch_service.launch(
        request=request,
        db=db_session,
        current_user=test_user,
        preset_key="bug_triage_swarm",
    )
    await db_session.refresh(profile)

    assert profile.latest_job_id == job.id
    assert job.config["commands"] == ["pytest -q backend/tests/test_parser.py"]
    assert job.config["file_paths"] == ["backend/app/parser.py"]
    assert job.config["search_query"] == "parser regression"
    collaboration = job.results["swarm_collaboration"]
    assert collaboration["shared_review"] is True
    assert collaboration["shared_with_user_ids"] == [str(shared_user_id)]


@pytest.mark.asyncio
async def test_launch_rejects_empty_repository_and_unsafe_profile_commands(
    db_session,
    test_user,
):
    empty_source = await _create_repository_source(
        db_session,
        test_user,
        with_document=False,
    )
    with pytest.raises(AgentCodingSwarmLaunchError) as empty_error:
        await agent_coding_swarm_launch_service.launch(
            request=AgentJobQuickStartBugTriageSwarmRequest(
                source_id=empty_source.id,
                failure_symptom="No indexed context",
                start_immediately=False,
            ),
            db=db_session,
            current_user=test_user,
            preset_key="bug_triage_swarm",
        )
    assert empty_error.value.status_code == 422
    assert "Source has no documents" in str(empty_error.value.detail)

    source = await _create_repository_source(db_session, test_user)
    profile = CodingSwarmProfile(
        user_id=test_user.id,
        source_id=source.id,
        title="Unsafe profile",
        status="active",
        preset_key="bug_triage_swarm",
        scope_default="auto",
        default_commands=["sudo rm -rf /tmp/project"],
        max_agents=4,
        safe_command_policy="standard",
        is_default=False,
        visibility="private",
    )
    db_session.add(profile)
    await db_session.commit()
    await db_session.refresh(profile)

    with pytest.raises(AgentCodingSwarmLaunchError) as unsafe_error:
        await agent_coding_swarm_launch_service.launch(
            request=AgentJobQuickStartBugTriageSwarmRequest(
                source_id=source.id,
                profile_id=profile.id,
                failure_symptom="Unsafe profile command",
                start_immediately=False,
            ),
            db=db_session,
            current_user=test_user,
            preset_key="bug_triage_swarm",
        )
    assert unsafe_error.value.status_code == 422
    assert unsafe_error.value.detail["blocked_commands"] == ["sudo rm -rf /tmp/project"]


@pytest.mark.asyncio
async def test_endpoint_adapter_dispatches_immediate_launch(
    db_session,
    test_user,
    monkeypatch,
):
    source = await _create_repository_source(db_session, test_user)
    dispatched: list[tuple[str, str]] = []
    monkeypatch.setattr(
        agent_jobs.execute_agent_job_task,
        "delay",
        lambda job_id, user_id: dispatched.append((job_id, user_id)),
    )

    response = await agent_jobs.quick_start_bug_triage_swarm_job(
        request=AgentJobQuickStartBugTriageSwarmRequest(
            source_id=source.id,
            failure_symptom="Dispatch this swarm",
            start_immediately=True,
        ),
        db=db_session,
        current_user=test_user,
    )

    assert dispatched == [(str(response.id), str(test_user.id))]
    assert response.config["launch_mode"] == "quick_start_bug_triage_swarm"
