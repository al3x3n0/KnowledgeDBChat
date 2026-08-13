"""Agents can author a reusable tool.

Custom tools existed but the whole surface was registered for chat only, so an
autonomous job could neither create one nor call one. Authoring was reachable
only through workflow synthesis, which is the least reliable path to it.
"""

import pytest
from sqlalchemy import select

from app.models.workflow import UserTool
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_workspace_mutation_provider,
)


def _ctx(db, user_id):
    return AgentToolExecutionContext(
        mode="autonomous", db=db, service=None, user_id=user_id, job=None, state={}
    )


async def _create(db, user_id, **params):
    provider = build_autonomous_workspace_mutation_provider(object())
    return await provider.execute("create_custom_tool", params, _ctx(db, user_id))


@pytest.mark.asyncio
async def test_a_tool_is_created_and_persisted(db_session, test_user):
    result = await _create(
        db_session,
        test_user.id,
        name="vector_count_compare",
        description="Compare vector op counts between two flag sets",
        tool_type="transform",
        parameters_schema={"type": "object", "properties": {"a": {"type": "number"}}},
        config={"expression": "{{ inputs.a }}"},
    )

    assert result["success"] is True
    stored = (
        await db_session.execute(
            select(UserTool).where(UserTool.name == "vector_count_compare")
        )
    ).scalar_one()
    assert stored.user_id == test_user.id
    assert stored.tool_type == "transform"
    assert stored.is_enabled is True


@pytest.mark.asyncio
async def test_creating_a_tool_is_recorded_as_a_finding(db_session, test_user):
    result = await _create(
        db_session,
        test_user.id,
        name="finding_tool",
        tool_type="transform",
        config={"expression": "x"},
    )

    assert result["findings"][0]["type"] == "tool_created"
    assert "finding_tool" in result["findings"][0]["title"]


@pytest.mark.asyncio
async def test_a_duplicate_name_is_refused_with_advice(db_session, test_user):
    await _create(
        db_session, test_user.id, name="dupe", tool_type="transform", config={"e": 1}
    )

    again = await _create(
        db_session, test_user.id, name="dupe", tool_type="transform", config={"e": 1}
    )

    assert "already exists" in again["error"]
    assert "run_custom_tool" in again["error"]


@pytest.mark.asyncio
async def test_docker_tools_stay_behind_their_flag(db_session, test_user, monkeypatch):
    """CUSTOM_TOOL_DOCKER_ENABLED gates container execution; the agent must not
    be able to route around it."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "CUSTOM_TOOL_DOCKER_ENABLED", False, raising=False)

    result = await _create(
        db_session,
        test_user.id,
        name="container_tool",
        tool_type="docker_container",
        config={"image": "alpine", "command": "echo hi"},
    )

    # The message echoes the rejected type back, so check the allowed list
    # itself rather than the whole message.
    allowed = result["error"].split("one of:")[1].split(".")[0]
    assert "docker_container" not in allowed
    assert "transform" in allowed


@pytest.mark.asyncio
async def test_workflow_runner_cannot_be_forged(db_session, test_user):
    """workflow_runner points at a workflow id that synthesis fills in."""
    result = await _create(
        db_session,
        test_user.id,
        name="fake_runner",
        tool_type="workflow_runner",
        config={"workflow_id": "00000000-0000-0000-0000-000000000000"},
    )

    assert "tool_type must be one of" in result["error"]


@pytest.mark.asyncio
async def test_missing_config_is_refused(db_session, test_user):
    result = await _create(
        db_session, test_user.id, name="no_config", tool_type="transform"
    )

    assert "config is required" in result["error"]


@pytest.mark.asyncio
async def test_missing_name_is_refused(db_session, test_user):
    result = await _create(
        db_session, test_user.id, name="  ", tool_type="transform", config={"e": 1}
    )

    assert "name is required" in result["error"]


class _Job:
    def __init__(self, user_id):
        self.id = "job-1"
        self.user_id = user_id
        self.iteration = 1


def _job_ctx(db, job):
    return AgentToolExecutionContext(
        mode="autonomous", db=db, service=None, user_id=None, job=job, state={}
    )


@pytest.mark.asyncio
async def test_the_owner_comes_from_the_job_when_ctx_has_no_user(
    db_session, test_user
):
    """ctx.user_id is not populated in autonomous runs; a live job failed on a
    NOT NULL violation on user_id."""
    provider = build_autonomous_workspace_mutation_provider(object())

    result = await provider.execute(
        "create_custom_tool",
        {"name": "owned_by_job", "tool_type": "transform", "config": {"e": 1}},
        _job_ctx(db_session, _Job(test_user.id)),
    )

    assert result["success"] is True
    stored = (
        await db_session.execute(
            select(UserTool).where(UserTool.name == "owned_by_job")
        )
    ).scalar_one()
    assert stored.user_id == test_user.id


@pytest.mark.asyncio
async def test_with_no_owner_at_all_it_declines_instead_of_inserting(db_session):
    provider = build_autonomous_workspace_mutation_provider(object())

    result = await provider.execute(
        "create_custom_tool",
        {"name": "orphan", "tool_type": "transform", "config": {"e": 1}},
        _job_ctx(db_session, _Job(None)),
    )

    assert "owning user" in result["error"]


@pytest.mark.asyncio
async def test_a_rejected_insert_leaves_the_session_usable(db_session, test_user):
    """The failed insert used to poison the shared session, ending the job
    rather than the action."""
    provider = build_autonomous_workspace_mutation_provider(object())
    ctx = _job_ctx(db_session, _Job(test_user.id))

    await provider.execute(
        "create_custom_tool",
        {"name": "x" * 300, "tool_type": "transform", "config": {"e": 1}},
        ctx,
    )

    # The session must still work for the next action in the same run.
    rows = (await db_session.execute(select(UserTool))).scalars().all()
    assert isinstance(rows, list)


@pytest.mark.asyncio
async def test_a_created_tool_can_then_be_listed(db_session, test_user):
    """Creation and execution were both chat-only; a live run created a tool
    and then could not call it."""
    provider = build_autonomous_workspace_mutation_provider(object())
    ctx = _job_ctx(db_session, _Job(test_user.id))

    await provider.execute(
        "create_custom_tool",
        {
            "name": "listable",
            "tool_type": "transform",
            "config": {"expression": "1"},
            "description": "d",
        },
        ctx,
    )
    listed = await provider.execute("list_custom_tools", {}, ctx)

    names = [t["name"] for t in listed["data"]["tools"]]
    assert "listable" in names


@pytest.mark.asyncio
async def test_running_an_unknown_tool_says_so(db_session, test_user):
    provider = build_autonomous_workspace_mutation_provider(object())

    result = await provider.execute(
        "run_custom_tool",
        {"tool_name": "does_not_exist"},
        _job_ctx(db_session, _Job(test_user.id)),
    )

    assert "No custom tool named" in result["error"]


@pytest.mark.asyncio
async def test_running_without_a_name_is_refused(db_session, test_user):
    provider = build_autonomous_workspace_mutation_provider(object())

    result = await provider.execute(
        "run_custom_tool", {}, _job_ctx(db_session, _Job(test_user.id))
    )

    assert "tool_name is required" in result["error"]
