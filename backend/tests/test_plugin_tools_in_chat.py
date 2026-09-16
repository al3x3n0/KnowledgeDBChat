"""Chat was the last surface that could not see a contributed tool.

Autonomous jobs could call one, and so could workflows. Chat built its menu
from the global `AGENT_TOOLS` and its dispatch checked policy *before* reaching
the provider -- so a user's own tool was invisible in the one place they would
naturally ask for it.

Two things are tested here that are easy to get wrong in opposite directions:
resolution must be per call (the service is a module-level singleton, so
anything cached on it is offered to whoever chats next), and the policy engine
must stop treating an unknown-to-the-catalog name as a nonexistent tool without
thereby letting anything ungoverned through.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.agent_core.plugin_specs import NAME_PREFIX
from app.models.plugin import Plugin, PluginInstallation
from app.services.agent_service import AgentService
from app.services.agent_tools import AGENT_TOOLS, get_tools_description
from app.services.plugin_manifest import validate_manifest
from app.services.tool_policy_engine import evaluate_tool_policy

MANIFEST = {
    "id": "chatkit",
    "name": "Chat Kit",
    "version": "0.1.0",
    "contributes": {
        "tools": [
            {
                "name": "echo",
                "description": "Echo a note back",
                "tool_type": "transform",
                "parameters_schema": {
                    "type": "object",
                    "properties": {"note": {"type": "string"}},
                },
                "config": {"template": "echo: {{ note }}"},
                "job_types": ["research"],
            }
        ]
    },
}


async def _install(db, user, *, enabled=True):
    validated = validate_manifest(MANIFEST)
    plugin = Plugin(
        slug=validated["id"],
        name=validated["name"],
        version=validated["version"],
        source="user",
        owner_id=user.id,
        manifest=validated,
    )
    db.add(plugin)
    await db.flush()
    db.add(PluginInstallation(plugin_id=plugin.id, user_id=user.id, is_enabled=enabled))
    await db.flush()
    return plugin


# --------------------------------------------------------------------------
# Chat can see it
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_offers_a_contributed_tool(db_session, test_user):
    await _install(db_session, test_user)

    schemas = await AgentService()._contributed_tool_schemas(db_session, test_user.id)

    assert [s["name"] for s in schemas] == ["p_chatkit_echo"]
    assert schemas[0]["parameters"]["properties"]["note"]


@pytest.mark.asyncio
async def test_a_contributed_tool_reaches_the_planning_prompt(db_session, test_user):
    """Being executable is not the same as being offered: a model that is never
    told a tool exists will never plan a call to it."""
    await _install(db_session, test_user)
    schemas = await AgentService()._contributed_tool_schemas(db_session, test_user.id)

    described = get_tools_description(schemas)

    assert "p_chatkit_echo" in described
    assert "Echo a note back" in described
    # And the built-ins are still all there.
    assert AGENT_TOOLS[0]["name"] in described


@pytest.mark.asyncio
async def test_one_users_tool_is_not_offered_to_another(db_session, test_user):
    """`AgentService` is a module-level singleton, so resolution has to be per
    call. Anything cached on it would be offered to whoever chats next."""
    await _install(db_session, test_user)
    service = AgentService()

    mine = await service._contributed_tool_schemas(db_session, test_user.id)
    theirs = await service._contributed_tool_schemas(db_session, uuid4())

    assert mine and theirs == []


@pytest.mark.asyncio
async def test_a_disabled_plugin_offers_chat_nothing(db_session, test_user):
    await _install(db_session, test_user, enabled=False)

    schemas = await AgentService()._contributed_tool_schemas(db_session, test_user.id)

    assert schemas == []


@pytest.mark.asyncio
async def test_resolution_failure_costs_the_turn_its_plugins_not_the_turn(
    db_session, test_user, monkeypatch
):
    async def boom(*_args, **_kwargs):
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(
        "app.services.plugin_registry.contributions_for_user", boom, raising=False
    )

    assert (
        await AgentService()._contributed_tool_schemas(db_session, test_user.id) == []
    )


@pytest.mark.asyncio
async def test_no_user_means_no_contributed_tools(db_session):
    """A contributed tool is always somebody's."""
    assert await AgentService()._contributed_tool_schemas(db_session, None) == []


# --------------------------------------------------------------------------
# The policy engine had to learn the namespace
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_contributed_name_is_not_treated_as_a_nonexistent_tool(
    db_session, test_user
):
    """Chat checks policy *before* dispatch, and the engine failed closed on any
    name without catalog metadata -- so a real contributed tool was rejected as
    "Unknown tool". Jobs never hit it, because they evaluate under
    `user_tool:<id>`, which was already exempt.
    """
    decision = await evaluate_tool_policy(
        db=db_session, tool_name="p_chatkit_echo", tool_args={}, user=test_user
    )

    assert decision.allowed
    assert not decision.denied_reason


@pytest.mark.asyncio
async def test_a_genuinely_unknown_tool_is_still_refused(db_session, test_user):
    """Widening the existence check must not stop it failing closed."""
    decision = await evaluate_tool_policy(
        db=db_session, tool_name="not_a_tool_at_all", tool_args={}, user=test_user
    )

    assert not decision.allowed
    assert "Unknown tool" in (decision.denied_reason or "")


@pytest.mark.asyncio
async def test_a_deny_policy_still_denies_a_contributed_tool(db_session, test_user):
    """The existence check was the only thing relaxed; governance is untouched."""
    from app.models.tool_policy import ToolPolicy

    db_session.add(
        ToolPolicy(
            subject_type="user",
            subject_id=test_user.id,
            tool_name="p_chatkit_echo",
            effect="deny",
        )
    )
    await db_session.flush()

    decision = await evaluate_tool_policy(
        db=db_session, tool_name="p_chatkit_echo", tool_args={}, user=test_user
    )

    assert not decision.allowed


def test_the_namespace_is_read_not_restated():
    """If the prefix ever changes, the policy engine must follow it."""
    from app.services import tool_policy_engine

    assert tool_policy_engine.NAME_PREFIX == NAME_PREFIX


@pytest.mark.asyncio
async def test_a_contributed_tool_keeps_its_authors_description(db_session, test_user):
    """Regression: `ContributedTool` had no `description` field, so
    `spec_from_user_tool` fell back to "User-contributed transform tool" and
    every plugin tool reached the model indistinguishable from every other of
    its type -- while the catalogue UI, reading the manifest directly, showed
    the real text. The description is what a model chooses between tools on.
    """
    await _install(db_session, test_user)

    schemas = await AgentService()._contributed_tool_schemas(db_session, test_user.id)

    assert schemas[0]["description"] == "Echo a note back"
    assert "User-contributed" not in schemas[0]["description"]


def test_the_prompt_survives_a_schema_an_author_left_incomplete():
    """A built-in schema is hand-written and always complete. A contributed one
    is whatever its author typed, and the first tool missing `description` on a
    property used to raise while building the prompt -- taking down the whole
    turn, for every tool, because one plugin omitted a field.
    """
    partial = {
        "name": "p_x_y",
        "description": "does a thing",
        "parameters": {"type": "object", "properties": {"note": {"type": "string"}}},
    }
    bare = {"name": "p_x_z", "parameters": {}}

    described = get_tools_description([partial, bare])

    assert "p_x_y" in described and "p_x_z" in described
    assert "(no description)" in described


def test_both_chat_planners_accept_contributed_tools():
    """Streaming chat plans through a different entry point than the
    agent-routed path. Offering a user's tools on one and not the other is the
    "works over there" failure: the same question typed into the same box would
    find the tool or not, depending on which handler took it.
    """
    import inspect

    for planner in (
        AgentService._plan_tool_calls,
        AgentService._plan_tool_calls_for_agent,
    ):
        assert (
            "contributed" in inspect.signature(planner).parameters
        ), f"{planner.__name__} cannot be given contributed tools"


def test_the_streaming_path_actually_passes_them():
    """A parameter nothing passes is the same as no parameter at all."""
    import inspect

    from app.api.endpoints import agent as agent_endpoints

    source = inspect.getsource(agent_endpoints._process_message_with_streaming)

    assert "_contributed_tool_schemas" in source
    assert "contributed=contributed" in source
