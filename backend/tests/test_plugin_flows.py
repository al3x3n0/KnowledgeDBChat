"""Workflows a plugin ships.

Unlike tools and views, a shipped workflow is materialized -- real rows the
owner can edit. That single fact decides most of what is tested here: identity
has to survive a rename, re-installing must not clobber an edit, and the node
types a manifest may use have to be the ones the engine can actually run.
"""

from __future__ import annotations

import copy

import pytest
from sqlalchemy import select

from app.models.plugin import Plugin, PluginInstallation
from app.models.workflow import Workflow, WorkflowEdge, WorkflowNode
from app.services import plugin_registry as registry
from app.services.plugin_flows import (
    FlowContributionError,
    resolve_tool_names,
    validate_workflows,
)
from app.services.plugin_manifest import ManifestError, validate_manifest
from app.services.workflow_engine import NODE_TYPES

TOOL = {
    "name": "stamp",
    "tool_type": "transform",
    "parameters_schema": {"type": "object", "properties": {}},
    "config": {"template": "stamped: {{ note }}"},
}

FLOW = {
    "id": "stamp_flow",
    "name": "Stamp a note",
    "nodes": [
        {"node_id": "s", "node_type": "start"},
        {"node_id": "t", "node_type": "tool", "tool": "stamp"},
        {"node_id": "e", "node_type": "end"},
    ],
    "edges": [
        {"source": "s", "target": "t"},
        {"source": "t", "target": "e"},
    ],
}


def _flows(**changes):
    flow = copy.deepcopy(FLOW)
    flow.update(changes)
    return validate_workflows({"workflows": [flow]}, slug="ops", tools=[TOOL])


def _manifest():
    return {
        "id": "ops",
        "name": "Ops Kit",
        "version": "0.1.0",
        "contributes": {"tools": [TOOL], "workflows": [copy.deepcopy(FLOW)]},
    }


# --------------------------------------------------------------------------
# The graph has to be one the engine can run
# --------------------------------------------------------------------------


def test_the_node_types_come_from_the_engine():
    """A validator with its own copy accepts a type the engine cannot run,
    which surfaces as a workflow that installs cleanly and dies on its first
    execution."""
    with pytest.raises(FlowContributionError) as excinfo:
        nodes = copy.deepcopy(FLOW["nodes"])
        nodes[1]["node_type"] = "webassembly"
        _flows(nodes=nodes)

    for node_type in NODE_TYPES:
        assert node_type in str(excinfo.value)


@pytest.mark.parametrize("count", [0, 2])
def test_a_workflow_needs_exactly_one_start(count):
    nodes = [n for n in copy.deepcopy(FLOW["nodes"]) if n["node_type"] != "start"]
    nodes += [{"node_id": f"s{i}", "node_type": "start"} for i in range(count)]
    edges = [e for e in FLOW["edges"] if e["source"] != "s"]

    with pytest.raises(FlowContributionError, match="exactly one"):
        _flows(nodes=nodes, edges=edges)


def test_an_edge_must_name_nodes_that_exist():
    edges = copy.deepcopy(FLOW["edges"])
    edges[0]["target"] = "ghost"

    with pytest.raises(FlowContributionError) as excinfo:
        _flows(edges=edges)

    assert "ghost" in str(excinfo.value)
    # Naming what does exist is what saves the author a second guess.
    assert "s, t" in str(excinfo.value)


def test_a_repeated_node_id_is_refused():
    nodes = copy.deepcopy(FLOW["nodes"]) + [{"node_id": "s", "node_type": "end"}]

    with pytest.raises(FlowContributionError, match="more than once"):
        _flows(nodes=nodes)


def test_a_node_id_longer_than_the_column_is_refused():
    """String(50). Allowing more would fail at insert, after install had
    already reported success."""
    nodes = copy.deepcopy(FLOW["nodes"])
    nodes[1]["node_id"] = "x" * 80
    edges = [{"source": "s", "target": "x" * 80}]

    with pytest.raises(FlowContributionError):
        _flows(nodes=nodes, edges=edges)


# --------------------------------------------------------------------------
# Tools a flow may call
# --------------------------------------------------------------------------


def test_a_tool_node_may_name_this_plugins_own_tool():
    flows = _flows()

    assert flows[0]["nodes"][1]["_own_tool"] == "stamp"


def test_a_tool_node_may_name_a_builtin():
    nodes = copy.deepcopy(FLOW["nodes"])
    nodes[1]["tool"] = "search_documents"

    flows = _flows(nodes=nodes)

    assert flows[0]["nodes"][1]["builtin_tool"] == "search_documents"


def test_a_tool_node_may_not_name_a_tool_nobody_has():
    nodes = copy.deepcopy(FLOW["nodes"])
    nodes[1]["tool"] = "not_a_real_tool"

    with pytest.raises(FlowContributionError) as excinfo:
        _flows(nodes=nodes)

    assert "stamp" in str(excinfo.value), "must name what this plugin declares"


def test_own_tool_references_are_namespaced_only_at_materialization():
    """The stored manifest keeps saying what its author wrote -- one full of
    `p_<slug>_` prefixes would be unreadable when read back."""
    flow = _flows()[0]
    assert flow["nodes"][1]["builtin_tool"] is None

    resolved = resolve_tool_names(flow, slug="ops")

    assert resolved["nodes"][1]["builtin_tool"] == "p_ops_stamp"
    assert "_own_tool" not in resolved["nodes"][1]


def test_a_bad_flow_is_a_manifest_error():
    manifest = _manifest()
    manifest["contributes"]["workflows"][0]["nodes"][1]["node_type"] = "nope"

    with pytest.raises(ManifestError, match="must be one of"):
        validate_manifest(manifest)


# --------------------------------------------------------------------------
# Materialization
# --------------------------------------------------------------------------


async def _install(db, user, manifest=None):
    validated = validate_manifest(manifest or _manifest())
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
    db.add(PluginInstallation(plugin_id=plugin.id, user_id=user.id, is_enabled=True))
    await db.flush()
    return plugin


@pytest.mark.asyncio
async def test_installing_creates_the_workflow_with_its_graph(db_session, test_user):
    plugin = await _install(db_session, test_user)

    outcomes = await registry.materialize_workflows(db_session, plugin, test_user.id)

    assert outcomes == [("stamp_flow", "created")]
    workflow = (
        await db_session.execute(
            select(Workflow).where(Workflow.origin_plugin_slug == "ops")
        )
    ).scalar_one()
    assert workflow.origin_flow_id == "stamp_flow"
    assert workflow.user_id == test_user.id

    nodes = list(
        (
            await db_session.execute(
                select(WorkflowNode).where(WorkflowNode.workflow_id == workflow.id)
            )
        ).scalars()
    )
    edges = list(
        (
            await db_session.execute(
                select(WorkflowEdge).where(WorkflowEdge.workflow_id == workflow.id)
            )
        ).scalars()
    )
    assert len(nodes) == 3
    assert len(edges) == 2
    tool_node = next(n for n in nodes if n.node_type == "tool")
    assert tool_node.builtin_tool == "p_ops_stamp"


@pytest.mark.asyncio
async def test_reinstalling_keeps_a_workflow_that_was_edited(db_session, test_user):
    """Somebody's edits are worth more than a version bump. A plugin that
    silently reverted them would be a reason never to install one."""
    plugin = await _install(db_session, test_user)
    await registry.materialize_workflows(db_session, plugin, test_user.id)

    workflow = (
        await db_session.execute(
            select(Workflow).where(Workflow.origin_plugin_slug == "ops")
        )
    ).scalar_one()
    workflow.name = "Stamp a note (mine)"
    await db_session.flush()

    outcomes = await registry.materialize_workflows(db_session, plugin, test_user.id)

    assert outcomes == [("stamp_flow", "kept")]
    rows = list(
        (
            await db_session.execute(
                select(Workflow).where(Workflow.origin_plugin_slug == "ops")
            )
        ).scalars()
    )
    assert len(rows) == 1, "a rename must not produce a second copy"
    assert rows[0].name == "Stamp a note (mine)"


@pytest.mark.asyncio
async def test_identity_is_the_flow_id_not_the_name(db_session, test_user):
    """Regression: matching on name meant renaming the workflow made the next
    install create a duplicate -- observed against a live stack, not theorised.
    """
    plugin = await _install(db_session, test_user)
    await registry.materialize_workflows(db_session, plugin, test_user.id)
    workflow = (
        await db_session.execute(
            select(Workflow).where(Workflow.origin_plugin_slug == "ops")
        )
    ).scalar_one()
    workflow.name = "something else entirely"
    await db_session.flush()

    await registry.materialize_workflows(db_session, plugin, test_user.id)

    rows = list(
        (
            await db_session.execute(
                select(Workflow).where(Workflow.origin_plugin_slug == "ops")
            )
        ).scalars()
    )
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_a_plugin_with_no_workflows_materializes_nothing(db_session, test_user):
    manifest = _manifest()
    manifest["contributes"].pop("workflows")
    plugin = await _install(db_session, manifest=manifest, user=test_user)

    assert await registry.materialize_workflows(db_session, plugin, test_user.id) == []
