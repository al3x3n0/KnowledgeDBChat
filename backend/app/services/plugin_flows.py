"""Workflows a plugin ships, and why one is refused.

A plugin could already extend workflows -- a tool node names a contributed tool
and the engine runs it. What it could not do was *ship the workflow itself*, so
a capability that is really "these three steps in this order" had to be
described in a README and rebuilt by hand in the editor by everybody who
installed it.

Unlike tools and views, a shipped workflow is **materialized**: rows in
`workflows`, `workflow_nodes` and `workflow_edges` owned by the installing
user. It has to be. A workflow is an editable object with executions attached
to it, and keeping it as a manifest that renders on demand would mean nobody
could change one without editing the plugin.

That materialization decides the update rule. Installing creates; updating a
plugin does **not** overwrite a workflow already created from it. Somebody's
edits are worth more than a version bump, and a plugin that silently reverted
them would be a reason never to install one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from app.agent_core.plugin_specs import contributed_tool_name
from app.services.workflow_engine import NODE_TYPES

MAX_WORKFLOWS = 10
MAX_NODES = 60
MAX_EDGES = 120


class FlowContributionError(ValueError):
    """A workflow that cannot be installed, with the reason."""


def _text(value: Any, limit: int = 200) -> str:
    return str(value or "").strip()[:limit]


def _identifier(value: Any, *, where: str, limit: int) -> str:
    """A node id or an edge endpoint, refused rather than shortened.

    Truncating an identifier changes what the author wrote: two node ids that
    differ only after the limit would silently become the same node, and an
    edge naming the full id would still match because it was truncated the
    same way. The graph would install and be wrong.
    """
    text = str(value or "").strip()
    if not text:
        raise FlowContributionError(f"{where} is empty")
    if len(text) > limit:
        raise FlowContributionError(
            f"{where} is {len(text)} characters; at most {limit} are stored, "
            "and shortening it would change which node it names"
        )
    return text


def _validate_node(
    raw: Any, *, where: str, own_tools: Sequence[str], builtin_tools: Sequence[str]
) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise FlowContributionError(f"{where} must be an object")

    # The column is String(50); allowing more would fail at insert, after
    # install had already reported success.
    node_id = _identifier(raw.get("node_id"), where=f"{where}.node_id", limit=50)

    node_type = _text(raw.get("node_type"), 32).lower()
    if node_type not in NODE_TYPES:
        # Read from the engine, never restated: a validator with its own copy
        # accepts a type the engine cannot run, which surfaces as a workflow
        # that installs cleanly and dies on its first execution.
        raise FlowContributionError(
            f"{where} has node_type {node_type!r}; must be one of: "
            f"{', '.join(NODE_TYPES)}"
        )

    node: Dict[str, Any] = {
        "node_id": node_id,
        "node_type": node_type,
        "config": dict(raw.get("config") or {}),
        "position_x": int(raw.get("position_x") or 0),
        "position_y": int(raw.get("position_y") or 0),
        "builtin_tool": None,
    }

    if node_type == "tool":
        tool = _text(raw.get("tool") or raw.get("builtin_tool"), 100)
        if not tool:
            raise FlowContributionError(f"{where} is a tool node and names no tool")
        if tool in own_tools:
            # The manifest names its own tool by the short name its author
            # wrote; what the engine will look up is the namespaced one.
            node["builtin_tool"] = None
            node["_own_tool"] = tool
        elif tool in builtin_tools:
            node["builtin_tool"] = tool
        else:
            raise FlowContributionError(
                f"{where} names tool {tool!r}, which is neither a built-in tool "
                f"nor one this plugin contributes. This manifest declares: "
                f"{', '.join(sorted(own_tools)) or '(no tools)'}."
            )

    return node


def validate_workflows(
    contributes: Mapping[str, Any],
    *,
    slug: str,
    tools: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalize `contributes.workflows`, or raise saying what is wrong."""
    from app.services.agent_tools import AGENT_TOOLS

    raw_workflows = contributes.get("workflows") or []
    if not isinstance(raw_workflows, list):
        raise FlowContributionError("contributes.workflows must be a list")
    if len(raw_workflows) > MAX_WORKFLOWS:
        raise FlowContributionError(
            f"contributes.workflows declares {len(raw_workflows)}; at most "
            f"{MAX_WORKFLOWS} are allowed"
        )

    own_tools = [str(t.get("name")) for t in tools if t.get("name")]
    builtin_tools = [str(t.get("name")) for t in AGENT_TOOLS]

    flows: List[Dict[str, Any]] = []
    seen_ids: set = set()

    for index, raw in enumerate(raw_workflows):
        where = f"contributes.workflows[{index}]"
        if not isinstance(raw, dict):
            raise FlowContributionError(f"{where} must be an object")

        flow_id = _text(raw.get("id"), 60)
        if not flow_id:
            raise FlowContributionError(f"{where} has no id")
        if flow_id in seen_ids:
            raise FlowContributionError(f"{where} repeats the id {flow_id!r}")
        seen_ids.add(flow_id)

        name = _text(raw.get("name"), 200)
        if not name:
            raise FlowContributionError(f"{where} has no name")

        raw_nodes = raw.get("nodes") or []
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise FlowContributionError(f"{where} declares no nodes")
        if len(raw_nodes) > MAX_NODES:
            raise FlowContributionError(
                f"{where} declares {len(raw_nodes)} nodes; at most {MAX_NODES}"
            )

        nodes = [
            _validate_node(
                node,
                where=f"{where}.nodes[{i}]",
                own_tools=own_tools,
                builtin_tools=builtin_tools,
            )
            for i, node in enumerate(raw_nodes)
        ]

        ids = [n["node_id"] for n in nodes]
        duplicated = sorted({i for i in ids if ids.count(i) > 1})
        if duplicated:
            raise FlowContributionError(
                f"{where} uses node_id {', '.join(duplicated)} more than once; "
                "edges name nodes by id, so a repeat makes the graph ambiguous"
            )

        starts = [n for n in nodes if n["node_type"] == "start"]
        if len(starts) != 1:
            # The engine looks for start nodes and needs exactly one; none
            # means it never begins, several means it begins arbitrarily.
            raise FlowContributionError(
                f"{where} has {len(starts)} start nodes; it needs exactly one"
            )

        raw_edges = raw.get("edges") or []
        if not isinstance(raw_edges, list):
            raise FlowContributionError(f"{where}.edges must be a list")
        if len(raw_edges) > MAX_EDGES:
            raise FlowContributionError(
                f"{where} declares {len(raw_edges)} edges; at most {MAX_EDGES}"
            )

        known = set(ids)
        edges: List[Dict[str, Any]] = []
        for i, raw_edge in enumerate(raw_edges):
            if not isinstance(raw_edge, dict):
                raise FlowContributionError(f"{where}.edges[{i}] must be an object")
            source = _identifier(
                raw_edge.get("source"), where=f"{where}.edges[{i}].source", limit=50
            )
            target = _identifier(
                raw_edge.get("target"), where=f"{where}.edges[{i}].target", limit=50
            )
            for role, value in (("source", source), ("target", target)):
                if value not in known:
                    raise FlowContributionError(
                        f"{where}.edges[{i}] has {role} {value!r}, which is not "
                        f"a node this workflow declares: {', '.join(sorted(known))}"
                    )
            edges.append(
                {
                    "source": source,
                    "target": target,
                    "condition": dict(raw_edge.get("condition") or {}),
                }
            )

        flows.append(
            {
                "id": flow_id,
                "name": name,
                "description": _text(raw.get("description"), 1000),
                "nodes": nodes,
                "edges": edges,
            }
        )

    return flows


def resolve_tool_names(flow: Mapping[str, Any], *, slug: str) -> Dict[str, Any]:
    """Turn a flow's own-tool references into the names the engine looks up.

    Done at materialization rather than at validation so the manifest keeps
    saying what its author wrote -- a plugin whose stored manifest was full of
    `p_<slug>_` prefixes would be one nobody could read back.
    """
    nodes = []
    for node in flow["nodes"]:
        copy = dict(node)
        own = copy.pop("_own_tool", None)
        if own:
            copy["builtin_tool"] = contributed_tool_name(slug, own)
        nodes.append(copy)
    return {**flow, "nodes": nodes}
