"""Guards on the tool effects classification.

Tool effects are derived from a hardcoded name list in ``_default_metadata``,
so a tool that is not on the list is classified read-safe *by omission*. That is
the wrong default: today the field is only displayed, but the moment anything
enforces an ``allowed_effects`` policy, a mutating tool classified read would be
permitted under a read-only policy — false assurance, which is worse than no
enforcement.

An audit found 25 such tools, including ``execute_python``, which runs arbitrary
code and was classified read while its sibling ``write_and_run_script`` was
classified write.

These tests do not attempt to decide each tool's effects. They fail when a new
tool whose name implies mutation is added without an explicit decision, so the
classification stays deliberate.
"""

import pytest

from app.agent_core.tool_catalog import iter_builtin_tools, iter_mcp_tools
from app.services.agent_tools import AGENT_TOOLS
from app.services.tool_registry import get_tool_metadata

# Verbs that imply a tool changes something. A tool whose name starts with one
# of these must be classified explicitly rather than defaulting to read.
MUTATING_PREFIXES = (
    "add_",
    "apply_",
    "cancel_",
    "create_",
    "delete_",
    "execute_",
    "export_",
    "generate_",
    "ingest_",
    "link_",
    "merge_",
    "publish_",
    "remove_",
    "run_",
    "save_",
    "schedule_",
    "send_",
    "set_",
    "update_",
    "upload_",
    "write_",
)

# Tools whose names look mutating but are genuinely read-only, each with the
# reason. Adding to this list is a deliberate act; adding a tool without
# touching either list fails the test below.
READ_ONLY_DESPITE_NAME: dict[str, str] = {
    # "write" means an effect outside the agent's own run — a database row, a
    # file, a network call, a queued job. These only mutate in-run state, which
    # a read-only policy is not trying to prevent: it exists to stop side
    # effects on the world, not to stop the agent steering itself.
    "save_research_finding": "appends to the run's findings in state",
    "write_section": "edits the in-run document workspace",
    "write_progress_report": "records progress in run state",
    "set_focus_directive": "sets a directive for this run only",
    "set_output_schema": "sets an output shape for this run only",
    "create_workspace_checkpoint": "snapshots in-run workspace state",
    # The data-analysis tools. Every one was read at its implementation before
    # being listed here: the datasets live in an in-memory sandbox keyed by job
    # id, and the diagram and export tools return their product in the response
    # body -- base64, mermaid or JSON -- rather than persisting it. Nothing
    # here reaches a database, a file or the network, so a read-only policy has
    # no side effect on the world to stop.
    "create_dataset": "builds a dataset in the run's in-memory sandbox",
    "create_chart_from_dataset": "returns the chart as base64 in the response",
    "create_correlation_heatmap": "returns the heatmap as base64 in the response",
    "create_flowchart": "returns mermaid source in the response",
    "create_sequence_diagram": "returns mermaid source in the response",
    "create_er_diagram": "returns mermaid source in the response",
    "create_architecture_diagram": "returns mermaid source in the response",
    "create_gantt_chart": "returns mermaid source in the response",
    "create_drawio_diagram": "returns drawio XML in the response",
    "export_dataset_csv": "returns base64 CSV in the response; writes no file",
    "export_dataset_json": "returns JSON in the response; writes no file",
}


def _tool_names() -> list[str]:
    """Every executable builtin tool, from the catalog rather than one registry.

    This read AGENT_TOOLS directly, which is the registry the data-analysis
    tools are not in -- so 21 executable tools escaped every guard in this
    file, including the one written to catch tools with no metadata. A guard
    that derives its own universe from a partial list inherits the omission it
    was written to detect.
    """
    return sorted(meta.name for meta in iter_builtin_tools() if meta.name)


def test_every_mutating_looking_tool_is_classified_deliberately():
    unclassified = []
    for name in _tool_names():
        if not name.startswith(MUTATING_PREFIXES):
            continue
        if name in READ_ONLY_DESPITE_NAME:
            continue
        meta = get_tool_metadata(name)
        if meta is None or meta.effects != "write":
            unclassified.append(name)

    assert not unclassified, (
        "These tools have mutating-sounding names but are classified read-safe.\n"
        "Either add them to write_tools in agent_core/tool_catalog.py, or add\n"
        "them to READ_ONLY_DESPITE_NAME here with the reason they are safe:\n"
        + "\n".join(f"  - {name}" for name in unclassified)
    )


def test_code_execution_tools_are_never_classified_read():
    """The specific failure the audit found, pinned so it cannot come back."""
    for name in ("execute_python", "write_and_run_script", "run_command"):
        meta = get_tool_metadata(name)
        if meta is None:
            continue  # not all are registered as builtin agent tools
        assert meta.effects == "write", f"{name} must not be classified read-safe"


def test_known_read_tools_are_not_over_classified():
    """Fail-closed must not mean everything is a write."""
    for name in ("search_documents", "read_document_content", "list_documents"):
        meta = get_tool_metadata(name)
        if meta is None:
            continue
        assert meta.effects == "read", f"{name} should be read, got {meta.effects}"


def test_every_builtin_tool_resolves_to_metadata():
    """A tool with no metadata is denied by any constraint policy.

    _constraints_ok fails closed on unknown tools, so a missing entry silently
    makes a tool unusable wherever constraints are configured.
    """
    missing = [name for name in _tool_names() if get_tool_metadata(name) is None]
    assert not missing, f"tools with no catalog metadata: {missing}"


@pytest.mark.parametrize("field", ["effects", "network", "cost_tier", "pii_risk"])
def test_metadata_fields_use_known_values(field):
    allowed = {
        "effects": {"read", "write"},
        "network": {"none", "egress"},
        "cost_tier": {"low", "medium", "high"},
        "pii_risk": {"low", "medium", "high"},
    }[field]
    for name in _tool_names():
        meta = get_tool_metadata(name)
        assert meta is not None
        assert (
            getattr(meta, field) in allowed
        ), f"{name}.{field}={getattr(meta, field)!r} is outside {sorted(allowed)}"


def test_mcp_tools_are_classified_deliberately_too():
    """The MCP surface is policy-gated the same way and needs the same guard.

    These tools live in a separate table that get_tool_metadata consults only
    for mcp:-prefixed names. It sat outside this file's reach until it was
    lifted to module level — and two job-creating tools, create_presentation
    and create_repo_report, were sitting in it classified read-safe.
    """
    unclassified = []
    for name in sorted(iter_mcp_tools()):
        if not name.startswith(MUTATING_PREFIXES):
            continue
        if name in READ_ONLY_DESPITE_NAME:
            continue
        meta = get_tool_metadata(f"mcp:{name}")
        if meta is None or meta.effects != "write":
            unclassified.append(name)

    assert (
        not unclassified
    ), "MCP tools with mutating-sounding names classified read-safe:\n" + "\n".join(
        f"  - {name}" for name in unclassified
    )


def test_every_mcp_tool_resolves_only_under_its_prefix():
    """Prefix handling is load-bearing: policy checks pass mcp:<name>.

    A bare lookup returning None for an MCP tool is correct, not a defect — a
    fact worth pinning, since mistaking it for one sent an earlier audit down
    the wrong path entirely.
    """
    for name in iter_mcp_tools():
        assert get_tool_metadata(f"mcp:{name}") is not None, f"mcp:{name} must resolve"
        if not any(str(tool.get("name") or "") == name for tool in AGENT_TOOLS):
            assert (
                get_tool_metadata(name) is None
            ), f"{name} is MCP-only and should not resolve unprefixed"


def test_docker_execute_is_the_strictest_classification():
    """It runs arbitrary commands in a container; nothing should be laxer."""
    meta = get_tool_metadata("mcp:docker_execute")
    assert meta is not None
    assert meta.effects == "write"
    assert meta.cost_tier == "high"
    assert meta.pii_risk == "high"
    assert meta.network == "egress"
