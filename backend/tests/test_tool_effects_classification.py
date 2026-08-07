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
}


def _tool_names() -> list[str]:
    return sorted(
        str(tool.get("name") or "").strip()
        for tool in AGENT_TOOLS
        if str(tool.get("name") or "").strip()
    )


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
