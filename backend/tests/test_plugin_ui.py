"""What a plugin may contribute to the interface.

Two of these are security boundaries rather than conveniences, and both are
checked against the *executor type* rather than against anything the manifest
declared -- so neither can be relaxed by its author:

- a view may only be backed by a read-only tool, because rendering a page calls
  its source and a write-backed view would mean opening a page performs a write
  on every visit;
- a view may only call its own plugin's tools, because "install this small
  bundle" must not mean "grant it a reader for anything in the catalogue".
"""

from __future__ import annotations

import pytest

from app.services.plugin_manifest import ManifestError, validate_manifest
from app.services.plugin_ui import (
    ICONS,
    NAV_DOORS,
    PANEL_SLOTS,
    UiContributionError,
    validate_ui,
)

READ_TOOL = {"name": "list_runs", "tool_type": "transform", "parameters_schema": {}}
WRITE_TOOL = {"name": "push", "tool_type": "webhook", "parameters_schema": {}}

TABLE_VIEW = {
    "kind": "table",
    "title": "Runs",
    "source": {"tool": "list_runs"},
    "path": "items",
    "columns": ["name", "ms"],
}


def _ui(contributes, tools=(READ_TOOL,)):
    return validate_ui(contributes, tools=list(tools))


# --------------------------------------------------------------------------
# The two boundaries
# --------------------------------------------------------------------------


def test_a_view_may_not_be_backed_by_a_tool_that_writes():
    with pytest.raises(UiContributionError) as excinfo:
        _ui(
            {"views": {"x": {**TABLE_VIEW, "source": {"tool": "push"}}}},
            tools=(READ_TOOL, WRITE_TOOL),
        )

    message = str(excinfo.value)
    assert "classified 'write'" in message
    assert "perform a write every time it is viewed" in message


def test_a_view_may_not_call_a_tool_from_outside_its_plugin():
    with pytest.raises(UiContributionError) as excinfo:
        _ui({"views": {"x": {**TABLE_VIEW, "source": {"tool": "search_documents"}}}})

    assert "does not contribute" in str(excinfo.value)
    # Naming what it *does* declare is what stops the author guessing again.
    assert "list_runs" in str(excinfo.value)


def test_the_classification_comes_from_the_executor_not_the_manifest():
    """An author cannot opt a webhook into being read-only by saying so."""
    lying = {"name": "push", "tool_type": "webhook", "effects": "read"}

    with pytest.raises(UiContributionError, match="classified 'write'"):
        _ui(
            {"views": {"x": {**TABLE_VIEW, "source": {"tool": "push"}}}},
            tools=(lying,),
        )


# --------------------------------------------------------------------------
# Pointing at things that do not exist
# --------------------------------------------------------------------------


def test_a_nav_entry_must_point_at_a_view_this_manifest_declares():
    with pytest.raises(UiContributionError) as excinfo:
        _ui(
            {
                "views": {"runs": TABLE_VIEW},
                "nav": [{"door": "rnd", "name": "Runs", "view": "absent"}],
            }
        )

    assert "does not declare" in str(excinfo.value)
    assert "runs" in str(excinfo.value)


def test_a_panel_must_point_at_a_view_this_manifest_declares():
    with pytest.raises(UiContributionError, match="does not declare"):
        _ui(
            {
                "views": {"runs": TABLE_VIEW},
                "panels": [{"slot": PANEL_SLOTS[0], "view": "absent"}],
            }
        )


@pytest.mark.parametrize(
    "entry,fragment",
    [
        ({"door": "nowhere", "name": "X", "view": "runs"}, "must be one of"),
        ({"door": "rnd", "name": "", "view": "runs"}, "has no name"),
        ({"door": "rnd", "name": "X", "view": "runs", "icon": "skull"}, "icon"),
    ],
)
def test_a_nav_entry_is_refused_with_the_reason(entry, fragment):
    with pytest.raises(UiContributionError) as excinfo:
        _ui({"views": {"runs": TABLE_VIEW}, "nav": [entry]})

    assert fragment in str(excinfo.value)


def test_a_panel_slot_must_be_one_that_exists():
    with pytest.raises(UiContributionError) as excinfo:
        _ui({"views": {"runs": TABLE_VIEW}, "panels": [{"slot": "x", "view": "runs"}]})

    # Every declared slot is one a page actually hosts; naming them tells the
    # author where a panel can go.
    assert all(slot in str(excinfo.value) for slot in PANEL_SLOTS)


# --------------------------------------------------------------------------
# View shapes
# --------------------------------------------------------------------------


def test_a_view_kind_must_be_one_we_can_render():
    with pytest.raises(UiContributionError, match="must be one of"):
        _ui({"views": {"x": {"kind": "iframe", "source": {"tool": "list_runs"}}}})


def test_a_table_needs_columns_and_a_source():
    with pytest.raises(UiContributionError, match="needs at least one column"):
        _ui({"views": {"x": {"kind": "table", "source": {"tool": "list_runs"}}}})

    with pytest.raises(UiContributionError, match="needs a source"):
        _ui({"views": {"x": {"kind": "table", "columns": ["a"]}}})


def test_markdown_needs_either_text_or_a_source():
    with pytest.raises(UiContributionError, match="neither"):
        _ui({"views": {"x": {"kind": "markdown"}}})

    assert (
        _ui({"views": {"x": {"kind": "markdown", "text": "hi"}}})["views"]["x"]["text"]
        == "hi"
    )


def test_a_column_may_be_a_bare_string_or_a_labelled_object():
    out = _ui(
        {
            "views": {
                "x": {
                    **TABLE_VIEW,
                    "columns": ["name", {"key": "ms", "label": "Milliseconds"}],
                }
            }
        }
    )

    assert out["views"]["x"]["columns"] == [
        {"key": "name", "label": "name"},
        {"key": "ms", "label": "Milliseconds"},
    ]


# --------------------------------------------------------------------------
# Through the whole manifest
# --------------------------------------------------------------------------


def test_a_complete_manifest_with_ui_validates():
    manifest = {
        "id": "runs",
        "name": "Run Board",
        "version": "0.1.0",
        "contributes": {
            "tools": [
                {
                    "name": "recent",
                    "tool_type": "transform",
                    "parameters_schema": {},
                    "job_types": ["research"],
                }
            ],
            "views": {
                "board": {
                    "kind": "table",
                    "title": "Recent runs",
                    "source": {"tool": "recent"},
                    "path": "items",
                    "columns": ["name", "ms"],
                }
            },
            "nav": [
                {"door": "rnd", "name": "Run Board", "view": "board", "icon": "gauge"}
            ],
            "panels": [{"slot": PANEL_SLOTS[0], "view": "board"}],
        },
    }

    out = validate_manifest(manifest)

    assert out["contributes"]["nav"][0]["door"] == "rnd"
    assert out["contributes"]["panels"][0]["slot"] == PANEL_SLOTS[0]
    assert out["contributes"]["views"]["board"]["source"]["tool"] == "recent"


def test_a_bad_ui_contribution_is_a_manifest_error():
    """One exception type, so a caller catching ManifestError sees every
    reason a manifest was rejected rather than two kinds of failure."""
    manifest = {
        "id": "runs",
        "name": "R",
        "version": "0.1.0",
        "contributes": {
            "tools": [{"name": "recent", "tool_type": "transform"}],
            "views": {"x": {"kind": "wat"}},
        },
    }

    with pytest.raises(ManifestError, match="must be one of"):
        validate_manifest(manifest)


def test_the_declared_lists_are_not_empty():
    """Each is mirrored on the frontend; an empty one here would silently
    disable a whole kind of contribution."""
    assert NAV_DOORS and ICONS and PANEL_SLOTS
