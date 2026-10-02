"""The output formatting tools: format_as_table, format_as_report, set_output_schema.

These call the real handlers. The file used to restate each handler's logic
inline and assert on the restatement -- `title = str(params.get("title",
"")).strip(); assert not title` -- so forty-four tests passed whatever the
tools did, including while `format_as_report` could not persist anything.
"""

from types import SimpleNamespace

import pytest

from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_output_state_provider,
)

pytestmark = pytest.mark.unit


def _call(tool, params, state=None):
    """Run one handler against a state dict; returns (result, state)."""
    state = {} if state is None else state
    provider = build_autonomous_output_state_provider(SimpleNamespace())
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=None,
        service=None,
        user_id="u",
        job=SimpleNamespace(id="job-1", user_id="u", goal="g", config={}),
        state=state,
    )
    return provider._handlers[tool](params, ctx), state


async def _run(tool, params, state=None):
    pending, state = _call(tool, params, state)
    return await pending, state


class TestFormatAsTable:
    async def test_a_title_is_required(self):
        result, state = await _run(
            "format_as_table", {"columns": ["A"], "rows": [["1"]]}
        )
        assert result == {"error": "title parameter is required"}
        assert "formatted_outputs" not in state

    @pytest.mark.parametrize(
        "params",
        [
            {"title": "T", "rows": [["1"]]},
            {"title": "T", "columns": ["A"]},
            {"title": "T", "columns": "A", "rows": [["1"]]},
        ],
    )
    async def test_a_custom_table_needs_columns_and_rows(self, params):
        result, _ = await _run("format_as_table", params)
        assert "columns and rows are required" in result["error"]

    async def test_it_renders_a_markdown_table(self):
        result, state = await _run(
            "format_as_table",
            {
                "title": "Scores",
                "columns": ["Name", "Score"],
                "rows": [["Alice", 95], ["Bob", 87]],
            },
        )

        assert result["success"] is True
        assert result["data"]["row_count"] == 2 and result["data"]["columns"] == 2
        assert result["data"]["markdown"].splitlines() == [
            "## Scores",
            "",
            "| Name | Score |",
            "| --- | --- |",
            "| Alice | 95 |",
            "| Bob | 87 |",
        ]
        assert result["artifacts"] == [{"type": "formatted_table", "title": "Scores"}]
        stored = state["formatted_outputs"][0]
        assert stored["type"] == "table" and stored["row_count"] == 2
        assert stored["columns"] == ["Name", "Score"]

    async def test_cells_cannot_break_the_table(self):
        result, _ = await _run(
            "format_as_table",
            {
                "title": "T",
                "columns": ["A", "B"],
                "rows": [["a|b"], ["1", "2", "3"], "not a row", ["x" * 500, ""]],
            },
        )
        lines = result["data"]["markdown"].splitlines()[4:]

        assert lines[0] == "| a\\|b |  |"  # pipe escaped, short row padded
        assert lines[1] == "| 1 | 2 |"  # long row trimmed
        assert lines[2] == "|  |  |"  # a row that is not a list is empty
        assert lines[3] == "| " + "x" * 200 + " |  |"  # cell truncated

    async def test_rows_are_capped_at_100(self):
        result, _ = await _run(
            "format_as_table",
            {"title": "T", "columns": ["n"], "rows": [[i] for i in range(150)]},
        )
        assert result["data"]["row_count"] == 100

    async def test_findings_become_rows(self):
        state = {
            "findings": [
                {"title": "F1", "category": "perf", "confidence": 0.9},
                "not a finding",
                {"title": "F2"},
            ]
        }
        result, _ = await _run(
            "format_as_table", {"title": "T", "source": "findings"}, state
        )
        lines = result["data"]["markdown"].splitlines()

        assert lines[2] == "| title | category | confidence |"
        assert lines[4:] == ["| F1 | perf | 0.9 |", "| F2 |  |  |"]

    async def test_finding_fields_can_be_chosen_and_are_capped(self):
        state = {"findings": [{"title": "F1", "content": "c"}]}
        result, _ = await _run(
            "format_as_table",
            {
                "title": "T",
                "source": "findings",
                "finding_fields": ["content", " ", "title"],
            },
            state,
        )
        assert "| content | title |" in result["data"]["markdown"]

        result, _ = await _run(
            "format_as_table",
            {
                "title": "T",
                "source": "findings",
                "finding_fields": [f"f{i}" for i in range(15)],
            },
            state,
        )
        assert result["data"]["columns"] == 10

    async def test_no_findings_is_an_empty_table_not_an_error(self):
        result, _ = await _run("format_as_table", {"title": "T", "source": "findings"})
        assert result["success"] is True and result["data"]["row_count"] == 0


class TestFormatAsReport:
    async def test_a_title_is_required(self):
        result, state = await _run("format_as_report", {})
        assert result == {"error": "title parameter is required"}
        assert "formatted_outputs" not in state

    async def test_a_bare_report_is_its_title(self):
        result, state = await _run("format_as_report", {"title": "Study"})

        assert result["success"] is True
        assert result["data"] == {
            "markdown": "# Study\n\n",
            "length": len("# Study\n\n"),
            "document_id": None,
        }
        assert result["artifacts"] == [{"type": "formatted_report", "title": "Study"}]
        assert state["formatted_outputs"] == [
            {"type": "report", "title": "Study", "markdown": "# Study\n\n"}
        ]

    async def test_summary_and_sections_in_order(self):
        result, _ = await _run(
            "format_as_report",
            {
                "title": "Study",
                "executive_summary": "S" * 5000,
                "sections": [
                    {"heading": "Method", "content": "m"},
                    "not a section",
                    {"content": "untitled"},
                ],
            },
        )
        md = result["data"]["markdown"]

        assert "## Executive Summary\n\n" + "S" * 3000 + "\n\n" in md
        assert "S" * 3001 not in md
        assert md.index("## Executive Summary") < md.index("## Method\n\nm")
        assert "## Section\n\nuntitled" in md

    async def test_sections_are_capped_at_20(self):
        result, _ = await _run(
            "format_as_report",
            {
                "title": "T",
                "sections": [{"heading": f"H{i}", "content": "c"} for i in range(25)],
            },
        )
        assert "## H19\n" in result["data"]["markdown"]
        assert "## H20\n" not in result["data"]["markdown"]

    async def test_findings_are_included_unless_declined(self):
        state = {
            "findings": [
                {
                    "title": "F1",
                    "content": "body",
                    "category": "perf",
                    "confidence": 0.9,
                },
                {"content": "no title"},
            ]
        }
        result, _ = await _run("format_as_report", {"title": "T"}, state)
        md = result["data"]["markdown"]

        assert "### 1. F1\n\nbody\n\n*Category: perf | Confidence: 0.9*" in md
        assert "### 2. Untitled\n\nno title" in md

        # An explicit null means the default, not "no".
        result, _ = await _run(
            "format_as_report", {"title": "T", "include_findings": None}, state
        )
        assert "## Findings" in result["data"]["markdown"]

        result, _ = await _run(
            "format_as_report", {"title": "T", "include_findings": False}, state
        )
        assert "## Findings" not in result["data"]["markdown"]

    async def test_findings_are_capped_at_30(self):
        state = {"findings": [{"title": f"F{i}"} for i in range(40)]}
        result, _ = await _run("format_as_report", {"title": "T"}, state)
        assert "### 30. F29" in result["data"]["markdown"]
        assert "### 31." not in result["data"]["markdown"]

    async def test_only_the_last_five_progress_reports(self):
        state = {
            "progress_reports": [
                {"iteration": i, "summary": f"did {i}"} for i in range(1, 9)
            ]
        }
        result, _ = await _run("format_as_report", {"title": "T"}, state)
        md = result["data"]["markdown"]

        assert "### Iteration 4\n\ndid 4" in md and "### Iteration 8" in md
        assert "### Iteration 3" not in md

        result, _ = await _run(
            "format_as_report", {"title": "T", "include_progress": False}, state
        )
        assert "Progress History" not in result["data"]["markdown"]

    async def test_the_whole_report_is_capped(self):
        result, _ = await _run(
            "format_as_report",
            {
                "title": "T",
                "sections": [
                    {"heading": "H", "content": "x" * 5000} for _ in range(20)
                ],
            },
        )
        assert result["data"]["length"] == 50000

    # Persisting is covered against a real database in
    # tests/test_format_as_report_persists.py.


class TestSetOutputSchema:
    @pytest.mark.parametrize("schema", [None, {}, "summary", ["summary"]])
    async def test_a_schema_must_be_a_non_empty_object(self, schema):
        result, state = await _run("set_output_schema", {"schema": schema})
        assert result == {"error": "schema must be a non-empty object"}
        assert "output_schema" not in state

    async def test_it_merges_by_default(self):
        state = {"output_schema": {"summary": "old", "kept": 1}}
        result, state = await _run(
            "set_output_schema", {"schema": {"summary": "new", "added": 2}}, state
        )

        assert state["output_schema"] == {"summary": "new", "kept": 1, "added": 2}
        assert result["data"] == {
            "schema_keys": ["summary", "kept", "added"],
            "total_keys": 3,
            "mode": "merged",
        }

    async def test_an_explicit_null_merge_still_merges(self):
        state = {"output_schema": {"kept": 1}}
        _, state = await _run(
            "set_output_schema", {"schema": {"a": 1}, "merge": None}, state
        )
        assert state["output_schema"] == {"kept": 1, "a": 1}

    async def test_it_can_replace(self):
        schema = {"only": 1}
        state = {"output_schema": {"old": 1}}
        result, state = await _run(
            "set_output_schema", {"schema": schema, "merge": False}, state
        )

        assert state["output_schema"] == {"only": 1}
        assert state["output_schema"] is not schema  # a copy, not the caller's dict
        assert result["data"]["mode"] == "replaced"

    async def test_a_corrupt_existing_schema_is_replaced_not_crashed_on(self):
        state = {"output_schema": "garbage"}
        result, state = await _run("set_output_schema", {"schema": {"a": 1}}, state)
        assert result["success"] is True and state["output_schema"] == {"a": 1}


class TestOutputFormattingSchemas:
    """Tests for output formatting tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS

        names = {t["name"] for t in AGENT_TOOLS}
        assert "format_as_table" in names
        assert "format_as_report" in names
        assert "set_output_schema" in names

    def test_format_as_table_requires_title(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("format_as_table")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "title" in required

    def test_format_as_table_has_source_enum(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("format_as_table")
        source_prop = tool["parameters"]["properties"]["source"]
        assert "enum" in source_prop
        assert "findings" in source_prop["enum"]
        assert "custom" in source_prop["enum"]

    def test_format_as_report_requires_title(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("format_as_report")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "title" in required

    def test_format_as_report_has_persist(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("format_as_report")
        assert "persist" in tool["parameters"]["properties"]

    def test_set_output_schema_requires_schema(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("set_output_schema")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "schema" in required


class TestOutputFormattingRegistry:
    """Tests for output formatting tool registry classification."""

    def test_all_are_read(self):
        from app.services.tool_registry import get_tool_metadata

        for tool_name in ["format_as_table", "format_as_report", "set_output_schema"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.effects == "read"

    def test_all_are_low_cost(self):
        from app.services.tool_registry import get_tool_metadata

        for tool_name in ["format_as_table", "format_as_report", "set_output_schema"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.cost_tier == "low"

    def test_none_is_network_tool(self):
        from app.services.tool_registry import get_tool_metadata

        for tool_name in ["format_as_table", "format_as_report", "set_output_schema"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.network == "none"
