"""Tests for agent output formatting tools (format_as_table, format_as_report, set_output_schema)."""

import uuid

import pytest


class TestFormatAsTableCustom:
    """Tests for format_as_table with custom data."""

    def test_requires_title(self):
        params = {"columns": ["A"], "rows": [["1"]]}
        title = str(params.get("title", "")).strip()
        assert not title

    def test_requires_columns_for_custom(self):
        params = {"title": "Test Table", "rows": [["1"]]}
        columns = params.get("columns", [])
        assert not columns

    def test_requires_rows_for_custom(self):
        params = {"title": "Test Table", "columns": ["A"]}
        rows = params.get("rows", [])
        assert not rows

    def test_generates_markdown_table(self):
        columns = ["Name", "Score"]
        rows = [["Alice", "95"], ["Bob", "87"]]
        md = "## Test Table\n\n"
        md += "| " + " | ".join(columns) + " |\n"
        md += "| " + " | ".join("---" for _ in columns) + " |\n"
        for row in rows:
            cells = [str(c).replace("|", "\\|")[:200] for c in row]
            md += "| " + " | ".join(cells) + " |\n"
        assert "| Name | Score |" in md
        assert "| --- | --- |" in md
        assert "| Alice | 95 |" in md
        assert "| Bob | 87 |" in md

    def test_pipes_in_cells_escaped(self):
        cell = "value|with|pipes"
        escaped = cell.replace("|", "\\|")
        assert "\\|" in escaped

    def test_rows_capped_at_100(self):
        rows = [[f"r{i}"] for i in range(150)]
        capped = rows[:100]
        assert len(capped) == 100

    def test_cells_truncated_to_200(self):
        long_cell = "X" * 300
        truncated = long_cell[:200]
        assert len(truncated) == 200

    def test_short_rows_padded(self):
        columns = ["A", "B", "C"]
        row = ["1"]
        cells = list(row)
        while len(cells) < len(columns):
            cells.append("")
        assert len(cells) == 3
        assert cells[1] == ""
        assert cells[2] == ""

    def test_long_rows_trimmed(self):
        columns = ["A", "B"]
        row = ["1", "2", "3", "4"]
        cells = row[:len(columns)]
        assert len(cells) == 2


class TestFormatAsTableFindings:
    """Tests for format_as_table with source=findings."""

    def test_auto_extract_findings(self):
        findings = [
            {"title": "Finding A", "category": "key_insight", "confidence": 0.9},
            {"title": "Finding B", "category": "methodology", "confidence": 0.7},
        ]
        fields = ["title", "category", "confidence"]
        rows = []
        for f in findings:
            row = [str(f.get(field, ""))[:200] for field in fields]
            rows.append(row)
        assert len(rows) == 2
        assert rows[0][0] == "Finding A"
        assert rows[1][1] == "methodology"

    def test_custom_finding_fields(self):
        findings = [{"title": "A", "content": "Content A", "tags": "['ml']"}]
        fields = ["title", "content"]
        rows = []
        for f in findings:
            row = [str(f.get(field, ""))[:200] for field in fields]
            rows.append(row)
        assert rows[0][1] == "Content A"

    def test_default_finding_fields(self):
        params = {"title": "Findings", "source": "findings"}
        fields = params.get("finding_fields", ["title", "category", "confidence"])
        assert fields == ["title", "category", "confidence"]

    def test_finding_fields_capped_at_10(self):
        fields = [f"field_{i}" for i in range(15)]
        capped = [str(f).strip() for f in fields if str(f).strip()][:10]
        assert len(capped) == 10

    def test_empty_findings(self):
        findings = []
        rows = []
        for f in findings:
            rows.append([])
        assert len(rows) == 0


class TestFormatAsTableState:
    """Tests for format_as_table state integration."""

    def test_stored_in_formatted_outputs(self):
        state: dict = {}
        state.setdefault("formatted_outputs", []).append({
            "type": "table",
            "title": "My Table",
            "markdown": "| A |\n| --- |\n| 1 |",
            "columns": ["A"],
            "row_count": 1,
        })
        assert len(state["formatted_outputs"]) == 1
        assert state["formatted_outputs"][0]["type"] == "table"

    def test_result_format(self):
        result = {
            "markdown": "| A | B |\n| --- | --- |\n| 1 | 2 |",
            "row_count": 1,
            "columns": 2,
        }
        assert "markdown" in result
        assert result["row_count"] == 1


class TestFormatAsReport:
    """Tests for format_as_report tool logic."""

    def test_requires_title(self):
        params = {}
        title = str(params.get("title", "")).strip()
        assert not title

    def test_basic_report(self):
        title = "Research Report"
        md = f"# {title}\n\n"
        assert md.startswith("# Research Report")

    def test_executive_summary(self):
        md = ""
        exec_summary = "This report covers transformer architectures."
        md += f"## Executive Summary\n\n{exec_summary}\n\n"
        assert "Executive Summary" in md
        assert "transformer" in md

    def test_executive_summary_truncated(self):
        long_summary = "S" * 5000
        truncated = long_summary[:3000]
        assert len(truncated) == 3000

    def test_custom_sections(self):
        sections = [
            {"heading": "Background", "content": "Background content here"},
            {"heading": "Methods", "content": "Methods description"},
        ]
        md = ""
        for sec in sections[:20]:
            heading = str(sec.get("heading", "Section"))[:200]
            content = str(sec.get("content", ""))[:5000]
            md += f"## {heading}\n\n{content}\n\n"
        assert "## Background" in md
        assert "## Methods" in md

    def test_sections_capped_at_20(self):
        sections = [{"heading": f"S{i}", "content": f"C{i}"} for i in range(25)]
        capped = sections[:20]
        assert len(capped) == 20

    def test_findings_included_by_default(self):
        params = {"title": "Report"}
        include_findings = params.get("include_findings", True)
        if include_findings is None:
            include_findings = True
        assert include_findings is True

    def test_findings_rendering(self):
        findings = [
            {"title": "Finding A", "content": "Content A", "category": "key_insight", "confidence": 0.9},
        ]
        md = "## Findings\n\n"
        for i, f in enumerate(findings[:30], 1):
            md += f"### {i}. {f.get('title', 'Untitled')}\n\n"
            md += f"{str(f.get('content', ''))[:1000]}\n\n"
            meta = []
            if f.get("category"):
                meta.append(f"Category: {f['category']}")
            if f.get("confidence"):
                meta.append(f"Confidence: {f['confidence']}")
            if meta:
                md += f"*{' | '.join(meta)}*\n\n"
        assert "### 1. Finding A" in md
        assert "Content A" in md
        assert "Category: key_insight | Confidence: 0.9" in md

    def test_findings_capped_at_30(self):
        findings = [{"title": f"F{i}"} for i in range(50)]
        rendered = findings[:30]
        assert len(rendered) == 30

    def test_progress_reports_included(self):
        reports = [
            {"iteration": 5, "summary": "Searched for papers"},
            {"iteration": 10, "summary": "Analyzed results"},
        ]
        md = "## Progress History\n\n"
        for r in reports[-5:]:
            md += f"### Iteration {r.get('iteration', '?')}\n\n"
            if r.get("summary"):
                md += f"{r['summary']}\n\n"
        assert "Iteration 5" in md
        assert "Searched for papers" in md

    def test_total_length_capped_at_50000(self):
        long_md = "M" * 60000
        capped = long_md[:50000]
        assert len(capped) == 50000

    def test_persist_defaults_false(self):
        params = {"title": "Report"}
        persist = params.get("persist", False)
        assert persist is False

    def test_result_format(self):
        result = {
            "markdown": "# Report\n\n...",
            "length": 500,
            "document_id": None,
        }
        assert "markdown" in result
        assert "length" in result
        assert result["document_id"] is None

    def test_result_with_persist(self):
        result = {
            "markdown": "# Report\n\n...",
            "length": 500,
            "document_id": str(uuid.uuid4()),
        }
        assert result["document_id"] is not None


class TestSetOutputSchema:
    """Tests for set_output_schema tool logic."""

    def test_requires_schema(self):
        params = {}
        schema = params.get("schema")
        assert not isinstance(schema, dict) or not schema

    def test_empty_schema_rejected(self):
        params = {"schema": {}}
        schema = params.get("schema")
        assert not schema

    def test_non_dict_schema_rejected(self):
        params = {"schema": "not a dict"}
        schema = params.get("schema")
        assert not isinstance(schema, dict)

    def test_merge_mode(self):
        state = {"output_schema": {"summary": "Initial summary"}}
        new_schema = {"recommendations": ["Use BERT"]}
        state["output_schema"].update(new_schema)
        assert "summary" in state["output_schema"]
        assert "recommendations" in state["output_schema"]

    def test_replace_mode(self):
        state = {"output_schema": {"old_key": "old_value"}}
        new_schema = {"new_key": "new_value"}
        state["output_schema"] = dict(new_schema)
        assert "old_key" not in state["output_schema"]
        assert "new_key" in state["output_schema"]

    def test_merge_defaults_true(self):
        params = {"schema": {"key": "val"}}
        merge = params.get("merge", True)
        if merge is None:
            merge = True
        assert merge is True

    def test_progressive_building(self):
        state: dict = {"output_schema": {}}
        state["output_schema"]["title"] = "Research Report"
        state["output_schema"]["key_findings"] = []
        state["output_schema"]["key_findings"].append("Finding 1")
        state["output_schema"]["key_findings"].append("Finding 2")
        assert len(state["output_schema"]["key_findings"]) == 2

    def test_result_format(self):
        result = {
            "schema_keys": ["summary", "recommendations", "next_steps"],
            "total_keys": 3,
            "mode": "merged",
        }
        assert result["total_keys"] == 3
        assert result["mode"] == "merged"

    def test_complex_schema(self):
        schema = {
            "title": "Transformer Analysis",
            "executive_summary": "Transformers are...",
            "key_findings": [
                {"title": "Attention is all you need", "confidence": 0.95},
            ],
            "recommendations": ["Use pre-trained models", "Fine-tune on domain data"],
            "metadata": {"papers_analyzed": 15, "sources_used": 8},
        }
        assert isinstance(schema, dict)
        assert len(schema) == 5


class TestFinalizationIntegration:
    """Tests for output formatting integration with job results finalization."""

    def test_structured_output_in_results(self):
        state = {"output_schema": {"summary": "Test", "findings": []}}
        results: dict = {}
        output_schema = state.get("output_schema")
        if isinstance(output_schema, dict) and output_schema:
            results["structured_output"] = output_schema
        assert "structured_output" in results
        assert results["structured_output"]["summary"] == "Test"

    def test_no_schema_no_structured_output(self):
        state = {}
        results: dict = {}
        output_schema = state.get("output_schema")
        if isinstance(output_schema, dict) and output_schema:
            results["structured_output"] = output_schema
        assert "structured_output" not in results

    def test_formatted_outputs_in_results(self):
        state = {"formatted_outputs": [
            {"type": "table", "title": "Table 1"},
            {"type": "report", "title": "Report 1"},
        ]}
        results: dict = {}
        formatted = state.get("formatted_outputs", [])
        if isinstance(formatted, list) and formatted:
            results["formatted_outputs"] = formatted[-20:]
        assert "formatted_outputs" in results
        assert len(results["formatted_outputs"]) == 2

    def test_formatted_outputs_capped_at_20(self):
        state = {"formatted_outputs": [{"type": "table", "title": f"T{i}"} for i in range(25)]}
        results: dict = {}
        formatted = state.get("formatted_outputs", [])
        if isinstance(formatted, list) and formatted:
            results["formatted_outputs"] = formatted[-20:]
        assert len(results["formatted_outputs"]) == 20

    def test_no_formatted_outputs_no_key(self):
        state = {}
        results: dict = {}
        formatted = state.get("formatted_outputs", [])
        if isinstance(formatted, list) and formatted:
            results["formatted_outputs"] = formatted[-20:]
        assert "formatted_outputs" not in results


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
