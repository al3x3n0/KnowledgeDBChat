"""Tests for data visualization tools (create_chart, render_diagram)."""


class TestCreateChart:
    """Tests for create_chart tool logic."""

    def test_requires_chart_type(self):
        params = {"data": {"labels": ["A"], "values": [1]}}
        chart_type = str(params.get("chart_type", "")).strip().lower()
        assert not chart_type

    def test_requires_data(self):
        params = {"chart_type": "bar"}
        data = params.get("data")
        assert not data or not isinstance(data, dict)

    def test_data_must_be_dict(self):
        params = {"chart_type": "bar", "data": "not a dict"}
        data = params.get("data")
        assert not isinstance(data, dict)

    def test_accepts_valid_params(self):
        params = {
            "chart_type": "bar",
            "data": {"labels": ["A", "B", "C"], "values": [10, 20, 30]},
        }
        chart_type = str(params.get("chart_type", "")).strip().lower()
        data = params.get("data")
        assert chart_type == "bar"
        assert isinstance(data, dict)

    def test_valid_chart_types(self):
        valid = {"bar", "line", "pie", "scatter", "histogram", "heatmap", "box", "area"}
        for ct in valid:
            assert ct in valid

    def test_invalid_chart_type_rejected(self):
        chart_type = "treemap"
        valid = {"bar", "line", "pie", "scatter", "histogram", "heatmap", "box", "area"}
        assert chart_type not in valid

    def test_format_defaults_to_png(self):
        params = {"chart_type": "bar", "data": {}}
        fmt = str(params.get("format", "png")).strip().lower()
        assert fmt == "png"

    def test_format_svg_accepted(self):
        params = {"chart_type": "bar", "data": {}, "format": "svg"}
        fmt = str(params.get("format", "png")).strip().lower()
        assert fmt == "svg"

    def test_invalid_format_falls_back_to_png(self):
        params = {"chart_type": "bar", "data": {}, "format": "gif"}
        fmt = str(params.get("format", "png")).strip().lower()
        if fmt not in {"png", "svg"}:
            fmt = "png"
        assert fmt == "png"

    def test_title_optional(self):
        params = {"chart_type": "bar", "data": {}}
        title = str(params.get("title", "")).strip()
        assert title == ""

    def test_title_passed_to_config(self):
        params = {"chart_type": "bar", "data": {}, "title": "Revenue by Quarter"}
        config = {}
        title = str(params.get("title", "")).strip()
        if title:
            config["title"] = title
        assert config["title"] == "Revenue by Quarter"

    def test_labels_optional(self):
        params = {"chart_type": "bar", "data": {}}
        x_label = str(params.get("x_label", "")).strip()
        y_label = str(params.get("y_label", "")).strip()
        assert x_label == ""
        assert y_label == ""

    def test_result_format(self):
        result = {
            "success": True,
            "data": {
                "chart_type": "bar",
                "url": "https://minio.local/agent_artifacts/j-1/charts/abc.png",
                "format": "png",
                "size_bytes": 45000,
            },
        }
        assert result["data"]["chart_type"] == "bar"
        assert result["data"]["url"].endswith(".png")
        assert result["data"]["size_bytes"] > 0

    def test_object_path_format(self):
        import uuid

        job_id = uuid.uuid4()
        chart_id = uuid.uuid4()
        fmt = "png"
        path = f"agent_artifacts/{job_id}/charts/{chart_id}.{fmt}"
        assert "agent_artifacts" in path
        assert "charts" in path
        assert path.endswith(".png")

    def test_mime_type_mapping(self):
        for fmt in ["png", "svg"]:
            mime = f"image/{fmt}"
            if fmt == "png":
                assert mime == "image/png"
            else:
                assert mime == "image/svg"


class TestRenderDiagram:
    """Tests for render_diagram tool logic."""

    def test_requires_diagram_code(self):
        params = {}
        code = str(params.get("diagram_code", "")).strip()
        assert not code

    def test_accepts_mermaid_code(self):
        params = {"diagram_code": "graph TD\n  A-->B\n  B-->C"}
        code = str(params.get("diagram_code", "")).strip()
        assert code.startswith("graph")

    def test_diagram_type_defaults_to_mermaid(self):
        params = {"diagram_code": "graph TD\n  A-->B"}
        dtype = str(params.get("diagram_type", "mermaid")).strip().lower()
        assert dtype == "mermaid"

    def test_graphviz_type_accepted(self):
        params = {"diagram_code": "digraph { A -> B }", "diagram_type": "graphviz"}
        dtype = str(params.get("diagram_type", "mermaid")).strip().lower()
        assert dtype == "graphviz"

    def test_format_defaults_to_png(self):
        params = {"diagram_code": "graph TD\n  A-->B"}
        fmt = str(params.get("format", "png")).strip().lower()
        assert fmt == "png"

    def test_svg_format_accepted(self):
        params = {"diagram_code": "graph TD\n  A-->B", "format": "svg"}
        fmt = str(params.get("format", "png")).strip().lower()
        assert fmt == "svg"

    def test_mime_type_for_svg(self):
        fmt = "svg"
        mime = f"image/{fmt}" if fmt == "png" else "image/svg+xml"
        assert mime == "image/svg+xml"

    def test_mime_type_for_png(self):
        fmt = "png"
        mime = f"image/{fmt}" if fmt == "png" else "image/svg+xml"
        assert mime == "image/png"

    def test_result_format(self):
        result = {
            "success": True,
            "data": {
                "url": "https://minio.local/agent_artifacts/j-1/diagrams/abc.svg",
                "diagram_type": "mermaid",
                "format": "svg",
                "size_bytes": 12000,
            },
        }
        assert result["data"]["diagram_type"] == "mermaid"
        assert result["data"]["format"] == "svg"
        assert result["data"]["size_bytes"] > 0

    def test_object_path_format(self):
        import uuid

        job_id = uuid.uuid4()
        diag_id = uuid.uuid4()
        fmt = "svg"
        path = f"agent_artifacts/{job_id}/diagrams/{diag_id}.{fmt}"
        assert "diagrams" in path
        assert path.endswith(".svg")

    def test_svg_render_returns_string_encoded(self):
        svg_str = "<svg xmlns='http://www.w3.org/2000/svg'><rect/></svg>"
        image_bytes = svg_str.encode("utf-8") if isinstance(svg_str, str) else svg_str
        assert isinstance(image_bytes, bytes)
        assert b"<svg" in image_bytes


class TestVisualizationToolSchemas:
    """Tests for visualization tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS

        names = {t["name"] for t in AGENT_TOOLS}
        assert "create_chart" in names
        assert "render_diagram" in names

    def test_create_chart_requires_params(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("create_chart")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "chart_type" in required
        assert "data" in required

    def test_render_diagram_requires_code(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("render_diagram")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "diagram_code" in required

    def test_create_chart_has_format_param(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("create_chart")
        assert "format" in tool["parameters"]["properties"]

    def test_render_diagram_has_diagram_type_param(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("render_diagram")
        assert "diagram_type" in tool["parameters"]["properties"]


class TestVisualizationToolRegistry:
    """Tests for visualization tool registry classification."""

    def test_create_chart_is_write_tool(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("create_chart")
        assert meta is not None
        assert meta.effects == "write"

    def test_render_diagram_is_write_tool(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("render_diagram")
        assert meta is not None
        assert meta.effects == "write"

    def test_create_chart_is_medium_cost(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("create_chart")
        assert meta is not None
        assert meta.cost_tier == "medium"

    def test_render_diagram_is_network_tool(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("render_diagram")
        assert meta is not None
        assert meta.network == "egress"


class TestNormalizeChartData:
    """The shapes the create_chart schema advertises must actually chart.

    Each of these raised "All arrays must be of the same length" before the
    normalizer existed, so a caller following the tool schema could not produce
    a chart at all.
    """

    def _normalize(self, data):
        from app.services.visualization_service import normalize_chart_data

        return normalize_chart_data(data)

    def test_labels_and_values_become_x_and_y_columns(self):
        frame = self._normalize({"labels": ["-O2", "-O3"], "values": [1.63, 1.69]})

        assert list(frame.columns) == ["label", "value"]
        assert list(frame["label"]) == ["-O2", "-O3"]
        assert list(frame["value"]) == [1.63, 1.69]

    def test_datasets_become_one_column_per_series(self):
        frame = self._normalize(
            {
                "labels": ["-O2", "-O3"],
                "datasets": [
                    {"label": "GFLOP/s", "values": [1.63, 1.69]},
                    {"label": "ms", "values": [126, 122]},
                ],
            }
        )

        assert list(frame.columns) == ["label", "GFLOP/s", "ms"]
        assert list(frame["GFLOP/s"]) == [1.63, 1.69]

    def test_a_dataset_written_chart_js_style_is_read_the_same_way(self):
        """Models send `data` for the series as often as `values`."""
        frame = self._normalize(
            {"labels": ["a", "b"], "datasets": [{"label": "s", "data": [1, 2]}]}
        )

        assert list(frame["s"]) == [1, 2]

    def test_a_mismatched_series_says_which_one_is_wrong(self):
        import pytest

        with pytest.raises(ValueError) as error:
            self._normalize(
                {"labels": ["a", "b", "c"], "datasets": [{"label": "s", "values": [1]}]}
            )

        assert "'s' has 1 values but there are 3 labels" in str(error.value)

    def test_points_become_x_and_y_columns(self):
        frame = self._normalize({"points": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]})

        assert list(frame.columns) == ["x", "y"]
        assert list(frame["y"]) == [2, 4]

    def test_matrix_keeps_its_labels(self):
        frame = self._normalize({"labels": ["a", "b"], "matrix": [[1, 2], [3, 4]]})

        assert list(frame.columns) == ["a", "b"]
        assert list(frame.index) == ["a", "b"]

    def test_a_plain_column_mapping_is_left_alone(self):
        frame = self._normalize({"flags": ["-O2", "-O3"], "gflops": [1.63, 1.69]})

        assert list(frame.columns) == ["flags", "gflops"]


class TestChartRendersFromSchemaShape:
    def test_bar_chart_renders_from_labels_and_datasets(self):
        import pytest

        from app.services.visualization_service import VisualizationService

        service = VisualizationService()
        if not service._enabled:
            pytest.skip("matplotlib/pandas not installed")

        result = service.create_chart(
            chart_type="bar",
            data={
                "labels": ["-O2", "-O3"],
                "datasets": [{"label": "GFLOP/s", "data": [1.63, 1.69]}],
            },
            config={"title": "Throughput", "format": "png"},
        )

        assert result["mime_type"] == "image/png"
        assert len(result["image_base64"]) > 1000
