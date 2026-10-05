"""`create_chart` and `render_diagram`, called through their real handlers.

The earlier version of this file restated each handler's parameter handling
inline and asserted on its own copy, so it passed whatever the tools did. These
tests call the handlers registered in the provider. Charts are rendered by the
real matplotlib and Graphviz diagrams by the real `dot`; only two edges are
replaced: object storage, and the HTTP call to the Mermaid renderer.
"""

import shutil
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_notification_visualization_provider,
)
from app.services.mermaid_renderer import MermaidRenderer
from app.services.storage_service import storage_service
from app.services.visualization_service import VisualizationService

pytestmark = pytest.mark.unit

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
FAKE_PNG = PNG_MAGIC + b"rendered-by-the-fake-mermaid-service"
FAKE_SVG = b"<svg xmlns='http://www.w3.org/2000/svg'><rect/></svg>"

ADVERTISED_CHART_TYPES = (
    "bar",
    "line",
    "pie",
    "scatter",
    "histogram",
    "heatmap",
    "box",
    "area",
)

# One payload per chart type, each in a shape the tool schema documents.
SCHEMA_SHAPED_DATA = {
    "bar": {"labels": ["-O2", "-O3"], "values": [1.63, 1.69]},
    "line": {
        "labels": ["q1", "q2", "q3"],
        "datasets": [
            {"label": "revenue", "values": [10, 20, 15]},
            {"label": "cost", "values": [7, 9, 12]},
        ],
    },
    "pie": {"labels": ["a", "b", "c"], "values": [5, 3, 2]},
    "scatter": {"points": [{"x": 1, "y": 2}, {"x": 2, "y": 4}, {"x": 3, "y": 5}]},
    "histogram": {"labels": ["a", "b", "c", "d"], "values": [1, 2, 2, 5]},
    "heatmap": {"labels": ["a", "b"], "matrix": [[1.0, 0.2], [0.2, 1.0]]},
    "box": {"labels": ["a", "b", "c", "d"], "values": [1, 2, 2, 9]},
    "area": {"labels": ["q1", "q2", "q3"], "values": [3, 4, 6]},
}

needs_matplotlib = pytest.mark.skipif(
    not VisualizationService()._enabled, reason="matplotlib/pandas not installed"
)
needs_dot = pytest.mark.skipif(
    shutil.which("dot") is None, reason="graphviz `dot` binary not installed"
)


class FakeStorage:
    """Records what a handler stores, in place of MinIO."""

    def __init__(self):
        self.uploads = []
        self.initialized = 0
        self.fail_upload = False

    def install(self, monkeypatch):
        async def initialize():
            self.initialized += 1

        async def upload_to_path(object_path, content, content_type=None):
            if self.fail_upload:
                raise RuntimeError("bucket unavailable")
            self.uploads.append(
                {"path": object_path, "content": content, "content_type": content_type}
            )
            return object_path

        async def get_presigned_download_url(object_path, expiry=None):
            return f"https://storage.test/{object_path}?signed=1"

        monkeypatch.setattr(storage_service, "initialize", initialize)
        monkeypatch.setattr(storage_service, "upload_to_path", upload_to_path)
        monkeypatch.setattr(
            storage_service, "get_presigned_download_url", get_presigned_download_url
        )
        return self


@pytest.fixture
def storage(monkeypatch):
    return FakeStorage().install(monkeypatch)


@pytest.fixture
def job():
    return SimpleNamespace(id=uuid4(), user_id=uuid4(), name="viz job", config={})


@pytest.fixture
def call(job):
    """Call a visualization tool the way the dispatcher does."""
    provider = build_autonomous_notification_visualization_provider(SimpleNamespace())

    async def _call(tool, params):
        return await provider._handlers[tool](
            params,
            AgentToolExecutionContext(
                mode="autonomous",
                db=None,
                service=None,
                user_id=str(job.user_id),
                job=job,
                state={},
            ),
        )

    return _call


@pytest.fixture
def mermaid(monkeypatch):
    """Stand in for the Mermaid renderer's HTTP service, and record its calls.

    Only the network hop is replaced: cleaning and validating the diagram
    source still happen in the real `MermaidRenderer`.
    """
    calls = []
    behaviour = {"fail": False}

    async def _render_via_kroki(self, code, format="png", base_url=None):
        calls.append({"code": code, "format": format, "base_url": base_url})
        if behaviour["fail"]:
            raise RuntimeError("connection refused")
        return FAKE_SVG if format == "svg" else FAKE_PNG

    monkeypatch.setattr(MermaidRenderer, "_render_via_kroki", _render_via_kroki)
    return SimpleNamespace(calls=calls, behaviour=behaviour)


# ---------------------------------------------------------------------------
# create_chart
# ---------------------------------------------------------------------------


class TestCreateChartRefusals:
    async def test_chart_type_is_required(self, call, storage):
        result = await call("create_chart", {"data": SCHEMA_SHAPED_DATA["bar"]})

        assert result == {"error": "chart_type is required"}
        assert storage.uploads == []

    async def test_data_is_required(self, call, storage):
        result = await call("create_chart", {"chart_type": "bar"})

        assert "data is required" in result["error"]
        assert storage.uploads == []

    @pytest.mark.parametrize("data", ["not a dict", [1, 2, 3], 7, {}])
    async def test_data_must_be_a_non_empty_object(self, call, storage, data):
        result = await call("create_chart", {"chart_type": "bar", "data": data})

        assert "data is required and must be an object" in result["error"]
        assert storage.uploads == []

    async def test_an_unadvertised_chart_type_is_refused_by_name(self, call, storage):
        result = await call(
            "create_chart",
            {"chart_type": "treemap", "data": SCHEMA_SHAPED_DATA["bar"]},
        )

        assert "treemap" in result["error"]
        for advertised in ADVERTISED_CHART_TYPES:
            assert advertised in result["error"]
        assert storage.uploads == []

    @needs_matplotlib
    async def test_a_series_shorter_than_its_labels_is_an_error(self, call, storage):
        result = await call(
            "create_chart",
            {
                "chart_type": "bar",
                "data": {
                    "labels": ["a", "b", "c"],
                    "datasets": [{"label": "speed", "values": [1]}],
                },
            },
        )

        assert "success" not in result
        assert "'speed' has 1 values but there are 3 labels" in result["error"]
        assert storage.uploads == []

    @needs_matplotlib
    async def test_a_storage_failure_is_reported_not_swallowed(self, call, storage):
        storage.fail_upload = True

        result = await call(
            "create_chart", {"chart_type": "bar", "data": SCHEMA_SHAPED_DATA["bar"]}
        )

        assert "success" not in result
        assert "bucket unavailable" in result["error"]


@needs_matplotlib
class TestCreateChartRenders:
    def test_the_spec_advertises_exactly_the_types_tested_here(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("create_chart")
        text = (
            tool["description"]
            + " "
            + tool["parameters"]["properties"]["chart_type"]["description"]
        )

        for chart_type in ADVERTISED_CHART_TYPES:
            assert chart_type in text

    @pytest.mark.parametrize("chart_type", ADVERTISED_CHART_TYPES)
    async def test_every_advertised_type_renders_a_real_png(
        self, call, storage, job, chart_type
    ):
        result = await call(
            "create_chart",
            {"chart_type": chart_type, "data": SCHEMA_SHAPED_DATA[chart_type]},
        )

        assert result.get("success") is True, result
        assert len(storage.uploads) == 1
        upload = storage.uploads[0]
        assert upload["content"].startswith(PNG_MAGIC)
        assert len(upload["content"]) > 1000
        assert upload["content_type"] == "image/png"
        assert upload["path"].startswith(f"agent_artifacts/{job.id}/charts/")
        assert upload["path"].endswith(".png")
        assert result["data"] == {
            "chart_type": chart_type,
            "url": f"https://storage.test/{upload['path']}?signed=1",
            "format": "png",
            "size_bytes": len(upload["content"]),
        }

    async def test_chart_type_is_matched_whatever_its_case(self, call, storage):
        result = await call(
            "create_chart",
            {"chart_type": "  BAR ", "data": SCHEMA_SHAPED_DATA["bar"]},
        )

        assert result.get("success") is True, result
        assert result["data"]["chart_type"] == "bar"

    async def test_svg_is_rendered_as_svg(self, call, storage):
        result = await call(
            "create_chart",
            {
                "chart_type": "line",
                "data": SCHEMA_SHAPED_DATA["line"],
                "format": "svg",
            },
        )

        assert result.get("success") is True, result
        upload = storage.uploads[0]
        assert b"<svg" in upload["content"]
        assert not upload["content"].startswith(PNG_MAGIC)
        assert upload["path"].endswith(".svg")
        assert result["data"]["format"] == "svg"
        assert result["data"]["url"].startswith(
            f"https://storage.test/{upload['path']}"
        )

    async def test_an_svg_chart_is_stored_with_the_svg_media_type(self, call, storage):
        await call(
            "create_chart",
            {"chart_type": "bar", "data": SCHEMA_SHAPED_DATA["bar"], "format": "svg"},
        )

        assert storage.uploads[0]["content_type"] == "image/svg+xml"

    async def test_an_unknown_format_falls_back_to_png_consistently(
        self, call, storage
    ):
        result = await call(
            "create_chart",
            {"chart_type": "bar", "data": SCHEMA_SHAPED_DATA["bar"], "format": "gif"},
        )

        assert result.get("success") is True, result
        upload = storage.uploads[0]
        assert upload["content"].startswith(PNG_MAGIC)
        assert upload["path"].endswith(".png")
        assert upload["content_type"] == "image/png"
        assert result["data"]["format"] == "png"

    @pytest.mark.parametrize(
        "label_param", [{"title": "Throughput"}, {"x_label": "flag"}, {"y_label": "x"}]
    )
    async def test_title_and_axis_labels_change_the_image(
        self, call, storage, label_param
    ):
        base = {"chart_type": "bar", "data": SCHEMA_SHAPED_DATA["bar"]}

        await call("create_chart", dict(base))
        await call("create_chart", dict(base))
        await call("create_chart", {**base, **label_param})

        plain, plain_again, labelled = (u["content"] for u in storage.uploads)
        assert plain == plain_again, "rendering is not deterministic"
        assert labelled != plain, f"{label_param} did not reach the chart"

    async def test_each_chart_gets_its_own_object(self, call, storage):
        params = {"chart_type": "bar", "data": SCHEMA_SHAPED_DATA["bar"]}

        first = await call("create_chart", dict(params))
        second = await call("create_chart", dict(params))

        paths = [upload["path"] for upload in storage.uploads]
        assert len(set(paths)) == 2
        assert first["data"]["url"] != second["data"]["url"]

    async def test_several_datasets_are_all_drawn(self, call, storage):
        one = {
            "labels": ["q1", "q2"],
            "datasets": [{"label": "revenue", "values": [10, 20]}],
        }
        two = {
            "labels": ["q1", "q2"],
            "datasets": [
                {"label": "revenue", "values": [10, 20]},
                {"label": "cost", "values": [7, 9]},
            ],
        }

        await call("create_chart", {"chart_type": "bar", "data": one})
        await call("create_chart", {"chart_type": "bar", "data": two})

        assert storage.uploads[0]["content"] != storage.uploads[1]["content"]


# ---------------------------------------------------------------------------
# render_diagram
# ---------------------------------------------------------------------------

MERMAID = "graph TD\n  A-->B\n  B-->C"
DOT = "digraph G { A -> B; B -> C; }"


class TestRenderDiagramRefusals:
    @pytest.mark.parametrize("params", [{}, {"diagram_code": "   \n "}])
    async def test_diagram_code_is_required(self, call, storage, mermaid, params):
        result = await call("render_diagram", params)

        assert result == {"error": "diagram_code is required"}
        assert mermaid.calls == []
        assert storage.uploads == []

    async def test_source_that_is_not_mermaid_is_refused_before_rendering(
        self, call, storage, mermaid
    ):
        result = await call("render_diagram", {"diagram_code": "hello world"})

        assert "success" not in result
        assert "Invalid" in result["error"]
        assert mermaid.calls == []
        assert storage.uploads == []

    async def test_a_renderer_that_is_down_is_an_error(self, call, storage, mermaid):
        mermaid.behaviour["fail"] = True

        result = await call("render_diagram", {"diagram_code": MERMAID})

        assert "success" not in result
        assert "connection refused" in result["error"]
        assert storage.uploads == []

    async def test_a_storage_failure_is_reported(self, call, storage, mermaid):
        storage.fail_upload = True

        result = await call("render_diagram", {"diagram_code": MERMAID})

        assert "success" not in result
        assert "bucket unavailable" in result["error"]

    async def test_an_unsupported_diagram_type_is_refused(self, call, storage, mermaid):
        result = await call(
            "render_diagram", {"diagram_code": MERMAID, "diagram_type": "plantuml"}
        )

        assert "success" not in result
        assert "plantuml" in result["error"]
        assert storage.uploads == []


class TestRenderMermaid:
    async def test_png_is_the_default_and_is_stored_under_the_job(
        self, call, storage, mermaid, job
    ):
        result = await call("render_diagram", {"diagram_code": MERMAID})

        assert result.get("success") is True, result
        assert [c["format"] for c in mermaid.calls] == ["png"]
        assert mermaid.calls[0]["code"] == MERMAID
        assert len(storage.uploads) == 1
        upload = storage.uploads[0]
        assert upload["content"] == FAKE_PNG
        assert upload["content_type"] == "image/png"
        assert upload["path"].startswith(f"agent_artifacts/{job.id}/diagrams/")
        assert upload["path"].endswith(".png")
        assert result["data"] == {
            "url": f"https://storage.test/{upload['path']}?signed=1",
            "diagram_type": "mermaid",
            "format": "png",
            "size_bytes": len(FAKE_PNG),
        }

    async def test_svg_is_stored_as_svg_bytes(self, call, storage, mermaid):
        result = await call(
            "render_diagram", {"diagram_code": MERMAID, "format": "svg"}
        )

        assert result.get("success") is True, result
        assert [c["format"] for c in mermaid.calls] == ["svg"]
        upload = storage.uploads[0]
        assert isinstance(upload["content"], bytes)
        assert upload["content"] == FAKE_SVG
        assert upload["content_type"] == "image/svg+xml"
        assert upload["path"].endswith(".svg")
        assert result["data"]["format"] == "svg"
        assert result["data"]["size_bytes"] == len(FAKE_SVG)

    async def test_a_markdown_fence_is_stripped_before_rendering(
        self, call, storage, mermaid
    ):
        result = await call(
            "render_diagram", {"diagram_code": f"```mermaid\n{MERMAID}\n```"}
        )

        assert result.get("success") is True, result
        assert mermaid.calls[0]["code"] == MERMAID

    async def test_an_unknown_format_falls_back_to_png(self, call, storage, mermaid):
        result = await call(
            "render_diagram", {"diagram_code": MERMAID, "format": "gif"}
        )

        assert result.get("success") is True, result
        assert mermaid.calls[0]["format"] == "png"
        assert storage.uploads[0]["path"].endswith(".png")
        assert storage.uploads[0]["content_type"] == "image/png"
        assert result["data"]["format"] == "png"

    async def test_each_diagram_gets_its_own_object(self, call, storage, mermaid):
        await call("render_diagram", {"diagram_code": MERMAID})
        await call("render_diagram", {"diagram_code": MERMAID})

        assert len({upload["path"] for upload in storage.uploads}) == 2


@needs_dot
class TestRenderGraphviz:
    async def test_graphviz_renders_a_real_png(self, call, storage, mermaid, job):
        result = await call(
            "render_diagram", {"diagram_code": DOT, "diagram_type": "graphviz"}
        )

        assert result.get("success") is True, result
        assert mermaid.calls == [], "DOT source was sent to the Mermaid renderer"
        upload = storage.uploads[0]
        assert upload["content"].startswith(PNG_MAGIC)
        assert upload["content_type"] == "image/png"
        assert upload["path"].startswith(f"agent_artifacts/{job.id}/diagrams/")
        assert upload["path"].endswith(".png")
        assert result["data"]["diagram_type"] == "graphviz"
        assert result["data"]["format"] == "png"
        assert result["data"]["size_bytes"] == len(upload["content"])

    async def test_graphviz_renders_a_real_svg(self, call, storage, mermaid):
        result = await call(
            "render_diagram",
            {"diagram_code": DOT, "diagram_type": "Graphviz", "format": "svg"},
        )

        assert result.get("success") is True, result
        upload = storage.uploads[0]
        assert b"<svg" in upload["content"]
        assert upload["content_type"] == "image/svg+xml"
        assert upload["path"].endswith(".svg")

    async def test_dot_that_does_not_parse_is_an_error(self, call, storage, mermaid):
        result = await call(
            "render_diagram",
            {"diagram_code": "digraph {{{ nope", "diagram_type": "graphviz"},
        )

        assert "success" not in result
        assert result["error"].startswith("Failed to render diagram")
        assert storage.uploads == []


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


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

    def test_every_declared_parameter_is_one_the_handler_reads(self):
        """A parameter the schema offers and the handler ignores is a lie."""
        import inspect

        from app.services import agent_tool_dispatch
        from app.services.agent_tools import get_tool_by_name

        source = inspect.getsource(
            agent_tool_dispatch.build_autonomous_notification_visualization_provider
        )
        for tool_name in ("create_chart", "render_diagram"):
            for param in get_tool_by_name(tool_name)["parameters"]["properties"]:
                assert f'"{param}"' in source, f"{tool_name} never reads {param}"


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

    def test_both_tools_are_answered_by_the_provider(self):
        provider = build_autonomous_notification_visualization_provider(
            SimpleNamespace()
        )

        assert "create_chart" in provider._handlers
        assert "render_diagram" in provider._handlers


# ---------------------------------------------------------------------------
# The service underneath create_chart
# ---------------------------------------------------------------------------


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
    @needs_matplotlib
    def test_bar_chart_renders_from_labels_and_datasets(self):
        import base64

        result = VisualizationService().create_chart(
            chart_type="bar",
            data={
                "labels": ["-O2", "-O3"],
                "datasets": [{"label": "GFLOP/s", "data": [1.63, 1.69]}],
            },
            config={"title": "Throughput", "format": "png"},
        )

        assert result["mime_type"] == "image/png"
        assert base64.b64decode(result["image_base64"]).startswith(PNG_MAGIC)
