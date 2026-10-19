"""Research programs: domain code installed as a package, not kept in core.

A program is found through an entry point, and its tools join the built-in
catalog. These tests drive the loader with stand-in entry points, and one runs
a fresh interpreter with a real installed distribution, because the catalog is
built at import and only a new process shows what a real install does.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from app.agent_core import research_programs as programs
from app.agent_core.tool_specs.spec import ToolSpec
from app.services.agent_tool_providers.base import (
    AgentToolExecutionContext,
    FunctionToolProvider,
)

pytestmark = pytest.mark.unit

BACKEND = Path(__file__).resolve().parents[1]


def spec(name):
    return ToolSpec(
        name=name,
        description=f"{name} does a thing.",
        parameters={"type": "object", "properties": {}, "required": []},
    )


async def _answer(params, ctx):
    return {"success": True, "data": {"answered": True}}


def provider(executor):
    return FunctionToolProvider(
        name="fake_lab", handlers={"fake_lab_measure": _answer}, modes={"autonomous"}
    )


FAKE = programs.ResearchProgram(
    name="fake_lab",
    version="0.1",
    tool_specs=(spec("fake_lab_measure"),),
    providers=(provider,),
    sandbox_images=("ghcr.io/example/fake-lab:1",),
)


class _EntryPoint:
    def __init__(self, name, value):
        self.name = name
        self.value = f"fake:{name}"
        self._value = value

    def load(self):
        if isinstance(self._value, Exception):
            raise self._value
        return self._value


@pytest.fixture
def installed(monkeypatch):
    def install(*entry_points, disabled=""):
        monkeypatch.setattr(programs, "_entry_points", lambda: list(entry_points))
        monkeypatch.setattr(
            programs, "_disabled_names", lambda: {x for x in disabled.split(",") if x}
        )
        return programs.load(reserved_tool_names=["search_documents"])

    return install


class TestLoading:
    def test_a_program_is_loaded_with_its_tools(self, installed):
        report = installed(_EntryPoint("fake_lab", FAKE))

        assert report.loaded == [FAKE]
        assert report.failed == []

    def test_an_entry_point_may_be_a_factory(self, installed):
        report = installed(_EntryPoint("fake_lab", lambda: FAKE))

        assert report.loaded == [FAKE]

    def test_a_program_that_cannot_load_is_reported_not_raised(self, installed):
        report = installed(
            _EntryPoint("broken", ImportError("no module named gem5_driver")),
            _EntryPoint("fake_lab", FAKE),
        )

        assert report.loaded == [FAKE]
        ((name, reason),) = report.failed
        assert name == "broken" and "gem5_driver" in reason

    def test_something_that_is_not_a_program_is_refused(self, installed):
        report = installed(_EntryPoint("odd", {"name": "odd"}))

        assert report.loaded == []
        assert "not a ResearchProgram" in report.failed[0][1]

    def test_a_program_may_not_shadow_a_built_in(self, installed):
        rogue = programs.ResearchProgram(
            name="rogue", tool_specs=(spec("search_documents"), spec("rogue_x"))
        )

        report = installed(_EntryPoint("rogue", rogue))

        assert report.loaded == []
        assert "search_documents" in report.failed[0][1]

    def test_two_programs_may_not_declare_the_same_tool(self, installed):
        twin = programs.ResearchProgram(
            name="twin", tool_specs=(spec("fake_lab_measure"),)
        )

        report = installed(_EntryPoint("fake_lab", FAKE), _EntryPoint("twin", twin))

        assert report.loaded == [FAKE]
        assert report.failed[0][0] == "twin"

    def test_a_disabled_program_is_left_out_and_said(self, installed):
        report = installed(_EntryPoint("fake_lab", FAKE), disabled="fake_lab")

        assert report.loaded == []
        assert report.disabled == ["fake_lab"]


class TestWhatALoadedProgramReaches:
    @pytest.fixture(autouse=True)
    def loaded(self, monkeypatch):
        monkeypatch.setattr(programs, "_REPORT", programs.LoadReport(loaded=[FAKE]))

    async def test_its_handlers_are_registered_on_the_executor(self):
        from app.services.autonomous_agent_executor import AutonomousAgentExecutor

        executor = AutonomousAgentExecutor()
        ctx = AgentToolExecutionContext(mode="autonomous", db=None, service=executor)

        handled, result = await executor.tool_registry.try_execute(
            "fake_lab_measure", {}, ctx
        )

        assert handled is True
        assert result["data"] == {"answered": True}

    def test_its_images_are_allowed(self):
        from app.services import agent_sandbox_runtime

        assert "ghcr.io/example/fake-lab:1" in agent_sandbox_runtime.allowed_images()

    def test_status_names_what_it_contributes(self):
        status = programs.status()

        (entry,) = status["loaded"]
        assert entry["tools"] == ["fake_lab_measure"]
        assert entry["sandbox_images"] == ["ghcr.io/example/fake-lab:1"]


def test_an_installed_distribution_joins_the_catalog(tmp_path):
    """A real entry point, in a fresh process: the catalog is built at import."""
    package = tmp_path / "fake_lab_pkg"
    package.mkdir()
    (package / "__init__.py").write_text(
        textwrap.dedent(
            """
            from app.agent_core.research_programs import ResearchProgram
            from app.agent_core.tool_specs.spec import ToolSpec

            PROGRAM = ResearchProgram(
                name="fake_lab",
                version="0.1",
                sandbox_images=("ghcr.io/example/fake-lab:1",),
                tool_specs=(
                    ToolSpec(
                        name="fake_lab_measure",
                        description="Measure a fake thing.",
                        parameters={"type": "object", "properties": {}},
                        produces=("fake_lab_measurement",),
                    ),
                ),
            )
            """
        )
    )
    dist = tmp_path / "fake_lab_pkg-0.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: fake-lab-pkg\nVersion: 0.1\n"
    )
    (dist / "entry_points.txt").write_text(
        "[kdbc.research_programs]\nfake_lab = fake_lab_pkg:PROGRAM\n"
    )
    probe = textwrap.dedent(
        """
        # Asked first, before anything has imported the catalog that loads
        # programs: the answer must not depend on import order.
        from app.services import agent_sandbox_runtime
        print("image", "ghcr.io/example/fake-lab:1" in agent_sandbox_runtime.allowed_images())

        from app.agent_core import tool_specs
        from app.services import agent_evidence_map

        names = [s["name"] for s in tool_specs.schemas()]
        print("schema", "fake_lab_measure" in names)
        print("domain", tool_specs.TOOL_DOMAINS.get("fake_lab_measure"))
        print("evidence", "fake_lab_measurement" in agent_evidence_map._PRODUCERS)
        """
    )

    run = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=BACKEND,
        env={
            **os.environ,
            "PYTHONPATH": f"{tmp_path}:{BACKEND}",
        },
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert run.returncode == 0, run.stderr[-2000:]
    assert "image True" in run.stdout
    assert "schema True" in run.stdout
    assert "domain fake_lab" in run.stdout
    assert "evidence True" in run.stdout
