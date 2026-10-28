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
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

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
    def test_tool_identity_only_comes_from_loaded_programs(
        self, installed, monkeypatch
    ):
        report = installed(
            _EntryPoint("disabled", replace(FAKE, name="disabled")),
            _EntryPoint("broken", ImportError("unavailable")),
            _EntryPoint("fake_lab", FAKE),
            disabled="disabled",
        )
        monkeypatch.setattr(programs, "_REPORT", report)
        identity = programs.tool_program("fake_lab_measure")
        assert identity == {"name": "fake_lab", "version": "0.1"}
        identity["version"] = "changed-by-caller"
        assert programs.tool_program("fake_lab_measure")["version"] == "0.1"
        assert programs.tool_program("search_documents") is None
        assert programs.tool_program("missing") is None
        monkeypatch.setattr(programs, "_REPORT", programs.LoadReport())
        assert programs.tool_program("fake_lab_measure") is None

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

    @pytest.mark.parametrize(
        "broken, reason",
        [
            (replace(FAKE, name=[]), "name"),
            (replace(FAKE, name=" "), "name"),
            (replace(FAKE, version=None), "version"),
            (replace(FAKE, tool_specs=None), "tool_specs"),
            (replace(FAKE, tool_specs=({"name": "bad_tool"},)), "ToolSpec"),
            (replace(FAKE, tool_specs=(replace(spec("bad"), name=[]),)), "name"),
            (
                replace(FAKE, tool_specs=(replace(spec("bad"), parameters=None),)),
                "parameters",
            ),
            (
                replace(FAKE, tool_specs=(replace(spec("bad"), produces=([],)),)),
                "produces",
            ),
            (replace(FAKE, providers=(None,)), "providers"),
            (replace(FAKE, sandbox_images="image:1"), "sandbox_images"),
        ],
    )
    def test_malformed_program_is_reported_without_poisoning_the_next(
        self, installed, broken, reason
    ):
        report = installed(_EntryPoint("broken", broken), _EntryPoint("fake_lab", FAKE))

        assert report.loaded == [FAKE]
        assert report.failed[0][0] == "broken"
        assert reason in report.failed[0][1]

    def test_a_program_cannot_declare_the_same_tool_twice(self, installed):
        broken = replace(FAKE, tool_specs=(spec("repeated"), spec("repeated")))
        report = installed(_EntryPoint("broken", broken), _EntryPoint("fake_lab", FAKE))

        assert report.loaded == [FAKE]
        assert "repeated" in report.failed[0][1]

    @pytest.mark.parametrize(
        "field, value",
        [
            ("evidence_tools", "fake_lab_measure"),
            ("evidence_tools", ([],)),
            ("image_origins", None),
            ("image_origins", {"image": "Dockerfile"}),
            ("image_origins", {"image": {"dockerfile": []}}),
            ("method_sources", None),
            ("method_sources", (("fake_lab_measure",),)),
            ("method_sources", (("fake_lab_measure", "fake.method"),)),
            ("method_sources", ((("fake_lab_measure",), ""),)),
            ("alternative_fields", None),
            ("alternative_fields", {"fake_lab_measure": None}),
            ("alternative_fields", {"fake_lab_measure": (("sources", "paths"),)}),
            ("alternative_fields", {"fake_lab_measure": (("sources", "paths", True),)}),
        ],
    )
    def test_malformed_metadata_does_not_load_or_block_a_valid_program(
        self, installed, field, value
    ):
        broken = replace(FAKE, **{field: value})
        report = installed(_EntryPoint("broken", broken), _EntryPoint("fake_lab", FAKE))

        assert report.loaded == [FAKE]
        assert report.failed[0][0] == "broken"
        assert field in report.failed[0][1]

    def test_valid_optional_metadata_is_preserved(self, installed, monkeypatch):
        from app.agent_core import tool_catalog
        from app.services import (
            agent_evidence_bundle,
            agent_evidence_map,
            agent_tool_validation,
        )

        metadata = replace(
            FAKE,
            evidence_tools=("fake_lab_measure",),
            image_origins={"fake-lab": {"dockerfile": "images/Dockerfile"}},
            method_sources=((("fake_lab_measure",), "fake_lab.methods"),),
            alternative_fields={
                "fake_lab_measure": (
                    ("sources", "paths", lambda value: isinstance(value, dict)),
                )
            },
        )
        report = installed(_EntryPoint("fake_lab", metadata))
        monkeypatch.setattr(programs, "_REPORT", report)

        assert report.failed == []
        assert programs.evidence_tools() == {"fake_lab_measure"}
        assert programs.image_origins() == metadata.image_origins
        assert programs.method_sources() == list(metadata.method_sources)
        assert programs.alternative_fields() == metadata.alternative_fields
        assert "fake_lab_measure" in agent_evidence_bundle.evidence_tools()
        assert agent_evidence_bundle.image_origin("ghcr.io/example/fake-lab:1") == {
            "dockerfile": "images/Dockerfile"
        }

        methodology = ModuleType("fake_lab.methods")
        methodology.describe = lambda: ["Compare measurements against the baseline."]
        monkeypatch.setitem(sys.modules, "fake_lab.methods", methodology)
        monkeypatch.setattr(
            agent_evidence_map, "chain_for", lambda required: ["fake_lab_measure"]
        )
        assert agent_evidence_map.method_notes(["measurement"]) == [
            "Compare measurements against the baseline."
        ]

        monkeypatch.setattr(
            tool_catalog,
            "get_tool_metadata",
            lambda name: SimpleNamespace(input_schema=spec(name).parameters),
        )
        params = {"sources": ["malformed"], "paths": ["input.c"]}
        assert agent_tool_validation.coerce_tool_params("fake_lab_measure", params) == [
            "sources"
        ]
        assert params == {"paths": ["input.c"]}


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

    def test_reachability_checks_include_its_actual_providers(self):
        from tests.test_capability_reachability import dispatchable_tools

        assert "fake_lab_measure" in dispatchable_tools()

    def test_status_names_what_it_contributes(self):
        status = programs.status()

        (entry,) = status["loaded"]
        assert entry["tools"] == ["fake_lab_measure"]
        assert entry["sandbox_images"] == ["ghcr.io/example/fake-lab:1"]


def _broken_provider(_executor):
    raise RuntimeError("simulator driver could not initialize")


def _empty_provider(_executor):
    return None


def _undeclared_provider(_executor):
    return FunctionToolProvider(name="broken", handlers={"search_documents": _answer})


def _partial_provider(_executor):
    return FunctionToolProvider(name="broken", handlers={"broken_first": _answer})


class TestProviderFailuresAreLocalToAProgram:
    @pytest.mark.parametrize(
        "factories, reason",
        [
            ((), "missing"),
            ((_broken_provider,), "simulator driver"),
            ((_empty_provider,), "supported_tools"),
            ((_undeclared_provider,), "undeclared"),
            ((_partial_provider,), "missing"),
            ((_partial_provider, _broken_provider), "simulator driver"),
        ],
    )
    async def test_bad_providers_do_not_prevent_other_tools_from_running(
        self, monkeypatch, factories, reason
    ):
        from app.services.autonomous_agent_executor import AutonomousAgentExecutor

        broken = programs.ResearchProgram(
            name="broken_lab",
            tool_specs=(spec("broken_first"), spec("broken_second")),
            providers=factories,
        )
        monkeypatch.setattr(
            programs, "_REPORT", programs.LoadReport(loaded=[broken, FAKE])
        )
        executor = AutonomousAgentExecutor()
        ctx = AgentToolExecutionContext(mode="autonomous", db=None, service=executor)

        for name in ("broken_first", "broken_second"):
            handled, result = await executor.tool_registry.try_execute(name, {}, ctx)
            assert handled
            assert result["success"] is False
            assert result["error_code"] == "research_program_unavailable"
            assert "broken_lab" in result["error"] and reason in result["error"]

        handled, result = await executor.tool_registry.try_execute(
            "fake_lab_measure", {}, ctx
        )
        assert handled and result["data"]["answered"]
        assert executor.tool_registry.resolve("search_documents", ctx) is not None

    async def test_a_later_executor_retries_instead_of_disabling_the_program(
        self, monkeypatch
    ):
        from app.services.agent_tool_providers.base import AgentToolRegistry

        attempts = 0

        def build(_executor):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("temporarily unavailable")
            return provider(_executor)

        current = programs.LoadReport(loaded=[replace(FAKE, providers=(build,))])
        monkeypatch.setattr(programs, "_REPORT", current)
        ctx = AgentToolExecutionContext(mode="autonomous", db=None, service=None)

        first = AgentToolRegistry(programs.build_providers(None))
        _, result = await first.try_execute("fake_lab_measure", {}, ctx)
        assert result["success"] is False

        second = AgentToolRegistry(programs.build_providers(None))
        _, result = await second.try_execute("fake_lab_measure", {}, ctx)
        assert result["data"]["answered"]
        assert attempts == 2
        assert current.failed == []

    async def test_providers_may_share_tools_for_different_execution_modes(
        self, monkeypatch
    ):
        from app.services.agent_tool_providers.base import AgentToolRegistry

        async def answer(params, ctx):
            return {"mode": ctx.mode}

        def build(mode):
            return lambda executor: FunctionToolProvider(
                name=mode, handlers={"fake_lab_measure": answer}, modes={mode}
            )

        current = replace(FAKE, providers=(build("autonomous"), build("chat")))
        monkeypatch.setattr(programs, "_REPORT", programs.LoadReport(loaded=[current]))
        registry = AgentToolRegistry(programs.build_providers(None))
        for mode in ("autonomous", "chat"):
            ctx = AgentToolExecutionContext(mode=mode, db=None, service=None)
            handled, result = await registry.try_execute("fake_lab_measure", {}, ctx)
            assert handled and result == {"mode": mode}


class TestMethodologyFailures:
    @pytest.mark.parametrize(
        "failure", ["missing", "raises", "string", "nontext", "partial"]
    )
    def test_optional_methodology_failure_is_visible_without_hiding_valid_notes(
        self, monkeypatch, failure
    ):
        from app.services import agent_evidence_map

        def broken_description():
            if failure == "raises":
                raise RuntimeError("methodology dependency unavailable")
            if failure == "string":
                return "This must not be split into characters."
            if failure == "nontext":
                return [None]

            def partial():
                yield "Discard this incomplete note."
                raise RuntimeError("methodology stopped midway")

            return partial()

        bad_module = ModuleType("kdbc_broken_method")
        bad_module.describe = broken_description
        if failure != "missing":
            monkeypatch.setitem(sys.modules, "kdbc_broken_method", bad_module)
        good_module = ModuleType("kdbc_good_method")
        good_module.describe = lambda: ["Use a matched baseline."]
        monkeypatch.setitem(sys.modules, "kdbc_good_method", good_module)
        bad_program = replace(
            FAKE,
            name="broken_lab",
            method_sources=((("fake_lab_measure",), "kdbc_broken_method"),),
        )
        good_program = replace(
            FAKE,
            method_sources=((("fake_lab_measure",), "kdbc_good_method"),),
        )
        monkeypatch.setattr(
            programs, "_REPORT", programs.LoadReport(loaded=[bad_program, good_program])
        )
        monkeypatch.setattr(
            agent_evidence_map, "chain_for", lambda required: ["fake_lab_measure"]
        )

        notes = agent_evidence_map.method_notes(["measurement"])
        assert "Use a matched baseline." in notes
        assert any(
            "broken_lab" in note and "methodology unavailable" in note for note in notes
        )
        assert "Discard this incomplete note." not in notes
        assert len(notes) == 2

    def test_unused_methodology_is_not_imported(self, monkeypatch):
        from app.services import agent_evidence_map

        program = replace(
            FAKE,
            method_sources=((("fake_lab_measure",), "kdbc_missing_unused_method"),),
        )
        monkeypatch.setattr(programs, "_REPORT", programs.LoadReport(loaded=[program]))
        monkeypatch.setattr(agent_evidence_map, "chain_for", lambda required: [])
        assert agent_evidence_map.method_notes([]) == []


class TestAdminStatus:
    def test_admin_sees_loaded_failed_and_disabled_programs(
        self, client, admin_headers, monkeypatch
    ):
        monkeypatch.setattr(
            programs,
            "_REPORT",
            programs.LoadReport(
                loaded=[FAKE],
                failed=[("broken_lab", "could not be loaded: missing driver")],
                disabled=["paused_lab"],
            ),
        )
        response = client.get("/api/v1/admin/research-programs", headers=admin_headers)

        assert response.status_code == 200
        assert response.json() == {
            "loaded": [
                {
                    "name": "fake_lab",
                    "version": "0.1",
                    "description": "",
                    "tools": ["fake_lab_measure"],
                    "sandbox_images": ["ghcr.io/example/fake-lab:1"],
                }
            ],
            "failed": [
                {"name": "broken_lab", "reason": "could not be loaded: missing driver"}
            ],
            "disabled": ["paused_lab"],
        }

    def test_regular_users_cannot_read_program_status(
        self, client, auth_headers, monkeypatch
    ):
        def unexpected_status():
            pytest.fail(
                "Program status must not be read before checking administrator access"
            )

        monkeypatch.setattr(programs, "status", unexpected_status)
        response = client.get("/api/v1/admin/research-programs", headers=auth_headers)
        assert response.status_code == 403

    def test_anonymous_users_cannot_read_program_status(self, client):
        response = client.get("/api/v1/admin/research-programs")
        assert response.status_code in (401, 403)


@pytest.mark.parametrize("disabled", [False, True])
def test_an_installed_distribution_joins_the_catalog(tmp_path, disabled):
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
            BROKEN = ResearchProgram(name="broken_lab", tool_specs=(object(),))
            """
        )
    )
    dist = tmp_path / "fake_lab_pkg-0.1.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: fake-lab-pkg\nVersion: 0.1\n"
    )
    (dist / "entry_points.txt").write_text(
        "[kdbc.research_programs]\n"
        "broken_lab = fake_lab_pkg:BROKEN\n"
        "fake_lab = fake_lab_pkg:PROGRAM\n"
    )
    probe = textwrap.dedent(
        """
        # Asked first, before anything has imported the catalog that loads
        # programs: the answer must not depend on import order.
        from app.services import agent_sandbox_runtime
        print("image", "ghcr.io/example/fake-lab:1" in agent_sandbox_runtime.allowed_images())

        from app.agent_core import tool_specs
        from app.agent_core import research_programs
        from app.services import agent_evidence_map
        from app.services import agent_evidence_bundle
        from pathlib import Path
        from tempfile import TemporaryDirectory

        names = [s["name"] for s in tool_specs.schemas()]
        print("schema", "fake_lab_measure" in names)
        print("domain", tool_specs.TOOL_DOMAINS.get("fake_lab_measure"))
        print("evidence", "fake_lab_measurement" in agent_evidence_map._PRODUCERS)
        failures = research_programs.status()["failed"]
        print("broken reported", any(f["name"] == "broken_lab" and "ToolSpec" in f["reason"] for f in failures))
        with TemporaryDirectory() as directory:
            entry = agent_evidence_bundle.record_entry(
                job_id="installed-program-probe", tool="fake_lab_measure",
                params={}, result={"success": True}, root=Path(directory),
            )
            print("recorded program", entry.get("research_program"))
        """
    )

    run = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=BACKEND,
        env={
            **os.environ,
            "PYTHONPATH": f"{tmp_path}:{BACKEND}",
            "RESEARCH_PROGRAMS_DISABLED": "fake_lab" if disabled else "",
        },
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert run.returncode == 0, run.stderr[-2000:]
    expected = "False" if disabled else "True"
    assert f"image {expected}" in run.stdout
    assert f"schema {expected}" in run.stdout
    assert f"domain {'None' if disabled else 'fake_lab'}" in run.stdout
    assert f"evidence {expected}" in run.stdout
    assert "broken reported True" in run.stdout
    identity = "None" if disabled else "{'name': 'fake_lab', 'version': '0.1'}"
    assert f"recorded program {identity}" in run.stdout
