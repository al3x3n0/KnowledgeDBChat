"""The compiler/BOLT/gem5 tool handlers, called through the real dispatch layer.

The services behind these tools have their own tests (test_agent_restructure,
test_agent_bolt, test_agent_pass_from_rewrite, ...). What nothing exercised was
the HANDLER in `agent_tool_dispatch`: which parameters it reads, whether what
it passes fits the service it calls, what it refuses before spending a
sandbox, and how a sandbox that never ran reaches the caller.

Two layers of fake, each binding every call against the real callee's
signature so a wrong keyword fails here rather than behind `except Exception`
in a run:

- the SERVICE function the handler delegates to (`Recorder`), for parameter
  plumbing and refusals;
- the SANDBOX (`agent_sandbox_runtime.run_in_sandbox`, `FakeSandbox`) and the
  MODEL (`LLMService.generate_structured`, `FakeModel`), for how the real
  service reports a run that could not happen.

`TestLive` runs a few handlers end to end against the local Docker daemon. It
is marked slow and skips where Docker or the image is missing.
"""

import asyncio
import base64
import inspect
import json
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.agent_core.tool_specs import spec_for
from app.services import (
    agent_binary_rewrite,
    agent_bolt,
    agent_gem5_sandbox,
    agent_gem5_studies,
    agent_pass_builder,
    agent_pass_from_rewrite,
    agent_restructure,
    agent_restructure_proposer,
    agent_sandbox_runtime,
)
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_workspace_mutation_provider,
)
from app.services.coding_workspace_manager import CodingWorkspaceManager
from app.services.llm_service import LLMService

unit = pytest.mark.unit

COMPILER_IMAGE = "ghcr.io/al3x3n0/kdbc-compiler-research:latest"
PASS_IMAGE = "ghcr.io/al3x3n0/kdbc-pass-dev:latest"
BOLT_IMAGE = "ghcr.io/al3x3n0/kdbc-bolt-research:latest"
GEM5_IMAGE = "ghcr.io/al3x3n0/kdbc-gem5-research:latest"

#: tool -> (module, function) the handler delegates to.
SERVICES = {
    "build_llvm_pass": (agent_pass_builder, "build_llvm_pass"),
    "disassemble_symbol": (agent_binary_rewrite, "disassemble_symbol"),
    "evaluate_restructuring": (agent_restructure, "evaluate_restructuring"),
    "propose_binary_rewrites": (agent_restructure_proposer, "propose_binary_rewrites"),
    "evaluate_binary_rewrite": (agent_binary_rewrite, "evaluate_binary_rewrite"),
    "synthesize_pass_from_rewrite": (
        agent_pass_from_rewrite,
        "synthesize_pass_from_rewrite",
    ),
    "propose_bolt_configurations": (agent_bolt, "propose_bolt_configurations"),
    "optimize_executable_with_bolt": (agent_bolt, "optimize_executable"),
    "measure_marginal": (agent_gem5_studies, "measure_marginal"),
}
TOOLS = tuple(SERVICES)
#: Tools that call a model, and so need to know whose model settings to use.
MODEL_TOOLS = (
    "propose_binary_rewrites",
    "synthesize_pass_from_rewrite",
    "propose_bolt_configurations",
)

KERNEL = "long work(long n) { long s = 0; for (long i = 0; i < n; i++) s += i % 7; return s; }\n"
DRIVER = (
    "#include <stdio.h>\nlong work(long n);\n"
    'int main(void) { long n; if (scanf("%ld", &n) != 1) return 1; '
    'printf("%ld\\n", work(n)); return 0; }\n'
)
ASM = ".text\n.globl work\n.type work, %function\nwork:\n  mov x0, #0\n  ret\n"
OBJECT_B64 = base64.b64encode(b"\x7fELF" + b"\0" * 60).decode("ascii")
DB = object()
NO_SUCH_IMAGE = "docker: Error response from daemon: No such image: {image}.\n"

_RUN_IN_SANDBOX = inspect.signature(agent_sandbox_runtime.run_in_sandbox)
_GENERATE_STRUCTURED = inspect.signature(LLMService.generate_structured)


# --------------------------------------------------------------------------- #
# Fakes and plumbing.
# --------------------------------------------------------------------------- #


class FakeWorkspaces:
    """The coding workspace manager: one workspace, or none."""

    def __init__(self, base_path=None):
        self.base_path = base_path
        self.asked = []

    def get_or_default(self, workspace_id, state, *, job=None):
        inspect.signature(CodingWorkspaceManager.get_or_default).bind(
            None, workspace_id, state, job=job
        )
        self.asked.append(workspace_id)
        if self.base_path is None:
            return None
        return SimpleNamespace(base_path=self.base_path, workspace_id="ws-1")


def _provider(base_path=None):
    executor = SimpleNamespace(workspace_manager=FakeWorkspaces(base_path))
    return build_autonomous_workspace_mutation_provider(executor)


def _job():
    return SimpleNamespace(id=uuid4(), user_id=uuid4(), config={}, iteration=1)


def _ctx(job=None, *, with_user_id=True):
    job = job or _job()
    return AgentToolExecutionContext(
        mode="autonomous",
        db=DB,
        service=None,
        user_id=job.user_id if with_user_id else None,
        job=job,
        state={},
    )


async def _call(tool, params, ctx=None, base_path=None):
    return await _provider(base_path)._handlers[tool](params, ctx or _ctx())


class Recorder:
    """A service function: binds each call against the real one, then answers."""

    def __init__(self, real, reply=None):
        self.signature = inspect.signature(real)
        self.calls = []
        self.reply = reply if reply is not None else {"success": True, "data": {}}

    async def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        self.calls.append(dict(bound.arguments))
        return self.reply


@pytest.fixture
def service(monkeypatch):
    """Replace the service behind `tool` with a Recorder, and return it."""

    def install(tool, reply=None):
        module, name = SERVICES[tool]
        recorder = Recorder(getattr(module, name), reply)
        monkeypatch.setattr(module, name, recorder)
        return recorder

    return install


class FakeSandbox:
    """`run_in_sandbox`: records each call; `respond(arguments)` answers it."""

    def __init__(self, respond):
        self.respond = respond
        self.calls = []

    async def __call__(self, *args, **kwargs):
        bound = _RUN_IN_SANDBOX.bind(*args, **kwargs)
        self.calls.append(dict(bound.arguments))
        out = self.respond(bound.arguments)
        if isinstance(out, BaseException):
            raise out
        return out


class FakeModel:
    """`LLMService.generate_structured`: answers from `replies`, in order."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def install(self, monkeypatch):
        model = self

        async def generate_structured(self_, *args, **kwargs):
            _GENERATE_STRUCTURED.bind(self_, *args, **kwargs)
            model.calls.append(kwargs)
            reply = model.replies[min(len(model.calls), len(model.replies)) - 1]
            return SimpleNamespace(
                structured=reply, text=json.dumps(reply), stop_reason="end_turn"
            )

        monkeypatch.setattr(LLMService, "generate_structured", generate_structured)
        return self


@pytest.fixture
def sandbox_enabled(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "ENABLE_UNSAFE_CODE_EXECUTION", True)
    monkeypatch.setattr(
        settings,
        "SCIENTIFIC_VALIDATION_ALLOWED_DOCKER_IMAGES",
        ",".join((COMPILER_IMAGE, PASS_IMAGE, BOLT_IMAGE, GEM5_IMAGE)),
    )
    agent_gem5_sandbox.forget_model_support()
    yield
    agent_gem5_sandbox.forget_model_support()


@pytest.fixture
def sandbox(monkeypatch, sandbox_enabled):
    def install(respond):
        fake = FakeSandbox(respond)
        monkeypatch.setattr(agent_sandbox_runtime, "run_in_sandbox", fake)
        return fake

    return install


# --------------------------------------------------------------------------- #
# What each handler passes to its service.
# --------------------------------------------------------------------------- #

PROGRAM = {
    "sources": {"prog.c": "int main(void) { return 0; }"},
    "inputs": ["10", "20"],
    "run_args": "- 3",
    "profile_run_args": "- 30",
    "build_flags": "-O3",
    "libs": "-lpthread",
    "profile_inputs": [1],
    "bench_input": 0,
    "label": "lua",
}

#: Every property each tool's spec declares, set to a value that can be told
#: apart, and the arguments the service must then receive.
CARRIED = {
    "build_llvm_pass": (
        {
            "source": "SRC",
            "pass_name": "mark-cold",
            "test_code": "int f(void);",
            "flags": "-O2",
            "label": "L",
        },
        {
            "source": "SRC",
            "pass_name": "mark-cold",
            "test_code": "int f(void);",
            "flags": "-O2",
            "label": "L",
        },
    ),
    "disassemble_symbol": (
        {"symbol": "work", "object_b64": OBJECT_B64, "kernel": KERNEL, "flags": "-O3"},
        {"symbol": "work", "object_b64": OBJECT_B64, "kernel": KERNEL, "flags": "-O3"},
    ),
    "evaluate_restructuring": (
        {
            "kernel": KERNEL,
            "candidate": "CAND",
            "driver": DRIVER,
            "inputs": ["1", "2"],
            "bench_input": 1,
            "flags": "-O1",
            "label": "L",
            "reference": {"adapter": "A", "paths": ["src/a.c"]},
            "value_preserving": False,
            "invariant": "n < 2^31",
            "trials": 9,
        },
        {
            "kernel": KERNEL,
            "candidate": "CAND",
            "driver": DRIVER,
            "inputs": ["1", "2"],
            "bench_input": 1,
            "flags": "-O1",
            "label": "L",
            "reference": {"adapter": "A", "paths": ["src/a.c"]},
            "value_preserving": False,
            "invariant": "n < 2^31",
            "trials": 9,
        },
    ),
    "propose_binary_rewrites": (
        {
            "symbol": "work",
            "object_b64": OBJECT_B64,
            "kernel": KERNEL,
            "driver": DRIVER,
            "inputs": ["1", "2"],
            "bench_input": 1,
            "flags": "-O1",
            "label": "L",
            "focus": "the modulo",
            "count": 2,
        },
        {
            "symbol": "work",
            "object_b64": OBJECT_B64,
            "kernel": KERNEL,
            "driver": DRIVER,
            "inputs": ["1", "2"],
            "bench_input": 1,
            "flags": "-O1",
            "label": "L",
            "focus": "the modulo",
            "count": 2,
            "db": DB,
        },
    ),
    "evaluate_binary_rewrite": (
        {
            "symbol": "work",
            "replacement_asm": ASM,
            "object_b64": OBJECT_B64,
            "kernel": KERNEL,
            "baseline_asm": "BASE",
            "driver": DRIVER,
            "inputs": ["1", "2"],
            "bench_input": 1,
            "flags": "-O1",
            "label": "L",
            "value_preserving": False,
            "invariant": "inv",
            "trials": 5,
        },
        {
            "symbol": "work",
            "replacement_asm": ASM,
            "object_b64": OBJECT_B64,
            "kernel": KERNEL,
            "baseline_asm": "BASE",
            "driver": DRIVER,
            "inputs": ["1", "2"],
            "bench_input": 1,
            "flags": "-O1",
            "label": "L",
            "value_preserving": False,
            "invariant": "inv",
            "trials": 5,
        },
    ),
    "synthesize_pass_from_rewrite": (
        {
            "kernel": KERNEL,
            "rewrite_kernel": "REWRITE",
            "driver": DRIVER,
            "inputs": ["1", "2"],
            "bench_input": 1,
            "flags": "-O1",
            "label": "L",
            "idea": "a table",
            "invariant": "bytes",
        },
        {
            "kernel": KERNEL,
            "rewrite_kernel": "REWRITE",
            "driver": DRIVER,
            "inputs": ["1", "2"],
            "bench_input": 1,
            "flags": "-O1",
            "label": "L",
            "idea": "a table",
            "invariant": "bytes",
            "db": DB,
        },
    ),
    "propose_bolt_configurations": (
        {
            **PROGRAM,
            "paths": ["src/a.c"],
            "include_dirs": ["src"],
            "workspace_id": "ws-1",
            "focus": "an interpreter",
            "count": 4,
        },
        {
            **{k: v for k, v in PROGRAM.items()},
            "focus": "an interpreter",
            "count": 4,
            "db": DB,
        },
    ),
    "optimize_executable_with_bolt": (
        {
            **PROGRAM,
            "paths": ["src/a.c"],
            "include_dirs": ["src"],
            "workspace_id": "ws-1",
            "options": "-reorder-blocks=ext-tsp",
            "rationale": "a dispatch loop",
            "measure": "cycles",
            "core": "O3CPU",
            "trials": 9,
        },
        {
            **{k: v for k, v in PROGRAM.items()},
            "options": "-reorder-blocks=ext-tsp",
            "rationale": "a dispatch loop",
            "measure": "cycles",
            "core": "O3CPU",
            "trials": 9,
        },
    ),
    "measure_marginal": (
        {
            "code": "for(int r=0;r<REPS;r++);",
            "configs": {
                "stride": {"caches": {"l2": {"prefetcher": "StridePrefetcher"}}}
            },
            "reps": [4, 16],
            "memory_bound": False,
            "flags": "-O1 -static",
            "label": "L",
        },
        {
            "code": "for(int r=0;r<REPS;r++);",
            "configs": {
                "stride": {"caches": {"l2": {"prefetcher": "StridePrefetcher"}}}
            },
            "reps": [4, 16],
            "memory_bound": False,
            "flags": "-O1 -static",
            "label": "L",
        },
    ),
}


class Tracking(dict):
    """A params dict that remembers which keys the handler looked at."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.read = set()

    def get(self, key, default=None):
        self.read.add(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        self.read.add(key)
        return super().__contains__(key)


@unit
class TestWhatEachHandlerPasses:
    @pytest.mark.parametrize("tool", TOOLS)
    def test_the_table_covers_every_declared_parameter(self, tool):
        declared = set(spec_for(tool).parameters["properties"])
        assert declared <= set(CARRIED[tool][0])

    @pytest.mark.parametrize("tool", TOOLS)
    async def test_every_argument_fits_the_service_and_arrives_intact(
        self, tool, service, tmp_path
    ):
        recorder = service(tool)
        params, expected = CARRIED[tool]

        result = await _call(tool, dict(params), base_path=tmp_path)

        assert result == recorder.reply
        assert len(recorder.calls) == 1
        call = recorder.calls[0]
        for key, value in expected.items():
            assert call[key] == value, key

    @pytest.mark.parametrize("tool", TOOLS)
    async def test_every_declared_parameter_is_read(self, tool, service, tmp_path):
        service(tool)
        params = Tracking(CARRIED[tool][0])
        if tool in ("propose_bolt_configurations", "optimize_executable_with_bolt"):
            # sources and paths are alternatives: without sources, the
            # workspace half of the program is what gets read.
            params["sources"] = None

        await _call(tool, params, base_path=tmp_path)

        declared = set(spec_for(tool).parameters["properties"])
        assert declared - params.read == set()

    async def test_a_reference_is_passed_with_the_workspace_it_names_files_in(
        self, service, tmp_path
    ):
        recorder = service("evaluate_restructuring")
        await _call(
            "evaluate_restructuring",
            dict(CARRIED["evaluate_restructuring"][0]),
            base_path=tmp_path,
        )
        assert recorder.calls[0]["reference_root"] == str(tmp_path)

    @pytest.mark.parametrize(
        "tool", ("propose_bolt_configurations", "optimize_executable_with_bolt")
    )
    async def test_a_workspace_program_is_passed_as_root_and_paths(
        self, tool, service, tmp_path
    ):
        recorder = service(tool)
        params = {
            "paths": ["src/a.c", "src/b.c"],
            "include_dirs": ["src/include"],
            "inputs": ["1"],
            "options": "-split-functions",
        }
        await _call(tool, params, base_path=tmp_path)

        call = recorder.calls[0]
        assert call["root"] == str(tmp_path)
        assert call["paths"] == ["src/a.c", "src/b.c"]
        assert call["include_dirs"] == ["src/include"]
        assert "sources" not in call

    async def test_defaults_when_optional_parameters_are_left_out(self, service):
        recorder = service("evaluate_binary_rewrite")
        await _call(
            "evaluate_binary_rewrite",
            {
                "symbol": "work",
                "replacement_asm": ASM,
                "driver": DRIVER,
                "inputs": ["1"],
            },
        )
        call = recorder.calls[0]
        assert call["value_preserving"] is True
        assert call["bench_input"] == 0
        assert call["trials"] == 7
        assert call["flags"] == "-O2"

    async def test_an_empty_libs_means_no_libraries_not_the_default(self, service):
        recorder = service("optimize_executable_with_bolt")
        await _call(
            "optimize_executable_with_bolt",
            {**PROGRAM, "libs": "", "options": "-split-functions"},
        )
        assert recorder.calls[0]["libs"] == ""

    async def test_a_lone_input_string_is_one_input(self, service):
        recorder = service("evaluate_restructuring")
        await _call(
            "evaluate_restructuring",
            {"kernel": KERNEL, "candidate": KERNEL, "driver": DRIVER, "inputs": "7 9"},
        )
        assert recorder.calls[0]["inputs"] == ["7 9"]

    async def test_measure_marginal_reads_configs_and_reps_spelled_as_text(
        self, service
    ):
        recorder = service("measure_marginal")
        await _call(
            "measure_marginal",
            {
                "code": "REPS",
                "configs": '{"base": {}}',
                "reps": "[3, 12]",
            },
        )
        call = recorder.calls[0]
        assert call["configs"] == {"base": {}}
        assert list(call["reps"]) == [3, 12]
        assert call["memory_bound"] is True

    @pytest.mark.parametrize("tool", MODEL_TOOLS)
    async def test_the_model_is_asked_as_the_jobs_owner(self, tool, service):
        recorder = service(tool)
        job = _job()
        # Built the way agent_action_service builds it: a job, no user_id.
        await _call(tool, dict(CARRIED[tool][0]), ctx=_ctx(job, with_user_id=False))
        assert recorder.calls[0]["user_id"] == job.user_id


# --------------------------------------------------------------------------- #
# What a handler refuses before a sandbox or a model is spent.
# --------------------------------------------------------------------------- #

HARNESS_TOOLS = (
    "evaluate_restructuring",
    "propose_binary_rewrites",
    "evaluate_binary_rewrite",
    "synthesize_pass_from_rewrite",
)
BOLT_TOOLS = ("propose_bolt_configurations", "optimize_executable_with_bolt")


@unit
class TestRefusals:
    @pytest.mark.parametrize("tool", HARNESS_TOOLS + BOLT_TOOLS)
    @pytest.mark.parametrize("inputs", [{"a": "1"}, 7])
    async def test_inputs_that_are_not_a_list_are_refused_by_shape(
        self, tool, inputs, service
    ):
        recorder = service(tool)
        params = {**CARRIED[tool][0], "inputs": inputs}

        result = await _call(tool, params)

        assert recorder.calls == []
        assert "inputs" in result["error"]

    @pytest.mark.parametrize("tool", BOLT_TOOLS)
    async def test_a_program_needs_sources_or_a_workspace(self, tool, service):
        recorder = service(tool)
        result = await _call(tool, {"inputs": ["1"], "options": "-split-functions"})
        assert recorder.calls == []
        assert "clone_and_index_repo" in result["error"]

    @pytest.mark.parametrize("tool", BOLT_TOOLS)
    async def test_profile_inputs_must_be_a_list(self, tool, service):
        recorder = service(tool)
        result = await _call(tool, {**PROGRAM, "profile_inputs": "1"})
        assert recorder.calls == []
        assert "profile_inputs" in result["error"]

    @pytest.mark.parametrize("tool", BOLT_TOOLS)
    async def test_workspace_paths_must_be_lists(self, tool, service, tmp_path):
        recorder = service(tool)
        result = await _call(
            tool, {"inputs": ["1"], "paths": "a.c b.c"}, base_path=tmp_path
        )
        assert recorder.calls == []
        assert "lists" in result["error"]

    async def test_a_reference_needs_the_repository(self, service):
        recorder = service("evaluate_restructuring")
        result = await _call(
            "evaluate_restructuring",
            dict(CARRIED["evaluate_restructuring"][0]),
            base_path=None,
        )
        assert recorder.calls == []
        assert "clone_and_index_repo" in result["error"]

    async def test_a_reference_that_is_not_an_object_is_refused(self, service):
        recorder = service("evaluate_restructuring")
        result = await _call(
            "evaluate_restructuring",
            {**CARRIED["evaluate_restructuring"][0], "reference": "src/a.c"},
        )
        assert recorder.calls == []
        assert "reference must be an object" in result["error"]

    async def test_a_policy_blocked_tool_never_reaches_its_handler(self, service):
        recorder = service("build_llvm_pass")
        job = _job()
        job.config = {"blocked_tools": ["build_llvm_pass"]}
        result = await _provider().execute(
            "build_llvm_pass", dict(CARRIED["build_llvm_pass"][0]), _ctx(job)
        )
        assert recorder.calls == []
        assert result["success"] is False

    @pytest.mark.parametrize("tool", TOOLS)
    async def test_execution_disabled_is_said_before_anything_runs(
        self, tool, monkeypatch
    ):
        from app.core.config import settings

        monkeypatch.setattr(settings, "ENABLE_UNSAFE_CODE_EXECUTION", False)
        fake = FakeSandbox(lambda a: AssertionError("the sandbox was called"))
        monkeypatch.setattr(agent_sandbox_runtime, "run_in_sandbox", fake)
        model = FakeModel([{}]).install(monkeypatch)
        params = dict(CARRIED[tool][0])
        params.pop("object_b64", None)
        params.pop("reference", None)
        params.pop("baseline_asm", None)
        if tool in BOLT_TOOLS:
            params.pop("profile_inputs")

        result = await _call(tool, params)

        assert fake.calls == []
        assert model.calls == []
        assert result.get("success") is not True
        assert "disabled" in result["error"].lower()
        assert not result.get("findings")


# --------------------------------------------------------------------------- #
# A sandbox that could not run, through the real services.
# --------------------------------------------------------------------------- #

#: Calls that pass every check the handler and service make before Docker.
REACHES_THE_SANDBOX = {
    "build_llvm_pass": {
        "source": "struct P {};",
        "pass_name": "mark-cold",
        "test_code": "int f(int x) { return x + 1; }",
    },
    "disassemble_symbol": {"symbol": "work", "kernel": KERNEL},
    "evaluate_restructuring": {
        "kernel": KERNEL,
        "candidate": KERNEL,
        "driver": DRIVER,
        "inputs": ["1", "100"],
    },
    "propose_binary_rewrites": {
        "symbol": "work",
        "kernel": KERNEL,
        "driver": DRIVER,
        "inputs": ["1", "100"],
        "count": 1,
    },
    "evaluate_binary_rewrite": {
        "symbol": "work",
        "replacement_asm": ASM,
        "kernel": KERNEL,
        "driver": DRIVER,
        "inputs": ["1", "100"],
    },
    "synthesize_pass_from_rewrite": {
        "kernel": KERNEL,
        "rewrite_kernel": KERNEL,
        "driver": DRIVER,
        "inputs": ["1", "100"],
    },
    "propose_bolt_configurations": {
        "sources": {"prog.c": "int main(void) { return 0; }"},
        "inputs": ["1", "2"],
        "count": 1,
    },
    "optimize_executable_with_bolt": {
        "sources": {"prog.c": "int main(void) { return 0; }"},
        "inputs": ["1", "2"],
        "options": "-reorder-blocks=ext-tsp",
    },
    "measure_marginal": {
        "code": "int main(void){for(int r=0;r<REPS;r++);return 0;}",
        "configs": {"base": {}},
        "memory_bound": False,
    },
}

PASS_REPLY = {
    "expressible": True,
    "reason": "the IR shows it",
    "pass_name": "mod-seven",
    "pass_source": "struct P {};",
    "must_decline": "",
    "value_preserving": True,
}
BINARY_REPLY = {
    "proposals": [
        {"name": "zero", "idea": "i", "invariant": "v", "assembly": ASM},
    ]
}
BOLT_REPLY = {
    "proposals": [
        {"name": "tsp", "idea": "i", "invariant": "v", "options": "-split-functions"}
    ]
}
MODEL_REPLIES = {
    "synthesize_pass_from_rewrite": PASS_REPLY,
    "propose_binary_rewrites": BINARY_REPLY,
    "propose_bolt_configurations": BOLT_REPLY,
}


@unit
class TestASandboxThatCouldNotRun:
    @pytest.mark.parametrize("tool", TOOLS)
    async def test_an_image_docker_cannot_start_is_named_as_such(
        self, tool, sandbox, monkeypatch
    ):
        fake = sandbox(lambda a: (125, "", NO_SUCH_IMAGE.format(image=a["image"])))
        FakeModel([MODEL_REPLIES.get(tool, {})]).install(monkeypatch)

        result = await _call(tool, dict(REACHES_THE_SANDBOX[tool]))

        assert fake.calls, "the call never reached the sandbox"
        assert result.get("success") is not True
        assert "could not start" in str(result.get("error") or "").lower()
        assert not result.get("findings")

    @pytest.mark.parametrize("tool", TOOLS)
    async def test_no_docker_on_this_host_is_said_in_one_failure(
        self, tool, sandbox, monkeypatch
    ):
        sandbox(lambda a: FileNotFoundError("docker"))
        FakeModel([MODEL_REPLIES.get(tool, {})]).install(monkeypatch)

        result = await _call(tool, dict(REACHES_THE_SANDBOX[tool]))

        assert result.get("success") is not True
        assert "docker" in str(result.get("error") or "").lower()
        assert not result.get("findings")


# --------------------------------------------------------------------------- #
# Tool-specific failure reporting, through the real services.
# --------------------------------------------------------------------------- #


def _nm_listing(arguments):
    """What disassemble_symbol's script prints for an object defining `work`."""
    return (
        0,
        "__nm__\n0000000000000000 T work\n__nm_end__\n__asm__\n"
        "Disassembly of section .text:\n\n0000000000000000 <work>:\n"
        "       0: mov x0, xzr\n       4: ret\n__asm_end__\n__calls__ 0\n",
        "",
    )


def _bolt_profiled(arguments):
    """Stage one of the BOLT tools: a built binary and a profile."""
    workdir = Path(arguments["workdir"])
    if "-instrument" in arguments["script"]:
        (workdir / "prog").write_bytes(b"\x7fELF" + b"\0" * 60)
        (workdir / "prof.fdata").write_text("1 main 0 1 main 10 0 5\n")
        return (0, "__std_rc__ 0\n__size__ 100 120\n__stdlog__\n__stdlog_end__\n", "")
    return AssertionError("only the build-and-profile stage was expected")


@unit
class TestToolSpecificReporting:
    async def test_a_pass_that_crashes_is_not_called_unregistered(self, sandbox):
        # The stdout shape the real image printed for a pass whose run()
        # calls report_fatal_error("boom from the pass").
        sandbox(
            lambda a: (
                0,
                "PB\tcompiled\t1\nPB\topt_rc\t134\nPB\tpass_out\n"
                "LLVM ERROR: boom from the pass\n"
                "PLEASE submit a bug report to https://github.com/llvm/"
                "llvm-project/issues/ and include the crash backtrace.\n"
                "Stack dump:\n0.\tProgram arguments: opt -load-pass-plugin=./pass.so "
                "-passes=mark-cold -S -o after.ll baseline.ll\nPB\tbefore\n"
                "define i32 @f(i32 %0) {\n  ret i32 %0\n}\nPB\tafter\n",
                "",
            )
        )
        result = await _call(
            "build_llvm_pass", dict(REACHES_THE_SANDBOX["build_llvm_pass"])
        )
        assert result["success"] is False
        assert result.get("registered") is not False
        assert "did not accept that name" not in result["error"]

    async def test_a_pass_that_does_not_compile_returns_the_compilers_words(
        self, sandbox
    ):
        sandbox(
            lambda a: (
                0,
                "PB\tcompiled\t0\nPB\tbuild_err\npass.cpp:1:1: error: unknown type "
                "name 'struc'\n",
                "",
            )
        )
        result = await _call(
            "build_llvm_pass", dict(REACHES_THE_SANDBOX["build_llvm_pass"])
        )
        assert result["success"] is False
        assert result["compiled"] is False
        assert "unknown type name 'struc'" in result["build_errors"]
        assert not result.get("findings")

    async def test_the_pass_tool_runs_in_the_image_that_has_the_headers(self, sandbox):
        fake = sandbox(lambda a: (0, "PB\tcompiled\t0\nPB\tbuild_err\nx\n", ""))
        await _call("build_llvm_pass", dict(REACHES_THE_SANDBOX["build_llvm_pass"]))
        assert fake.calls[0]["image"] == PASS_IMAGE
        assert fake.calls[0]["timeout_seconds"] > 0

    async def test_a_binary_rewrite_whose_judging_could_not_run_is_not_a_success(
        self, sandbox, monkeypatch
    ):
        def respond(arguments):
            if "__nm__" in arguments["script"]:
                return _nm_listing(arguments)
            return asyncio.TimeoutError()

        sandbox(respond)
        FakeModel([BINARY_REPLY]).install(monkeypatch)

        result = await _call(
            "propose_binary_rewrites",
            {
                "symbol": "work",
                "object_b64": OBJECT_B64,
                "driver": DRIVER,
                "inputs": ["1"],
                "count": 1,
            },
        )
        assert result.get("success") is not True
        assert "timed out" in str(result.get("error") or "")

    async def test_the_binary_proposer_reads_only_the_disassembly(
        self, sandbox, monkeypatch
    ):
        def respond(arguments):
            if "__nm__" in arguments["script"]:
                return _nm_listing(arguments)
            return asyncio.TimeoutError()

        sandbox(respond)
        model = FakeModel([BINARY_REPLY]).install(monkeypatch)

        await _call(
            "propose_binary_rewrites",
            {
                "symbol": "work",
                "object_b64": OBJECT_B64,
                "driver": DRIVER,
                "inputs": ["1"],
                "count": 1,
            },
        )
        assert len(model.calls) == 1
        message = model.calls[0]["user_message"]
        assert "mov x0, xzr" in message
        assert "s += i % 7" not in message

    async def test_a_disallowed_bolt_option_is_refused_before_building(self, sandbox):
        fake = sandbox(_bolt_profiled)
        result = await _call(
            "optimize_executable_with_bolt",
            {
                **REACHES_THE_SANDBOX["optimize_executable_with_bolt"],
                "options": "-instrumentation-file=/work/x",
            },
        )
        assert "-instrumentation-file" in str(result.get("error") or "")
        assert fake.calls == []

    async def test_a_disallowed_bolt_option_is_not_reported_as_success(self, sandbox):
        sandbox(_bolt_profiled)
        result = await _call(
            "optimize_executable_with_bolt",
            {
                **REACHES_THE_SANDBOX["optimize_executable_with_bolt"],
                "options": "-instrumentation-file=/work/x",
            },
        )
        assert result.get("success") is not True

    async def test_a_bolt_build_failure_is_the_compilers_words(self, sandbox):
        sandbox(
            lambda a: (
                0,
                "__build_failed__\nprog.c:1:5: error: expected ';'\n",
                "",
            )
        )
        result = await _call(
            "optimize_executable_with_bolt",
            dict(REACHES_THE_SANDBOX["optimize_executable_with_bolt"]),
        )
        assert result["success"] is False
        assert "did not build" in result["error"]
        assert "expected ';'" in result["error"]

    async def test_a_missing_kernel_is_refused_before_the_model_is_asked(
        self, sandbox, monkeypatch
    ):
        sandbox(lambda a: (0, "", ""))
        model = FakeModel([PASS_REPLY]).install(monkeypatch)

        result = await _call(
            "synthesize_pass_from_rewrite",
            {**REACHES_THE_SANDBOX["synthesize_pass_from_rewrite"], "kernel": ""},
        )
        assert "kernel" in result["error"]
        assert model.calls == []

    async def test_not_expressible_is_a_pass_evaluation_finding(
        self, sandbox, monkeypatch
    ):
        sandbox(lambda a: (0, "define i64 @work(i64 %0) {\n}\n", ""))
        FakeModel(
            [{"expressible": False, "reason": "relies on the inputs being bytes"}]
        ).install(monkeypatch)

        result = await _call(
            "synthesize_pass_from_rewrite",
            dict(REACHES_THE_SANDBOX["synthesize_pass_from_rewrite"]),
        )
        produces = spec_for("synthesize_pass_from_rewrite").produces
        assert result["success"] is True
        assert result["data"]["verdict"] == "not_expressible"
        assert {f["type"] for f in result["findings"]} <= set(produces)
        assert result["findings"]

    async def test_a_simulation_has_a_time_limit(self, sandbox, monkeypatch):
        async def usable(image, cpu_type):
            return {"usable": True, "probed": True}

        monkeypatch.setattr(agent_gem5_sandbox, "model_support", usable)
        fake = sandbox(lambda a: (91, "", "ARM_FAILED base\nfatal: stop here\n"))

        await _call("measure_marginal", dict(REACHES_THE_SANDBOX["measure_marginal"]))

        timeout = fake.calls[0]["timeout_seconds"]
        assert isinstance(timeout, int) and timeout > 0

    async def test_a_marginal_measurement_produces_its_declared_evidence(
        self, sandbox, monkeypatch
    ):
        async def usable(image, cpu_type):
            return {"usable": True, "probed": True}

        monkeypatch.setattr(agent_gem5_sandbox, "model_support", usable)

        def respond(arguments):
            workdir = Path(arguments["workdir"])
            reps = int(
                re.search(r"/\*R\*/(\d+)", (workdir / "workload.c").read_text())[1]
            )
            for arm in ("base", "stride"):
                (workdir / arm).mkdir()
                (workdir / arm / "stats.txt").write_text(
                    f"system.cpu.numCycles {1000 + 500 * reps}\n"
                    f"system.l2cache.overallMisses::total {10 + 5000 * reps}\n"
                )
                (workdir / f"{arm}.manifest.json").write_text("{}")
            return (0, "OK\n", "")

        sandbox(respond)
        result = await _call(
            "measure_marginal",
            {
                "code": "int main(void){for(int r=0;r</*R*/REPS;r++);return 0;}",
                "configs": json.dumps(
                    {
                        "base": {},
                        "stride": {
                            "caches": {"l2": {"prefetcher": "StridePrefetcher"}}
                        },
                    }
                ),
                "reps": [2, 6],
            },
        )

        assert result["success"] is True
        assert result["per_config"]["base"]["cycles_per_repetition"] == 500.0
        assert result["per_config"]["base"]["fixed_cost_cycles"] == 1000.0
        produces = spec_for("measure_marginal").produces
        assert [f["type"] for f in result["findings"]] == list(produces)


# --------------------------------------------------------------------------- #
# Live: the real handler against the real sandbox.
# --------------------------------------------------------------------------- #


def _image_available(image):
    if not shutil.which("docker"):
        return False
    try:
        done = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def _needs(image):
    return pytest.mark.skipif(
        not _image_available(image), reason=f"Docker or {image} is not available"
    )


def _assert_produces(tool, result):
    produces = set(spec_for(tool).produces)
    types = {f.get("type") for f in result.get("findings") or []}
    assert types and types <= produces, (types, produces)


#: Verdicts that would mean the harness, not the candidate, is wrong.
BROKEN = ("baseline_broken", "did_not_compile", "diverged", "crashed")

MARK_COLD_PASS = r"""
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
using namespace llvm;
namespace {
struct MarkCold : PassInfoMixin<MarkCold> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    if (F.isDeclaration() || F.hasFnAttribute(Attribute::Cold))
      return PreservedAnalyses::all();
    F.addFnAttr(Attribute::Cold);
    return PreservedAnalyses::none();
  }
};
}
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "mark-cold", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "mark-cold") {
                    FPM.addPass(MarkCold());
                    return true;
                  }
                  return false;
                });
          }};
}
"""

ADD3 = "long add3(long x) { return x + 3; }\n"
ADD3_DRIVER = (
    "#include <stdio.h>\nlong add3(long x);\n"
    'int main(void) { long n, s = 0; if (scanf("%ld", &n) != 1) return 1; '
    "for (long i = 0; i < n; i++) s += add3(i); "
    'printf("%ld\\n", s); return 0; }\n'
)
ADD3_ASM = (
    ".text\n.globl add3\n.p2align 2\n.type add3, %function\n"
    "add3:\n  add x0, x0, #3\n  ret\n.size add3, .-add3\n"
)
BOLT_PROGRAM = (
    "#include <stdio.h>\n"
    "static int classify(long x) { if (x % 3 == 0) return 1; "
    "if (x % 5 == 0) return 2; return 0; }\n"
    'int main(void) { long n, c[3] = {0}; if (scanf("%ld", &n) != 1) return 1; '
    "for (long i = 0; i < n; i++) c[classify(i)]++; "
    'printf("%ld %ld %ld\\n", c[0], c[1], c[2]); return 0; }\n'
)


async def _live(tool, params, seconds=240):
    # A bound of our own: on expiry the cancellation reaches run_in_sandbox,
    # which removes the container by name.
    return await asyncio.wait_for(_call(tool, params), seconds)


@pytest.mark.slow
class TestLive:
    @_needs(PASS_IMAGE)
    async def test_build_llvm_pass_builds_registers_and_fires(self, sandbox_enabled):
        result = await _live(
            "build_llvm_pass",
            {
                "source": MARK_COLD_PASS,
                "pass_name": "mark-cold",
                "test_code": "int f(int x) { return x + 1; }",
            },
        )
        assert result["success"] is True, result.get("error")
        assert result["registered"] is True
        assert result["fired"] is True
        _assert_produces("build_llvm_pass", result)

    @_needs(COMPILER_IMAGE)
    async def test_evaluate_restructuring_judges_an_identical_rewrite(
        self, sandbox_enabled
    ):
        result = await _live(
            "evaluate_restructuring",
            {
                "kernel": KERNEL,
                "candidate": KERNEL.replace("s += i % 7", "s = s + i % 7"),
                "driver": DRIVER,
                "inputs": ["1", "1000", "3000000"],
                "bench_input": 2,
                "trials": 6,
                "label": "identical",
            },
        )
        assert result["success"] is True, result.get("error")
        assert result["data"]["verdict"] not in BROKEN
        assert result["data"]["equivalence"]["status"] == "equivalent"
        _assert_produces("evaluate_restructuring", result)

    @_needs(COMPILER_IMAGE)
    async def test_disassemble_then_evaluate_a_binary_rewrite(self, sandbox_enabled):
        listing = await _live(
            "disassemble_symbol", {"symbol": "add3", "kernel": ADD3}, seconds=120
        )
        assert listing["success"] is True, listing.get("error")
        assert "add3" in listing["data"]["listing"]
        assert listing["data"]["instructions"] >= 1

        result = await _live(
            "evaluate_binary_rewrite",
            {
                "symbol": "add3",
                "replacement_asm": ADD3_ASM,
                "kernel": ADD3,
                "driver": ADD3_DRIVER,
                "inputs": ["0", "5", "20000000"],
                "bench_input": 2,
                "trials": 6,
            },
        )
        assert result["success"] is True, result.get("error")
        assert result["data"]["verdict"] not in BROKEN
        _assert_produces("evaluate_binary_rewrite", result)

    @_needs(BOLT_IMAGE)
    async def test_optimize_executable_with_bolt_judges_a_configuration(
        self, sandbox_enabled
    ):
        result = await _live(
            "optimize_executable_with_bolt",
            {
                "sources": {"prog.c": BOLT_PROGRAM},
                "inputs": ["3000000", "2000000", "1000000"],
                "options": "-reorder-blocks=ext-tsp",
                "trials": 6,
            },
        )
        assert result["success"] is True, result.get("error")
        assert result["data"]["verdict"] not in BROKEN
        assert result["data"]["hot_functions"]
        _assert_produces("optimize_executable_with_bolt", result)

    @_needs(GEM5_IMAGE)
    async def test_measure_marginal_cancels_setup_in_simulation(self, sandbox_enabled):
        result = await _live(
            "measure_marginal",
            {
                "code": (
                    "int main(void) { volatile long s = 0; "
                    "for (int r = 0; r < REPS; r++) "
                    "for (int i = 0; i < 2000; i++) s += i; return (int)(s & 1); }"
                ),
                "configs": {"base": {}},
                "reps": [2, 6],
                "memory_bound": False,
                "label": "adder",
            },
        )
        assert result["success"] is True, result.get("error")
        assert result["per_config"]["base"]["cycles_per_repetition"] > 0
        _assert_produces("measure_marginal", result)
