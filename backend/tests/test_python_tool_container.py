"""A custom Python tool's code runs in the sandbox container, in the caller's
working directory.

It used to run in the server's own process under RestrictedPython, which was
escaped once and could not stop a loop. `python_tool_container` hands it to
`agent_sandbox_runtime.run_in_sandbox` instead.

Most tests replace that one call with `_LocalSandbox`, which does what the
container does to the files -- runs the same command line with the working
directory standing in for `/work` -- so the runner script, the file protocol
and the error paths are all real. It is bound against the real signature.
The last test runs a real container and is skipped where there is none.
"""

import asyncio
import inspect
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.core.config import settings
from app.services import (
    agent_sandbox_runtime,
    python_tool_container,
    sandbox_skill_runtime,
)
from app.services.custom_tool_service import CustomToolService, ToolExecutionError

pytestmark = pytest.mark.unit

IMAGE = "ghcr.io/al3x3n0/kdbc-compiler-research:latest"
_REAL_RUN = agent_sandbox_runtime.run_in_sandbox
_RUN_SIGNATURE = inspect.signature(_REAL_RUN)


class _LocalSandbox:
    """`run_in_sandbox`, minus the container: same command, same files."""

    def __init__(self):
        self.calls = []
        self.returncode = None
        self.stderr = ""

    async def __call__(self, *args, **kwargs):
        bound = _RUN_SIGNATURE.bind(*args, **kwargs)
        bound.apply_defaults()
        call = dict(bound.arguments)
        self.calls.append(call)
        if self.returncode is not None:
            return self.returncode, "", self.stderr
        workdir = call["workdir"]
        command = call["script"].replace("/work/", f"{workdir}/").split()
        assert command[0] == "python3"
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            *command[1:],
            cwd=workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(
                process.communicate(), timeout=call["timeout_seconds"]
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise
        return process.returncode, out.decode(), err.decode()


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    fake = _LocalSandbox()
    monkeypatch.setattr(agent_sandbox_runtime, "run_in_sandbox", fake)
    monkeypatch.setattr(settings, "ENABLE_UNSAFE_CODE_EXECUTION", True)
    monkeypatch.setattr(settings, "CUSTOM_TOOL_PYTHON_IMAGE", IMAGE)
    monkeypatch.setattr(sandbox_skill_runtime, "root_dir", lambda: tmp_path / "skills")
    fake.root = tmp_path / "skills"
    return fake


def _user():
    return SimpleNamespace(id=uuid4())


async def _run(code, inputs=None, user=None, workdir=None, **config):
    return await CustomToolService()._execute_python(
        {"code": code, **config}, inputs or {}, user, workdir=workdir
    )


# --------------------------------------------------------------------------
# the contract with the code
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code, inputs, expected",
    [
        ("output = {'value': 6 * 7}", {}, {"value": 42}),
        ("result = input['a'] + 1", {"a": 1}, {"result": 2}),
        ("output['v'] = input['a']", {"a": 5}, {"v": 5}),
        (
            "t = 0\nfor x in input['items']:\n    t += x\nresult = t",
            {"items": [1, 2]},
            {"result": 3},
        ),
        (
            "def double(x):\n    return x * 2\noutput = {'r': [double(x) for x in input['items']]}",
            {"items": [1, 2]},
            {"r": [2, 4]},
        ),
        # Ordinary Python: the container is the boundary, not a list of names.
        (
            "import statistics, collections\nresult = statistics.mean([1, 2, 3])",
            {},
            {"result": 2},
        ),
        (
            "import datetime\nresult = datetime.date(2026, 10, 13)",
            {},
            {"result": "2026-10-13"},
        ),
        ("print('working')\nresult = 'done'", {}, {"result": "done"}),
    ],
)
async def test_code_gets_its_input_and_answers(sandbox, code, inputs, expected):
    assert await _run(code, inputs) == expected


async def test_it_runs_in_the_configured_image_with_the_asked_timeout(sandbox):
    await _run("result = 1", timeout_seconds=7)

    (call,) = sandbox.calls
    assert call["image"] == IMAGE
    assert call["timeout_seconds"] == 7
    assert call["script"].startswith("python3 /work/.pytool/")


@pytest.mark.parametrize(
    "asked, used", [(None, 30), (0, 1), (-5, 1), (10**6, 300), ("x", 30)]
)
async def test_the_timeout_is_bounded(sandbox, asked, used):
    await _run("result = 1", timeout_seconds=asked)

    assert sandbox.calls[0]["timeout_seconds"] == used


# --------------------------------------------------------------------------
# the working directory
# --------------------------------------------------------------------------


async def test_files_persist_in_the_callers_directory(sandbox):
    user = _user()

    await _run("open('notes.txt', 'w').write('built')\nresult = 'ok'", user=user)
    second = await _run("result = open('notes.txt').read()", user=user)

    assert second == {"result": "built"}
    # The same directory a sandbox skill uses for this user.
    assert (sandbox.root / f"user-{user.id}" / "notes.txt").read_text() == "built"


async def test_a_file_a_skill_left_is_there_to_read(sandbox):
    workdir = sandbox_skill_runtime.run_dir("job-42")
    (workdir / "kernel.sym").write_text("main\nwork\n")

    result = await _run("result = open('kernel.sym').read().split()", workdir=workdir)

    assert result == {"result": ["main", "work"]}


async def test_two_users_do_not_share_a_directory(sandbox):
    await _run("open('mine.txt', 'w').write('x')", user=_user())

    with pytest.raises(ToolExecutionError, match="FileNotFoundError"):
        await _run("result = open('mine.txt').read()", user=_user())


async def test_a_call_leaves_none_of_its_own_files_behind(sandbox):
    user = _user()
    await _run("result = 1", user=user)
    with pytest.raises(ToolExecutionError):
        await _run("raise ValueError('no')", user=user)

    assert not (sandbox.root / f"user-{user.id}" / ".pytool").exists()


async def test_a_call_with_no_caller_gets_a_directory_that_is_removed(sandbox):
    assert await _run("open('scratch', 'w').write('x')\nresult = 1") == {"result": 1}

    assert not any(sandbox.root.iterdir())


# --------------------------------------------------------------------------
# when it does not work, it says which thing failed
# --------------------------------------------------------------------------


async def test_a_syntax_error_is_a_compile_error(sandbox):
    with pytest.raises(ToolExecutionError) as failed:
        await _run("def broken(:")

    assert "compilation errors" in str(failed.value).lower()
    assert "line 1" in str(failed.value)


async def test_a_failure_names_the_line_and_what_was_printed(sandbox):
    with pytest.raises(ToolExecutionError) as failed:
        await _run("print('step one')\nx = input['missing']")

    message = str(failed.value)
    assert "KeyError: 'missing'" in message
    assert "line 2" in message
    assert "step one" in message


async def test_a_loop_that_never_ends_is_stopped(sandbox):
    with pytest.raises(ToolExecutionError, match="timed out after 1s"):
        await _run("while True:\n    pass", timeout_seconds=1)


async def test_code_that_exits_without_answering_says_so(sandbox):
    with pytest.raises(ToolExecutionError, match="without an answer"):
        await _run("import os\nos._exit(3)")


async def test_an_answer_too_large_to_return_is_refused(sandbox, monkeypatch):
    monkeypatch.setattr(python_tool_container, "MAX_OUTPUT_BYTES", 100)

    with pytest.raises(ToolExecutionError, match="larger than"):
        await _run("result = 'x' * 1000")


async def test_a_container_that_never_started_is_not_the_codes_fault(sandbox):
    sandbox.returncode = agent_sandbox_runtime.DOCKER_COULD_NOT_START
    sandbox.stderr = "Unable to find image locally"

    with pytest.raises(ToolExecutionError) as failed:
        await _run("result = 1")

    assert "KeyError" not in str(failed.value)
    assert IMAGE in str(failed.value)


async def test_it_does_not_run_where_sandboxed_execution_is_off(sandbox, monkeypatch):
    monkeypatch.setattr(settings, "ENABLE_UNSAFE_CODE_EXECUTION", False)

    with pytest.raises(ToolExecutionError, match="ENABLE_UNSAFE_CODE_EXECUTION"):
        await _run("result = 1")

    assert sandbox.calls == []


async def test_it_does_not_run_in_an_image_that_is_not_allowlisted(
    sandbox, monkeypatch
):
    monkeypatch.setattr(
        settings, "CUSTOM_TOOL_PYTHON_IMAGE", "example.org/not-listed:1"
    )

    with pytest.raises(ToolExecutionError, match="CUSTOM_TOOL_PYTHON_IMAGE"):
        await _run("result = 1")

    assert sandbox.calls == []


# --------------------------------------------------------------------------
# nothing is left running in this process
# --------------------------------------------------------------------------


def test_no_tool_code_is_executed_in_process():
    """The in-process path is gone, not kept as a fallback: a fallback is the
    path that runs when the safe one is unavailable."""
    backend = Path(__file__).resolve().parents[1]
    service = (backend / "app/services/custom_tool_service.py").read_text()
    assert "exec(" not in service
    assert "RestrictedPython" not in service
    assert "RestrictedPython" not in (backend / "requirements.txt").read_text()
    # The only `exec` is in the script that runs inside the container.
    container = (backend / "app/services/python_tool_container.py").read_text()
    assert container.count("exec(compiled, scope)") == 1


def test_the_default_image_is_one_the_sandbox_allows():
    from app.core.config import Settings

    fields = Settings.model_fields
    default_image = fields["CUSTOM_TOOL_PYTHON_IMAGE"].default
    allowed = fields["SCIENTIFIC_VALIDATION_ALLOWED_DOCKER_IMAGES"].default
    assert default_image in [item.strip() for item in allowed.split(",")]


# --------------------------------------------------------------------------
# a real container
# --------------------------------------------------------------------------


def _image_available(image):
    if not shutil.which("docker"):
        return False
    try:
        done = subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True, timeout=20
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


@pytest.mark.skipif(
    not _image_available(IMAGE), reason=f"Docker or {IMAGE} is not available"
)
async def test_in_a_real_container_the_code_is_confined(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "ENABLE_UNSAFE_CODE_EXECUTION", True)
    monkeypatch.setattr(settings, "CUSTOM_TOOL_PYTHON_IMAGE", IMAGE)
    # Somewhere the daemon can mount: the default temp directory.
    workdir = sandbox_skill_runtime.run_dir(f"pytest-{uuid4().hex[:10]}")
    try:
        result = await _run(
            "import os, socket\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
            "    reached = True\n"
            "except OSError:\n"
            "    reached = False\n"
            "open('made-here.txt', 'w').write('x')\n"
            "output = {'uid': os.getuid(), 'network': reached, 'cwd': os.getcwd()}",
            workdir=workdir,
        )

        assert result == {"uid": 65534, "network": False, "cwd": "/work"}
        assert (workdir / "made-here.txt").read_text() == "x"
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
