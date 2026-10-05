"""The code execution tools: execute_python, execute_data_pipeline, write_and_run_script.

These call the real handlers with the executor underneath replaced. The file
used to restate each handler inline and assert on the restatement, so all
fourteen tests passed while `write_and_run_script` could not run a script at
all: it handed the script to the container as stdin and then ran
`python /workspace/<name>`, a file nothing had written.
"""

import json
import os
from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.schemas.docker_tool import DockerToolConfig, DockerToolExecutionInput
from app.services import custom_tool_service
from app.services import docker_tool_executor as executor_module
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_workspace_mutation_provider,
)

pytestmark = pytest.mark.unit


class _Recorder:
    """Stands in for CustomToolService: records what it was asked to run."""

    calls = []
    fail_with = None

    async def _execute_python(self, config, inputs, user):
        return self._record("python", config, inputs)

    async def _execute_docker(self, config, inputs, user):
        return self._record("docker", config, inputs)

    def _record(self, kind, config, inputs):
        if _Recorder.fail_with:
            raise RuntimeError(_Recorder.fail_with)
        _Recorder.calls.append((kind, config, inputs))
        return {"stdout": "ok"}


@pytest.fixture
def run(monkeypatch):
    _Recorder.calls, _Recorder.fail_with = [], None
    monkeypatch.setattr(custom_tool_service, "CustomToolService", _Recorder)

    class _Db:
        async def execute(self, *_a, **_k):
            return SimpleNamespace(scalar_one_or_none=lambda: SimpleNamespace(id="u"))

    provider = build_autonomous_workspace_mutation_provider(SimpleNamespace())

    async def _run(tool, params, state=None):
        state = {} if state is None else state
        ctx = AgentToolExecutionContext(
            mode="autonomous",
            db=_Db(),
            service=None,
            user_id="u",
            job=SimpleNamespace(id="job-1", user_id="u", config={}),
            state=state,
        )
        return await provider._handlers[tool](params, ctx), state

    return _run


@pytest.fixture
def docker_enabled(monkeypatch):
    monkeypatch.setattr(settings, "CUSTOM_TOOL_DOCKER_ENABLED", True)


class TestExecutePython:
    async def test_code_is_required(self, run):
        result, _ = await run("execute_python", {"code": "   "})
        assert result == {"error": "No code provided"}
        assert _Recorder.calls == []

    async def test_it_runs_the_code_and_records_that_it_did(self, run):
        result, state = await run("execute_python", {"code": "x = 1"})

        assert result == {"success": True, "data": {"stdout": "ok"}}
        assert _Recorder.calls == [
            ("python", {"code": "x = 1", "timeout_seconds": 10}, {})
        ]
        entry = state["code_execution_history"][0]
        assert entry["tool"] == "execute_python" and entry["code_preview"] == "x = 1"

    async def test_the_timeout_is_capped(self, run):
        await run("execute_python", {"code": "x", "timeout_seconds": 9999})
        assert _Recorder.calls[0][1]["timeout_seconds"] == 30

    async def test_a_failed_run_is_an_error_and_is_not_recorded_as_a_success(self, run):
        _Recorder.fail_with = "import of 'subprocess' is not allowed"
        result, state = await run("execute_python", {"code": "import subprocess"})

        assert "not allowed" in result["error"]
        assert "code_execution_history" not in state

    async def test_history_keeps_the_last_fifty(self, run):
        state = {"code_execution_history": [{"tool": "old"}] * 60}
        _, state = await run("execute_python", {"code": "x"}, state)

        assert len(state["code_execution_history"]) == 50
        assert state["code_execution_history"][-1]["tool"] == "execute_python"


class TestExecuteDataPipeline:
    async def test_without_docker_it_runs_restricted_with_the_input_bound(self, run):
        await run(
            "execute_data_pipeline",
            {"code": "result = input_data['n'] + 1", "input_data": {"n": 1}},
        )
        kind, config, _ = _Recorder.calls[0]

        assert kind == "python"
        assert config["code"] == "input_data = {'n': 1}\nresult = input_data['n'] + 1"

    async def test_with_docker_the_input_is_stdin_and_there_is_no_network(
        self, run, docker_enabled
    ):
        await run(
            "execute_data_pipeline",
            {
                "code": "result = 1",
                "input_data": {"who": "O'Brien"},
                "timeout_seconds": 999,
            },
        )
        kind, config, inputs = _Recorder.calls[0]

        assert kind == "docker"
        assert config["network_enabled"] is False
        assert config["timeout_seconds"] == 300
        assert (
            config["command"][:2] == ["python", "-c"]
            and "result = 1" in config["command"][2]
        )
        assert json.loads(inputs["stdin"]) == {"who": "O'Brien"}

    async def test_input_that_is_not_an_object_is_no_input(self, run):
        await run("execute_data_pipeline", {"code": "x", "input_data": "a string"})
        assert _Recorder.calls[0][1]["code"].startswith("input_data = {}\n")


class TestWriteAndRunScript:
    async def test_it_needs_docker(self, run):
        result, _ = await run("write_and_run_script", {"script_content": "print(1)"})
        assert "requires Docker" in result["error"]
        assert _Recorder.calls == []

    async def test_a_script_is_required(self, run, docker_enabled):
        result, _ = await run("write_and_run_script", {"script_content": " "})
        assert result == {"error": "No script content provided"}

    async def test_the_script_is_delivered_as_the_file_that_is_run(
        self, run, docker_enabled
    ):
        result, state = await run(
            "write_and_run_script",
            {
                "script_name": "analysis.py",
                "script_content": "print(1)",
                "arguments": ["--n", 3],
            },
        )
        kind, config, inputs = _Recorder.calls[0]

        assert result["success"] is True and kind == "docker"
        # The file the executor writes is the file python is given.
        assert config["input_mode"] == "both"
        assert config["input_file_path"] == "/workspace/analysis.py"
        assert inputs["input_file_content"] == "print(1)"
        assert config["command"][3:] == ["/workspace/analysis.py", "--n", "3"]
        assert config["network_enabled"] is False
        assert state["code_execution_history"][0]["script_name"] == "analysis.py"

    async def test_data_never_reaches_the_shell_line(self, run, docker_enabled):
        """An apostrophe in the input used to close the quoted string the
        JSON had been pasted into; an argument could add commands."""
        await run(
            "write_and_run_script",
            {
                "script_content": "print(1)",
                "input_data": {"who": "O'Brien; rm -rf /"},
                "arguments": ["$(id)", "; echo pwned"],
            },
        )
        _, config, inputs = _Recorder.calls[0]
        shell_line = config["command"][2]

        assert "O'Brien" not in shell_line and "pwned" not in shell_line
        assert shell_line == 'cat > /workspace/input.json; exec python "$0" "$@"'
        assert json.loads(inputs["stdin"]) == {"who": "O'Brien; rm -rf /"}
        assert config["command"][4:] == ["$(id)", "; echo pwned"]

    @pytest.mark.parametrize(
        "name", ["../../etc/cron.d/x", "a b.py", "x;y.py", "", "/abs/path.py"]
    )
    async def test_the_script_name_is_a_bare_file_name(self, run, docker_enabled, name):
        await run("write_and_run_script", {"script_name": name, "script_content": "x"})
        path = _Recorder.calls[0][1]["input_file_path"]

        assert os.path.dirname(path) == "/workspace"
        assert ".." not in path and " " not in path and ";" not in path

    async def test_requirements_are_refused_rather_than_silently_failing(
        self, run, docker_enabled
    ):
        """There is no network in the container. `pip install` was chained in
        front of the script with `&&`, so asking for pandas guaranteed the
        script never ran."""
        result, _ = await run(
            "write_and_run_script",
            {"script_content": "import pandas", "requirements": ["pandas"]},
        )
        assert "no network" in result["error"]
        assert _Recorder.calls == []

    async def test_at_most_ten_arguments(self, run, docker_enabled):
        await run(
            "write_and_run_script",
            {"script_content": "x", "arguments": [str(i) for i in range(20)]},
        )
        assert len(_Recorder.calls[0][1]["command"][4:]) == 10


class TestTheDockerExecutorDeliversItsInputs:
    """The other half: what the executor does with what the tools hand it."""

    def _command(self, **kwargs):
        return executor_module.DockerToolExecutor()._build_docker_command(
            config=DockerToolConfig(image="img", command=["true"]),
            workspace_dir="/tmp/w",
            container_name="c",
            **kwargs,
        )

    def test_stdin_is_attached_only_when_there_is_some(self):
        """Without -i the container's stdin is not connected: data written to
        the `docker run` client goes nowhere."""
        assert "-i" in self._command(attach_stdin=True)
        assert "-i" not in self._command()
        attached = self._command(attach_stdin=True)
        assert attached.index("-i") < attached.index("img")

    async def test_the_input_file_has_the_name_the_tool_configured(self, monkeypatch):
        seen = {}

        class _Process:
            returncode = 0

            async def communicate(self, input=None):
                seen["stdin"] = input
                return b"out", b""

        async def create_subprocess_exec(*cmd, **_kwargs):
            seen["cmd"] = list(cmd)
            workspace = cmd[cmd.index("-v") + 1].split(":")[0]
            seen["files"] = {
                name: open(os.path.join(workspace, name)).read()
                for name in os.listdir(workspace)
            }
            return _Process()

        monkeypatch.setattr(
            executor_module.asyncio, "create_subprocess_exec", create_subprocess_exec
        )

        result = await executor_module.DockerToolExecutor().execute(
            config=DockerToolConfig(
                image="img",
                command=["true"],
                input_mode="both",
                input_file_path="/workspace/analysis.py",
            ),
            execution_input=DockerToolExecutionInput(
                stdin_data='{"n": 1}', input_content="print(1)"
            ),
        )

        assert result.success is True
        assert seen["files"] == {"analysis.py": "print(1)"}
        assert seen["stdin"] == b'{"n": 1}'
        assert "-i" in seen["cmd"]
