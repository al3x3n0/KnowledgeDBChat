"""A sandbox run that is abandoned must not leave its container running.

`process.kill()` -- what the timeout path did before -- kills the `docker run`
client. The container belongs to the daemon and keeps going, holding its
--cpus share. One orphaned gem5 burned 150% CPU for an hour on this machine
and corrupted every wall-clock measurement taken while it ran. That is the
shape of the bug worth testing: it is silent, and it comes back as bad numbers
rather than as an error.
"""

import asyncio

import pytest

from app.services import agent_sandbox_runtime as runtime


class FakeProcess:
    """A `docker run` client that never finishes on its own."""

    def __init__(self):
        self.killed = False
        self.returncode = None

    async def communicate(self):
        await asyncio.sleep(3600)
        return b"", b""  # pragma: no cover - the wait is the point

    def kill(self):
        self.killed = True


@pytest.fixture
def abandoned(monkeypatch):
    """A run whose client hangs, with the container removal recorded."""
    removed = []
    process = FakeProcess()

    async def _spawn(*args, **_kwargs):
        _spawn.command = list(args)
        return process

    async def _remove(name):
        removed.append(name)
        return True

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _spawn)
    monkeypatch.setattr(runtime, "remove_container", _remove)
    return process, removed, _spawn


class TestTheContainerIsTornDown:
    async def test_a_timed_out_run_removes_its_container(self, abandoned):
        process, removed, _spawn = abandoned

        with pytest.raises(asyncio.TimeoutError):
            await runtime.run_in_sandbox(
                "sleep 600", "/tmp", image="img", timeout_seconds=0.05
            )

        assert process.killed, "the client was not killed"
        assert removed, "the container was left running"

    async def test_a_cancelled_run_removes_its_container(self, abandoned):
        """A job cancelled mid-run abandons its container just as thoroughly,
        and that path is the one a user actually triggers."""
        process, removed, _spawn = abandoned

        task = asyncio.create_task(
            runtime.run_in_sandbox(
                "sleep 600", "/tmp", image="img", timeout_seconds=600
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert removed, "the container was left running"

    async def test_the_container_removed_is_the_one_that_was_started(self, abandoned):
        """Naming it is the whole mechanism: without a name there is no handle
        on the container at all once the client is gone."""
        _process, removed, spawn = abandoned

        with pytest.raises(asyncio.TimeoutError):
            await runtime.run_in_sandbox(
                "sleep 600", "/tmp", image="img", timeout_seconds=0.05
            )

        command = spawn.command
        assert "--name" in command
        assert command[command.index("--name") + 1] == removed[0]


class TestTheCommandItself:
    def test_a_named_run_carries_its_name(self):
        command = runtime.docker_command(
            image="img", workdir="/w", script="true", timeout_seconds=10, name="box"
        )

        assert command[command.index("--name") + 1] == "box"

    def test_an_unnamed_run_is_unchanged(self):
        """Callers that build a command themselves keep the posture they had."""
        command = runtime.docker_command(
            image="img", workdir="/w", script="true", timeout_seconds=10
        )

        assert "--name" not in command
        assert "--network" in command and "--cap-drop" in command


class TestRemovalIsBestEffort:
    async def test_a_failing_removal_does_not_mask_the_timeout(self, monkeypatch):
        """The caller is already reporting a failure. A cleanup error raised
        into that path would replace a truthful timeout with a confusing one."""

        async def _explode(*_args, **_kwargs):
            raise OSError("docker is not there")

        monkeypatch.setattr(asyncio, "create_subprocess_exec", _explode)

        assert await runtime.remove_container("box") is False

    async def test_nothing_to_remove_is_not_an_error(self):
        assert await runtime.remove_container("") is False


class TestThereIsOneConfinedRun:
    """Five places built `docker run` by hand. They agreed on the posture and
    only one of them named its container, so a timeout in the other four left
    the container running -- the leak this file's first tests are about, fixed
    in one copy and not in the rest."""

    def test_nothing_else_builds_the_confined_command(self):
        import re
        from pathlib import Path

        app = Path(__file__).resolve().parents[1] / "app"
        builders = sorted(
            str(path.relative_to(app))
            for path in app.rglob("*.py")
            if re.search(r'"--cap-drop"', path.read_text(encoding="utf-8"))
        )
        assert builders == ["services/agent_sandbox_runtime.py"], (
            "the confined `docker run` is built outside agent_sandbox_runtime; "
            f"call docker_command() instead: {builders}"
        )

    def test_a_program_can_be_run_directly_and_the_mount_made_read_only(self):
        command = runtime.docker_command(
            image="img",
            workdir="/tmp/w",
            argv=["python", "-I", "-S", "demo.py"],
            pids_limit="64",
            name="n",
            read_only=True,
        )
        assert command[-4:] == ["python", "-I", "-S", "demo.py"]
        assert "/tmp/w:/work:ro" in command
        assert command[command.index("--pids-limit") + 1] == "64"
        # The posture is not optional, whatever else is.
        for flag, value in (
            ("--network", "none"),
            ("--cap-drop", "ALL"),
            ("--security-opt", "no-new-privileges"),
            ("--user", "65534:65534"),
            ("--name", "n"),
        ):
            assert command[command.index(flag) + 1] == value

    def test_every_caller_names_its_container_and_removes_it(self):
        """A name nobody removes is the same leak with a label on it."""
        from pathlib import Path

        app = Path(__file__).resolve().parents[1] / "app"
        for relative in (
            "services/agent_experiment_runner_service.py",
            "services/agent_ingestion_demo_runner_service.py",
            "api/endpoints/admin.py",
            "services/docker_tool_executor.py",
        ):
            source = (app / relative).read_text(encoding="utf-8")
            assert "new_container_name()" in source, relative
            assert "remove_container" in source, relative

    def test_a_thread_can_remove_a_container_too(self, monkeypatch):
        calls = []

        def fake_run(argv, **_kwargs):
            calls.append(argv)

            class Done:
                returncode = 0

            return Done()

        monkeypatch.setattr(runtime.subprocess, "run", fake_run)
        assert runtime.remove_container_sync("kdbc-sandbox-abc")
        assert calls == [["docker", "rm", "--force", "kdbc-sandbox-abc"]]
        assert not runtime.remove_container_sync("")


class TestACustomDockerTool:
    def _command(self, **config):
        from app.schemas.docker_tool import DockerToolConfig
        from app.services.docker_tool_executor import DockerToolExecutor

        return DockerToolExecutor()._build_docker_command(
            config=DockerToolConfig(image="img", command=["true"], **config),
            workspace_dir="/tmp/w",
            container_name="kdbc-sandbox-abc",
        )

    def test_it_is_named_so_a_timeout_can_remove_it(self):
        command = self._command()
        assert command[command.index("--name") + 1] == "kdbc-sandbox-abc"

    def test_it_cannot_gain_privileges_or_fork_without_bound(self):
        command = self._command()
        assert command[command.index("--security-opt") + 1] == "no-new-privileges"
        assert "--pids-limit" in command
