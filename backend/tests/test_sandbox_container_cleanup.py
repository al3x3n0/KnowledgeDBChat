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
