"""Sandbox runs on one daemon are counted across processes.

Nothing limited them: every worker process could start as many containers as
it had tool calls, each allowed two CPUs, on a daemon none of them could see
the load of. Beyond exhausting the host, that corrupts results -- a run timed
beside too many others measures the others.

The counting itself is ``shared_slots`` (tested, against a real Redis, in
test_llm_concurrency). These check what the sandbox adds: which daemon a run
is counted against, that waiting is not charged to the run, and that a slot is
returned on every way out.
"""

import asyncio
import time

import pytest

from app.core.config import settings
from app.services import agent_sandbox_runtime as runtime
from app.services import shared_slots

pytestmark = pytest.mark.unit


class FakeRedis:
    def __init__(self):
        self.sets = {}

    async def eval(self, script, numkeys, key, now, expiry, limit, holder):
        slots = self.sets.setdefault(key, {})
        for gone in [h for h, e in slots.items() if e <= float(now)]:
            del slots[gone]
        if len(slots) < int(limit):
            slots[holder] = float(expiry)
            return 1
        return 0

    async def zrem(self, key, holder):
        self.sets.get(key, {}).pop(holder, None)


class FakeProcess:
    """A `docker run` client that takes `seconds` and can be killed."""

    running = 0
    peak = 0
    started = []

    def __init__(self, seconds: float):
        self.seconds = seconds
        self.returncode = 0

    async def communicate(self):
        cls = type(self)
        cls.running += 1
        cls.peak = max(cls.peak, cls.running)
        cls.started.append(time.monotonic())
        try:
            await asyncio.sleep(self.seconds)
        finally:
            cls.running -= 1
        return b"out", b""

    def kill(self):
        pass


@pytest.fixture
def sandbox(monkeypatch):
    redis = FakeRedis()
    FakeProcess.running = FakeProcess.peak = 0
    FakeProcess.started = []
    state = {"seconds": 0.05}

    async def client():
        return redis

    async def spawn(*argv, **kwargs):
        return FakeProcess(state["seconds"])

    async def removed(name):
        return True

    monkeypatch.setattr(shared_slots, "_redis", client)
    monkeypatch.setattr(shared_slots, "_redis_down_until", 0.0)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(runtime, "remove_container", removed)
    monkeypatch.setattr(settings, "SANDBOX_MAX_CONCURRENT_RUNS", 2)
    monkeypatch.setattr(settings, "SANDBOX_SLOT_WAIT_SECONDS", 30.0)
    monkeypatch.setenv("DOCKER_HOST", "tcp://sandbox-docker:2376")
    state["redis"] = redis
    return state


def _run(timeout: int = 5):
    return runtime.run_in_sandbox("true", "/w", image="img", timeout_seconds=timeout)


def _held(state) -> int:
    return sum(len(slots) for slots in state["redis"].sets.values())


async def test_runs_beyond_the_cap_wait_their_turn(sandbox):
    results = await asyncio.gather(*(_run() for _ in range(6)))
    assert FakeProcess.peak == 2
    assert all(rc == 0 for rc, _, _ in results)
    assert _held(sandbox) == 0


async def test_waiting_for_a_slot_is_not_charged_to_the_run(sandbox):
    # Each run takes 0.6s of a 1s limit; the third waits about 0.6s for a
    # slot first. Counted against its own limit, it would time out.
    sandbox["seconds"] = 0.6
    results = await asyncio.gather(*(_run(timeout=1) for _ in range(3)))
    assert [rc for rc, _, _ in results] == [0, 0, 0]
    assert FakeProcess.started[2] - FakeProcess.started[0] > 0.4


async def test_a_timed_out_run_gives_its_slot_back(sandbox):
    sandbox["seconds"] = 5
    with pytest.raises(asyncio.TimeoutError):
        await _run(timeout=0.2)
    assert _held(sandbox) == 0


async def test_a_cancelled_run_gives_its_slot_back(sandbox):
    sandbox["seconds"] = 5
    task = asyncio.create_task(_run())
    await asyncio.sleep(0.1)
    assert _held(sandbox) == 1
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.05)  # the shielded release
    assert _held(sandbox) == 0


async def test_zero_turns_the_cap_off(sandbox, monkeypatch):
    monkeypatch.setattr(settings, "SANDBOX_MAX_CONCURRENT_RUNS", 0)
    await asyncio.gather(*(_run() for _ in range(5)))
    assert FakeProcess.peak == 5
    assert sandbox["redis"].sets == {}


async def test_a_full_cap_only_delays_a_run(sandbox, monkeypatch):
    monkeypatch.setattr(settings, "SANDBOX_SLOT_WAIT_SECONDS", 0.3)
    key = runtime.sandbox_slots_key()
    sandbox["redis"].sets[key] = {"a": time.time() + 600, "b": time.time() + 600}
    started = time.monotonic()
    rc, _, _ = await _run()
    assert rc == 0 and time.monotonic() - started >= 0.25


def test_a_shared_daemon_is_one_count_and_a_local_one_is_per_host(monkeypatch):
    monkeypatch.setenv("DOCKER_HOST", "tcp://sandbox-docker:2376")
    shared = runtime.sandbox_slots_key()
    assert shared == "sandbox:slots:tcp://sandbox-docker:2376"

    # The chart's sidecar: every pod has the same socket path and its own
    # daemon, so the host name keeps two pods from counting against each other.
    monkeypatch.setenv("DOCKER_HOST", "unix:///run/sandbox/docker.sock")
    monkeypatch.setattr(runtime.socket, "gethostname", lambda: "celery-agents-abc")
    one = runtime.sandbox_slots_key()
    monkeypatch.setattr(runtime.socket, "gethostname", lambda: "celery-agents-xyz")
    assert one != runtime.sandbox_slots_key()
    assert "celery-agents-abc" in one

    monkeypatch.delenv("DOCKER_HOST")
    assert runtime.sandbox_slots_key().endswith(":default")
