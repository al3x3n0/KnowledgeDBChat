"""A measurement claim must cover the remote Job's actual lifetime."""

import asyncio

import httpx
import pytest

from app.core.config import settings
from app.services import agent_measurement_lock as lock
from app.services import kubernetes_sandbox as runner
from app.services.sandbox_invocation import SandboxStepPending, invocation
from tests.test_kubernetes_sandbox import IMAGE, FakeAPI
from tests.test_measurement_lock import FakeRedis

pytestmark = pytest.mark.unit


@pytest.fixture
def measurement(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SANDBOX_EXECUTION_BACKEND", "kubernetes")
    redis = FakeRedis()
    monkeypatch.setattr(lock, "_client", lambda: asyncio.sleep(0, result=redis))
    config = runner.Configuration(
        "experiments", "data", tmp_path / "records", poll_seconds=0.001
    )
    api = FakeAPI(config)
    api.complete = False
    workdir = tmp_path / "source"
    workdir.mkdir()
    (workdir / "input.txt").write_text("benchmark")

    async def execute(*, enabled=True, identity="measurement"):
        with invocation("agent", identity, defer=True):
            async with lock.exclusive_measurement(
                enabled=enabled, ttl_seconds=lock.ttl_for(30)
            ):
                return await runner.run(
                    "benchmark",
                    str(workdir),
                    image=IMAGE,
                    timeout_seconds=30,
                    memory="2048m",
                    cpus="2",
                    config=config,
                    api=api,
                )

    return redis, api, execute


@pytest.mark.parametrize("cancelled", [False, True])
async def test_protected_measurement_waits_until_the_job_finishes(
    measurement, cancelled
):
    redis, api, execute = measurement
    task = asyncio.create_task(execute())
    try:
        await asyncio.wait_for(api.created.wait(), timeout=5)
        await asyncio.sleep(0.03)
        assert not task.done(), "suspension would release the lock while the Job runs"
        assert redis.store and not redis.deleted
        if cancelled:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert redis.store and not redis.deleted
            assert api.jobs and api.deletes == []
            async with lock.exclusive_measurement(wait_seconds=0) as contender:
                assert contender.serialized is False
        else:
            api.complete = True
            assert (await asyncio.wait_for(task, timeout=5))[0] == 0
            assert not redis.store and len(redis.deleted) == 1
    finally:
        if not task.done():
            task.cancel()
        try:
            await task
        except BaseException:
            pass


async def test_unknown_submission_outcome_keeps_the_claim_until_its_ttl(
    measurement, monkeypatch
):
    redis, api, execute = measurement
    actual = api.request

    async def lost_response(method, path, **kwargs):
        response = await actual(method, path, **kwargs)
        if method == "POST":
            return api.response(method, path, 503, {})
        return response

    monkeypatch.setattr(api, "request", lost_response)
    with pytest.raises(httpx.HTTPStatusError):
        await execute()
    assert api.jobs and redis.store and not redis.deleted


async def test_unprotected_measurement_can_still_release_its_celery_slot(measurement):
    redis, api, execute = measurement
    with pytest.raises(SandboxStepPending):
        await execute(enabled=False)
    assert api.jobs and not redis.store and not redis.deleted


async def test_a_protected_measurement_does_not_change_another_tasks_waiting_mode(
    measurement,
):
    redis, api, execute = measurement
    task = asyncio.create_task(execute())
    try:
        await asyncio.wait_for(api.created.wait(), timeout=5)
        await asyncio.sleep(0.03)
        assert not task.done()
        with pytest.raises(SandboxStepPending):
            await execute(enabled=False, identity="independent-work")
        assert api.posts == 2 and redis.store
        api.complete = True
        assert (await asyncio.wait_for(task, timeout=5))[0] == 0
        assert not redis.store
    finally:
        if not task.done():
            task.cancel()
        try:
            await task
        except BaseException:
            pass
