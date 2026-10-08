"""Model calls are capped per process and, through Redis, across processes.

The per-process cap was one module-level semaphore, which made the real
ceiling "the setting times the number of processes" and, in a Celery worker,
tied the semaphore to the first task's event loop.

The shared cap's counting is a Lua script, which a fake cannot check, so the
tests under ``live`` need a real Redis (``TEST_REDIS_URL``; the Postgres CI
job provides one) and skip without it. The rest use a small in-memory stand-in
that implements the same contract.
"""

import asyncio
import os
import time

import pytest

from app.core.config import settings
from app.services import llm_concurrency
from app.services.llm_concurrency import SLOTS_KEY, llm_slot

pytestmark = pytest.mark.unit


class FakeRedis:
    """The sorted-set contract of the acquire script, in Python."""

    def __init__(self):
        self.slots = {}
        self.fail = False

    async def eval(self, script, numkeys, key, now, expiry, limit, holder):
        if self.fail:
            raise ConnectionError("redis is down")
        self.slots = {h: e for h, e in self.slots.items() if e > float(now)}
        if len(self.slots) < int(limit):
            self.slots[holder] = float(expiry)
            return 1
        return 0

    async def zrem(self, key, holder):
        self.slots.pop(holder, None)

    async def zremrangebyscore(self, key, low, high):
        self.slots = {h: e for h, e in self.slots.items() if e > float(high)}

    async def zcard(self, key):
        return len(self.slots)


@pytest.fixture
def redis(monkeypatch):
    fake = FakeRedis()

    async def client():
        return fake

    monkeypatch.setattr(llm_concurrency, "_redis", client)
    monkeypatch.setattr(llm_concurrency, "_redis_down_until", 0.0)
    monkeypatch.setattr(settings, "LLM_MAX_CONCURRENCY", 50)
    monkeypatch.setattr(settings, "LLM_GLOBAL_MAX_CONCURRENCY", 2)
    monkeypatch.setattr(settings, "LLM_GLOBAL_ACQUIRE_TIMEOUT_SECONDS", 5.0)
    return fake


async def _peak(calls: int, hold: float = 0.05) -> int:
    running = peak = 0

    async def one():
        nonlocal running, peak
        async with llm_slot():
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(hold)
            running -= 1

    await asyncio.gather(*(one() for _ in range(calls)))
    return peak


async def test_the_shared_cap_holds_whatever_the_local_one_allows(redis):
    assert await _peak(8) == 2
    assert redis.slots == {}, "every slot is returned"


async def test_the_local_cap_still_applies(redis, monkeypatch):
    monkeypatch.setattr(settings, "LLM_MAX_CONCURRENCY", 1)
    monkeypatch.setattr(settings, "LLM_GLOBAL_MAX_CONCURRENCY", 10)
    assert await _peak(5) == 1


async def test_zero_turns_the_shared_cap_off(redis, monkeypatch):
    monkeypatch.setattr(settings, "LLM_GLOBAL_MAX_CONCURRENCY", 0)
    monkeypatch.setattr(settings, "LLM_MAX_CONCURRENCY", 3)
    assert await _peak(6) == 3
    assert redis.slots == {}


async def test_a_slot_is_returned_when_the_call_raises(redis):
    with pytest.raises(RuntimeError):
        async with llm_slot():
            assert len(redis.slots) == 1
            raise RuntimeError("the provider failed")
    assert redis.slots == {}


async def test_redis_being_down_lets_calls_through_and_stops_asking(redis):
    redis.fail = True
    async with llm_slot():
        pass
    # It does not probe a dead Redis on every call.
    assert llm_concurrency._redis_down_until > time.monotonic()
    redis.fail = False
    async with llm_slot():
        assert redis.slots == {}, "still inside the back-off"


async def test_a_full_cap_delays_a_call_and_then_lets_it_through(redis, monkeypatch):
    monkeypatch.setattr(settings, "LLM_GLOBAL_ACQUIRE_TIMEOUT_SECONDS", 0.3)
    redis.slots = {"dead-1": time.time() + 600, "dead-2": time.time() + 600}
    started = time.monotonic()
    async with llm_slot():
        waited = time.monotonic() - started
    assert 0.25 <= waited < 3.0
    assert set(redis.slots) == {"dead-1", "dead-2"}, "it took no slot it must return"


async def test_a_dead_holders_slot_expires(redis):
    redis.slots = {"dead-1": time.time() - 1, "dead-2": time.time() - 1}
    started = time.monotonic()
    async with llm_slot():
        assert len(redis.slots) == 1
    assert time.monotonic() - started < 1.0


def test_each_event_loop_gets_its_own_local_semaphore(monkeypatch):
    # A Celery task runs in a fresh loop. One semaphore shared across them
    # raised "bound to a different event loop" on its first contended wait.
    monkeypatch.setattr(settings, "LLM_GLOBAL_MAX_CONCURRENCY", 0)
    monkeypatch.setattr(settings, "LLM_MAX_CONCURRENCY", 1)
    for _ in range(3):
        assert asyncio.run(_peak(3, hold=0.01)) == 1


def test_the_service_has_no_semaphore_of_its_own():
    from app.services import llm_service

    assert not hasattr(llm_service, "_LLM_SEMAPHORE")
    source = open(llm_service.__file__, encoding="utf-8").read()
    assert source.count("async with llm_slot()") == 2
    assert "Semaphore(" not in source


# -- against a real Redis ----------------------------------------------------

REDIS_URL = os.environ.get("TEST_REDIS_URL", "")
live = pytest.mark.skipif(not REDIS_URL, reason="TEST_REDIS_URL is not set")


@pytest.fixture
async def real_redis(monkeypatch):
    import redis.asyncio as aioredis

    client = aioredis.from_url(REDIS_URL)
    await client.delete(SLOTS_KEY)

    async def get():
        return client

    monkeypatch.setattr(llm_concurrency, "_redis", get)
    monkeypatch.setattr(llm_concurrency, "_redis_down_until", 0.0)
    monkeypatch.setattr(settings, "LLM_MAX_CONCURRENCY", 50)
    monkeypatch.setattr(settings, "LLM_GLOBAL_MAX_CONCURRENCY", 3)
    monkeypatch.setattr(settings, "LLM_GLOBAL_ACQUIRE_TIMEOUT_SECONDS", 10.0)
    yield client
    await client.delete(SLOTS_KEY)
    await (getattr(client, "aclose", None) or client.close)()


@live
async def test_live_the_script_caps_and_returns_every_slot(real_redis):
    assert await _peak(12) == 3
    assert await real_redis.zcard(SLOTS_KEY) == 0
    assert llm_concurrency._redis_down_until == 0.0, "the script ran without error"


@live
async def test_live_an_expired_holder_is_forgotten(real_redis):
    await real_redis.zadd(SLOTS_KEY, {"dead-1": 1.0, "dead-2": 2.0, "dead-3": 3.0})
    started = time.monotonic()
    async with llm_slot():
        assert await real_redis.zcard(SLOTS_KEY) == 1
    assert time.monotonic() - started < 2.0
    assert await llm_concurrency.slots_in_use() == 0


@live
async def test_live_the_key_expires_by_itself(real_redis):
    async with llm_slot():
        ttl = await real_redis.ttl(SLOTS_KEY)
    assert 0 < ttl <= settings.LLM_GLOBAL_SLOT_TTL_SECONDS + 61
