"""How many model calls run at once: per process, and across the deployment.

The cap was one module-level ``asyncio.Semaphore(LLM_MAX_CONCURRENCY)``. Two
things were wrong with it.

It is per process. The API runs several, each worker several more, so the
deployment's real ceiling was the setting times the number of processes --
64 on the chart's defaults for a setting that reads 4 -- and adding a replica
raised it without anyone deciding to. ``LLM_GLOBAL_MAX_CONCURRENCY`` is a
second cap counted in Redis, shared by every process.

And a semaphore belongs to the event loop it first waited on. Celery runs
each task in a fresh loop, so in a worker the first contended acquire bound
the semaphore to a loop that then closed, and a later contended one raised
"bound to a different event loop". The local cap is now kept per loop.

The global cap is **soft**, on purpose. Redis being unreachable disables it
for a while rather than failing calls, a caller that has waited
``LLM_GLOBAL_ACQUIRE_TIMEOUT_SECONDS`` proceeds anyway, and a slot expires
after ``LLM_GLOBAL_SLOT_TTL_SECONDS`` so a process that died holding one does
not hold it for ever. Overshooting a provider's quota costs a 429 the caller
already handles; a limiter that can stop every model call costs the platform.
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from loguru import logger

from app.core.config import settings
from app.utils.per_loop import PerLoop

SLOTS_KEY = "llm:concurrency:slots"

#: Atomically: forget expired holders, then take a slot if one is free.
#: KEYS[1] the sorted set (member = holder, score = when its slot expires).
#: ARGV: now, expiry, limit, holder. Returns 1 when the slot was taken.
_ACQUIRE = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', ARGV[1])
if redis.call('ZCARD', KEYS[1]) < tonumber(ARGV[3]) then
  redis.call('ZADD', KEYS[1], ARGV[2], ARGV[4])
  redis.call('EXPIRE', KEYS[1], math.ceil(ARGV[2] - ARGV[1]) + 60)
  return 1
end
return 0
"""

#: After Redis fails, how long the global cap is skipped before trying again.
REDIS_BACKOFF_SECONDS = 30.0

_redis_down_until = 0.0


class _Sized:
    """A loop's semaphore and the size it was made at."""

    def __init__(self) -> None:
        self.size = 0
        self.semaphore: Optional[asyncio.Semaphore] = None


_local: PerLoop[_Sized] = PerLoop(_Sized)


def _local_semaphore() -> asyncio.Semaphore:
    """This event loop's semaphore, at the currently configured size."""
    held = _local.get()
    size = max(1, int(settings.LLM_MAX_CONCURRENCY))
    if held.semaphore is None or held.size != size:
        # Resized: holders of the old one keep their reference and release it.
        held.size, held.semaphore = size, asyncio.Semaphore(size)
    return held.semaphore


async def _redis():
    from app.core.cache import get_redis_client

    return await get_redis_client()


async def _acquire_global(limit: int) -> Optional[str]:
    """Take a shared slot; None when the cap is off, unreachable or timed out."""
    global _redis_down_until
    if limit <= 0 or time.monotonic() < _redis_down_until:
        return None
    holder = uuid.uuid4().hex
    ttl = max(30.0, float(settings.LLM_GLOBAL_SLOT_TTL_SECONDS))
    deadline = time.monotonic() + max(
        0.0, float(settings.LLM_GLOBAL_ACQUIRE_TIMEOUT_SECONDS)
    )
    delay = 0.05
    while True:
        try:
            client = await _redis()
            now = time.time()
            taken = await client.eval(
                _ACQUIRE, 1, SLOTS_KEY, now, now + ttl, limit, holder
            )
        except Exception as exc:
            _redis_down_until = time.monotonic() + REDIS_BACKOFF_SECONDS
            logger.warning(
                "LLM global concurrency cap skipped for "
                f"{REDIS_BACKOFF_SECONDS:.0f}s: Redis unavailable ({exc})"
            )
            return None
        if int(taken or 0) == 1:
            return holder
        if time.monotonic() >= deadline:
            logger.warning(
                f"LLM global concurrency cap ({limit}) still full after "
                f"{settings.LLM_GLOBAL_ACQUIRE_TIMEOUT_SECONDS}s; proceeding"
            )
            return None
        # Jittered, so waiters across processes do not retry in step.
        await asyncio.sleep(delay * (0.5 + random.random()))
        delay = min(delay * 2, 1.0)


async def _release_global(holder: Optional[str]) -> None:
    if not holder:
        return
    try:
        client = await _redis()
        await client.zrem(SLOTS_KEY, holder)
    except Exception as exc:
        # The slot expires by itself; nothing else depends on this.
        logger.debug(f"LLM slot release failed, will expire: {exc}")


@asynccontextmanager
async def llm_slot() -> AsyncIterator[None]:
    """Hold one model-call slot: this process's, then the deployment's.

    Local first, so a process never queues more than its own share on Redis.
    """
    semaphore = _local_semaphore()
    await semaphore.acquire()
    holder: Optional[str] = None
    try:
        holder = await _acquire_global(int(settings.LLM_GLOBAL_MAX_CONCURRENCY))
        yield
    finally:
        semaphore.release()
        await _release_global(holder)


async def slots_in_use() -> Optional[int]:
    """Shared slots currently held, or None when Redis cannot say."""
    try:
        client = await _redis()
        await client.zremrangebyscore(SLOTS_KEY, "-inf", time.time())
        return int(await client.zcard(SLOTS_KEY))
    except Exception:
        return None
