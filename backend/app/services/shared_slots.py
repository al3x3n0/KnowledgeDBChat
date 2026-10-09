"""A counted resource shared by every process: N holders at once, in Redis.

Two things are rationed this way -- model calls and sandbox runs -- and the
reason is the same for both: an in-process semaphore caps one process, and the
deployment's real ceiling is that number times however many processes there
are, which grows with every replica without anyone deciding it should.

A holder is a member of a sorted set whose score is when its slot expires, so
a process that dies holding one gives it back by doing nothing.

The cap is **soft**, on purpose. Redis being unreachable switches every cap
off for a while rather than failing the work, and a caller that has waited
its ``wait`` proceeds anyway. A limiter that can stop all work costs more than
the overshoot it prevents.
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from typing import Optional

from loguru import logger

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

#: After Redis fails, how long every cap is skipped before trying again.
REDIS_BACKOFF_SECONDS = 30.0

_redis_down_until = 0.0


async def _redis():
    from app.core.cache import get_redis_client

    return await get_redis_client()


async def acquire(
    key: str, limit: int, *, ttl: float, wait: float, what: str
) -> Optional[str]:
    """Take a slot under ``key``; the holder id, or None when none was taken.

    None means "go ahead without one": the cap is off (``limit`` <= 0), Redis
    is unreachable, or ``wait`` seconds passed with the cap still full.
    ``what`` names the resource in the log line.
    """
    global _redis_down_until
    if limit <= 0 or time.monotonic() < _redis_down_until:
        return None
    holder = uuid.uuid4().hex
    ttl = max(30.0, float(ttl))
    deadline = time.monotonic() + max(0.0, float(wait))
    delay = 0.05
    while True:
        try:
            client = await _redis()
            now = time.time()
            taken = await client.eval(_ACQUIRE, 1, key, now, now + ttl, limit, holder)
        except Exception as exc:
            _redis_down_until = time.monotonic() + REDIS_BACKOFF_SECONDS
            logger.warning(
                f"Shared cap on {what} skipped for {REDIS_BACKOFF_SECONDS:.0f}s: "
                f"Redis unavailable ({exc})"
            )
            return None
        if int(taken or 0) == 1:
            return holder
        if time.monotonic() >= deadline:
            logger.warning(
                f"Shared cap on {what} ({limit}) still full after {wait:.0f}s; "
                "proceeding"
            )
            return None
        # Jittered, so waiters across processes do not retry in step.
        await asyncio.sleep(delay * (0.5 + random.random()))
        delay = min(delay * 2, 1.0)


async def release(key: str, holder: Optional[str]) -> None:
    if not holder:
        return
    try:
        client = await _redis()
        await client.zrem(key, holder)
    except Exception as exc:
        # The slot expires by itself; nothing else depends on this.
        logger.debug(f"Slot release failed, will expire: {exc}")


async def in_use(key: str) -> Optional[int]:
    """Slots currently held under ``key``, or None when Redis cannot say."""
    try:
        client = await _redis()
        await client.zremrangebyscore(key, "-inf", time.time())
        return int(await client.zcard(key))
    except Exception:
        return None
