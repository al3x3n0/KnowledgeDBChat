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

The global cap is counted by ``shared_slots`` and is **soft** (see there): a
caller that has waited ``LLM_GLOBAL_ACQUIRE_TIMEOUT_SECONDS`` proceeds, and a
slot expires after ``LLM_GLOBAL_SLOT_TTL_SECONDS``. Overshooting a provider's
quota costs a 429 the caller already handles.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from app.core.config import settings
from app.services import shared_slots
from app.utils.per_loop import PerLoop

SLOTS_KEY = "llm:concurrency:slots"


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


@asynccontextmanager
async def llm_slot() -> AsyncIterator[None]:
    """Hold one model-call slot: this process's, then the deployment's.

    Local first, so a process never queues more than its own share on Redis.
    """
    semaphore = _local_semaphore()
    await semaphore.acquire()
    holder: Optional[str] = None
    try:
        holder = await shared_slots.acquire(
            SLOTS_KEY,
            int(settings.LLM_GLOBAL_MAX_CONCURRENCY),
            ttl=settings.LLM_GLOBAL_SLOT_TTL_SECONDS,
            wait=settings.LLM_GLOBAL_ACQUIRE_TIMEOUT_SECONDS,
            what="model calls",
        )
        yield
    finally:
        semaphore.release()
        await shared_slots.release(SLOTS_KEY, holder)


async def slots_in_use() -> Optional[int]:
    """Shared slots currently held, or None when Redis cannot say."""
    return await shared_slots.in_use(SLOTS_KEY)
