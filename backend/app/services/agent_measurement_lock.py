"""One wall-clock measurement at a time, per host.

A swarm fans its roles out to work in parallel, which is the point of a swarm
-- and then two of those roles each start a CPU-bound timing loop on the same
machine, at the same second, and time each other's contention.

Measured, on the first swarm whose roles both benchmarked the same kernel:

    Verifier   | start=13:51:13 | end=13:53:18
    Researcher | start=13:51:13 | end=13:53:58

Both reported the host `busy`/`saturated` and their own trials spread by 130%
and 142%. Neither was wrong about the host. They *were* the host's load. The
merge then had two numbers 33% apart, each with an error bar wide enough to
swallow the difference, and the honest verdict over such a pair is that
nothing was established -- so the swarm paid for two independent measurements
and bought one unusable one.

Independence is what makes corroboration worth anything, and two measurements
taken through the same contended CPU are not independent. So wall-clock
measurement is serialised: a role that wants to time something waits for the
one ahead of it.

Deliberately narrow. Only measurement takes this lock -- compiles, correctness
checks and every other sandbox run stay parallel, because they are not
claiming a number that the machine's state can invalidate. That leaves some
residual load from the other roles' non-timing work, which is real but small
beside two simultaneous benchmark loops, and which the existing `/proc/loadavg`
sampling still reports.

Never fails the work. If Redis is unreachable, or the wait runs out, the
measurement proceeds unserialised and says so in its own result, because a
measurement with a caveat beats no measurement at all. `serialized: false` on
a finding is the signal that it carries the old risk.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from loguru import logger

#: One key per host. Every worker in this deployment shares one Docker daemon
#: and therefore one set of CPUs, so one key is the right granularity. A
#: deployment spreading workers over several machines wants one scope per
#: machine, or it will serialise measurements that never competed.
KEY_PREFIX = "agent:measurement:lock"

#: How long a holder's claim survives without being released. A crashed or
#: killed worker must not block measurement for ever, so the claim expires on
#: its own; the value is derived from the caller's own timeout rather than
#: guessed, because a lock that expires mid-measurement is worse than no lock
#: -- two runs would then be timing each other while both believe they hold it.
TTL_MARGIN_SECONDS = 60

_POLL_SECONDS = 0.5


@dataclass(frozen=True)
class LockOutcome:
    """Whether this measurement had the machine to itself, and what it cost.

    `held` and `serialized` are deliberately not the same question. With
    serialisation switched off the block must still run, so `held` is true --
    it means "go ahead", not "you have the machine". Reporting that as
    `serialized: true` claimed every measurement was protected the moment
    somebody turned the feature off, which is the precise shape of bug this
    whole area keeps producing: a field that reads as evidence and is actually
    a default. Caught in an A/B run where the control, with the lock disabled,
    reported `serialized=True` while visibly sharing the CPU.
    """

    held: bool
    waited_seconds: float
    detail: str
    enabled: bool = True

    @property
    def serialized(self) -> bool:
        """Whether this timing actually had the host's CPUs to itself."""
        return bool(self.enabled and self.held)

    def as_quality(self) -> dict:
        """The part that belongs on the measurement's own result."""
        quality = {"serialized": self.serialized}
        if self.waited_seconds >= 0.5:
            quality["measurement_queue_wait_seconds"] = round(self.waited_seconds, 1)
        if not self.serialized and self.detail:
            quality["serialization_note"] = self.detail
        return quality


async def _client():
    from app.core.cache import get_redis_client

    return await get_redis_client()


@asynccontextmanager
async def exclusive_measurement(
    *,
    scope: str = "default",
    wait_seconds: float = 180.0,
    ttl_seconds: float = 180.0,
    enabled: bool = True,
) -> AsyncIterator[LockOutcome]:
    """Hold the host's measurement lock for the duration of the block.

    Yields a `LockOutcome` describing whether the lock was actually held. The
    block always runs: this serialises measurement, it does not gate it.
    """
    if not enabled:
        yield LockOutcome(
            True,
            0.0,
            "measurement serialisation is switched off; this timing may have "
            "shared the machine with another one",
            enabled=False,
        )
        return

    key = f"{KEY_PREFIX}:{scope}"
    token = uuid.uuid4().hex
    ttl_ms = int(max(ttl_seconds, 1) * 1000)
    started = time.monotonic()
    acquired = False
    redis = None

    try:
        redis = await _client()
        deadline = started + max(wait_seconds, 0.0)
        while True:
            acquired = bool(await redis.set(key, token, nx=True, px=ttl_ms))
            if acquired or time.monotonic() >= deadline:
                break
            await asyncio.sleep(_POLL_SECONDS)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(f"Measurement lock unavailable ({exc}); measuring anyway")
        yield LockOutcome(
            False,
            time.monotonic() - started,
            "could not reach the measurement lock; this timing may have shared "
            "the machine with another one",
        )
        return

    waited = time.monotonic() - started
    if not acquired:
        yield LockOutcome(
            False,
            waited,
            f"another measurement held the machine for longer than "
            f"{wait_seconds:.0f}s; this timing may have shared it",
        )
        return

    try:
        yield LockOutcome(True, waited, "")
    finally:
        # Release only our own claim. If the TTL expired mid-measurement the
        # key now belongs to somebody else, and deleting it blindly would hand
        # the machine to a third run while the second is still timing.
        try:
            current = await redis.get(key)
            if current is not None and _as_text(current) == token:
                await redis.delete(key)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Could not release the measurement lock: {exc}")


def _as_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


def ttl_for(timeout_seconds: Optional[float]) -> float:
    """A claim that outlives the measurement it protects."""
    try:
        base = float(timeout_seconds or 0)
    except (TypeError, ValueError):
        base = 0.0
    return max(base, 0.0) + TTL_MARGIN_SECONDS
