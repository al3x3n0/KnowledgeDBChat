"""How many agent jobs one user may have running at once.

A running job holds a worker process for hours, and the number of processes is
the whole capacity of the agents queue. Nothing bounded one user's share of it:
a campaign, a swarm or a loop creating jobs could hold every process while
everyone else's jobs sat pending behind them.

The cap is applied where a worker claims a job, not where a job is created.
Pipelines, chains, swarms and campaigns create jobs on a user's behalf, and
refusing one of those would break the run that asked for it; holding it
pending until one of the user's other jobs ends does not.

"Running" means holding a live execution lease. The status column cannot say
it: a job whose worker died stays ``running`` until the sweep notices, and
counting it would hold a user's queue for work nobody is doing.

The count and the claim are two statements, so on Postgres a lock on the
user's row makes one user's claims take turns (`lock_user_claims`). Without
it, six jobs delivered together each counted none running and all six
started. A job whose worker was killed still counts until its lease lapses.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import func, select

from app.core.config import settings
from app.models.agent_job import AgentJob
from app.models.user import User
from app.services.config_values import safe_int

#: `queue_reason` in a job's scheduler state while it waits on this cap.
QUEUE_REASON = "user_job_cap"

#: How long past its next attempt a waiting job is still presumed to be coming
#: back. After that the redelivery was lost and the sweep may queue it again.
REDELIVERY_GRACE_SECONDS = 300


def limit() -> int:
    """The cap, or 0 when there is none."""
    return max(0, safe_int(getattr(settings, "AGENT_JOBS_MAX_RUNNING_PER_USER", 0)))


def retry_seconds() -> int:
    configured = safe_int(getattr(settings, "AGENT_JOBS_USER_CAP_RETRY_SECONDS", 60))
    return max(5, min(configured or 60, 3600))


async def running_for_user(
    db: Any,
    user_id: Any,
    *,
    excluding: Any = None,
    now: Optional[datetime] = None,
) -> int:
    """Jobs of this user some worker holds right now, other than `excluding`."""
    # Compared by the database, as the lease itself is: the column comes back
    # timezone-aware from Postgres and this value is naive.
    moment = now or datetime.utcnow()
    statement = select(func.count(AgentJob.id)).where(
        AgentJob.user_id == user_id,
        AgentJob.execution_lease_expires_at > moment,
    )
    if excluding is not None:
        statement = statement.where(AgentJob.id != excluding)
    return int((await db.execute(statement)).scalar() or 0)


async def lock_user_claims(db: Any, user_id: Any) -> None:
    """Make this user's claims take turns, until the transaction ends.

    A lock on the user's row that does not block inserts referring to it
    (FOR NO KEY UPDATE), so creating a job for the user is never held up by
    one being claimed.
    """
    await db.execute(
        select(User.id).where(User.id == user_id).with_for_update(key_share=True)
    )


def waiting_state(*, now: datetime, running: int, cap: int, retry_in: int) -> dict:
    """What a held job records about why it is not running."""
    return {
        "queue_reason": QUEUE_REASON,
        "deferred_at": now.isoformat(),
        "deferred_until": (now + timedelta(seconds=retry_in)).isoformat(),
        "user_running_jobs": int(running),
        "user_job_cap": int(cap),
    }


def is_waiting(state: Any, now: datetime) -> bool:
    """True while a held job's next attempt is still expected to arrive."""
    if not isinstance(state, dict) or state.get("queue_reason") != QUEUE_REASON:
        return False
    try:
        until = datetime.fromisoformat(str(state.get("deferred_until") or ""))
    except ValueError:
        return False
    if until.tzinfo is not None:
        until = until.replace(tzinfo=None)
    return until + timedelta(seconds=REDELIVERY_GRACE_SECONDS) > now


__all__ = [
    "QUEUE_REASON",
    "REDELIVERY_GRACE_SECONDS",
    "is_waiting",
    "limit",
    "lock_user_claims",
    "retry_seconds",
    "running_for_user",
    "waiting_state",
]
