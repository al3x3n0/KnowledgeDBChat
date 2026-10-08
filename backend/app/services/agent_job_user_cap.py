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

Soft, like the other shared limits: the count and the claim are two
statements, so two workers claiming for one user in the same instant can both
pass. That overshoots by a job, which costs nothing a hard limit would save.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy import func, select

from app.core.config import settings
from app.models.agent_job import AgentJob
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
    "retry_seconds",
    "running_for_user",
    "waiting_state",
]
