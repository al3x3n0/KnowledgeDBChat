"""Whether a run has ended, and if not what it is waiting for.

A run is a job and every job sharing its root: a proposed plan's stages, a
stage's retries. A campaign settles a question from its run and an eval
grades a trial from its run, and both must agree on when the run is over --
reading the first job alone settles an item before its plan has begun.

An attempt that has been superseded is not part of the answer. A stage that
ends short of its contract is retried by a new job (`auto_retry_of`), and a
person's restart also adds one; the old attempt stays paused or completed.
Counted, a paused attempt read as "waiting on a person" while its retry was
running.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

from app.models.agent_job import AgentJobStatus
from app.services.agent_stage_auto_retry import RETRY_OF_KEY
from app.utils.datetimes import age, as_aware_utc

TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled", "stopped"}

# A stage that has just completed may not have created the stage after it
# yet: completing and starting the next are two writes. For this long, a
# completed job with follow-up work it has not started is not "finished".
STAGE_HANDOVER_GRACE = timedelta(minutes=5)

WAITING_ON_JOB = "job"
WAITING_ON_APPROVAL = "approval"
WAITING_ON_PERSON = "person"

_WAIT_PRIORITY = {
    "queued": 0,
    "finalizing": 1,
    "running": 2,
    "sandbox": 3,
    "external": 4,
    "paused": 5,
    "review": 6,
    WAITING_ON_PERSON: 7,
    WAITING_ON_APPROVAL: 8,
}

_PAUSED_PHASES = {
    "waiting_sandbox": "sandbox",
    "awaiting_external": "external",
    "awaiting_approval": WAITING_ON_APPROVAL,
    "blocked_needs_input": WAITING_ON_PERSON,
    "awaiting_reconciliation": "review",
    "needs_review": "review",
}


def _config(job: Any) -> Dict[str, Any]:
    config = getattr(job, "config", None)
    return config if isinstance(config, dict) else {}


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _created(job: Any) -> datetime:
    return as_aware_utc(getattr(job, "created_at", None)) or _EPOCH


def final_jobs(run: Sequence[Any]) -> List[Any]:
    """The run without the attempts something later replaced.

    Replaced means a retry names it (`auto_retry_of`), or a newer job runs
    the same pipeline stage (a person's restart, which names nothing).
    """
    retried = {
        str(_config(job).get(RETRY_OF_KEY))
        for job in run
        if _config(job).get(RETRY_OF_KEY)
    }
    newest_per_stage: Dict[str, Any] = {}
    for job in run:
        stage = _config(job).get("pipeline_stage")
        if not stage:
            continue
        held = newest_per_stage.get(stage)
        if held is None or _created(job) >= _created(held):
            newest_per_stage[stage] = job
    return [
        job
        for job in run
        if str(job.id) not in retried
        and (
            not _config(job).get("pipeline_stage")
            or newest_per_stage.get(_config(job)["pipeline_stage"]) is job
        )
    ]


def hands_over(job: Any) -> bool:
    """True while a completed job may still be about to start its successor."""
    chain = job.chain_config if isinstance(job.chain_config, dict) else {}
    if not chain.get("child_jobs") or getattr(job, "chain_triggered", False):
        return False
    if str(job.status or "") != "completed":
        return False
    since = age(job.completed_at)
    return since is not None and since < STAGE_HANDOVER_GRACE


def still_writing_up(job: Any, now: Optional[datetime] = None) -> bool:
    """True while a worker still holds the job, whatever its status says.

    A run is marked completed before its finaliser writes the conclusion,
    the library record and its memories, which can take minutes; an eval
    graded a trial in that gap and found no conclusion. The worker releases
    the lease when it is done, and a dead worker's lease lapses.
    """
    expires = as_aware_utc(getattr(job, "execution_lease_expires_at", None))
    return expires is not None and expires > (now or datetime.now(timezone.utc))


def run_wait_state(run: Sequence[Any]) -> tuple[Optional[str], Any]:
    """Return the most relevant wait and the current attempt that owns it."""
    selected: tuple[Optional[str], Any] = (None, None)
    for job in final_jobs(run):
        status = str(job.status or "")
        phase = str(getattr(job, "current_phase", None) or "")
        if status in TERMINAL_JOB_STATUSES:
            if not (still_writing_up(job) or hands_over(job)):
                continue
            state = "finalizing"
        elif status == AgentJobStatus.PAUSED.value:
            state = _PAUSED_PHASES.get(phase, "paused")
            if state not in ("sandbox", "external") and still_writing_up(job):
                state = "running"
        else:
            state = "queued" if status == AgentJobStatus.PENDING.value else "running"
        if selected[0] is None or _WAIT_PRIORITY[state] >= _WAIT_PRIORITY[selected[0]]:
            selected = (state, job)
    return selected


def waiting_on(run: Sequence[Any]) -> Optional[str]:
    """Return the coarse wait used by the scheduler and evaluations."""
    state, _ = run_wait_state(run)
    return wait_category(state)


def wait_category(state: Optional[str]) -> Optional[str]:
    if state is None or state in (WAITING_ON_APPROVAL, WAITING_ON_PERSON):
        return state
    if state in ("paused", "review"):
        return WAITING_ON_PERSON
    return WAITING_ON_JOB


__all__ = [
    "STAGE_HANDOVER_GRACE",
    "TERMINAL_JOB_STATUSES",
    "WAITING_ON_APPROVAL",
    "WAITING_ON_JOB",
    "WAITING_ON_PERSON",
    "final_jobs",
    "hands_over",
    "run_wait_state",
    "still_writing_up",
    "wait_category",
    "waiting_on",
]
