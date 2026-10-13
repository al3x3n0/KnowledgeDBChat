"""Queueing a job from an operator action, once the action is committed.

Every action that sends a job back to a worker (restart, resume, approving a
mid-run checkpoint, a swarm relaunch) changed the row and called the task's
``.delay`` in the same breath, with the commit still to come. A worker in
another process reads only committed rows: restarting a completed pipeline
stage on the live stack, the worker took the message 45 ms later, read the
row as still ``completed``, called it a duplicate delivery and dropped it.
The commit then set the job to ``pending``, where it stayed, with a task id
and nothing coming.

The task is still the one the dependencies carry, so a caller's stand-in
receives the call as before.
"""

from __future__ import annotations

from typing import Any

from app.modules.autonomy.application.job_action_contracts import JobActionDependencies
from app.services import job_dispatch


def queue_job(db: Any, deps: JobActionDependencies, job_id: Any, user_id: Any) -> None:
    """Send the job to a worker when what this action wrote is committed."""

    def _send() -> Any:
        return deps.execute_agent_job_task.delay(str(job_id), str(user_id))

    session = getattr(db, "sync_session", db)
    if not hasattr(session, "info"):
        # Not a session (a caller's stand-in): there is no commit to wait for.
        _send()
        return
    job_dispatch.after_commit(db, _send)
