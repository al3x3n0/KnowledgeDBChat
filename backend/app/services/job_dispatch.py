"""Queueing work for the Celery worker once the rows it reads are committed.

A worker runs in another process and sees only committed rows. Services used to
call ``execute_agent_job_task.delay(...)`` right after ``db.flush()``, so a
worker that picked the message up before the caller committed found no job --
and a caller that rolled back afterwards had already queued a job for a row
that never existed. Delegation, peer review and handoff were each fixed by
hand ("commit before queue"); the coding backlog's orchestrator start still
queued before its commit, from three call sites.

``enqueue_agent_job`` takes the session instead of trusting the caller's order:
with a transaction open, the message is sent when it commits and dropped if it
rolls back; with none open, the row is already committed and it is sent now.
It works whoever commits, which matters for code such as the blocker filer and
the opportunity reprioritiser, whose callers commit later. (Sessions here do
not expire on commit, so reading an id after a commit opens no transaction.)
If a deferred message is lost anyway -- nobody commits that session -- the
job stays pending with no task id, which ``check_stalled_agent_jobs``
re-delivers.

This module is also the one place in ``app.services`` that imports a task
module: services depend on it, and it on the worker, so the service layer no
longer reaches up into ``app.tasks`` (``tests/test_layering.py``).
"""

from __future__ import annotations

from typing import Any, Callable

from loguru import logger
from sqlalchemy import event

_PENDING = "kdbc_after_commit"
_LISTENING = "kdbc_after_commit_listening"


def _sync_session(db: Any) -> Any:
    return getattr(db, "sync_session", db)


def _run_pending_one(callback: Callable[[], None]) -> None:
    try:
        callback()
    except Exception as exc:  # a lost message must not undo the commit
        logger.warning(f"Could not queue work after commit: {exc}")


def _run_pending(session: Any) -> None:
    for callback in session.info.pop(_PENDING, []):
        _run_pending_one(callback)


def _drop_pending(session: Any) -> None:
    session.info.pop(_PENDING, None)


def after_commit(db: Any, callback: Callable[[], None]) -> None:
    """Run ``callback`` when ``db``'s transaction commits; never if it rolls back.

    With no transaction open there is nothing to wait for: it runs now, and an
    error (a broker that cannot be reached) reaches the caller, which may
    record it. A deferred callback runs inside the commit, where raising would
    reach nobody useful, so its error is logged instead.
    """
    session = _sync_session(db)
    if not session.in_transaction():
        callback()
        return
    session.info.setdefault(_PENDING, []).append(callback)
    if not session.info.get(_LISTENING):
        event.listen(session, "after_commit", _run_pending)
        event.listen(session, "after_rollback", _drop_pending)
        session.info[_LISTENING] = True


def enqueue_agent_job(db: Any, job_id: Any, user_id: Any) -> None:
    """Queue an agent job to run once the session holding its row commits."""

    def _send() -> None:
        from app.tasks.agent_job_tasks import execute_agent_job_task

        execute_agent_job_task.delay(str(job_id), str(user_id))

    after_commit(db, _send)


__all__ = ["after_commit", "enqueue_agent_job"]
