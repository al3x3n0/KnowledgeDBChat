"""Queueing work for the Celery worker once the rows it reads are committed.

A worker runs in another process and sees only committed rows. Services used to
call ``execute_agent_job_task.delay(...)`` right after ``db.flush()``, so a
worker that picked the message up before the caller committed found no job --
and a caller that rolled back afterwards had already queued a job for a row
that never existed. Delegation, peer review and handoff were each fixed by
hand ("commit before queue"); the coding backlog's orchestrator start, its
repair and apply spawns, the coding runner's spawns and the opportunity
reprioritiser still queued first.

``enqueue`` takes the session instead of trusting the caller's order. If the
session's transaction has written something (``core.session_writes``), the
message is sent when it commits and dropped if it rolls back. Otherwise the
rows are already committed -- or this transaction only read, and may never be
committed by anyone -- and it is sent now, with a broker error reaching the
caller. The test is "has written", not "is open": any SELECT opens a
transaction, and deferring a read-only caller's message to a commit that never
comes would lose it.

Tasks are named by dotted path and resolved when sent, so a service queues
work without importing the task module -- which would pull the worker's whole
import graph into the service layer (``tests/test_layering.py``). This is the
one service module that imports task modules. Sending goes through the task's
own ``.delay``, so a test that patches it still intercepts.

A deferred message lost anyway (nobody commits the session that wrote) leaves
an agent job pending with no task id, which ``check_stalled_agent_jobs``
re-delivers.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Optional

from loguru import logger
from sqlalchemy import event
from sqlalchemy.orm import Session

from app.core.session_writes import has_uncommitted_writes

_PENDING = "kdbc_after_commit"

AGENT_JOB_TASK = "app.tasks.agent_job_tasks.execute_agent_job_task"


def _sync_session(db: Any) -> Any:
    return getattr(db, "sync_session", db)


@event.listens_for(Session, "after_commit")
def _run_pending(session: Session) -> None:
    for callback in session.info.pop(_PENDING, []):
        try:
            callback()
        except Exception as exc:  # a lost message must not undo the commit
            logger.warning(f"Could not queue work after commit: {exc}")


@event.listens_for(Session, "after_rollback")
def _drop_pending(session: Session) -> None:
    session.info.pop(_PENDING, None)


def after_commit(db: Any, callback: Callable[[], Any]) -> Any:
    """Run ``callback`` once what ``db``'s transaction wrote is committed.

    Deferred to the commit if the transaction has written, and dropped if it
    rolls back. Otherwise it runs now and its result -- or its error, such as
    a broker that cannot be reached -- goes to the caller. A deferred callback
    runs inside the commit, where an error would reach nobody useful, so it is
    logged instead, and the return value is None.
    """
    session = _sync_session(db)
    if not has_uncommitted_writes(session):
        return callback()
    session.info.setdefault(_PENDING, []).append(callback)
    return None


def resolve_task(path: str) -> Any:
    """The Celery task at dotted ``path`` (module path, then the task's name)."""
    module, _, name = path.rpartition(".")
    return getattr(import_module(module), name)


def _sender(
    task: str, args: tuple, kwargs: dict, options: Optional[dict]
) -> Callable[[], Any]:
    resolve_task(task)  # a misspelt task fails here, not inside a commit

    def _send() -> Any:
        celery_task = resolve_task(task)
        if options:
            return celery_task.apply_async(args=args, kwargs=kwargs, **options)
        return celery_task.delay(*args, **kwargs)

    return _send


def enqueue(
    db: Any,
    task: str,
    *args: Any,
    _options: Optional[dict] = None,
    **kwargs: Any,
) -> Optional[Any]:
    """Queue ``task`` with these arguments once ``db``'s writes are committed.

    Returns the ``AsyncResult`` when it was sent now, None when deferred. A
    caller that needs the task id must commit first, so that it is sent now.
    ``_options`` go to ``apply_async`` (time limits, queue).
    """
    return after_commit(db, _sender(task, args, kwargs, _options))


def send_now(
    task: str, *args: Any, _options: Optional[dict] = None, **kwargs: Any
) -> Any:
    """Send ``task`` at once, for a caller holding no session to wait on."""
    return _sender(task, args, kwargs, _options)()


def enqueue_agent_job(db: Any, job_id: Any, user_id: Any) -> Optional[Any]:
    """Queue an agent job to run once the session holding its row commits."""
    return enqueue(db, AGENT_JOB_TASK, str(job_id), str(user_id))


__all__ = [
    "AGENT_JOB_TASK",
    "after_commit",
    "enqueue",
    "enqueue_agent_job",
    "resolve_task",
    "send_now",
]
