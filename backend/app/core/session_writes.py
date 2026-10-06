"""Whether a session's open transaction has written anything.

``services/job_dispatch`` needs this to decide when a message may be sent. A
worker in another process sees only committed rows, so a message about rows
this transaction wrote must wait for its commit. But "a transaction is open"
is not the same question: any SELECT opens one (autobegin), and a function
that only read has an open transaction nobody may ever commit -- deferring its
message to that commit would lose it.

So the session records writes as they happen: an ORM flush with changes, or an
ORM-enabled INSERT/UPDATE/DELETE (``db.execute(update(Model)...)``, which
never flushes anything), and forgets them when the transaction ends. Raw
``text()`` DML is not seen; nothing that queues work after such a statement
exists, and the dispatcher's fallback is the stalled-job sweep.

Registered on the ``Session`` class, so it covers every session, including the
sync session inside each ``AsyncSession``. Imported by ``core.database`` so it
is in force before any session exists.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import event
from sqlalchemy.orm import Session

_WROTE = "kdbc_transaction_wrote"


def has_uncommitted_writes(session: Any) -> bool:
    """True if ``session``'s transaction has written, or holds unflushed changes."""
    session = getattr(session, "sync_session", session)
    if session.info.get(_WROTE):
        return True
    return bool(session.new or session.dirty or session.deleted)


@event.listens_for(Session, "after_flush")
def _flushed(session: Session, _context: Any) -> None:
    session.info[_WROTE] = True


@event.listens_for(Session, "do_orm_execute")
def _executed(state: Any) -> None:
    if state.is_insert or state.is_update or state.is_delete:
        state.session.info[_WROTE] = True


@event.listens_for(Session, "after_commit")
@event.listens_for(Session, "after_rollback")
def _ended(session: Session) -> None:
    session.info.pop(_WROTE, None)
