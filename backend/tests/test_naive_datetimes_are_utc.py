"""A naive datetime written to the database means UTC, whatever the host's zone.

The application writes ``datetime.utcnow()`` (naive) into ``TIMESTAMP WITH TIME
ZONE`` columns, and asyncpg reads a naive value in the process's local zone. On
a host at UTC+3 a naive 12:00 was stored as 09:00 UTC. ``app.core.database``
pins the process to UTC; on Postgres (the backend-postgres CI job) the
round trip below checks the stored instant itself.
"""

import time
from datetime import datetime, timezone

import pytest

from app.models.agent_job import AgentJob

pytestmark = pytest.mark.unit


def test_the_process_runs_in_utc():
    import app.core.database  # noqa: F401  (pins the zone on import)

    assert time.strftime("%z", time.localtime()) == "+0000"


async def test_a_naive_utc_time_is_stored_as_that_instant(db_session, test_user):
    written = datetime(2026, 1, 1, 12, 0, 0)
    job = AgentJob(
        name="tz", goal="g", job_type="research", user_id=test_user.id, config={}
    )
    job.created_at = written
    db_session.add(job)
    await db_session.commit()

    db_session.expunge_all()
    stored = (await db_session.get(AgentJob, job.id)).created_at
    if stored.tzinfo is None:  # SQLite keeps what it was given
        stored = stored.replace(tzinfo=timezone.utc)
    assert stored == written.replace(tzinfo=timezone.utc)
