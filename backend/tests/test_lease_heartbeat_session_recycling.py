"""A timed-out lease renewal must not poison every renewal after it.

Measured live: a `specify` stage died at iteration 25 with "Execution lease
lost at fence 2". The fence had not moved and the stalled-job sweep had
requeued nothing -- the lease had simply expired, because one renewal timed out
and left its session mid-statement in the holder. Every tick after that failed
with "Can't reconnect until invalid transaction is rolled back", and 120s later
the TTL killed a job that was working perfectly.

The cleanup existed. It caught `Exception`, and `asyncio.wait_for` cancels the
coroutine it bounds -- `CancelledError` is a `BaseException`. The one failure
mode the bound exists to create was the one that skipped the cleanup.
"""

import asyncio

import pytest

from app.tasks.agent_job_tasks import renew_lease_on_a_recycled_session

pytestmark = pytest.mark.unit


class _Session:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


def _factory_making(sessions):
    def _factory():
        session = _Session()
        sessions.append(session)
        return session

    return _factory


@pytest.mark.asyncio
class TestTheSessionIsRecycledOnEveryKindOfFailure:
    async def test_a_cancelled_renewal_discards_the_session(self, monkeypatch):
        """The regression. A bounded renewal that times out arrives here as
        CancelledError."""
        sessions = []
        holder = {}

        async def _cancelled(**kwargs):
            raise asyncio.CancelledError()

        monkeypatch.setattr(
            "app.tasks.agent_job_tasks.agent_execution_lease_service.renew",
            _cancelled,
        )

        with pytest.raises(asyncio.CancelledError):
            await renew_lease_on_a_recycled_session(
                session_holder=holder,
                session_factory=_factory_making(sessions),
                lease=object(),
                ttl_seconds=120,
            )

        assert "session" not in holder, (
            "a half-used session left in the holder fails every later tick "
            "until the lease expires"
        )
        assert sessions[0].closed is True

    async def test_an_ordinary_error_still_discards_it(self, monkeypatch):
        sessions = []
        holder = {}

        async def _boom(**kwargs):
            raise RuntimeError("connection reset")

        monkeypatch.setattr(
            "app.tasks.agent_job_tasks.agent_execution_lease_service.renew", _boom
        )

        with pytest.raises(RuntimeError):
            await renew_lease_on_a_recycled_session(
                session_holder=holder,
                session_factory=_factory_making(sessions),
                lease=object(),
                ttl_seconds=120,
            )

        assert "session" not in holder

    async def test_the_next_tick_gets_a_fresh_session(self, monkeypatch):
        """What the discarding is for: recovery, not just tidiness."""
        sessions = []
        holder = {}
        calls = []

        async def _fail_then_succeed(**kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise asyncio.CancelledError()
            return "renewed"

        monkeypatch.setattr(
            "app.tasks.agent_job_tasks.agent_execution_lease_service.renew",
            _fail_then_succeed,
        )
        factory = _factory_making(sessions)

        with pytest.raises(asyncio.CancelledError):
            await renew_lease_on_a_recycled_session(
                session_holder=holder,
                session_factory=factory,
                lease=object(),
                ttl_seconds=120,
            )
        result = await renew_lease_on_a_recycled_session(
            session_holder=holder,
            session_factory=factory,
            lease=object(),
            ttl_seconds=120,
        )

        assert result == "renewed"
        assert len(sessions) == 2, "the second tick must not reuse the poisoned one"

    async def test_a_success_keeps_the_session_warm(self, monkeypatch):
        """The control. Recycling on every tick would defeat the reason the
        session is held at all -- one small UPDATE on a warm connection."""
        sessions = []
        holder = {}

        async def _ok(**kwargs):
            return "renewed"

        monkeypatch.setattr(
            "app.tasks.agent_job_tasks.agent_execution_lease_service.renew", _ok
        )
        factory = _factory_making(sessions)

        for _ in range(3):
            await renew_lease_on_a_recycled_session(
                session_holder=holder,
                session_factory=factory,
                lease=object(),
                ttl_seconds=120,
            )

        assert len(sessions) == 1
        assert holder["session"] is sessions[0]
        assert sessions[0].closed is False
