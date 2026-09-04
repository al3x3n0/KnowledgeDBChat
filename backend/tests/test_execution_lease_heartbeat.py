"""The heartbeat that keeps a running job's lease alive.

A job died mid-run with "Execution lease lost at fence 3" -- two takeovers,
discovered only when it tried to commit. The lease is heartbeated by a
background task for exactly as long as the run, so on paper this cannot
happen.

It could, because the renewal was unguarded. Any failure with nothing to do
with ownership -- a saturated pool, a dropped connection, a database restart --
raised out of the coroutine and ended the heartbeat. `asyncio.create_task`
holds that exception until someone retrieves it, and the only retrieval was in
a `finally` clause that ran after the job finished, catching `CancelledError`
alone. So renewals stopped in silence: no log, no signal, no failure. The lease
lapsed, the stalled-job sweep declared the job an orphan, another worker picked
it up, and the original discovered the theft at commit time -- with the blame
landing on the job rather than on a heartbeat that had quietly died.

Two behaviours are load-bearing and pull in opposite directions: a transient
blip must NOT surrender a running job, because that throws away the work it has
done; and a lease that genuinely cannot be renewed must stop the run promptly,
because another worker is entitled to take it. The dividing line is the TTL.
"""

import asyncio

import pytest

from app.tasks.agent_job_tasks import run_execution_lease_heartbeat

pytestmark = pytest.mark.unit


async def _beat(renew, *, ttl_seconds=120, interval=0.01, stop=None, run_for=None):
    """Drive the REAL heartbeat loop and report what it did."""
    stop = stop or asyncio.Event()
    lease_lost = asyncio.Event()
    task = asyncio.create_task(
        run_execution_lease_heartbeat(
            job_id="job-under-test",
            fence=1,
            renew=renew,
            interval=interval,
            ttl_seconds=ttl_seconds,
            stop=stop,
            lease_lost=lease_lost,
        )
    )
    if run_for is not None:
        await asyncio.sleep(run_for)
    return task, stop, lease_lost


class TestABlipDoesNotSurrenderTheJob:
    @pytest.mark.asyncio
    async def test_a_failing_renewal_is_retried_not_fatal(self):
        calls = {"n": 0, "ok": 0}

        async def renew():
            calls["n"] += 1
            if calls["n"] <= 3:
                raise RuntimeError("connection reset by peer")
            calls["ok"] += 1
            return object()

        task, stop, lease_lost = await _beat(renew, run_for=0.15)
        stop.set()
        await asyncio.wait_for(task, timeout=2)

        assert calls["n"] > 3, "it stopped trying after the first failure"
        assert calls["ok"] >= 1, "it must recover once the database returns"
        assert not lease_lost.is_set(), "a transient blip surrendered a running job"

    @pytest.mark.asyncio
    async def test_the_heartbeat_survives_its_own_exception(self):
        # The defect itself: before the guard, the first raise ended the
        # coroutine and every later renewal simply never happened -- silently,
        # because create_task holds the exception until someone retrieves it.
        async def renew():
            raise RuntimeError("pool exhausted")

        task, stop, _ = await _beat(renew, ttl_seconds=9999, run_for=0.1)
        assert not task.done(), "the heartbeat died on the first failure"
        stop.set()
        await asyncio.wait_for(task, timeout=2)

    @pytest.mark.asyncio
    async def test_cancellation_is_not_mistaken_for_a_failed_renewal(self):
        # Cancellation must propagate, not be counted as a blip and retried.
        async def renew():
            await asyncio.sleep(10)

        task, _stop, lease_lost = await _beat(renew, run_for=0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not lease_lost.is_set()


class TestALostLeaseStopsTheRunPromptly:
    @pytest.mark.asyncio
    async def test_renewals_failing_past_the_ttl_give_up(self):
        # Past the TTL the lease has certainly expired and another worker may
        # already hold it. Continuing would put two workers on one job.
        async def renew():
            raise RuntimeError("database is down")

        task, _stop, lease_lost = await _beat(renew, ttl_seconds=0)
        await asyncio.wait_for(task, timeout=2)
        assert lease_lost.is_set()

    @pytest.mark.asyncio
    async def test_a_stolen_lease_is_noticed_at_once(self):
        # renew() returning None means the fence moved: someone else owns it.
        # That is not a blip and gets no retries.
        calls = {"n": 0}

        async def renew():
            calls["n"] += 1
            return None

        task, _stop, lease_lost = await _beat(renew)
        await asyncio.wait_for(task, timeout=2)
        assert lease_lost.is_set()
        assert calls["n"] == 1, "a stolen lease must not be retried"

    @pytest.mark.asyncio
    async def test_stopping_the_run_ends_the_heartbeat(self):
        async def renew():
            return object()

        task, stop, lease_lost = await _beat(renew, run_for=0.05)
        stop.set()
        await asyncio.wait_for(task, timeout=2)
        assert not lease_lost.is_set()


class TestTheTaskStillReportsADeadHeartbeat:
    def test_the_teardown_does_not_mask_it(self):
        """The one thing a unit test cannot drive: the celery task's cleanup.

        Awaiting the heartbeat while catching only CancelledError lets a
        heartbeat that died of anything else raise from the cleanup path and
        replace whatever the job was reporting.
        """
        from pathlib import Path

        source = Path("app/tasks/agent_job_tasks.py")
        if not source.exists():  # pragma: no cover
            source = (
                Path(__file__).resolve().parents[1]
                / "app"
                / "tasks"
                / "agent_job_tasks.py"
            )
        text = source.read_text()
        teardown = text[text.index("heartbeat_task.cancel()") :][:700]
        assert "except Exception as heartbeat_error" in teardown
