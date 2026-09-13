"""The loop has to stop failing, and it cannot count on the database to help.

A run whose iteration fails the same way every time must give up. It did not:
one job produced 60,697 identical errors in eight minutes, every one of them
reporting "iteration 1", pegging a worker, while the job row still said
error_count 0. A cap of five existed and never fired once.

The mechanism is the part worth a test. Both limits the loop relied on -- how
many iterations had run, and how many had failed -- lived on the AgentJob, and
`can_continue` ends with `await self.db.refresh(self.job)`. Refresh re-reads
the row and overwrites the instance, so every increment made during an
iteration that failed before committing was thrown away. `iteration` went back
to 0 and was set to 1 again next time round; `error_count` went back to 0 and
could never climb to five. A rollback anywhere -- including one issued inside a
failing tool, on a session this loop does not own -- does the same thing.

So the limits now live on the adapter, which no session knows about, and are
checked before anything touches the database. These tests drive the REAL
adapter, with a session whose refresh reverts the job exactly as the live one
did.
"""

from datetime import datetime, timedelta

import pytest

from app.agent_core.runtime import AgentRuntimeRunner
from app.services.autonomous_agent_executor import _AutonomousRuntimeAdapter

pytestmark = pytest.mark.unit


class _Job:
    """Only what the loop reads off an AgentJob."""

    def __init__(self, max_iterations: int = 8):
        self.id = "job-under-test"
        self.iteration = 0
        self.error_count = 0
        self.max_iterations = max_iterations
        self.max_tool_calls = 1000
        self.max_llm_calls = 1000
        self.max_runtime_minutes = 60
        self.tool_calls_used = 0
        self.llm_calls_used = 0
        self.status = "running"
        self.config = {}
        self.error = None
        self.last_error_at = None
        self.last_activity_at = None
        self.execution_log = []

    def add_log_entry(self, entry):
        self.execution_log.append(entry)

    def can_continue(self):
        return self.iteration < self.max_iterations


class _RevertingSession:
    """A session whose refresh puts back what the row still says.

    This is not a caricature: `refresh` genuinely overwrites the instance from
    the database, and a run that fails before committing has written nothing.
    """

    def __init__(self, revert: bool = True):
        self.revert = revert
        self.refreshes = 0

    async def refresh(self, obj):
        self.refreshes += 1
        if self.revert:
            obj.iteration = 0
            obj.error_count = 0


def _adapter(job, session, **kwargs):
    return _AutonomousRuntimeAdapter(
        executor=None,
        job=job,
        agent_def=None,
        user_settings=None,
        state={"findings": [], "observations": []},
        db=session,
        start_time=datetime.utcnow(),
        max_runtime=timedelta(minutes=60),
        progress_callback=None,
        **kwargs,
    )


class TestTheLimitsSurviveTheDatabase:
    @pytest.mark.asyncio
    async def test_the_error_cap_fires_when_the_job_is_reverted(self):
        """The live failure, against the real adapter.

        With the counters on the job this never terminated. The run must stop
        at the cap, not near infinity.
        """
        job = _Job()
        adapter = _adapter(job, _RevertingSession())

        seen = 0
        while await adapter.can_continue():
            await adapter.on_iteration_start()
            keep = await adapter.on_iteration_error(RuntimeError("boom"))
            seen += 1
            if not keep:
                break
            assert seen < 50, "the loop rearmed itself; the cap is not holding"

        assert seen == adapter.MAX_ITERATION_ERRORS

    @pytest.mark.asyncio
    async def test_the_iteration_budget_survives_a_revert(self):
        """The other half. A reverted `job.iteration` reads zero every time, so
        the job's own budget check never sees it spent."""
        job = _Job(max_iterations=3)
        adapter = _adapter(job, _RevertingSession())

        started = 0
        while await adapter.can_continue():
            await adapter.on_iteration_start()
            started += 1
            assert started < 50, "the iteration budget rearmed itself"

        assert started == 3

    @pytest.mark.asyncio
    async def test_a_resumed_job_keeps_the_budget_it_has_left(self):
        # A job resuming from a checkpoint has already spent iterations; the
        # budget is what remains, not the whole allowance again.
        job = _Job(max_iterations=5)
        job.iteration = 3
        adapter = _adapter(job, _RevertingSession())

        started = 0
        while await adapter.can_continue():
            await adapter.on_iteration_start()
            started += 1
            assert started < 50

        assert started == 2

    @pytest.mark.asyncio
    async def test_nothing_changes_when_the_session_behaves(self):
        job = _Job(max_iterations=4)
        adapter = _adapter(job, _RevertingSession(revert=False))

        started = 0
        while await adapter.can_continue():
            await adapter.on_iteration_start()
            started += 1
            assert started < 50

        assert started == 4


class TestWhatTheRunReports:
    @pytest.mark.asyncio
    async def test_the_error_says_how_many_and_which(self):
        # "Too many errors" alone left nobody able to tell a run that failed
        # five different ways from one that failed the same way five times.
        job = _Job()
        adapter = _adapter(job, _RevertingSession())
        for _ in range(adapter.MAX_ITERATION_ERRORS):
            keep = await adapter.on_iteration_error(RuntimeError("perishable"))
        assert keep is False
        assert "5" in job.error
        assert "perishable" in job.error

    @pytest.mark.asyncio
    async def test_every_error_is_numbered_in_the_log(self):
        job = _Job()
        adapter = _adapter(job, _RevertingSession())
        await adapter.on_iteration_error(RuntimeError("first"))
        await adapter.on_iteration_error(RuntimeError("second"))
        numbered = [e for e in job.execution_log if e.get("phase") == "error"]
        assert [e["error_number"] for e in numbered] == [1, 2]

    @pytest.mark.asyncio
    async def test_the_job_still_carries_the_counts_for_the_ui(self):
        # Written, just not trusted: the UI and the log read these.
        job = _Job()
        adapter = _adapter(job, _RevertingSession(revert=False))
        await adapter.on_iteration_start()
        await adapter.on_iteration_error(RuntimeError("x"))
        assert job.iteration == 1
        assert job.error_count == 1


class TestTheRunnerHonoursTheStop:
    @pytest.mark.asyncio
    async def test_a_failing_run_ends_rather_than_spinning(self):
        """End to end through the real runner, with the real adapter's limits."""
        job = _Job()
        session = _RevertingSession()
        adapter = _adapter(job, session)

        calls = {"observe": 0}

        class _Runaway(RuntimeError):
            # The runner catches every Exception and hands it to
            # on_iteration_error, so a plain assert inside a phase is swallowed
            # and becomes part of the very loop it was meant to stop. `fatal`
            # is the runner's own escape hatch: it re-raises those.
            fatal = True

        async def observe_phase():
            calls["observe"] += 1
            if calls["observe"] > 50:
                raise _Runaway("the loop did not terminate")
            raise RuntimeError("'ToolSpec' object has no attribute 'perishable'")

        async def build_run_result():
            return {"errors": adapter._errors_seen}

        adapter.observe_phase = observe_phase
        adapter.build_run_result = build_run_result

        result = await AgentRuntimeRunner().run(adapter)
        assert result["errors"] == adapter.MAX_ITERATION_ERRORS
        assert calls["observe"] == adapter.MAX_ITERATION_ERRORS


class TestRefreshDoesNotEraseTheRecord:
    """`can_continue` refreshes the job to notice an external cancellation.

    Refresh also overwrites every other attribute with the row's values, so an
    uncommitted iteration's counts and log entries were discarded on the way
    past. That is why a job with 60,697 errors reported error_count 0 and three
    log entries: not a logging failure, the record was being overwritten.
    """

    @pytest.mark.asyncio
    async def test_the_counts_survive_the_refresh(self):
        job = _Job()
        adapter = _adapter(job, _RevertingSession())
        await adapter.on_iteration_start()
        await adapter.on_iteration_error(RuntimeError("boom"))
        assert job.iteration == 1 and job.error_count == 1

        await adapter.can_continue()  # refreshes, and used to wipe both

        assert job.iteration == 1, "the refresh discarded the iteration count"
        assert job.error_count == 1, "the refresh discarded the error count"

    @pytest.mark.asyncio
    async def test_the_log_entries_survive_the_refresh(self):
        job = _Job()
        adapter = _adapter(job, _RevertingSession())

        # A session that also drops the log, which is what a real refresh of an
        # uncommitted JSON column does.
        async def refresh(obj):
            obj.iteration = 0
            obj.error_count = 0
            obj.execution_log = []

        adapter.db.refresh = refresh

        await adapter.on_iteration_error(RuntimeError("boom"))
        assert len(job.execution_log) == 1
        await adapter.can_continue()
        assert len(job.execution_log) == 1, "the refresh discarded the log entry"

    @pytest.mark.asyncio
    async def test_an_external_cancellation_is_still_noticed(self):
        # The reason the refresh exists at all: preserving our own fields must
        # not blind the loop to someone else stopping the job.
        job = _Job()
        adapter = _adapter(job, _RevertingSession())

        async def refresh(obj):
            obj.status = "cancelled"

        adapter.db.refresh = refresh
        assert await adapter.can_continue() is False
