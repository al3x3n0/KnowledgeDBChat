"""What a failed agent job may and may not do next.

Every case here is one that happened. A chained simulation study hit the
Celery soft time limit at 13 iterations; the task retried, re-running the whole
job from scratch against a record already written to FAILED; the retry held the
only worker slot; and the next job's heartbeat could not run while it did, so
that job lost its lease and failed too. One timeout, three dead runs.
"""

import asyncio

from celery.exceptions import SoftTimeLimitExceeded

from app.services.agent_execution_lease_service import ExecutionLeaseLostError
from app.tasks.agent_job_tasks import is_terminal_task_error


class TestWhatMustNotBeRetried:
    def test_running_out_of_wall_clock_is_not_transient(self):
        """Re-running the same work takes the same time and hits the same
        wall. The old code called this a transient error."""
        assert is_terminal_task_error(SoftTimeLimitExceeded()) is True

    def test_a_lost_lease_is_not_transient(self):
        """Another owner holds the job; retrying only contends again."""
        assert is_terminal_task_error(ExecutionLeaseLostError("gone")) is True

    def test_cancellation_is_not_retried(self):
        assert is_terminal_task_error(asyncio.CancelledError()) is True


class TestWhatMayStillBeRetried:
    def test_a_dropped_connection_is_worth_another_go(self):
        """The retry policy exists for these, and narrowing it must not
        remove the case it was written for."""
        assert is_terminal_task_error(ConnectionError("reset by peer")) is False

    def test_an_unexpected_error_is_still_retried(self):
        assert is_terminal_task_error(ValueError("surprise")) is False
