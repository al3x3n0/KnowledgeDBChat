"""Tests for the extracted scheduler-state reader helpers.

Complements test_agent_job_scheduler_state.py (which tests the writers in
agent_job_tasks) — this covers the reader/label helpers carved out of
api/endpoints/agent_jobs.py into services/agent_job_scheduler_state.py.
"""

from types import SimpleNamespace

from app.services.agent_job_scheduler_state import (
    extract_scheduler_state,
    queue_reason_label,
)


def _job(scheduler_state=None, results=None):
    if results is None:
        results = (
            {"execution_strategy": {"scheduler_state": scheduler_state}}
            if scheduler_state is not None
            else {}
        )
    return SimpleNamespace(results=results)


class TestExtractSchedulerState:
    def test_none_job_returns_none(self):
        assert extract_scheduler_state(None) is None

    def test_no_scheduler_state_returns_none(self):
        assert extract_scheduler_state(_job(results={})) is None
        assert extract_scheduler_state(_job(scheduler_state={})) is None

    def test_extracts_and_normalizes(self):
        state = extract_scheduler_state(
            _job(
                scheduler_state={
                    "last_run_status": "  failed  ",
                    "failure_streak": "3",
                    "backoff_seconds": -5,
                    "queue_reason": " execution_failure ",
                    "last_scheduled_at": "2026-01-01T00:00:00Z",
                }
            )
        )
        assert state["last_run_status"] == "failed"
        assert state["failure_streak"] == 3
        assert state["backoff_seconds"] == 0  # clamped to >= 0
        assert state["queue_reason"] == "execution_failure"
        assert state["last_scheduled_at"] == "2026-01-01T00:00:00Z"

    def test_blank_strings_become_none(self):
        state = extract_scheduler_state(
            _job(scheduler_state={"last_run_status": "   ", "queue_reason": ""})
        )
        assert state["last_run_status"] is None
        assert state["queue_reason"] is None

    def test_non_dict_results_safe(self):
        assert extract_scheduler_state(SimpleNamespace(results="not a dict")) is None


class TestQueueReasonLabel:
    def test_known_codes(self):
        assert queue_reason_label("approval_required") == "Approval required"
        assert queue_reason_label("budget_throttle") == "Autonomy budget review"

    def test_unknown_code_titlecased(self):
        assert queue_reason_label("some_new_reason") == "Some New Reason"

    def test_empty_falls_back(self):
        assert queue_reason_label("") == "Needs review"
        assert queue_reason_label(None) == "Needs review"


def test_backward_compat_aliases_in_agent_jobs():
    """agent_jobs.py must keep the private-name aliases for existing callers."""
    from app.api.endpoints import agent_jobs

    assert agent_jobs._extract_scheduler_state is extract_scheduler_state
    assert agent_jobs._queue_reason_label is queue_reason_label
