"""Tests for extracted agent-job queue field helpers."""

from datetime import datetime
from types import SimpleNamespace

from app.models.agent_job import AgentJobStatus
from app.services.agent_job_queue_helpers import (
    extract_approval_checkpoint,
    extract_launch_mode,
    parse_optional_datetime,
    queue_age_minutes,
    queue_customer_for_job,
    queue_evidence_summary_for_job,
)


def _job(**kw):
    base = dict(
        results={},
        config={},
        status=AgentJobStatus.RUNNING.value,
        current_phase="",
        phase_details="",
        error=None,
    )
    base.update(kw)
    return SimpleNamespace(**base)


class TestParseOptionalDatetime:
    def test_empty_and_garbage_return_none(self):
        assert parse_optional_datetime("") is None
        assert parse_optional_datetime(None) is None
        assert parse_optional_datetime("not a date") is None

    def test_parses_z_suffix_to_naive_local(self):
        parsed = parse_optional_datetime("2026-01-01T12:00:00Z")
        assert parsed is not None
        assert parsed.tzinfo is None

    def test_parses_naive(self):
        assert parse_optional_datetime("2026-01-01T12:00:00") == datetime(2026, 1, 1, 12, 0, 0)


class TestQueueAgeMinutes:
    def test_none_is_zero(self):
        assert queue_age_minutes(None) == 0

    def test_computes_minutes(self):
        created = datetime(2026, 1, 1, 10, 0, 0)
        now = datetime(2026, 1, 1, 10, 45, 0)
        assert queue_age_minutes(created, now=now) == 45

    def test_future_clamped_to_zero(self):
        created = datetime(2026, 1, 1, 11, 0, 0)
        now = datetime(2026, 1, 1, 10, 0, 0)
        assert queue_age_minutes(created, now=now) == 0


class TestExtractLaunchMode:
    def test_non_dict(self):
        assert extract_launch_mode(None) == ""

    def test_lowercases(self):
        assert extract_launch_mode({"launch_mode": "  Manual "}) == "manual"


class TestExtractApprovalCheckpoint:
    def test_none_when_no_checkpoint(self):
        assert extract_approval_checkpoint(_job()) is None

    def test_direct_checkpoint_pending_when_paused(self):
        job = _job(
            status=AgentJobStatus.PAUSED.value,
            results={"approval_checkpoint": {"message": "approve?", "iteration": 3}},
        )
        cp = extract_approval_checkpoint(job)
        assert cp["required"] is True
        assert cp["status"] == "pending"
        assert cp["iteration"] == 3

    def test_stale_when_not_paused(self):
        job = _job(results={"approval_checkpoint": {"message": "x"}})
        assert extract_approval_checkpoint(job)["status"] == "stale"

    def test_reads_from_execution_strategy_pending(self):
        job = _job(
            status=AgentJobStatus.PAUSED.value,
            results={"execution_strategy": {"approval_checkpoints": {"pending": {"message": "y"}}}},
        )
        assert extract_approval_checkpoint(job)["message"] == "y"


class TestQueueCustomerForJob:
    def test_none_when_absent(self):
        assert queue_customer_for_job(_job()) is None

    def test_from_config(self):
        assert queue_customer_for_job(_job(config={"customer": "Acme"})) == "Acme"

    def test_strips_customer_prefix(self):
        assert queue_customer_for_job(_job(config={"customer": "customer: Acme"})) == "Acme"

    def test_from_results_profile(self):
        job = _job(results={"customer_profile": {"name": "Globex"}})
        assert queue_customer_for_job(job) == "Globex"


class TestQueueEvidenceSummary:
    def test_uses_checkpoint_reasons(self):
        job = _job(
            status=AgentJobStatus.PAUSED.value,
            results={"approval_checkpoint": {"reasons": ["risky tool", "high cost"]}},
        )
        assert queue_evidence_summary_for_job(job) == "risky tool; high cost"

    def test_falls_back_to_scheduler_reason(self):
        job = _job(
            results={"execution_strategy": {"scheduler_state": {"queue_reason": "execution_failure"}}}
        )
        summary = queue_evidence_summary_for_job(job)
        assert "Recovery reason" in summary
        assert "Execution failure" in summary

    def test_falls_back_to_error_then_phase(self):
        assert queue_evidence_summary_for_job(_job(error="boom")) == "boom"
        assert queue_evidence_summary_for_job(_job(phase_details="thinking")) == "thinking"

    def test_none_when_nothing(self):
        assert queue_evidence_summary_for_job(_job()) is None


def test_backward_compat_aliases_in_agent_jobs():
    from app.api.endpoints import agent_jobs

    assert agent_jobs._extract_approval_checkpoint is extract_approval_checkpoint
    assert agent_jobs._queue_customer_for_job is queue_customer_for_job
    assert agent_jobs._parse_optional_datetime is parse_optional_datetime
