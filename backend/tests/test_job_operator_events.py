"""Tests for normalized job operator-action decision events."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.modules.autonomy.application.job_operator_events import (
    JobOperatorEventDependencies,
    record_job_operator_event,
)

NOW = datetime(2026, 8, 1, 9, 30, 0)


def _dependencies():
    return JobOperatorEventDependencies(
        record_event=AsyncMock(),
        queue_customer_for_job=Mock(return_value="Acme"),
        reason_label=Mock(return_value="Execution failure"),
        utcnow=lambda: NOW,
    )


@pytest.mark.asyncio
async def test_record_job_operator_event_builds_normalized_trace_payload():
    job = SimpleNamespace(id=uuid4(), name=" Recovery Job ", status="failed")
    current_user = SimpleNamespace(id=uuid4())
    dependencies = _dependencies()
    db = object()
    scheduler_state = {"queue_reason": "execution_failure"}
    metadata = {"reason_code": " execution_failure ", "spawned": True}

    await record_job_operator_event(
        db=db,
        job=job,
        current_user=current_user,
        action=" RESTART ",
        note="Retry after repair.",
        previous_status="failed",
        next_status="running",
        scheduler_state=scheduler_state,
        metadata=metadata,
        summary="Recovery Job: restart",
        deps=dependencies,
    )

    assert dependencies.record_event.await_args.args == (db,)
    payload = dependencies.record_event.await_args.kwargs
    assert payload["user_id"] == current_user.id
    assert payload["event_type"] == "job_operator_action"
    assert payload["event_time"] == NOW
    assert payload["source_kind"] == "job"
    assert payload["source_id"] == str(job.id)
    assert payload["source_label"] == "Recovery Job"
    assert payload["customer"] == "Acme"
    assert payload["decision_type"] == "restart"
    assert payload["reason_code"] == "execution_failure"
    assert payload["reason_label"] == "Execution failure"
    assert payload["status"] == "running"
    assert payload["scheduler_state"] is scheduler_state
    assert payload["before_state"] == {"job_status": "failed"}
    assert payload["after_state"] == {"job_status": "running"}
    assert payload["metadata"] is metadata
    assert payload["deep_link"] == {
        "target_tab": "jobs",
        "job_id": str(job.id),
        "params": {"job": str(job.id)},
        "label": "Open Job",
    }
    dependencies.queue_customer_for_job.assert_called_once_with(job)
    dependencies.reason_label.assert_called_once_with("execution_failure")


@pytest.mark.asyncio
async def test_record_job_operator_event_uses_fallbacks_and_discards_bad_scheduler():
    job = SimpleNamespace(id=None, name="", status="paused")
    dependencies = _dependencies()

    await record_job_operator_event(
        db=object(),
        job=job,
        current_user=SimpleNamespace(id=uuid4()),
        action="",
        note=None,
        previous_status=None,
        next_status=None,
        scheduler_state="invalid",
        metadata=None,
        summary=None,
        deps=dependencies,
    )

    payload = dependencies.record_event.await_args.kwargs
    assert payload["source_id"] is None
    assert payload["source_label"] == "Agent job"
    assert payload["decision_type"] == "operator_intervention"
    assert payload["status"] == "paused"
    assert payload["summary"] == "Agent job: operator action"
    assert payload["scheduler_state"] is None
    assert payload["before_state"] is None
    assert payload["after_state"] is None
    assert payload["deep_link"]["params"] == {}
    assert payload["metadata"] is None
