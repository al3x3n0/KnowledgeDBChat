"""Tests for persisted follow-up queue operator-decision events."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.api.endpoints import agent_jobs
from app.modules.autonomy.application.follow_up_queue_events import (
    FollowUpQueueEventDependencies,
    record_follow_up_queue_decision,
)

NOW = datetime(2026, 7, 31, 12, 30, 0)


async def _record(*, action, scheduler_state):
    recorder = AsyncMock()
    db = object()
    user_id = uuid4()
    await record_follow_up_queue_decision(
        db=db,
        user_id=user_id,
        action=action,
        operator_note="Reviewed by operator.",
        source_kind="research_inbox",
        source_id="inbox-1",
        source_label="Compiler follow-up",
        customer="Acme",
        reason_code="deep_dive_chain",
        reason_label="Deep dive chain",
        scheduler_state=scheduler_state,
        follow_up_launch_status=" launched ",
        deep_link={"path": "/research/inbox-1"},
        metadata={"recommendation_key": "deep_dive_chain"},
        after_state={"follow_up_operator_decision": action},
        deps=FollowUpQueueEventDependencies(
            record_event=recorder,
            utcnow=lambda: NOW,
        ),
    )
    return recorder, db, user_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "decision_type", "verb"),
    [
        (" APPROVE_LAUNCH ", "follow_up_approved", "approved"),
        ("reject_launch", "follow_up_rejected", "rejected"),
    ],
)
async def test_record_follow_up_queue_decision_maps_operator_action(
    action,
    decision_type,
    verb,
):
    recorder, db, user_id = await _record(action=action, scheduler_state=None)

    assert recorder.await_args.args == (db,)
    payload = recorder.await_args.kwargs
    assert payload["user_id"] == user_id
    assert payload["event_type"] == decision_type
    assert payload["decision_type"] == decision_type
    assert payload["event_time"] == NOW
    assert payload["status"] == "launched"
    assert payload["severity"] == "medium"
    assert payload["actor_mode"] == "operator"
    assert payload["summary"] == f"Compiler follow-up: {verb} queued follow-up"


@pytest.mark.asyncio
async def test_record_follow_up_queue_decision_removes_empty_scheduler_values():
    recorder, _, _ = await _record(
        action="approve_launch",
        scheduler_state={
            "queue_reason": "execution_failure",
            "retry_count": 0,
            "last_scheduled_at": "",
            "last_dispatched_at": None,
            "enabled": False,
        },
    )

    assert recorder.await_args.kwargs["scheduler_state"] == {
        "queue_reason": "execution_failure"
    }


@pytest.mark.asyncio
async def test_endpoint_adapter_maps_user_and_discards_legacy_operator_field(
    monkeypatch,
):
    captured = {}

    async def fake_record(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        agent_jobs.follow_up_queue_events,
        "record_follow_up_queue_decision",
        fake_record,
    )
    current_user = SimpleNamespace(id=uuid4())

    await agent_jobs._record_follow_up_queue_decision_event(
        db=object(),
        current_user=current_user,
        action="approve_launch",
        operator_note=None,
        source_kind="research_inbox",
        source_id="inbox-1",
        source_label="Compiler follow-up",
        customer=None,
        reason_code=None,
        reason_label=None,
        scheduler_state=None,
        follow_up_launch_status="launched",
        follow_up_operator_decision="approved_launch",
        deep_link={},
        metadata={},
        after_state={},
    )

    assert captured["user_id"] == current_user.id
    assert "current_user" not in captured
    assert "follow_up_operator_decision" not in captured
    assert isinstance(captured["deps"], FollowUpQueueEventDependencies)
