"""Tests for checkpoint-driven execution-plan state mutation."""

from datetime import datetime

from app.modules.autonomy.application.job_action_plan_state import (
    set_current_plan_step_status,
)

NOW = datetime(2026, 8, 2, 10, 0, 0)


def test_plan_state_returns_empty_metadata_without_a_plan():
    assert set_current_plan_step_status(None, status="done", utcnow=lambda: NOW) == {
        "step_id": "",
        "plan_step_index": -1,
    }
    assert set_current_plan_step_status(
        {"execution_plan": "invalid"},
        status="done",
        utcnow=lambda: NOW,
    ) == {"step_id": "", "plan_step_index": -1}


def test_plan_state_clamps_index_and_replaces_malformed_step():
    state = {
        "execution_plan": ["invalid", {"step_id": "step_2", "status": "pending"}],
        "plan_step_index": -8,
    }

    result = set_current_plan_step_status(
        state,
        status=" ",
        utcnow=lambda: NOW,
    )

    assert result == {"step_id": "step_1", "plan_step_index": 0}
    assert state["execution_plan"][0] == {
        "status": "pending",
        "updated_at": NOW.isoformat(),
    }


def test_plan_state_updates_and_advances_to_pending_next_step():
    timestamps = iter(
        [
            datetime(2026, 8, 2, 10, 0, 0),
            datetime(2026, 8, 2, 10, 0, 1),
        ]
    )
    state = {
        "execution_plan": [
            {"step_id": "step_1", "status": "in_progress"},
            {"step_id": "step_2", "status": "pending"},
        ],
        "plan_step_index": 0,
    }

    result = set_current_plan_step_status(
        state,
        status="skipped",
        advance_next=True,
        utcnow=lambda: next(timestamps),
    )

    assert result == {"step_id": "step_1", "plan_step_index": 0}
    assert state["plan_step_index"] == 1
    assert state["execution_plan"][0]["status"] == "skipped"
    assert state["execution_plan"][1]["status"] == "in_progress"
    assert state["execution_plan"][1]["updated_at"].endswith("10:00:01")


def test_plan_state_does_not_reopen_completed_next_step():
    state = {
        "execution_plan": [
            {"step_id": "step_1", "status": "in_progress"},
            {"step_id": "step_2", "status": "done"},
        ],
        "plan_step_index": 0,
    }

    set_current_plan_step_status(
        state,
        status="done",
        advance_next=True,
        utcnow=lambda: NOW,
    )

    assert state["plan_step_index"] == 1
    assert state["execution_plan"][1] == {
        "step_id": "step_2",
        "status": "done",
    }
