"""Tests for structured, bounded operator-intervention history."""

from datetime import datetime

from app.modules.autonomy.application.job_action_interventions import (
    append_operator_intervention,
)

NOW = datetime(2026, 8, 1, 14, 30, 0)


def test_append_operator_intervention_normalizes_structured_row():
    payload = {"execution_strategy": {}}
    metadata = {"reason_code": "execution_failure"}

    row = append_operator_intervention(
        payload,
        action=" restart ",
        actor_user_id=" user-1 ",
        note=" Retry after repair. ",
        job_status_before=" failed ",
        job_status_after=" pending ",
        metadata=metadata,
        now=NOW,
    )

    assert row == {
        "action": "restart",
        "actor_user_id": "user-1",
        "at": NOW.isoformat(),
        "note": "Retry after repair.",
        "job_status_before": "failed",
        "job_status_after": "pending",
        "metadata": metadata,
    }
    assert payload["execution_strategy"]["operator_interventions"] == [row]


def test_append_operator_intervention_replaces_malformed_execution_payload():
    payload = {"execution_strategy": "invalid"}

    row = append_operator_intervention(
        payload,
        action="",
        actor_user_id="",
        metadata={"ignored": False},
        now=NOW,
    )

    assert row["action"] == "unknown"
    assert row["actor_user_id"] is None
    assert row["metadata"] == {"ignored": False}
    assert payload["execution_strategy"]["operator_interventions"] == [row]


def test_append_operator_intervention_enforces_minimum_history_limit():
    existing = [{"action": f"action-{index}"} for index in range(25)]
    payload = {
        "execution_strategy": {
            "operator_interventions": existing,
        }
    }

    append_operator_intervention(
        payload,
        action="restart",
        actor_user_id="user-1",
        max_events=1,
        now=NOW,
    )

    rows = payload["execution_strategy"]["operator_interventions"]
    assert len(rows) == 20
    assert rows[0]["action"] == "action-6"
    assert rows[-1]["action"] == "restart"
