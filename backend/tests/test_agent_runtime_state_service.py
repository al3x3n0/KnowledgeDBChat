"""Tests for autonomous runtime state bootstrap helpers."""

from app.services.agent_runtime_state_service import initialize_runtime_state


def test_initialize_runtime_state_builds_expected_defaults():
    state = initialize_runtime_state()

    assert state["observations"] == []
    assert state["execution_plan"] == []
    assert state["goal_progress"] == 0
    assert state["goal_contract_last"] == {}
    assert state["decision_parse_metrics"]["parse_success"] == 0
    assert state["coding_workspace_id"] is None
    assert state["document_workspace"] is None
    assert state["delegated_subtask_results"] == {}


def test_initialize_runtime_state_preserves_checkpoint_values_and_backfills_missing_keys():
    checkpoint_state = {
        "goal_progress": 55,
        "last_progress": 21,
        "findings": [{"title": "cached"}],
        "project_profile": {"source_name": "repo"},
    }

    state = initialize_runtime_state(checkpoint_state)

    assert state["goal_progress"] == 55
    assert state["last_progress"] == 21
    assert state["findings"] == [{"title": "cached"}]
    assert state["project_profile"] == {"source_name": "repo"}
    assert state["execution_plan"] == []
    assert state["decision_parse_metrics"]["parse_failure"] == 0
    assert state["coding_command_history"] == []
