from app.api.endpoints.agent_jobs import (
    _append_operator_intervention,
    _append_step_event,
    _apply_checkpoint_action_patch,
    _approval_payload_from_results,
    _normalize_checkpoint_action_patch,
    _set_current_plan_step_status,
    _sync_execution_strategy_state,
)


def test_approval_payload_from_results_prefers_execution_pending_then_direct_fallback():
    results_with_execution = {
        "execution_strategy": {
            "approval_checkpoints": {
                "pending": {"iteration": 2, "action": {"tool": "search_documents"}},
            }
        },
        "approval_checkpoint": {"iteration": 1, "action": {"tool": "old_tool"}},
    }
    payload, approval, pending = _approval_payload_from_results(results_with_execution)

    assert payload is results_with_execution
    assert isinstance(approval, dict)
    assert pending is not None
    assert int(pending.get("iteration", 0)) == 2

    direct_only = {
        "approval_checkpoint": {
            "iteration": 7,
            "action": {"tool": "create_document_from_text"},
        },
    }
    _, _, pending_direct = _approval_payload_from_results(direct_only)
    assert pending_direct is not None
    assert int(pending_direct.get("iteration", 0)) == 7


def test_normalize_checkpoint_action_patch_normalizes_scope_keys():
    patch = _normalize_checkpoint_action_patch(
        {
            "tool": "search_documents",
            "purpose": "Refine scope",
            "params": {
                "target_source_id": "scope-1",
                "nested": {"target_source_id": "scope-2"},
            },
        }
    )

    assert patch["tool"] == "search_documents"
    assert patch["purpose"] == "Refine scope"
    assert isinstance(patch.get("params"), dict)
    assert patch["params"]["source_id"] == "scope-1"
    assert "target_source_id" not in patch["params"]
    assert patch["params"]["nested"]["source_id"] == "scope-2"


def test_normalize_checkpoint_action_patch_rejects_invalid_tool_name():
    try:
        _normalize_checkpoint_action_patch({"tool": "bad tool with spaces"})
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "invalid" in str(exc).lower()


def test_apply_checkpoint_action_patch_merges_with_existing_action():
    pending = {
        "action": {
            "tool": "search_documents",
            "purpose": "Old purpose",
            "params": {"query": "alpha", "limit": 5},
        }
    }
    merged = _apply_checkpoint_action_patch(
        pending,
        {"purpose": "New purpose", "params": {"query": "beta", "limit": 8}},
    )

    assert merged["tool"] == "search_documents"
    assert merged["purpose"] == "New purpose"
    assert merged["params"]["query"] == "beta"
    assert int(merged["params"]["limit"]) == 8
    assert isinstance(pending.get("updated_at"), str)


def test_set_current_plan_step_status_marks_and_advances():
    state = {
        "execution_plan": [
            {"step_id": "step_1", "status": "in_progress"},
            {"step_id": "step_2", "status": "pending"},
        ],
        "plan_step_index": 0,
    }

    out = _set_current_plan_step_status(state, status="skipped", advance_next=True)

    assert out["step_id"] == "step_1"
    assert int(out["plan_step_index"]) == 0
    assert state["execution_plan"][0]["status"] == "skipped"
    assert int(state["plan_step_index"]) == 1
    assert state["execution_plan"][1]["status"] == "in_progress"


def test_append_step_event_and_sync_execution_strategy_state():
    state = {"step_events": []}
    _append_step_event(
        state,
        {
            "type": "checkpoint_waiting",
            "plan_step_id": "step_2",
            "plan_step_index": 1,
        },
    )
    assert isinstance(state.get("step_events"), list)
    assert len(state["step_events"]) == 1
    assert state["step_events"][0]["type"] == "checkpoint_waiting"
    assert isinstance(state["step_events"][0].get("at"), str)

    payload = {}
    approval = {"pending": {"iteration": 3}}
    execution = _sync_execution_strategy_state(
        payload, approval_payload=approval, state=state
    )

    assert isinstance(execution, dict)
    assert execution.get("approval_checkpoints") == approval
    assert isinstance(execution.get("step_events"), list)
    assert len(execution.get("step_events") or []) == 1


def test_append_operator_intervention_tracks_structured_action_history():
    payload = {}

    row = _append_operator_intervention(
        payload,
        action="restart",
        actor_user_id="user-1",
        note="Retry after fallback failure",
        job_status_before="failed",
        job_status_after="pending",
        metadata={"new_job_id": None, "launch_mode": "quick_start_claude_backend"},
    )

    execution = payload.get("execution_strategy")
    assert isinstance(execution, dict)
    interventions = execution.get("operator_interventions")
    assert isinstance(interventions, list)
    assert len(interventions) == 1
    assert row["action"] == "restart"
    assert row["actor_user_id"] == "user-1"
    assert row["job_status_before"] == "failed"
    assert row["job_status_after"] == "pending"
    assert row["note"] == "Retry after fallback failure"
    assert row["metadata"]["launch_mode"] == "quick_start_claude_backend"
    assert isinstance(row.get("at"), str)
    assert "outcome_status" not in row
