from app.api.endpoints.coding_backlog import _default_decomposition, _normalize_policy


def test_normalize_policy_applies_safe_defaults():
    policy = _normalize_policy({})

    assert policy["max_auto_retries"] == 1
    assert policy["max_files_touched"] == 3
    assert policy["blocked_path_prefixes"] == []
    assert policy["require_experiments_ok"] is True
    assert policy["confidence_threshold"] == 0.55


def test_normalize_policy_clamps_threshold_and_filters_prefixes():
    policy = _normalize_policy(
        {
            "max_auto_retries": 2,
            "max_files_touched": 5,
            "blocked_path_prefixes": ["frontend/src", "", " backend/secret "],
            "require_experiments_ok": False,
            "confidence_threshold": 2,
        }
    )

    assert policy["max_auto_retries"] == 2
    assert policy["max_files_touched"] == 5
    assert policy["blocked_path_prefixes"] == ["frontend/src", "backend/secret"]
    assert policy["require_experiments_ok"] is False
    assert policy["confidence_threshold"] == 1.0


def test_default_decomposition_initializes_portfolio_tracking():
    decomposition = _default_decomposition()

    assert decomposition["strategy"] == "portfolio_goal"
    assert decomposition["planned_slices"] == []
    assert decomposition["completed_slices"] == []
    assert decomposition["failed_slices"] == []
    assert decomposition["promotion_decisions"] == []
    assert decomposition["backlog_timeline"] == []
    assert decomposition["lineage_summary"] == {
        "repair_job_count": 0,
        "apply_job_count": 0,
        "patch_pr_count": 0,
        "proposal_count": 0,
        "operator_action_count": 0,
    }
    assert decomposition["active_slice_id"] is None
    assert decomposition["portfolio_progress"] == {
        "total_slices": 0,
        "pending_slices": 0,
        "completed_slices": 0,
        "failed_slices": 0,
        "auto_applied_slices": 0,
        "proposal_only_slices": 0,
    }
