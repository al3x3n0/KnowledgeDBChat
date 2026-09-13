from app.services.agent_scope_service import (
    merge_chain_step_config,
    normalize_scope_config,
    normalize_scope_keys_deep,
)


def test_normalize_scope_config_promotes_legacy_key_without_mutating_input():
    original = {
        "target_source_id": "legacy-source",
        "nested": {"value": 1},
    }

    normalized = normalize_scope_config(original)

    assert normalized == {
        "source_id": "legacy-source",
        "nested": {"value": 1},
    }
    assert original["target_source_id"] == "legacy-source"


def test_normalize_scope_config_preserves_canonical_source_id():
    normalized = normalize_scope_config(
        {
            "source_id": "canonical-source",
            "target_source_id": "legacy-source",
        }
    )

    assert normalized == {"source_id": "canonical-source"}


def test_normalize_scope_keys_deep_handles_nested_lists():
    normalized = normalize_scope_keys_deep(
        {
            "target_source_id": "root",
            "steps": [
                {"target_source_id": "step"},
                {"config": {"target_source_id": "nested"}},
            ],
        }
    )

    assert normalized == {
        "source_id": "root",
        "steps": [
            {"source_id": "step"},
            {"config": {"source_id": "nested"}},
        ],
    }


def test_merge_chain_step_config_preserves_only_root_default_scope():
    defaults = {
        "target_source_id": "root-default",
        "nested": {
            "source_id": "nested-default",
            "limits": {"timeout": 30, "retries": 1},
        },
    }
    step = {
        "target_source_id": "root-step",
        "nested": {
            "target_source_id": "nested-step",
            "limits": {"timeout": 60},
        },
    }

    merged = merge_chain_step_config(defaults, step)

    assert merged == {
        "source_id": "root-default",
        "nested": {
            "source_id": "nested-step",
            "limits": {"timeout": 60, "retries": 1},
        },
    }


def test_merge_chain_step_config_returns_independent_collections():
    defaults = {"source_id": "root", "values": ["default"]}
    step = {"extra": {"items": ["step"]}}

    merged = merge_chain_step_config(defaults, step)
    merged["values"].append("changed")
    merged["extra"]["items"].append("changed")

    assert defaults["values"] == ["default"]
    assert step["extra"]["items"] == ["step"]
