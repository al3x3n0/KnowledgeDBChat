"""Tests for operator queue and decision-trace context normalization."""

from app.modules.autonomy.application.operator_queue_context import (
    build_operator_queue_context,
    clean_text_list,
)


def test_clean_text_list_deduplicates_preserves_order_and_limits_rows():
    assert clean_text_list(
        [" repo-1 ", "", None, "repo-1", "repo-2", "repo-3"],
        limit=2,
    ) == ["repo-1", "repo-2"]


def test_clean_text_list_rejects_non_lists_and_empty_lists():
    assert clean_text_list(("repo-1",)) is None
    assert clean_text_list([" ", None]) is None


def test_build_operator_queue_context_normalizes_all_supported_fields():
    policy = {"mode": "auto_launch_safe"}

    context = build_operator_queue_context(
        objective=" Track compiler regressions ",
        domain=" Compilers ",
        track_type=" benchmark ",
        source_scope=" owned ",
        repo_source_ids=[" repo-1 ", "repo-1", "repo-2"],
        benchmark_queries=[" llvm ", "gcc"],
        sandbox_profile_id=" sandbox-1 ",
        automation_profile=" guarded ",
        effective_policy=policy,
        confidence="0.876543",
        readiness=0.712345,
        linked_note_ids=["note-1"],
        linked_experiment_plan_ids=["plan-1"],
        linked_validation_run_ids=["validation-1"],
        child_job_ids=["job-1"],
    )

    assert context == {
        "domain": "Compilers",
        "objective": "Track compiler regressions",
        "track_type": "benchmark",
        "source_scope": "owned",
        "repo_source_ids": ["repo-1", "repo-2"],
        "benchmark_queries": ["llvm", "gcc"],
        "sandbox_profile_id": "sandbox-1",
        "automation_profile": "guarded",
        "effective_policy": policy,
        "confidence": 0.8765,
        "readiness": 0.7123,
        "linked_note_ids": ["note-1"],
        "linked_experiment_plan_ids": ["plan-1"],
        "linked_validation_run_ids": ["validation-1"],
        "child_job_ids": ["job-1"],
    }
    assert context["effective_policy"] is not policy


def test_build_operator_queue_context_discards_invalid_optional_values():
    context = build_operator_queue_context(
        objective=" ",
        effective_policy="manual_only",
        confidence="unknown",
        readiness=None,
        repo_source_ids="repo-1",
    )

    assert context["objective"] is None
    assert context["effective_policy"] is None
    assert context["confidence"] is None
    assert context["readiness"] is None
    assert context["repo_source_ids"] is None
