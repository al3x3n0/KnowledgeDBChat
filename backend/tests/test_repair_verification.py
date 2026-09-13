"""Tests for repair verification evidence precedence and fallbacks."""

from types import SimpleNamespace

import pytest

from app.modules.autonomy.application.repair_verification import (
    derive_repair_verification_status,
)


def _job(*, status="completed", results=None, execution_log=None):
    return SimpleNamespace(
        status=status,
        results=results if results is not None else {},
        execution_log=execution_log if execution_log is not None else [],
    )


@pytest.mark.parametrize(
    ("recovery_state", "retry_reason", "expected"),
    [
        (
            "verified_fix",
            "Promoted fix passed verification.",
            ("succeeded", "Promoted fix passed verification."),
        ),
        (
            "verification_error",
            None,
            ("failed", "Verification failed."),
        ),
    ],
)
def test_recovery_metadata_has_highest_precedence(
    recovery_state,
    retry_reason,
    expected,
):
    job = _job(
        results={
            "code_patch_execution": {
                "recovery": {
                    "recovery_state": recovery_state,
                    "retry_reason": retry_reason,
                }
            },
            "experiment_run": {"runs": [{"ok": False}]},
        },
        execution_log=[{"verify_success": False}],
    )

    assert derive_repair_verification_status(job) == expected


def test_latest_explicit_log_event_wins_after_recovery_metadata():
    job = _job(
        results={"experiment_run": {"runs": [{"ok": False}]}},
        execution_log=[
            {"verify_success": False},
            {"message": "unrelated"},
            {"verify_success": True},
        ],
    )

    assert derive_repair_verification_status(job) == (
        "succeeded",
        "Verification succeeded.",
    )


@pytest.mark.parametrize(
    ("runs", "expected"),
    [
        (
            [{"ok": True}, {"ok": True}],
            ("succeeded", "Experiment verification runs succeeded."),
        ),
        (
            [{"ok": True}, {"ok": False}],
            ("failed", "Experiment verification runs failed."),
        ),
    ],
)
def test_experiment_runs_supply_verification_when_logs_are_absent(runs, expected):
    job = _job(results={"experiment_run": {"runs": runs}})

    assert derive_repair_verification_status(job) == expected


@pytest.mark.parametrize(
    ("job_status", "expected"),
    [
        ("running", ("pending", "Verification is still in progress.")),
        (
            "completed",
            (
                "incomplete",
                "Repair completed without explicit verification evidence.",
            ),
        ),
    ],
)
def test_code_execution_without_evidence_uses_lifecycle_fallback(
    job_status,
    expected,
):
    job = _job(
        status=job_status,
        results={"code_patch_execution": {"attempted": True}},
    )

    assert derive_repair_verification_status(job) == expected


def test_job_without_repair_evidence_has_no_verification_status():
    assert derive_repair_verification_status(_job()) == (None, None)
