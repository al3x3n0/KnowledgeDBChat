"""Derive repair verification state from durable execution evidence."""

from typing import Any

from app.models.agent_job import AgentJob, AgentJobStatus

SUCCEEDED_RECOVERY_STATES = frozenset(
    {"verified", "verification_succeeded", "verified_fix"}
)
FAILED_RECOVERY_STATES = frozenset(
    {"verification_failed", "failed", "verification_error"}
)
ACTIVE_JOB_STATUSES = frozenset(
    {
        AgentJobStatus.PENDING.value,
        AgentJobStatus.RUNNING.value,
        AgentJobStatus.PAUSED.value,
    }
)


def derive_repair_verification_status(
    job: AgentJob,
) -> tuple[str | None, str | None]:
    """Return the strongest available repair-verification status and reason."""
    results = job.results if isinstance(job.results, dict) else {}
    code_execution = _mapping(results.get("code_patch_execution"))
    recovery = _mapping(code_execution.get("recovery"))
    recovery_state = _normalized_text(recovery.get("recovery_state"))
    if recovery_state in SUCCEEDED_RECOVERY_STATES:
        return "succeeded", _reason_or_default(
            recovery.get("retry_reason"),
            "Verification succeeded.",
        )
    if recovery_state in FAILED_RECOVERY_STATES:
        return "failed", _reason_or_default(
            recovery.get("retry_reason"),
            "Verification failed.",
        )

    execution_log = job.execution_log if isinstance(job.execution_log, list) else []
    verify_events = [
        entry
        for entry in execution_log
        if isinstance(entry, dict) and entry.get("verify_success") is not None
    ]
    if verify_events:
        if bool(verify_events[-1].get("verify_success")):
            return "succeeded", "Verification succeeded."
        return "failed", "Verification failed."

    experiment_run = _mapping(results.get("experiment_run"))
    runs = (
        experiment_run.get("runs")
        if isinstance(experiment_run.get("runs"), list)
        else []
    )
    if runs:
        normalized_runs = [row for row in runs if isinstance(row, dict)]
        if normalized_runs and all(bool(row.get("ok")) for row in normalized_runs):
            return "succeeded", "Experiment verification runs succeeded."
        if any(not bool(row.get("ok")) for row in normalized_runs):
            return "failed", "Experiment verification runs failed."

    if code_execution:
        if _normalized_text(job.status) in ACTIVE_JOB_STATUSES:
            return "pending", "Verification is still in progress."
        return "incomplete", "Repair completed without explicit verification evidence."
    return None, None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _normalized_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _reason_or_default(value: Any, default: str) -> str:
    return str(value or default).strip()
