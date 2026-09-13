"""Normalize operator-facing queue and decision-trace context."""

from typing import Any


def clean_text_list(value: Any, *, limit: int = 8) -> list[str] | None:
    """Return unique, non-empty text values in stable input order."""
    if not isinstance(value, list):
        return None
    cleaned: list[str] = []
    for row in value:
        text = str(row or "").strip()
        if not text or text in cleaned:
            continue
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned or None


def build_operator_queue_context(
    *,
    objective: str | None,
    domain: str | None = None,
    track_type: str | None = None,
    source_scope: str | None = None,
    repo_source_ids: Any = None,
    benchmark_queries: Any = None,
    sandbox_profile_id: str | None = None,
    automation_profile: str | None = None,
    effective_policy: dict[str, Any] | None = None,
    confidence: Any = None,
    readiness: Any = None,
    linked_note_ids: Any = None,
    linked_experiment_plan_ids: Any = None,
    linked_validation_run_ids: Any = None,
    child_job_ids: Any = None,
) -> dict[str, Any]:
    """Build normalized evidence and automation context for an operator."""
    return {
        "domain": _clean_text(domain),
        "objective": _clean_text(objective),
        "track_type": _clean_text(track_type),
        "source_scope": _clean_text(source_scope),
        "repo_source_ids": clean_text_list(repo_source_ids),
        "benchmark_queries": clean_text_list(benchmark_queries),
        "sandbox_profile_id": _clean_text(sandbox_profile_id),
        "automation_profile": _clean_text(automation_profile),
        "effective_policy": (
            dict(effective_policy) if isinstance(effective_policy, dict) else None
        ),
        "confidence": _optional_rounded_float(confidence),
        "readiness": _optional_rounded_float(readiness),
        "linked_note_ids": clean_text_list(linked_note_ids),
        "linked_experiment_plan_ids": clean_text_list(linked_experiment_plan_ids),
        "linked_validation_run_ids": clean_text_list(linked_validation_run_ids),
        "child_job_ids": clean_text_list(child_job_ids),
    }


def _clean_text(value: Any) -> str | None:
    return str(value or "").strip() or None


def _optional_rounded_float(value: Any) -> float | None:
    try:
        return round(float(value), 4) if value is not None else None
    except Exception:
        return None
