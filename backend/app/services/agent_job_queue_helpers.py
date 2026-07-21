"""Pure field-level helpers for agent-job checkpoint/queue serialization.

Extracted from ``api/endpoints/agent_jobs.py``. These are the leaf primitives
the larger queue/trace builders compose; keeping them here makes them
independently testable and lets those builders move out of the endpoint file
later without dragging the whole helper web with them.

All pure — no DB or request coupling.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.agent_job_scheduler_state import (
    extract_scheduler_state,
    queue_reason_label,
)


def parse_optional_datetime(raw: Any) -> Optional[datetime]:
    """Parse an ISO-8601 string to a naive local datetime, or None."""
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            return parsed.astimezone().replace(tzinfo=None)
        return parsed
    except Exception:
        return None


def queue_age_minutes(
    created_at: Optional[datetime], *, now: Optional[datetime] = None
) -> int:
    """Whole minutes elapsed since ``created_at`` (clamped to >= 0)."""
    if created_at is None:
        return 0
    reference = now or datetime.utcnow()
    return max(0, int((reference - created_at).total_seconds() // 60))


def extract_launch_mode(config: Optional[dict]) -> str:
    """Lowercased launch_mode from a job config."""
    if not isinstance(config, dict):
        return ""
    return str(config.get("launch_mode") or "").strip().lower()


def extract_approval_checkpoint(job: AgentJob) -> Optional[dict]:
    """Extract pending approval checkpoint summary for paused jobs."""
    results = job.results if isinstance(job.results, dict) else {}
    direct = results.get("approval_checkpoint") if isinstance(results.get("approval_checkpoint"), dict) else None
    execution = results.get("execution_strategy") if isinstance(results.get("execution_strategy"), dict) else {}
    approval = execution.get("approval_checkpoints") if isinstance(execution.get("approval_checkpoints"), dict) else {}
    pending = approval.get("pending") if isinstance(approval.get("pending"), dict) else None
    data = direct or pending
    if not isinstance(data, dict):
        return None
    return {
        "required": True,
        "status": "pending" if str(job.status or "") == AgentJobStatus.PAUSED.value else "stale",
        "current_phase": str(job.current_phase or ""),
        "message": str(data.get("message") or job.phase_details or "").strip()[:300],
        "iteration": int(data.get("iteration", 0) or 0),
        "reasons": [str(x)[:140] for x in (data.get("reasons") if isinstance(data.get("reasons"), list) else [])[:8]],
        "action": data.get("action") if isinstance(data.get("action"), dict) else {},
        "created_at": data.get("created_at"),
    }


def queue_customer_for_job(job: AgentJob) -> Optional[str]:
    """Best-effort customer label for a job (config or results profile)."""
    cfg = job.config if isinstance(job.config, dict) else {}
    values = [
        cfg.get("customer"),
        cfg.get("customer_context"),
        ((job.results or {}).get("customer_profile") or {}).get("name") if isinstance((job.results or {}).get("customer_profile"), dict) else None,
    ]
    for raw in values:
        text = str(raw or "").strip()
        if text:
            if ":" in text and text.lower().startswith("customer:"):
                text = text.split(":", 1)[1].strip()
            return text[:200]
    return None


def queue_evidence_summary_for_job(job: AgentJob) -> Optional[str]:
    """Short human-readable reason a job is sitting in the review queue."""
    checkpoint = extract_approval_checkpoint(job)
    if checkpoint:
        reasons = checkpoint.get("reasons") if isinstance(checkpoint.get("reasons"), list) else []
        tool = str(((checkpoint.get("action") or {}).get("tool") or "")).strip()
        if reasons:
            return "; ".join(str(x).strip() for x in reasons[:3] if str(x).strip())[:320] or None
        if tool:
            return f"Pending tool: {tool}"[:320]
    scheduler_state = extract_scheduler_state(job) or {}
    reason = str(scheduler_state.get("queue_reason") or "").strip()
    if reason:
        return f"Recovery reason: {queue_reason_label(reason)}"[:320]
    if job.error:
        return str(job.error).strip()[:320]
    if job.phase_details:
        return str(job.phase_details).strip()[:320]
    return None
