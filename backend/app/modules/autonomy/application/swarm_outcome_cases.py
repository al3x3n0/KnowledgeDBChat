"""Project a coding-swarm root and downstream work into a terminal outcome case."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from app.models.agent_job import AgentJob
from app.models.coding_backlog import CodingBacklogItem
from app.models.user import User
from app.schemas.agent_job import (
    AgentJobSwarmOutcomeCaseResponse,
    CollaborationSummaryResponse,
)
from app.services.collaboration_service import build_collaboration_summary


@dataclass(frozen=True)
class SwarmOutcomeCaseDependencies:
    extract_swarm_summary: Callable[..., Any]
    extract_collaboration: Callable[..., dict[str, Any]]
    infer_preset_key: Callable[..., Any]
    extract_launch_mode: Callable[..., Any]
    safe_float: Callable[..., Any]
    derive_verification_status: Callable[..., Any]
    derive_terminal_outcome: Callable[..., Any]
    extract_backlog_route_mode: Callable[..., Any]


def derive_swarm_outcome_case(
    swarm_job: AgentJob,
    *,
    repair_jobs_by_id: dict[str, AgentJob],
    backlog_by_swarm_job_id: dict[str, list[CodingBacklogItem]],
    deps: SwarmOutcomeCaseDependencies,
    current_user_id: str | None = None,
    user_lookup: dict[str, User] | None = None,
) -> AgentJobSwarmOutcomeCaseResponse:
    cfg = swarm_job.config if isinstance(swarm_job.config, dict) else {}
    quick_start = (
        cfg.get("quick_start") if isinstance(cfg.get("quick_start"), dict) else {}
    )
    summary = deps.extract_swarm_summary(swarm_job) or {}
    collaboration = deps.extract_collaboration(swarm_job)
    preset_key = deps.infer_preset_key(swarm_job)
    launch_mode = deps.extract_launch_mode(cfg)
    source_id = str(cfg.get("source_id") or "").strip() or None
    source_label = (
        str(quick_start.get("source_name") or source_id or "").strip() or None
    )
    review_state = str(summary.get("review_state") or "").strip().lower()
    review_reason = (
        str(
            summary.get("review_reason") or summary.get("promotion_reason") or ""
        ).strip()
        or None
    )
    confidence = deps.safe_float(
        (
            summary.get("confidence")
            if isinstance(summary.get("confidence"), dict)
            else {}
        ).get("overall")
    )
    repair_job_id = str(summary.get("repair_chain_job_id") or "").strip()
    repair_job = repair_jobs_by_id.get(repair_job_id) if repair_job_id else None
    verification_status, verification_reason = (
        deps.derive_verification_status(repair_job)
        if repair_job is not None
        else (None, None)
    )
    backlog_items = backlog_by_swarm_job_id.get(str(swarm_job.id), [])
    backlog_item = backlog_items[0] if backlog_items else None
    promotion_mode = _promotion_mode(review_state, summary, repair_job)
    repair_handoff_at = repair_job.created_at if repair_job is not None else None
    backlog_routed_at = backlog_item.created_at if backlog_item is not None else None
    latest_downstream_at = _latest_downstream_at(repair_job, backlog_item)
    handoff_latency = _handoff_latency(swarm_job, repair_handoff_at)
    terminal_outcome, terminal_reason = deps.derive_terminal_outcome(
        review_state=review_state,
        repair_job=repair_job,
        verification_status=verification_status,
        backlog_item=backlog_item,
    )
    return AgentJobSwarmOutcomeCaseResponse(
        swarm_job_id=str(swarm_job.id),
        swarm_job_name=str(swarm_job.name or "").strip() or None,
        preset_key=preset_key,
        launch_mode=launch_mode,
        source_id=source_id,
        source_label=source_label,
        swarm_status=str(swarm_job.status or "").strip() or None,
        swarm_completed_at=(
            swarm_job.completed_at or swarm_job.last_activity_at or swarm_job.created_at
        ),
        review_state=review_state or None,
        review_reason=review_reason,
        owner_user_id=str(collaboration.get("owner_user_id") or swarm_job.user_id),
        assigned_user_id=_text(collaboration.get("assigned_user_id")),
        assigned_at=(
            datetime.fromisoformat(str(collaboration.get("assigned_at")))
            if _text(collaboration.get("assigned_at"))
            else None
        ),
        assigned_by_user_id=_text(collaboration.get("assigned_by_user_id")),
        review_note=_text(collaboration.get("review_note")),
        collaboration_summary=CollaborationSummaryResponse.model_validate(
            build_collaboration_summary(
                owner_user_id=str(
                    collaboration.get("owner_user_id") or swarm_job.user_id
                ),
                visibility=(
                    "shared" if collaboration.get("shared_with_user_ids") else "private"
                ),
                shared_with_user_ids=list(
                    collaboration.get("shared_with_user_ids") or []
                ),
                assigned_user_id=_text(collaboration.get("assigned_user_id")),
                assigned_by_user_id=_text(collaboration.get("assigned_by_user_id")),
                assigned_at=_text(collaboration.get("assigned_at")),
                note=_text(collaboration.get("review_note")),
                current_user_id=current_user_id,
                user_lookup=user_lookup,
            )
        ),
        promotion_mode=promotion_mode,
        confidence_overall=round(confidence, 4) if confidence is not None else None,
        tie_breaker_attempted=bool(
            summary.get("tie_breaker_attempted") or summary.get("tie_breaker_job_id")
        ),
        repair_job_id=str(repair_job.id) if repair_job is not None else None,
        repair_job_name=_text(repair_job.name) if repair_job is not None else None,
        repair_status=_text(repair_job.status) if repair_job is not None else None,
        repair_handoff_at=repair_handoff_at,
        verification_status=verification_status,
        verification_reason=verification_reason,
        backlog_item_id=str(backlog_item.id) if backlog_item is not None else None,
        backlog_title=_text(backlog_item.title) if backlog_item is not None else None,
        backlog_status=_text(backlog_item.status) if backlog_item is not None else None,
        backlog_route_mode=deps.extract_backlog_route_mode(backlog_item),
        backlog_routed_at=backlog_routed_at,
        latest_downstream_at=latest_downstream_at,
        handoff_latency_minutes=handoff_latency,
        terminal_outcome=terminal_outcome,
        terminal_reason=terminal_reason,
    )


def _promotion_mode(
    review_state: str, summary: dict, repair_job: AgentJob | None
) -> str:
    if review_state == "auto_promoted":
        return "auto"
    if review_state == "manual_promotion" or (
        repair_job is not None
        and str(summary.get("promotion_reason") or "")
        .strip()
        .lower()
        .startswith("manually promoted")
    ):
        return "manual"
    return "none"


def _latest_downstream_at(
    repair_job: AgentJob | None,
    backlog_item: CodingBacklogItem | None,
) -> datetime | None:
    candidates = [
        repair_job.completed_at if repair_job is not None else None,
        repair_job.last_activity_at if repair_job is not None else None,
        backlog_item.updated_at if backlog_item is not None else None,
    ]
    return max(
        (value for value in candidates if isinstance(value, datetime)), default=None
    )


def _handoff_latency(
    swarm_job: AgentJob,
    repair_handoff_at: datetime | None,
) -> float | None:
    if not isinstance(repair_handoff_at, datetime):
        return None
    origin = (
        swarm_job.completed_at
        if isinstance(swarm_job.completed_at, datetime)
        else swarm_job.created_at
    )
    if not isinstance(origin, datetime):
        return None
    return max(0.0, round((repair_handoff_at - origin).total_seconds() / 60.0, 2))


def _text(value: Any) -> str | None:
    return str(value or "").strip() or None
