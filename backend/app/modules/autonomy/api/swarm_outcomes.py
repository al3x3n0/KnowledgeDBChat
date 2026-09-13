"""Composed HTTP boundary for coding-swarm terminal outcome analytics."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.agent_job import AgentJob
from app.models.coding_backlog import CodingBacklogItem
from app.models.user import User
from app.schemas.agent_job import (
    AgentJobSwarmOutcomeAnalyticsResponse,
    AgentJobSwarmOutcomeCaseResponse,
    AgentJobSwarmOutcomePresetRowResponse,
)

JobVisibility = Callable[[AgentJob, User], bool]
BacklogVisibility = Callable[[CodingBacklogItem, User], bool]
LaunchModeExtractor = Callable[[Optional[dict]], str]
PresetInferrer = Callable[[AgentJob], str]
OutcomeDeriver = Callable[..., AgentJobSwarmOutcomeCaseResponse]
SwarmSummaryExtractor = Callable[[AgentJob], Optional[dict[str, Any]]]
DatetimeSortKey = Callable[[Optional[datetime]], float]
IsoFormatter = Callable[[Optional[datetime]], Optional[str]]
CollaborationLookupLoader = Callable[..., Awaitable[dict[str, User]]]


@dataclass(frozen=True)
class SwarmOutcomesApi:
    router: APIRouter
    get_swarm_outcomes: Callable[..., Any]


def build_swarm_outcomes_api(
    *,
    router: APIRouter,
    presets: dict[str, dict[str, Any]],
    is_job_visible: JobVisibility,
    is_backlog_visible: BacklogVisibility,
    extract_launch_mode: LaunchModeExtractor,
    infer_preset_key: PresetInferrer,
    derive_outcome_case: OutcomeDeriver,
    extract_swarm_summary: SwarmSummaryExtractor,
    datetime_sort_key: DatetimeSortKey,
    iso_or_none: IsoFormatter,
    load_collaboration_user_lookup: CollaborationLookupLoader,
) -> SwarmOutcomesApi:
    """Register terminal outcome analytics at its static route position."""

    @router.get(
        "/swarm-outcomes",
        response_model=AgentJobSwarmOutcomeAnalyticsResponse,
    )
    async def get_swarm_outcomes(
        source_id: Optional[UUID] = Query(None),
        preset_key: Optional[str] = Query(None),
        terminal_outcome: Optional[str] = Query(None),
        promotion_mode: Optional[str] = Query(None),
        visibility_scope: str = Query("mine"),
        date_from: Optional[datetime] = Query(None),
        date_to: Optional[datetime] = Query(None),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        normalized_preset = str(preset_key or "").strip().lower() or None
        if normalized_preset and normalized_preset not in presets:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Unknown coding swarm preset",
            )
        normalized_outcome = str(terminal_outcome or "").strip().lower() or None
        normalized_promotion = str(promotion_mode or "").strip().lower() or None
        normalized_visibility = (
            str(visibility_scope or "mine").strip().lower() or "mine"
        )

        jobs = (await db.execute(select(AgentJob))).scalars().all()
        if normalized_visibility == "mine":
            jobs = [job for job in jobs if str(job.user_id) == str(current_user.id)]
        elif normalized_visibility == "shared":
            jobs = [
                job
                for job in jobs
                if str(job.user_id) != str(current_user.id)
                and is_job_visible(job, current_user)
            ]
        else:
            jobs = [job for job in jobs if is_job_visible(job, current_user)]
        backlog_items = [
            item
            for item in (await db.execute(select(CodingBacklogItem))).scalars().all()
            if is_backlog_visible(item, current_user)
        ]

        jobs_by_id = {str(job.id): job for job in jobs}
        backlog_by_swarm_job_id: dict[
            str,
            list[CodingBacklogItem],
        ] = {}
        for item in backlog_items:
            lineage = item.lineage if isinstance(item.lineage, dict) else {}
            swarm_job_id = str(lineage.get("originating_swarm_job_id") or "").strip()
            if swarm_job_id:
                backlog_by_swarm_job_id.setdefault(
                    swarm_job_id,
                    [],
                ).append(item)

        user_lookup = await load_collaboration_user_lookup(
            db,
            current_user=current_user,
        )
        allowed_launch_modes = {
            metadata["launch_mode"] for metadata in presets.values()
        }
        cases = []
        for job in jobs:
            config = job.config if isinstance(job.config, dict) else {}
            if extract_launch_mode(config) not in allowed_launch_modes:
                continue
            if source_id and str(config.get("source_id") or "") != str(source_id):
                continue
            event_at = job.completed_at or job.last_activity_at or job.created_at
            if date_from and event_at and event_at < date_from:
                continue
            if date_to and event_at and event_at > date_to:
                continue
            row_preset = infer_preset_key(job)
            if not row_preset or (
                normalized_preset and row_preset != normalized_preset
            ):
                continue
            case = derive_outcome_case(
                job,
                repair_jobs_by_id=jobs_by_id,
                backlog_by_swarm_job_id=backlog_by_swarm_job_id,
                current_user_id=str(current_user.id),
                user_lookup=user_lookup,
            )
            if normalized_outcome and case.terminal_outcome != normalized_outcome:
                continue
            if normalized_promotion and case.promotion_mode != normalized_promotion:
                continue
            cases.append(case)

        cases.sort(
            key=lambda item: datetime_sort_key(
                item.latest_downstream_at
                or item.repair_handoff_at
                or item.backlog_routed_at
                or item.swarm_completed_at
            ),
            reverse=True,
        )
        accumulators = {
            key: {
                "launch_mode": metadata["launch_mode"],
                "label": metadata["label"],
                "total_swarm_roots": 0,
                "auto_promoted_runs": 0,
                "manual_promoted_runs": 0,
                "tie_breaker_runs": 0,
                "repair_handoff_runs": 0,
                "verified_fix_runs": 0,
                "repair_failed_runs": 0,
                "backlog_routed_runs": 0,
                "auto_backlog_routed_runs": 0,
                "manual_backlog_routed_runs": 0,
                "backlog_auto_suppressed_runs": 0,
                "needs_review_runs": 0,
                "stalled_after_handoff_runs": 0,
                "confidence_values": [],
                "handoff_minutes": [],
            }
            for key, metadata in presets.items()
        }

        for case in cases:
            accumulator = accumulators[case.preset_key]
            accumulator["total_swarm_roots"] += 1
            if case.promotion_mode == "auto":
                accumulator["auto_promoted_runs"] += 1
            elif case.promotion_mode == "manual":
                accumulator["manual_promoted_runs"] += 1
            if case.tie_breaker_attempted:
                accumulator["tie_breaker_runs"] += 1
            if case.repair_job_id:
                accumulator["repair_handoff_runs"] += 1
            if case.terminal_outcome == "verified_fix":
                accumulator["verified_fix_runs"] += 1
            elif case.terminal_outcome == "repair_failed":
                accumulator["repair_failed_runs"] += 1
            elif case.terminal_outcome == "backlog_routed":
                accumulator["backlog_routed_runs"] += 1
                route_key = (
                    "auto_backlog_routed_runs"
                    if case.backlog_route_mode == "auto"
                    else "manual_backlog_routed_runs"
                )
                accumulator[route_key] += 1
            elif case.terminal_outcome == "needs_review":
                accumulator["needs_review_runs"] += 1
            elif case.terminal_outcome == "stalled_after_handoff":
                accumulator["stalled_after_handoff_runs"] += 1

            source_job = jobs_by_id.get(case.swarm_job_id)
            source_summary = (
                extract_swarm_summary(source_job) if source_job is not None else {}
            )
            if str(
                (source_summary or {}).get("backlog_auto_route_suppressed_reason") or ""
            ).strip():
                accumulator["backlog_auto_suppressed_runs"] += 1
            if case.confidence_overall is not None:
                accumulator["confidence_values"].append(float(case.confidence_overall))
            if case.handoff_latency_minutes is not None:
                accumulator["handoff_minutes"].append(
                    float(case.handoff_latency_minutes)
                )

        preset_rows = []
        for key in presets:
            accumulator = accumulators[key]
            confidence_values = [
                float(value)
                for value in accumulator.pop("confidence_values", [])
                if isinstance(value, (int, float))
            ]
            handoff_minutes = [
                float(value)
                for value in accumulator.pop("handoff_minutes", [])
                if isinstance(value, (int, float))
            ]
            average_confidence = (
                sum(confidence_values) / len(confidence_values)
                if confidence_values
                else None
            )
            average_handoff = (
                sum(handoff_minutes) / len(handoff_minutes) if handoff_minutes else None
            )
            preset_rows.append(
                AgentJobSwarmOutcomePresetRowResponse(
                    preset_key=key,
                    launch_mode=str(accumulator["launch_mode"]),
                    label=str(accumulator["label"]),
                    total_swarm_roots=int(accumulator["total_swarm_roots"]),
                    auto_promoted_runs=int(accumulator["auto_promoted_runs"]),
                    manual_promoted_runs=int(accumulator["manual_promoted_runs"]),
                    tie_breaker_runs=int(accumulator["tie_breaker_runs"]),
                    repair_handoff_runs=int(accumulator["repair_handoff_runs"]),
                    verified_fix_runs=int(accumulator["verified_fix_runs"]),
                    repair_failed_runs=int(accumulator["repair_failed_runs"]),
                    backlog_routed_runs=int(accumulator["backlog_routed_runs"]),
                    auto_backlog_routed_runs=int(
                        accumulator["auto_backlog_routed_runs"]
                    ),
                    manual_backlog_routed_runs=int(
                        accumulator["manual_backlog_routed_runs"]
                    ),
                    backlog_auto_suppressed_runs=int(
                        accumulator["backlog_auto_suppressed_runs"]
                    ),
                    needs_review_runs=int(accumulator["needs_review_runs"]),
                    stalled_after_handoff_runs=int(
                        accumulator["stalled_after_handoff_runs"]
                    ),
                    avg_confidence=round(average_confidence, 4)
                    if average_confidence is not None
                    else None,
                    avg_handoff_minutes=round(average_handoff, 2)
                    if average_handoff is not None
                    else None,
                )
            )

        filtered_rows = [
            row
            for row in preset_rows
            if not normalized_preset or row.preset_key == normalized_preset
        ]
        sum_fields = (
            "total_swarm_roots",
            "auto_promoted_runs",
            "manual_promoted_runs",
            "tie_breaker_runs",
            "repair_handoff_runs",
            "verified_fix_runs",
            "repair_failed_runs",
            "backlog_routed_runs",
            "auto_backlog_routed_runs",
            "manual_backlog_routed_runs",
            "backlog_auto_suppressed_runs",
            "needs_review_runs",
            "stalled_after_handoff_runs",
        )
        totals = {
            field: sum(int(getattr(row, field) or 0) for row in filtered_rows)
            for field in sum_fields
        }
        confidence_pool = [
            row.avg_confidence
            for row in filtered_rows
            if row.avg_confidence is not None
        ]
        handoff_pool = [
            row.avg_handoff_minutes
            for row in filtered_rows
            if row.avg_handoff_minutes is not None
        ]
        totals["avg_confidence"] = (
            round(sum(confidence_pool) / len(confidence_pool), 4)
            if confidence_pool
            else None
        )
        totals["avg_handoff_minutes"] = (
            round(sum(handoff_pool) / len(handoff_pool), 2) if handoff_pool else None
        )
        return AgentJobSwarmOutcomeAnalyticsResponse(
            preset_rows=filtered_rows,
            cases=cases[:200],
            totals=totals,
            filters={
                "source_id": str(source_id) if source_id else None,
                "preset_key": normalized_preset,
                "terminal_outcome": normalized_outcome,
                "promotion_mode": normalized_promotion,
                "visibility_scope": normalized_visibility,
                "date_from": iso_or_none(date_from),
                "date_to": iso_or_none(date_to),
            },
        )

    return SwarmOutcomesApi(
        router=router,
        get_swarm_outcomes=get_swarm_outcomes,
    )
