"""Composed HTTP boundary for coding-swarm aggregate analytics."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Optional
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
    AgentJobSwarmAnalyticsPresetRowResponse,
    AgentJobSwarmAnalyticsResponse,
)

JobVisibility = Callable[[AgentJob, User], bool]
BacklogVisibility = Callable[[CodingBacklogItem, User], bool]
LaunchModeExtractor = Callable[[Optional[dict]], str]
PresetInferrer = Callable[[AgentJob], str]
SwarmSummaryExtractor = Callable[[AgentJob], Optional[dict[str, Any]]]
ConfidenceBucket = Callable[[float], str]
BacklogRouteMode = Callable[[Optional[CodingBacklogItem]], Optional[str]]


@dataclass(frozen=True)
class SwarmAnalyticsApi:
    router: APIRouter
    get_swarm_analytics: Callable[..., Any]


def build_swarm_analytics_api(
    *,
    router: APIRouter,
    presets: dict[str, dict[str, Any]],
    is_job_visible: JobVisibility,
    is_backlog_visible: BacklogVisibility,
    extract_launch_mode: LaunchModeExtractor,
    infer_preset_key: PresetInferrer,
    extract_swarm_summary: SwarmSummaryExtractor,
    confidence_bucket: ConfidenceBucket,
    backlog_route_mode: BacklogRouteMode,
) -> SwarmAnalyticsApi:
    """Register swarm analytics at its static-route precedence point."""

    @router.get(
        "/swarm-analytics",
        response_model=AgentJobSwarmAnalyticsResponse,
    )
    async def get_swarm_analytics(
        source_id: Optional[UUID] = Query(None),
        preset_key: Optional[str] = Query(None),
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
        accumulators = {
            key: {
                "preset_key": key,
                "launch_mode": metadata["launch_mode"],
                "label": metadata["label"],
                "total_runs": 0,
                "confidence_values": [],
                "high_confidence_runs": 0,
                "medium_confidence_runs": 0,
                "low_confidence_runs": 0,
                "auto_promoted_runs": 0,
                "review_needed_runs": 0,
                "tie_breaker_runs": 0,
                "manual_promotion_runs": 0,
                "repair_handoff_runs": 0,
                "backlog_handoff_runs": 0,
                "auto_backlog_handoff_runs": 0,
                "manual_backlog_handoff_runs": 0,
                "backlog_auto_suppressed_runs": 0,
            }
            for key, metadata in presets.items()
        }
        allowed_launch_modes = {
            metadata["launch_mode"] for metadata in presets.values()
        }

        for job in jobs:
            config = job.config if isinstance(job.config, dict) else {}
            if extract_launch_mode(config) not in allowed_launch_modes:
                continue
            if source_id and str(config.get("source_id") or "") != str(source_id):
                continue
            if date_from and job.created_at and job.created_at < date_from:
                continue
            if date_to and job.created_at and job.created_at > date_to:
                continue
            row_preset = infer_preset_key(job)
            if not row_preset or (
                normalized_preset and row_preset != normalized_preset
            ):
                continue

            accumulator = accumulators[row_preset]
            accumulator["total_runs"] += 1
            summary = extract_swarm_summary(job) or {}
            confidence = (
                summary.get("confidence")
                if isinstance(summary.get("confidence"), dict)
                else {}
            )
            overall = float(confidence.get("overall") or 0.0)
            accumulator["confidence_values"].append(overall)
            bucket = confidence_bucket(overall)
            accumulator[f"{bucket}_confidence_runs"] += 1
            review_state = str(summary.get("review_state") or "").strip().lower()
            if review_state == "auto_promoted":
                accumulator["auto_promoted_runs"] += 1
            if bool(summary.get("review_required")) or review_state in {
                "needs_review",
                "insufficient_swarm_consensus",
                "consensus_failed",
            }:
                accumulator["review_needed_runs"] += 1
            if (
                bool(summary.get("tie_breaker_attempted"))
                or str(summary.get("tie_breaker_job_id") or "").strip()
            ):
                accumulator["tie_breaker_runs"] += 1
            if review_state == "manual_promotion":
                accumulator["manual_promotion_runs"] += 1
            if str(summary.get("repair_chain_job_id") or "").strip():
                accumulator["repair_handoff_runs"] += 1
            if str(summary.get("backlog_auto_route_suppressed_reason") or "").strip():
                accumulator["backlog_auto_suppressed_runs"] += 1

        for item in backlog_items:
            if date_from and item.created_at and item.created_at < date_from:
                continue
            if date_to and item.created_at and item.created_at > date_to:
                continue
            if source_id and str(getattr(item, "source_id", "") or "") != str(
                source_id
            ):
                continue
            lineage = item.lineage if isinstance(item.lineage, dict) else {}
            row_preset = (
                str(lineage.get("originating_swarm_preset") or "").strip().lower()
            )
            if row_preset not in accumulators:
                continue
            if normalized_preset and row_preset != normalized_preset:
                continue
            accumulator = accumulators[row_preset]
            accumulator["backlog_handoff_runs"] += 1
            if backlog_route_mode(item) == "auto":
                accumulator["auto_backlog_handoff_runs"] += 1
            else:
                accumulator["manual_backlog_handoff_runs"] += 1

        preset_rows = []
        for key in presets:
            accumulator = accumulators[key]
            total_runs = int(accumulator["total_runs"] or 0)
            confidence_values = [
                float(value)
                for value in accumulator.pop("confidence_values", [])
                if isinstance(value, (int, float))
            ]
            average_confidence = (
                sum(confidence_values) / len(confidence_values)
                if confidence_values
                else None
            )
            promotion_rate = (
                float(accumulator["repair_handoff_runs"]) / total_runs
                if total_runs > 0
                else None
            )
            review_rate = (
                float(accumulator["review_needed_runs"]) / total_runs
                if total_runs > 0
                else None
            )
            tie_breaker_rate = (
                float(accumulator["tie_breaker_runs"]) / total_runs
                if total_runs > 0
                else None
            )
            preset_rows.append(
                AgentJobSwarmAnalyticsPresetRowResponse(
                    preset_key=key,
                    launch_mode=str(accumulator["launch_mode"]),
                    label=str(accumulator["label"]),
                    total_runs=total_runs,
                    avg_confidence=round(average_confidence, 4)
                    if average_confidence is not None
                    else None,
                    high_confidence_runs=int(accumulator["high_confidence_runs"]),
                    medium_confidence_runs=int(accumulator["medium_confidence_runs"]),
                    low_confidence_runs=int(accumulator["low_confidence_runs"]),
                    auto_promoted_runs=int(accumulator["auto_promoted_runs"]),
                    review_needed_runs=int(accumulator["review_needed_runs"]),
                    tie_breaker_runs=int(accumulator["tie_breaker_runs"]),
                    manual_promotion_runs=int(accumulator["manual_promotion_runs"]),
                    repair_handoff_runs=int(accumulator["repair_handoff_runs"]),
                    backlog_handoff_runs=int(accumulator["backlog_handoff_runs"]),
                    auto_backlog_handoff_runs=int(
                        accumulator["auto_backlog_handoff_runs"]
                    ),
                    manual_backlog_handoff_runs=int(
                        accumulator["manual_backlog_handoff_runs"]
                    ),
                    backlog_auto_suppressed_runs=int(
                        accumulator["backlog_auto_suppressed_runs"]
                    ),
                    promotion_rate=round(promotion_rate, 4)
                    if promotion_rate is not None
                    else None,
                    review_rate=round(review_rate, 4)
                    if review_rate is not None
                    else None,
                    tie_breaker_rate=round(tie_breaker_rate, 4)
                    if tie_breaker_rate is not None
                    else None,
                )
            )

        filtered_rows = [
            row
            for row in preset_rows
            if not normalized_preset or row.preset_key == normalized_preset
        ]
        totals = {
            "total_runs": sum(row.total_runs for row in filtered_rows),
            "auto_promoted_runs": sum(row.auto_promoted_runs for row in filtered_rows),
            "review_needed_runs": sum(row.review_needed_runs for row in filtered_rows),
            "tie_breaker_runs": sum(row.tie_breaker_runs for row in filtered_rows),
            "repair_handoff_runs": sum(
                row.repair_handoff_runs for row in filtered_rows
            ),
            "backlog_handoff_runs": sum(
                row.backlog_handoff_runs for row in filtered_rows
            ),
            "auto_backlog_handoff_runs": sum(
                row.auto_backlog_handoff_runs for row in filtered_rows
            ),
            "manual_backlog_handoff_runs": sum(
                row.manual_backlog_handoff_runs for row in filtered_rows
            ),
            "backlog_auto_suppressed_runs": sum(
                row.backlog_auto_suppressed_runs for row in filtered_rows
            ),
        }
        confidence_pool = [
            row.avg_confidence
            for row in filtered_rows
            if row.avg_confidence is not None
        ]
        totals["avg_confidence"] = (
            round(sum(confidence_pool) / len(confidence_pool), 4)
            if confidence_pool
            else None
        )
        return AgentJobSwarmAnalyticsResponse(
            preset_rows=filtered_rows,
            totals=totals,
            filters={
                "source_id": str(source_id) if source_id else None,
                "preset_key": normalized_preset,
                "visibility_scope": normalized_visibility,
                "date_from": date_from.isoformat() if date_from else None,
                "date_to": date_to.isoformat() if date_to else None,
            },
        )

    return SwarmAnalyticsApi(
        router=router,
        get_swarm_analytics=get_swarm_analytics,
    )
