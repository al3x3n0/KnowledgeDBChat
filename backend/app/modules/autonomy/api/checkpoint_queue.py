"""Composed HTTP boundary for the autonomous operator checkpoint queue."""

from collections import Counter
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.agent_job import AgentJob
from app.models.domain_research_profile import DomainResearchProfile
from app.models.research_inbox import ResearchInboxItem
from app.models.research_portfolio import ResearchPortfolio
from app.models.user import User
from app.schemas.agent_job import (
    AgentCheckpointQueueItemResponse,
    AgentCheckpointQueueResponse,
)

QueueItemsBuilder = Callable[..., list[AgentCheckpointQueueItemResponse]]
CustomerProfileKey = Callable[[Optional[str]], str]
LearningProfileLoader = Callable[..., Awaitable[dict[str, Any]]]
MonitorSnapshotBuilder = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class CheckpointQueueApi:
    router: APIRouter
    get_checkpoint_queue: Callable[..., Any]


def build_checkpoint_queue_api(
    *,
    router: APIRouter,
    build_queue_items: QueueItemsBuilder,
    customer_profile_key: CustomerProfileKey,
    load_learning_profile: LearningProfileLoader,
    build_monitor_snapshot: MonitorSnapshotBuilder,
) -> CheckpointQueueApi:
    """Register the operator queue at its static-route precedence point."""

    @router.get(
        "/checkpoint-queue",
        response_model=AgentCheckpointQueueResponse,
    )
    async def get_checkpoint_queue(
        item_type: Optional[str] = Query(
            None,
            description="Filter by queue item type",
        ),
        status: Optional[str] = Query(
            None,
            description="Filter by queue item/job status",
        ),
        customer: Optional[str] = Query(
            None,
            description="Filter by customer",
        ),
        job_type: Optional[str] = Query(
            None,
            description="Filter by job type",
        ),
        sla_bucket: Optional[str] = Query(
            None,
            description="Filter by SLA bucket",
        ),
        escalation_level: Optional[str] = Query(
            None,
            description="Filter by escalation level",
        ),
        overdue_only: bool = Query(
            False,
            description="Only include overdue queue items",
        ),
        sort_by: str = Query(
            "priority_score_desc",
            description=(
                "priority_score_desc|sla_desc|age_desc|priority_desc|"
                "created_desc|created_asc"
            ),
        ),
        limit: int = Query(
            100,
            ge=1,
            le=300,
            description="Maximum queue items to return",
        ),
        offset: int = Query(0, ge=0, description="Queue offset"),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        jobs = list(
            (
                await db.execute(
                    select(AgentJob)
                    .options(selectinload(AgentJob.agent_definition))
                    .where(AgentJob.user_id == current_user.id)
                    .order_by(AgentJob.created_at.desc())
                )
            )
            .scalars()
            .all()
        )
        inbox_items = list(
            (
                await db.execute(
                    select(ResearchInboxItem)
                    .where(
                        and_(
                            ResearchInboxItem.user_id == current_user.id,
                            ResearchInboxItem.status == "accepted",
                        )
                    )
                    .order_by(
                        ResearchInboxItem.updated_at.desc(),
                        ResearchInboxItem.discovered_at.desc(),
                    )
                )
            )
            .scalars()
            .all()
        )
        portfolios = list(
            (
                await db.execute(
                    select(ResearchPortfolio)
                    .where(ResearchPortfolio.user_id == current_user.id)
                    .order_by(ResearchPortfolio.updated_at.desc())
                )
            )
            .scalars()
            .all()
        )
        profiles = list(
            (
                await db.execute(
                    select(DomainResearchProfile)
                    .where(DomainResearchProfile.user_id == current_user.id)
                    .order_by(DomainResearchProfile.updated_at.desc())
                )
            )
            .scalars()
            .all()
        )

        learning_profiles = {}
        customers = sorted({str(item.customer or "").strip() for item in inbox_items})
        for customer_key in customers:
            learning_profiles[
                customer_profile_key(customer_key)
            ] = await load_learning_profile(
                db=db,
                user_id=current_user.id,
                customer=customer_key or None,
            )

        monitor_rows = build_monitor_snapshot(
            items=inbox_items,
            jobs_by_id={job.id: job for job in jobs if job.id is not None},
        ).get("monitors", [])
        all_items = build_queue_items(
            jobs,
            inbox_items,
            portfolios,
            profiles,
            learning_profiles=learning_profiles,
            monitor_health_rows=monitor_rows,
        )

        counters = {
            "by_type": Counter(
                str(row.item_type or "").strip() or "unknown" for row in all_items
            ),
            "by_status": Counter(
                str(row.status or "").strip() or "unknown" for row in all_items
            ),
            "by_customer": Counter(
                str(row.customer or "").strip() or "Unassigned" for row in all_items
            ),
            "by_sla_bucket": Counter(
                str(row.sla_bucket or "").strip() or "unknown" for row in all_items
            ),
            "by_escalation_level": Counter(
                str(row.escalation_level or "").strip() or "unknown"
                for row in all_items
            ),
        }
        filters = {
            "item_type": str(item_type or "").strip().lower(),
            "status": str(status or "").strip().lower(),
            "customer": str(customer or "").strip().lower(),
            "job_type": str(job_type or "").strip().lower(),
            "sla_bucket": str(sla_bucket or "").strip().lower(),
            "escalation_level": str(escalation_level or "").strip().lower(),
        }
        filtered_items = []
        for row in all_items:
            if filters["item_type"] and (
                str(row.item_type or "").strip().lower() != filters["item_type"]
            ):
                continue
            if filters["status"] and (
                str(row.status or "").strip().lower() != filters["status"]
            ):
                continue
            if filters["customer"] and (
                str(row.customer or "").strip().lower() != filters["customer"]
            ):
                continue
            if filters["job_type"] and (
                str(row.job_type or "").strip().lower() != filters["job_type"]
            ):
                continue
            if filters["sla_bucket"] and (
                str(row.sla_bucket or "").strip().lower() != filters["sla_bucket"]
            ):
                continue
            if filters["escalation_level"] and (
                str(row.escalation_level or "").strip().lower()
                != filters["escalation_level"]
            ):
                continue
            if overdue_only and not bool(row.is_overdue):
                continue
            filtered_items.append(row)

        sort_mode = str(sort_by or "priority_desc").strip().lower()
        if sort_mode == "created_asc":
            filtered_items.sort(
                key=lambda row: (
                    row.created_at.timestamp() if row.created_at else 0.0,
                    -int(row.priority or 0),
                )
            )
        elif sort_mode == "created_desc":
            filtered_items.sort(
                key=lambda row: (
                    row.created_at.timestamp() if row.created_at else 0.0,
                    int(row.priority or 0),
                ),
                reverse=True,
            )
        elif sort_mode == "age_desc":
            filtered_items.sort(
                key=lambda row: (
                    int(row.age_minutes or 0),
                    float(row.priority_score or 0),
                ),
                reverse=True,
            )
        elif sort_mode == "sla_desc":
            sla_rank = {"overdue": 3, "at_risk": 2, "normal": 1}
            escalation_rank = {"high": 3, "medium": 2, "normal": 1}
            filtered_items.sort(
                key=lambda row: (
                    sla_rank.get(str(row.sla_bucket or ""), 0),
                    escalation_rank.get(
                        str(row.escalation_level or ""),
                        0,
                    ),
                    bool(row.is_overdue),
                    float(row.priority_score or 0),
                    int(row.age_minutes or 0),
                ),
                reverse=True,
            )
        elif sort_mode == "priority_score_desc":
            filtered_items.sort(
                key=lambda row: (
                    float(row.priority_score or 0),
                    bool(row.is_overdue),
                    int(row.age_minutes or 0),
                    row.created_at.timestamp() if row.created_at else 0.0,
                ),
                reverse=True,
            )
        else:
            filtered_items.sort(
                key=lambda row: (
                    int(row.priority or 0),
                    float(row.priority_score or 0),
                    row.created_at.timestamp() if row.created_at else 0.0,
                ),
                reverse=True,
            )

        total = len(filtered_items)
        return AgentCheckpointQueueResponse(
            items=filtered_items[offset : offset + limit],
            total=total,
            limit=limit,
            offset=offset,
            approvals=sum(
                row.item_type == "approval_checkpoint" for row in filtered_items
            ),
            recoveries=sum(row.item_type == "job_recovery" for row in filtered_items),
            follow_ups=sum(
                row.item_type == "follow_up_recommendation" for row in filtered_items
            ),
            policy_reviews=sum(
                row.item_type == "policy_review" for row in filtered_items
            ),
            budget_reviews=sum(
                row.item_type == "budget_review" for row in filtered_items
            ),
            by_type=dict(counters["by_type"]),
            by_status=dict(counters["by_status"]),
            by_customer=dict(counters["by_customer"]),
            by_sla_bucket=dict(counters["by_sla_bucket"]),
            by_escalation_level=dict(counters["by_escalation_level"]),
        )

    return CheckpointQueueApi(
        router=router,
        get_checkpoint_queue=get_checkpoint_queue,
    )
