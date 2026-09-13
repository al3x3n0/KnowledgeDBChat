"""Export and analytics HTTP boundaries for autonomous decision traces."""

import csv
import io
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi import status as http_status
from fastapi.responses import Response
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.autonomy_decision_event import AutonomyDecisionEvent
from app.models.user import User
from app.schemas.agent_job import (
    AgentDecisionTraceAnalyticsBucketResponse,
    AgentDecisionTraceAnalyticsResponse,
    AgentDecisionTraceAnalyticsTrendPointResponse,
    AgentDecisionTraceEventResponse,
    AgentDecisionTraceResponse,
)
from app.services.autonomy_event_service import event_to_trace_payload

DecisionTraceQuery = Callable[..., Awaitable[AgentDecisionTraceResponse]]
VisibleUserIdsLoader = Callable[..., Awaitable[set[UUID]]]

_CSV_FIELDS = (
    "event_id",
    "event_time",
    "event_type",
    "source_kind",
    "source_id",
    "source_label",
    "customer",
    "decision_type",
    "reason_code",
    "reason_label",
    "status",
    "severity",
    "actor_mode",
    "summary",
    "operator_note",
    "triage_status",
    "pinned",
    "is_derived",
    "record_origin",
    "scheduler_state",
    "metadata",
    "before_state",
    "after_state",
    "deep_link",
    "team_bucket",
    "due_at",
    "escalation_state",
    "escalation_reason",
    "escalated_at",
)
_CSV_JSON_FIELDS = {
    "scheduler_state",
    "metadata",
    "before_state",
    "after_state",
    "deep_link",
}


@dataclass(frozen=True)
class DecisionTraceReportingApi:
    router: APIRouter
    export_decision_trace: Callable[..., Any]
    get_decision_trace_analytics: Callable[..., Any]
    load_full_decision_trace_events: Callable[..., Any]


def _trace_analytics_bucket_rows(
    counter: Counter[str],
    *,
    limit: int = 5,
) -> list[AgentDecisionTraceAnalyticsBucketResponse]:
    return [
        AgentDecisionTraceAnalyticsBucketResponse(value=value, count=count)
        for value, count in sorted(
            counter.items(),
            key=lambda item: (-item[1], item[0]),
        )[:limit]
    ]


def _build_trace_response(
    items: list[AgentDecisionTraceEventResponse],
) -> AgentDecisionTraceResponse:
    field_map = {
        "by_source_kind": "source_kind",
        "by_decision_type": "decision_type",
        "by_status": "status",
        "by_customer": "customer",
        "by_severity": "severity",
        "by_actor_mode": "actor_mode",
        "by_triage_status": "triage_status",
        "by_assignee": "assigned_to_user_id",
        "by_escalation_state": "escalation_state",
    }
    empty_labels = {
        "by_customer": "Unassigned",
        "by_assignee": "unassigned",
        "by_escalation_state": "none",
    }
    counters = {
        response_field: Counter(
            str(getattr(item, item_field) or "").strip()
            or empty_labels.get(response_field, "unknown")
            for item in items
        )
        for response_field, item_field in field_map.items()
    }
    overdue_count = sum(
        1
        for item in items
        if item.due_at
        and str(item.triage_status or "").strip().lower() != "resolved"
        and item.due_at <= datetime.utcnow()
    )
    return AgentDecisionTraceResponse(
        items=items,
        total=len(items),
        limit=len(items),
        offset=0,
        by_source_kind=dict(counters["by_source_kind"]),
        by_decision_type=dict(counters["by_decision_type"]),
        by_status=dict(counters["by_status"]),
        by_customer=dict(counters["by_customer"]),
        by_severity=dict(counters["by_severity"]),
        by_actor_mode=dict(counters["by_actor_mode"]),
        by_triage_status=dict(counters["by_triage_status"]),
        by_assignee=dict(counters["by_assignee"]),
        by_escalation_state=dict(counters["by_escalation_state"]),
        overdue_count=overdue_count,
        has_more=False,
    )


def build_decision_trace_reporting_api(
    *,
    router: APIRouter,
    get_decision_trace: DecisionTraceQuery,
    list_visible_user_ids: VisibleUserIdsLoader,
) -> DecisionTraceReportingApi:
    """Register reporting routes backed by the canonical trace query."""

    async def load_full_decision_trace_events(
        *,
        source_kind: Optional[str],
        decision_type: Optional[str],
        customer: Optional[str],
        status: Optional[str],
        severity: Optional[str],
        actor_mode: Optional[str],
        triage_status: Optional[str],
        assigned_to_user_id: Optional[UUID],
        unassigned_only: bool,
        escalation_state: Optional[str],
        pinned: Optional[bool],
        actionable_only: bool,
        start_at: Optional[datetime],
        end_at: Optional[datetime],
        db: AsyncSession,
        current_user: User,
        page_size: int = 300,
    ) -> list[AgentDecisionTraceEventResponse]:
        offset = 0
        collected: list[AgentDecisionTraceEventResponse] = []
        while True:
            page = await get_decision_trace(
                source_kind=source_kind,
                decision_type=decision_type,
                customer=customer,
                status=status,
                severity=severity,
                actor_mode=actor_mode,
                triage_status=triage_status,
                assigned_to_user_id=assigned_to_user_id,
                unassigned_only=unassigned_only,
                escalation_state=escalation_state,
                pinned=pinned,
                actionable_only=actionable_only,
                start_at=start_at,
                end_at=end_at,
                limit=page_size,
                offset=offset,
                db=db,
                current_user=current_user,
            )
            collected.extend(page.items)
            if not page.has_more or not page.items:
                break
            offset += page.limit
        return collected

    @router.get("/decision-trace/export")
    async def export_decision_trace(
        format: str = Query("json", description="Export format: json or csv"),
        source_kind: Optional[str] = Query(
            None,
            description="Filter by event source kind",
        ),
        decision_type: Optional[str] = Query(
            None,
            description="Filter by normalized decision type",
        ),
        customer: Optional[str] = Query(None, description="Filter by customer"),
        status: Optional[str] = Query(
            None,
            description="Filter by derived event status",
        ),
        severity: Optional[str] = Query(
            None,
            description="Filter by event severity",
        ),
        actor_mode: Optional[str] = Query(
            None,
            description="Filter by actor mode: operator|autonomous",
        ),
        triage_status: Optional[str] = Query(
            None,
            description="Filter by operator triage status",
        ),
        assigned_to_user_id: Optional[UUID] = Query(
            None,
            description="Filter by assignee",
        ),
        unassigned_only: bool = Query(
            False,
            description="Only include unassigned persisted events",
        ),
        escalation_state: Optional[str] = Query(
            None,
            description="Filter by escalation state",
        ),
        pinned: Optional[bool] = Query(
            None,
            description="Filter by pinned state for persisted events",
        ),
        actionable_only: bool = Query(
            False,
            description="Only include persisted actionable events",
        ),
        start_at: Optional[datetime] = Query(
            None,
            description="Only include events at or after this time",
        ),
        end_at: Optional[datetime] = Query(
            None,
            description="Only include events at or before this time",
        ),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        export_format = str(format or "json").strip().lower()
        if export_format not in {"json", "csv"}:
            raise HTTPException(
                status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Unsupported export format",
            )
        items = await load_full_decision_trace_events(
            source_kind=source_kind,
            decision_type=decision_type,
            customer=customer,
            status=status,
            severity=severity,
            actor_mode=actor_mode,
            triage_status=triage_status,
            assigned_to_user_id=assigned_to_user_id,
            unassigned_only=unassigned_only,
            escalation_state=escalation_state,
            pinned=pinned,
            actionable_only=actionable_only,
            start_at=start_at,
            end_at=end_at,
            db=db,
            current_user=current_user,
        )
        if export_format == "json":
            payload = _build_trace_response(items)
            return Response(
                content=json.dumps(
                    payload.model_dump(mode="json"),
                    ensure_ascii=False,
                ),
                media_type="application/json",
            )

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"decision_trace_export_{timestamp}.csv"
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(_CSV_FIELDS))
        writer.writeheader()
        for item in items:
            payload = item.model_dump(mode="json")
            row = {}
            for field in _CSV_FIELDS:
                value = payload.get(field)
                row[field] = (
                    json.dumps(value, ensure_ascii=False)
                    if field in _CSV_JSON_FIELDS
                    else value
                )
            writer.writerow(row)
        return Response(
            content=buffer.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @router.get(
        "/decision-trace/analytics",
        response_model=AgentDecisionTraceAnalyticsResponse,
    )
    async def get_decision_trace_analytics(
        source_kind: Optional[str] = Query(
            None,
            description="Filter by event source kind",
        ),
        decision_type: Optional[str] = Query(
            None,
            description="Filter by normalized decision type",
        ),
        customer: Optional[str] = Query(None, description="Filter by customer"),
        status: Optional[str] = Query(
            None,
            description="Filter by derived event status",
        ),
        severity: Optional[str] = Query(
            None,
            description="Filter by event severity",
        ),
        actor_mode: Optional[str] = Query(
            None,
            description="Filter by actor mode: operator|autonomous",
        ),
        triage_status: Optional[str] = Query(
            None,
            description="Filter by operator triage status",
        ),
        assigned_to_user_id: Optional[UUID] = Query(
            None,
            description="Filter by assignee",
        ),
        unassigned_only: bool = Query(
            False,
            description="Only include unassigned persisted events",
        ),
        escalation_state: Optional[str] = Query(
            None,
            description="Filter by escalation state",
        ),
        pinned: Optional[bool] = Query(
            None,
            description="Filter by pinned state for persisted events",
        ),
        actionable_only: bool = Query(
            False,
            description="Only include persisted actionable events",
        ),
        start_at: Optional[datetime] = Query(
            None,
            description="Only include events at or after this time",
        ),
        end_at: Optional[datetime] = Query(
            None,
            description="Only include events at or before this time",
        ),
        days: int = Query(7, ge=1, le=30, description="Trend window in days"),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        del actionable_only
        filters = {
            "source_kind": str(source_kind or "").strip().lower(),
            "decision_type": str(decision_type or "").strip().lower(),
            "customer": str(customer or "").strip().lower(),
            "status": str(status or "").strip().lower(),
            "severity": str(severity or "").strip().lower(),
            "actor_mode": str(actor_mode or "").strip().lower(),
            "triage_status": str(triage_status or "").strip().lower(),
            "escalation_state": str(escalation_state or "").strip().lower(),
        }
        visible_user_ids = await list_visible_user_ids(
            db,
            current_user=current_user,
        )
        if current_user.id not in visible_user_ids:
            visible_user_ids.add(current_user.id)
        query = select(AutonomyDecisionEvent).where(
            AutonomyDecisionEvent.user_id.in_(visible_user_ids)
        )
        field_filters = (
            (AutonomyDecisionEvent.source_kind, filters["source_kind"], False),
            (AutonomyDecisionEvent.decision_type, filters["decision_type"], False),
            (AutonomyDecisionEvent.customer, filters["customer"], True),
            (AutonomyDecisionEvent.status, filters["status"], True),
            (AutonomyDecisionEvent.severity, filters["severity"], True),
            (AutonomyDecisionEvent.actor_mode, filters["actor_mode"], True),
            (AutonomyDecisionEvent.triage_status, filters["triage_status"], True),
            (
                AutonomyDecisionEvent.escalation_state,
                filters["escalation_state"],
                True,
            ),
        )
        for column, value, coalesce_empty in field_filters:
            if value:
                expression = func.coalesce(column, "") if coalesce_empty else column
                query = query.where(func.lower(expression) == value)
        if assigned_to_user_id is not None:
            query = query.where(
                AutonomyDecisionEvent.assigned_to_user_id == assigned_to_user_id
            )
        if unassigned_only:
            query = query.where(AutonomyDecisionEvent.assigned_to_user_id.is_(None))
        if pinned is not None:
            query = query.where(AutonomyDecisionEvent.pinned == bool(pinned))
        if start_at is not None:
            query = query.where(AutonomyDecisionEvent.event_time >= start_at)
        if end_at is not None:
            query = query.where(AutonomyDecisionEvent.event_time <= end_at)
        query = query.order_by(
            AutonomyDecisionEvent.event_time.desc(),
            AutonomyDecisionEvent.created_at.desc(),
        )
        rows = list((await db.execute(query)).scalars().all())

        by_source_kind: Counter[str] = Counter()
        by_triage_status: Counter[str] = Counter()
        by_decision_type: Counter[str] = Counter()
        by_reason_label: Counter[str] = Counter()
        by_queue_reason: Counter[str] = Counter()
        daily_counts: Counter[str] = Counter()
        for row in rows:
            payload = event_to_trace_payload(row)
            by_source_kind[str(payload["source_kind"] or "").strip() or "unknown"] += 1
            by_triage_status[
                str(payload["triage_status"] or "").strip() or "unknown"
            ] += 1
            by_decision_type[
                str(payload["decision_type"] or "").strip() or "unknown"
            ] += 1
            by_reason_label[
                str(payload["reason_label"] or "").strip() or "unknown"
            ] += 1
            scheduler_state = payload.get("scheduler_state")
            queue_reason = (
                str((scheduler_state or {}).get("queue_reason") or "").strip()
                if isinstance(scheduler_state, dict)
                else ""
            )
            by_queue_reason[queue_reason or "unknown"] += 1
            event_day = (
                payload["event_time"].date().isoformat()
                if isinstance(payload.get("event_time"), datetime)
                else datetime.utcnow().date().isoformat()
            )
            daily_counts[event_day] += 1

        today = datetime.utcnow().date()
        daily_trend = [
            AgentDecisionTraceAnalyticsTrendPointResponse(
                day=(today - timedelta(days=days - 1 - index)).isoformat(),
                count=int(
                    daily_counts.get(
                        (today - timedelta(days=days - 1 - index)).isoformat(),
                        0,
                    )
                    or 0
                ),
            )
            for index in range(days)
        ]
        return AgentDecisionTraceAnalyticsResponse(
            window_days=days,
            total=len(rows),
            by_source_kind=dict(
                sorted(
                    by_source_kind.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            by_triage_status=dict(
                sorted(
                    by_triage_status.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            top_decision_types=_trace_analytics_bucket_rows(by_decision_type),
            top_reason_labels=_trace_analytics_bucket_rows(by_reason_label),
            top_queue_reasons=_trace_analytics_bucket_rows(by_queue_reason),
            daily_trend=daily_trend,
        )

    return DecisionTraceReportingApi(
        router=router,
        export_decision_trace=export_decision_trace,
        get_decision_trace_analytics=get_decision_trace_analytics,
        load_full_decision_trace_events=load_full_decision_trace_events,
    )
