"""Composed HTTP boundary for querying autonomous decision traces."""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.autonomy_decision_event import AutonomyDecisionEvent
from app.models.user import User
from app.schemas.agent_job import (
    AgentDecisionTraceEventResponse,
    AgentDecisionTraceResponse,
)
from app.services.autonomy_event_service import (
    apply_decision_trace_escalation,
    event_to_trace_payload,
    maybe_emit_escalation_transition_notification,
)
from app.utils.datetimes import is_past

VisibleUserIdsLoader = Callable[..., Awaitable[set[UUID]]]
TraceEventDecorator = Callable[..., dict[str, Any]]
DerivedEventsLoader = Callable[..., Awaitable[list[AgentDecisionTraceEventResponse]]]


@dataclass(frozen=True)
class DecisionTraceQueryApi:
    router: APIRouter
    get_decision_trace: Callable[..., Any]


def build_decision_trace_query_api(
    *,
    router: APIRouter,
    list_visible_user_ids: VisibleUserIdsLoader,
    decorate_trace_event_payload: TraceEventDecorator,
    load_derived_events: DerivedEventsLoader,
) -> DecisionTraceQueryApi:
    """Register the decision-trace query at its static-route precedence point."""

    @router.get("/decision-trace", response_model=AgentDecisionTraceResponse)
    async def get_decision_trace(
        source_kind: Optional[str] = Query(
            None,
            description="Filter by event source kind",
        ),
        decision_type: Optional[str] = Query(
            None,
            description="Filter by normalized decision type",
        ),
        customer: Optional[str] = Query(
            None,
            description="Filter by customer",
        ),
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
        limit: int = Query(
            100,
            ge=1,
            le=300,
            description="Maximum decision trace items to return",
        ),
        offset: int = Query(0, ge=0, description="Decision trace offset"),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
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
        visible_users = list(
            (await db.execute(select(User).where(User.id.in_(visible_user_ids))))
            .scalars()
            .all()
        )
        user_lookup = {str(user.id): user for user in visible_users}

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
            if not value:
                continue
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
        persisted_rows = list((await db.execute(query)).scalars().all())

        escalation_mutated = False
        for row in persisted_rows:
            previous_escalation_state = (
                str(row.escalation_state or "none").strip().lower() or "none"
            )
            apply_decision_trace_escalation(row)
            if (
                str(row.escalation_state or "none").strip().lower()
                != previous_escalation_state
            ):
                escalation_mutated = True
                await maybe_emit_escalation_transition_notification(
                    db,
                    row,
                    previous_state=previous_escalation_state,
                )
        if escalation_mutated:
            await db.commit()

        events = [
            AgentDecisionTraceEventResponse.model_validate(
                decorate_trace_event_payload(
                    event_to_trace_payload(row),
                    user_lookup=user_lookup,
                    current_user_id=current_user.id,
                )
            )
            for row in persisted_rows
        ]
        persisted_source_kinds = {
            str(row.source_kind or "").strip().lower()
            for row in persisted_rows
            if str(row.source_kind or "").strip()
        }
        need_derived_fallback = (not actionable_only) and (
            not persisted_rows or not filters["source_kind"]
        )
        if (
            filters["source_kind"]
            and filters["source_kind"] not in persisted_source_kinds
        ):
            need_derived_fallback = not actionable_only

        if need_derived_fallback:
            derived_events = await load_derived_events(
                db=db,
                current_user=current_user,
            )
            for item in derived_events:
                normalized_kind = str(item.source_kind or "").strip().lower()
                if filters["source_kind"]:
                    if (
                        normalized_kind != filters["source_kind"]
                        or normalized_kind in persisted_source_kinds
                    ):
                        continue
                elif normalized_kind in persisted_source_kinds:
                    continue
                events.append(
                    item.model_copy(
                        update={
                            "is_derived": True,
                            "record_origin": "derived_fallback",
                        }
                    )
                )

        filtered_items = []
        for item in events:
            if filters["source_kind"] and (
                str(item.source_kind or "").strip().lower() != filters["source_kind"]
            ):
                continue
            if filters["decision_type"] and (
                str(item.decision_type or "").strip().lower()
                != filters["decision_type"]
            ):
                continue
            if filters["customer"] and (
                str(item.customer or "").strip().lower() != filters["customer"]
            ):
                continue
            if filters["status"] and (
                str(item.status or "").strip().lower() != filters["status"]
            ):
                continue
            if filters["severity"] and (
                str(item.severity or "").strip().lower() != filters["severity"]
            ):
                continue
            if filters["actor_mode"] and (
                str(item.actor_mode or "").strip().lower() != filters["actor_mode"]
            ):
                continue
            if filters["triage_status"] and (
                str(item.triage_status or "").strip().lower()
                != filters["triage_status"]
            ):
                continue
            if assigned_to_user_id is not None and (
                str(item.assigned_to_user_id or "").strip() != str(assigned_to_user_id)
            ):
                continue
            if unassigned_only and item.assigned_to_user_id:
                continue
            if filters["escalation_state"] and (
                str(item.escalation_state or "").strip().lower()
                != filters["escalation_state"]
            ):
                continue
            if pinned is not None and bool(item.pinned) != bool(pinned):
                continue
            if actionable_only and bool(item.is_derived):
                continue
            if start_at is not None and item.event_time < start_at:
                continue
            if end_at is not None and item.event_time > end_at:
                continue
            filtered_items.append(item)

        filtered_items.sort(
            key=lambda row: (
                row.event_time.timestamp() if row.event_time else 0.0,
                str(row.event_id or ""),
            ),
            reverse=True,
        )
        total = len(filtered_items)
        counter_fields = {
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
                for item in filtered_items
            )
            for response_field, item_field in counter_fields.items()
        }
        overdue_count = sum(
            1
            for item in filtered_items
            if item.due_at
            and str(item.triage_status or "").strip().lower() != "resolved"
            and is_past(item.due_at)
        )
        items = filtered_items[offset : offset + limit]

        return AgentDecisionTraceResponse(
            items=items,
            total=total,
            limit=limit,
            offset=offset,
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
            has_more=offset + len(items) < total,
        )

    return DecisionTraceQueryApi(
        router=router,
        get_decision_trace=get_decision_trace,
    )
