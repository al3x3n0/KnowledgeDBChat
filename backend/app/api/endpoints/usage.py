"""
Usage/analytics endpoints (LLM token usage).
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.llm_usage import LLMUsageEvent
from app.models.agent_definition import AgentDefinition
from app.models.user import User
from app.schemas.common import PaginatedResponse
from app.schemas.usage import (
    LLMUsageEventResponse,
    LLMUsageSummaryResponse,
    LLMUsageSummaryItem,
    LLMRoutingSummaryResponse,
    LLMRoutingSummaryItem,
    LLMRoutingExperimentRecommendationResponse,
    LLMRoutingExperimentVariantStat,
    LLMRoutingExperimentListResponse,
    LLMRoutingExperimentListItem,
)
from app.services.auth_service import get_current_user

router = APIRouter()


def _event_to_response(row: LLMUsageEvent) -> LLMUsageEventResponse:
    return LLMUsageEventResponse(
        id=row.id,
        user_id=row.user_id,
        provider=row.provider,
        model=row.model,
        task_type=row.task_type,
        prompt_tokens=row.prompt_tokens,
        completion_tokens=row.completion_tokens,
        total_tokens=row.total_tokens,
        input_chars=row.input_chars,
        output_chars=row.output_chars,
        latency_ms=row.latency_ms,
        error=row.error,
        extra=row.extra,
        created_at=row.created_at,
    )


@router.get("/llm/events", response_model=PaginatedResponse[LLMUsageEventResponse])
async def list_llm_usage_events(
    provider: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    task_type: Optional[str] = Query(None),
    user_id: Optional[UUID] = Query(None),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List recent LLM usage events.

    Non-admin users may only view their own events.
    """
    if user_id and (not current_user.is_admin()) and user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to view other users' usage")

    effective_user_id = user_id
    if (not current_user.is_admin()) and effective_user_id is None:
        effective_user_id = current_user.id

    query = select(LLMUsageEvent)
    if effective_user_id is not None:
        query = query.where(LLMUsageEvent.user_id == effective_user_id)
    if provider:
        query = query.where(LLMUsageEvent.provider == provider)
    if model:
        query = query.where(LLMUsageEvent.model == model)
    if task_type:
        query = query.where(LLMUsageEvent.task_type == task_type)
    if date_from:
        query = query.where(LLMUsageEvent.created_at >= date_from)
    if date_to:
        query = query.where(LLMUsageEvent.created_at <= date_to)

    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar() or 0

    query = query.order_by(desc(LLMUsageEvent.created_at)).offset((page - 1) * page_size).limit(page_size)
    rows = (await db.execute(query)).scalars().all()

    return PaginatedResponse.create(
        items=[_event_to_response(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/llm/summary", response_model=LLMUsageSummaryResponse)
async def llm_usage_summary(
    provider: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    task_type: Optional[str] = Query(None),
    user_id: Optional[UUID] = Query(None),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Aggregate token usage grouped by provider/model/task_type.

    Non-admin users may only view their own usage.
    """
    if user_id and (not current_user.is_admin()) and user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to view other users' usage")

    effective_user_id = user_id
    if (not current_user.is_admin()) and effective_user_id is None:
        effective_user_id = current_user.id

    stmt = select(
        LLMUsageEvent.provider,
        LLMUsageEvent.model,
        LLMUsageEvent.task_type,
        func.count().label("request_count"),
        func.coalesce(func.sum(LLMUsageEvent.prompt_tokens), 0).label("total_prompt_tokens"),
        func.coalesce(func.sum(LLMUsageEvent.completion_tokens), 0).label("total_completion_tokens"),
        func.coalesce(func.sum(LLMUsageEvent.total_tokens), 0).label("total_tokens"),
        func.avg(LLMUsageEvent.latency_ms).label("avg_latency_ms"),
    ).group_by(LLMUsageEvent.provider, LLMUsageEvent.model, LLMUsageEvent.task_type)

    if effective_user_id is not None:
        stmt = stmt.where(LLMUsageEvent.user_id == effective_user_id)
    if provider:
        stmt = stmt.where(LLMUsageEvent.provider == provider)
    if model:
        stmt = stmt.where(LLMUsageEvent.model == model)
    if task_type:
        stmt = stmt.where(LLMUsageEvent.task_type == task_type)
    if date_from:
        stmt = stmt.where(LLMUsageEvent.created_at >= date_from)
    if date_to:
        stmt = stmt.where(LLMUsageEvent.created_at <= date_to)

    stmt = stmt.order_by(desc(func.coalesce(func.sum(LLMUsageEvent.total_tokens), 0)))
    rows = (await db.execute(stmt)).all()

    items: List[LLMUsageSummaryItem] = []
    for r in rows:
        items.append(
            LLMUsageSummaryItem(
                provider=r.provider,
                model=r.model,
                task_type=r.task_type,
                request_count=int(r.request_count or 0),
                total_prompt_tokens=int(r.total_prompt_tokens or 0),
                total_completion_tokens=int(r.total_completion_tokens or 0),
                total_tokens=int(r.total_tokens or 0),
                avg_latency_ms=float(r.avg_latency_ms) if r.avg_latency_ms is not None else None,
            )
        )

    return LLMUsageSummaryResponse(items=items, date_from=date_from, date_to=date_to)



def _percentile_int(values: List[int], pct: float) -> Optional[int]:
    if not values:
        return None
    vs = sorted([int(v) for v in values if v is not None])
    if not vs:
        return None
    if pct <= 0:
        return int(vs[0])
    if pct >= 100:
        return int(vs[-1])
    # Nearest-rank on 0..n-1
    n = len(vs)
    idx = int(round((pct / 100.0) * (n - 1)))
    idx = max(0, min(idx, n - 1))
    return int(vs[idx])


@router.get("/llm/routing/summary", response_model=LLMRoutingSummaryResponse)
async def llm_routing_summary(
    provider: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    task_type: Optional[str] = Query(None),
    user_id: Optional[UUID] = Query(None),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    limit: int = Query(20000, ge=1, le=100000),
    include_unrouted: bool = Query(True),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Routing observability summary based on `LLMUsageEvent.extra['routing']`.

    Groups by provider/model/task and routing metadata (tier/attempt).

    Non-admin users may only view their own usage.
    """
    if user_id and (not current_user.is_admin()) and user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to view other users' usage")

    effective_user_id = user_id
    if (not current_user.is_admin()) and effective_user_id is None:
        effective_user_id = current_user.id

    stmt = select(
        LLMUsageEvent.provider,
        LLMUsageEvent.model,
        LLMUsageEvent.task_type,
        LLMUsageEvent.total_tokens,
        LLMUsageEvent.latency_ms,
        LLMUsageEvent.error,
        LLMUsageEvent.extra,
    )

    if effective_user_id is not None:
        stmt = stmt.where(LLMUsageEvent.user_id == effective_user_id)
    if provider:
        stmt = stmt.where(LLMUsageEvent.provider == provider)
    if model:
        stmt = stmt.where(LLMUsageEvent.model == model)
    if task_type:
        stmt = stmt.where(LLMUsageEvent.task_type == task_type)
    if date_from:
        stmt = stmt.where(LLMUsageEvent.created_at >= date_from)
    if date_to:
        stmt = stmt.where(LLMUsageEvent.created_at <= date_to)

    stmt = stmt.order_by(desc(LLMUsageEvent.created_at)).limit(limit)

    rows = (await db.execute(stmt)).all()

    # Aggregate in Python to avoid DB-specific JSON operators.
    groups: dict[tuple, dict] = {}

    def _opt_int(v: object) -> Optional[int]:
        if v is None:
            return None
        if isinstance(v, int):
            return int(v)
        try:
            s = str(v).strip()
            if s.isdigit():
                return int(s)
        except Exception:
            return None
        return None

    for r in rows:
        extra = r.extra if isinstance(r.extra, dict) else None
        routing = extra.get("routing") if isinstance(extra, dict) else None
        routing = routing if isinstance(routing, dict) else None

        routing_tier = routing.get("tier") if routing else None
        routing_requested_tier = routing.get("requested_tier") if routing else None
        routing_attempt = routing.get("attempt") if routing else None
        routing_attempts = routing.get("attempts") if routing else None
        routing_tier_provider = routing.get("tier_provider") if routing else None
        routing_tier_model = routing.get("tier_model") if routing else None
        routing_experiment_id = routing.get("experiment_id") if routing else None
        routing_experiment_variant_id = routing.get("experiment_variant_id") if routing else None

        if not include_unrouted and not routing:
            continue

        key = (
            r.provider,
            r.model,
            r.task_type,
            str(routing_tier).lower() if isinstance(routing_tier, str) else None,
            str(routing_requested_tier).lower() if isinstance(routing_requested_tier, str) else None,
            _opt_int(routing_attempt),
            _opt_int(routing_attempts),
            str(routing_tier_provider).lower() if isinstance(routing_tier_provider, str) else None,
            str(routing_tier_model) if isinstance(routing_tier_model, str) else None,
            str(routing_experiment_id) if isinstance(routing_experiment_id, str) else None,
            str(routing_experiment_variant_id) if isinstance(routing_experiment_variant_id, str) else None,
        )

        g = groups.get(key)
        if g is None:
            g = {
                "provider": r.provider,
                "model": r.model,
                "task_type": r.task_type,
                "routing_tier": key[3],
                "routing_requested_tier": key[4],
                "routing_attempt": key[5],
                "routing_attempts": key[6],
                "routing_tier_provider": key[7],
                "routing_tier_model": key[8],
                "routing_experiment_id": key[9],
                "routing_experiment_variant_id": key[10],
                "request_count": 0,
                "success_count": 0,
                "error_count": 0,
                "total_tokens": 0,
                "latencies": [],
                "latency_sum": 0,
            }
            groups[key] = g

        g["request_count"] += 1
        if r.error:
            g["error_count"] += 1
        else:
            g["success_count"] += 1

        if isinstance(r.total_tokens, int):
            g["total_tokens"] += int(r.total_tokens)

        if isinstance(r.latency_ms, int):
            g["latencies"].append(int(r.latency_ms))
            g["latency_sum"] += int(r.latency_ms)

    items: List[LLMRoutingSummaryItem] = []
    for g in groups.values():
        rc = int(g["request_count"] or 0)
        sc = int(g["success_count"] or 0)
        ec = int(g["error_count"] or 0)
        success_rate = (float(sc) / float(rc)) if rc else 0.0
        latencies = g.get("latencies") or []
        avg_latency = (float(g.get("latency_sum") or 0) / float(len(latencies))) if latencies else None

        items.append(
            LLMRoutingSummaryItem(
                provider=g["provider"],
                model=g["model"],
                task_type=g["task_type"],
                routing_tier=g["routing_tier"],
                routing_requested_tier=g["routing_requested_tier"],
                routing_attempt=g["routing_attempt"],
                routing_attempts=g["routing_attempts"],
                routing_tier_provider=g["routing_tier_provider"],
                routing_tier_model=g["routing_tier_model"],
                routing_experiment_id=g.get("routing_experiment_id"),
                routing_experiment_variant_id=g.get("routing_experiment_variant_id"),
                request_count=rc,
                success_count=sc,
                error_count=ec,
                success_rate=success_rate,
                total_tokens=int(g.get("total_tokens") or 0),
                avg_latency_ms=avg_latency,
                p50_latency_ms=_percentile_int(latencies, 50),
                p95_latency_ms=_percentile_int(latencies, 95),
            )
        )

    items.sort(key=lambda x: (x.error_count, x.request_count), reverse=True)

    return LLMRoutingSummaryResponse(
        items=items,
        date_from=date_from,
        date_to=date_to,
        scanned_events=len(rows),
        truncated=len(rows) >= limit,
    )



@router.get("/llm/routing/experiments/recommendation", response_model=LLMRoutingExperimentRecommendationResponse)
async def llm_routing_experiment_recommendation(
    experiment_id: str = Query(..., min_length=1),
    agent_id: Optional[UUID] = Query(None),
    user_id: Optional[UUID] = Query(None),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    limit: int = Query(50000, ge=1, le=200000),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Recommend an experiment variant based on success rate and p95 latency.

    Uses `LLMUsageEvent.extra['routing'].experiment_id/experiment_variant_id`.
    """
    if user_id and (not current_user.is_admin()) and user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to view other users' usage")

    effective_user_id = user_id
    if (not current_user.is_admin()) and effective_user_id is None:
        effective_user_id = current_user.id

    stmt = select(
        LLMUsageEvent.provider,
        LLMUsageEvent.model,
        LLMUsageEvent.task_type,
        LLMUsageEvent.latency_ms,
        LLMUsageEvent.error,
        LLMUsageEvent.extra,
    )

    if effective_user_id is not None:
        stmt = stmt.where(LLMUsageEvent.user_id == effective_user_id)
    if date_from:
        stmt = stmt.where(LLMUsageEvent.created_at >= date_from)
    if date_to:
        stmt = stmt.where(LLMUsageEvent.created_at <= date_to)

    stmt = stmt.order_by(desc(LLMUsageEvent.created_at)).limit(limit)
    rows = (await db.execute(stmt)).all()

    def _get_routing(ev_extra: object) -> Optional[dict]:
        if not isinstance(ev_extra, dict):
            return None
        r = ev_extra.get('routing')
        return r if isinstance(r, dict) else None

    # Aggregate by variant
    groups: dict[str, dict] = {}
    for r in rows:
        routing = _get_routing(r.extra)
        if not routing:
            continue
        if str(routing.get('experiment_id') or '') != str(experiment_id):
            continue
        if agent_id is not None:
            rid = routing.get('agent_id')
            if rid and str(rid) != str(agent_id):
                continue

        variant_id = str(routing.get('experiment_variant_id') or '').strip()
        if not variant_id:
            continue

        g = groups.get(variant_id)
        if g is None:
            g = {
                'request_count': 0,
                'success_count': 0,
                'error_count': 0,
                'latencies': [],
                'latency_sum': 0,
            }
            groups[variant_id] = g

        g['request_count'] += 1
        if r.error:
            g['error_count'] += 1
        else:
            g['success_count'] += 1

        if isinstance(r.latency_ms, int):
            g['latencies'].append(int(r.latency_ms))
            g['latency_sum'] += int(r.latency_ms)

    variants = []
    for vid, g in groups.items():
        rc = int(g['request_count'] or 0)
        sc = int(g['success_count'] or 0)
        ec = int(g['error_count'] or 0)
        sr = (float(sc) / float(rc)) if rc else 0.0
        lat = g.get('latencies') or []
        avg = (float(g.get('latency_sum') or 0) / float(len(lat))) if lat else None
        variants.append(
            LLMRoutingExperimentVariantStat(
                experiment_id=str(experiment_id),
                variant_id=str(vid),
                request_count=rc,
                success_count=sc,
                error_count=ec,
                success_rate=sr,
                avg_latency_ms=avg,
                p95_latency_ms=_percentile_int(lat, 95),
            )
        )

    variants.sort(key=lambda v: (v.success_rate, -(v.p95_latency_ms or 10**9)), reverse=True)

    recommended = variants[0].variant_id if variants else None
    rationale = 'No data for this experiment in the selected window.'
    if variants:
        best = variants[0]
        rationale = f"Best success_rate={best.success_rate:.3f}, p95_latency_ms={best.p95_latency_ms or 'n/a'}"

    return LLMRoutingExperimentRecommendationResponse(
        experiment_id=str(experiment_id),
        agent_id=agent_id,
        recommended_variant_id=recommended,
        rationale=rationale,
        variants=variants,
        date_from=date_from,
        date_to=date_to,
        scanned_events=len(rows),
        truncated=len(rows) >= limit,
    )


@router.get("/llm/routing/experiments", response_model=LLMRoutingExperimentListResponse)
async def list_llm_routing_experiments(
    enabled_only: bool = Query(False),
    include_system: bool = Query(False),
    search: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List agent routing experiments defined in `AgentDefinition.routing_defaults.experiment`.

    Admins can view all agents (optionally including system agents). Non-admins can only view
    agents they own and never system agents.
    """
    effective_include_system = bool(include_system) if current_user.is_admin() else False

    stmt = select(AgentDefinition).order_by(AgentDefinition.name.asc())

    if not current_user.is_admin():
        stmt = stmt.where(AgentDefinition.owner_user_id == current_user.id)

    if not effective_include_system:
        stmt = stmt.where(AgentDefinition.is_system.is_(False))

    if search:
        s = f"%{search.strip()}%"
        stmt = stmt.where(
            func.lower(AgentDefinition.name).like(func.lower(s))
            | func.lower(AgentDefinition.display_name).like(func.lower(s))
        )

    rows = (await db.execute(stmt)).scalars().all()

    items: List[LLMRoutingExperimentListItem] = []

    for a in rows:
        rd = a.routing_defaults if isinstance(a.routing_defaults, dict) else None
        if not rd:
            continue
        exp = rd.get("experiment")
        if not isinstance(exp, dict):
            continue
        if enabled_only and not bool(exp.get("enabled")):
            continue

        items.append(
            LLMRoutingExperimentListItem(
                agent_id=a.id,
                agent_name=a.name,
                agent_display_name=a.display_name,
                agent_is_system=bool(a.is_system),
                agent_owner_user_id=a.owner_user_id,
                agent_lifecycle_status=getattr(a, "lifecycle_status", None),
                routing_defaults=rd,
                experiment=exp,
            )
        )

    return LLMRoutingExperimentListResponse(items=items, total=len(items))
