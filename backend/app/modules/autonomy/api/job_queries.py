"""HTTP query boundary for autonomous-job lists and aggregate statistics."""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import String, and_, cast, func, literal, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased, selectinload

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.user import User
from app.schemas.agent_job import (
    AgentJobListResponse,
    AgentJobResponse,
    AgentJobStatsResponse,
)

JobSerializer = Callable[..., AgentJobResponse]
JobVisibility = Callable[[AgentJob, User], bool]
SwarmSummaryExtractor = Callable[[AgentJob], Optional[dict[str, Any]]]
RelaunchCountsLoader = Callable[..., Awaitable[dict[UUID, int]]]
CollaborationLookupLoader = Callable[..., Awaitable[dict[str, User]]]
LaunchModeStatsBuilder = Callable[
    [list[Optional[dict]]],
    tuple[dict[str, int], int],
]


@dataclass(frozen=True)
class JobQueryApi:
    router: APIRouter
    list_agent_jobs: Callable[..., Any]
    get_job_stats: Callable[..., Any]


def _json_config_text(model: Any, key: str) -> Any:
    try:
        return model.config[key].as_string()
    except Exception:
        return model.config[key].astext


def build_job_query_api(
    *,
    router: APIRouter,
    job_serializer: JobSerializer,
    is_job_visible: JobVisibility,
    extract_swarm_summary: SwarmSummaryExtractor,
    load_relaunch_children_counts: RelaunchCountsLoader,
    load_collaboration_user_lookup: CollaborationLookupLoader,
    build_launch_mode_stats: LaunchModeStatsBuilder,
) -> JobQueryApi:
    """Register list and statistics routes at their precedence-sensitive point."""

    @router.get("", response_model=AgentJobListResponse)
    async def list_agent_jobs(
        status: Optional[str] = Query(None, description="Filter by status"),
        job_type: Optional[str] = Query(None, description="Filter by job type"),
        launch_mode: Optional[str] = Query(
            None,
            description="Filter by launch mode",
        ),
        relaunch_from_job_id: Optional[str] = Query(
            None,
            description="Filter jobs relaunched from a specific parent job id",
        ),
        has_relaunch_children: Optional[bool] = Query(
            None,
            description="Filter by whether jobs have relaunch descendants",
        ),
        swarm_only: bool = Query(
            False,
            description="Only return jobs with swarm summary data",
        ),
        swarm_min_consensus: int = Query(
            0,
            ge=0,
            le=100,
            description="Minimum swarm consensus findings",
        ),
        visibility_scope: str = Query(
            "mine",
            description="Visibility scope: mine|shared|all",
        ),
        sort_by: str = Query(
            "created_desc",
            description=(
                "created_desc|created_asc|swarm_confidence_desc|"
                "swarm_consensus_desc|swarm_conflicts_desc"
            ),
        ),
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(
            20,
            ge=1,
            le=100,
            description="Items per page",
        ),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        normalized_visibility = (
            str(visibility_scope or "mine").strip().lower() or "mine"
        )
        statement = select(AgentJob)
        if normalized_visibility == "mine":
            statement = statement.where(AgentJob.user_id == current_user.id)
        elif normalized_visibility == "shared":
            statement = statement.where(AgentJob.user_id != current_user.id)

        if status:
            statement = statement.where(AgentJob.status == status)
        if job_type:
            statement = statement.where(AgentJob.job_type == job_type)

        launch_mode_filter = str(launch_mode or "").strip().lower()
        launch_mode_expr = func.lower(
            func.trim(
                func.coalesce(
                    _json_config_text(AgentJob, "launch_mode"),
                    "",
                )
            )
        )
        if launch_mode_filter:
            if launch_mode_filter in {"__none__", "none", "manual"}:
                statement = statement.where(
                    launch_mode_expr.in_(["", "__none__", "none", "manual"])
                )
            else:
                statement = statement.where(launch_mode_expr == launch_mode_filter)

        parent_filter_raw = str(relaunch_from_job_id or "").strip()
        parent_filter_id = None
        if parent_filter_raw:
            try:
                parent_filter_id = UUID(parent_filter_raw)
            except Exception as error:
                raise HTTPException(
                    status_code=422,
                    detail="Invalid relaunch_from_job_id",
                ) from error
        parent_expr = _json_config_text(
            AgentJob,
            "relaunch_from_job_id",
        )
        if parent_filter_id is not None:
            statement = statement.where(parent_expr == str(parent_filter_id))

        if has_relaunch_children is not None and normalized_visibility == "mine":
            child = aliased(AgentJob)
            child_parent_expr = _json_config_text(
                child,
                "relaunch_from_job_id",
            )
            has_children_expr = (
                select(literal(1))
                .where(
                    and_(
                        child.user_id == current_user.id,
                        child_parent_expr == cast(AgentJob.id, String),
                    )
                )
                .exists()
            )
            statement = statement.where(
                has_children_expr if has_relaunch_children else ~has_children_expr
            )

        children_counts = (
            await load_relaunch_children_counts(
                db,
                user_id=current_user.id,
            )
            if normalized_visibility == "mine"
            else {}
        )
        sort_mode = str(sort_by or "created_desc").strip().lower()
        if sort_mode not in {
            "created_desc",
            "created_asc",
            "swarm_confidence_desc",
            "swarm_consensus_desc",
            "swarm_conflicts_desc",
        }:
            sort_mode = "created_desc"

        requires_swarm_projection = (
            bool(swarm_only)
            or int(swarm_min_consensus or 0) > 0
            or sort_mode.startswith("swarm_")
            or sort_mode == "created_asc"
        )
        statement = statement.options(selectinload(AgentJob.agent_definition)).order_by(
            AgentJob.created_at.desc()
        )
        result = await db.execute(statement)
        all_jobs = result.scalars().all()
        if normalized_visibility != "mine":
            all_jobs = [job for job in all_jobs if is_job_visible(job, current_user)]

        if not requires_swarm_projection:
            total = len(all_jobs)
            offset = (page - 1) * page_size
            jobs = all_jobs[offset : offset + page_size]
        else:
            rows = []
            for job in all_jobs:
                swarm_summary = extract_swarm_summary(job)
                if swarm_only and not swarm_summary:
                    continue
                if int(swarm_min_consensus or 0) > 0:
                    consensus_count = int(
                        (swarm_summary or {}).get("consensus_count", 0) or 0
                    )
                    if consensus_count < int(swarm_min_consensus or 0):
                        continue
                rows.append((job, swarm_summary))

            def created_timestamp(job: AgentJob) -> float:
                try:
                    return float(job.created_at.timestamp()) if job.created_at else 0.0
                except Exception:
                    return 0.0

            if sort_mode == "created_asc":
                rows.sort(key=lambda row: created_timestamp(row[0]))
            elif sort_mode == "swarm_confidence_desc":
                rows.sort(
                    key=lambda row: (
                        float(
                            (
                                ((row[1] or {}).get("confidence") or {}).get("overall")
                                or 0.0
                            )
                        ),
                        int((row[1] or {}).get("consensus_count", 0) or 0),
                        created_timestamp(row[0]),
                    ),
                    reverse=True,
                )
            elif sort_mode == "swarm_consensus_desc":
                rows.sort(
                    key=lambda row: (
                        int((row[1] or {}).get("consensus_count", 0) or 0),
                        float(
                            (
                                ((row[1] or {}).get("confidence") or {}).get("overall")
                                or 0.0
                            )
                        ),
                        created_timestamp(row[0]),
                    ),
                    reverse=True,
                )
            elif sort_mode == "swarm_conflicts_desc":
                rows.sort(
                    key=lambda row: (
                        int((row[1] or {}).get("conflict_count", 0) or 0),
                        int((row[1] or {}).get("consensus_count", 0) or 0),
                        created_timestamp(row[0]),
                    ),
                    reverse=True,
                )

            total = len(rows)
            offset = (page - 1) * page_size
            jobs = [job for job, _summary in rows[offset : offset + page_size]]

        user_lookup = await load_collaboration_user_lookup(
            db,
            current_user=current_user,
        )
        return AgentJobListResponse(
            jobs=[
                job_serializer(
                    job,
                    relaunch_children_count=int(children_counts.get(job.id, 0) or 0),
                    current_user_id=str(current_user.id),
                    user_lookup=user_lookup,
                )
                for job in jobs
            ],
            total=total,
            page=page,
            page_size=page_size,
            has_more=(page * page_size) < total,
        )

    @router.get("/stats", response_model=AgentJobStatsResponse)
    async def get_job_stats(
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        status_counts = {}
        for job_status in AgentJobStatus:
            count_result = await db.execute(
                select(func.count()).where(
                    and_(
                        AgentJob.user_id == current_user.id,
                        AgentJob.status == job_status.value,
                    )
                )
            )
            status_counts[job_status.value] = count_result.scalar()

        total_result = await db.execute(
            select(
                func.sum(AgentJob.iteration),
                func.sum(AgentJob.tool_calls_used),
                func.sum(AgentJob.llm_calls_used),
            ).where(AgentJob.user_id == current_user.id)
        )
        totals = total_result.one()

        launch_rows = await db.execute(
            select(AgentJob.config).where(AgentJob.user_id == current_user.id)
        )
        launch_configs = [row[0] for row in launch_rows.all()]
        launch_mode_counts, launch_mode_none_count = build_launch_mode_stats(
            launch_configs
        )

        completed_result = await db.execute(
            select(AgentJob).where(
                and_(
                    AgentJob.user_id == current_user.id,
                    AgentJob.status == AgentJobStatus.COMPLETED.value,
                    AgentJob.started_at.isnot(None),
                    AgentJob.completed_at.isnot(None),
                )
            )
        )
        completed_jobs = completed_result.scalars().all()
        average_time = None
        if completed_jobs:
            durations = [
                (job.completed_at - job.started_at).total_seconds() / 60
                for job in completed_jobs
            ]
            average_time = sum(durations) / len(durations)

        total_finished = status_counts.get(
            "completed",
            0,
        ) + status_counts.get("failed", 0)
        success_rate = None
        if total_finished > 0:
            success_rate = status_counts.get("completed", 0) / total_finished

        return AgentJobStatsResponse(
            total_jobs=sum(status_counts.values()),
            running_jobs=status_counts.get("running", 0),
            pending_jobs=status_counts.get("pending", 0),
            completed_jobs=status_counts.get("completed", 0),
            failed_jobs=status_counts.get("failed", 0),
            total_iterations=totals[0] or 0,
            total_tool_calls=totals[1] or 0,
            total_llm_calls=totals[2] or 0,
            avg_completion_time_minutes=average_time,
            success_rate=success_rate,
            launch_mode_counts=launch_mode_counts,
            launch_mode_none_count=launch_mode_none_count,
        )

    return JobQueryApi(
        router=router,
        list_agent_jobs=list_agent_jobs,
        get_job_stats=get_job_stats,
    )
