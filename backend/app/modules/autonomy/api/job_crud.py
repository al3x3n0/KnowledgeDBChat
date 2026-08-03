"""Composed HTTP boundaries for autonomous-job creation and record CRUD."""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.agent_job import AgentJob, AgentJobCheckpoint, AgentJobStatus
from app.models.user import User
from app.schemas.agent_job import (
    AgentJobCreate,
    AgentJobDetailResponse,
    AgentJobFromTemplate,
    AgentJobResponse,
    AgentJobUpdate,
)
from app.services.agent_job_creation_service import (
    AgentJobCreationError,
    agent_job_creation_service,
)

JobSerializer = Callable[..., AgentJobResponse]
JobVisibility = Callable[[AgentJob, User], bool]
ScopeNormalizer = Callable[[Any], Any]
RelaunchCountsLoader = Callable[..., Awaitable[dict[UUID, int]]]
CollaborationLookupLoader = Callable[..., Awaitable[dict[str, User]]]


@dataclass(frozen=True)
class JobCreationApi:
    router: APIRouter
    create_agent_job: Callable[..., Any]
    create_job_from_template: Callable[..., Any]


@dataclass(frozen=True)
class JobRecordApi:
    router: APIRouter
    get_agent_job: Callable[..., Any]
    update_agent_job: Callable[..., Any]
    delete_agent_job: Callable[..., Any]


def _creation_http_exception(error: AgentJobCreationError) -> HTTPException:
    return HTTPException(status_code=error.status_code, detail=error.detail)


def build_job_creation_api(
    *,
    job_serializer: JobSerializer,
    execute_job_task: Any,
    router: APIRouter | None = None,
) -> JobCreationApi:
    """Register create routes with task dispatch and presentation injected."""
    target_router = router if router is not None else APIRouter()

    @target_router.post(
        "",
        response_model=AgentJobResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_agent_job(
        job_create: AgentJobCreate,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        try:
            job = await agent_job_creation_service.create_from_request(
                request=job_create,
                user_id=current_user.id,
                db=db,
            )
        except AgentJobCreationError as error:
            raise _creation_http_exception(error) from error

        logger.info(f"Created agent job {job.id} for user {current_user.id}")
        if job_create.start_immediately:
            execute_job_task.delay(str(job.id), str(current_user.id))
            logger.info(f"Queued agent job {job.id} for immediate execution")
            await agent_job_creation_service.mark_immediately_dispatched(
                job=job,
                db=db,
            )
        return job_serializer(job)

    @target_router.post(
        "/from-template",
        response_model=AgentJobResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_job_from_template(
        request: AgentJobFromTemplate,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        try:
            job = await agent_job_creation_service.create_from_template(
                request=request,
                user_id=current_user.id,
                db=db,
            )
        except AgentJobCreationError as error:
            raise _creation_http_exception(error) from error

        logger.info(f"Created agent job {job.id} from template {request.template_id}")
        if request.start_immediately:
            execute_job_task.delay(str(job.id), str(current_user.id))
        return job_serializer(job)

    return JobCreationApi(
        router=target_router,
        create_agent_job=create_agent_job,
        create_job_from_template=create_job_from_template,
    )


def build_job_record_api(
    *,
    job_serializer: JobSerializer,
    is_job_visible: JobVisibility,
    normalize_scope_config: ScopeNormalizer,
    load_relaunch_children_counts: RelaunchCountsLoader,
    load_collaboration_user_lookup: CollaborationLookupLoader,
) -> JobRecordApi:
    """Build detail/update/delete routes with legacy projection edges injected."""
    router = APIRouter()

    @router.get("/{job_id}", response_model=AgentJobDetailResponse)
    async def get_agent_job(
        job_id: UUID,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        result = await db.execute(
            select(AgentJob)
            .options(selectinload(AgentJob.agent_definition))
            .where(AgentJob.id == job_id)
        )
        job = result.scalar_one_or_none()
        if job is None or not is_job_visible(job, current_user):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent job not found",
            )

        children_counts = await load_relaunch_children_counts(
            db,
            user_id=current_user.id,
        )
        user_lookup = await load_collaboration_user_lookup(
            db,
            current_user=current_user,
        )
        response = job_serializer(
            job,
            relaunch_children_count=int(children_counts.get(job.id, 0) or 0),
            current_user_id=str(current_user.id),
            user_lookup=user_lookup,
        )
        return AgentJobDetailResponse(
            **response.model_dump(),
            execution_log=job.execution_log,
        )

    @router.patch("/{job_id}", response_model=AgentJobResponse)
    async def update_agent_job(
        job_id: UUID,
        job_update: AgentJobUpdate,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        result = await db.execute(
            select(AgentJob)
            .options(selectinload(AgentJob.agent_definition))
            .where(
                and_(
                    AgentJob.id == job_id,
                    AgentJob.user_id == current_user.id,
                )
            )
        )
        job = result.scalar_one_or_none()
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent job not found",
            )
        if job.status not in {
            AgentJobStatus.PENDING.value,
            AgentJobStatus.PAUSED.value,
        }:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot update job in status: {job.status}",
            )

        update_data = job_update.model_dump(exclude_unset=True)
        if "config" in update_data:
            update_data["config"] = normalize_scope_config(update_data.get("config"))
        for field, value in update_data.items():
            setattr(job, field, value)
        await db.commit()
        await db.refresh(job)
        return job_serializer(job)

    @router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_agent_job(
        job_id: UUID,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        result = await db.execute(
            select(AgentJob).where(
                and_(
                    AgentJob.id == job_id,
                    AgentJob.user_id == current_user.id,
                )
            )
        )
        job = result.scalar_one_or_none()
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent job not found",
            )
        if job.status == AgentJobStatus.RUNNING.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete running job. Cancel it first.",
            )

        await db.execute(
            select(AgentJobCheckpoint).where(AgentJobCheckpoint.job_id == job_id)
        )
        await db.delete(job)
        await db.commit()

    return JobRecordApi(
        router=router,
        get_agent_job=get_agent_job,
        update_agent_job=update_agent_job,
        delete_agent_job=delete_agent_job,
    )
