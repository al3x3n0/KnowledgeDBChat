"""Single-job operator action HTTP boundary."""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.agent_job import AgentJob
from app.models.user import User
from app.schemas.agent_job import AgentJobActionRequest, AgentJobResponse

JobVisibility = Callable[[AgentJob, User], bool]
JobActionPerformer = Callable[..., Awaitable[AgentJob]]
OperatorEventRecorder = Callable[..., Awaitable[Any]]
SchedulerStateExtractor = Callable[[AgentJob | None], dict[str, Any] | None]
JobPresenter = Callable[..., AgentJobResponse]

_OWNER_ONLY_ACTIONS = {
    "pause",
    "resume",
    "cancel",
    "restart",
    "relaunch",
    "generate_summary",
    "approve",
    "reject",
    "edit",
    "skip",
}


@dataclass(frozen=True)
class JobActionApi:
    router: APIRouter
    job_action: Callable[..., Any]


def build_job_action_api(
    *,
    router: APIRouter,
    is_job_visible: JobVisibility,
    perform_job_action: JobActionPerformer,
    record_operator_event: OperatorEventRecorder,
    extract_scheduler_state: SchedulerStateExtractor,
    present_job: JobPresenter,
) -> JobActionApi:
    """Register ownership-aware operator actions for one job."""

    @router.post("/{job_id}/action", response_model=AgentJobResponse)
    async def job_action(
        job_id: UUID,
        request: AgentJobActionRequest,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        job = (
            await db.execute(
                select(AgentJob)
                .options(selectinload(AgentJob.agent_definition))
                .where(AgentJob.id == job_id)
            )
        ).scalar_one_or_none()
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent job not found",
            )

        if request.action.lower() in _OWNER_ONLY_ACTIONS:
            if not (
                current_user.is_admin() or str(job.user_id) == str(current_user.id)
            ):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Agent job not found",
                )
        elif not is_job_visible(job, current_user):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent job not found",
            )

        previous_status = str(job.status or "")
        result_job = await perform_job_action(
            job,
            request,
            db=db,
            current_user=current_user,
        )
        await record_operator_event(
            db=db,
            job=job,
            current_user=current_user,
            action=request.action,
            note=request.checkpoint_note,
            previous_status=previous_status,
            next_status=str(job.status or ""),
            scheduler_state=extract_scheduler_state(job),
            metadata={
                "returned_job_id": (
                    str(result_job.id) if getattr(result_job, "id", None) else None
                ),
                "spawned_new_job": (
                    str(getattr(result_job, "id", None) or "") != str(job.id)
                ),
            },
            summary=(
                f"{str(job.name or 'Agent job').strip()}: "
                f"{str(request.action or '').strip().replace('_', ' ')}"
            ),
        )
        await db.commit()
        await db.refresh(result_job)
        return present_job(result_job)

    return JobActionApi(router=router, job_action=job_action)
