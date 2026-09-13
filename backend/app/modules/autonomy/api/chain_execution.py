"""HTTP composition for launching, observing, and saving autonomous job chains."""

from dataclasses import dataclass
from typing import Any, Callable
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.agent_job import (
    AgentJobChainDefinitionResponse,
    AgentJobChainStatusResponse,
    AgentJobFromChainCreate,
    AgentJobResponse,
    AgentJobSaveAsChainRequest,
)
from app.services.agent_chain_definition_service import AgentChainDefinitionError
from app.services.agent_chain_launch_service import (
    AgentChainLaunchError,
    agent_chain_launch_service,
)
from app.services.agent_chain_status_service import (
    AgentChainStatusError,
    agent_chain_status_service,
)
from app.services.agent_playbook_service import (
    AgentPlaybookError,
    agent_playbook_service,
)

JobSerializer = Callable[..., AgentJobResponse]


@dataclass(frozen=True)
class ChainExecutionApi:
    """Composed router plus compatibility-callable endpoint handlers."""

    router: APIRouter
    create_job_from_chain: Callable[..., Any]
    get_chain_status: Callable[..., Any]
    save_job_as_chain_definition: Callable[..., Any]


def build_chain_execution_api(
    *,
    job_serializer: JobSerializer,
    execute_job_task: Any,
) -> ChainExecutionApi:
    """Build chain routes with legacy presentation and task edges injected."""
    router = APIRouter()

    @router.post(
        "/from-chain",
        response_model=AgentJobResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_job_from_chain(
        request: AgentJobFromChainCreate,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        """Create and optionally start the root job for a chain definition."""
        try:
            job = await agent_chain_launch_service.launch(
                request=request,
                user_id=current_user.id,
                db=db,
            )
        except (AgentChainDefinitionError, AgentChainLaunchError) as error:
            raise HTTPException(
                status_code=error.status_code,
                detail=error.detail,
            ) from error

        logger.info(
            f"Created chain root job {job.id} from definition "
            f"{request.chain_definition_id}"
        )
        if request.start_immediately:
            execute_job_task.delay(str(job.id), str(current_user.id))
            logger.info(f"Queued chain root job {job.id} for immediate execution")
        return job_serializer(job)

    @router.get(
        "/{job_id}/chain-status",
        response_model=AgentJobChainStatusResponse,
    )
    async def get_chain_status(
        job_id: UUID,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        """Return aggregate status for the chain containing a visible job."""
        try:
            snapshot = await agent_chain_status_service.get_snapshot(
                job_id=job_id,
                user_id=current_user.id,
                db=db,
            )
        except AgentChainStatusError as error:
            raise HTTPException(
                status_code=error.status_code,
                detail=error.detail,
            ) from error

        return AgentJobChainStatusResponse(
            root_job_id=snapshot.root_job_id,
            chain_definition_id=snapshot.chain_definition_id,
            total_steps=snapshot.total_steps,
            completed_steps=snapshot.completed_steps,
            current_step=snapshot.current_step,
            overall_progress=snapshot.overall_progress,
            status=snapshot.status,
            jobs=[job_serializer(job) for job in snapshot.jobs],
        )

    @router.post(
        "/{job_id}/save-as-chain",
        response_model=AgentJobChainDefinitionResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def save_job_as_chain_definition(
        job_id: UUID,
        request: AgentJobSaveAsChainRequest,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        """Save a job or chain as a reusable playbook definition."""
        try:
            chain = await agent_playbook_service.save(
                job_id=job_id,
                request=request,
                user_id=current_user.id,
                db=db,
            )
        except (AgentPlaybookError, AgentChainDefinitionError) as error:
            raise HTTPException(
                status_code=error.status_code,
                detail=error.detail,
            ) from error
        return agent_playbook_service.definition_service.to_response(chain)

    return ChainExecutionApi(
        router=router,
        create_job_from_chain=create_job_from_chain,
        get_chain_status=get_chain_status,
        save_job_as_chain_definition=save_job_as_chain_definition,
    )
