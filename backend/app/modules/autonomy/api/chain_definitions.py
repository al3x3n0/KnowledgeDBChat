"""HTTP boundary for reusable autonomous-job chain definitions."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.agent_job import (
    AgentJobChainDefinitionCreate,
    AgentJobChainDefinitionListResponse,
    AgentJobChainDefinitionResponse,
    AgentJobChainDefinitionUpdate,
)
from app.services.agent_chain_definition_service import (
    AgentChainDefinitionError,
    agent_chain_definition_service,
)

router = APIRouter()


def _chain_definition_http_exception(
    error: AgentChainDefinitionError,
) -> HTTPException:
    """Translate chain-definition domain failures at the HTTP boundary."""
    return HTTPException(status_code=error.status_code, detail=error.detail)


@router.get("/chains", response_model=AgentJobChainDefinitionListResponse)
async def list_chain_definitions(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List system chains and chain definitions owned by the current user."""
    chains = await agent_chain_definition_service.list_for_user(
        user_id=current_user.id,
        db=db,
    )
    return AgentJobChainDefinitionListResponse(
        chains=[agent_chain_definition_service.to_response(chain) for chain in chains],
        total=len(chains),
    )


@router.post(
    "/chains",
    response_model=AgentJobChainDefinitionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_chain_definition(
    chain_create: AgentJobChainDefinitionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create a reusable autonomous-job chain definition."""
    try:
        chain = await agent_chain_definition_service.create(
            request=chain_create,
            user_id=current_user.id,
            db=db,
        )
    except AgentChainDefinitionError as error:
        raise _chain_definition_http_exception(error) from error
    logger.info(f"Created chain definition {chain.id} for user {current_user.id}")
    return agent_chain_definition_service.to_response(chain)


@router.get("/chains/{chain_id}", response_model=AgentJobChainDefinitionResponse)
async def get_chain_definition(
    chain_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Return one visible autonomous-job chain definition."""
    try:
        chain = await agent_chain_definition_service.get_visible(
            chain_id=chain_id,
            user_id=current_user.id,
            db=db,
        )
    except AgentChainDefinitionError as error:
        raise _chain_definition_http_exception(error) from error
    return agent_chain_definition_service.to_response(chain)


@router.patch("/chains/{chain_id}", response_model=AgentJobChainDefinitionResponse)
async def update_chain_definition(
    chain_id: UUID,
    chain_update: AgentJobChainDefinitionUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Update a non-system chain definition owned by the current user."""
    try:
        chain = await agent_chain_definition_service.update_owned(
            chain_id=chain_id,
            request=chain_update,
            user_id=current_user.id,
            db=db,
        )
    except AgentChainDefinitionError as error:
        raise _chain_definition_http_exception(error) from error
    return agent_chain_definition_service.to_response(chain)


@router.delete("/chains/{chain_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chain_definition(
    chain_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Delete a non-system chain definition owned by the current user."""
    try:
        await agent_chain_definition_service.delete_owned(
            chain_id=chain_id,
            user_id=current_user.id,
            db=db,
        )
    except AgentChainDefinitionError as error:
        raise _chain_definition_http_exception(error) from error
