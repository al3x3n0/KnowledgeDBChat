"""
API endpoints for model registry.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.training import (
    ModelAdapterUpdate,
    ModelAdapterResponse,
    ModelAdapterListResponse,
    DeployAdapterRequest,
    DeploymentStatusResponse,
    TestAdapterRequest,
    TestAdapterResponse,
)
from app.services.model_registry_service import model_registry_service

router = APIRouter()


def _adapter_to_response(adapter) -> ModelAdapterResponse:
    """Convert adapter model to response schema."""
    return ModelAdapterResponse(
        id=adapter.id,
        name=adapter.name,
        display_name=adapter.display_name,
        description=adapter.description,
        base_model=adapter.base_model,
        adapter_type=adapter.adapter_type,
        adapter_config=adapter.adapter_config,
        adapter_path=adapter.adapter_path,
        adapter_size=adapter.adapter_size,
        training_job_id=adapter.training_job_id,
        training_metrics=adapter.training_metrics,
        user_id=adapter.user_id,
        is_public=adapter.is_public,
        status=adapter.status,
        is_deployed=adapter.is_deployed,
        deployment_config=adapter.deployment_config,
        version=adapter.version,
        tags=adapter.tags,
        usage_count=adapter.usage_count,
        created_at=adapter.created_at,
        updated_at=adapter.updated_at,
    )


@router.get("", response_model=ModelAdapterListResponse)
async def list_adapters(
    status: Optional[str] = Query(None, description="Filter by status"),
    base_model: Optional[str] = Query(None, description="Filter by base model"),
    is_deployed: Optional[bool] = Query(None, description="Filter by deployment status"),
    include_public: bool = Query(True, description="Include public adapters"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List model adapters for the current user."""
    adapters, total = await model_registry_service.list_adapters(
        db=db,
        user_id=current_user.id,
        include_public=include_public,
        status=status,
        base_model=base_model,
        is_deployed=is_deployed,
        page=page,
        page_size=page_size,
    )

    return ModelAdapterListResponse(
        adapters=[_adapter_to_response(a) for a in adapters],
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )


@router.get("/stats")
async def get_adapter_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get adapter statistics for the current user."""
    stats = await model_registry_service.get_stats(db=db, user_id=current_user.id)
    return stats


@router.get("/deployed")
async def list_deployed_models(
    current_user: User = Depends(get_current_active_user),
):
    """Get list of deployed custom models from Ollama."""
    models = await model_registry_service.get_deployed_models()
    return {"models": models}


@router.get("/{adapter_id}", response_model=ModelAdapterResponse)
async def get_adapter(
    adapter_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get a model adapter by ID."""
    adapter = await model_registry_service.get_adapter(
        db=db,
        adapter_id=adapter_id,
        user_id=current_user.id,
    )

    if not adapter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Adapter not found",
        )

    return _adapter_to_response(adapter)


@router.patch("/{adapter_id}", response_model=ModelAdapterResponse)
async def update_adapter(
    adapter_id: UUID,
    data: ModelAdapterUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Update a model adapter."""
    adapter = await model_registry_service.update_adapter(
        db=db,
        adapter_id=adapter_id,
        user_id=current_user.id,
        data=data,
    )

    if not adapter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Adapter not found or not authorized",
        )

    return _adapter_to_response(adapter)


@router.delete("/{adapter_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_adapter(
    adapter_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Delete a model adapter."""
    success = await model_registry_service.delete_adapter(
        db=db,
        adapter_id=adapter_id,
        user_id=current_user.id,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Adapter not found or not authorized",
        )


@router.post("/{adapter_id}/deploy", response_model=ModelAdapterResponse)
async def deploy_adapter(
    adapter_id: UUID,
    request: Optional[DeployAdapterRequest] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Deploy an adapter to Ollama as a custom model."""
    try:
        adapter = await model_registry_service.deploy_to_ollama(
            db=db,
            adapter_id=adapter_id,
            user_id=current_user.id,
            request=request,
        )
        return _adapter_to_response(adapter)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/{adapter_id}/undeploy", response_model=ModelAdapterResponse)
async def undeploy_adapter(
    adapter_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Undeploy an adapter from Ollama."""
    try:
        adapter = await model_registry_service.undeploy_from_ollama(
            db=db,
            adapter_id=adapter_id,
            user_id=current_user.id,
        )
        return _adapter_to_response(adapter)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/{adapter_id}/deployment-status", response_model=DeploymentStatusResponse)
async def get_deployment_status(
    adapter_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get deployment status for an adapter."""
    adapter = await model_registry_service.get_adapter(
        db=db,
        adapter_id=adapter_id,
        user_id=current_user.id,
    )

    if not adapter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Adapter not found",
        )

    deployed_at = None
    if adapter.deployment_config and "deployed_at" in adapter.deployment_config:
        from datetime import datetime
        deployed_at = datetime.fromisoformat(adapter.deployment_config["deployed_at"])

    return DeploymentStatusResponse(
        adapter_id=adapter.id,
        is_deployed=adapter.is_deployed,
        ollama_model_name=adapter.get_ollama_model_name(),
        deployed_at=deployed_at,
        status=adapter.status,
    )


@router.post("/{adapter_id}/test", response_model=TestAdapterResponse)
async def test_adapter(
    adapter_id: UUID,
    request: TestAdapterRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Test an adapter with a prompt."""
    try:
        result = await model_registry_service.test_adapter(
            db=db,
            adapter_id=adapter_id,
            user_id=current_user.id,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        return TestAdapterResponse(
            prompt=result["prompt"],
            response=result["response"],
            tokens_generated=result["tokens_generated"],
            generation_time_ms=result["generation_time_ms"],
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
