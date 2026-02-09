"""
API endpoints for training jobs.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.training import (
    TrainingJobCreate,
    TrainingJobUpdate,
    TrainingJobResponse,
    TrainingJobListResponse,
    TrainingJobDetailResponse,
    TrainingJobActionRequest,
    TrainingCheckpointResponse,
    TrainingStatsResponse,
    BaseModelInfo,
)
from app.services.training_service import training_service

router = APIRouter()


def _job_to_response(job) -> TrainingJobResponse:
    """Convert job model to response schema."""
    return TrainingJobResponse(
        id=job.id,
        name=job.name,
        description=job.description,
        training_method=job.training_method,
        training_backend=job.training_backend,
        base_model=job.base_model,
        base_model_provider=job.base_model_provider,
        dataset_id=job.dataset_id,
        hyperparameters=job.hyperparameters,
        resource_config=job.resource_config,
        user_id=job.user_id,
        status=job.status,
        progress=job.progress,
        current_step=job.current_step,
        total_steps=job.total_steps,
        current_epoch=job.current_epoch,
        total_epochs=job.total_epochs,
        training_metrics=job.training_metrics,
        final_metrics=job.final_metrics,
        output_adapter_id=job.output_adapter_id,
        error=job.error,
        celery_task_id=job.celery_task_id,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.post(
    "",
    response_model=TrainingJobResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_training_job(
    data: TrainingJobCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Create a new training job."""
    try:
        job = await training_service.create_training_job(
            db=db,
            user_id=current_user.id,
            data=data,
        )
        return _job_to_response(job)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("", response_model=TrainingJobListResponse)
async def list_training_jobs(
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """List training jobs for the current user."""
    jobs, total = await training_service.list_jobs(
        db=db,
        user_id=current_user.id,
        status=status_filter,
        page=page,
        page_size=page_size,
    )

    return TrainingJobListResponse(
        jobs=[_job_to_response(j) for j in jobs],
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )


@router.get("/stats", response_model=TrainingStatsResponse)
async def get_training_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get training statistics for the current user."""
    stats = await training_service.get_stats(db=db, user_id=current_user.id)

    return TrainingStatsResponse(
        total_jobs=stats.get("total_jobs", 0),
        running_jobs=stats.get("running_jobs", 0),
        completed_jobs=stats.get("completed_jobs", 0),
        failed_jobs=stats.get("failed_jobs", 0),
        total_training_hours=0.0,  # TODO: Calculate from completed jobs
        total_samples_trained=0,   # TODO: Calculate from completed jobs
        avg_final_loss=None,
    )


@router.get("/base-models")
async def list_base_models(
    current_user: User = Depends(get_current_active_user),
):
    """Get list of available base models for training."""
    models = await training_service.get_available_base_models()

    return {
        "models": [
            BaseModelInfo(
                name=m["name"],
                display_name=m.get("display_name", m["name"]),
                provider=m.get("provider", "ollama"),
                size_gb=m.get("size_gb"),
                parameters=m.get("parameters"),
                context_length=m.get("context_length"),
                is_available=m.get("is_available", True),
            )
            for m in models
        ]
    }


@router.get("/backends")
async def list_backends(
    current_user: User = Depends(get_current_active_user),
):
    """Get list of available training backends."""
    backends = training_service.get_available_backends()

    return {
        "backends": [
            {
                "name": backend,
                "display_name": backend.capitalize(),
                "is_available": True,
            }
            for backend in backends
        ]
    }


@router.get("/{job_id}", response_model=TrainingJobDetailResponse)
async def get_training_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get a training job by ID."""
    job = await training_service.get_job(
        db=db,
        job_id=job_id,
        user_id=current_user.id,
    )

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found",
        )

    response = _job_to_response(job)
    return TrainingJobDetailResponse(
        **response.model_dump(),
        dataset_name=job.dataset.name if job.dataset else None,
        dataset_sample_count=job.dataset.sample_count if job.dataset else None,
    )


@router.patch("/{job_id}", response_model=TrainingJobResponse)
async def update_training_job(
    job_id: UUID,
    data: TrainingJobUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Update a training job (name/description only for pending jobs)."""
    job = await training_service.get_job(
        db=db,
        job_id=job_id,
        user_id=current_user.id,
    )

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found",
        )

    if job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update this job",
        )

    # Only allow updates for pending jobs
    if not job.can_start():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Can only update pending jobs",
        )

    update_data = data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(job, field, value)

    await db.commit()
    await db.refresh(job)

    return _job_to_response(job)


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_training_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Delete a training job."""
    success = await training_service.delete_job(
        db=db,
        job_id=job_id,
        user_id=current_user.id,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found",
        )


@router.post("/{job_id}/start", response_model=TrainingJobResponse)
async def start_training_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Start a pending training job."""
    try:
        job = await training_service.start_job(db=db, job_id=job_id)
        return _job_to_response(job)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/{job_id}/cancel", response_model=TrainingJobResponse)
async def cancel_training_job(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Cancel a running or pending training job."""
    try:
        job = await training_service.cancel_job(
            db=db,
            job_id=job_id,
            user_id=current_user.id,
        )
        return _job_to_response(job)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/{job_id}/action", response_model=TrainingJobResponse)
async def perform_job_action(
    job_id: UUID,
    action: TrainingJobActionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Perform an action on a training job (start, cancel)."""
    if action.action == "start":
        return await start_training_job(job_id, db, current_user)
    elif action.action == "cancel":
        return await cancel_training_job(job_id, db, current_user)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown action: {action.action}",
        )


@router.get("/{job_id}/checkpoints")
async def get_checkpoints(
    job_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Get checkpoints for a training job."""
    job = await training_service.get_job(
        db=db,
        job_id=job_id,
        user_id=current_user.id,
    )

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Training job not found",
        )

    checkpoints = await training_service.get_checkpoints(db=db, job_id=job_id)

    return {
        "checkpoints": [
            TrainingCheckpointResponse(
                id=c.id,
                job_id=c.job_id,
                step=c.step,
                epoch=c.epoch,
                checkpoint_path=c.checkpoint_path,
                loss=c.loss,
                metrics=c.metrics,
                created_at=c.created_at,
            )
            for c in checkpoints
        ]
    }


@router.websocket("/{job_id}/progress")
async def training_progress_websocket(
    websocket: WebSocket,
    job_id: UUID,
):
    """WebSocket endpoint for real-time training progress updates."""
    import json
    import redis.asyncio as redis
    from sqlalchemy import select, and_

    from app.core.config import settings
    from app.core.database import AsyncSessionLocal
    from app.models.training_job import TrainingJob
    from app.utils.websocket_auth import require_websocket_auth

    # Authenticate + accept
    try:
        user = await require_websocket_auth(websocket)
    except WebSocketDisconnect:
        return

    # Verify job ownership
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(TrainingJob).where(
                and_(TrainingJob.id == job_id, TrainingJob.user_id == user.id)
            )
        )
        job = result.scalar_one_or_none()

    if not job:
        try:
            await websocket.send_json({"type": "error", "error": "Training job not found"})
        except Exception:
            pass
        await websocket.close(code=4004, reason="Training job not found")
        return

    # Subscribe to progress channel
    redis_client = None
    pubsub = None
    channel = f"training_job:{job_id}:progress"

    try:
        redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel)

        # Send initial state
        await websocket.send_json(
            {
                "type": "connected",
                "job_id": str(job_id),
                "status": job.status,
                "progress": job.progress,
            }
        )

        async for message in pubsub.listen():
            if message["type"] != "message":
                continue

            data = message["data"]
            await websocket.send_text(data)

            # Close on terminal state updates
            try:
                payload = json.loads(data)
                if payload.get("type") == "progress" and payload.get("status") in (
                    "completed",
                    "failed",
                    "cancelled",
                ):
                    break
            except Exception:
                pass

    except WebSocketDisconnect:
        return
    except Exception as e:
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass
    finally:
        try:
            if pubsub:
                await pubsub.unsubscribe(channel)
        except Exception:
            pass
        try:
            if redis_client:
                await redis_client.close()
        except Exception:
            pass
