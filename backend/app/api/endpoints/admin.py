"""
Admin API endpoints for system management.
"""

from typing import List, Dict, Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.database import get_db
from app.core.logging import log_error
from app.core.config import settings
from app.models.user import User
from app.services.auth_service import require_admin
from app.tasks.sync_tasks import sync_all_sources, ingest_from_source
from app.tasks.monitoring_tasks import health_check, generate_stats
from app.services.vector_store import VectorStoreService
from app.schemas.admin import (
    SystemStatsResponse,
    HealthCheckResponse,
    TaskTriggerResponse
)

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
async def get_system_health(
    current_user: User = Depends(require_admin)
):
    """
    Get comprehensive system health status.
    
    Checks the health of all system components:
    - Database connectivity
    - Celery worker status
    - Vector store status
    - LLM service availability
    
    Args:
        current_user: Current authenticated admin user
        
    Returns:
        HealthCheckResponse with status of all services
        
    Raises:
        HTTPException: 500 if health check fails
    """
    try:
        # Trigger health check task
        task = health_check.delay()
        
        # Wait for result with timeout
        try:
            result = task.get(timeout=30)  # Wait up to 30 seconds
            return HealthCheckResponse(**result)
        except Exception as task_error:
            # If task fails or times out, return basic health check
            logger.warning(f"Health check task failed or timed out: {task_error}")
            
            # Perform basic health checks directly
            from datetime import datetime
            basic_health = {
                "timestamp": datetime.utcnow().isoformat(),
                "overall_status": "degraded",
                "services": {
                    "celery": {
                        "status": "unhealthy",
                        "error": "Health check task failed or timed out"
                    }
                }
            }
            
            # Try to check database directly
            try:
                from app.core.database import AsyncSessionLocal
                from sqlalchemy import select, func
                from app.models.document import DocumentSource
                
                async with AsyncSessionLocal() as db:
                    result = await db.execute(select(func.count(DocumentSource.id)))
                    source_count = result.scalar()
                    basic_health["services"]["database"] = {
                        "status": "healthy",
                        "message": f"Connected successfully, {source_count} sources configured"
                    }
            except Exception as db_error:
                basic_health["services"]["database"] = {
                    "status": "unhealthy",
                    "error": str(db_error)
                }
                basic_health["overall_status"] = "unhealthy"
            
            return HealthCheckResponse(**basic_health)
    
    except Exception as e:
        log_error(e, context={"endpoint": "health_check"})
        raise HTTPException(status_code=500, detail="Failed to get system health")


@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    current_user: User = Depends(require_admin)
):
    """
    Get comprehensive system statistics.
    
    Returns statistics about:
    - Total documents and sources
    - User counts
    - Chat session statistics
    - Vector store statistics
    
    Args:
        current_user: Current authenticated admin user
        
    Returns:
        SystemStatsResponse with system statistics
        
    Raises:
        HTTPException: 500 if stats generation fails
    """
    try:
        # Trigger stats generation task
        task = generate_stats.delay()
        result = task.get(timeout=30)  # Wait up to 30 seconds
        
        return SystemStatsResponse(**result)
    
    except Exception as e:
        log_error(e, context={"endpoint": "system_stats"})
        raise HTTPException(status_code=500, detail="Failed to get system statistics")


@router.post("/sync/all", response_model=TaskTriggerResponse)
async def trigger_full_sync(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_admin)
):
    """Trigger synchronization of all data sources."""
    try:
        # Trigger sync task
        task = sync_all_sources.delay()
        
        return TaskTriggerResponse(
            task_id=task.id,
            message="Full synchronization started",
            status="triggered"
        )
    
    except Exception as e:
        logger.error(f"Error triggering full sync: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger synchronization")


@router.post("/sync/source/{source_id}", response_model=TaskTriggerResponse)
async def trigger_source_sync(
    source_id: UUID,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_admin)
):
    """Trigger synchronization of a specific data source."""
    try:
        # Trigger source-specific sync task
        task = ingest_from_source.delay(str(source_id))
        
        return TaskTriggerResponse(
            task_id=task.id,
            message=f"Source synchronization started for {source_id}",
            status="triggered"
        )
    
    except Exception as e:
        logger.error(f"Error triggering source sync: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger source synchronization")


@router.post("/vector-store/reset")
async def reset_vector_store(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Reset the vector store (delete all embeddings)."""
    try:
        vector_store = VectorStoreService()
        if not vector_store._initialized:
            await vector_store.initialize()
        
        await vector_store.reset_collection()
        
        # Reset processing status for all documents
        from sqlalchemy import update
        from app.models.document import Document
        
        await db.execute(
            update(Document).values(
                is_processed=False,
                processing_error=None
            )
        )
        await db.commit()
        
        logger.warning("Vector store reset by admin user")
        
        return {"message": "Vector store reset successfully"}
    
    except Exception as e:
        logger.error(f"Error resetting vector store: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset vector store")


@router.get("/vector-store/stats")
async def get_vector_store_stats(
    current_user: User = Depends(require_admin)
):
    """
    Get vector store statistics.
    
    Returns:
        Dictionary with vector store statistics including:
        - total_chunks: Number of chunks in the collection
        - collection_name: Name of the ChromaDB collection
        - embedding_model: Currently active embedding model
        - available_models: List of available embedding models
    """
    try:
        vector_store = VectorStoreService()
        if not vector_store._initialized:
            await vector_store.initialize()
        
        stats = await vector_store.get_collection_stats()
        return stats
    
    except Exception as e:
        log_error(e, context={"endpoint": "vector_store_stats"})
        raise HTTPException(status_code=500, detail="Failed to get vector store statistics")


@router.post("/vector-store/switch-model")
async def switch_embedding_model(
    model_name: str,
    current_user: User = Depends(require_admin)
):
    """
    Switch the embedding model (admin only).
    
    Args:
        model_name: Name of the embedding model to switch to
        current_user: Current authenticated admin user
        
    Returns:
        Success message with model information
        
    Note:
        Changing models requires reprocessing all documents for best results.
    """
    try:
        vector_store = VectorStoreService()
        if not vector_store._initialized:
            await vector_store.initialize()
        
        success = await vector_store.switch_embedding_model(model_name)
        
        if success:
            return {
                "message": f"Switched to embedding model: {model_name}",
                "current_model": vector_store.get_current_model(),
                "warning": "Existing documents should be reprocessed for consistency"
            }
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to switch model. Available models: {settings.EMBEDDING_MODEL_OPTIONS}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, context={"endpoint": "switch_embedding_model", "model": model_name})
        raise HTTPException(status_code=500, detail="Failed to switch embedding model")


@router.get("/tasks/status")
async def get_task_status(
    current_user: User = Depends(require_admin)
):
    """Get status of background tasks."""
    try:
        from app.core.celery import celery_app
        
        # Get active tasks
        active_tasks = celery_app.control.inspect().active()
        
        # Get scheduled tasks
        scheduled_tasks = celery_app.control.inspect().scheduled()
        
        # Get reserved tasks
        reserved_tasks = celery_app.control.inspect().reserved()
        
        return {
            "active_tasks": active_tasks,
            "scheduled_tasks": scheduled_tasks,
            "reserved_tasks": reserved_tasks
        }
    
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get task status")


@router.post("/tasks/purge")
async def purge_tasks(
    current_user: User = Depends(require_admin)
):
    """Purge all pending tasks."""
    try:
        from app.core.celery import celery_app
        
        # Purge all tasks
        celery_app.control.purge()
        
        logger.warning("All pending tasks purged by admin user")
        
        return {"message": "All pending tasks purged"}
    
    except Exception as e:
        logger.error(f"Error purging tasks: {e}")
        raise HTTPException(status_code=500, detail="Failed to purge tasks")


@router.get("/logs")
async def get_system_logs(
    lines: int = 100,
    current_user: User = Depends(require_admin)
):
    """Get recent system logs."""
    try:
        import os
        from app.core.config import settings
        
        log_file = settings.LOG_FILE
        
        if not os.path.exists(log_file):
            return {"logs": [], "message": "Log file not found"}
        
        # Read last N lines
        with open(log_file, 'r') as f:
            log_lines = f.readlines()
        
        # Get last 'lines' number of lines
        recent_logs = log_lines[-lines:] if len(log_lines) > lines else log_lines
        
        return {
            "logs": [line.strip() for line in recent_logs],
            "total_lines": len(log_lines),
            "returned_lines": len(recent_logs)
        }
    
    except Exception as e:
        logger.error(f"Error getting system logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system logs")
