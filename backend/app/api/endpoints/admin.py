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
from app.tasks.ingestion_tasks import dry_run_source as dry_run_task
from app.core.celery import celery_app
from celery.result import AsyncResult
from croniter import croniter
from datetime import datetime, timedelta
from app.tasks.monitoring_tasks import health_check, generate_stats, _async_generate_stats
from app.utils.ingestion_state import (
    set_ingestion_task_mapping,
    get_ingestion_task_mapping,
    set_ingestion_cancel_flag,
    set_force_full_flag,
)
from app.services.vector_store import VectorStoreService
from app.schemas.admin import (
    SystemStatsResponse,
    HealthCheckResponse,
    TaskTriggerResponse
)
from app.core.feature_flags import get_flags as get_feature_flags, set_flag as set_feature_flag, set_str as set_feature_str
from app.services.llm_service import LLMService
from fastapi import WebSocket, WebSocketDisconnect

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
        try:
            result = task.get(timeout=30)  # Wait up to 30 seconds
            return SystemStatsResponse(**result)
        except Exception as task_error:
            logger.warning(f"Stat task failed or timed out, falling back to direct computation: {task_error}")
            # Fallback: compute stats directly to avoid total failure when Celery unavailable
            try:
                fallback_result = await _async_generate_stats()
                return SystemStatsResponse(**fallback_result)
            except Exception as fallback_error:
                log_error(fallback_error, context={"endpoint": "system_stats_fallback"})
                raise HTTPException(status_code=500, detail="Failed to generate system statistics")
    
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, context={"endpoint": "system_stats"})
        raise HTTPException(status_code=500, detail="Failed to get system statistics")


@router.get("/flags")
async def get_flags(current_user: User = Depends(require_admin)):
    """Get runtime feature flags (admin)."""
    try:
        return await get_feature_flags()
    except Exception as e:
        logger.error(f"Error getting flags: {e}")
        raise HTTPException(status_code=500, detail="Failed to get flags")


@router.post("/flags")
async def update_flags(
    flags: dict,
    current_user: User = Depends(require_admin)
):
    """Update runtime feature flags (admin)."""
    try:
        updated = {}
        for name, val in flags.items():
            if name in {"knowledge_graph_enabled", "summarization_enabled", "auto_summarize_on_process"}:
                ok = await set_feature_flag(name, bool(val))
                updated[name] = ok
        return {"updated": updated}
    except Exception as e:
        logger.error(f"Error updating flags: {e}")
        raise HTTPException(status_code=500, detail="Failed to update flags")


@router.get("/llm/models")
async def list_llm_models(current_user: User = Depends(require_admin)):
    try:
        svc = LLMService()
        models = await svc.list_available_models()
        # Extract names for brevity
        names = [m.get('name') for m in models if isinstance(m, dict)]
        return {"models": names, "default_model": svc.default_model}
    except Exception as e:
        logger.error(f"Error listing LLM models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list LLM models")


@router.post("/llm/switch-model")
async def switch_llm_model(model_name: str, current_user: User = Depends(require_admin)):
    """Switch the default LLM model (runtime flag)."""
    try:
        # Save to feature flags; LLMService reads this at call time
        ok = await set_feature_str("llm_default_model", model_name)
        if not ok:
            raise HTTPException(status_code=400, detail="Invalid model name")
        return {"message": "LLM model updated", "model": model_name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching LLM model: {e}")
        raise HTTPException(status_code=500, detail="Failed to switch LLM model")


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
    current_user: User = Depends(require_admin),
    force_full: bool = False,
):
    """Trigger synchronization of a specific data source."""
    try:
        # Optionally set force_full flag for ingestion
        if force_full:
            await set_force_full_flag(str(source_id), ttl=600)
        # Trigger source-specific sync task
        task = ingest_from_source.delay(str(source_id))
        # Map source->task for cancellation
        await set_ingestion_task_mapping(str(source_id), task.id, ttl=3600)
        
        return TaskTriggerResponse(
            task_id=task.id,
            message=f"Source synchronization started for {source_id}",
            status="triggered"
        )
    
    except Exception as e:
        logger.error(f"Error triggering source sync: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger source synchronization")


@router.websocket("/sources/{source_id}/ingestion-progress")
async def ingestion_progress_websocket(websocket: WebSocket, source_id: UUID):
    """WebSocket endpoint for real-time ingestion progress updates (admin only)."""
    from app.utils.websocket_auth import require_websocket_auth
    from app.utils.websocket_manager import websocket_manager
    try:
        user = await require_websocket_auth(websocket)
        if not user.is_admin():
            await websocket.close(code=1008, reason="Admin required")
            return
    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close(code=1008, reason="Authentication failed")
        return
    await websocket_manager.connect(websocket, str(source_id))
    try:
        while True:
            try:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break
    finally:
        websocket_manager.disconnect(websocket, str(source_id))


@router.post("/sources/{source_id}/clear-error")
async def clear_source_error(
    source_id: UUID,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Clear last_error for a document source (admin)."""
    try:
        from sqlalchemy import select
        from app.models.document import DocumentSource as _DS
        result = await db.execute(select(_DS).where(_DS.id == source_id))
        src = result.scalar_one_or_none()
        if not src:
            raise HTTPException(status_code=404, detail="Source not found")
        src.last_error = None
        await db.commit()
        return {"message": "Cleared last error"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing source error: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear error")


@router.post("/sources/{source_id}/dry-run")
async def dry_run_source(
    source_id: UUID,
    current_user: User = Depends(require_admin),
    overrides: Dict[str, Any] | None = None,
):
    """Run a dry-run ingestion; returns counts and a small sample inline."""
    try:
        task = dry_run_task.delay(str(source_id), overrides or {})
        result = task.get(timeout=60)
        return result
    except Exception as e:
        logger.error(f"Dry-run failed: {e}")
        raise HTTPException(status_code=500, detail="Dry-run failed")


@router.post("/sources/{source_id}/cancel")
async def cancel_source_sync(
    source_id: UUID,
    current_user: User = Depends(require_admin)
):
    """Cancel an active ingestion task for a source (best-effort)."""
    try:
        task_id = None
        task_id = await get_ingestion_task_mapping(str(source_id))
        # Set cancel flag so task loop can exit gracefully
        await set_ingestion_cancel_flag(str(source_id), ttl=600)
        if task_id:
            try:
                AsyncResult(task_id, app=celery_app).revoke(terminate=True)
            except Exception:
                pass
        return {"message": "Cancellation requested", "task_id": task_id}
    except Exception as e:
        logger.error(f"Cancel sync failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to request cancellation")


@router.get("/sources/next-run")
async def get_sources_next_run(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Return next scheduled run time for all active sources based on config (cron or interval)."""
    try:
        from sqlalchemy import select
        from app.models.document import DocumentSource as _DS
        res = await db.execute(select(_DS).where(_DS.is_active == True))
        sources = res.scalars().all()
        now = datetime.utcnow()
        items = []
        for s in sources:
            cfg = s.config or {}
            if not cfg.get('auto_sync'):
                items.append({"source_id": str(s.id), "next_run": None})
                continue
            next_run = None
            cron_expr = cfg.get('cron')
            interval_min = int(cfg.get('sync_interval_minutes', 0) or 0)
            try:
                if cron_expr:
                    it = croniter(cron_expr, now)
                    nr = it.get_next(datetime)
                    next_run = nr.isoformat()
                elif interval_min > 0:
                    last = s.last_sync
                    base = last or now
                    nr = base + timedelta(minutes=interval_min)
                    if nr < now:
                        nr = now + timedelta(minutes=interval_min)
                    next_run = nr.isoformat()
            except Exception:
                next_run = None
            items.append({"source_id": str(s.id), "next_run": next_run})
        return {"items": items, "count": len(items)}
    except Exception as e:
        logger.error(f"Get next-run failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute next run")


@router.post("/validate-cron")
async def validate_cron(cron: str, current_user: User = Depends(require_admin)):
    """Validate a cron expression and return the next run timestamp if valid."""
    try:
        now = datetime.utcnow()
        it = croniter(cron, now)
        nr = it.get_next(datetime)
        return {"valid": True, "next_run": nr.isoformat()}
    except Exception as e:
        return {"valid": False, "error": str(e)}


@router.get("/sources/{source_id}/sync-logs")
async def get_source_sync_logs(
    source_id: UUID,
    limit: int = 20,
    offset: int = 0,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Return recent sync logs for a source."""
    try:
        from sqlalchemy import select, desc
        from app.models.document import DocumentSourceSyncLog as _Log
        limit = max(1, min(200, limit))
        offset = max(0, offset)
        res = await db.execute(
            select(_Log)
            .where(_Log.source_id == source_id)
            .order_by(desc(_Log.started_at))
            .offset(offset)
            .limit(limit)
        )
        logs = res.scalars().all()
        items = []
        for l in logs:
            items.append({
                "id": str(l.id),
                "status": l.status,
                "task_id": l.task_id,
                "started_at": l.started_at.isoformat() if l.started_at else None,
                "finished_at": l.finished_at.isoformat() if l.finished_at else None,
                "total_documents": l.total_documents,
                "processed": l.processed,
                "created": l.created,
                "updated": l.updated,
                "errors": l.errors,
                "error_message": l.error_message,
            })
        return {"items": items}
    except Exception as e:
        logger.error(f"Get sync logs failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get sync logs")


@router.get("/sources/{source_id}/sync-logs.csv")
async def export_source_sync_logs_csv(
    source_id: UUID,
    limit: int = 1000,
    offset: int = 0,
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Export sync logs as CSV for a source."""
    try:
        from sqlalchemy import select, desc
        from app.models.document import DocumentSourceSyncLog as _Log
        limit = max(1, min(5000, limit))
        offset = max(0, offset)
        res = await db.execute(
            select(_Log)
            .where(_Log.source_id == source_id)
            .order_by(desc(_Log.started_at))
            .offset(offset)
            .limit(limit)
        )
        logs = res.scalars().all()
        import csv
        import io
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            'id','task_id','status','started_at','finished_at','total_documents','processed','created','updated','errors','error_message'
        ])
        for l in logs:
            writer.writerow([
                str(l.id),
                l.task_id or '',
                l.status or '',
                (l.started_at.isoformat() if l.started_at else ''),
                (l.finished_at.isoformat() if l.finished_at else ''),
                l.total_documents or 0,
                l.processed or 0,
                l.created or 0,
                l.updated or 0,
                l.errors or 0,
                (l.error_message or '').replace('\n',' ').replace('\r',' '),
            ])
        from fastapi import Response
        csv_data = buf.getvalue()
        headers = {
            'Content-Disposition': f'attachment; filename="sync_logs_{source_id}.csv"'
        }
        return Response(content=csv_data, media_type='text/csv', headers=headers)
    except Exception as e:
        logger.error(f"Export sync logs CSV failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to export CSV")


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
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get status of background tasks including ingestion tasks."""
    try:
        from app.core.celery import celery_app
        from sqlalchemy import select
        from app.models.document import DocumentSource as _DocumentSource
        
        # Get active tasks (returns None if no workers available)
        active_tasks = celery_app.control.inspect().active()
        if active_tasks is None:
            active_tasks = {}
        
        # Get scheduled tasks (returns None if no workers available)
        scheduled_tasks = celery_app.control.inspect().scheduled()
        if scheduled_tasks is None:
            scheduled_tasks = {}
        
        # Get reserved tasks (returns None if no workers available)
        reserved_tasks = celery_app.control.inspect().reserved()
        if reserved_tasks is None:
            reserved_tasks = {}
        
        # Also get active ingestion tasks from sources
        ingestion_tasks = []
        try:
            stmt = select(_DocumentSource).where(_DocumentSource.is_syncing == True)
            result = await db.execute(stmt)
            syncing_sources = result.scalars().all()
            
            for source in syncing_sources:
                task_id = None
                try:
                    task_id = await get_ingestion_task_mapping(str(source.id))
                except Exception:
                    pass
                
                ingestion_tasks.append({
                    "source_id": str(source.id),
                    "source_name": source.name,
                    "source_type": source.source_type,
                    "task_id": task_id,
                    "status": "syncing"
                })
            
            # Also check for pending tasks (have task_id but not syncing yet)
            all_sources_stmt = select(_DocumentSource)
            all_result = await db.execute(all_sources_stmt)
            all_sources = all_result.scalars().all()
            
            for source in all_sources:
                if source.is_syncing:
                    continue  # Already included above
                try:
                    task_id = await get_ingestion_task_mapping(str(source.id))
                    if task_id:
                        ingestion_tasks.append({
                            "source_id": str(source.id),
                            "source_name": source.name,
                            "source_type": source.source_type,
                            "task_id": task_id,
                            "status": "pending"
                        })
                except Exception:
                    pass
        except Exception as ing_err:
            logger.warning(f"Error getting ingestion tasks: {ing_err}")
        
        return {
            "active_tasks": active_tasks,
            "scheduled_tasks": scheduled_tasks,
            "reserved_tasks": reserved_tasks,
            "ingestion_tasks": ingestion_tasks
        }
    
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        # Return empty dicts on error instead of raising exception
        return {
            "active_tasks": {},
            "scheduled_tasks": {},
            "reserved_tasks": {},
            "ingestion_tasks": []
        }


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
