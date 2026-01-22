"""
Sync tasks for automated data source synchronization.
"""

import asyncio
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger

from app.core.celery import celery_app
from app.core.database import create_celery_session
from app.models.document import DocumentSource
from app.tasks.ingestion_tasks import ingest_from_source
from app.core.cache import cache_service
from croniter import croniter


@celery_app.task(name="app.tasks.sync_tasks.sync_all_gitlab_sources")
def sync_all_gitlab_sources() -> Dict[str, Any]:
    """Sync all active GitLab document sources."""
    return asyncio.run(_async_sync_sources_by_type("gitlab"))


@celery_app.task(name="app.tasks.sync_tasks.sync_all_confluence_sources")
def sync_all_confluence_sources() -> Dict[str, Any]:
    """Sync all active Confluence document sources."""
    return asyncio.run(_async_sync_sources_by_type("confluence"))


@celery_app.task(name="app.tasks.sync_tasks.sync_all_web_sources")
def sync_all_web_sources() -> Dict[str, Any]:
    """Sync all active web document sources."""
    return asyncio.run(_async_sync_sources_by_type("web"))


@celery_app.task(name="app.tasks.sync_tasks.sync_all_sources")
def sync_all_sources() -> Dict[str, Any]:
    """Sync all active document sources."""
    return asyncio.run(_async_sync_all_sources())


async def _async_sync_sources_by_type(source_type: str) -> Dict[str, Any]:
    """Sync all sources of a specific type."""
    async with create_celery_session()() as db:
        try:
            # Get all active sources of the specified type
            result = await db.execute(
                select(DocumentSource).where(
                    DocumentSource.source_type == source_type,
                    DocumentSource.is_active == True
                )
            )
            sources = result.scalars().all()
            
            if not sources:
                logger.info(f"No active {source_type} sources found")
                return {
                    "source_type": source_type,
                    "total_sources": 0,
                    "synced_sources": 0,
                    "failed_sources": 0,
                    "success": True
                }
            
            logger.info(f"Starting sync for {len(sources)} {source_type} sources")
            
            synced = 0
            failed = 0
            results = []
            
            for source in sources:
                try:
                    # Trigger ingestion task
                    task_result = ingest_from_source.delay(str(source.id))
                    
                    # For now, we'll just mark as triggered
                    # In a real implementation, you might want to wait for results
                    synced += 1
                    results.append({
                        "source_id": str(source.id),
                        "source_name": source.name,
                        "task_id": task_result.id,
                        "status": "triggered"
                    })
                    
                    logger.info(f"Triggered sync for {source_type} source: {source.name}")
                    
                except Exception as e:
                    failed += 1
                    logger.error(f"Failed to trigger sync for {source_type} source {source.name}: {e}")
                    results.append({
                        "source_id": str(source.id),
                        "source_name": source.name,
                        "error": str(e),
                        "status": "failed"
                    })
            
            return {
                "source_type": source_type,
                "total_sources": len(sources),
                "synced_sources": synced,
                "failed_sources": failed,
                "results": results,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error in sync task for {source_type} sources: {e}")
            return {
                "source_type": source_type,
                "error": str(e),
                "success": False
            }


async def _async_sync_all_sources() -> Dict[str, Any]:
    """Sync all active document sources."""
    async with create_celery_session()() as db:
        try:
            # Get all active sources
            result = await db.execute(
                select(DocumentSource).where(DocumentSource.is_active == True)
            )
            sources = result.scalars().all()
            
            if not sources:
                logger.info("No active sources found")
                return {
                    "total_sources": 0,
                    "synced_sources": 0,
                    "failed_sources": 0,
                    "success": True
                }
            
            logger.info(f"Starting sync for {len(sources)} sources")
            
            synced = 0
            failed = 0
            results_by_type = {}
            
            for source in sources:
                try:
                    # Trigger ingestion task
                    task_result = ingest_from_source.delay(str(source.id))
                    
                    synced += 1
                    
                    # Group results by source type
                    if source.source_type not in results_by_type:
                        results_by_type[source.source_type] = []
                    
                    results_by_type[source.source_type].append({
                        "source_id": str(source.id),
                        "source_name": source.name,
                        "task_id": task_result.id,
                        "status": "triggered"
                    })
                    
                    logger.info(f"Triggered sync for {source.source_type} source: {source.name}")
                    
                except Exception as e:
                    failed += 1
                    logger.error(f"Failed to trigger sync for source {source.name}: {e}")
                    
                    if source.source_type not in results_by_type:
                        results_by_type[source.source_type] = []
                    
                    results_by_type[source.source_type].append({
                        "source_id": str(source.id),
                        "source_name": source.name,
                        "error": str(e),
                        "status": "failed"
                    })
            
            return {
                "total_sources": len(sources),
                "synced_sources": synced,
                "failed_sources": failed,
                "results_by_type": results_by_type,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error in sync all sources task: {e}")
            return {
                "error": str(e),
                "success": False
            }


@celery_app.task(name="app.tasks.sync_tasks.scan_scheduled_sources")
def scan_scheduled_sources() -> Dict[str, Any]:
    """Scan sources for auto-sync scheduling and trigger ingestion as needed."""
    return asyncio.run(_async_scan_scheduled_sources())


async def _async_scan_scheduled_sources() -> Dict[str, Any]:
    async with create_celery_session()() as db:
        try:
            result = await db.execute(select(DocumentSource).where(DocumentSource.is_active == True))
            sources = result.scalars().all()
            triggered = []
            from datetime import datetime, timedelta
            now = datetime.utcnow()
            for src in sources:
                try:
                    cfg = src.config or {}
                    auto = bool(cfg.get('auto_sync', False))
                    interval_min = int(cfg.get('sync_interval_minutes', 0) or 0)
                    cron_expr = cfg.get('cron')
                    if not auto:
                        continue
                    if getattr(src, 'is_syncing', False):
                        continue
                    last = src.last_sync
                    due = False
                    # Interval-based due
                    if interval_min and interval_min > 0:
                        due = (last is None) or (now - last >= timedelta(minutes=interval_min))
                    # Cron-based due
                    if cron_expr:
                        try:
                            it = croniter(cron_expr, now)
                            last_sched = it.get_prev(datetime)
                            if last is None or last < last_sched:
                                due = True
                        except Exception as ce:
                            logger.warning(f"Invalid cron for source {getattr(src,'name','?')}: {cron_expr} ({ce})")
                    if not due:
                        continue
                    # Trigger ingestion
                    # Respect sync_only_changed flag to optionally force full
                    try:
                        sync_only_changed = bool(cfg.get('sync_only_changed', True))
                        if not sync_only_changed:
                            await cache_service.set(f"ingestion:force_full:{src.id}", 1, ttl=600)
                    except Exception:
                        pass
                    task_res = ingest_from_source.delay(str(src.id))
                    try:
                        await cache_service.set(f"ingestion:task:{src.id}", task_res.id, ttl=3600)
                    except Exception:
                        pass
                    triggered.append({"source_id": str(src.id), "task_id": task_res.id})
                except Exception as e:
                    logger.warning(f"Auto-sync check failed for source {getattr(src,'name','?')}: {e}")
                    continue
            return {"triggered": triggered, "count": len(triggered), "success": True}
        except Exception as e:
            logger.error(f"Scheduled scan failed: {e}")
            return {"success": False, "error": str(e)}







