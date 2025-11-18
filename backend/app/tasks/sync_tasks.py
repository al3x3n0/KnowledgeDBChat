"""
Sync tasks for automated data source synchronization.
"""

import asyncio
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger

from app.core.celery import celery_app
from app.core.database import AsyncSessionLocal
from app.models.document import DocumentSource
from app.tasks.ingestion_tasks import ingest_from_source


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
    async with AsyncSessionLocal() as db:
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
    async with AsyncSessionLocal() as db:
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







