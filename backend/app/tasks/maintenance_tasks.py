"""
Background tasks for system maintenance and cleanup.
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from loguru import logger

from app.core.celery import celery_app
from app.core.database import AsyncSessionLocal
from app.models.chat import ChatSession


@celery_app.task(name="app.tasks.maintenance_tasks.cleanup_old_data")
def cleanup_old_data() -> Dict[str, Any]:
    """Clean up old logs and temporary files."""
    return asyncio.run(_async_cleanup_old_data())


async def _async_cleanup_old_data() -> Dict[str, Any]:
    """Async implementation of data cleanup."""
    try:
        cleanup_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "cleaned_items": 0,
            "freed_space_mb": 0,
            "errors": []
        }
        
        # Clean up old log files (older than 30 days)
        log_dir = "./data/logs"
        if os.path.exists(log_dir):
            cutoff_date = datetime.now() - timedelta(days=30)
            
            for filename in os.listdir(log_dir):
                file_path = os.path.join(log_dir, filename)
                
                if os.path.isfile(file_path):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if file_mtime < cutoff_date:
                        try:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            cleanup_results["cleaned_items"] += 1
                            cleanup_results["freed_space_mb"] += file_size / (1024 * 1024)
                            logger.info(f"Deleted old log file: {filename}")
                        except Exception as e:
                            cleanup_results["errors"].append(f"Failed to delete {filename}: {str(e)}")
        
        # Clean up old chat sessions (older than 1 year, inactive)
        async with AsyncSessionLocal() as db:
            cutoff_date = datetime.utcnow() - timedelta(days=365)
            
            old_sessions_result = await db.execute(
                select(ChatSession).where(
                    and_(
                        ChatSession.last_message_at < cutoff_date,
                        ChatSession.is_active == False
                    )
                )
            )
            old_sessions = old_sessions_result.scalars().all()
            
            for session in old_sessions:
                try:
                    await db.delete(session)
                    cleanup_results["cleaned_items"] += 1
                except Exception as e:
                    cleanup_results["errors"].append(f"Failed to delete session {session.id}: {str(e)}")
            
            if old_sessions:
                await db.commit()
                logger.info(f"Deleted {len(old_sessions)} old chat sessions")
        
        cleanup_results["freed_space_mb"] = round(cleanup_results["freed_space_mb"], 2)
        
        logger.info(f"Cleanup completed: {cleanup_results['cleaned_items']} items, "
                   f"{cleanup_results['freed_space_mb']} MB freed")
        
        return cleanup_results
        
    except Exception as e:
        logger.error(f"Error in cleanup task: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

