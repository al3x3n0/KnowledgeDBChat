"""
Celery tasks for document synthesis.

Handles async execution of synthesis jobs.
"""

import asyncio
import json
from typing import Dict, Any, Optional
from uuid import UUID
from loguru import logger
import redis

from app.core.celery import celery_app
from app.core.config import settings
from app.core.database import create_celery_session
from app.services.synthesis_service import synthesis_service
from app.services.llm_service import UserLLMSettings


async def _load_user_settings(db, user_id: str) -> Optional[UserLLMSettings]:
    """Load user LLM settings from preferences."""
    if not user_id:
        return None
    try:
        from sqlalchemy import select
        from app.models.memory import UserPreferences
        result = await db.execute(
            select(UserPreferences).where(UserPreferences.user_id == UUID(user_id))
        )
        user_prefs = result.scalar_one_or_none()
        if user_prefs:
            return UserLLMSettings.from_preferences(user_prefs)
    except Exception as e:
        logger.debug(f"Could not load user preferences for synthesis task: {e}")
    return None


def _get_redis_client():
    """Get Redis client for progress publishing."""
    try:
        return redis.from_url(settings.REDIS_URL, decode_responses=True)
    except Exception as e:
        logger.warning(f"Failed to connect to Redis: {e}")
        return None


def _publish_progress(job_id: str, progress: int, stage: str):
    """Publish progress update via Redis."""
    try:
        client = _get_redis_client()
        if client:
            channel = f"synthesis_progress:{job_id}"
            msg = json.dumps({
                "type": "progress",
                "job_id": job_id,
                "progress": progress,
                "stage": stage,
            })
            client.publish(channel, msg)
    except Exception as e:
        logger.debug(f"Failed to publish synthesis progress: {e}")


def _publish_complete(job_id: str, result: dict):
    """Publish completion via Redis."""
    try:
        client = _get_redis_client()
        if client:
            channel = f"synthesis_progress:{job_id}"
            msg = json.dumps({
                "type": "complete",
                "job_id": job_id,
                "result": {
                    "word_count": result.get("metadata", {}).get("word_count", 0),
                    "documents_analyzed": result.get("metadata", {}).get("documents_analyzed", 0),
                },
            })
            client.publish(channel, msg)
    except Exception as e:
        logger.debug(f"Failed to publish synthesis complete: {e}")


def _publish_error(job_id: str, error: str):
    """Publish error via Redis."""
    try:
        client = _get_redis_client()
        if client:
            channel = f"synthesis_progress:{job_id}"
            msg = json.dumps({
                "type": "error",
                "job_id": job_id,
                "error": error,
            })
            client.publish(channel, msg)
    except Exception as e:
        logger.debug(f"Failed to publish synthesis error: {e}")


@celery_app.task(bind=True, name="app.tasks.synthesis_tasks.execute_synthesis_task")
def execute_synthesis_task(self, job_id: str, user_id: str) -> Dict[str, Any]:
    """Execute a synthesis job asynchronously."""
    return asyncio.run(_async_execute_synthesis(self, job_id, user_id))


async def _async_execute_synthesis(task, job_id: str, user_id: str) -> Dict[str, Any]:
    """Async implementation of synthesis execution."""
    async with create_celery_session()() as db:
        try:
            logger.info(f"Starting synthesis job {job_id}")

            # Load job
            from sqlalchemy import select
            from app.models.synthesis_job import SynthesisJob, SynthesisJobStatus

            result = await db.execute(
                select(SynthesisJob).where(SynthesisJob.id == UUID(job_id))
            )
            job = result.scalar_one_or_none()

            if not job:
                logger.error(f"Synthesis job {job_id} not found")
                return {"success": False, "error": "Job not found"}

            # Check if cancelled
            if job.status == SynthesisJobStatus.CANCELLED.value:
                logger.info(f"Synthesis job {job_id} was cancelled")
                return {"success": False, "error": "Job cancelled"}

            # Load user settings
            user_settings = await _load_user_settings(db, user_id)

            # Progress callback
            async def progress_callback(progress: int, stage: str):
                _publish_progress(job_id, progress, stage)
                # Update job in database
                job.progress = progress
                job.current_stage = stage
                await db.commit()

            # Execute synthesis
            result = await synthesis_service.execute_synthesis(
                db=db,
                job=job,
                user_settings=user_settings,
                progress_callback=progress_callback,
            )

            _publish_complete(job_id, result)

            logger.info(f"Synthesis job {job_id} completed successfully")
            return {
                "success": True,
                "job_id": job_id,
                "word_count": result.get("metadata", {}).get("word_count", 0),
            }

        except Exception as e:
            logger.error(f"Synthesis job {job_id} failed: {e}")
            _publish_error(job_id, str(e))

            # Update job status
            try:
                from app.models.synthesis_job import SynthesisJob, SynthesisJobStatus
                from datetime import datetime

                result = await db.execute(
                    select(SynthesisJob).where(SynthesisJob.id == UUID(job_id))
                )
                job = result.scalar_one_or_none()
                if job:
                    job.status = SynthesisJobStatus.FAILED.value
                    job.error = str(e)
                    job.completed_at = datetime.utcnow()
                    await db.commit()
            except Exception as e2:
                logger.error(f"Failed to update job status: {e2}")

            return {"success": False, "job_id": job_id, "error": str(e)}


@celery_app.task(name="app.tasks.synthesis_tasks.cleanup_old_synthesis_jobs")
def cleanup_old_synthesis_jobs(days: int = 30) -> Dict[str, Any]:
    """Clean up old synthesis jobs and their files."""
    return asyncio.run(_async_cleanup_old_jobs(days))


async def _async_cleanup_old_jobs(days: int) -> Dict[str, Any]:
    """Async implementation of job cleanup."""
    async with create_celery_session()() as db:
        try:
            from datetime import datetime, timedelta
            from sqlalchemy import select, delete
            from app.models.synthesis_job import SynthesisJob, SynthesisJobStatus
            from app.services.storage_service import storage_service

            cutoff = datetime.utcnow() - timedelta(days=days)

            # Find old completed/failed jobs
            result = await db.execute(
                select(SynthesisJob).where(
                    SynthesisJob.created_at < cutoff,
                    SynthesisJob.status.in_([
                        SynthesisJobStatus.COMPLETED.value,
                        SynthesisJobStatus.FAILED.value,
                        SynthesisJobStatus.CANCELLED.value,
                    ])
                )
            )
            old_jobs = result.scalars().all()

            deleted_count = 0
            files_deleted = 0

            for job in old_jobs:
                try:
                    # Delete file if exists
                    if job.file_path:
                        try:
                            await storage_service.delete_file(job.file_path)
                            files_deleted += 1
                        except Exception as e:
                            logger.warning(f"Failed to delete file {job.file_path}: {e}")

                    await db.delete(job)
                    deleted_count += 1
                except Exception as e:
                    logger.warning(f"Failed to delete job {job.id}: {e}")

            await db.commit()

            logger.info(f"Cleaned up {deleted_count} old synthesis jobs, {files_deleted} files")
            return {
                "success": True,
                "jobs_deleted": deleted_count,
                "files_deleted": files_deleted,
            }

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return {"success": False, "error": str(e)}
