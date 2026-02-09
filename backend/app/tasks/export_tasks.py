"""
Celery tasks for DOCX/PDF export generation.
"""

import asyncio
import json
from datetime import datetime
from typing import Optional
from uuid import UUID

from celery import current_task
from loguru import logger

from app.core.celery import celery_app
from app.core.database import create_celery_session
from app.models.export_job import ExportJob
from app.services.export_service import export_service
from sqlalchemy import select


async def _publish_progress(job_id: str, progress: int, stage: str, status: str, error: Optional[str] = None):
    """Publish progress update to Redis for WebSocket subscribers."""
    import redis.asyncio as redis
    from app.core.config import settings

    try:
        redis_client = redis.from_url(settings.REDIS_URL)
        channel = f"export:{job_id}:progress"

        message = {
            "type": "progress",
            "progress": progress,
            "stage": stage,
            "status": status,
        }
        if error:
            message["error"] = error

        await redis_client.publish(channel, json.dumps(message))
        await redis_client.close()
    except Exception as e:
        logger.warning(f"Failed to publish progress for export job {job_id}: {e}")


async def _process_export_async(job_id: str):
    """Async implementation of export processing."""
    job_uuid = UUID(job_id)
    session_factory = create_celery_session()

    async with session_factory() as db:
        # Load job from database
        result = await db.execute(
            select(ExportJob).where(ExportJob.id == job_uuid)
        )
        job = result.scalar_one_or_none()

        if not job:
            logger.error(f"Export job {job_id} not found")
            return

        # Check if cancelled
        if job.status == "cancelled":
            logger.info(f"Export job {job_id} was cancelled")
            return

        await _publish_progress(job_id, 0, "starting", "processing")

        try:
            # Progress callback that publishes to Redis
            def progress_callback(progress: int, stage: str):
                # Publish progress synchronously (within async context)
                asyncio.create_task(_publish_progress(job_id, progress, stage, "processing"))
                # Update job in database
                job.progress = progress
                job.current_stage = stage

            # Process the export job
            await export_service.process_export_job(
                db=db,
                job_id=job_uuid,
                progress_callback=progress_callback
            )

            await _publish_progress(job_id, 100, "completed", "completed")
            logger.info(f"Export job {job_id} completed successfully")

        except Exception as e:
            # Update job with error
            job.status = "failed"
            job.error = str(e)
            job.completed_at = datetime.utcnow()
            await db.commit()

            await _publish_progress(job_id, job.progress, "error", "failed", str(e))
            logger.error(f"Export job {job_id} failed: {e}")


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def process_export_task(self, job_id: str):
    """
    Celery task for processing an export job.

    This task:
    1. Loads the export job from the database
    2. Retrieves content based on source type
    3. Converts content to document format (DOCX/PDF)
    4. Uploads to MinIO
    5. Updates job status

    Progress is published to Redis for WebSocket subscribers.

    Args:
        job_id: UUID of the ExportJob
    """
    logger.info(f"Starting export task for job {job_id}")

    try:
        # Run the async processing
        asyncio.run(_process_export_async(job_id))

    except Exception as e:
        logger.exception(f"Export task failed for job {job_id}")

        async def _mark_failed():
            job_uuid = UUID(job_id)
            session_factory = create_celery_session()
            async with session_factory() as db:
                result = await db.execute(
                    select(ExportJob).where(ExportJob.id == job_uuid)
                )
                job = result.scalar_one_or_none()
                if job and job.status not in ("completed", "failed", "cancelled"):
                    job.status = "failed"
                    job.error = f"Task error: {str(e)}"
                    job.completed_at = datetime.utcnow()
                    await db.commit()

        try:
            asyncio.run(_mark_failed())
        except Exception:
            logger.warning("Failed to persist export task failure status")

        # Retry on transient errors
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)


@celery_app.task
def cleanup_old_exports(days: int = 30):
    """
    Cleanup task to remove old export jobs and files.

    Removes exports older than the specified number of days.
    Scheduled to run periodically via Celery Beat.

    Args:
        days: Number of days to keep exports
    """
    from datetime import timedelta
    from sqlalchemy import and_

    logger.info(f"Starting cleanup of exports older than {days} days")

    async def _cleanup():
        session_factory = create_celery_session()
        async with session_factory() as db:
            from app.services.storage_service import StorageService
            storage = StorageService()

            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Find old completed/failed jobs
            result = await db.execute(
                select(ExportJob).where(
                    and_(
                        ExportJob.created_at < cutoff_date,
                        ExportJob.status.in_(["completed", "failed", "cancelled"])
                    )
                )
            )
            old_jobs = result.scalars().all()

            deleted_count = 0
            for job in old_jobs:
                try:
                    # Delete file from MinIO
                    if job.file_path:
                        await storage.delete_file(job.file_path)

                    # Delete job from database
                    await db.delete(job)
                    deleted_count += 1

                except Exception as e:
                    logger.warning(f"Failed to cleanup export job {job.id}: {e}")

            await db.commit()
            logger.info(f"Cleaned up {deleted_count} old export jobs")

    asyncio.run(_cleanup())
