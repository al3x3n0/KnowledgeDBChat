"""
Celery tasks for AI-powered presentation generation.
"""

import asyncio
import json
from datetime import datetime
from typing import Optional

from celery import current_task
from loguru import logger

from app.core.celery import celery_app
from app.core.database import async_session_maker
from app.models.presentation import PresentationJob
from app.services.presentation_generator import PresentationGeneratorService, PresentationGenerationError
from sqlalchemy import select


async def _publish_progress(job_id: str, progress: int, stage: str, status: str, error: Optional[str] = None):
    """Publish progress update to Redis for WebSocket subscribers."""
    import redis.asyncio as redis
    from app.core.config import settings

    try:
        redis_client = redis.from_url(settings.REDIS_URL)
        channel = f"presentation:{job_id}:progress"

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
        logger.warning(f"Failed to publish progress for job {job_id}: {e}")


async def _generate_presentation_async(job_id: str, user_id: str):
    """Async implementation of presentation generation."""
    async with async_session_maker() as db:
        # Load job from database
        result = await db.execute(
            select(PresentationJob).where(PresentationJob.id == job_id)
        )
        job = result.scalar_one_or_none()

        if not job:
            logger.error(f"Presentation job {job_id} not found")
            return

        # Check if cancelled
        if job.status == "cancelled":
            logger.info(f"Presentation job {job_id} was cancelled")
            return

        # Update status to generating
        job.status = "generating"
        job.current_stage = "starting"
        await db.commit()

        await _publish_progress(job_id, 0, "starting", "generating")

        try:
            # Initialize generator service
            generator = PresentationGeneratorService()

            # Progress callback that publishes to Redis
            async def progress_callback(progress: int, stage: str):
                # Check for cancellation
                await db.refresh(job)
                if job.status == "cancelled":
                    raise PresentationGenerationError("Job cancelled by user")

                await _publish_progress(job_id, progress, stage, "generating")

            # Generate the presentation
            file_path = await generator.generate_presentation(
                job=job,
                db=db,
                progress_callback=progress_callback
            )

            # Update job as completed
            job.status = "completed"
            job.file_path = file_path
            job.completed_at = datetime.utcnow()
            await db.commit()

            await _publish_progress(job_id, 100, "completed", "completed")

            logger.info(f"Presentation job {job_id} completed successfully: {file_path}")

        except PresentationGenerationError as e:
            # Known error - update job with error message
            job.status = "failed"
            job.error = str(e)
            job.completed_at = datetime.utcnow()
            await db.commit()

            await _publish_progress(job_id, job.progress, job.current_stage or "failed", "failed", str(e))

            logger.error(f"Presentation job {job_id} failed: {e}")

        except Exception as e:
            # Unexpected error
            job.status = "failed"
            job.error = f"Unexpected error: {str(e)}"
            job.completed_at = datetime.utcnow()
            await db.commit()

            await _publish_progress(job_id, job.progress, "error", "failed", str(e))

            logger.exception(f"Presentation job {job_id} failed with unexpected error")

        finally:
            # Clean up Mermaid renderer
            try:
                from app.services.mermaid_renderer import get_mermaid_renderer
                renderer = get_mermaid_renderer()
                await renderer.close()
            except:
                pass


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def generate_presentation_task(self, job_id: str, user_id: str):
    """
    Celery task for generating a presentation.

    This task:
    1. Loads the presentation job from the database
    2. Gathers context from documents using RAG
    3. Generates presentation outline with LLM
    4. Generates content for each slide
    5. Renders Mermaid diagrams to PNG
    6. Builds the PPTX file
    7. Uploads to MinIO
    8. Updates job status

    Progress is published to Redis for WebSocket subscribers.

    Args:
        job_id: UUID of the PresentationJob
        user_id: UUID of the user who created the job
    """
    logger.info(f"Starting presentation generation task for job {job_id}")

    try:
        # Run the async generation
        asyncio.run(_generate_presentation_async(job_id, user_id))

    except Exception as e:
        logger.exception(f"Presentation task failed for job {job_id}")

        # Update job status synchronously as fallback
        try:
            from app.core.database import SessionLocal
            with SessionLocal() as db:
                job = db.query(PresentationJob).filter(PresentationJob.id == job_id).first()
                if job and job.status not in ("completed", "failed", "cancelled"):
                    job.status = "failed"
                    job.error = f"Task error: {str(e)}"
                    job.completed_at = datetime.utcnow()
                    db.commit()
        except:
            pass

        # Retry on transient errors
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)


@celery_app.task
def cleanup_old_presentations(days: int = 30):
    """
    Cleanup task to remove old presentation jobs and files.

    Removes presentations older than the specified number of days.
    Scheduled to run periodically via Celery Beat.

    Args:
        days: Number of days to keep presentations
    """
    from datetime import timedelta
    from sqlalchemy import and_

    logger.info(f"Starting cleanup of presentations older than {days} days")

    async def _cleanup():
        async with async_session_maker() as db:
            from app.services.storage_service import StorageService
            storage = StorageService()

            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Find old completed/failed jobs
            result = await db.execute(
                select(PresentationJob).where(
                    and_(
                        PresentationJob.created_at < cutoff_date,
                        PresentationJob.status.in_(["completed", "failed", "cancelled"])
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
                    logger.warning(f"Failed to cleanup presentation job {job.id}: {e}")

            await db.commit()
            logger.info(f"Cleaned up {deleted_count} old presentation jobs")

    asyncio.run(_cleanup())
