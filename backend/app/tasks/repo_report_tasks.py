"""
Celery tasks for repository report and presentation generation.
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
from app.models.repo_report import RepoReportJob
from app.services.repo_analysis_service import RepoAnalysisService
from app.services.repo_report_generator import RepoReportGenerator
from app.services.repo_presentation_generator import RepoPresentationGenerator
from app.services.storage_service import StorageService
from sqlalchemy import select


class RepoReportGenerationError(Exception):
    """Raised when report generation fails."""
    pass


async def _publish_progress(job_id: str, progress: int, stage: str, status: str, error: Optional[str] = None):
    """Publish progress update to Redis for WebSocket subscribers."""
    import redis.asyncio as redis
    from app.core.config import settings

    try:
        redis_client = redis.from_url(settings.REDIS_URL)
        channel = f"repo_report:{job_id}:progress"

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


async def _generate_repo_report_async(job_id: str, user_id: str):
    """Async implementation of repository report generation."""
    job_uuid = UUID(job_id)
    user_uuid = UUID(user_id)
    session_factory = create_celery_session()
    storage = StorageService()

    async with session_factory() as db:
        # Load job from database
        result = await db.execute(
            select(RepoReportJob).where(RepoReportJob.id == job_uuid)
        )
        job = result.scalar_one_or_none()

        if not job:
            logger.error(f"Repo report job {job_id} not found")
            return

        # Check if cancelled
        if job.status == "cancelled":
            logger.info(f"Repo report job {job_id} was cancelled")
            return

        # Update status to analyzing
        job.status = "analyzing"
        job.current_stage = "Starting analysis"
        job.started_at = datetime.utcnow()
        await db.commit()

        await _publish_progress(job_id, 5, "Starting analysis", "analyzing")

        try:
            # Progress callback
            async def progress_callback(progress: int, stage: str):
                # Check for cancellation
                await db.refresh(job)
                if job.status == "cancelled":
                    raise RepoReportGenerationError("Job cancelled by user")

                job.progress = progress
                job.current_stage = stage
                await db.commit()
                await _publish_progress(job_id, progress, stage, "analyzing")

            # Stage 1: Analyze repository
            analysis_service = RepoAnalysisService()

            if job.source_id:
                # From existing DocumentSource
                analysis = await analysis_service.analyze_from_source(
                    source_id=job.source_id,
                    db=db,
                    sections=job.sections,
                    progress_callback=progress_callback,
                    user_id=user_uuid,
                )
            elif job.adhoc_url:
                # From ad-hoc URL
                analysis = await analysis_service.analyze_from_url(
                    repo_url=job.adhoc_url,
                    token=job.adhoc_token,
                    sections=job.sections,
                    progress_callback=progress_callback,
                    user_id=user_uuid,
                    db=db,
                )
            else:
                raise RepoReportGenerationError("No source_id or adhoc_url provided")

            # Cache analysis data (use mode='json' to serialize datetime objects)
            job.analysis_data = analysis.model_dump(mode='json')
            job.status = "generating"
            job.current_stage = "Analysis complete, generating output"
            await db.commit()

            await _publish_progress(job_id, 45, "Generating output", "generating")

            # Stage 2: Generate output based on format
            output_format = job.output_format.lower()
            file_bytes = None

            if output_format == "pptx":
                generator = RepoPresentationGenerator(
                    style=job.style,
                    custom_theme=job.custom_theme
                )
                file_bytes = await generator.generate_pptx(
                    analysis=analysis,
                    title=job.title,
                    sections=job.sections,
                    slide_count=job.slide_count or 10,
                    include_diagrams=job.include_diagrams,
                    progress_callback=progress_callback
                )
                file_extension = "pptx"
                content_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

            elif output_format == "pdf":
                generator = RepoReportGenerator(
                    style=job.style,
                    custom_theme=job.custom_theme
                )
                file_bytes = await generator.generate_pdf(
                    analysis=analysis,
                    title=job.title,
                    sections=job.sections,
                    progress_callback=progress_callback
                )
                file_extension = "pdf"
                content_type = "application/pdf"

            else:  # docx (default)
                generator = RepoReportGenerator(
                    style=job.style,
                    custom_theme=job.custom_theme
                )
                file_bytes = await generator.generate_docx(
                    analysis=analysis,
                    title=job.title,
                    sections=job.sections,
                    progress_callback=progress_callback
                )
                file_extension = "docx"
                content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

            # Stage 3: Upload to MinIO
            job.current_stage = "Uploading to storage"
            await db.commit()
            await _publish_progress(job_id, 90, "Uploading to storage", "uploading")

            await storage.initialize()
            filename = f"{job.repo_name.replace('/', '_')}_{job.id}.{file_extension}"
            file_path = f"repo_reports/{user_id}/{filename}"

            await storage.upload_file(
                file_path=file_path,
                data=file_bytes,
                content_type=content_type
            )

            # Update job as completed
            job.status = "completed"
            job.file_path = file_path
            job.file_size = len(file_bytes)
            job.progress = 100
            job.current_stage = "Completed"
            job.completed_at = datetime.utcnow()
            await db.commit()

            await _publish_progress(job_id, 100, "Completed", "completed")

            logger.info(f"Repo report job {job_id} completed successfully: {file_path}")

        except RepoReportGenerationError as e:
            # Known error - update job with error message
            job.status = "failed"
            job.error = str(e)
            job.completed_at = datetime.utcnow()
            await db.commit()

            await _publish_progress(job_id, job.progress, job.current_stage or "failed", "failed", str(e))

            logger.error(f"Repo report job {job_id} failed: {e}")

        except Exception as e:
            # Unexpected error
            job.status = "failed"
            job.error = f"Unexpected error: {str(e)}"
            job.completed_at = datetime.utcnow()
            await db.commit()

            await _publish_progress(job_id, job.progress, "error", "failed", str(e))

            logger.exception(f"Repo report job {job_id} failed with unexpected error")


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def generate_repo_report_task(self, job_id: str, user_id: str):
    """
    Celery task for generating a repository report or presentation.

    This task:
    1. Loads the repo report job from the database
    2. Analyzes the repository (via connector or ad-hoc URL)
    3. Generates insights using LLM
    4. Builds the output document (DOCX, PDF, or PPTX)
    5. Uploads to MinIO
    6. Updates job status

    Progress is published to Redis for WebSocket subscribers.

    Args:
        job_id: UUID of the RepoReportJob
        user_id: UUID of the user who created the job
    """
    logger.info(f"Starting repo report generation task for job {job_id}")

    try:
        # Run the async generation
        asyncio.run(_generate_repo_report_async(job_id, user_id))

    except Exception as e:
        logger.exception(f"Repo report task failed for job {job_id}")

        async def _mark_failed():
            job_uuid = UUID(job_id)
            session_factory = create_celery_session()
            async with session_factory() as db:
                result = await db.execute(
                    select(RepoReportJob).where(RepoReportJob.id == job_uuid)
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
            logger.warning("Failed to persist repo report task failure status")

        # Retry on transient errors
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)


@celery_app.task
def cleanup_old_repo_reports(days: int = 30):
    """
    Cleanup task to remove old repository report jobs and files.

    Removes reports older than the specified number of days.
    Scheduled to run periodically via Celery Beat.

    Args:
        days: Number of days to keep reports
    """
    from datetime import timedelta
    from sqlalchemy import and_

    logger.info(f"Starting cleanup of repo reports older than {days} days")

    async def _cleanup():
        session_factory = create_celery_session()
        async with session_factory() as db:
            storage = StorageService()
            await storage.initialize()

            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Find old completed/failed jobs
            result = await db.execute(
                select(RepoReportJob).where(
                    and_(
                        RepoReportJob.created_at < cutoff_date,
                        RepoReportJob.status.in_(["completed", "failed", "cancelled"])
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
                    logger.warning(f"Failed to cleanup repo report job {job.id}: {e}")

            await db.commit()
            logger.info(f"Cleaned up {deleted_count} old repo report jobs")

    asyncio.run(_cleanup())
