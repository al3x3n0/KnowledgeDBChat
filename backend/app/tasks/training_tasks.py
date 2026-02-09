"""
Celery tasks for AI Hub training.

Handles background execution of training jobs, including:
- Training job execution with progress tracking
- Dataset validation
- Dataset generation from documents
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime
from typing import Optional
from uuid import UUID

from celery import current_task
from loguru import logger

from app.core.celery import celery_app
from app.core.config import settings
from app.core.database import create_celery_session
from app.models.training_job import TrainingJob, TrainingJobStatus
from app.models.training_dataset import TrainingDataset
from app.services.trainers.base_trainer import TrainingProgress
from sqlalchemy import select
from sqlalchemy.orm import selectinload


def _run_async(coroutine):
    """Run async coroutine in sync context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # Create new loop for this thread
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coroutine)
        finally:
            new_loop.close()
    else:
        return loop.run_until_complete(coroutine)


async def _publish_training_progress(
    job_id: str,
    progress: int,
    status: str,
    current_step: Optional[int] = None,
    total_steps: Optional[int] = None,
    current_epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
    current_loss: Optional[float] = None,
    learning_rate: Optional[float] = None,
    eta_seconds: Optional[int] = None,
    error: Optional[str] = None,
):
    """Publish training progress update to Redis for WebSocket subscribers."""
    import redis.asyncio as redis

    try:
        redis_client = redis.from_url(settings.REDIS_URL)
        channel = f"training_job:{job_id}:progress"

        message = {
            "type": "progress",
            "job_id": job_id,
            "progress": progress,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if current_step is not None:
            message["current_step"] = current_step
        if total_steps is not None:
            message["total_steps"] = total_steps
        if current_epoch is not None:
            message["current_epoch"] = current_epoch
        if total_epochs is not None:
            message["total_epochs"] = total_epochs
        if current_loss is not None:
            message["current_loss"] = current_loss
        if learning_rate is not None:
            message["learning_rate"] = learning_rate
        if eta_seconds is not None:
            message["eta_seconds"] = eta_seconds
        if error:
            message["error"] = error

        await redis_client.publish(channel, json.dumps(message))
        await redis_client.close()
    except Exception as e:
        logger.warning(f"Failed to publish training progress for job {job_id}: {e}")


async def _execute_training_async(job_id: str, user_id: str):
    """Async implementation of training job execution."""
    job_uuid = UUID(job_id)
    session_factory = create_celery_session()

    async with session_factory() as db:
        # Load job from database
        result = await db.execute(
            select(TrainingJob)
            .options(selectinload(TrainingJob.dataset))
            .where(TrainingJob.id == job_uuid)
        )
        job = result.scalar_one_or_none()

        if not job:
            logger.error(f"Training job {job_id} not found")
            return

        # Check if cancelled
        if job.status == TrainingJobStatus.CANCELLED.value:
            logger.info(f"Training job {job_id} was cancelled")
            return

        # Store celery task ID
        if current_task:
            job.celery_task_id = current_task.request.id
            await db.commit()

        # Update status to preparing
        job.status = TrainingJobStatus.PREPARING.value
        job.started_at = datetime.utcnow()
        await db.commit()

        await _publish_training_progress(job_id, 0, TrainingJobStatus.PREPARING.value)

        try:
            # Get trainer
            from app.services.training_service import training_service

            trainer = training_service.get_trainer(job.training_backend)
            if not trainer:
                raise ValueError(f"Trainer '{job.training_backend}' not available")

            # Check if dataset is exported
            dataset = job.dataset
            if not dataset or not dataset.file_path:
                # Export dataset first
                from app.services.training_dataset_service import training_dataset_service
                await training_dataset_service.export_to_jsonl(db, dataset.id)
                await db.refresh(dataset)

            # Download dataset from MinIO to temp file
            from app.services.storage_service import storage_service

            dataset_content = await storage_service.download_file(dataset.file_path)

            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".jsonl",
                delete=False,
            ) as f:
                f.write(dataset_content)
                local_dataset_path = f.name

            # Create output directory
            output_dir = os.path.join(
                settings.TRAINING_OUTPUT_DIR,
                str(job.user_id),
                str(job.id),
            )
            os.makedirs(output_dir, exist_ok=True)

            # Update status to training
            job.status = TrainingJobStatus.TRAINING.value
            await db.commit()

            await _publish_training_progress(job_id, 5, TrainingJobStatus.TRAINING.value)

            # Progress bridging: trainer runs in a thread; we must not touch the async DB session from that thread.
            progress_queue: asyncio.Queue[TrainingProgress | None] = asyncio.Queue()
            loop = asyncio.get_running_loop()

            async def progress_consumer():
                while True:
                    progress = await progress_queue.get()
                    if progress is None:
                        return

                    # Stop quickly if job got cancelled
                    try:
                        await db.refresh(job)
                        if job.status == TrainingJobStatus.CANCELLED.value:
                            logger.info(f"Training job {job_id} cancelled during training")
                            return
                    except Exception:
                        pass

                    # Persist progress to DB
                    try:
                        await training_service.update_job_progress(db, job_uuid, progress)
                    except Exception as e:
                        logger.warning(f"Failed to update training job progress in DB: {e}")

                    # Publish progress (cap below 100; completion is a separate event)
                    pct = int((progress.step / progress.total_steps) * 95) if progress.total_steps > 0 else 0
                    await _publish_training_progress(
                        job_id=job_id,
                        progress=pct,
                        status=TrainingJobStatus.TRAINING.value,
                        current_step=progress.step,
                        total_steps=progress.total_steps,
                        current_epoch=progress.epoch,
                        total_epochs=progress.total_epochs,
                        current_loss=progress.loss,
                        learning_rate=progress.learning_rate,
                        eta_seconds=progress.eta_seconds,
                    )

                    # Update Celery task state
                    if current_task:
                        current_task.update_state(
                            state="PROGRESS",
                            meta={
                                "progress": pct,
                                "step": progress.step,
                                "total_steps": progress.total_steps,
                                "loss": progress.loss,
                            },
                        )

            consumer_task: Optional[asyncio.Task] = asyncio.create_task(progress_consumer())

            def progress_callback(progress: TrainingProgress):
                try:
                    loop.call_soon_threadsafe(progress_queue.put_nowait, progress)
                except Exception:
                    # If the loop is gone, just drop progress updates
                    pass

            # Define cancellation check
            def cancel_check() -> bool:
                # Check Redis for cancellation signal
                import redis
                r = redis.from_url(settings.REDIS_URL)
                return r.get(f"training_job:{job_id}:cancel") is not None

            # Get hyperparameters
            hyperparameters = job.get_hyperparameters()
            hyperparameters["training_method"] = job.training_method

            # Execute training
            logger.info(f"Starting training for job {job_id}")
            try:
                training_result = await trainer.train(
                    job_id=job_uuid,
                    dataset_path=local_dataset_path,
                    base_model=job.base_model,
                    output_path=output_dir,
                    hyperparameters=hyperparameters,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                )
            finally:
                # Stop consumer task even if training errors out
                await progress_queue.put(None)
                try:
                    if consumer_task:
                        await consumer_task
                except Exception:
                    pass

            # Clean up temp file
            try:
                os.unlink(local_dataset_path)
            except Exception:
                pass

            if training_result.success:
                # Update status to saving (adapter upload/registration)
                job.status = TrainingJobStatus.SAVING.value
                await db.commit()

                await _publish_training_progress(job_id, 95, TrainingJobStatus.SAVING.value)

                # Upload adapter to MinIO
                adapter_minio_path = f"training/adapters/{job.user_id}/{job.id}"

                if training_result.adapter_path and os.path.exists(training_result.adapter_path):
                    # Upload all files in adapter directory
                    for root, dirs, files in os.walk(training_result.adapter_path):
                        for file in files:
                            local_path = os.path.join(root, file)
                            rel_path = os.path.relpath(local_path, training_result.adapter_path)
                            minio_path = f"{adapter_minio_path}/{rel_path}"

                            with open(local_path, "rb") as f:
                                await storage_service.upload_file(
                                    object_name=minio_path,
                                    data=f.read(),
                                )

                # Complete job with adapter
                await training_service.complete_job(
                    db=db,
                    job_id=job_uuid,
                    adapter_path=adapter_minio_path,
                    adapter_size=training_result.adapter_size or 0,
                    final_metrics=training_result.metrics,
                )

                await _publish_training_progress(
                    job_id=job_id,
                    progress=100,
                    status=TrainingJobStatus.COMPLETED.value,
                )

                logger.info(f"Training job {job_id} completed successfully")

            else:
                # Training failed
                await training_service.fail_job(
                    db=db,
                    job_id=job_uuid,
                    error=training_result.error or "Unknown error",
                )

                await _publish_training_progress(
                    job_id=job_id,
                    progress=job.progress,
                    status=TrainingJobStatus.FAILED.value,
                    error=training_result.error,
                )

        except Exception as e:
            logger.exception(f"Training job {job_id} failed: {e}")

            from app.services.training_service import training_service
            await training_service.fail_job(
                db=db,
                job_id=job_uuid,
                error=str(e),
            )

            await _publish_training_progress(
                job_id=job_id,
                progress=job.progress if job else 0,
                status=TrainingJobStatus.FAILED.value,
                error=str(e),
            )


@celery_app.task(bind=True, name="app.tasks.training_tasks.execute_training_job")
def execute_training_job_task(self, job_id: str, user_id: str):
    """
    Celery task for executing a training job.

    This is the main entry point for training job execution.
    """
    logger.info(f"Starting training job execution for {job_id}")

    try:
        _run_async(_execute_training_async(job_id, user_id))
    except Exception as e:
        logger.exception(f"Training task failed for job {job_id}: {e}")
        raise


@celery_app.task(name="app.tasks.training_tasks.validate_dataset")
def validate_dataset_task(dataset_id: str, user_id: str):
    """Validate a training dataset asynchronously."""

    async def _validate():
        from app.services.training_dataset_service import training_dataset_service

        session_factory = create_celery_session()
        async with session_factory() as db:
            result = await training_dataset_service.validate_dataset(
                db=db,
                dataset_id=UUID(dataset_id),
            )
            logger.info(f"Dataset {dataset_id} validation: valid={result.is_valid}")
            return result.is_valid

    return _run_async(_validate())


@celery_app.task(name="app.tasks.training_tasks.generate_dataset_from_documents")
def generate_dataset_from_documents_task(
    user_id: str,
    name: str,
    description: str,
    document_ids: list,
    dataset_type: str = "instruction",
    samples_per_document: int = 5,
):
    """Generate training dataset from documents using LLM."""

    async def _generate():
        from app.services.training_dataset_service import training_dataset_service
        from app.schemas.training import GenerateDatasetRequest

        session_factory = create_celery_session()
        async with session_factory() as db:
            request = GenerateDatasetRequest(
                name=name,
                description=description,
                document_ids=[UUID(doc_id) for doc_id in document_ids],
                dataset_type=dataset_type,
                samples_per_document=samples_per_document,
            )

            dataset = await training_dataset_service.generate_from_documents(
                db=db,
                user_id=UUID(user_id),
                request=request,
            )

            logger.info(f"Generated dataset {dataset.id} with {dataset.sample_count} samples")
            return str(dataset.id)

    return _run_async(_generate())


@celery_app.task(name="app.tasks.training_tasks.cleanup_old_checkpoints")
def cleanup_old_checkpoints_task(job_id: str, keep_last: int = 3):
    """Clean up old training checkpoints, keeping only the most recent."""

    async def _cleanup():
        from app.services.storage_service import storage_service

        session_factory = create_celery_session()
        async with session_factory() as db:
            from app.models.training_job import TrainingCheckpoint

            result = await db.execute(
                select(TrainingCheckpoint)
                .where(TrainingCheckpoint.job_id == UUID(job_id))
                .order_by(TrainingCheckpoint.step.desc())
            )
            checkpoints = list(result.scalars().all())

            if len(checkpoints) <= keep_last:
                return 0

            deleted = 0
            for checkpoint in checkpoints[keep_last:]:
                if checkpoint.checkpoint_path:
                    try:
                        await storage_service.delete_file(checkpoint.checkpoint_path)
                    except Exception as e:
                        logger.warning(f"Failed to delete checkpoint file: {e}")

                await db.delete(checkpoint)
                deleted += 1

            await db.commit()
            logger.info(f"Cleaned up {deleted} old checkpoints for job {job_id}")
            return deleted

    return _run_async(_cleanup())


@celery_app.task(name="app.tasks.training_tasks.export_dataset")
def export_dataset_task(dataset_id: str, user_id: str):
    """Export a dataset to JSONL format in MinIO."""

    async def _export():
        from app.services.training_dataset_service import training_dataset_service

        session_factory = create_celery_session()
        async with session_factory() as db:
            file_path = await training_dataset_service.export_to_jsonl(
                db=db,
                dataset_id=UUID(dataset_id),
            )
            logger.info(f"Exported dataset {dataset_id} to {file_path}")
            return file_path

    return _run_async(_export())
