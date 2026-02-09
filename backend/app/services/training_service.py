"""
Training Service for AI Hub.

Orchestrates training jobs: creation, execution, monitoring, and cancellation.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import settings
from app.models.training_job import (
    TrainingJob,
    TrainingCheckpoint,
    TrainingJobStatus,
    TrainingMethod,
    TrainingBackend,
)
from app.models.training_dataset import TrainingDataset, DatasetStatus
from app.models.model_registry import ModelAdapter, AdapterStatus, AdapterType
from app.schemas.training import TrainingJobCreate, HyperparametersConfig
from app.services.trainers import LocalTrainer, SimulatedTrainer
from app.services.trainers.base_trainer import TrainingProgress


class TrainingService:
    """Service for managing training jobs."""

    def __init__(self):
        self._trainers: Dict[str, Any] = {}
        self._init_trainers()

    def _init_trainers(self):
        """Initialize available training backends."""
        # Simulated trainer (always available; demo/dev happy-path)
        self._trainers["simulated"] = SimulatedTrainer()
        logger.info("Simulated trainer initialized")

        # Local trainer (PEFT-based)
        local_trainer = LocalTrainer()
        if local_trainer.is_available():
            self._trainers["local"] = local_trainer
            logger.info("Local trainer initialized")
        else:
            logger.warning("Local trainer not available (missing dependencies)")

    def get_available_backends(self) -> List[str]:
        """Get list of available training backends."""
        return list(self._trainers.keys())

    def get_trainer(self, backend: str):
        """Get a trainer by backend name."""
        return self._trainers.get(backend)

    async def create_training_job(
        self,
        db: AsyncSession,
        user_id: UUID,
        data: TrainingJobCreate,
    ) -> TrainingJob:
        """
        Create a new training job.

        Args:
            db: Database session
            user_id: Owner user ID
            data: Job creation data

        Returns:
            Created TrainingJob
        """
        # Validate dataset exists and is ready
        dataset = await db.get(TrainingDataset, data.dataset_id)
        if not dataset:
            raise ValueError(f"Dataset {data.dataset_id} not found")
        if not dataset.is_ready_for_training():
            raise ValueError(f"Dataset is not ready for training (status: {dataset.status})")

        # Validate training backend
        if data.training_backend not in self._trainers:
            available = ", ".join(self.get_available_backends()) or "none"
            raise ValueError(
                f"Training backend '{data.training_backend}' not available. "
                f"Available backends: {available}"
            )

        # Validate training method for currently supported trainers
        if data.training_method not in [TrainingMethod.LORA.value, TrainingMethod.QLORA.value]:
            raise ValueError(
                f"Training method '{data.training_method}' not supported. "
                f"Supported: {TrainingMethod.LORA.value}, {TrainingMethod.QLORA.value}"
            )

        # Get default hyperparameters if not provided
        hyperparameters = {}
        if data.hyperparameters:
            hyperparameters = data.hyperparameters.model_dump()
        else:
            trainer = self._trainers.get(data.training_backend)
            if trainer:
                hyperparameters = trainer.get_default_hyperparameters(data.base_model)

        # Get resource config
        resource_config = {}
        if data.resource_config:
            resource_config = data.resource_config.model_dump()

        # Create job
        job = TrainingJob(
            name=data.name,
            description=data.description,
            training_method=data.training_method,
            training_backend=data.training_backend,
            base_model=data.base_model,
            base_model_provider=data.base_model_provider,
            dataset_id=data.dataset_id,
            hyperparameters=hyperparameters,
            resource_config=resource_config,
            user_id=user_id,
            status=TrainingJobStatus.PENDING.value,
            total_epochs=hyperparameters.get("num_epochs", 3),
        )

        db.add(job)
        await db.commit()
        await db.refresh(job)

        logger.info(f"Created training job {job.id} for user {user_id}")

        # Start immediately if requested
        if data.start_immediately:
            await self.start_job(db, job.id)

        return job

    async def get_job(
        self,
        db: AsyncSession,
        job_id: UUID,
        user_id: Optional[UUID] = None,
    ) -> Optional[TrainingJob]:
        """Get a training job by ID."""
        query = select(TrainingJob).where(TrainingJob.id == job_id)

        if user_id:
            query = query.where(TrainingJob.user_id == user_id)

        query = query.options(selectinload(TrainingJob.dataset))

        result = await db.execute(query)
        return result.scalar_one_or_none()

    async def list_jobs(
        self,
        db: AsyncSession,
        user_id: UUID,
        status: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[List[TrainingJob], int]:
        """List training jobs for a user."""
        query = select(TrainingJob).where(TrainingJob.user_id == user_id)

        if status:
            query = query.where(TrainingJob.status == status)

        # Count total
        count_query = select(func.count()).select_from(query.subquery())
        count_result = await db.execute(count_query)
        total = count_result.scalar() or 0

        # Apply pagination
        query = query.options(selectinload(TrainingJob.dataset))
        query = query.order_by(TrainingJob.created_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)

        result = await db.execute(query)
        jobs = list(result.scalars().all())

        return jobs, total

    async def start_job(
        self,
        db: AsyncSession,
        job_id: UUID,
    ) -> TrainingJob:
        """
        Start a pending training job via Celery.

        Returns:
            Updated TrainingJob
        """
        job = await db.get(TrainingJob, job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        if not job.can_start():
            raise ValueError(f"Job cannot be started (status: {job.status})")

        # Queue job in Celery
        from app.tasks.training_tasks import execute_training_job_task

        task = execute_training_job_task.delay(str(job_id), str(job.user_id))

        job.status = TrainingJobStatus.QUEUED.value
        job.celery_task_id = task.id

        await db.commit()
        await db.refresh(job)

        logger.info(f"Started training job {job_id} with Celery task {task.id}")
        return job

    async def cancel_job(
        self,
        db: AsyncSession,
        job_id: UUID,
        user_id: UUID,
    ) -> TrainingJob:
        """Cancel a running or pending job."""
        job = await self.get_job(db, job_id, user_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        if not job.can_cancel():
            raise ValueError(f"Job cannot be cancelled (status: {job.status})")

        # Request trainer cancellation
        trainer = self._trainers.get(job.training_backend)
        if trainer and job.is_running():
            await trainer.cancel(job_id)

        # Signal cancellation to any worker using Redis polling
        try:
            import redis.asyncio as redis
            redis_client = redis.from_url(settings.REDIS_URL)
            # Keep the flag around briefly so worker threads polling Redis can see it
            await redis_client.set(f"training_job:{job_id}:cancel", "1", ex=3600)
            await redis_client.close()
        except Exception as e:
            logger.warning(f"Failed to set training cancel flag in Redis: {e}")

        # Revoke Celery task if exists
        if job.celery_task_id:
            from app.core.celery import celery_app
            celery_app.control.revoke(job.celery_task_id, terminate=True)

        job.status = TrainingJobStatus.CANCELLED.value
        job.completed_at = datetime.utcnow()

        await db.commit()
        await db.refresh(job)

        logger.info(f"Cancelled training job {job_id}")
        return job

    async def delete_job(
        self,
        db: AsyncSession,
        job_id: UUID,
        user_id: UUID,
    ) -> bool:
        """Delete a training job."""
        job = await self.get_job(db, job_id, user_id)
        if not job:
            return False

        # Cancel if running
        if job.can_cancel():
            await self.cancel_job(db, job_id, user_id)

        await db.delete(job)
        await db.commit()

        logger.info(f"Deleted training job {job_id}")
        return True

    async def get_checkpoints(
        self,
        db: AsyncSession,
        job_id: UUID,
    ) -> List[TrainingCheckpoint]:
        """Get checkpoints for a training job."""
        result = await db.execute(
            select(TrainingCheckpoint)
            .where(TrainingCheckpoint.job_id == job_id)
            .order_by(TrainingCheckpoint.step.desc())
        )
        return list(result.scalars().all())

    async def save_checkpoint(
        self,
        db: AsyncSession,
        job_id: UUID,
        step: int,
        epoch: float,
        checkpoint_path: str,
        loss: float,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> TrainingCheckpoint:
        """Save a training checkpoint."""
        checkpoint = TrainingCheckpoint(
            job_id=job_id,
            step=step,
            epoch=epoch,
            checkpoint_path=checkpoint_path,
            loss=loss,
            metrics=metrics,
        )

        db.add(checkpoint)
        await db.commit()
        await db.refresh(checkpoint)

        return checkpoint

    async def update_job_progress(
        self,
        db: AsyncSession,
        job_id: UUID,
        progress: TrainingProgress,
    ):
        """Update job progress from trainer callback."""
        job = await db.get(TrainingJob, job_id)
        if not job:
            return

        job.status = TrainingJobStatus.TRAINING.value
        # Keep 100% reserved for the adapter registration/completion phase.
        pct = int((progress.step / progress.total_steps) * 95) if progress.total_steps > 0 else 0
        job.progress = max(0, min(95, pct))
        job.current_step = progress.step
        job.total_steps = progress.total_steps
        job.current_epoch = progress.epoch
        job.total_epochs = progress.total_epochs

        # Update metrics
        if job.training_metrics is None:
            job.training_metrics = {}

        job.training_metrics["current_loss"] = progress.loss
        job.training_metrics["learning_rate"] = progress.learning_rate

        if "loss_history" not in job.training_metrics:
            job.training_metrics["loss_history"] = []
        job.training_metrics["loss_history"].append(progress.loss)

        if progress.loss < job.training_metrics.get("best_loss", float("inf")):
            job.training_metrics["best_loss"] = progress.loss

        await db.commit()

    async def complete_job(
        self,
        db: AsyncSession,
        job_id: UUID,
        adapter_path: str,
        adapter_size: int,
        final_metrics: Dict[str, Any],
    ) -> TrainingJob:
        """Mark a job as completed and register the adapter."""
        job = await db.get(TrainingJob, job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        # Create adapter record
        adapter_name = f"{job.name.lower().replace(' ', '-')}-{str(job_id)[:8]}"
        if job.training_method == TrainingMethod.QLORA.value:
            adapter_type = AdapterType.QLORA.value
        else:
            adapter_type = AdapterType.LORA.value
        adapter = ModelAdapter(
            name=adapter_name,
            display_name=job.name,
            description=f"Trained on {job.base_model}",
            base_model=job.base_model,
            adapter_type=adapter_type,
            adapter_config=job.hyperparameters,
            adapter_path=adapter_path,
            adapter_size=adapter_size,
            training_job_id=job.id,
            training_metrics=final_metrics,
            user_id=job.user_id,
            status=AdapterStatus.READY.value,
        )

        db.add(adapter)
        await db.flush()

        # Update job
        job.status = TrainingJobStatus.COMPLETED.value
        job.progress = 100
        job.final_metrics = final_metrics
        job.output_adapter_id = adapter.id
        job.completed_at = datetime.utcnow()

        await db.commit()
        await db.refresh(job)

        logger.info(f"Training job {job_id} completed with adapter {adapter.id}")
        return job

    async def fail_job(
        self,
        db: AsyncSession,
        job_id: UUID,
        error: str,
    ) -> TrainingJob:
        """Mark a job as failed."""
        job = await db.get(TrainingJob, job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        job.status = TrainingJobStatus.FAILED.value
        job.error = error
        job.completed_at = datetime.utcnow()

        await db.commit()
        await db.refresh(job)

        logger.error(f"Training job {job_id} failed: {error}")
        return job

    async def get_stats(
        self,
        db: AsyncSession,
        user_id: UUID,
    ) -> Dict[str, Any]:
        """Get training statistics for a user."""
        running_statuses = [
            TrainingJobStatus.QUEUED.value,
            TrainingJobStatus.PREPARING.value,
            TrainingJobStatus.TRAINING.value,
            TrainingJobStatus.SAVING.value,
        ]
        result = await db.execute(
            select(
                func.count(TrainingJob.id).label("total"),
                func.count(TrainingJob.id).filter(
                    TrainingJob.status.in_(running_statuses)
                ).label("running"),
                func.count(TrainingJob.id).filter(
                    TrainingJob.status == TrainingJobStatus.COMPLETED.value
                ).label("completed"),
                func.count(TrainingJob.id).filter(
                    TrainingJob.status == TrainingJobStatus.FAILED.value
                ).label("failed"),
            ).where(TrainingJob.user_id == user_id)
        )
        row = result.one()

        return {
            "total_jobs": row.total,
            "running_jobs": row.running,
            "completed_jobs": row.completed,
            "failed_jobs": row.failed,
        }

    async def get_available_base_models(self) -> List[Dict[str, Any]]:
        """Get list of available base models from Ollama."""
        import httpx

        models = []

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{settings.OLLAMA_BASE_URL}/api/tags",
                    timeout=10.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    for model in data.get("models", []):
                        models.append({
                            "name": model["name"],
                            "display_name": model["name"],
                            "provider": "ollama",
                            "size_gb": model.get("size", 0) / (1024**3),
                            "is_available": True,
                        })

        except Exception as e:
            logger.warning(f"Failed to fetch Ollama models: {e}")

        # Add common HuggingFace models as options
        hf_models = [
            {"name": "meta-llama/Llama-3.2-1B", "display_name": "Llama 3.2 1B", "size_gb": 2.0},
            {"name": "meta-llama/Llama-3.2-3B", "display_name": "Llama 3.2 3B", "size_gb": 6.0},
            {"name": "mistralai/Mistral-7B-v0.1", "display_name": "Mistral 7B", "size_gb": 14.0},
            {"name": "microsoft/Phi-3-mini-4k-instruct", "display_name": "Phi-3 Mini", "size_gb": 4.0},
        ]

        for model in hf_models:
            models.append({
                "name": model["name"],
                "display_name": model["display_name"],
                "provider": "huggingface",
                "size_gb": model["size_gb"],
                "is_available": True,
            })

        return models


# Global service instance
training_service = TrainingService()
