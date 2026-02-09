"""
Training Job models for AI Hub.

Tracks training job execution for LoRA/PEFT fine-tuning.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
import enum

from app.core.database import Base


class TrainingMethod(str, enum.Enum):
    """Training method/algorithm."""
    LORA = "lora"                   # Low-Rank Adaptation
    QLORA = "qlora"                 # Quantized LoRA (4-bit)
    FULL_FINETUNE = "full_finetune" # Full model fine-tuning


class TrainingBackend(str, enum.Enum):
    """Training compute backend."""
    LOCAL = "local"     # Local machine with PEFT
    MODAL = "modal"     # Modal.com cloud training
    RUNPOD = "runpod"   # RunPod cloud training


class TrainingJobStatus(str, enum.Enum):
    """Status of a training job."""
    PENDING = "pending"       # Job created, waiting to start
    QUEUED = "queued"         # Queued in Celery
    PREPARING = "preparing"   # Downloading model/data
    TRAINING = "training"     # Actively training
    SAVING = "saving"         # Saving adapter weights
    COMPLETED = "completed"   # Successfully finished
    FAILED = "failed"         # Failed with error
    CANCELLED = "cancelled"   # Cancelled by user


class TrainingJob(Base):
    """
    Training job for LoRA/PEFT model fine-tuning.

    Tracks the full lifecycle of a training run including
    configuration, progress, metrics, and output artifacts.
    """

    __tablename__ = "training_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Job identification
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)

    # Training configuration
    training_method = Column(String(30), nullable=False, default=TrainingMethod.LORA.value)
    training_backend = Column(String(30), nullable=False, default=TrainingBackend.LOCAL.value)

    # Base model
    base_model = Column(String(200), nullable=False)  # e.g., "llama3.2:3b", "mistral:7b"
    base_model_provider = Column(String(50), nullable=False, default="ollama")

    # Dataset
    dataset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("training_datasets.id", ondelete="CASCADE"),
        nullable=False
    )

    # Hyperparameters
    hyperparameters = Column(JSON, nullable=True)
    # Structure:
    # {
    #   "lora_r": 16,
    #   "lora_alpha": 32,
    #   "lora_dropout": 0.05,
    #   "learning_rate": 2e-4,
    #   "num_epochs": 3,
    #   "batch_size": 4,
    #   "gradient_accumulation_steps": 4,
    #   "warmup_steps": 100,
    #   "max_seq_length": 2048,
    #   "target_modules": ["q_proj", "v_proj"]
    # }

    # Resource configuration
    resource_config = Column(JSON, nullable=True)
    # Structure:
    # {
    #   "device": "cuda",
    #   "max_memory_gb": 24,
    #   "mixed_precision": "bf16",
    #   "gradient_checkpointing": true
    # }

    # Ownership
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )

    # Status and progress
    status = Column(String(30), nullable=False, default=TrainingJobStatus.PENDING.value)
    progress = Column(Integer, nullable=False, default=0)  # 0-100

    # Step tracking
    current_step = Column(Integer, nullable=True)
    total_steps = Column(Integer, nullable=True)
    current_epoch = Column(Integer, nullable=True)
    total_epochs = Column(Integer, nullable=True)

    # Training metrics - updated during training
    training_metrics = Column(JSON, nullable=True)
    # Structure:
    # {
    #   "loss_history": [2.5, 2.3, 2.1, ...],
    #   "learning_rate_history": [...],
    #   "grad_norm_history": [...],
    #   "current_loss": 1.8,
    #   "best_loss": 1.5,
    #   "samples_processed": 1000
    # }

    # Final metrics after training
    final_metrics = Column(JSON, nullable=True)
    # Structure:
    # {
    #   "final_loss": 1.2,
    #   "eval_loss": 1.4,
    #   "perplexity": 3.5,
    #   "total_training_time_seconds": 3600,
    #   "total_samples": 5000,
    #   "epochs_completed": 3
    # }

    # Output
    output_adapter_id = Column(
        UUID(as_uuid=True),
        ForeignKey("model_adapters.id", ondelete="SET NULL"),
        nullable=True
    )

    # Celery task tracking
    celery_task_id = Column(String(100), nullable=True)

    # Error tracking
    error = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    user = relationship("User", backref="training_jobs")
    dataset = relationship("TrainingDataset")
    checkpoints = relationship("TrainingCheckpoint", back_populates="job", cascade="all, delete-orphan")
    output_adapter = relationship("ModelAdapter", foreign_keys=[output_adapter_id])

    def __repr__(self):
        return f"<TrainingJob(id={self.id}, name='{self.name}', status={self.status})>"

    def can_start(self) -> bool:
        """Check if job can be started."""
        return self.status in [TrainingJobStatus.PENDING.value, TrainingJobStatus.QUEUED.value]

    def can_cancel(self) -> bool:
        """Check if job can be cancelled."""
        return self.status in [
            TrainingJobStatus.PENDING.value,
            TrainingJobStatus.QUEUED.value,
            TrainingJobStatus.PREPARING.value,
            TrainingJobStatus.TRAINING.value,
        ]

    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status in [
            TrainingJobStatus.PREPARING.value,
            TrainingJobStatus.TRAINING.value,
            TrainingJobStatus.SAVING.value,
        ]

    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.status in [
            TrainingJobStatus.COMPLETED.value,
            TrainingJobStatus.FAILED.value,
            TrainingJobStatus.CANCELLED.value,
        ]

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters with defaults."""
        defaults = {
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "learning_rate": 2e-4,
            "num_epochs": 3,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "warmup_steps": 100,
            "max_seq_length": 2048,
        }
        if self.hyperparameters:
            defaults.update(self.hyperparameters)
        return defaults

    def get_current_loss(self) -> Optional[float]:
        """Get the current training loss."""
        if self.training_metrics and "current_loss" in self.training_metrics:
            return self.training_metrics["current_loss"]
        return None

    def get_training_time_seconds(self) -> Optional[int]:
        """Get elapsed training time in seconds."""
        if self.started_at:
            end = self.completed_at or datetime.utcnow()
            return int((end - self.started_at).total_seconds())
        return None

    def update_progress(self, step: int, total: int, loss: Optional[float] = None):
        """Update training progress."""
        self.current_step = step
        self.total_steps = total
        self.progress = int((step / total) * 100) if total > 0 else 0

        if loss is not None:
            if self.training_metrics is None:
                self.training_metrics = {"loss_history": []}
            self.training_metrics["current_loss"] = loss
            if "loss_history" in self.training_metrics:
                self.training_metrics["loss_history"].append(loss)
            if "best_loss" not in self.training_metrics or loss < self.training_metrics["best_loss"]:
                self.training_metrics["best_loss"] = loss


class TrainingCheckpoint(Base):
    """
    Training checkpoint saved during training.

    Allows resuming training and selecting the best checkpoint.
    """

    __tablename__ = "training_checkpoints"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Parent job
    job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("training_jobs.id", ondelete="CASCADE"),
        nullable=False
    )

    # Checkpoint info
    step = Column(Integer, nullable=False)
    epoch = Column(Float, nullable=True)

    # Storage
    checkpoint_path = Column(String(500), nullable=True)  # MinIO path

    # Metrics at checkpoint
    loss = Column(Float, nullable=True)
    metrics = Column(JSON, nullable=True)  # Additional metrics

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    job = relationship("TrainingJob", back_populates="checkpoints")

    def __repr__(self):
        return f"<TrainingCheckpoint(job_id={self.job_id}, step={self.step}, loss={self.loss})>"
