"""
Model Registry models for AI Hub.

Manages trained LoRA adapters and their deployment to Ollama.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, BigInteger, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
import enum

from app.core.database import Base


class AdapterType(str, enum.Enum):
    """Type of adapter."""
    LORA = "lora"       # Standard LoRA
    QLORA = "qlora"     # Quantized LoRA


class AdapterStatus(str, enum.Enum):
    """Status of a model adapter."""
    TRAINING = "training"     # Being trained
    READY = "ready"           # Trained and ready for use
    DEPLOYING = "deploying"   # Being deployed to Ollama
    DEPLOYED = "deployed"     # Deployed and available
    FAILED = "failed"         # Deployment or training failed
    ARCHIVED = "archived"     # No longer active


class ModelAdapter(Base):
    """
    Trained LoRA/PEFT adapter.

    Represents a trained adapter that can be deployed to Ollama
    for inference as a custom model.
    """

    __tablename__ = "model_adapters"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Adapter identification
    name = Column(String(100), nullable=False, unique=True)  # Unique internal name
    display_name = Column(String(200), nullable=False)       # Human-readable name
    description = Column(Text, nullable=True)

    # Base model info
    base_model = Column(String(200), nullable=False)  # e.g., "llama3.2:3b"

    # Adapter type and config
    adapter_type = Column(String(30), nullable=False, default=AdapterType.LORA.value)
    adapter_config = Column(JSON, nullable=True)
    # Structure:
    # {
    #   "lora_r": 16,
    #   "lora_alpha": 32,
    #   "target_modules": ["q_proj", "v_proj"],
    #   "lora_dropout": 0.05
    # }

    # Storage
    adapter_path = Column(String(500), nullable=True)  # MinIO path to adapter weights
    adapter_size = Column(BigInteger, nullable=True)   # Size in bytes

    # Training info
    training_job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("training_jobs.id", ondelete="SET NULL"),
        nullable=True
    )
    training_metrics = Column(JSON, nullable=True)
    # Structure:
    # {
    #   "final_loss": 1.2,
    #   "training_time_seconds": 3600,
    #   "samples_trained": 5000,
    #   "epochs": 3
    # }

    # Ownership
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )

    # Visibility
    is_public = Column(Boolean, nullable=False, default=False)

    # Status
    status = Column(String(30), nullable=False, default=AdapterStatus.READY.value)

    # Deployment
    is_deployed = Column(Boolean, nullable=False, default=False)
    deployment_config = Column(JSON, nullable=True)
    # Structure:
    # {
    #   "ollama_model_name": "my-custom-model",
    #   "deployed_at": "2024-01-29T12:00:00Z",
    #   "modelfile_path": "/path/to/modelfile"
    # }

    # Versioning
    version = Column(Integer, nullable=False, default=1)

    # Organization
    tags = Column(JSON, nullable=True)  # List of tags

    # Usage tracking
    usage_count = Column(Integer, nullable=False, default=0)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", backref="model_adapters")
    training_job = relationship("TrainingJob", foreign_keys=[training_job_id])

    def __repr__(self):
        return f"<ModelAdapter(id={self.id}, name='{self.name}', status={self.status})>"

    def can_deploy(self) -> bool:
        """Check if adapter can be deployed."""
        return (
            self.status == AdapterStatus.READY.value and
            not self.is_deployed and
            self.adapter_path is not None
        )

    def can_undeploy(self) -> bool:
        """Check if adapter can be undeployed."""
        return self.is_deployed

    def get_ollama_model_name(self) -> Optional[str]:
        """Get the Ollama model name if deployed."""
        if self.deployment_config:
            return self.deployment_config.get("ollama_model_name")
        return None

    def get_display_info(self) -> Dict[str, Any]:
        """Get display info for UI."""
        return {
            "id": str(self.id),
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "base_model": self.base_model,
            "adapter_type": self.adapter_type,
            "status": self.status,
            "is_deployed": self.is_deployed,
            "ollama_model_name": self.get_ollama_model_name(),
            "usage_count": self.usage_count,
            "tags": self.tags or [],
        }

    def increment_usage(self):
        """Increment the usage counter."""
        self.usage_count = (self.usage_count or 0) + 1

    def get_size_mb(self) -> Optional[float]:
        """Get adapter size in MB."""
        if self.adapter_size:
            return self.adapter_size / (1024 * 1024)
        return None
