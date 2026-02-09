"""
Training Dataset models for AI Hub.

Manages training datasets and individual samples for model fine-tuning.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, BigInteger, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
import enum

from app.core.database import Base


class DatasetType(str, enum.Enum):
    """Type of training dataset."""
    INSTRUCTION = "instruction"  # Alpaca-style instruction following
    CHAT = "chat"               # Multi-turn conversation format
    COMPLETION = "completion"   # Text completion format
    PREFERENCE = "preference"   # DPO/RLHF preference pairs


class DatasetFormat(str, enum.Enum):
    """Format of training data."""
    ALPACA = "alpaca"           # {"instruction": ..., "input": ..., "output": ...}
    SHAREGPT = "sharegpt"       # {"conversations": [{"from": "human/gpt", "value": ...}]}
    CUSTOM = "custom"           # Custom JSON schema


class DatasetStatus(str, enum.Enum):
    """Status of a training dataset."""
    DRAFT = "draft"             # Being created/edited
    VALIDATING = "validating"   # Being validated
    READY = "ready"             # Validated and ready for training
    ERROR = "error"             # Validation failed
    ARCHIVED = "archived"       # No longer active


class TrainingDataset(Base):
    """
    Training dataset for model fine-tuning.

    Stores metadata and configuration for a collection of training samples.
    Actual sample data is stored in DatasetSample records.
    """

    __tablename__ = "training_datasets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Dataset identification
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)

    # Dataset type and format
    dataset_type = Column(String(50), nullable=False, default=DatasetType.INSTRUCTION.value)
    format = Column(String(50), nullable=False, default=DatasetFormat.ALPACA.value)

    # Source tracking
    source_document_ids = Column(JSON, nullable=True)  # Array of document UUIDs used to generate

    # Storage
    file_path = Column(String(500), nullable=True)  # MinIO path for exported JSONL
    file_size = Column(BigInteger, nullable=True)   # Size in bytes

    # Statistics
    sample_count = Column(Integer, nullable=False, default=0)
    token_count = Column(BigInteger, nullable=False, default=0)

    # Validation
    is_validated = Column(Boolean, nullable=False, default=False)
    validation_errors = Column(JSON, nullable=True)  # List of validation errors if any

    # Versioning
    version = Column(Integer, nullable=False, default=1)
    parent_dataset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("training_datasets.id", ondelete="SET NULL"),
        nullable=True
    )

    # Ownership
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False
    )

    # Visibility
    is_public = Column(Boolean, nullable=False, default=False)

    # Status
    status = Column(String(30), nullable=False, default=DatasetStatus.DRAFT.value)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", backref="training_datasets")
    samples = relationship("DatasetSample", back_populates="dataset", cascade="all, delete-orphan")
    parent_dataset = relationship("TrainingDataset", remote_side=[id], backref="child_datasets")

    def __repr__(self):
        return f"<TrainingDataset(id={self.id}, name='{self.name}', status={self.status})>"

    def is_ready_for_training(self) -> bool:
        """Check if dataset is ready for training."""
        return (
            self.status == DatasetStatus.READY.value and
            self.is_validated and
            self.sample_count > 0
        )

    def get_token_statistics(self) -> Dict[str, Any]:
        """Get token distribution statistics."""
        return {
            "total_tokens": self.token_count,
            "sample_count": self.sample_count,
            "avg_tokens_per_sample": (
                self.token_count / self.sample_count if self.sample_count > 0 else 0
            ),
        }


class DatasetSample(Base):
    """
    Individual training sample in a dataset.

    Stores the actual training data (instruction/input/output or conversation).
    """

    __tablename__ = "dataset_samples"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Parent dataset
    dataset_id = Column(
        UUID(as_uuid=True),
        ForeignKey("training_datasets.id", ondelete="CASCADE"),
        nullable=False
    )

    # Sample ordering
    sample_index = Column(Integer, nullable=False)

    # Sample content - structure depends on dataset format
    # Alpaca: {"instruction": "...", "input": "...", "output": "..."}
    # ShareGPT: {"conversations": [{"from": "human", "value": "..."}, ...]}
    content = Column(JSON, nullable=False)

    # Source tracking
    source_document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True
    )

    # Token counts
    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)

    # Quality flags
    is_flagged = Column(Boolean, nullable=False, default=False)
    flag_reason = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)

    # Relationships
    dataset = relationship("TrainingDataset", back_populates="samples")

    def __repr__(self):
        return f"<DatasetSample(id={self.id}, dataset_id={self.dataset_id}, index={self.sample_index})>"

    @property
    def total_tokens(self) -> int:
        """Get total token count for this sample."""
        return self.input_tokens + self.output_tokens

    def get_instruction(self) -> Optional[str]:
        """Extract instruction from content (for alpaca format)."""
        if isinstance(self.content, dict):
            return self.content.get("instruction")
        return None

    def get_output(self) -> Optional[str]:
        """Extract output from content (for alpaca format)."""
        if isinstance(self.content, dict):
            return self.content.get("output")
        return None

    def to_alpaca_dict(self) -> Dict[str, str]:
        """Convert sample to alpaca format dict."""
        if isinstance(self.content, dict):
            return {
                "instruction": self.content.get("instruction", ""),
                "input": self.content.get("input", ""),
                "output": self.content.get("output", ""),
            }
        return {"instruction": "", "input": "", "output": ""}
