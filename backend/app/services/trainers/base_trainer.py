"""
Base trainer interface for AI Hub training backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID


@dataclass
class TrainingProgress:
    """Training progress information."""

    job_id: str
    step: int
    total_steps: int
    epoch: int
    total_epochs: int
    loss: float
    learning_rate: float
    samples_processed: int
    eta_seconds: Optional[int] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    message: str = ""


@dataclass
class TrainingResult:
    """Result of a training run."""

    success: bool
    adapter_path: Optional[str] = None
    adapter_size: Optional[int] = None
    final_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    total_steps: int = 0
    total_epochs: int = 0
    training_time_seconds: int = 0
    samples_trained: int = 0
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeviceInfo:
    """Information about available compute devices."""

    device: str  # cuda, cpu, mps
    device_name: Optional[str] = None
    memory_total_gb: Optional[float] = None
    memory_available_gb: Optional[float] = None
    compute_capability: Optional[str] = None
    cuda_version: Optional[str] = None


class BaseTrainer(ABC):
    """
    Abstract base class for training backends.

    Implementations handle the actual model training using different
    compute backends (local, Modal, RunPod, etc.).
    """

    @abstractmethod
    async def train(
        self,
        job_id: UUID,
        dataset_path: str,
        base_model: str,
        output_path: str,
        hyperparameters: Dict[str, Any],
        progress_callback: Optional[Callable[[TrainingProgress], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> TrainingResult:
        """
        Execute training and return the result.

        Args:
            job_id: Unique job identifier
            dataset_path: Path to training dataset (local or MinIO)
            base_model: Base model identifier (e.g., "llama3.2:3b")
            output_path: Where to save the trained adapter
            hyperparameters: Training hyperparameters
            progress_callback: Optional callback for progress updates
            cancel_check: Optional function that returns True if training should be cancelled

        Returns:
            TrainingResult with success status and output path
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this training backend is available.

        Returns:
            True if the backend can be used for training
        """
        pass

    @abstractmethod
    def get_device_info(self) -> DeviceInfo:
        """
        Get information about available compute devices.

        Returns:
            DeviceInfo with device details
        """
        pass

    @abstractmethod
    async def cancel(self, job_id: UUID) -> bool:
        """
        Request cancellation of a running training job.

        Args:
            job_id: Job to cancel

        Returns:
            True if cancellation was requested successfully
        """
        pass

    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """
        Get list of supported base models for this backend.

        Returns:
            List of model identifiers that can be fine-tuned
        """
        pass

    def get_default_hyperparameters(self, model_name: str) -> Dict[str, Any]:
        """
        Get recommended hyperparameters for a given model.

        Args:
            model_name: Base model name

        Returns:
            Dictionary of recommended hyperparameters
        """
        # Default hyperparameters suitable for most models
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
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
        }

        # Adjust based on model size
        model_lower = model_name.lower()
        if "7b" in model_lower or "8b" in model_lower:
            defaults["batch_size"] = 2
            defaults["gradient_accumulation_steps"] = 8
        elif "13b" in model_lower or "14b" in model_lower:
            defaults["batch_size"] = 1
            defaults["gradient_accumulation_steps"] = 16
            defaults["lora_r"] = 8
        elif "70b" in model_lower:
            defaults["batch_size"] = 1
            defaults["gradient_accumulation_steps"] = 32
            defaults["lora_r"] = 8
            defaults["max_seq_length"] = 1024

        return defaults

    def estimate_memory_requirements(
        self,
        model_name: str,
        training_method: str,
        batch_size: int,
        max_seq_length: int,
    ) -> float:
        """
        Estimate GPU memory requirements in GB.

        Args:
            model_name: Base model name
            training_method: lora, qlora, or full_finetune
            batch_size: Batch size
            max_seq_length: Maximum sequence length

        Returns:
            Estimated memory requirement in GB
        """
        # Base memory estimates per model size
        model_lower = model_name.lower()
        if "1b" in model_lower:
            base_memory = 2.0
        elif "3b" in model_lower:
            base_memory = 6.0
        elif "7b" in model_lower or "8b" in model_lower:
            base_memory = 14.0
        elif "13b" in model_lower or "14b" in model_lower:
            base_memory = 26.0
        elif "70b" in model_lower:
            base_memory = 140.0
        else:
            base_memory = 8.0  # Default estimate

        # Adjust for training method
        if training_method == "qlora":
            base_memory *= 0.25  # 4-bit quantization
        elif training_method == "lora":
            base_memory *= 0.5   # Only training adapters
        # full_finetune uses full memory

        # Adjust for batch size and sequence length
        memory = base_memory * (batch_size / 4) * (max_seq_length / 2048)

        return round(memory, 1)
