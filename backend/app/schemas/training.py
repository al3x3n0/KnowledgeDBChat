"""
Pydantic schemas for AI Hub training.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================================================
# Dataset Schemas
# ============================================================================

class DatasetSampleCreate(BaseModel):
    """Request schema for creating a dataset sample."""

    instruction: str = Field(..., min_length=1, description="The instruction/prompt")
    input: Optional[str] = Field(None, description="Optional input context")
    output: str = Field(..., min_length=1, description="The expected output/response")
    source_document_id: Optional[UUID] = Field(None, description="Source document if generated")


class DatasetSampleResponse(BaseModel):
    """Response schema for a dataset sample."""

    id: UUID
    dataset_id: UUID
    sample_index: int
    content: Dict[str, Any]
    source_document_id: Optional[UUID]
    input_tokens: int
    output_tokens: int
    is_flagged: bool
    flag_reason: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class TrainingDatasetCreate(BaseModel):
    """Request schema for creating a training dataset."""

    name: str = Field(..., min_length=1, max_length=200, description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    dataset_type: str = Field("instruction", description="Type: instruction, chat, completion, preference")
    format: str = Field("alpaca", description="Format: alpaca, sharegpt, custom")
    samples: Optional[List[DatasetSampleCreate]] = Field(None, description="Initial samples to add")
    is_public: bool = Field(False, description="Make dataset public")


class TrainingDatasetUpdate(BaseModel):
    """Request schema for updating a dataset."""

    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    is_public: Optional[bool] = None


class TrainingDatasetResponse(BaseModel):
    """Response schema for a training dataset."""

    id: UUID
    name: str
    description: Optional[str]
    dataset_type: str
    format: str
    source_document_ids: Optional[List[UUID]]
    file_path: Optional[str]
    file_size: Optional[int]
    sample_count: int
    token_count: int
    is_validated: bool
    validation_errors: Optional[List[Dict[str, Any]]]
    version: int
    parent_dataset_id: Optional[UUID]
    user_id: UUID
    is_public: bool
    status: str
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class TrainingDatasetListResponse(BaseModel):
    """Response schema for listing datasets."""

    datasets: List[TrainingDatasetResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class DatasetValidationResult(BaseModel):
    """Result of dataset validation."""

    is_valid: bool
    sample_count: int
    token_count: int
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]


class GenerateDatasetRequest(BaseModel):
    """Request to generate dataset from documents."""

    name: str = Field(..., min_length=1, max_length=200, description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    document_ids: List[UUID] = Field(..., min_length=1, description="Documents to generate from")
    dataset_type: str = Field("instruction", description="Type of dataset to generate")
    samples_per_document: int = Field(5, ge=1, le=50, description="Samples to generate per document")
    # If preset_id is provided, generation_prompt is ignored and the preset's prompt is used.
    preset_id: Optional[str] = Field(None, description="AI Hub preset id (plugin) for generation")
    extra_instructions: Optional[str] = Field(None, description="Additional instructions appended to the preset prompt")
    generation_prompt: Optional[str] = Field(None, description="Custom prompt for generation (advanced)")


class AddSamplesResponse(BaseModel):
    """Response for adding samples."""

    added_count: int
    total_count: int
    token_count: int


# ============================================================================
# Hyperparameter Schemas
# ============================================================================

class HyperparametersConfig(BaseModel):
    """Training hyperparameters configuration."""

    # LoRA parameters
    lora_r: int = Field(16, ge=4, le=256, description="LoRA rank")
    lora_alpha: int = Field(32, ge=8, le=512, description="LoRA alpha")
    lora_dropout: float = Field(0.05, ge=0.0, le=0.5, description="LoRA dropout")
    target_modules: Optional[List[str]] = Field(None, description="Target modules for LoRA")

    # Training parameters
    learning_rate: float = Field(2e-4, ge=1e-6, le=1e-2, description="Learning rate")
    num_epochs: int = Field(3, ge=1, le=50, description="Number of epochs")
    batch_size: int = Field(4, ge=1, le=64, description="Batch size")
    gradient_accumulation_steps: int = Field(4, ge=1, le=128, description="Gradient accumulation steps")
    warmup_steps: int = Field(100, ge=0, le=10000, description="Warmup steps")
    max_seq_length: int = Field(2048, ge=128, le=8192, description="Maximum sequence length")

    # Optimization
    weight_decay: float = Field(0.01, ge=0.0, le=1.0, description="Weight decay")
    max_grad_norm: float = Field(1.0, ge=0.0, le=10.0, description="Max gradient norm")


class ResourceConfig(BaseModel):
    """Resource configuration for training."""

    device: str = Field("auto", description="Device: cuda, cpu, mps, auto")
    max_memory_gb: Optional[float] = Field(None, description="Maximum GPU memory to use")
    mixed_precision: str = Field("bf16", description="Mixed precision: bf16, fp16, fp32")
    gradient_checkpointing: bool = Field(True, description="Enable gradient checkpointing")


# ============================================================================
# Training Job Schemas
# ============================================================================

class TrainingJobCreate(BaseModel):
    """Request schema for creating a training job."""

    name: str = Field(..., min_length=1, max_length=200, description="Job name")
    description: Optional[str] = Field(None, description="Job description")

    # Training configuration
    training_method: str = Field("lora", description="Method: lora, qlora, full_finetune")
    training_backend: str = Field("local", description="Backend: local, modal, runpod")

    # Model and dataset
    base_model: str = Field(..., min_length=1, description="Base model name (e.g., llama3.2:3b)")
    base_model_provider: str = Field("ollama", description="Model provider: ollama, huggingface")
    dataset_id: UUID = Field(..., description="Training dataset ID")

    # Hyperparameters (optional - uses defaults if not provided)
    hyperparameters: Optional[HyperparametersConfig] = Field(None, description="Training hyperparameters")
    resource_config: Optional[ResourceConfig] = Field(None, description="Resource configuration")

    # Execution
    start_immediately: bool = Field(True, description="Start job immediately after creation")


class TrainingJobUpdate(BaseModel):
    """Request schema for updating a training job."""

    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None


class TrainingJobResponse(BaseModel):
    """Response schema for a training job."""

    id: UUID
    name: str
    description: Optional[str]

    # Configuration
    training_method: str
    training_backend: str
    base_model: str
    base_model_provider: str
    dataset_id: UUID
    hyperparameters: Optional[Dict[str, Any]]
    resource_config: Optional[Dict[str, Any]]

    # Ownership
    user_id: UUID

    # Status and progress
    status: str
    progress: int
    current_step: Optional[int]
    total_steps: Optional[int]
    current_epoch: Optional[int]
    total_epochs: Optional[int]

    # Metrics
    training_metrics: Optional[Dict[str, Any]]
    final_metrics: Optional[Dict[str, Any]]

    # Output
    output_adapter_id: Optional[UUID]

    # Error tracking
    error: Optional[str]
    celery_task_id: Optional[str]

    # Timestamps
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class TrainingJobListResponse(BaseModel):
    """Response schema for listing training jobs."""

    jobs: List[TrainingJobResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class TrainingJobDetailResponse(TrainingJobResponse):
    """Detailed response including dataset info."""

    dataset_name: Optional[str] = None
    dataset_sample_count: Optional[int] = None
    adapter_name: Optional[str] = None


class TrainingJobActionRequest(BaseModel):
    """Request schema for job actions."""

    action: str = Field(..., description="Action: start, pause, resume, cancel")


class TrainingCheckpointResponse(BaseModel):
    """Response schema for a training checkpoint."""

    id: UUID
    job_id: UUID
    step: int
    epoch: Optional[float]
    checkpoint_path: Optional[str]
    loss: Optional[float]
    metrics: Optional[Dict[str, Any]]
    created_at: datetime

    class Config:
        from_attributes = True


class TrainingProgressUpdate(BaseModel):
    """WebSocket message for training progress updates."""

    type: str = "progress"
    job_id: str
    progress: int
    status: str
    current_step: Optional[int]
    total_steps: Optional[int]
    current_epoch: Optional[int]
    total_epochs: Optional[int]
    current_loss: Optional[float]
    learning_rate: Optional[float]
    eta_seconds: Optional[int]
    timestamp: str


class TrainingStatsResponse(BaseModel):
    """Response schema for training statistics."""

    total_jobs: int
    running_jobs: int
    completed_jobs: int
    failed_jobs: int
    total_training_hours: float
    total_samples_trained: int
    avg_final_loss: Optional[float]


class BaseModelInfo(BaseModel):
    """Information about an available base model."""

    name: str
    display_name: str
    provider: str
    size_gb: Optional[float]
    parameters: Optional[str]
    context_length: Optional[int]
    is_available: bool


# ============================================================================
# Model Adapter Schemas
# ============================================================================

class ModelAdapterCreate(BaseModel):
    """Request schema for manually creating an adapter."""

    name: str = Field(..., min_length=1, max_length=100, description="Unique adapter name")
    display_name: str = Field(..., min_length=1, max_length=200, description="Display name")
    description: Optional[str] = Field(None, description="Description")
    base_model: str = Field(..., description="Base model name")
    adapter_type: str = Field("lora", description="Adapter type: lora, qlora")
    adapter_config: Optional[Dict[str, Any]] = Field(None, description="Adapter configuration")
    is_public: bool = Field(False, description="Make adapter public")
    tags: Optional[List[str]] = Field(None, description="Tags for organization")


class ModelAdapterUpdate(BaseModel):
    """Request schema for updating an adapter."""

    display_name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    is_public: Optional[bool] = None
    tags: Optional[List[str]] = None


class ModelAdapterResponse(BaseModel):
    """Response schema for a model adapter."""

    id: UUID
    name: str
    display_name: str
    description: Optional[str]
    base_model: str
    adapter_type: str
    adapter_config: Optional[Dict[str, Any]]
    adapter_path: Optional[str]
    adapter_size: Optional[int]
    training_job_id: Optional[UUID]
    training_metrics: Optional[Dict[str, Any]]
    user_id: UUID
    is_public: bool
    status: str
    is_deployed: bool
    deployment_config: Optional[Dict[str, Any]]
    version: int
    tags: Optional[List[str]]
    usage_count: int
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True


class ModelAdapterListResponse(BaseModel):
    """Response schema for listing adapters."""

    adapters: List[ModelAdapterResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class DeployAdapterRequest(BaseModel):
    """Request to deploy an adapter to Ollama."""

    ollama_model_name: Optional[str] = Field(
        None,
        min_length=1,
        max_length=100,
        description="Custom Ollama model name (default: adapter name)"
    )


class DeploymentStatusResponse(BaseModel):
    """Response for deployment status."""

    adapter_id: UUID
    is_deployed: bool
    ollama_model_name: Optional[str]
    deployed_at: Optional[datetime]
    status: str


class TestAdapterRequest(BaseModel):
    """Request to test an adapter with a prompt."""

    prompt: str = Field(..., min_length=1, max_length=4000, description="Test prompt")
    max_tokens: int = Field(256, ge=1, le=2000, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature")


class TestAdapterResponse(BaseModel):
    """Response from testing an adapter."""

    prompt: str
    response: str
    tokens_generated: int
    generation_time_ms: int
