from datetime import datetime
from typing import Any, Dict, Optional, List
from uuid import UUID

from pydantic import BaseModel


class LLMUsageEventResponse(BaseModel):
    id: UUID
    user_id: Optional[UUID] = None
    provider: str
    model: Optional[str] = None
    task_type: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    input_chars: Optional[int] = None
    output_chars: Optional[int] = None
    latency_ms: Optional[int] = None
    error: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None
    created_at: datetime


class LLMUsageSummaryItem(BaseModel):
    provider: str
    model: Optional[str] = None
    task_type: Optional[str] = None
    request_count: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    avg_latency_ms: Optional[float] = None


class LLMUsageSummaryResponse(BaseModel):
    items: List[LLMUsageSummaryItem]
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None



class LLMRoutingSummaryItem(BaseModel):
    provider: str
    model: Optional[str] = None
    task_type: Optional[str] = None

    routing_tier: Optional[str] = None
    routing_requested_tier: Optional[str] = None
    routing_attempt: Optional[int] = None
    routing_attempts: Optional[int] = None
    routing_tier_provider: Optional[str] = None
    routing_tier_model: Optional[str] = None

    routing_experiment_id: Optional[str] = None
    routing_experiment_variant_id: Optional[str] = None

    request_count: int
    success_count: int
    error_count: int
    success_rate: float

    total_tokens: int
    avg_latency_ms: Optional[float] = None
    p50_latency_ms: Optional[int] = None
    p95_latency_ms: Optional[int] = None


class LLMRoutingSummaryResponse(BaseModel):
    items: List[LLMRoutingSummaryItem]
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    scanned_events: int
    truncated: bool



class LLMRoutingExperimentVariantStat(BaseModel):
    experiment_id: str
    variant_id: str
    request_count: int
    success_count: int
    error_count: int
    success_rate: float
    avg_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[int] = None


class LLMRoutingExperimentRecommendationResponse(BaseModel):
    experiment_id: str
    agent_id: Optional[UUID] = None
    recommended_variant_id: Optional[str] = None
    rationale: str
    variants: List[LLMRoutingExperimentVariantStat]
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    scanned_events: int
    truncated: bool



class LLMRoutingExperimentListItem(BaseModel):
    agent_id: UUID
    agent_name: str
    agent_display_name: str
    agent_is_system: bool
    agent_owner_user_id: Optional[UUID] = None
    agent_lifecycle_status: Optional[str] = None

    routing_defaults: Optional[Dict[str, Any]] = None
    experiment: Dict[str, Any]


class LLMRoutingExperimentListResponse(BaseModel):
    items: List[LLMRoutingExperimentListItem]
    total: int
