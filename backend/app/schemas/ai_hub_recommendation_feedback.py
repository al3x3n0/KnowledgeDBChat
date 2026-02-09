"""
Schemas for AI Hub recommendation feedback (learning loop).
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Literal
from uuid import UUID

from pydantic import BaseModel, Field


WorkflowId = Literal["triage", "extraction", "literature"]
ItemType = Literal["dataset_preset", "eval_template"]
Decision = Literal["accept", "reject"]


class AIHubRecommendationFeedbackCreate(BaseModel):
    workflow: WorkflowId
    item_type: ItemType
    item_id: str = Field(..., min_length=2, max_length=200)
    decision: Decision
    reason: Optional[str] = Field(None, max_length=2000)


class AIHubRecommendationFeedbackResponse(BaseModel):
    id: UUID
    created_at: datetime
    user_id: UUID
    agent_job_id: Optional[UUID] = None
    customer_profile_name: Optional[str] = None
    workflow: WorkflowId
    item_type: ItemType
    item_id: str
    decision: Decision
    reason: Optional[str] = None

    class Config:
        from_attributes = True


class AIHubRecommendationFeedbackListResponse(BaseModel):
    items: List[AIHubRecommendationFeedbackResponse]
    total: int

