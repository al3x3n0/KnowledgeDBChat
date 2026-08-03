"""
Pydantic schemas for the Research Inbox.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ResearchInboxItemResponse(BaseModel):
    id: UUID
    user_id: UUID
    job_id: Optional[UUID] = None
    customer: Optional[str] = None

    item_type: str
    item_key: str
    title: str
    summary: Optional[str] = None
    url: Optional[str] = None
    published_at: Optional[datetime] = None
    discovered_at: datetime

    status: str
    feedback: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, validation_alias="item_metadata"
    )
    follow_up_decision: Optional[str] = None
    follow_up_policy_mode: Optional[str] = None
    follow_up_launch_status: Optional[str] = None
    follow_up_block_reason: Optional[str] = None
    follow_up_budget_decision: Optional[str] = None
    follow_up_budget_reason: Optional[str] = None
    follow_up_budget_throttle_state: Optional[str] = None
    follow_up_customer_budget_decision: Optional[str] = None
    follow_up_customer_budget_reason: Optional[str] = None
    follow_up_customer_budget_throttle_state: Optional[str] = None
    follow_up_recommendation_key: Optional[str] = None
    follow_up_operator_decision: Optional[str] = None
    follow_up_operator_note: Optional[str] = None
    follow_up_operator_acted_at: Optional[datetime] = None
    follow_up_operator_user_id: Optional[UUID] = None
    follow_up_job_id: Optional[UUID] = None
    follow_up_last_job_id: Optional[UUID] = None
    follow_up_chain_definition_id: Optional[UUID] = None
    follow_up_launched_at: Optional[datetime] = None
    follow_up_outcome_status: Optional[str] = None
    follow_up_outcome_recorded_at: Optional[datetime] = None
    follow_up_outcome_summary: Optional[str] = None
    origin_source_kind: Optional[str] = None
    origin_source_id: Optional[str] = None
    origin_opportunity_id: Optional[str] = None

    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ResearchInboxListResponse(BaseModel):
    items: List[ResearchInboxItemResponse]
    total: int
    limit: int
    offset: int


class ResearchInboxItemUpdateRequest(BaseModel):
    status: Optional[str] = Field(None, description="new | accepted | rejected")
    feedback: Optional[str] = Field(None, max_length=4000)
    metadata_patch: Optional[Dict[str, Any]] = Field(
        None,
        description="Merge patch for item.metadata (allowlisted keys only)",
    )


class ResearchInboxStatsResponse(BaseModel):
    total: int
    new: int
    accepted: int
    rejected: int


class ResearchInboxFollowUpRelaunchRequest(BaseModel):
    operator_note: Optional[str] = Field(default=None, max_length=2000)


class ResearchInboxBulkFollowUpRelaunchRequest(BaseModel):
    item_ids: List[UUID] = Field(default_factory=list, min_length=1)
    operator_note: Optional[str] = Field(default=None, max_length=2000)


class ResearchInboxBulkFollowUpRelaunchResult(BaseModel):
    item_id: UUID
    ok: bool
    follow_up_job_id: Optional[UUID] = None
    error: Optional[str] = None


class ResearchInboxBulkFollowUpRelaunchResponse(BaseModel):
    requested_count: int
    applied: int
    failed: int
    results: List[ResearchInboxBulkFollowUpRelaunchResult]
