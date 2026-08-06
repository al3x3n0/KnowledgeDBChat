from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class CollaborationSummaryResponse(BaseModel):
    owner_user_id: Optional[UUID] = None
    owner_label: Optional[str] = None
    assigned_user_id: Optional[UUID] = None
    assignee_label: Optional[str] = None
    assigned_by_user_id: Optional[UUID] = None
    assigned_at: Optional[datetime] = None
    shared_with_user_ids: List[UUID] = Field(default_factory=list)
    visibility_scope: str = "private"
    is_owned_by_current_user: bool = False
    is_assigned_to_current_user: bool = False
    is_shared_with_current_user: bool = False
    note: Optional[str] = None


class CodingBacklogItemCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    portfolio_goal: str = Field(..., min_length=1, max_length=8000)
    source_id: UUID
    scope: Optional[str] = Field(default="auto", max_length=32)
    priority: int = Field(default=50, ge=0, le=100)
    failure_symptom: Optional[str] = Field(default=None, max_length=4000)
    error_output: Optional[str] = Field(default=None, max_length=8000)
    file_paths: Optional[List[str]] = None
    commands: Optional[List[str]] = None
    auto_apply_enabled: bool = True
    require_patch_pr: bool = False
    visibility: Optional[str] = Field(default="private", max_length=24)
    shared_with_user_ids: Optional[List[UUID]] = None
    assigned_user_id: Optional[UUID] = None
    assigned_by_user_id: Optional[UUID] = None
    assigned_at: Optional[datetime] = None
    collaboration: Optional[Dict[str, Any]] = None
    policy: Optional[Dict[str, Any]] = None
    lineage: Optional[Dict[str, Any]] = None
    start_immediately: bool = True


class CodingBacklogItemUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=200)
    portfolio_goal: Optional[str] = Field(default=None, min_length=1, max_length=8000)
    scope: Optional[str] = Field(default=None, max_length=32)
    priority: Optional[int] = Field(default=None, ge=0, le=100)
    failure_symptom: Optional[str] = Field(default=None, max_length=4000)
    error_output: Optional[str] = Field(default=None, max_length=8000)
    file_paths: Optional[List[str]] = None
    commands: Optional[List[str]] = None
    auto_apply_enabled: Optional[bool] = None
    require_patch_pr: Optional[bool] = None
    visibility: Optional[str] = Field(default=None, max_length=24)
    shared_with_user_ids: Optional[List[UUID]] = None
    assigned_user_id: Optional[UUID] = None
    assigned_by_user_id: Optional[UUID] = None
    assigned_at: Optional[datetime] = None
    collaboration: Optional[Dict[str, Any]] = None
    policy: Optional[Dict[str, Any]] = None
    lineage: Optional[Dict[str, Any]] = None
    decomposition: Optional[Dict[str, Any]] = None


class CodingBacklogItemActionRequest(BaseModel):
    action: str = Field(
        ...,
        description=(
            "start | pause | resume | cancel | close | assign_backlog | clear_backlog_assignment | "
            "update_backlog_note | apply_override | create_patch_pr | keep_proposal_only | relaunch_slice | skip_slice"
        ),
    )
    slice_id: Optional[str] = Field(default=None, max_length=120)
    assigned_user_id: Optional[UUID] = None
    closure_reason: Optional[str] = Field(default=None, max_length=64)
    operator_note: Optional[str] = Field(default=None, max_length=5000)


class CodingBacklogItemResponse(BaseModel):
    id: UUID
    user_id: UUID
    source_id: Optional[UUID] = None
    title: str
    portfolio_goal: str
    status: str
    priority: int
    scope: Optional[str] = None
    failure_symptom: Optional[str] = None
    error_output: Optional[str] = None
    file_paths: Optional[List[str]] = None
    commands: Optional[List[str]] = None
    auto_apply_enabled: bool
    require_patch_pr: bool
    visibility: str = "private"
    shared_with_user_ids: List[UUID] = Field(default_factory=list)
    assigned_user_id: Optional[UUID] = None
    assigned_by_user_id: Optional[UUID] = None
    assigned_at: Optional[datetime] = None
    collaboration: Optional[Dict[str, Any]] = None
    collaboration_summary: Optional[CollaborationSummaryResponse] = None
    operator_queue_state: Optional[str] = None
    closure_reason: Optional[str] = None
    why_not_repair: Optional[Dict[str, Any]] = None
    policy: Optional[Dict[str, Any]] = None
    lineage: Optional[Dict[str, Any]] = None
    decomposition: Optional[Dict[str, Any]] = None
    child_job_ids: Optional[List[str]] = None
    latest_summary: Optional[Dict[str, Any]] = None
    orchestrator_job_id: Optional[UUID] = None
    current_job_id: Optional[UUID] = None
    latest_apply_job_id: Optional[UUID] = None
    latest_proposal_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class CodingBacklogItemListResponse(BaseModel):
    items: List[CodingBacklogItemResponse]
    total: int
    limit: int
    offset: int
