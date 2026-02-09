"""
Schemas for Patch PRs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class PatchPRListItem(BaseModel):
    id: UUID
    source_id: Optional[UUID] = None
    title: str
    status: str
    selected_proposal_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PatchPRListResponse(BaseModel):
    items: List[PatchPRListItem]
    total: int
    limit: int
    offset: int


class PatchPRResponse(BaseModel):
    id: UUID
    user_id: UUID
    source_id: Optional[UUID] = None
    title: str
    description: Optional[str] = None
    status: str
    selected_proposal_id: Optional[UUID] = None
    proposal_ids: List[str] = []
    checks: Optional[Dict[str, Any]] = None
    approvals: Optional[List[Dict[str, Any]]] = None
    merged_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PatchPRCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    description: Optional[str] = Field(default=None, max_length=20000)
    source_id: Optional[str] = None
    initial_proposal_id: Optional[str] = None


class PatchPRFromChainRequest(BaseModel):
    root_job_id: str = Field(..., min_length=1)
    title: Optional[str] = Field(default=None, max_length=500)
    description: Optional[str] = Field(default=None, max_length=20000)
    proposal_strategy: str = Field(default="best_passing", pattern="^(best_passing|latest)$")
    open_after_create: bool = True


class PatchPRUpdateRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=500)
    description: Optional[str] = Field(default=None, max_length=20000)
    status: Optional[str] = Field(default=None, max_length=24)
    selected_proposal_id: Optional[str] = None


class PatchPRApproveRequest(BaseModel):
    note: Optional[str] = Field(default=None, max_length=5000)


class PatchPRMergeRequest(BaseModel):
    dry_run: bool = True
    require_approved: bool = True


class PatchPRMergeResponse(BaseModel):
    pr_id: str
    dry_run: bool
    ok: bool
    selected_proposal_id: Optional[str] = None
    applied_files: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

