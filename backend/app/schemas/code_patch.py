"""
Schemas for code patch proposals.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class CodePatchProposalResponse(BaseModel):
    id: UUID
    user_id: UUID
    job_id: Optional[UUID] = None
    source_id: Optional[UUID] = None
    title: str
    summary: Optional[str] = None
    diff_unified: str
    metadata: Optional[Dict[str, Any]] = Field(default=None, validation_alias="proposal_metadata")
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CodePatchProposalListItem(BaseModel):
    id: UUID
    job_id: Optional[UUID] = None
    source_id: Optional[UUID] = None
    title: str
    summary: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CodePatchProposalListResponse(BaseModel):
    items: list[CodePatchProposalListItem]
    total: int
    limit: int
    offset: int


class CodePatchProposalUpdateRequest(BaseModel):
    status: Optional[str] = Field(None, description="proposed | applied | rejected")
