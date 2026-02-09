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
    metadata: Optional[Dict[str, Any]] = Field(default=None, validation_alias="item_metadata")

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
