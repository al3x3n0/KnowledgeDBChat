"""Pydantic schemas for artifact draft review flow."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ArtifactDraftListItem(BaseModel):
    id: UUID
    artifact_type: str
    source_id: Optional[UUID] = None
    title: str
    status: str
    created_at: datetime
    updated_at: datetime
    published_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ArtifactDraftListResponse(BaseModel):
    items: List[ArtifactDraftListItem]
    total: int
    limit: int
    offset: int


class ArtifactDraftResponse(BaseModel):
    id: UUID
    user_id: UUID
    artifact_type: str
    source_id: Optional[UUID] = None
    title: str
    description: Optional[str] = None
    status: str
    draft_payload: Dict[str, Any] = Field(default_factory=dict)
    published_payload: Optional[Dict[str, Any]] = None
    approvals: Optional[List[Dict[str, Any]]] = None
    created_at: datetime
    updated_at: datetime
    published_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ArtifactDraftApproveRequest(BaseModel):
    note: Optional[str] = Field(None, max_length=1000)


class ArtifactDraftSubmitRequest(BaseModel):
    note: Optional[str] = Field(None, max_length=1000)

