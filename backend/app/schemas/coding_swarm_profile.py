from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.coding_backlog import CollaborationSummaryResponse


class CodingSwarmProfileCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    source_id: UUID
    preset_key: str = Field(..., min_length=1, max_length=48)
    description: Optional[str] = Field(default=None, max_length=4000)
    scope_default: str = Field(default="auto", max_length=32)
    default_commands: Optional[List[str]] = None
    default_file_paths: Optional[List[str]] = None
    max_agents: int = Field(default=4, ge=1, le=4)
    safe_command_policy: str = Field(default="standard", max_length=32)
    saved_search_query: Optional[str] = Field(default=None, max_length=500)
    is_default: bool = False
    visibility: str = Field(default="private", max_length=24)
    shared_with_user_ids: Optional[List[UUID]] = None
    profile_metadata: Optional[Dict[str, Any]] = None


class CodingSwarmProfileUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=4000)
    preset_key: Optional[str] = Field(default=None, min_length=1, max_length=48)
    scope_default: Optional[str] = Field(default=None, max_length=32)
    default_commands: Optional[List[str]] = None
    default_file_paths: Optional[List[str]] = None
    max_agents: Optional[int] = Field(default=None, ge=1, le=4)
    safe_command_policy: Optional[str] = Field(default=None, max_length=32)
    saved_search_query: Optional[str] = Field(default=None, max_length=500)
    is_default: Optional[bool] = None
    status: Optional[str] = Field(default=None, max_length=24)
    visibility: Optional[str] = Field(default=None, max_length=24)
    shared_with_user_ids: Optional[List[UUID]] = None
    profile_metadata: Optional[Dict[str, Any]] = None


class CodingSwarmProfileResponse(BaseModel):
    id: UUID
    user_id: UUID
    source_id: UUID
    title: str
    description: Optional[str] = None
    status: str
    preset_key: str
    scope_default: str
    default_commands: Optional[List[str]] = None
    default_file_paths: Optional[List[str]] = None
    max_agents: int
    safe_command_policy: str
    saved_search_query: Optional[str] = None
    is_default: bool
    visibility: str
    shared_with_user_ids: List[UUID] = Field(default_factory=list)
    collaboration_summary: Optional[CollaborationSummaryResponse] = None
    latest_job_id: Optional[UUID] = None
    profile_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CodingSwarmProfileListResponse(BaseModel):
    items: List[CodingSwarmProfileResponse] = Field(default_factory=list)
    total: int
    limit: int
    offset: int
