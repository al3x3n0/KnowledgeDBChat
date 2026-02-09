"""
Schemas for tool policies.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ToolPolicyResponse(BaseModel):
    id: UUID
    subject_type: str
    subject_id: Optional[UUID] = None
    subject_key: Optional[str] = None
    tool_name: str
    effect: str
    require_approval: bool
    constraints: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ToolPolicyCreate(BaseModel):
    tool_name: str = Field(..., min_length=1, max_length=120, description="Exact tool name or '*'")
    effect: str = Field(default="deny", pattern="^(allow|deny)$")
    require_approval: bool = False
    constraints: Optional[Dict[str, Any]] = None


class AdminToolPolicyCreate(ToolPolicyCreate):
    subject_type: str = Field(default="user", pattern="^(global|role|user|agent_definition|api_key)$")
    subject_id: Optional[str] = None
    subject_key: Optional[str] = None


class ToolPolicyEvaluateRequest(BaseModel):
    tool_name: str = Field(..., min_length=1, max_length=200)
    tool_args: Optional[Dict[str, Any]] = None
    agent_definition_id: Optional[UUID] = None
    api_key_id: Optional[UUID] = None


class ToolPolicyEvaluateResponse(BaseModel):
    tool_name: str
    allowed: bool
    require_approval: bool
    denied_reason: Optional[str] = None
    matched_policies: Optional[list[dict]] = None
