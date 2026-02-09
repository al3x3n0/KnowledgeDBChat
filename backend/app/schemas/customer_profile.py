"""
Customer profile schemas.

Stored as a deployment-level configuration (admin-managed), used to tailor
AI Scientist recommendations without hard-coding vertical-specific logic.
"""

from __future__ import annotations

from typing import List, Optional, Literal
from uuid import UUID

from pydantic import BaseModel, Field


WorkflowId = Literal["triage", "extraction", "literature"]


class CustomerProfile(BaseModel):
    id: Optional[UUID] = None
    name: str = Field(..., min_length=1, max_length=200)
    keywords: List[str] = Field(default_factory=list, description="Lowercase keywords/phrases describing the customer domain")
    preferred_workflows: List[WorkflowId] = Field(default_factory=list)
    notes: Optional[str] = Field(None, max_length=2000)


class CustomerProfileGetResponse(BaseModel):
    profile: Optional[CustomerProfile] = None
    raw: Optional[str] = None


class CustomerProfileSetRequest(BaseModel):
    profile: CustomerProfile


class CustomerProfileSetResponse(BaseModel):
    ok: bool
    profile: CustomerProfile
    raw: str
