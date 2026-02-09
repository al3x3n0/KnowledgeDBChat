"""
Pydantic schemas for research monitor profiles.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ResearchMonitorProfileResponse(BaseModel):
    id: UUID
    user_id: UUID
    customer: Optional[str] = None
    token_scores: Optional[Dict[str, int]] = None
    muted_tokens: Optional[List[str]] = None
    muted_patterns: Optional[List[str]] = None
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ResearchMonitorProfileUpdateRequest(BaseModel):
    muted_tokens: Optional[List[str]] = Field(default=None)
    muted_patterns: Optional[List[str]] = Field(default=None)
    notes: Optional[str] = Field(default=None, max_length=4000)

