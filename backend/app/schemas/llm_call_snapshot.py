"""Pydantic schemas for LLM call snapshots (replay/debug observability)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel


class LLMCallSnapshotSummary(BaseModel):
    """List-view summary: correlation + outcome, no payloads."""

    id: UUID
    job_id: Optional[UUID] = None
    iteration: Optional[int] = None
    phase: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    task_type: Optional[str] = None
    error: Optional[str] = None
    latency_ms: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True


class LLMCallSnapshotResponse(LLMCallSnapshotSummary):
    """Detail view: includes the full request/response payloads."""

    request: Dict[str, Any]
    response_text: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    structured: Optional[Dict[str, Any]] = None
