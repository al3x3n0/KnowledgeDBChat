"""Schemas for retrieval traces."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel


class RetrievalTraceResponse(BaseModel):
    id: UUID
    user_id: Optional[UUID] = None
    session_id: Optional[UUID] = None
    chat_message_id: Optional[UUID] = None
    trace_type: str
    query: str
    processed_query: Optional[str] = None
    provider: Optional[str] = None
    settings_snapshot: Optional[Dict[str, Any]] = None
    trace: Dict[str, Any]
    created_at: datetime

    class Config:
        from_attributes = True

