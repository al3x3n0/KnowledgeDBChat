from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel


class ToolAuditResponse(BaseModel):
    id: UUID
    user_id: UUID
    agent_definition_id: Optional[UUID] = None
    conversation_id: Optional[UUID] = None
    tool_name: str
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: Optional[Any] = None
    policy_decision: Optional[Dict[str, Any]] = None
    status: str
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None
    approval_required: bool
    approval_mode: Optional[str] = None
    approval_status: Optional[str] = None
    approved_by: Optional[UUID] = None
    approved_at: Optional[datetime] = None
    approval_note: Optional[str] = None
    owner_approved_by: Optional[UUID] = None
    owner_approved_at: Optional[datetime] = None
    admin_approved_by: Optional[UUID] = None
    admin_approved_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class ToolApprovalRequest(BaseModel):
    note: Optional[str] = None
