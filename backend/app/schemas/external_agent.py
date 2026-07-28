"""Schemas for the external-agent gateway."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


class ExternalAgentCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=2000)
    provider_type: Literal["generic_agent", "compops"] = "generic_agent"
    endpoint_url: str = Field(..., min_length=1, max_length=2000)
    capabilities: List[str] = Field(..., min_length=1, max_length=100)
    auth_type: Literal["none", "bearer", "api_key"] = "none"
    secret_id: Optional[UUID] = None
    auth_header_name: str = Field("X-API-Key", min_length=1, max_length=64)
    timeout_seconds: int = Field(30, ge=2, le=120)
    is_enabled: bool = True

    @field_validator("name", "endpoint_url", "auth_header_name")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("capabilities")
    @classmethod
    def normalize_capabilities(cls, values: List[str]) -> List[str]:
        normalized = []
        for value in values:
            capability = str(value or "").strip().lower()
            if capability and capability not in normalized:
                normalized.append(capability)
        if not normalized:
            raise ValueError("At least one capability is required")
        return normalized

    @model_validator(mode="after")
    def require_secret_for_auth(self):
        if self.auth_type != "none" and self.secret_id is None:
            raise ValueError("secret_id is required for authenticated agents")
        return self


class ExternalAgentResponse(BaseModel):
    id: UUID
    name: str
    description: Optional[str] = None
    provider_type: str
    endpoint_url: str
    capabilities: List[str]
    auth_type: str
    secret_id: Optional[UUID] = None
    auth_header_name: Optional[str] = None
    timeout_seconds: int
    is_enabled: bool
    version: int
    created_at: datetime
    updated_at: datetime


class ExternalAgentListResponse(BaseModel):
    agents: List[ExternalAgentResponse]
    total: int


class ExternalAgentInvokeRequest(BaseModel):
    capability: str = Field(..., min_length=1, max_length=120)
    payload: Dict[str, Any] = Field(default_factory=dict)
    request_id: Optional[str] = Field(None, min_length=1, max_length=200)
    agent_job_id: Optional[UUID] = None

    @field_validator("capability")
    @classmethod
    def normalize_capability(cls, value: str) -> str:
        return value.strip().lower()


class ExternalAgentInvokeResponse(BaseModel):
    status: Literal["completed", "requires_approval", "failed"]
    audit_id: UUID
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    evidence_linked: bool = False


class CompOpsEvidenceSubscriptionCreateRequest(BaseModel):
    tool_id: UUID
    capability: str = Field(..., min_length=1, max_length=120)
    payload: Dict[str, Any]
    interval_minutes: int = Field(15, ge=5, le=1440)
    sync_immediately: bool = True

    @field_validator("capability")
    @classmethod
    def normalize_sync_capability(cls, value: str) -> str:
        return value.strip().lower()


class CompOpsEvidenceSubscriptionUpdateRequest(BaseModel):
    interval_minutes: Optional[int] = Field(None, ge=5, le=1440)
    is_enabled: Optional[bool] = None


class CompOpsEvidenceSubscriptionResponse(BaseModel):
    id: UUID
    user_id: UUID
    job_id: UUID
    tool_id: UUID
    capability: str
    remote_id: str
    payload: Dict[str, Any]
    interval_minutes: int
    is_enabled: bool
    status: str
    last_response_sha256: Optional[str] = None
    last_audit_id: Optional[UUID] = None
    last_attempt_at: Optional[datetime] = None
    last_success_at: Optional[datetime] = None
    next_sync_at: Optional[datetime] = None
    last_error: Optional[str] = None
    webhook_enabled: bool = False
    last_webhook_at: Optional[datetime] = None
    last_webhook_event_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class CompOpsEvidenceSubscriptionListResponse(BaseModel):
    subscriptions: List[CompOpsEvidenceSubscriptionResponse]
    total: int


class CompOpsEvidenceSyncResponse(BaseModel):
    subscription: CompOpsEvidenceSubscriptionResponse
    evidence_changed: bool


class CompOpsWebhookSetupResponse(BaseModel):
    subscription: CompOpsEvidenceSubscriptionResponse
    callback_path: str
    signing_secret: str
    signature_header: str = "X-CompOps-Signature"
    timestamp_header: str = "X-CompOps-Timestamp"
    event_id_header: str = "X-CompOps-Event-ID"
    signing_format: str = "v1=hex(hmac_sha256(secret, timestamp.event_id.raw_body))"


class CompOpsWebhookReceiptResponse(BaseModel):
    accepted: bool
    duplicate: bool
    event_id: str
