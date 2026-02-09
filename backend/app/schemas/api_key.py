"""
Pydantic schemas for API key operations.
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID
from pydantic import BaseModel, Field


class APIKeyCreate(BaseModel):
    """Schema for creating a new API key."""
    name: str = Field(..., min_length=1, max_length=100, description="Human-readable name for the key")
    description: Optional[str] = Field(None, max_length=500, description="Optional description")
    scopes: Optional[List[str]] = Field(
        None,
        description="List of allowed scopes. None means full access. Options: read, write, chat, documents, workflows, admin"
    )
    expires_in_days: Optional[int] = Field(
        None,
        ge=1,
        le=365,
        description="Days until expiration. None means never expires"
    )
    rate_limit_per_minute: int = Field(60, ge=1, le=1000, description="Max requests per minute")
    rate_limit_per_day: int = Field(10000, ge=1, le=1000000, description="Max requests per day")


class APIKeyUpdate(BaseModel):
    """Schema for updating an API key."""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    scopes: Optional[List[str]] = None
    rate_limit_per_minute: Optional[int] = Field(None, ge=1, le=1000)
    rate_limit_per_day: Optional[int] = Field(None, ge=1, le=1000000)
    is_active: Optional[bool] = None


class APIKeyResponse(BaseModel):
    """Schema for API key response (without the actual key)."""
    id: UUID
    name: str
    description: Optional[str] = None
    key_prefix: str  # First 8 chars for identification
    scopes: Optional[List[str]] = None
    rate_limit_per_minute: int
    rate_limit_per_day: int
    is_active: bool
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    last_used_ip: Optional[str] = None
    usage_count: int
    created_at: datetime
    revoked_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class APIKeyCreateResponse(BaseModel):
    """Schema for API key creation response (includes the actual key - shown only once!)."""
    id: UUID
    name: str
    description: Optional[str] = None
    key_prefix: str
    api_key: str  # The full API key - ONLY shown at creation time!
    scopes: Optional[List[str]] = None
    rate_limit_per_minute: int
    rate_limit_per_day: int
    expires_at: Optional[datetime] = None
    created_at: datetime
    message: str = "Store this API key securely. It will not be shown again!"


class APIKeyListResponse(BaseModel):
    """Schema for listing API keys."""
    api_keys: List[APIKeyResponse]
    total: int


class APIKeyUsageStats(BaseModel):
    """Schema for API key usage statistics."""
    key_id: str
    key_name: str
    period_days: int
    total_requests: int
    lifetime_requests: int
    last_used_at: Optional[str] = None
    top_endpoints: List[dict]


class APIKeyValidationResult(BaseModel):
    """Schema for API key validation result (internal use)."""
    is_valid: bool
    user_id: Optional[UUID] = None
    key_id: Optional[UUID] = None
    scopes: Optional[List[str]] = None
    error: Optional[str] = None
