"""
API Key model for external tool authentication.
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, String, Text, DateTime, Boolean, JSON, ForeignKey, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

from app.core.database import Base


class APIKey(Base):
    """API Key model for authenticating external tools and services."""

    __tablename__ = "api_keys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Key identification
    name = Column(String(100), nullable=False)  # Human-readable name
    description = Column(Text, nullable=True)  # Optional description
    key_prefix = Column(String(8), nullable=False, index=True)  # First 8 chars for identification
    key_hash = Column(String(128), nullable=False)  # SHA256 hash of the full key

    # Ownership
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    user = relationship("User", backref="api_keys")

    # Permissions and scopes
    scopes = Column(JSON, nullable=True)  # List of allowed scopes/permissions
    # Scopes can be: "read", "write", "admin", "chat", "documents", "workflows", etc.

    # Rate limiting
    rate_limit_per_minute = Column(Integer, default=60)  # Requests per minute
    rate_limit_per_day = Column(Integer, default=10000)  # Requests per day

    # Status
    is_active = Column(Boolean, default=True)

    # Expiration
    expires_at = Column(DateTime(timezone=True), nullable=True)  # Null = never expires

    # Usage tracking
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    last_used_ip = Column(String(45), nullable=True)  # IPv4 or IPv6
    usage_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    revoked_at = Column(DateTime(timezone=True), nullable=True)

    # MCP Configuration
    mcp_enabled = Column(Boolean, default=True)  # Whether MCP access is enabled
    allowed_tools = Column(JSON, nullable=True)  # List of allowed tool names, null = all tools
    source_access_mode = Column(String(20), default="all")  # "all" or "restricted"

    def __repr__(self):
        return f"<APIKey(id={self.id}, name='{self.name}', user_id={self.user_id})>"

    def is_valid(self) -> bool:
        """Check if the API key is currently valid."""
        if not self.is_active:
            return False
        if self.revoked_at is not None:
            return False
        if self.expires_at is not None and datetime.utcnow() > self.expires_at:
            return False
        return True

    def has_scope(self, scope: str) -> bool:
        """Check if the API key has a specific scope."""
        if not self.scopes:
            return True  # No scopes means full access
        if "admin" in self.scopes:
            return True
        if "*" in self.scopes:
            return True
        return scope in self.scopes

    def has_any_scope(self, scopes: List[str]) -> bool:
        """Check if the API key has any of the specified scopes."""
        return any(self.has_scope(scope) for scope in scopes)

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a specific MCP tool is allowed for this API key."""
        if not self.mcp_enabled:
            return False
        if self.allowed_tools is None:
            return True  # No restrictions = all tools allowed
        return tool_name in self.allowed_tools

    def is_mcp_enabled(self) -> bool:
        """Check if MCP access is enabled for this key."""
        return self.mcp_enabled if self.mcp_enabled is not None else True


class APIKeyUsageLog(Base):
    """Log of API key usage for auditing and analytics."""

    __tablename__ = "api_key_usage_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Key reference
    api_key_id = Column(UUID(as_uuid=True), ForeignKey("api_keys.id", ondelete="CASCADE"), nullable=False)
    api_key = relationship("APIKey", backref="usage_logs")

    # Request details
    endpoint = Column(String(500), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=True)

    # Client info
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)

    # Timing
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow, index=True)
    response_time_ms = Column(Integer, nullable=True)

    def __repr__(self):
        return f"<APIKeyUsageLog(api_key_id={self.api_key_id}, endpoint='{self.endpoint}')>"
