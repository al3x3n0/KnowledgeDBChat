from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class ScientificSandboxProfile(Base):
    __tablename__ = "scientific_sandbox_profiles"

    id = Column(String(80), primary_key=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    track_type = Column(String(32), nullable=False, default="generic", index=True)
    backend = Column(String(24), nullable=False, default="docker")
    docker_image = Column(String(255), nullable=True)
    timeout_seconds = Column(Integer, nullable=False, default=900)
    resource_caps = Column(JSON, nullable=False, default=dict)
    allowed_benchmark_families = Column(JSON, nullable=False, default=list)
    allowed_perf_collectors = Column(JSON, nullable=False, default=list)
    required_capabilities = Column(JSON, nullable=False, default=list)
    toolchains = Column(JSON, nullable=False, default=list)
    budget_limit_default = Column(Float, nullable=False, default=25.0)
    enabled = Column(Boolean, nullable=False, default=True, index=True)
    system_managed = Column(Boolean, nullable=False, default=False, index=True)
    is_default = Column(Boolean, nullable=False, default=False)
    created_by_user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "track_type": self.track_type,
            "backend": self.backend,
            "docker_image": self.docker_image,
            "timeout_seconds": self.timeout_seconds,
            "resource_caps": dict(self.resource_caps or {}),
            "allowed_benchmark_families": list(self.allowed_benchmark_families or []),
            "allowed_perf_collectors": list(self.allowed_perf_collectors or []),
            "required_capabilities": list(self.required_capabilities or []),
            "toolchains": list(self.toolchains or []),
            "budget_limit_default": float(self.budget_limit_default or 0.0),
            "enabled": bool(self.enabled),
            "system_managed": bool(self.system_managed),
            "is_default": bool(self.is_default),
            "created_by_user_id": str(self.created_by_user_id) if self.created_by_user_id else None,
            "created_at": self.created_at.isoformat() if isinstance(self.created_at, datetime) else None,
            "updated_at": self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else None,
        }
