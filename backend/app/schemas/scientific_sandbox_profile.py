from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _normalize_string(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _normalize_string_list(value: Any, *, limit: int = 24) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= limit:
            break
    return out


class ScientificSandboxProfileResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    track_type: str
    backend: str
    docker_image: Optional[str] = None
    timeout_seconds: int
    resource_caps: Dict[str, Any] = Field(default_factory=dict)
    allowed_benchmark_families: List[str] = Field(default_factory=list)
    allowed_perf_collectors: List[str] = Field(default_factory=list)
    required_capabilities: List[str] = Field(default_factory=list)
    toolchains: List[str] = Field(default_factory=list)
    budget_limit_default: float = 25.0
    enabled: bool = True
    system_managed: bool = True
    is_default: bool = False
    created_by_user_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class ScientificSandboxProfileCreate(BaseModel):
    id: str = Field(..., min_length=3, max_length=80)
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    track_type: str = Field(default="generic", min_length=1, max_length=32)
    backend: str = Field(default="docker", min_length=1, max_length=24)
    docker_image: Optional[str] = None
    timeout_seconds: int = Field(default=900, ge=30, le=7200)
    resource_caps: Dict[str, Any] = Field(default_factory=dict)
    allowed_benchmark_families: List[str] = Field(default_factory=list)
    allowed_perf_collectors: List[str] = Field(default_factory=list)
    required_capabilities: List[str] = Field(default_factory=list)
    toolchains: List[str] = Field(default_factory=list)
    budget_limit_default: float = Field(default=25.0, ge=1.0, le=10000.0)
    enabled: bool = True
    is_default: bool = False

    @field_validator(
        "id",
        "name",
        "description",
        "track_type",
        "backend",
        "docker_image",
        mode="before",
    )
    @classmethod
    def _normalize_text(cls, value: Any) -> Optional[str]:
        return _normalize_string(value)

    @field_validator(
        "allowed_benchmark_families",
        "allowed_perf_collectors",
        "required_capabilities",
        "toolchains",
        mode="before",
    )
    @classmethod
    def _normalize_lists(cls, value: Any) -> List[str]:
        return _normalize_string_list(value)


class ScientificSandboxProfileUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = None
    track_type: Optional[str] = Field(default=None, min_length=1, max_length=32)
    backend: Optional[str] = Field(default=None, min_length=1, max_length=24)
    docker_image: Optional[str] = None
    timeout_seconds: Optional[int] = Field(default=None, ge=30, le=7200)
    resource_caps: Optional[Dict[str, Any]] = None
    allowed_benchmark_families: Optional[List[str]] = None
    allowed_perf_collectors: Optional[List[str]] = None
    required_capabilities: Optional[List[str]] = None
    toolchains: Optional[List[str]] = None
    budget_limit_default: Optional[float] = Field(default=None, ge=1.0, le=10000.0)
    enabled: Optional[bool] = None
    is_default: Optional[bool] = None

    @field_validator(
        "name", "description", "track_type", "backend", "docker_image", mode="before"
    )
    @classmethod
    def _normalize_optional_text(cls, value: Any) -> Optional[str]:
        return _normalize_string(value)

    @field_validator(
        "allowed_benchmark_families",
        "allowed_perf_collectors",
        "required_capabilities",
        "toolchains",
        mode="before",
    )
    @classmethod
    def _normalize_optional_lists(cls, value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        return _normalize_string_list(value)


class ScientificSandboxProfileListResponse(BaseModel):
    items: List[ScientificSandboxProfileResponse] = Field(default_factory=list)
    total: int = 0
