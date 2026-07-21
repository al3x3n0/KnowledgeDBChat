from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BenchmarkCaseResponse(BaseModel):
    id: str
    suite_id: str
    name: str
    description: Optional[str] = None
    rank: int = 0
    source_ref: Optional[str] = None
    benchmark_query: Optional[str] = None
    compile_command_template: Optional[str] = None
    run_command_template: Optional[str] = None
    expected_artifacts: List[str] = Field(default_factory=list)
    metrics: List[Dict[str, Any]] = Field(default_factory=list)
    observability: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class BenchmarkBaselineResponse(BaseModel):
    id: str
    suite_id: str
    case_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    compiler_revision: Optional[str] = None
    toolchain_id: Optional[str] = None
    sandbox_profile_id: Optional[str] = None
    measurements: Dict[str, Any] = Field(default_factory=dict)
    environment_snapshot: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    system_managed: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class BenchmarkSuiteResponse(BaseModel):
    id: str
    user_id: Optional[str] = None
    name: str
    description: Optional[str] = None
    track_type: str
    benchmark_family: str
    suite_version: int = 1
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    system_managed: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    cases: List[BenchmarkCaseResponse] = Field(default_factory=list)
    baselines: List[BenchmarkBaselineResponse] = Field(default_factory=list)


class BenchmarkSuiteListResponse(BaseModel):
    items: List[BenchmarkSuiteResponse] = Field(default_factory=list)
    total: int = 0
