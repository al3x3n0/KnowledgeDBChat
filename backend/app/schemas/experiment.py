"""
Pydantic schemas for experiment planning + run tracking.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ExperimentPlanGenerateRequest(BaseModel):
    note_id: UUID
    max_note_chars: int = Field(default=12000, ge=500, le=60000)
    prefer_section: str = Field(default="hypothesis", pattern="^(hypothesis|full_note)$")
    include_ablations: bool = True
    include_timeline: bool = True
    include_risks: bool = True
    include_repro_checklist: bool = True


class ExperimentPlanResponse(BaseModel):
    id: UUID
    user_id: UUID
    research_note_id: UUID
    title: str
    hypothesis_text: Optional[str] = None
    plan: Dict[str, Any]
    generator: Optional[str] = None
    generator_details: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ExperimentPlanListResponse(BaseModel):
    plans: List[ExperimentPlanResponse]


class ExperimentPlanUpdateRequest(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=500)
    hypothesis_text: Optional[str] = None
    plan: Optional[Dict[str, Any]] = None


class ExperimentRunCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=500)
    config: Optional[Dict[str, Any]] = None
    summary: Optional[str] = Field(default=None, max_length=20000)


class ExperimentRunUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=500)
    status: Optional[str] = Field(default=None, pattern="^(planned|running|completed|failed|cancelled)$")
    progress: Optional[int] = Field(default=None, ge=0, le=100)
    config: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    summary: Optional[str] = Field(default=None, max_length=20000)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ExperimentRunResponse(BaseModel):
    id: UUID
    user_id: UUID
    experiment_plan_id: UUID
    agent_job_id: Optional[UUID] = None
    name: str
    status: str
    config: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    progress: int
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ExperimentRunListResponse(BaseModel):
    runs: List[ExperimentRunResponse]


class ExperimentRunStartRequest(BaseModel):
    source_id: UUID
    commands: List[str] = Field(default_factory=list, max_length=12)
    latex_project_id: Optional[UUID] = None
    timeout_seconds: int = Field(default=30, ge=5, le=600)
    start_immediately: bool = True


class ExperimentRunStartResponse(BaseModel):
    run: ExperimentRunResponse
    agent_job_id: UUID


class ExperimentRunSyncResponse(BaseModel):
    run: ExperimentRunResponse
