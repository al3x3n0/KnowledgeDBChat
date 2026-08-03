"""
Pydantic schemas for experiment planning + run tracking.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    # Imported lazily to break the schema import cycle:
    # experiment -> agent_job -> domain_research_profile -> experiment.
    # agent_job.py injects these names and rebuilds ExperimentRunResponse
    # once it finishes loading.
    from app.schemas.agent_job import (
        AgentJobExperimentRunResponse,
        AgentJobOperatorInterventionResponse,
    )


EXPERIMENT_RUN_STATUS_PATTERN = "^(pending|planned|queued|provisioning|running|paused|succeeded|completed|failed|blocked|cancelled)$"


class ExperimentPlanGenerateRequest(BaseModel):
    note_id: UUID
    max_note_chars: int = Field(default=12000, ge=500, le=60000)
    prefer_section: str = Field(
        default="hypothesis", pattern="^(hypothesis|full_note)$"
    )
    plan_mode: Optional[str] = Field(
        default=None,
        pattern="^(aggregate_note|single_hypothesis|compiler_regression_followup)$",
    )
    hypothesis_id: Optional[str] = None
    benchmark_suite_id: Optional[str] = Field(default=None, max_length=120)
    benchmark_case_ids: List[str] = Field(default_factory=list, max_length=24)
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
    benchmark_family: Optional[str] = None
    benchmark_suite_id: Optional[str] = None
    benchmark_case_ids: List[str] = Field(default_factory=list)
    benchmark_baseline_id: Optional[str] = None
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
    name: Optional[str] = Field(default=None, min_length=1, max_length=500)
    config: Optional[Dict[str, Any]] = None
    summary: Optional[str] = Field(default=None, max_length=20000)


class ExperimentRunUpdateRequest(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=500)
    status: Optional[str] = Field(default=None, pattern=EXPERIMENT_RUN_STATUS_PATTERN)
    progress: Optional[int] = Field(default=None, ge=0, le=100)
    config: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    summary: Optional[str] = Field(default=None, max_length=20000)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ExperimentRunOperatorActionResponse(BaseModel):
    action: str
    actor_user_id: Optional[str] = None
    at: Optional[str] = None
    note: Optional[str] = None
    previous_status: Optional[str] = None
    new_status: Optional[str] = None
    linked_job_id: Optional[str] = None
    linked_job_action: Optional[str] = None
    outcome_status: Optional[str] = None
    outcome_reason: Optional[str] = None
    parent_run_id: Optional[str] = None
    child_run_id: Optional[str] = None


class CompilerArtifactSummaryResponse(BaseModel):
    source_run_ids: List[str] = Field(default_factory=list)
    primary_run_id: Optional[str] = None
    comparison_run_id: Optional[str] = None
    explanation_note_id: Optional[UUID] = None
    explanation_synthesis_job_id: Optional[UUID] = None
    explanation_synthesis_status: Optional[str] = None
    proposal_note_id: Optional[UUID] = None
    proposal_synthesis_job_id: Optional[UUID] = None
    proposal_synthesis_status: Optional[str] = None
    patch_draft_note_id: Optional[UUID] = None
    patch_draft_synthesis_job_id: Optional[UUID] = None
    patch_draft_synthesis_status: Optional[str] = None
    source_explanation_note_id: Optional[UUID] = None
    source_proposal_note_id: Optional[UUID] = None
    source_id: Optional[str] = None
    source_name: Optional[str] = None
    available_actions: List[str] = Field(default_factory=list)


class ScientificValidationRunSummaryResponse(BaseModel):
    id: UUID
    agent_job_id: Optional[UUID] = None
    name: str
    status: str
    progress: int
    validation_kind: Optional[str] = None
    sandbox_profile_id: Optional[str] = None
    sandbox_profile_name: Optional[str] = None
    recipe_family: Optional[str] = None
    recipe_id: Optional[str] = None
    benchmark_family: Optional[str] = None
    benchmark_suite_id: Optional[str] = None
    benchmark_case_ids: List[str] = Field(default_factory=list)
    blocked_reason_code: Optional[str] = None
    hypothesis_id: Optional[str] = None
    track_type: Optional[str] = None
    domain_research_profile_id: Optional[UUID] = None
    research_portfolio_id: Optional[UUID] = None
    parent_run_id: Optional[UUID] = None
    latest_child_run_id: Optional[UUID] = None
    retry_count: int = 0
    latest_operator_action: Optional[str] = None
    latest_operator_outcome_status: Optional[str] = None
    compiler_artifact_summary: Optional[CompilerArtifactSummaryResponse] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class ExperimentRunResponse(BaseModel):
    id: UUID
    user_id: UUID
    experiment_plan_id: UUID
    agent_job_id: Optional[UUID] = None
    parent_run_id: Optional[UUID] = None
    latest_child_run_id: Optional[UUID] = None
    name: str
    status: str
    config: Optional[Dict[str, Any]] = None
    results: Optional[Dict[str, Any]] = None
    validation_kind: Optional[str] = None
    sandbox_profile_id: Optional[str] = None
    recipe_family: Optional[str] = None
    recipe_id: Optional[str] = None
    recipe_version: Optional[int] = None
    domain_research_profile_id: Optional[str] = None
    research_portfolio_id: Optional[str] = None
    hypothesis_id: Optional[str] = None
    originating_job_id: Optional[str] = None
    blocked_reason_code: Optional[str] = None
    capability_check: Optional[Dict[str, Any]] = None
    profile_snapshot: Optional[Dict[str, Any]] = None
    recipe_snapshot: Optional[Dict[str, Any]] = None
    benchmark_family: Optional[str] = None
    benchmark_suite_id: Optional[str] = None
    benchmark_case_ids: List[str] = Field(default_factory=list)
    benchmark_baseline_id: Optional[str] = None
    measurement_summary: Optional[Dict[str, Any]] = None
    compiler_artifacts: Optional[Dict[str, Any]] = None
    perf_counters: Optional[Dict[str, Any]] = None
    artifact_inventory: List[str] = Field(default_factory=list)
    repeat_count: Optional[int] = None
    experiment_run: Optional[AgentJobExperimentRunResponse] = None
    operator_interventions: Optional[List[AgentJobOperatorInterventionResponse]] = None
    operator_actions: Optional[List[ExperimentRunOperatorActionResponse]] = None
    summary: Optional[str] = None
    progress: int
    retry_count: int = 0
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


class ExperimentRunActionRequest(BaseModel):
    action: str = Field(
        ...,
        pattern="^(start|sync|pause|resume|cancel|retry|requeue)$",
    )
    note: Optional[str] = Field(default=None, max_length=2000)
    start_immediately: bool = True


class ExperimentRunActionResponse(BaseModel):
    run: ExperimentRunResponse
    agent_job_id: Optional[UUID] = None
