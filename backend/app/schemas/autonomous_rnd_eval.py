"""Schemas for grading persisted autonomous R&D job trajectories."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator


class AutonomousRnDEvalTrialBinding(BaseModel):
    task_id: str = Field(..., min_length=1, max_length=200)
    job_ids: List[UUID] = Field(..., min_length=1, max_length=100)

    @field_validator("task_id")
    @classmethod
    def normalize_task_id(cls, value: str) -> str:
        return value.strip()

    @field_validator("job_ids")
    @classmethod
    def unique_job_ids(cls, value: List[UUID]) -> List[UUID]:
        if len(value) != len(set(value)):
            raise ValueError("job_ids must be unique within a task")
        return value


class AutonomousRnDEvalGradeJobsRequest(BaseModel):
    suite_id: str = Field("compiler_research_v1", min_length=1, max_length=200)
    trials: List[AutonomousRnDEvalTrialBinding] = Field(
        ..., min_length=1, max_length=100
    )
    persist: bool = False
    label: Optional[str] = Field(None, min_length=1, max_length=200)

    @field_validator("suite_id")
    @classmethod
    def normalize_suite_id(cls, value: str) -> str:
        return value.strip()

    @field_validator("label")
    @classmethod
    def normalize_label(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() or None if value else None

    @model_validator(mode="after")
    def unique_task_bindings(self):
        task_ids = [binding.task_id for binding in self.trials]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task_id bindings must be unique")
        return self


class AutonomousRnDEvalGradeJobsResponse(BaseModel):
    report: Dict[str, Any]
    evaluated_job_count: int
    task_bindings: Dict[str, List[str]]
    run_id: Optional[UUID] = None


class AutonomousRnDEvalSuiteListResponse(BaseModel):
    suites: List[Dict[str, Any]]


class AutonomousRnDEvalRunSummary(BaseModel):
    id: UUID
    suite_id: str
    suite_name: str
    suite_version: int
    label: Optional[str] = None
    source: str
    is_baseline: bool
    task_count: int
    trial_count: int
    mean_score: float
    pass_at_k: float
    pass_pow_k: float
    created_at: datetime

    class Config:
        from_attributes = True


class AutonomousRnDEvalRunListResponse(BaseModel):
    runs: List[AutonomousRnDEvalRunSummary]


class AutonomousRnDEvalRunDetailResponse(AutonomousRnDEvalRunSummary):
    report: Dict[str, Any]
    task_bindings: Optional[Dict[str, List[str]]] = None


class AutonomousRnDEvalRunComparisonResponse(BaseModel):
    comparison: Dict[str, Any]


class AutonomousRnDVerificationLaunchRequest(BaseModel):
    approval_confirmed: Literal[True]
    approval_note: str = Field(..., min_length=3, max_length=2000)
    research_note_id: UUID
    source_id: UUID
    sandbox_profile_id: str = Field(..., min_length=1, max_length=80)
    commands: List[str] = Field(..., min_length=1, max_length=4)
    repeat_count: int = Field(..., ge=2, le=10)
    timeout_seconds: int = Field(..., ge=5, le=600)
    max_runtime_minutes: int = Field(..., ge=1, le=60)
    budget_limit: float = Field(..., gt=0, le=1000)
    start_immediately: bool = False

    @field_validator("approval_note", "sandbox_profile_id")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("commands")
    @classmethod
    def normalize_commands(cls, value: List[str]) -> List[str]:
        commands = [item.strip() for item in value if item.strip()]
        if not commands:
            raise ValueError("commands must contain at least one non-empty command")
        if len(commands) != len(set(commands)):
            raise ValueError("commands must be unique")
        if any(len(item) > 1000 for item in commands):
            raise ValueError("commands must not exceed 1000 characters")
        return commands


class AutonomousRnDVerificationLaunchResponse(BaseModel):
    created: bool
    queued: bool
    experiment_plan_id: UUID
    experiment_run_id: UUID
    agent_job_id: UUID
    audit_id: UUID
    status: str
    budget: Dict[str, Any]


class AutonomousRnDJobOutcomeResponse(BaseModel):
    job_id: UUID
    job_status: str
    outcome: Dict[str, Any]
    verification_lifecycle: Dict[str, Any]


class AutonomousRnDVerificationAuditSnapshotRequest(BaseModel):
    task_id: Optional[str] = Field(None, min_length=1, max_length=200)
    status: Optional[str] = Field(None, min_length=1, max_length=80)

    @field_validator("task_id", "status")
    @classmethod
    def normalize_optional_filter(cls, value: Optional[str]) -> Optional[str]:
        return value.strip() if value else None


class AutonomousRnDVerificationAuditEnvelope(BaseModel):
    snapshot: Dict[str, Any]
    integrity: Dict[str, Any]


class AutonomousRnDVerificationAuditVerifyResponse(BaseModel):
    valid: bool
    reason: str
    job_id: Optional[str] = None
    registry_id: Optional[str] = None
    sha256: Optional[str] = None
    key_id: Optional[str] = None


class AutonomousRnDVerificationAuditSnapshotListItem(BaseModel):
    registry_id: UUID
    job_id: UUID
    created_at: datetime
    filters: Dict[str, Any]
    sha256: str
    key_id: str
    signature_algorithm: str


class AutonomousRnDVerificationAuditSnapshotListResponse(BaseModel):
    items: List[AutonomousRnDVerificationAuditSnapshotListItem]


class AutonomousRnDVerificationAuditKey(BaseModel):
    key_id: str
    algorithm: str
    encoding: str
    public_key: str
    fingerprint_sha256: str
    status: str


class AutonomousRnDVerificationAuditKeyListResponse(BaseModel):
    keys: List[AutonomousRnDVerificationAuditKey]
