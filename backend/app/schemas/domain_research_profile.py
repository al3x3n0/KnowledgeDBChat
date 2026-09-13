from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.experiment import ScientificValidationRunSummaryResponse
from app.services.scientific_validation_service import (
    normalize_portfolio_automation_policy,
    normalize_portfolio_automation_profile,
    normalize_validation_policy,
)


class ResearchOpportunityActionRequest(BaseModel):
    action: str = Field(
        ...,
        description="accept, suppress, reopen, create_plan, launch_validation, materialize_experiment, launch_follow_up, relaunch_follow_up",
    )
    operator_note: Optional[str] = Field(default=None, max_length=2000)
    start_immediately: Optional[bool] = None

    @field_validator("action", "operator_note", mode="before")
    @classmethod
    def _normalize_action_text(cls, value: Any) -> Any:
        if value is None:
            return value
        return str(value).strip()


def _normalize_query_list(value: Any) -> list[str]:
    if value is None:
        return []
    rows = (
        value if isinstance(value, list) else str(value).replace("\n", ",").split(",")
    )
    out: list[str] = []
    for row in rows:
        text = str(row or "").strip()
        if not text or text in out:
            continue
        out.append(text[:240])
        if len(out) >= 12:
            break
    return out


def _normalize_uuid_list(value: Any, *, max_items: int = 24) -> list[str]:
    if value is None:
        return []
    rows = (
        value if isinstance(value, list) else str(value).replace("\n", ",").split(",")
    )
    out: list[str] = []
    for row in rows:
        text = str(row or "").strip()
        if not text or text in out:
            continue
        out.append(text)
        if len(out) >= max_items:
            break
    return out


def _normalize_track_type(value: Any) -> str:
    text = str(value or "generic").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"micro_arch", "microarch", "uarch"}:
        return "microarchitecture"
    if text not in {"compiler", "microarchitecture", "generic"}:
        return "generic"
    return text


def _normalize_source_scope(value: Any) -> str:
    text = (
        str(value or "kb_plus_arxiv")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    if text in {"kb", "documents", "kb_first"}:
        return "kb_only"
    if text in {"arxiv", "papers"}:
        return "arxiv_only"
    if text in {"kb_plus_repo", "kb_repo", "repo"}:
        return "kb_plus_arxiv_plus_repo"
    if text not in {
        "kb_only",
        "arxiv_only",
        "kb_plus_arxiv",
        "kb_plus_arxiv_plus_repo",
    }:
        return "kb_plus_arxiv"
    return text


def _normalize_validation_policy(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    return normalize_validation_policy(value)


def _normalize_automation_policy(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    return normalize_portfolio_automation_policy(value)


def _normalize_automation_profile(value: Any, *, default: str = "balanced") -> str:
    return normalize_portfolio_automation_profile(value, default=default)


class DomainResearchProfileCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    domain: str = Field(..., min_length=1, max_length=300)
    objective: str = Field(..., min_length=1)
    customer_context: Optional[str] = None
    source_scope: str = Field(default="kb_plus_arxiv_plus_repo")
    track_type: str = Field(default="compiler")
    research_mode: str = Field(default="literature_to_hypothesis")
    monitor_queries: Optional[List[str]] = None
    repo_source_ids: Optional[List[UUID]] = None
    benchmark_queries: Optional[List[str]] = None
    report_format: str = Field(default="brief_and_report")
    scoring_policy: Optional[Dict[str, Any]] = None
    selection_policy: Optional[Dict[str, Any]] = None
    validation_policy: Optional[Dict[str, Any]] = Field(
        default=None, description="Compatibility-only legacy validation policy mirror"
    )
    automation_profile: str = Field(default="balanced")
    automation_policy: Optional[Dict[str, Any]] = None
    sandbox_profile_id: Optional[str] = None
    interval_minutes: int = Field(default=1440, ge=15, le=10080)
    persist_artifacts: bool = True
    auto_launch_follow_up: bool = True
    auto_create_experiment_plans: bool = True
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_documents: int = Field(default=10, ge=1, le=25)
    max_papers: int = Field(default=8, ge=0, le=25)
    start_immediately: bool = True

    @field_validator(
        "title",
        "domain",
        "objective",
        "customer_context",
        "sandbox_profile_id",
        mode="before",
    )
    @classmethod
    def _normalize_text(cls, value: Any) -> Any:
        if value is None:
            return value
        return str(value).strip()

    @field_validator("source_scope", mode="before")
    @classmethod
    def _normalize_source_scope(cls, value: Any) -> str:
        return _normalize_source_scope(value)

    @field_validator("track_type", mode="before")
    @classmethod
    def _normalize_track_type(cls, value: Any) -> str:
        return _normalize_track_type(value)

    @field_validator("research_mode", mode="before")
    @classmethod
    def _normalize_research_mode(cls, value: Any) -> str:
        text = (
            str(value or "literature_to_hypothesis")
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
        if text not in {"literature_to_hypothesis"}:
            return "literature_to_hypothesis"
        return text

    @field_validator("report_format", mode="before")
    @classmethod
    def _normalize_report_format(cls, value: Any) -> str:
        text = (
            str(value or "brief_and_report")
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
        if text not in {"brief_only", "report_only", "brief_and_report"}:
            return "brief_and_report"
        return text

    @field_validator("monitor_queries", mode="before")
    @classmethod
    def _normalize_monitor_queries(cls, value: Any) -> Any:
        rows = _normalize_query_list(value)
        return rows or None

    @field_validator("repo_source_ids", mode="before")
    @classmethod
    def _normalize_repo_source_ids(cls, value: Any) -> Any:
        rows = _normalize_uuid_list(value)
        return rows or None

    @field_validator("benchmark_queries", mode="before")
    @classmethod
    def _normalize_benchmark_queries(cls, value: Any) -> Any:
        rows = _normalize_query_list(value)
        return rows or None

    @field_validator("validation_policy", mode="before")
    @classmethod
    def _normalize_validation_policy(cls, value: Any) -> Any:
        return _normalize_validation_policy(value)

    @field_validator("automation_profile", mode="before")
    @classmethod
    def _normalize_automation_profile(cls, value: Any) -> Any:
        return _normalize_automation_profile(value, default="balanced")

    @field_validator("automation_policy", mode="before")
    @classmethod
    def _normalize_automation_policy(cls, value: Any) -> Any:
        return _normalize_automation_policy(value)


class DomainResearchProfileUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=200)
    objective: Optional[str] = Field(default=None, min_length=1)
    customer_context: Optional[str] = None
    source_scope: Optional[str] = None
    track_type: Optional[str] = None
    research_mode: Optional[str] = None
    monitor_queries: Optional[List[str]] = None
    repo_source_ids: Optional[List[UUID]] = None
    benchmark_queries: Optional[List[str]] = None
    report_format: Optional[str] = None
    scoring_policy: Optional[Dict[str, Any]] = None
    selection_policy: Optional[Dict[str, Any]] = None
    validation_policy: Optional[Dict[str, Any]] = Field(
        default=None, description="Compatibility-only legacy validation policy mirror"
    )
    automation_profile: Optional[str] = None
    automation_policy: Optional[Dict[str, Any]] = None
    sandbox_profile_id: Optional[str] = None
    interval_minutes: Optional[int] = Field(default=None, ge=15, le=10080)
    persist_artifacts: Optional[bool] = None
    auto_launch_follow_up: Optional[bool] = None
    auto_create_experiment_plans: Optional[bool] = None
    confidence_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_documents: Optional[int] = Field(default=None, ge=1, le=25)
    max_papers: Optional[int] = Field(default=None, ge=0, le=25)

    @field_validator(
        "title",
        "objective",
        "customer_context",
        "source_scope",
        "track_type",
        "research_mode",
        "report_format",
        "sandbox_profile_id",
        mode="before",
    )
    @classmethod
    def _normalize_update_text(cls, value: Any) -> Any:
        if value is None:
            return value
        return str(value).strip()

    @field_validator("monitor_queries", mode="before")
    @classmethod
    def _normalize_update_monitor_queries(cls, value: Any) -> Any:
        rows = _normalize_query_list(value)
        return rows or None

    @field_validator("source_scope", mode="before")
    @classmethod
    def _normalize_update_source_scope(cls, value: Any) -> Any:
        if value is None:
            return None
        return _normalize_source_scope(value)

    @field_validator("track_type", mode="before")
    @classmethod
    def _normalize_update_track_type(cls, value: Any) -> Any:
        if value is None:
            return None
        return _normalize_track_type(value)

    @field_validator("repo_source_ids", mode="before")
    @classmethod
    def _normalize_update_repo_source_ids(cls, value: Any) -> Any:
        if value is None:
            return None
        rows = _normalize_uuid_list(value)
        return rows or None

    @field_validator("benchmark_queries", mode="before")
    @classmethod
    def _normalize_update_benchmark_queries(cls, value: Any) -> Any:
        if value is None:
            return None
        rows = _normalize_query_list(value)
        return rows or None

    @field_validator("validation_policy", mode="before")
    @classmethod
    def _normalize_update_validation_policy(cls, value: Any) -> Any:
        if value is None:
            return None
        return _normalize_validation_policy(value)

    @field_validator("automation_profile", mode="before")
    @classmethod
    def _normalize_update_automation_profile(cls, value: Any) -> Any:
        if value is None:
            return None
        return _normalize_automation_profile(value, default="balanced")

    @field_validator("automation_policy", mode="before")
    @classmethod
    def _normalize_update_automation_policy(cls, value: Any) -> Any:
        return _normalize_automation_policy(value)


class DomainResearchProfileActionRequest(BaseModel):
    action: str = Field(..., description="start, pause, resume, cancel, run_now")

    @field_validator("action", mode="before")
    @classmethod
    def _normalize_action(cls, value: Any) -> str:
        return str(value or "").strip().lower()


class DomainResearchProfileResponse(BaseModel):
    id: UUID
    user_id: UUID
    title: str
    domain: str
    objective: str
    customer_context: Optional[str] = None
    status: str
    source_scope: str
    track_type: str
    research_mode: str
    monitor_queries: Optional[List[str]] = None
    repo_source_ids: Optional[List[str]] = None
    benchmark_queries: Optional[List[str]] = None
    report_format: str
    scoring_policy: Optional[Dict[str, Any]] = None
    selection_policy: Optional[Dict[str, Any]] = None
    validation_policy: Optional[Dict[str, Any]] = Field(
        default=None, description="Compatibility-only legacy validation policy mirror"
    )
    automation_profile: str = "balanced"
    automation_policy: Optional[Dict[str, Any]] = None
    effective_policy: Optional[Dict[str, Any]] = None
    sandbox_profile_id: Optional[str] = None
    interval_minutes: int
    persist_artifacts: bool
    auto_launch_follow_up: bool
    auto_create_experiment_plans: bool
    confidence_threshold: float
    max_documents: int
    max_papers: int
    opportunities: Optional[List[Dict[str, Any]]] = None
    latest_summary: Optional[Dict[str, Any]] = None
    latest_note_ids: Optional[List[str]] = None
    latest_experiment_plan_ids: Optional[List[str]] = None
    latest_validation_run_ids: Optional[List[str]] = None
    latest_validation_runs: Optional[
        List[ScientificValidationRunSummaryResponse]
    ] = None
    latest_run_job_id: Optional[UUID] = None
    active_job_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    last_run_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class DomainResearchProfileListResponse(BaseModel):
    items: List[DomainResearchProfileResponse] = Field(default_factory=list)
    total: int = 0
