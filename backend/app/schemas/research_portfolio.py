from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.experiment import ScientificValidationRunSummaryResponse
from app.services.scientific_validation_service import (
    normalize_portfolio_automation_policy,
    normalize_portfolio_automation_profile,
    resolve_portfolio_automation_policy,
)


def _normalize_uuid_list(value: Any, *, max_items: int = 24) -> list[str]:
    rows = (
        value
        if isinstance(value, list)
        else str(value or "").replace("\n", ",").split(",")
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


def _normalize_policy(value: Any) -> dict[str, Any]:
    return normalize_portfolio_automation_policy(value)


def _normalize_profile(value: Any, *, default: str = "balanced") -> str:
    return normalize_portfolio_automation_profile(value, default=default)


class ResearchPortfolioCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    objective: str = Field(..., min_length=1)
    linked_profile_ids: List[UUID] = Field(default_factory=list, max_length=24)
    automation_profile: str = Field(default="balanced")
    automation_policy: Optional[Dict[str, Any]] = None
    sandbox_profile_id: Optional[str] = None
    start_immediately: bool = True

    @field_validator("title", "objective", "sandbox_profile_id", mode="before")
    @classmethod
    def _normalize_text(cls, value: Any) -> Any:
        if value is None:
            return value
        return str(value).strip()

    @field_validator("linked_profile_ids", mode="before")
    @classmethod
    def _normalize_profiles(cls, value: Any) -> Any:
        rows = _normalize_uuid_list(value)
        return rows

    @field_validator("automation_profile", mode="before")
    @classmethod
    def _normalize_profile_before(cls, value: Any) -> Any:
        return _normalize_profile(value, default="balanced")

    @field_validator("automation_policy", mode="before")
    @classmethod
    def _normalize_policy_before(cls, value: Any) -> Any:
        if value is None:
            return None
        return dict(value) if isinstance(value, dict) else value

    @model_validator(mode="after")
    def _resolve_policy(self) -> "ResearchPortfolioCreate":
        self.automation_profile = _normalize_profile(
            self.automation_profile, default="balanced"
        )
        self.automation_policy = resolve_portfolio_automation_policy(
            self.automation_profile, self.automation_policy
        )
        return self


class ResearchPortfolioUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=200)
    objective: Optional[str] = Field(default=None, min_length=1)
    linked_profile_ids: Optional[List[UUID]] = Field(default=None, max_length=24)
    automation_profile: Optional[str] = None
    automation_policy: Optional[Dict[str, Any]] = None
    sandbox_profile_id: Optional[str] = None

    @field_validator("title", "objective", "sandbox_profile_id", mode="before")
    @classmethod
    def _normalize_update_text(cls, value: Any) -> Any:
        if value is None:
            return value
        return str(value).strip()

    @field_validator("linked_profile_ids", mode="before")
    @classmethod
    def _normalize_update_profiles(cls, value: Any) -> Any:
        if value is None:
            return None
        return _normalize_uuid_list(value)

    @field_validator("automation_profile", mode="before")
    @classmethod
    def _normalize_profile_update(cls, value: Any) -> Any:
        if value is None:
            return None
        return _normalize_profile(value, default="balanced")

    @field_validator("automation_policy", mode="before")
    @classmethod
    def _normalize_policy_update(cls, value: Any) -> Any:
        if value is None:
            return None
        return _normalize_policy(value)


class ResearchPortfolioActionRequest(BaseModel):
    action: str = Field(..., description="start, pause, resume, cancel, run_now")

    @field_validator("action", mode="before")
    @classmethod
    def _normalize_action(cls, value: Any) -> str:
        return str(value or "").strip().lower()


class ResearchPortfolioOpportunityActionRequest(BaseModel):
    action: str = Field(
        ...,
        description="accept, suppress, reopen, create_plan, launch_validation, materialize_experiment, launch_follow_up, relaunch_follow_up",
    )
    operator_note: Optional[str] = Field(default=None, max_length=2000)
    start_immediately: Optional[bool] = None

    @field_validator("action", "operator_note", mode="before")
    @classmethod
    def _normalize_text(cls, value: Any) -> Any:
        if value is None:
            return value
        return str(value).strip()


class ResearchPortfolioResponse(BaseModel):
    id: UUID
    user_id: UUID
    title: str
    objective: str
    status: str
    linked_profile_ids: Optional[List[str]] = None
    automation_profile: str = "balanced"
    automation_policy: Optional[Dict[str, Any]] = None
    effective_policy: Optional[Dict[str, Any]] = None
    sandbox_profile_id: Optional[str] = None
    opportunities: Optional[List[Dict[str, Any]]] = None
    latest_summary: Optional[Dict[str, Any]] = None
    latest_note_ids: Optional[List[str]] = None
    latest_experiment_plan_ids: Optional[List[str]] = None
    latest_validation_run_ids: Optional[List[str]] = None
    latest_validation_runs: Optional[
        List[ScientificValidationRunSummaryResponse]
    ] = None
    child_job_ids: Optional[List[str]] = None
    active_job_id: Optional[UUID] = None
    latest_run_job_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    last_run_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class ResearchPortfolioListResponse(BaseModel):
    items: List[ResearchPortfolioResponse] = Field(default_factory=list)
    total: int = 0
