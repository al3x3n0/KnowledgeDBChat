"""
Schemas for LangGraph issue -> PR draft orchestration.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

DecisionType = Literal["pass", "revise", "escalate"]
StatusType = Literal["pr_ready", "needs_human_review", "blocked"]
ResultType = Literal["pass", "fail"]
StatusReasonCodeType = Literal[
    "passed",
    "policy_escalation",
    "manual_escalation",
    "human_review_required",
    "revision_exhausted",
    "needs_revision",
    "escalated_blocked",
    "blocked_unknown",
    "unknown",
]


class IssuePrIssueInput(BaseModel):
    id: str = Field(..., min_length=1, max_length=200)
    title: str = Field(..., min_length=1, max_length=500)
    body: str = Field("", max_length=20000)
    labels: List[str] = Field(default_factory=list)

    @field_validator("labels", mode="before")
    @classmethod
    def _normalize_labels(cls, value: Any) -> Any:
        if value is None:
            return []
        if not isinstance(value, list):
            return value
        labels: List[str] = []
        for raw in value:
            token = str(raw or "").strip().lower()
            if token and token not in labels:
                labels.append(token[:120])
            if len(labels) >= 50:
                break
        return labels


class PlanStep(BaseModel):
    id: str = Field(..., min_length=1, max_length=20)
    action: str = Field(..., min_length=1, max_length=500)
    rationale: str = Field(..., min_length=1, max_length=800)


class PlannerOutput(BaseModel):
    plan_steps: List[PlanStep] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)
    test_plan: List[str] = Field(default_factory=list)
    risks: List[Dict[str, str]] = Field(default_factory=list)
    out_of_scope: List[str] = Field(default_factory=list)


class Finding(BaseModel):
    claim: str = Field(..., min_length=1, max_length=600)
    evidence: str = Field(..., min_length=1, max_length=1200)
    file_path: str = Field(..., min_length=1, max_length=600)


class RelatedTest(BaseModel):
    file_path: str = Field(..., min_length=1, max_length=600)
    why: str = Field(..., min_length=1, max_length=600)


class ResearcherOutput(BaseModel):
    findings: List[Finding] = Field(default_factory=list)
    related_tests: List[RelatedTest] = Field(default_factory=list)
    unknowns: List[str] = Field(default_factory=list)
    risk_flags: List[Dict[str, str]] = Field(default_factory=list)


class ChangeItem(BaseModel):
    file: str = Field(..., min_length=1, max_length=600)
    summary: str = Field(..., min_length=1, max_length=800)


class CommandRun(BaseModel):
    cmd: str = Field(..., min_length=1, max_length=800)
    result: ResultType
    output_ref: str = Field(..., min_length=1, max_length=300)


class ExecutorOutput(BaseModel):
    changes: List[ChangeItem] = Field(default_factory=list)
    tests_added: List[ChangeItem] = Field(default_factory=list)
    commands_run: List[CommandRun] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class ReviewFailure(BaseModel):
    criterion: str = Field(..., min_length=1, max_length=500)
    reason: str = Field(..., min_length=1, max_length=1000)
    evidence: str = Field(..., min_length=1, max_length=1000)


class RequiredFix(BaseModel):
    file: str = Field(..., min_length=1, max_length=600)
    change_request: str = Field(..., min_length=1, max_length=1000)


class PolicyCheck(BaseModel):
    check: str = Field(..., min_length=1, max_length=500)
    status: ResultType
    evidence: str = Field(..., min_length=1, max_length=1000)


class ReviewerOutput(BaseModel):
    decision: DecisionType
    failures: List[ReviewFailure] = Field(default_factory=list)
    required_fixes: List[RequiredFix] = Field(default_factory=list)
    policy_checks: List[PolicyCheck] = Field(default_factory=list)
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class ChecklistItem(BaseModel):
    item: str = Field(..., min_length=1, max_length=500)
    status: ResultType


class PrDraftBodySections(BaseModel):
    Summary: str = ""
    Root_Cause: str = Field("", alias="Root Cause")
    Changes: str = ""
    Test_Plan: str = Field("", alias="Test Plan")
    Risks: str = ""
    Rollback: str = ""

    model_config = {"populate_by_name": True}


class PrDraftPackage(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    body_sections: Dict[str, str] = Field(default_factory=dict)
    checklist: Dict[str, List[ChecklistItem]] = Field(default_factory=dict)
    artifacts: List[Dict[str, str]] = Field(default_factory=list)


class EventLogItem(BaseModel):
    ts: datetime
    agent: str = Field(..., min_length=1, max_length=80)
    action: str = Field(..., min_length=1, max_length=120)
    result: str = Field(..., min_length=1, max_length=120)
    ref: str = Field("", max_length=300)


class LangGraphIssuePrRequest(BaseModel):
    issue: IssuePrIssueInput
    repo_context: Dict[str, Any] = Field(default_factory=dict)
    constraints: List[str] = Field(default_factory=list)
    policy_profile: Dict[str, Any] = Field(default_factory=dict)
    max_revision_loops: int = Field(2, ge=0, le=2)
    reviewer_min_confidence: float = Field(0.75, ge=0.0, le=1.0)


class LangGraphIssuePrResponse(BaseModel):
    status: StatusType
    reason: str = Field("", max_length=1000)
    status_reason_code: StatusReasonCodeType = "unknown"
    planner: PlannerOutput
    researcher: ResearcherOutput
    executor: ExecutorOutput
    reviewer: ReviewerOutput
    pr_draft: Optional[PrDraftPackage] = None
    repo_context_meta: Dict[str, Any] = Field(default_factory=dict)
    repo_context_summary: Dict[str, Any] = Field(default_factory=dict)
    confidence_breakdown: Dict[str, float] = Field(default_factory=dict)
    decision_trace: List[str] = Field(default_factory=list)
    event_log: List[EventLogItem] = Field(default_factory=list)
