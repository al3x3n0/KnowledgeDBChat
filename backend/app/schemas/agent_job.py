"""
Pydantic schemas for autonomous agent jobs.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from app.schemas.domain_research_profile import DomainResearchProfileResponse
from app.schemas.research_portfolio import ResearchPortfolioResponse
from app.services.agent_scope_service import normalize_scope_config
from app.services.scientific_validation_service import (
    normalize_portfolio_automation_policy,
    normalize_portfolio_automation_profile,
    normalize_validation_policy,
)


def _normalize_text_list(
    value: Any, *, max_items: int = 12, max_len: int = 240
) -> Optional[List[str]]:
    if value is None:
        return None
    rows = (
        value if isinstance(value, list) else str(value).replace("\n", ",").split(",")
    )
    out: List[str] = []
    for row in rows:
        text = str(row or "").strip()
        if not text or text in out:
            continue
        out.append(text[:max_len])
        if len(out) >= max_items:
            break
    return out or None


def _normalize_uuid_text_list(
    value: Any, *, max_items: int = 24
) -> Optional[List[str]]:
    if value is None:
        return None
    rows = (
        value if isinstance(value, list) else str(value).replace("\n", ",").split(",")
    )
    out: List[str] = []
    for row in rows:
        text = str(row or "").strip()
        if not text or text in out:
            continue
        out.append(text)
        if len(out) >= max_items:
            break
    return out or None


def _normalize_domain_source_scope(value: Any) -> str:
    text = (
        str(value or "kb_plus_arxiv")
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )
    if text in {"kb", "documents", "kb_first"}:
        text = "kb_only"
    elif text in {"arxiv", "papers"}:
        text = "arxiv_only"
    elif text in {"kb_plus_repo", "kb_repo", "repo"}:
        text = "kb_plus_arxiv_plus_repo"
    elif text not in {
        "kb_only",
        "arxiv_only",
        "kb_plus_arxiv",
        "kb_plus_arxiv_plus_repo",
    }:
        text = "kb_plus_arxiv"
    return text


def _normalize_track_type(value: Any) -> str:
    text = str(value or "generic").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"micro_arch", "microarch", "uarch"}:
        return "microarchitecture"
    if text not in {"compiler", "microarchitecture", "generic"}:
        return "generic"
    return text


def _normalize_validation_policy(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    return normalize_validation_policy(value)


def _normalize_automation_policy(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    return normalize_portfolio_automation_policy(value)


class AgentJobCreate(BaseModel):
    """Request schema for creating an agent job."""

    name: str = Field(..., min_length=1, max_length=200, description="Job name")
    description: Optional[str] = Field(None, description="Job description")
    job_type: str = Field(
        "custom",
        description="Type of job: research, monitor, analysis, synthesis, knowledge_expansion, custom",
    )
    goal: str = Field(
        ..., min_length=1, description="The goal for the agent to achieve"
    )
    goal_criteria: Optional[Dict[str, Any]] = Field(
        None, description="Structured success criteria"
    )
    config: Optional[Dict[str, Any]] = Field(
        None, description="Job-specific configuration"
    )
    agent_definition_id: Optional[UUID] = Field(
        None, description="ID of agent definition to use"
    )

    # Resource limits (optional overrides)
    max_iterations: Optional[int] = Field(
        None, ge=1, le=1000, description="Maximum iterations"
    )
    max_tool_calls: Optional[int] = Field(
        None, ge=1, le=5000, description="Maximum tool calls"
    )
    max_llm_calls: Optional[int] = Field(
        None, ge=1, le=2000, description="Maximum LLM calls"
    )
    max_runtime_minutes: Optional[int] = Field(
        None, ge=1, le=480, description="Maximum runtime in minutes"
    )

    # Scheduling
    schedule_type: Optional[str] = Field(
        None, description="Scheduling type: once, recurring, continuous"
    )
    schedule_cron: Optional[str] = Field(
        None, description="Cron expression for recurring jobs"
    )
    start_immediately: bool = Field(
        True, description="Start job immediately after creation"
    )

    # Job chaining
    chain_config: Optional[Dict[str, Any]] = Field(
        None, description="Chain configuration for triggering child jobs"
    )
    # Structure:
    # {
    #   "trigger_condition": "on_complete",  # on_complete, on_fail, on_any_end, on_progress, on_findings
    #   "progress_threshold": 50,            # For on_progress trigger
    #   "findings_threshold": 10,            # For on_findings trigger
    #   "inherit_results": true,             # Pass parent results to child
    #   "inherit_config": false,             # Inherit parent config
    #   "child_jobs": [
    #     {
    #       "name": "Follow-up Analysis",
    #       "job_type": "analysis",
    #       "goal": "Analyze the research findings",
    #       "config": {...}
    #     }
    #   ]
    # }
    parent_job_id: Optional[UUID] = Field(
        None, description="ID of parent job (for manually chained jobs)"
    )

    @field_validator("config", mode="before")
    @classmethod
    def _normalize_config(cls, value: Any) -> Any:
        return normalize_scope_config(value)


class AgentJobFromTemplate(BaseModel):
    """Request schema for creating a job from a template."""

    template_id: UUID = Field(..., description="ID of the template to use")
    name: str = Field(..., min_length=1, max_length=200, description="Job name")
    goal: Optional[str] = Field(None, description="Override the default goal")
    config: Optional[Dict[str, Any]] = Field(
        None, description="Override template config"
    )
    start_immediately: bool = Field(True, description="Start job immediately")
    chain_config: Optional[Dict[str, Any]] = Field(
        None, description="Optional chain configuration (triggers child jobs)"
    )

    @field_validator("config", mode="before")
    @classmethod
    def _normalize_config(cls, value: Any) -> Any:
        return normalize_scope_config(value)


class AgentJobQuickStartClaudeBackendRequest(BaseModel):
    """Quick-start request for launching the Claude-style backend coding loop."""

    name: Optional[str] = Field(
        None, min_length=1, max_length=200, description="Optional job name"
    )
    goal: str = Field(..., min_length=1, description="Backend coding goal")
    source_id: UUID = Field(..., description="Target git DocumentSource UUID")
    search_query: Optional[str] = Field(
        "backend", description="Optional retrieval hint for code patch context"
    )
    file_paths: Optional[List[str]] = Field(
        None, description="Optional list of focused file paths"
    )
    commands: Optional[List[str]] = Field(
        None, description="Optional verification commands"
    )
    start_immediately: bool = Field(True, description="Start job immediately")
    config_overrides: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional additional config merged into template config",
    )

    @field_validator("config_overrides", mode="before")
    @classmethod
    def _normalize_config_overrides(cls, value: Any) -> Any:
        return normalize_scope_config(value)

    @field_validator("search_query", mode="before")
    @classmethod
    def _normalize_search_query(cls, value: Any) -> Any:
        if value is None:
            return value
        text = str(value).strip()
        return text[:500]

    @field_validator("file_paths", mode="before")
    @classmethod
    def _normalize_file_paths(cls, value: Any) -> Any:
        if value is None:
            return value
        if not isinstance(value, list):
            return value
        paths = [str(p).strip() for p in value if str(p).strip()]
        out: List[str] = []
        for p in paths:
            path = p.replace("\\", "/").strip()
            while path.startswith("./"):
                path = path[2:]
            if not path or path.startswith("/") or ":" in path:
                continue
            parts = [seg for seg in path.split("/") if seg not in {"", "."}]
            if any(seg == ".." for seg in parts):
                continue
            normalized = "/".join(parts)[:500]
            if normalized and normalized not in out:
                out.append(normalized)
            if len(out) >= 32:
                break
        return out

    @field_validator("commands", mode="before")
    @classmethod
    def _normalize_commands(cls, value: Any) -> Any:
        if value is None:
            return value
        if not isinstance(value, list):
            return value
        commands = [str(c).strip() for c in value if str(c).strip()]
        out: List[str] = []
        for cmd in commands:
            if cmd not in out:
                out.append(cmd[:500])
            if len(out) >= 6:
                break
        return out


class AgentJobQuickStartRoleWorkflowRequest(BaseModel):
    """Quick-start request for launching role-based swarm workflows."""

    name: Optional[str] = Field(
        None, min_length=1, max_length=200, description="Optional job name"
    )
    goal: str = Field(..., min_length=1, description="Goal for the role workflow")
    roles: Optional[List[str]] = Field(
        None,
        description="Optional role list (e.g. researcher_documents, researcher_arxiv, analyst, synthesizer)",
    )
    max_agents: Optional[int] = Field(
        None, ge=1, le=12, description="Maximum swarm child roles"
    )
    memory_profile: Optional[str] = Field(
        "balanced",
        description="Memory profile: off, minimal, balanced, evidence, synthesis",
    )
    approval_mode: Optional[str] = Field(
        "high_impact",
        description="Approval mode: high_impact or none",
    )
    execution_mode: Optional[str] = Field(
        "plan_and_execute",
        description="Execution mode: plan_and_execute or adaptive",
    )
    extract_memory_on_failure: Optional[bool] = Field(
        True,
        description="When true, auto-extract memories on failed runs in addition to completed runs",
    )
    memory_failed_types: Optional[List[str]] = Field(
        None,
        description="Optional memory types to extract on failed runs",
    )
    memory_completed_types: Optional[List[str]] = Field(
        None,
        description="Optional memory types to extract on completed runs",
    )
    start_immediately: bool = Field(True, description="Start job immediately")
    config_overrides: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional additional config merged into quick-start config",
    )

    @field_validator("config_overrides", mode="before")
    @classmethod
    def _normalize_config_overrides(cls, value: Any) -> Any:
        return normalize_scope_config(value)

    @field_validator("roles", mode="before")
    @classmethod
    def _normalize_roles(cls, value: Any) -> Any:
        if value is None:
            return value
        if not isinstance(value, list):
            return value
        out: List[str] = []
        for raw in value:
            role = str(raw or "").strip().lower()
            if not role:
                continue
            role = role.replace("-", "_").replace(" ", "_")
            if not role or len(role) > 120:
                continue
            if role not in out:
                out.append(role)
            if len(out) >= 12:
                break
        return out

    @field_validator("memory_profile", mode="before")
    @classmethod
    def _normalize_memory_profile(cls, value: Any) -> Any:
        if value is None:
            return "balanced"
        text = str(value).strip().lower()
        if text in {"off", "minimal", "balanced", "evidence", "synthesis"}:
            return text
        return "balanced"

    @field_validator("approval_mode", mode="before")
    @classmethod
    def _normalize_approval_mode(cls, value: Any) -> Any:
        if value is None:
            return "high_impact"
        text = str(value).strip().lower()
        if text in {"high_impact", "none"}:
            return text
        return "high_impact"

    @field_validator("execution_mode", mode="before")
    @classmethod
    def _normalize_execution_mode(cls, value: Any) -> Any:
        if value is None:
            return "plan_and_execute"
        text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        if text in {"plan_then_act", "plan_execute", "planner_executor"}:
            text = "plan_and_execute"
        if text in {"plan_and_execute", "adaptive"}:
            return text
        return "plan_and_execute"

    @field_validator("memory_failed_types", "memory_completed_types", mode="before")
    @classmethod
    def _normalize_memory_type_lists(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, str):
            rows = [r.strip() for r in value.replace("\n", ",").split(",") if r.strip()]
        elif isinstance(value, list):
            rows = [str(r).strip() for r in value if str(r).strip()]
        else:
            return None

        allowed = {
            "finding",
            "insight",
            "pattern",
            "lesson",
            "fact",
            "preference",
            "context",
            "summary",
        }
        out: List[str] = []
        for raw in rows:
            token = str(raw or "").strip().lower().replace("-", "_").replace(" ", "_")
            if token in allowed and token not in out:
                out.append(token)
            if len(out) >= 12:
                break
        return out or None


class AgentJobQuickStartDomainResearchRequest(BaseModel):
    """Quick-start request for launching a domain research orchestrator."""

    name: Optional[str] = Field(
        None, min_length=1, max_length=200, description="Optional job name"
    )
    domain: str = Field(
        ..., min_length=1, max_length=300, description="Domain or topic to research"
    )
    objective: str = Field(
        ..., min_length=1, description="Research objective or thesis prompt"
    )
    customer_context: Optional[str] = Field(
        None, description="Optional customer or operating context"
    )
    source_scope: Optional[str] = Field(
        "kb_plus_arxiv_plus_repo",
        description="Evidence scope: kb_only, arxiv_only, kb_plus_arxiv, kb_plus_arxiv_plus_repo",
    )
    track_type: Optional[str] = Field(
        "compiler",
        description="Track specialization: compiler, microarchitecture, generic",
    )
    research_mode: Optional[str] = Field(
        "literature_to_hypothesis",
        description="Research loop shape; v1 supports literature_to_hypothesis",
    )
    monitor_queries: Optional[List[str]] = Field(
        None, description="Optional domain monitoring queries"
    )
    repo_source_ids: Optional[List[UUID]] = Field(
        None, description="Optional repository source UUIDs for code/benchmark evidence"
    )
    benchmark_queries: Optional[List[str]] = Field(
        None, description="Optional benchmark or perf-counter retrieval queries"
    )
    sandbox_profile_id: Optional[str] = Field(
        None,
        description="Optional approved sandbox profile identifier for recipe-backed validation execution",
    )
    report_format: Optional[str] = Field(
        "brief_and_report",
        description="Output format: brief_only, report_only, brief_and_report",
    )
    scoring_policy: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional scoring policy override for novelty/evidence/testability",
    )
    selection_policy: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional selection policy override for surfaced hypothesis count",
    )
    persist_artifacts: bool = Field(
        True, description="Persist generated artifacts as Research Notes"
    )
    auto_launch_follow_up: bool = Field(
        True, description="Auto-launch a deep-dive follow-up when confidence passes"
    )
    max_documents: Optional[int] = Field(
        10, ge=1, le=25, description="Maximum KB documents to use"
    )
    max_papers: Optional[int] = Field(
        8, ge=0, le=25, description="Maximum arXiv papers to use"
    )
    profile_id: Optional[UUID] = Field(
        None, description="Optional saved domain research profile UUID"
    )
    auto_create_experiment_plans: bool = Field(
        True,
        description="Auto-create experiment plans for strong ideas when policy passes",
    )
    automation_profile: Optional[str] = Field(
        "balanced",
        description="Canonical automation mode for domain research quick starts",
    )
    automation_policy: Optional[Dict[str, Any]] = Field(
        None,
        description="Canonical automation policy controlling follow-up and validation behavior",
    )
    validation_policy: Optional[Dict[str, Any]] = Field(
        None,
        description="Compatibility-only legacy validation policy mirror",
    )
    confidence_threshold: Optional[float] = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence required to auto-launch follow-up research",
    )
    start_immediately: bool = Field(True, description="Start job immediately")
    config_overrides: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional additional config merged into quick-start config",
    )

    @field_validator("config_overrides", mode="before")
    @classmethod
    def _normalize_config_overrides(cls, value: Any) -> Any:
        return normalize_scope_config(value)

    @field_validator(
        "domain",
        "objective",
        "customer_context",
        "research_mode",
        "report_format",
        "sandbox_profile_id",
        mode="before",
    )
    @classmethod
    def _normalize_text_fields(cls, value: Any) -> Any:
        if value is None:
            return value
        return str(value).strip()

    @field_validator("source_scope", mode="before")
    @classmethod
    def _normalize_source_scope(cls, value: Any) -> Any:
        return _normalize_domain_source_scope(value)

    @field_validator("track_type", mode="before")
    @classmethod
    def _normalize_track_type(cls, value: Any) -> Any:
        return _normalize_track_type(value)

    @field_validator("research_mode", mode="before")
    @classmethod
    def _normalize_research_mode(cls, value: Any) -> Any:
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
    def _normalize_report_format(cls, value: Any) -> Any:
        text = (
            str(value or "brief_and_report")
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
        if text not in {"brief_only", "report_only", "brief_and_report"}:
            text = "brief_and_report"
        return text

    @field_validator("monitor_queries", mode="before")
    @classmethod
    def _normalize_monitor_queries(cls, value: Any) -> Any:
        return _normalize_text_list(value, max_items=12, max_len=240)

    @field_validator("repo_source_ids", mode="before")
    @classmethod
    def _normalize_repo_source_ids(cls, value: Any) -> Any:
        return _normalize_uuid_text_list(value, max_items=24)

    @field_validator("benchmark_queries", mode="before")
    @classmethod
    def _normalize_benchmark_queries(cls, value: Any) -> Any:
        return _normalize_text_list(value, max_items=16, max_len=240)

    @field_validator("validation_policy", mode="before")
    @classmethod
    def _normalize_validation_policy(cls, value: Any) -> Any:
        return _normalize_validation_policy(value)

    @field_validator("automation_profile", mode="before")
    @classmethod
    def _normalize_automation_profile(cls, value: Any) -> Any:
        return normalize_portfolio_automation_profile(value, default="balanced")

    @field_validator("automation_policy", mode="before")
    @classmethod
    def _normalize_automation_policy(cls, value: Any) -> Any:
        return _normalize_automation_policy(value)


class AgentJobPromoteDomainResearchProfileRequest(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=200)
    domain: Optional[str] = Field(default=None, min_length=1, max_length=300)
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
        "domain",
        "objective",
        "customer_context",
        "research_mode",
        "report_format",
        "sandbox_profile_id",
        mode="before",
    )
    @classmethod
    def _normalize_text_fields(cls, value: Any) -> Any:
        if value is None:
            return value
        return str(value).strip()

    @field_validator("source_scope", mode="before")
    @classmethod
    def _normalize_source_scope_field(cls, value: Any) -> Any:
        if value is None:
            return None
        return _normalize_domain_source_scope(value)

    @field_validator("track_type", mode="before")
    @classmethod
    def _normalize_track_type_field(cls, value: Any) -> Any:
        if value is None:
            return None
        return _normalize_track_type(value)

    @field_validator("research_mode", mode="before")
    @classmethod
    def _normalize_research_mode_field(cls, value: Any) -> Any:
        if value is None:
            return None
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
    def _normalize_report_format_field(cls, value: Any) -> Any:
        if value is None:
            return None
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
    def _normalize_monitor_queries_field(cls, value: Any) -> Any:
        if value is None:
            return None
        return _normalize_text_list(value, max_items=12, max_len=240)

    @field_validator("repo_source_ids", mode="before")
    @classmethod
    def _normalize_repo_source_ids_field(cls, value: Any) -> Any:
        if value is None:
            return None
        return _normalize_uuid_text_list(value, max_items=24)

    @field_validator("benchmark_queries", mode="before")
    @classmethod
    def _normalize_benchmark_queries_field(cls, value: Any) -> Any:
        if value is None:
            return None
        return _normalize_text_list(value, max_items=16, max_len=240)

    @field_validator("automation_profile", mode="before")
    @classmethod
    def _normalize_automation_profile_field(cls, value: Any) -> Any:
        if value is None:
            return None
        return normalize_portfolio_automation_profile(value, default="balanced")

    @field_validator("automation_policy", mode="before")
    @classmethod
    def _normalize_automation_policy_field(cls, value: Any) -> Any:
        if value is None:
            return None
        return _normalize_automation_policy(value)


class AgentJobPromoteDomainResearchPortfolioRequest(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=200)
    objective: Optional[str] = Field(default=None, min_length=1)
    sandbox_profile_id: Optional[str] = None
    automation_profile: Optional[str] = None
    automation_policy: Optional[Dict[str, Any]] = None

    @field_validator("title", "objective", "sandbox_profile_id", mode="before")
    @classmethod
    def _normalize_text_fields(cls, value: Any) -> Any:
        if value is None:
            return value
        return str(value).strip()

    @field_validator("automation_profile", mode="before")
    @classmethod
    def _normalize_automation_profile_field(cls, value: Any) -> Any:
        if value is None:
            return None
        return normalize_portfolio_automation_profile(value, default="balanced")

    @field_validator("automation_policy", mode="before")
    @classmethod
    def _normalize_automation_policy_field(cls, value: Any) -> Any:
        if value is None:
            return None
        return _normalize_automation_policy(value)


class AgentJobPromoteDomainResearchRequest(BaseModel):
    target_mode: str = Field(
        default="profile_only", description="profile_only or profile_with_portfolio"
    )
    profile: AgentJobPromoteDomainResearchProfileRequest = Field(
        default_factory=AgentJobPromoteDomainResearchProfileRequest
    )
    portfolio_id: Optional[UUID] = None
    portfolio: Optional[AgentJobPromoteDomainResearchPortfolioRequest] = None
    start_profile_now: bool = True
    run_portfolio_now: bool = False

    @field_validator("target_mode", mode="before")
    @classmethod
    def _normalize_target_mode(cls, value: Any) -> str:
        text = (
            str(value or "profile_only")
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )
        if text not in {"profile_only", "profile_with_portfolio"}:
            return "profile_only"
        return text


class AgentJobPromoteDomainResearchResponse(BaseModel):
    source_job_id: UUID
    promotion_status: str
    domain_research_profile_id: UUID
    research_portfolio_id: Optional[UUID] = None
    profile: DomainResearchProfileResponse
    portfolio: Optional[ResearchPortfolioResponse] = None
    source_job: Optional["AgentJobResponse"] = None


class AgentJobQuickStartRepoBugTriageRequest(BaseModel):
    """Quick-start request for launching the repo-wide bug triage + repair loop."""

    name: Optional[str] = Field(
        None, min_length=1, max_length=200, description="Optional job name"
    )
    goal: Optional[str] = Field(
        None, min_length=1, description="Optional desired fix outcome"
    )
    failure_symptom: Optional[str] = Field(
        None, min_length=1, description="Observed bug symptom or failure description"
    )
    source_id: UUID = Field(..., description="Target git DocumentSource UUID")
    scope: Optional[str] = Field(
        "auto", description="Repo scope profile: auto, backend, frontend, worker"
    )
    search_query: Optional[str] = Field(
        None, description="Optional retrieval hint for code patch context"
    )
    file_paths: Optional[List[str]] = Field(
        None, description="Optional list of focused file paths"
    )
    commands: Optional[List[str]] = Field(
        None, description="Optional verification commands"
    )
    error_output: Optional[str] = Field(
        None, description="Optional logs or stack trace text"
    )
    start_immediately: bool = Field(True, description="Start job immediately")
    config_overrides: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional additional config merged into quick-start config",
    )

    @model_validator(mode="after")
    def _require_goal_or_symptom(self) -> "AgentJobQuickStartRepoBugTriageRequest":
        if (
            not str(self.goal or "").strip()
            and not str(self.failure_symptom or "").strip()
        ):
            raise ValueError("Either goal or failure_symptom is required")
        return self

    @field_validator("config_overrides", mode="before")
    @classmethod
    def _normalize_config_overrides(cls, value: Any) -> Any:
        return normalize_scope_config(value)

    @field_validator("scope", mode="before")
    @classmethod
    def _normalize_scope(cls, value: Any) -> Any:
        text = (
            str(value or "auto")
            .strip()
            .lower()
            .replace("-", "")
            .replace("_", "")
            .replace(" ", "")
        )
        if text not in {"auto", "backend", "frontend", "worker"}:
            text = "auto"
        return text

    @field_validator("search_query", mode="before")
    @classmethod
    def _normalize_search_query(cls, value: Any) -> Any:
        if value is None:
            return value
        return str(value).strip()[:500]

    @field_validator("error_output", mode="before")
    @classmethod
    def _normalize_error_output(cls, value: Any) -> Any:
        if value is None:
            return value
        return str(value).strip()[:12000]

    @field_validator("file_paths", mode="before")
    @classmethod
    def _normalize_file_paths(cls, value: Any) -> Any:
        return AgentJobQuickStartClaudeBackendRequest._normalize_file_paths(value)

    @field_validator("commands", mode="before")
    @classmethod
    def _normalize_commands(cls, value: Any) -> Any:
        return AgentJobQuickStartClaudeBackendRequest._normalize_commands(value)


class _AgentJobQuickStartCodingSwarmRequestBase(BaseModel):
    """Shared quick-start request shape for coding swarm presets."""

    name: Optional[str] = Field(
        None, min_length=1, max_length=200, description="Optional job name"
    )
    goal: Optional[str] = Field(
        None, min_length=1, description="Optional desired fix outcome"
    )
    failure_symptom: Optional[str] = Field(
        None, min_length=1, description="Observed bug symptom or failure description"
    )
    source_id: UUID = Field(..., description="Target git DocumentSource UUID")
    scope: Optional[str] = Field(
        "auto", description="Repo scope profile: auto, backend, frontend, worker"
    )
    search_query: Optional[str] = Field(
        None, description="Optional retrieval hint for swarm triage context"
    )
    file_paths: Optional[List[str]] = Field(
        None, description="Optional list of focused file paths"
    )
    commands: Optional[List[str]] = Field(
        None, description="Optional verification commands"
    )
    error_output: Optional[str] = Field(
        None, description="Optional logs or stack trace text"
    )
    max_agents: Optional[int] = Field(
        4, ge=1, le=4, description="Maximum coding swarm agents"
    )
    profile_id: Optional[UUID] = Field(
        None, description="Optional saved coding swarm profile id"
    )
    start_immediately: bool = Field(True, description="Start job immediately")
    config_overrides: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional additional config merged into quick-start config",
    )

    @model_validator(mode="after")
    def _require_goal_or_symptom(self) -> "_AgentJobQuickStartCodingSwarmRequestBase":
        if (
            not str(self.goal or "").strip()
            and not str(self.failure_symptom or "").strip()
        ):
            raise ValueError("Either goal or failure_symptom is required")
        return self

    @field_validator("config_overrides", mode="before")
    @classmethod
    def _normalize_config_overrides(cls, value: Any) -> Any:
        return normalize_scope_config(value)

    @field_validator("scope", mode="before")
    @classmethod
    def _normalize_scope(cls, value: Any) -> Any:
        text = (
            str(value or "auto")
            .strip()
            .lower()
            .replace("-", "")
            .replace("_", "")
            .replace(" ", "")
        )
        if text not in {"auto", "backend", "frontend", "worker"}:
            text = "auto"
        return text

    @field_validator("search_query", mode="before")
    @classmethod
    def _normalize_search_query(cls, value: Any) -> Any:
        if value is None:
            return value
        return str(value).strip()[:500]

    @field_validator("error_output", mode="before")
    @classmethod
    def _normalize_error_output(cls, value: Any) -> Any:
        if value is None:
            return value
        return str(value).strip()[:12000]

    @field_validator("file_paths", mode="before")
    @classmethod
    def _normalize_file_paths(cls, value: Any) -> Any:
        return AgentJobQuickStartClaudeBackendRequest._normalize_file_paths(value)

    @field_validator("commands", mode="before")
    @classmethod
    def _normalize_commands(cls, value: Any) -> Any:
        return AgentJobQuickStartClaudeBackendRequest._normalize_commands(value)


class AgentJobQuickStartBugTriageSwarmRequest(
    _AgentJobQuickStartCodingSwarmRequestBase
):
    """Quick-start request for launching a coding-focused bug triage swarm."""


class AgentJobQuickStartBuildBreakSwarmRequest(
    _AgentJobQuickStartCodingSwarmRequestBase
):
    """Quick-start request for launching a build-break coding swarm."""


class AgentJobQuickStartFrontendRegressionSwarmRequest(
    _AgentJobQuickStartCodingSwarmRequestBase
):
    """Quick-start request for launching a frontend-regression coding swarm."""


class AgentJobUpdate(BaseModel):
    """Request schema for updating an agent job."""

    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    goal: Optional[str] = None
    goal_criteria: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None
    max_iterations: Optional[int] = Field(None, ge=1, le=1000)
    max_tool_calls: Optional[int] = Field(None, ge=1, le=5000)
    max_llm_calls: Optional[int] = Field(None, ge=1, le=2000)
    max_runtime_minutes: Optional[int] = Field(None, ge=1, le=480)
    schedule_type: Optional[str] = None
    schedule_cron: Optional[str] = None

    @field_validator("config", mode="before")
    @classmethod
    def _normalize_config(cls, value: Any) -> Any:
        return normalize_scope_config(value)


class AgentJobLogEntry(BaseModel):
    """Schema for a job log entry."""

    iteration: int
    phase: str
    timestamp: str
    thought: Optional[str] = None
    action: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None


class AgentJobExperimentRunResponse(BaseModel):
    """Typed projection for deterministic experiment runner results."""

    source_id: Optional[str] = None
    source_name: Optional[str] = None
    enabled: Optional[bool] = None
    backend: Optional[str] = None
    commands: List[str] = Field(default_factory=list)
    verification_commands: List[str] = Field(default_factory=list)
    bootstrap_commands: List[str] = Field(default_factory=list)
    fallback_commands: List[str] = Field(default_factory=list)
    runs: List[Dict[str, Any]] = Field(default_factory=list)
    ok: Optional[bool] = None
    final_phase: Optional[str] = None
    phases: List[str] = Field(default_factory=list)
    verification_phases: List[str] = Field(default_factory=list)
    failed_commands: List[str] = Field(default_factory=list)
    proposal_id: Optional[str] = None
    latex_project_id: Optional[str] = None
    latex_updated: Optional[bool] = None
    inferred_project_profile: Optional[Dict[str, Any]] = None
    bootstrap_attempted: Optional[bool] = None
    bootstrap_ok: Optional[bool] = None
    bootstrap_used: Optional[bool] = None
    fallback_attempted: Optional[bool] = None
    fallback_ok: Optional[bool] = None
    fallback_used: Optional[bool] = None
    measurement_summary: Optional[Dict[str, Any]] = None
    compiler_artifacts: Optional[Dict[str, Any]] = None
    perf_counters: Optional[Dict[str, Any]] = None
    artifact_inventory: List[str] = Field(default_factory=list)
    repeat_count: Optional[int] = None
    note: Optional[str] = None
    summary: Optional[str] = None


class AgentJobOperatorInterventionResponse(BaseModel):
    """Typed projection for operator intervention history."""

    action: str
    actor_user_id: Optional[str] = None
    at: Optional[str] = None
    note: Optional[str] = None
    job_status_before: Optional[str] = None
    job_status_after: Optional[str] = None
    outcome_status: Optional[str] = None
    outcome_reason: Optional[str] = None
    resolved_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AgentJobResponse(BaseModel):
    """Response schema for an agent job."""

    id: UUID
    name: str
    description: Optional[str]
    job_type: str
    goal: str
    goal_criteria: Optional[Dict[str, Any]]
    config: Optional[Dict[str, Any]]
    launch_mode: Optional[str] = None
    relaunch_from_job_id: Optional[UUID] = None
    relaunch_children_count: int = 0
    promotion_status: Optional[str] = None
    promoted_domain_research_profile_id: Optional[UUID] = None
    promoted_research_portfolio_id: Optional[UUID] = None

    # Agent assignment
    agent_definition_id: Optional[UUID]
    agent_definition_name: Optional[str] = None

    # Ownership
    user_id: UUID

    # Status and progress
    status: str
    progress: int
    current_phase: Optional[str]
    phase_details: Optional[str]

    # Execution tracking
    iteration: int
    max_iterations: int

    # Resource limits
    max_tool_calls: int
    max_llm_calls: int
    max_runtime_minutes: int

    # Usage tracking
    tool_calls_used: int
    llm_calls_used: int
    tokens_used: int

    # Error tracking
    error: Optional[str]
    error_count: int

    # Scheduling
    schedule_type: Optional[str]
    schedule_cron: Optional[str]
    next_run_at: Optional[datetime]
    scheduler_state: Optional[Dict[str, Any]] = None

    # Results
    results: Optional[Dict[str, Any]]
    experiment_run: Optional[AgentJobExperimentRunResponse] = None
    experiment_runs: Optional[List[AgentJobExperimentRunResponse]] = None
    operator_interventions: Optional[List[AgentJobOperatorInterventionResponse]] = None
    output_artifacts: Optional[List[Dict[str, Any]]]

    # Timestamps
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    last_activity_at: Optional[datetime]

    # Celery task tracking
    celery_task_id: Optional[str]
    execution_lease_owner: Optional[str] = None
    execution_lease_expires_at: Optional[datetime] = None
    execution_lease_heartbeat_at: Optional[datetime] = None
    execution_fence: int = 0

    # Job chaining
    parent_job_id: Optional[UUID] = None
    root_job_id: Optional[UUID] = None
    chain_depth: int = 0
    chain_triggered: bool = False
    chain_config: Optional[Dict[str, Any]] = None
    swarm_summary: Optional[Dict[str, Any]] = None
    goal_contract_summary: Optional[Dict[str, Any]] = None
    approval_checkpoint: Optional[Dict[str, Any]] = None
    executive_digest: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class AgentJobListResponse(BaseModel):
    """Response schema for listing agent jobs."""

    jobs: List[AgentJobResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class AgentJobDetailResponse(AgentJobResponse):
    """Detailed response schema including execution log."""

    execution_log: Optional[List[Dict[str, Any]]]


class AgentJobRelaunchLineageNode(BaseModel):
    """Single node in a relaunch lineage graph."""

    id: UUID
    name: str
    status: str
    created_at: datetime
    launch_mode: Optional[str] = None


class AgentJobRelaunchLineageResponse(BaseModel):
    """Relaunch ancestry/descendant summary for one job."""

    job_id: UUID
    root_job_id: UUID
    parent_job_id: Optional[UUID] = None
    latest_child_job_id: Optional[UUID] = None
    ancestors_truncated: bool = False
    descendants_truncated: bool = False
    ancestors: List[AgentJobRelaunchLineageNode] = Field(default_factory=list)
    descendants: List[AgentJobRelaunchLineageNode] = Field(default_factory=list)


class AgentJobTemplateResponse(BaseModel):
    """Response schema for an agent job template."""

    id: UUID
    name: str
    display_name: str
    description: Optional[str]
    category: Optional[str]
    job_type: str
    default_goal: Optional[str]
    default_config: Optional[Dict[str, Any]]
    default_chain_config: Optional[Dict[str, Any]] = None
    agent_definition_id: Optional[UUID]

    # Resource defaults
    default_max_iterations: int
    default_max_tool_calls: int
    default_max_llm_calls: int
    default_max_runtime_minutes: int

    # Visibility
    is_system: bool
    is_active: bool
    owner_user_id: Optional[UUID]

    # Optional recommendation metadata for UI/API ranking.
    recommended: Optional[bool] = None
    recommendation_score: Optional[int] = None
    recommendation_reasons: Optional[List[str]] = None

    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AgentJobTemplateListResponse(BaseModel):
    """Response schema for listing job templates."""

    templates: List[AgentJobTemplateResponse]
    total: int


class AgentJobProgressUpdate(BaseModel):
    """WebSocket message for job progress updates."""

    type: str = "progress"
    job_id: str
    progress: int
    phase: str
    status: str
    iteration: int
    phase_details: Optional[str]
    execution_graph_runtime: Optional[Dict[str, Any]] = None
    scope_observability_runtime: Optional[Dict[str, Any]] = None
    error: Optional[str]
    timestamp: str


class AgentJobActionRequest(BaseModel):
    """Request schema for job actions."""

    action: str = Field(
        ...,
        description=(
            "Action to perform: pause, resume, cancel, restart, relaunch, "
            "generate_summary, approve, reject, edit, skip, launch_tie_breaker, "
            "promote_swarm_candidate, assign_swarm_review, clear_swarm_assignment, "
            "update_swarm_review_note"
        ),
    )
    checkpoint_note: Optional[str] = Field(
        None,
        max_length=2000,
        description="Optional operator note for approval checkpoint actions.",
    )
    checkpoint_action_patch: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Optional action patch for checkpoint actions. Supported keys: tool, purpose, params."
        ),
    )
    action_payload: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Optional payload for non-checkpoint actions. Swarm actions support keys such as "
            "candidate_job_id, candidate_index, assigned_user_id, and review_note."
        ),
    )


class AgentCheckpointQueueBulkActionRequest(BaseModel):
    """Request schema for safe homogeneous bulk queue actions."""

    item_type: str = Field(
        ...,
        description="Queue item type. Supported: approval_checkpoint, job_recovery",
    )
    action: str = Field(
        ...,
        description="Bulk action. Supported by item_type validation on the server.",
    )
    job_ids: List[str] = Field(
        ...,
        min_length=1,
        description="Job ids to process in one homogeneous bulk action.",
    )
    checkpoint_note: Optional[str] = Field(
        None,
        max_length=2000,
        description="Optional shared operator note applied to each processed item.",
    )


class AgentCheckpointQueueBulkActionResultResponse(BaseModel):
    """Per-item result for a bulk queue action."""

    job_id: Optional[str] = None
    ok: bool
    status: Optional[str] = None
    error: Optional[str] = None
    queue_key: Optional[str] = None


class AgentCheckpointQueueBulkActionResponse(BaseModel):
    """Response for a bulk queue action."""

    requested_count: int
    applied: int
    failed: int
    results: List[AgentCheckpointQueueBulkActionResultResponse] = Field(
        default_factory=list
    )


class AgentCheckpointQueueFollowUpActionRequest(BaseModel):
    """Request schema for follow-up approval-bridge queue actions."""

    inbox_item_id: Optional[UUID] = None
    portfolio_id: Optional[UUID] = None
    portfolio_opportunity_id: Optional[str] = Field(None, max_length=200)
    domain_research_profile_id: Optional[UUID] = None
    profile_opportunity_id: Optional[str] = Field(None, max_length=200)
    action: str = Field(
        ..., description="Action to perform: approve_launch or reject_launch"
    )
    operator_note: Optional[str] = Field(None, max_length=2000)

    @model_validator(mode="after")
    def _validate_target(self) -> "AgentCheckpointQueueFollowUpActionRequest":
        has_inbox = self.inbox_item_id is not None
        has_portfolio = self.portfolio_id is not None and bool(
            str(self.portfolio_opportunity_id or "").strip()
        )
        has_profile = self.domain_research_profile_id is not None and bool(
            str(self.profile_opportunity_id or "").strip()
        )
        if sum(1 for flag in (has_inbox, has_portfolio, has_profile) if flag) != 1:
            raise ValueError(
                "Provide either inbox_item_id, portfolio_id + portfolio_opportunity_id, or domain_research_profile_id + profile_opportunity_id"
            )
        if self.portfolio_opportunity_id is not None:
            self.portfolio_opportunity_id = (
                str(self.portfolio_opportunity_id).strip() or None
            )
        if self.profile_opportunity_id is not None:
            self.profile_opportunity_id = (
                str(self.profile_opportunity_id).strip() or None
            )
        return self


class AgentCheckpointQueueFollowUpActionResponse(BaseModel):
    """Response for a follow-up queue action."""

    inbox_item_id: Optional[UUID] = None
    portfolio_id: Optional[UUID] = None
    portfolio_opportunity_id: Optional[str] = None
    domain_research_profile_id: Optional[UUID] = None
    profile_opportunity_id: Optional[str] = None
    ok: bool = True
    follow_up_launch_status: Optional[str] = None
    follow_up_operator_decision: Optional[str] = None
    follow_up_job_id: Optional[UUID] = None
    follow_up_chain_definition_id: Optional[UUID] = None
    detail: Optional[str] = None


class AgentCheckpointQueueBulkFollowUpActionRequest(BaseModel):
    """Request schema for homogeneous bulk follow-up approvals within one owner scope."""

    portfolio_id: Optional[UUID] = None
    portfolio_opportunity_ids: List[str] = Field(default_factory=list, max_length=100)
    domain_research_profile_id: Optional[UUID] = None
    profile_opportunity_ids: List[str] = Field(default_factory=list, max_length=100)
    action: str = Field(
        ..., description="Action to perform: approve_launch or reject_launch"
    )
    operator_note: Optional[str] = Field(None, max_length=2000)

    @model_validator(mode="after")
    def _validate_target(self) -> "AgentCheckpointQueueBulkFollowUpActionRequest":
        self.portfolio_opportunity_ids = [
            str(value).strip()
            for value in self.portfolio_opportunity_ids
            if str(value).strip()
        ]
        self.profile_opportunity_ids = [
            str(value).strip()
            for value in self.profile_opportunity_ids
            if str(value).strip()
        ]
        has_portfolio = (
            self.portfolio_id is not None and len(self.portfolio_opportunity_ids) > 0
        )
        has_profile = (
            self.domain_research_profile_id is not None
            and len(self.profile_opportunity_ids) > 0
        )
        if sum(1 for flag in (has_portfolio, has_profile) if flag) != 1:
            raise ValueError(
                "Provide either portfolio_id + portfolio_opportunity_ids or domain_research_profile_id + profile_opportunity_ids"
            )
        return self


class AgentCheckpointQueueBulkFollowUpActionResultResponse(BaseModel):
    """Per-opportunity result for a bulk follow-up action."""

    portfolio_id: Optional[UUID] = None
    portfolio_opportunity_id: Optional[str] = None
    domain_research_profile_id: Optional[UUID] = None
    profile_opportunity_id: Optional[str] = None
    ok: bool
    follow_up_launch_status: Optional[str] = None
    follow_up_operator_decision: Optional[str] = None
    follow_up_job_id: Optional[UUID] = None
    detail: Optional[str] = None
    error: Optional[str] = None


class AgentCheckpointQueueBulkFollowUpActionResponse(BaseModel):
    """Response for a bulk follow-up action."""

    requested_count: int
    applied: int
    failed: int
    results: List[AgentCheckpointQueueBulkFollowUpActionResultResponse] = Field(
        default_factory=list
    )


class AgentJobFeedbackCreate(BaseModel):
    """Create human feedback that can tune future autonomous behavior."""

    rating: int = Field(
        ..., ge=1, le=5, description="User rating for this output/checkpoint"
    )
    feedback: Optional[str] = Field(
        None, max_length=4000, description="Optional feedback text"
    )
    target_type: str = Field(
        "job", description="Target: job, checkpoint, finding, action, or tool"
    )
    target_id: Optional[str] = Field(
        None, max_length=200, description="Optional target identifier"
    )
    scope: str = Field("user", description="Learning scope: user, customer, or team")
    team_key: Optional[str] = Field(
        None, max_length=120, description="Team key when scope=team"
    )
    preferred_tools: Optional[List[str]] = Field(
        default_factory=list, description="Tools to favor in future runs"
    )
    discouraged_tools: Optional[List[str]] = Field(
        default_factory=list, description="Tools to avoid in future runs"
    )
    checkpoint: Optional[str] = Field(
        None,
        max_length=200,
        description="Checkpoint label if feedback is checkpoint-specific",
    )


class AgentJobFeedbackResponse(BaseModel):
    """Stored human feedback item."""

    id: UUID
    job_id: Optional[UUID] = None
    rating: int
    feedback: Optional[str]
    target_type: str
    target_id: Optional[str]
    scope: str
    preferred_tools: List[str] = Field(default_factory=list)
    discouraged_tools: List[str] = Field(default_factory=list)
    checkpoint: Optional[str]
    created_at: Optional[datetime]


class AgentJobFeedbackListResponse(BaseModel):
    """Paginated feedback list for a job or user scope."""

    items: List[AgentJobFeedbackResponse]
    total: int


class AgentJobExtractedMemoryResponse(BaseModel):
    """Single memory record returned by manual extraction endpoint."""

    id: str
    type: str
    content: str
    importance_score: float
    tags: List[str] = Field(default_factory=list)


class AgentJobMemoryResponse(BaseModel):
    """Single memory record returned by job-memory read/create endpoints."""

    id: str
    job_id: str
    type: str
    content: str
    importance_score: float
    tags: List[str] = Field(default_factory=list)
    context: Optional[Dict[str, Any]] = None
    access_count: int = 0
    created_at: Optional[str] = None


class AgentJobMemoryExtractResponse(BaseModel):
    """Manual extraction response with dedup and relaunch stats."""

    job_id: str
    memories_created: int
    parsed_count: int = 0
    candidate_count: int = 0
    skipped_duplicates: int = 0
    is_relaunch_chain: bool = False
    relaunch_root_job_id: Optional[str] = None
    memories: List[AgentJobExtractedMemoryResponse] = Field(default_factory=list)


class AgentJobMemoryListResponse(BaseModel):
    """List response for all memories associated with one agent job."""

    job_id: str
    memories: List[AgentJobMemoryResponse] = Field(default_factory=list)
    total: int


class AgentJobMemoryDeleteResponse(BaseModel):
    """Delete response for job memories."""

    job_id: str
    deleted_count: int


class AgentJobMemoryStatsMostAccessedItem(BaseModel):
    """Most-accessed memory item in stats response."""

    id: str
    type: str
    content: str
    access_count: int


class AgentJobMemoryStatsMostImportantItem(BaseModel):
    """Most-important memory item in stats response."""

    id: str
    type: str
    content: str
    importance: float


class AgentJobMemoryStatsResponse(BaseModel):
    """Aggregate stats for user agent-job memories."""

    total_memories: int
    by_type: Dict[str, int] = Field(default_factory=dict)
    job_sourced: int
    chat_sourced: int
    manual: int
    most_accessed: List[AgentJobMemoryStatsMostAccessedItem] = Field(
        default_factory=list
    )
    most_important: List[AgentJobMemoryStatsMostImportantItem] = Field(
        default_factory=list
    )


class AgentJobMemorySearchItemResponse(BaseModel):
    """Single search result item for agent-job memory search."""

    id: str
    type: str
    content: str
    importance_score: float
    tags: List[str] = Field(default_factory=list)
    job_id: Optional[str] = None
    access_count: int = 0
    created_at: Optional[str] = None


class AgentJobMemorySearchResponse(BaseModel):
    """Search response for agent-job memory search endpoint."""

    query: str
    memories: List[AgentJobMemorySearchItemResponse] = Field(default_factory=list)
    total: int


class AgentJobMemoryGraphNodeResponse(BaseModel):
    """Node in task-memory graph response."""

    id: str
    type: str
    content: str
    importance_score: float
    tags: List[str] = Field(default_factory=list)
    job_id: Optional[str] = None
    created_at: Optional[str] = None
    project_scope: Optional[str] = None
    execution_outcome: Optional[str] = None
    strategy_signal: Optional[str] = None
    access_count: int = 0


class AgentJobMemoryGraphEdgeResponse(BaseModel):
    """Edge in task-memory graph response."""

    source: str
    target: str
    weight: float
    reasons: List[str] = Field(default_factory=list)


class AgentJobMemoryGraphResponse(BaseModel):
    """Task-memory graph response."""

    nodes: List[AgentJobMemoryGraphNodeResponse] = Field(default_factory=list)
    edges: List[AgentJobMemoryGraphEdgeResponse] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)
    job_id: Optional[str] = None


class AgentCheckpointQueueActionResponse(BaseModel):
    """Action or launch recommendation surfaced in the checkpoint queue."""

    kind: str
    label: str
    description: Optional[str] = None
    action: Optional[str] = None
    recommended: bool = False
    launch_label: Optional[str] = None
    recommendation_key: Optional[str] = None
    autonomy_eligibility: Optional[str] = None
    recommendation_score: Optional[int] = None
    recommendation_reasons: Optional[List[str]] = None
    job_create_payload: Optional[Dict[str, Any]] = None
    chain_create_payload: Optional[Dict[str, Any]] = None
    follow_up_action_payload: Optional[Dict[str, Any]] = None
    policy_update_payload: Optional[Dict[str, Any]] = None
    policy_rollback_payload: Optional[Dict[str, Any]] = None


class AgentCheckpointQueueItemResponse(BaseModel):
    """One operator-facing item in the checkpoint/recovery queue."""

    queue_key: str
    item_type: str
    priority: int
    title: str
    summary: Optional[str] = None
    evidence_summary: Optional[str] = None
    status: Optional[str] = None
    customer: Optional[str] = None
    job_name: Optional[str] = None
    job_type: Optional[str] = None
    reason_code: Optional[str] = None
    reason_label: Optional[str] = None
    recommended_action: Optional[str] = None
    priority_score: float = 0
    age_minutes: int = 0
    sla_bucket: Optional[str] = None
    escalation_level: Optional[str] = None
    is_overdue: bool = False
    is_stale: bool = False
    next_run_at: Optional[datetime] = None
    backoff_until: Optional[datetime] = None
    action_count: int = 0
    created_at: Optional[datetime] = None
    job_id: Optional[UUID] = None
    inbox_item_id: Optional[UUID] = None
    portfolio_id: Optional[UUID] = None
    portfolio_title: Optional[str] = None
    portfolio_opportunity_id: Optional[str] = None
    portfolio_opportunity_key: Optional[str] = None
    domain_research_profile_id: Optional[UUID] = None
    domain_research_profile_title: Optional[str] = None
    profile_opportunity_id: Optional[str] = None
    profile_opportunity_key: Optional[str] = None
    domain: Optional[str] = None
    objective: Optional[str] = None
    track_type: Optional[str] = None
    source_scope: Optional[str] = None
    repo_source_ids: Optional[List[str]] = None
    benchmark_queries: Optional[List[str]] = None
    sandbox_profile_id: Optional[str] = None
    automation_profile: Optional[str] = None
    effective_policy: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    readiness: Optional[float] = None
    linked_note_ids: Optional[List[str]] = None
    linked_experiment_plan_ids: Optional[List[str]] = None
    linked_validation_run_ids: Optional[List[str]] = None
    child_job_ids: Optional[List[str]] = None
    job: Optional[AgentJobResponse] = None
    checkpoint: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    inbox_item: Optional[Dict[str, Any]] = None
    follow_up_decision: Optional[str] = None
    follow_up_policy_mode: Optional[str] = None
    follow_up_launch_status: Optional[str] = None
    follow_up_block_reason: Optional[str] = None
    follow_up_budget_decision: Optional[str] = None
    follow_up_budget_reason: Optional[str] = None
    follow_up_budget_throttle_state: Optional[str] = None
    follow_up_customer_budget_decision: Optional[str] = None
    follow_up_customer_budget_reason: Optional[str] = None
    follow_up_customer_budget_throttle_state: Optional[str] = None
    follow_up_recommendation_key: Optional[str] = None
    follow_up_job_id: Optional[UUID] = None
    follow_up_chain_definition_id: Optional[UUID] = None
    follow_up_operator_decision: Optional[str] = None
    follow_up_operator_note: Optional[str] = None
    follow_up_operator_acted_at: Optional[datetime] = None
    follow_up_operator_user_id: Optional[UUID] = None
    policy_guardrail_status: Optional[str] = None
    policy_guardrail_action: Optional[str] = None
    policy_guardrail_target_history_entry_id: Optional[str] = None
    policy_guardrail_reasons: List[str] = Field(default_factory=list)
    policy_guardrail_target_policy: Optional[Dict[str, Any]] = None
    policy_guardrail_follow_up_autonomy: Optional[Dict[str, Any]] = None
    budget_throttle_state: Optional[str] = None
    budget_reason: Optional[str] = None
    customer_budget_throttle_state: Optional[str] = None
    customer_budget_reason: Optional[str] = None
    actions: List[AgentCheckpointQueueActionResponse] = Field(default_factory=list)


class AgentCheckpointQueueResponse(BaseModel):
    """Queue response for operator-first review of approvals and follow-ups."""

    items: List[AgentCheckpointQueueItemResponse] = Field(default_factory=list)
    total: int
    limit: int = 100
    offset: int = 0
    approvals: int = 0
    recoveries: int = 0
    follow_ups: int = 0
    policy_reviews: int = 0
    budget_reviews: int = 0
    by_type: Dict[str, int] = Field(default_factory=dict)
    by_status: Dict[str, int] = Field(default_factory=dict)
    by_customer: Dict[str, int] = Field(default_factory=dict)
    by_sla_bucket: Dict[str, int] = Field(default_factory=dict)
    by_escalation_level: Dict[str, int] = Field(default_factory=dict)


class AgentDecisionTraceDeepLinkResponse(BaseModel):
    """Deep-link target metadata for a decision trace row."""

    target_tab: str
    job_id: Optional[UUID] = None
    params: Dict[str, str] = Field(default_factory=dict)
    label: Optional[str] = None


class CollaborationSummaryResponse(BaseModel):
    owner_user_id: Optional[UUID] = None
    owner_label: Optional[str] = None
    assigned_user_id: Optional[UUID] = None
    assignee_label: Optional[str] = None
    assigned_by_user_id: Optional[UUID] = None
    assigned_at: Optional[datetime] = None
    shared_with_user_ids: List[UUID] = Field(default_factory=list)
    visibility_scope: str = "private"
    is_owned_by_current_user: bool = False
    is_assigned_to_current_user: bool = False
    is_shared_with_current_user: bool = False
    note: Optional[str] = None


class AgentDecisionTraceEventResponse(BaseModel):
    """One normalized operator-facing autonomy or intervention event."""

    event_id: str
    event_type: str
    event_time: datetime
    source_kind: str
    source_id: Optional[str] = None
    source_label: Optional[str] = None
    customer: Optional[str] = None
    decision_type: str
    reason_code: Optional[str] = None
    reason_label: Optional[str] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    severity: Optional[str] = None
    actor_mode: Optional[str] = None
    summary: str
    operator_note: Optional[str] = None
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    deep_link: Optional[AgentDecisionTraceDeepLinkResponse] = None
    metadata: Optional[Dict[str, Any]] = None
    is_derived: bool = False
    record_origin: Optional[str] = None
    triage_status: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by_user_id: Optional[UUID] = None
    resolved_at: Optional[datetime] = None
    resolved_by_user_id: Optional[UUID] = None
    resolution_note: Optional[str] = None
    pinned: bool = False
    last_viewed_at: Optional[datetime] = None
    owner_user_id: Optional[UUID] = None
    owner_label: Optional[str] = None
    assigned_to_user_id: Optional[UUID] = None
    assigned_at: Optional[datetime] = None
    assigned_by_user_id: Optional[UUID] = None
    assignee_label: Optional[str] = None
    is_owned_by_current_user: bool = False
    is_assigned_to_current_user: bool = False
    team_bucket: Optional[str] = None
    due_at: Optional[datetime] = None
    escalation_state: Optional[str] = None
    escalation_reason: Optional[str] = None
    escalated_at: Optional[datetime] = None
    domain: Optional[str] = None
    objective: Optional[str] = None
    track_type: Optional[str] = None
    source_scope: Optional[str] = None
    repo_source_ids: Optional[List[str]] = None
    benchmark_queries: Optional[List[str]] = None
    sandbox_profile_id: Optional[str] = None
    automation_profile: Optional[str] = None
    effective_policy: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    readiness: Optional[float] = None
    linked_note_ids: Optional[List[str]] = None
    linked_experiment_plan_ids: Optional[List[str]] = None
    linked_validation_run_ids: Optional[List[str]] = None
    child_job_ids: Optional[List[str]] = None


class AgentDecisionTraceResponse(BaseModel):
    """Read-only decision trace feed across autonomy-backed surfaces."""

    items: List[AgentDecisionTraceEventResponse] = Field(default_factory=list)
    total: int
    limit: int = 100
    offset: int = 0
    by_source_kind: Dict[str, int] = Field(default_factory=dict)
    by_decision_type: Dict[str, int] = Field(default_factory=dict)
    by_status: Dict[str, int] = Field(default_factory=dict)
    by_customer: Dict[str, int] = Field(default_factory=dict)
    by_severity: Dict[str, int] = Field(default_factory=dict)
    by_actor_mode: Dict[str, int] = Field(default_factory=dict)
    by_triage_status: Dict[str, int] = Field(default_factory=dict)
    by_assignee: Dict[str, int] = Field(default_factory=dict)
    by_escalation_state: Dict[str, int] = Field(default_factory=dict)
    overdue_count: int = 0
    has_more: bool = False


class AgentDecisionTraceAnalyticsBucketResponse(BaseModel):
    value: str
    count: int


class AgentDecisionTraceAnalyticsTrendPointResponse(BaseModel):
    day: str
    count: int


class AgentDecisionTraceAnalyticsResponse(BaseModel):
    """Aggregated trace trends for the decision-trace dashboard."""

    window_days: int = 7
    total: int
    by_source_kind: Dict[str, int] = Field(default_factory=dict)
    by_triage_status: Dict[str, int] = Field(default_factory=dict)
    top_decision_types: List[AgentDecisionTraceAnalyticsBucketResponse] = Field(
        default_factory=list
    )
    top_reason_labels: List[AgentDecisionTraceAnalyticsBucketResponse] = Field(
        default_factory=list
    )
    top_queue_reasons: List[AgentDecisionTraceAnalyticsBucketResponse] = Field(
        default_factory=list
    )
    daily_trend: List[AgentDecisionTraceAnalyticsTrendPointResponse] = Field(
        default_factory=list
    )


class AgentDecisionTraceActionRequest(BaseModel):
    """Workflow action for a persisted decision-trace event."""

    action: str = Field(
        ...,
        description="Action to perform: acknowledge, start_investigation, resolve, reopen, toggle_pin, assign, unassign, set_due_at, clear_due_at, approve_launch, reject_launch, relaunch_follow_up",
    )
    note: Optional[str] = Field(None, max_length=2000)
    assigned_to_user_id: Optional[UUID] = None
    due_at: Optional[datetime] = None


class AgentDecisionTraceActionResponse(BaseModel):
    """Response for a decision-trace event workflow action."""

    event: AgentDecisionTraceEventResponse


class AgentDecisionTraceViewBase(BaseModel):
    """User-scoped saved filter preset for the decision trace."""

    name: str = Field(..., min_length=1, max_length=255)
    filters: Dict[str, Any] = Field(default_factory=dict)
    is_default: bool = False


class AgentDecisionTraceViewCreate(AgentDecisionTraceViewBase):
    pass


class AgentDecisionTraceViewUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    filters: Optional[Dict[str, Any]] = None
    is_default: Optional[bool] = None


class AgentDecisionTraceViewResponse(AgentDecisionTraceViewBase):
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AgentDecisionTraceViewListResponse(BaseModel):
    items: List[AgentDecisionTraceViewResponse] = Field(default_factory=list)
    total: int


class AgentJobStatsResponse(BaseModel):
    """Response schema for job statistics."""

    total_jobs: int
    running_jobs: int
    pending_jobs: int
    completed_jobs: int
    failed_jobs: int
    total_iterations: int
    total_tool_calls: int
    total_llm_calls: int
    avg_completion_time_minutes: Optional[float]
    success_rate: Optional[float]
    launch_mode_counts: Dict[str, int] = Field(default_factory=dict)
    launch_mode_none_count: int = 0


class AgentJobSwarmAnalyticsPresetRowResponse(BaseModel):
    preset_key: str
    launch_mode: str
    label: str
    total_runs: int = 0
    avg_confidence: Optional[float] = None
    high_confidence_runs: int = 0
    medium_confidence_runs: int = 0
    low_confidence_runs: int = 0
    auto_promoted_runs: int = 0
    review_needed_runs: int = 0
    tie_breaker_runs: int = 0
    manual_promotion_runs: int = 0
    repair_handoff_runs: int = 0
    backlog_handoff_runs: int = 0
    auto_backlog_handoff_runs: int = 0
    manual_backlog_handoff_runs: int = 0
    backlog_auto_suppressed_runs: int = 0
    promotion_rate: Optional[float] = None
    review_rate: Optional[float] = None
    tie_breaker_rate: Optional[float] = None


class AgentJobSwarmAnalyticsResponse(BaseModel):
    preset_rows: List[AgentJobSwarmAnalyticsPresetRowResponse] = Field(
        default_factory=list
    )
    totals: Dict[str, Any] = Field(default_factory=dict)
    filters: Dict[str, Any] = Field(default_factory=dict)


class AgentJobSwarmOutcomeCaseResponse(BaseModel):
    swarm_job_id: str
    swarm_job_name: Optional[str] = None
    preset_key: str
    launch_mode: str
    source_id: Optional[str] = None
    source_label: Optional[str] = None
    swarm_status: Optional[str] = None
    swarm_completed_at: Optional[datetime] = None
    review_state: Optional[str] = None
    review_reason: Optional[str] = None
    owner_user_id: Optional[str] = None
    assigned_user_id: Optional[str] = None
    assigned_at: Optional[datetime] = None
    assigned_by_user_id: Optional[str] = None
    review_note: Optional[str] = None
    collaboration_summary: Optional[CollaborationSummaryResponse] = None
    promotion_mode: str = "none"
    confidence_overall: Optional[float] = None
    tie_breaker_attempted: bool = False
    repair_job_id: Optional[str] = None
    repair_job_name: Optional[str] = None
    repair_status: Optional[str] = None
    repair_handoff_at: Optional[datetime] = None
    verification_status: Optional[str] = None
    verification_reason: Optional[str] = None
    backlog_item_id: Optional[str] = None
    backlog_title: Optional[str] = None
    backlog_status: Optional[str] = None
    backlog_route_mode: Optional[str] = None
    backlog_routed_at: Optional[datetime] = None
    latest_downstream_at: Optional[datetime] = None
    handoff_latency_minutes: Optional[float] = None
    terminal_outcome: str
    terminal_reason: Optional[str] = None


class AgentJobSwarmOutcomePresetRowResponse(BaseModel):
    preset_key: str
    launch_mode: str
    label: str
    total_swarm_roots: int = 0
    auto_promoted_runs: int = 0
    manual_promoted_runs: int = 0
    tie_breaker_runs: int = 0
    repair_handoff_runs: int = 0
    verified_fix_runs: int = 0
    repair_failed_runs: int = 0
    backlog_routed_runs: int = 0
    auto_backlog_routed_runs: int = 0
    manual_backlog_routed_runs: int = 0
    backlog_auto_suppressed_runs: int = 0
    needs_review_runs: int = 0
    stalled_after_handoff_runs: int = 0
    avg_confidence: Optional[float] = None
    avg_handoff_minutes: Optional[float] = None


class AgentJobSwarmOutcomeAnalyticsResponse(BaseModel):
    preset_rows: List[AgentJobSwarmOutcomePresetRowResponse] = Field(
        default_factory=list
    )
    cases: List[AgentJobSwarmOutcomeCaseResponse] = Field(default_factory=list)
    totals: Dict[str, Any] = Field(default_factory=dict)
    filters: Dict[str, Any] = Field(default_factory=dict)


class AgentJobCheckpointResponse(BaseModel):
    """Response schema for a job checkpoint."""

    id: UUID
    job_id: UUID
    iteration: int
    phase: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# Chain Definition schemas
class ChainStepConfig(BaseModel):
    """Configuration for a single step in a job chain."""

    step_name: str = Field(..., description="Name of this step")
    template_id: Optional[UUID] = Field(None, description="Optional template to use")
    job_type: str = Field("custom", description="Job type for this step")
    goal_template: str = Field(
        ..., description="Goal template with {variable} placeholders"
    )
    config: Optional[Dict[str, Any]] = Field(
        None, description="Step-specific configuration"
    )
    trigger_condition: str = Field(
        "on_complete", description="When to trigger next step"
    )
    trigger_thresholds: Optional[Dict[str, int]] = Field(
        None, description="Thresholds for progress/findings triggers"
    )

    @field_validator("config", mode="before")
    @classmethod
    def _normalize_config(cls, value: Any) -> Any:
        return normalize_scope_config(value)


class AgentJobChainDefinitionCreate(BaseModel):
    """Request schema for creating a chain definition."""

    name: str = Field(
        ..., min_length=1, max_length=100, description="Unique chain name"
    )
    display_name: str = Field(
        ..., min_length=1, max_length=200, description="Display name"
    )
    description: Optional[str] = Field(None, description="Chain description")
    chain_steps: List[ChainStepConfig] = Field(
        ..., min_length=1, description="Ordered list of chain steps"
    )
    default_settings: Optional[Dict[str, Any]] = Field(
        None, description="Default settings for all jobs"
    )

    @field_validator("default_settings", mode="before")
    @classmethod
    def _normalize_default_settings(cls, value: Any) -> Any:
        return normalize_scope_config(value)


class AgentJobChainDefinitionUpdate(BaseModel):
    """Request schema for updating a chain definition."""

    display_name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    chain_steps: Optional[List[ChainStepConfig]] = Field(None, min_length=1)
    default_settings: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None

    @field_validator("default_settings", mode="before")
    @classmethod
    def _normalize_default_settings(cls, value: Any) -> Any:
        return normalize_scope_config(value)


class AgentJobChainDefinitionResponse(BaseModel):
    """Response schema for a chain definition."""

    id: UUID
    name: str
    display_name: str
    description: Optional[str]
    chain_steps: List[Dict[str, Any]]
    default_settings: Optional[Dict[str, Any]]
    owner_user_id: Optional[UUID]
    is_system: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class AgentJobChainDefinitionListResponse(BaseModel):
    """Response schema for listing chain definitions."""

    chains: List[AgentJobChainDefinitionResponse]
    total: int


class AgentJobFromChainCreate(BaseModel):
    """Request schema for creating a job chain from a definition."""

    chain_definition_id: UUID = Field(
        ..., description="ID of the chain definition to use"
    )
    name_prefix: str = Field(
        ..., min_length=1, max_length=150, description="Prefix for job names"
    )
    variables: Dict[str, str] = Field(
        default_factory=dict, description="Variables to substitute in goal templates"
    )
    config_overrides: Optional[Dict[str, Any]] = Field(
        None, description="Override chain default settings"
    )
    start_immediately: bool = Field(True, description="Start first job immediately")

    @field_validator("config_overrides", mode="before")
    @classmethod
    def _normalize_config_overrides(cls, value: Any) -> Any:
        return normalize_scope_config(value)


class AgentJobSaveAsChainRequest(BaseModel):
    """Request schema for saving an executed job chain as a reusable chain definition (playbook)."""

    name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=100,
        description="Unique chain name (optional)",
    )
    display_name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Display name (optional)",
    )
    description: Optional[str] = Field(
        default=None, max_length=2000, description="Description (optional)"
    )


class AgentJobChainStatusResponse(BaseModel):
    """Response schema for chain status."""

    root_job_id: UUID
    chain_definition_id: Optional[UUID]
    total_steps: int
    completed_steps: int
    current_step: int
    overall_progress: int
    status: str  # pending, running, completed, failed, partially_completed
    jobs: List[AgentJobResponse]


AgentJobPromoteDomainResearchResponse.model_rebuild()

# Resolve the forward references deferred in app.schemas.experiment to break
# the import cycle: experiment -> agent_job -> domain_research_profile -> experiment.
from app.schemas import experiment as _experiment_schemas  # noqa: E402

_experiment_schemas.AgentJobExperimentRunResponse = AgentJobExperimentRunResponse
_experiment_schemas.AgentJobOperatorInterventionResponse = (
    AgentJobOperatorInterventionResponse
)
_experiment_schemas.ExperimentRunResponse.model_rebuild()
