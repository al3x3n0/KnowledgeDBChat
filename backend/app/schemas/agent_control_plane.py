from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.agent_job import AgentDecisionTraceEventResponse, AgentJobMemoryGraphResponse


class AgentControlRunRoutingSummary(BaseModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    routing_tier: Optional[str] = None
    requested_tier: Optional[str] = None
    request_count: int = 0
    summary: Optional[str] = None


class AgentControlRunNode(BaseModel):
    id: str
    kind: str
    label: str
    status: Optional[str] = None
    stage: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentControlRunEdge(BaseModel):
    source: str
    target: str
    relation: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentControlRunReplaySummary(BaseModel):
    replayability_status: str = "partial_lineage"
    planner_summary: Optional[str] = None
    router_summary: Optional[str] = None
    executor_summary: Optional[str] = None
    ended_at: Optional[datetime] = None


class AgentControlRunLinkResponse(BaseModel):
    label: str
    path: str


class AgentControlRunReviewItemResponse(BaseModel):
    run_id: Optional[str] = None
    run_title: Optional[str] = None
    run_source_type: Optional[str] = None
    run_status: Optional[str] = None
    review_type: Optional[str] = None
    review_status: Optional[str] = None
    reason_code: Optional[str] = None
    reason_label: Optional[str] = None
    source_kind: Optional[str] = None
    source_id: Optional[str] = None
    opportunity_id: Optional[str] = None
    canonical_key: Optional[str] = None
    title: Optional[str] = None
    evidence_revision: Optional[str] = None
    autonomy_state: Optional[str] = None
    operator_note: Optional[str] = None
    created_at: Optional[datetime] = None
    action_path: Optional[str] = None
    queue_path: Optional[str] = None
    note_path: Optional[str] = None
    synthesis_path: Optional[str] = None
    item_type: Optional[str] = None
    queue_item_key: Optional[str] = None
    status: Optional[str] = None
    summary: Optional[str] = None
    evidence_summary: Optional[str] = None
    customer: Optional[str] = None
    job_id: Optional[str] = None
    job_name: Optional[str] = None
    job_type: Optional[str] = None
    age_minutes: Optional[int] = None
    priority_score: Optional[float] = None
    sla_bucket: Optional[str] = None
    escalation_level: Optional[str] = None
    next_run_at: Optional[datetime] = None
    backoff_until: Optional[datetime] = None
    checkpoint: Optional[Dict[str, Any]] = None
    checkpoint_action_draft: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    follow_up_launch_status: Optional[str] = None
    follow_up_review_status: Optional[str] = None
    follow_up_recommendation_key: Optional[str] = None
    recommendation_score: Optional[float] = None
    follow_up_block_reason: Optional[str] = None
    follow_up_budget_decision: Optional[str] = None
    follow_up_budget_reason: Optional[str] = None
    follow_up_customer_budget_decision: Optional[str] = None
    follow_up_customer_budget_reason: Optional[str] = None
    recommended_action: Optional[str] = None
    policy_update_payload: Optional[Dict[str, Any]] = None
    policy_rollback_payload: Optional[Dict[str, Any]] = None
    policy_guardrail_action: Optional[str] = None
    policy_guardrail_target_history_entry_id: Optional[str] = None
    policy_guardrail_reasons: List[str] = Field(default_factory=list)
    budget_throttle_state: Optional[str] = None
    budget_reason: Optional[str] = None
    customer_budget_throttle_state: Optional[str] = None
    customer_budget_reason: Optional[str] = None
    available_actions: List[str] = Field(default_factory=list)
    can_acknowledge: bool = False
    can_approve: bool = False
    can_reject: bool = False
    can_defer: bool = False
    can_launch_follow_up: bool = False
    can_relaunch_follow_up: bool = False
    can_skip: bool = False
    can_restart: bool = False
    can_resume: bool = False
    can_cancel: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentControlRunReviewActionRequest(BaseModel):
    review_type: str
    source_kind: str
    source_id: str
    opportunity_id: str
    action: str
    operator_note: Optional[str] = Field(default=None, max_length=2000)
    reason_code: Optional[str] = Field(default=None, max_length=200)
    checkpoint_action_patch: Optional[Dict[str, Any]] = None


class AgentControlRunReviewActionResponse(BaseModel):
    ok: bool = True
    action: str
    review_type: Optional[str] = None
    source_kind: Optional[str] = None
    source_id: Optional[str] = None
    opportunity_id: Optional[str] = None
    detail: Optional[str] = None
    monitor_job_id: Optional[str] = None
    follow_up_launch_status: Optional[str] = None
    follow_up_operator_decision: Optional[str] = None
    follow_up_job_id: Optional[str] = None


class AgentControlRunBulkReviewActionRequest(BaseModel):
    item_type: str
    action: str
    job_ids: List[str] = Field(default_factory=list)
    domain_research_profile_id: Optional[str] = None
    profile_opportunity_ids: List[str] = Field(default_factory=list)
    portfolio_id: Optional[str] = None
    portfolio_opportunity_ids: List[str] = Field(default_factory=list)
    operator_note: Optional[str] = Field(default=None, max_length=2000)


class AgentControlRunBulkReviewActionResultResponse(BaseModel):
    item_key: Optional[str] = None
    job_id: Optional[str] = None
    opportunity_id: Optional[str] = None
    ok: bool
    detail: Optional[str] = None
    error: Optional[str] = None
    status: Optional[str] = None
    follow_up_launch_status: Optional[str] = None
    follow_up_operator_decision: Optional[str] = None
    follow_up_job_id: Optional[str] = None


class AgentControlRunBulkReviewActionResponse(BaseModel):
    ok: bool = True
    item_type: str
    action: str
    requested_count: int
    applied: int
    failed: int
    results: List[AgentControlRunBulkReviewActionResultResponse] = Field(default_factory=list)


class AgentControlRunSummary(BaseModel):
    id: str
    source_type: str
    title: str
    subtitle: Optional[str] = None
    status: str
    outcome: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    root_job_id: Optional[str] = None
    workflow_execution_id: Optional[str] = None
    child_job_count: int = 0
    child_execution_count: int = 0
    linked_note_count: int = 0
    linked_experiment_count: int = 0
    decision_count: int = 0
    replayability_status: str = "partial_lineage"
    automation_profile: Optional[str] = None
    routing: Optional[AgentControlRunRoutingSummary] = None
    queued_operator_review_count: int = 0
    queued_operator_reviews_by_type: Dict[str, int] = Field(default_factory=dict)


class AgentControlRunListResponse(BaseModel):
    items: List[AgentControlRunSummary] = Field(default_factory=list)
    total: int = 0


class AgentControlRunReviewListResponse(BaseModel):
    items: List[AgentControlRunReviewItemResponse] = Field(default_factory=list)
    total: int = 0
    summary: Dict[str, Dict[str, int] | int] = Field(default_factory=dict)
    offset: int = 0
    limit: int = 0
    has_more: bool = False


class AgentControlRunViewBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    filters: Dict[str, Any] = Field(default_factory=dict)
    is_default: bool = False


class AgentControlRunViewCreate(AgentControlRunViewBase):
    pass


class AgentControlRunViewUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    filters: Optional[Dict[str, Any]] = None
    is_default: Optional[bool] = None


class AgentControlRunViewResponse(AgentControlRunViewBase):
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime


class AgentControlRunViewListResponse(BaseModel):
    items: List[AgentControlRunViewResponse] = Field(default_factory=list)
    total: int = 0


class AgentControlRunDetail(BaseModel):
    run: AgentControlRunSummary
    nodes: List[AgentControlRunNode] = Field(default_factory=list)
    edges: List[AgentControlRunEdge] = Field(default_factory=list)
    decision_trace: List[AgentDecisionTraceEventResponse] = Field(default_factory=list)
    memory_graph: Optional[AgentJobMemoryGraphResponse] = None
    routing: Optional[AgentControlRunRoutingSummary] = None
    replay: AgentControlRunReplaySummary
    related_links: List[AgentControlRunLinkResponse] = Field(default_factory=list)
    queued_operator_review_count: int = 0
    queued_operator_reviews: List[AgentControlRunReviewItemResponse] = Field(default_factory=list)
    policy_summary: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
