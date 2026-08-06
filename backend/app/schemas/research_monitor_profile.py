"""
Pydantic schemas for research monitor profiles.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ResearchMonitorBudgetConfigResponse(BaseModel):
    auto_launch_limit_24h: int = 0
    approval_queue_limit_24h: int = 0
    alert_limit_24h: int = 0
    queue_backlog_cap: int = 0


class ResearchMonitorProfileResponse(BaseModel):
    id: UUID
    user_id: UUID
    customer: Optional[str] = None
    token_scores: Optional[Dict[str, int]] = None
    phrase_scores: Optional[Dict[str, int]] = None
    recommendation_scores: Optional[Dict[str, int]] = None
    source_type_scores: Optional[Dict[str, int]] = None
    outcome_counters: Optional[Dict[str, int]] = None
    customer_budget_config: Optional[ResearchMonitorBudgetConfigResponse] = None
    muted_tokens: Optional[List[str]] = None
    muted_patterns: Optional[List[str]] = None
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ResearchMonitorProfileUpdateRequest(BaseModel):
    muted_tokens: Optional[List[str]] = Field(default=None)
    muted_patterns: Optional[List[str]] = Field(default=None)
    notes: Optional[str] = Field(default=None, max_length=4000)


class ResearchMonitorRecommendationAnalyticsResponse(BaseModel):
    recommendation_key: str
    launch_count: int = 0
    auto_launch_count: int = 0
    approval_launch_count: int = 0
    blocked_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    cancelled_count: int = 0
    success_rate: float = 0.0
    score_trend: str = "mixed"
    monitor_count: int = 0


class ResearchMonitorPolicyConfigResponse(BaseModel):
    mode: str = "manual_only"
    allowed_recommendations: List[str] = Field(default_factory=list)
    automation_profile: str = "balanced"
    automation_policy: Dict[str, Any] = Field(default_factory=dict)
    effective_policy: Dict[str, Any] = Field(default_factory=dict)


class ResearchMonitorBudgetUsageResponse(BaseModel):
    auto_launch_count_24h: int = 0
    approval_queue_count_24h: int = 0
    alert_count_24h: int = 0
    queue_backlog_count: int = 0


class ResearchMonitorCustomerTopContributorResponse(BaseModel):
    monitor_job_id: Optional[UUID] = None
    monitor_name: str
    customer: Optional[str] = None
    value: int = 0
    throttle_state: Optional[str] = None


class ResearchMonitorBudgetHistoryEntryResponse(BaseModel):
    id: str
    at: datetime
    actor_user_id: Optional[str] = None
    change_source: str = "manual_override"
    change_reason: Optional[str] = None
    previous_autonomy_budget: ResearchMonitorBudgetConfigResponse
    next_autonomy_budget: ResearchMonitorBudgetConfigResponse
    guidance_context: Dict[str, Any] = Field(default_factory=dict)


class ResearchMonitorCustomerRebalanceChangeResponse(BaseModel):
    monitor_job_id: UUID
    monitor_name: str
    customer: Optional[str] = None
    current_budget: ResearchMonitorBudgetConfigResponse
    proposed_budget: ResearchMonitorBudgetConfigResponse
    delta_budget: ResearchMonitorBudgetConfigResponse
    reasons: List[str] = Field(default_factory=list)


class ResearchMonitorCustomerRebalancePreviewResponse(BaseModel):
    customer: str
    guidance_status: str = "none"
    guidance_summary: Optional[str] = None
    guidance_reasons: List[str] = Field(default_factory=list)
    before_capacity: ResearchMonitorBudgetConfigResponse = Field(
        default_factory=ResearchMonitorBudgetConfigResponse
    )
    after_capacity: ResearchMonitorBudgetConfigResponse = Field(
        default_factory=ResearchMonitorBudgetConfigResponse
    )
    changes: List[ResearchMonitorCustomerRebalanceChangeResponse] = Field(
        default_factory=list
    )


class ResearchMonitorCustomerRebalanceEvaluationCountsResponse(BaseModel):
    accepted_count: int = 0
    blocked_count: int = 0
    follow_up_completed_count: int = 0
    follow_up_failed_count: int = 0
    follow_up_cancelled_count: int = 0
    auto_launch_used_24h: int = 0
    approval_queue_used_24h: int = 0
    alert_used_24h: int = 0
    backlog_used: int = 0
    throttled_monitor_count: int = 0


class ResearchMonitorCustomerRebalanceHistoryEntryResponse(BaseModel):
    id: str
    at: datetime
    actor_user_id: Optional[str] = None
    change_source: str = "customer_rebalance_guidance"
    change_reason: Optional[str] = None
    changes: List[ResearchMonitorCustomerRebalanceChangeResponse] = Field(
        default_factory=list
    )
    before_capacity: ResearchMonitorBudgetConfigResponse = Field(
        default_factory=ResearchMonitorBudgetConfigResponse
    )
    after_capacity: ResearchMonitorBudgetConfigResponse = Field(
        default_factory=ResearchMonitorBudgetConfigResponse
    )
    evaluation_target_count: int = 8
    evaluation_state: str = "active"
    evaluation_status: Optional[str] = None
    evaluation_sample_count: int = 0
    evaluation_reasons: List[str] = Field(default_factory=list)
    before_counts: ResearchMonitorCustomerRebalanceEvaluationCountsResponse = Field(
        default_factory=ResearchMonitorCustomerRebalanceEvaluationCountsResponse
    )
    after_counts: ResearchMonitorCustomerRebalanceEvaluationCountsResponse = Field(
        default_factory=ResearchMonitorCustomerRebalanceEvaluationCountsResponse
    )
    delta_counts: ResearchMonitorCustomerRebalanceEvaluationCountsResponse = Field(
        default_factory=ResearchMonitorCustomerRebalanceEvaluationCountsResponse
    )


class ResearchMonitorCustomerRebalanceEvaluationSampleResponse(BaseModel):
    item_id: UUID
    title: str
    period: str
    launch_status: Optional[str] = None
    outcome_status: Optional[str] = None
    recommendation_key: Optional[str] = None
    summary: Optional[str] = None
    monitor_job_id: Optional[UUID] = None
    monitor_name: Optional[str] = None


class ResearchMonitorCustomerRebalanceEvaluationDetailResponse(BaseModel):
    customer: str
    history_entry_id: str
    evaluation_status: str = "insufficient_data"
    evaluation_sample_count: int = 0
    evaluation_target_count: int = 8
    evaluation_reasons: List[str] = Field(default_factory=list)
    before_counts: ResearchMonitorCustomerRebalanceEvaluationCountsResponse = Field(
        default_factory=ResearchMonitorCustomerRebalanceEvaluationCountsResponse
    )
    after_counts: ResearchMonitorCustomerRebalanceEvaluationCountsResponse = Field(
        default_factory=ResearchMonitorCustomerRebalanceEvaluationCountsResponse
    )
    delta_counts: ResearchMonitorCustomerRebalanceEvaluationCountsResponse = Field(
        default_factory=ResearchMonitorCustomerRebalanceEvaluationCountsResponse
    )
    sample_items: List[
        ResearchMonitorCustomerRebalanceEvaluationSampleResponse
    ] = Field(default_factory=list)


class ResearchMonitorCustomerPortfolioResponse(BaseModel):
    customer: str
    monitor_count: int = 0
    strong_monitor_count: int = 0
    mixed_monitor_count: int = 0
    weak_monitor_count: int = 0
    auto_launch_used_24h: int = 0
    auto_launch_capacity_24h: int = 0
    approval_queue_used_24h: int = 0
    approval_queue_capacity_24h: int = 0
    alert_used_24h: int = 0
    alert_capacity_24h: int = 0
    backlog_used: int = 0
    backlog_capacity: int = 0
    throttled_monitor_count: int = 0
    customer_budget: ResearchMonitorBudgetConfigResponse = Field(
        default_factory=ResearchMonitorBudgetConfigResponse
    )
    customer_budget_usage: ResearchMonitorBudgetUsageResponse = Field(
        default_factory=ResearchMonitorBudgetUsageResponse
    )
    customer_budget_remaining: ResearchMonitorBudgetUsageResponse = Field(
        default_factory=ResearchMonitorBudgetUsageResponse
    )
    customer_budget_throttle_state: str = "normal"
    customer_budget_throttle_reasons: List[str] = Field(default_factory=list)
    accepted_count: int = 0
    blocked_count: int = 0
    follow_up_completed_count: int = 0
    follow_up_failed_count: int = 0
    follow_up_cancelled_count: int = 0
    portfolio_status: str = "normal"
    portfolio_reasons: List[str] = Field(default_factory=list)
    top_launch_monitors: List[ResearchMonitorCustomerTopContributorResponse] = Field(
        default_factory=list
    )
    top_backlog_monitors: List[ResearchMonitorCustomerTopContributorResponse] = Field(
        default_factory=list
    )
    top_alert_monitors: List[ResearchMonitorCustomerTopContributorResponse] = Field(
        default_factory=list
    )
    throttled_monitors: List[ResearchMonitorCustomerTopContributorResponse] = Field(
        default_factory=list
    )
    rebalance_guidance_status: str = "none"
    rebalance_guidance_reasons: List[str] = Field(default_factory=list)
    rebalance_guidance_summary: Optional[str] = None
    rebalance_guidance_changes: List[
        ResearchMonitorCustomerRebalanceChangeResponse
    ] = Field(default_factory=list)
    latest_rebalance_evaluation_status: Optional[str] = None
    latest_rebalance_evaluation_sample_count: int = 0
    latest_rebalance_evaluation_target_count: int = 0
    latest_rebalance_evaluation_reasons: List[str] = Field(default_factory=list)
    recent_rebalance_history: List[
        ResearchMonitorCustomerRebalanceHistoryEntryResponse
    ] = Field(default_factory=list)


class ResearchMonitorPolicyEvaluationCountsResponse(BaseModel):
    accepted_count: int = 0
    auto_launched_count: int = 0
    approval_launched_count: int = 0
    queued_for_approval_count: int = 0
    manual_only_count: int = 0
    blocked_count: int = 0
    follow_up_completed_count: int = 0
    follow_up_failed_count: int = 0
    follow_up_cancelled_count: int = 0


class ResearchMonitorPolicyEvaluationSampleResponse(BaseModel):
    item_id: UUID
    title: str
    period: str
    launch_status: Optional[str] = None
    outcome_status: Optional[str] = None
    recommendation_key: Optional[str] = None
    summary: Optional[str] = None


class ResearchMonitorPolicyHistoryEntryResponse(BaseModel):
    id: str
    at: datetime
    actor_user_id: Optional[str] = None
    change_source: str = "manual_override"
    change_reason: Optional[str] = None
    previous_follow_up_autonomy: ResearchMonitorPolicyConfigResponse = Field(
        description="Compatibility-only legacy mirror of the previous canonical policy snapshot"
    )
    next_follow_up_autonomy: ResearchMonitorPolicyConfigResponse = Field(
        description="Compatibility-only legacy mirror of the next canonical policy snapshot"
    )
    previous_automation_profile: str = "balanced"
    next_automation_profile: str = "balanced"
    previous_automation_policy: Dict[str, Any] = Field(default_factory=dict)
    next_automation_policy: Dict[str, Any] = Field(default_factory=dict)
    previous_effective_policy: Dict[str, Any] = Field(default_factory=dict)
    next_effective_policy: Dict[str, Any] = Field(default_factory=dict)
    effective_clamp_state: Optional[str] = None
    effective_clamp_reasons: List[str] = Field(default_factory=list)
    analytics_context: Dict[str, Any] = Field(default_factory=dict)
    evaluation_target_count: int = 8
    evaluation_state: str = "active"
    evaluation_status: Optional[str] = None
    evaluation_sample_count: int = 0
    evaluation_reasons: List[str] = Field(default_factory=list)
    before_counts: ResearchMonitorPolicyEvaluationCountsResponse = Field(
        default_factory=ResearchMonitorPolicyEvaluationCountsResponse
    )
    after_counts: ResearchMonitorPolicyEvaluationCountsResponse = Field(
        default_factory=ResearchMonitorPolicyEvaluationCountsResponse
    )
    delta_counts: ResearchMonitorPolicyEvaluationCountsResponse = Field(
        default_factory=ResearchMonitorPolicyEvaluationCountsResponse
    )


class ResearchMonitorPolicyUpdateRequest(BaseModel):
    automation_profile: Optional[str] = None
    automation_policy: Optional[Dict[str, Any]] = None
    mode: Optional[str] = Field(
        default=None,
        description="Compatibility-only legacy alias for automation_policy.follow_up_review_mode",
    )
    allowed_recommendations: Optional[List[str]] = Field(
        default=None,
        description="Compatibility-only legacy alias for automation_policy.allowed_recommendations",
    )
    reset_to_default: bool = False
    change_source: Optional[str] = None
    change_reason: Optional[str] = Field(default=None, max_length=1000)
    analytics_context: Optional[Dict[str, Any]] = None


class ResearchMonitorPolicyRollbackRequest(BaseModel):
    history_entry_id: str
    change_reason: Optional[str] = Field(default=None, max_length=1000)


class ResearchMonitorPolicySimulationRequest(BaseModel):
    automation_profile: Optional[str] = None
    automation_policy: Optional[Dict[str, Any]] = None
    mode: Optional[str] = Field(
        default=None,
        description="Compatibility-only legacy alias for automation_policy.follow_up_review_mode",
    )
    allowed_recommendations: Optional[List[str]] = Field(
        default=None,
        description="Compatibility-only legacy alias for automation_policy.allowed_recommendations",
    )
    history_limit: int = Field(default=25, ge=5, le=100)


class ResearchMonitorPolicyUpdateResponse(BaseModel):
    monitor_job_id: UUID
    follow_up_autonomy: ResearchMonitorPolicyConfigResponse = Field(
        description="Compatibility-only legacy mirror of the resolved canonical policy"
    )
    automation_profile: str = "balanced"
    automation_policy: Dict[str, Any] = Field(default_factory=dict)
    effective_policy: Dict[str, Any] = Field(default_factory=dict)
    latest_history_entry: Optional[ResearchMonitorPolicyHistoryEntryResponse] = None
    policy_history_count: int = 0


class ResearchMonitorBudgetUpdateRequest(BaseModel):
    auto_launch_limit_24h: Optional[int] = Field(default=None, ge=0, le=10000)
    approval_queue_limit_24h: Optional[int] = Field(default=None, ge=0, le=10000)
    alert_limit_24h: Optional[int] = Field(default=None, ge=0, le=10000)
    queue_backlog_cap: Optional[int] = Field(default=None, ge=0, le=10000)
    reset_to_default: bool = False
    change_source: Optional[str] = None
    change_reason: Optional[str] = Field(default=None, max_length=1000)


class ResearchMonitorBudgetUpdateResponse(BaseModel):
    monitor_job_id: UUID
    autonomy_budget: ResearchMonitorBudgetConfigResponse
    latest_history_entry: Optional[ResearchMonitorBudgetHistoryEntryResponse] = None


class ResearchMonitorCustomerBudgetUpdateRequest(BaseModel):
    customer: str = Field(min_length=1, max_length=255)
    auto_launch_limit_24h: Optional[int] = Field(default=None, ge=0, le=10000)
    approval_queue_limit_24h: Optional[int] = Field(default=None, ge=0, le=10000)
    alert_limit_24h: Optional[int] = Field(default=None, ge=0, le=10000)
    queue_backlog_cap: Optional[int] = Field(default=None, ge=0, le=10000)
    reset_to_default: bool = False


class ResearchMonitorCustomerBudgetUpdateResponse(BaseModel):
    customer: str
    customer_budget: ResearchMonitorBudgetConfigResponse


class ResearchMonitorCustomerRebalanceApplyMonitorRequest(BaseModel):
    monitor_job_id: UUID
    auto_launch_limit_24h: int = Field(ge=0, le=10000)
    approval_queue_limit_24h: int = Field(ge=0, le=10000)
    alert_limit_24h: int = Field(ge=0, le=10000)
    queue_backlog_cap: int = Field(ge=0, le=10000)


class ResearchMonitorCustomerRebalancePreviewRequest(BaseModel):
    customer: str = Field(min_length=1, max_length=255)
    monitor_budget_updates: List[
        ResearchMonitorCustomerRebalanceApplyMonitorRequest
    ] = Field(default_factory=list)


class ResearchMonitorCustomerRebalanceApplyRequest(BaseModel):
    customer: str = Field(min_length=1, max_length=255)
    monitor_budget_updates: List[
        ResearchMonitorCustomerRebalanceApplyMonitorRequest
    ] = Field(min_length=1)
    change_reason: Optional[str] = Field(default=None, max_length=1000)


class ResearchMonitorCustomerRebalanceApplyResponse(BaseModel):
    customer: str
    updated_monitor_ids: List[UUID] = Field(default_factory=list)
    guidance_status: str = "none"
    guidance_summary: Optional[str] = None
    latest_history_entries: List[ResearchMonitorBudgetHistoryEntryResponse] = Field(
        default_factory=list
    )


class ResearchMonitorPolicySimulationCountsResponse(BaseModel):
    auto_launch_safe_count: int = 0
    queue_for_approval_count: int = 0
    manual_only_count: int = 0
    blocked_count: int = 0
    insufficient_context_count: int = 0


class ResearchMonitorPolicySimulationRecommendationDeltaResponse(BaseModel):
    recommendation_key: str
    baseline_count: int = 0
    simulated_count: int = 0
    delta_count: int = 0


class ResearchMonitorPolicySimulationSampleResponse(BaseModel):
    item_id: UUID
    title: str
    recommendation_key: Optional[str] = None
    current_outcome: str
    simulated_outcome: str
    reason: str


class ResearchMonitorPolicySimulationResponse(BaseModel):
    monitor_job_id: UUID
    current_policy: ResearchMonitorPolicyConfigResponse
    proposed_policy: ResearchMonitorPolicyConfigResponse
    current_automation_profile: str = "balanced"
    current_automation_policy: Dict[str, Any] = Field(default_factory=dict)
    current_effective_policy: Dict[str, Any] = Field(default_factory=dict)
    proposed_automation_profile: str = "balanced"
    proposed_automation_policy: Dict[str, Any] = Field(default_factory=dict)
    proposed_effective_policy: Dict[str, Any] = Field(default_factory=dict)
    history_limit: int = 25
    baseline_counts: ResearchMonitorPolicySimulationCountsResponse
    simulated_counts: ResearchMonitorPolicySimulationCountsResponse
    delta_counts: ResearchMonitorPolicySimulationCountsResponse
    top_recommendation_deltas: List[
        ResearchMonitorPolicySimulationRecommendationDeltaResponse
    ] = Field(default_factory=list)
    sample_items: List[ResearchMonitorPolicySimulationSampleResponse] = Field(
        default_factory=list
    )
    insufficient_context_count: int = 0


class ResearchMonitorPolicyEvaluationDetailResponse(BaseModel):
    monitor_job_id: UUID
    history_entry_id: str
    evaluation_status: str = "insufficient_data"
    evaluation_sample_count: int = 0
    evaluation_target_count: int = 8
    evaluation_reasons: List[str] = Field(default_factory=list)
    before_counts: ResearchMonitorPolicyEvaluationCountsResponse = Field(
        default_factory=ResearchMonitorPolicyEvaluationCountsResponse
    )
    after_counts: ResearchMonitorPolicyEvaluationCountsResponse = Field(
        default_factory=ResearchMonitorPolicyEvaluationCountsResponse
    )
    delta_counts: ResearchMonitorPolicyEvaluationCountsResponse = Field(
        default_factory=ResearchMonitorPolicyEvaluationCountsResponse
    )
    sample_items: List[ResearchMonitorPolicyEvaluationSampleResponse] = Field(
        default_factory=list
    )


class ResearchMonitorHealthSummaryResponse(BaseModel):
    monitor_job_id: Optional[UUID] = None
    monitor_name: str
    monitor_job_type: Optional[str] = None
    customer: Optional[str] = None
    discovered_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    acceptance_rate: float = 0.0
    auto_launched_count: int = 0
    approval_launched_count: int = 0
    queued_for_approval_count: int = 0
    manual_only_count: int = 0
    blocked_count: int = 0
    follow_up_completed_count: int = 0
    follow_up_failed_count: int = 0
    follow_up_cancelled_count: int = 0
    relaunch_count: int = 0
    health_score: float = 0.0
    health_bucket: str = "weak"
    health_reasons: List[str] = Field(default_factory=list)
    current_policy_mode: Optional[str] = Field(
        default="manual_only",
        description="Compatibility-only legacy mirror of effective_policy.follow_up_review_mode",
    )
    current_allowed_recommendations: List[str] = Field(
        default_factory=list,
        description="Compatibility-only legacy mirror of the resolved allowed recommendation list",
    )
    automation_profile: str = "balanced"
    automation_policy: Dict[str, Any] = Field(default_factory=dict)
    effective_policy: Dict[str, Any] = Field(default_factory=dict)
    autonomy_mode: str = "balanced"
    autonomy_budget: ResearchMonitorBudgetConfigResponse = Field(
        default_factory=ResearchMonitorBudgetConfigResponse
    )
    budget_usage: ResearchMonitorBudgetUsageResponse = Field(
        default_factory=ResearchMonitorBudgetUsageResponse
    )
    budget_remaining: ResearchMonitorBudgetUsageResponse = Field(
        default_factory=ResearchMonitorBudgetUsageResponse
    )
    budget_throttle_state: str = "normal"
    budget_throttle_reasons: List[str] = Field(default_factory=list)
    budget_clamp_state: Optional[str] = None
    budget_clamp_reasons: List[str] = Field(default_factory=list)
    budget_history_count: int = 0
    latest_budget_changed_at: Optional[datetime] = None
    latest_budget_change_source: Optional[str] = None
    latest_budget_actor_user_id: Optional[str] = None
    latest_budget_change_reason: Optional[str] = None
    recommended_policy_mode: str = "manual_only"
    recommended_allowed_recommendations: List[str] = Field(default_factory=list)
    policy_reasons: List[str] = Field(default_factory=list)
    policy_confidence: str = "low"
    policy_history_count: int = 0
    latest_policy_changed_at: Optional[datetime] = None
    latest_policy_change_source: Optional[str] = None
    latest_policy_actor_user_id: Optional[str] = None
    latest_policy_evaluation_status: Optional[str] = None
    latest_policy_evaluation_sample_count: int = 0
    latest_policy_evaluation_target_count: int = 0
    latest_policy_evaluation_reasons: List[str] = Field(default_factory=list)
    policy_guardrail_status: Optional[str] = None
    policy_guardrail_action: Optional[str] = None
    policy_guardrail_reasons: List[str] = Field(default_factory=list)
    policy_guardrail_target_history_entry_id: Optional[str] = None
    policy_guardrail_follow_up_autonomy: Optional[
        ResearchMonitorPolicyConfigResponse
    ] = Field(
        default=None,
        description="Compatibility-only legacy mirror of policy_guardrail_target_policy",
    )
    policy_guardrail_state: Optional[str] = None
    policy_guardrail_target_policy: Optional[Dict[str, Any]] = None
    policy_mode_counts: Dict[str, int] = Field(default_factory=dict)
    follow_up_review_counts: Dict[str, int] = Field(default_factory=dict)
    scheduler_summary: Dict[str, Any] = Field(default_factory=dict)
    suppressed_relaunches_count: int = 0
    recent_policy_history: List[ResearchMonitorPolicyHistoryEntryResponse] = Field(
        default_factory=list
    )
    top_recommendations: List[ResearchMonitorRecommendationAnalyticsResponse] = Field(
        default_factory=list
    )


class ResearchMonitorAnalyticsTotalsResponse(BaseModel):
    total_monitors: int = 0
    discovered_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    auto_launched_count: int = 0
    approval_launched_count: int = 0
    blocked_count: int = 0
    follow_up_completed_count: int = 0
    follow_up_failed_count: int = 0
    follow_up_cancelled_count: int = 0
    strong_monitors: int = 0
    mixed_monitors: int = 0
    weak_monitors: int = 0


class ResearchMonitorAnalyticsResponse(BaseModel):
    generated_at: datetime
    totals: ResearchMonitorAnalyticsTotalsResponse
    customers: List[ResearchMonitorCustomerPortfolioResponse] = Field(
        default_factory=list
    )
    monitors: List[ResearchMonitorHealthSummaryResponse] = Field(default_factory=list)
    recommendations: List[ResearchMonitorRecommendationAnalyticsResponse] = Field(
        default_factory=list
    )
