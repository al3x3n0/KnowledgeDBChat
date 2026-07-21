import json
import pytest
from sqlalchemy import select
from uuid import uuid4
from types import SimpleNamespace
from datetime import datetime, timedelta

from app.api.endpoints import agent_jobs as agent_jobs_endpoint
from app.api.endpoints.agent_jobs import (
    act_on_decision_trace_event,
    _build_decision_trace_from_queue_items,
    _build_decision_trace_from_monitor_snapshot,
    _build_decision_trace_from_job,
    _build_decision_trace_from_opportunities,
    _build_checkpoint_queue_items,
    _build_follow_up_actions_for_inbox_item,
    _job_matches_bulk_queue_item_type,
    _job_to_response,
    _queue_priority_fields,
    _score_follow_up_action_for_item,
    _validate_bulk_queue_action,
    _build_extract_job_memories_response,
    _build_job_memory_response,
    _build_job_memories_list_response,
    _build_memory_search_response,
    _build_memory_stats_response,
    _build_memory_graph_response,
    create_decision_trace_view,
    delete_decision_trace_view,
    _record_job_operator_event,
    get_decision_trace,
    get_decision_trace_analytics,
    export_decision_trace,
    list_decision_trace_views,
    update_decision_trace_view,
)
from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.autonomy_decision_event import AutonomyDecisionEvent
from app.models.notification import Notification
from app.models.research_inbox import ResearchInboxItem
from app.models.research_portfolio import ResearchPortfolio
from app.models.domain_research_profile import DomainResearchProfile
from app.models.experiment import ExperimentRun
from app.models.user import User
from app.schemas.agent_job import AgentDecisionTraceActionRequest, AgentDecisionTraceViewCreate, AgentDecisionTraceViewUpdate
from app.services.autonomy_event_service import record_autonomy_decision_event
from fastapi import HTTPException


def test_job_to_response_exposes_launch_mode_field():
    job = AgentJob(
        name="Launch Mode Job",
        goal="Run quick start",
        job_type="analysis",
        user_id=uuid4(),
        status=AgentJobStatus.PENDING.value,
        config={"launch_mode": "quick_start_claude_backend"},
    )

    res = _job_to_response(job)

    assert res.launch_mode == "quick_start_claude_backend"


def test_job_to_response_exposes_relaunch_relation_fields():
    parent_id = uuid4()
    job = AgentJob(
        name="Relaunch Child",
        goal="Run again",
        job_type="analysis",
        user_id=uuid4(),
        status=AgentJobStatus.PENDING.value,
        config={"launch_mode": "quick_start_claude_backend", "relaunch_from_job_id": str(parent_id)},
    )

    res = _job_to_response(job, relaunch_children_count=3)

    assert str(res.relaunch_from_job_id) == str(parent_id)
    assert res.relaunch_children_count == 3


def test_job_to_response_exposes_domain_research_promotion_fields():
    profile_id = uuid4()
    portfolio_id = uuid4()
    job = AgentJob(
        name="Promoted Domain Research",
        goal="Promote research findings",
        job_type="research",
        user_id=uuid4(),
        status=AgentJobStatus.COMPLETED.value,
        config={
            "launch_mode": "quick_start_domain_research",
            "quick_start": {
                "promotion": {
                    "status": "promoted_to_profile_and_portfolio",
                    "domain_research_profile_id": str(profile_id),
                    "research_portfolio_id": str(portfolio_id),
                }
            },
        },
    )

    res = _job_to_response(job)

    assert res.promotion_status == "promoted_to_profile_and_portfolio"
    assert str(res.promoted_domain_research_profile_id) == str(profile_id)
    assert str(res.promoted_research_portfolio_id) == str(portfolio_id)


def test_job_to_response_normalizes_scope_keys_in_config_and_chain_config():
    job = AgentJob(
        name="Scope Key Mapping",
        goal="Normalize nested scope keys",
        job_type="analysis",
        user_id=uuid4(),
        status=AgentJobStatus.PENDING.value,
        config={"target_source_id": "src-123"},
        chain_config={
            "default_settings": {"target_source_id": "root-src"},
            "child_jobs": [
                {"name": "Step 1", "config": {"target_source_id": "child-src-1"}},
                {"name": "Step 2", "config": {"source_id": "child-src-2", "target_source_id": "legacy"}},
            ],
        },
    )

    res = _job_to_response(job)

    assert res.config is not None
    assert res.config.get("source_id") == "src-123"
    assert "target_source_id" not in res.config
    assert isinstance(res.chain_config, dict)
    assert res.chain_config["default_settings"]["source_id"] == "root-src"
    assert "target_source_id" not in res.chain_config["default_settings"]
    assert res.chain_config["child_jobs"][0]["config"]["source_id"] == "child-src-1"
    assert "target_source_id" not in res.chain_config["child_jobs"][0]["config"]
    assert res.chain_config["child_jobs"][1]["config"]["source_id"] == "child-src-2"
    assert "target_source_id" not in res.chain_config["child_jobs"][1]["config"]


def test_job_to_response_projects_typed_experiment_run_fields():
    job = AgentJob(
        name="Experiment Projection",
        goal="Expose bootstrap execution details",
        job_type="analysis",
        user_id=uuid4(),
        status=AgentJobStatus.COMPLETED.value,
        results={
            "experiment_run": {
                "source_id": "repo-1",
                "source_name": "Knowledge Repo",
                "ok": True,
                "final_phase": "retry_primary",
                "verification_commands": ["CI=true npm --prefix frontend test -- --watchAll=false"],
                "bootstrap_commands": ["npm --prefix frontend install"],
                "fallback_commands": ["python3 -m pytest -q backend/tests"],
                "phases": ["primary", "bootstrap", "retry_primary"],
                "failed_commands": ["CI=true npm --prefix frontend test -- --watchAll=false"],
                "bootstrap_attempted": True,
                "bootstrap_ok": True,
            },
            "experiment_runs": [
                {"source_id": "repo-older", "ok": False, "final_phase": "primary"},
            ],
        },
    )

    res = _job_to_response(job)

    assert res.experiment_run is not None
    assert res.experiment_run.source_id == "repo-1"
    assert res.experiment_run.final_phase == "retry_primary"
    assert res.experiment_run.bootstrap_ok is True
    assert res.experiment_run.verification_commands == [
        "CI=true npm --prefix frontend test -- --watchAll=false"
    ]
    assert res.experiment_runs is not None
    assert len(res.experiment_runs) == 1
    assert res.experiment_runs[0].source_id == "repo-older"


def test_job_to_response_ignores_invalid_experiment_run_shapes():
    job = AgentJob(
        name="Experiment Projection Invalid",
        goal="Skip malformed experiment payloads",
        job_type="analysis",
        user_id=uuid4(),
        status=AgentJobStatus.RUNNING.value,
        results={
            "experiment_run": "bad-payload",
            "experiment_runs": ["x", {"source_id": "repo-1", "ok": True}],
        },
    )

    res = _job_to_response(job)

    assert res.experiment_run is None
    assert res.experiment_runs is not None
    assert len(res.experiment_runs) == 1
    assert res.experiment_runs[0].source_id == "repo-1"


def test_job_to_response_projects_typed_operator_interventions():
    job = AgentJob(
        name="Operator Projection",
        goal="Expose intervention history",
        job_type="analysis",
        user_id=uuid4(),
        status=AgentJobStatus.RUNNING.value,
        results={
            "execution_strategy": {
                "operator_interventions": [
                    {
                        "action": "pause",
                        "actor_user_id": "user-1",
                        "at": "2026-03-10T00:00:00Z",
                        "job_status_before": "running",
                        "job_status_after": "paused",
                        "note": "Inspect fallback output",
                        "metadata": {"tool": "python -m pytest -q backend/tests"},
                    }
                ]
            }
        },
    )

    res = _job_to_response(job)

    assert res.operator_interventions is not None
    assert len(res.operator_interventions) == 1
    assert res.operator_interventions[0].action == "pause"
    assert res.operator_interventions[0].actor_user_id == "user-1"
    assert res.operator_interventions[0].job_status_before == "running"
    assert res.operator_interventions[0].job_status_after == "paused"
    assert res.operator_interventions[0].note == "Inspect fallback output"
    assert res.operator_interventions[0].outcome_status == "applied"
    assert res.operator_interventions[0].outcome_reason == "Job remains paused after intervention"
    assert res.operator_interventions[0].metadata == {"tool": "python -m pytest -q backend/tests"}


def test_build_checkpoint_queue_items_includes_approval_recovery_and_follow_up():
    approval_job = AgentJob(
        id=uuid4(),
        name="Approval Required Job",
        goal="Create a document",
        job_type="research",
        user_id=uuid4(),
        status=AgentJobStatus.PAUSED.value,
        phase_details="Waiting on approval",
        results={
            "approval_checkpoint": {
                "message": "Approval required before autonomous action",
                "iteration": 3,
                "action": {"tool": "create_document_from_text"},
            }
        },
    )
    recovery_job = AgentJob(
        id=uuid4(),
        name="Recurring Monitor",
        goal="Monitor for updates",
        job_type="monitor",
        user_id=uuid4(),
        status=AgentJobStatus.FAILED.value,
        schedule_type="continuous",
        error="Network timeout",
        results={
            "execution_strategy": {
                "scheduler_state": {
                    "failure_streak": 2,
                    "backoff_until": "2026-03-16T12:00:00Z",
                    "queue_reason": "execution_failure",
                }
            }
        },
    )
    inbox_item = ResearchInboxItem(
        id=uuid4(),
        user_id=uuid4(),
        item_type="document",
        item_key="doc-1",
        title="Accepted Design Note",
        summary="Worth a deeper follow-up.",
        status="accepted",
    )

    items = _build_checkpoint_queue_items([approval_job, recovery_job], [inbox_item])

    assert [row.item_type for row in items] == [
        "approval_checkpoint",
        "job_recovery",
        "follow_up_recommendation",
    ]
    assert items[0].checkpoint is not None
    assert items[0].checkpoint["action"]["tool"] == "create_document_from_text"
    assert items[0].reason_code == "approval_required"
    assert items[0].reason_label == "Approval required"
    assert items[0].recommended_action == "approve"
    assert items[0].action_count == 4
    assert items[0].priority_score >= 100
    assert items[0].sla_bucket in {"normal", "at_risk", "overdue"}
    assert items[0].escalation_level in {"normal", "medium", "high"}
    assert items[1].scheduler_state is not None
    assert items[1].scheduler_state["queue_reason"] == "execution_failure"
    assert items[1].scheduler_state["last_scheduled_at"] is None
    assert items[1].scheduler_state["last_dispatched_at"] is None
    assert items[1].scheduler_state["current_run_started_at"] is None
    assert items[1].reason_label == "Execution failure"
    assert items[1].recommended_action == "restart"
    assert items[1].backoff_until is not None
    assert items[1].job is not None
    assert items[1].job.scheduler_state is not None
    assert items[1].job.scheduler_state["queue_reason"] == "execution_failure"
    assert items[1].priority_score > items[2].priority_score
    assert items[1].sla_bucket in {"normal", "at_risk", "overdue"}
    assert items[2].reason_label == "Accepted inbox signal"
    assert any(action.chain_create_payload for action in items[2].actions)
    assert any(bool(action.recommended) for action in items[2].actions)
    assert items[2].actions[0].recommendation_key == "deep_dive_chain"
    assert items[2].actions[0].autonomy_eligibility == "auto_launchable"
    assert items[2].actions
    assert items[2].reason_code == "accepted_inbox_item"

    trace_events = _build_decision_trace_from_queue_items(items[:2])
    recovery_event = next(row for row in trace_events if row.decision_type == "job_recovery_queued")
    assert recovery_event.reason_label == "Execution failure"
    assert recovery_event.scheduler_state is not None
    assert recovery_event.scheduler_state["queue_reason"] == "execution_failure"
    assert recovery_event.metadata is not None
    assert recovery_event.metadata["scheduler_state"]["queue_reason"] == "execution_failure"


def test_build_decision_trace_from_opportunities_projects_stateful_events():
    events = _build_decision_trace_from_opportunities(
        source_kind="portfolio",
        source_id="portfolio-1",
        source_label="Scientific Fleet",
        customer=None,
        deep_link_params={"tab": "fleet", "fleetId": "portfolio-1"},
        opportunities=[
            {
                "opportunity_id": "opp-1",
                "canonical_key": "compiler_hotspot",
                "title": "Compiler hotspot",
                "autonomy_state": "blocked_structural",
                "stage": "blocked",
                "last_decision_type": "blocked",
                "last_decision_reason_code": "sandbox_policy_rejected",
                "updated_at": "2026-03-20T00:00:00Z",
            },
            {
                "opportunity_id": "opp-2",
                "canonical_key": "prefetch_gap",
                "title": "Prefetch gap",
                "autonomy_state": "completed_waiting_change",
                "stage": "completed",
                "updated_at": "2026-03-21T00:00:00Z",
            },
        ],
    )

    assert [row.event_type for row in events] == [
        "opportunity_blocked",
        "opportunity_completed_waiting_change",
    ]
    assert events[0].reason_code == "sandbox_policy_rejected"
    assert events[0].deep_link is not None
    assert events[0].deep_link.target_tab == "fleet"
    assert events[0].deep_link.params["fleetId"] == "portfolio-1"
    assert events[0].deep_link.params["opportunityId"] == "opp-1"
    assert events[1].deep_link is not None
    assert events[1].deep_link.params["opportunityId"] == "opp-2"
    assert events[1].status == "completed_waiting_change"


def test_build_decision_trace_from_monitor_snapshot_projects_policy_and_rebalance_events():
    events = _build_decision_trace_from_monitor_snapshot(
        {
            "generated_at": "2026-03-22T12:00:00Z",
            "monitors": [
                {
                    "monitor_job_id": "job-monitor-1",
                    "monitor_name": "Acme Monitor",
                    "customer": "Acme",
                    "policy_guardrail_status": "active",
                    "policy_guardrail_state": "active",
                    "policy_guardrail_action": "rollback",
                    "policy_guardrail_reasons": ["Outcomes degraded"],
                    "policy_guardrail_target_policy": {"follow_up_review_mode": "manual_only"},
                    "latest_policy_changed_at": "2026-03-22T11:30:00Z",
                    "budget_clamp_state": "customer_backlog",
                    "budget_clamp_reasons": ["Queue backlog full"],
                    "latest_budget_changed_at": "2026-03-22T11:45:00Z",
                    "recent_policy_history": [
                        {
                            "id": "hist-1",
                            "at": "2026-03-22T10:00:00Z",
                            "change_source": "rollback",
                            "change_reason": "Rollback triggered",
                            "previous_effective_policy": {"follow_up_review_mode": "queue_for_approval"},
                            "next_effective_policy": {"follow_up_review_mode": "manual_only"},
                        }
                    ],
                }
            ],
            "customers": [
                {
                    "customer": "Acme",
                    "recent_rebalance_history": [
                        {
                            "id": "rebalance-1",
                            "at": "2026-03-22T09:00:00Z",
                            "change_source": "customer_rebalance_guidance",
                            "change_reason": "Reduce queue pressure",
                            "before_capacity": {"queue_backlog_cap": 10},
                            "after_capacity": {"queue_backlog_cap": 6},
                            "evaluation_status": "stable",
                        }
                    ],
                }
            ],
        }
    )

    decision_types = {row.decision_type for row in events}
    assert "policy_rollback" in decision_types
    assert "policy_guardrail_triggered" in decision_types
    assert "budget_clamped" in decision_types
    assert "customer_rebalanced" in decision_types


def test_build_decision_trace_from_domain_opportunities_includes_profile_and_opportunity_targets():
    events = _build_decision_trace_from_opportunities(
        source_kind="domain_profile",
        source_id="profile-1",
        source_label="Compiler Frontier",
        customer=None,
        deep_link_params={"tab": "domain"},
        domain="Compiler",
        objective="Track compiler opportunities",
        track_type="compiler",
        source_scope="kb_plus_arxiv_plus_repo",
        repo_source_ids=["repo-source-1"],
        benchmark_queries=["llvm-test-suite"],
        sandbox_profile_id="scientific-compiler-sandbox",
        automation_profile="balanced",
        effective_policy={"follow_up_review_mode": "queue_for_approval"},
        opportunities=[
            {
                "opportunity_id": "opp-domain-1",
                "canonical_key": "compiler_hotspot",
                "title": "Compiler hotspot",
                "follow_up_review_status": "pending_approval",
                "stage": "planned",
                "updated_at": "2026-03-20T00:00:00Z",
                "confidence": 0.82,
                "readiness": 0.77,
                "linked_experiment_plan_ids": ["plan-1"],
                "linked_validation_run_ids": ["run-1"],
                "child_job_ids": ["job-follow-up-1"],
            },
        ],
    )

    assert len(events) == 1
    assert events[0].event_type == "follow_up_queued"
    assert events[0].deep_link is not None
    assert events[0].deep_link.target_tab == "domain"
    assert events[0].deep_link.params["profileId"] == "profile-1"
    assert events[0].deep_link.params["opportunityId"] == "opp-domain-1"
    assert events[0].domain == "Compiler"
    assert events[0].objective == "Track compiler opportunities"
    assert events[0].track_type == "compiler"
    assert events[0].source_scope == "kb_plus_arxiv_plus_repo"
    assert events[0].repo_source_ids == ["repo-source-1"]
    assert events[0].benchmark_queries == ["llvm-test-suite"]
    assert events[0].sandbox_profile_id == "scientific-compiler-sandbox"
    assert events[0].automation_profile == "balanced"
    assert events[0].effective_policy == {"follow_up_review_mode": "queue_for_approval"}
    assert events[0].confidence == 0.82
    assert events[0].readiness == 0.77
    assert events[0].linked_experiment_plan_ids == ["plan-1"]
    assert events[0].linked_validation_run_ids == ["run-1"]
    assert events[0].child_job_ids == ["job-follow-up-1"]


def test_build_decision_trace_from_opportunities_projects_follow_up_terminal_outcomes():
    events = _build_decision_trace_from_opportunities(
        source_kind="portfolio",
        source_id="portfolio-1",
        source_label="Scientific Fleet",
        customer=None,
        deep_link_params={"tab": "fleet", "fleetId": "portfolio-1"},
        opportunities=[
            {
                "opportunity_id": "opp-complete",
                "canonical_key": "compiler_hotspot",
                "title": "Compiler hotspot",
                "follow_up_outcome_status": "completed",
                "follow_up_outcome_recorded_at": "2026-03-22T10:00:00Z",
                "follow_up_outcome_summary": "Validated the hotspot and documented next steps.",
                "follow_up_last_job_id": "job-follow-up-1",
                "last_decision_type": "follow_up_completed",
                "last_decision_reason_code": "follow_up_completed",
                "stage": "completed",
            },
            {
                "opportunity_id": "opp-failed",
                "canonical_key": "prefetch_gap",
                "title": "Prefetch gap",
                "follow_up_outcome_status": "failed",
                "follow_up_outcome_recorded_at": "2026-03-23T10:00:00Z",
                "follow_up_outcome_summary": "Benchmark verification failed.",
                "follow_up_last_job_id": "job-follow-up-2",
                "last_decision_type": "follow_up_failed",
                "last_decision_reason_code": "follow_up_failed",
                "stage": "blocked",
            },
        ],
    )

    assert [row.event_type for row in events] == [
        "follow_up_completed",
        "follow_up_failed",
    ]
    assert events[0].status == "completed"
    assert events[0].metadata is not None
    assert events[0].metadata["follow_up_outcome_summary"] == "Validated the hotspot and documented next steps."
    assert events[1].status == "failed"
    assert events[1].after_state is not None
    assert events[1].after_state["follow_up_last_job_id"] == "job-follow-up-2"


def test_build_decision_trace_from_opportunities_prefers_relaunched_active_state_over_stale_outcome():
    events = _build_decision_trace_from_opportunities(
        source_kind="domain_profile",
        source_id="profile-1",
        source_label="Compiler Frontier",
        customer=None,
        deep_link_params={"tab": "domain", "profileId": "profile-1"},
        opportunities=[
            {
                "opportunity_id": "opp-relaunched",
                "canonical_key": "compiler_hotspot",
                "title": "Compiler hotspot",
                "follow_up_outcome_status": None,
                "follow_up_outcome_recorded_at": None,
                "follow_up_outcome_summary": None,
                "follow_up_last_job_id": "job-follow-up-3",
                "follow_up_launched_at": "2026-03-24T10:00:00Z",
                "last_decision_type": "follow_up_launched",
                "last_decision_reason_code": "follow_up_relaunched",
                "autonomy_state": "active",
                "stage": "accepted",
            },
        ],
    )

    assert len(events) == 1
    assert events[0].event_type == "follow_up_launched"
    assert events[0].reason_code == "follow_up_relaunched"
    assert events[0].status == "active"
    assert events[0].after_state is not None
    assert events[0].after_state["follow_up_last_job_id"] == "job-follow-up-3"


def test_build_decision_trace_from_validation_runs_projects_compiler_context_and_exact_links():
    run = ExperimentRun(
        id=uuid4(),
        user_id=uuid4(),
        experiment_plan_id=uuid4(),
        agent_job_id=uuid4(),
        name="Validation Run: Compiler hotspot",
        status="blocked",
        progress=100,
        created_at=datetime(2026, 3, 24, 10, 0, 0),
        updated_at=datetime(2026, 3, 24, 10, 5, 0),
        config={
            "execution_handoff": {
                "autonomous_origin": {
                    "source_kind": "profile",
                    "source_id": "profile-1",
                    "opportunity_id": "opp-compiler-1",
                }
            },
            "scientific_validation": {
                "blocked_reason_code": "sandbox_policy_rejected",
                "domain_research_profile_id": "profile-1",
                "hypothesis_id": "opp-compiler-1",
                "track_type": "compiler",
                "sandbox_profile_id": "scientific-compiler-sandbox",
                "benchmark_queries": ["llvm-test-suite"],
                "automation_profile": "balanced",
                "effective_policy": {"follow_up_review_mode": "queue_for_approval"},
                "confidence": 0.81,
                "readiness": 0.79,
                "profile_snapshot": {"domain": "Compiler"},
            },
        },
        summary="Compiler validation blocked by sandbox policy",
    )

    events = agent_jobs_endpoint._build_decision_trace_from_validation_runs([run])

    assert len(events) == 1
    assert events[0].event_type == "validation_blocked"
    assert events[0].deep_link is not None
    assert events[0].deep_link.target_tab == "domain"
    assert events[0].deep_link.params["profileId"] == "profile-1"
    assert events[0].deep_link.params["opportunityId"] == "opp-compiler-1"
    assert events[0].linked_experiment_plan_ids == [str(run.experiment_plan_id)]
    assert events[0].linked_validation_run_ids == [str(run.id)]
    assert events[0].child_job_ids == [str(run.agent_job_id)]
    assert events[0].track_type == "compiler"
    assert events[0].domain == "Compiler"
    assert events[0].benchmark_queries == ["llvm-test-suite"]
    assert events[0].sandbox_profile_id == "scientific-compiler-sandbox"
    assert events[0].automation_profile == "balanced"
    assert events[0].effective_policy == {"follow_up_review_mode": "queue_for_approval"}
    assert events[0].confidence == 0.81
    assert events[0].readiness == 0.79


def test_build_decision_trace_from_job_projects_scheduler_recovery_metadata():
    job = AgentJob(
        id=uuid4(),
        name="Recurring Monitor",
        goal="Monitor for updates",
        job_type="monitor",
        user_id=uuid4(),
        status=AgentJobStatus.FAILED.value,
        created_at=datetime(2026, 3, 16, 9, 0, 0),
        results={
            "execution_strategy": {
                "scheduler_state": {
                    "queue_reason": "execution_failure",
                    "backoff_until": "not-a-datetime",
                }
            }
        },
    )

    events = _build_decision_trace_from_job(job)

    assert len(events) == 1
    event = events[0]
    assert event.event_type == "job_recovery_queued"
    assert event.reason_code == "execution_failure"
    assert event.reason_label == "Execution failure"
    assert event.scheduler_state is not None
    assert event.scheduler_state["queue_reason"] == "execution_failure"
    assert event.event_time == datetime(2026, 3, 16, 9, 0, 0)
    assert event.metadata is not None
    assert event.metadata["reason_label"] == "Execution failure"
    assert event.metadata["scheduler_state"]["queue_reason"] == "execution_failure"
    assert event.metadata["scheduler_state"]["backoff_until"] == "not-a-datetime"


def test_build_checkpoint_queue_items_skips_auto_launched_follow_ups_and_marks_blocked_items():
    blocked_item = ResearchInboxItem(
        id=uuid4(),
        user_id=uuid4(),
        item_type="document",
        item_key="doc-2",
        title="Blocked follow-up",
        summary="Needs manual review.",
        status="accepted",
        follow_up_decision="manual",
        follow_up_policy_mode="auto_launch_safe",
        follow_up_launch_status="blocked",
        follow_up_block_reason="Recommendation is not allowlisted by this monitor policy.",
        follow_up_recommendation_key="repo_patch_chain",
    )
    launched_item = ResearchInboxItem(
        id=uuid4(),
        user_id=uuid4(),
        item_type="document",
        item_key="doc-3",
        title="Already launched",
        status="accepted",
        follow_up_decision="auto_launched",
        follow_up_policy_mode="auto_launch_safe",
        follow_up_launch_status="launched",
    )

    items = _build_checkpoint_queue_items([], [blocked_item, launched_item])

    assert len(items) == 1
    assert items[0].title == "Blocked follow-up"
    assert items[0].reason_code == "follow_up_blocked"
    assert items[0].reason_label == "Follow-up blocked by policy"
    assert items[0].follow_up_policy_mode == "auto_launch_safe"
    assert items[0].follow_up_launch_status == "blocked"
    assert items[0].follow_up_block_reason == "Recommendation is not allowlisted by this monitor policy."


def test_build_checkpoint_queue_items_projects_pending_follow_up_approval_actions():
    pending_item = ResearchInboxItem(
        id=uuid4(),
        user_id=uuid4(),
        item_type="document",
        item_key="doc-4",
        title="Pending approval follow-up",
        summary="Queued safe follow-up.",
        status="accepted",
        follow_up_decision="queued_for_approval",
        follow_up_policy_mode="queue_for_approval",
        follow_up_launch_status="pending_approval",
        follow_up_recommendation_key="deep_dive_chain",
    )

    items = _build_checkpoint_queue_items([], [pending_item])

    assert len(items) == 1
    assert items[0].reason_code == "follow_up_launch_approval"
    assert items[0].reason_label == "Follow-up launch approval"
    assert [action.action for action in items[0].actions] == ["approve_launch", "reject_launch"]
    assert items[0].actions[0].follow_up_action_payload == {"inbox_item_id": str(pending_item.id)}


def test_build_checkpoint_queue_items_projects_policy_review_guardrail():
    monitor_id = uuid4()
    job = AgentJob(
        id=monitor_id,
        name="Beta Watch",
        goal="Monitor Beta updates",
        job_type="monitor",
        user_id=uuid4(),
        status=AgentJobStatus.COMPLETED.value,
        progress=100,
        iteration=1,
        max_iterations=1,
        max_tool_calls=0,
        max_llm_calls=0,
        max_runtime_minutes=5,
        tool_calls_used=0,
        llm_calls_used=0,
        tokens_used=0,
        error_count=0,
        created_at=datetime(2026, 3, 18, 9, 0),
        chain_depth=0,
        chain_triggered=False,
    )

    items = _build_checkpoint_queue_items(
        [job],
        [],
        monitor_health_rows=[
            {
                "monitor_job_id": monitor_id,
                "monitor_name": "Beta Watch",
                "customer": "Beta",
                "latest_policy_changed_at": datetime(2026, 3, 18, 10, 0),
                "policy_guardrail_status": "active",
                "policy_guardrail_action": "rollback",
                "budget_throttle_state": "normal",
                "policy_guardrail_reasons": ["More accepted items are getting blocked by policy"],
                "policy_guardrail_target_history_entry_id": "history-2",
                "policy_guardrail_follow_up_autonomy": {
                    "mode": "queue_for_approval",
                    "allowed_recommendations": ["deep_dive_chain", "single_research_job"],
                },
                "policy_guardrail_target_policy": {
                    "follow_up_review_mode": "queue_for_approval",
                    "allowed_recommendations": ["deep_dive_chain", "single_research_job"],
                },
            }
        ],
    )

    assert len(items) == 1
    assert items[0].item_type == "policy_review"
    assert items[0].policy_guardrail_action == "rollback"
    assert items[0].reason_label == "Policy safeguard review"
    assert items[0].policy_guardrail_target_policy is not None
    assert items[0].policy_guardrail_target_policy["follow_up_review_mode"] == "queue_for_approval"
    assert items[0].actions[0].action == "apply_guardrail"
    assert items[0].actions[0].policy_rollback_payload == {"history_entry_id": "history-2"}


def test_build_checkpoint_queue_items_projects_portfolio_follow_up_approval():
    portfolio_id = uuid4()
    portfolio = ResearchPortfolio(
        id=portfolio_id,
        user_id=uuid4(),
        title="Scientific Fleet",
        objective="Validate promising opportunities",
        status="running",
        automation_profile="max_autonomy",
        automation_policy={
            "follow_up_review_mode": "queue_for_approval",
            "auto_launch_follow_up": True,
        },
        opportunities=[
            {
                "opportunity_id": "opp-follow-up",
                "canonical_key": "cache_hotspot",
                "title": "Cache hotspot",
                "hypothesis": "L2 reuse is regressing",
                "autonomy_state": "eligible",
                "decision_state": "accepted",
                "stage": "accepted",
                "evidence_revision": "rev-1",
                "track_type": "compiler",
                "confidence": 0.81,
                "readiness": 0.73,
                "source_repo_ids": ["repo-1"],
            }
        ],
        sandbox_profile_id="scientific-compiler-sandbox",
        latest_summary={"portfolio_config_revision": "cfg-1"},
        latest_note_ids=["note-1"],
        latest_experiment_plan_ids=["plan-1"],
        latest_validation_run_ids=["run-1"],
    )

    items = _build_checkpoint_queue_items([], [], [portfolio])

    assert len(items) == 1
    assert items[0].item_type == "follow_up_recommendation"
    assert str(items[0].portfolio_id) == str(portfolio_id)
    assert items[0].portfolio_title == "Scientific Fleet"
    assert items[0].portfolio_opportunity_id == "opp-follow-up"
    assert items[0].objective == "Validate promising opportunities"
    assert items[0].track_type == "compiler"
    assert items[0].source_scope == "kb_plus_arxiv_plus_repo"
    assert items[0].repo_source_ids == ["repo-1"]
    assert items[0].sandbox_profile_id == "scientific-compiler-sandbox"
    assert items[0].automation_profile == "max_autonomy"
    assert items[0].effective_policy["follow_up_review_mode"] == "queue_for_approval"
    assert items[0].confidence == 0.81
    assert items[0].readiness == 0.73
    assert items[0].linked_note_ids == ["note-1"]
    assert items[0].linked_experiment_plan_ids == ["plan-1"]
    assert items[0].linked_validation_run_ids == ["run-1"]
    assert items[0].child_job_ids is None
    assert items[0].actions[0].follow_up_action_payload == {
        "portfolio_id": str(portfolio_id),
        "portfolio_opportunity_id": "opp-follow-up",
    }


def test_build_checkpoint_queue_items_projects_portfolio_policy_and_budget_reviews():
    portfolio_id = uuid4()
    portfolio = ResearchPortfolio(
        id=portfolio_id,
        user_id=uuid4(),
        title="Scientific Fleet",
        objective="Validate promising opportunities",
        status="running",
        automation_policy={},
        opportunities=[
            {
                "opportunity_id": "opp-policy",
                "canonical_key": "sandbox_gap",
                "title": "Sandbox gap",
                "hypothesis": "Missing capability blocks validation",
                "autonomy_state": "blocked_structural",
                "stage": "blocked",
                "last_blocked_reason_code": "missing_required_capability",
                "evidence_revision": "rev-policy",
            },
            {
                "opportunity_id": "opp-budget",
                "canonical_key": "budget_stop",
                "title": "Budget stop",
                "hypothesis": "Budget clamp blocks validation",
                "autonomy_state": "blocked_structural",
                "stage": "blocked",
                "last_blocked_reason_code": "budget_exhausted",
                "evidence_revision": "rev-budget",
            },
        ],
        latest_summary={"portfolio_config_revision": "cfg-2"},
    )

    items = _build_checkpoint_queue_items([], [], [portfolio])

    assert [item.item_type for item in items] == ["policy_review", "budget_review"]
    assert all(action.action == "open_fleet" for item in items for action in item.actions)


def test_build_checkpoint_queue_items_projects_profile_follow_up_approval():
    profile_id = uuid4()
    profile = DomainResearchProfile(
        id=profile_id,
        user_id=uuid4(),
        title="Compiler Frontier",
        domain="Compiler",
        objective="Validate compiler opportunities",
        status="running",
        automation_profile="max_autonomy",
        automation_policy={
            "follow_up_review_mode": "queue_for_approval",
            "auto_launch_follow_up": True,
        },
        latest_summary={
            "profile_config_revision": "cfg-profile-1",
            "opportunities": [
                {
                    "opportunity_id": "opp-follow-up",
                    "canonical_key": "compiler_hotspot",
                    "title": "Compiler hotspot",
                    "hypothesis": "Pass ordering regressed",
                    "autonomy_state": "eligible",
                    "decision_state": "accepted",
                    "stage": "accepted",
                    "evidence_revision": "rev-1",
                    "confidence": 0.84,
                    "readiness": 0.78,
                }
            ],
        },
        source_scope="kb_plus_arxiv_plus_repo",
        track_type="compiler",
        repo_source_ids=["repo-source-1"],
        benchmark_queries=["llvm-test-suite"],
        sandbox_profile_id="scientific-compiler-sandbox",
        latest_note_ids=["note-compiler-1"],
        latest_experiment_plan_ids=["plan-compiler-1"],
        latest_validation_run_ids=["run-compiler-1"],
    )

    items = _build_checkpoint_queue_items([], [], [], [profile])

    assert len(items) == 1
    assert items[0].item_type == "follow_up_recommendation"
    assert str(items[0].domain_research_profile_id) == str(profile_id)
    assert items[0].domain_research_profile_title == "Compiler Frontier"
    assert items[0].profile_opportunity_id == "opp-follow-up"
    assert items[0].domain == "Compiler"
    assert items[0].objective == "Validate compiler opportunities"
    assert items[0].track_type == "compiler"
    assert items[0].source_scope == "kb_plus_arxiv_plus_repo"
    assert items[0].repo_source_ids == ["repo-source-1"]
    assert items[0].benchmark_queries == ["llvm-test-suite"]
    assert items[0].sandbox_profile_id == "scientific-compiler-sandbox"
    assert items[0].automation_profile == "max_autonomy"
    assert items[0].effective_policy["follow_up_review_mode"] == "queue_for_approval"
    assert items[0].confidence == 0.84
    assert items[0].readiness == 0.78
    assert items[0].linked_note_ids == ["note-compiler-1"]
    assert items[0].linked_experiment_plan_ids == ["plan-compiler-1"]
    assert items[0].linked_validation_run_ids == ["run-compiler-1"]
    assert items[0].child_job_ids is None
    assert items[0].actions[0].follow_up_action_payload == {
        "domain_research_profile_id": str(profile_id),
        "profile_opportunity_id": "opp-follow-up",
    }


def test_build_checkpoint_queue_items_projects_profile_policy_and_budget_reviews():
    profile_id = uuid4()
    profile = DomainResearchProfile(
        id=profile_id,
        user_id=uuid4(),
        title="Compiler Frontier",
        domain="Compiler",
        objective="Validate compiler opportunities",
        status="running",
        latest_summary={
            "profile_config_revision": "cfg-profile-2",
            "opportunities": [
                {
                    "opportunity_id": "opp-policy",
                    "canonical_key": "sandbox_gap",
                    "title": "Sandbox gap",
                    "hypothesis": "Missing capability blocks validation",
                    "autonomy_state": "blocked_structural",
                    "stage": "blocked",
                    "last_blocked_reason_code": "missing_required_capability",
                    "evidence_revision": "rev-policy",
                },
                {
                    "opportunity_id": "opp-budget",
                    "canonical_key": "budget_stop",
                    "title": "Budget stop",
                    "hypothesis": "Budget clamp blocks validation",
                    "autonomy_state": "blocked_structural",
                    "stage": "blocked",
                    "last_blocked_reason_code": "budget_exhausted",
                    "evidence_revision": "rev-budget",
                },
            ],
        },
    )

    items = _build_checkpoint_queue_items([], [], [], [profile])

    assert [item.item_type for item in items] == ["policy_review", "budget_review"]
    assert all(action.action == "open_fleet" for item in items for action in item.actions)


def test_follow_up_action_scoring_prefers_learned_single_job_recommendation():
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=uuid4(),
        item_type="document",
        item_key="doc-5",
        title="Latency regression investigation",
        summary="Need a focused document-driven follow-up.",
        status="accepted",
    )

    actions = _build_follow_up_actions_for_inbox_item(
        item,
        learning_profile={
            "token_scores": {"latency": 3, "focused": 2},
            "phrase_scores": {"latency regression": 2},
            "recommendation_scores": {"single_research_job": 7, "deep_dive_chain": 0},
            "source_type_scores": {"document": 2},
            "outcome_counters": {},
        },
    )

    assert actions[0].recommendation_key == "single_research_job"
    assert actions[0].recommended is True
    assert int(actions[0].recommendation_score or 0) > int(actions[1].recommendation_score or 0)
    assert "learned_recommendation:7" in (actions[0].recommendation_reasons or [])


def test_score_follow_up_action_for_item_uses_repo_presence_and_source_bias():
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=uuid4(),
        item_type="arxiv",
        item_key="paper-2",
        title="Paper with code release",
        summary="Runnable implementation available.",
        status="accepted",
        item_metadata={"repos": [{"provider": "github", "repo": "acme/project"}]},
    )
    action = _build_follow_up_actions_for_inbox_item(item)[-1]

    score, reasons = _score_follow_up_action_for_item(
        item,
        action,
        learning_profile={
            "token_scores": {},
            "phrase_scores": {},
            "recommendation_scores": {"repo_patch_chain": 3},
            "source_type_scores": {"arxiv": 2},
            "outcome_counters": {},
        },
    )

    assert score > 0
    assert "repos_present" in reasons
    assert "paper_repo_fit" in reasons


def test_queue_priority_fields_escalate_overdue_and_stale_items():
    now = datetime(2026, 3, 16, 12, 0, 0)

    approval = _queue_priority_fields(
        item_type="approval_checkpoint",
        reason_code="approval_required",
        created_at=datetime(2026, 3, 16, 6, 0, 0),
        next_run_at=None,
        backoff_until=None,
        stale=False,
        now=now,
    )
    recovery = _queue_priority_fields(
        item_type="job_recovery",
        reason_code="stalled_run",
        created_at=datetime(2026, 3, 16, 9, 0, 0),
        next_run_at=datetime(2026, 3, 16, 10, 0, 0),
        backoff_until=datetime(2026, 3, 16, 11, 0, 0),
        stale=True,
        now=now,
    )
    followup = _queue_priority_fields(
        item_type="follow_up_recommendation",
        reason_code="accepted_inbox_item",
        created_at=datetime(2026, 3, 16, 11, 0, 0),
        next_run_at=None,
        backoff_until=None,
        stale=False,
        now=now,
    )

    assert approval["is_overdue"] is True
    assert approval["sla_bucket"] == "overdue"
    assert approval["escalation_level"] == "high"
    assert recovery["is_stale"] is True
    assert recovery["is_overdue"] is True
    assert recovery["priority_score"] > followup["priority_score"]
    assert followup["sla_bucket"] == "normal"


def test_validate_bulk_queue_action_allows_safe_homogeneous_actions():
    _validate_bulk_queue_action("approval_checkpoint", "approve")
    _validate_bulk_queue_action("approval_checkpoint", "reject")
    _validate_bulk_queue_action("job_recovery", "restart")
    _validate_bulk_queue_action("job_recovery", "resume")


def test_validate_bulk_queue_action_rejects_mixed_or_unsupported_actions():
    try:
        _validate_bulk_queue_action("follow_up_recommendation", "approve")
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "only supported for approval_checkpoint and job_recovery" in str(exc.detail)

    try:
        _validate_bulk_queue_action("approval_checkpoint", "restart")
        assert False, "expected HTTPException"
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "not allowed" in str(exc.detail)


def test_job_matches_bulk_queue_item_type_uses_current_queue_eligibility():
    approval_job = AgentJob(
        id=uuid4(),
        name="Approval Required Job",
        goal="Create a document",
        job_type="research",
        user_id=uuid4(),
        status=AgentJobStatus.PAUSED.value,
        results={
            "approval_checkpoint": {
                "message": "Approval required before autonomous action",
                "action": {"tool": "create_document_from_text"},
            }
        },
    )
    recovery_job = AgentJob(
        id=uuid4(),
        name="Recurring Monitor",
        goal="Monitor for updates",
        job_type="monitor",
        user_id=uuid4(),
        status=AgentJobStatus.FAILED.value,
        results={
            "execution_strategy": {
                "scheduler_state": {
                    "queue_reason": "execution_failure",
                }
            }
        },
    )
    non_queue_job = AgentJob(
        id=uuid4(),
        name="Normal Job",
        goal="Nothing queued",
        job_type="analysis",
        user_id=uuid4(),
        status=AgentJobStatus.RUNNING.value,
        results={},
    )

    ok_approval, reason_approval = _job_matches_bulk_queue_item_type(approval_job, "approval_checkpoint")
    ok_recovery, reason_recovery = _job_matches_bulk_queue_item_type(recovery_job, "job_recovery")
    bad_match, bad_reason = _job_matches_bulk_queue_item_type(non_queue_job, "job_recovery")

    assert ok_approval is True
    assert reason_approval is None
    assert ok_recovery is True
    assert reason_recovery is None
    assert bad_match is False
    assert "not currently represented as a recovery queue item" in str(bad_reason)


@pytest.mark.asyncio
async def test_record_job_operator_event_threads_scheduler_state_into_trace_payload(monkeypatch):
    job = AgentJob(
        id=uuid4(),
        user_id=uuid4(),
        name="Recovery Job",
        goal="Recover failed work",
        job_type="research",
        status=AgentJobStatus.FAILED.value,
        results={
            "execution_strategy": {
                "scheduler_state": {
                    "queue_reason": "execution_failure",
                    "last_scheduled_at": "2026-03-16T09:00:00Z",
                    "last_dispatched_at": "2026-03-16T09:05:00Z",
                }
            }
        },
    )
    captured = {}

    async def _fake_record(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr("app.api.endpoints.agent_jobs.record_autonomy_decision_event", _fake_record)

    await _record_job_operator_event(
        db=SimpleNamespace(),
        job=job,
        current_user=SimpleNamespace(id=job.user_id),
        action="restart",
        note="Retry after failure",
        previous_status=AgentJobStatus.FAILED.value,
        next_status=AgentJobStatus.RUNNING.value,
        scheduler_state={
            "queue_reason": "execution_failure",
            "last_scheduled_at": "2026-03-16T09:00:00Z",
            "last_dispatched_at": "2026-03-16T09:05:00Z",
        },
        metadata={"reason_code": "execution_failure"},
        summary="Recovery Job: restart",
    )

    assert captured["reason_label"] == "Execution failure"
    assert captured["scheduler_state"] == {
        "queue_reason": "execution_failure",
        "last_scheduled_at": "2026-03-16T09:00:00Z",
        "last_dispatched_at": "2026-03-16T09:05:00Z",
    }
    assert captured["metadata"] == {"reason_code": "execution_failure"}


@pytest.mark.asyncio
async def test_record_job_operator_event_ignores_malformed_scheduler_state(monkeypatch):
    job = AgentJob(
        id=uuid4(),
        user_id=uuid4(),
        name="Recovery Job",
        goal="Recover failed work",
        job_type="research",
        status=AgentJobStatus.FAILED.value,
    )
    captured = {}

    async def _fake_record(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr("app.api.endpoints.agent_jobs.record_autonomy_decision_event", _fake_record)

    await _record_job_operator_event(
        db=SimpleNamespace(),
        job=job,
        current_user=SimpleNamespace(id=job.user_id),
        action="restart",
        note=None,
        previous_status=AgentJobStatus.FAILED.value,
        next_status=AgentJobStatus.RUNNING.value,
        scheduler_state="bad-payload",
        metadata={"reason_code": "execution_failure"},
        summary="Recovery Job: restart",
    )

    assert captured["reason_label"] == "Execution failure"
    assert captured["scheduler_state"] is None


def test_job_to_response_ignores_invalid_operator_intervention_shapes():
    job = AgentJob(
        name="Operator Projection Invalid",
        goal="Skip malformed intervention payloads",
        job_type="analysis",
        user_id=uuid4(),
        status=AgentJobStatus.RUNNING.value,
        results={
            "execution_strategy": {
                "operator_interventions": [
                    "bad-payload",
                    {"action": "resume", "job_status_before": "paused", "job_status_after": "running"},
                ]
            }
        },
    )

    res = _job_to_response(job)

    assert res.operator_interventions is not None
    assert len(res.operator_interventions) == 1
    assert res.operator_interventions[0].action == "resume"
    assert res.operator_interventions[0].job_status_before == "paused"
    assert res.operator_interventions[0].job_status_after == "running"
    assert res.operator_interventions[0].outcome_status == "pending"
    assert res.operator_interventions[0].outcome_reason == "Awaiting job outcome"


def test_job_to_response_projects_scheduler_state_metadata():
    job = AgentJob(
        name="Scheduler Projection",
        goal="Expose scheduler state",
        job_type="monitor",
        user_id=uuid4(),
        status=AgentJobStatus.FAILED.value,
        schedule_type="continuous",
        results={
            "execution_strategy": {
                "scheduler_state": {
                    "last_run_status": " failed ",
                    "failure_streak": "3",
                    "last_scheduled_at": "2026-03-16T09:00:00Z",
                    "last_dispatched_at": "2026-03-16T09:05:00Z",
                    "current_run_started_at": "2026-03-16T09:06:00Z",
                    "last_successful_run_at": "2026-03-16T08:00:00Z",
                    "last_completed_run_at": "2026-03-16T08:01:00Z",
                    "last_failure_at": "2026-03-16T09:10:00Z",
                    "backoff_until": "2026-03-16T10:00:00Z",
                    "backoff_seconds": "1800",
                    "queue_reason": " execution_failure ",
                }
            }
        },
    )

    res = _job_to_response(job)

    assert res.scheduler_state is not None
    assert res.scheduler_state["last_run_status"] == AgentJobStatus.FAILED.value
    assert res.scheduler_state["failure_streak"] == 3
    assert res.scheduler_state["last_scheduled_at"] == "2026-03-16T09:00:00Z"
    assert res.scheduler_state["last_dispatched_at"] == "2026-03-16T09:05:00Z"
    assert res.scheduler_state["current_run_started_at"] == "2026-03-16T09:06:00Z"
    assert res.scheduler_state["last_successful_run_at"] == "2026-03-16T08:00:00Z"
    assert res.scheduler_state["last_completed_run_at"] == "2026-03-16T08:01:00Z"
    assert res.scheduler_state["last_failure_at"] == "2026-03-16T09:10:00Z"
    assert res.scheduler_state["backoff_until"] == "2026-03-16T10:00:00Z"
    assert res.scheduler_state["backoff_seconds"] == 1800
    assert res.scheduler_state["queue_reason"] == "execution_failure"


def test_job_to_response_omits_scheduler_state_for_malformed_payloads():
    job = AgentJob(
        name="Malformed Scheduler Projection",
        goal="Skip malformed scheduler payloads",
        job_type="analysis",
        user_id=uuid4(),
        status=AgentJobStatus.RUNNING.value,
        results={
            "execution_strategy": {
                "scheduler_state": "bad-payload",
            }
        },
    )

    res = _job_to_response(job)

    assert res.scheduler_state is None


def test_build_extract_job_memories_response_includes_stats_and_memories():
    job_id = uuid4()
    memories = [
        SimpleNamespace(
            id=uuid4(),
            memory_type="lesson",
            content="Prefer small validation loops before full execution.",
            importance_score=0.8,
            tags=["lesson", "outcome:success"],
        )
    ]

    payload = _build_extract_job_memories_response(
        job_id=job_id,
        memories=memories,
        extraction_stats={
            "parsed_count": 4,
            "candidate_count": 3,
            "skipped_duplicates": 2,
            "is_relaunch_chain": True,
            "relaunch_root_job_id": str(uuid4()),
        },
    )

    assert payload.job_id == str(job_id)
    assert payload.memories_created == 1
    assert payload.parsed_count == 4
    assert payload.candidate_count == 3
    assert payload.skipped_duplicates == 2
    assert payload.is_relaunch_chain is True
    assert payload.relaunch_root_job_id is not None
    assert len(payload.memories) == 1
    assert payload.memories[0].type == "lesson"


def test_build_extract_job_memories_response_defaults_stats_when_missing():
    job_id = uuid4()
    payload = _build_extract_job_memories_response(
        job_id=job_id,
        memories=[],
        extraction_stats=None,
    )

    assert payload.job_id == str(job_id)
    assert payload.memories_created == 0
    assert payload.parsed_count == 0
    assert payload.candidate_count == 0
    assert payload.skipped_duplicates == 0
    assert payload.is_relaunch_chain is False
    assert payload.relaunch_root_job_id is None
    assert payload.memories == []


def test_build_extract_job_memories_response_handles_invalid_numeric_fields():
    job_id = uuid4()
    payload = _build_extract_job_memories_response(
        job_id=job_id,
        memories=[
            SimpleNamespace(
                id=uuid4(),
                memory_type="insight",
                content="Keep tool output snippets concise.",
                importance_score="bad-float",
                tags="not-a-list",
            )
        ],
        extraction_stats={
            "parsed_count": "x",
            "candidate_count": object(),
            "skipped_duplicates": "oops",
            "is_relaunch_chain": 1,
        },
    )

    assert payload.parsed_count == 0
    assert payload.candidate_count == 0
    assert payload.skipped_duplicates == 0
    assert payload.is_relaunch_chain is True
    assert len(payload.memories) == 1
    assert payload.memories[0].importance_score == 0.0
    assert payload.memories[0].tags == []


def test_build_extract_job_memories_response_normalizes_tag_lists():
    payload = _build_extract_job_memories_response(
        job_id=uuid4(),
        memories=[
            SimpleNamespace(
                id=uuid4(),
                memory_type="pattern",
                content="retry clusters",
                importance_score=0.5,
                tags=[" pattern ", "", None, 42],
            )
        ],
        extraction_stats={},
    )

    assert payload.memories[0].tags == ["pattern", "42"]


def test_build_extract_job_memories_response_normalizes_missing_type_and_content():
    payload = _build_extract_job_memories_response(
        job_id=uuid4(),
        memories=[
            SimpleNamespace(
                id=uuid4(),
                memory_type=None,
                content=None,
                importance_score=0.4,
                tags=[],
            )
        ],
        extraction_stats={},
    )

    assert payload.memories[0].type == "unknown"
    assert payload.memories[0].content == ""


def test_build_job_memory_response_serializes_optional_fields():
    job_id = uuid4()
    created_at = datetime.utcnow()
    payload = _build_job_memory_response(
        job_id=job_id,
        memory=SimpleNamespace(
            id=uuid4(),
            memory_type="insight",
            content="Cache expensive prompts by query hash.",
            importance_score=0.77,
            tags=["insight", "cache"],
            context={"project_scope": "backend"},
            access_count=4,
            created_at=created_at,
        ),
    )

    assert payload.job_id == str(job_id)
    assert payload.type == "insight"
    assert payload.tags == ["insight", "cache"]
    assert payload.context == {"project_scope": "backend"}
    assert payload.access_count == 4
    assert payload.created_at == created_at.isoformat()


def test_build_job_memory_response_handles_invalid_numbers():
    payload = _build_job_memory_response(
        job_id=uuid4(),
        memory=SimpleNamespace(
            id=uuid4(),
            memory_type="lesson",
            content="Prefer idempotent job actions.",
            importance_score="invalid",
            tags="invalid-tags",
            context=None,
            access_count="invalid",
            created_at=None,
        ),
    )

    assert payload.importance_score == 0.0
    assert payload.access_count == 0
    assert payload.tags == []


def test_build_job_memory_response_normalizes_missing_type_and_content():
    payload = _build_job_memory_response(
        job_id=uuid4(),
        memory=SimpleNamespace(
            id=uuid4(),
            memory_type=None,
            content=None,
            importance_score=0.2,
            tags=[],
            context={},
            access_count=0,
            created_at=None,
        ),
    )

    assert payload.type == "unknown"
    assert payload.content == ""


def test_build_job_memories_list_response_uses_builder_for_each_memory():
    job_id = uuid4()
    memories = [
        SimpleNamespace(
            id=uuid4(),
            memory_type="lesson",
            content="Start with a 5-minute dry run.",
            importance_score=0.85,
            tags=["lesson"],
            context=None,
            access_count=0,
            created_at=None,
        ),
        SimpleNamespace(
            id=uuid4(),
            memory_type="pattern",
            content="Retry failures cluster by endpoint.",
            importance_score=0.66,
            tags=["pattern", "retry"],
            context={"execution_outcome": "failed"},
            access_count=2,
            created_at=None,
        ),
    ]

    payload = _build_job_memories_list_response(job_id=job_id, memories=memories)

    assert payload.job_id == str(job_id)
    assert payload.total == 2
    assert len(payload.memories) == 2
    assert payload.memories[0].job_id == str(job_id)
    assert payload.memories[1].type == "pattern"
    assert payload.memories[1].access_count == 2


def test_build_memory_search_response_serializes_job_links_and_totals():
    query = "retry"
    source_job_id = uuid4()
    payload = _build_memory_search_response(
        query=query,
        memories=[
            SimpleNamespace(
                id=uuid4(),
                memory_type="finding",
                content="Retries improved throughput by 23%.",
                importance_score=0.72,
                tags=["finding", "metric"],
                job_id=source_job_id,
                access_count=3,
                created_at=None,
            ),
            SimpleNamespace(
                id=uuid4(),
                memory_type="context",
                content="Endpoint timeout increased during deploy window.",
                importance_score=0.41,
                tags=None,
                job_id=None,
                access_count=0,
                created_at=None,
            ),
        ],
    )

    assert payload.query == query
    assert payload.total == 2
    assert len(payload.memories) == 2
    assert payload.memories[0].job_id == str(source_job_id)
    assert payload.memories[1].job_id is None
    assert payload.memories[1].tags == []


def test_build_memory_search_response_normalizes_missing_type_and_content():
    payload = _build_memory_search_response(
        query="q",
        memories=[
            SimpleNamespace(
                id=uuid4(),
                memory_type=None,
                content=None,
                importance_score=0.5,
                tags=[],
                job_id=None,
                access_count=0,
                created_at=None,
            )
        ],
    )

    assert payload.memories[0].type == "unknown"
    assert payload.memories[0].content == ""


def test_build_memory_stats_response_normalizes_counts_and_rows():
    payload = _build_memory_stats_response(
        stats={
            "total_memories": "7",
            "by_type": {"lesson": "2", "finding": 3, "": 99},
            "job_sourced": "4",
            "chat_sourced": 1,
            "manual": "2",
            "most_accessed": [
                {"id": uuid4(), "type": "lesson", "content": "dry run first", "access_count": "5"},
                "invalid",
            ],
            "most_important": [
                {"id": uuid4(), "type": "finding", "content": "throughput +23%", "importance": "0.88"},
            ],
        }
    )

    assert payload.total_memories == 7
    assert payload.by_type == {"lesson": 2, "finding": 3}
    assert payload.job_sourced == 4
    assert payload.chat_sourced == 1
    assert payload.manual == 2
    assert len(payload.most_accessed) == 1
    assert payload.most_accessed[0].access_count == 5
    assert len(payload.most_important) == 1
    assert payload.most_important[0].importance == 0.88


def test_build_memory_stats_response_handles_invalid_numeric_values():
    payload = _build_memory_stats_response(
        stats={
            "total_memories": "nope",
            "by_type": {"lesson": "bad"},
            "job_sourced": object(),
            "chat_sourced": "bad",
            "manual": None,
            "most_accessed": [{"id": "m1", "type": "lesson", "content": "x", "access_count": "NaN"}],
            "most_important": [{"id": "m2", "type": "finding", "content": "y", "importance": "NaN"}],
        }
    )

    assert payload.total_memories == 0
    assert payload.by_type == {"lesson": 0}
    assert payload.job_sourced == 0
    assert payload.chat_sourced == 0
    assert payload.manual == 0
    assert payload.most_accessed[0].access_count == 0
    assert payload.most_important[0].importance == 0.0


def test_build_memory_graph_response_normalizes_nodes_edges_and_job_id():
    payload = _build_memory_graph_response(
        graph={
            "job_id": "  ",
            "nodes": [
                {
                    "id": uuid4(),
                    "type": "pattern",
                    "content": "retry clusters at deploy time",
                    "importance_score": "0.91",
                    "tags": ["pattern", 42],
                    "job_id": uuid4(),
                    "access_count": "3",
                },
                "skip",
            ],
            "edges": [
                {
                    "source": "n1",
                    "target": "n2",
                    "weight": "1.25",
                    "reasons": ["shared:retry", 7],
                },
                None,
            ],
            "stats": {"memory_count": 2, "": "drop"},
        },
        job_id="job-123",
    )

    assert payload.job_id == "job-123"
    assert len(payload.nodes) == 1
    assert payload.nodes[0].importance_score == 0.91
    assert payload.nodes[0].tags == ["pattern", "42"]
    assert payload.nodes[0].access_count == 3
    assert len(payload.edges) == 1
    assert payload.edges[0].weight == 1.25
    assert payload.edges[0].reasons == ["shared:retry", "7"]
    assert payload.stats == {"memory_count": 2}


def test_build_memory_graph_response_handles_invalid_numeric_values():
    payload = _build_memory_graph_response(
        graph={
            "nodes": [
                {
                    "id": "n1",
                    "type": "finding",
                    "content": "x",
                    "importance_score": "bad",
                    "access_count": "bad",
                    "tags": "not-list",
                }
            ],
            "edges": [
                {
                    "source": "n1",
                    "target": "n2",
                    "weight": "bad",
                    "reasons": "not-list",
                }
            ],
            "stats": {"memory_count": 1},
        }
    )

    assert len(payload.nodes) == 1
    assert payload.nodes[0].importance_score == 0.0
    assert payload.nodes[0].access_count == 0
    assert payload.nodes[0].tags == []
    assert len(payload.edges) == 1
    assert payload.edges[0].weight == 0.0
    assert payload.edges[0].reasons == []


@pytest.mark.asyncio
async def test_decision_trace_endpoint_filters_triage_and_actionable_only(db_session, test_user):
    event = await record_autonomy_decision_event(
        db_session,
        user_id=test_user.id,
        event_type="job_recovery_queued",
        source_kind="job",
        source_id="job-1",
        source_label="Recovery Job",
        decision_type="job_recovery_queued",
        summary="Recovery queued",
        severity="high",
    )
    event.triage_status = "investigating"
    event.pinned = True
    await db_session.commit()

    response = await get_decision_trace(
        source_kind=None,
        decision_type=None,
        customer=None,
        status=None,
        severity=None,
        actor_mode=None,
        triage_status="investigating",
        assigned_to_user_id=None,
        unassigned_only=False,
        escalation_state=None,
        pinned=True,
        actionable_only=True,
        start_at=None,
        end_at=None,
        limit=100,
        offset=0,
        db=db_session,
        current_user=test_user,
    )

    assert response.total == 1
    assert response.items[0].event_id == str(event.id)
    assert response.items[0].triage_status == "investigating"
    assert response.items[0].pinned is True
    assert response.by_triage_status["investigating"] == 1

    notifications = list(
        (
            await db_session.execute(
                select(Notification).where(Notification.related_entity_type == "autonomy_decision_event")
            )
        ).scalars().all()
    )
    assert len(notifications) == 1
    assert notifications[0].action_url.endswith(f"trace_event={event.id}")


@pytest.mark.asyncio
async def test_decision_trace_shared_assignment_and_escalation_filters(db_session, test_user):
    collaborator = User(
        username="trace-collaborator",
        email="trace-collaborator@example.com",
        hashed_password="hashed",
        is_active=True,
    )
    db_session.add(collaborator)
    await db_session.flush()
    old_event_time = datetime.utcnow() - timedelta(hours=30)
    event = await record_autonomy_decision_event(
        db_session,
        user_id=collaborator.id,
        event_type="validation_blocked",
        source_kind="validation_run",
        source_id="run-1",
        source_label="Validation Run",
        decision_type="validation_blocked",
        summary="Validation run blocked",
        severity="high",
        event_time=old_event_time,
    )
    event.assigned_to_user_id = test_user.id
    event.assigned_at = datetime.utcnow()
    event.assigned_by_user_id = test_user.id
    await db_session.commit()

    response = await get_decision_trace(
        source_kind=None,
        decision_type=None,
        customer=None,
        status=None,
        severity=None,
        actor_mode=None,
        triage_status=None,
        assigned_to_user_id=test_user.id,
        unassigned_only=False,
        escalation_state="escalated",
        pinned=None,
        actionable_only=True,
        start_at=None,
        end_at=None,
        limit=100,
        offset=0,
        db=db_session,
        current_user=test_user,
    )

    assert response.total == 1
    assert response.items[0].event_id == str(event.id)
    assert str(response.items[0].owner_user_id) == str(collaborator.id)
    assert response.items[0].assigned_to_user_id == test_user.id
    assert response.items[0].escalation_state == "escalated"
    assert response.by_assignee[str(test_user.id)] == 1
    assert response.by_escalation_state["escalated"] == 1


@pytest.mark.asyncio
async def test_decision_trace_hides_unrelated_collaboration_events(db_session, test_user):
    unrelated = User(
        username="trace-unrelated",
        email="trace-unrelated@example.com",
        hashed_password="hashed",
        is_active=True,
    )
    db_session.add(unrelated)
    await db_session.flush()
    await record_autonomy_decision_event(
        db_session,
        user_id=unrelated.id,
        event_type="policy_guardrail_triggered",
        source_kind="monitor",
        source_id="monitor-1",
        source_label="Unrelated Monitor",
        decision_type="policy_guardrail_triggered",
        summary="Guardrail triggered",
        severity="high",
    )
    await db_session.commit()

    response = await get_decision_trace(
        source_kind=None,
        decision_type=None,
        customer=None,
        status=None,
        severity=None,
        actor_mode=None,
        triage_status=None,
        assigned_to_user_id=None,
        unassigned_only=False,
        escalation_state=None,
        pinned=None,
        actionable_only=True,
        start_at=None,
        end_at=None,
        limit=100,
        offset=0,
        db=db_session,
        current_user=test_user,
    )

    assert response.total == 0


@pytest.mark.asyncio
async def test_decision_trace_endpoint_surfaces_derived_job_recovery_events(db_session, test_user):
    job = AgentJob(
        id=uuid4(),
        name="Recovery Job",
        goal="Retry a failed monitor",
        job_type="monitor",
        user_id=test_user.id,
        status=AgentJobStatus.FAILED.value,
        created_at=datetime(2026, 3, 16, 9, 0, 0),
        results={
            "execution_strategy": {
                "scheduler_state": {
                    "queue_reason": "execution_failure",
                    "backoff_until": "not-a-datetime",
                }
            }
        },
    )
    db_session.add(job)
    await db_session.commit()

    response = await get_decision_trace(
        source_kind="job",
        decision_type=None,
        customer=None,
        status=None,
        severity=None,
        actor_mode=None,
        triage_status=None,
        assigned_to_user_id=None,
        unassigned_only=False,
        escalation_state=None,
        pinned=None,
        actionable_only=False,
        start_at=None,
        end_at=None,
        limit=100,
        offset=0,
        db=db_session,
        current_user=test_user,
    )

    assert response.total == 1
    assert response.by_source_kind["job"] == 1
    assert response.by_decision_type["job_recovery_queued"] == 1
    event = response.items[0]
    assert event.is_derived is True
    assert event.record_origin == "derived_fallback"
    assert event.decision_type == "job_recovery_queued"
    assert event.reason_code == "execution_failure"
    assert event.reason_label == "Execution failure"
    assert event.scheduler_state is not None
    assert event.scheduler_state["queue_reason"] == "execution_failure"
    assert event.metadata is not None
    assert event.metadata["reason_label"] == "Execution failure"
    assert event.metadata["scheduler_state"]["queue_reason"] == "execution_failure"


@pytest.mark.asyncio
async def test_decision_trace_endpoint_surfaces_persisted_job_recovery_event_context(db_session, test_user):
    event = await record_autonomy_decision_event(
        db_session,
        user_id=test_user.id,
        event_type="job_recovery_queued",
        source_kind="job",
        source_id="job-structured",
        source_label="Structured Recovery Job",
        decision_type="job_recovery_queued",
        summary="Structured Recovery Job: queued for scheduler recovery",
        reason_code="execution_failure",
        metadata={
            "reason_label": "Execution failure",
            "scheduler_state": {
                "queue_reason": "execution_failure",
                "last_run_status": "failed",
                "failure_streak": 3,
                "last_scheduled_at": "2026-03-16T09:00:00Z",
                "last_dispatched_at": "2026-03-16T09:05:00Z",
                "backoff_until": "2026-03-16T10:00:00Z",
            },
        },
    )
    await db_session.commit()

    response = await get_decision_trace(
        source_kind="job",
        decision_type=None,
        customer=None,
        status=None,
        severity=None,
        actor_mode=None,
        triage_status=None,
        assigned_to_user_id=None,
        unassigned_only=False,
        escalation_state=None,
        pinned=None,
        actionable_only=False,
        start_at=None,
        end_at=None,
        limit=100,
        offset=0,
        db=db_session,
        current_user=test_user,
    )

    assert response.total == 1
    row = response.items[0]
    assert row.event_id == str(event.id)
    assert row.reason_label == "Execution failure"
    assert row.scheduler_state is not None
    assert row.scheduler_state["queue_reason"] == "execution_failure"
    assert row.metadata is not None
    assert row.metadata["reason_label"] == "Execution failure"
    assert row.metadata["scheduler_state"]["failure_streak"] == 3

    notification = (
        await db_session.execute(
            select(Notification).where(
                Notification.related_entity_type == "autonomy_decision_event",
                Notification.related_entity_id == event.id,
            )
        )
    ).scalars().first()
    assert notification is not None
    assert notification.data is not None
    assert notification.data["reason_label"] == "Execution failure"
    assert notification.data["scheduler_state"] is not None
    assert notification.data["scheduler_state"]["queue_reason"] == "execution_failure"


@pytest.mark.asyncio
async def test_record_autonomy_decision_event_normalizes_structured_trace_context(db_session, test_user):
    event = await record_autonomy_decision_event(
        db_session,
        user_id=test_user.id,
        event_type="job_recovery_queued",
        source_kind="queue",
        source_id="queue-structured",
        source_label="Structured Recovery Queue",
        decision_type="job_recovery_queued",
        summary="Structured Recovery Queue: queued for scheduler recovery",
        reason_code="execution_failure",
        reason_label="Execution failure",
        scheduler_state={
            "queue_reason": "execution_failure",
            "last_run_status": "failed",
            "failure_streak": 4,
            "last_scheduled_at": "2026-03-16T09:00:00Z",
            "last_dispatched_at": "2026-03-16T09:05:00Z",
            "backoff_until": "2026-03-16T10:00:00Z",
        },
        metadata={"source": "scheduler"},
    )
    await db_session.commit()

    response = await get_decision_trace(
        source_kind="queue",
        decision_type=None,
        customer=None,
        status=None,
        severity=None,
        actor_mode=None,
        triage_status=None,
        assigned_to_user_id=None,
        unassigned_only=False,
        escalation_state=None,
        pinned=None,
        actionable_only=False,
        start_at=None,
        end_at=None,
        limit=100,
        offset=0,
        db=db_session,
        current_user=test_user,
    )

    assert response.total == 1
    row = response.items[0]
    assert row.event_id == str(event.id)
    assert row.reason_label == "Execution failure"
    assert row.scheduler_state is not None
    assert row.scheduler_state["queue_reason"] == "execution_failure"
    assert row.metadata is not None
    assert row.metadata["reason_label"] == "Execution failure"
    assert row.metadata["scheduler_state"]["failure_streak"] == 4
    assert row.metadata["source"] == "scheduler"

    notification = (
        await db_session.execute(
            select(Notification).where(
                Notification.related_entity_type == "autonomy_decision_event",
                Notification.related_entity_id == event.id,
            )
        )
    ).scalars().first()
    assert notification is not None
    assert notification.data is not None
    assert notification.data["reason_label"] == "Execution failure"
    assert notification.data["scheduler_state"] is not None
    assert notification.data["scheduler_state"]["queue_reason"] == "execution_failure"


@pytest.mark.asyncio
async def test_record_autonomy_decision_event_strips_malformed_scheduler_context(db_session, test_user):
    event = await record_autonomy_decision_event(
        db_session,
        user_id=test_user.id,
        event_type="job_recovery_queued",
        source_kind="queue",
        source_id="queue-malformed",
        source_label="Malformed Recovery Queue",
        decision_type="job_recovery_queued",
        summary="Malformed Recovery Queue: queued for scheduler recovery",
        reason_code="execution_failure",
        reason_label="Execution failure",
        scheduler_state="bad-payload",
        metadata={"source": "scheduler"},
    )
    await db_session.commit()

    response = await get_decision_trace(
        source_kind="queue",
        decision_type=None,
        customer=None,
        status=None,
        severity=None,
        actor_mode=None,
        triage_status=None,
        assigned_to_user_id=None,
        unassigned_only=False,
        escalation_state=None,
        pinned=None,
        actionable_only=False,
        start_at=None,
        end_at=None,
        limit=100,
        offset=0,
        db=db_session,
        current_user=test_user,
    )

    assert response.total == 1
    row = response.items[0]
    assert row.event_id == str(event.id)
    assert row.reason_label == "Execution failure"
    assert row.scheduler_state is None
    assert row.metadata is not None
    assert row.metadata["reason_label"] == "Execution failure"
    assert "scheduler_state" not in row.metadata

    notification = (
        await db_session.execute(
            select(Notification).where(
                Notification.related_entity_type == "autonomy_decision_event",
                Notification.related_entity_id == event.id,
            )
        )
    ).scalars().first()
    assert notification is not None
    assert notification.data is not None
    assert notification.data["reason_label"] == "Execution failure"
    assert notification.data["scheduler_state"] is None


@pytest.mark.asyncio
async def test_decision_trace_endpoint_preserves_label_only_customer_rebalance_events(db_session, test_user):
    event = await record_autonomy_decision_event(
        db_session,
        user_id=test_user.id,
        event_type="customer_rebalanced",
        source_kind="monitor",
        source_id="beta-customer",
        source_label="Beta",
        customer="Beta",
        decision_type="customer_rebalanced",
        summary="Beta: customer rebalance applied",
        reason_code="customer_rebalance_guidance",
        reason_label="Customer rebalance guidance",
        metadata={
            "updated_monitor_ids": ["monitor-1", "monitor-2"],
            "change_source": "customer_rebalance_guidance",
        },
    )
    await db_session.commit()

    response = await get_decision_trace(
        source_kind="monitor",
        decision_type=None,
        customer="Beta",
        status=None,
        severity=None,
        actor_mode=None,
        triage_status=None,
        assigned_to_user_id=None,
        unassigned_only=False,
        escalation_state=None,
        pinned=None,
        actionable_only=False,
        start_at=None,
        end_at=None,
        limit=100,
        offset=0,
        db=db_session,
        current_user=test_user,
    )

    assert response.total == 1
    row = response.items[0]
    assert row.event_id == str(event.id)
    assert row.decision_type == "customer_rebalanced"
    assert row.reason_label == "Customer rebalance guidance"
    assert row.scheduler_state is None
    assert row.metadata is not None
    assert row.metadata["updated_monitor_ids"] == ["monitor-1", "monitor-2"]
    assert row.metadata["change_source"] == "customer_rebalance_guidance"


@pytest.mark.asyncio
async def test_decision_trace_analytics_endpoint_summarizes_trace_trends_and_queue_reasons(db_session, test_user):
    today = datetime.utcnow().date()
    day_one = datetime.combine(today - timedelta(days=1), datetime.min.time())
    day_two = datetime.combine(today - timedelta(days=2), datetime.min.time())

    await record_autonomy_decision_event(
        db_session,
        user_id=test_user.id,
        event_type="job_recovery_queued",
        source_kind="job",
        source_id="job-analytics-1",
        source_label="Recovery Job",
        decision_type="job_recovery_queued",
        summary="Recovery Job: queued for scheduler recovery",
        reason_code="execution_failure",
        reason_label="Execution failure",
        scheduler_state={"queue_reason": "execution_failure", "last_run_status": "failed"},
        event_time=day_one,
    )
    await record_autonomy_decision_event(
        db_session,
        user_id=test_user.id,
        event_type="validation_blocked",
        source_kind="validation_run",
        source_id="run-analytics-1",
        source_label="Validation Run",
        decision_type="validation_blocked",
        summary="Validation Run: blocked pending approval",
        reason_code="approval_required",
        reason_label="Approval required",
        scheduler_state={"queue_reason": "approval_required", "last_run_status": "pending"},
        event_time=day_one,
    )
    await record_autonomy_decision_event(
        db_session,
        user_id=test_user.id,
        event_type="job_recovery_queued",
        source_kind="job",
        source_id="job-analytics-2",
        source_label="Recovery Job 2",
        decision_type="job_recovery_queued",
        summary="Recovery Job 2: queued for scheduler recovery",
        reason_code="execution_failure",
        reason_label="Execution failure",
        scheduler_state="bad-payload",
        event_time=day_two,
    )
    await record_autonomy_decision_event(
        db_session,
        user_id=test_user.id,
        event_type="job_recovery_queued",
        source_kind="job",
        source_id="job-analytics-3",
        source_label="Recovery Job 3",
        decision_type="job_recovery_queued",
        summary="Recovery Job 3: queued for scheduler recovery",
        reason_code="execution_failure",
        reason_label="Execution failure",
        scheduler_state={"queue_reason": "execution_failure", "last_run_status": "failed"},
        event_time=day_two,
    )
    await db_session.commit()

    response = await get_decision_trace_analytics(
        source_kind=None,
        decision_type=None,
        customer=None,
        status=None,
        severity=None,
        actor_mode=None,
        triage_status=None,
        assigned_to_user_id=None,
        unassigned_only=False,
        escalation_state=None,
        pinned=None,
        actionable_only=False,
        start_at=None,
        end_at=None,
        days=7,
        db=db_session,
        current_user=test_user,
    )

    assert response.total == 4
    assert response.window_days == 7
    assert response.by_source_kind["job"] == 3
    assert response.by_source_kind["validation_run"] == 1
    assert response.by_triage_status["new"] == 4
    assert response.top_decision_types[0].value == "job_recovery_queued"
    assert response.top_decision_types[0].count == 3
    assert response.top_reason_labels[0].value == "Execution failure"
    assert response.top_reason_labels[0].count == 3
    assert response.top_queue_reasons[0].value == "execution_failure"
    assert response.top_queue_reasons[0].count == 2
    assert any(bucket.value == "unknown" and bucket.count == 1 for bucket in response.top_queue_reasons)
    assert len(response.daily_trend) == 7
    assert sum(point.count for point in response.daily_trend) == 4


@pytest.mark.asyncio
async def test_decision_trace_analytics_endpoint_returns_empty_trend_for_no_matches(db_session, test_user):
    response = await get_decision_trace_analytics(
        source_kind="portfolio",
        decision_type=None,
        customer=None,
        status=None,
        severity=None,
        actor_mode=None,
        triage_status=None,
        assigned_to_user_id=None,
        unassigned_only=False,
        escalation_state=None,
        pinned=None,
        actionable_only=False,
        start_at=None,
        end_at=None,
        days=7,
        db=db_session,
        current_user=test_user,
    )

    assert response.total == 0
    assert response.by_source_kind == {}
    assert response.by_triage_status == {}
    assert response.top_decision_types == []
    assert response.top_reason_labels == []
    assert response.top_queue_reasons == []
    assert len(response.daily_trend) == 7
    assert all(point.count == 0 for point in response.daily_trend)


@pytest.mark.asyncio
async def test_decision_trace_export_json_preserves_scheduler_context_and_filters(db_session, test_user):
    await record_autonomy_decision_event(
        db_session,
        user_id=test_user.id,
        event_type="job_recovery_queued",
        source_kind="job",
        source_id="job-export-1",
        source_label="Export Job",
        decision_type="job_recovery_queued",
        summary="Export Job: queued for scheduler recovery",
        reason_code="execution_failure",
        reason_label="Execution failure",
        scheduler_state={"queue_reason": "execution_failure", "last_run_status": "failed"},
    )
    await record_autonomy_decision_event(
        db_session,
        user_id=test_user.id,
        event_type="job_recovery_queued",
        source_kind="job",
        source_id="job-export-2",
        source_label="Malformed Export Job",
        decision_type="job_recovery_queued",
        summary="Malformed Export Job: queued for scheduler recovery",
        reason_code="execution_failure",
        reason_label="Execution failure",
        scheduler_state="bad-payload",
    )
    await db_session.commit()

    response = await export_decision_trace(
        format="json",
        source_kind="job",
        decision_type=None,
        customer=None,
        status=None,
        severity=None,
        actor_mode=None,
        triage_status=None,
        assigned_to_user_id=None,
        unassigned_only=False,
        escalation_state=None,
        pinned=None,
        actionable_only=False,
        start_at=None,
        end_at=None,
        db=db_session,
        current_user=test_user,
    )

    payload = json.loads(response.body.decode())
    assert payload["total"] == 2
    assert payload["by_source_kind"]["job"] == 2
    assert all(item["reason_label"] == "Execution failure" for item in payload["items"])
    items_by_source_id = {str(item["source_id"]): item for item in payload["items"]}
    assert items_by_source_id["job-export-1"]["scheduler_state"]["queue_reason"] == "execution_failure"
    assert items_by_source_id["job-export-2"]["scheduler_state"] is None
    assert response.media_type == "application/json"


@pytest.mark.asyncio
async def test_decision_trace_export_csv_returns_header_only_for_empty_result(db_session, test_user):
    response = await export_decision_trace(
        format="csv",
        source_kind="portfolio",
        decision_type=None,
        customer=None,
        status=None,
        severity=None,
        actor_mode=None,
        triage_status=None,
        assigned_to_user_id=None,
        unassigned_only=False,
        escalation_state=None,
        pinned=None,
        actionable_only=False,
        start_at=None,
        end_at=None,
        db=db_session,
        current_user=test_user,
    )

    csv_text = response.body.decode()
    assert csv_text.startswith("event_id,event_time,event_type,source_kind")
    assert csv_text.count("\n") == 1
    assert response.headers["Content-Disposition"].startswith('attachment; filename="decision_trace_export_')
    assert response.media_type == "text/csv; charset=utf-8"


@pytest.mark.asyncio
async def test_decision_trace_event_actions_update_triage_state(db_session, test_user):
    event = await record_autonomy_decision_event(
        db_session,
        user_id=test_user.id,
        event_type="opportunity_blocked",
        source_kind="portfolio",
        source_id="portfolio-1",
        source_label="Fleet",
        decision_type="opportunity_blocked",
        summary="Opportunity blocked",
    )
    await db_session.commit()

    acknowledged = await act_on_decision_trace_event(
        event_id=event.id,
        request=AgentDecisionTraceActionRequest(action="acknowledge"),
        db=db_session,
        current_user=test_user,
    )
    assert acknowledged.event.triage_status == "acknowledged"

    resolved = await act_on_decision_trace_event(
        event_id=event.id,
        request=AgentDecisionTraceActionRequest(action="resolve", note="Handled"),
        db=db_session,
        current_user=test_user,
    )
    assert resolved.event.triage_status == "resolved"
    assert resolved.event.resolution_note == "Handled"

    pinned = await act_on_decision_trace_event(
        event_id=event.id,
        request=AgentDecisionTraceActionRequest(action="toggle_pin"),
        db=db_session,
        current_user=test_user,
    )
    assert pinned.event.pinned is True

    assigned = await act_on_decision_trace_event(
        event_id=event.id,
        request=AgentDecisionTraceActionRequest(action="assign", assigned_to_user_id=str(test_user.id)),
        db=db_session,
        current_user=test_user,
    )
    assert str(assigned.event.assigned_to_user_id) == str(test_user.id)

    reopened = await act_on_decision_trace_event(
        event_id=event.id,
        request=AgentDecisionTraceActionRequest(action="reopen", note="Needs another pass"),
        db=db_session,
        current_user=test_user,
    )
    assert reopened.event.triage_status == "new"

    due_dated = await act_on_decision_trace_event(
        event_id=event.id,
        request=AgentDecisionTraceActionRequest(action="set_due_at", due_at=datetime.utcnow() - timedelta(hours=1)),
        db=db_session,
        current_user=test_user,
    )
    assert due_dated.event.due_at is not None
    assert due_dated.event.escalation_state == "escalated"
    assert due_dated.event.resolved_at is None


@pytest.mark.asyncio
async def test_decision_trace_follow_up_approve_reuses_profile_queue_action(db_session, test_user, monkeypatch):
    profile = DomainResearchProfile(
        id=uuid4(),
        user_id=test_user.id,
        title="Compiler Frontier",
        domain="Compiler",
        objective="Track compiler opportunities",
        status="running",
    )
    db_session.add(profile)
    await db_session.flush()
    event = await record_autonomy_decision_event(
        db_session,
        user_id=test_user.id,
        event_type="follow_up_queued_for_approval",
        source_kind="domain_profile",
        source_id=str(profile.id),
        source_label=profile.title,
        decision_type="follow_up_queued_for_approval",
        reason_code="follow_up_pending_approval",
        summary="Compiler Frontier: queued follow-up approval for compiler hotspot",
        metadata={"opportunity_id": "opp-compiler-1"},
    )
    await db_session.commit()

    captured: dict[str, object] = {}

    async def _fake_follow_up_queue_action(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            follow_up_launch_status="launched",
            follow_up_operator_decision="approved_launch",
            follow_up_job_id=uuid4(),
        )

    monkeypatch.setattr(agent_jobs_endpoint, "_perform_follow_up_queue_action", _fake_follow_up_queue_action)

    response = await act_on_decision_trace_event(
        event_id=event.id,
        request=AgentDecisionTraceActionRequest(action="approve_launch", note="Ship it"),
        db=db_session,
        current_user=test_user,
    )

    assert captured["profile"] == profile
    assert captured["profile_opportunity_id"] == "opp-compiler-1"
    assert captured["action"] == "approve_launch"
    assert captured["operator_note"] == "Ship it"
    assert response.event.event_type == "follow_up_approved"
    assert response.event.decision_type == "follow_up_approved"
    assert response.event.reason_code == "operator_approved_follow_up"
    assert response.event.status == "launched"
    assert response.event.triage_status == "resolved"
    assert response.event.resolution_note == "Ship it"
    assert response.event.after_state is not None
    assert response.event.after_state["opportunity_id"] == "opp-compiler-1"
    assert response.event.after_state["follow_up_launch_status"] == "launched"
    assert response.event.after_state["follow_up_operator_decision"] == "approved_launch"
    assert response.event.after_state["follow_up_job_id"]

    refreshed = await db_session.get(AutonomyDecisionEvent, event.id)
    assert refreshed is not None
    assert refreshed.event_type == "follow_up_approved"
    assert refreshed.triage_status == "resolved"


@pytest.mark.asyncio
async def test_decision_trace_follow_up_reject_reuses_portfolio_queue_action(db_session, test_user, monkeypatch):
    portfolio = ResearchPortfolio(
        id=uuid4(),
        user_id=test_user.id,
        title="Scientific Fleet",
        objective="Track fleet opportunities",
        status="running",
    )
    db_session.add(portfolio)
    await db_session.flush()
    event = await record_autonomy_decision_event(
        db_session,
        user_id=test_user.id,
        event_type="follow_up_queued",
        source_kind="portfolio",
        source_id=str(portfolio.id),
        source_label=portfolio.title,
        decision_type="follow_up_queued",
        reason_code="follow_up_pending_approval",
        summary="Scientific Fleet: queued follow-up approval for hotspot",
        metadata={"opportunity_id": "opp-fleet-1"},
    )
    await db_session.commit()

    captured: dict[str, object] = {}

    async def _fake_follow_up_queue_action(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            follow_up_launch_status="rejected",
            follow_up_operator_decision="rejected",
            follow_up_job_id=None,
        )

    monkeypatch.setattr(agent_jobs_endpoint, "_perform_follow_up_queue_action", _fake_follow_up_queue_action)

    response = await act_on_decision_trace_event(
        event_id=event.id,
        request=AgentDecisionTraceActionRequest(action="reject_launch", note="Not yet"),
        db=db_session,
        current_user=test_user,
    )

    assert captured["portfolio"] == portfolio
    assert captured["portfolio_opportunity_id"] == "opp-fleet-1"
    assert captured["action"] == "reject_launch"
    assert captured["operator_note"] == "Not yet"
    assert response.event.event_type == "follow_up_rejected"
    assert response.event.decision_type == "follow_up_rejected"
    assert response.event.reason_code == "operator_rejected_follow_up"
    assert response.event.status == "rejected"
    assert response.event.triage_status == "resolved"
    assert response.event.after_state is not None
    assert response.event.after_state["opportunity_id"] == "opp-fleet-1"
    assert response.event.after_state["follow_up_launch_status"] == "rejected"
    assert response.event.after_state["follow_up_operator_decision"] == "rejected"


@pytest.mark.asyncio
async def test_decision_trace_follow_up_relaunch_reuses_inbox_relaunch(db_session, test_user, monkeypatch):
    old_job_id = uuid4()
    new_job_id = uuid4()
    inbox_item = ResearchInboxItem(
        id=uuid4(),
        user_id=test_user.id,
        item_type="document",
        item_key="trace-follow-up-relaunch",
        title="Compiler hotspot",
        summary="Failed validation run",
        status="accepted",
        follow_up_launch_status="launched",
        follow_up_outcome_status="failed",
        follow_up_job_id=old_job_id,
    )
    db_session.add(inbox_item)
    await db_session.flush()
    event = await record_autonomy_decision_event(
        db_session,
        user_id=test_user.id,
        event_type="follow_up_failed",
        source_kind="domain_profile",
        source_id="profile-1",
        source_label="Compiler Frontier",
        decision_type="follow_up_failed",
        status="failed",
        summary="Compiler Frontier: compiler hotspot is follow up failed",
        metadata={"opportunity_id": "opp-compiler-1"},
        after_state={
            "opportunity_id": "opp-compiler-1",
            "follow_up_outcome_status": "failed",
            "follow_up_last_job_id": str(old_job_id),
        },
    )
    await db_session.commit()

    captured: dict[str, object] = {}

    async def _fake_relaunch_inbox_follow_up_item(**kwargs):
        captured.update(kwargs)
        item = kwargs["item"]
        item.follow_up_job_id = new_job_id
        item.follow_up_launch_status = "launched"
        item.follow_up_outcome_status = None
        return SimpleNamespace(
            inbox_item_id=item.id,
            follow_up_launch_status="launched",
            follow_up_job_id=new_job_id,
        )

    monkeypatch.setattr(agent_jobs_endpoint, "_relaunch_follow_up_inbox_item", _fake_relaunch_inbox_follow_up_item)

    response = await act_on_decision_trace_event(
        event_id=event.id,
        request=AgentDecisionTraceActionRequest(action="relaunch_follow_up", note="Retry now"),
        db=db_session,
        current_user=test_user,
    )

    assert captured["item"].id == inbox_item.id
    assert captured["operator_note"] == "Retry now"
    assert response.event.event_type == "follow_up_launched"
    assert response.event.decision_type == "follow_up_launched"
    assert response.event.reason_code == "operator_relaunched_follow_up"
    assert response.event.status == "active"
    assert response.event.triage_status == "resolved"
    assert response.event.resolution_note == "Retry now"
    assert response.event.after_state is not None
    assert response.event.after_state["follow_up_launch_status"] == "launched"
    assert response.event.after_state["follow_up_last_job_id"] == str(new_job_id)
    assert response.event.after_state["follow_up_outcome_status"] is None

    refreshed = await db_session.get(AutonomyDecisionEvent, event.id)
    assert refreshed is not None
    assert refreshed.event_type == "follow_up_launched"
    assert refreshed.reason_code == "operator_relaunched_follow_up"
    assert refreshed.triage_status == "resolved"


@pytest.mark.asyncio
async def test_decision_trace_follow_up_actions_reject_unsupported_events(db_session, test_user):
    derived_event = await record_autonomy_decision_event(
        db_session,
        user_id=test_user.id,
        event_type="follow_up_queued",
        source_kind="portfolio",
        source_id="portfolio-1",
        source_label="Fleet",
        decision_type="follow_up_queued",
        summary="Queued follow-up",
        metadata={"opportunity_id": "opp-1"},
        is_derived=True,
    )
    blocked_event = await record_autonomy_decision_event(
        db_session,
        user_id=test_user.id,
        event_type="opportunity_blocked",
        source_kind="portfolio",
        source_id="portfolio-1",
        source_label="Fleet",
        decision_type="opportunity_blocked",
        summary="Blocked opportunity",
        metadata={"opportunity_id": "opp-2"},
    )
    await db_session.commit()

    with pytest.raises(HTTPException) as derived_exc:
        await act_on_decision_trace_event(
            event_id=derived_event.id,
            request=AgentDecisionTraceActionRequest(action="approve_launch"),
            db=db_session,
            current_user=test_user,
        )
    assert derived_exc.value.status_code == 422

    with pytest.raises(HTTPException) as blocked_exc:
        await act_on_decision_trace_event(
            event_id=blocked_event.id,
            request=AgentDecisionTraceActionRequest(action="approve_launch"),
            db=db_session,
            current_user=test_user,
        )
    assert blocked_exc.value.status_code == 422

    completed_event = await record_autonomy_decision_event(
        db_session,
        user_id=test_user.id,
        event_type="follow_up_completed",
        source_kind="domain_profile",
        source_id="profile-1",
        source_label="Compiler Frontier",
        decision_type="follow_up_completed",
        status="completed",
        summary="Completed follow-up",
        after_state={"follow_up_outcome_status": "completed", "follow_up_last_job_id": str(uuid4())},
    )
    await db_session.commit()

    with pytest.raises(HTTPException) as completed_exc:
        await act_on_decision_trace_event(
            event_id=completed_event.id,
            request=AgentDecisionTraceActionRequest(action="relaunch_follow_up"),
            db=db_session,
            current_user=test_user,
        )
    assert completed_exc.value.status_code == 422


@pytest.mark.asyncio
async def test_decision_trace_assignment_requires_collaboration_visibility(db_session, test_user):
    outsider = User(
        username="trace-outsider",
        email="trace-outsider@example.com",
        hashed_password="hashed",
        is_active=True,
    )
    db_session.add(outsider)
    await db_session.flush()
    event = await record_autonomy_decision_event(
        db_session,
        user_id=test_user.id,
        event_type="opportunity_blocked",
        source_kind="portfolio",
        source_id="portfolio-1",
        source_label="Fleet",
        decision_type="opportunity_blocked",
        summary="Opportunity blocked",
    )
    await db_session.commit()

    with pytest.raises(HTTPException) as exc_info:
        await act_on_decision_trace_event(
            event_id=event.id,
            request=AgentDecisionTraceActionRequest(action="assign", assigned_to_user_id=str(outsider.id)),
            db=db_session,
            current_user=test_user,
        )

    assert exc_info.value.status_code == 422


@pytest.mark.asyncio
async def test_decision_trace_escalation_notifies_assignee_instead_of_owner(db_session, test_user):
    collaborator = User(
        username="trace-assignee",
        email="trace-assignee@example.com",
        hashed_password="hashed",
        is_active=True,
    )
    db_session.add(collaborator)
    await db_session.flush()
    event = await record_autonomy_decision_event(
        db_session,
        user_id=test_user.id,
        event_type="validation_blocked",
        source_kind="validation_run",
        source_id="run-assign",
        source_label="Validation Run",
        decision_type="validation_blocked",
        summary="Validation run blocked",
        severity="high",
    )
    event.assigned_to_user_id = collaborator.id
    event.assigned_at = datetime.utcnow()
    event.assigned_by_user_id = test_user.id
    await db_session.commit()

    await act_on_decision_trace_event(
        event_id=event.id,
        request=AgentDecisionTraceActionRequest(action="set_due_at", due_at=datetime.utcnow() - timedelta(hours=1)),
        db=db_session,
        current_user=test_user,
    )

    notifications = list(
        (
            await db_session.execute(
                select(Notification).where(Notification.related_entity_type == "autonomy_decision_event")
            )
        ).scalars().all()
    )
    assert notifications
    assert notifications[-1].user_id == collaborator.id
    assert notifications[-1].user_id != test_user.id


@pytest.mark.asyncio
async def test_decision_trace_views_round_trip(db_session, test_user):
    created = await create_decision_trace_view(
        AgentDecisionTraceViewCreate(
            name="High Signal",
            filters={"severity": "high", "actionable_only": True, "date_range": "30d"},
            is_default=True,
        ),
        db=db_session,
        current_user=test_user,
    )
    assert created.name == "High Signal"
    assert created.filters["severity"] == "high"
    assert created.is_default is True

    listed = await list_decision_trace_views(db=db_session, current_user=test_user)
    assert listed.total == 1
    assert listed.items[0].is_default is True

    promoted = await create_decision_trace_view(
        AgentDecisionTraceViewCreate(
            name="Pinned Secondary",
            filters={"pinned": True, "date_range": "7d"},
            is_default=False,
        ),
        db=db_session,
        current_user=test_user,
    )
    assert promoted.is_default is False

    updated = await update_decision_trace_view(
        view_id=promoted.id,
        request=AgentDecisionTraceViewUpdate(
            name="Pinned High Signal",
            filters={"pinned": True, "date_range": "7d"},
            is_default=True,
        ),
        db=db_session,
        current_user=test_user,
    )
    assert updated.name == "Pinned High Signal"
    assert updated.filters["pinned"] is True
    assert updated.filters["date_range"] == "7d"
    assert updated.is_default is True

    listed_after_update = await list_decision_trace_views(db=db_session, current_user=test_user)
    assert listed_after_update.total == 2
    assert listed_after_update.items[0].is_default is True
    assert listed_after_update.items[0].id == promoted.id
    assert listed_after_update.items[1].is_default is False
    assert listed_after_update.items[1].id == created.id

    await delete_decision_trace_view(view_id=created.id, db=db_session, current_user=test_user)
    await delete_decision_trace_view(view_id=promoted.id, db=db_session, current_user=test_user)
    listed_after_delete = await list_decision_trace_views(db=db_session, current_user=test_user)
    assert listed_after_delete.total == 0
