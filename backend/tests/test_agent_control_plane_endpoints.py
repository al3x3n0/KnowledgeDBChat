import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.endpoints import agent_control_plane
from app.core.database import get_db
from app.models.agent_job import AgentJob
from app.models.autonomy_decision_event import AutonomyDecisionEvent
from app.models.domain_research_profile import DomainResearchProfile
from app.models.research_portfolio import ResearchPortfolio
from app.models.user import User
from app.models.workflow import Workflow, WorkflowExecution


@pytest.fixture
def control_plane_client(db_session, test_user):
    app = FastAPI()
    app.include_router(agent_control_plane.router, prefix="/api/v1/agent-control-plane")

    def override_get_db():
        return db_session

    async def override_current_user():
        return test_user

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[agent_control_plane.get_current_active_user] = override_current_user

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_list_agent_control_runs_groups_job_and_workflow_roots_for_current_user(
    control_plane_client,
    db_session,
    test_user,
):
    other_user = User(
        id=uuid4(),
        username="other",
        email="other@example.com",
        hashed_password="x",
        is_active=True,
    )

    root_job = AgentJob(
        name="Root planner job",
        goal="Plan and execute validation",
        job_type="analysis",
        user_id=test_user.id,
        status="running",
        progress=40,
        iteration=1,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=10,
        config={"automation_profile": "balanced"},
    )
    approval_job = AgentJob(
        name="Approval-gated job",
        goal="Needs operator approval",
        job_type="analysis",
        user_id=test_user.id,
        status="paused",
        progress=70,
        iteration=1,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=10,
        results={
            "approval_checkpoint": {
                "message": "Approve the next tool step",
                "action": {"tool": "web.search", "params": {"q": "compiler regression"}},
            }
        },
    )
    recovery_job = AgentJob(
        name="Recurring validation job",
        goal="Recover recurring validation",
        job_type="analysis",
        user_id=test_user.id,
        status="failed",
        progress=55,
        iteration=1,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=10,
        schedule_type="recurring",
        last_activity_at=datetime.utcnow() - timedelta(hours=2),
        results={"execution_strategy": {"scheduler_state": {"queue_reason": "execution_failure", "retry_count": 2}}},
    )
    workflow = Workflow(
        user_id=test_user.id,
        name="Validation workflow",
        is_active=True,
    )
    hidden_job = AgentJob(
        name="Hidden other job",
        goal="Should not appear",
        job_type="analysis",
        user_id=other_user.id,
        status="completed",
        progress=100,
        iteration=1,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=10,
    )

    async def _seed():
        db_session.add(other_user)
        db_session.add(root_job)
        db_session.add(approval_job)
        db_session.add(recovery_job)
        db_session.add(workflow)
        db_session.add(hidden_job)
        await db_session.flush()

        child_job = AgentJob(
            name="Child executor job",
            goal="Execute validation",
            job_type="analysis",
            user_id=test_user.id,
            status="completed",
            progress=100,
            iteration=1,
            max_iterations=10,
            max_tool_calls=10,
            max_llm_calls=10,
            max_runtime_minutes=10,
            parent_job_id=root_job.id,
            root_job_id=root_job.id,
            chain_depth=1,
            results={"research_note_id": "note-1"},
        )
        root_execution = WorkflowExecution(
            workflow_id=workflow.id,
            user_id=test_user.id,
            trigger_type="manual",
            status="running",
            progress=25,
            context={"job_id": str(root_job.id)},
        )
        db_session.add(child_job)
        db_session.add(root_execution)
        await db_session.flush()

        child_execution = WorkflowExecution(
            workflow_id=workflow.id,
            user_id=test_user.id,
            parent_execution_id=root_execution.id,
            trigger_type="subworkflow",
            status="completed",
            progress=100,
            context={},
        )
        db_session.add(child_execution)
        await db_session.flush()

        db_session.add(
            AutonomyDecisionEvent(
                user_id=test_user.id,
                event_type="job_progress",
                source_kind="agent_job",
                source_id=str(child_job.id),
                decision_type="delegated_execution",
                summary="Child job executed",
            )
        )
        db_session.add(
            AutonomyDecisionEvent(
                user_id=test_user.id,
                event_type="workflow_progress",
                source_kind="workflow_execution",
                source_id=str(child_execution.id),
                decision_type="workflow_step_complete",
                summary="Workflow step completed",
            )
        )
        profile = DomainResearchProfile(
            user_id=test_user.id,
            title="Compiler domain",
            domain="compiler",
            objective="Track regressions",
            status="running",
            latest_summary={
                "queued_operator_reviews_count": 1,
                "queued_operator_reviews_by_type": {"policy_review": 1},
                "queued_operator_reviews": [
                    {
                        "review_type": "policy_review",
                        "reason_code": "sandbox_policy_gate",
                        "reason_label": "Policy review",
                        "opportunity_id": "opp-policy-1",
                        "title": "Compiler policy gate",
                    }
                ],
                "manual_follow_up_recommendations": [
                    {
                        "opportunity_id": "opp-manual-profile-1",
                        "title": "Manual compiler follow-up",
                        "reason_code": "manual_follow_up_recommendation",
                    }
                ],
                "opportunities": [
                    {
                        "opportunity_id": "opp-manual-profile-1",
                        "title": "Manual compiler follow-up",
                        "autonomy_state": "eligible",
                        "follow_up_outcome_status": "",
                    }
                ],
            },
            latest_note_ids=["note-1"],
            active_job_id=root_job.id,
            latest_run_job_id=root_job.id,
        )
        portfolio = ResearchPortfolio(
            user_id=test_user.id,
            title="Compiler fleet",
            objective="Fleet objective",
            status="running",
            latest_summary={
                "queued_operator_reviews_count": 1,
                "queued_operator_reviews_by_type": {"budget_review": 1},
                "queued_operator_reviews": [
                    {
                        "review_type": "budget_review",
                        "reason_code": "portfolio_budget_cap",
                        "reason_label": "Budget review",
                        "opportunity_id": "opp-budget-1",
                        "title": "Workflow budget gate",
                    }
                ],
                "manual_follow_up_recommendations": [
                    {
                        "opportunity_id": "opp-manual-fleet-1",
                        "title": "Manual fleet follow-up",
                        "reason_code": "manual_follow_up_recommendation",
                    }
                ],
                "opportunities": [
                    {
                        "opportunity_id": "opp-manual-fleet-1",
                        "title": "Manual fleet follow-up",
                        "autonomy_state": "eligible",
                        "follow_up_outcome_status": "",
                    }
                ],
            },
            latest_note_ids=["note-fleet-1"],
            active_job_id=root_job.id,
            latest_run_job_id=root_job.id,
            child_job_ids=[str(root_job.id)],
        )
        db_session.add(profile)
        db_session.add(portfolio)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = control_plane_client.get("/api/v1/agent-control-plane/runs")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    items = payload["items"]
    job_summary = next(item for item in items if item["source_type"] == "job")
    workflow_summary = next(item for item in items if item["source_type"] == "workflow")

    assert job_summary["title"] == "Root planner job"
    assert job_summary["child_job_count"] == 1
    assert job_summary["decision_count"] == 1
    assert job_summary["replayability_status"] == "full_lineage"
    assert job_summary["queued_operator_review_count"] == 6
    assert job_summary["queued_operator_reviews_by_type"]["approval_checkpoint"] == 1
    assert job_summary["queued_operator_reviews_by_type"]["job_recovery"] == 1
    assert job_summary["queued_operator_reviews_by_type"]["manual_follow_up_recommendation"] == 2
    assert workflow_summary["title"] == "Validation workflow"
    assert workflow_summary["child_execution_count"] == 1
    assert workflow_summary["child_job_count"] == 1
    assert workflow_summary["decision_count"] == 1
    assert workflow_summary["replayability_status"] == "full_lineage"
    assert workflow_summary["queued_operator_review_count"] == 4


def test_agent_control_run_detail_for_job_includes_related_links_and_memory_graph(
    control_plane_client,
    db_session,
    test_user,
    monkeypatch,
):
    root_job = AgentJob(
        name="Compiler monitor",
        goal="Track compiler regressions",
        job_type="monitor",
        user_id=test_user.id,
        status="completed",
        progress=100,
        iteration=1,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=10,
        config={},
        results={
            "experiment_plan_id": "plan-1",
            "experiment_run_id": "run-1",
        },
    )
    approval_job = AgentJob(
        name="Checkpointed compiler step",
        goal="Pause for approval",
        job_type="analysis",
        user_id=test_user.id,
        status="paused",
        progress=60,
        iteration=1,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=10,
        parent_job_id=None,
        root_job_id=None,
        results={
            "approval_checkpoint": {
                "message": "Approve compiler benchmark step",
                "action": {"tool": "bash.exec"},
            }
        },
    )
    recovery_job = AgentJob(
        name="Recover compiler monitor",
        goal="Recover recurring compiler monitor",
        job_type="monitor",
        user_id=test_user.id,
        status="failed",
        progress=25,
        iteration=1,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=10,
        schedule_type="recurring",
        last_activity_at=datetime.utcnow() - timedelta(hours=3),
        results={"execution_strategy": {"scheduler_state": {"queue_reason": "execution_failure", "retry_count": 3}}},
    )

    async def _seed():
        db_session.add(root_job)
        db_session.add(approval_job)
        db_session.add(recovery_job)
        await db_session.flush()
        workflow = Workflow(
            user_id=test_user.id,
            name="Compiler workflow",
            is_active=True,
        )
        db_session.add(workflow)
        await db_session.flush()
        workflow_execution = WorkflowExecution(
            workflow_id=workflow.id,
            user_id=test_user.id,
            trigger_type="manual",
            status="completed",
            progress=100,
            context={},
        )
        db_session.add(workflow_execution)
        await db_session.flush()
        root_job.config = {
            "automation_profile": "balanced",
            "workflow_execution_id": str(workflow_execution.id),
            "research_note_id": "note-control-1",
            "routing_tier": "balanced",
            "provider": "openai",
            "model": "gpt-5.4",
        }
        child_job = AgentJob(
            name="Compiler validator",
            goal="Validate regression",
            job_type="analysis",
            user_id=test_user.id,
            status="completed",
            progress=100,
            iteration=1,
            max_iterations=10,
            max_tool_calls=10,
            max_llm_calls=10,
            max_runtime_minutes=10,
            parent_job_id=root_job.id,
            root_job_id=root_job.id,
            chain_depth=1,
        )
        db_session.add(child_job)
        await db_session.flush()
        approval_job.parent_job_id = root_job.id
        approval_job.root_job_id = root_job.id
        recovery_job.parent_job_id = root_job.id
        recovery_job.root_job_id = root_job.id
        db_session.add(
            AutonomyDecisionEvent(
                user_id=test_user.id,
                event_type="job_completed",
                source_kind="agent_job",
                source_id=str(child_job.id),
                decision_type="materialize_experiment",
                summary="Validation linked to experiment",
                event_metadata={
                    "linked_note_ids": ["note-control-1"],
                    "linked_experiment_plan_ids": ["plan-1"],
                    "linked_validation_run_ids": ["run-1"],
                },
            )
        )
        db_session.add(
            DomainResearchProfile(
                user_id=test_user.id,
                title="Compiler domain",
                domain="compiler",
                objective="Track regressions",
                status="running",
                latest_summary={
                    "queued_operator_reviews_count": 1,
                    "queued_operator_reviews_by_type": {"policy_review": 1},
                    "queued_operator_reviews": [
                        {
                            "review_type": "policy_review",
                            "reason_code": "sandbox_policy_gate",
                            "reason_label": "Policy review",
                            "opportunity_id": "opp-policy-1",
                            "title": "Compiler policy gate",
                            "monitor_job_id": str(root_job.id),
                            "customer": "compiler",
                            "recommended_action": "apply_guardrail",
                            "policy_guardrail_action": "rollback",
                            "policy_guardrail_target_history_entry_id": "history-123",
                            "policy_guardrail_reasons": ["sandbox restriction", "manual review required"],
                            "policy_rollback_payload": {"history_entry_id": "history-123"},
                        }
                    ],
                    "manual_follow_up_recommendations": [
                        {
                            "opportunity_id": "opp-manual-1",
                            "title": "Manual compiler follow-up",
                            "reason_code": "manual_follow_up_recommendation",
                        }
                    ],
                    "opportunities": [
                        {
                            "opportunity_id": "opp-manual-1",
                            "title": "Manual compiler follow-up",
                            "autonomy_state": "eligible",
                            "follow_up_outcome_status": "",
                            "follow_up_block_reason": "Operator requested manual approval.",
                        }
                    ],
                },
                latest_note_ids=["note-control-1"],
                active_job_id=root_job.id,
                latest_run_job_id=root_job.id,
            )
        )
        await db_session.commit()
        return workflow_execution.id

    workflow_execution_id = asyncio.get_event_loop().run_until_complete(_seed())

    async def _fake_slice_memory_graph_for_job_ids(*, db, current_user, job_ids):
        return {
            "nodes": [{"id": "mem-1", "type": "lesson", "content": "bootstrap", "importance_score": 0.9, "tags": [], "job_id": str(root_job.id)}],
            "edges": [],
            "stats": {"memory_count": 1, "edge_count": 0, "job_count": len(job_ids)},
            "job_id": str(root_job.id),
        }

    monkeypatch.setattr(agent_control_plane, "_slice_memory_graph_for_job_ids", _fake_slice_memory_graph_for_job_ids)

    response = control_plane_client.get(f"/api/v1/agent-control-plane/runs/job:{root_job.id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run"]["title"] == "Compiler monitor"
    assert payload["run"]["decision_count"] == 1
    assert payload["run"]["replayability_status"] == payload["replay"]["replayability_status"] == "full_lineage"
    assert payload["routing"]["routing_tier"] == "balanced"
    assert payload["memory_graph"]["stats"]["memory_count"] == 1
    assert payload["queued_operator_review_count"] == 4
    policy_review = next(item for item in payload["queued_operator_reviews"] if item["review_type"] == "policy_review")
    assert policy_review["action_path"] == f"/autonomous-agents?tab=domain&profileId={policy_review['source_id']}&opportunityId=opp-policy-1"
    assert policy_review["queue_path"] == (
        f"/autonomous-agents?tab=queue&queue_item_type=policy_review&profileId={policy_review['source_id']}"
        f"&opportunityId=opp-policy-1&queue_customer=compiler&queue_job={root_job.id}"
    )
    assert policy_review["can_approve"] is False
    assert policy_review["can_reject"] is False
    assert policy_review["job_id"] == str(root_job.id)
    assert policy_review["customer"] == "compiler"
    assert policy_review["recommended_action"] == "apply_guardrail"
    assert policy_review["policy_guardrail_action"] == "rollback"
    assert policy_review["policy_guardrail_target_history_entry_id"] == "history-123"
    assert policy_review["policy_rollback_payload"] == {"history_entry_id": "history-123"}
    assert policy_review["policy_guardrail_reasons"] == ["sandbox restriction", "manual review required"]
    manual_review = next(item for item in payload["queued_operator_reviews"] if item["review_type"] == "manual_follow_up_recommendation")
    assert manual_review["can_launch_follow_up"] is True
    assert manual_review["can_relaunch_follow_up"] is False
    assert "queue_health_drilldown=manual_follow_up_recommendations" in manual_review["queue_path"]
    assert manual_review["follow_up_block_reason"] == "Operator requested manual approval."
    approval_review = next(item for item in payload["queued_operator_reviews"] if item["review_type"] == "approval_checkpoint")
    assert approval_review["source_kind"] == "job"
    assert approval_review["source_id"] == str(approval_job.id)
    assert approval_review["can_approve"] is True
    assert approval_review["can_skip"] is True
    assert approval_review["checkpoint"]["action"]["tool"] == "bash.exec"
    assert approval_review["queue_path"] == f"/autonomous-agents?tab=queue&queue_item_type=approval_checkpoint&queue_job={approval_job.id}"
    recovery_review = next(item for item in payload["queued_operator_reviews"] if item["review_type"] == "job_recovery")
    assert recovery_review["source_kind"] == "job"
    assert recovery_review["can_restart"] is True
    assert recovery_review["can_resume"] is True
    assert recovery_review["can_cancel"] is True
    assert recovery_review["scheduler_state"]["queue_reason"] == "execution_failure"
    assert {link["path"] for link in payload["related_links"]} >= {
        f"/autonomous-agents?job={root_job.id}",
        f"/workflows?executionId={workflow_execution_id}",
        "/research-notes?note=note-control-1",
        "/synthesis",
    }
    routing_link = next(link for link in payload["related_links"] if link["label"] == "Routing Observability")
    assert "provider=openai" in routing_link["path"]
    assert "model=gpt-5.4" in routing_link["path"]
    assert "routing_tier=balanced" in routing_link["path"]
    workflow_node = next(node for node in payload["nodes"] if node["kind"] == "workflow_execution")
    assert workflow_node["metadata"]["workflow_execution_id"] == str(workflow_execution_id)
    experiment_node = next(node for node in payload["nodes"] if node["kind"] == "experiment_run")
    assert experiment_node["metadata"]["experiment_run_id"] == "run-1"


def test_agent_control_run_detail_for_sparse_workflow_returns_partial_lineage(
    control_plane_client,
    db_session,
    test_user,
):
    workflow = Workflow(
        user_id=test_user.id,
        name="Sparse workflow",
        is_active=True,
    )

    async def _seed():
        db_session.add(workflow)
        await db_session.flush()
        execution = WorkflowExecution(
            workflow_id=workflow.id,
            user_id=test_user.id,
            trigger_type="manual",
            status="pending",
            progress=0,
            context={},
        )
        db_session.add(execution)
        await db_session.commit()
        return execution.id

    execution_id = asyncio.get_event_loop().run_until_complete(_seed())

    response = control_plane_client.get(f"/api/v1/agent-control-plane/runs/workflow:{execution_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run"]["source_type"] == "workflow"
    assert payload["run"]["decision_count"] == 0
    assert payload["run"]["replayability_status"] == "partial_lineage"
    assert payload["replay"]["replayability_status"] == "partial_lineage"
    assert payload["memory_graph"] is None
    assert payload["routing"] is None
    assert payload["related_links"] == [{"label": "Workflows", "path": f"/workflows?executionId={execution_id}"}]


def test_list_agent_control_reviews_returns_flattened_cross_run_queue_items(
    control_plane_client,
    db_session,
    test_user,
):
    root_job = AgentJob(
        name="Compiler root job",
        goal="Track compiler regressions",
        job_type="monitor",
        user_id=test_user.id,
        status="running",
        progress=50,
        iteration=1,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=10,
    )
    workflow = Workflow(user_id=test_user.id, name="Compiler workflow", is_active=True)

    async def _seed():
        db_session.add(root_job)
        db_session.add(workflow)
        await db_session.flush()
        approval_job = AgentJob(
            name="Approval job",
            goal="Needs approval",
            job_type="analysis",
            user_id=test_user.id,
            status="paused",
            progress=50,
            iteration=1,
            max_iterations=10,
            max_tool_calls=10,
            max_llm_calls=10,
            max_runtime_minutes=10,
            parent_job_id=root_job.id,
            root_job_id=root_job.id,
            results={"approval_checkpoint": {"message": "Approve next step", "action": {"tool": "web.search"}}},
        )
        execution = WorkflowExecution(
            workflow_id=workflow.id,
            user_id=test_user.id,
            trigger_type="manual",
            status="running",
            progress=10,
            context={"job_id": str(root_job.id)},
        )
        db_session.add(approval_job)
        db_session.add(execution)
        await db_session.flush()
        profile = DomainResearchProfile(
            user_id=test_user.id,
            title="Compiler domain",
            domain="compiler",
            objective="Track regressions",
            status="running",
            latest_summary={
                "queued_operator_reviews": [
                    {
                        "review_type": "policy_review",
                        "reason_code": "policy_guardrail",
                        "opportunity_id": "opp-policy-1",
                        "title": "Compiler guardrail",
                        "monitor_job_id": str(root_job.id),
                        "customer": "compiler",
                    }
                ]
            },
            active_job_id=root_job.id,
            latest_run_job_id=root_job.id,
        )
        db_session.add(profile)
        await db_session.commit()
        return str(root_job.id), str(execution.id), str(approval_job.id)

    root_job_id, execution_id, approval_job_id = asyncio.get_event_loop().run_until_complete(_seed())

    response = control_plane_client.get("/api/v1/agent-control-plane/reviews?queue_preset=approval_required")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["summary"]["total"] == 1
    assert payload["summary"]["by_type"]["approval_checkpoint"] == 1
    assert payload["offset"] == 0
    assert payload["has_more"] is False
    assert payload["items"][0]["review_type"] == "approval_checkpoint"
    assert payload["items"][0]["run_id"] == f"job:{root_job_id}"
    assert payload["items"][0]["run_title"] == "Compiler root job"
    assert payload["items"][0]["job_id"] == approval_job_id

    response = control_plane_client.get("/api/v1/agent-control-plane/reviews?review_type=policy_review")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert payload["summary"]["by_type"]["policy_review"] == 2
    run_ids = {item["run_id"] for item in payload["items"]}
    assert f"job:{root_job_id}" in run_ids
    assert f"workflow:{execution_id}" in run_ids


def test_list_agent_control_reviews_supports_priority_sort_and_pagination(
    control_plane_client,
    db_session,
    test_user,
):
    root_job = AgentJob(
        name="Compiler root job",
        goal="Track compiler regressions",
        job_type="monitor",
        user_id=test_user.id,
        status="running",
        progress=50,
        iteration=1,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=10,
    )

    async def _seed():
        db_session.add(root_job)
        await db_session.flush()
        approval_job = AgentJob(
            name="Approval job",
            goal="Needs approval",
            job_type="analysis",
            user_id=test_user.id,
            status="paused",
            progress=50,
            iteration=1,
            max_iterations=10,
            max_tool_calls=10,
            max_llm_calls=10,
            max_runtime_minutes=10,
            parent_job_id=root_job.id,
            root_job_id=root_job.id,
            created_at=datetime.utcnow() - timedelta(minutes=20),
            results={"approval_checkpoint": {"message": "Approve next step", "action": {"tool": "web.search"}}},
        )
        recovery_job = AgentJob(
            name="Recovery job",
            goal="Recover monitor",
            job_type="monitor",
            user_id=test_user.id,
            status="failed",
            progress=25,
            iteration=1,
            max_iterations=10,
            max_tool_calls=10,
            max_llm_calls=10,
            max_runtime_minutes=10,
            schedule_type="recurring",
            parent_job_id=root_job.id,
            root_job_id=root_job.id,
            created_at=datetime.utcnow() - timedelta(hours=5),
            last_activity_at=datetime.utcnow() - timedelta(hours=5),
            results={"execution_strategy": {"scheduler_state": {"queue_reason": "execution_failure", "retry_count": 3}}},
        )
        db_session.add(approval_job)
        db_session.add(recovery_job)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    response = control_plane_client.get("/api/v1/agent-control-plane/reviews?sort=priority&limit=1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 2
    assert payload["summary"]["by_type"]["approval_checkpoint"] == 1
    assert payload["summary"]["by_type"]["job_recovery"] == 1
    assert payload["summary"]["by_sla_bucket"]["overdue"] == 1
    assert payload["limit"] == 1
    assert payload["offset"] == 0
    assert payload["has_more"] is True
    assert payload["items"][0]["review_type"] == "job_recovery"

    second_page = control_plane_client.get("/api/v1/agent-control-plane/reviews?sort=priority&limit=1&offset=1")
    assert second_page.status_code == 200
    second_payload = second_page.json()
    assert second_payload["items"][0]["review_type"] == "approval_checkpoint"
    assert second_payload["has_more"] is False


def test_agent_control_review_action_approves_follow_up_for_profile(
    control_plane_client,
    db_session,
    test_user,
    monkeypatch,
):
    root_job = AgentJob(
        name="Compiler monitor",
        goal="Track compiler regressions",
        job_type="monitor",
        user_id=test_user.id,
        status="running",
        progress=50,
        iteration=1,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=10,
    )

    async def _seed():
        db_session.add(root_job)
        await db_session.flush()
        profile = DomainResearchProfile(
            user_id=test_user.id,
            title="Compiler domain",
            domain="compiler",
            objective="Track regressions",
            status="running",
            latest_summary={
                "effective_policy": {"follow_up_review_mode": "queue_for_approval", "auto_launch_follow_up": True},
                "queued_operator_reviews": [
                    {
                        "review_type": "follow_up_recommendation",
                        "reason_code": "follow_up_launch_approval",
                        "reason_label": "Follow-up launch approval",
                        "opportunity_id": "opp-follow-up-1",
                        "title": "Launch compiler follow-up",
                        "evidence_revision": "rev-1",
                        "autonomy_state": "eligible",
                    }
                ],
                "opportunities": [
                    {
                        "opportunity_id": "opp-follow-up-1",
                        "title": "Launch compiler follow-up",
                        "autonomy_state": "eligible",
                        "evidence_revision": "rev-1",
                        "follow_up_review_status": "pending_approval",
                        "child_job_ids": [],
                    }
                ],
            },
            active_job_id=root_job.id,
            latest_run_job_id=root_job.id,
        )
        db_session.add(profile)
        await db_session.commit()
        return str(profile.id)

    profile_id = asyncio.get_event_loop().run_until_complete(_seed())

    class _HelperResponse:
        detail = "Follow-up launched from queue approval"
        follow_up_launch_status = "launched"
        follow_up_operator_decision = "approved_launch"
        follow_up_job_id = uuid4()

    async def _fake_helper(**kwargs):
        assert kwargs["profile"] is not None
        assert kwargs["profile_opportunity_id"] == "opp-follow-up-1"
        assert kwargs["action"] == "approve_launch"
        assert kwargs["operator_note"] == "Looks good."
        return _HelperResponse()

    monkeypatch.setattr(agent_control_plane, "_perform_follow_up_queue_action", _fake_helper)

    response = control_plane_client.post(
        "/api/v1/agent-control-plane/reviews/action",
        json={
            "review_type": "follow_up_recommendation",
            "source_kind": "profile",
            "source_id": profile_id,
            "opportunity_id": "opp-follow-up-1",
            "action": "approve_follow_up",
            "operator_note": "Looks good.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["action"] == "approve_follow_up"
    assert payload["follow_up_launch_status"] == "launched"
    assert payload["follow_up_operator_decision"] == "approved_launch"
    assert payload["source_id"] == profile_id


def test_agent_control_review_action_applies_policy_guardrail_with_rollback(
    control_plane_client,
    db_session,
    test_user,
    monkeypatch,
):
    async def _seed():
        profile = DomainResearchProfile(
            user_id=test_user.id,
            title="Compiler domain",
            domain="compiler",
            objective="Track regressions",
            status="running",
            latest_summary={
                "queued_operator_reviews": [
                    {
                        "review_type": "policy_review",
                        "opportunity_id": "opp-policy-1",
                        "monitor_job_id": "3d7b1d8c-c01a-4ffd-9011-7a4cf962de18",
                        "policy_rollback_payload": {"history_entry_id": "history-abc"},
                    }
                ],
                "opportunities": [
                    {
                        "opportunity_id": "opp-policy-1",
                        "monitor_job_id": "3d7b1d8c-c01a-4ffd-9011-7a4cf962de18",
                    }
                ],
            },
        )
        db_session.add(profile)
        await db_session.commit()
        return str(profile.id)

    profile_id = asyncio.get_event_loop().run_until_complete(_seed())

    async def _fake_rollback_monitor_policy(*, monitor_job_id, payload, current_user, db):
        assert monitor_job_id == "3d7b1d8c-c01a-4ffd-9011-7a4cf962de18"
        assert payload.history_entry_id == "history-abc"
        assert payload.change_reason == "Apply the safeguard."
        assert current_user.id == test_user.id
        return None

    monkeypatch.setattr(agent_control_plane, "rollback_monitor_policy", _fake_rollback_monitor_policy)

    response = control_plane_client.post(
        "/api/v1/agent-control-plane/reviews/action",
        json={
            "review_type": "policy_review",
            "source_kind": "profile",
            "source_id": profile_id,
            "opportunity_id": "opp-policy-1",
            "action": "apply_guardrail",
            "operator_note": "Apply the safeguard.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["action"] == "apply_guardrail"
    assert payload["monitor_job_id"] == "3d7b1d8c-c01a-4ffd-9011-7a4cf962de18"
    assert payload["detail"] == "Policy safeguard rollback applied"


def test_agent_control_review_action_rejects_read_only_budget_review(
    control_plane_client,
    db_session,
    test_user,
):
    async def _seed():
        profile = DomainResearchProfile(
            user_id=test_user.id,
            title="Compiler domain",
            domain="compiler",
            objective="Track regressions",
            status="running",
            latest_summary={},
        )
        db_session.add(profile)
        await db_session.commit()
        return str(profile.id)

    profile_id = asyncio.get_event_loop().run_until_complete(_seed())

    response = control_plane_client.post(
        "/api/v1/agent-control-plane/reviews/action",
        json={
            "review_type": "budget_review",
            "source_kind": "profile",
            "source_id": profile_id,
            "opportunity_id": "opp-budget-1",
            "action": "open_monitor",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "This review type is currently read-only in the control plane"


def test_agent_control_review_action_launches_manual_follow_up_for_profile(
    control_plane_client,
    db_session,
    test_user,
    monkeypatch,
):
    async def _seed():
        profile = DomainResearchProfile(
            user_id=test_user.id,
            title="Compiler domain",
            domain="compiler",
            objective="Track regressions",
            status="running",
            latest_summary={},
        )
        db_session.add(profile)
        await db_session.commit()
        return str(profile.id)

    profile_id = asyncio.get_event_loop().run_until_complete(_seed())

    async def _fake_action(**kwargs):
        assert str(kwargs["profile_id"]) == profile_id
        assert kwargs["opportunity_id"] == "opp-manual-1"
        assert kwargs["payload"].action == "launch_follow_up"
        assert kwargs["payload"].operator_note == "Launch it."

        class _Response:
            latest_summary = {
                "opportunities": [
                    {
                        "opportunity_id": "opp-manual-1",
                        "child_job_ids": ["job-follow-up-123"],
                    }
                ]
            }

        return _Response()

    monkeypatch.setattr(agent_control_plane, "act_on_domain_research_opportunity", _fake_action)

    response = control_plane_client.post(
        "/api/v1/agent-control-plane/reviews/action",
        json={
            "review_type": "manual_follow_up_recommendation",
            "source_kind": "profile",
            "source_id": profile_id,
            "opportunity_id": "opp-manual-1",
            "action": "launch_follow_up",
            "operator_note": "Launch it.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == "launch_follow_up"
    assert payload["follow_up_launch_status"] == "launched"
    assert payload["follow_up_job_id"] == "job-follow-up-123"


def test_agent_control_review_action_approves_job_checkpoint(
    control_plane_client,
    db_session,
    test_user,
    monkeypatch,
):
    async def _seed():
        job = AgentJob(
            name="Approval job",
            goal="Needs approval",
            job_type="analysis",
            user_id=test_user.id,
            status="paused",
            progress=50,
            iteration=1,
            max_iterations=10,
            max_tool_calls=10,
            max_llm_calls=10,
            max_runtime_minutes=10,
            results={"approval_checkpoint": {"message": "Approve next step", "action": {"tool": "web.search"}}},
        )
        db_session.add(job)
        await db_session.commit()
        return str(job.id)

    job_id = asyncio.get_event_loop().run_until_complete(_seed())

    async def _fake_perform_job_action(job, request, *, db, current_user):
        assert str(job.id) == job_id
        assert request.action == "approve"
        assert request.checkpoint_note == "Ship it."
        return job

    monkeypatch.setattr(agent_control_plane, "_perform_job_action", _fake_perform_job_action)

    response = control_plane_client.post(
        "/api/v1/agent-control-plane/reviews/action",
        json={
            "review_type": "approval_checkpoint",
            "source_kind": "job",
            "source_id": job_id,
            "opportunity_id": job_id,
            "action": "approve",
            "operator_note": "Ship it.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["action"] == "approve"
    assert payload["source_id"] == job_id
    assert payload["follow_up_job_id"] == job_id


def test_agent_control_review_action_edits_job_checkpoint(
    control_plane_client,
    db_session,
    test_user,
    monkeypatch,
):
    async def _seed():
        job = AgentJob(
            name="Approval job",
            goal="Needs approval",
            job_type="analysis",
            user_id=test_user.id,
            status="paused",
            progress=50,
            iteration=1,
            max_iterations=10,
            max_tool_calls=10,
            max_llm_calls=10,
            max_runtime_minutes=10,
            results={"approval_checkpoint": {"message": "Approve next step", "action": {"tool": "web.search", "purpose": "Find regressions", "params": {"q": "compiler regression"}}}},
        )
        db_session.add(job)
        await db_session.commit()
        return str(job.id)

    job_id = asyncio.get_event_loop().run_until_complete(_seed())

    async def _fake_perform_job_action(job, request, *, db, current_user):
        assert str(job.id) == job_id
        assert request.action == "edit"
        assert request.checkpoint_note == "Tighten the query."
        assert request.checkpoint_action_patch == {
            "tool": "web.search",
            "purpose": "Investigate narrower regressions",
            "params": {"q": "gcc regression bug"},
        }
        return job

    monkeypatch.setattr(agent_control_plane, "_perform_job_action", _fake_perform_job_action)

    response = control_plane_client.post(
        "/api/v1/agent-control-plane/reviews/action",
        json={
            "review_type": "approval_checkpoint",
            "source_kind": "job",
            "source_id": job_id,
            "opportunity_id": job_id,
            "action": "edit",
            "operator_note": "Tighten the query.",
            "checkpoint_action_patch": {
                "tool": "web.search",
                "purpose": "Investigate narrower regressions",
                "params": {"q": "gcc regression bug"},
            },
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["action"] == "edit"
    assert payload["source_id"] == job_id
    assert payload["follow_up_job_id"] == job_id


def test_agent_control_review_action_restarts_job_recovery(
    control_plane_client,
    db_session,
    test_user,
    monkeypatch,
):
    async def _seed():
        job = AgentJob(
            name="Recovery job",
            goal="Needs restart",
            job_type="analysis",
            user_id=test_user.id,
            status="failed",
            progress=15,
            iteration=1,
            max_iterations=10,
            max_tool_calls=10,
            max_llm_calls=10,
            max_runtime_minutes=10,
            schedule_type="recurring",
            last_activity_at=datetime.utcnow() - timedelta(hours=2),
            results={"execution_strategy": {"scheduler_state": {"queue_reason": "execution_failure"}}},
        )
        db_session.add(job)
        await db_session.commit()
        return str(job.id)

    job_id = asyncio.get_event_loop().run_until_complete(_seed())

    async def _fake_perform_job_action(job, request, *, db, current_user):
        assert str(job.id) == job_id
        assert request.action == "restart"
        assert request.checkpoint_note == "Retry now."
        return job

    monkeypatch.setattr(agent_control_plane, "_perform_job_action", _fake_perform_job_action)

    response = control_plane_client.post(
        "/api/v1/agent-control-plane/reviews/action",
        json={
            "review_type": "job_recovery",
            "source_kind": "job",
            "source_id": job_id,
            "opportunity_id": job_id,
            "action": "restart",
            "operator_note": "Retry now.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["action"] == "restart"
    assert payload["source_id"] == job_id


def test_agent_control_bulk_review_action_delegates_job_queue_bulk_action(
    control_plane_client,
    test_user,
    monkeypatch,
):
    class _BulkResponse:
        requested_count = 2
        applied = 2
        failed = 0
        results = [
            type("Row", (), {"queue_key": "approval:job-1", "job_id": uuid4(), "ok": True, "error": None, "status": "pending"})(),
            type("Row", (), {"queue_key": "approval:job-2", "job_id": uuid4(), "ok": True, "error": None, "status": "pending"})(),
        ]

    async def _fake_bulk(*, request, db, current_user):
        assert request.item_type == "approval_checkpoint"
        assert request.action == "approve"
        assert request.job_ids == ["job-1", "job-2"]
        assert request.checkpoint_note == "Approve all."
        assert current_user.id == test_user.id
        return _BulkResponse()

    monkeypatch.setattr(agent_control_plane, "checkpoint_queue_bulk_action", _fake_bulk)

    response = control_plane_client.post(
        "/api/v1/agent-control-plane/reviews/bulk-action",
        json={
            "item_type": "approval_checkpoint",
            "action": "approve",
            "job_ids": ["job-1", "job-2"],
            "operator_note": "Approve all.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["item_type"] == "approval_checkpoint"
    assert payload["action"] == "approve"
    assert payload["requested_count"] == 2
    assert payload["applied"] == 2


def test_agent_control_bulk_review_action_delegates_follow_up_bulk_action(
    control_plane_client,
    test_user,
    monkeypatch,
):
    profile_id = str(uuid4())

    class _BulkResponse:
        requested_count = 2
        applied = 1
        failed = 1
        results = [
            type(
                "Row",
                (),
                {
                    "profile_opportunity_id": "opp-1",
                    "portfolio_opportunity_id": None,
                    "ok": True,
                    "error": None,
                    "detail": "launched",
                    "follow_up_launch_status": "launched",
                    "follow_up_operator_decision": "approved_launch",
                    "follow_up_job_id": uuid4(),
                },
            )(),
            type(
                "Row",
                (),
                {
                    "profile_opportunity_id": "opp-2",
                    "portfolio_opportunity_id": None,
                    "ok": False,
                    "error": "blocked",
                    "detail": None,
                    "follow_up_launch_status": None,
                    "follow_up_operator_decision": None,
                    "follow_up_job_id": None,
                },
            )(),
        ]

    async def _fake_follow_up_bulk(*, request, db, current_user):
        assert str(request.domain_research_profile_id) == profile_id
        assert request.profile_opportunity_ids == ["opp-1", "opp-2"]
        assert request.action == "approve_launch"
        assert request.operator_note == "Launch these."
        assert current_user.id == test_user.id
        return _BulkResponse()

    monkeypatch.setattr(agent_control_plane, "checkpoint_queue_bulk_follow_up_action", _fake_follow_up_bulk)

    response = control_plane_client.post(
        "/api/v1/agent-control-plane/reviews/bulk-action",
        json={
            "item_type": "follow_up_recommendation",
            "action": "approve_launch",
            "domain_research_profile_id": profile_id,
            "profile_opportunity_ids": ["opp-1", "opp-2"],
            "operator_note": "Launch these.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["item_type"] == "follow_up_recommendation"
    assert payload["requested_count"] == 2
    assert payload["applied"] == 1
    assert payload["failed"] == 1


def test_agent_control_bulk_review_action_rejects_unsupported_item_type(
    control_plane_client,
):
    response = control_plane_client.post(
        "/api/v1/agent-control-plane/reviews/bulk-action",
        json={
            "item_type": "manual_follow_up_recommendation",
            "action": "launch_follow_up",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Bulk actions are not supported for this control-plane item type"


def test_agent_control_run_detail_rejects_malformed_or_foreign_ids(
    control_plane_client,
    db_session,
    test_user,
):
    other_user = User(
        id=uuid4(),
        username="foreign",
        email="foreign@example.com",
        hashed_password="x",
        is_active=True,
    )
    foreign_job = AgentJob(
        name="Foreign job",
        goal="Should stay hidden",
        job_type="analysis",
        user_id=other_user.id,
        status="completed",
        progress=100,
        iteration=1,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=10,
        max_runtime_minutes=10,
    )

    async def _seed():
        db_session.add(other_user)
        db_session.add(foreign_job)
        await db_session.commit()

    asyncio.get_event_loop().run_until_complete(_seed())

    malformed = control_plane_client.get("/api/v1/agent-control-plane/runs/not-a-run")
    foreign = control_plane_client.get(f"/api/v1/agent-control-plane/runs/job:{foreign_job.id}")

    assert malformed.status_code == 404
    assert foreign.status_code == 404


def test_agent_control_run_views_crud_and_default_uniqueness(
    control_plane_client,
):
    created = control_plane_client.post(
        "/api/v1/agent-control-plane/views",
        json={
            "name": "Compiler Runs",
            "filters": {
                "source_type": "job",
                "outcome": "running",
                "routing_tier": "balanced",
                "selected_run_id": "job:run-1",
                "queue_status": "pending_approval",
                "queue_health_drilldown": "pending_follow_up_approvals",
                "queue_sort": "age_desc",
                "ignored": "nope",
            },
            "is_default": True,
        },
    )
    assert created.status_code == 201
    created_body = created.json()
    assert created_body["name"] == "Compiler Runs"
    assert created_body["filters"] == {
        "source_type": "job",
        "outcome": "running",
        "routing_tier": "balanced",
        "selected_run_id": "job:run-1",
        "queue_status": "pending_approval",
        "queue_health_drilldown": "pending_follow_up_approvals",
        "queue_sort": "age_desc",
    }
    assert created_body["is_default"] is True

    second = control_plane_client.post(
        "/api/v1/agent-control-plane/views",
        json={
            "name": "Workflow Runs",
            "filters": {"source_type": "workflow"},
            "is_default": True,
        },
    )
    assert second.status_code == 201
    second_body = second.json()
    assert second_body["is_default"] is True

    listed = control_plane_client.get("/api/v1/agent-control-plane/views")
    assert listed.status_code == 200
    listed_body = listed.json()
    assert listed_body["total"] == 2
    assert listed_body["items"][0]["id"] == second_body["id"]
    assert listed_body["items"][0]["is_default"] is True
    first_view = next(item for item in listed_body["items"] if item["id"] == created_body["id"])
    assert first_view["is_default"] is False

    updated = control_plane_client.patch(
        f"/api/v1/agent-control-plane/views/{created_body['id']}",
        json={
            "name": "Compiler Runs Updated",
            "filters": {"source_type": "job", "routing_tier": "premium", "queue_preset": "compiler", "queue_sort": "created_at_desc"},
            "is_default": True,
        },
    )
    assert updated.status_code == 200
    updated_body = updated.json()
    assert updated_body["name"] == "Compiler Runs Updated"
    assert updated_body["filters"] == {"source_type": "job", "routing_tier": "premium", "queue_preset": "compiler", "queue_sort": "created_at_desc"}
    assert updated_body["is_default"] is True

    relisted = control_plane_client.get("/api/v1/agent-control-plane/views")
    relisted_items = relisted.json()["items"]
    updated_row = next(item for item in relisted_items if item["id"] == created_body["id"])
    other_row = next(item for item in relisted_items if item["id"] == second_body["id"])
    assert updated_row["is_default"] is True
    assert other_row["is_default"] is False

    deleted = control_plane_client.delete(f"/api/v1/agent-control-plane/views/{second_body['id']}")
    assert deleted.status_code == 204

    final_list = control_plane_client.get("/api/v1/agent-control-plane/views")
    assert final_list.status_code == 200
    assert final_list.json()["total"] == 1
