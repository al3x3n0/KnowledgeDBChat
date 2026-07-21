"""Tests for extracted goal-contract service."""

from uuid import uuid4

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.agent_goal_contract_service import AgentGoalContractService
from app.services.autonomous_agent_executor import AutonomousAgentExecutor


def _make_job(config=None) -> AgentJob:
    return AgentJob(
        name="Goal Contract Test",
        goal="Summarize results",
        job_type="research",
        user_id=uuid4(),
        status=AgentJobStatus.RUNNING.value,
        config=config or {},
        max_iterations=20,
        max_tool_calls=20,
        max_llm_calls=20,
        max_runtime_minutes=30,
    )


def test_goal_contract_service_evaluates_required_counts_and_types():
    executor = AutonomousAgentExecutor()
    service = AgentGoalContractService()
    job = _make_job(
        {
            "goal_contract_enabled": True,
            "goal_contract_min_progress": 70,
            "goal_contract_min_findings": 2,
            "goal_contract_required_finding_types": ["paper"],
        }
    )
    state = {
        "goal_progress": 65,
        "findings": [{"type": "document", "id": "d1"}],
        "artifacts": [],
    }

    result = service.evaluate_goal_contract(executor, job, state)

    assert result["enabled"] is True
    assert result["satisfied"] is False
    assert "progress>=70" in result["missing"]
    assert "findings>=2" in result["missing"]
    assert "finding_type:paper" in result["missing"]


def test_goal_contract_service_builds_executive_digest():
    executor = AutonomousAgentExecutor()
    service = AgentGoalContractService()
    job = _make_job({"goal_contract_enabled": True, "goal_contract_min_findings": 2})
    job.results = {"summary": "Partial outcome", "research_bundle": {"next_steps": ["Validate metrics"]}}
    state = {
        "goal_progress": 60,
        "findings": [{"type": "document", "title": "Internal bottleneck"}],
        "artifacts": [{"type": "note", "id": "a1"}],
        "actions_taken": [
            {"action": {"tool": "search_documents"}, "result": {"success": True}},
            {"action": {"tool": "search_arxiv"}, "result": {"success": False, "error": "timeout"}},
        ],
        "critic_notes": [{"severity": "high", "pivot": "Need stronger baselines"}],
    }

    digest = service.build_executive_digest(executor, job, state)

    assert digest["outcome"] == "Partial outcome"
    assert digest["metrics"]["failed_actions"] == 1
    assert digest["key_findings"]
    assert digest["risks"]
    assert digest["goal_contract"]["enabled"] is True
    assert digest["goal_contract"]["satisfied"] is False
    assert digest["next_actions"][0] == "Validate metrics"
