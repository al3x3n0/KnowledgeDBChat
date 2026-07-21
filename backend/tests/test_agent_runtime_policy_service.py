"""Tests for extracted runtime policy service."""

from uuid import uuid4

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.agent_runtime_policy_service import AgentRuntimePolicyService
from app.services.autonomous_agent_executor import AutonomousAgentExecutor


def _make_job(config=None) -> AgentJob:
    return AgentJob(
        name="Policy Test",
        goal="Improve retrieval quality",
        job_type="research",
        user_id=uuid4(),
        status=AgentJobStatus.RUNNING.value,
        config=config or {},
        max_iterations=20,
        max_tool_calls=20,
        max_llm_calls=20,
        max_runtime_minutes=30,
    )


def test_runtime_policy_service_resolves_ab_assignment_and_effective_mode():
    executor = AutonomousAgentExecutor()
    service = AgentRuntimePolicyService()
    job = _make_job(
        {
            "tool_selection_policy_mode": "adaptive",
            "tool_selection_ab_test_enabled": True,
            "tool_selection_ab_test_split": 1.0,
            "tool_selection_ab_test_variant_a": "baseline",
            "tool_selection_ab_test_variant_b": "thompson",
        }
    )
    state = {}

    mode, assignment = service.resolve_tool_selection_mode(executor, job, state=state)

    assert mode == "baseline"
    assert assignment["variant"] == "A"
    assert state["tool_selection_effective_mode"] == "baseline"


def test_runtime_policy_service_applies_live_fallback_override():
    executor = AutonomousAgentExecutor()
    service = AgentRuntimePolicyService()
    job = _make_job(
        {
            "tool_selection_policy_mode": "thompson",
            "tool_selection_live_fallback_enabled": True,
            "tool_selection_live_fallback_min_samples": 3,
            "tool_selection_live_fallback_min_success_rate": 0.5,
            "tool_selection_live_fallback_to_mode": "adaptive",
        }
    )
    job.iteration = 9
    state = {
        "tool_selection_mode_metrics": {
            "thompson": {"success": 0, "failure": 4},
        }
    }

    mode, assignment = service.resolve_tool_selection_mode(executor, job, state=state)

    assert mode == "adaptive"
    assert assignment["mode"] == "adaptive"
    assert state["tool_selection_mode_override"] == "adaptive"
