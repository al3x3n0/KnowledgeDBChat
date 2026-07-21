"""Tests for extracted skill profile service."""

from uuid import uuid4

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.agent_skill_profile_service import AgentSkillProfileService
from app.services.autonomous_agent_executor import AutonomousAgentExecutor


def _make_job(config=None, *, name="Skill Test", goal="Research docs") -> AgentJob:
    return AgentJob(
        name=name,
        goal=goal,
        job_type="research",
        user_id=uuid4(),
        status=AgentJobStatus.RUNNING.value,
        config=config or {},
        max_iterations=20,
        max_tool_calls=20,
        max_llm_calls=20,
        max_runtime_minutes=30,
    )


def test_skill_profile_service_maps_alias_role_to_critic():
    executor = AutonomousAgentExecutor()
    service = AgentSkillProfileService()
    job = _make_job(config={"agent_role": "analyst"})

    profile = service.resolve_agent_skill_profile(executor, job, state={})

    assert profile["role"] == "critic"
    assert profile["display_name"] == "Critic"


def test_skill_profile_service_honors_explicit_override():
    executor = AutonomousAgentExecutor()
    service = AgentSkillProfileService()
    job = _make_job(config={"agent_role": "researcher"})

    profile = service.resolve_agent_skill_profile(executor, job, state={}, override_role="author")

    assert profile["role"] == "author"
    assert "plan_document" in profile["preferred_tools"]
