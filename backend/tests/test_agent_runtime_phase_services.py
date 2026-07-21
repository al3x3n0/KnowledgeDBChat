from datetime import datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import AsyncMock

import pytest

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.agent_action_service import AgentActionService
from app.services.agent_observation_service import AgentObservationService
from app.services.agent_progress_evaluation_service import AgentProgressEvaluationService
from app.services.agent_thinking_service import AgentThinkingService
from app.services.autonomous_agent_executor import AutonomousAgentExecutor, _AutonomousRuntimeAdapter


def _make_job(*, job_type: str = "research", config=None) -> AgentJob:
    return AgentJob(
        name="Phase Service Test",
        goal="Improve retrieval quality",
        job_type=job_type,
        user_id=uuid4(),
        status=AgentJobStatus.RUNNING.value,
        config=config or {},
        max_iterations=20,
        max_tool_calls=20,
        max_llm_calls=20,
        max_runtime_minutes=30,
    )


@pytest.mark.asyncio
async def test_observation_service_tracks_data_analysis_runtime():
    executor = AutonomousAgentExecutor()
    job = _make_job(job_type="data_analysis")
    state = {
        "actions_taken": [
            {"action": {"tool": "create_chart"}},
            {"action": {"tool": "create_flowchart"}},
            {"action": {"tool": "aggregate_data"}},
        ]
    }
    executor._data_analysis_tools[str(job.id)] = SimpleNamespace(
        list_datasets=lambda: {"count": 2, "datasets": [{"name": "sales"}, {"name": "profit"}]}
    )
    executor._get_execution_graph_runtime_snapshot = lambda runtime_state: {"graph_health": {"score": 1.0}}

    observation = await AgentObservationService().observe(executor, job, state, db=AsyncMock())

    assert observation["datasets_loaded"] == 2
    assert observation["charts_created"] == 1
    assert observation["diagrams_created"] == 1
    assert observation["transformations_applied"] == 1
    assert observation["execution_graph"]["graph_health"]["score"] == 1.0


def test_thinking_service_parse_decision_response_recovers_invalid_tool():
    executor = AutonomousAgentExecutor()
    service = AgentThinkingService()
    job = _make_job()
    state = {"findings": [], "actions_taken": []}
    available_tools = executor._get_tools_for_job_type(job.job_type, job.config)

    raw = """{
      "goal_achieved": false,
      "should_stop": false,
      "reasoning": "Try a tool",
      "action": {"tool": "non_existent_tool", "params": {"foo": "bar"}}
    }"""
    decision = service.parse_decision_response(
        executor,
        raw_response=raw,
        job=job,
        state=state,
        available_tools=available_tools,
    )

    assert decision["goal_achieved"] is False
    assert decision["should_stop"] is False
    assert decision["action"] is not None
    assert decision["action"]["tool"] in set(available_tools)


@pytest.mark.asyncio
async def test_action_service_blocks_cross_scope_write():
    executor = AutonomousAgentExecutor()
    service = AgentActionService()
    job = _make_job(config={"source_id": "source-a"})
    state = {}

    result = await service.act(
        executor,
        job,
        {"tool": "create_document_from_text", "params": {"source_id": "source-b", "text": "hello"}},
        state,
        db=AsyncMock(),
    )

    assert result["success"] is False
    assert "Scope guard blocked cross-source write" in result["error"]
    assert state["scope_guard_blocks"] == 1


@pytest.mark.asyncio
async def test_progress_service_marks_research_with_document_artifact_near_done():
    executor = AutonomousAgentExecutor()
    service = AgentProgressEvaluationService()
    job = _make_job(
        config={"max_documents": 10, "max_papers": 10, "prefer_sources": ["documents"]}
    )
    job.iteration = 3
    state = {
        "findings": [{"type": "document", "id": "doc-1"}],
        "actions_taken": [],
        "artifacts": [{"type": "document", "id": "artifact-1"}],
    }

    progress = await service.evaluate_progress(executor, job, state, user_settings=None, db=AsyncMock())

    assert progress >= 85


@pytest.mark.asyncio
async def test_runtime_adapter_observe_phase_uses_observation_service_directly():
    executor = AutonomousAgentExecutor()
    calls = []

    class _FakeObservationService:
        async def observe(self, passed_executor, job, state, db):
            calls.append((passed_executor, job, state, db))
            return {"context": [], "iteration": 1}

    executor.observation_service = _FakeObservationService()
    executor._observe = AsyncMock(side_effect=AssertionError("wrapper should not be called"))
    executor._resolve_default_source_scope = lambda job: None
    executor._resolve_scope_source = lambda job: "none"
    executor._append_scope_event = lambda state, event: None
    executor._ensure_causal_experiment_plan = AsyncMock(return_value=False)
    executor._resolve_execution_mode = lambda job, state=None: "autonomous"
    executor._ensure_execution_plan = AsyncMock(return_value=False)
    executor._ensure_subgoals = lambda job, state: None
    executor._ensure_swarm_chain_config = lambda job, state: None
    executor._ensure_subgoal_chain_config = lambda job, state: None
    executor._should_run_critic = lambda job, state: False

    job = _make_job()
    state = {"observations": [], "execution_plan": []}
    adapter = _AutonomousRuntimeAdapter(
        executor=executor,
        job=job,
        agent_def=None,
        user_settings=None,
        state=state,
        db=AsyncMock(),
        start_time=datetime.utcnow(),
        max_runtime=timedelta(minutes=5),
        progress_callback=None,
    )

    observation = await adapter.observe_phase()

    assert observation["iteration"] == 1
    assert len(calls) == 1
