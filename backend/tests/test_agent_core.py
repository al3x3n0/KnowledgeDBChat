"""Tests for the extracted app.agent_core package."""

from unittest.mock import AsyncMock

import pytest

from app.agent_core.planning import AgentExecutionPlanner, ExecutionPlan, PlanStep
from app.agent_core.runtime import AgentRuntimeRunner
from app.agent_core.routing import AgentRouter
from app.agent_core.tool_catalog import get_tool_metadata
from app.agent_core.types import AgentSpec, agent_spec_from_model, job_spec_from_model


class DummyAgent:
    def __init__(self) -> None:
        self.id = "agent-1"
        self.name = "document_expert"
        self.display_name = "Document Expert"
        self.system_prompt = "helpful"
        self.capabilities = ["document_search"]
        self.tool_whitelist = ["search_documents"]
        self.priority = 80
        self.is_active = True
        self.routing_defaults = {"tier": "balanced"}


class DummyJob:
    def __init__(self) -> None:
        self.goal = "Research quantum computing"
        self.job_type = "research"
        self.config = {"replan_enabled": True}
        self.goal_criteria = {"criteria": [{"name": "Find papers", "description": "At least 5"}]}
        self.max_iterations = 25
        self.iteration = 3


class _DummyRuntimeAdapter:
    def __init__(
        self,
        *,
        decision=None,
        action_bundle=None,
        observe_error=None,
        continue_after_error=False,
    ) -> None:
        self.events = []
        self.iteration = 0
        self._decision = decision or {"goal_achieved": False, "should_stop": False}
        self._action_bundle = action_bundle or {}
        self._observe_error = observe_error
        self._continue_after_error = continue_after_error

    async def can_continue(self) -> bool:
        self.events.append("can_continue")
        return self.iteration == 0

    async def on_iteration_start(self) -> None:
        self.events.append("start")
        self.iteration += 1

    async def observe_phase(self):
        self.events.append("observe")
        if self._observe_error is not None:
            raise self._observe_error
        return {"iteration": self.iteration}

    async def think_phase(self, observation):
        self.events.append("think")
        return dict(self._decision)

    async def act_phase(self, decision):
        self.events.append("act")
        return dict(self._action_bundle)

    async def evaluate_phase(self, decision, action_bundle):
        self.events.append("evaluate")
        return {"progress": 100, "should_stop": False}

    async def on_iteration_complete(self, observation, decision, action_bundle, evaluation) -> None:
        self.events.append("complete")

    async def on_iteration_error(self, exc: Exception) -> bool:
        self.events.append("error")
        return self._continue_after_error

    async def build_run_result(self):
        self.events.append("build")
        return {"status": "completed"}


def test_agent_spec_from_model_maps_fields():
    spec = agent_spec_from_model(DummyAgent())
    assert spec.name == "document_expert"
    assert spec.priority == 80
    assert spec.metadata["routing_defaults"] == {"tier": "balanced"}


def test_job_spec_from_model_maps_fields():
    spec = job_spec_from_model(DummyJob())
    assert spec.goal == "Research quantum computing"
    assert spec.job_type == "research"
    assert spec.max_iterations == 25


@pytest.mark.asyncio
async def test_core_planner_fallback():
    planner = AgentExecutionPlanner(AsyncMock(generate_response=AsyncMock(side_effect=Exception("down"))))
    plan = await planner.create_plan(
        job=DummyJob(),
        observation={},
        user_settings=None,
        available_tools=["search_documents", "read_document_content", "summarize_document"],
    )
    assert len(plan.steps) == 3
    assert plan.steps[0].title == "Gather information"


def test_core_planner_dependency_annotation():
    steps = [PlanStep(title="A"), PlanStep(title="B")]
    annotated = AgentExecutionPlanner.annotate_dependencies(steps)
    assert annotated[1].depends_on == [annotated[0].step_id]


@pytest.mark.asyncio
async def test_core_router_selects_specialist():
    router = AgentRouter()
    agents = {
        "document_expert": AgentSpec(
            name="document_expert",
            display_name="Document Expert",
            system_prompt="helpful",
            capabilities=["document_search"],
            priority=80,
        ),
        "generalist": AgentSpec(
            name="generalist",
            display_name="Generalist",
            system_prompt="helpful",
            capabilities=["general"],
            priority=50,
        ),
    }
    selected, reason = await router.select_agent(
        {"capabilities_needed": ["document_search"]},
        available_agents=agents,
    )
    assert selected.name == "document_expert"
    assert "Matched capabilities" in reason


@pytest.mark.asyncio
async def test_core_router_keyword_analysis():
    router = AgentRouter()
    result = await router.analyze_intent("Find documents about transformers", use_llm=False)
    assert "document_search" in result["capabilities_needed"]
    assert result["method"] == "keyword"


def test_tool_catalog_returns_mcp_metadata():
    meta = get_tool_metadata("mcp:chat")
    assert meta is not None
    assert meta.name == "mcp:chat"
    assert meta.network == "none"


@pytest.mark.asyncio
async def test_runtime_runner_completes_normal_iteration_flow():
    adapter = _DummyRuntimeAdapter()
    result = await AgentRuntimeRunner().run(adapter)
    assert result == {"status": "completed"}
    assert adapter.events == [
        "can_continue",
        "start",
        "observe",
        "think",
        "act",
        "evaluate",
        "complete",
        "can_continue",
        "build",
    ]


@pytest.mark.asyncio
async def test_runtime_runner_stops_after_think_when_goal_achieved():
    adapter = _DummyRuntimeAdapter(decision={"goal_achieved": True, "should_stop": False})
    result = await AgentRuntimeRunner().run(adapter)
    assert result == {"status": "completed"}
    assert "act" not in adapter.events
    assert adapter.events[-1] == "build"


@pytest.mark.asyncio
async def test_runtime_runner_returns_terminal_result_from_act():
    adapter = _DummyRuntimeAdapter(action_bundle={"terminal_result": {"status": "paused"}})
    result = await AgentRuntimeRunner().run(adapter)
    assert result == {"status": "paused"}
    assert "evaluate" not in adapter.events


@pytest.mark.asyncio
async def test_runtime_runner_uses_error_handler_and_stops_when_requested():
    adapter = _DummyRuntimeAdapter(observe_error=RuntimeError("boom"), continue_after_error=False)
    result = await AgentRuntimeRunner().run(adapter)
    assert result == {"status": "completed"}
    assert adapter.events == [
        "can_continue",
        "start",
        "observe",
        "error",
        "build",
    ]
