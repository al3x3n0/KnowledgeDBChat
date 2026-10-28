"""Observation planners share one remaining job allowance, including fallbacks."""

import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.agent_core.runtime import AgentRuntimeRunner
from app.models.agent_job import AgentJob
from app.services.autonomous_agent_executor import (
    AutonomousAgentExecutor,
    _AutonomousRuntimeAdapter,
)

pytestmark = pytest.mark.unit


def _adapter(
    remaining, *, causal="valid", execution="valid", critic="valid", replan_error=False
):
    executor = AutonomousAgentExecutor()
    job = AgentJob(
        id=uuid4(),
        user_id=uuid4(),
        name="Observation budget",
        goal="Compare evidence",
        job_type="research",
        status="running",
        iteration=1,
        llm_calls_used=2,
        max_llm_calls=2 + remaining,
        tool_calls_used=0,
        max_tool_calls=10,
        max_iterations=10,
        max_runtime_minutes=5,
        config={
            "allowed_tools": ["get_research_findings"],
            "auto_subgoal_child_jobs_enabled": False,
            "swarm_child_jobs_enabled": False,
        },
    )
    state = {
        "observations": [],
        "findings": [{"type": "dynamic_profile", "title": "retained"}],
        "actions_taken": [],
    }
    calls = []

    async def generate(**kwargs):
        phase = (kwargs.get("snapshot_context") or {}).get("phase", "replan")
        calls.append(phase)
        if phase == "causal_plan":
            if causal == "error":
                raise RuntimeError("fixture causal request failed")
            if causal == "invalid":
                return "not JSON"
            return json.dumps(
                {
                    "hypotheses": ["A change improves the result"],
                    "experiments": [
                        {
                            "name": "Compare",
                            "minimal_design": "Compare baseline and intervention",
                        }
                    ],
                }
            )
        if phase == "execution_plan":
            if execution == "error":
                raise RuntimeError("fixture execution request failed")
            if execution == "invalid":
                return "not JSON"
            return json.dumps(
                {
                    "plan_steps": [
                        {
                            "title": "Read evidence",
                            "objective": "Read evidence",
                            "suggested_tools": ["get_research_findings"],
                        }
                    ]
                }
            )
        if phase == "critic":
            if critic == "error":
                raise RuntimeError("fixture critic request failed")
            return (
                ""
                if critic == "empty"
                else json.dumps({"trajectory_assessment": "Continue"})
            )
        if replan_error:
            raise RuntimeError("fixture replan request failed")
        return "not JSON"  # The real replanner retains its prior plan.

    executor.llm_service.generate_response = AsyncMock(side_effect=generate)
    executor.observation_service.observe = AsyncMock(return_value={"context": []})
    # Force both optional stages to exercise each adapter boundary independently
    # of the critic's scheduling policy and replanner's trigger heuristics.
    executor._should_run_critic = lambda job, state: True
    executor.planner.evaluate_replan_triggers = lambda *args, **kwargs: "fixture"
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
    return adapter, calls


@pytest.mark.parametrize("remaining", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("causal", ["valid", "invalid", "error"])
async def test_each_observation_step_rechecks_budget_and_counts_fallback_attempts(
    remaining, causal
):
    adapter, calls = _adapter(remaining, causal=causal)

    observation = await adapter.observe_phase()

    assert observation == {"context": []}
    assert calls == ["causal_plan", "execution_plan", "critic", "replan"][:remaining]
    assert adapter.job.llm_calls_used == 2 + len(calls)
    assert adapter.job.llm_calls_used <= adapter.job.max_llm_calls
    assert adapter.state["findings"] == [
        {"type": "dynamic_profile", "title": "retained"}
    ]
    if remaining:
        assert adapter.state["causal_experiment_plan"]["source"] == (
            "llm" if causal == "valid" else "fallback"
        )


async def test_disabled_causal_planning_does_not_consume_a_slot():
    adapter, calls = _adapter(1)
    adapter.job.config["causal_experiment_planner_enabled"] = False

    await adapter.observe_phase()

    assert calls == ["execution_plan"]
    assert adapter.job.llm_calls_used == 3
    assert "causal_plan_generation_attempted" not in adapter.state


@pytest.mark.parametrize("critic", ["empty", "error"])
async def test_unusable_critic_result_still_spends_its_attempt_and_blocks_replanning(
    critic,
):
    adapter, calls = _adapter(3, critic=critic)

    await adapter.observe_phase()

    assert calls == ["causal_plan", "execution_plan", "critic"]
    assert adapter.job.llm_calls_used == 5
    assert not adapter.state.get("critic_notes")


async def test_replan_fallback_spends_one_slot_and_retains_usable_steps():
    adapter, calls = _adapter(4, replan_error=True)

    await adapter.observe_phase()

    assert calls == ["causal_plan", "execution_plan", "critic", "replan"]
    assert adapter.job.llm_calls_used == 6
    assert adapter.state["execution_plan"]
    assert adapter.state["execution_plan_version"] == 2


@pytest.mark.parametrize("execution", ["invalid", "error"])
async def test_execution_fallback_retains_a_plan_without_authorizing_another_call(
    execution,
):
    adapter, calls = _adapter(2, execution=execution)

    await adapter.observe_phase()

    assert calls == ["causal_plan", "execution_plan"]
    assert adapter.job.llm_calls_used == 4
    assert adapter.state["execution_plan"]
    assert not adapter.state.get("critic_notes")


@pytest.mark.parametrize("critic_enabled", [False, True])
async def test_cached_plans_leave_the_remaining_slot_for_later_work(critic_enabled):
    adapter, calls = _adapter(1)
    causal = {"source": "retained", "experiments": [{"id": "existing"}]}
    adapter.state["causal_experiment_plan"] = causal
    adapter.state["execution_plan"] = adapter.executor._annotate_execution_plan_graph(
        adapter.executor._fallback_execution_plan(adapter.job)
    )
    adapter.executor._should_run_critic = lambda job, state: critic_enabled

    await adapter.observe_phase()

    assert calls == (["critic"] if critic_enabled else ["replan"])
    assert adapter.job.llm_calls_used == 3
    assert adapter.state["causal_experiment_plan"] == causal
    assert "causal_plan_generation_attempted" not in adapter.state
    assert "plan_generation_attempted" not in adapter.state


async def test_real_observation_fallback_exhausts_budget_before_thinking():
    adapter, calls = _adapter(1, causal="error")
    thinking = AsyncMock(side_effect=AssertionError("No model allowance remains"))
    adapter.executor.thinking_service.think = thinking
    adapter.executor._finalize_job = AsyncMock(return_value={"finalized": True})

    result = await AgentRuntimeRunner().run(adapter)

    assert result == {"finalized": True}
    assert calls == ["causal_plan"]
    assert adapter.job.llm_calls_used == adapter.job.max_llm_calls == 3
    assert adapter.state["causal_experiment_plan"]["source"] == "fallback"
    assert adapter.state["findings"][0]["title"] == "retained"
    thinking.assert_not_awaited()
