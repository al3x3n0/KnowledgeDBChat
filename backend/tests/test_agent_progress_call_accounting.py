"""Deterministic progress must not spend the run's remaining model budget."""

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


def _adapter(*, job_type="analysis", used=0, limit=3):
    executor = AutonomousAgentExecutor()
    job = AgentJob(
        id=uuid4(),
        user_id=uuid4(),
        name="Progress accounting",
        goal="Read the research evidence",
        job_type=job_type,
        status="running",
        config={"goal_contract": {"enabled": False}, "stall_detection_enabled": False},
        iteration=0,
        tool_calls_used=0,
        llm_calls_used=used,
        max_iterations=10,
        max_tool_calls=10,
        max_llm_calls=limit,
        max_runtime_minutes=5,
    )
    state = {
        "observations": [],
        "findings": [{"type": "dynamic_profile", "title": "kernel"}],
        "actions_taken": [],
        "goal_progress": 0,
    }
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
    return adapter


@pytest.mark.parametrize(
    "job_type", ["research", "analysis", "data_analysis", "custom"]
)
@pytest.mark.parametrize("used", [0, 2])
async def test_progress_scoring_preserves_model_budget_across_repeated_evaluations(
    job_type, used
):
    adapter = _adapter(job_type=job_type, used=used)
    for _ in range(3):
        result = await adapter.evaluate_phase(
            {}, {"action": None, "action_result": None}
        )
        assert (
            result["progress"] == adapter.state["goal_progress"] == adapter.job.progress
        )
        assert adapter.job.llm_calls_used == used
        assert adapter.job.can_continue()


@pytest.mark.parametrize("finish", [True, False], ids=["finishes", "budget_exhausted"])
async def test_a_resumed_loop_can_use_only_its_remaining_model_calls(finish):
    adapter = _adapter(used=2, limit=5)
    decisions = [
        {"goal_achieved": False, "should_stop": False},
        {"goal_achieved": False, "should_stop": False},
        {"goal_achieved": finish, "should_stop": False},
    ]
    thinking = AsyncMock(side_effect=decisions)
    adapter.executor.thinking_service.think = thinking
    adapter.observe_phase = AsyncMock(return_value={})
    adapter.act_phase = AsyncMock(return_value={"action": None, "action_result": None})
    adapter.on_iteration_complete = AsyncMock()
    adapter._close_instrument_bracket = AsyncMock()
    adapter.executor._finalize_job = AsyncMock(return_value={"finished": True})

    result = await AgentRuntimeRunner().run(adapter)

    assert result == {"finished": True}
    assert thinking.await_count == 3
    assert adapter.on_iteration_complete.await_count == (2 if finish else 3)
    assert adapter.job.llm_calls_used == 5
    assert (adapter.state["goal_progress"] == 100) is finish
    assert (
        any(e["phase"] == "goal_achieved" for e in adapter.job.execution_log or [])
        is finish
    )
    assert not adapter.job.can_continue()


async def test_contract_completion_still_runs_at_the_model_budget_limit():
    adapter = _adapter(used=3)
    adapter.job.config = {
        "goal_contract": {
            "enabled": True,
            "required_finding_types": ["dynamic_profile"],
        }
    }

    result = await adapter.evaluate_phase({}, {"action": None, "action_result": None})

    assert (
        result["should_stop"] and result["stop_reason"] == "goal_contract_autocomplete"
    )
    assert adapter.state["goal_contract_last"]["satisfied"]
    assert adapter.job.progress == 100
    assert adapter.job.llm_calls_used == 3
