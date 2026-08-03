"""Cache-friendliness tests for the thinking prompt stable/volatile split.

The stable prompt must stay byte-identical across iterations (it keys the
provider prompt cache); all per-iteration context must land in the volatile
part instead.
"""

from uuid import uuid4

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.autonomous_agent_executor import AutonomousAgentExecutor


def _make_job(config=None) -> AgentJob:
    return AgentJob(
        name="Prompt Split Test",
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


def test_stable_prompt_is_byte_stable_across_iteration_state_changes():
    executor = AutonomousAgentExecutor()
    job = _make_job()

    state_iter_1 = {"goal_progress": 10}
    state_iter_5 = {
        "goal_progress": 60,
        "focus_directive": "focus on reranking benchmarks",
        "compressed_history": "iterations 1-4 searched the KB and found 3 docs",
        "critic_notes": [{"note": "consider arxiv"}],
    }

    stable_a = executor._build_thinking_prompt_stable(
        job, None, state_iter_1, profile={}
    )
    stable_b = executor._build_thinking_prompt_stable(
        job, None, state_iter_5, profile={}
    )

    assert stable_a == stable_b
    assert "GOAL:" in stable_a
    assert "RESPONSE FORMAT:" in stable_a
    # Volatile content must never leak into the stable prefix.
    assert "reranking benchmarks" not in stable_b
    assert "COMPRESSED HISTORY" not in stable_b


def test_volatile_prompt_carries_iteration_context():
    executor = AutonomousAgentExecutor()
    job = _make_job()
    state = {
        "focus_directive": "focus on reranking benchmarks",
        "compressed_history": "iterations 1-4 summary",
    }

    volatile = executor._build_thinking_prompt_volatile(job, state)

    assert volatile.startswith("CURRENT EXECUTION CONTEXT:")
    assert "reranking benchmarks" in volatile
    assert "COMPRESSED HISTORY" in volatile


def test_volatile_prompt_empty_when_no_iteration_context():
    executor = AutonomousAgentExecutor()
    assert executor._build_thinking_prompt_volatile(_make_job(), {}) == ""


def test_legacy_combined_prompt_is_stable_plus_volatile():
    executor = AutonomousAgentExecutor()
    job = _make_job()
    state = {"focus_directive": "check the KG"}

    stable = executor._build_thinking_prompt_stable(job, None, state, profile={})
    combined = executor._build_thinking_prompt(job, None, state, {}, profile={})

    assert combined.startswith(stable)
    assert "check the KG" in combined
