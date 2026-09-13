"""Context a run is handed before it starts.

A run launched from a chat answer should not spend its first iterations
rediscovering what the corpus already said. It should also not mistake what it
was handed for something it measured — that distinction is the whole point of
the goal contract, and this is the one place a run is given numbers it did not
produce.
"""

import uuid

import pytest

from app.models.agent_job import AgentJob
from app.services.autonomous_agent_executor import AutonomousAgentExecutor

pytestmark = pytest.mark.unit


def _job(config=None, goal="Measure the INT8 attention ceiling"):
    return AgentJob(
        id=uuid.uuid4(),
        name="throughput study",
        goal=goal,
        job_type="experiment",
        user_id=uuid.uuid4(),
        status="pending",
        config=config,
    )


@pytest.fixture
def executor():
    return AutonomousAgentExecutor()


class TestStartingContextReading:
    def test_absent_config_gives_no_context(self, executor):
        assert executor._starting_context_for_prompt(_job(None)) == ""
        assert executor._starting_context_for_prompt(_job({})) == ""

    def test_blank_and_whitespace_are_no_context(self, executor):
        # An empty block in the prompt would read as "nothing is known", which
        # is a claim rather than the absence of one.
        assert (
            executor._starting_context_for_prompt(_job({"starting_context": ""})) == ""
        )
        assert (
            executor._starting_context_for_prompt(_job({"starting_context": "   \n  "}))
            == ""
        )

    def test_a_non_string_is_ignored_rather_than_stringified(self, executor):
        # str({...}) of a dict in a prompt is noise the model would try to read.
        assert (
            executor._starting_context_for_prompt(_job({"starting_context": {"a": 1}}))
            == ""
        )
        assert (
            executor._starting_context_for_prompt(_job({"starting_context": 42})) == ""
        )

    def test_it_is_truncated_at_a_fixed_length(self, executor):
        long_text = "x" * (AutonomousAgentExecutor.STARTING_CONTEXT_MAX_CHARS + 500)
        out = executor._starting_context_for_prompt(
            _job({"starting_context": long_text})
        )
        assert out.endswith("[...truncated]")
        assert len(out) < len(long_text)

    def test_truncation_is_deterministic(self, executor):
        # The stable prompt half keys the provider's cache, so the same job
        # must render identically every iteration. A summarising truncation
        # would not.
        long_text = "y" * (AutonomousAgentExecutor.STARTING_CONTEXT_MAX_CHARS + 500)
        job = _job({"starting_context": long_text})
        assert executor._starting_context_for_prompt(job) == (
            executor._starting_context_for_prompt(job)
        )


class TestStartingContextInThePrompt:
    def test_the_prompt_carries_it_under_a_heading(self, executor):
        job = _job(
            {
                "starting_context": "An earlier run measured 3.81 GB/s at four chains.",
            }
        )
        prompt = executor._build_thinking_prompt_stable(job, None, {})

        assert "ALREADY ESTABLISHED, BEFORE THIS RUN:" in prompt
        assert "3.81 GB/s" in prompt

    def test_it_says_the_context_is_not_this_runs_measurement(self, executor):
        job = _job({"starting_context": "Someone reported 3.81 GB/s."})
        prompt = executor._build_thinking_prompt_stable(job, None, {})

        # Without this the run can settle a prediction with a number it was
        # handed, which is the failure the contract exists to prevent.
        assert "not as a measurement of your own" in prompt

    def test_a_job_without_context_gets_no_heading(self, executor):
        prompt = executor._build_thinking_prompt_stable(_job(None), None, {})

        assert "ALREADY ESTABLISHED" not in prompt
        assert "Measure the INT8 attention ceiling" in prompt

    def test_the_prompt_is_byte_stable_across_calls(self, executor):
        job = _job({"starting_context": "Prior finding f-104."})
        assert executor._build_thinking_prompt_stable(job, None, {}) == (
            executor._build_thinking_prompt_stable(job, None, {})
        )
