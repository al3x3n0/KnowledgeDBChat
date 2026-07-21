"""Tests for automatic context compaction (agent_context_compaction)."""

import asyncio
from types import SimpleNamespace

import pytest

from app.services.agent_context_compaction import (
    AgentContextCompactionService,
    context_compaction_service,
)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class FakeLLM:
    def __init__(self, response="Compacted narrative summary.", fail=False):
        self.response = response
        self.fail = fail
        self.calls = []

    async def generate_response(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("llm down")
        return self.response


def _executor(llm=None):
    async def _get_user_settings(user_id, db):
        return None

    return SimpleNamespace(
        llm_service=llm or FakeLLM(),
        _get_user_settings=_get_user_settings,
        _llm_routing_from_job_config=lambda config: {"llm_tier": "balanced"},
    )


def _job(iteration=10, config=None):
    log = []
    job = SimpleNamespace(
        id="job-1",
        iteration=iteration,
        config=config or {},
        user_id=None,
        add_log_entry=lambda entry: log.append(entry),
    )
    job.log = log
    return job


def _big_state(action_count=12, result_size=8000):
    return {
        "actions_taken": [
            {
                "action": {"tool": f"search_documents"},
                "result": {"success": True, "data": {"blob": "x" * result_size}},
                "iteration": i,
            }
            for i in range(action_count)
        ],
        "findings": [],
    }


class TestMaybeCompact:
    def test_compacts_when_over_threshold(self):
        llm = FakeLLM()
        executor = _executor(llm)
        job = _job()
        state = _big_state()

        compacted = run(context_compaction_service.maybe_compact(executor, job, state, None))

        assert compacted is True
        assert state["compressed_history"] == "Compacted narrative summary."
        assert len(state["actions_taken"]) == 5  # keep_recent default
        assert state["auto_compaction_last"]["iteration"] == 10
        assert state["auto_compaction_last"]["compacted_actions"] == 7
        assert job.log[0]["phase"] == "auto_compaction"
        # Summary call uses fast tier and the summarization task type.
        assert llm.calls[0]["routing"]["llm_tier"] == "fast"
        assert llm.calls[0]["task_type"] == "summarization"

    def test_below_threshold_skips(self):
        executor = _executor()
        state = _big_state(action_count=12, result_size=10)  # tiny
        compacted = run(context_compaction_service.maybe_compact(executor, _job(), state, None))
        assert compacted is False
        assert "compressed_history" not in state

    def test_too_few_actions_skips(self):
        executor = _executor()
        state = _big_state(action_count=4, result_size=50000)
        compacted = run(context_compaction_service.maybe_compact(executor, _job(), state, None))
        assert compacted is False

    def test_cooldown_between_compactions(self):
        executor = _executor()
        state = _big_state()
        state["auto_compaction_last"] = {"iteration": 9}
        compacted = run(
            context_compaction_service.maybe_compact(executor, _job(iteration=10), state, None)
        )
        assert compacted is False
        # After the cooldown window it runs again.
        compacted = run(
            context_compaction_service.maybe_compact(executor, _job(iteration=12), state, None)
        )
        assert compacted is True

    def test_disabled_via_job_config(self):
        executor = _executor()
        state = _big_state()
        job = _job(config={"auto_compaction": False})
        compacted = run(context_compaction_service.maybe_compact(executor, job, state, None))
        assert compacted is False

    def test_llm_failure_falls_back_to_digest(self):
        executor = _executor(FakeLLM(fail=True))
        state = _big_state()
        state["compressed_history"] = "prior summary"

        compacted = run(context_compaction_service.maybe_compact(executor, _job(), state, None))

        assert compacted is True
        assert "prior summary" in state["compressed_history"]
        assert "search_documents" in state["compressed_history"]
        assert len(state["actions_taken"]) == 5

    def test_existing_summary_included_in_prompt(self):
        llm = FakeLLM()
        executor = _executor(llm)
        state = _big_state()
        state["compressed_history"] = "earlier compact summary"

        run(context_compaction_service.maybe_compact(executor, _job(), state, None))

        assert "earlier compact summary" in llm.calls[0]["user_message"]


class TestConfigResolution:
    def test_defaults(self):
        cfg = context_compaction_service.resolve_config(SimpleNamespace(config={}))
        assert cfg["enabled"] is True
        assert cfg["threshold_chars"] == 60000
        assert cfg["keep_recent_actions"] == 5
        assert cfg["min_iterations_between"] == 3

    def test_job_dict_override(self):
        cfg = context_compaction_service.resolve_config(
            SimpleNamespace(
                config={
                    "auto_compaction": {
                        "threshold_chars": 9000,
                        "keep_recent_actions": 2,
                        "min_iterations_between": 1,
                    }
                }
            )
        )
        assert cfg == {
            "enabled": True,
            "threshold_chars": 9000,
            "keep_recent_actions": 2,
            "min_iterations_between": 1,
        }


class TestEstimate:
    def test_counts_state_payloads(self):
        estimate = AgentContextCompactionService.estimate_context_chars(
            {
                "actions_taken": [{"a": "x" * 100}],
                "findings": [{"f": "y" * 50}],
                "compressed_history": "z" * 30,
            }
        )
        assert estimate > 180

    def test_empty_state(self):
        assert AgentContextCompactionService.estimate_context_chars({}) == 0
