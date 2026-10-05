"""The context window tools: compress_history, summarize_findings.

These call the real handlers. The file used to restate each handler's logic
inline and assert on the restatement -- `state = {}; state["compressed_history"]
= "..."; assert "compressed_history" in state` -- so twenty-four tests passed
whatever the tools did, including while an empty model reply made
compress_history throw away the history it was asked to preserve.

The executor is the real `AutonomousAgentExecutor`; only the model call is
replaced. `FakeLLM` binds every call against the real
`LLMService.generate_response` signature, so a keyword the service does not
accept fails here rather than in a run.

`compress_history` shares a contract with automatic compaction
(`agent_context_compaction`): both leave `state["compressed_history"]` and a
trimmed `state["actions_taken"]`, and the thinking prompt reads the former.
`TestTheContractSharedWithAutomaticCompaction` runs the two side by side.
"""

import copy
import inspect
from types import SimpleNamespace

import pytest

from app.models.memory import UserPreferences
from app.services.agent_context_compaction import (
    _SUMMARY_MAX_CHARS,
    AgentContextCompactionService,
)
from app.services.agent_runtime_state_service import initialize_runtime_state
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_observability_provider,
)
from app.services.agent_tools import AGENT_TOOLS, get_tool_by_name
from app.services.autonomous_agent_executor import AutonomousAgentExecutor
from app.services.llm_service import LLMService
from app.services.tool_registry import get_tool_metadata

pytestmark = pytest.mark.unit

TOOLS = ("compress_history", "summarize_findings")

_GENERATE_RESPONSE = inspect.signature(LLMService.generate_response)


class FakeLLM:
    """The model: answers with `reply`, or raises `error`."""

    def __init__(self, reply="A summary.", error=None):
        self.reply = reply
        self.error = error
        self.calls = []

    async def generate_response(self, *args, **kwargs):
        _GENERATE_RESPONSE.bind(self, *args, **kwargs)
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.reply

    @property
    def prompt(self):
        return self.calls[-1]["user_message"]


def _executor(reply="A summary.", error=None):
    executor = AutonomousAgentExecutor()
    executor.llm_service = FakeLLM(reply, error)
    return executor


def _job(**fields):
    fields.setdefault("id", "job-1")
    fields.setdefault("user_id", None)
    fields.setdefault("config", {})
    fields.setdefault("iteration", 9)
    fields.setdefault("add_log_entry", lambda entry: None)
    return SimpleNamespace(**fields)


async def _run(tool, params, state, executor, job=None, db=None):
    provider = build_autonomous_observability_provider(executor)
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=db,
        service=None,
        user_id="u",
        job=job or _job(),
        state=state,
    )
    return await provider._handlers[tool](params, ctx)


def _state(**overrides):
    """The state a real run starts with."""
    state = initialize_runtime_state()
    state.update(overrides)
    return state


def _actions(count, tool="search_documents"):
    """Actions shaped the way the act phase appends them."""
    return [
        {
            "action": {"tool": f"{tool}_{i}", "params": {}},
            "result": {"success": True, "data": {"total": i, "results": []}},
            "iteration": i,
            "node": "act",
            "step_id": f"step_{i}",
        }
        for i in range(count)
    ]


def _iterations(state):
    return [a["iteration"] for a in state["actions_taken"]]


class TestCompressHistory:
    async def test_a_fresh_run_has_nothing_to_compress(self):
        executor = _executor()
        state = _state()
        result = await _run("compress_history", {}, state, executor)
        assert result["success"] is True
        assert result["data"]["actions_count"] == 0
        assert "compressed_actions" not in result["data"]
        assert executor.llm_service.calls == []
        assert "compressed_history" not in state

    async def test_history_no_longer_than_keep_last_is_left_alone(self):
        executor = _executor()
        state = _state(actions_taken=_actions(5))
        before = copy.deepcopy(state)
        result = await _run("compress_history", {}, state, executor)
        assert result["data"]["actions_count"] == 5
        assert executor.llm_service.calls == []
        assert state == before

    async def test_the_default_keeps_the_last_five(self):
        executor = _executor("Searched three times.")
        state = _state(actions_taken=_actions(8))
        result = await _run("compress_history", {}, state, executor)
        assert result == {
            "success": True,
            "data": {
                "compressed_actions": 3,
                "kept_actions": 5,
                "summary_length": len("Searched three times."),
            },
        }
        assert _iterations(state) == [3, 4, 5, 6, 7]
        assert state["compressed_history"] == "Searched three times."

    async def test_keep_last_is_honoured(self):
        executor = _executor()
        state = _state(actions_taken=_actions(10))
        result = await _run("compress_history", {"keep_last": 3}, state, executor)
        assert result["data"]["compressed_actions"] == 7
        assert result["data"]["kept_actions"] == 3
        assert _iterations(state) == [7, 8, 9]

    async def test_keep_last_is_capped_at_twenty(self):
        executor = _executor()
        state = _state(actions_taken=_actions(30))
        result = await _run("compress_history", {"keep_last": 50}, state, executor)
        assert result["data"]["kept_actions"] == 20
        assert result["data"]["compressed_actions"] == 10
        assert _iterations(state) == list(range(10, 30))

    async def test_a_numeric_string_keep_last_is_a_number(self):
        executor = _executor()
        state = _state(actions_taken=_actions(10))
        result = await _run("compress_history", {"keep_last": "2"}, state, executor)
        assert result["data"]["kept_actions"] == 2

    async def test_a_keep_last_that_is_not_a_number_is_refused(self):
        executor = _executor()
        state = _state(actions_taken=_actions(10))
        before = copy.deepcopy(state)
        result = await _run("compress_history", {"keep_last": "lots"}, state, executor)
        assert "success" not in result
        assert "error" in result
        assert executor.llm_service.calls == []
        assert state == before

    async def test_keep_last_zero_compresses_everything(self):
        executor = _executor()
        state = _state(actions_taken=_actions(8))
        result = await _run("compress_history", {"keep_last": 0}, state, executor)
        assert result["data"]["compressed_actions"] == 8
        assert result["data"]["kept_actions"] == 0
        assert state["actions_taken"] == []

    async def test_the_model_is_shown_the_compressed_actions_and_not_the_kept(self):
        executor = _executor()
        actions = _actions(8)
        actions[1]["result"] = {"success": False, "error": "upstream answered 406"}
        state = _state(actions_taken=actions)
        await _run("compress_history", {}, state, executor)
        prompt = executor.llm_service.prompt
        for i in range(3):
            assert f"search_documents_{i}" in prompt
        for i in range(3, 8):
            assert f"search_documents_{i}" not in prompt
        assert "failed: upstream answered 406" in prompt
        assert "success, data keys: ['total', 'results']" in prompt

    async def test_an_earlier_summary_is_carried_into_the_next(self):
        executor = _executor("Both rounds, merged.")
        state = _state(
            actions_taken=_actions(8), compressed_history="Round one found nothing."
        )
        await _run("compress_history", {}, state, executor)
        assert "Round one found nothing." in executor.llm_service.prompt
        assert state["compressed_history"] == "Both rounds, merged."

    async def test_the_summary_is_capped_at_two_thousand_characters(self):
        executor = _executor("X" * 5000)
        state = _state(actions_taken=_actions(8))
        result = await _run("compress_history", {}, state, executor)
        assert len(state["compressed_history"]) == 2000
        assert result["data"]["summary_length"] == 2000

    async def test_the_call_is_attributed_to_the_job(self):
        executor = _executor()
        db = object()
        state = _state(actions_taken=_actions(8))
        await _run("compress_history", {}, state, executor, _job(iteration=4), db)
        call = executor.llm_service.calls[-1]
        assert call["db"] is db
        assert call["snapshot_context"] == {
            "job_id": "job-1",
            "iteration": 4,
            "phase": "tool:compress_history",
        }

    async def test_a_model_failure_is_reported_and_loses_nothing(self):
        executor = _executor(error=RuntimeError("provider timed out"))
        state = _state(actions_taken=_actions(8), compressed_history="Earlier summary.")
        before = copy.deepcopy(state)
        result = await _run("compress_history", {}, state, executor)
        assert "success" not in result
        assert "provider timed out" in result["error"]
        assert state == before

    @pytest.mark.parametrize("reply", ["", "   ", None])
    async def test_an_empty_model_reply_does_not_erase_the_history(self, reply):
        executor = _executor(reply)
        state = _state(actions_taken=_actions(8), compressed_history="Earlier summary.")
        result = await _run("compress_history", {}, state, executor)
        kept_everything = len(state["actions_taken"]) == 8
        assert "error" in result or state["compressed_history"].strip()
        assert kept_everything or state["compressed_history"].strip()
        assert "Earlier summary." in state["compressed_history"] or kept_everything


class TestTheContractSharedWithAutomaticCompaction:
    """`compress_history` and `maybe_compact` must leave the same state."""

    def _auto_job(self):
        return _job(
            config={
                "auto_compaction": {
                    "enabled": True,
                    "threshold_chars": 1000,
                    "keep_recent_actions": 5,
                    "min_iterations_between": 1,
                }
            }
        )

    def _big_state(self, **overrides):
        actions = _actions(12)
        for action in actions:
            action["result"]["data"]["results"] = ["x" * 200]
        return _state(actions_taken=actions, **overrides)

    async def test_both_paths_leave_the_same_state(self):
        tool_state = self._big_state(compressed_history="Earlier summary.")
        auto_state = copy.deepcopy(tool_state)

        tool_executor = _executor("Merged summary.")
        await _run("compress_history", {"keep_last": 5}, tool_state, tool_executor)

        auto_executor = _executor("Merged summary.")
        compacted = await AgentContextCompactionService().maybe_compact(
            auto_executor, self._auto_job(), auto_state, None
        )
        assert compacted is True

        assert tool_state["compressed_history"] == auto_state["compressed_history"]
        assert tool_state["actions_taken"] == auto_state["actions_taken"]
        assert isinstance(tool_state["compressed_history"], str)

    async def test_both_paths_describe_the_actions_to_the_model_identically(self):
        actions = _actions(8)
        actions[2]["result"] = {"success": False, "error": "refused: bad argument"}
        state = _state(actions_taken=actions)
        executor = _executor()
        await _run("compress_history", {}, state, executor)
        digest = AgentContextCompactionService._build_actions_digest(actions[:3])
        assert digest
        assert digest in executor.llm_service.prompt

    async def test_both_paths_cap_the_summary_at_the_same_length(self):
        executor = _executor("X" * 5000)
        state = _state(actions_taken=_actions(8))
        await _run("compress_history", {}, state, executor)
        assert len(state["compressed_history"]) == _SUMMARY_MAX_CHARS

    async def test_the_thinking_prompt_reads_what_the_tool_writes(self):
        executor = _executor("The agent searched and found three papers.")
        state = _state(actions_taken=_actions(8))
        job = _job(goal="a goal", job_type="research", max_iterations=100)
        await _run("compress_history", {}, state, executor, job)
        prompt = executor._build_thinking_prompt_volatile(job, state)
        assert "COMPRESSED HISTORY" in prompt
        assert "The agent searched and found three papers." in prompt

    async def test_the_tool_is_stricter_than_the_automatic_path_on_failure(self):
        """A failed model call: the automatic path degrades to its digest (it
        must not block the iteration); the tool reports the failure and
        leaves the history as it was, which loses nothing either."""
        error = RuntimeError("provider timed out")
        tool_state = self._big_state()
        auto_state = copy.deepcopy(tool_state)
        before = copy.deepcopy(tool_state)

        result = await _run("compress_history", {}, tool_state, _executor(error=error))
        assert "provider timed out" in result["error"]
        assert tool_state == before

        await AgentContextCompactionService().maybe_compact(
            _executor(error=error), self._auto_job(), auto_state, None
        )
        assert "search_documents_0" in auto_state["compressed_history"]
        assert len(auto_state["actions_taken"]) == 5


def _findings():
    return [
        {
            "id": "f1",
            "title": "Finding A",
            "content": "Tables beat branches.",
            "category": "key_insight",
            "confidence": 0.8,
        },
        {
            "id": "f2",
            "title": "Finding B",
            "content": "Measured at 15 trials.",
            "category": "methodology",
            "confidence": 0.7,
        },
        {
            "id": "f3",
            "title": "Finding C",
            "content": "Noise was 34 percent.",
            "category": "key_insight",
            "confidence": 0.9,
        },
    ]


class TestSummarizeFindings:
    async def test_a_fresh_run_has_nothing_to_summarize(self):
        executor = _executor()
        result = await _run(
            "summarize_findings", {"consolidate": True}, _state(), executor
        )
        assert result == {
            "success": True,
            "data": {"message": "No findings to summarize", "count": 0},
        }
        assert executor.llm_service.calls == []

    async def test_a_category_nothing_has_is_nothing_to_summarize(self):
        executor = _executor()
        state = _state(findings=_findings())
        before = copy.deepcopy(state)
        result = await _run(
            "summarize_findings",
            {"category": "absent", "consolidate": True},
            state,
            executor,
        )
        assert result["data"]["count"] == 0
        assert executor.llm_service.calls == []
        assert state == before

    async def test_it_returns_the_synthesis_and_leaves_the_findings_alone(self):
        executor = _executor("Two themes emerge.")
        state = _state(findings=_findings())
        before = copy.deepcopy(state)
        result = await _run("summarize_findings", {}, state, executor)
        assert result == {
            "success": True,
            "data": {
                "synthesis": "Two themes emerge.",
                "findings_summarized": 3,
                "consolidated": False,
            },
        }
        assert state == before
        assert executor._job_findings == {}

    async def test_the_model_is_shown_every_finding(self):
        executor = _executor()
        await _run("summarize_findings", {}, _state(findings=_findings()), executor)
        prompt = executor.llm_service.prompt
        assert "[key_insight] Finding A: Tables beat branches." in prompt
        assert "[methodology] Finding B: Measured at 15 trials." in prompt
        assert "[key_insight] Finding C: Noise was 34 percent." in prompt

    async def test_a_category_filter_shows_only_that_category(self):
        executor = _executor()
        result = await _run(
            "summarize_findings",
            {"category": "key_insight"},
            _state(findings=_findings()),
            executor,
        )
        prompt = executor.llm_service.prompt
        assert result["data"]["findings_summarized"] == 2
        assert "Finding A" in prompt and "Finding C" in prompt
        assert "Finding B" not in prompt

    async def test_long_content_is_cut_to_three_hundred_characters(self):
        executor = _executor()
        finding = {"title": "Long", "category": "c", "content": "Y" * 500 + "TAIL"}
        await _run("summarize_findings", {}, _state(findings=[finding]), executor)
        prompt = executor.llm_service.prompt
        assert "Y" * 300 in prompt
        assert "Y" * 301 not in prompt
        assert "TAIL" not in prompt

    async def test_the_synthesis_is_capped_at_three_thousand_characters(self):
        executor = _executor("Z" * 5000)
        result = await _run(
            "summarize_findings", {}, _state(findings=_findings()), executor
        )
        assert len(result["data"]["synthesis"]) == 3000

    async def test_the_call_is_attributed_to_the_job(self):
        executor = _executor()
        db = object()
        await _run(
            "summarize_findings",
            {},
            _state(findings=_findings()),
            executor,
            _job(iteration=6),
            db,
        )
        call = executor.llm_service.calls[-1]
        assert call["db"] is db
        assert call["snapshot_context"] == {
            "job_id": "job-1",
            "iteration": 6,
            "phase": "tool:summarize_findings",
        }

    async def test_consolidating_everything_returns_one_synthesis_finding(self):
        executor = _executor("Two themes emerge.")
        state = _state(findings=_findings())
        result = await _run(
            "summarize_findings", {"consolidate": True}, state, executor
        )
        assert result["data"]["consolidated"] is True
        assert result["data"]["findings_summarized"] == 3
        (finding,) = result["findings"]
        assert finding["category"] == "synthesis"
        assert finding["content"] == "Two themes emerge."
        assert "3 items" in finding["title"]
        assert finding["id"] and finding["created_at"]
        assert 0 <= finding["confidence"] <= 1
        originals = [f for f in state["findings"] if f.get("category") != "synthesis"]
        assert originals == []
        assert executor._job_findings["job-1"] == [finding]

    async def test_consolidating_a_category_keeps_the_others(self):
        executor = _executor("Insights, merged.")
        state = _state(findings=_findings())
        result = await _run(
            "summarize_findings",
            {"consolidate": True, "category": "key_insight"},
            state,
            executor,
        )
        (finding,) = result["findings"]
        assert "key_insight" in finding["title"]
        assert "2 items" in finding["title"]
        kept = [f["id"] for f in state["findings"] if f.get("category") != "synthesis"]
        assert kept == ["f2"]

    async def test_consolidating_keeps_the_evidence_a_contract_counts(self):
        """A finding with a `type` is evidence. Goal contracts count them in
        this same list, so folding them into one untyped synthesis let a run
        un-satisfy a contract it had already met."""
        executor = _executor("One measurement, two remarks.")
        measured = {
            "id": "m1",
            "type": "dynamic_profile",
            "title": "Profile of kernel A",
            "content": "62% of cycles in the inner loop",
            "category": "result",
        }
        state = _state(findings=[measured] + _findings())

        result = await _run(
            "summarize_findings", {"consolidate": True}, state, executor
        )

        assert result["data"]["findings_summarized"] == 4
        assert result["data"]["findings_folded"] == 3
        assert result["data"]["typed_findings_kept"] == 1
        assert state["findings"] == [measured]
        assert [f["category"] for f in result["findings"]] == ["synthesis"]

    async def test_the_consolidated_finding_is_recorded_once(self):
        executor = _executor("Two themes emerge.")
        state = _state(findings=_findings())
        result = await _run(
            "summarize_findings", {"consolidate": True}, state, executor
        )
        # What the act phase does with every tool result that carries findings.
        state["findings"].extend(result["findings"])
        syntheses = [f for f in state["findings"] if f.get("category") == "synthesis"]
        assert len(syntheses) == 1

    async def test_a_model_failure_is_reported_and_loses_nothing(self):
        executor = _executor(error=RuntimeError("provider timed out"))
        state = _state(findings=_findings())
        before = copy.deepcopy(state)
        result = await _run(
            "summarize_findings", {"consolidate": True}, state, executor
        )
        assert "success" not in result
        assert "provider timed out" in result["error"]
        assert state == before
        assert executor._job_findings == {}

    @pytest.mark.parametrize("reply", ["", "   ", None])
    async def test_an_empty_synthesis_does_not_replace_the_findings(self, reply):
        executor = _executor(reply)
        state = _state(findings=_findings())
        await _run("summarize_findings", {"consolidate": True}, state, executor)
        assert [f["id"] for f in state["findings"]][:3] == ["f1", "f2", "f3"]


class TestTheModelCallUsesTheOwnersSettings:
    @pytest.mark.parametrize("tool", TOOLS)
    async def test_the_owners_llm_preferences_reach_the_model_call(
        self, tool, db_session, test_user
    ):
        db_session.add(UserPreferences(user_id=test_user.id, llm_model="owner-model"))
        await db_session.commit()

        executor = _executor()
        state = _state(actions_taken=_actions(8), findings=_findings())
        job = _job(user_id=test_user.id)
        result = await _run(tool, {}, state, executor, job, db_session)
        assert result.get("success") is True, result

        settings = executor.llm_service.calls[-1]["user_settings"]
        assert settings is not None
        assert settings.model == "owner-model"


class TestContextManagementSchemas:
    def test_schemas_exist(self):
        names = {t["name"] for t in AGENT_TOOLS}
        assert set(TOOLS) <= names

    @pytest.mark.parametrize("name", TOOLS)
    def test_nothing_is_required(self, name):
        tool = get_tool_by_name(name)
        assert tool is not None
        assert tool["parameters"].get("required", []) == []

    def test_compress_history_declares_keep_last(self):
        tool = get_tool_by_name("compress_history")
        assert set(tool["parameters"]["properties"]) == {"keep_last"}

    def test_summarize_findings_declares_consolidate_and_category(self):
        tool = get_tool_by_name("summarize_findings")
        assert set(tool["parameters"]["properties"]) == {"consolidate", "category"}

    def test_one_provider_answers_both(self):
        provider = build_autonomous_observability_provider(SimpleNamespace())
        assert set(TOOLS) <= set(provider._handlers)


class TestContextManagementRegistry:
    @pytest.mark.parametrize("name", TOOLS)
    def test_classification(self, name):
        meta = get_tool_metadata(name)
        assert meta is not None
        # "read" here follows the registry's convention that rewriting the
        # run's own state (as capture_snapshot and switch_strategy also do)
        # is not an external effect.
        assert meta.effects == "read"
        assert meta.cost_tier == "medium"
        assert meta.network == "none"
