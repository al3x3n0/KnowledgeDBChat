"""Tests for the native tool-calling loop and its think-phase glue."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services.agent_native_tool_loop import (
    NativeToolLoopService,
    native_tool_loop_service,
)
from app.services.agent_thinking_service import AgentThinkingService
from app.services.llm_providers.base import LLMCompletion, LLMToolCall


class FakeLLM:
    """Scripted generate_structured responses; records call kwargs."""

    def __init__(self, completions):
        self.completions = list(completions)
        self.calls = []

    async def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        completion = self.completions.pop(0)
        if isinstance(completion, Exception):
            raise completion
        return completion


def _tool_call(name="search_documents", arguments=None, call_id="tc1"):
    return LLMToolCall(id=call_id, name=name, arguments=arguments or {"query": "x"})


def _completion(text="", tool_calls=None, structured=None):
    return LLMCompletion(text=text, tool_calls=tool_calls or [], structured=structured)


SAMPLE_TOOLS = [
    {
        "name": "search_documents",
        "description": "search",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    }
]

DECISION = {"goal_achieved": False, "should_stop": False, "action": None}


def run(coro):
    return asyncio.run(coro)


class TestNativeToolLoopService:
    def test_tool_round_then_final_structured(self):
        llm = FakeLLM(
            [
                _completion(tool_calls=[_tool_call()]),
                _completion(text=json.dumps(DECISION), structured=DECISION),
            ]
        )
        executed = []

        async def execute_tool(name, params):
            executed.append((name, params))
            return {"success": True, "data": {"hits": 3}}

        result = run(
            native_tool_loop_service.run(
                llm_service=llm,
                messages=[{"role": "user", "content": "go"}],
                tools=SAMPLE_TOOLS,
                execute_tool=execute_tool,
                final_response_schema={"type": "object"},
            )
        )
        assert result.stop_reason == "completed"
        assert result.structured == DECISION
        assert result.tool_calls_executed == 1
        assert result.llm_calls_used == 2
        assert executed == [("search_documents", {"query": "x"})]
        # Second LLM call must see assistant tool_calls + tool result messages.
        second_messages = llm.calls[1]["messages"]
        roles = [m["role"] for m in second_messages]
        assert roles == ["user", "assistant", "tool"]
        assert "hits" in second_messages[2]["content"]

    def test_the_reasoning_goes_back_with_its_turn(self):
        """A thinking model keeps its chain of thought across tool calls."""
        thinking = _completion(tool_calls=[_tool_call()])
        thinking.reasoning = "Search first, then decide."
        llm = FakeLLM(
            [thinking, _completion(text=json.dumps(DECISION), structured=DECISION)]
        )

        async def execute_tool(name, params):
            return {"success": True}

        run(
            native_tool_loop_service.run(
                llm_service=llm,
                messages=[{"role": "user", "content": "go"}],
                tools=SAMPLE_TOOLS,
                execute_tool=execute_tool,
                final_response_schema={"type": "object"},
            )
        )
        assistant = llm.calls[1]["messages"][1]
        assert assistant["reasoning_content"] == "Search first, then decide."

        from app.services.llm_providers.openai_compatible import (
            OpenAICompatibleProvider,
        )

        (_, sent, _) = OpenAICompatibleProvider._convert_messages(
            llm.calls[1]["messages"]
        )
        assert sent["reasoning_content"] == "Search first, then decide."
        assert sent["tool_calls"][0]["function"]["name"] == "search_documents"

    def test_deferred_action_stops_without_execution(self):
        llm = FakeLLM([_completion(tool_calls=[_tool_call(name="delete_document")])])

        async def execute_tool(name, params):
            raise AssertionError("gated tool must not execute")

        result = run(
            native_tool_loop_service.run(
                llm_service=llm,
                messages=[{"role": "user", "content": "go"}],
                tools=SAMPLE_TOOLS,
                execute_tool=execute_tool,
                should_defer=lambda action: action["tool"] == "delete_document",
            )
        )
        assert result.stop_reason == "deferred_action"
        assert result.pending_action == {
            "tool": "delete_document",
            "params": {"query": "x"},
        }
        assert result.tool_calls_executed == 0

    def test_tool_budget_answers_all_calls_and_finalizes(self):
        llm = FakeLLM(
            [
                _completion(
                    tool_calls=[
                        _tool_call(call_id="a", arguments={"query": "1"}),
                        _tool_call(call_id="b", arguments={"query": "2"}),
                    ]
                ),
                _completion(structured=DECISION, text=json.dumps(DECISION)),
            ]
        )

        async def execute_tool(name, params):
            return {"success": True}

        result = run(
            native_tool_loop_service.run(
                llm_service=llm,
                messages=[{"role": "user", "content": "go"}],
                tools=SAMPLE_TOOLS,
                execute_tool=execute_tool,
                final_response_schema={"type": "object"},
                max_tool_calls=1,
            )
        )
        assert result.stop_reason == "max_tool_calls"
        assert result.tool_calls_executed == 1
        assert result.structured == DECISION
        # Both tool calls got a tool message (real + budget placeholder).
        final_messages = llm.calls[1]["messages"]
        tool_msgs = [m for m in final_messages if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        assert "budget" in tool_msgs[1]["content"].lower()
        # Finalization call carries the schema and no tools.
        assert llm.calls[1]["response_schema"] == {"type": "object"}
        assert llm.calls[1]["tools"] is None

    def test_repeated_identical_call_not_re_executed(self):
        llm = FakeLLM(
            [
                _completion(tool_calls=[_tool_call(call_id="a")]),
                _completion(tool_calls=[_tool_call(call_id="b")]),  # identical
                _completion(text="done", structured=None),
            ]
        )
        count = {"n": 0}

        async def execute_tool(name, params):
            count["n"] += 1
            return {"success": True}

        result = run(
            native_tool_loop_service.run(
                llm_service=llm,
                messages=[{"role": "user", "content": "go"}],
                tools=SAMPLE_TOOLS,
                execute_tool=execute_tool,
            )
        )
        assert count["n"] == 1
        assert result.tool_calls_executed == 1
        assert result.stop_reason == "completed"

    def test_tool_exception_fed_back_to_model(self):
        llm = FakeLLM(
            [
                _completion(tool_calls=[_tool_call()]),
                _completion(text="ok"),
            ]
        )

        async def execute_tool(name, params):
            raise RuntimeError("boom")

        result = run(
            native_tool_loop_service.run(
                llm_service=llm,
                messages=[{"role": "user", "content": "go"}],
                tools=SAMPLE_TOOLS,
                execute_tool=execute_tool,
            )
        )
        assert result.steps[0]["success"] is False
        assert "boom" in result.steps[0]["error"]
        tool_msg = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]
        assert "boom" in tool_msg["content"]

    def test_max_llm_calls_stops_loop(self):
        llm = FakeLLM(
            [
                _completion(tool_calls=[_tool_call(call_id="a", arguments={"q": "1"})]),
                _completion(tool_calls=[_tool_call(call_id="b", arguments={"q": "2"})]),
            ]
        )

        async def execute_tool(name, params):
            return {"success": True}

        result = run(
            native_tool_loop_service.run(
                llm_service=llm,
                messages=[{"role": "user", "content": "go"}],
                tools=SAMPLE_TOOLS,
                execute_tool=execute_tool,
                max_llm_calls=2,
            )
        )
        assert result.stop_reason == "max_llm_calls"
        assert result.llm_calls_used == 2

    @pytest.mark.parametrize("limit", [0, 1, 2])
    def test_schema_finalization_shares_the_hard_model_call_limit(self, limit):
        llm = FakeLLM(
            [
                _completion(tool_calls=[_tool_call(call_id="a", arguments={"q": "1"})]),
                _completion(tool_calls=[_tool_call(call_id="b", arguments={"q": "2"})]),
                _completion(structured=DECISION),
            ]
        )
        executed = []

        async def execute_tool(name, params):
            executed.append(params)
            return {"success": True}

        result = run(
            native_tool_loop_service.run(
                llm_service=llm,
                messages=[],
                tools=SAMPLE_TOOLS,
                execute_tool=execute_tool,
                final_response_schema={"type": "object"},
                max_llm_calls=limit,
            )
        )

        assert len(llm.calls) == result.llm_calls_used == limit
        assert len(executed) == limit
        assert result.stop_reason == "max_llm_calls"
        assert result.structured is None

    @pytest.mark.parametrize("fails", [False, True])
    def test_finalization_uses_and_counts_an_available_slot(self, fails):
        llm = FakeLLM(
            [
                _completion(text="A free-text answer"),
                RuntimeError("schema request failed")
                if fails
                else _completion(structured=DECISION),
            ]
        )

        async def execute_tool(name, params):
            raise AssertionError("No tool was requested")

        result = run(
            native_tool_loop_service.run(
                llm_service=llm,
                messages=[],
                tools=SAMPLE_TOOLS,
                execute_tool=execute_tool,
                final_response_schema={"type": "object"},
                max_llm_calls=2,
            )
        )

        assert len(llm.calls) == result.llm_calls_used == 2
        assert result.structured == (None if fails else DECISION)
        assert llm.calls[1]["tools"] is None
        assert llm.calls[1]["response_schema"] == {"type": "object"}

    def test_result_truncation(self):
        payload = NativeToolLoopService._serialize_result({"data": "x" * 10000}, 100)
        assert len(payload) <= 120
        assert payload.endswith("[truncated]")


class _FakeActionService:
    def __init__(self):
        self.calls = []

    async def act(self, executor, job, action, state, db):
        self.calls.append(action)
        return {"success": True, "findings": [{"f": 1}]}


def _fake_executor(llm):
    executor = SimpleNamespace()
    executor.llm_service = llm
    executor.action_service = _FakeActionService()
    executor._apply_default_scope_to_action = lambda action, job: action
    executor._evaluate_approval_checkpoint = lambda job, state, action: {
        "required": False
    }
    return executor


def _fake_job(**overrides):
    job = SimpleNamespace(
        config={"native_tool_loop": True},
        max_tool_calls=10,
        tool_calls_used=0,
        max_llm_calls=10,
        llm_calls_used=0,
        iteration=1,
        user_id=None,
    )
    for k, v in overrides.items():
        setattr(job, k, v)
    return job


class TestThinkingServiceNativeGlue:
    @pytest.mark.parametrize("remaining", [0, 1, 2])
    def test_exhausted_budget_returns_a_decision_without_prompted_fallback(
        self, remaining
    ):
        llm = FakeLLM(
            [
                _completion(tool_calls=[_tool_call(call_id="a", arguments={"q": "1"})]),
                _completion(tool_calls=[_tool_call(call_id="b", arguments={"q": "2"})]),
                _completion(structured=DECISION),
            ]
        )
        executor = _fake_executor(llm)
        job = _fake_job(max_llm_calls=5, llm_calls_used=5 - remaining)
        state = {"actions_taken": [], "findings": []}
        response = run(
            AgentThinkingService()._maybe_run_native_tool_loop(
                executor,
                job,
                state,
                system_prompt="s",
                user_message="u",
                available_tools=["search_documents"],
                user_settings=None,
                routing=None,
                db=None,
            )
        )

        decision = json.loads(response)
        assert decision["should_stop"] is True and decision["goal_achieved"] is False
        assert decision["action"] is None and "budget" in decision["stop_reason"]
        assert len(llm.calls) == remaining
        assert len(state["findings"]) == remaining
        assert state["native_tool_loop_last"]["llm_calls"] == remaining
        # The adapter accounts for one call only when it actually enters think.
        assert job.llm_calls_used == (4 if remaining else 5)

    @pytest.mark.parametrize(
        "text",
        [
            "not JSON",
            json.dumps(DECISION),
            "```json\n" + json.dumps(DECISION) + "\n```",
        ],
    )
    def test_final_text_at_the_cap_is_validated_without_a_repair_call(self, text):
        llm = FakeLLM([_completion(text=text)])
        job = _fake_job(max_llm_calls=1)
        response = run(
            AgentThinkingService()._maybe_run_native_tool_loop(
                _fake_executor(llm),
                job,
                {},
                system_prompt="s",
                user_message="u",
                available_tools=["search_documents"],
                user_settings=None,
                routing=None,
                db=None,
            )
        )

        from app.services.agent_decision_parser import extract_first_json_object

        decision = extract_first_json_object(response)
        assert len(llm.calls) == 1
        if text == "not JSON":
            assert decision["should_stop"] and "budget" in decision["stop_reason"]
        else:
            assert decision == DECISION

    @pytest.mark.parametrize("malformed_structured", [False, True])
    def test_think_does_not_generate_or_repair_after_native_budget_exhaustion(
        self, malformed_structured
    ):
        from app.services.agent_decision_parser import AgentDecisionParser

        llm = FakeLLM(
            [
                _completion(structured={"action": {"tool": []}})
                if malformed_structured
                else _completion(tool_calls=[_tool_call()])
            ]
        )
        executor = _fake_executor(llm)
        executor._resolve_agent_skill_profile = lambda job, state=None: {}
        executor._build_thinking_prompt = lambda *args, **kwargs: "system"
        executor._get_tools_for_job_type = lambda *args, **kwargs: ["search_documents"]
        executor._llm_routing_from_job_config = lambda config: None
        executor.decision_parser = AgentDecisionParser(llm)
        job = _fake_job(max_llm_calls=5, llm_calls_used=4)
        job.goal, job.job_type, job.max_iterations = "Read evidence", "analysis", 10
        state = {"actions_taken": [], "findings": []}
        service = AgentThinkingService()
        service._generate_decision_text = AsyncMock(
            side_effect=AssertionError("The native loop spent the final slot")
        )

        decision = run(service.think(executor, job, None, state, {}, None, None))

        assert decision["should_stop"] and not decision["goal_achieved"]
        assert "budget" in decision["stop_reason"]
        assert len(llm.calls) == 1
        service._generate_decision_text.assert_not_awaited()
        assert executor.decision_parser.metrics["parse_retry"] == 0
        assert executor.decision_parser.metrics["parse_repair"] == 0
        assert state["findings"] == ([] if malformed_structured else [{"f": 1}])

    def test_disabled_returns_none(self):
        service = AgentThinkingService()
        job = _fake_job(config={})
        result = run(
            service._maybe_run_native_tool_loop(
                _fake_executor(FakeLLM([])),
                job,
                {},
                system_prompt="s",
                user_message="u",
                available_tools=["search_documents"],
                user_settings=None,
                routing=None,
                db=None,
            )
        )
        assert result is None

    def test_native_loop_produces_decision_and_updates_budgets(self):
        llm = FakeLLM(
            [
                _completion(tool_calls=[_tool_call()]),
                _completion(structured=DECISION, text=json.dumps(DECISION)),
            ]
        )
        executor = _fake_executor(llm)
        service = AgentThinkingService()
        job = _fake_job()
        state = {"actions_taken": [], "findings": []}

        response = run(
            service._maybe_run_native_tool_loop(
                executor,
                job,
                state,
                system_prompt="system",
                user_message="decide",
                available_tools=["search_documents"],
                user_settings=None,
                routing=None,
                db=None,
            )
        )
        assert json.loads(response) == DECISION
        assert job.tool_calls_used == 1
        # Two loop LLM calls; adapter accounts for one, glue adds the extra.
        assert job.llm_calls_used == 1
        assert state["native_tool_loop_last"]["tool_calls"] == 1
        assert state["actions_taken"][0]["node"] == "native_think"
        assert state["findings"] == [{"f": 1}]
        assert executor.action_service.calls[0]["tool"] == "search_documents"

    def test_dangerous_tool_deferred_into_decision_action(self):
        llm = FakeLLM([_completion(tool_calls=[_tool_call(name="delete_document")])])
        executor = _fake_executor(llm)
        service = AgentThinkingService()
        job = _fake_job()
        state = {"actions_taken": []}

        response = run(
            service._maybe_run_native_tool_loop(
                executor,
                job,
                state,
                system_prompt="system",
                user_message="decide",
                available_tools=["search_documents", "delete_document"],
                user_settings=None,
                routing=None,
                db=None,
            )
        )
        decision = json.loads(response)
        assert decision["action"]["tool"] == "delete_document"
        assert decision["goal_achieved"] is False
        # Gated tool never executed through the action service.
        assert executor.action_service.calls == []

    def test_no_tool_budget_returns_none(self):
        service = AgentThinkingService()
        job = _fake_job(max_tool_calls=3, tool_calls_used=3)
        result = run(
            service._maybe_run_native_tool_loop(
                _fake_executor(FakeLLM([])),
                job,
                {},
                system_prompt="s",
                user_message="u",
                available_tools=["search_documents"],
                user_settings=None,
                routing=None,
                db=None,
            )
        )
        assert result is None

    def test_resolve_config_precedence(self):
        service = AgentThinkingService()
        assert service._resolve_native_loop_config(
            SimpleNamespace(config={"native_tool_loop": True})
        )["enabled"]
        assert not service._resolve_native_loop_config(
            SimpleNamespace(config={"native_tool_loop": False})
        )["enabled"]
        cfg = service._resolve_native_loop_config(
            SimpleNamespace(
                config={"native_tool_loop": {"max_tool_calls": 2, "max_llm_calls": 3}}
            )
        )
        assert cfg == {"enabled": True, "max_tool_calls": 2, "max_llm_calls": 3}
