"""A chat turn plans its tool calls in rounds, each seeing the last one's results.

A turn used to plan every call before any result existed. Asked to use a
sandbox skill, chat planned `list` and `load` -- correctly -- and stopped,
because the command to run is in the procedure `load` had not yet returned.

The loop is tested with the planner and the executor scripted, since what
matters here is when it asks again and when it stops; the planner's prompt is
tested separately, against the real method.
"""

from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.core.config import settings
from app.schemas.agent import AgentToolCall
from app.services.agent_service import AgentService

pytestmark = pytest.mark.unit


def call(name, **tool_input):
    return AgentToolCall(tool_name=name, tool_input=tool_input)


def agent(allowed=None):
    return SimpleNamespace(
        id=uuid4(),
        name="generalist",
        display_name="Generalist",
        system_prompt="You help.",
        has_tool=lambda tool: allowed is None or tool in allowed,
    )


class Scripted:
    """An AgentService whose planner serves queued rounds and whose executor
    records what ran."""

    def __init__(self, rounds, *, statuses=None):
        self.service = AgentService()
        self.rounds = list(rounds)
        self.statuses = statuses or {}
        self.seen_prior = []
        self.ran = []
        self.service._plan_tool_calls_for_agent = self._plan
        self.service._execute_tool = self._execute

    async def _plan(self, *, prior_results=None, **_kwargs):
        self.seen_prior.append(
            None if prior_results is None else [r.tool_name for r in prior_results]
        )
        return self.rounds.pop(0) if self.rounds else []

    async def _execute(self, tool_call, user_id, db, **_kwargs):
        self.ran.append(tool_call.tool_name)
        tool_call.status = self.statuses.get(tool_call.tool_name, "completed")
        tool_call.tool_output = {"ok": tool_call.tool_name}
        return tool_call

    async def run(self, who=None):
        return await self.service._run_tool_rounds(
            message="do it",
            history=[],
            agent=who or agent(),
            memory_context="",
            user_settings=None,
            contributed=None,
            user_id=uuid4(),
            db=None,
            conversation_id=uuid4(),
        )


async def test_a_second_round_is_planned_from_the_first_rounds_results():
    scripted = Scripted(
        [
            [call("list_sandbox_skills"), call("load_sandbox_skill", skill="x")],
            [call("run_sandbox_skill", skill="x", command="make")],
            [],
        ]
    )
    results, _, _ = await scripted.run()

    assert scripted.ran == [
        "list_sandbox_skills",
        "load_sandbox_skill",
        "run_sandbox_skill",
    ]
    assert [r.tool_name for r in results] == scripted.ran
    # The first round has nothing to go on; later ones are shown what ran.
    assert scripted.seen_prior == [
        None,
        ["list_sandbox_skills", "load_sandbox_skill"],
        ["list_sandbox_skills", "load_sandbox_skill", "run_sandbox_skill"],
    ]


async def test_a_turn_that_needs_no_tools_asks_once():
    scripted = Scripted([[]])
    results, _, _ = await scripted.run()
    assert results == [] and scripted.seen_prior == [None]


async def test_a_round_that_only_repeats_itself_ends_the_turn():
    """A model shown a result it cannot improve on tends to ask again."""
    scripted = Scripted(
        [
            [call("search_documents", query="gem5")],
            [call("search_documents", query="gem5")],
            [call("search_documents", query="never reached")],
        ]
    )
    await scripted.run()
    assert scripted.ran == ["search_documents"]


async def test_the_same_tool_with_different_input_is_not_a_repeat():
    scripted = Scripted(
        [
            [call("search_documents", query="gem5")],
            [call("search_documents", query="prefetcher")],
            [],
        ]
    )
    await scripted.run()
    assert scripted.ran == ["search_documents", "search_documents"]


async def test_the_round_limit_ends_a_turn_that_never_says_done(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_CHAT_MAX_TOOL_ROUNDS", 3, raising=False)
    scripted = Scripted([[call("search_documents", query=str(n))] for n in range(9)])
    await scripted.run()
    assert len(scripted.ran) == 3


async def test_one_round_is_the_old_single_shot_behaviour(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_CHAT_MAX_TOOL_ROUNDS", 1, raising=False)
    scripted = Scripted([[call("a_tool")], [call("b_tool")]])
    await scripted.run()
    assert scripted.ran == ["a_tool"] and scripted.seen_prior == [None]


async def test_the_call_limit_holds_across_rounds(monkeypatch):
    monkeypatch.setattr(settings, "AGENT_CHAT_MAX_TOOL_CALLS", 3, raising=False)
    scripted = Scripted(
        [
            [call("t", n=1), call("t", n=2)],
            [call("t", n=3), call("t", n=4)],
            [call("t", n=5)],
        ]
    )
    results, _, _ = await scripted.run()
    assert len(results) == 3 and len(scripted.ran) == 3


async def test_a_call_waiting_for_approval_ends_the_turn():
    """Planning past a question nobody has answered is planning on a guess."""
    scripted = Scripted(
        [[call("delete_document", document_id="d")], [call("search_documents")]],
        statuses={"delete_document": "requires_approval"},
    )
    results, _, _ = await scripted.run()
    assert scripted.ran == ["delete_document"]
    assert results[0].status == "requires_approval"


async def test_asking_for_a_file_ends_the_turn_and_says_why():
    scripted = Scripted([[call("request_file_upload")], [call("search_documents")]])
    _, requires_user_action, action_type = await scripted.run()
    assert scripted.ran == ["request_file_upload"]
    assert requires_user_action and action_type == "upload_file"


async def test_a_tool_the_agent_may_not_use_is_refused_and_reported():
    scripted = Scripted([[call("delete_document"), call("search_documents")], []])
    results, _, _ = await scripted.run(agent(allowed={"search_documents"}))
    assert scripted.ran == ["search_documents"]
    assert results[0].status == "failed"
    assert "not available for this agent" in results[0].error


class TestWhatThePlannerIsTold:
    async def _prompt(self, monkeypatch, **kwargs):
        service = AgentService()
        seen = {}

        async def fake_generate(**call_kwargs):
            seen.update(call_kwargs)
            return "[]"

        monkeypatch.setattr(service.llm_service, "generate_response", fake_generate)
        monkeypatch.setattr(service, "_routing_from_agent", lambda *a, **k: None)
        await service._plan_tool_calls_for_agent(
            message="use the skill",
            history=[],
            agent=SimpleNamespace(
                name="g",
                display_name="G",
                system_prompt="You help.",
                tool_whitelist=None,
            ),
            memory_context="",
            **kwargs,
        )
        return seen

    async def test_the_first_round_is_told_not_to_guess(self, monkeypatch):
        seen = await self._prompt(monkeypatch)
        assert "plan only the calls whose\ninputs you already know" in seen["query"]
        assert "ALREADY MADE" not in seen["query"]

    async def test_a_later_round_sees_what_came_back(self, monkeypatch):
        done = call("load_sandbox_skill", skill="x")
        done.status = "completed"
        done.tool_output = {"data": {"procedure": "run make, then ./prog"}}
        failed = call("run_sandbox_skill", skill="x", command="mak")
        failed.status = "failed"
        failed.error = "The command exited 127"

        seen = await self._prompt(monkeypatch, prior_results=[done, failed])
        prompt = seen["query"]
        assert "ALREADY MADE" in prompt
        assert "run make, then ./prog" in prompt
        assert "FAILED: The command exited 127" in prompt
        assert "Never repeat a call" in prompt

    async def test_a_huge_result_is_truncated_rather_than_sent_whole(self, monkeypatch):
        big = call("search_documents", query="x")
        big.status = "completed"
        big.tool_output = {"text": "y" * 100_000}
        seen = await self._prompt(monkeypatch, prior_results=[big])
        assert "[truncated]" in seen["query"]
        assert len(seen["query"]) < 60_000

    async def test_there_is_room_for_a_call_that_carries_a_file(self, monkeypatch):
        seen = await self._prompt(monkeypatch)
        assert seen["max_tokens"] >= 1500


async def test_the_answer_may_not_describe_a_run_that_did_not_happen(monkeypatch):
    """Seen live: a skill request that ran no tool was answered with the IR the
    compiler "typically" emits, formatted as a result."""
    service = AgentService()
    seen = {}

    async def fake_generate(**call_kwargs):
        seen.update(call_kwargs)
        return "ok"

    monkeypatch.setattr(service.llm_service, "generate_response", fake_generate)
    monkeypatch.setattr(service, "_routing_from_agent", lambda *a, **k: None)
    await service._generate_response_for_agent(
        message="run the skill",
        tool_results=[],
        history=[],
        agent=SimpleNamespace(name="g", display_name="G", system_prompt="You help."),
        memory_context="",
    )
    assert "No tools were executed." in seen["query"]
    assert 'Never describe what it "would" or "should" have' in seen["query"]
