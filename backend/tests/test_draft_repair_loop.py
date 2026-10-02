"""The loop three drafters share: ask, judge, and ask again with the complaint.

Each drafter carried its own copy, so each had to learn separately what to do
when the model is unreachable, when the reply is not JSON, and when a progress
callback throws. These pin the plumbing once; each drafter's own tests still
pin its judge.
"""

from __future__ import annotations

import pytest

from app.services import draft_repair_loop as loop
from app.services.draft_repair_loop import Verdict

pytestmark = pytest.mark.unit


class Completion:
    def __init__(self, structured=None, text=""):
        self.structured = structured
        self.text = text


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = list(replies)
        self.messages = []
        self.kwargs = []

    async def generate_structured(self, **kwargs):
        self.messages.append(kwargs["user_message"])
        self.kwargs.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


async def run(replies, judge, **kwargs):
    llm = ScriptedLLM(replies)
    outcome = await loop.run(
        system="s", message="make it", schema={}, judge=judge, llm=llm, **kwargs
    )
    return outcome, llm


async def accept(payload, _attempt):
    return Verdict(value=payload)


async def test_an_accepted_draft_ends_the_loop_at_once():
    outcome, llm = await run([Completion({"a": 1}), Completion({"a": 2})], accept)
    assert outcome.value == {"a": 1} and outcome.attempts == 1 and outcome.notes == []
    assert len(llm.messages) == 1


async def test_a_complaint_goes_back_to_the_model_and_into_the_notes():
    async def judge(payload, attempt):
        if payload["a"] == 1:
            return Verdict(
                complaint="a must be 2", note="a was 1", instruction="Set a to 2."
            )
        return Verdict(value=payload)

    outcome, llm = await run([Completion({"a": 1}), Completion({"a": 2})], judge)
    assert outcome.value == {"a": 2} and outcome.attempts == 2
    assert outcome.notes == ["Attempt 1: a was 1"]
    # The second request carries the first, the complaint and what to do.
    assert llm.messages[1].startswith("make it")
    assert "a must be 2" in llm.messages[1] and "Set a to 2." in llm.messages[1]


async def test_it_stops_at_the_limit_with_every_refusal_noted():
    async def refuse(_payload, _attempt):
        return Verdict(complaint="still wrong")

    outcome, llm = await run([Completion({"a": 1})] * 5, refuse, max_attempts=3)
    assert outcome.value is None and outcome.attempts == 3
    assert len(outcome.notes) == 3 and len(llm.messages) == 3


async def test_a_flawed_draft_can_be_kept_while_it_is_sent_back():
    """A flaw a person can see and fix is worth more than nothing."""

    async def judge(payload, _attempt):
        return Verdict(value=payload, complaint="the path misses")

    outcome, _ = await run(
        [Completion({"n": 1}), Completion({"n": 2})], judge, max_attempts=2
    )
    assert outcome.value == {"n": 2} and len(outcome.notes) == 2


async def test_a_refusal_can_discard_what_was_kept_before():
    async def judge(payload, attempt):
        if attempt == 1:
            return Verdict(value=payload, complaint="flawed")
        return Verdict(complaint="invalid", discard=True)

    outcome, _ = await run(
        [Completion({"n": 1}), Completion({"n": 2})], judge, max_attempts=2
    )
    assert outcome.value is None


async def test_stop_ends_the_loop_without_spending_another_call():
    """When the failure is not in the draft, no rewrite can fix it."""

    async def judge(payload, _attempt):
        return Verdict(value=payload, complaint="x", note="sandbox disabled", stop=True)

    outcome, llm = await run([Completion({"a": 1}), Completion({"a": 2})], judge)
    assert outcome.value == {"a": 1} and outcome.attempts == 1
    assert outcome.notes == ["sandbox disabled"] and len(llm.messages) == 1


async def test_a_reply_that_is_not_json_is_asked_for_again():
    outcome, llm = await run(
        [Completion(text="sorry, no"), Completion({"a": 1})], accept
    )
    assert outcome.value == {"a": 1}
    assert outcome.notes == ["Attempt 1: the reply was not JSON."]
    assert "not a JSON object" in llm.messages[1]


async def test_json_left_in_the_text_counts_as_a_reply():
    outcome, _ = await run([Completion(text='Here: {"a": 1}')], accept)
    assert outcome.value == {"a": 1}


async def test_an_unreachable_model_is_a_note_not_an_exception():
    outcome, _ = await run([RuntimeError("503 Service Unavailable")], accept)
    assert outcome.value is None
    assert outcome.notes == ["The model could not be reached: 503 Service Unavailable"]


async def test_progress_is_reported_and_a_broken_reporter_is_ignored():
    seen = []
    outcome, _ = await run(
        [Completion({"a": 1})], accept, on_progress=lambda *a: seen.append(a[:2])
    )
    assert seen == [("drafting", 1), ("checking", 1), ("done", 1)]

    def boom(*_args):
        raise RuntimeError("socket closed")

    outcome, _ = await run([Completion({"a": 1})], accept, on_progress=boom)
    assert outcome.value == {"a": 1}


async def test_the_snapshot_phase_is_passed_only_when_named():
    _, llm = await run([Completion({"a": 1})], accept)
    assert "snapshot_context" not in llm.kwargs[0]
    _, llm = await run([Completion({"a": 1})], accept, snapshot_phase="skill_draft")
    assert llm.kwargs[0]["snapshot_context"] == {"phase": "skill_draft"}


def test_every_drafter_uses_the_loop():
    """The point of the module: a drafter that grows its own loop again is the
    fourth copy."""
    import inspect

    from app.services import (
        agent_definition_author_service,
        plugin_author_service,
        sandbox_skill_author_service,
    )

    for module, function in (
        (plugin_author_service, "draft_manifest"),
        (agent_definition_author_service, "draft_definition"),
        (sandbox_skill_author_service, "draft_skill"),
    ):
        source = inspect.getsource(getattr(module, function))
        assert "draft_repair_loop.run(" in source, module.__name__
        assert "generate_structured" not in source, module.__name__
