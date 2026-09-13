"""Attacking an experiment's design before it is paid for.

Every test here stubs the model. What is being checked is not whether a critic
is clever -- that is the model's business -- but that a critic's answer is
never turned into a clean bill of health by the code that reads it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services import agent_design_critique as critic


class _LLM:
    """Serves queued payloads, recording what it was asked."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.messages = []

    async def generate_structured(self, *, system_prompt, user_message, **_kw):
        self.messages.append(user_message)
        payload = self.payloads.pop(0) if self.payloads else {"concerns": []}
        if isinstance(payload, Exception):
            raise payload
        return SimpleNamespace(structured=payload, text="")


BLOCKED = {
    "concerns": [
        {
            "summary": "phases are blocked, not alternating",
            "why_it_matters": "the trace is two experiments",
            "remedy": "interleave the phases",
            "severity": "blocking",
        }
    ]
}


@pytest.mark.asyncio
async def test_a_concern_in_the_declared_shape_is_read():
    llm = _LLM([BLOCKED, {"concerns": []}, {"concerns": []}])

    result = await critic.critique(llm, artifact="int main(){}", goal="g")

    assert result["reviewed"] is True
    assert len(result["blocking"]) == 1
    assert "blocked" in result["concerns"][0]["summary"]


@pytest.mark.asyncio
async def test_a_concern_in_the_wrong_shape_is_still_read():
    """The live failure this module was rebuilt around. Asked for
    summary/severity, the provider returned concern/location -- a correct
    finding, precisely located -- and the first version discarded it and
    reported a clean design. Tolerance comes before retrying: a good answer in
    the wrong shape must not cost another call."""
    drifted = {
        "concerns": [
            {
                "concern": "runs 100 cache phases then 100 memory phases",
                "location": "main() for loops",
            }
        ]
    }
    llm = _LLM([drifted, {"concerns": []}, {"concerns": []}])

    result = await critic.critique(llm, artifact="x", goal="g")

    assert len(result["concerns"]) == 1
    assert "100 cache phases" in result["concerns"][0]["summary"]
    assert result["concerns"][0]["location"] == "main() for loops"
    assert llm.messages.count("Report your concerns.") == 3, "no retry was needed"


@pytest.mark.asyncio
async def test_an_absent_severity_is_unrated_not_middling():
    """Four concerns arrived live without a severity, were defaulted to
    "serious", and "0 blocking" then read as a reviewer that had declined to
    escalate rather than one that never rated anything."""
    llm = _LLM(
        [
            {"concerns": [{"concern": "no severity given"}]},
            {"concerns": []},
            {"concerns": []},
        ]
    )

    result = await critic.critique(llm, artifact="x", goal="g")

    assert result["concerns"][0]["severity"] == "unrated"
    assert result["blocking"] == []
    assert len(result["unrated"]) == 1


@pytest.mark.asyncio
async def test_a_bare_list_is_read():
    """Also seen live: the concerns arrived without their envelope, which
    raised AttributeError rather than being read."""
    llm = _LLM([[{"summary": "x", "severity": "minor"}], [], []])

    result = await critic.critique(llm, artifact="x", goal="g")

    assert len(result["concerns"]) == 1


@pytest.mark.asyncio
async def test_an_unreadable_answer_is_retried_with_a_correction():
    llm = _LLM(
        [{"concerns": [{"unusable": 42}]}, BLOCKED, {"concerns": []}, {"concerns": []}]
    )

    result = await critic.critique(
        llm, artifact="x", goal="g", lenses=["answers_the_question"]
    )

    assert len(result["concerns"]) == 1, "the retry's answer is kept"
    assert "could not be read" in llm.messages[1], "the retry says what was wrong"
    assert (
        "still there" in llm.messages[1]
    ), "a correction must not invite the reviewer to soften its judgement"


@pytest.mark.asyncio
async def test_a_lens_that_never_answers_is_unreviewed_not_clean():
    """The distinction this whole package turns on: 'nothing was found' and
    'we could not look' are opposite statements."""
    llm = _LLM([{"concerns": [{"bad": 1}]}] * critic.MAX_ATTEMPTS)

    result = await critic.critique(
        llm, artifact="x", goal="g", lenses=["answers_the_question"]
    )

    assert result["reviewed"] is False
    assert result["unreviewed_lenses"][0]["lens"] == "answers_the_question"
    assert result["concerns"] == []


@pytest.mark.asyncio
async def test_an_empty_list_is_an_answer_and_is_not_retried():
    """A critic that always finds something is noise, so 'nothing' has to be
    sayable -- and saying it must not cost three calls."""
    llm = _LLM([{"concerns": []}])

    result = await critic.critique(
        llm, artifact="x", goal="g", lenses=["answers_the_question"]
    )

    assert result["reviewed"] is True
    assert result["concerns"] == []
    assert len(llm.messages) == 1


@pytest.mark.asyncio
async def test_a_raising_lens_is_retried_then_reported():
    llm = _LLM([RuntimeError("boom")] * critic.MAX_ATTEMPTS)

    result = await critic.critique(
        llm, artifact="x", goal="g", lenses=["cost_and_feasibility"]
    )

    assert result["reviewed"] is False
    assert result["unreviewed_lenses"]


@pytest.mark.asyncio
async def test_blocking_concerns_sort_first():
    llm = _LLM(
        [
            {
                "concerns": [
                    {"summary": "small", "severity": "minor"},
                    {"summary": "fatal", "severity": "blocking"},
                ]
            },
            {"concerns": []},
            {"concerns": []},
        ]
    )

    result = await critic.critique(llm, artifact="x", goal="g")

    assert result["concerns"][0]["summary"] == "fatal"


def test_the_lenses_ask_different_questions():
    """Asking one prompt three times produces one concern three times."""
    assert len(critic.LENSES) >= 3
    texts = list(critic.LENSES.values())
    assert len(set(texts)) == len(texts)
