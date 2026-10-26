"""Final conclusions share the job's model allowance, including JSON fallback."""

import json
from types import SimpleNamespace

import pytest

from app.services.agent_run_synthesis_service import synthesize_conclusion
from app.services.llm_providers.base import LLMCompletion

pytestmark = pytest.mark.unit
PAYLOAD = {
    "answer": "The measured result held.",
    "confidence": "high",
    "evidence": ["profile"],
    "gaps": [],
}
FINDINGS = [{"type": "dynamic_profile", "title": "A measured profile"}]


class Model:
    def __init__(self, structured="valid", prompt_fails=False):
        self.structured = structured
        self.prompt_fails = prompt_fails
        self.calls = []

    async def generate_structured(self, **kwargs):
        self.calls.append(("structured", kwargs))
        if self.structured == "error":
            raise RuntimeError("fixture schema failure")
        return LLMCompletion(structured=PAYLOAD) if self.structured == "valid" else None

    async def generate_response(self, **kwargs):
        self.calls.append(("prompted", kwargs))
        if self.prompt_fails:
            raise RuntimeError("fixture prompt failure")
        return json.dumps(PAYLOAD)


def job(remaining):
    return SimpleNamespace(
        id="job-1",
        goal="Explain the evidence",
        iteration=2,
        llm_calls_used=3,
        max_llm_calls=3 + remaining,
    )


@pytest.mark.parametrize("remaining", [0, -1])
async def test_no_allowance_produces_no_model_request_or_answer(remaining):
    model, run = Model(), job(remaining)

    result = await synthesize_conclusion(
        SimpleNamespace(llm_service=model), run, {"findings": FINDINGS}
    )

    assert model.calls == [] and run.llm_calls_used == 3
    assert result["answer"] is None and result["generated_by"] == "budget_exhausted"


async def test_structured_success_charges_its_one_request_and_keeps_context():
    model, run = Model(), job(1)

    result = await synthesize_conclusion(
        SimpleNamespace(llm_service=model), run, {"findings": FINDINGS}
    )

    assert result["generated_by"] == "llm" and run.llm_calls_used == 4
    assert [kind for kind, _ in model.calls] == ["structured"]
    assert model.calls[0][1]["snapshot_context"] == {
        "job_id": "job-1",
        "iteration": 2,
        "phase": "conclusion",
    }


@pytest.mark.parametrize("structured", ["empty", "error"])
@pytest.mark.parametrize("remaining", [1, 2])
async def test_json_fallback_rechecks_and_charges_the_same_allowance(
    structured, remaining
):
    model, run = Model(structured), job(remaining)

    result = await synthesize_conclusion(
        SimpleNamespace(llm_service=model), run, {"findings": FINDINGS}
    )

    assert len(model.calls) == remaining and run.llm_calls_used == 3 + remaining
    assert result["generated_by"] == ("budget_exhausted" if remaining == 1 else "llm")
    assert (result["answer"] is None) is (remaining == 1)


async def test_failed_prompt_request_still_spends_its_slot():
    model, run = Model("empty", prompt_fails=True), job(2)

    result = await synthesize_conclusion(
        SimpleNamespace(llm_service=model), run, {"findings": FINDINGS}
    )

    assert len(model.calls) == 2 and run.llm_calls_used == 5
    assert result["generated_by"] == "error" and result["answer"] is None


async def test_absent_structured_method_does_not_spend_a_request():
    model, run = Model(), job(1)
    prompt_only = SimpleNamespace(generate_response=model.generate_response)

    result = await synthesize_conclusion(
        SimpleNamespace(llm_service=prompt_only), run, {"findings": FINDINGS}
    )

    assert [kind for kind, _ in model.calls] == ["prompted"]
    assert run.llm_calls_used == 4 and result["generated_by"] == "llm"


async def test_no_evidence_does_not_spend_a_request():
    model, run = Model(), job(2)

    result = await synthesize_conclusion(
        SimpleNamespace(llm_service=model), run, {"findings": []}
    )

    assert model.calls == [] and run.llm_calls_used == 3
    assert result["generated_by"] == "no_evidence"
