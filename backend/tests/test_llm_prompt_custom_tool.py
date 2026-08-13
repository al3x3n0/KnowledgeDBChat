"""llm_prompt custom tools must actually call the model.

The executor called LLMService.generate_response(messages=...), which is not a
parameter it accepts, so every llm_prompt tool failed on invocation. The line
after it read .get("content") off the return value, which is a string, so a
fixed call would have failed too.
"""

from types import SimpleNamespace

import pytest

from app.services.custom_tool_service import CustomToolService


class _DB:
    async def execute(self, *_a, **_k):
        return SimpleNamespace(scalar_one_or_none=lambda: None)


@pytest.fixture
def captured(monkeypatch):
    calls = {}

    async def _generate_response(**kwargs):
        calls.update(kwargs)
        return "Vectorization was unlocked by the tuned flags."

    from app.services import custom_tool_service as module

    monkeypatch.setattr(
        module,
        "LLMService",
        lambda: SimpleNamespace(generate_response=_generate_response),
    )
    return calls


@pytest.mark.asyncio
async def test_the_model_is_called_with_supported_arguments(captured):
    out = await CustomToolService()._execute_llm_prompt(
        {"user_prompt": "baseline {{ baseline }}, tuned {{ tuned }}"},
        {"baseline": 0, "tuned": 17},
        SimpleNamespace(id="u1"),
        _DB(),
    )

    assert "messages" not in captured, "generate_response has no messages parameter"
    assert "user_message" in captured
    assert out["text"].startswith("Vectorization was unlocked")


@pytest.mark.asyncio
async def test_a_string_reply_is_handled_not_treated_as_a_mapping(captured):
    out = await CustomToolService()._execute_llm_prompt(
        {"user_prompt": "x"}, {}, SimpleNamespace(id="u1"), _DB()
    )

    assert isinstance(out["text"], str) and out["text"]


@pytest.mark.asyncio
async def test_prompt_is_accepted_as_an_alias_for_user_prompt(captured):
    """An agent authoring a tool wrote config.prompt, and the executor read
    user_prompt, so it rendered an empty prompt."""
    await CustomToolService()._execute_llm_prompt(
        {"prompt": "baseline {{ baseline }}"},
        {"baseline": 0},
        SimpleNamespace(id="u1"),
        _DB(),
    )

    assert "baseline 0" in captured["user_message"]


def test_single_brace_placeholders_are_filled_for_known_inputs():
    """Jinja needs {{ name }}; an agent wrote {name} and the model answered
    about the literal placeholder."""
    svc = CustomToolService()

    out = svc._render_template(
        "baseline {baseline}, tuned {tuned}", {"baseline": 0, "tuned": 17}
    )

    assert out == "baseline 0, tuned 17"


def test_braces_that_name_nothing_are_left_alone():
    svc = CustomToolService()

    out = svc._render_template("keep {unknown} and {}", {"baseline": 1})

    assert "{unknown}" in out and "{}" in out


def test_jinja_syntax_still_works():
    svc = CustomToolService()

    assert svc._render_template("v={{ baseline }}", {"baseline": 5}) == "v=5"
