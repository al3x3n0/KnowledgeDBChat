"""Tests for the single JSON-asking path.

The point of ask_for_json is that callers stop caring whether the provider
constrained the schema or the model answered in prose, so these pin both paths
and the transitions between them.
"""

import pytest

from app.services.llm_structured import ask_for_json, schema_hint

SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


class FakeCompletion:
    def __init__(self, structured=None, text=""):
        self.structured = structured
        self.text = text


class FakeLLM:
    """Records what was called so tests can assert which path ran."""

    def __init__(self, *, structured=None, structured_error=None, response=""):
        self._structured = structured
        self._structured_error = structured_error
        self._response = response
        self.structured_calls = 0
        self.response_calls = 0
        self.last_schema = None

    async def generate_structured(self, **kwargs):
        self.structured_calls += 1
        self.last_schema = kwargs.get("response_schema")
        if self._structured_error is not None:
            raise self._structured_error
        return self._structured

    async def generate_response(self, **kwargs):
        self.response_calls += 1
        return self._response


@pytest.mark.asyncio
async def test_uses_the_structured_payload_when_the_provider_constrains_it():
    llm = FakeLLM(structured=FakeCompletion(structured={"answer": "yes"}))

    result = await ask_for_json(llm, schema=SCHEMA, user_message="q")

    assert result == {"answer": "yes"}
    assert llm.structured_calls == 1
    assert llm.response_calls == 0, "the prompted path must not run unnecessarily"
    assert llm.last_schema == SCHEMA


@pytest.mark.asyncio
async def test_falls_back_to_the_prompted_path_when_structured_raises():
    llm = FakeLLM(
        structured_error=RuntimeError("provider has no schema support"),
        response='```json\n{"answer": "from prose"}\n```',
    )

    result = await ask_for_json(llm, schema=SCHEMA, user_message="q")

    assert result == {"answer": "from prose"}
    assert llm.structured_calls == 1
    assert llm.response_calls == 1


@pytest.mark.asyncio
async def test_falls_back_when_the_provider_returns_nothing():
    llm = FakeLLM(structured=None, response='{"answer": "fallback"}')

    assert await ask_for_json(llm, schema=SCHEMA, user_message="q") == {
        "answer": "fallback"
    }
    assert llm.response_calls == 1


@pytest.mark.asyncio
async def test_parses_text_when_a_provider_honours_the_schema_but_answers_in_prose():
    llm = FakeLLM(
        structured=FakeCompletion(text='Sure!\n```json\n{"answer": "text"}\n```')
    )

    result = await ask_for_json(llm, schema=SCHEMA, user_message="q")

    assert result == {"answer": "text"}
    assert llm.response_calls == 0, "a usable answer must not trigger a second call"


@pytest.mark.asyncio
async def test_parses_a_structured_payload_delivered_as_a_string():
    llm = FakeLLM(structured=FakeCompletion(structured='{"answer": "stringified"}'))

    assert await ask_for_json(llm, schema=SCHEMA, user_message="q") == {
        "answer": "stringified"
    }


@pytest.mark.asyncio
async def test_returns_none_when_neither_path_produces_an_object():
    llm = FakeLLM(structured=FakeCompletion(text="no json here"), response="still none")

    assert await ask_for_json(llm, schema=SCHEMA, user_message="q") is None
    assert llm.response_calls == 1


def test_schema_hint_renders_the_schema_for_a_prompt():
    rendered = schema_hint(SCHEMA)
    assert '"answer"' in rendered
    assert rendered.startswith("{")
