"""Guard: the suite must not reach a real LLM provider.

A test that stubs only generate_response still lets
llm_structured.ask_for_json try generate_structured first. With a provider key
configured that call succeeds against the live API, so the stub is bypassed:
the test spends credits and its assertions depend on whatever the model said
that minute. CI has no keys, so the failure only ever appeared on developer
machines, which made it look like a flaky test rather than a live call.
"""

import pytest

from app.core.config import settings


@pytest.mark.parametrize(
    "name",
    [
        "DEEPSEEK_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "QWEN_API_KEY",
        "KIMI_API_KEY",
    ],
)
def test_provider_credentials_are_cleared_during_tests(name):
    assert getattr(settings, name, None) in (None, ""), (
        f"{name} is visible to the test suite; an unstubbed call would reach a "
        "real provider"
    )


def test_ollama_is_not_pointed_at_a_reachable_server():
    """Ollama needs no key, so clearing credentials alone would not stop it."""
    assert settings.OLLAMA_BASE_URL == "http://127.0.0.1:1"


@pytest.mark.asyncio
async def test_ask_for_json_falls_back_when_structured_is_unavailable():
    """The behaviour the scripted tests rely on: structured returns None and the
    prompted stub is used."""
    from app.services import llm_structured

    class _Stub:
        async def generate_structured(self, **kwargs):
            raise RuntimeError("no schema support")

        async def generate_response(self, **kwargs):
            return '{"answer": "from the prompted path"}'

    result = await llm_structured.ask_for_json(
        _Stub(),
        schema={"type": "object"},
        system_prompt="s",
        user_message="u",
    )

    assert result == {"answer": "from the prompted path"}
