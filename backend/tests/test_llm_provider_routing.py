"""Provider routing for the prompted-text chat path.

Both call paths -- ``generate_response`` (prompted text) and
``generate_structured`` (native tool calling) -- must send each provider to
its own endpoint with its own credentials. The historical bug was that
``LLM_PROVIDER=openai`` on the text path fell into the DeepSeek branch and
resolved ``DEEPSEEK_API_KEY``/``DEEPSEEK_API_BASE`` -- empty for a gateway
that holds the target credentials -- and died with "DEEPSEEK_API_KEY is not
set" while the structured path already went to the right place.

The tests stop at the provider: ``build_provider`` and the DeepSeek request
helper are replaced by recorders, so what is checked is the name, URL, key
and model each call would have used -- the same seam
``test_llm_provider_selection.py`` uses.
"""

from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.services import llm_providers
from app.services.llm_service import LLMService

pytestmark = pytest.mark.unit


class _Recorder:
    def __init__(self):
        self.calls = []


@pytest.fixture
def sent(monkeypatch):
    rec = _Recorder()

    class _FakeProvider:
        def __init__(self, name, api_url, api_key):
            self.name, self.api_url, self.api_key = name, api_url, api_key

        async def complete(self, messages, *, model=None, **kwargs):
            rec.calls.append(
                {
                    "branch": "provider",
                    "provider": self.name,
                    "api_url": self.api_url,
                    "api_key": self.api_key,
                    "model": model,
                }
            )
            return SimpleNamespace(
                text="pong",
                model=model or "rec-model",
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                raw={},
                provider=self.name,
                tool_calls=[],
                structured=None,
                finish_reason="stop",
            )

    def _fake_build(name, *, api_url=None, api_key=None, http_client=None):
        return _FakeProvider(name, api_url, api_key)

    async def _fake_deepseek(
        self, *, model, messages, api_key_override=None, api_base_override=None, **kw
    ):
        rec.calls.append(
            {
                "branch": "deepseek",
                "api_url": api_base_override,
                "api_key": api_key_override,
                "model": model,
            }
        )
        return "pong", {}

    monkeypatch.setattr(llm_providers, "build_provider", _fake_build)
    monkeypatch.setattr(LLMService, "_make_deepseek_chat_request", _fake_deepseek)
    # Keep the suite off the live model host: the autouse _no_live_llm_calls
    # fixture clears the keys, so the URL below is where an unpatched call
    # would land.
    monkeypatch.setattr(settings, "OPENAI_API_BASE", "http://127.0.0.1:1/v1")
    monkeypatch.setattr(settings, "OPENAI_MODEL", "deepseek-v4-flash")
    return rec


def _service(provider):
    svc = LLMService()
    svc.provider = provider
    svc.default_model = "deepseek-v4-pro"
    return svc


async def test_text_and_structured_openai_go_to_openai_not_deepseek(sent):
    svc = _service("openai")
    text = await svc.generate_response(query="ping", model="deepseek-v4-flash")
    completion = await svc.generate_structured(user_message="ping")

    assert text == "pong"
    assert completion.text == "pong"
    for call in sent.calls:
        # The openai provider must take the provider path (the one that
        # resolves the OPENAI_* settings), never the DeepSeek one.
        assert call["branch"] == "provider" and call["provider"] == "openai"
        # build_provider fills the OPENAI_* settings in; nothing of DeepSeek's.
        assert call["api_url"] is None and call["api_key"] is None
    assert sent.calls[0]["model"] == "deepseek-v4-flash"
    assert sent.calls[1]["model"] is None
