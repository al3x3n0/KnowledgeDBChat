"""Where a call goes is what the settings say, on both call paths.

Reported: tasks ignored an OpenAI-compatible local endpoint and went to
DeepSeek. Two causes. The prompted-text path (``generate_response``) shared
its DeepSeek branch with ``openai``, so ``LLM_PROVIDER=openai`` was sent to
DeepSeek's URL with DeepSeek's key while the structured path went where it
was told. And ``prefer_deepseek`` sent context compression, branch summaries
and long document summaries to DeepSeek whenever a DeepSeek key existed,
whatever provider anyone had chosen.

The tests stop at the provider: ``build_provider`` and the three request
helpers are replaced by recorders, so what is checked is the name, URL, key
and model each call would have used.
"""

from types import SimpleNamespace

import pytest

from app.core.config import settings
from app.services import llm_provider_catalog as catalog
from app.services import llm_providers
from app.services.llm_routing import resolve_effective_provider_model
from app.services.llm_service import LLMService, UserLLMSettings

pytestmark = pytest.mark.unit


class Recorder:
    def __init__(self):
        self.calls = []


@pytest.fixture
def sent(monkeypatch):
    rec = Recorder()

    class FakeProvider:
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
                text="ok",
                model=model,
                prompt_tokens=1,
                completion_tokens=1,
                total_tokens=2,
                raw={},
                provider=self.name,
                tool_calls=[],
                parsed=None,
                structured=None,
                finish_reason="stop",
            )

    def fake_build(name, *, api_url=None, api_key=None, http_client=None):
        return FakeProvider(name, api_url, api_key)

    monkeypatch.setattr(llm_providers, "build_provider", fake_build)

    async def deepseek(
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
        return "ok", {}

    async def custom(self, *, api_url, api_key, model, messages, **kw):
        rec.calls.append(
            {"branch": "custom", "api_url": api_url, "api_key": api_key, "model": model}
        )
        return "ok", {}

    async def ollama(self, *, model, prompt, **kw):
        rec.calls.append({"branch": "ollama", "model": model})
        return "ok", {}

    monkeypatch.setattr(LLMService, "_make_deepseek_chat_request", deepseek)
    monkeypatch.setattr(LLMService, "_make_openai_compatible_request", custom)
    monkeypatch.setattr(LLMService, "_make_ollama_request", ollama, raising=False)
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", "ds-key")
    monkeypatch.setattr(settings, "OPENAI_API_BASE", "http://gpu-box:8000/v1")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "local-key")
    monkeypatch.setattr(settings, "OPENAI_MODEL", "qwen3-32b")
    return rec


def _service(provider):
    svc = LLMService()
    svc.provider = provider
    svc.default_model = "deepseek-v4-pro"
    return svc


def _saved(**over):
    entry = {
        "id": "gpu",
        "name": "GPU box",
        "kind": "openai_compatible",
        "api_url": "http://gpu-box:8000/v1",
        "default_model": "qwen3-32b",
        "api_key": "box-key",
    }
    entry.update(over)
    return catalog.normalize_custom_providers([entry], None)


async def test_openai_is_sent_to_openai_not_deepseek(sent):
    svc = _service("openai")
    await svc.generate_response(query="hi")

    (call,) = sent.calls
    assert call["branch"] == "provider" and call["provider"] == "openai"
    # build_provider fills the OPENAI_* settings in; nothing of DeepSeek's.
    assert call["api_url"] is None and call["api_key"] is None
    # Not DEFAULT_MODEL (deepseek-v4-pro): the provider's own default applies.
    assert call["model"] is None


async def test_prefer_deepseek_does_not_override_a_configured_provider(sent):
    await _service("openai").generate_response(query="hi", prefer_deepseek=True)
    assert sent.calls[-1]["branch"] == "provider"

    user = UserLLMSettings(provider="ollama")
    await _service("deepseek").generate_response(
        query="hi", prefer_deepseek=True, user_settings=user
    )
    assert sent.calls[-1]["branch"] == "ollama"


async def test_prefer_deepseek_still_moves_the_deployments_own_ollama(sent):
    await _service("ollama").generate_response(query="hi", prefer_deepseek=True)
    assert sent.calls[-1]["branch"] == "deepseek"


async def test_a_saved_provider_brings_its_own_endpoint_key_and_model(sent):
    user = UserLLMSettings(provider="custom:gpu", custom_providers=_saved())
    await _service("deepseek").generate_response(query="hi", user_settings=user)
    await _service("deepseek").generate_structured(
        user_message="hi", user_settings=user
    )

    for call in sent.calls:
        assert call["branch"] == "provider"
        assert call["provider"] == "openai_compatible"
        assert call["api_url"] == "http://gpu-box:8000/v1"
        assert call["api_key"] == "box-key"
        assert call["model"] == "qwen3-32b"


async def test_a_task_can_go_to_a_different_saved_provider(sent):
    stored = catalog.normalize_custom_providers(
        [
            {
                "name": "Box",
                "id": "gpu",
                "api_url": "http://gpu-box:8000/v1",
                "default_model": "big",
            },
            {
                "name": "Laptop",
                "id": "lap",
                "api_url": "http://laptop:1234/v1",
                "default_model": "small",
            },
        ],
        None,
    )
    user = UserLLMSettings(
        provider="custom:gpu",
        model="big-tuned",
        task_providers={"title_generation": "custom:lap"},
        custom_providers=stored,
    )
    svc = _service("deepseek")
    await svc.generate_response(query="t", user_settings=user, task_type="chat")
    await svc.generate_response(
        query="t", user_settings=user, task_type="title_generation"
    )
    chat, title = sent.calls
    assert (chat["api_url"], chat["model"]) == ("http://gpu-box:8000/v1", "big-tuned")
    # The default model belongs to the default provider, not to this one.
    assert (title["api_url"], title["model"]) == ("http://laptop:1234/v1", "small")


async def test_the_single_api_url_belongs_to_the_default_provider(sent):
    user = UserLLMSettings(
        api_url="http://proxy:9000/v1",
        api_key="k",
        task_providers={"summarization": "ollama"},
    )
    svc = _service("deepseek")
    await svc.generate_response(query="s", user_settings=user, task_type="chat")
    await svc.generate_response(
        query="s", user_settings=user, task_type="summarization"
    )
    chat, summary = sent.calls
    assert chat["branch"] == "custom" and chat["api_url"] == "http://proxy:9000/v1"
    assert summary["branch"] == "ollama"


async def test_a_deleted_saved_provider_falls_back_to_the_deployment(sent):
    user = UserLLMSettings(provider="custom:gone", custom_providers=_saved())
    await _service("deepseek").generate_response(query="hi", user_settings=user)
    assert sent.calls[-1]["branch"] == "deepseek"


def test_the_routing_preview_agrees():
    user = UserLLMSettings(provider="custom:gpu", custom_providers=_saved())
    got = resolve_effective_provider_model(
        system_provider="deepseek",
        system_default_model="deepseek-v4-pro",
        feature_default_model=None,
        user_settings=user,
        task_type="chat",
        model=None,
        provider_override=None,
        api_url_override=None,
    )
    assert got["provider_used"] == "openai_compatible"
    assert got["api_url"] == "http://gpu-box:8000/v1"
    assert got["effective_model"] == "qwen3-32b"

    plain = resolve_effective_provider_model(
        system_provider="openai",
        system_default_model="deepseek-v4-pro",
        feature_default_model=None,
        user_settings=None,
        task_type="chat",
        model=None,
        provider_override=None,
        api_url_override=None,
        prefer_deepseek=True,
    )
    assert plain["provider_used"] == "openai"


# --------------------------------------------------------------------------
# Storing saved providers


def test_a_key_is_stored_encrypted_and_kept_when_not_resent():
    first = _saved()
    assert "api_key" not in first[0]
    assert first[0]["api_key_enc"] != "box-key"
    assert catalog.decrypt_key(first[0]) == "box-key"

    again = catalog.normalize_custom_providers(
        [{"id": "gpu", "name": "GPU box", "api_url": "http://gpu-box:8000/v1/"}],
        first,
    )
    assert again[0]["api_key_enc"] == first[0]["api_key_enc"]
    assert again[0]["api_url"] == "http://gpu-box:8000/v1"

    cleared = catalog.normalize_custom_providers(
        [
            {
                "id": "gpu",
                "name": "GPU box",
                "api_url": "http://gpu-box:8000/v1",
                "clear_api_key": True,
            }
        ],
        first,
    )
    assert "api_key_enc" not in cleared[0]
    assert catalog.public_view(first)[0]["api_key_set"] is True
    assert "api_key_enc" not in catalog.public_view(first)[0]


@pytest.mark.parametrize(
    "entry, complaint",
    [
        ({"name": "x", "api_url": "ftp://host"}, r"http\(s\)"),
        ({"name": "", "api_url": "http://h"}, "name"),
        ({"name": "x", "api_url": "http://h", "kind": "grpc"}, "kind"),
        ({"name": "x", "api_url": "http://h", "id": "Bad Id"}, "may use only"),
    ],
)
def test_a_bad_entry_is_refused_with_its_reason(entry, complaint):
    with pytest.raises(ValueError, match=complaint):
        catalog.normalize_custom_providers([entry], None)


def test_every_template_kind_is_a_kind():
    for t in catalog.TEMPLATES:
        assert t["kind"] in catalog.CUSTOM_KINDS
    assert set(catalog.KIND_TO_PROVIDER) == set(catalog.CUSTOM_KINDS)


# --------------------------------------------------------------------------
# The settings API


def test_saved_providers_round_trip_without_their_key(client, auth_headers):
    put = client.put(
        "/api/v1/users/me/llm-settings",
        headers=auth_headers,
        json={
            "llm_providers": [
                {
                    "id": "gpu",
                    "name": "GPU box",
                    "api_url": "http://gpu-box:8000/v1",
                    "default_model": "qwen3-32b",
                    "api_key": "secret",
                }
            ],
            "llm_provider": "custom:gpu",
            "llm_task_providers": {"summarization": "openai"},
        },
    )
    assert put.status_code == 200, put.text
    body = put.json()
    assert body["llm_provider"] == "custom:gpu"
    (saved,) = body["llm_providers"]
    assert saved["api_key_set"] is True
    assert "secret" not in put.text

    got = client.get("/api/v1/users/me/llm-settings", headers=auth_headers).json()
    assert got["llm_providers"][0]["default_model"] == "qwen3-32b"


def test_a_choice_naming_nothing_is_refused(client, auth_headers):
    r = client.put(
        "/api/v1/users/me/llm-settings",
        headers=auth_headers,
        json={"llm_task_providers": {"chat": "custom:nope"}},
    )
    assert r.status_code == 400
    assert "custom:nope" in r.json()["detail"]


def test_deleting_a_provider_still_in_use_is_refused(client, auth_headers):
    client.put(
        "/api/v1/users/me/llm-settings",
        headers=auth_headers,
        json={
            "llm_providers": [{"id": "gpu", "name": "B", "api_url": "http://b/v1"}],
            "llm_provider": "custom:gpu",
        },
    )
    r = client.put(
        "/api/v1/users/me/llm-settings",
        headers=auth_headers,
        json={"llm_providers": []},
    )
    assert r.status_code == 400
    ok = client.put(
        "/api/v1/users/me/llm-settings",
        headers=auth_headers,
        json={"llm_providers": [], "llm_provider": None},
    )
    assert ok.status_code == 200 and ok.json()["llm_providers"] == []


def test_the_catalog_lists_builtins_and_templates(client, auth_headers):
    r = client.get("/api/v1/users/me/llm-providers", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    ids = {p["id"] for p in body["builtin"]}
    assert {"deepseek", "openai", "anthropic", "ollama"} <= ids
    assert any(t["id"] == "vllm" for t in body["templates"])


def test_testing_an_unreachable_endpoint_says_so(client, auth_headers):
    r = client.post(
        "/api/v1/users/me/llm-providers/test",
        headers=auth_headers,
        json={"kind": "openai_compatible", "api_url": "http://127.0.0.1:9/v1"},
    )
    assert r.status_code == 200
    # "Could not reach", or a status where a system proxy answers for it
    # (this happens on macOS, whose proxies httpx reads as the call paths do).
    assert r.json()["ok"] is False and r.json()["error"]
    assert r.json()["models"] == []


def test_the_preferences_response_does_not_return_the_key(client, auth_headers):
    client.put(
        "/api/v1/users/me/llm-settings",
        headers=auth_headers,
        json={"llm_api_key": "sk-very-secret"},
    )
    for path in ("/api/v1/users/me/preferences", "/api/v1/memory/preferences"):
        r = client.get(path, headers=auth_headers)
        assert r.status_code == 200, (path, r.text)
        assert "sk-very-secret" not in r.text
        assert "llm_api_key" not in r.json()
        assert r.json()["llm_api_key_set"] is True


# --------------------------------------------------------------------------
# Recording the model that ran


async def test_last_call_names_what_the_call_ran_on(sent):
    from app.services.llm_service import last_call

    await _service("openai").generate_response(query="hi")
    # The fake provider reports no model, so the provider's own default.
    assert last_call() == ("openai", "qwen3-32b")

    user = UserLLMSettings(provider="custom:gpu", custom_providers=_saved())
    await _service("deepseek").generate_response(query="hi", user_settings=user)
    assert last_call() == ("openai_compatible", "qwen3-32b")


async def test_a_failed_call_leaves_no_last_call(sent, monkeypatch):
    from app.services.llm_service import last_call

    await _service("openai").generate_response(query="hi")
    assert last_call() is not None

    async def broken(self, **kw):
        raise RuntimeError("down")

    monkeypatch.setattr(LLMService, "_make_deepseek_chat_request", broken)
    with pytest.raises(Exception):
        await _service("deepseek").generate_response(query="hi")
    assert last_call() is None


def test_active_model_is_the_providers_own(monkeypatch):
    monkeypatch.setattr(settings, "OPENAI_MODEL", "gpt-4o")
    monkeypatch.setattr(settings, "DEFAULT_MODEL", "llama3.2")
    assert _service("openai").get_active_model() == "gpt-4o"
    svc = _service("ollama")
    assert catalog.default_model_for("ollama") == "llama3.2"
    assert svc.get_active_model() == "llama3.2"
    assert catalog.default_model_for("openai_compatible") is None


async def test_a_summary_records_the_model_its_call_ran_on(sent, db_session):
    """It recorded DEFAULT_MODEL (deepseek-v4-pro) whatever the call was
    routed to; routed to a saved provider, that was a model never used."""
    from app.services.document_service import DocumentService
    from tests.factories import create_test_document, create_test_document_source

    source = await create_test_document_source(db_session)
    doc = await create_test_document(db_session, source, content="Some text. " * 20)
    service = DocumentService()
    service.llm = _service("deepseek")

    user = UserLLMSettings(provider="custom:gpu", custom_providers=_saved())
    await service.summarize_document(doc.id, db_session, user_settings=user)
    await db_session.refresh(doc)
    assert doc.summary_model == "qwen3-32b"
