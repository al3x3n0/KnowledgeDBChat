"""What a person can choose as their LLM provider, and their own saved ones.

Two kinds of choice. A *built-in* provider is one the deployment knows by
name (``deepseek``, ``openai``, ...): its endpoint, key and default model come
from settings unless the person supplies their own. A *custom* provider is a
connection the person saved -- a vLLM or LM Studio on their own machine, a
LiteLLM gateway, OpenRouter -- stored in ``user_preferences.llm_providers``
and referred to everywhere else as ``custom:<id>``, so a per-task override can
name it the same way it names a built-in one.

The catalog is served to the settings page rather than restated there, so the
list of providers, their defaults and whether the deployment holds a key for
each cannot drift from what the call paths actually do.

A saved key is encrypted with the secrets store's key and never returned:
the page is told only whether one is set.
"""

from __future__ import annotations

import re
import secrets as _secrets
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx
from loguru import logger

from app.core.config import settings

CUSTOM_PREFIX = "custom:"

#: How a saved connection is spoken to. ``openai_compatible`` covers nearly
#: every local server and gateway; the other two are for an Anthropic- or
#: Ollama-native endpoint somewhere other than the deployment's own.
CUSTOM_KINDS = ("openai_compatible", "anthropic", "ollama")

#: The provider name the call paths use for each kind. ``openai_compatible``
#: is its own name rather than ``custom``: ``custom`` is what a legacy
#: single API URL resolves to, and that path appends ``/v1`` to a URL that
#: lacks it, while a saved connection's URL is used exactly as entered.
KIND_TO_PROVIDER = {
    "openai_compatible": "openai_compatible",
    "anthropic": "anthropic",
    "ollama": "ollama",
}

MAX_CUSTOM_PROVIDERS = 20
MAX_MODELS_PER_PROVIDER = 50
_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")


@dataclass(frozen=True)
class ResolvedConnection:
    """Where one call goes: the provider name the call paths understand, and
    whatever the person's settings say about endpoint, key and model. Each
    field is None when the settings say nothing about it."""

    provider: Optional[str] = None
    model: Optional[str] = None
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    # True when a person's setting (not the deployment) chose the provider.
    chosen_by_user: bool = False


# --------------------------------------------------------------------------
# Built-in providers and templates


def builtin_providers() -> List[Dict[str, Any]]:
    """Every provider the deployment knows by name, with its defaults and
    whether it can be used without a key of one's own."""
    s = settings
    rows = [
        (
            "deepseek",
            "DeepSeek",
            s.DEEPSEEK_API_BASE,
            s.DEEPSEEK_MODEL,
            s.DEEPSEEK_API_KEY,
            ["deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp"],
        ),
        (
            "openai",
            "OpenAI",
            s.OPENAI_API_BASE,
            s.OPENAI_MODEL,
            s.OPENAI_API_KEY,
            ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "o1", "o1-mini"],
        ),
        (
            "anthropic",
            "Anthropic",
            "https://api.anthropic.com",
            s.ANTHROPIC_MODEL,
            s.ANTHROPIC_API_KEY,
            ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"],
        ),
        (
            "qwen",
            "Qwen (DashScope)",
            s.QWEN_API_BASE,
            s.QWEN_MODEL,
            s.QWEN_API_KEY,
            [],
        ),
        (
            "kimi",
            "Kimi (Moonshot)",
            s.KIMI_API_BASE,
            s.KIMI_MODEL,
            s.KIMI_API_KEY,
            [],
        ),
        (
            "glm",
            "GLM (Zhipu)",
            s.GLM_API_BASE,
            s.GLM_MODEL,
            s.GLM_API_KEY,
            [],
        ),
    ]
    out: List[Dict[str, Any]] = []
    for pid, label, base, model, key, known in rows:
        models = [model] + [m for m in known if m != model] if model else known
        out.append(
            {
                "id": pid,
                "label": label,
                "api_url": base,
                "default_model": model,
                "models": models,
                "system_configured": bool(key),
                "requires_api_key": True,
            }
        )
    out.append(
        {
            "id": "ollama",
            "label": "Ollama",
            "api_url": s.OLLAMA_BASE_URL,
            "default_model": s.DEFAULT_MODEL,
            "models": [],
            # Whether it answers is a question for the model list, which asks.
            "system_configured": (s.LLM_PROVIDER or "").lower() == "ollama",
            "requires_api_key": False,
        }
    )
    return out


BUILTIN_IDS = frozenset(p["id"] for p in builtin_providers())


def default_model_for(provider: Optional[str]) -> Optional[str]:
    """The model a built-in provider falls back to when no call names one
    (``ollama`` falls back to DEFAULT_MODEL), or None for anything else."""
    name = (provider or "").strip().lower()
    for p in builtin_providers():
        if p["id"] == name:
            return p["default_model"]
    return None


#: Starting points for a custom provider. Only a template: everything is
#: editable, and nothing here is reachable until a person saves it.
TEMPLATES: List[Dict[str, Any]] = [
    {
        "id": "vllm",
        "label": "vLLM",
        "kind": "openai_compatible",
        "api_url": "http://localhost:8000/v1",
        "requires_api_key": False,
        "description": "vLLM's OpenAI-compatible server (vllm serve <model>).",
    },
    {
        "id": "lm-studio",
        "label": "LM Studio",
        "kind": "openai_compatible",
        "api_url": "http://localhost:1234/v1",
        "requires_api_key": False,
        "description": "LM Studio's local server.",
    },
    {
        "id": "llama-cpp",
        "label": "llama.cpp server",
        "kind": "openai_compatible",
        "api_url": "http://localhost:8080/v1",
        "requires_api_key": False,
        "description": "llama-server from llama.cpp.",
    },
    {
        "id": "ollama",
        "label": "Ollama (another host)",
        "kind": "ollama",
        "api_url": "http://localhost:11434",
        "requires_api_key": False,
        "description": "An Ollama instance other than the deployment's own.",
    },
    {
        "id": "litellm",
        "label": "LiteLLM proxy",
        "kind": "openai_compatible",
        "api_url": "http://localhost:4000/v1",
        "requires_api_key": True,
        "description": "A LiteLLM gateway in front of other providers.",
    },
    {
        "id": "openrouter",
        "label": "OpenRouter",
        "kind": "openai_compatible",
        "api_url": "https://openrouter.ai/api/v1",
        "requires_api_key": True,
        "description": "Hosted router to many model vendors.",
    },
    {
        "id": "groq",
        "label": "Groq",
        "kind": "openai_compatible",
        "api_url": "https://api.groq.com/openai/v1",
        "requires_api_key": True,
        "description": "Groq's OpenAI-compatible API.",
    },
    {
        "id": "together",
        "label": "Together AI",
        "kind": "openai_compatible",
        "api_url": "https://api.together.xyz/v1",
        "requires_api_key": True,
        "description": "Together AI's OpenAI-compatible API.",
    },
    {
        "id": "openai-compatible",
        "label": "Other OpenAI-compatible",
        "kind": "openai_compatible",
        "api_url": "",
        "requires_api_key": False,
        "description": "Any server speaking /chat/completions.",
    },
]


# --------------------------------------------------------------------------
# Saved custom providers


def _secret_service():
    from app.services.secret_service import SecretService

    return SecretService()


def _clean_str(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _check_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"API URL must be an http(s) URL, got {url!r}")
    return url.rstrip("/")


def normalize_custom_providers(
    submitted: Any, existing: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """The stored form of a submitted list of custom providers.

    Each entry keeps its ``id`` when it sends one; a new entry gets one. A
    key is encrypted when sent; an entry that sends none keeps the key it
    had (the page never holds the old key to send back), and
    ``clear_api_key`` removes it. Raises ValueError naming what is wrong.
    """
    if submitted is None:
        return []
    if not isinstance(submitted, list):
        raise ValueError("llm_providers must be a list")
    if len(submitted) > MAX_CUSTOM_PROVIDERS:
        raise ValueError(f"At most {MAX_CUSTOM_PROVIDERS} custom providers")

    previous = {str(p.get("id")): p for p in (existing or []) if isinstance(p, dict)}
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for raw in submitted:
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump()
        if not isinstance(raw, dict):
            raise ValueError("Each custom provider must be an object")
        pid = _clean_str(raw.get("id"), 32).lower()
        if not pid:
            pid = _secrets.token_hex(4)
            while pid in seen or pid in previous:
                pid = _secrets.token_hex(4)
        if not _ID.match(pid):
            raise ValueError(
                f"Custom provider id {pid!r} may use only a-z, 0-9 and '-'"
            )
        if pid in seen:
            raise ValueError(f"Two custom providers share the id {pid!r}")
        seen.add(pid)

        name = _clean_str(raw.get("name"), 60)
        if not name:
            raise ValueError("Each custom provider needs a name")
        kind = _clean_str(raw.get("kind") or "openai_compatible", 32)
        if kind not in CUSTOM_KINDS:
            raise ValueError(
                f"Unknown kind {kind!r}; expected one of {', '.join(CUSTOM_KINDS)}"
            )
        api_url = _clean_str(raw.get("api_url"), 500)
        if not api_url:
            raise ValueError(f"Custom provider {name!r} needs an API URL")
        api_url = _check_url(api_url)

        models_raw = raw.get("models") or []
        if not isinstance(models_raw, list):
            raise ValueError("models must be a list of names")
        models: List[str] = []
        for m in models_raw:
            text = _clean_str(m, 100)
            if text and text not in models:
                models.append(text)
        models = models[:MAX_MODELS_PER_PROVIDER]

        entry: Dict[str, Any] = {
            "id": pid,
            "name": name,
            "kind": kind,
            "api_url": api_url,
            "default_model": _clean_str(raw.get("default_model"), 100) or None,
            "models": models,
        }
        new_key = raw.get("api_key")
        if new_key:
            entry["api_key_enc"] = _secret_service().encrypt(str(new_key)[:500])
        elif not raw.get("clear_api_key"):
            kept = (previous.get(pid) or {}).get("api_key_enc")
            if kept:
                entry["api_key_enc"] = kept
        out.append(entry)
    return out


def public_view(stored: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """A saved list as the page sees it: everything but the key."""
    out = []
    for p in stored or []:
        if not isinstance(p, dict):
            continue
        out.append(
            {
                "id": p.get("id"),
                "name": p.get("name"),
                "kind": p.get("kind"),
                "api_url": p.get("api_url"),
                "default_model": p.get("default_model"),
                "models": list(p.get("models") or []),
                "api_key_set": bool(p.get("api_key_enc")),
            }
        )
    return out


def find_custom(
    stored: Optional[List[Dict[str, Any]]], ref: Optional[str]
) -> Optional[Dict[str, Any]]:
    """The saved provider ``ref`` (``custom:<id>``) names, or None."""
    if not ref or not str(ref).startswith(CUSTOM_PREFIX):
        return None
    pid = str(ref)[len(CUSTOM_PREFIX) :]
    for p in stored or []:
        if isinstance(p, dict) and p.get("id") == pid:
            return p
    return None


def decrypt_key(entry: Optional[Dict[str, Any]]) -> Optional[str]:
    token = (entry or {}).get("api_key_enc")
    if not token:
        return None
    return _secret_service().decrypt(str(token))


def unknown_references(
    refs: List[Optional[str]], stored: Optional[List[Dict[str, Any]]]
) -> List[str]:
    """The provider choices in ``refs`` that name nothing: neither a
    built-in provider nor one of the saved ones."""
    bad = []
    for ref in refs:
        if not ref:
            continue
        text = str(ref).strip().lower()
        if text in BUILTIN_IDS:
            continue
        if find_custom(stored, text) is not None:
            continue
        bad.append(str(ref))
    return bad


# --------------------------------------------------------------------------
# Asking an endpoint which models it serves


async def list_models(
    kind: str, api_url: str, api_key: Optional[str], *, timeout: float = 10.0
) -> Dict[str, Any]:
    """Ask an endpoint for its models: ``{"ok", "models", "error"}``.

    Only model names come back, never the response body, and an error says
    what failed (unreachable, a status) rather than quoting the server.
    """
    base = (api_url or "").rstrip("/")
    try:
        _check_url(base)
    except ValueError as exc:
        return {"ok": False, "models": [], "error": str(exc)}

    headers: Dict[str, str] = {}
    if kind == "ollama":
        url = f"{base}/api/tags"
    elif kind == "anthropic":
        url = f"{base}/v1/models" if not base.endswith("/v1") else f"{base}/models"
        if api_key:
            headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    else:
        url = f"{base}/models"
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as c:
            response = await c.get(url, headers=headers)
    except httpx.HTTPError as exc:
        return {
            "ok": False,
            "models": [],
            "error": f"Could not reach {base}: {type(exc).__name__}",
        }
    if response.status_code in (401, 403):
        return {
            "ok": False,
            "models": [],
            "error": f"{base} refused the key (HTTP {response.status_code})",
        }
    if response.status_code >= 400:
        return {
            "ok": False,
            "models": [],
            "error": f"{url} answered HTTP {response.status_code}",
        }
    try:
        data = response.json()
    except ValueError:
        return {"ok": False, "models": [], "error": f"{url} did not answer JSON"}

    names: List[str] = []
    items = []
    if isinstance(data, dict):
        items = data.get("models") if kind == "ollama" else data.get("data")
    for item in items or []:
        if isinstance(item, dict):
            name = item.get("name") if kind == "ollama" else item.get("id")
            if name:
                names.append(str(name)[:100])
    if not names:
        logger.debug(f"{url} answered with no model names")
    return {"ok": True, "models": sorted(set(names)), "error": None}
