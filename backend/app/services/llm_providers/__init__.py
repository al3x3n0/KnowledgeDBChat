"""Native LLM provider module.

Exposes a common provider interface (:class:`BaseLLMProvider`) with native
tool calling and structured output, plus :func:`build_provider` which maps
the platform's provider names ("ollama", "deepseek", "openai", "anthropic",
"qwen", "kimi", "custom") to concrete implementations.

Used by ``LLMService.generate_structured()``; the legacy prompted-text path
in ``llm_service.py`` remains unchanged for existing callers.
"""

from typing import Any, Optional

from app.core.config import settings

from .base import (
    BaseLLMProvider,
    LLMCompletion,
    LLMToolCall,
    normalize_tool,
    to_anthropic_tools,
    to_openai_tools,
    try_parse_json_object,
)
from .anthropic_sdk import STRUCTURED_OUTPUT_TOOL, AnthropicProvider
from .ollama import OllamaProvider
from .openai_compatible import OpenAICompatibleProvider

__all__ = [
    "BaseLLMProvider",
    "LLMCompletion",
    "LLMToolCall",
    "AnthropicProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "STRUCTURED_OUTPUT_TOOL",
    "build_provider",
    "normalize_tool",
    "to_anthropic_tools",
    "to_openai_tools",
    "try_parse_json_object",
]


def build_provider(
    provider: Optional[str],
    *,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None,
    http_client: Optional[Any] = None,
) -> BaseLLMProvider:
    """Build a provider instance for the given provider name.

    ``api_url``/``api_key`` are per-call overrides (user settings or agent
    routing); system settings fill the gaps.
    """
    name = (provider or "").strip().lower()

    if name == "anthropic":
        return AnthropicProvider(
            api_key=api_key or settings.ANTHROPIC_API_KEY,
            base_url=api_url,
        )
    if name == "deepseek":
        return OpenAICompatibleProvider(
            api_key=api_key or settings.DEEPSEEK_API_KEY,
            base_url=api_url or settings.DEEPSEEK_API_BASE,
            default_model=settings.DEEPSEEK_MODEL,
            schema_mode="json_object",
            provider_label="deepseek",
            timeout_seconds=int(settings.DEEPSEEK_TIMEOUT_SECONDS or 120),
        )
    if name == "openai":
        return OpenAICompatibleProvider(
            api_key=api_key or settings.OPENAI_API_KEY,
            base_url=api_url or settings.OPENAI_API_BASE,
            default_model=settings.OPENAI_MODEL,
            provider_label="openai",
        )
    if name == "qwen":
        # DashScope compatible mode: native tool calling; JSON output is
        # json_object only (no arbitrary schema enforcement).
        return OpenAICompatibleProvider(
            api_key=api_key or settings.QWEN_API_KEY,
            base_url=api_url or settings.QWEN_API_BASE,
            default_model=settings.QWEN_MODEL,
            schema_mode="json_object",
            provider_label="qwen",
        )
    if name == "kimi":
        # Moonshot AI: native tool calling; JSON output is json_object only.
        return OpenAICompatibleProvider(
            api_key=api_key or settings.KIMI_API_KEY,
            base_url=api_url or settings.KIMI_API_BASE,
            default_model=settings.KIMI_MODEL,
            schema_mode="json_object",
            provider_label="kimi",
        )
    if name == "ollama" or (not name and not api_url):
        return OllamaProvider(base_url=api_url, http_client=http_client)

    # Unknown provider with an explicit endpoint: treat as OpenAI-compatible.
    return OpenAICompatibleProvider(
        api_key=api_key,
        base_url=api_url,
        schema_mode="json_object",
        provider_label="custom",
    )
