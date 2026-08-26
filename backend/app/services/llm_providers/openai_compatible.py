"""OpenAI-compatible provider built on the official ``openai`` SDK.

Serves three configurations from one implementation:

- OpenAI itself (``provider_label="openai"``, ``schema_mode="json_schema"``)
- DeepSeek (``provider_label="deepseek"``, ``schema_mode="json_object"`` —
  DeepSeek's chat completions API does not enforce arbitrary JSON schemas)
- User-configured custom endpoints (``provider_label="custom"``)

The SDK is imported lazily so the module can be imported (and unit-tested)
in environments where ``openai`` is not installed.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from loguru import logger

from app.utils.exceptions import LLMServiceError

from .base import (
    BaseLLMProvider,
    LLMCompletion,
    LLMToolCall,
    to_openai_tools,
    try_parse_json_object,
)

_CLIENT_CACHE: Dict[tuple, Any] = {}


class OpenAICompatibleProvider(BaseLLMProvider):
    name = "openai"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        schema_mode: str = "json_schema",
        provider_label: str = "openai",
        timeout_seconds: int = 120,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/") if base_url else None
        self.default_model = default_model
        self.schema_mode = schema_mode
        self.provider_label = provider_label
        self.name = provider_label
        self.timeout_seconds = timeout_seconds

    def _get_client(self):
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise LLMServiceError(
                "The 'openai' package is required for OpenAI-compatible providers "
                "(pip install openai)"
            )
        cache_key = (self.base_url, self.api_key)
        client = _CLIENT_CACHE.get(cache_key)
        if client is None:
            client = AsyncOpenAI(
                api_key=self.api_key or "not-set",
                base_url=self.base_url,
                timeout=float(self.timeout_seconds),
            )
            _CLIENT_CACHE[cache_key] = client
        return client

    async def complete(
        self,
        messages: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
    ) -> LLMCompletion:
        resolved_model = model or self.default_model
        if not resolved_model:
            raise LLMServiceError(
                f"No model configured for provider '{self.provider_label}'"
            )

        kwargs: Dict[str, Any] = {
            "model": resolved_model,
            "messages": self._convert_messages(messages),
            "stream": False,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = to_openai_tools(tools)
            kwargs["tool_choice"] = "auto"
        if response_schema:
            if self.schema_mode == "json_schema":
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "schema": response_schema},
                }
            else:
                # json_object mode guarantees valid JSON but not schema shape;
                # callers keep validating the payload downstream.
                kwargs["response_format"] = {"type": "json_object"}

        client = self._get_client()
        if timeout_seconds:
            client = client.with_options(timeout=float(timeout_seconds))

        try:
            response = await client.chat.completions.create(**kwargs)
        except LLMServiceError:
            raise
        except Exception as e:  # noqa: BLE001 - normalize SDK errors below
            status = getattr(e, "status_code", None)
            label = self.provider_label
            if status is not None:
                logger.error(f"{label} API error {status}: {e}")
                raise LLMServiceError(f"{label} API error: {status}")
            logger.error(f"{label} request error: {e}")
            raise LLMServiceError(f"{label} request error: {str(e)}")

        return self._parse_response(
            response,
            provider=self.provider_label,
            fallback_model=resolved_model,
            expect_structured=bool(response_schema),
        )

    @staticmethod
    def _convert_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "tool":
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(message.get("tool_call_id") or ""),
                        "content": str(message.get("content") or ""),
                    }
                )
                continue
            entry: Dict[str, Any] = {
                "role": role,
                "content": str(message.get("content") or ""),
            }
            tool_calls = message.get("tool_calls")
            if role == "assistant" and tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": str(tc.get("id") or f"call_{idx}"),
                        "type": "function",
                        "function": {
                            "name": tc.get("name"),
                            "arguments": json.dumps(tc.get("arguments") or {}),
                        },
                    }
                    for idx, tc in enumerate(tool_calls)
                ]
            converted.append(entry)
        return converted

    @staticmethod
    def _parse_response(
        response: Any,
        *,
        provider: str,
        fallback_model: str,
        expect_structured: bool = False,
    ) -> LLMCompletion:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMServiceError(f"{provider} returned no choices")
        choice = choices[0]
        message = getattr(choice, "message", None)
        text = str(getattr(message, "content", None) or "").strip()

        tool_calls: List[LLMToolCall] = []
        for idx, tc in enumerate(getattr(message, "tool_calls", None) or []):
            function = getattr(tc, "function", None)
            raw_args = getattr(function, "arguments", None) or "{}"
            arguments = try_parse_json_object(raw_args)
            if arguments is None:
                arguments = {"_raw": raw_args}
            tool_calls.append(
                LLMToolCall(
                    id=str(getattr(tc, "id", None) or f"call_{idx}"),
                    name=str(getattr(function, "name", None) or ""),
                    arguments=arguments,
                )
            )

        # The chain of thought, on the path the agent's decisions actually take.
        # The prompted-text client already records this; the structured client
        # did not, so the thinking phase -- the one call whose reasoning is
        # worth having -- was the only one still discarding it.
        reasoning = getattr(message, "reasoning_content", None) or getattr(
            message, "reasoning", None
        )
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None)
        if reasoning:
            try:
                from app.services.llm_service import _LAST_REASONING

                _LAST_REASONING.set((str(reasoning), reasoning_tokens))
            except Exception:  # pragma: no cover - capture must never break a call
                pass

        return LLMCompletion(
            text=text,
            tool_calls=tool_calls,
            structured=try_parse_json_object(text) if expect_structured else None,
            provider=provider,
            model=str(getattr(response, "model", None) or fallback_model),
            stop_reason=getattr(choice, "finish_reason", None),
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            total_tokens=getattr(usage, "total_tokens", None),
            raw={
                "id": getattr(response, "id", None),
                "reasoning_tokens": reasoning_tokens,
                "reasoning_chars": len(str(reasoning)) if reasoning else 0,
            },
        )
