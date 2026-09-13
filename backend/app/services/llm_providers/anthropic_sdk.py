"""Anthropic provider built on the official ``anthropic`` SDK.

Uses the Messages API with native tool use. Structured output is requested
via a forced synthetic tool (``emit_structured_output``) whose ``input_schema``
is the caller's response schema — this tolerates open-ended sub-schemas
(e.g. free-form ``params`` objects) that strict JSON-schema output modes
reject.

Notes:
- ``temperature``/``top_p`` are never sent — they are rejected by
  Opus 4.7+ models; prompting is the steering mechanism.
- A ``refusal`` stop reason is surfaced as ``LLMServiceError`` so the tier
  fallback in ``LLMService`` can try another provider.
- The SDK is imported lazily so the module stays importable in test
  environments without the dependency.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from app.core.config import settings
from app.utils.exceptions import LLMServiceError

from .base import BaseLLMProvider, LLMCompletion, LLMToolCall, to_anthropic_tools

STRUCTURED_OUTPUT_TOOL = "emit_structured_output"

_CLIENT_CACHE: Dict[tuple, Any] = {}


class AnthropicProvider(BaseLLMProvider):
    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        default_max_tokens: Optional[int] = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model or settings.ANTHROPIC_MODEL
        self.default_max_tokens = default_max_tokens or settings.ANTHROPIC_MAX_TOKENS

    def _get_client(self):
        if not self.api_key:
            raise LLMServiceError("ANTHROPIC_API_KEY is not set")
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise LLMServiceError(
                "The 'anthropic' package is required for the Anthropic provider "
                "(pip install anthropic)"
            )
        cache_key = (self.base_url, self.api_key)
        client = _CLIENT_CACHE.get(cache_key)
        if client is None:
            kwargs: Dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            client = AsyncAnthropic(**kwargs)
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
        client = self._get_client()

        system_text, converted = self._convert_messages(messages)
        cache_enabled = bool(getattr(settings, "ANTHROPIC_PROMPT_CACHE_ENABLED", True))
        if cache_enabled:
            self._mark_last_message_cacheable(converted)
        kwargs: Dict[str, Any] = {
            "model": model or self.default_model,
            "max_tokens": max_tokens or self.default_max_tokens,
            "messages": converted,
        }
        if system_text:
            kwargs["system"] = self._system_param(system_text, cache_enabled)

        anthropic_tools: List[Dict[str, Any]] = (
            to_anthropic_tools(tools) if tools else []
        )
        if response_schema and not tools:
            anthropic_tools.append(
                {
                    "name": STRUCTURED_OUTPUT_TOOL,
                    "description": (
                        "Emit the final answer as a structured object matching "
                        "the required schema. Always call this tool exactly once."
                    ),
                    "input_schema": response_schema,
                }
            )
            kwargs["tool_choice"] = {"type": "tool", "name": STRUCTURED_OUTPUT_TOOL}
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        if timeout_seconds:
            client = client.with_options(timeout=float(timeout_seconds))

        try:
            response = await client.messages.create(**kwargs)
        except LLMServiceError:
            raise
        except Exception as e:  # noqa: BLE001 - normalize SDK errors below
            status = getattr(e, "status_code", None)
            if status is not None:
                logger.error(f"Anthropic API error {status}: {e}")
                raise LLMServiceError(f"Anthropic API error: {status}")
            logger.error(f"Anthropic request error: {e}")
            raise LLMServiceError(f"Anthropic request error: {str(e)}")

        if getattr(response, "stop_reason", None) == "refusal":
            raise LLMServiceError("Anthropic model refused the request")

        return self._parse_message(response)

    @staticmethod
    def _system_param(system_text: str, cache_enabled: bool) -> Any:
        """System prompt param; with caching, a block carrying a breakpoint.

        The breakpoint on the system block caches tools + system together
        (render order is tools -> system -> messages), so a byte-stable
        system prompt makes every later request read that prefix from cache.
        """
        if not cache_enabled:
            return system_text
        return [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    @staticmethod
    def _mark_last_message_cacheable(converted: List[Dict[str, Any]]) -> None:
        """Add an incremental cache breakpoint on the newest message.

        Multi-round tool loops grow the conversation monotonically; a
        breakpoint on the newest message lets each round read the previous
        round's cache write. Two breakpoints total (system + here), well
        under the 4-breakpoint limit.
        """
        if not converted:
            return
        last = converted[-1]
        content = last.get("content")
        if isinstance(content, str) and content:
            last["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        elif isinstance(content, list) and content and isinstance(content[-1], dict):
            content[-1] = {**content[-1], "cache_control": {"type": "ephemeral"}}

    @staticmethod
    def _convert_messages(
        messages: List[Dict[str, Any]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Split out system text and convert to Anthropic message blocks."""
        system_parts: List[str] = []
        converted: List[Dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role == "system":
                if content:
                    system_parts.append(str(content))
                continue
            if role == "tool":
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": str(message.get("tool_call_id") or ""),
                                "content": str(content or ""),
                            }
                        ],
                    }
                )
                continue
            tool_calls = message.get("tool_calls")
            if role == "assistant" and tool_calls:
                blocks: List[Dict[str, Any]] = []
                if content:
                    blocks.append({"type": "text", "text": str(content)})
                for idx, tc in enumerate(tool_calls):
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": str(tc.get("id") or f"call_{idx}"),
                            "name": str(tc.get("name") or ""),
                            "input": tc.get("arguments") or {},
                        }
                    )
                converted.append({"role": "assistant", "content": blocks})
                continue
            converted.append({"role": role, "content": str(content or "")})
        return "\n\n".join(system_parts), converted

    @staticmethod
    def _parse_message(response: Any) -> LLMCompletion:
        text_parts: List[str] = []
        tool_calls: List[LLMToolCall] = []
        structured: Optional[Dict[str, Any]] = None

        for block in getattr(response, "content", None) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(str(getattr(block, "text", "") or ""))
            elif block_type == "tool_use":
                name = str(getattr(block, "name", "") or "")
                block_input = getattr(block, "input", None)
                if not isinstance(block_input, dict):
                    block_input = {}
                if name == STRUCTURED_OUTPUT_TOOL:
                    structured = block_input
                else:
                    tool_calls.append(
                        LLMToolCall(
                            id=str(getattr(block, "id", "") or ""),
                            name=name,
                            arguments=block_input,
                        )
                    )

        text = "\n".join(part for part in text_parts if part).strip()
        if structured is not None and not text:
            text = json.dumps(structured)

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", None)
        completion_tokens = getattr(usage, "output_tokens", None)
        total_tokens = None
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            total_tokens = prompt_tokens + completion_tokens

        return LLMCompletion(
            text=text,
            tool_calls=tool_calls,
            structured=structured,
            provider="anthropic",
            model=str(getattr(response, "model", "") or ""),
            stop_reason=getattr(response, "stop_reason", None),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw={
                "id": getattr(response, "id", None),
                "cache_read_input_tokens": getattr(
                    usage, "cache_read_input_tokens", None
                ),
                "cache_creation_input_tokens": getattr(
                    usage, "cache_creation_input_tokens", None
                ),
            },
        )
