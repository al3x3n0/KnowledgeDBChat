"""Ollama provider using the native ``/api/chat`` endpoint.

Unlike the legacy ``/api/generate`` text path in ``llm_service.py``, this
provider uses chat messages plus Ollama's native ``tools`` (function calling)
and ``format`` (JSON-schema constrained output) parameters.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from loguru import logger

from app.core.config import settings
from app.utils.exceptions import LLMServiceError

from .base import (
    BaseLLMProvider,
    LLMCompletion,
    LLMToolCall,
    to_openai_tools,
    try_parse_json_object,
)


class OllamaProvider(BaseLLMProvider):
    name = "ollama"

    def __init__(
        self,
        base_url: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self._client = http_client

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
        resolved_model = model or settings.DEFAULT_MODEL
        payload: Dict[str, Any] = {
            "model": resolved_model,
            "messages": self._convert_messages(messages),
            "stream": False,
            "options": {
                "temperature": temperature
                if temperature is not None
                else settings.TEMPERATURE,
                "num_predict": max_tokens or settings.MAX_RESPONSE_LENGTH,
                "top_p": settings.TOP_P,
            },
        }
        if tools:
            payload["tools"] = to_openai_tools(tools)
        if response_schema:
            payload["format"] = response_schema

        timeout = float(timeout_seconds or 120)
        client = self._client
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=timeout)
        try:
            response = await client.post(
                f"{self.base_url}/api/chat", json=payload, timeout=timeout
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Ollama chat API error: {e.response.status_code} - {e.response.text}"
            )
            raise LLMServiceError(f"Ollama API error: {e.response.status_code}")
        except httpx.TimeoutException:
            logger.error("Ollama chat request timed out")
            raise LLMServiceError("Request timed out")
        except httpx.RequestError as e:
            logger.error(f"Ollama chat request error: {e}")
            raise LLMServiceError(f"Request error: {str(e)}")
        finally:
            if owns_client and client is not None:
                await client.aclose()

        return self._parse_chat_response(
            data, resolved_model, expect_structured=bool(response_schema)
        )

    @staticmethod
    def _convert_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "tool":
                converted.append(
                    {"role": "tool", "content": str(message.get("content") or "")}
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
                        "function": {
                            "name": tc.get("name"),
                            "arguments": tc.get("arguments") or {},
                        }
                    }
                    for tc in tool_calls
                ]
            converted.append(entry)
        return converted

    @staticmethod
    def _parse_chat_response(
        data: Dict[str, Any],
        model: str,
        *,
        expect_structured: bool = False,
    ) -> LLMCompletion:
        message = data.get("message") or {}
        text = str(message.get("content") or "").strip()

        tool_calls: List[LLMToolCall] = []
        for idx, tc in enumerate(message.get("tool_calls") or []):
            function = (tc or {}).get("function") or {}
            arguments = function.get("arguments")
            if not isinstance(arguments, dict):
                arguments = try_parse_json_object(str(arguments or "")) or {}
            tool_calls.append(
                LLMToolCall(
                    id=str(tc.get("id") or f"call_{idx}"),
                    name=str(function.get("name") or ""),
                    arguments=arguments,
                )
            )

        prompt_tokens = data.get("prompt_eval_count")
        completion_tokens = data.get("eval_count")
        total_tokens = None
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            total_tokens = prompt_tokens + completion_tokens

        return LLMCompletion(
            text=text,
            tool_calls=tool_calls,
            structured=try_parse_json_object(text) if expect_structured else None,
            provider="ollama",
            model=str(data.get("model") or model),
            stop_reason=data.get("done_reason"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            raw={
                "total_duration": data.get("total_duration"),
                "load_duration": data.get("load_duration"),
                "prompt_eval_duration": data.get("prompt_eval_duration"),
                "eval_duration": data.get("eval_duration"),
            },
        )
