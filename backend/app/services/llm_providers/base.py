"""Native LLM provider abstraction: tool calling and structured output.

Canonical formats shared by all providers:

- Messages use OpenAI-style dicts::

    {"role": "system" | "user" | "assistant", "content": "..."}
    {"role": "assistant", "content": "...", "tool_calls": [{"id": ..., "name": ..., "arguments": {...}}]}
    {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}

- Tools use the registry format already used by ``agent_tools.py``::

    {"name": "...", "description": "...", "parameters": {<JSON Schema>}}

Each provider translates these into its wire format and normalizes results
into :class:`LLMCompletion`, so callers never parse provider payloads.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class LLMToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMCompletion:
    """Normalized result of one provider completion."""

    text: str = ""
    tool_calls: List[LLMToolCall] = field(default_factory=list)
    structured: Optional[Dict[str, Any]] = None
    provider: str = ""
    model: str = ""
    stop_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class BaseLLMProvider(ABC):
    """Interface all native providers implement."""

    name: str = "base"
    supports_tools: bool = True
    supports_structured_output: bool = True

    @abstractmethod
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
        """Run one completion and return a normalized result.

        ``tools`` requests native tool calling; ``response_schema`` requests
        schema-constrained JSON output (populated on ``LLMCompletion.structured``).
        """


def normalize_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a registry tool dict to {name, description, parameters}."""
    parameters = (
        tool.get("parameters")
        or tool.get("input_schema")
        or {
            "type": "object",
            "properties": {},
        }
    )
    return {
        "name": str(tool.get("name") or ""),
        "description": str(tool.get("description") or ""),
        "parameters": parameters,
    }


def to_openai_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert registry tools to OpenAI/Ollama function-calling format."""
    result = []
    for tool in tools:
        t = normalize_tool(tool)
        result.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
        )
    return result


def to_anthropic_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert registry tools to Anthropic Messages API format."""
    result = []
    for tool in tools:
        t = normalize_tool(tool)
        result.append(
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
        )
    return result


def try_parse_json_object(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """Best-effort parse of a JSON object from model output text."""
    if not text:
        return None
    try:
        parsed = json.loads(text.strip())
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None
