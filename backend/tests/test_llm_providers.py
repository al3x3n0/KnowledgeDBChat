"""Unit tests for the native LLM provider module (app.services.llm_providers).

These tests exercise message/tool conversion and response normalization with
stubbed payloads — no network and no provider SDK imports required.
"""

import json
from types import SimpleNamespace

import pytest

from app.services.llm_providers import build_provider, try_parse_json_object
from app.services.llm_providers.anthropic_sdk import STRUCTURED_OUTPUT_TOOL, AnthropicProvider
from app.services.llm_providers.base import to_anthropic_tools, to_openai_tools
from app.services.llm_providers.ollama import OllamaProvider
from app.services.llm_providers.openai_compatible import OpenAICompatibleProvider

SAMPLE_TOOLS = [
    {
        "name": "search_documents",
        "description": "Search the knowledge base",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    }
]


class TestToolConversion:
    def test_to_openai_tools(self):
        converted = to_openai_tools(SAMPLE_TOOLS)
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "search_documents"
        assert converted[0]["function"]["parameters"]["required"] == ["query"]

    def test_to_anthropic_tools(self):
        converted = to_anthropic_tools(SAMPLE_TOOLS)
        assert converted[0]["name"] == "search_documents"
        assert converted[0]["input_schema"]["properties"]["query"]["type"] == "string"

    def test_missing_parameters_defaults_to_empty_object_schema(self):
        converted = to_openai_tools([{"name": "noop", "description": ""}])
        assert converted[0]["function"]["parameters"] == {
            "type": "object",
            "properties": {},
        }


class TestTryParseJsonObject:
    def test_parses_object(self):
        assert try_parse_json_object('{"a": 1}') == {"a": 1}

    def test_rejects_non_object(self):
        assert try_parse_json_object("[1, 2]") is None
        assert try_parse_json_object("not json") is None
        assert try_parse_json_object("") is None


class TestOllamaProvider:
    def test_parse_chat_response_with_tool_calls(self):
        data = {
            "model": "llama3.2:3b",
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "search_documents", "arguments": {"query": "x"}}}
                ],
            },
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "eval_count": 20,
        }
        completion = OllamaProvider._parse_chat_response(data, "llama3.2:3b")
        assert completion.provider == "ollama"
        assert completion.has_tool_calls
        assert completion.tool_calls[0].name == "search_documents"
        assert completion.tool_calls[0].arguments == {"query": "x"}
        assert completion.total_tokens == 120

    def test_parse_chat_response_structured(self):
        data = {"message": {"content": '{"goal_achieved": false}'}}
        completion = OllamaProvider._parse_chat_response(
            data, "m", expect_structured=True
        )
        assert completion.structured == {"goal_achieved": False}

    def test_convert_messages_tool_roles(self):
        messages = [
            {"role": "system", "content": "sys"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "name": "t", "arguments": {"a": 1}}],
            },
            {"role": "tool", "tool_call_id": "1", "name": "t", "content": "result"},
        ]
        converted = OllamaProvider._convert_messages(messages)
        assert converted[0] == {"role": "system", "content": "sys"}
        assert converted[1]["tool_calls"][0]["function"]["name"] == "t"
        assert converted[2] == {"role": "tool", "content": "result"}


class TestOpenAICompatibleProvider:
    def _response(self, message, finish_reason="stop"):
        return SimpleNamespace(
            id="resp-1",
            model="gpt-4o",
            choices=[SimpleNamespace(message=message, finish_reason=finish_reason)],
            usage=SimpleNamespace(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )

    def test_parse_response_text(self):
        message = SimpleNamespace(content="hello", tool_calls=None)
        completion = OpenAICompatibleProvider._parse_response(
            self._response(message), provider="openai", fallback_model="gpt-4o"
        )
        assert completion.text == "hello"
        assert completion.total_tokens == 15
        assert completion.stop_reason == "stop"

    def test_parse_response_tool_calls(self):
        tool_call = SimpleNamespace(
            id="call_abc",
            function=SimpleNamespace(
                name="search_documents", arguments='{"query": "kg"}'
            ),
        )
        message = SimpleNamespace(content=None, tool_calls=[tool_call])
        completion = OpenAICompatibleProvider._parse_response(
            self._response(message, "tool_calls"),
            provider="deepseek",
            fallback_model="deepseek-chat",
        )
        assert completion.tool_calls[0].id == "call_abc"
        assert completion.tool_calls[0].arguments == {"query": "kg"}

    def test_parse_response_malformed_tool_arguments(self):
        tool_call = SimpleNamespace(
            id="call_1",
            function=SimpleNamespace(name="t", arguments="{broken"),
        )
        message = SimpleNamespace(content=None, tool_calls=[tool_call])
        completion = OpenAICompatibleProvider._parse_response(
            self._response(message), provider="openai", fallback_model="m"
        )
        assert completion.tool_calls[0].arguments == {"_raw": "{broken"}

    def test_convert_messages_round_trip(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "name": "t", "arguments": {"x": 1}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ]
        converted = OpenAICompatibleProvider._convert_messages(messages)
        assert converted[0]["tool_calls"][0]["function"]["arguments"] == '{"x": 1}'
        assert converted[1]["tool_call_id"] == "c1"


class TestAnthropicProvider:
    def test_convert_messages(self):
        system, converted = AnthropicProvider._convert_messages(
            [
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "using tool",
                    "tool_calls": [{"id": "tu_1", "name": "t", "arguments": {"a": 1}}],
                },
                {"role": "tool", "tool_call_id": "tu_1", "content": "42"},
            ]
        )
        assert system == "be helpful"
        assert converted[0] == {"role": "user", "content": "hi"}
        assert converted[1]["content"][0] == {"type": "text", "text": "using tool"}
        assert converted[1]["content"][1]["type"] == "tool_use"
        assert converted[2]["content"][0]["type"] == "tool_result"
        assert converted[2]["content"][0]["tool_use_id"] == "tu_1"

    def test_parse_message_text_and_tool_use(self):
        response = SimpleNamespace(
            id="msg_1",
            model="claude-opus-4-8",
            stop_reason="tool_use",
            content=[
                SimpleNamespace(type="text", text="checking"),
                SimpleNamespace(
                    type="tool_use", id="tu_9", name="search_documents", input={"query": "x"}
                ),
            ],
            usage=SimpleNamespace(
                input_tokens=50,
                output_tokens=10,
                cache_read_input_tokens=None,
                cache_creation_input_tokens=None,
            ),
        )
        completion = AnthropicProvider._parse_message(response)
        assert completion.text == "checking"
        assert completion.tool_calls[0].name == "search_documents"
        assert completion.total_tokens == 60

    def test_parse_message_structured_output_tool(self):
        response = SimpleNamespace(
            id="msg_2",
            model="claude-opus-4-8",
            stop_reason="tool_use",
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="tu_1",
                    name=STRUCTURED_OUTPUT_TOOL,
                    input={"goal_achieved": True, "action": None},
                )
            ],
            usage=SimpleNamespace(
                input_tokens=1,
                output_tokens=1,
                cache_read_input_tokens=None,
                cache_creation_input_tokens=None,
            ),
        )
        completion = AnthropicProvider._parse_message(response)
        assert completion.structured == {"goal_achieved": True, "action": None}
        assert json.loads(completion.text)["goal_achieved"] is True
        assert not completion.tool_calls


class TestBuildProvider:
    def test_anthropic(self):
        provider = build_provider("anthropic", api_key="k")
        assert isinstance(provider, AnthropicProvider)
        assert provider.api_key == "k"

    def test_deepseek_uses_json_object_mode(self):
        provider = build_provider("deepseek", api_key="k")
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.schema_mode == "json_object"
        assert provider.provider_label == "deepseek"

    def test_openai(self):
        provider = build_provider("openai", api_key="k")
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.schema_mode == "json_schema"

    def test_ollama_default(self):
        provider = build_provider("ollama")
        assert isinstance(provider, OllamaProvider)

    def test_custom_endpoint_falls_back_to_openai_compatible(self):
        provider = build_provider("custom", api_url="http://localhost:8081/v1")
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.provider_label == "custom"


class TestAnthropicPromptCaching:
    def test_system_param_with_cache(self):
        param = AnthropicProvider._system_param("stable prefix", True)
        assert param == [
            {
                "type": "text",
                "text": "stable prefix",
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def test_system_param_without_cache(self):
        assert AnthropicProvider._system_param("stable prefix", False) == "stable prefix"

    def test_mark_last_message_string_content(self):
        converted = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "latest"},
        ]
        AnthropicProvider._mark_last_message_cacheable(converted)
        assert converted[0] == {"role": "user", "content": "first"}
        assert converted[1]["content"] == [
            {
                "type": "text",
                "text": "latest",
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def test_mark_last_message_block_content(self):
        converted = [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "42"}
                ],
            }
        ]
        AnthropicProvider._mark_last_message_cacheable(converted)
        block = converted[0]["content"][-1]
        assert block["cache_control"] == {"type": "ephemeral"}
        assert block["tool_use_id"] == "t1"

    def test_mark_last_message_empty_list_noop(self):
        AnthropicProvider._mark_last_message_cacheable([])  # must not raise


class TestQwenKimiProviders:
    def test_qwen_maps_to_dashscope_compatible_mode(self):
        provider = build_provider("qwen", api_key="k")
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.provider_label == "qwen"
        assert provider.schema_mode == "json_object"
        assert "dashscope" in provider.base_url
        assert provider.default_model == "qwen-plus"

    def test_kimi_maps_to_moonshot(self):
        provider = build_provider("kimi", api_key="k")
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.provider_label == "kimi"
        assert provider.schema_mode == "json_object"
        assert "moonshot" in provider.base_url
        assert provider.default_model == "kimi-latest"

    def test_api_url_and_key_overrides_win(self):
        provider = build_provider(
            "qwen", api_url="https://proxy.internal/v1", api_key="override"
        )
        assert provider.base_url == "https://proxy.internal/v1"
        assert provider.api_key == "override"
