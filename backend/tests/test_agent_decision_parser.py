"""Tests for agent_decision_parser module."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.agent_decision_parser import (
    AgentActionDecision,
    AgentDecision,
    AgentDecisionParser,
    extract_first_json_object,
    DECISION_JSON_SCHEMA,
)


# ---------------------------------------------------------------------------
# extract_first_json_object
# ---------------------------------------------------------------------------

class TestExtractFirstJsonObject:
    def test_direct_json(self):
        text = '{"goal_achieved": true, "reasoning": "done"}'
        result = extract_first_json_object(text)
        assert result == {"goal_achieved": True, "reasoning": "done"}

    def test_markdown_fence(self):
        text = 'Here is the response:\n```json\n{"goal_achieved": false}\n```\nDone.'
        result = extract_first_json_object(text)
        assert result == {"goal_achieved": False}

    def test_markdown_fence_no_lang(self):
        text = '```\n{"action": null}\n```'
        result = extract_first_json_object(text)
        assert result == {"action": None}

    def test_embedded_json_with_commentary(self):
        text = 'I think the answer is: {"goal_achieved": false, "action": {"tool": "search_documents"}} and that is it.'
        result = extract_first_json_object(text)
        assert result is not None
        assert result["goal_achieved"] is False

    def test_empty_string(self):
        assert extract_first_json_object("") is None

    def test_no_json(self):
        assert extract_first_json_object("There is no JSON here at all.") is None

    def test_array_not_object(self):
        assert extract_first_json_object('[1, 2, 3]') is None

    def test_nested_braces(self):
        text = '{"action": {"tool": "search", "params": {"query": "test {with} braces"}}}'
        result = extract_first_json_object(text)
        assert result is not None
        assert result["action"]["tool"] == "search"


# ---------------------------------------------------------------------------
# AgentDecision model
# ---------------------------------------------------------------------------

class TestAgentDecisionModel:
    def test_valid_decision(self):
        d = AgentDecision.model_validate({
            "goal_achieved": False,
            "should_stop": False,
            "reasoning": "Continuing research",
            "action": {"tool": "search_documents", "params": {"query": "test"}, "purpose": "find docs"},
        })
        assert d.goal_achieved is False
        assert d.action is not None
        assert d.action.tool == "search_documents"

    def test_string_bool_coercion(self):
        d = AgentDecision.model_validate({
            "goal_achieved": "true",
            "should_stop": "no",
        })
        assert d.goal_achieved is True
        assert d.should_stop is False

    def test_numeric_bool_coercion(self):
        d = AgentDecision.model_validate({
            "goal_achieved": 1,
            "should_stop": 0,
        })
        assert d.goal_achieved is True
        assert d.should_stop is False

    def test_action_from_string(self):
        d = AgentDecision.model_validate({
            "action": "search_documents",
        })
        assert d.action is not None
        assert d.action.tool == "search_documents"
        assert d.action.params == {}

    def test_null_action(self):
        d = AgentDecision.model_validate({
            "goal_achieved": True,
            "action": None,
        })
        assert d.action is None

    def test_reasoning_truncation(self):
        long_reason = "x" * 1000
        d = AgentDecision.model_validate({"reasoning": long_reason})
        assert len(d.reasoning) == 800

    def test_defaults(self):
        d = AgentDecision.model_validate({})
        assert d.goal_achieved is False
        assert d.should_stop is False
        assert d.action is None

    def test_action_params_coerced(self):
        d = AgentDecision.model_validate({
            "action": {"tool": "test", "params": "not_a_dict"},
        })
        assert d.action.params == {}


# ---------------------------------------------------------------------------
# AgentDecisionParser
# ---------------------------------------------------------------------------

class TestAgentDecisionParser:
    def setup_method(self):
        self.llm = MagicMock()
        self.parser = AgentDecisionParser(self.llm)
        self.tools = ["search_documents", "read_document_content", "summarize_document"]

    def test_parse_valid_json(self):
        text = json.dumps({
            "goal_achieved": False,
            "reasoning": "searching",
            "action": {"tool": "search_documents", "params": {"query": "test"}},
        })
        decision, err = self.parser.parse(text, self.tools)
        assert decision is not None
        assert err == ""
        assert decision.action.tool == "search_documents"

    def test_parse_validates_tool(self):
        text = json.dumps({
            "action": {"tool": "nonexistent_tool", "params": {}},
        })
        decision, err = self.parser.parse(text, self.tools)
        assert decision is not None
        assert decision.action is None  # cleared

    def test_parse_no_json(self):
        decision, err = self.parser.parse("No JSON here", self.tools)
        assert decision is None
        assert "No valid JSON" in err

    def test_parse_metrics_incremented(self):
        text = json.dumps({"goal_achieved": True})
        self.parser.parse(text, self.tools)
        assert self.parser.metrics["parse_success"] == 1

    def test_schema_for_prompt(self):
        schema = AgentDecisionParser.get_schema_for_prompt()
        assert "goal_achieved" in schema
        assert "action" in schema
        assert len(schema) > 50

    def test_validate_action_tool_clears_unavailable(self):
        decision = AgentDecision(
            action=AgentActionDecision(tool="bad_tool", params={})
        )
        result = AgentDecisionParser.validate_action_tool(decision, self.tools)
        assert result.action is None

    def test_validate_action_tool_keeps_available(self):
        decision = AgentDecision(
            action=AgentActionDecision(tool="search_documents", params={"query": "test"})
        )
        result = AgentDecisionParser.validate_action_tool(decision, self.tools)
        assert result.action is not None
        assert result.action.tool == "search_documents"


# ---------------------------------------------------------------------------
# Async tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestAgentDecisionParserAsync:
    async def test_parse_with_retry_success_first_attempt(self):
        llm = AsyncMock()
        parser = AgentDecisionParser(llm)
        tools = ["search_documents"]

        text = json.dumps({"goal_achieved": False, "action": {"tool": "search_documents", "params": {"query": "x"}}})
        result = await parser.parse_with_retry(
            raw_response=text,
            available_tools=tools,
            job=MagicMock(),
            state={},
            user_settings=None,
            system_prompt="test",
            user_message="test",
        )
        assert result is not None
        assert result.action.tool == "search_documents"
        assert llm.generate_response.call_count == 0  # no retries needed

    async def test_parse_with_retry_retries_on_failure(self):
        llm = AsyncMock()
        # First retry returns valid JSON
        llm.generate_response.return_value = json.dumps({"goal_achieved": True})
        parser = AgentDecisionParser(llm)
        tools = ["search_documents"]

        result = await parser.parse_with_retry(
            raw_response="not json at all",
            available_tools=tools,
            job=MagicMock(),
            state={},
            user_settings=None,
            system_prompt="test",
            user_message="test",
        )
        assert result is not None
        assert result.goal_achieved is True
        assert llm.generate_response.call_count == 1  # one retry

    async def test_parse_with_retry_falls_back_to_none(self):
        llm = AsyncMock()
        llm.generate_response.return_value = "still not json"
        parser = AgentDecisionParser(llm)
        tools = ["search_documents"]

        result = await parser.parse_with_retry(
            raw_response="not json",
            available_tools=tools,
            job=MagicMock(),
            state={},
            user_settings=None,
            system_prompt="test",
            user_message="test",
        )
        assert result is None
        assert parser.metrics["parse_failure"] == 1

    async def test_repair_json_calls_llm(self):
        llm = AsyncMock()
        llm.generate_response.return_value = '{"goal_achieved": true}'
        parser = AgentDecisionParser(llm)

        result = await parser.repair_json(
            '{"goal_achieved": true,}',  # trailing comma
            "Expecting value",
            user_settings=None,
        )
        assert result is not None
        assert llm.generate_response.call_count == 1
        assert parser.metrics["parse_repair"] == 1
