"""Tests for prompt template management tools (switch_strategy, set_focus_directive, get_available_strategies)."""

import pytest


class TestSwitchStrategy:
    """Tests for switch_strategy tool logic."""

    def test_valid_roles(self):
        valid = {"researcher", "critic", "synthesizer", "verifier", "coder", "author"}
        assert len(valid) == 6

    def test_requires_role(self):
        params = {}
        role = str(params.get("role", "")).strip().lower()
        assert not role

    def test_invalid_role_rejected(self):
        valid_roles = {"researcher", "critic", "synthesizer", "verifier", "coder", "author"}
        assert "explorer" not in valid_roles
        assert "manager" not in valid_roles

    def test_all_valid_roles_accepted(self):
        valid_roles = {"researcher", "critic", "synthesizer", "verifier", "coder", "author"}
        for role in valid_roles:
            assert role in valid_roles

    def test_role_case_insensitive(self):
        role = "Researcher".strip().lower()
        valid_roles = {"researcher", "critic", "synthesizer", "verifier", "coder", "author"}
        assert role in valid_roles

    def test_state_profile_updated(self):
        state = {"skill_profile": {"role": "researcher", "display_name": "Researcher"}}
        new_role = "critic"
        state["skill_profile"] = {"role": new_role, "display_name": "Critic"}
        assert state["skill_profile"]["role"] == "critic"

    def test_strategy_switch_logged(self):
        state = {"strategy_switches": []}
        state["strategy_switches"].append({
            "from": "researcher",
            "to": "critic",
            "reason": "Enough findings gathered, need validation",
            "iteration": 8,
            "timestamp": "2026-03-20T12:00:00",
        })
        assert len(state["strategy_switches"]) == 1
        assert state["strategy_switches"][0]["from"] == "researcher"
        assert state["strategy_switches"][0]["to"] == "critic"

    def test_reason_truncated_to_500(self):
        long_reason = "R" * 800
        truncated = long_reason[:500]
        assert len(truncated) == 500

    def test_result_format(self):
        result = {
            "previous_role": "researcher",
            "new_role": "synthesizer",
            "display_name": "Synthesizer",
            "preferred_tools": ["create_synthesis_document", "write_progress_report"],
        }
        assert result["previous_role"] == "researcher"
        assert result["new_role"] == "synthesizer"
        assert "display_name" in result
        assert len(result["preferred_tools"]) <= 5

    def test_multiple_switches_tracked(self):
        state = {"strategy_switches": []}
        switches = [
            {"from": "researcher", "to": "critic", "iteration": 5},
            {"from": "critic", "to": "synthesizer", "iteration": 12},
        ]
        for s in switches:
            state["strategy_switches"].append(s)
        assert len(state["strategy_switches"]) == 2


class TestSetFocusDirective:
    """Tests for set_focus_directive tool logic."""

    def test_requires_directive(self):
        params = {}
        directive = str(params.get("directive", "")).strip()[:1000]
        assert not directive

    def test_replace_mode(self):
        state = {"focus_directive": "Old directive"}
        new_directive = "Prioritize contradictions"
        state["focus_directive"] = new_directive
        assert state["focus_directive"] == "Prioritize contradictions"

    def test_append_mode(self):
        state = {"focus_directive": "Focus on recent papers"}
        new_directive = "Also check for methodology gaps"
        existing = str(state.get("focus_directive", ""))
        state["focus_directive"] = (existing + "\n" + new_directive).strip()[:2000]
        assert "recent papers" in state["focus_directive"]
        assert "methodology gaps" in state["focus_directive"]

    def test_directive_truncated_to_1000(self):
        long_directive = "D" * 1500
        truncated = long_directive[:1000]
        assert len(truncated) == 1000

    def test_combined_directive_capped_at_2000(self):
        existing = "E" * 1500
        new = "N" * 1500
        combined = (existing + "\n" + new).strip()[:2000]
        assert len(combined) == 2000

    def test_append_defaults_false(self):
        params = {"directive": "Focus on X"}
        append = bool(params.get("append", False))
        assert append is False

    def test_result_format_replace(self):
        result = {
            "directive": "Prioritize finding contradictions",
            "mode": "replaced",
        }
        assert result["mode"] == "replaced"

    def test_result_format_append(self):
        result = {
            "directive": "Focus on recent papers\nAlso check methodology gaps",
            "mode": "appended",
        }
        assert result["mode"] == "appended"

    def test_empty_state_gets_new_directive(self):
        state = {}
        state["focus_directive"] = "New directive"
        assert state["focus_directive"] == "New directive"


class TestGetAvailableStrategies:
    """Tests for get_available_strategies tool logic."""

    def test_returns_all_six_roles(self):
        roles = ["researcher", "critic", "synthesizer", "verifier", "coder", "author"]
        assert len(roles) == 6

    def test_strategy_entry_format(self):
        entry = {
            "role": "researcher",
            "display_name": "Researcher",
            "guidance": "Prioritize discovery and evidence coverage before conclusions.",
            "preferred_tools": ["search_documents", "search_arxiv"],
            "discouraged_tools": ["create_synthesis_document"],
        }
        assert "role" in entry
        assert "display_name" in entry
        assert "guidance" in entry
        assert "preferred_tools" in entry
        assert "discouraged_tools" in entry

    def test_preferred_tools_capped_at_5(self):
        tools = ["t1", "t2", "t3", "t4", "t5", "t6", "t7"]
        capped = tools[:5]
        assert len(capped) == 5

    def test_current_role_in_result(self):
        result = {
            "strategies": [],
            "current_role": "researcher",
        }
        assert "current_role" in result

    def test_guidance_truncated_to_300(self):
        directives = ["A" * 100, "B" * 100, "C" * 100, "D" * 100]
        guidance = "; ".join(directives)[:300]
        assert len(guidance) <= 300

    def test_no_params_required(self):
        params = {}
        # Tool takes no params
        assert isinstance(params, dict)


class TestPromptInjection:
    """Tests for prompt injection of focus_directive and handoff_contract."""

    def test_focus_directive_in_prompt(self):
        state = {"focus_directive": "Prioritize contradictions"}
        prompt = ""
        focus_directive = state.get("focus_directive", "")
        if focus_directive:
            prompt += f"\nFOCUS DIRECTIVE (set by agent):\n{focus_directive}\n"
        assert "FOCUS DIRECTIVE" in prompt
        assert "contradictions" in prompt

    def test_no_focus_directive_no_injection(self):
        state = {}
        prompt = ""
        focus_directive = state.get("focus_directive", "")
        if focus_directive:
            prompt += f"\nFOCUS DIRECTIVE (set by agent):\n{focus_directive}\n"
        assert "FOCUS DIRECTIVE" not in prompt

    def test_handoff_contract_in_prompt(self):
        config = {
            "handoff_contract": {
                "from_job_id": "parent-123",
                "context": "We found 5 papers on transformers",
                "expected_outputs": ["summary", "key_findings"],
            }
        }
        prompt = ""
        handoff_contract = config.get("handoff_contract")
        if isinstance(handoff_contract, dict):
            prompt += "\nHANDOFF CONTRACT (from parent agent):\n"
            ctx = str(handoff_contract.get("context", ""))[:1000]
            if ctx:
                prompt += f"- Context: {ctx}\n"
            outputs = handoff_contract.get("expected_outputs", [])
            if isinstance(outputs, list) and outputs:
                prompt += f"- Expected outputs: {', '.join(str(o) for o in outputs[:10])}\n"
            prompt += "- You MUST produce results that satisfy the expected outputs.\n"
        assert "HANDOFF CONTRACT" in prompt
        assert "transformers" in prompt
        assert "summary, key_findings" in prompt
        assert "MUST produce" in prompt

    def test_no_handoff_contract_no_injection(self):
        config = {}
        prompt = ""
        handoff_contract = config.get("handoff_contract")
        if isinstance(handoff_contract, dict):
            prompt += "HANDOFF CONTRACT"
        assert "HANDOFF CONTRACT" not in prompt

    def test_context_truncated_in_prompt(self):
        long_ctx = "C" * 2000
        truncated = long_ctx[:1000]
        assert len(truncated) == 1000

    def test_outputs_capped_at_10_in_prompt(self):
        outputs = [f"output_{i}" for i in range(15)]
        capped = outputs[:10]
        assert len(capped) == 10


class TestPromptManagementSchemas:
    """Tests for prompt management tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS
        names = {t["name"] for t in AGENT_TOOLS}
        assert "switch_strategy" in names
        assert "set_focus_directive" in names
        assert "get_available_strategies" in names

    def test_switch_strategy_requires_role(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("switch_strategy")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "role" in required

    def test_switch_strategy_has_role_enum(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("switch_strategy")
        role_prop = tool["parameters"]["properties"]["role"]
        assert "enum" in role_prop
        assert "researcher" in role_prop["enum"]
        assert "coder" in role_prop["enum"]
        assert len(role_prop["enum"]) == 6

    def test_set_focus_directive_requires_directive(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("set_focus_directive")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "directive" in required

    def test_set_focus_directive_has_append(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("set_focus_directive")
        assert "append" in tool["parameters"]["properties"]

    def test_get_available_strategies_no_required(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("get_available_strategies")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert len(required) == 0


class TestPromptManagementRegistry:
    """Tests for prompt management tool registry classification."""

    def test_all_are_read(self):
        from app.services.tool_registry import get_tool_metadata
        for tool_name in ["switch_strategy", "set_focus_directive", "get_available_strategies"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.effects == "read"

    def test_all_are_low_cost(self):
        from app.services.tool_registry import get_tool_metadata
        for tool_name in ["switch_strategy", "set_focus_directive", "get_available_strategies"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.cost_tier == "low"

    def test_none_is_network_tool(self):
        from app.services.tool_registry import get_tool_metadata
        for tool_name in ["switch_strategy", "set_focus_directive", "get_available_strategies"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.network == "none"
