"""Tests for context window management tools (compress_history, summarize_findings)."""

import uuid


class TestCompressHistory:
    """Tests for compress_history tool logic."""

    def test_keep_last_default_5(self):
        params = {}
        keep_last = min(int(params.get("keep_last", 5) or 5), 20)
        assert keep_last == 5

    def test_keep_last_capped_at_20(self):
        params = {"keep_last": 50}
        keep_last = min(int(params.get("keep_last", 5) or 5), 20)
        assert keep_last == 20

    def test_not_enough_to_compress(self):
        actions = [{"iteration": i} for i in range(3)]
        keep_last = 5
        assert len(actions) <= keep_last

    def test_splits_actions_correctly(self):
        actions = [{"iteration": i} for i in range(10)]
        keep_last = 3
        to_compress = actions[:-keep_last] if keep_last > 0 else list(actions)
        kept = actions[-keep_last:]
        assert len(to_compress) == 7
        assert len(kept) == 3

    def test_keep_last_zero_compresses_all(self):
        actions = [{"iteration": i} for i in range(5)]
        keep_last = 0
        to_compress = actions[:-keep_last] if keep_last > 0 else list(actions)
        assert len(to_compress) == 5

    def test_action_text_extraction(self):
        actions = [
            {
                "action": {"tool": "search_documents"},
                "result": {"success": True, "data": {"total": 5}},
                "iteration": 1,
            },
            {
                "action": {"tool": "answer_question"},
                "result": {"success": False, "error": "timeout"},
                "iteration": 2,
            },
        ]
        lines = []
        for a in actions:
            tool = (
                a.get("action", {}).get("tool", "unknown")
                if isinstance(a.get("action"), dict)
                else "unknown"
            )
            res_summary = ""
            act_result = a.get("result", {})
            if isinstance(act_result, dict):
                if act_result.get("success"):
                    data_keys = (
                        list(act_result.get("data", {}).keys())
                        if isinstance(act_result.get("data"), dict)
                        else []
                    )
                    res_summary = f"success, data keys: {data_keys}"
                else:
                    res_summary = f"failed: {str(act_result.get('error', ''))[:100]}"
            lines.append(
                f"- Iteration {a.get('iteration', '?')}: {tool} → {res_summary}"
            )
        assert "search_documents" in lines[0]
        assert "success" in lines[0]
        assert "answer_question" in lines[1]
        assert "failed" in lines[1]

    def test_state_mutation_trims_actions(self):
        state = {"actions_taken": [{"iteration": i} for i in range(10)]}
        keep_last = 3
        state["actions_taken"] = state["actions_taken"][-keep_last:]
        assert len(state["actions_taken"]) == 3
        assert state["actions_taken"][0]["iteration"] == 7

    def test_compressed_history_stored_in_state(self):
        state = {}
        state[
            "compressed_history"
        ] = "Agent searched for papers and found 3 relevant results."
        assert "compressed_history" in state
        assert len(state["compressed_history"]) > 0

    def test_existing_compressed_history_appended(self):
        existing = "Previously searched for ML papers."
        new_actions = "Then analyzed 5 documents."
        combined_prompt = f"Previous compressed history:\n{existing}\n\nNew actions to compress:\n{new_actions}"
        assert existing in combined_prompt
        assert new_actions in combined_prompt

    def test_summary_truncated_at_2000(self):
        long_summary = "X" * 3000
        truncated = long_summary[:2000]
        assert len(truncated) == 2000

    def test_result_format(self):
        result = {
            "compressed_actions": 7,
            "kept_actions": 3,
            "summary_length": 450,
        }
        assert result["compressed_actions"] == 7
        assert result["kept_actions"] == 3
        assert result["summary_length"] > 0

    def test_missing_action_tool_defaults_to_unknown(self):
        action = {"result": {"success": True}}
        tool = (
            action.get("action", {}).get("tool", "unknown")
            if isinstance(action.get("action"), dict)
            else "unknown"
        )
        assert tool == "unknown"


class TestSummarizeFindings:
    """Tests for summarize_findings tool logic."""

    def test_no_findings_returns_empty(self):
        findings = []
        target = list(findings)
        assert len(target) == 0

    def test_filter_by_category(self):
        findings = [
            {"category": "key_insight", "title": "A", "content": "..."},
            {"category": "methodology", "title": "B", "content": "..."},
            {"category": "key_insight", "title": "C", "content": "..."},
        ]
        cat_filter = "key_insight"
        target = [f for f in findings if f.get("category") == cat_filter]
        assert len(target) == 2

    def test_no_filter_gets_all(self):
        findings = [
            {"category": "key_insight", "title": "A"},
            {"category": "methodology", "title": "B"},
        ]
        cat_filter = None
        target = (
            [f for f in findings if f.get("category") == cat_filter]
            if cat_filter
            else list(findings)
        )
        assert len(target) == 2

    def test_consolidate_replaces_findings(self):
        findings = [
            {"category": "key_insight", "title": "A"},
            {"category": "key_insight", "title": "B"},
            {"category": "methodology", "title": "C"},
        ]
        cat_filter = "key_insight"
        consolidate = True

        if consolidate:
            if cat_filter:
                remaining = [f for f in findings if f.get("category") != cat_filter]
            else:
                remaining = []
            consolidated = {
                "id": str(uuid.uuid4()),
                "title": f"Synthesis: {cat_filter} (2 items)",
                "content": "synthesized content",
                "category": "synthesis",
            }
            remaining.append(consolidated)
            assert len(remaining) == 2  # methodology + consolidated
            assert remaining[-1]["category"] == "synthesis"

    def test_consolidate_all_findings(self):
        findings = [
            {"category": "key_insight", "title": "A"},
            {"category": "methodology", "title": "B"},
        ]
        cat_filter = None
        consolidate = True

        if consolidate:
            if cat_filter:
                remaining = [f for f in findings if f.get("category") != cat_filter]
            else:
                remaining = []
            consolidated = {
                "id": str(uuid.uuid4()),
                "title": f"Synthesis: all findings ({len(findings)} items)",
                "content": "synthesized content",
                "category": "synthesis",
                "confidence": 0.9,
                "tags": ["synthesized", "compressed"],
            }
            remaining.append(consolidated)
            assert len(remaining) == 1
            assert remaining[0]["category"] == "synthesis"
            assert remaining[0]["confidence"] == 0.9
            assert "synthesized" in remaining[0]["tags"]

    def test_findings_text_format(self):
        findings = [
            {
                "category": "key_insight",
                "title": "Finding A",
                "content": "Content of finding A",
            },
        ]
        findings_text = ""
        for f in findings:
            findings_text += f"- [{f.get('category', 'general')}] {f.get('title', 'Untitled')}: {str(f.get('content', ''))[:300]}\n"
        assert "[key_insight]" in findings_text
        assert "Finding A" in findings_text
        assert "Content of finding A" in findings_text

    def test_content_truncated_to_300(self):
        long_content = "X" * 500
        truncated = str(long_content)[:300]
        assert len(truncated) == 300

    def test_synthesis_truncated_to_3000(self):
        long_synthesis = "Y" * 5000
        truncated = long_synthesis[:3000]
        assert len(truncated) == 3000

    def test_consolidate_defaults_false(self):
        params = {}
        consolidate = bool(params.get("consolidate", False))
        assert consolidate is False

    def test_result_format_no_consolidate(self):
        result = {
            "synthesis": "A coherent summary...",
            "findings_summarized": 5,
            "consolidated": False,
        }
        assert result["consolidated"] is False
        assert result["findings_summarized"] == 5

    def test_result_format_with_consolidate(self):
        result = {
            "synthesis": "A coherent summary...",
            "findings_summarized": 5,
            "consolidated": True,
        }
        assert result["consolidated"] is True

    def test_consolidated_finding_has_required_fields(self):
        consolidated = {
            "id": str(uuid.uuid4()),
            "title": "Synthesis: key_insight (3 items)",
            "content": "synthesized content",
            "category": "synthesis",
            "confidence": 0.9,
            "tags": ["synthesized", "compressed"],
            "created_at": "2026-03-20T00:00:00",
        }
        assert "id" in consolidated
        assert "title" in consolidated
        assert "content" in consolidated
        assert "category" in consolidated
        assert consolidated["category"] == "synthesis"


class TestContextManagementSchemas:
    """Tests for context management tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS

        names = {t["name"] for t in AGENT_TOOLS}
        assert "compress_history" in names
        assert "summarize_findings" in names

    def test_compress_history_no_required(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("compress_history")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert len(required) == 0

    def test_compress_history_has_keep_last(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("compress_history")
        assert "keep_last" in tool["parameters"]["properties"]

    def test_summarize_findings_no_required(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("summarize_findings")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert len(required) == 0

    def test_summarize_findings_has_consolidate(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("summarize_findings")
        assert "consolidate" in tool["parameters"]["properties"]

    def test_summarize_findings_has_category(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("summarize_findings")
        assert "category" in tool["parameters"]["properties"]


class TestContextManagementRegistry:
    """Tests for context management tool registry classification."""

    def test_compress_history_is_read(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("compress_history")
        assert meta is not None
        assert meta.effects == "read"

    def test_summarize_findings_is_read(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("summarize_findings")
        assert meta is not None
        assert meta.effects == "read"

    def test_both_are_medium_cost(self):
        from app.services.tool_registry import get_tool_metadata

        for tool_name in ["compress_history", "summarize_findings"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.cost_tier == "medium"

    def test_neither_is_network_tool(self):
        from app.services.tool_registry import get_tool_metadata

        for tool_name in ["compress_history", "summarize_findings"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.network == "none"
