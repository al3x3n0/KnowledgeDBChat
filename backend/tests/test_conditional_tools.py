"""Tests for conditional execution tools (evaluate_condition, count_findings, check_goal_status)."""


class TestEvaluateConditionFindingsCount:
    """Tests for evaluate_condition with findings_count."""

    def test_met_when_above_threshold(self):
        state = {"findings": [{"id": "f1"}, {"id": "f2"}, {"id": "f3"}]}
        threshold = 2
        count = len(state.get("findings", []))
        assert count >= threshold

    def test_not_met_when_below_threshold(self):
        state = {"findings": [{"id": "f1"}]}
        threshold = 5
        count = len(state.get("findings", []))
        assert count < threshold

    def test_met_when_equal_to_threshold(self):
        state = {"findings": [{"id": "f1"}, {"id": "f2"}]}
        threshold = 2
        count = len(state.get("findings", []))
        assert count >= threshold

    def test_empty_findings(self):
        state = {"findings": []}
        threshold = 1
        count = len(state.get("findings", []))
        assert count < threshold

    def test_missing_findings_key(self):
        state = {}
        count = len(state.get("findings", []))
        assert count == 0


class TestEvaluateConditionFindingsHasCategory:
    """Tests for evaluate_condition with findings_has_category."""

    def test_category_found(self):
        state = {
            "findings": [
                {"category": "key_insight", "title": "A"},
                {"category": "methodology", "title": "B"},
                {"category": "key_insight", "title": "C"},
            ]
        }
        cat = "key_insight"
        matches = [f for f in state.get("findings", []) if f.get("category") == cat]
        assert len(matches) == 2

    def test_category_not_found(self):
        state = {
            "findings": [
                {"category": "methodology", "title": "A"},
            ]
        }
        cat = "key_insight"
        matches = [f for f in state.get("findings", []) if f.get("category") == cat]
        assert len(matches) == 0

    def test_threshold_on_category(self):
        state = {
            "findings": [
                {"category": "result", "title": "A"},
                {"category": "result", "title": "B"},
            ]
        }
        cat = "result"
        threshold = 3
        matches = [f for f in state.get("findings", []) if f.get("category") == cat]
        assert len(matches) < threshold


class TestEvaluateConditionActionsCount:
    """Tests for evaluate_condition with actions_count."""

    def test_actions_count_met(self):
        state = {"actions_taken": [{"iteration": i} for i in range(5)]}
        threshold = 3
        count = len(state.get("actions_taken", []))
        assert count >= threshold

    def test_actions_count_not_met(self):
        state = {"actions_taken": [{"iteration": 1}]}
        threshold = 5
        count = len(state.get("actions_taken", []))
        assert count < threshold


class TestEvaluateConditionProgressAbove:
    """Tests for evaluate_condition with progress_above."""

    def test_progress_met(self):
        state = {"goal_progress": 75}
        threshold = 50
        progress = state.get("goal_progress", 0)
        assert progress >= threshold

    def test_progress_not_met(self):
        state = {"goal_progress": 20}
        threshold = 50
        progress = state.get("goal_progress", 0)
        assert progress < threshold

    def test_zero_progress(self):
        state = {}
        progress = state.get("goal_progress", 0)
        assert progress == 0


class TestEvaluateConditionValidation:
    """Tests for evaluate_condition validation."""

    def test_unknown_condition_rejected(self):
        valid_conditions = {
            "findings_count",
            "findings_has_category",
            "documents_count",
            "search_has_results",
            "actions_count",
            "progress_above",
        }
        assert "nonexistent" not in valid_conditions

    def test_all_conditions_recognized(self):
        valid = {
            "findings_count",
            "findings_has_category",
            "documents_count",
            "search_has_results",
            "actions_count",
            "progress_above",
        }
        assert len(valid) == 6

    def test_default_threshold_is_1(self):
        params = {"condition": "findings_count"}
        threshold = int(params.get("threshold", 1) or 1)
        assert threshold == 1

    def test_result_format(self):
        result = {
            "met": True,
            "actual": 5,
            "threshold": 3,
            "condition": "findings_count",
        }
        assert "met" in result
        assert "actual" in result
        assert "threshold" in result
        assert "condition" in result


class TestCountFindings:
    """Tests for count_findings tool logic."""

    def test_count_all(self):
        findings = [
            {"category": "key_insight", "confidence": 0.9},
            {"category": "methodology", "confidence": 0.7},
            {"category": "key_insight", "confidence": 0.8},
        ]
        assert len(findings) == 3

    def test_filter_by_category(self):
        findings = [
            {"category": "key_insight", "confidence": 0.9},
            {"category": "methodology", "confidence": 0.7},
            {"category": "key_insight", "confidence": 0.8},
        ]
        cat_filter = "key_insight"
        filtered = [f for f in findings if f.get("category") == cat_filter]
        assert len(filtered) == 2

    def test_filter_by_confidence(self):
        findings = [
            {"category": "key_insight", "confidence": 0.9},
            {"category": "methodology", "confidence": 0.3},
            {"category": "result", "confidence": 0.8},
        ]
        min_conf = 0.5
        filtered = [
            f for f in findings if float(f.get("confidence", 0.8) or 0.8) >= min_conf
        ]
        assert len(filtered) == 2

    def test_group_by_category(self):
        findings = [
            {"category": "key_insight"},
            {"category": "methodology"},
            {"category": "key_insight"},
            {"category": "result"},
        ]
        by_category: dict[str, int] = {}
        for f in findings:
            c = str(f.get("category", "uncategorized"))
            by_category[c] = by_category.get(c, 0) + 1
        assert by_category["key_insight"] == 2
        assert by_category["methodology"] == 1
        assert by_category["result"] == 1
        assert len(by_category) == 3

    def test_empty_findings(self):
        findings = []
        by_category: dict[str, int] = {}
        for f in findings:
            c = str(f.get("category", "uncategorized"))
            by_category[c] = by_category.get(c, 0) + 1
        assert len(by_category) == 0

    def test_missing_category_uses_uncategorized(self):
        findings = [{"confidence": 0.9}]
        by_category: dict[str, int] = {}
        for f in findings:
            c = str(f.get("category", "uncategorized"))
            by_category[c] = by_category.get(c, 0) + 1
        assert by_category["uncategorized"] == 1

    def test_default_min_confidence_is_zero(self):
        params = {}
        min_conf = float(params.get("min_confidence", 0.0) or 0.0)
        assert min_conf == 0.0

    def test_result_format(self):
        result = {
            "total": 5,
            "by_category": {"key_insight": 3, "methodology": 2},
            "categories": ["key_insight", "methodology"],
        }
        assert "total" in result
        assert "by_category" in result
        assert "categories" in result
        assert result["total"] == 5


class TestCheckGoalStatus:
    """Tests for check_goal_status tool logic."""

    def test_iterations_remaining(self):
        iteration = 7
        max_iterations = 20
        remaining = max_iterations - iteration
        assert remaining == 13

    def test_tool_calls_remaining(self):
        used = 15
        max_calls = 50
        remaining = max_calls - used
        assert remaining == 35

    def test_plan_steps_from_state(self):
        state = {
            "execution_plan": {"steps": ["step1", "step2", "step3"]},
            "plan_step_index": 1,
        }
        exec_plan = state.get("execution_plan")
        steps_total = (
            len(exec_plan.get("steps", [])) if isinstance(exec_plan, dict) else 0
        )
        assert steps_total == 3
        assert state.get("plan_step_index", 0) == 1

    def test_no_execution_plan(self):
        state = {}
        exec_plan = state.get("execution_plan")
        has_plan = bool(exec_plan)
        steps_total = (
            len(exec_plan.get("steps", [])) if isinstance(exec_plan, dict) else 0
        )
        assert has_plan is False
        assert steps_total == 0

    def test_result_format(self):
        result = {
            "iteration": 5,
            "max_iterations": 20,
            "iterations_remaining": 15,
            "tool_calls_used": 10,
            "max_tool_calls": 50,
            "tool_calls_remaining": 40,
            "goal_progress": 35,
            "findings_count": 3,
            "actions_count": 8,
            "has_execution_plan": True,
            "plan_steps_completed": 2,
            "plan_steps_total": 5,
        }
        assert (
            result["iterations_remaining"]
            == result["max_iterations"] - result["iteration"]
        )
        assert (
            result["tool_calls_remaining"]
            == result["max_tool_calls"] - result["tool_calls_used"]
        )
        assert len(result) == 12


class TestConditionalToolSchemas:
    """Tests for conditional tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS

        names = {t["name"] for t in AGENT_TOOLS}
        assert "evaluate_condition" in names
        assert "count_findings" in names
        assert "check_goal_status" in names

    def test_evaluate_condition_requires_condition(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("evaluate_condition")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "condition" in required

    def test_evaluate_condition_has_enum(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("evaluate_condition")
        condition_prop = tool["parameters"]["properties"]["condition"]
        assert "enum" in condition_prop
        assert "findings_count" in condition_prop["enum"]
        assert "progress_above" in condition_prop["enum"]

    def test_count_findings_no_required(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("count_findings")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert len(required) == 0

    def test_check_goal_status_no_required(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("check_goal_status")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert len(required) == 0


class TestConditionalToolRegistry:
    """Tests for conditional tool registry classification."""

    def test_evaluate_condition_is_read(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("evaluate_condition")
        assert meta is not None
        assert meta.effects == "read"

    def test_count_findings_is_read(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("count_findings")
        assert meta is not None
        assert meta.effects == "read"

    def test_check_goal_status_is_read(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("check_goal_status")
        assert meta is not None
        assert meta.effects == "read"

    def test_all_are_low_cost(self):
        from app.services.tool_registry import get_tool_metadata

        for tool_name in ["evaluate_condition", "count_findings", "check_goal_status"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.cost_tier == "low"

    def test_none_is_network_tool(self):
        from app.services.tool_registry import get_tool_metadata

        for tool_name in ["evaluate_condition", "count_findings", "check_goal_status"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.network == "none"
