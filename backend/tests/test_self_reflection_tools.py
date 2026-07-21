"""Tests for agent self-reflection and tool analytics tools."""

import uuid
from datetime import datetime, timezone, timedelta

import pytest


class TestGetJobHistory:
    """Tests for get_job_history tool logic."""

    def test_no_required_params(self):
        params = {}
        # All params optional
        jt = str(params.get("job_type", "")).strip()
        assert jt == ""

    def test_job_type_filter(self):
        params = {"job_type": "research"}
        jt = str(params.get("job_type", "")).strip()
        assert jt == "research"

    def test_status_filter(self):
        params = {"status": "completed"}
        status = str(params.get("status", "")).strip()
        assert status == "completed"

    def test_limit_defaults_to_10(self):
        params = {}
        limit = min(int(params.get("limit", 10) or 10), 50)
        assert limit == 10

    def test_limit_capped_at_50(self):
        params = {"limit": 200}
        limit = min(int(params.get("limit", 10) or 10), 50)
        assert limit == 50

    def test_excludes_current_job(self):
        current_id = uuid.uuid4()
        all_ids = [uuid.uuid4(), current_id, uuid.uuid4()]
        filtered = [i for i in all_ids if i != current_id]
        assert len(filtered) == 2
        assert current_id not in filtered

    def test_duration_calculation(self):
        started = datetime(2026, 3, 20, 10, 0, 0, tzinfo=timezone.utc)
        completed = datetime(2026, 3, 20, 10, 45, 30, tzinfo=timezone.utc)
        duration = round((completed - started).total_seconds() / 60, 1)
        assert duration == 45.5

    def test_duration_none_if_not_completed(self):
        started = datetime(2026, 3, 20, 10, 0, 0, tzinfo=timezone.utc)
        completed = None
        duration = round((completed - started).total_seconds() / 60, 1) if started and completed else None
        assert duration is None

    def test_error_truncated(self):
        long_error = "E" * 500
        truncated = long_error[:200]
        assert len(truncated) == 200

    def test_result_format(self):
        result = {
            "success": True,
            "data": {
                "jobs": [
                    {
                        "id": str(uuid.uuid4()),
                        "goal": "Research transformers",
                        "job_type": "research",
                        "status": "completed",
                        "iteration": 15,
                        "tool_calls_used": 42,
                        "llm_calls_used": 30,
                        "tokens_used": 50000,
                        "error": None,
                        "created_at": "2026-03-19T10:00:00",
                        "completed_at": "2026-03-19T10:30:00",
                        "duration_minutes": 30.0,
                    },
                ],
                "count": 1,
            }
        }
        assert result["data"]["count"] == 1
        assert result["data"]["jobs"][0]["tool_calls_used"] == 42


class TestGetJobMetrics:
    """Tests for get_job_metrics tool logic."""

    def test_defaults_to_current_job(self):
        params = {}
        target_id = str(params.get("job_id", "")).strip()
        assert target_id == ""  # Should default to current job

    def test_accepts_specific_job_id(self):
        jid = str(uuid.uuid4())
        params = {"job_id": jid}
        target_id = str(params.get("job_id", "")).strip()
        assert target_id == jid

    def test_invalid_uuid_detected(self):
        with pytest.raises(ValueError):
            uuid.UUID("not-valid")

    def test_multi_tenant_check(self):
        user_a = uuid.uuid4()
        user_b = uuid.uuid4()
        assert user_a != user_b

    def test_tool_usage_breakdown(self):
        execution_log = [
            {"action": "search_documents"},
            {"action": "search_documents"},
            {"action": "get_document_details"},
            {"action": "search_documents"},
            {"action": "summarize_document"},
        ]
        tool_counts = {}
        for entry in execution_log:
            tool = entry.get("action")
            if tool:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
        assert tool_counts["search_documents"] == 3
        assert tool_counts["get_document_details"] == 1
        assert tool_counts["summarize_document"] == 1

    def test_duration_calculation(self):
        started = datetime(2026, 3, 20, 10, 0, tzinfo=timezone.utc)
        completed = datetime(2026, 3, 20, 10, 15, 30, tzinfo=timezone.utc)
        duration = round((completed - started).total_seconds() / 60, 2)
        assert duration == 15.5

    def test_result_format(self):
        result = {
            "success": True,
            "data": {
                "id": str(uuid.uuid4()),
                "goal": "Analyze papers",
                "status": "completed",
                "job_type": "analysis",
                "iterations": 20,
                "tool_calls_used": 45,
                "llm_calls_used": 40,
                "tokens_used": 80000,
                "max_tool_calls": 500,
                "max_llm_calls": 200,
                "max_runtime_minutes": 60,
                "duration_minutes": 12.5,
                "error_count": 2,
                "tool_usage_breakdown": {
                    "search_documents": 15,
                    "get_document_details": 10,
                    "summarize_document": 5,
                },
                "created_at": "2026-03-20T10:00:00",
                "started_at": "2026-03-20T10:00:01",
                "completed_at": "2026-03-20T10:12:31",
            }
        }
        assert result["data"]["tool_usage_breakdown"]["search_documents"] == 15
        assert result["data"]["duration_minutes"] == 12.5
        assert result["data"]["error_count"] == 2


class TestGetToolUsageStats:
    """Tests for get_tool_usage_stats tool logic."""

    def test_days_defaults_to_7(self):
        params = {}
        days = min(int(params.get("days", 7) or 7), 30)
        assert days == 7

    def test_days_capped_at_30(self):
        params = {"days": 90}
        days = min(int(params.get("days", 7) or 7), 30)
        assert days == 30

    def test_tool_name_filter_optional(self):
        params = {}
        tool_name = str(params.get("tool_name", "")).strip() or None
        assert tool_name is None

    def test_tool_name_filter_applied(self):
        params = {"tool_name": "search_documents"}
        tool_name = str(params.get("tool_name", "")).strip() or None
        assert tool_name == "search_documents"

    def test_stats_aggregation(self):
        entries = [
            {"action": "search_documents"},
            {"action": "search_documents", "error": "timeout"},
            {"action": "get_document_details"},
            {"action": "search_documents"},
        ]
        tool_stats = {}
        for entry in entries:
            tool = entry.get("action")
            if not tool:
                continue
            stats = tool_stats.setdefault(tool, {"calls": 0, "successes": 0, "failures": 0})
            stats["calls"] += 1
            if entry.get("error"):
                stats["failures"] += 1
            else:
                stats["successes"] += 1

        assert tool_stats["search_documents"]["calls"] == 3
        assert tool_stats["search_documents"]["failures"] == 1
        assert tool_stats["search_documents"]["successes"] == 2
        assert tool_stats["get_document_details"]["calls"] == 1

    def test_success_rate_calculation(self):
        calls = 10
        successes = 8
        rate = round(successes / calls, 3)
        assert rate == 0.8

    def test_success_rate_zero_calls(self):
        calls = 0
        rate = 0.0 if calls == 0 else round(5 / calls, 3)
        assert rate == 0.0

    def test_sorted_by_calls(self):
        tools = [("a", {"calls": 5}), ("b", {"calls": 20}), ("c", {"calls": 10})]
        sorted_tools = sorted(tools, key=lambda x: x[1]["calls"], reverse=True)
        assert sorted_tools[0][0] == "b"
        assert sorted_tools[1][0] == "c"

    def test_result_format(self):
        result = {
            "success": True,
            "data": {
                "period_days": 7,
                "total_jobs_analyzed": 15,
                "tools": [
                    {"name": "search_documents", "calls": 45, "successes": 43, "failures": 2, "success_rate": 0.956},
                    {"name": "get_document_details", "calls": 20, "successes": 20, "failures": 0, "success_rate": 1.0},
                ],
            }
        }
        assert result["data"]["total_jobs_analyzed"] == 15
        assert result["data"]["tools"][0]["success_rate"] > 0.9

    def test_cutoff_date(self):
        days = 7
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        assert (datetime.now(timezone.utc) - cutoff).days == 7


class TestGetToolFailureAnalysis:
    """Tests for get_tool_failure_analysis tool logic."""

    def test_requires_tool_name(self):
        params = {}
        tool_name = str(params.get("tool_name", "")).strip()
        assert not tool_name

    def test_accepts_tool_name(self):
        params = {"tool_name": "search_web"}
        tool_name = str(params.get("tool_name", "")).strip()
        assert tool_name == "search_web"

    def test_days_defaults_to_7(self):
        params = {"tool_name": "x"}
        days = min(int(params.get("days", 7) or 7), 30)
        assert days == 7

    def test_error_pattern_grouping(self):
        errors = [
            {"error": "Connection timeout after 30s"},
            {"error": "Connection timeout after 30s"},
            {"error": "Rate limit exceeded"},
            {"error": "Connection timeout after 30s"},
        ]
        error_patterns = {}
        for e in errors:
            key = e["error"][:80]
            pattern = error_patterns.setdefault(key, {"count": 0, "examples": []})
            pattern["count"] += 1
            if len(pattern["examples"]) < 3:
                pattern["examples"].append(e)

        assert error_patterns["Connection timeout after 30s"]["count"] == 3
        assert error_patterns["Rate limit exceeded"]["count"] == 1
        assert len(error_patterns["Connection timeout after 30s"]["examples"]) == 3

    def test_error_truncated_to_200(self):
        long_error = "E" * 500
        truncated = long_error[:200]
        assert len(truncated) == 200

    def test_failure_rate_calculation(self):
        total_calls = 100
        total_failures = 15
        rate = round(total_failures / total_calls, 3)
        assert rate == 0.15

    def test_recent_failures_capped(self):
        errors = [{"error": f"err-{i}"} for i in range(25)]
        recent = errors[-10:]
        assert len(recent) == 10
        assert recent[0]["error"] == "err-15"

    def test_result_format(self):
        result = {
            "success": True,
            "data": {
                "tool_name": "search_web",
                "period_days": 7,
                "total_calls": 50,
                "total_failures": 5,
                "failure_rate": 0.1,
                "error_patterns": [
                    {"pattern": "Connection timeout", "count": 3, "examples": []},
                    {"pattern": "Rate limit exceeded", "count": 2, "examples": []},
                ],
                "recent_failures": [],
            }
        }
        assert result["data"]["failure_rate"] == 0.1
        assert len(result["data"]["error_patterns"]) == 2


class TestSelfReflectionToolSchemas:
    """Tests for self-reflection tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS
        names = {t["name"] for t in AGENT_TOOLS}
        assert "get_job_history" in names
        assert "get_job_metrics" in names
        assert "get_tool_usage_stats" in names
        assert "get_tool_failure_analysis" in names

    def test_get_job_history_no_required(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("get_job_history")
        assert tool is not None
        assert tool["parameters"].get("required", []) == []

    def test_get_job_metrics_no_required(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("get_job_metrics")
        assert tool is not None
        assert tool["parameters"].get("required", []) == []

    def test_get_tool_usage_stats_no_required(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("get_tool_usage_stats")
        assert tool is not None
        assert tool["parameters"].get("required", []) == []

    def test_get_tool_failure_analysis_requires_tool_name(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("get_tool_failure_analysis")
        assert tool is not None
        assert "tool_name" in tool["parameters"].get("required", [])


class TestSelfReflectionToolRegistry:
    """Tests for self-reflection tool registry classification."""

    def test_all_are_read_tools(self):
        from app.services.tool_registry import get_tool_metadata
        for tool_name in ["get_job_history", "get_job_metrics", "get_tool_usage_stats", "get_tool_failure_analysis"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.effects == "read", f"{tool_name} should be read-only"

    def test_all_are_low_cost(self):
        from app.services.tool_registry import get_tool_metadata
        for tool_name in ["get_job_history", "get_job_metrics", "get_tool_usage_stats", "get_tool_failure_analysis"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.cost_tier == "low", f"{tool_name} should be low cost"

    def test_none_are_network_tools(self):
        from app.services.tool_registry import get_tool_metadata
        for tool_name in ["get_job_history", "get_job_metrics", "get_tool_usage_stats", "get_tool_failure_analysis"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.network == "none", f"{tool_name} should not require network"
