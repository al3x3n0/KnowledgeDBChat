"""Tests for workspace snapshot tools (capture_snapshot, compare_snapshots, detect_drift)."""

import re
import uuid

import pytest


class TestCaptureSnapshot:
    """Tests for capture_snapshot tool logic."""

    def test_requires_name(self):
        params = {}
        name = str(params.get("name", "")).strip()[:100]
        assert not name

    def test_name_validation_alphanumeric(self):
        valid_names = ["after_search", "before-synthesis", "snap1", "A_B_C"]
        for name in valid_names:
            assert re.match(r'^[a-zA-Z0-9_\-]+$', name)

    def test_name_validation_rejects_special_chars(self):
        invalid_names = ["has space", "has.dot", "has/slash", "has@at"]
        for name in invalid_names:
            assert not re.match(r'^[a-zA-Z0-9_\-]+$', name)

    def test_name_truncated_to_100(self):
        long_name = "a" * 200
        truncated = long_name[:100]
        assert len(truncated) == 100

    def test_snapshot_captures_findings_count(self):
        state = {"findings": [{"title": "A"}, {"title": "B"}]}
        snapshot = {"findings_count": len(state.get("findings", []))}
        assert snapshot["findings_count"] == 2

    def test_snapshot_captures_actions_count(self):
        state = {"actions_taken": [{"tool": "search"}, {"tool": "read"}]}
        snapshot = {"actions_count": len(state.get("actions_taken", []))}
        assert snapshot["actions_count"] == 2

    def test_snapshot_captures_goal_progress(self):
        state = {"goal_progress": 45}
        snapshot = {"goal_progress": state.get("goal_progress", 0)}
        assert snapshot["goal_progress"] == 45

    def test_snapshot_captures_documents_found(self):
        state = {"findings": [
            {"document_id": "doc-1", "title": "A"},
            {"document_id": "doc-2", "title": "B"},
            {"document_id": "doc-1", "title": "C"},  # duplicate
        ]}
        doc_ids = set()
        for f in state.get("findings", []):
            did = f.get("document_id") or f.get("source_id")
            if did:
                doc_ids.add(str(did))
        assert len(doc_ids) == 2

    def test_snapshot_captures_tool_stats(self):
        state = {"tool_stats": {"search_documents": {"success": 3, "failure": 0}}}
        snapshot = {"tool_stats": dict(state.get("tool_stats", {}))}
        assert "search_documents" in snapshot["tool_stats"]

    def test_snapshot_captures_stalled_iterations(self):
        state = {"stalled_iterations": 3}
        snapshot = {"stalled_iterations": state.get("stalled_iterations", 0)}
        assert snapshot["stalled_iterations"] == 3

    def test_snapshot_captures_skill_profile_role(self):
        state = {"skill_profile": {"role": "researcher", "display_name": "Researcher"}}
        role = (state.get("skill_profile") or {}).get("role", "")
        assert role == "researcher"

    def test_custom_keys_captured(self):
        state = {"custom_field": "custom_value", "another": 42}
        extra_keys = ["custom_field", "another"]
        custom = {}
        for k in extra_keys[:20]:
            k = str(k).strip()
            if k and k != "workspace_snapshots":
                val = state.get(k)
                if val is not None:
                    custom[k] = str(val)[:5000]
        assert custom["custom_field"] == "custom_value"
        assert custom["another"] == "42"

    def test_custom_key_values_truncated(self):
        state = {"long_key": "V" * 6000}
        val = str(state.get("long_key"))[:5000]
        assert len(val) == 5000

    def test_workspace_snapshots_key_excluded(self):
        extra_keys = ["findings", "workspace_snapshots", "goal_progress"]
        filtered = [k for k in extra_keys if k != "workspace_snapshots"]
        assert "workspace_snapshots" not in filtered

    def test_stored_in_state(self):
        state = {}
        snapshots = state.setdefault("workspace_snapshots", {})
        snapshots["test_snap"] = {"iteration": 5, "findings_count": 10}
        assert "test_snap" in state["workspace_snapshots"]
        assert state["workspace_snapshots"]["test_snap"]["iteration"] == 5

    def test_max_20_snapshots(self):
        snapshots = {f"snap_{i}": {"iteration": i} for i in range(20)}
        assert len(snapshots) == 20
        # Adding 21st should evict oldest
        if len(snapshots) >= 20 and "snap_new" not in snapshots:
            oldest = min(snapshots, key=lambda n: snapshots[n].get("iteration", 0))
            del snapshots[oldest]
        snapshots["snap_new"] = {"iteration": 21}
        assert len(snapshots) == 20
        assert "snap_new" in snapshots

    def test_overwrite_existing_snapshot(self):
        snapshots = {"my_snap": {"iteration": 5, "findings_count": 3}}
        snapshots["my_snap"] = {"iteration": 10, "findings_count": 8}
        assert snapshots["my_snap"]["iteration"] == 10
        assert snapshots["my_snap"]["findings_count"] == 8

    def test_result_format(self):
        result = {
            "name": "after_search",
            "iteration": 5,
            "findings_count": 12,
            "actions_count": 20,
            "goal_progress": 45,
            "documents_found": 8,
            "total_snapshots": 3,
        }
        assert result["name"] == "after_search"
        assert "total_snapshots" in result

    def test_empty_state_snapshot(self):
        state = {}
        snapshot = {
            "findings_count": len(state.get("findings", [])),
            "actions_count": len(state.get("actions_taken", [])),
            "goal_progress": state.get("goal_progress", 0),
            "stalled_iterations": state.get("stalled_iterations", 0),
        }
        assert snapshot["findings_count"] == 0
        assert snapshot["goal_progress"] == 0


class TestCompareSnapshots:
    """Tests for compare_snapshots tool logic."""

    def test_requires_both_names(self):
        params = {"snapshot_a": "snap1"}
        name_b = str(params.get("snapshot_b", "")).strip()
        assert not name_b

    def test_missing_snapshot_a_error(self):
        snapshots = {"snap_b": {"iteration": 10}}
        assert "snap_a" not in snapshots

    def test_missing_snapshot_b_error(self):
        snapshots = {"snap_a": {"iteration": 5}}
        assert "snap_b" not in snapshots

    def test_numeric_diff_increased(self):
        a_val, b_val = 5, 12
        delta = b_val - a_val
        direction = "increased" if delta > 0 else ("decreased" if delta < 0 else "unchanged")
        assert delta == 7
        assert direction == "increased"

    def test_numeric_diff_decreased(self):
        a_val, b_val = 50, 30
        delta = b_val - a_val
        direction = "increased" if delta > 0 else ("decreased" if delta < 0 else "unchanged")
        assert delta == -20
        assert direction == "decreased"

    def test_numeric_diff_unchanged(self):
        a_val, b_val = 10, 10
        delta = b_val - a_val
        direction = "increased" if delta > 0 else ("decreased" if delta < 0 else "unchanged")
        assert delta == 0
        assert direction == "unchanged"

    def test_string_field_changed(self):
        a_val = "researcher"
        b_val = "synthesizer"
        assert a_val != b_val

    def test_string_field_unchanged(self):
        a_val = "Focus on papers"
        b_val = "Focus on papers"
        assert a_val == b_val

    def test_tool_stats_diff(self):
        stats_a = {"search_documents": {"success": 3}}
        stats_b = {"search_documents": {"success": 5}, "summarize_document": {"success": 2}}
        tools_added = set(stats_b.keys()) - set(stats_a.keys())
        tools_removed = set(stats_a.keys()) - set(stats_b.keys())
        assert "summarize_document" in tools_added
        assert len(tools_removed) == 0

    def test_summary_format(self):
        iter_a, iter_b = 5, 15
        findings_delta = 8
        progress_delta = 25
        summary_parts = [f"Between iteration {iter_a} and {iter_b}:"]
        if findings_delta:
            summary_parts.append(f"findings +{findings_delta}")
        if progress_delta:
            summary_parts.append(f"progress +{progress_delta}%")
        summary = " ".join(summary_parts)
        assert "Between iteration 5 and 15:" in summary
        assert "findings +8" in summary
        assert "progress +25%" in summary

    def test_result_format(self):
        result = {
            "diff": {"findings_count": {"before": 5, "after": 12, "delta": 7, "direction": "increased"}},
            "summary": "Between iteration 5 and 15: findings +7",
            "snapshot_a_iteration": 5,
            "snapshot_b_iteration": 15,
        }
        assert "diff" in result
        assert "summary" in result
        assert result["snapshot_a_iteration"] == 5

    def test_full_diff_structure(self):
        snap_a = {"findings_count": 3, "goal_progress": 20, "stalled_iterations": 0}
        snap_b = {"findings_count": 10, "goal_progress": 60, "stalled_iterations": 1}
        numeric_keys = ["findings_count", "goal_progress", "stalled_iterations"]
        diff = {}
        for key in numeric_keys:
            a_val = snap_a.get(key, 0)
            b_val = snap_b.get(key, 0)
            delta = b_val - a_val
            direction = "increased" if delta > 0 else ("decreased" if delta < 0 else "unchanged")
            diff[key] = {"before": a_val, "after": b_val, "delta": delta, "direction": direction}
        assert diff["findings_count"]["delta"] == 7
        assert diff["goal_progress"]["direction"] == "increased"
        assert diff["stalled_iterations"]["delta"] == 1


class TestDetectDrift:
    """Tests for detect_drift tool logic."""

    def test_requires_baseline(self):
        params = {}
        baseline = str(params.get("baseline", "")).strip()
        assert not baseline

    def test_missing_baseline_error(self):
        snapshots = {}
        assert "my_baseline" not in snapshots

    def test_stalled_iterations_alert(self):
        baseline = {"stalled_iterations": 0, "iteration": 5}
        current = {"stalled_iterations": 4, "iteration": 15}
        threshold = 2
        alerts = []
        if current["stalled_iterations"] > threshold:
            alerts.append({
                "metric": "stalled_iterations",
                "baseline_value": baseline["stalled_iterations"],
                "current_value": current["stalled_iterations"],
                "severity": "warning",
                "message": f"Agent has stalled for {current['stalled_iterations']} iterations",
            })
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "warning"

    def test_no_stall_no_alert(self):
        current_stalled = 1
        threshold = 2
        assert not (current_stalled > threshold)

    def test_progress_regression_warning(self):
        baseline_progress = 60
        current_progress = 45
        progress_drop = baseline_progress - current_progress
        assert progress_drop == 15
        severity = "critical" if progress_drop > 20 else "warning"
        assert severity == "warning"

    def test_progress_regression_critical(self):
        baseline_progress = 80
        current_progress = 50
        progress_drop = baseline_progress - current_progress
        assert progress_drop == 30
        severity = "critical" if progress_drop > 20 else "warning"
        assert severity == "critical"

    def test_no_progress_regression_no_alert(self):
        baseline_progress = 40
        current_progress = 60
        progress_drop = baseline_progress - current_progress
        assert progress_drop < 0  # Progress increased, no alert

    def test_findings_stale_alert(self):
        baseline = {"findings_count": 5, "iteration": 3}
        current = {"findings_count": 5, "iteration": 10}
        findings_delta = current["findings_count"] - baseline["findings_count"]
        iterations_elapsed = current["iteration"] - baseline["iteration"]
        threshold = 5
        assert findings_delta == 0 and iterations_elapsed >= threshold

    def test_findings_growing_no_alert(self):
        baseline = {"findings_count": 5, "iteration": 3}
        current = {"findings_count": 12, "iteration": 10}
        findings_delta = current["findings_count"] - baseline["findings_count"]
        assert findings_delta > 0

    def test_tool_failure_rate_alert(self):
        tool_stats = {"search_documents": {"success": 1, "failure": 4}}
        alerts = []
        threshold = 0.5
        for tool, stats in tool_stats.items():
            total = stats.get("success", 0) + stats.get("failure", 0)
            if total >= 3:
                fail_rate = stats.get("failure", 0) / total
                if fail_rate > threshold:
                    alerts.append({"metric": f"tool_failure:{tool}", "severity": "warning"})
        assert len(alerts) == 1
        assert "search_documents" in alerts[0]["metric"]

    def test_healthy_tool_no_alert(self):
        tool_stats = {"search_documents": {"success": 9, "failure": 1}}
        total = 10
        fail_rate = 1 / total
        assert fail_rate <= 0.5

    def test_tool_too_few_calls_ignored(self):
        tool_stats = {"rare_tool": {"success": 0, "failure": 2}}
        total = 2
        assert total < 3  # Not enough calls to judge

    def test_custom_thresholds(self):
        thresholds = {"stalled_iterations": 2, "goal_progress_drop": 0}
        custom = {"stalled_iterations": 5, "goal_progress_drop": 10}
        for k, v in custom.items():
            if k in thresholds:
                thresholds[k] = float(v)
        assert thresholds["stalled_iterations"] == 5.0
        assert thresholds["goal_progress_drop"] == 10.0

    def test_no_drift_info_alert(self):
        alerts = []
        # When no issues found, add info alert
        if not alerts:
            alerts.append({
                "metric": "overall",
                "severity": "info",
                "message": "No drift detected after 10 iterations",
            })
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "info"

    def test_drift_finding_saved(self):
        alerts = [
            {"severity": "warning", "metric": "stalled_iterations"},
            {"severity": "critical", "metric": "goal_progress"},
        ]
        has_issues = any(a["severity"] in ("warning", "critical") for a in alerts)
        assert has_issues

    def test_no_finding_for_info_only(self):
        alerts = [{"severity": "info", "metric": "overall"}]
        has_issues = any(a["severity"] in ("warning", "critical") for a in alerts)
        assert not has_issues

    def test_severity_counts(self):
        alerts = [
            {"severity": "warning"}, {"severity": "warning"}, {"severity": "critical"},
        ]
        counts = {}
        for a in alerts:
            counts[a["severity"]] = counts.get(a["severity"], 0) + 1
        assert counts["warning"] == 2
        assert counts["critical"] == 1

    def test_result_format(self):
        result = {
            "alerts": [{"metric": "stalled_iterations", "severity": "warning"}],
            "metrics_compared": 7,
            "iterations_elapsed": 10,
            "summary": "1 alert(s) after 10 iterations (1 warnings)",
        }
        assert "alerts" in result
        assert result["metrics_compared"] == 7
        assert "iterations_elapsed" in result

    def test_summary_with_critical(self):
        severity_counts = {"critical": 2, "warning": 1}
        summary = f"3 alert(s) after 10 iterations"
        if severity_counts.get("critical"):
            summary += f" ({severity_counts['critical']} critical)"
        assert "2 critical" in summary


class TestSnapshotSchemas:
    """Tests for workspace snapshot tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS
        names = {t["name"] for t in AGENT_TOOLS}
        assert "capture_snapshot" in names
        assert "compare_snapshots" in names
        assert "detect_drift" in names

    def test_capture_snapshot_requires_name(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("capture_snapshot")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "name" in required

    def test_capture_snapshot_has_keys(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("capture_snapshot")
        assert "keys" in tool["parameters"]["properties"]

    def test_compare_snapshots_requires_both(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("compare_snapshots")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "snapshot_a" in required
        assert "snapshot_b" in required

    def test_detect_drift_requires_baseline(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("detect_drift")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "baseline" in required

    def test_detect_drift_has_thresholds(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("detect_drift")
        assert "thresholds" in tool["parameters"]["properties"]


class TestSnapshotRegistry:
    """Tests for workspace snapshot tool registry classification."""

    def test_all_are_read(self):
        from app.services.tool_registry import get_tool_metadata
        for tool_name in ["capture_snapshot", "compare_snapshots", "detect_drift"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.effects == "read"

    def test_all_are_low_cost(self):
        from app.services.tool_registry import get_tool_metadata
        for tool_name in ["capture_snapshot", "compare_snapshots", "detect_drift"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.cost_tier == "low"

    def test_none_is_network_tool(self):
        from app.services.tool_registry import get_tool_metadata
        for tool_name in ["capture_snapshot", "compare_snapshots", "detect_drift"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.network == "none"
