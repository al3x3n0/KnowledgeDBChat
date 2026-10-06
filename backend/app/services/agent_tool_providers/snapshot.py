"""Autonomous-job tools: the ``snapshot`` provider.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

from app.services.agent_tool_providers.base import (
    AgentToolExecutionContext,
    FunctionToolProvider,
)


def build_autonomous_snapshot_provider(executor: Any) -> FunctionToolProvider:
    """Workspace snapshot and drift-detection tools for AutonomousAgentExecutor."""

    def _run_iteration(ctx: AgentToolExecutionContext) -> int:
        # The iteration is the job's. Runtime state has no such key, so
        # reading it there stamped every snapshot 0, and drift never saw
        # an iteration pass.
        state = ctx.state if isinstance(ctx.state, dict) else {}
        value = state.get("iteration")
        if value is None:
            value = getattr(ctx.job, "iteration", 0)
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    async def _capture_snapshot(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import re
        from datetime import datetime as _dt

        state = ctx.state if isinstance(ctx.state, dict) else {}
        snap_name = str(params.get("name", "")).strip()[:100]
        extra_keys = params.get("keys", [])
        if not snap_name:
            return {"error": "Missing required parameter: name"}
        if not re.match(r"^[a-zA-Z0-9_\-]+$", snap_name):
            return {
                "error": "Snapshot name must be alphanumeric with underscores/hyphens only"
            }
        doc_ids = {
            str(f.get("document_id") or f.get("source_id"))
            for f in state.get("findings", [])
            if f.get("document_id") or f.get("source_id")
        }
        snapshot = {
            "iteration": _run_iteration(ctx),
            "timestamp": _dt.utcnow().isoformat(),
            "findings_count": len(state.get("findings", [])),
            "actions_count": len(state.get("actions_taken", [])),
            "goal_progress": state.get("goal_progress", 0),
            "documents_found": len(doc_ids),
            # Deep: the executor updates a tool's counters in place.
            "tool_stats": copy.deepcopy(state.get("tool_stats", {})),
            "stalled_iterations": state.get("stalled_iterations", 0),
            "artifacts_count": len(state.get("artifacts", [])),
            "formatted_outputs_count": len(state.get("formatted_outputs", [])),
            "focus_directive": state.get("focus_directive", ""),
            "skill_profile_role": (state.get("skill_profile") or {}).get("role", ""),
        }
        if isinstance(extra_keys, list) and extra_keys:
            custom = {}
            for key in extra_keys[:20]:
                key = str(key).strip()
                if key and key != "workspace_snapshots":
                    val = state.get(key)
                    if val is not None:
                        custom[key] = str(val)[:5000]
            if custom:
                snapshot["custom_keys"] = custom
        snapshots = state.setdefault("workspace_snapshots", {})
        if len(snapshots) >= 20 and snap_name not in snapshots:
            oldest = min(snapshots, key=lambda n: snapshots[n].get("iteration", 0))
            del snapshots[oldest]
        snapshots[snap_name] = snapshot
        return {
            "success": True,
            "data": {
                "name": snap_name,
                "iteration": snapshot["iteration"],
                "findings_count": snapshot["findings_count"],
                "actions_count": snapshot["actions_count"],
                "goal_progress": snapshot["goal_progress"],
                "documents_found": snapshot["documents_found"],
                "total_snapshots": len(snapshots),
            },
            "findings": [
                {
                    "type": "workspace_snapshot",
                    "name": snap_name,
                    "iteration": snapshot["iteration"],
                    "findings_count": snapshot["findings_count"],
                    "actions_count": snapshot["actions_count"],
                    "goal_progress": snapshot["goal_progress"],
                }
            ],
        }

    async def _compare_snapshots(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        name_a = str(params.get("snapshot_a", "")).strip()
        name_b = str(params.get("snapshot_b", "")).strip()
        if not name_a or not name_b:
            return {"error": "Both snapshot_a and snapshot_b are required"}
        snapshots = state.get("workspace_snapshots", {})
        snap_a = snapshots.get(name_a)
        snap_b = snapshots.get(name_b)
        if not snap_a:
            return {"error": f"Snapshot '{name_a}' not found"}
        if not snap_b:
            return {"error": f"Snapshot '{name_b}' not found"}
        numeric_keys = [
            "findings_count",
            "actions_count",
            "goal_progress",
            "documents_found",
            "stalled_iterations",
            "artifacts_count",
            "formatted_outputs_count",
        ]
        diff = {}
        for key in numeric_keys:
            a_val = float(snap_a.get(key, 0) or 0)
            b_val = float(snap_b.get(key, 0) or 0)
            delta = b_val - a_val
            diff[key] = {
                "before": a_val,
                "after": b_val,
                "delta": delta,
                "direction": "increased"
                if delta > 0
                else ("decreased" if delta < 0 else "unchanged"),
            }
        for key in ["focus_directive", "skill_profile_role"]:
            a_val = str(snap_a.get(key, ""))
            b_val = str(snap_b.get(key, ""))
            diff[key] = {"before": a_val, "after": b_val, "changed": a_val != b_val}
        # Keys the run asked to have captured are compared too; they were
        # stored and then never read.
        custom_a = snap_a.get("custom_keys") or {}
        custom_b = snap_b.get("custom_keys") or {}
        if custom_a or custom_b:
            diff["custom_keys"] = {
                key: {
                    "before": custom_a.get(key),
                    "after": custom_b.get(key),
                    "changed": custom_a.get(key) != custom_b.get(key),
                }
                for key in sorted(set(custom_a) | set(custom_b))
            }
        stats_a = snap_a.get("tool_stats", {})
        stats_b = snap_b.get("tool_stats", {})
        tools_added = set(stats_b.keys()) - set(stats_a.keys())
        tools_removed = set(stats_a.keys()) - set(stats_b.keys())
        diff["tool_stats"] = {
            "tools_added": sorted(list(tools_added)),
            "tools_removed": sorted(list(tools_removed)),
            "total_before": len(stats_a),
            "total_after": len(stats_b),
        }
        iter_a = snap_a.get("iteration", "?")
        iter_b = snap_b.get("iteration", "?")
        summary_parts = [f"Between iteration {iter_a} and {iter_b}:"]
        if diff["findings_count"]["delta"]:
            summary_parts.append(
                f"findings {'+' if diff['findings_count']['delta'] > 0 else ''}{int(diff['findings_count']['delta'])}"
            )
        if diff["goal_progress"]["delta"]:
            summary_parts.append(
                f"progress {'+' if diff['goal_progress']['delta'] > 0 else ''}{int(diff['goal_progress']['delta'])}%"
            )
        if tools_added:
            summary_parts.append(f"{len(tools_added)} new tools used")
        return {
            "success": True,
            "data": {
                "diff": diff,
                "summary": " ".join(summary_parts),
                "snapshot_a_iteration": iter_a,
                "snapshot_b_iteration": iter_b,
            },
            "findings": [
                {
                    "type": "snapshot_diff",
                    "summary": " ".join(summary_parts),
                    "snapshot_a_iteration": iter_a,
                    "snapshot_b_iteration": iter_b,
                    "diff": diff,
                }
            ],
        }

    async def _detect_drift(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        state = ctx.state if isinstance(ctx.state, dict) else {}
        baseline_name = str(params.get("baseline", "")).strip()
        custom_thresholds = params.get("thresholds", {})
        if not baseline_name:
            return {"error": "Missing required parameter: baseline"}
        snapshots = state.get("workspace_snapshots", {})
        baseline = snapshots.get(baseline_name)
        if not baseline:
            return {"error": f"Baseline snapshot '{baseline_name}' not found"}
        doc_ids = {
            str(f.get("document_id") or f.get("source_id"))
            for f in state.get("findings", [])
            if f.get("document_id") or f.get("source_id")
        }
        current = {
            "iteration": _run_iteration(ctx),
            "findings_count": len(state.get("findings", [])),
            "actions_count": len(state.get("actions_taken", [])),
            "goal_progress": state.get("goal_progress", 0),
            "documents_found": len(doc_ids),
            "stalled_iterations": state.get("stalled_iterations", 0),
            "artifacts_count": len(state.get("artifacts", [])),
        }
        thresholds = {
            "stalled_iterations": 2,
            "goal_progress_drop": 0,
            "findings_stale_iterations": 5,
            "tool_failure_rate": 0.5,
        }
        if isinstance(custom_thresholds, dict):
            for key, val in custom_thresholds.items():
                if key in thresholds:
                    try:
                        thresholds[key] = float(val)
                    except (TypeError, ValueError):
                        pass
        iterations_elapsed = current["iteration"] - baseline.get("iteration", 0)
        alerts = []
        if current["stalled_iterations"] > thresholds["stalled_iterations"]:
            alerts.append(
                {
                    "metric": "stalled_iterations",
                    "baseline_value": baseline.get("stalled_iterations", 0),
                    "current_value": current["stalled_iterations"],
                    "severity": "warning",
                    "message": f"Agent has stalled for {current['stalled_iterations']} iterations",
                }
            )
        progress_drop = baseline.get("goal_progress", 0) - current["goal_progress"]
        if progress_drop > thresholds["goal_progress_drop"]:
            alerts.append(
                {
                    "metric": "goal_progress",
                    "baseline_value": baseline.get("goal_progress", 0),
                    "current_value": current["goal_progress"],
                    "severity": "critical" if progress_drop > 20 else "warning",
                    "message": f"Goal progress dropped by {progress_drop}% since baseline",
                }
            )
        findings_delta = current["findings_count"] - baseline.get("findings_count", 0)
        if (
            findings_delta == 0
            and iterations_elapsed >= thresholds["findings_stale_iterations"]
        ):
            alerts.append(
                {
                    "metric": "findings_count",
                    "baseline_value": baseline.get("findings_count", 0),
                    "current_value": current["findings_count"],
                    "severity": "warning",
                    "message": f"No new findings in {iterations_elapsed} iterations",
                }
            )
        for tool, stats in state.get("tool_stats", {}).items():
            if isinstance(stats, dict):
                total = (stats.get("success", 0) or 0) + (stats.get("failure", 0) or 0)
                if total >= 3:
                    fail_rate = (stats.get("failure", 0) or 0) / total
                    if fail_rate > thresholds["tool_failure_rate"]:
                        alerts.append(
                            {
                                "metric": f"tool_failure:{tool}",
                                "baseline_value": 0,
                                "current_value": round(fail_rate, 2),
                                "severity": "warning",
                                "message": f"Tool '{tool}' failure rate is {round(fail_rate * 100)}%",
                            }
                        )
        if not alerts:
            alerts.append(
                {
                    "metric": "overall",
                    "baseline_value": baseline.get("goal_progress", 0),
                    "current_value": current["goal_progress"],
                    "severity": "info",
                    "message": f"No drift detected after {iterations_elapsed} iterations",
                }
            )
        severity_counts = {}
        for alert in alerts:
            severity_counts[alert["severity"]] = (
                severity_counts.get(alert["severity"], 0) + 1
            )
        summary = f"{len(alerts)} alert(s) after {iterations_elapsed} iterations"
        if severity_counts.get("critical"):
            summary += f" ({severity_counts['critical']} critical)"
        elif severity_counts.get("warning"):
            summary += f" ({severity_counts['warning']} warnings)"
        payload = {
            "success": True,
            "data": {
                "alerts": alerts,
                "metrics_compared": len(current),
                "iterations_elapsed": iterations_elapsed,
                "summary": summary,
            },
        }
        if any(a["severity"] in ("warning", "critical") for a in alerts):
            payload["findings"] = [
                {
                    "type": "drift_detected",
                    "title": f"Drift detected: {summary}",
                    "baseline": baseline_name,
                    "alert_count": len(alerts),
                    "severity_counts": severity_counts,
                }
            ]
        return payload

    return FunctionToolProvider(
        name="autonomous_snapshot_tools",
        modes={"autonomous"},
        handlers={
            "capture_snapshot": _capture_snapshot,
            "compare_snapshots": _compare_snapshots,
            "detect_drift": _detect_drift,
        },
    )
