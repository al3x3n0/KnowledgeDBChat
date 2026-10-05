"""The workspace snapshot tools: capture_snapshot, compare_snapshots, detect_drift.

These call the real handlers. The file used to restate each handler's logic
inline and assert on the restatement -- `params = {}; name =
str(params.get("name", "")).strip()[:100]; assert not name` -- so fifty-one
tests passed whatever the tools did, including while every snapshot a real
run took was stamped iteration 0.

Two kinds of state are used on purpose. `_runtime_state()` is what the
executor actually hands a tool (`initialize_runtime_state`), with the
iteration on the job, where the runtime keeps it. `_state()` is a hand-built
dict that also sets `state["iteration"]`, the key the handlers read; it is
used only where a test is about some *other* rule and needs iterations to
differ, so that one defect does not turn every test red.
"""

from types import SimpleNamespace

import pytest

from app.services.agent_runtime_state_service import initialize_runtime_state
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_snapshot_provider,
)

pytestmark = pytest.mark.unit

TOOLS = ("capture_snapshot", "compare_snapshots", "detect_drift")


async def _run(tool, params, state, iteration=0):
    """Run one real handler against a state dict."""
    provider = build_autonomous_snapshot_provider(SimpleNamespace())
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=None,
        service=None,
        user_id="u",
        job=SimpleNamespace(
            id="job-1", user_id="u", goal="g", config={}, iteration=iteration
        ),
        state=state,
    )
    return await provider._handlers[tool](params, ctx)


def _runtime_state(**overrides):
    """The state a real run starts with."""
    state = initialize_runtime_state()
    state.update(overrides)
    return state


def _state(iteration=0, findings=0, progress=0, **extra):
    state = {
        "iteration": iteration,
        "findings": [{"title": f"f{i}"} for i in range(findings)],
        "goal_progress": progress,
    }
    state.update(extra)
    return state


async def _capture(name, state, **params):
    result = await _run("capture_snapshot", {"name": name, **params}, state)
    assert result.get("success") is True, result
    return result


def _alerts(result, metric=None):
    alerts = result["data"]["alerts"]
    if metric is None:
        return alerts
    return [a for a in alerts if a["metric"] == metric]


class TestCaptureSnapshot:
    @pytest.mark.parametrize("params", [{}, {"name": ""}, {"name": "   "}])
    async def test_a_name_is_required(self, params):
        state = _runtime_state()
        result = await _run("capture_snapshot", params, state)
        assert result == {"error": "Missing required parameter: name"}
        assert "workspace_snapshots" not in state

    @pytest.mark.parametrize(
        "name", ["has space", "has.dot", "has/slash", "has@at", "né"]
    )
    async def test_a_name_with_other_characters_is_refused(self, name):
        state = _runtime_state()
        result = await _run("capture_snapshot", {"name": name}, state)
        assert "alphanumeric" in result["error"]
        assert "success" not in result
        assert "workspace_snapshots" not in state

    @pytest.mark.parametrize(
        "name", ["after_search", "before-synthesis", "snap1", "A_B_C"]
    )
    async def test_letters_digits_underscores_and_hyphens_are_accepted(self, name):
        state = _runtime_state()
        result = await _capture(name, state)
        assert result["data"]["name"] == name
        assert name in state["workspace_snapshots"]

    async def test_surrounding_whitespace_is_not_part_of_the_name(self):
        state = _runtime_state()
        result = await _capture("  after_search  ", state)
        assert result["data"]["name"] == "after_search"
        assert list(state["workspace_snapshots"]) == ["after_search"]

    async def test_a_long_name_is_capped_at_100_characters(self):
        state = _runtime_state()
        result = await _capture("a" * 200, state)
        assert result["data"]["name"] == "a" * 100
        assert list(state["workspace_snapshots"]) == ["a" * 100]

    async def test_it_records_the_metrics_of_the_run(self):
        state = _runtime_state(
            findings=[
                {"title": "A", "document_id": "doc-1"},
                {"title": "B", "source_id": "src-2"},
                {"title": "C", "document_id": "doc-1"},  # same document again
                {"title": "D"},  # no document at all
            ],
            actions_taken=[{"tool": "search_documents"}, {"tool": "read_document"}],
            goal_progress=45,
            stalled_iterations=3,
            artifacts=[{"id": "a1"}],
            formatted_outputs=[{"title": "t1"}, {"title": "t2"}],
            focus_directive="Focus on papers",
            skill_profile={"role": "researcher", "display_name": "Researcher"},
            tool_stats={"search_documents": {"success": 3, "failure": 0}},
        )

        result = await _capture("after_search", state)

        snapshot = state["workspace_snapshots"]["after_search"]
        assert snapshot["findings_count"] == 4
        assert snapshot["actions_count"] == 2
        assert snapshot["goal_progress"] == 45
        assert snapshot["documents_found"] == 2
        assert snapshot["stalled_iterations"] == 3
        assert snapshot["artifacts_count"] == 1
        assert snapshot["formatted_outputs_count"] == 2
        assert snapshot["focus_directive"] == "Focus on papers"
        assert snapshot["skill_profile_role"] == "researcher"
        assert snapshot["tool_stats"] == {
            "search_documents": {"success": 3, "failure": 0}
        }
        assert snapshot["timestamp"]
        assert result["data"] == {
            "name": "after_search",
            "iteration": snapshot["iteration"],
            "findings_count": 4,
            "actions_count": 2,
            "goal_progress": 45,
            "documents_found": 2,
            "total_snapshots": 1,
        }

    async def test_it_returns_a_workspace_snapshot_finding(self):
        # The spec declares produces=("workspace_snapshot",), and contracts
        # count findings, so the evidence has to arrive on that channel.
        result = await _capture("s", _runtime_state(goal_progress=10))
        assert [f["type"] for f in result["findings"]] == ["workspace_snapshot"]
        assert result["findings"][0]["name"] == "s"
        assert result["findings"][0]["goal_progress"] == 10

    async def test_a_fresh_run_snapshots_as_zeroes(self):
        state = _runtime_state()
        result = await _capture("start", state)
        assert result["data"]["findings_count"] == 0
        assert result["data"]["actions_count"] == 0
        assert result["data"]["goal_progress"] == 0
        assert result["data"]["documents_found"] == 0
        assert state["workspace_snapshots"]["start"]["skill_profile_role"] == ""

    async def test_a_snapshot_is_stamped_with_the_iteration_it_was_taken_at(self):
        state = _runtime_state()
        result = await _run("capture_snapshot", {"name": "s"}, state, iteration=7)
        assert result["data"]["iteration"] == 7
        assert state["workspace_snapshots"]["s"]["iteration"] == 7

    async def test_a_snapshot_is_not_changed_by_later_progress(self):
        state = _runtime_state(findings=[{"title": "A"}], goal_progress=20)
        await _capture("early", state)

        state["findings"].append({"title": "B"})
        state["actions_taken"].append({"tool": "search_documents"})
        state["goal_progress"] = 80
        state["tool_stats"]["read_document"] = {"success": 1, "failure": 0}

        snapshot = state["workspace_snapshots"]["early"]
        assert snapshot["findings_count"] == 1
        assert snapshot["actions_count"] == 0
        assert snapshot["goal_progress"] == 20
        assert "read_document" not in snapshot["tool_stats"]

    async def test_a_snapshots_tool_counts_are_frozen(self):
        state = _runtime_state(
            tool_stats={"search_documents": {"success": 3, "failure": 0}}
        )
        await _capture("early", state)

        # What _record_tool_outcome does: mutate the tool's own slot.
        state["tool_stats"]["search_documents"]["failure"] += 4

        snapshot = state["workspace_snapshots"]["early"]
        assert snapshot["tool_stats"]["search_documents"]["failure"] == 0

    async def test_extra_keys_are_captured_as_text(self):
        state = _runtime_state(custom_field="custom_value", another=42)
        await _capture("s", state, keys=["custom_field", " another ", "absent"])
        assert state["workspace_snapshots"]["s"]["custom_keys"] == {
            "custom_field": "custom_value",
            "another": "42",
        }

    async def test_an_extra_value_is_capped_at_5000_characters(self):
        state = _runtime_state(long_key="V" * 6000)
        await _capture("s", state, keys=["long_key"])
        assert state["workspace_snapshots"]["s"]["custom_keys"]["long_key"] == (
            "V" * 5000
        )

    async def test_at_most_20_extra_keys_are_captured(self):
        state = _runtime_state(**{f"k{i}": i for i in range(25)})
        await _capture("s", state, keys=[f"k{i}" for i in range(25)])
        custom = state["workspace_snapshots"]["s"]["custom_keys"]
        assert sorted(custom) == sorted(f"k{i}" for i in range(20))

    async def test_a_snapshot_never_contains_the_snapshots(self):
        state = _runtime_state()
        await _capture("first", state)
        await _capture("second", state, keys=["workspace_snapshots"])
        assert "custom_keys" not in state["workspace_snapshots"]["second"]

    async def test_no_extra_keys_means_no_custom_section(self):
        state = _runtime_state()
        await _capture("s", state)
        assert "custom_keys" not in state["workspace_snapshots"]["s"]

    async def test_snapshots_accumulate_under_their_names(self):
        state = _runtime_state()
        first = await _capture("one", state)
        second = await _capture("two", state)
        assert first["data"]["total_snapshots"] == 1
        assert second["data"]["total_snapshots"] == 2
        assert sorted(state["workspace_snapshots"]) == ["one", "two"]

    async def test_reusing_a_name_replaces_that_snapshot(self):
        state = _runtime_state(goal_progress=10)
        await _capture("mine", state)
        state["goal_progress"] = 70
        result = await _capture("mine", state)
        assert result["data"]["total_snapshots"] == 1
        assert state["workspace_snapshots"]["mine"]["goal_progress"] == 70

    async def test_the_21st_snapshot_evicts_the_oldest(self):
        state = _state()
        for i in range(1, 21):
            state["iteration"] = i
            await _capture(f"snap_{i}", state)
        assert len(state["workspace_snapshots"]) == 20

        state["iteration"] = 21
        result = await _capture("snap_new", state)

        assert result["data"]["total_snapshots"] == 20
        assert "snap_new" in state["workspace_snapshots"]
        assert "snap_1" not in state["workspace_snapshots"]
        assert "snap_2" in state["workspace_snapshots"]

    async def test_replacing_a_snapshot_at_the_cap_evicts_nothing(self):
        state = _state()
        for i in range(1, 21):
            state["iteration"] = i
            await _capture(f"snap_{i}", state)

        state["iteration"] = 21
        await _capture("snap_5", state)

        assert len(state["workspace_snapshots"]) == 20
        assert "snap_1" in state["workspace_snapshots"]
        assert state["workspace_snapshots"]["snap_5"]["iteration"] == 21


class TestCompareSnapshots:
    @pytest.mark.parametrize(
        "params",
        [
            {},
            {"snapshot_a": "a"},
            {"snapshot_b": "b"},
            {"snapshot_a": "a", "snapshot_b": " "},
        ],
    )
    async def test_both_names_are_required(self, params):
        state = _runtime_state()
        await _capture("a", state)
        await _capture("b", state)
        result = await _run("compare_snapshots", params, state)
        assert result == {"error": "Both snapshot_a and snapshot_b are required"}

    async def test_an_unknown_snapshot_is_named_in_the_refusal(self):
        state = _runtime_state()
        await _capture("known", state)

        missing_a = await _run(
            "compare_snapshots", {"snapshot_a": "nope", "snapshot_b": "known"}, state
        )
        missing_b = await _run(
            "compare_snapshots", {"snapshot_a": "known", "snapshot_b": "gone"}, state
        )

        assert missing_a == {"error": "Snapshot 'nope' not found"}
        assert missing_b == {"error": "Snapshot 'gone' not found"}

    async def test_comparing_before_any_snapshot_exists_is_refused(self):
        result = await _run(
            "compare_snapshots",
            {"snapshot_a": "a", "snapshot_b": "b"},
            _runtime_state(),
        )
        assert result == {"error": "Snapshot 'a' not found"}

    async def test_it_reports_what_grew_between_two_snapshots(self):
        state = _state(
            iteration=5,
            findings=3,
            progress=20,
            actions_taken=[{"tool": "search_documents"}],
            tool_stats={"search_documents": {"success": 3, "failure": 0}},
        )
        await _capture("before", state)

        state["iteration"] = 15
        state["findings"] += [{"title": "x", "document_id": f"d{i}"} for i in range(7)]
        state["actions_taken"] += [{"tool": "summarize_document"}] * 4
        state["goal_progress"] = 60
        state["stalled_iterations"] = 1
        state["artifacts"] = [{"id": "a1"}]
        state["tool_stats"]["summarize_document"] = {"success": 2, "failure": 0}
        await _capture("after", state)

        result = await _run(
            "compare_snapshots", {"snapshot_a": "before", "snapshot_b": "after"}, state
        )

        assert result["success"] is True
        diff = result["data"]["diff"]
        assert diff["findings_count"] == {
            "before": 3,
            "after": 10,
            "delta": 7,
            "direction": "increased",
        }
        assert diff["actions_count"]["delta"] == 4
        assert diff["goal_progress"]["delta"] == 40
        assert diff["goal_progress"]["direction"] == "increased"
        assert diff["documents_found"]["delta"] == 7
        assert diff["stalled_iterations"]["delta"] == 1
        assert diff["artifacts_count"]["delta"] == 1
        assert diff["formatted_outputs_count"]["direction"] == "unchanged"
        assert diff["tool_stats"] == {
            "tools_added": ["summarize_document"],
            "tools_removed": [],
            "total_before": 1,
            "total_after": 2,
        }
        assert result["data"]["snapshot_a_iteration"] == 5
        assert result["data"]["snapshot_b_iteration"] == 15
        summary = result["data"]["summary"]
        assert summary.startswith("Between iteration 5 and 15:")
        assert "findings +7" in summary
        assert "progress +40%" in summary
        assert "1 new tools used" in summary

    async def test_a_regression_is_reported_as_a_decrease(self):
        state = _state(iteration=1, findings=5, progress=50)
        await _capture("before", state)
        state["iteration"] = 2
        state["findings"] = state["findings"][:2]
        state["goal_progress"] = 30
        await _capture("after", state)

        result = await _run(
            "compare_snapshots", {"snapshot_a": "before", "snapshot_b": "after"}, state
        )

        diff = result["data"]["diff"]
        assert diff["findings_count"]["delta"] == -3
        assert diff["findings_count"]["direction"] == "decreased"
        assert diff["goal_progress"]["delta"] == -20
        assert diff["goal_progress"]["direction"] == "decreased"
        assert "findings -3" in result["data"]["summary"]
        assert "progress -20%" in result["data"]["summary"]

    async def test_the_order_of_the_names_decides_the_sign(self):
        state = _state(iteration=1, findings=1)
        await _capture("early", state)
        state["findings"].append({"title": "more"})
        await _capture("late", state)

        result = await _run(
            "compare_snapshots", {"snapshot_a": "late", "snapshot_b": "early"}, state
        )

        assert result["data"]["diff"]["findings_count"]["delta"] == -1

    async def test_identical_snapshots_report_nothing_changed(self):
        state = _state(iteration=4, findings=2, progress=30)
        await _capture("a", state)
        await _capture("b", state)

        result = await _run(
            "compare_snapshots", {"snapshot_a": "a", "snapshot_b": "b"}, state
        )

        diff = result["data"]["diff"]
        for key in ("findings_count", "actions_count", "goal_progress"):
            assert diff[key]["delta"] == 0
            assert diff[key]["direction"] == "unchanged"
        assert diff["focus_directive"]["changed"] is False
        assert result["data"]["summary"] == "Between iteration 4 and 4:"

    async def test_a_snapshot_can_be_compared_with_itself(self):
        state = _state(findings=2)
        await _capture("a", state)
        result = await _run(
            "compare_snapshots", {"snapshot_a": "a", "snapshot_b": "a"}, state
        )
        assert result["data"]["diff"]["findings_count"]["direction"] == "unchanged"

    async def test_a_change_of_focus_or_role_is_reported(self):
        state = _state(
            focus_directive="Focus on papers", skill_profile={"role": "researcher"}
        )
        await _capture("a", state)
        state["focus_directive"] = "Focus on code"
        state["skill_profile"] = {"role": "synthesizer"}
        await _capture("b", state)

        result = await _run(
            "compare_snapshots", {"snapshot_a": "a", "snapshot_b": "b"}, state
        )

        diff = result["data"]["diff"]
        assert diff["focus_directive"] == {
            "before": "Focus on papers",
            "after": "Focus on code",
            "changed": True,
        }
        assert diff["skill_profile_role"] == {
            "before": "researcher",
            "after": "synthesizer",
            "changed": True,
        }

    async def test_a_tool_no_longer_present_is_reported_removed(self):
        state = _state(tool_stats={"old_tool": {"success": 1, "failure": 0}})
        await _capture("a", state)
        state["tool_stats"] = {"new_tool": {"success": 1, "failure": 0}}
        await _capture("b", state)

        result = await _run(
            "compare_snapshots", {"snapshot_a": "a", "snapshot_b": "b"}, state
        )

        assert result["data"]["diff"]["tool_stats"]["tools_added"] == ["new_tool"]
        assert result["data"]["diff"]["tool_stats"]["tools_removed"] == ["old_tool"]

    async def test_it_returns_a_snapshot_diff_finding(self):
        # produces=("snapshot_diff",): the diff must be a finding, not only data.
        state = _state(iteration=1, findings=1)
        await _capture("a", state)
        state["findings"].append({"title": "more"})
        await _capture("b", state)

        result = await _run(
            "compare_snapshots", {"snapshot_a": "a", "snapshot_b": "b"}, state
        )

        assert [f["type"] for f in result["findings"]] == ["snapshot_diff"]
        assert result["findings"][0]["diff"] == result["data"]["diff"]
        assert result["findings"][0]["summary"] == result["data"]["summary"]

    async def test_comparing_does_not_alter_the_snapshots(self):
        state = _state(findings=1)
        await _capture("a", state)
        await _capture("b", state)
        before = {k: dict(v) for k, v in state["workspace_snapshots"].items()}

        await _run("compare_snapshots", {"snapshot_a": "a", "snapshot_b": "b"}, state)

        assert state["workspace_snapshots"] == before

    async def test_a_change_in_a_captured_extra_key_is_reported(self):
        state = _state(execution_mode="plan")
        await _capture("a", state, keys=["execution_mode"])
        state["execution_mode"] = "explore"
        await _capture("b", state, keys=["execution_mode"])

        result = await _run(
            "compare_snapshots", {"snapshot_a": "a", "snapshot_b": "b"}, state
        )

        assert "execution_mode" in str(result["data"]["diff"])

    async def test_the_diff_says_which_iterations_it_spans(self):
        state = _runtime_state()
        await _run("capture_snapshot", {"name": "a"}, state, iteration=5)
        await _run("capture_snapshot", {"name": "b"}, state, iteration=15)

        result = await _run(
            "compare_snapshots",
            {"snapshot_a": "a", "snapshot_b": "b"},
            state,
            iteration=15,
        )

        assert result["data"]["snapshot_a_iteration"] == 5
        assert result["data"]["snapshot_b_iteration"] == 15


class TestDetectDrift:
    @pytest.mark.parametrize("params", [{}, {"baseline": ""}, {"baseline": "  "}])
    async def test_a_baseline_is_required(self, params):
        result = await _run("detect_drift", params, _runtime_state())
        assert result == {"error": "Missing required parameter: baseline"}

    async def test_an_unknown_baseline_is_named_in_the_refusal(self):
        state = _runtime_state()
        await _capture("other", state)
        result = await _run("detect_drift", {"baseline": "my_baseline"}, state)
        assert result == {"error": "Baseline snapshot 'my_baseline' not found"}

    async def test_a_healthy_run_reports_no_drift(self):
        state = _state(iteration=5, findings=2, progress=30)
        await _capture("base", state)
        state["iteration"] = 15
        state["findings"].append({"title": "new"})
        state["goal_progress"] = 55
        state["tool_stats"] = {"search_documents": {"success": 9, "failure": 1}}

        result = await _run("detect_drift", {"baseline": "base"}, state)

        assert result["success"] is True
        alerts = _alerts(result)
        assert [a["severity"] for a in alerts] == ["info"]
        assert alerts[0]["metric"] == "overall"
        assert "No drift detected after 10 iterations" in alerts[0]["message"]
        assert result["data"]["iterations_elapsed"] == 10
        assert result["data"]["summary"] == "1 alert(s) after 10 iterations"
        # Nothing is wrong, so nothing is recorded as a finding.
        assert "findings" not in result

    async def test_checking_immediately_after_the_baseline_is_quiet(self):
        state = _runtime_state(goal_progress=40, findings=[{"title": "A"}])
        await _capture("base", state)
        result = await _run("detect_drift", {"baseline": "base"}, state)
        assert [a["severity"] for a in _alerts(result)] == ["info"]

    @pytest.mark.parametrize("stalled, alerted", [(2, False), (3, True), (4, True)])
    async def test_stalling_alerts_above_two_iterations(self, stalled, alerted):
        state = _state(iteration=1, findings=1)
        await _capture("base", state)
        state["stalled_iterations"] = stalled

        result = await _run("detect_drift", {"baseline": "base"}, state)

        alerts = _alerts(result, "stalled_iterations")
        assert bool(alerts) is alerted
        if alerted:
            assert alerts[0]["severity"] == "warning"
            assert alerts[0]["baseline_value"] == 0
            assert alerts[0]["current_value"] == stalled
            assert f"stalled for {stalled} iterations" in alerts[0]["message"]

    @pytest.mark.parametrize(
        "now, severity",
        [(60, None), (75, None), (59, "warning"), (40, "warning"), (39, "critical")],
    )
    async def test_progress_regression_is_graded_by_its_size(self, now, severity):
        # Baseline 60: any drop warns, a drop of more than 20 points is critical.
        state = _state(iteration=1, findings=1, progress=60)
        await _capture("base", state)
        state["goal_progress"] = now

        result = await _run("detect_drift", {"baseline": "base"}, state)

        alerts = _alerts(result, "goal_progress")
        if severity is None:
            assert alerts == []
        else:
            assert [a["severity"] for a in alerts] == [severity]
            assert alerts[0]["baseline_value"] == 60
            assert alerts[0]["current_value"] == now
            assert f"dropped by {60 - now}%" in alerts[0]["message"]

    @pytest.mark.parametrize("elapsed, alerted", [(4, False), (5, True), (9, True)])
    async def test_no_new_findings_alerts_after_five_iterations(self, elapsed, alerted):
        state = _state(iteration=3, findings=5)
        await _capture("base", state)
        state["iteration"] = 3 + elapsed

        result = await _run("detect_drift", {"baseline": "base"}, state)

        alerts = _alerts(result, "findings_count")
        assert bool(alerts) is alerted
        if alerted:
            assert alerts[0]["severity"] == "warning"
            assert f"No new findings in {elapsed} iterations" in alerts[0]["message"]

    async def test_new_findings_are_not_stale_however_long_it_took(self):
        state = _state(iteration=3, findings=5)
        await _capture("base", state)
        state["iteration"] = 30
        state["findings"].append({"title": "new"})

        result = await _run("detect_drift", {"baseline": "base"}, state)

        assert _alerts(result, "findings_count") == []

    async def test_a_real_run_with_no_new_findings_is_flagged(self):
        state = _runtime_state(findings=[{"title": "A"}])
        await _run("capture_snapshot", {"name": "base"}, state, iteration=3)

        result = await _run("detect_drift", {"baseline": "base"}, state, iteration=10)

        assert result["data"]["iterations_elapsed"] == 7
        assert [a["severity"] for a in _alerts(result, "findings_count")] == ["warning"]

    @pytest.mark.parametrize(
        "stats, alerted",
        [
            ({"success": 1, "failure": 4}, True),
            ({"success": 0, "failure": 3}, True),
            ({"success": 2, "failure": 2}, False),  # exactly half is not "high"
            ({"success": 9, "failure": 1}, False),
            ({"success": 0, "failure": 2}, False),  # too few calls to judge
        ],
    )
    async def test_a_tool_failing_more_than_half_the_time_alerts(self, stats, alerted):
        state = _state(iteration=1, findings=1)
        await _capture("base", state)
        state["tool_stats"] = {"search_documents": stats}

        result = await _run("detect_drift", {"baseline": "base"}, state)

        alerts = _alerts(result, "tool_failure:search_documents")
        assert bool(alerts) is alerted
        if alerted:
            total = stats["success"] + stats["failure"]
            assert alerts[0]["severity"] == "warning"
            assert alerts[0]["current_value"] == round(stats["failure"] / total, 2)
            assert "search_documents" in alerts[0]["message"]

    async def test_each_failing_tool_gets_its_own_alert(self):
        state = _state(iteration=1, findings=1)
        await _capture("base", state)
        state["tool_stats"] = {
            "search_documents": {"success": 0, "failure": 5},
            "read_document": {"success": 8, "failure": 0},
            "web_search": {"success": 1, "failure": 3, "last_error": "timeout"},
        }

        result = await _run("detect_drift", {"baseline": "base"}, state)

        assert sorted(a["metric"] for a in _alerts(result)) == [
            "tool_failure:search_documents",
            "tool_failure:web_search",
        ]

    async def test_a_custom_stall_threshold_replaces_the_default(self):
        state = _state(iteration=1, findings=1)
        await _capture("base", state)
        state["stalled_iterations"] = 4

        default = await _run("detect_drift", {"baseline": "base"}, state)
        raised = await _run(
            "detect_drift",
            {"baseline": "base", "thresholds": {"stalled_iterations": 5}},
            state,
        )

        assert _alerts(default, "stalled_iterations")
        assert _alerts(raised, "stalled_iterations") == []

    async def test_a_custom_progress_threshold_tolerates_a_small_drop(self):
        state = _state(iteration=1, findings=1, progress=60)
        await _capture("base", state)
        state["goal_progress"] = 52
        params = {"baseline": "base", "thresholds": {"goal_progress_drop": 10}}

        tolerated = await _run("detect_drift", params, state)
        state["goal_progress"] = 49
        flagged = await _run("detect_drift", params, state)

        assert _alerts(tolerated, "goal_progress") == []
        assert [a["severity"] for a in _alerts(flagged, "goal_progress")] == ["warning"]

    async def test_custom_stale_and_failure_thresholds_apply(self):
        state = _state(iteration=1, findings=1)
        await _capture("base", state)
        state["iteration"] = 3
        state["tool_stats"] = {"web_search": {"success": 3, "failure": 2}}

        result = await _run(
            "detect_drift",
            {
                "baseline": "base",
                "thresholds": {
                    "findings_stale_iterations": 2,
                    "tool_failure_rate": 0.3,
                },
            },
            state,
        )

        assert sorted(a["metric"] for a in _alerts(result)) == [
            "findings_count",
            "tool_failure:web_search",
        ]

    @pytest.mark.parametrize(
        "thresholds",
        [
            {"stalled_iterations": "not a number"},
            {"stalled_iterations": None},
            {"no_such_threshold": 99},
            "stalled_iterations=99",
            None,
        ],
    )
    async def test_an_unusable_threshold_leaves_the_default_in_force(self, thresholds):
        state = _state(iteration=1, findings=1)
        await _capture("base", state)
        state["stalled_iterations"] = 3

        result = await _run(
            "detect_drift", {"baseline": "base", "thresholds": thresholds}, state
        )

        assert [a["severity"] for a in _alerts(result, "stalled_iterations")] == [
            "warning"
        ]

    async def test_drift_is_recorded_as_a_finding(self):
        state = _state(iteration=1, findings=1, progress=80)
        await _capture("base", state)
        state["goal_progress"] = 50
        state["stalled_iterations"] = 4

        result = await _run("detect_drift", {"baseline": "base"}, state)

        assert sorted(a["metric"] for a in _alerts(result)) == [
            "goal_progress",
            "stalled_iterations",
        ]
        assert "overall" not in [a["metric"] for a in _alerts(result)]
        assert result["data"]["summary"] == "2 alert(s) after 0 iterations (1 critical)"
        (finding,) = result["findings"]
        assert finding["type"] == "drift_detected"
        assert finding["baseline"] == "base"
        assert finding["alert_count"] == 2
        assert finding["severity_counts"] == {"warning": 1, "critical": 1}

    async def test_warnings_alone_are_counted_in_the_summary(self):
        state = _state(iteration=1, findings=1, progress=50)
        await _capture("base", state)
        state["goal_progress"] = 45
        state["stalled_iterations"] = 3

        result = await _run("detect_drift", {"baseline": "base"}, state)

        assert result["data"]["summary"] == "2 alert(s) after 0 iterations (2 warnings)"
        assert result["findings"][0]["severity_counts"] == {"warning": 2}

    async def test_detecting_drift_does_not_move_the_baseline(self):
        state = _state(iteration=1, findings=1, progress=80)
        await _capture("base", state)
        before = dict(state["workspace_snapshots"]["base"])
        state["goal_progress"] = 10

        await _run("detect_drift", {"baseline": "base"}, state)
        again = await _run("detect_drift", {"baseline": "base"}, state)

        assert state["workspace_snapshots"]["base"] == before
        assert list(state["workspace_snapshots"]) == ["base"]
        assert [a["severity"] for a in _alerts(again, "goal_progress")] == ["critical"]


class TestSnapshotProvider:
    def test_the_provider_answers_all_three_tools(self):
        provider = build_autonomous_snapshot_provider(SimpleNamespace())
        assert set(provider._handlers) == set(TOOLS)


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

        for tool_name in TOOLS:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.effects == "read"

    def test_all_are_low_cost(self):
        from app.services.tool_registry import get_tool_metadata

        for tool_name in TOOLS:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.cost_tier == "low"

    def test_none_is_network_tool(self):
        from app.services.tool_registry import get_tool_metadata

        for tool_name in TOOLS:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.network == "none"


def test_a_snapshot_needs_no_repository():
    """It records the run's own state, so the plan must not clone first."""
    from app.services.agent_evidence_map import chain_for

    assert chain_for(["workspace_snapshot"], job_type="research") == [
        "capture_snapshot"
    ]
    assert chain_for(["snapshot_diff"], job_type="research") == [
        "capture_snapshot",
        "compare_snapshots",
    ]
