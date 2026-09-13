"""Unit tests for the pure tool-selection scoring extracted from the executor."""

from datetime import datetime, timedelta, timezone

from app.services import agent_tool_scoring as scoring


def _stat(success, failure, last_error=""):
    return {"success": success, "failure": failure, "last_error": last_error}


def test_success_ratio_is_smoothed_so_one_sample_is_not_certainty():
    assert scoring.tool_success_ratio(_stat(1, 0)) == 2 / 3
    assert scoring.tool_success_ratio(_stat(0, 1)) == 1 / 3
    assert scoring.tool_success_ratio(_stat(0, 0)) == 0.5
    assert scoring.tool_success_ratio("not a stat") == 0.0


def test_merge_sums_counts_and_keeps_the_latest_error():
    merged = scoring.merge_tool_stats(
        {"search_docs": _stat(2, 1, "timeout")},
        {"search_docs": _stat(3, 0), "write_note": _stat(1, 1, "bad input")},
        {"ignored": "not a dict"},
    )
    assert merged["search_docs"] == {
        "success": 5,
        "failure": 1,
        "last_error": "timeout",
    }
    assert merged["write_note"]["last_error"] == "bad input"
    assert "ignored" not in merged


def test_normalize_drops_unusable_rows_and_bounds_error_text():
    normalized = scoring.normalize_tool_stats_map(
        {"": _stat(1, 1), "  ": _stat(1, 1), "ok": _stat(1, 1, "x" * 500), "bad": 7}
    )
    assert list(normalized) == ["ok"]
    assert len(normalized["ok"]["last_error"]) == 200


def test_tool_family_classification():
    assert scoring.tool_family("search_documents") == "retrieval"
    assert scoring.tool_family("ingest_repository") == "ingestion"
    assert scoring.tool_family("summarize_document") == "analysis"
    assert scoring.tool_family("create_research_note") == "synthesis"
    assert scoring.tool_family("generate_flowchart") == "visualization"
    assert scoring.tool_family("pause_job") == "other"
    assert scoring.tool_family("") == "unknown"


def test_visualization_wins_over_prefix_family():
    # "generate_" would be synthesis, but a chart is a chart.
    assert scoring.tool_family("generate_chart") == "visualization"


def test_family_bonus_favours_an_unused_family():
    state = {
        "actions_taken": [
            {"action": {"tool": "search_documents"}},
            {"action": {"tool": "search_papers"}},
        ]
    }
    unused = scoring.family_diversification_bonus(
        "create_research_note", state=state, selection_cfg={}
    )
    used = scoring.family_diversification_bonus(
        "search_documents", state=state, selection_cfg={}
    )
    assert unused > used


def test_family_bonus_is_disabled_by_config_and_by_empty_history():
    state = {"actions_taken": [{"action": {"tool": "search_documents"}}]}
    assert (
        scoring.family_diversification_bonus(
            "create_note",
            state=state,
            selection_cfg={"family_diversification_enabled": False},
        )
        == 0.0
    )
    assert (
        scoring.family_diversification_bonus(
            "create_note", state={"actions_taken": []}, selection_cfg={}
        )
        == 0.0
    )


def test_baseline_mode_scores_purely_on_observed_quality():
    good = scoring.tool_priority_score(_stat(9, 1), mode="baseline")
    poor = scoring.tool_priority_score(_stat(1, 9), mode="baseline")
    unseen = scoring.tool_priority_score(_stat(0, 0), mode="baseline")
    assert good > unseen > poor


def test_exploration_lifts_an_unseen_tool_above_its_raw_ratio():
    unseen = _stat(0, 0)
    explored = scoring.tool_priority_score(unseen, total_trials=20, mode="adaptive")
    baseline = scoring.tool_priority_score(unseen, mode="baseline")
    assert explored > baseline


def test_thompson_sampling_is_deterministic_for_identical_identity():
    kwargs = dict(
        mode="thompson",
        tool_name="search_documents",
        job_id="job-1",
        iteration=3,
        context_tag="think",
    )
    first = scoring.tool_priority_score(_stat(4, 2), **kwargs)
    second = scoring.tool_priority_score(_stat(4, 2), **kwargs)
    other_iteration = scoring.tool_priority_score(
        _stat(4, 2), **{**kwargs, "iteration": 4}
    )
    assert first == second, "replay of the same state must reproduce the same draw"
    assert first != other_iteration


def test_feedback_bias_is_bounded_and_directional():
    state = {"feedback_learning": {"tool_bias": {"a": 1.0, "b": -1.0, "c": "junk"}}}
    assert scoring.feedback_tool_bias("a", state, weight=0.5, max_abs=0.3) == 0.3
    assert scoring.feedback_tool_bias("b", state, weight=0.5, max_abs=0.3) == -0.3
    assert scoring.feedback_tool_bias("c", state) == 0.0
    assert scoring.feedback_tool_bias("a", state, enabled=False) == 0.0
    assert scoring.feedback_tool_bias("a", None) == 0.0


def test_ranking_prefers_the_better_tool_and_breaks_ties_by_name():
    stats = {"beta": _stat(9, 1), "alpha": _stat(1, 9)}
    assert scoring.rank_tools_for_selection(
        ["alpha", "beta"], stats, mode="baseline"
    ) == ["beta", "alpha"]
    # Identical stats must still produce a stable, name-ordered ranking.
    tied = {"beta": _stat(2, 2), "alpha": _stat(2, 2)}
    assert scoring.rank_tools_for_selection(
        ["beta", "alpha"], tied, mode="baseline"
    ) == ["alpha", "beta"]


def test_ranking_handles_empty_and_blank_input():
    assert scoring.rank_tools_for_selection([], {}) == []
    assert scoring.rank_tools_for_selection(["  ", ""], {}, mode="baseline") == []


def test_cooldown_expires_once_the_iteration_passes_it():
    cooldowns = {"run_command": 5}
    assert scoring.is_tool_in_cooldown("run_command", cooldowns, 4) is True
    assert scoring.is_tool_in_cooldown("run_command", cooldowns, 5) is True
    assert scoring.is_tool_in_cooldown("run_command", cooldowns, 6) is False
    assert scoring.is_tool_in_cooldown("other", cooldowns, 1) is False
    assert scoring.is_tool_in_cooldown("run_command", {"run_command": "x"}, 1) is False


def test_priors_decay_by_half_over_one_half_life():
    updated = datetime(2026, 1, 1, tzinfo=timezone.utc)
    now = updated + timedelta(days=45)
    assert scoring.apply_decay_to_prior_counts(
        100, 40, updated, now=now, half_life_days=45.0
    ) == (50, 20)


def test_priors_are_untouched_when_decay_is_off_or_age_is_unknown():
    updated = datetime(2026, 1, 1, tzinfo=timezone.utc)
    now = updated + timedelta(days=400)
    assert scoring.apply_decay_to_prior_counts(
        10, 4, updated, now=now, enabled=False
    ) == (10, 4)
    assert scoring.apply_decay_to_prior_counts(10, 4, None, now=now) == (10, 4)
    # A clock that moved backwards must not inflate counts.
    assert scoring.apply_decay_to_prior_counts(
        10, 4, updated, now=updated - timedelta(days=5)
    ) == (10, 4)


def test_decay_floor_keeps_ancient_evidence_from_vanishing_entirely():
    updated = datetime(2020, 1, 1, tzinfo=timezone.utc)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    successes, failures = scoring.apply_decay_to_prior_counts(
        1000, 1000, updated, now=now, half_life_days=45.0, min_factor=0.01
    )
    assert (successes, failures) == (10, 10)


def test_stable_fraction_is_deterministic_and_in_range():
    values = [scoring.stable_fraction(f"key-{i}") for i in range(50)]
    assert all(0.0 <= value < 1.0 for value in values)
    assert scoring.stable_fraction("key-1") == scoring.stable_fraction("key-1")
    assert len(set(values)) > 40, "hash bucketing should spread across the range"
