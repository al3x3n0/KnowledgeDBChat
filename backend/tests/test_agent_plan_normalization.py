"""Tests for plan normalization and deterministic fallbacks.

Planner output is LLM-shaped, so the interesting cases are the malformed ones:
alternate key names, wrong types, junk entries, and unbounded lists. A plan that
survives normalization is what the whole run then follows.
"""

from app.services import agent_plan_normalization as plans


def test_normalize_accepts_bare_strings_as_steps():
    normalized = plans.normalize_execution_plan({"plan_steps": ["  Scope the work  "]})
    assert normalized == [
        {
            "title": "Scope the work",
            "objective": "Scope the work",
            "exit_criteria": "",
            "suggested_tools": [],
            "status": "pending",
        }
    ]


def test_normalize_accepts_alternate_key_names():
    normalized = plans.normalize_execution_plan(
        {
            "steps": [
                {
                    "name": "Gather",
                    "purpose": "collect evidence",
                    "done_when": "3 docs read",
                    "tools": ["search_documents", "  ", "read_document_content"],
                }
            ]
        }
    )
    assert normalized == [
        {
            "title": "Gather",
            "objective": "collect evidence",
            "exit_criteria": "3 docs read",
            "suggested_tools": ["search_documents", "read_document_content"],
            "status": "pending",
        }
    ]


def test_normalize_derives_a_title_from_the_objective_when_missing():
    normalized = plans.normalize_execution_plan(
        {"plan_steps": [{"objective": "no title here"}]}
    )
    assert normalized[0]["title"] == "no title here"


def test_normalize_drops_untitled_steps_and_respects_the_cap():
    normalized = plans.normalize_execution_plan(
        {"plan_steps": [{"title": ""}, "a", "b", "c"]}, max_steps=2
    )
    assert [step["title"] for step in normalized] == ["a", "b"]


def test_normalize_returns_nothing_for_unusable_payloads():
    assert plans.normalize_execution_plan("junk") == []
    assert plans.normalize_execution_plan({}) == []
    assert plans.normalize_execution_plan({"steps": "not a list"}) == []


def test_normalize_bounds_oversized_text_and_tool_lists():
    normalized = plans.normalize_execution_plan(
        {
            "plan_steps": [
                {
                    "title": "t" * 500,
                    "objective": "o" * 900,
                    "exit_criteria": "e" * 900,
                    "suggested_tools": [f"tool_{i}" for i in range(20)],
                }
            ]
        }
    )
    step = normalized[0]
    assert len(step["title"]) == 220
    assert len(step["objective"]) == 500
    assert len(step["exit_criteria"]) == 300
    assert len(step["suggested_tools"]) == 6


def test_fallback_plan_adds_an_external_research_step_only_for_research_types():
    research = [step["title"] for step in plans.fallback_execution_plan("research")]
    analysis = [step["title"] for step in plans.fallback_execution_plan("analysis")]
    assert "Expand with external research" in research
    assert "Expand with external research" not in analysis
    assert research[0] == "Scope the goal and constraints"
    assert research[-1] == "Publish results"


def test_fallback_plan_respects_the_step_cap_and_tolerates_no_job_type():
    assert len(plans.fallback_execution_plan("research", max_steps=3)) == 3
    assert plans.fallback_execution_plan("") == plans.fallback_execution_plan(
        "analysis"
    )


def test_fallback_plan_steps_are_all_pending_and_carry_exit_criteria():
    for step in plans.fallback_execution_plan("research"):
        assert step["status"] == "pending"
        assert step["exit_criteria"]
        assert step["suggested_tools"]


def test_causal_normalize_assigns_ids_and_clamps_confidence():
    plan = plans.normalize_causal_experiment_plan(
        {
            "hypotheses": [
                " inlining regressed ",
                {"id": "HX", "hypothesis": "regalloc", "confidence": 3.7},
            ],
            "experiments": [{"name": "toggle inlining"}],
        }
    )
    assert [h["id"] for h in plan["hypotheses"]] == ["H1", "HX"]
    assert plan["hypotheses"][0]["confidence"] == 0.5
    assert plan["hypotheses"][1]["confidence"] == 1.0


def test_causal_normalize_repoints_an_experiment_at_a_real_hypothesis():
    plan = plans.normalize_causal_experiment_plan(
        {
            "hypotheses": [{"id": "H1", "statement": "a"}],
            "experiments": [{"name": "e", "hypothesis": "does-not-exist"}],
        }
    )
    # A dangling reference is repaired rather than left pointing at nothing.
    assert plan["experiments"][0]["hypothesis_id"] == "H1"


def test_causal_normalize_defaults_effort_and_supplies_decision_rules():
    plan = plans.normalize_causal_experiment_plan(
        {
            "hypotheses": ["h"],
            "experiments": [{"name": "e", "effort": "EXTREME"}],
            "decision_rules": [],
        }
    )
    assert plan["experiments"][0]["estimated_effort"] == "medium"
    assert len(plan["decision_rules"]) == 2


def test_causal_normalize_filters_priority_order_to_known_experiments():
    plan = plans.normalize_causal_experiment_plan(
        {
            "hypotheses": ["h"],
            "experiments": [{"id": "E9", "name": "second"}],
            "priority_order": ["E9", "nope"],
        }
    )
    assert plan["priority_order"] == ["E9"]


def test_causal_normalize_falls_back_to_experiment_order_without_priority():
    plan = plans.normalize_causal_experiment_plan(
        {
            "hypotheses": ["h"],
            "experiments": [{"id": "E1", "name": "a"}, {"id": "E2", "name": "b"}],
        }
    )
    assert plan["priority_order"] == ["E1", "E2"]


def test_causal_normalize_requires_both_hypotheses_and_experiments():
    assert plans.normalize_causal_experiment_plan("junk") == {}
    assert plans.normalize_causal_experiment_plan({}) == {}
    assert (
        plans.normalize_causal_experiment_plan({"hypotheses": ["h"], "experiments": []})
        == {}
    )
    assert (
        plans.normalize_causal_experiment_plan(
            {"hypotheses": [], "experiments": [{"name": "e"}]}
        )
        == {}
    )


def test_causal_fallback_is_falsifiable_and_marked_as_a_fallback():
    plan = plans.fallback_causal_experiment_plan("Reduce compile time by 10%")
    assert plan["source"] == "fallback"
    assert "Reduce compile time by 10%" in plan["hypotheses"][0]["statement"]
    assert plan["priority_order"] == [e["id"] for e in plan["experiments"]]
    # Every fallback experiment must state what would disprove it.
    for experiment in plan["experiments"]:
        assert experiment["expected_evidence"]["falsifies"]


def test_causal_fallback_respects_caps():
    plan = plans.fallback_causal_experiment_plan(
        "G", max_hypotheses=1, max_experiments=1
    )
    assert len(plan["hypotheses"]) == 1
    assert len(plan["experiments"]) == 1
    assert plan["priority_order"] == ["E1"]
