"""Golden tests for the volatile prompt sections.

Prompt text is contract surface: a whitespace or ordering change alters what
every agent sees. These assert exact rendered output, captured from the
implementation before it was extracted from the executor.
"""

from app.services import agent_prompt_sections as sections

STATE = {
    "execution_plan": [
        {
            "title": "Reproduce",
            "objective": "Rebuild the regression",
            "exit_criteria": "benchmark reproduces",
            "suggested_tools": ["run_experiment", "search_documents"],
            "status": "done",
        },
        {
            "title": "Isolate",
            "objective": "Bisect the commit range",
            "status": "active",
        },
    ],
    "plan_step_index": 1,
    "execution_mode": "Plan_Then_Act",
    "subgoals": [
        {"title": "collect evidence", "status": "done"},
        {"title": "verify", "status": "active"},
    ],
    "subgoal_index": 1,
    "critic_notes": [
        {
            "trajectory_assessment": "drifting",
            "severity": "high",
            "confidence": 0.7312,
            "pivot": "return to benchmark",
            "recommended_tools": ["run_experiment"],
            "risks": ["no baseline"],
        }
    ],
    "tool_stats": {
        "run_experiment": {"success": 3, "failure": 1},
        "search_documents": {"success": 1, "failure": 4},
    },
    "tool_priors": {
        "run_experiment": {"success": 10, "failure": 2, "last_error": "timeout"}
    },
    "skill_profile": {
        "display_name": "Compiler researcher",
        "prompt_directives": ["cite evidence", "no speculation"],
        "preferred_tools": ["run_experiment"],
        "discouraged_tools": ["create_note"],
    },
    "feedback_learning": {
        "feedback_count": 4,
        "avg_rating": 4.25,
        "preferred_tools": ["run_experiment"],
        "discouraged_tools": ["web_search"],
        "highlights": ["good repro steps"],
        "tool_bias": {"run_experiment": 0.4},
    },
    "causal_experiment_plan": {
        "hypotheses": [
            {"id": "H1", "statement": "inlining regressed"},
            {"id": "H2", "statement": "regalloc regressed"},
        ],
        "experiments": [
            {
                "id": "E1",
                "name": "toggle inlining",
                "hypothesis_id": "H1",
                "expected_evidence": {
                    "supports": ["speedup returns"],
                    "falsifies": ["no change"],
                },
            },
            {"id": "E2", "name": "toggle regalloc", "hypothesis_id": "H2"},
        ],
        "priority_order": ["E2", "E1"],
    },
}

GRAPH_RUNTIME = {
    "dag_stats": {"total_nodes": 12, "total_edges": 14, "critical_path_length": 5},
    "graph_health": {
        "status": "degraded",
        "severity_score": 3,
        "reasons": ["stalled node", "retry loop"],
    },
    "verification_successes": 2,
    "verification_attempts": 3,
    "summarization_successes": 1,
    "summarization_attempts": 1,
    "recommended_actions": ["re-run verification", "compact history"],
}


def test_execution_plan_section():
    assert sections.format_execution_plan(STATE) == (
        "EXECUTION PLAN (Plan-Then-Act):\n"
        "- Execution mode: plan_then_act\n"
        "- Current step 2/2: Isolate\n"
        "- Current objective: Bisect the commit range\n"
        "- Completed steps: 1"
    )


def test_causal_experiment_section_follows_priority_order():
    assert sections.format_causal_experiment_plan(STATE) == (
        "CAUSAL EXPERIMENT PLAN:\n"
        "- Hypotheses: 2\n"
        "  - H1: inlining regressed\n"
        "  - H2: regalloc regressed\n"
        "- Next experiment IDs: E2, E1\n"
        "  - E2 (H2): toggle regalloc\n"
        "  - E1 (H1): toggle inlining\n"
        "    support signal: speedup returns\n"
        "    falsify signal: no change"
    )


def test_subgoals_section():
    assert sections.format_subgoals(STATE) == (
        "SUBGOALS:\n- Current subgoal 2/2: verify\n- Subgoals completed: 1"
    )


def test_critic_section_renders_the_latest_note():
    assert sections.format_critic(STATE) == (
        "CRITIC FEEDBACK:\n"
        "- Assessment: drifting\n"
        "- Severity: high\n"
        "- Confidence: 0.73\n"
        "- Pivot: return to benchmark\n"
        "- Recommended tools: run_experiment\n"
        "- Top risk: no baseline"
    )


def test_tool_stats_section_merges_priors_into_current_counts():
    assert sections.format_tool_stats(STATE) == (
        "ADAPTIVE TOOL HINTS:\n"
        "- Historical priors loaded for 1 tools.\n"
        "- Strong tools:\n"
        "  - run_experiment: success=13, failure=3\n"
        "  - search_documents: success=1, failure=4\n"
        "- Weak tools (avoid repeats unless needed):\n"
        "  - search_documents: success=1, failure=4\n"
        "  - run_experiment: success=13, failure=3"
    )


def test_skill_profile_section():
    assert sections.format_skill_profile(STATE) == (
        "ROLE PROFILE: Compiler researcher\n"
        "- cite evidence\n"
        "- no speculation\n"
        "- Preferred tools: run_experiment\n"
        "- Discouraged tools: create_note"
    )


def test_feedback_learning_section():
    assert sections.format_feedback_learning(STATE) == (
        "HUMAN FEEDBACK LEARNING:\n"
        "- Average rating context: 4.25/5\n"
        "- Prefer tools: run_experiment\n"
        "- Avoid tools: web_search\n"
        "- Recent feedback note: good repro steps"
    )


def test_execution_graph_section():
    assert sections.format_execution_graph(GRAPH_RUNTIME) == (
        "EXECUTION GRAPH:\n"
        "- Health: degraded (severity=3)\n"
        "- Health reasons: stalled node, retry loop\n"
        "- Nodes=12, edges=14, critical_path=5\n"
        "- Verify/summarize: 2/3 verifications succeeded; 1/1 summaries succeeded\n"
        "- Recommended actions:\n"
        "  - re-run verification\n"
        "  - compact history"
    )


def test_every_section_is_empty_for_empty_state():
    # An absent section must contribute nothing rather than a stray heading,
    # otherwise the prompt fills with empty scaffolding.
    for render in (
        sections.format_execution_plan,
        sections.format_causal_experiment_plan,
        sections.format_subgoals,
        sections.format_critic,
        sections.format_tool_stats,
        sections.format_skill_profile,
        sections.format_feedback_learning,
        sections.format_execution_graph,
    ):
        assert render({}) == ""


def test_sections_tolerate_malformed_state():
    malformed = {
        "execution_plan": "not a list",
        "subgoals": [None, "junk"],
        "critic_notes": ["not a dict"],
        "tool_stats": "junk",
        "tool_priors": None,
        "skill_profile": [],
        "feedback_learning": {"feedback_count": 0},
        "causal_experiment_plan": {"hypotheses": [], "experiments": []},
    }
    assert sections.format_execution_plan(malformed) == ""
    assert sections.format_tool_stats(malformed) == ""
    assert sections.format_skill_profile(malformed) == ""
    assert sections.format_feedback_learning(malformed) == ""
    assert sections.format_causal_experiment_plan(malformed) == ""
    # A subgoal list of junk still reports position, never raises.
    assert sections.format_subgoals(malformed).startswith("SUBGOALS:")


def test_a_malformed_critic_note_still_emits_a_hollow_section():
    """Pins pre-existing behaviour carried over from the executor.

    A non-dict latest note degrades to ``{}``, which passes the dict check and
    renders a heading plus a default confidence line. That injects a section
    carrying no information into the prompt. Preserved deliberately so the
    extraction changed no prompt text; worth fixing on its own, since the
    remedy is a prompt-content change and should be visible as one.
    """
    assert sections.format_critic({"critic_notes": ["not a dict"]}) == (
        "CRITIC FEEDBACK:\n- Confidence: 0.00"
    )


def test_graph_section_is_empty_when_the_graph_has_no_nodes_or_edges():
    assert (
        sections.format_execution_graph(
            {"dag_stats": {"total_nodes": 0, "total_edges": 0}}
        )
        == ""
    )
    assert sections.format_execution_graph("not a dict") == ""
