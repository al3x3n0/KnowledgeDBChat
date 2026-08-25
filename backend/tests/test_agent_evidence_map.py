"""Knowing which tool yields which evidence, and what must come first.

A run given a goal and no method chose its tools well and then got the order
and the hand-offs wrong: it mined before profiling, and spent three attempts
passing raw assembly where a mined pattern belongs. The chain was right; the
knowledge about the tools was missing.
"""

from __future__ import annotations

import pytest

from app.services import agent_evidence_map as evidence


def test_evidence_names_the_tool_that_produces_it():
    assert evidence.producer_of("dynamic_profile") == "profile_c_workload"
    assert evidence.producer_of("fusion_candidate") == "find_fusion_candidates"
    assert evidence.producer_of("prediction_settled") == "record_measurement"


def test_a_chain_puts_prerequisites_first():
    """Mining before profiling is the mistake this exists to prevent."""
    chain = evidence.chain_for(["fusion_candidate"])

    assert chain.index("profile_c_workload") < chain.index("find_fusion_candidates")


def test_prerequisites_are_pulled_in_even_when_not_asked_for():
    """A contract wants a settled prediction; nothing settles what was never
    recorded."""
    chain = evidence.chain_for(["prediction_settled"])

    assert chain == ["record_prediction", "record_measurement"]


def test_a_transitive_prerequisite_comes_first():
    chain = evidence.chain_for(["fusion_cost_bound"])

    assert chain.index("profile_c_workload") < chain.index("find_fusion_candidates")
    assert chain.index("find_fusion_candidates") < chain.index("cost_fusion_candidate")


def test_a_tool_is_named_once_however_many_things_need_it():
    chain = evidence.chain_for(
        ["fusion_candidate", "fusion_cost_bound", "dynamic_profile"]
    )

    assert chain.count("profile_c_workload") == 1


def test_the_description_says_what_each_tool_is_fed():
    """Three attempts were lost to passing the instructions a pattern was
    found in rather than the pattern."""
    lines = "\n".join(evidence.describe_chain(["fusion_cost_bound"]))

    assert "pattern" in lines
    assert "not the instructions it was found in" in lines


def test_evidence_nothing_can_produce_is_reported():
    """A contract asking for the unobtainable cannot be satisfied however well
    a run behaves, and that is the contract's fault."""
    assert evidence.unobtainable(["fusion_candidate", "vibes"]) == ["vibes"]


def test_an_empty_requirement_yields_no_chain():
    assert evidence.chain_for([]) == []
    assert evidence.describe_chain([]) == []


class _Job:
    def __init__(self, config):
        self.config = config
        self.results = {}
        self.iteration = 1
        self.goal = "find something worth proposing"
        self.status = "running"
        self.job_type = "research"
        self.name = "job"

    def __getattr__(self, name):
        return None


def test_the_prompt_tells_a_run_how_to_get_what_its_contract_wants():
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    job = _Job(
        {
            "goal_contract": {
                "enabled": True,
                "required_finding_types": ["fusion_candidate", "prediction_settled"],
            }
        }
    )

    prompt = AutonomousAgentExecutor()._build_thinking_prompt_stable(job, None, {})

    assert "HOW THIS RUN'S REQUIRED EVIDENCE IS PRODUCED" in prompt
    assert "profile_c_workload" in prompt
    assert "find_fusion_candidates" in prompt
    # It must not read as the method itself: choosing what to measure is the
    # run's job, and this only says where evidence comes from.
    assert "not the whole method" in prompt


def test_a_contract_with_no_named_evidence_adds_nothing():
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    job = _Job({"goal_contract": {"enabled": True, "min_findings": 2}})

    prompt = AutonomousAgentExecutor()._build_thinking_prompt_stable(job, None, {})

    assert "HOW THIS RUN'S REQUIRED EVIDENCE" not in prompt


@pytest.mark.asyncio
async def test_raw_assembly_is_named_as_the_mistake():
    from app.services.agent_compiler_sandbox import cost_fusion_candidate

    result = await cost_fusion_candidate(
        pattern="fmul s0, s0, s0; fmadd s0, s2, s2, s0", cpu="neoverse-n1"
    )

    assert "looks like assembly" in result["error"]
    assert "find_fusion_candidates" in result["error"]


def test_a_validity_rule_contributes_to_the_chain():
    """`predictions_measured` is satisfied by record_measurement, which the
    counting requirements never name. A run given no method got everything
    else right and left its prediction unsettled."""
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    job = _Job(
        {
            "goal_contract": {
                "enabled": True,
                "required_finding_types": ["fusion_candidate"],
                "validity": {"predictions_measured": True, "records_method": True},
            }
        }
    )

    prompt = AutonomousAgentExecutor()._build_thinking_prompt_stable(job, None, {})
    chain = prompt.split("HOW THIS RUN")[1].split("You are not required")[0]

    assert "record_measurement" in chain
    assert "record_method" in chain
    assert chain.index("record_prediction") < chain.index("record_measurement")


# --- the predictability chain ----------------------------------------------


def test_the_counter_tools_are_obtainable():
    """A contract asking for evidence no tool claims to produce tells the run
    it cannot be satisfied. Four tools existed for a week without an entry
    here, so a predictability contract would have said exactly that."""
    required = [
        "counter_trace",
        "predictability_ceiling",
        "counter_tap_selection",
        "predictor_design_result",
    ]

    assert evidence.unobtainable(required) == []


def test_the_predictability_chain_derives_in_the_order_that_works():
    """Sample before measuring, measure before choosing which counters to tap
    together, choose before evaluating a design that reads one. Derived from
    the requirements rather than recited -- asking only for the last one still
    yields the whole chain."""
    chain = evidence.chain_for(["predictor_design_result"])

    assert chain == [
        "sample_hardware_counters",
        "measure_predictability",
        "select_counter_taps",
        "evaluate_predictor_design",
    ]


def test_method_notes_reach_a_run_that_needs_them():
    """Every module carried a describe() saying what makes its numbers worth
    having, and the only caller of any describe() was the validity block -- so
    the notes belonging to modules implementing no validity predicate were
    written and never read."""
    notes = " ".join(evidence.method_notes(["predictability_ceiling"]))

    assert "beyond persistence" in notes


def test_method_notes_are_keyed_to_the_evidence_actually_required():
    """A run measuring a ceiling should not be lectured about held-out splits
    it is not doing; a prompt that says everything says nothing."""
    ceiling_only = evidence.method_notes(["predictability_ceiling"])
    with_design = evidence.method_notes(["predictor_design_result"])

    assert not any("contiguous" in line for line in ceiling_only)
    assert any("contiguous" in line for line in with_design)
    # And the design run still inherits the traps of the chain it must run
    # through to get there.
    assert any("beyond persistence" in line for line in with_design)


def test_evidence_unrelated_to_counters_draws_no_counter_notes():
    assert evidence.method_notes(["fusion_candidate"]) == []


def test_the_prompt_carries_the_traps_of_the_work_this_run_was_given():
    """The chain says which tool yields what. It does not say that a counter
    looks predictive until you ask what it adds over last-value, which is how
    the number goes wrong while every tool reports success."""
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    job = _Job(
        {
            "goal_contract": {
                "enabled": True,
                "required_finding_types": ["predictor_design_result"],
            }
        }
    )

    prompt = AutonomousAgentExecutor()._build_thinking_prompt_stable(job, None, {})

    assert "WHAT MAKES THIS EVIDENCE WORTH HAVING" in prompt
    assert "beyond persistence" in prompt
    assert "contiguous" in prompt
    # And the chain that gets there, derived from the one type asked for.
    assert "sample_hardware_counters" in prompt
    assert "select_counter_taps" in prompt


def test_a_contract_wanting_none_of_this_carries_none_of_it():
    """A prompt that says everything says nothing, and the stable half of the
    thinking prompt keys the provider's cache -- it must not grow for runs that
    are not doing this work."""
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    job = _Job(
        {
            "goal_contract": {
                "enabled": True,
                "required_finding_types": ["fusion_candidate"],
            }
        }
    )

    prompt = AutonomousAgentExecutor()._build_thinking_prompt_stable(job, None, {})

    assert "WHAT MAKES THIS EVIDENCE WORTH HAVING" not in prompt
