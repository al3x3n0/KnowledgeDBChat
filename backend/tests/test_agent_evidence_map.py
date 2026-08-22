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
