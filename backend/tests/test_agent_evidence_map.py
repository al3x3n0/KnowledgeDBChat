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


# --- can this job afford its own contract ----------------------------------


GEM5_CONTRACT = ["counter_trace", "predictability_ceiling", "counter_tap_selection"]


def test_the_default_budget_cannot_afford_a_simulation_contract():
    """max_runtime_minutes defaults to 60 and one counter-sampling call has
    been observed at 105. Such a job does not fail -- it stops with its
    contract unmet, which reads as an agent that gave up rather than a budget
    that expired."""
    result = evidence.check_runtime_budget(GEM5_CONTRACT, 60, max_iterations=12)

    assert result["feasible"] is False
    assert "sample_hardware_counters" in result["message"]
    assert "Raise max_runtime_minutes" in result["message"]


def test_a_budget_in_the_right_range_passes():
    result = evidence.check_runtime_budget(GEM5_CONTRACT, 240, max_iterations=12)

    assert result["feasible"] is True


def test_a_contract_needing_no_simulator_fits_the_default():
    """The default is right for most work and must not be condemned for it."""
    result = evidence.check_runtime_budget(
        ["prediction_settled", "method_recorded"], 60, max_iterations=12
    )

    assert result["feasible"] is True


def test_an_expensive_tool_is_counted_more_than_once():
    """The live run called the sampler twice, at 38 and 105 minutes, and
    neither call was wasted -- an agent refines a workload after seeing its
    output. Counting one call each pronounced a budget that had just expired
    to be ample."""
    once = sum(
        row["floor_seconds"]
        for row in evidence.check_runtime_budget(GEM5_CONTRACT, 0)["breakdown"]
    )
    estimated = evidence.estimate_chain_seconds(GEM5_CONTRACT)

    assert estimated > once, "an expensive tool must be budgeted for a retry"


def test_no_budget_declared_is_not_a_shortfall():
    result = evidence.check_runtime_budget(GEM5_CONTRACT, 0)

    assert result["feasible"] is True


def test_the_estimate_is_presented_as_a_range_not_a_prediction():
    """It is arithmetic over order-of-magnitude figures, and a reader who
    treats it as planning data will be wrong by the size of their workload."""
    message = evidence.check_runtime_budget(GEM5_CONTRACT, 60, 12)["message"]

    assert "order-of-magnitude" in message


def test_a_job_that_cannot_afford_its_contract_is_told_at_launch():
    """Not after ninety minutes. The stable prompt is where it belongs: a run
    that knows its budget is short can spend it on the evidence that matters."""
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    job = _Job(
        {
            "goal_contract": {
                "enabled": True,
                "required_finding_types": ["counter_trace", "predictability_ceiling"],
            }
        }
    )
    job.max_runtime_minutes = 60
    job.max_iterations = 12

    prompt = AutonomousAgentExecutor()._build_thinking_prompt_stable(job, None, {})

    assert "BUDGET WARNING" in prompt
    assert "sample_hardware_counters" in prompt


def test_an_affordable_job_is_not_warned():
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    job = _Job(
        {
            "goal_contract": {
                "enabled": True,
                "required_finding_types": ["counter_trace", "predictability_ceiling"],
            }
        }
    )
    job.max_runtime_minutes = 480
    job.max_iterations = 12

    prompt = AutonomousAgentExecutor()._build_thinking_prompt_stable(job, None, {})

    assert "BUDGET WARNING" not in prompt


class TestSeveralRoutesToOneFact:
    """Two tools may yield the same evidence; neither may vanish because of it.

    `_BY_EVIDENCE` used to be a dict comprehension, so a second tool declaring
    an existing evidence type silently replaced the first as its producer.
    Nothing warned. Three pairs already existed -- a paper can be ingested by
    arXiv search or by id -- and in each case one route was invisible to
    planning: `producer_of` named one tool and `describe_chain` told the model
    about that one only.

    Found while adding Rust: a `benchmark_rust_snippet` producing
    `benchmark_measurement` would have quietly hijacked every pipeline that
    asks for a timing. That is why the language became a parameter of the
    existing benchmark tool rather than a second tool.
    """

    def test_every_route_is_reachable(self):
        from app.services import agent_evidence_map as em

        # Two ways to get a paper in, and the map knows both.
        producers = em.producers_of("papers_ingested")
        assert "ingest_arxiv_papers" in producers
        assert "ingest_paper_by_id" in producers

    def test_the_planned_tool_is_the_first_declared_not_the_last(self):
        # Either is arbitrary; a dict comprehension made it the last one
        # *accidentally*, so reordering two specs in a file would have changed
        # what every pipeline planned with nothing to notice it.
        from app.services import agent_evidence_map as em

        for finding_type in (
            "papers_ingested",
            "documents_ingested",
            "literature_review",
        ):
            assert em.producer_of(finding_type) == em.producers_of(finding_type)[0]

    def test_a_single_producer_still_reads_as_one(self):
        from app.services import agent_evidence_map as em

        assert em.producers_of("reproduction_verdict") == ["compare_to_claim"]
        assert em.producer_of("reproduction_verdict") == "compare_to_claim"

    def test_unknown_evidence_has_no_producers(self):
        from app.services import agent_evidence_map as em

        assert em.producers_of("nothing_produces_this") == []
        assert em.producer_of("nothing_produces_this") == ""

    def test_the_guidance_names_the_alternative(self):
        # Otherwise the second route is unreachable in practice: nothing the
        # model reads would ever mention it.
        from app.services import agent_evidence_map as em

        line = " ".join(em.describe_chain(["papers_ingested"]))
        assert "ingest_arxiv_papers" in line
        assert "ingest_paper_by_id" in line
