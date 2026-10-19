"""Knowing which tool yields which evidence, and what must come first.

A run given a goal and no method chose its tools well and then got the order
and the hand-offs wrong: it mined before profiling, and spent three attempts
passing raw assembly where a mined pattern belongs. The chain was right; the
knowledge about the tools was missing.

Core's own tools are used here. The chains that first needed this (profile,
mine, cost; sample, measure, select, evaluate) belong to the uarch_lab research
program, which tests them through this same machinery.
"""

from __future__ import annotations


from app.services import agent_evidence_map as evidence


def test_evidence_names_the_tool_that_produces_it():
    assert evidence.producer_of("algorithm_spec") == "extract_algorithm_spec"
    assert evidence.producer_of("implementation_verified") == "check_implementation"
    assert evidence.producer_of("prediction_settled") == "record_measurement"


def test_a_chain_puts_prerequisites_first():
    """Checking an implementation before knowing what it implements is the
    shape of mistake this exists to prevent."""
    chain = evidence.chain_for(["implementation_verified"])

    assert chain.index("extract_algorithm_spec") < chain.index("check_implementation")


def test_prerequisites_are_pulled_in_even_when_not_asked_for():
    """A contract wants a settled prediction; nothing settles what was never
    recorded."""
    chain = evidence.chain_for(["prediction_settled"])

    assert chain == ["record_prediction", "record_measurement"]


def test_a_transitive_prerequisite_comes_first():
    chain = evidence.chain_for(["reproduction_verdict"])

    assert chain.index("extract_algorithm_spec") < chain.index("check_implementation")
    assert chain.index("check_implementation") < chain.index("compare_to_claim")


def test_a_tool_is_named_once_however_many_things_need_it():
    chain = evidence.chain_for(
        ["implementation_verified", "reproduction_verdict", "algorithm_spec"]
    )

    assert chain.count("extract_algorithm_spec") == 1


def test_evidence_nothing_can_produce_is_reported():
    """A contract asking for the unobtainable cannot be satisfied however well
    a run behaves, and that is the contract's fault."""
    assert evidence.unobtainable(["implementation_verified", "vibes"]) == ["vibes"]


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
                "required_finding_types": [
                    "implementation_verified",
                    "prediction_settled",
                ],
            }
        }
    )

    prompt = AutonomousAgentExecutor()._build_thinking_prompt_stable(job, None, {})

    assert "HOW THIS RUN'S REQUIRED EVIDENCE IS PRODUCED" in prompt
    assert "extract_algorithm_spec" in prompt
    assert "check_implementation" in prompt
    # It must not read as the method itself: choosing what to measure is the
    # run's job, and this only says where evidence comes from.
    assert "not the whole method" in prompt


def test_a_contract_with_no_named_evidence_adds_nothing():
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    job = _Job({"goal_contract": {"enabled": True, "min_findings": 2}})

    prompt = AutonomousAgentExecutor()._build_thinking_prompt_stable(job, None, {})

    assert "HOW THIS RUN'S REQUIRED EVIDENCE" not in prompt


def test_a_validity_rule_contributes_to_the_chain():
    """`predictions_measured` is satisfied by record_measurement, which the
    counting requirements never name. A run given no method got everything
    else right and left its prediction unsettled."""
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    job = _Job(
        {
            "goal_contract": {
                "enabled": True,
                "required_finding_types": ["implementation_verified"],
                "validity": {"predictions_measured": True, "records_method": True},
            }
        }
    )

    prompt = AutonomousAgentExecutor()._build_thinking_prompt_stable(job, None, {})
    chain = prompt.split("HOW THIS RUN")[1].split("You are not required")[0]

    assert "record_measurement" in chain
    assert "record_method" in chain
    assert chain.index("record_prediction") < chain.index("record_measurement")


# --- method notes ------------------------------------------------------------


def test_method_notes_reach_a_run_that_needs_them():
    """Every module carried a describe() saying what makes its numbers worth
    having, and the only caller of any describe() was the validity block -- so
    the notes belonging to modules implementing no validity predicate were
    written and never read."""
    from app.services import agent_implementation_check

    notes = evidence.method_notes(["implementation_verified"])

    assert notes
    assert set(agent_implementation_check.describe()) <= set(notes)


def test_evidence_with_no_method_source_draws_no_notes():
    assert evidence.method_notes(["algorithm_spec"]) == []


def test_a_contract_wanting_none_of_this_carries_none_of_it():
    """A prompt that says everything says nothing, and the stable half of the
    thinking prompt keys the provider's cache -- it must not grow for runs that
    are not doing this work."""
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    job = _Job(
        {
            "goal_contract": {
                "enabled": True,
                "required_finding_types": ["algorithm_spec"],
            }
        }
    )

    prompt = AutonomousAgentExecutor()._build_thinking_prompt_stable(job, None, {})

    assert "WHAT MAKES THIS EVIDENCE WORTH HAVING" not in prompt


# --- can this job afford its own contract ----------------------------------


#: The slowest evidence core's own tools produce, about nine minutes of tool
#: time. The program's simulator contracts are the case this was written for
#: and are tested there; this is the same arithmetic on core's figures.
SLOW_CONTRACT = ["literature_review", "research_presentation", "synthesis_document"]


def test_a_budget_too_small_for_the_contract_is_refused():
    """Such a job does not fail -- it stops with its contract unmet, which
    reads as an agent that gave up rather than a budget that expired."""
    result = evidence.check_runtime_budget(SLOW_CONTRACT, 5, max_iterations=12)

    assert result["feasible"] is False
    assert "literature_review_arxiv" in result["message"]
    assert "Raise max_runtime_minutes" in result["message"]


def test_a_budget_in_the_right_range_passes():
    result = evidence.check_runtime_budget(SLOW_CONTRACT, 240, max_iterations=12)

    assert result["feasible"] is True


def test_a_contract_needing_no_simulator_fits_the_default():
    """The default is right for most work and must not be condemned for it."""
    result = evidence.check_runtime_budget(
        ["prediction_settled", "method_recorded"], 60, max_iterations=12
    )

    assert result["feasible"] is True


def test_no_budget_declared_is_not_a_shortfall():
    result = evidence.check_runtime_budget(SLOW_CONTRACT, 0)

    assert result["feasible"] is True


def test_the_estimate_is_presented_as_a_range_not_a_prediction():
    """It is arithmetic over order-of-magnitude figures, and a reader who
    treats it as planning data will be wrong by the size of their workload."""
    message = evidence.check_runtime_budget(SLOW_CONTRACT, 5, 12)["message"]

    assert "order-of-magnitude" in message


def test_a_job_that_cannot_afford_its_contract_is_told_at_launch():
    """Not after ninety minutes. The stable prompt is where it belongs: a run
    that knows its budget is short can spend it on the evidence that matters."""
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    job = _Job(
        {
            "goal_contract": {
                "enabled": True,
                "required_finding_types": SLOW_CONTRACT,
            }
        }
    )
    job.max_runtime_minutes = 5
    job.max_iterations = 12

    prompt = AutonomousAgentExecutor()._build_thinking_prompt_stable(job, None, {})

    assert "BUDGET WARNING" in prompt
    assert "literature_review_arxiv" in prompt


def test_an_affordable_job_is_not_warned():
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    job = _Job(
        {
            "goal_contract": {
                "enabled": True,
                "required_finding_types": SLOW_CONTRACT,
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
