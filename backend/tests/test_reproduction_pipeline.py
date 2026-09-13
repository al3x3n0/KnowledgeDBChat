"""The pipeline this whole chain exists to make expressible.

Read a paper, implement the algorithm it describes, measure it, and say whether
the number reproduces the paper's claim. Every stage is a contract over
evidence, and the tools are derived from the contracts rather than named -- so
these tests are really asking whether the evidence types connect end to end. If
any link is missing the planner silently plans a stage with no way to satisfy
it, which is the failure this file is here to catch.
"""

import json

import pytest

from app.services import agent_pipeline_spec as ps

pytestmark = pytest.mark.unit


#: The pipeline as an author would write it in the studio.
REPRODUCE_PAPER = {
    "name": "reproduce-paper-algorithm",
    "stages": [
        {
            "id": "find",
            "goal": "Ingest the paper describing the algorithm to reproduce",
            "contract": {"required_finding_types": ["papers_ingested"]},
        },
        {
            "id": "specify",
            "goal": (
                "Read the paper into an implementable specification: the steps, "
                "its worked examples, and the numbers it claims -- written down "
                "before any code exists"
            ),
            "depends_on": ["find"],
            "assumes": ["papers_ingested"],
            "contract": {"required_finding_types": ["algorithm_spec"]},
        },
        {
            "id": "implement",
            "goal": (
                "Implement the algorithm in C or Rust and establish that it "
                "computes the paper's worked examples"
            ),
            "depends_on": ["specify"],
            "assumes": ["algorithm_spec"],
            # Writing an algorithm from a paper's prose does not work first
            # try. The loop is write, check, read the failing case, fix.
            "loop": {"max_iterations": 6, "until": "contract_satisfied"},
            "contract": {"required_finding_types": ["implementation_verified"]},
        },
        {
            "id": "measure",
            "goal": "Time the verified implementation",
            "depends_on": ["implement"],
            "assumes": ["implementation_verified"],
            "contract": {
                "required_finding_types": ["benchmark_measurement"],
                "validity": {"require_uncertainty": ["benchmark_measurement"]},
            },
        },
        {
            "id": "compare",
            "goal": (
                "Score the measurement against the paper's claim, or say which "
                "condition makes them incomparable"
            ),
            "depends_on": ["measure"],
            "assumes": ["benchmark_measurement", "implementation_verified"],
            "checkpoint": True,
            "contract": {"required_finding_types": ["reproduction_verdict"]},
        },
    ],
}


@pytest.fixture
def pipeline():
    return ps.normalize(REPRODUCE_PAPER)


class TestItIsAValidPipeline:
    def test_it_validates(self, pipeline):
        assert ps.validate(pipeline) == []

    def test_every_stage_can_reach_its_evidence(self, pipeline):
        # The real question. A contract requiring a finding type no tool
        # produces is a stage that cannot finish, and before these three tools
        # existed `algorithm_spec`, `implementation_verified` and
        # `reproduction_verdict` were exactly that.
        plan = ps.plan(pipeline)
        for stage in plan.stages:
            assert (
                stage.tools
            ), f"stage {stage.stage_id} has no way to produce its evidence"

    def test_the_stages_run_in_the_only_order_that_makes_sense(self, pipeline):
        assert ps.topological_order(pipeline) == [
            "find",
            "specify",
            "implement",
            "measure",
            "compare",
        ]


class TestTheChainIsDerivedNotRecited:
    def test_the_verdict_stage_pulls_in_its_prerequisites(self, pipeline):
        plan = ps.plan(pipeline)
        by_stage = {s.stage_id: s.tools for s in plan.stages}
        assert "compare_to_claim" in by_stage["compare"]

    def test_specifying_comes_from_the_paper_not_from_nowhere(self, pipeline):
        plan = ps.plan(pipeline)
        by_stage = {s.stage_id: s.tools for s in plan.stages}
        assert "extract_algorithm_spec" in by_stage["specify"]

    def test_the_implement_stage_checks_rather_than_assuming(self, pipeline):
        plan = ps.plan(pipeline)
        by_stage = {s.stage_id: s.tools for s in plan.stages}
        assert "check_implementation" in by_stage["implement"]

    def test_a_spec_is_not_re_extracted_downstream(self, pipeline):
        # The durable half of the incremental chain: the paper does not change,
        # so reading it twice is waste rather than rigour.
        plan = ps.plan(pipeline)
        by_stage = {s.stage_id: s.tools for s in plan.stages}
        assert "extract_algorithm_spec" not in by_stage["compare"]


class TestTheCheckIsRetakenNotInherited:
    """A correctness check that survives an edit certifies the wrong program."""

    def test_a_second_implement_round_rechecks(self):
        # Two implementation stages in sequence, the second editing what the
        # first produced. Without `perishable` on check_implementation the
        # second inherits the first's verdict and derives no tools at all --
        # it would report a verified implementation it never checked.
        spec = ps.normalize(
            {
                "name": "two-rounds",
                "stages": [
                    {
                        "id": "first",
                        "goal": "implement",
                        "contract": {
                            "required_finding_types": ["implementation_verified"]
                        },
                    },
                    {
                        "id": "second",
                        "goal": "optimise it",
                        "depends_on": ["first"],
                        "assumes": ["implementation_verified"],
                        "contract": {
                            "required_finding_types": ["implementation_verified"]
                        },
                    },
                ],
            }
        )
        assert ps.validate(spec) == []
        by_stage = {s.stage_id: s.tools for s in ps.plan(spec).stages}
        assert "check_implementation" in by_stage["second"]


class TestWhatTheAuthorSees:
    def test_it_describes_itself(self, pipeline):
        lines = "\n".join(ps.describe(pipeline))
        assert "reproduce-paper-algorithm" in lines

    def test_it_carries_a_cost(self, pipeline):
        plan = ps.plan(pipeline)
        assert plan.total_seconds > 0

    def test_it_is_the_json_an_author_would_paste(self):
        # Guards the studio starter against drifting from a spec that validates.
        reparsed = ps.normalize(json.loads(json.dumps(REPRODUCE_PAPER)))
        assert ps.validate(reparsed) == []
