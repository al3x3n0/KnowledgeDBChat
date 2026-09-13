"""A stage must be allowed to call the tools its contract requires.

Tools may restrict themselves to certain job types, and the runtime hides the
rest. Nothing checked that against the tools a stage's contract implies, so a
stage could validate, plan, and start with none of the tools it needed.

Observed: a coding stage left at the default job type `research` was planned as
`clone_and_index_repo, apply_patch, run_repo_tests` -- the planner named all
three -- and then spent eight iterations calling search_documents and
save_research_finding, because every one of those tools is restricted to
`analysis` and `coding` and the agent could not see a single one of them.

The planner naming a tool the runtime forbids is the same failure as evidence
declared but never emitted: a statement about the system that nothing checked
against the system.
"""

import pytest

from app.services import agent_pipeline_spec as ps

pytestmark = pytest.mark.unit

CODING_CONTRACT = {
    "required_finding_types": ["repo_workspace", "patch_applied", "test_result"]
}


def _pipeline(job_type=None):
    stage = {"id": "fix", "goal": "make the tests pass", "contract": CODING_CONTRACT}
    if job_type is not None:
        stage["job_type"] = job_type
    return ps.normalize({"name": "p", "stages": [stage]})


class TestAStarvedStageIsRefused:
    def test_the_default_job_type_cannot_run_a_coding_stage(self):
        problems = ps.validate(_pipeline())
        assert problems, "a stage with none of its tools available validated"
        assert any("clone_and_index_repo" in p for p in problems)

    def test_the_message_names_the_fix(self):
        # "Something is wrong" would leave an author guessing; the job types
        # that would work are the whole answer.
        problems = ps.validate(_pipeline())
        joined = " ".join(problems)
        assert "job_type" in joined
        assert "coding" in joined and "analysis" in joined

    def test_every_blocked_tool_is_listed(self):
        # Fixing one and rediscovering the next is three round trips.
        problems = " ".join(ps.validate(_pipeline()))
        for tool in ("clone_and_index_repo", "apply_patch", "run_repo_tests"):
            assert tool in problems


class TestAWellTypedStagePasses:
    def test_coding(self):
        assert ps.validate(_pipeline("coding")) == []

    def test_analysis_is_also_allowed(self):
        assert ps.validate(_pipeline("analysis")) == []

    def test_a_research_stage_with_research_tools_is_fine(self):
        pipeline = ps.normalize(
            {
                "name": "p",
                "stages": [
                    {
                        "id": "read",
                        "goal": "read a paper",
                        "contract": {"required_finding_types": ["algorithm_spec"]},
                    }
                ],
            }
        )
        assert ps.validate(pipeline) == []

    def test_an_unrestricted_tool_never_blocks_a_stage(self):
        # Most tools name no job types at all and must stay available to every
        # stage; treating "unrestricted" as "restricted to nothing" would
        # refuse almost every pipeline.
        pipeline = ps.normalize(
            {
                "name": "p",
                "stages": [
                    {
                        "id": "measure",
                        "goal": "benchmark",
                        "contract": {
                            "required_finding_types": ["benchmark_measurement"]
                        },
                    }
                ],
            }
        )
        assert ps.validate(pipeline) == []
