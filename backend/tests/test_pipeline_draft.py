"""Drafting a pipeline from a sentence, and why the checker judges it.

The value of this path is not that a model can emit JSON. It is that whatever
comes back is put through the same checks a hand-typed pipeline faces, and the
complaints are fed back once. So the tests here are about the judging, not the
generating: the model is a stub, and what is asserted is what happens to what
it returns.

The case that motivated `tidy` is worth stating plainly, because it is the
failure mode this whole design is supposed to prevent and it still got through:
a real draft scored ZERO problems while every one of its five stages had an
empty goal, because the model had put each goal inside `contract.goal` where
nothing reads it. A checker that passes that is measuring shape, not content.
"""

import pytest

from app.services import agent_pipeline_draft as draft
from app.services import agent_pipeline_spec

pytestmark = pytest.mark.unit


def _stage(stage_id, **over):
    stage = {
        "id": stage_id,
        "goal": f"do {stage_id}",
        "contract": {"required_finding_types": ["papers_ingested"]},
    }
    stage.update(over)
    return stage


class TestTheGoalIsNotOptional:
    """The contract says when a stage may stop; the goal says what it is for,
    and the goal is what the running stage is actually given."""

    def test_a_stage_with_no_goal_is_refused(self):
        pipeline = agent_pipeline_spec.normalize(
            {"name": "p", "stages": [_stage("gather", goal="")]}
        )
        problems = agent_pipeline_spec.validate(pipeline)

        assert any("has no goal" in p for p in problems)

    def test_a_stage_with_a_goal_is_not(self):
        pipeline = agent_pipeline_spec.normalize(
            {"name": "p", "stages": [_stage("gather")]}
        )

        assert not [p for p in agent_pipeline_spec.validate(pipeline) if "goal" in p]


class TestTidyFixesWhatAModelMisplacesRatherThanMisjudges:
    def test_a_goal_written_into_the_contract_is_hoisted(self):
        """Everything else about a stage lives in the contract, so that is
        where the goal gets put -- and nothing reads it there."""
        spec = {
            "stages": [
                {
                    "id": "gather",
                    "contract": {
                        "goal": "Collect the papers",
                        "required_finding_types": ["papers_ingested"],
                    },
                }
            ]
        }

        fixed = draft.tidy(spec, "collect some papers")

        assert fixed["stages"][0]["goal"] == "Collect the papers"
        assert (
            "goal" not in fixed["stages"][0]["contract"]
        ), "a second copy nothing reads is a second copy to maintain"

    def test_a_real_goal_is_not_overwritten_by_a_misplaced_one(self):
        spec = {
            "stages": [
                {
                    "id": "gather",
                    "goal": "The real one",
                    "contract": {"goal": "The stray one", "required_finding_types": []},
                }
            ]
        }

        assert draft.tidy(spec, "x")["stages"][0]["goal"] == "The real one"

    def test_a_nameless_draft_gets_a_name_from_the_request(self):
        """Harmless to the checker, which is exactly why nothing catches it.
        It reaches the library as "None"."""
        fixed = draft.tidy({"stages": []}, "Reproduce the FlashAttention speedup")

        assert fixed["name"] == "reproduce-the-flashattention-speedup"

    def test_a_description_with_nothing_usable_still_yields_a_name(self):
        assert draft.tidy({"stages": []}, "!!! ???")["name"] == "drafted-pipeline"


class TestTheCheckerIsTheJudge:
    def test_an_invented_finding_type_is_a_problem(self):
        """The most common way an authored pipeline fails, and the one a model
        commits most readily."""
        problems = draft.problems_with(
            {
                "name": "p",
                "stages": [
                    _stage(
                        "gather",
                        contract={"required_finding_types": ["vibes_measured"]},
                    )
                ],
            }
        )

        assert any("vibes_measured" in p for p in problems)

    def test_a_spec_that_is_not_a_pipeline_at_all_is_reported_not_raised(self):
        """A drafter that raises here leaves the author with nothing; the
        editor is where a pipeline gets fixed."""
        problems = draft.problems_with({"stages": "not a list"})

        assert problems, "it must say something rather than pass silently"


class _StubLLM:
    """Returns queued payloads, and records what it was asked."""

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.prompts = []

    async def generate_structured(self, **kwargs):
        self.prompts.append(kwargs)

        class _Completion:
            structured = self.payloads.pop(0)
            text = ""

        return _Completion()


@pytest.mark.asyncio
class TestTheRepairRound:
    async def test_a_clean_draft_costs_one_call(self):
        good = {"name": "p", "stages": [_stage("gather")]}
        llm = _StubLLM(good)

        spec, problems, repaired = await draft.draft_pipeline(
            description="collect papers", llm_service=llm
        )

        assert problems == []
        assert repaired is False
        assert len(llm.prompts) == 1

    async def test_a_broken_draft_is_shown_its_own_problems(self):
        broken = {
            "name": "p",
            "stages": [
                _stage(
                    "gather", contract={"required_finding_types": ["vibes_measured"]}
                )
            ],
        }
        fixed = {"name": "p", "stages": [_stage("gather")]}
        llm = _StubLLM(broken, fixed)

        spec, problems, repaired = await draft.draft_pipeline(
            description="collect papers", llm_service=llm
        )

        assert repaired is True
        assert problems == []
        # The repair prompt has to contain the complaint. A model told to be
        # careful in advance does not fix this; a model told what is wrong does.
        assert "vibes_measured" in llm.prompts[1]["user_message"]

    async def test_a_repair_that_made_it_worse_is_discarded(self):
        """Measured once as a model 'fixing' an unknown finding type by
        inventing two more."""
        broken = {
            "name": "p",
            "stages": [
                _stage(
                    "gather", contract={"required_finding_types": ["vibes_measured"]}
                )
            ],
        }
        worse = {
            "name": "p",
            "stages": [
                _stage(
                    "gather",
                    goal="",
                    contract={"required_finding_types": ["vibes", "more_vibes"]},
                )
            ],
        }
        llm = _StubLLM(broken, worse)

        spec, problems, repaired = await draft.draft_pipeline(
            description="collect papers", llm_service=llm
        )

        assert repaired is False
        assert spec["stages"][0]["contract"]["required_finding_types"] == [
            "vibes_measured"
        ], "the less broken of the two is what the author gets"

    async def test_the_vocabulary_reaches_the_prompt(self):
        """The whole reason a drafted contract can be satisfied at all: the
        model may only choose from types a tool actually produces."""
        llm = _StubLLM({"name": "p", "stages": [_stage("gather")]})

        await draft.draft_pipeline(description="collect papers", llm_service=llm)

        system = llm.prompts[0]["system_prompt"]
        assert "papers_ingested" in system
        assert "vibes_measured" not in system

    async def test_an_empty_description_is_refused_before_any_call(self):
        llm = _StubLLM()

        with pytest.raises(draft.PipelineDraftError):
            await draft.draft_pipeline(description="   ", llm_service=llm)

        assert llm.prompts == []


class TestTheGoalRuleDoesNotBreakAGate:
    """A stage that only waits for a person does no work, so there is nothing
    to instruct it to do. The same carve-out that lets it demand nothing.

    Caught by the existing suite when the goal rule was added without it: the
    rule is about stages that RUN, and a gate is the one stage that does not."""

    def test_a_pure_checkpoint_gate_needs_no_goal(self):
        pipeline = agent_pipeline_spec.normalize(
            {"stages": [{"id": "gate", "checkpoint": True}]}
        )

        assert agent_pipeline_spec.validate(pipeline) == []

    def test_a_checkpoint_that_also_does_work_still_needs_one(self):
        """Waiting for a person does not excuse a stage from saying what the
        work before the wait was for."""
        pipeline = agent_pipeline_spec.normalize(
            {
                "stages": [
                    {
                        "id": "writeup",
                        "checkpoint": True,
                        "contract": {"required_finding_types": ["papers_ingested"]},
                    }
                ]
            }
        )

        assert any("has no goal" in p for p in agent_pipeline_spec.validate(pipeline))


class TestADraftIsUntrustedInput:
    """`tidy` is the first thing to touch what the model returned, and it was
    written against its own well-formed fixtures.

    Measured on a live call: the model returned `contract` as a LIST, and
    `dict(stage.get("contract") or {})` raised `ValueError: dictionary update
    sequence element #0 has length 1; 2 is required` -- turning a recoverable
    draft into a 500. Everything here is a shape a model actually can return.
    """

    def test_a_contract_that_is_a_list_does_not_raise(self):
        spec = {
            "name": "p",
            "stages": [
                {"id": "gather", "goal": "collect", "contract": ["papers_ingested"]}
            ],
        }

        fixed = draft.tidy(spec, "collect papers")

        # Left exactly as it came: the checker names it and the repair round
        # gets a chance. Guessing here would hide the mistake from the model.
        assert fixed["stages"][0]["contract"] == ["papers_ingested"]

    def test_the_checker_then_reports_it_rather_than_crashing(self):
        problems = draft.problems_with(
            {
                "name": "p",
                "stages": [
                    {"id": "gather", "goal": "collect", "contract": ["papers_ingested"]}
                ],
            }
        )

        assert problems, "a malformed contract has to be said out loud"

    def test_a_contract_that_is_a_string(self):
        spec = {
            "name": "p",
            "stages": [{"id": "gather", "goal": "g", "contract": "papers_ingested"}],
        }

        assert draft.tidy(spec, "x")["stages"][0]["contract"] == "papers_ingested"

    def test_stages_that_are_not_a_list(self):
        assert draft.tidy({"name": "p", "stages": "gather then read"}, "x")

    def test_a_stage_that_is_not_an_object(self):
        fixed = draft.tidy({"name": "p", "stages": ["gather", {"id": "a"}]}, "x")

        assert [s["id"] for s in fixed["stages"]] == ["a"]

    def test_a_stage_with_no_contract_key_at_all(self):
        fixed = draft.tidy({"name": "p", "stages": [{"id": "a", "goal": "g"}]}, "x")

        assert "contract" not in fixed["stages"][0], (
            "inventing an empty contract would make a stage nothing can fail "
            "look deliberate"
        )


class TestImpliedDependencies:
    """A stage that waits for a stage and for that stage's predecessor.

    Found by sending real requests to a real model: the extra edge was the
    only thing wrong with two of four drafts, and the binding read it as a
    fan-in between stages that are not siblings.
    """

    @staticmethod
    def _spec():
        return {
            "name": "fastmod",
            "stages": [
                {
                    "id": "extract",
                    "goal": "Ingest the paper",
                    "contract": {"required_finding_types": ["papers_ingested"]},
                },
                {
                    "id": "specify",
                    "goal": "Write the algorithm down",
                    "depends_on": ["extract"],
                    "contract": {"required_finding_types": ["algorithm_spec"]},
                },
                {
                    "id": "review",
                    "goal": "Review the literature around it",
                    "depends_on": ["extract", "specify"],
                    "contract": {"required_finding_types": ["related_paper_set"]},
                },
            ],
        }

    def test_the_draft_as_the_model_wrote_it_is_refused(self):
        problems = draft.problems_with(self._spec())

        assert any("Fan-in" in problem for problem in problems), problems

    def test_the_implied_edge_is_dropped_and_the_draft_passes(self):
        fixed = draft.tidy(self._spec(), "x")

        assert fixed["stages"][2]["depends_on"] == ["specify"]
        assert draft.problems_with(fixed) == []

    def test_a_real_fan_in_is_left_alone(self):
        spec = self._spec()
        spec["stages"][2]["depends_on"] = ["extract"]
        spec["stages"].append(
            {
                "id": "join",
                "goal": "Bring both together",
                "depends_on": ["specify", "review"],
                "contract": {"required_finding_types": ["literature_review"]},
            }
        )

        assert draft.tidy(spec, "x")["stages"][3]["depends_on"] == [
            "specify",
            "review",
        ]

    def test_a_cycle_is_not_hidden(self):
        spec = self._spec()
        spec["stages"][0]["depends_on"] = ["specify"]

        fixed = draft.tidy(spec, "x")

        assert fixed["stages"][2]["depends_on"] == ["extract", "specify"]

    def test_a_malformed_dependency_list_is_left_for_the_checker(self):
        spec = self._spec()
        spec["stages"][2]["depends_on"] = "extract, specify"

        assert draft.tidy(spec, "x")["stages"][2]["depends_on"] == "extract, specify"


class TestStageFieldsWrittenIntoTheContract:
    """`job_type` and `assumes` inside `contract`, where nothing reads them."""

    @staticmethod
    def _spec():
        return {
            "name": "fix",
            "stages": [
                {
                    "id": "clone",
                    "goal": "Have the repository",
                    "contract": {
                        "required_finding_types": ["repo_workspace"],
                        "job_type": "coding",
                        "assumes": [],
                    },
                },
                {
                    "id": "patch",
                    "goal": "Propose the fix",
                    "depends_on": ["clone"],
                    "contract": {
                        "required_finding_types": ["code_patch_proposal"],
                        "job_type": "coding",
                        "assumes": ["repo_workspace"],
                    },
                },
            ],
        }

    def test_as_written_the_stage_is_planned_as_research_and_refused(self):
        problems = draft.problems_with(self._spec())

        assert any("job_type" in p or "job type" in p for p in problems), problems

    def test_they_are_moved_to_the_stage(self):
        fixed = draft.tidy(self._spec(), "x")

        patch_stage = fixed["stages"][1]
        assert patch_stage["job_type"] == "coding"
        assert patch_stage["assumes"] == ["repo_workspace"]
        assert patch_stage["contract"] == {
            "required_finding_types": ["code_patch_proposal"]
        }
        assert draft.problems_with(fixed) == []

    def test_what_the_stage_says_itself_wins(self):
        spec = self._spec()
        spec["stages"][1]["job_type"] = "analysis"

        assert draft.tidy(spec, "x")["stages"][1]["job_type"] == "analysis"


class TestTheEvidenceListSaysWhatEachTypeIs:
    """A bare name was all the drafter had to choose by.

    Asked to implement fastmod in C, check it and benchmark it, it required
    `code_patch_proposal`, which means changing a repository; the stage
    cloned one that already had the function and had nothing to change.
    """

    def test_each_type_names_the_tool_that_makes_it_in_that_tools_words(self):
        from app.services.agent_pipeline_draft import _vocabulary_block

        block = _vocabulary_block()
        (line,) = [
            entry
            for entry in block.splitlines()
            if entry.strip().startswith("implementation_verified ")
        ]
        assert "made by check_implementation: Run an implementation against" in line

    def test_the_repository_kind_reads_as_a_repository_change(self):
        from app.services.agent_pipeline_draft import _vocabulary_block

        (line,) = [
            entry
            for entry in _vocabulary_block().splitlines()
            if entry.strip().startswith("code_patch_proposal ")
        ]
        assert "workspace changes" in line

    def test_a_long_description_is_cut_to_one_short_sentence(self):
        from types import SimpleNamespace

        from app.services import agent_pipeline_draft

        entry = SimpleNamespace(producers=("benchmark_c_snippet",))
        made_by = agent_pipeline_draft._what_makes(entry, most=60)

        assert made_by.startswith("made by benchmark_c_snippet: ")
        assert len(made_by) <= len("made by benchmark_c_snippet: ") + 60
        assert made_by.endswith("...")

    def test_evidence_nothing_produces_says_nothing(self):
        from types import SimpleNamespace

        from app.services import agent_pipeline_draft

        assert agent_pipeline_draft._what_makes(SimpleNamespace(producers=())) == ""


class TestAStageShapedLikeAStageWithNoContract:
    """Live drafts of an ordinary repository fix (2026-10-14): `name` for
    `id`, the evidence beside the goal, and `loop` as a yes or no."""

    @staticmethod
    def _spec():
        return {
            "name": "fix",
            "stages": [
                {
                    "name": "clone",
                    "goal": "Have the repository",
                    "required_finding_types": ["repo_workspace"],
                    "job_type": "coding",
                    "loop": False,
                    "max_iterations": 1,
                },
                {
                    "name": "patch",
                    "goal": "Propose the fix",
                    "depends_on": ["clone"],
                    "assumes": ["repo_workspace"],
                    "required_finding_types": ["code_patch_proposal"],
                    "job_type": "coding",
                    "loop": True,
                    "max_iterations": 2,
                },
            ],
        }

    def test_as_written_it_is_refused(self):
        problems = draft.problems_with(self._spec())

        assert any("requires no finding types" in p for p in problems), problems

    def test_tidied_it_passes_and_says_the_same_thing(self):
        fixed = draft.tidy(self._spec(), "x")

        clone, patch = fixed["stages"]
        assert [clone["id"], patch["id"]] == ["clone", "patch"]
        assert patch["contract"] == {"required_finding_types": ["code_patch_proposal"]}
        assert "required_finding_types" not in patch
        assert "loop" not in clone
        assert patch["loop"] == {"max_iterations": 2}
        assert draft.problems_with(fixed) == []

    def test_a_contract_that_already_names_evidence_is_not_overwritten(self):
        spec = self._spec()
        spec["stages"][1]["contract"] = {
            "required_finding_types": ["test_result"],
        }

        patch = draft.tidy(spec, "x")["stages"][1]

        assert patch["contract"]["required_finding_types"] == ["test_result"]
        # Left where it was, for the checker to see, not silently dropped.
        assert patch["required_finding_types"] == ["code_patch_proposal"]

    def test_an_id_is_never_replaced_by_a_name(self):
        spec = self._spec()
        spec["stages"][0]["id"] = "clone_repo"
        spec["stages"][1]["depends_on"] = ["clone_repo"]

        assert draft.tidy(spec, "x")["stages"][0]["id"] == "clone_repo"

    def test_a_loop_with_no_usable_bound_is_left_for_the_checker(self):
        spec = self._spec()
        spec["stages"][1]["max_iterations"] = "two"

        assert draft.tidy(spec, "x")["stages"][1]["loop"] is True


class TestCodeThatBelongsToNoRepository:
    def test_the_drafter_is_told_where_writing_such_code_belongs(self):
        prompt = " ".join(draft.SYSTEM_PROMPT.split())

        assert "Code that belongs to no repository" in prompt
        assert "not a stage of its own" in prompt
        for name in ("implementation_verified", "code_patch_proposal"):
            assert name in prompt
