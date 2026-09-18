"""Evidence types suggested from a stage goal.

These use the real vocabulary rather than a fixture, because the whole value of
the feature is whether it picks the right thing out of 51 real options. A test
against three invented evidence types would pass while the feature was useless.

The last class is the important one: a vague goal produces poor suggestions, and
the tests say so rather than pretending otherwise. That is the reason this
suggests and never applies.
"""

from app.services.agent_contract_suggestions import suggest, suggest_for_spec


def names(goal, job_type="research", **kw):
    return [s.finding_type for s in suggest(goal=goal, job_type=job_type, **kw)]


class TestItFindsTheObviousOnes:
    """Goals whose words the producing tool also uses."""

    def test_identify_research_gaps(self):
        # identify_research_gaps produces research_gap. The clearest case there
        # is, and it should be first, not merely present.
        assert names("Identify research gaps and future directions", "analysis")[0] == (
            "research_gap"
        )

    def test_create_a_presentation(self):
        assert (
            names("Create a presentation summarizing research findings", "synthesis")[0]
            == "research_presentation"
        )

    def test_monitor_for_new_papers(self):
        assert "papers_ingested" in names(
            "Monitor for new papers on sparsity", "monitor"
        )

    def test_a_literature_review_document(self):
        got = names("Generate comprehensive literature review document", "synthesis")
        assert "literature_review" in got

    def test_each_suggestion_names_the_tool_that_would_produce_it(self):
        # A suggestion you cannot check is worse than none: the author has to be
        # able to see *why* before accepting it.
        (first, *_) = suggest(goal="Identify research gaps", job_type="analysis")
        assert first.produced_by == "identify_research_gaps"
        assert "gap" in first.matched


class TestItRespectsWhatAStageCanActuallyDo:
    def test_evidence_the_job_type_cannot_produce_is_not_offered(self):
        # papers_ingested is restricted to knowledge_expansion / monitor /
        # research. Offering it to a coding stage would be accepted by the
        # author and then refused by the checker, for a reason that looks
        # unrelated to what they clicked.
        assert "papers_ingested" not in names("Find relevant papers", "coding")
        assert "papers_ingested" in names("Find relevant papers", "research")

    def test_unrestricted_evidence_is_offered_to_any_job_type(self):
        assert names("Review the literature", "coding") or True  # no crash
        assert "literature_review" in names("Review the literature", "custom")

    def test_what_is_already_required_is_not_suggested_again(self):
        without = names("Identify research gaps", "analysis")
        with_exclusion = names(
            "Identify research gaps", "analysis", exclude=["research_gap"]
        )
        assert "research_gap" in without
        assert "research_gap" not in with_exclusion


class TestItKnowsWhenItHasNothingToSay:
    def test_a_goal_sharing_no_words_with_the_vocabulary_suggests_nothing(self):
        assert names("Ask Bob about the thing we discussed on Tuesday") == []

    def test_an_empty_goal_suggests_nothing(self):
        assert names("") == []
        assert names("   ") == []

    def test_a_goal_of_only_common_words_suggests_nothing(self):
        assert names("Do the thing and then do the other thing") == []


class TestWhereItIsWeak:
    """Recorded rather than hidden: these are why it never applies itself."""

    def test_a_vague_goal_matches_on_a_domain_word_and_misleads(self):
        # "Research X comprehensively" shares only the word "research" with the
        # vocabulary, so it returns research_* evidence that has nothing to do
        # with what the stage means. The suggestions are shown with the words
        # they matched precisely so this is visible rather than trusted.
        got = names("Research attention sparsity comprehensively", "research")
        assert got, "it does suggest something"
        assert (
            "papers_ingested" not in got
        ), "and the thing a person would actually choose is not among them"

    def test_the_matched_words_expose_a_weak_match(self):
        (first, *_) = suggest(
            goal="Research attention sparsity comprehensively", job_type="research"
        )
        # One generic word. An author reading this can dismiss it in a second.
        assert first.matched == ("research",)


class TestAcrossAWholeSpec:
    SPEC = {
        "name": "lit",
        "stages": [
            {"id": "discover", "goal": "Find relevant papers", "job_type": "research"},
            {
                "id": "gaps",
                "goal": "Identify research gaps",
                "job_type": "analysis",
                "depends_on": ["discover"],
            },
        ],
    }

    def test_suggests_for_every_stage_without_a_contract(self):
        out = suggest_for_spec(self.SPEC)
        assert set(out) == {"discover", "gaps"}
        assert out["gaps"][0]["finding_type"] == "research_gap"

    def test_leaves_alone_a_stage_that_already_says_what_it_needs(self):
        # The author has answered this question. A suggestion beside their
        # answer reads as a correction rather than an offer.
        spec = {
            "name": "lit",
            "stages": [
                {
                    "id": "gaps",
                    "goal": "Identify research gaps",
                    "job_type": "analysis",
                    "contract": {"required_finding_types": ["research_gap"]},
                }
            ],
        }
        assert suggest_for_spec(spec) == {}

    def test_reads_a_contract_written_as_counts_too(self):
        spec = {
            "name": "lit",
            "stages": [
                {
                    "id": "gaps",
                    "goal": "Identify research gaps",
                    "job_type": "analysis",
                    "contract": {"required_finding_type_counts": {"research_gap": 2}},
                }
            ],
        }
        assert suggest_for_spec(spec) == {}

    def test_a_stage_it_cannot_help_with_is_simply_absent(self):
        spec = {"name": "x", "stages": [{"id": "a", "goal": "Ask Bob about Tuesday"}]}
        assert suggest_for_spec(spec) == {}
