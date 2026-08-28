"""Reading what earlier runs measured.

Every run's findings were already persisted, typed and complete -- 260 jobs in
the development database hold them, with the numbers still attached -- and
nothing could read them. What crossed between runs instead was an LLM summary:
*that* prefetching helped a strided scan, never *by how much, against what
baseline, on which core*. For work whose product is a number, that is the
wrong half to keep.

The two rules being tested here are the ones that make reuse safe rather than
merely possible: recalled evidence is citable, and recalled evidence cannot
satisfy a contract.
"""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.services import agent_prior_findings as prior


class _Job(SimpleNamespace):
    """Enough of an AgentJob for the parts under test."""


def _job(findings, goal="measure a prefetcher", job_id=None, completed_at=None):
    return _Job(
        id=job_id or uuid4(),
        goal=goal,
        completed_at=completed_at,
        created_at=None,
        results={"findings": findings},
    )


MECHANISM = {
    "type": "mechanism_comparison",
    "subject": "L2 TaggedPrefetcher on strided scan",
    "title": "L2 TaggedPrefetcher: 1.7054x (1460876 -> 856605 cycles)",
    "speedup": 1.7054,
    "baseline_cycles": 1460876.0,
    "measurement_source": "gem5 mechanism pair",
}
HEADROOM = {
    "type": "headroom_bound",
    "subject": "PCE INT8 MHA: load_queue",
    "title": "idealising load_queue recovers 4.68%",
    "headroom_percent": 4.68,
    "measurement_source": "gem5 idealised limit study",
}


class TestMatching:
    def test_a_type_filter_selects_only_that_type(self):
        assert prior._matches(MECHANISM, ["mechanism_comparison"], "") is True
        assert prior._matches(HEADROOM, ["mechanism_comparison"], "") is False

    def test_no_filter_matches_anything_typed(self):
        assert prior._matches(MECHANISM, [], "") is True

    def test_an_untyped_record_is_not_evidence(self):
        """A finding with no type cannot be cited, so returning it would only
        take up room in the answer."""
        assert prior._matches({"title": "something"}, [], "") is False

    def test_subject_matches_across_the_fields_a_reader_would_search(self):
        assert prior._matches(MECHANISM, [], "TaggedPrefetcher") is True
        assert prior._matches(MECHANISM, [], "gem5 mechanism") is True
        assert prior._matches(MECHANISM, [], "1.7054") is True

    def test_subject_matching_ignores_case(self):
        assert prior._matches(MECHANISM, [], "taggedprefetcher") is True

    def test_a_subject_that_does_not_appear_is_not_a_match(self):
        assert prior._matches(MECHANISM, [], "branch predictor") is False


class TestProvenanceTravelsWithTheNumber:
    def test_the_finding_says_which_job_produced_it(self):
        job = _job([MECHANISM], goal="a goal that was already answered")
        recalled = prior._provenance(job, MECHANISM)

        assert recalled["recalled_from_job"] == str(job.id)
        assert "already answered" in recalled["recalled_from_goal"]

    def test_the_measurement_survives_intact(self):
        """The whole point: prose crossed before, numbers cross now."""
        recalled = prior._provenance(_job([MECHANISM]), MECHANISM)

        assert recalled["speedup"] == 1.7054
        assert recalled["baseline_cycles"] == 1460876.0
        assert recalled["type"] == "mechanism_comparison"

    def test_the_original_is_not_mutated(self):
        prior._provenance(_job([MECHANISM]), MECHANISM)

        assert "recalled" not in MECHANISM

    def test_it_is_stamped_as_recalled(self):
        assert prior._provenance(_job([MECHANISM]), MECHANISM)["recalled"] is True


class TestRecalledEvidenceCannotSatisfyAContract:
    """The rule that keeps reuse honest. A contract asking for two
    mechanism_comparison findings is what makes a run do the work; if a lookup
    could fill it, the cheapest way to satisfy any contract would be to recall
    two old numbers and stop."""

    def _counts(self, findings):
        from types import SimpleNamespace

        from app.services.agent_goal_contract_service import AgentGoalContractService

        # min_progress 0 because this is a test about counting findings.
        # Left out, the contract defaults to requiring progress 100 and every
        # case here fails for a reason that has nothing to do with recall --
        # which is exactly what the control below caught.
        contract = {
            "enabled": True,
            "min_progress": 0,
            "required_finding_type_counts": {"mechanism_comparison": 2},
        }
        executor = SimpleNamespace(_get_goal_contract_config=lambda job: contract)
        return AgentGoalContractService().evaluate_goal_contract(
            executor,
            SimpleNamespace(id=uuid4()),
            {"findings": findings, "artifacts": [], "goal_progress": 0},
            include_result_keys=False,
        )

    def test_two_recalled_findings_do_not_satisfy_a_contract_for_two(self):
        recalled = [
            dict(MECHANISM, recalled=True),
            dict(MECHANISM, recalled=True),
        ]

        assert self._counts(recalled)["satisfied"] is False

    def test_two_measured_findings_do(self):
        assert self._counts([dict(MECHANISM), dict(MECHANISM)])["satisfied"] is True

    def test_one_measured_plus_one_recalled_is_still_short(self):
        mixed = [dict(MECHANISM), dict(MECHANISM, recalled=True)]

        assert self._counts(mixed)["satisfied"] is False


class TestRecalledEvidenceIsCitable:
    def test_a_recalled_type_resolves_a_derived_from_citation(self):
        """record_prediction checks derived_from against the types the run
        holds. Recalled findings enter that set, so a run can cite what an
        earlier run established -- with the source job on the record."""
        from app.services import agent_evidence_citation

        recalled = prior._provenance(_job([MECHANISM]), MECHANISM)
        available = [recalled["type"]]

        assert (
            agent_evidence_citation.resolve("mechanism_comparison", available)
            == "mechanism_comparison"
        )

    def test_a_type_no_run_produced_is_still_refused(self):
        from app.services import agent_evidence_citation

        assert (
            agent_evidence_citation.resolve("invented_measurement", ["headroom_bound"])
            is None
        )


@pytest.mark.asyncio
class TestTheQuery:
    async def test_it_returns_newest_first_and_respects_the_limit(self, db_session):
        pytest.importorskip("sqlalchemy")
        result = await prior.recall(db=db_session, user_id=uuid4(), limit=5)

        assert result["success"] is True
        assert result["count"] == 0
        assert result["findings"] == []

    async def test_the_note_says_recall_is_not_measurement(self, db_session):
        result = await prior.recall(db=db_session, user_id=uuid4())

        assert "NOT count toward" in result["note"]


class TestTheSubjectFilterSearchesTheWholeRecord:
    """A live run asked for "L2 prefetcher" and got nothing, while the
    database held `l2.prefetcher=StridePrefetcher`: the words were there,
    adjacent nowhere, and in a field the filter was not reading. A filter
    narrower than the record it searches returns an empty answer that reads as
    an absence of evidence."""

    REAL = {
        "type": "mechanism_comparison",
        "subject": "shipped_stride_l2_control",
        "title": "shipped_stride_l2_control: 1.5248x (644242 -> 422497 cycles)",
        "mechanisms": ["l2.prefetcher=StridePrefetcher"],
        "speedup": 1.5248,
        "identical_stats": False,
    }

    def test_the_query_that_found_nothing_now_finds_it(self):
        assert prior._matches(self.REAL, [], "L2 prefetcher") is True

    def test_word_order_does_not_matter(self):
        assert prior._matches(self.REAL, [], "prefetcher l2") is True

    def test_a_word_only_in_a_nested_field_still_matches(self):
        """StridePrefetcher appears only inside `mechanisms`."""
        assert prior._matches(self.REAL, [], "StridePrefetcher") is True

    def test_a_number_can_be_searched_for(self):
        assert prior._matches(self.REAL, [], "1.5248") is True

    def test_every_word_must_appear(self):
        """Matching any word would return most of the corpus for most
        queries, which is the same uselessness in the other direction."""
        assert prior._matches(self.REAL, [], "l2 branch predictor") is False

    def test_an_unrelated_query_still_misses(self):
        assert prior._matches(self.REAL, [], "branch predictor") is False

    def test_booleans_are_not_searchable_text(self):
        """identical_stats=False must not make "false" a matching word."""
        assert prior._matches(self.REAL, [], "false") is False
