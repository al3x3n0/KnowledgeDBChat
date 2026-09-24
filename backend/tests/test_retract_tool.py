"""Withdrawing a number a run has shown to be wrong.

The retraction machinery existed and nothing could reach it. A run that found
an `IrregularStreamBufferPrefetcher` result described a mechanism issuing no
prefetches could add a contradicting finding, but the false one stayed
recallable at equal standing, with a job id and a measurement source attached.
"""

from __future__ import annotations

from app.services import agent_retract_tool as rt

STATE = {
    "findings": [
        {
            "type": "mechanism_comparison",
            "title": "IrregularStreamBuffer issues zero prefetches at L2",
        },
        {
            "type": "simulated_measurement",
            "title": "cycles equal the no-prefetcher run",
        },
    ]
}
REASON = "The arm never prefetched: pfIdentified and pfIssued are both zero."


class TestWhatAValidRetractionNeeds:
    def test_a_grounded_retraction_is_accepted(self):
        assert (
            rt.check(
                "job1#3",
                REASON,
                ["IrregularStreamBuffer issues zero prefetches at L2"],
                STATE,
            )
            is None
        )

    def test_a_finding_type_may_be_cited_instead_of_a_title(self):
        assert rt.check("job1#3", REASON, ["mechanism_comparison"], STATE) is None


class TestWhatItRefuses:
    def test_a_job_id_alone_does_not_name_a_claim(self):
        problem = rt.check("job1", REASON, ["mechanism_comparison"], STATE)
        assert problem and "#" in problem

    def test_a_reason_too_thin_to_act_on(self):
        """A later run has to tell a harness defect from a changed question."""
        problem = rt.check("job1#3", "wrong", ["mechanism_comparison"], STATE)
        assert problem and "reason" in problem

    def test_citing_nothing_is_an_opinion(self):
        problem = rt.check("job1#3", REASON, [], STATE)
        assert problem and "cites nothing" in problem

    def test_citing_something_this_run_never_produced(self):
        """The failure mode is a run reasoning its way out of an inconvenient
        number, so the citation is checked against what it actually measured."""
        problem = rt.check("job1#3", REASON, ["a study I did not run"], STATE)
        assert problem and "did not produce" in problem

    def test_a_run_with_no_findings_of_its_own_cannot_overturn_anything(self):
        problem = rt.check("job1#3", REASON, ["anything"], {"findings": []})
        assert problem and "no findings of its own" in problem

    def test_recalled_findings_do_not_count_as_this_run_s_evidence(self):
        """Recalling a number is not establishing one; a retraction built on a
        recall would let one wrong finding withdraw another."""
        recalled = {
            "findings": [
                {"type": "mechanism_evaluation", "title": "something", "recalled": True}
            ]
        }
        problem = rt.check("job1#3", REASON, ["something"], recalled)
        assert problem and "no findings of its own" in problem
