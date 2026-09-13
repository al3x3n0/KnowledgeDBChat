"""Telling a run what its contract still wants, without spending an action.

The volatile prompt listed the finding TYPES a run had produced, so a run
needing two `headroom_bound` findings and holding one saw the type present and
could not tell it was short. A live 16-iteration study called
`get_research_findings` five times asking exactly that, and ran out of
iterations before reaching the tools it was created to exercise.
"""

from app.services.autonomous_agent_executor import AutonomousAgentExecutor


class _Job:
    def __init__(self, contract):
        self.config = {"goal_contract": contract} if contract is not None else {}


def outstanding(contract, counts):
    return AutonomousAgentExecutor()._outstanding_contract_evidence(
        _Job(contract), counts
    )


class TestWhatIsStillMissing:
    def test_a_partial_count_says_how_many_more(self):
        """The case the prompt could not express: the type is present and the
        run is still short."""
        line = outstanding(
            {"required_finding_types": {"headroom_bound": 2}}, {"headroom_bound": 1}
        )

        assert line == "headroom_bound (1 more; 1 of 2)"

    def test_a_type_never_produced_is_named_with_its_count(self):
        line = outstanding(
            {"required_finding_types": {"mechanism_sweep": 1}}, {"dynamic_profile": 3}
        )

        assert line == "mechanism_sweep (1)"

    def test_several_shortfalls_are_all_reported(self):
        line = outstanding(
            {
                "required_finding_types": {
                    "headroom_bound": 2,
                    "mechanism_comparison": 1,
                    "mechanism_sweep": 1,
                }
            },
            {"headroom_bound": 1},
        )

        assert "headroom_bound (1 more; 1 of 2)" in line
        assert "mechanism_comparison (1)" in line
        assert "mechanism_sweep (1)" in line


class TestWhenToSayNothing:
    def test_a_satisfied_contract_is_not_nagged(self):
        """A run that has finished should not be told to keep going."""
        assert (
            outstanding(
                {"required_finding_types": {"headroom_bound": 2}},
                {"headroom_bound": 2},
            )
            == ""
        )

    def test_more_than_required_is_still_satisfied(self):
        assert (
            outstanding(
                {"required_finding_types": {"headroom_bound": 1}},
                {"headroom_bound": 5},
            )
            == ""
        )

    def test_a_job_with_no_contract_says_nothing(self):
        assert outstanding(None, {"anything": 1}) == ""

    def test_a_contract_requiring_no_types_says_nothing(self):
        assert outstanding({"min_progress": 0}, {"anything": 1}) == ""


class TestToleratingHowItWasWritten:
    def test_a_bare_list_of_types_means_one_each(self):
        """Contracts are written both ways; the counting form arrived later."""
        line = outstanding(
            {"required_finding_types": ["mechanism_sweep", "headroom_bound"]},
            {"headroom_bound": 1},
        )

        assert line == "mechanism_sweep (1)"

    def test_a_non_numeric_requirement_counts_as_one(self):
        line = outstanding({"required_finding_types": {"x": "two"}}, {})

        assert line == "x (1)"
