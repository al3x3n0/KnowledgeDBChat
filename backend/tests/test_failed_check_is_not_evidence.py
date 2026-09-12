"""A check that never ran must not satisfy the contract that required it.

Measured live, in the reproduction pipeline's implement stage:

    {"type": "implementation_verified", "verified": false, "ran": false,
     "cases_run": 0, "cases_passed": 0, "note": "Check failed: ", "reason": ""}

That satisfied `required_finding_types: [implementation_verified]` at 100%,
which fired the chain and started the benchmark stage on code nothing had
verified. The gate between "it works" and "time it" opened for a check that
never ran -- and the fastest implementation of any algorithm is one that
returns garbage.

Emitting the finding either way is deliberate and stays: `compare_to_claim`
reads it to refuse scoring unverified code. It just must not COUNT as the
evidence whose absence it reports.
"""

import asyncio

import pytest

from app.services.agent_goal_contract_service import AgentGoalContractService
from app.services.autonomous_agent_executor import AutonomousAgentExecutor

pytestmark = pytest.mark.unit

CONTRACT = {
    "goal_contract": {
        "enabled": True,
        "required_finding_types": {"implementation_verified": 1},
    }
}


def _evaluate(findings):
    from app.models.agent_job import AgentJob

    job = AgentJob(name="implement", goal="implement it", config=CONTRACT)
    return AgentGoalContractService().evaluate_goal_contract(
        AutonomousAgentExecutor(),
        job,
        {"findings": findings, "goal_progress": 100, "artifacts": []},
    )


class TestAFailedCheckDoesNotCount:
    def test_the_exact_finding_from_the_live_run(self):
        result = _evaluate(
            [
                {
                    "type": "implementation_verified",
                    "subject": "implementation",
                    "verified": False,
                    "ran": False,
                    "language": "c",
                    "cases_run": 0,
                    "cases_passed": 0,
                    "note": "Check failed: ",
                    "reason": "",
                }
            ]
        )

        assert result["satisfied"] is False
        assert "finding_type:implementation_verified" in result["missing"]

    def test_a_check_whose_cases_ran_and_failed_also_does_not_count(self):
        """Ran but failed is a real result about the code -- and still not
        evidence that the implementation is verified."""
        result = _evaluate(
            [
                {
                    "type": "implementation_verified",
                    "verified": False,
                    "ran": True,
                    "cases_run": 3,
                    "cases_passed": 1,
                }
            ]
        )

        assert result["satisfied"] is False

    def test_a_passing_check_still_counts(self):
        """The control. A guard that rejected every check would make the
        contract unsatisfiable."""
        result = _evaluate(
            [
                {
                    "type": "implementation_verified",
                    "verified": True,
                    "ran": True,
                    "cases_run": 3,
                    "cases_passed": 3,
                }
            ]
        )

        assert result["satisfied"] is True

    def test_a_finding_with_no_verified_field_is_unaffected(self):
        """Most evidence types carry no such field and must not be caught by
        a guard aimed at self-declared failure."""
        result = _evaluate(
            [{"type": "implementation_verified", "cases_run": 3, "cases_passed": 3}]
        )

        assert result["satisfied"] is True


class TestTheFailureNamesItself:
    def test_an_exception_with_no_message_still_reports_its_class(self):
        """asyncio.TimeoutError stringifies to "", which produced
        `note="Check failed: "` -- a run could not tell a timeout from a broken
        toolchain from its own bad code."""
        from app.services import agent_compiler_sandbox as sandbox
        from app.services import agent_implementation_check as impl

        async def _timeout(*args, **kwargs):
            raise asyncio.TimeoutError()

        original = sandbox._run
        sandbox._run = _timeout
        try:
            outcome = asyncio.get_event_loop().run_until_complete(
                impl.check_implementation(
                    code="int main(void){return 0;}",
                    cases=[{"input": "", "expected_output": "ok"}],
                    language="c",
                )
            )
        finally:
            sandbox._run = original

        assert outcome.ran is False
        assert outcome.reason == "error"
        assert "TimeoutError" in outcome.note
        assert "says nothing about the implementation" in outcome.note
