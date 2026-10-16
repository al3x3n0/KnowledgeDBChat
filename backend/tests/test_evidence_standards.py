"""A plan a run proposed for itself is held to an evidence standard.

Found live: the first campaign to approve its own plan ran a benchmark stage
whose contract asked for one `benchmark_measurement`; it took one trial of one
program, met its contract, and the campaign concluded that nothing had been
established. The standard is written into the plan's contracts before it is
checked, priced and approved, and enforced by the existing validity rules.
"""

import copy

import pytest

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services import agent_evidence_standards as standards
from app.services import agent_measurement_validity as validity
from app.services import agent_self_planned_pipeline as planning

pytestmark = pytest.mark.unit

IMPLEMENT = {
    "id": "implement",
    "job_type": "research",
    "goal": "Implement fastmod in C and check it against plain %",
    "contract": {"required_finding_types": ["implementation_verified"]},
}
BENCHMARK = {
    "id": "benchmark",
    "job_type": "research",
    "goal": "Time fastmod against plain %",
    "depends_on": ["implement"],
    "contract": {"required_finding_types": ["benchmark_measurement"]},
}
SPEC = {"name": "fastmod", "stages": [IMPLEMENT, BENCHMARK]}


def contract_of(spec, stage_id):
    return next(s for s in spec["stages"] if s["id"] == stage_id)["contract"]


class TestWhatIsWrittenIn:
    def test_a_benchmark_must_report_a_spread(self):
        raised, notes = standards.apply(SPEC)

        assert contract_of(raised, "benchmark")["validity"]["require_uncertainty"] == [
            "benchmark_measurement"
        ]
        assert any("spread" in n for n in notes)

    def test_an_implementation_must_pass_an_oracle(self):
        raised, _ = standards.apply(SPEC)

        assert contract_of(raised, "implement")["validity"]["bounds"][
            "implementation_verified"
        ] == {"field": "generated_cases", "min": standards.ORACLE_MIN_CASES}

    def test_the_plan_given_is_not_changed(self):
        before = copy.deepcopy(SPEC)

        standards.apply(SPEC)

        assert SPEC == before

    def test_a_stronger_bound_is_kept_and_a_weaker_one_raised(self):
        stronger = copy.deepcopy(SPEC)
        contract_of(stronger, "implement")["validity"] = {
            "bounds": {
                "implementation_verified": {"field": "generated_cases", "min": 500}
            }
        }
        weaker = copy.deepcopy(SPEC)
        contract_of(weaker, "implement")["validity"] = {
            "bounds": {
                "implementation_verified": {"field": "generated_cases", "min": 5}
            }
        }

        kept = contract_of(standards.apply(stronger)[0], "implement")
        raised = contract_of(standards.apply(weaker)[0], "implement")

        assert kept["validity"]["bounds"]["implementation_verified"]["min"] == 500
        assert raised["validity"]["bounds"]["implementation_verified"]["min"] == 50

    def test_a_bound_on_another_field_is_kept_and_said(self):
        own = copy.deepcopy(SPEC)
        contract_of(own, "implement")["validity"] = {
            "bounds": {"implementation_verified": {"field": "cases_passed", "min": 3}}
        }

        raised, notes = standards.apply(own)

        assert (
            contract_of(raised, "implement")["validity"]["bounds"][
                "implementation_verified"
            ]["field"]
            == "cases_passed"
        )
        assert any("kept its own bound" in n for n in notes)

    def test_every_spelling_of_a_contract_is_read(self):
        counted = copy.deepcopy(SPEC)
        contract_of(counted, "benchmark").pop("required_finding_types")
        contract_of(counted, "benchmark")["required_finding_type_counts"] = {
            "benchmark_measurement": 2
        }

        raised, _ = standards.apply(counted)

        assert "validity" in contract_of(raised, "benchmark")


class TestWhatItThenRefuses:
    """The raised contracts, judged by the validity rules the runtime uses."""

    def test_one_trial_does_not_meet_the_benchmark_stage(self):
        contract = contract_of(standards.apply(SPEC)[0], "benchmark")
        one = {"findings": [{"type": "benchmark_measurement", "all_ms": [68]}]}
        five = {
            "findings": [
                {"type": "benchmark_measurement", "all_ms": [68, 69, 70, 68, 71]}
            ]
        }

        assert (
            "validity:uncertainty:benchmark_measurement"
            in validity.evaluate(contract, one)["missing"]
        )
        assert validity.evaluate(contract, five)["missing"] == []

    def test_hand_written_cases_do_not_meet_the_implementation_stage(self):
        contract = contract_of(standards.apply(SPEC)[0], "implement")
        cases = {
            "findings": [
                {
                    "type": "implementation_verified",
                    "verified": True,
                    "generated_cases": 0,
                }
            ]
        }
        oracle = {
            "findings": [
                {
                    "type": "implementation_verified",
                    "verified": True,
                    "generated_cases": 200,
                }
            ]
        }

        assert (
            "validity:bounds:implementation_verified"
            in validity.evaluate(contract, cases)["missing"]
        )
        assert validity.evaluate(contract, oracle)["missing"] == []


class _Model:
    def __init__(self, spec):
        self.spec = spec
        self.asked = []

    async def generate_structured(self, **kwargs):
        self.asked.append(kwargs["user_message"])
        return copy.deepcopy(self.spec)


async def test_a_proposed_plan_carries_the_standard_to_its_stages(
    db_session, test_user
):
    job = AgentJob(
        name="scope fastmod",
        goal="See whether fastmod is worth adopting",
        job_type="research",
        user_id=test_user.id,
        status=AgentJobStatus.RUNNING.value,
        config={},
    )
    db_session.add(job)
    await db_session.commit()
    model = _Model(SPEC)

    result = await planning.propose(
        job,
        "Implement fastmod, check it, and time it against plain %.",
        db=db_session,
        llm_service=model,
        state={},
    )

    assert result.get("success") is True, result
    assert "Evidence standard for this plan" in model.asked[0]
    proposal = planning.proposal_of(job)
    assert len(proposal["evidence_standards"]) == 2
    assert "Held to:" in planning.approval_message(proposal)
    (first,) = job.chain_config["child_jobs"]
    contract = (first.get("config") or {}).get("goal_contract") or {}
    assert contract.get("validity", {}).get("bounds", {}).get("implementation_verified")


def test_the_run_is_told_what_the_oracle_bound_counts():
    """Found live: read as "50 cases", it led a stage to write 70 by hand."""
    contract = contract_of(standards.apply(SPEC)[0], "implement")

    (line,) = validity.describe(contract["validity"])

    assert "reference" in line and "generator" in line
    assert "Cases you write yourself do not count" in line


class TestASpreadIsJudgedOnWhatWasLastMeasured:
    """Found live: one trial, then seven, and the one blocked the stage."""

    contract = {"validity": {"require_uncertainty": ["benchmark_measurement"]}}

    @staticmethod
    def bench(subject, trials):
        return {
            "type": "benchmark_measurement",
            "subject": subject,
            "all_ms": [68 + i for i in range(trials)],
        }

    def test_measuring_again_with_repeats_recovers(self):
        state = {"findings": [self.bench("fastmod", 1), self.bench("fastmod", 7)]}

        assert validity.evaluate(self.contract, state)["missing"] == []

    def test_a_comparison_needs_a_spread_on_both_sides(self):
        state = {"findings": [self.bench("plain", 1), self.bench("fastmod", 7)]}

        result = validity.evaluate(self.contract, state)

        assert result["missing"] == ["validity:uncertainty:benchmark_measurement"]
        assert result["details"]["bare_subjects"]["benchmark_measurement"] == ["plain"]

    def test_the_run_is_told_which_one_to_measure_again(self):
        state = {"findings": [self.bench("plain", 1), self.bench("fastmod", 7)]}
        result = validity.evaluate(self.contract, state)

        (line,) = validity.explain(result["missing"], result["details"])

        assert "for: plain" in line and "same label" in line
