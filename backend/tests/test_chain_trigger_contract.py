"""What `on_complete` promises the next stage.

A run that exhausts its iterations is marked completed -- deliberately, since
nothing failed -- and the finalizer records the contract it did not meet:
progress capped below 100, a `completed_contract_unmet` log entry, and
`results["goal_contract"]["satisfied"] = False`. All of that was written and
none of it was read by the thing that decides whether the next stage starts.

Observed end to end: a measure stage spent its whole budget getting an
implementation right, never benchmarked, and reported completed at 49% with
`benchmark_measurement` and `reproduction_verdict` missing. The compare stage
fired anyway and set about scoring a measurement that did not exist.

`on_any_end` still fires either way -- that is what it is for. `on_complete`
is a claim about the outcome, and now has to be true.
"""

import pytest

from app.models.agent_job import AgentJob, ChainTriggerCondition

pytestmark = pytest.mark.unit


def _job(*, condition, contract=None):
    job = AgentJob()
    job.chain_triggered = False
    job.chain_config = {"trigger_condition": condition.value}
    job.results = {"goal_contract": contract} if contract is not None else {}
    return job


MET = {"enabled": True, "satisfied": True, "missing": []}
UNMET = {
    "enabled": True,
    "satisfied": False,
    "missing": ["finding_type:benchmark_measurement"],
}


class TestOnCompleteMeansTheContractWasMet:
    def test_a_met_contract_fires_the_next_stage(self):
        job = _job(condition=ChainTriggerCondition.ON_COMPLETE, contract=MET)
        assert job.should_trigger_chain("complete") is True

    def test_an_unmet_contract_does_not(self):
        # The failure this exists for: the next stage would begin on premises
        # nobody established.
        job = _job(condition=ChainTriggerCondition.ON_COMPLETE, contract=UNMET)
        assert job.should_trigger_chain("complete") is False

    def test_a_job_with_no_contract_still_chains(self):
        # Most jobs declare none, and treating "nothing to prove" as failure
        # would stop every chain that never used contracts.
        job = _job(condition=ChainTriggerCondition.ON_COMPLETE)
        assert job.should_trigger_chain("complete") is True

    def test_a_disabled_contract_is_not_an_unmet_one(self):
        job = _job(
            condition=ChainTriggerCondition.ON_COMPLETE,
            contract={"enabled": False, "satisfied": False},
        )
        assert job.should_trigger_chain("complete") is True


class TestOnAnyEndStillMeansAnyEnd:
    def test_it_fires_on_an_unmet_contract(self):
        # Its whole purpose is to run whatever happened; gating it would take
        # away the only condition that can clean up after a bad stage.
        job = _job(condition=ChainTriggerCondition.ON_ANY_END, contract=UNMET)
        assert job.should_trigger_chain("complete") is True

    def test_it_fires_on_failure_too(self):
        job = _job(condition=ChainTriggerCondition.ON_ANY_END, contract=UNMET)
        assert job.should_trigger_chain("fail") is True


class TestTheOtherConditionsAreUntouched:
    def test_on_fail_is_unaffected_by_the_contract(self):
        job = _job(condition=ChainTriggerCondition.ON_FAIL, contract=UNMET)
        assert job.should_trigger_chain("fail") is True
        assert job.should_trigger_chain("complete") is False

    def test_on_findings_still_counts_findings(self):
        job = _job(condition=ChainTriggerCondition.ON_FINDINGS, contract=UNMET)
        job.chain_config = {
            "trigger_condition": ChainTriggerCondition.ON_FINDINGS.value,
            "findings_threshold": 3,
        }
        assert job.should_trigger_chain("findings", 5) is True
        assert job.should_trigger_chain("findings", 1) is False

    def test_an_already_triggered_chain_does_not_fire_twice(self):
        job = _job(condition=ChainTriggerCondition.ON_COMPLETE, contract=MET)
        job.chain_triggered = True
        assert job.should_trigger_chain("complete") is False


class TestTheHelperOnItsOwn:
    def test_it_reads_the_record_the_finalizer_writes(self):
        job = AgentJob()
        job.results = {"goal_contract": UNMET}
        assert job.goal_contract_satisfied() is False
        job.results = {"goal_contract": MET}
        assert job.goal_contract_satisfied() is True

    def test_missing_or_malformed_results_are_not_a_failure(self):
        job = AgentJob()
        for results in (None, {}, {"goal_contract": None}, {"goal_contract": "no"}):
            job.results = results
            assert job.goal_contract_satisfied() is True
