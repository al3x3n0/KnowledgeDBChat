"""A contract must not go vacuous because of which key it was written under.

`agent_pipeline_spec` reads `required_finding_type_counts` first and the
authoring path emits it, while the runtime normaliser read only
`required_finding_types`. A contract written the first way survived validation,
reached the job config intact, and was then normalised to *nothing required* --
while staying `enabled`. An enabled contract that requires nothing is satisfied
immediately, so the stage autocompleted at progress 100 having produced none of
the evidence it named.

Measured live: a pipeline stage contracted for three `fusion_candidate`
findings completed with zero, logging "deterministic goal contract satisfied".
The evaluator was correct throughout; it was asked the wrong question.
"""

from types import SimpleNamespace

import pytest

from app.services.agent_goal_contract_service import AgentGoalContractService
from app.services.autonomous_agent_executor import AutonomousAgentExecutor


def _config_for(contract: dict) -> dict:
    job = SimpleNamespace(config={"goal_contract": contract}, job_type="research")
    return AutonomousAgentExecutor._get_goal_contract_config(
        object.__new__(AutonomousAgentExecutor), job
    )


class TestEverySpellingIsRead:
    @pytest.mark.parametrize(
        "contract,expected",
        [
            # The key the pipeline authoring path writes, and the one
            # agent_pipeline_spec._required_types() reads first.
            ({"required_finding_type_counts": {"fusion_candidate": 3}}, 3),
            # A bare list means one of each.
            ({"required_finding_types": ["fusion_candidate"]}, 1),
            # A mapping under the older key.
            ({"required_finding_types": {"fusion_candidate": 3}}, 3),
        ],
    )
    def test_the_requirement_survives_normalisation(self, contract, expected):
        counts = _config_for(contract)["required_finding_type_counts"]
        assert counts.get("fusion_candidate") == expected

    def test_artifacts_are_read_under_both_spellings_too(self):
        assert _config_for({"required_artifact_type_counts": {"deck": 2}})[
            "required_artifact_type_counts"
        ] == {"deck": 2}


class TestAnEnabledContractNeverRequiresNothing:
    """The failure was not a refusal, it was an acceptance."""

    def test_a_counts_contract_is_not_silently_emptied(self):
        config = _config_for({"required_finding_type_counts": {"fusion_candidate": 3}})
        assert config["enabled"] is True
        assert config[
            "required_finding_type_counts"
        ], "an enabled contract that requires nothing is satisfied on the spot"

    def test_the_evaluator_refuses_the_run_that_produced_nothing(self):
        # End to end over the two pieces: normalise the contract the authoring
        # path writes, then ask the evaluator about a run with no such finding.
        contract = _config_for(
            {"required_finding_type_counts": {"fusion_candidate": 3}}
        )
        job = SimpleNamespace(
            config={"goal_contract": contract},
            iteration=9,
            job_type="research",
            results={},
            id="x",
        )
        executor = SimpleNamespace(_get_goal_contract_config=lambda _job: contract)
        state = {"findings": [{"type": "paper"} for _ in range(54)]}

        out = AgentGoalContractService().evaluate_goal_contract(
            executor, job, state, include_result_keys=False
        )
        assert out["satisfied"] is False
        assert any("fusion_candidate" in item for item in out["missing"])
