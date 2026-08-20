"""Contracts must be able to reject results that counting would accept.

Every case here is a run that really happened during the microarchitecture
calibration work and really produced a confident wrong number: a throughput
figure that was an artifact of the harness, a frequency implying a dependent
integer add ran in 0.83 cycles, and a prediction recorded and never settled.
"""

from __future__ import annotations

from app.services import agent_measurement_validity as validity


def _prediction_action(prediction_id: str, success: bool = True):
    return {
        "action": {"tool": "record_prediction", "params": {"subject": "fsqrt"}},
        "result": {"success": success, "data": {"prediction_id": prediction_id}},
    }


def _measurement_action(prediction_id: str, success: bool = True):
    return {
        "action": {
            "tool": "record_measurement",
            "params": {"prediction_id": prediction_id},
        },
        "result": {"success": success},
    }


def test_no_validity_block_means_no_opinion():
    """A contract that declares nothing must not start failing runs."""
    result = validity.evaluate(
        {"min_findings": 2}, {"findings": [], "actions_taken": []}
    )

    assert result["declared"] is False
    assert result["missing"] == []


def test_prediction_without_measurement_is_unmet():
    state = {"actions_taken": [_prediction_action("pred-1")], "findings": []}

    result = validity.evaluate({"validity": {"predictions_measured": True}}, state)

    assert "validity:predictions_measured" in result["missing"]
    assert result["details"]["unsettled_predictions"] == ["pred-1"]


def test_prediction_settled_by_its_own_measurement_passes():
    state = {
        "actions_taken": [_prediction_action("pred-1"), _measurement_action("pred-1")],
        "findings": [],
    }

    result = validity.evaluate({"validity": {"predictions_measured": True}}, state)

    assert result["missing"] == []


def test_measurement_of_a_different_prediction_does_not_settle_it():
    """Closing the loop means settling *that* claim, not any claim."""
    state = {
        "actions_taken": [_prediction_action("pred-1"), _measurement_action("pred-2")],
        "findings": [],
    }

    result = validity.evaluate({"validity": {"predictions_measured": True}}, state)

    assert "validity:predictions_measured" in result["missing"]
    assert result["details"]["unsettled_predictions"] == ["pred-1"]


def test_failed_calls_settle_nothing_and_record_nothing():
    state = {
        "actions_taken": [
            _prediction_action("pred-1", success=False),
            _measurement_action("pred-9", success=False),
        ],
        "findings": [],
    }

    result = validity.evaluate({"validity": {"predictions_measured": True}}, state)

    assert result["missing"] == []


def test_bare_measurement_without_a_spread_is_unmet():
    state = {
        "findings": [{"type": "cycle_model_measurement", "cycles_per_op": 10.14}],
        "actions_taken": [],
    }
    contract = {"validity": {"require_uncertainty": ["cycle_model_measurement"]}}

    result = validity.evaluate(contract, state)

    assert "validity:uncertainty:cycle_model_measurement" in result["missing"]


def test_a_reported_spread_satisfies_the_requirement():
    state = {
        "findings": [
            {"type": "cycle_model_measurement", "cycles_per_op": 10.14, "spread": 0.13}
        ],
        "actions_taken": [],
    }
    contract = {"validity": {"require_uncertainty": ["cycle_model_measurement"]}}

    assert validity.evaluate(contract, state)["missing"] == []


def test_uncertainty_is_found_one_level_down():
    """Tools put their numbers under `data`; a check that missed those would
    pass every finding whose spread it could not see."""
    state = {
        "findings": [
            {"type": "cycle_model_measurement", "data": {"samples": 9, "value": 10.1}}
        ],
        "actions_taken": [],
    }
    contract = {"validity": {"require_uncertainty": ["cycle_model_measurement"]}}

    assert validity.evaluate(contract, state)["missing"] == []


def test_a_type_that_never_appeared_is_not_blamed_here():
    """Absence is the counting requirements' job to report, not this one's."""
    contract = {"validity": {"require_uncertainty": ["cycle_model_measurement"]}}

    assert (
        validity.evaluate(contract, {"findings": [], "actions_taken": []})["missing"]
        == []
    )


def test_impossible_value_is_rejected():
    """A dependent integer add cannot take less than one cycle.

    This is the run that reported 0.83 and would otherwise have counted as a
    perfectly good measurement.
    """
    state = {
        "findings": [{"type": "anchor_check", "residual_cycles": 0.83}],
        "actions_taken": [],
    }
    contract = {
        "validity": {
            "bounds": {
                "anchor_check": {"field": "residual_cycles", "min": 0.95, "max": 1.15}
            }
        }
    }

    result = validity.evaluate(contract, state)

    assert "validity:bounds:anchor_check" in result["missing"]
    assert result["details"]["out_of_bounds"]["anchor_check"]["values"] == [0.83]


def test_value_inside_the_bounds_passes():
    state = {
        "findings": [{"type": "anchor_check", "residual_cycles": 1.01}],
        "actions_taken": [],
    }
    contract = {
        "validity": {
            "bounds": {
                "anchor_check": {"field": "residual_cycles", "min": 0.95, "max": 1.15}
            }
        }
    }

    assert validity.evaluate(contract, state)["missing"] == []


def test_a_one_sided_bound_is_allowed():
    state = {
        "findings": [{"type": "cycle_model_measurement", "cycles_per_op": 0.0}],
        "actions_taken": [],
    }
    contract = {
        "validity": {
            "bounds": {
                "cycle_model_measurement": {"field": "cycles_per_op", "min": 0.1}
            }
        }
    }

    assert (
        "validity:bounds:cycle_model_measurement"
        in validity.evaluate(contract, state)["missing"]
    )


def test_unmet_requirements_explain_their_remedy():
    """A label the model cannot act on is not a useful gate."""
    missing = [
        "validity:predictions_measured",
        "validity:bounds:anchor_check",
        "validity:uncertainty:cycle_model_measurement",
    ]
    details = {
        "unsettled_predictions": ["pred-1"],
        "out_of_bounds": {
            "anchor_check": {
                "field": "residual_cycles",
                "min": 0.95,
                "max": 1.15,
                "values": [0.83],
            }
        },
    }

    lines = validity.explain(missing, details)

    assert any("record_measurement" in line and "pred-1" in line for line in lines)
    assert any("0.83" in line and "residual_cycles" in line for line in lines)
    assert any("spread or sample count" in line for line in lines)


def test_describe_lists_what_a_contract_demands():
    spec = {
        "predictions_measured": True,
        "require_uncertainty": ["cycle_model_measurement"],
        "bounds": {
            "anchor_check": {"field": "residual_cycles", "min": 0.95, "max": 1.15}
        },
    }

    lines = list(validity.describe(spec))

    assert len(lines) == 3
    assert any("settled" in line for line in lines)


class _Job:
    """Enough of an AgentJob for the contract and prompt code under test.

    Unset attributes read as None rather than raising, so this stub does not
    have to track every optional field the prompt builder consults.
    """

    def __init__(self, config):
        self.config = config
        self.results = {}
        self.iteration = 1
        self.goal = "calibrate"
        self.status = "running"
        self.job_type = "research"
        self.name = "job"

    def __getattr__(self, name):
        return None


def _executor():
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    return AutonomousAgentExecutor()


def test_validity_block_survives_contract_normalization():
    """A block dropped in normalization would be a gate that never fires."""
    job = _Job(
        {
            "goal_contract": {
                "enabled": True,
                "validity": {
                    "predictions_measured": True,
                    "require_uncertainty": ["cycle_model_measurement"],
                    "bounds": {
                        "anchor_check": {
                            "field": "residual_cycles",
                            "min": 0.95,
                            "max": 1.15,
                        }
                    },
                },
            }
        }
    )

    contract = _executor()._get_goal_contract_config(job)

    assert contract["validity"]["predictions_measured"] is True
    assert contract["validity"]["require_uncertainty"] == ["cycle_model_measurement"]
    assert contract["validity"]["bounds"]["anchor_check"]["min"] == 0.95


def test_malformed_validity_rules_are_dropped_not_guessed():
    job = _Job(
        {
            "goal_contract": {
                "enabled": True,
                "validity": {
                    "predictions_measured": "no",
                    "bounds": {
                        "no_field": {"min": 1},
                        "no_edges": {"field": "x"},
                        "bad_edge": {"field": "y", "min": "abc"},
                    },
                },
            }
        }
    )

    contract = _executor()._get_goal_contract_config(job)

    assert "predictions_measured" not in contract["validity"]
    assert "no_field" not in contract["validity"].get("bounds", {})
    assert "no_edges" not in contract["validity"].get("bounds", {})
    assert "bad_edge" not in contract["validity"].get("bounds", {})


def test_contract_is_unsatisfied_while_a_prediction_is_open():
    """The end-to-end gate: counting requirements met, soundness not."""
    from app.services.agent_goal_contract_service import AgentGoalContractService

    job = _Job(
        {
            "goal_contract": {
                "enabled": True,
                "min_progress": 0,
                "required_finding_types": ["prediction_recorded"],
                "validity": {"predictions_measured": True},
            }
        }
    )
    state = {
        "goal_progress": 100,
        "findings": [{"type": "prediction_recorded"}],
        "actions_taken": [_prediction_action("pred-1")],
    }

    result = AgentGoalContractService().evaluate_goal_contract(_executor(), job, state)

    assert result["satisfied"] is False
    assert "validity:predictions_measured" in result["missing"]

    state["actions_taken"].append(_measurement_action("pred-1"))
    settled = AgentGoalContractService().evaluate_goal_contract(_executor(), job, state)

    assert settled["satisfied"] is True


def test_validity_requirements_appear_in_the_stable_prompt():
    """The model must know the rules before it measures, not after."""
    job = _Job(
        {
            "goal_contract": {
                "enabled": True,
                "validity": {
                    "predictions_measured": True,
                    "require_uncertainty": ["cycle_model_measurement"],
                },
            }
        }
    )
    job.job_type = "research"
    job.name = "calibrate"

    executor = _executor()
    prompt = executor._build_thinking_prompt_stable(job, None, {})

    assert "MUST SATISFY" in prompt
    assert "settled" in prompt
    assert "cycle_model_measurement" in prompt


def test_stable_prompt_stays_byte_identical_across_iterations():
    """It keys the provider prompt cache; drift there is a silent cost."""
    job = _Job(
        {
            "goal_contract": {
                "enabled": True,
                "validity": {"predictions_measured": True},
            }
        }
    )
    job.job_type = "research"
    job.name = "calibrate"
    executor = _executor()

    first = executor._build_thinking_prompt_stable(job, None, {"findings": []})
    job.iteration = 7
    second = executor._build_thinking_prompt_stable(
        job, None, {"findings": [{"type": "x"}], "actions_taken": [1, 2]}
    )

    assert first == second


def test_a_job_without_validity_gets_no_extra_prompt_text():
    job = _Job({"goal_contract": {"enabled": True, "min_findings": 2}})
    job.job_type = "research"
    job.name = "plain"

    prompt = _executor()._build_thinking_prompt_stable(job, None, {})

    assert "MUST SATISFY" not in prompt


async def _call_check_goal_status(state):
    from app.services.agent_tool_dispatch import AgentToolExecutionContext
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    executor = AutonomousAgentExecutor()
    job = _Job({})
    job.max_iterations = 10
    job.max_tool_calls = 50
    job.tool_calls_used = 3
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id=None, job=job, state=state
    )
    provider = executor.tool_registry.resolve("check_goal_status", ctx)
    return await provider.execute("check_goal_status", {}, ctx)


async def test_check_goal_status_reports_unmet_validity_with_remedies():
    """Self-check mid-run, rather than learning the rules by being refused."""
    state = {
        "goal_progress": 100,
        "findings": [],
        "actions_taken": [],
        "goal_contract_last": {
            "enabled": True,
            "satisfied": False,
            "missing": ["validity:predictions_measured"],
            "metrics": {"validity": {"details": {"unsettled_predictions": ["pred-1"]}}},
        },
    }

    result = await _call_check_goal_status(state)

    data = result["data"]
    assert data["goal_contract_satisfied"] is False
    assert "validity:predictions_measured" in data["goal_contract_missing"]
    assert any("record_measurement" in line for line in data["goal_contract_remedies"])


async def test_check_goal_status_is_quiet_without_a_contract():
    result = await _call_check_goal_status(
        {"goal_progress": 10, "findings": [], "actions_taken": []}
    )

    assert result["data"]["goal_contract_enabled"] is False
    assert "goal_contract_remedies" not in result["data"]


def test_a_run_that_records_no_method_is_unmet_when_required():
    """What a run learned about *how* to work should outlive the run."""
    contract = {"validity": {"records_method": True}}
    state = {"findings": [{"type": "cycle_model_measurement"}], "actions_taken": []}

    result = validity.evaluate(contract, state)

    assert "validity:records_method" in result["missing"]
    remedy = validity.explain(result["missing"], result["details"])
    assert any("record_method" in line for line in remedy)


def test_recording_a_method_satisfies_the_requirement():
    contract = {"validity": {"records_method": True}}
    state = {
        "findings": [
            {"type": "cycle_model_measurement"},
            {"type": "method_recorded", "title": "Method: chains (validated)"},
        ],
        "actions_taken": [],
    }

    assert validity.evaluate(contract, state)["missing"] == []


def test_the_method_requirement_survives_normalization_and_the_prompt():
    job = _Job(
        {"goal_contract": {"enabled": True, "validity": {"records_method": True}}}
    )
    executor = _executor()

    contract = executor._get_goal_contract_config(job)
    assert contract["validity"]["records_method"] is True

    prompt = executor._build_thinking_prompt_stable(job, None, {})
    assert "record_method" in prompt
