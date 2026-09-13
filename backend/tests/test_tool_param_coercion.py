"""A lone string where a list is wanted is repaired, not refused.

Observed twice in one live run: record_prediction rejected with "field
derived_from should be array, got str" while the value it carried was exactly
right. The model then spent an iteration re-sending the same value in brackets.
Real shape mistakes still fail.
"""

from __future__ import annotations

from app.services.agent_tool_validation import coerce_tool_params, validate_tool_params


def test_a_lone_string_becomes_a_single_element_list():
    params = {
        "subject": "fsqrt latency",
        "metric": "cycles",
        "predicted_value": 24.0,
        "methodology": "gem5 O3CPU default opLat",
        "derived_from": "benchmark_measurement",
    }

    repaired = coerce_tool_params("record_prediction", params)

    assert repaired == ["derived_from"]
    assert params["derived_from"] == ["benchmark_measurement"]
    assert validate_tool_params("record_prediction", params) is None


def test_a_list_is_left_alone():
    params = {"derived_from": ["benchmark_measurement"]}

    assert coerce_tool_params("record_prediction", params) == []
    assert params["derived_from"] == ["benchmark_measurement"]


def test_record_method_accepts_a_string_procedure_after_coercion():
    params = {
        "name": "measure fsqrt latency",
        "procedure": "Emit a dependent chain of fsqrt and time it.",
        "prevents": "The compiler reshaping the loop under test.",
        "derived_from": "benchmark_measurement",
    }

    coerce_tool_params("record_method", params)

    assert params["procedure"] == ["Emit a dependent chain of fsqrt and time it."]
    assert params["derived_from"] == ["benchmark_measurement"]
    assert validate_tool_params("record_method", params) is None


def test_an_empty_string_is_not_turned_into_a_list():
    """An empty required field is a real error and must keep failing."""
    params = {"derived_from": "   "}

    assert coerce_tool_params("record_prediction", params) == []
    assert params["derived_from"] == "   "


def test_a_wrong_type_that_is_not_a_string_still_fails():
    params = {
        "subject": "x",
        "metric": "cycles",
        "predicted_value": 1.0,
        "methodology": "m",
        "derived_from": 42,
    }

    assert coerce_tool_params("record_prediction", params) == []
    assert "derived_from" in (validate_tool_params("record_prediction", params) or "")


def test_unknown_tools_are_not_touched():
    params = {"anything": "value"}

    assert coerce_tool_params("no_such_tool", params) == []
    assert params == {"anything": "value"}


def test_a_one_item_list_where_a_string_is_wanted_is_unwrapped():
    """A live run sent run_args as a list twice and lost both attempts."""
    params = {"code": "int main(void){return 0;}", "run_args": ["--iterations=1000"]}

    repaired = coerce_tool_params("simulate_c_workload", params)

    assert repaired == ["run_args"]
    assert params["run_args"] == "--iterations=1000"
    assert validate_tool_params("simulate_c_workload", params) is None


def test_a_multi_item_list_is_not_joined_into_a_string():
    """Joining would guess the separator; that is a real mistake to report."""
    params = {"code": "int main(void){return 0;}", "run_args": ["--n", "1000"]}

    assert coerce_tool_params("simulate_c_workload", params) == []
    assert "run_args" in (validate_tool_params("simulate_c_workload", params) or "")


def test_a_structure_serialised_as_json_is_parsed_back():
    """Refused twice in one live run while the value was exactly right.

    Uses share_findings because it still declares a strict array of objects;
    find_fusion_candidates now accepts either shape so its own handler can
    fall back to the profile the run already took.
    """
    params = {"findings": '[{"title": "a", "detail": "b"}]', "agent_name": "x"}

    repaired = coerce_tool_params("share_findings", params)

    assert repaired == ["findings"]
    assert isinstance(params["findings"], list)
    assert params["findings"][0]["title"] == "a"


def test_a_single_json_object_becomes_a_one_item_array():
    params = {"findings": '{"title": "a"}', "agent_name": "x"}

    coerce_tool_params("share_findings", params)

    assert isinstance(params["findings"], list) and len(params["findings"]) == 1


def test_text_that_is_not_json_is_left_for_validation_to_refuse():
    """An array of objects cannot be salvaged from prose."""
    params = {"findings": "the things I found", "agent_name": "x"}

    assert coerce_tool_params("share_findings", params) == []
    assert "findings" in (validate_tool_params("share_findings", params) or "")
