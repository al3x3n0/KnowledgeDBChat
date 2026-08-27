"""Per-tool argument checking against the catalog's own schemas.

Tools validated their arguments to wildly different standards: some returned a
clear message, some raised, some accepted nonsense and failed obscurely inside.
The message matters as much as the rejection, because an agent told which field
is wrong fixes its next call.
"""

import pytest

from app.services.agent_tool_validation import validate_tool_params


def test_a_missing_required_field_is_named():
    message = validate_tool_params("search_web", {})

    assert "missing required field: query" in message
    assert "search_web" in message


def test_the_message_lists_the_accepted_parameters():
    message = validate_tool_params("search_web", {})

    assert "Accepted parameters:" in message
    assert "query" in message


def test_a_valid_call_passes():
    assert validate_tool_params("search_web", {"query": "vectorization"}) is None


def test_an_empty_required_value_is_rejected():
    message = validate_tool_params("search_web", {"query": ""})

    assert "is empty" in message


def test_a_wrong_type_is_reported_with_what_arrived():
    message = validate_tool_params("search_web", {"query": 42})

    assert "should be string" in message
    assert "got int" in message


def test_booleans_do_not_satisfy_integer_fields():
    """bool subclasses int, so True would otherwise pass an integer check."""
    message = validate_tool_params(
        "compile_c_snippet", {"code": "int main(){}", "flags": True}
    )

    assert message is not None and "flags" in message


def test_runtime_bookkeeping_keys_are_ignored():
    assert (
        validate_tool_params(
            "search_web", {"query": "x", "_idempotency_key": "abc", "_depth": 1}
        )
        is None
    )


def test_unknown_tools_are_not_rejected_here():
    """This guards against malformed calls, not against undescribed tools."""
    assert validate_tool_params("no_such_tool", {"anything": 1}) is None


def test_extra_parameters_are_tolerated():
    assert validate_tool_params("search_web", {"query": "x", "limit": 5}) is None


@pytest.mark.parametrize("params", [None, {}, {"code": None}])
def test_missing_code_is_caught_for_the_compiler_tool(params):
    message = validate_tool_params("compile_c_snippet", params)

    assert message is not None and "code" in message


def test_an_arxiv_id_is_rejected_where_a_document_uuid_is_required():
    """The failure this replaces named neither the tool nor the field.

    Passing an arXiv id to summarize_document raised "badly formed hexadecimal
    UUID string" from deep inside the handler, and the agent retried it.
    """
    message = validate_tool_params("summarize_document", {"document_id": "2401.00001"})

    assert "field document_id should be a UUID" in message
    assert "2401.00001" in message
    assert "not an external identifier" in message


def test_a_real_uuid_passes():
    message = validate_tool_params(
        "summarize_document",
        {"document_id": "f873f8a1-842f-42d7-947e-5781eb997125"},
    )

    assert message is None


def test_a_field_that_promises_no_uuid_is_left_alone():
    """Only fields whose description promises a UUID are held to one."""
    message = validate_tool_params("search_web", {"query": "not-a-uuid"})

    assert message is None


def test_an_empty_list_where_a_string_is_wanted_means_nothing():
    """Three attempts across two live runs were refused for `run_args = []` --
    a call that wanted no arguments, said so, and was told its way of saying so
    was the wrong type. The one-item repair beside this did not cover it."""
    from app.services.agent_tool_validation import coerce_tool_params

    params = {"code": "int main(void){return 0;}", "run_args": []}
    repaired = coerce_tool_params("simulate_c_workload", params)

    assert params["run_args"] == ""
    assert "run_args" in repaired
