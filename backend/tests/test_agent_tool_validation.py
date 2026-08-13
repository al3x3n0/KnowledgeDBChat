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
