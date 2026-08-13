"""A failed tool must carry its reason at the top level.

A sweep of the catalog found fourteen data-analysis tools reporting
{"success": False, "data": {"error": ...}} while the rest report a top-level
"error". Recovery, the fallback suggestion and every reader key off the
top-level field, so those failures arrived with no reason attached.
"""

from app.services.agent_action_service import _surface_nested_error


def test_a_nested_reason_is_lifted():
    result = {
        "tool": "describe_dataset",
        "success": False,
        "data": {"success": False, "error": "Dataset 'x' not found"},
    }

    _surface_nested_error(result)

    assert result["error"] == "Dataset 'x' not found"


def test_an_existing_top_level_reason_is_left_alone():
    result = {"success": False, "error": "original", "data": {"error": "nested"}}

    _surface_nested_error(result)

    assert result["error"] == "original"


def test_a_successful_result_is_untouched():
    result = {"success": True, "data": {"error": "not a real failure"}}

    _surface_nested_error(result)

    assert "error" not in result


def test_results_without_a_nested_reason_are_untouched():
    for result in ({"success": False}, {"success": False, "data": "text"}, {}):
        before = dict(result)
        _surface_nested_error(result)
        assert result == before


def test_a_very_long_reason_is_clipped():
    result = {"success": False, "data": {"error": "x" * 2000}}

    _surface_nested_error(result)

    assert len(result["error"]) <= 500
