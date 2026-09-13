"""Tests for the shared model-output JSON parser.

Before this module existed, subsystems disagreed about what counted as a
parseable reply: the decision parser recovered an object from a reply
containing two, the runners returned nothing for the same text, and two runners
called bare json.loads so any fenced reply failed outright. These tests pin the
tolerances everything now shares.
"""

from app.services.llm_json import extract_json_object


def test_parses_a_plain_object():
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_parses_a_fenced_object():
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object('```\n{"a": 1}\n```') == {"a": 1}


def test_parses_an_object_wrapped_in_prose():
    assert extract_json_object('Here is the plan:\n{"a": 1}\nHope that helps') == {
        "a": 1
    }


def test_parses_a_fenced_object_surrounded_by_chat():
    text = 'Sure!\n```json\n{"x": "y"}\n```\nLet me know if that works.'
    assert extract_json_object(text) == {"x": "y"}


def test_returns_the_first_object_when_a_reply_contains_several():
    # The runners used to drop this reply entirely; the decision parser did not.
    assert extract_json_object('{"a": 1} and {"b": 2}') == {"a": 1}
    assert extract_json_object('first {"a": 1}\nsecond {"b": 2}') == {"a": 1}


def test_a_brace_inside_a_string_does_not_end_the_object():
    assert extract_json_object('{"s": "text with } brace"}') == {
        "s": "text with } brace"
    }
    assert extract_json_object('{"s": "escaped \\" and } brace"}') == {
        "s": 'escaped " and } brace'
    }


def test_handles_nesting():
    assert extract_json_object('noise {"deep": {"x": [1, 2, {"y": "}"}]}} noise') == {
        "deep": {"x": [1, 2, {"y": "}"}]}
    }


def test_passes_through_an_already_parsed_dict():
    payload = {"already": "parsed"}
    assert extract_json_object(payload) is payload


def test_returns_none_when_there_is_no_object():
    for value in ("", "prose only, no json", "[1, 2, 3]", None, 42, ["a"]):
        assert extract_json_object(value) is None


def test_returns_none_for_malformed_json():
    assert extract_json_object('{"a": 1,}') is None
    assert extract_json_object('{"unclosed": ') is None


def test_prefers_the_whole_string_over_an_embedded_span():
    # A reply that is itself valid JSON must not be re-scanned for inner spans.
    assert extract_json_object('{"outer": {"inner": 1}}') == {"outer": {"inner": 1}}


def test_recovers_an_inner_object_when_the_outer_span_is_malformed():
    assert extract_json_object('{ bad {"a": 1} }') == {"a": 1}
    assert extract_json_object('{oops} {"a": 1}') == {"a": 1}


def test_scanning_stays_linear_on_malformed_input():
    """Guards against the quadratic scan this replaced.

    28KB of unbalanced braces took 37 seconds before, in a path that parses
    untrusted model output. A generous ceiling still catches a regression.
    """
    import time

    noisy = "text { " * 4000
    started = time.perf_counter()
    assert extract_json_object(noisy) is None
    assert time.perf_counter() - started < 1.0
