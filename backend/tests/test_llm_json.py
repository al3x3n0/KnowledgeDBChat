"""Tests for the shared model-output JSON parser.

Before this module existed, subsystems disagreed about what counted as a
parseable reply: the decision parser recovered an object from a reply
containing two, the runners returned nothing for the same text, and two runners
called bare json.loads so any fenced reply failed outright. These tests pin the
tolerances everything now shares.
"""

from app.services import llm_json
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


# ---------------------------------------------------------------------- arrays


def test_parses_a_plain_array():
    assert llm_json.extract_json_array('[{"a": 1}]') == [{"a": 1}]


def test_parses_a_fenced_array():
    assert llm_json.extract_json_array("```json\n[1, 2]\n```") == [1, 2]


def test_parses_an_array_wrapped_in_prose():
    reply = 'Here are the calls: [{"tool_name": "x"}] -- done.'
    assert llm_json.extract_json_array(reply) == [{"tool_name": "x"}]


def test_an_empty_array_is_an_answer_not_a_failure():
    """ "No tools needed" is `[]`, which must not be confused with a reply
    that could not be parsed."""
    assert llm_json.extract_json_array("[]") == []
    assert llm_json.extract_json_array("no json here") is None


def test_a_bracket_inside_a_string_does_not_end_the_array():
    assert llm_json.extract_json_array('x ["a]b", "c"] y') == ["a]b", "c"]


def test_an_object_is_not_an_array_and_the_reverse():
    assert llm_json.extract_json_array('{"a": [1]}') == [1]
    assert llm_json.extract_json_object("[1, 2]") is None
    assert llm_json.extract_json_array([1]) == [1]


def test_array_scanning_stays_linear_on_malformed_input():
    import time

    started = time.perf_counter()
    assert llm_json.extract_json_array("[" * 28000) is None
    assert time.perf_counter() - started < 2


def test_every_helper_a_caller_uses_exists():
    """`extract_json_array` was deleted while two callers still used it. Both
    wrapped the call in `except Exception`, so nothing raised: the chat planner
    dropped every tool call for eight weeks and logged a line nobody read.

    A missing attribute is invisible to a test that only exercises the helper,
    so this reads the callers.
    """
    import re
    from pathlib import Path

    app = Path(__file__).resolve().parents[1] / "app"
    used = set()
    for path in app.rglob("*.py"):
        used |= set(re.findall(r"\bllm_json\.([A-Za-z_]\w*)", path.read_text()))
    assert used, "found no callers; this guard would pass vacuously"
    missing = sorted(name for name in used if not hasattr(llm_json, name))
    assert not missing, f"called but not defined in llm_json: {missing}"


def test_the_chat_planner_keeps_the_calls_the_model_made():
    """The path a caller takes, not the helper: this is where it was lost."""
    from app.services.agent_service import AgentService

    calls = AgentService()._parse_tool_calls(
        '[{"tool_name": "search_documents", "tool_input": {"query": "gem5"}}]'
    )
    assert [c.tool_name for c in calls] == ["search_documents"]
    assert calls[0].tool_input == {"query": "gem5"}


def test_deep_nesting_is_unparseable_not_a_crash():
    """The decoder recurses per bracket and raises RecursionError, which is
    not a ValueError: a hostile or runaway reply took the caller down."""
    assert llm_json.extract_json_array("[" * 28000 + "]" * 28000) is None
    assert extract_json_object('{"a":' * 28000 + "1" + "}" * 28000) is None


# ------------------------------------------------- one parser, used everywhere


def test_nothing_else_digs_json_out_of_a_reply_by_hand():
    """About twenty places found the JSON in a model's reply themselves, with
    `text.find("{")` to `text.rfind("}")` or a greedy `\\{.*\\}`. They did not
    agree: a reply holding two objects parsed in one service and failed in
    another, a fenced reply was handled by some and not others, and one of
    them needed raw newlines accepted and so was the only one that did.

    The idiom is refused here, so the next parser is this one.
    """
    import re
    from pathlib import Path

    app = Path(__file__).resolve().parents[1] / "app"
    idioms = re.compile(r"""rfind\(\s*["'][}\]]["']\s*\)|\\\{\.\*\\\}|\\\[\.\*\\\]""")
    offenders = []
    for path in sorted(app.rglob("*.py")):
        if path.name == "llm_json.py":
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if idioms.search(line):
                offenders.append(
                    f"{path.relative_to(app)}:{number}: {line.strip()[:70]}"
                )
    assert not offenders, (
        "Use llm_json.extract_json_object / extract_json_array / "
        "completion_object instead:\n" + "\n".join(offenders)
    )


class _Completion:
    def __init__(self, text="", structured=None):
        self.text = text
        self.structured = structured


class TestTheObjectOutOfACompletion:
    def test_native_structured_output_wins(self):
        completion = _Completion(text='{"a": 2}', structured={"a": 1})
        assert llm_json.completion_object(completion) == {"a": 1}

    def test_json_left_in_the_text_is_found(self):
        assert llm_json.completion_object(_Completion('{"a": 1}')) == {"a": 1}
        assert llm_json.completion_object(_Completion('```json\n{"a": 1}\n```')) == {
            "a": 1
        }
        assert llm_json.completion_object(_Completion('Sure: {"a": 1}. Done.')) == {
            "a": 1
        }

    def test_a_mapping_and_a_bare_string_are_accepted(self):
        assert llm_json.completion_object({"a": 1}) == {"a": 1}
        assert llm_json.completion_object('{"a": 1}') == {"a": 1}

    def test_nothing_usable_is_an_empty_object_not_an_error(self):
        assert llm_json.completion_object(_Completion("not json at all")) == {}
        assert llm_json.completion_object(_Completion("")) == {}
        assert llm_json.completion_object(None) == {}
        # An array is not an object.
        assert llm_json.completion_object(_Completion("[1, 2]")) == {}

    def test_an_empty_structured_falls_back_to_the_text(self):
        """Providers without schema output leave `structured` empty."""
        completion = _Completion(text='{"a": 1}', structured={})
        assert llm_json.completion_object(completion) == {"a": 1}


def test_raw_newlines_in_a_string_need_strict_false():
    """A reply carrying a source file in a JSON string is written with literal
    newlines as often as with an escape. Strict JSON rejects them."""
    reply = '{"code": "int f() {\n\treturn 1;\n}"}'
    assert extract_json_object(reply) is None
    assert extract_json_object(reply, strict=False) == {
        "code": "int f() {\n\treturn 1;\n}"
    }
    assert llm_json.extract_json_array('["a\nb"]', strict=False) == ["a\nb"]


def test_require_raises_with_the_callers_own_message():
    import pytest

    assert llm_json.require_json_object('x {"a": 1} y') == {"a": 1}
    with pytest.raises(ValueError, match="No JSON object found in LLM response"):
        llm_json.require_json_object("nothing", "No JSON object found in LLM response")


def test_every_drafter_parses_through_the_shared_helper():
    """Four drafters each carried a copy of `_payload`. A fix to one -- an
    empty `structured`, a reply wrapped in a sentence -- did not reach the
    others."""
    from app.services import (
        agent_campaign_intent,
        agent_definition_author_service,
        agent_restructure_proposer,
        plugin_author_service,
    )

    completion = _Completion('Here you go: {"name": "x"}', structured={})
    for module in (
        agent_campaign_intent,
        agent_definition_author_service,
        agent_restructure_proposer,
        plugin_author_service,
    ):
        assert module._payload(completion) == {"name": "x"}, module.__name__
