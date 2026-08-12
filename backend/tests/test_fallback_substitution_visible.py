"""A substituted tool must never look like the requested one succeeding.

When a tool fails, the action service may run a fallback and rewrite the result
to success. The result keeps primary_tool/primary_error, but every summary read
downstream kept only the requested tool's name and a success flag. That is how a
broken arXiv search was recorded as seven successful findings that were really
unrelated knowledge-base documents.
"""

from types import SimpleNamespace

from app.services.agent_job_transcript_service import build_actions
from app.services.autonomous_rnd_trajectory_service import (
    autonomous_rnd_trajectory_adapter as adapter,
)


def _substituted_action():
    return {
        "iteration": 1,
        "node": "act",
        "action": {"tool": "search_arxiv", "purpose": "find papers"},
        "result": {
            "success": True,
            "tool": "search_documents",
            "primary_tool": "search_arxiv",
            "primary_error": "'ArxivSearchResult' object is not subscriptable",
            "note": "Primary tool failed; used fallback tool: search_documents",
            "findings": [{"title": "Unrelated KB doc"}],
        },
    }


def _plain_action():
    return {
        "iteration": 1,
        "node": "act",
        "action": {"tool": "search_documents", "purpose": "search the KB"},
        "result": {"success": True, "tool": "search_documents", "findings": []},
    }


def test_ledger_marks_a_substituted_tool():
    row = adapter.compact_action_ledger([_substituted_action()])[0]

    assert row["substituted"] is True
    assert row["requested_tool"] == "search_arxiv"
    assert row["executed_tool"] == "search_documents"
    assert "not subscriptable" in row["primary_error"]


def test_ledger_leaves_an_ordinary_action_unmarked():
    row = adapter.compact_action_ledger([_plain_action()])[0]

    assert "substituted" not in row
    assert "requested_tool" not in row


def test_transcript_marks_a_substituted_tool():
    checkpoint = SimpleNamespace(state={"actions_taken": [_substituted_action()]})

    action = build_actions([checkpoint])[0]

    assert action["substituted"] is True
    assert action["requested_tool"] == "search_arxiv"
    assert action["executed_tool"] == "search_documents"


def test_transcript_leaves_an_ordinary_action_unmarked():
    checkpoint = SimpleNamespace(state={"actions_taken": [_plain_action()]})

    action = build_actions([checkpoint])[0]

    assert "substituted" not in action


def test_a_substituted_row_is_distinguishable_from_a_real_success():
    """The property that matters: the two rows must not be equal."""
    substituted = adapter.compact_action_ledger([_substituted_action()])[0]
    genuine = adapter.compact_action_ledger([_plain_action()])[0]

    assert substituted != genuine
    assert substituted.get("substituted") and not genuine.get("substituted")


def test_ledger_says_when_it_dropped_earlier_actions():
    """A truncated ledger reads as the whole run, and the beginning is where a
    trajectory usually goes wrong."""
    actions = [
        {
            "iteration": i,
            "action": {"tool": f"tool_{i}"},
            "result": {"success": True, "tool": f"tool_{i}"},
        }
        for i in range(10)
    ]

    ledger = adapter.compact_action_ledger(actions, max_actions=4)

    assert ledger[0]["tool"] == "__omitted__"
    assert "6 earlier action(s) omitted" in ledger[0]["note"]
    assert [row["tool"] for row in ledger[1:]] == [
        "tool_6",
        "tool_7",
        "tool_8",
        "tool_9",
    ]


def test_ledger_adds_no_marker_when_nothing_was_dropped():
    actions = [
        {"iteration": 1, "action": {"tool": "t"}, "result": {"success": True}},
    ]

    ledger = adapter.compact_action_ledger(actions, max_actions=200)

    assert all(row["tool"] != "__omitted__" for row in ledger)
