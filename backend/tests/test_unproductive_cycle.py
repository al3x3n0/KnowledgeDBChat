"""A run going in circles through calls that all succeed.

Every case here is taken from a live job that produced every finding its
contract named, then spent its remaining iterations alternating
`write_progress_report` and `recall_prior_findings` -- the reports carrying
different text each time so the identical-call detector never matched them,
the recalls returning the same ten findings so the raw count grew while the
claims did not (101 findings, 24 distinct).
"""

from __future__ import annotations

import pytest

from app.services import agent_unproductive_cycle as cycle


def _act(tool, *, success=True, findings=None):
    return {
        "action": {"tool": tool, "params": {}},
        "result": {
            "success": success,
            "findings": findings if findings is not None else [],
        },
    }


def _doc(n):
    return {"type": "document", "title": f"doc {n}", "id": f"id{n}"}


class TestWhatCountsAsGoingNowhere:
    def test_the_measured_loop_is_caught(self):
        """Reports alternating with recalls of findings already in hand."""
        docs = [_doc(i) for i in range(10)]
        state = {
            "actions_taken": [
                _act("recall_prior_findings", findings=docs),  # first one is real
                _act("write_progress_report"),
                _act("recall_prior_findings", findings=docs),  # same ten again
                _act("write_progress_report"),
                _act("recall_prior_findings", findings=docs),
                _act("write_progress_report"),
            ]
        }
        assert len(cycle.streak(state)) >= cycle.NOTE_AT

    def test_the_first_recall_is_credited_not_blamed(self):
        """Bringing findings into view is work; doing it again is not."""
        docs = [_doc(i) for i in range(3)]
        state = {
            "actions_taken": [
                _act("write_progress_report"),
                _act("recall_prior_findings", findings=docs),
            ]
        }
        assert cycle.streak(state) == []

    def test_measurement_work_is_never_a_loop(self):
        """Four simulations produce no findings either, and are exactly what a
        careful run should be doing."""
        state = {"actions_taken": [_act("simulate_c_workload") for _ in range(6)]}
        assert cycle.streak(state) == []
        assert cycle.analyze(state) is None

    def test_a_failure_breaks_the_streak(self):
        """A run hitting errors is not idling; that is the other detector."""
        state = {
            "actions_taken": [
                _act("write_progress_report"),
                _act("write_progress_report"),
                _act("search_documents", success=False),
                _act("write_progress_report"),
            ]
        }
        assert len(cycle.streak(state)) == 1

    def test_writing_the_answer_breaks_the_streak(self):
        """set_output_schema emits no finding and is the productive act when
        the contract wants a result key."""
        state = {
            "actions_taken": [
                _act("write_progress_report"),
                _act("write_progress_report"),
                _act("set_output_schema"),
            ]
        }
        assert cycle.streak(state) == []

    def test_an_unknown_tool_is_not_accused(self):
        """A plugin tool this catalog does not carry is treated as productive."""
        state = {"actions_taken": [_act("p_thing_do") for _ in range(6)]}
        assert cycle.streak(state) == []


class TestWhatItTellsTheRun:
    def _note(self, n, missing=()):
        state = {"actions_taken": [_act("write_progress_report") for _ in range(n)]}
        return cycle.analyze(state, missing=missing)

    def test_below_the_threshold_it_says_nothing(self):
        assert self._note(cycle.NOTE_AT - 1) is None

    def test_it_names_the_tools_and_why_they_cannot_help(self):
        note = self._note(cycle.NOTE_AT)["note"]
        assert "write_progress_report" in note
        assert "no evidence" in note

    def test_it_names_the_way_out_when_it_knows_it(self):
        """A run told it is looping and not told the remedy has been given the
        same information twice."""
        note = self._note(cycle.NOTE_AT, missing=["result_key:structured_output"])[
            "note"
        ]
        assert "result_key:structured_output" in note

    def test_it_turns_imperative_when_the_loop_persists(self):
        assert "Stop reporting" in self._note(cycle.DIRECTIVE_AT)["note"]
        assert "Stop reporting" not in self._note(cycle.NOTE_AT)["note"]

    def test_malformed_state_is_tolerated(self):
        for bad in (None, {}, {"actions_taken": "nope"}, {"actions_taken": [None]}):
            assert cycle.analyze(bad) is None
