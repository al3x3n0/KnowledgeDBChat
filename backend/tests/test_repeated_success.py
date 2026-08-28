"""Noticing that a run already has the answer it is asking for again.

There was machinery for the same call failing repeatedly and none for it
succeeding repeatedly, which costs as much and is easier to miss because
nothing looks wrong. A live run asked to build on earlier work spent eight of
its nine iterations alternating two tools, every call identical to one it had
already made, every answer the same sixteen findings.
"""

from app.services import agent_repeated_success as repeated

CALL = {"tool": "recall_prior_findings", "params": {"subject": "L2 prefetcher"}}
OK = {"success": True, "findings": [1, 2]}
FAILED = {"success": False, "error": "boom"}


def _state(*entries):
    return {"actions_taken": list(entries)}


def _entry(action, result, iteration):
    return {"action": action, "result": result, "iteration": iteration}


class TestWhenItSpeaks:
    def test_the_first_call_is_not_a_repeat(self):
        assert repeated.analyze(CALL, OK, _state()) is None

    def test_the_second_identical_call_is_noted(self):
        note = repeated.analyze(CALL, OK, _state(_entry(CALL, OK, 1)))

        assert note["attempt"] == 2
        assert "already succeeded at iteration 1" in note["note"]

    def test_the_third_is_called_a_loop(self):
        note = repeated.analyze(
            CALL, OK, _state(_entry(CALL, OK, 1), _entry(CALL, OK, 2))
        )

        assert note["attempt"] == 3
        assert "is a loop" in note["note"]


class TestWhenItStaysQuiet:
    def test_different_arguments_are_a_different_question(self):
        other = {"tool": "recall_prior_findings", "params": {"subject": "branch"}}

        assert repeated.analyze(CALL, OK, _state(_entry(other, OK, 1))) is None

    def test_a_different_tool_is_not_a_repeat(self):
        other = {"tool": "get_job_history", "params": {"subject": "L2 prefetcher"}}

        assert repeated.analyze(CALL, OK, _state(_entry(other, OK, 1))) is None

    def test_retrying_after_a_failure_is_progress_not_a_loop(self):
        """The whole point of a retry is that the earlier call did not work.
        Calling that a repeat would scold a run for recovering."""
        assert repeated.analyze(CALL, OK, _state(_entry(CALL, FAILED, 1))) is None

    def test_a_call_that_failed_is_left_to_the_failure_diagnosis(self):
        assert repeated.analyze(CALL, FAILED, _state(_entry(CALL, OK, 1))) is None

    def test_a_label_does_not_make_two_calls_different(self):
        """label and reason vary between otherwise identical calls and say
        nothing about what was asked."""
        labelled = {
            "tool": "recall_prior_findings",
            "params": {"subject": "L2 prefetcher", "label": "second look"},
        }

        assert repeated.analyze(labelled, OK, _state(_entry(CALL, OK, 1))) is not None


class TestItDoesNotServeACachedAnswer:
    """Repeating a measurement is sometimes exactly right -- a second trial is
    a sample, not a duplicate, and this project's calibration rests on being
    able to take one. Short-circuiting would turn a re-measurement into a copy
    of the first."""

    def test_the_module_only_describes_and_never_returns_a_prior_result(self):
        note = repeated.analyze(CALL, OK, _state(_entry(CALL, OK, 1)))

        assert set(note) == {"attempt", "earlier_iterations", "note"}
        assert "findings" not in note

    def test_malformed_state_is_survivable(self):
        assert repeated.analyze(CALL, OK, None) is None
        assert repeated.analyze(CALL, OK, {"actions_taken": "nonsense"}) is None
        assert repeated.analyze(None, OK, _state()) is None
