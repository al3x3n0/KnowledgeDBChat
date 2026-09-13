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


class TestJsonColumnsRejectNaN:
    """A run died at iteration 10 of 14 with `invalid input syntax for type
    json: Token "NaN" is invalid` while checkpointing. Postgres JSON has no
    NaN and no Infinity; Python's json.dumps emits them happily, so the
    mismatch surfaces at INSERT, after the work is done. A plain gem5 run
    emits fourteen NaN statistics for averages whose denominator was zero, and
    those reach state through tool results."""

    def test_nan_becomes_null(self):
        from app.services.agent_checkpoint_service import strip_non_finite as json_safe

        assert json_safe(float("nan")) is None

    def test_infinity_becomes_null_too(self):
        from app.services.agent_checkpoint_service import strip_non_finite as json_safe

        assert json_safe(float("inf")) is None
        assert json_safe(float("-inf")) is None

    def test_it_reaches_inside_findings(self):
        from app.services.agent_checkpoint_service import strip_non_finite as json_safe

        state = {"findings": [{"speedup": float("nan"), "cycles": 1.5}]}

        assert json_safe(state) == {"findings": [{"speedup": None, "cycles": 1.5}]}

    def test_the_key_survives_rather_than_being_dropped(self):
        """A key vanishing changes the shape a resume reads back, and a
        missing average and an unmeasurable one are the same claim."""
        from app.services.agent_checkpoint_service import strip_non_finite as json_safe

        assert "speedup" in json_safe({"speedup": float("nan")})

    def test_the_result_is_what_postgres_will_take(self):
        import json

        from app.services.agent_checkpoint_service import strip_non_finite as json_safe

        payload = json_safe({"a": float("nan"), "b": [float("inf"), 2], "c": "ok"})

        assert json.dumps(payload, allow_nan=False)

    def test_ordinary_values_are_untouched(self):
        from app.services.agent_checkpoint_service import strip_non_finite as json_safe

        payload = {"n": 1, "f": 1.5, "s": "x", "b": True, "none": None, "l": [1, 2]}

        assert json_safe(payload) == payload
