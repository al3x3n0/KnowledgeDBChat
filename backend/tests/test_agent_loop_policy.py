"""When a looping stage stops, beyond running out of iterations.

A patch-and-test loop is the case this exists for. A run that patches, tests,
reads the failures and patches again is working; a run that produces nothing
new two rounds running is stuck, and the iterations it has left will go the
same way.

`loop_until` and `loop_dry_rounds` were written into the job config by the
pipeline binding and read by nothing — documented options that silently did
nothing, which is worse than not offering them.
"""

import pytest

from app.services import agent_loop_policy as policy

pytestmark = pytest.mark.unit


def _state(counts):
    state = {}
    for n in counts:
        policy.record_round(state, n)
    return state


class TestNoNewFindings:
    def test_a_productive_loop_keeps_going(self):
        # Each round established something: 2, then 4, then 5.
        stop, _ = policy.should_stop(
            {"loop_until": "no_new_findings"}, _state([2, 4, 5])
        )
        assert stop is False

    def test_two_dry_rounds_stop_it(self):
        stop, reason = policy.should_stop(
            {"loop_until": "no_new_findings"}, _state([3, 3, 3])
        )
        assert stop is True
        # The reason goes in the job log, so it has to say what happened.
        assert "no new findings" in reason
        assert "3" in reason

    def test_one_dry_round_is_not_enough(self):
        # A round can come up empty while the agent reads context. Cutting the
        # run off there would stop exactly the patient work that lands.
        stop, _ = policy.should_stop({"loop_until": "no_new_findings"}, _state([3, 3]))
        assert stop is False

    def test_a_late_finding_rescues_the_run(self):
        stop, _ = policy.should_stop(
            {"loop_until": "no_new_findings"}, _state([3, 3, 3, 4])
        )
        assert stop is False

    def test_the_window_is_configurable(self):
        # Three dry rounds asked for, three dry rounds given.
        config = {"loop_until": "no_new_findings", "loop_dry_rounds": 3}
        assert policy.should_stop(config, _state([2, 2, 2]))[0] is False
        assert policy.should_stop(config, _state([2, 2, 2, 2]))[0] is True

    def test_a_nonsense_window_falls_back_rather_than_stopping_at_once(self):
        # dry_rounds=0 would stop the run before it did anything.
        config = {"loop_until": "no_new_findings", "loop_dry_rounds": 0}
        assert policy.should_stop(config, _state([1, 1]))[0] is False
        assert policy.should_stop(config, _state([1, 1, 1]))[0] is True

    def test_an_early_run_is_never_dry(self):
        # Nothing to compare against yet.
        for counts in ([], [0], [0, 0]):
            assert (
                policy.should_stop({"loop_until": "no_new_findings"}, _state(counts))[0]
                is False
            )


class TestOtherPolicies:
    def test_contract_satisfied_defers_to_the_executor(self):
        # The executor already refuses to stop on an unmet contract; this
        # policy has nothing to add and must not stop the run itself.
        stop, _ = policy.should_stop(
            {"loop_until": "contract_satisfied"}, _state([3, 3, 3])
        )
        assert stop is False

    def test_no_policy_means_no_extra_stopping(self):
        assert policy.should_stop({}, _state([3, 3, 3]))[0] is False
        assert policy.should_stop(None, None)[0] is False

    def test_an_unknown_policy_is_reported_rather_than_guessed(self):
        # Silently picking a policy the author did not choose is how a run
        # stops for a reason nobody can explain.
        config = {"loop_until": "whenever_ready"}
        assert policy.should_stop(config, _state([3, 3, 3]))[0] is False
        warning = policy.policy_warning(config)
        assert "whenever_ready" in warning
        assert "no_new_findings" in warning

    def test_a_known_policy_warns_about_nothing(self):
        assert policy.policy_warning({"loop_until": "no_new_findings"}) == ""
        assert policy.policy_warning({}) == ""


class TestRecordingRounds:
    def test_it_keeps_only_a_recent_tail(self):
        # The history rides in the job state, which is serialised every
        # iteration, so it cannot grow without bound.
        state = _state(range(50))
        assert len(state["loop_finding_counts"]) <= 20

    def test_it_survives_a_state_that_has_the_wrong_shape(self):
        state = {"loop_finding_counts": "not a list"}
        policy.record_round(state, 3)
        assert state["loop_finding_counts"] == [3]
