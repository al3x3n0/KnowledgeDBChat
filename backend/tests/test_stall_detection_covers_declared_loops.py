"""`until` says when a stage is DONE. Dry rounds say when it is STUCK.

Declaring the first used to silence the second. Measured live: an implement
stage declaring `until: contract_satisfied` got a real diagnostic at iteration
4 -- "the program exited 3 on this input" -- and then spent iterations 5 to 10
on ten consecutive reads and progress reports. No code written, no checks run,
nothing new recorded, and nothing in place to notice, because declaring any
loop at all bought the stage out of stall detection entirely.

A stage going in circles is stuck whichever way its author wrote the success
condition.
"""

import pytest

from app.services import agent_loop_policy as policy

pytestmark = pytest.mark.unit


def _stuck(rounds):
    """A history with `rounds` observations and no growth."""
    return {"loop_finding_counts": [7] * rounds}


class TestADeclaredLoopIsStillWatched:
    def test_contract_satisfied_no_longer_silences_stall_detection(self):
        config = {"loop_until": "contract_satisfied"}

        stop, reason = policy.should_stop(
            config, _stuck(policy.DECLARED_LOOP_DRY_ROUNDS + 1)
        )

        assert stop is True
        assert "no new findings" in reason

    def test_the_reason_says_why_a_declared_until_did_not_save_it(self):
        config = {"loop_until": "contract_satisfied"}

        _, reason = policy.should_stop(
            config, _stuck(policy.DECLARED_LOOP_DRY_ROUNDS + 1)
        )

        assert "contract_satisfied" in reason
        assert "not whether it is getting anywhere" in reason

    def test_a_declared_loop_gets_more_rope_than_the_default(self):
        """Its author chose a different success condition, so being stopped
        this way is not what they asked for."""
        assert policy.DECLARED_LOOP_DRY_ROUNDS > policy.DEFAULT_DRY_ROUNDS

        config = {"loop_until": "contract_satisfied"}
        # Enough rounds to stop a default stage, not enough for a declared one.
        stop, _ = policy.should_stop(config, _stuck(policy.DEFAULT_DRY_ROUNDS + 1))

        assert stop is False


class TestTheOriginalBehaviourIsIntact:
    def test_no_new_findings_still_stops_at_its_own_threshold(self):
        config = {"loop_until": "no_new_findings", "loop_dry_rounds": 2}

        stop, _ = policy.should_stop(config, _stuck(3))

        assert stop is True

    def test_progress_is_never_stopped(self):
        """The control that matters most: a stage still recording findings
        must run, whatever policy it declared."""
        for declared in ("contract_satisfied", "no_new_findings", ""):
            config = {"loop_until": declared}
            progressing = {"loop_finding_counts": [1, 2, 3, 4, 5, 6, 7, 8]}

            stop, _ = policy.should_stop(config, progressing)

            assert stop is False, declared

    def test_too_little_history_to_judge_is_not_a_stall(self):
        config = {"loop_until": "contract_satisfied"}

        stop, _ = policy.should_stop(config, _stuck(2))

        assert stop is False

    def test_an_author_may_still_set_their_own_threshold(self):
        config = {"loop_until": "contract_satisfied", "loop_dry_rounds": 9}

        assert policy.should_stop(config, _stuck(6))[0] is False
        assert policy.should_stop(config, _stuck(10))[0] is True
