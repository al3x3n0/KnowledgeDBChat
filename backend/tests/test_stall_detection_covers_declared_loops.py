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


class TestDuplicateFindingsAreNotProgress:
    """A repeated call defeated the stall detector by inflating the count.

    Measured on a verdict stage: it called `find_fusion_candidates` four times
    with the same arguments, appending the same candidate each time, and the
    round counts climbed 5, 6, 7, 8, 9. The detector read growth as progress
    and let it spend half its budget re-mining a candidate it already had --
    while `agent_repeated_success` was separately telling it, in the result,
    that the call was a repeat. Two mechanisms disagreeing, and the one that
    could have stopped it was the one being fooled.
    """

    def test_the_same_finding_twice_counts_once(self):
        same = {"type": "fusion_candidate", "title": "fmadd fsqrt fcmp"}

        assert policy.distinct_findings([same, dict(same), dict(same)]) == 1

    def test_different_findings_of_one_type_both_count(self):
        findings = [
            {"type": "fusion_candidate", "title": "fmadd fsqrt fcmp"},
            {"type": "fusion_candidate", "title": "ldr add str"},
        ]

        assert policy.distinct_findings(findings) == 2

    def test_subject_identifies_a_finding_with_no_title(self):
        findings = [
            {"type": "benchmark_measurement", "subject": "kernel_a"},
            {"type": "benchmark_measurement", "subject": "kernel_b"},
            {"type": "benchmark_measurement", "subject": "kernel_a"},
        ]

        assert policy.distinct_findings(findings) == 2

    def test_a_repeated_round_now_reads_as_dry(self):
        """The behaviour that matters: four identical candidates in a row is a
        stalled run, and the counts must say so."""
        state = {"findings": []}
        candidate = {"type": "fusion_candidate", "title": "fmadd fsqrt fcmp"}
        for _ in range(5):
            state["findings"].append(dict(candidate))
            policy.record_round(state, len(state["findings"]))

        assert state["loop_finding_counts"] == [1, 1, 1, 1, 1]
        stop, reason = policy.should_stop(
            {"loop_until": "no_new_findings", "loop_dry_rounds": 2}, state
        )
        assert stop is True

    def test_real_progress_still_counts(self):
        """The control. A run establishing different things each round must
        never be stopped by this."""
        state = {"findings": []}
        for i in range(5):
            state["findings"].append({"type": "fusion_candidate", "title": f"pair {i}"})
            policy.record_round(state, len(state["findings"]))

        assert state["loop_finding_counts"] == [1, 2, 3, 4, 5]
        assert policy.should_stop({"loop_until": "no_new_findings"}, state)[0] is False

    def test_a_state_with_no_findings_uses_the_callers_number(self):
        """A caller that passes a count without findings in state knows
        something this function does not; it is not overruled."""
        state = {}
        policy.record_round(state, 7)

        assert state["loop_finding_counts"] == [7]
