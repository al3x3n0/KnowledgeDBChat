"""Reading a test run well enough to gate on it.

The distinction this exists for: an exit code cannot tell "your change broke
three tests" from "the tests never ran". Those call for opposite responses —
one says fix the patch, the other says fix the harness — and a gate that reads
only the code sends the run to do the wrong thing.
"""

import pytest

from app.services.coding_test_gate import read_test_output

pytestmark = pytest.mark.unit


class TestReadingPytest:
    def test_a_green_run(self):
        outcome = read_test_output("collected 20 items\n\n20 passed in 3.4s\n", "", 0)
        assert outcome.ran is True
        assert outcome.green is True
        assert outcome.passed == 20
        assert outcome.framework == "pytest"

    def test_a_red_run_names_what_failed(self):
        stdout = (
            "collected 23 items\n"
            "FAILED tests/test_math.py::test_off_by_one\n"
            "FAILED tests/test_math.py::test_rounding\n"
            "2 failed, 21 passed in 4.1s\n"
        )
        outcome = read_test_output(stdout, "", 1)
        assert outcome.green is False
        assert outcome.failed == 2
        assert outcome.passed == 21
        # A gate that says "tests failed" and not which ones makes the run
        # guess where to look.
        assert "tests/test_math.py::test_off_by_one" in outcome.failing

    def test_errors_close_the_gate_as_firmly_as_failures(self):
        outcome = read_test_output("1 error in 0.2s\n", "", 1)
        assert outcome.errors == 1
        assert outcome.green is False

    def test_skips_do_not_close_the_gate(self):
        outcome = read_test_output("18 passed, 2 skipped in 1.1s\n", "", 0)
        assert outcome.skipped == 2
        assert outcome.green is True


class TestReadingJest:
    def test_it_reads_the_summary_line(self):
        stdout = (
            "  ● pipeline › refuses an unbounded loop\n"
            "Tests:       2 failed, 15 passed, 17 total\n"
        )
        outcome = read_test_output(stdout, "", 1)
        assert outcome.framework == "jest"
        assert outcome.failed == 2
        assert outcome.passed == 15
        assert outcome.green is False
        assert outcome.failing[0].startswith("pipeline")

    def test_a_green_jest_run(self):
        outcome = read_test_output("Tests:       41 passed, 41 total\n", "", 0)
        assert outcome.green is True
        assert outcome.passed == 41


class TestReadingGo:
    def test_failures_are_counted_by_line(self):
        outcome = read_test_output(
            "--- FAIL: TestParse (0.00s)\n--- FAIL: TestRound (0.00s)\nFAIL\n", "", 1
        )
        assert outcome.framework == "go"
        assert outcome.failed == 2
        assert outcome.green is False

    def test_ok_is_green(self):
        outcome = read_test_output("ok  \texample.com/pkg\t0.02s\n", "", 0)
        assert outcome.green is True


class TestTheDistinctionThatMatters:
    def test_a_harness_that_never_ran_is_not_a_failing_suite(self):
        # The case an exit code cannot express. pytest not installed exits
        # non-zero exactly like a broken test, and the response is completely
        # different: install the harness, do not touch the patch.
        outcome = read_test_output("", "bash: pytest: command not found\n", 127)
        assert outcome.ran is False
        assert outcome.green is False
        assert "cannot distinguish" in outcome.note

    def test_silence_is_not_success(self):
        # Exit code zero with no output at all. A gate reading only the code
        # would open here, which is the worst possible reading: it would let
        # an untested patch through believing it verified.
        outcome = read_test_output("", "", 0)
        assert outcome.ran is False
        assert outcome.green is False

    def test_the_evidence_says_both_things_separately(self):
        outcome = read_test_output("", "", 0)
        evidence = outcome.as_evidence()
        # `ran` and `green` are separate keys on purpose: a contract can then
        # require that the tests ran AND passed, rather than one word that
        # collapses the two.
        assert evidence["ran"] is False
        assert evidence["green"] is False
        assert evidence["exit_code"] == 0


class TestPerishableEvidenceIsRetaken:
    """A gate that inherits its own answer is not a gate.

    Pipeline stages normally inherit what upstream produced, so the planner
    does not charge twice for one measurement. That is right for a fact and
    wrong for a test run: a suite that was green before the patch describes a
    tree that no longer exists.
    """

    def test_a_verify_stage_reruns_the_tests(self):
        from app.services import agent_pipeline_spec as ps

        pipeline = ps.normalize(
            {
                "name": "fix-and-prove",
                "stages": [
                    {
                        "id": "clone",
                        "goal": "c",
                        "contract": {"required_finding_types": ["repo_workspace"]},
                    },
                    {
                        "id": "baseline",
                        "goal": "b",
                        "depends_on": ["clone"],
                        "assumes": ["repo_workspace"],
                        "contract": {"required_finding_types": ["test_result"]},
                    },
                    {
                        "id": "fix",
                        "goal": "f",
                        "depends_on": ["baseline"],
                        "assumes": ["repo_workspace"],
                        "contract": {"required_finding_types": ["patch_applied"]},
                    },
                    {
                        "id": "verify",
                        "goal": "v",
                        "depends_on": ["fix"],
                        "assumes": ["patch_applied"],
                        "contract": {"required_finding_types": ["test_result"]},
                    },
                ],
            }
        )
        assert ps.validate(pipeline) == []
        plan = ps.plan(pipeline)
        by_stage = {s.stage_id: s.tools for s in plan.stages}

        # Before `perishable`, verify inherited the baseline's test_result and
        # derived nothing at all — a gate on a measurement it never took.
        assert "run_repo_tests" in by_stage["verify"]
        # And it is priced, so a pipeline that verifies is not free.
        verify = next(s for s in plan.stages if s.stage_id == "verify")
        assert verify.seconds > 0

    def test_a_durable_fact_is_still_inherited(self):
        # The optimisation is right for evidence a later stage cannot
        # invalidate: cloning twice is waste, not rigour.
        from app.services import agent_pipeline_spec as ps

        pipeline = ps.normalize(
            {
                "name": "reuse",
                "stages": [
                    {
                        "id": "clone",
                        "goal": "c",
                        "contract": {"required_finding_types": ["repo_workspace"]},
                    },
                    {
                        "id": "search",
                        "goal": "s",
                        "depends_on": ["clone"],
                        "assumes": ["repo_workspace"],
                        "contract": {"required_finding_types": ["code_search_result"]},
                    },
                ],
            }
        )
        plan = ps.plan(pipeline)
        by_stage = {s.stage_id: s.tools for s in plan.stages}
        assert "clone_and_index_repo" not in by_stage["search"]
        assert "search_code" in by_stage["search"]
