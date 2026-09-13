"""Checking an implementation before anyone times it.

The failure this guards: the fastest implementation of any algorithm is one
that returns garbage. A benchmark of unverified code is a perfectly accurate
number for work nobody checked, and it looks better than a correct result
because the broken version is usually the fast one.
"""

import pytest

from app.services import agent_implementation_check as impl

pytestmark = pytest.mark.unit


class TestComparingOneOutput:
    def test_numbers_match_within_tolerance(self):
        # An algorithm rebuilt from prose reassociates its sums; demanding the
        # last bits rejects correct implementations.
        assert impl.compare_output("3.14159265", "3.14159270").passed is True

    def test_numbers_outside_tolerance_do_not(self):
        result = impl.compare_output("3.2", "3.14159")
        assert result.passed is False
        assert "expected 3.14159" in result.detail

    def test_a_count_mismatch_says_so(self):
        # Printing three numbers where four were expected is a real failure,
        # and naming it beats "output does not match".
        result = impl.compare_output("1 2 3", "1 2 3 4")
        assert result.passed is False
        assert "Expected 4 numbers" in result.detail

    def test_whitespace_is_never_the_difference(self):
        assert impl.compare_output(" 1  2\n3 ", "1 2 3").passed is True

    def test_text_output_falls_back_to_text(self):
        assert impl.compare_output("SORTED", "sorted").passed is False
        assert impl.compare_output("sorted\n", " sorted ").passed is True

    def test_tolerance_is_relative_so_it_means_the_same_at_any_scale(self):
        assert impl.compare_output("1e9", "1000000001").passed is True
        assert impl.compare_output("2e9", "1000000001").passed is False


class TestZeroCasesIsNotVerification:
    """The vacuous-truth hole, which is the worst way this could fail.

    Zero passing out of zero is true under any naive counting, so a check with
    no cases would certify untested code -- and certify it as *verified*, which
    is worse than never checking.
    """

    def test_no_cases_is_unverified(self):
        check = impl.ImplementationCheck(ran=True, cases=[])
        assert check.verified is False

    def test_cases_without_an_expected_output_are_dropped(self):
        # A case with nothing to compare against cannot pass or fail; keeping
        # it would inflate cases_run with checks that check nothing.
        cases = impl.normalize_cases(
            [
                {"name": "real", "input": "1", "expected_output": "1"},
                {"name": "empty", "input": "2"},
                {"name": "blank", "input": "3", "expected_output": "   "},
            ]
        )
        assert [c["name"] for c in cases] == ["real"]

    def test_nonsense_cases_are_dropped_not_guessed_at(self):
        assert impl.normalize_cases(None) == []
        assert impl.normalize_cases(["not a dict"]) == []

    def test_the_case_list_is_capped(self):
        many = [{"expected_output": str(n)} for n in range(100)]
        assert len(impl.normalize_cases(many)) == impl.MAX_CASES


class TestRanAndPassedAreDifferentFacts:
    def test_all_cases_passing_verifies(self):
        check = impl.ImplementationCheck(
            ran=True,
            cases=[
                impl.CaseResult(name="a", passed=True),
                impl.CaseResult(name="b", passed=True),
            ],
        )
        assert check.verified is True
        assert check.cases_passed == 2

    def test_one_failure_is_enough_to_refuse(self):
        check = impl.ImplementationCheck(
            ran=True,
            cases=[
                impl.CaseResult(name="a", passed=True),
                impl.CaseResult(name="b", passed=False, detail="wrong"),
            ],
        )
        assert check.verified is False
        assert check.as_evidence()["failing"][0]["name"] == "b"

    def test_code_that_never_compiled_is_not_a_wrong_answer(self):
        # A different failure with a different fix: repair the build, do not
        # go looking for a bug in the algorithm.
        check = impl.ImplementationCheck(ran=False, compile_error="undefined ref")
        assert check.verified is False
        assert check.cases_run == 0

    def test_a_case_that_produced_no_output_did_not_run(self):
        check = impl.ImplementationCheck(
            ran=True,
            cases=[
                impl.CaseResult(name="a", passed=True),
                impl.CaseResult(name="b", passed=False, ran=False),
            ],
        )
        # cases_run counts what actually executed. Comparing passed against
        # run would make this pass -- 1 of 1 -- because the case that never
        # ran drops out of both sides. Both are counted against the cases
        # supplied instead.
        assert check.cases_run == 1
        assert check.verified is False


class TestAttributingOutputToCases:
    def test_each_case_gets_its_own_lines(self):
        stdout = (
            "__case_begin__ 0\n42\n__case_exit__ 0 0\n"
            "__case_begin__ 1\n7\n8\n__case_exit__ 1 0\n"
        )
        outputs = impl._split_case_output(stdout, 2)
        assert outputs[0] == ("42", 0)
        assert outputs[1] == ("7\n8", 0)

    def test_a_nonzero_exit_is_carried_through(self):
        outputs = impl._split_case_output("__case_begin__ 0\n__case_exit__ 0 139\n", 1)
        assert outputs[0][1] == 139


class TestTheSandboxPath:
    @pytest.mark.asyncio
    async def test_no_usable_cases_never_reaches_the_sandbox(self):
        # And says why, rather than reporting a mysterious pass.
        check = await impl.check_implementation(code="int main(){}", cases=[])
        assert check.verified is False
        assert check.ran is False
        assert "vacuously" in check.note

    @pytest.mark.asyncio
    async def test_unsafe_flags_are_refused(self, monkeypatch):
        from app.services import agent_compiler_sandbox as sandbox

        monkeypatch.setattr(sandbox, "_execution_enabled", lambda: True)
        monkeypatch.setattr(sandbox, "_allowed_images", lambda: {sandbox.DEFAULT_IMAGE})
        check = await impl.check_implementation(
            code="int main(){return 0;}",
            cases=[{"expected_output": "0"}],
            flags="-O2; rm -rf /",
        )
        assert check.ran is False
        assert "unsupported characters" in check.note


class TestLanguages:
    """Rust is a first-class implementation language, not a translation step."""

    @pytest.mark.asyncio
    async def test_an_unknown_language_is_refused_rather_than_built_as_c(self):
        check = await impl.check_implementation(
            code="package main",
            cases=[{"expected_output": "1"}],
            language="go",
        )
        assert check.ran is False
        assert check.verified is False
        assert "Unsupported language" in check.note

    @pytest.mark.asyncio
    async def test_the_language_rides_on_the_evidence(self, monkeypatch):
        # A later stage reading "verified" has to know what was verified: a
        # Rust implementation and a C one are different programs, and a
        # benchmark stage that assumes the wrong one compiles the source with
        # the wrong compiler.
        from app.services import agent_compiler_sandbox as sandbox

        monkeypatch.setattr(sandbox, "_execution_enabled", lambda: True)
        monkeypatch.setattr(sandbox, "_allowed_images", lambda: {sandbox.DEFAULT_IMAGE})

        captured = {}

        async def fake_run(script, workdir, *, image, timeout_seconds):
            captured["script"] = script
            return 0, "__case_begin__ 0\n42\n__case_exit__ 0 0\n", ""

        monkeypatch.setattr(sandbox, "_run", fake_run)
        check = await impl.check_implementation(
            code='fn main(){println!("42");}',
            cases=[{"expected_output": "42"}],
            language="rust",
        )
        assert check.verified is True
        assert check.as_evidence()["language"] == "rust"
        # And it really built it as Rust.
        assert "rustc" in captured["script"]
        assert "prog.rs" in captured["script"]

    @pytest.mark.asyncio
    async def test_rust_gets_its_own_default_flags(self, monkeypatch):
        # Passing C's -O2 to rustc fails outright, so the default cannot be
        # shared between languages.
        from app.services import agent_compiler_sandbox as sandbox

        monkeypatch.setattr(sandbox, "_execution_enabled", lambda: True)
        monkeypatch.setattr(sandbox, "_allowed_images", lambda: {sandbox.DEFAULT_IMAGE})

        captured = {}

        async def fake_run(script, workdir, *, image, timeout_seconds):
            captured["script"] = script
            return 0, "__case_begin__ 0\n1\n__case_exit__ 0 0\n", ""

        monkeypatch.setattr(sandbox, "_run", fake_run)
        await impl.check_implementation(
            code="fn main(){}", cases=[{"expected_output": "1"}], language="rust"
        )
        assert "-O2" not in captured["script"]
        assert "-C linker=clang" in captured["script"]


class TestAMistakeInTheCallIsReportedAsOne:
    """The difference between "you called this wrong" and "your code is wrong".

    A check reporting `verified: false` is a SUCCESSFUL tool call, and the
    repeat-failure diagnosis only looks at failures. So an agent that called
    the gate with no reference cases got a polite note and no escalation, and
    one run made that identical mistake three times running -- three
    iterations of its budget spent on a correction nothing was pressing it to
    make.

    Supplying nothing to check against is a mistake in the call. Cases that
    ran and failed are a fact about the implementation, and must stay a
    success, or the gate would hide the very thing it exists to report.
    """

    @pytest.mark.asyncio
    async def test_the_reason_says_which_kind_of_not_run_it_was(self):
        check = await impl.check_implementation(code="int main(){}", cases=[])
        assert check.reason == "no_cases"
        assert check.as_evidence()["reason"] == "no_cases"

    @pytest.mark.asyncio
    async def test_an_unknown_language_is_also_a_call_mistake(self):
        check = await impl.check_implementation(
            code="x", cases=[{"expected_output": "1"}], language="go"
        )
        assert check.reason == "bad_language"

    def test_a_failing_case_is_not_a_call_mistake(self):
        # It ran. The answer was wrong, which is a result, not a misuse.
        check = impl.ImplementationCheck(
            ran=True,
            cases=[impl.CaseResult(name="a", passed=False, detail="wrong")],
        )
        assert check.reason == ""
        assert check.verified is False

    def test_the_repeat_detector_can_see_a_no_cases_call(self):
        """The behaviour that makes escalation possible at all."""
        from app.services import agent_failure_diagnosis

        as_error = {"error": "No usable reference cases were supplied.", "data": {}}
        as_result = {"success": True, "data": {"verified": False}}
        assert agent_failure_diagnosis._failed(as_error) is True
        assert agent_failure_diagnosis._failed(as_result) is False
