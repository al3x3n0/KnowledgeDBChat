"""Fixing your own code is the loop working, not a run going in circles.

`check_implementation` and `benchmark_c_snippet` fail with a compiler
diagnostic whenever the code the run just wrote does not compile. That is the
development loop: write, read the diagnostic, fix, compile again. Counting
those as evidence the TOOL is at fault escalates after four of them and tells
the run

    "Editing the input and trying again is not working. Establish what the
     tool does accept before changing it further."

plus a protocol whose first step is to push a trivial control through the tool
to see whether the tool itself is broken. For a syntax error on line 12 that is
the one investigation that cannot help, and it costs iterations from a stage
whose budget is six.

The distinction is not the error class -- `Compilation failed: clang: not
found` is also a `compilation` error, and that one IS the tool being broken.
It is whether the compiler pointed at a place in the submitted source.
"""

import pytest

from app.services import agent_failure_diagnosis as diag

pytestmark = pytest.mark.unit


class TestTellingTheCodeFromTheToolchain:
    @pytest.mark.parametrize(
        "message",
        [
            "prog.c:12:5: error: expected ';' before '}' token",
            "error[E0308]: mismatched types\n --> prog.rs:7:19",
            'File "prog.py", line 1\n    def broken(:\n               ^\nSyntaxError: invalid syntax',
            "Compilation failed: prog.c:3:1: warning: implicit declaration",
        ],
    )
    def test_a_diagnostic_naming_a_line_blames_the_code(self, message):
        assert diag.blames_the_submitted_code(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "Compilation failed: clang: not found",
            "Sandboxed execution is disabled on this server",
            "Docker is not available to this process",
            "Benchmark timed out after 120s",
            "",
        ],
    )
    def test_a_broken_environment_does_not(self, message):
        assert diag.blames_the_submitted_code(message) is False


def _history(tool, error, n):
    """n prior failures of one tool with varied arguments and one error."""
    return {
        "actions_taken": [
            {
                "action": {"tool": tool, "params": {"code": f"attempt {i}"}},
                "result": {"success": False, "error": error},
            }
            for i in range(n)
        ]
    }


class TestEscalationSkipsTheRunsOwnCode:
    SYNTAX = "prog.c:12:5: error: expected ';' before '}' token"
    BROKEN = "Compilation failed: clang: not found"

    def test_four_syntax_errors_do_not_escalate(self):
        state = _history("check_implementation", self.SYNTAX, 4)
        action = {"tool": "check_implementation", "params": {"code": "attempt 5"}}

        verdict = diag.analyze(action, {"success": False, "error": self.SYNTAX}, state)

        assert verdict is None, (
            "a run fixing its own compile errors is converging; telling it the "
            "tool is broken sends it to test the compiler"
        )

    def test_four_broken_toolchain_errors_still_escalate(self):
        """The control, and the reason this is not a blanket exclusion: both
        are `compilation` errors and only one is the run's fault."""
        state = _history("check_implementation", self.BROKEN, 4)
        action = {"tool": "check_implementation", "params": {"code": "attempt 5"}}

        verdict = diag.analyze(action, {"success": False, "error": self.BROKEN}, state)

        assert verdict is not None
        assert verdict["attempt"] >= diag.CLASS_ESCALATE_AFTER
        assert verdict["protocol"], "a broken tool is what the protocol is for"

    def test_the_identical_code_twice_is_still_called_out(self):
        """Not everything is forgiven. Submitting the SAME source and getting
        the same diagnostic is a verbatim retry, and that path is untouched."""
        same = {"tool": "check_implementation", "params": {"code": "the same"}}
        state = {
            "actions_taken": [
                {"action": same, "result": {"success": False, "error": self.SYNTAX}}
                for _ in range(diag.CALL_OUT_AFTER)
            ]
        }

        verdict = diag.analyze(same, {"success": False, "error": self.SYNTAX}, state)

        assert verdict is not None
        assert verdict["attempt"] >= diag.CALL_OUT_AFTER

    def test_a_first_compile_error_was_never_escalated_anyway(self):
        state = {"actions_taken": []}
        action = {"tool": "check_implementation", "params": {"code": "first"}}

        assert (
            diag.analyze(action, {"success": False, "error": self.SYNTAX}, state)
            is None
        )
