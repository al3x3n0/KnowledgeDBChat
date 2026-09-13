"""`exit code 125` is docker saying the container never started.

Measured: profile_c_workload returned "Profiling failed with exit code 125" for
an entire run, because kdbc-profiling-research had never been built on this
machine. The number already meant "the code never ran", and nothing the agent
read said so -- it kept rewriting a workload that had never executed. Worse,
`make sandbox-check` had reported that image MISSING hours earlier, so the
answer existed and nothing pointed at it.

125 is docker's own failure, distinct from 126/127 (the image exists, the
command could not be run) and from any other code (the program's own exit).
Only the first two are the sandbox's fault, and only they get an explanation:
a program exiting 3 is a fact about the program.
"""

import pytest

from app.services.agent_sandbox_runtime import explain_sandbox_exit

pytestmark = pytest.mark.unit

IMAGE = "ghcr.io/al3x3n0/kdbc-profiling-research:latest"


class TestItNamesTheCauseNotTheNumber:
    def test_125_points_at_the_image_and_at_sandbox_check(self):
        message = explain_sandbox_exit(125, "Unable to find image locally", IMAGE)

        assert IMAGE in message
        assert "never ran" in message, "the run must know its code was not tested"
        assert "sandbox-check" in message, "point at the thing that answers it"

    def test_dockers_own_words_are_kept(self):
        message = explain_sandbox_exit(125, "Unable to find image locally", IMAGE)

        assert "Unable to find image locally" in message

    def test_a_missing_command_is_a_different_failure(self):
        """126/127 mean the image exists and the entrypoint does not -- a
        different repair from a missing image."""
        message = explain_sandbox_exit(127, "exec: clang: not found", IMAGE)

        assert "image exists" in message
        assert "sandbox-check" not in message


class TestItStaysQuietWhenTheProgramFailed:
    @pytest.mark.parametrize("code", [1, 2, 3, 42, 91])
    def test_an_ordinary_exit_gets_no_explanation(self, code):
        """A program exiting 3 is a fact about the program. Dressing it up as
        a sandbox problem would send the run to check the sandbox."""
        assert explain_sandbox_exit(code, "assertion failed", IMAGE) == ""
