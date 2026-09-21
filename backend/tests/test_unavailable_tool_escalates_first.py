"""A tool that could not run says so on the first failure.

Silence at attempt 1 is right when a tool ran and refused the input: its own
message is the remedy, and repeating it as guidance is noise. It is wrong when
the tool never got that far, because no edit to the call can help and the run
will otherwise spend its iterations proving that one at a time.

Every message below was produced by this deployment during one session. A
discovery stage spent 19 iterations rewriting calls while arXiv answered 406 to
each; a gem5 tool reported "Cannot connect to the Docker daemon" eight times
before anyone looked.
"""

from app.services import agent_failure_diagnosis as diagnosis

DOCKER = (
    "docker: Cannot connect to the Docker daemon at unix:///var/run/docker.sock. "
    "Is the docker daemon running?"
)
ARXIV = "HTTP Error 406: Not Acceptable"
MISSING_CLANG = "Compilation failed: clang: not found"
MNEMONIC = "unknown mnemonic 'uaddw': add it to operand_arity"
SYNTAX = "/work/prog.c:12:5: error: expected ';' after expression"
REFUSAL = "The program never calls M5_SAMPLE(), so there is nothing to sample"


class TestWhatCountsAsUnavailable:
    def test_a_daemon_that_is_not_listening(self):
        assert diagnosis.could_not_run(DOCKER) is True

    def test_an_upstream_answering_with_a_status(self):
        assert diagnosis.could_not_run(ARXIV) is True
        assert diagnosis.could_not_run("503 Service Unavailable") is True

    def test_a_binary_missing_from_the_image(self):
        # The docstring's own example: nothing about the source is wrong.
        assert diagnosis.could_not_run(MISSING_CLANG) is True


class TestWhatDoesNot:
    def test_a_tool_that_ran_and_refused_the_input(self):
        # The tool judged the input and said what it needs. Telling the run
        # "this happened before it looked at your input" would be false.
        assert diagnosis.could_not_run(MNEMONIC) is False
        assert diagnosis.could_not_run(REFUSAL) is False

    def test_the_run_s_own_broken_code(self):
        assert diagnosis.could_not_run(SYNTAX) is False

    def test_a_number_that_merely_looks_like_a_status(self):
        assert diagnosis.could_not_run("the kernel ran in 503 cycles") is False

    def test_silence(self):
        assert diagnosis.could_not_run("") is False
        assert diagnosis.could_not_run(None) is False


class TestTheEscalationArrivesImmediately:
    def _first_failure(self, error):
        return diagnosis.analyze(
            {"tool": "simulate_mechanism", "params": {"code": "int main(){}"}},
            {"success": False, "error": error},
            {"actions_taken": []},
        )

    def test_an_unavailable_tool_is_diagnosed_at_attempt_one(self):
        found = self._first_failure(DOCKER)
        assert found is not None
        assert found["attempt"] == 1
        assert found["unavailable"] is True
        assert found["protocol"], "the control-run protocol should be attached"

    def test_the_guidance_says_editing_the_call_will_not_help(self):
        # The one thing the run needs to stop doing.
        found = self._first_failure(ARXIV)
        assert "before it looked at your input" in found["guidance"]

    def test_a_tool_that_refused_the_input_is_still_silent_at_first(self):
        # Unchanged behaviour: the tool's message is the remedy.
        assert self._first_failure(MNEMONIC) is None
