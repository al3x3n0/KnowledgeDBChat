"""A blocked run says what would end the stall, not just its shape.

"3 consecutive rounds produced no new findings (still 11)" is honest and
unanswerable: it describes the stall, not the thing a person could supply. The
run's own history usually names one -- which tool failed, in which words, and
whether it ran at all. That last distinction separates "this is broken, is it
coming back?" from "this refused my call, what does it accept?", which are
different questions with different answerers.
"""

from app.services import agent_unblock_request as unblock

DAEMON = "docker: Cannot connect to the Docker daemon at unix:///var/run/docker.sock"
CONFIG = (
    "run has key(s) l2, which are not part of a configuration. It takes: "
    "cpu_type, clock, caches, branch_pred. A mechanism is named inside `caches`"
)
SYNTAX = "/work/prog.c:12:5: error: expected ';' after expression"


def _state(*failures):
    return {
        "actions_taken": [
            {"action": {"tool": tool}, "result": {"success": False, "error": err}}
            for tool, err in failures
        ]
    }


class TestATooThatNeverRan:
    def test_it_asks_whether_the_tool_is_available(self):
        need = unblock.describe(_state(("simulate_mechanism", DAEMON)))
        assert need["kind"] == unblock.TOOL_UNAVAILABLE
        assert "Is simulate_mechanism available" in need["question"]

    def test_a_person_cannot_fix_it_by_answering(self):
        # Nothing an operator types makes a daemon listen.
        need = unblock.describe(_state(("simulate_mechanism", DAEMON)))
        assert need["answerable_by"] == "platform_change"

    def test_it_outranks_a_refusal_seen_earlier(self):
        # A tool that never ran has nothing to say about what it accepts.
        need = unblock.describe(
            _state(("explain_bottleneck", CONFIG), ("explain_bottleneck", DAEMON))
        )
        assert need["kind"] == unblock.TOOL_UNAVAILABLE


class TestAToolThatRefusedTheCall:
    def test_it_asks_what_the_tool_accepts_and_quotes_it(self):
        need = unblock.describe(_state(("explain_bottleneck", CONFIG)))
        assert need["kind"] == unblock.TOOL_REFUSES_INPUT
        assert "named inside `caches`" in need["question"]

    def test_an_operator_can_answer_that_one(self):
        need = unblock.describe(_state(("explain_bottleneck", CONFIG)))
        assert need["answerable_by"] == "operator"

    def test_the_run_s_own_broken_code_is_not_a_question_for_anyone(self):
        assert unblock.describe(_state(("compile_c_snippet", SYNTAX))) is None


class TestEvidenceNothingHereCanProduce:
    def test_it_asks_about_the_contract_rather_than_the_call(self):
        # literature_review's producers are not callable under synthesis.
        need = unblock.describe(
            {"actions_taken": []},
            missing=["finding_type:literature_review"],
            job_type="synthesis",
        )
        assert need["kind"] == unblock.EVIDENCE_UNREACHABLE
        assert "literature_review" in need["question"]

    def test_evidence_the_job_type_can_produce_raises_no_question(self):
        assert (
            unblock.describe(
                {"actions_taken": []},
                missing=["finding_type:related_paper_set"],
                job_type="research",
            )
            is None
        )


class TestSilence:
    def test_a_stall_with_no_nameable_blocker_invents_none(self):
        # Dressing "it stopped finding things" up as a question would waste the
        # reader's attention on something nobody can answer.
        assert unblock.describe({"actions_taken": []}) is None


class TestOnlyAJudgedInputBecomesAQuestionForAPerson:
    """Every tool failure wears the same shape in the ledger.

    Measured by replaying ten real blocked runs: seven produced a question and
    five of those asked an operator "what does X accept" about a failure where
    nothing they could type would help -- an upstream that errored, a
    simulation that aborted, an ingestion that never landed. Those reach a
    person through the backlog instead, and the queue row still shows the stall
    either way; what is refused here is dressing a platform problem up as a
    question about arguments.
    """

    # Verbatim from the runs that motivated the rule.
    OUTAGE = (
        "Failed to summarize findings: LLM service error: Failed to generate response"
    )
    CRASH = "A simulation failed."
    ASYNC_GAP = "Ingestion of 2605.20868v1 was started (source bce256eb) but no document appeared"
    SEQUENCING = (
        "No hot blocks to mine. Run profile_c_workload first and this tool "
        "will pick up its blocks."
    )
    JUDGED = (
        "get_document_details was called with invalid parameters: field "
        "document_id should be a UUID"
    )

    def test_a_tool_that_judged_the_arguments_is_asked_about(self):
        need = unblock.describe(_state(("get_document_details", self.JUDGED)))
        assert need is not None
        assert need["kind"] == unblock.TOOL_REFUSES_INPUT
        assert need["answerable_by"] == "operator"

    def test_a_failure_that_never_judged_the_arguments_asks_nothing(self):
        for error in (self.OUTAGE, self.CRASH, self.ASYNC_GAP, self.SEQUENCING):
            assert unblock.describe(_state(("some_tool", error))) is None, error

    def test_a_judged_refusal_is_still_found_behind_unjudged_noise(self):
        """The run's last failure is usually not the informative one."""
        need = unblock.describe(
            _state(
                ("get_document_details", self.JUDGED),
                ("summarize_findings", self.OUTAGE),
            )
        )
        assert need is not None
        assert need["tool"] == "get_document_details"
