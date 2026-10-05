"""What a run learned about calling a tool, kept for the next one.

415 methods in this deployment, none of them about tool usage — and that is the
knowledge runs pay for twice. Measured: one gem5 study was refused for passing a
mechanism at the top level of a config, was told the accepted shape, corrected
itself and finished; the next study made the identical mistake and spent five
iterations on it.
"""

from app.services import agent_tool_usage_methods as usage

CONFIG_REFUSAL = (
    "run has key(s) l2, which are not part of a configuration. It takes: "
    "cpu_type, clock, mem_size, cache_line_size, cpu_params, caches, "
    'branch_pred. A mechanism is named inside `caches`, like {"caches": '
    '{"l2": {"prefetcher": "StridePrefetcher"}}}'
)
DAEMON = "docker: Cannot connect to the Docker daemon at unix:///var/run/docker.sock"
SYNTAX = "/work/prog.c:12:5: error: expected ';' after expression"


def _state(*entries, findings=()):
    return {
        "actions_taken": [
            {"action": {"tool": tool}, "result": result} for tool, result in entries
        ],
        "findings": [{"type": t} for t in findings],
    }


def FAILED(err):
    return {"success": False, "error": err}


def OK(types=()):
    return {"success": True, "findings": [{"type": t} for t in types]}


class TestWhatCountsAsALesson:
    def test_a_refusal_the_run_recovered_from(self):
        state = _state(
            ("explain_bottleneck", FAILED(CONFIG_REFUSAL)),
            ("explain_bottleneck", OK(("bottleneck_attribution",))),
        )
        (learned,) = usage.corrections(state)
        assert learned["tool"] == "explain_bottleneck"
        assert "named inside `caches`" in learned["refusal"]
        assert learned["produced"] == ["bottleneck_attribution"]

    def test_one_lesson_per_tool_however_often_it_was_refused(self):
        state = _state(
            ("explain_bottleneck", FAILED(CONFIG_REFUSAL)),
            ("explain_bottleneck", FAILED(CONFIG_REFUSAL)),
            ("explain_bottleneck", OK(("bottleneck_attribution",))),
        )
        assert len(usage.corrections(state)) == 1


class TestWhatIsNotALesson:
    def test_a_refusal_never_recovered_from(self):
        # A correction nobody demonstrated is a guess.
        state = _state(("measure_headroom", FAILED(CONFIG_REFUSAL)))
        assert usage.corrections(state) == []

    def test_a_tool_that_could_not_run(self):
        # An unreachable daemon teaches nothing about how to call the tool.
        state = _state(
            ("simulate_mechanism", FAILED(DAEMON)),
            ("simulate_mechanism", OK(("mechanism_comparison",))),
        )
        assert usage.corrections(state) == []

    def test_the_run_s_own_broken_code(self):
        state = _state(
            ("compile_c_snippet", FAILED(SYNTAX)),
            ("compile_c_snippet", OK(("codegen_measurement",))),
        )
        assert usage.corrections(state) == []

    def test_a_run_that_never_failed(self):
        assert usage.corrections(_state(("t", OK()))) == []


class TestTheRecordItBuilds:
    def _record(self, produced=("bottleneck_attribution",)):
        return usage.build(
            {
                "tool": "explain_bottleneck",
                "refusal": CONFIG_REFUSAL,
                "error_class": "invalid_argument",
                "produced": list(produced),
            },
            available_finding_types=["bottleneck_attribution"],
        )

    def test_it_keeps_the_tool_s_own_corrective_words(self):
        record = self._record()
        assert record is not None
        assert any("named inside `caches`" in step for step in record["procedure"])

    def test_it_says_what_it_prevents(self):
        assert "iterations" in self._record()["prevents"]

    def test_evidence_is_what_the_corrected_call_produced(self):
        from app.services import agent_method_record

        record = self._record()
        assert record["status"] == agent_method_record.VALIDATED

    def test_a_correction_that_produced_nothing_is_recorded_unvalidated(self):
        # Still true and still worth having, but not dressed up as evidence.
        from app.services import agent_method_record

        record = self._record(produced=())
        assert record["status"] == agent_method_record.UNVALIDATED

    def test_the_arguments_that_worked_are_not_stored(self):
        # They routinely contain whole programs, and the reusable part is the
        # shape the tool described rather than one kernel's source.
        record = self._record()
        assert "int main" not in agent_method_render(record)


def agent_method_render(record):
    from app.services import agent_method_record

    return agent_method_record.render(record)


class TestNamingTheFailureClass:
    """The class is written into the lesson, so it has to read as English.

    A classified refusal is worth naming; an unclassified one is not, because
    "refused for its shape (unknown)" tells a later run less than stopping at
    "shape" does.
    """

    def _prevents(self, error_class):
        record = usage.build(
            {
                "tool": "evaluate_across_kernels",
                "refusal": "At least two kernels are needed.",
                "error_class": error_class,
                "produced": ["mechanism_evaluation"],
            },
            available_finding_types=["mechanism_evaluation"],
        )
        assert record is not None
        return record["prevents"]

    def test_a_classified_refusal_names_its_class(self):
        assert "(invalid_argument)" in self._prevents("invalid_argument")

    def test_an_unclassified_refusal_says_nothing_rather_than_unknown(self):
        for missing in ("unknown", "", None):
            prevents = self._prevents(missing)
            assert "unknown" not in prevents, missing
            assert "()" not in prevents, missing
            assert "for its shape," in prevents, missing
