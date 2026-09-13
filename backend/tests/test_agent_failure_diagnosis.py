"""A repeated failure must escalate from a message into a protocol.

The cases here are the two that actually cost runs: a compiler flag that is
unsupported on this architecture at all, retried four times because the error
read like something a retry might fix, and a gem5 timeout whose advice --
shrink the workload -- pointed away from the answer on a workload of 3,200
instructions that could never have been slow.
"""

from __future__ import annotations

from app.services import agent_failure_diagnosis as diagnosis


def _entry(tool: str, params: dict, error: str):
    return {
        "action": {"tool": tool, "params": params},
        "result": {"success": False, "error": error},
    }


def _state(*entries):
    return {"actions_taken": list(entries)}


NATIVE = {"code": "int main(void){return 0;}", "flags": "-O3 -march=native"}
NATIVE_ERROR = "Compilation failed: clang: error: unsupported argument 'native'"


def test_a_first_failure_is_left_alone():
    """The tool's own message is the remedy; repeating it is noise."""
    action = {"tool": "compile_c_snippet", "params": NATIVE}

    result = diagnosis.analyze(
        action, {"success": False, "error": NATIVE_ERROR}, _state()
    )

    assert result is None


def test_a_second_identical_failure_is_called_out():
    action = {"tool": "compile_c_snippet", "params": NATIVE}
    state = _state(_entry("compile_c_snippet", NATIVE, NATIVE_ERROR))

    result = diagnosis.analyze(action, {"success": False, "error": NATIVE_ERROR}, state)

    assert result["attempt"] == 2
    assert "same arguments" in result["guidance"]
    # Not yet the full protocol: one repeat may still be a slip.
    assert "protocol" not in result


def test_a_third_identical_failure_gets_the_protocol():
    action = {"tool": "compile_c_snippet", "params": NATIVE}
    state = _state(
        _entry("compile_c_snippet", NATIVE, NATIVE_ERROR),
        _entry("compile_c_snippet", NATIVE, NATIVE_ERROR),
    )

    result = diagnosis.analyze(action, {"success": False, "error": NATIVE_ERROR}, state)

    assert result["attempt"] == 3
    assert "Stop retrying" in result["guidance"]
    steps = result["protocol"]
    assert any("smallest input" in step for step in steps)
    assert any("control also fails" in step for step in steps)
    assert any("one element at a time" in step for step in steps)


def test_a_timeout_warns_that_size_advice_can_mislead():
    """The gem5 deadlock: shrinking a 3,200-instruction workload never helps."""
    params = {"code": "int main(void){return 0;}", "cpu_type": "NeoverseV2"}
    action = {"tool": "simulate_c_workload", "params": params}
    error = "Simulation timed out after 900s. Shrink the workload's input."
    state = _state(
        _entry("simulate_c_workload", params, error),
        _entry("simulate_c_workload", params, error),
    )

    result = diagnosis.analyze(action, {"success": False, "error": error}, state)

    assert result["error_class"] == "timeout"
    assert "stuck rather than slow" in result["note"]


def test_changing_the_call_resets_the_count():
    """Varying the input is the wanted behaviour and must not be punished."""
    state = _state(
        _entry("compile_c_snippet", NATIVE, NATIVE_ERROR),
        _entry("compile_c_snippet", NATIVE, NATIVE_ERROR),
    )
    changed = {"code": NATIVE["code"], "flags": "-O3 -mcpu=neoverse-n1"}
    action = {"tool": "compile_c_snippet", "params": changed}

    result = diagnosis.analyze(
        action, {"success": False, "error": "Compilation failed: something else"}, state
    )

    assert result is None


def test_labels_do_not_disguise_a_verbatim_retry():
    """Relabelling the same call is still the same call."""
    first = {**NATIVE, "label": "attempt-one"}
    second = {**NATIVE, "label": "attempt-two"}
    state = _state(_entry("compile_c_snippet", first, NATIVE_ERROR))

    result = diagnosis.analyze(
        {"tool": "compile_c_snippet", "params": second},
        {"success": False, "error": NATIVE_ERROR},
        state,
    )

    assert result is not None and result["attempt"] == 2


def test_a_different_tool_failing_is_a_different_problem():
    state = _state(_entry("compile_c_snippet", NATIVE, NATIVE_ERROR))

    result = diagnosis.analyze(
        {"tool": "simulate_c_workload", "params": NATIVE},
        {"success": False, "error": NATIVE_ERROR},
        state,
    )

    assert result is None


def test_timeouts_differing_only_by_duration_count_as_the_same_failure():
    """Exact-text matching would treat these as unrelated and never escalate."""
    params = {"code": "x"}
    state = _state(
        _entry("simulate_c_workload", params, "Simulation timed out after 900s"),
        _entry("simulate_c_workload", params, "Simulation timed out after 1500s"),
    )

    result = diagnosis.analyze(
        {"tool": "simulate_c_workload", "params": params},
        {"success": False, "error": "Simulation timed out after 1800s"},
        state,
    )

    assert result["attempt"] == 3
    assert "protocol" in result


def test_successes_are_never_diagnosed():
    assert (
        diagnosis.analyze(
            {"tool": "compile_c_snippet", "params": NATIVE},
            {"success": True, "data": {"output": "..."}},
            _state(),
        )
        is None
    )


def test_a_successful_result_between_failures_still_counts_the_failures():
    """Succeeding once does not mean the failing call has become sound."""
    state = _state(
        _entry("compile_c_snippet", NATIVE, NATIVE_ERROR),
        {
            "action": {"tool": "compile_c_snippet", "params": {"code": "other"}},
            "result": {"success": True},
        },
        _entry("compile_c_snippet", NATIVE, NATIVE_ERROR),
    )

    result = diagnosis.analyze(
        {"tool": "compile_c_snippet", "params": NATIVE},
        {"success": False, "error": NATIVE_ERROR},
        state,
    )

    assert result["attempt"] == 3


def test_errors_reported_inside_data_are_seen():
    """Some tools report failure one level down; missing those hides repeats."""
    params = {"code": "x"}
    failing = {
        "action": {"tool": "profile_c_workload", "params": params},
        "result": {"success": False, "data": {"error": "profiling failed"}},
    }

    result = diagnosis.analyze(
        {"tool": "profile_c_workload", "params": params},
        {"success": False, "data": {"error": "profiling failed"}},
        _state(failing),
    )

    assert result is not None and result["attempt"] == 2


def test_classification_buckets_the_common_shapes():
    assert diagnosis.classify_error("Simulation timed out after 900s") == "timeout"
    assert diagnosis.classify_error("Compilation failed: ...") == "compilation"
    assert diagnosis.classify_error("Image is not allowlisted") == "permission"
    assert diagnosis.classify_error("cc1plus killed: out of memory") == "resource"
    assert diagnosis.classify_error("") == "unknown"


def _varied(tool: str, code: str, error: str):
    return {
        "action": {"tool": tool, "params": {"code": code, "flags": "-O2"}},
        "result": {"success": False, "error": error},
    }


def test_editing_the_input_between_attempts_still_escalates_eventually():
    """The failure mode a live run actually produced.

    benchmark_c_snippet failed seven times with five different compile errors.
    Every call had different arguments, so the identical-call check never
    fired, and the run kept rewriting the code instead of finding out what the
    tool accepts.
    """
    state = _state(
        _varied(
            "benchmark_c_snippet", "v1", "Compilation failed: unknown FP unit '387'"
        ),
        _varied("benchmark_c_snippet", "v2", "Compilation failed: constraint 'x'"),
        _varied("benchmark_c_snippet", "v3", "Compilation failed: implicit sqrtf"),
    )

    result = diagnosis.analyze(
        {"tool": "benchmark_c_snippet", "params": {"code": "v4", "flags": "-O2"}},
        {"success": False, "error": "Compilation failed: no -march=native"},
        state,
    )

    assert result is not None, "four different compile failures went unremarked"
    assert result["varied_arguments"] is True
    assert result["attempt"] == 4
    assert "different arguments" in result["guidance"]
    assert result["protocol"]


def test_a_few_varied_failures_are_left_alone():
    """Working through two or three different errors is normal progress."""
    state = _state(
        _varied("benchmark_c_snippet", "v1", "Compilation failed: one"),
        _varied("benchmark_c_snippet", "v2", "Compilation failed: two"),
    )

    result = diagnosis.analyze(
        {"tool": "benchmark_c_snippet", "params": {"code": "v3", "flags": "-O2"}},
        {"success": False, "error": "Compilation failed: three"},
        state,
    )

    assert result is None


def test_varied_failures_of_different_kinds_do_not_accumulate():
    """A compile error then a timeout is not one wall being hit repeatedly."""
    state = _state(
        _varied("simulate_c_workload", "v1", "Compilation failed: one"),
        _varied("simulate_c_workload", "v2", "Simulation timed out after 900s"),
        _varied("simulate_c_workload", "v3", "Compilation failed: three"),
    )

    result = diagnosis.analyze(
        {"tool": "simulate_c_workload", "params": {"code": "v4", "flags": "-O2"}},
        {"success": False, "error": "Simulation timed out after 900s"},
        state,
    )

    assert result is None
