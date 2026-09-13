"""Guard: LLM calls made during an agent run must be recordable.

The snapshot recorder returns early without a session, so a call site that
passes snapshot_context but not db records nothing, silently. Enabling
LLM_CALL_SNAPSHOT_ENABLED then produced an empty table with no error, which is
how the gap survived: a run with 9 LLM calls wrote 0 snapshots.
"""

import ast
import pathlib

import pytest

SERVICES = pathlib.Path(__file__).resolve().parents[1] / "app" / "services"

# Modules that run inside an agent job and therefore have a session to hand.
IN_LOOP_MODULES = (
    "agent_thinking_service.py",
    "agent_decision_parser.py",
    "agent_context_compaction.py",
    "agent_native_tool_loop.py",
    "autonomous_agent_executor.py",
    # Added after a run captured 12 of 15 real provider calls: these make LLM
    # calls during a job too, and each unlisted module is a blind spot the
    # export reports as a shortfall without saying where.
    "agent_job_memory_service.py",
    "agent_tool_dispatch.py",
    "agent_runtime_finalizer.py",
    "agent_action_service.py",
    "agent_observation_service.py",
)

LLM_METHODS = {"generate_response", "generate_structured"}


def _llm_calls(path: pathlib.Path):
    """Yield (line, kwargs) for each LLM call in a module."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in LLM_METHODS:
            continue
        yield node.lineno, {kw.arg for kw in node.keywords if kw.arg}


@pytest.mark.parametrize("module", IN_LOOP_MODULES)
def test_in_loop_llm_calls_pass_a_session_with_their_context(module):
    path = SERVICES / module
    if not path.exists():
        pytest.skip(f"{module} not present")
    offenders = [
        line
        for line, kwargs in _llm_calls(path)
        if "snapshot_context" in kwargs and "db" not in kwargs
    ]

    assert not offenders, (
        f"{module} lines {offenders} pass snapshot_context without db; "
        "the recorder returns early and captures nothing"
    )


@pytest.mark.parametrize("module", IN_LOOP_MODULES)
def test_in_loop_llm_calls_declare_their_context(module):
    """A call with a session but no context records an unattributable row."""
    path = SERVICES / module
    if not path.exists():
        pytest.skip(f"{module} not present")
    offenders = [
        line
        for line, kwargs in _llm_calls(path)
        if "db" in kwargs and "snapshot_context" not in kwargs
    ]

    assert not offenders, (
        f"{module} lines {offenders} pass db without snapshot_context; "
        "the snapshot would not be attributable to a job phase"
    )


def test_the_decision_path_is_instrumented_end_to_end():
    """The think phase is the conversation; it must be captured."""
    thinking = (SERVICES / "agent_thinking_service.py").read_text()

    assert "db=db" in thinking
    assert '"phase": "thinking"' in thinking


def test_the_critic_and_planning_passes_are_instrumented():
    executor = (SERVICES / "autonomous_agent_executor.py").read_text()

    for phase in ("critic", "execution_plan", "causal_plan"):
        assert f'"phase": "{phase}"' in executor, f"{phase} pass is not captured"


@pytest.mark.parametrize("module", IN_LOOP_MODULES)
def test_no_in_loop_llm_call_is_left_uninstrumented(module):
    """Every in-loop call must be recordable, not merely consistent.

    The two tests above pass when a call declares neither db nor
    snapshot_context, which is exactly the state that leaves a run's export
    short of the calls it actually made.
    """
    path = SERVICES / module
    if not path.exists():
        pytest.skip(f"{module} not present")

    offenders = [
        line
        for line, kwargs in _llm_calls(path)
        if "snapshot_context" not in kwargs and "db" not in kwargs
    ]

    assert (
        not offenders
    ), f"{module} lines {offenders} make an LLM call that cannot be captured"
