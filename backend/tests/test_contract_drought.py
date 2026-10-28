"""A run is told when its last few actions produced nothing its contract needs.

The first stage of the first self-proposed plan spent all ten iterations
orienting -- reading, cloning, browsing -- and ended with its contract unmet.
The loop policy never called it stalled, because cloning and browsing record
findings; what never grew was anything the contract asked for.
"""

import pytest

from app.services import agent_contract_drought as drought
from app.services import agent_loop_policy

pytestmark = pytest.mark.unit


def _eval(missing, have=None, satisfied=False):
    return {
        "enabled": True,
        "satisfied": satisfied,
        "missing": missing,
        "metrics": {"finding_types": have or {}},
    }


BENCH = ["finding_type:benchmark_measurement>=2"]


def _run(evals):
    """Record one contract evaluation per iteration, starting at 1."""
    state = {}
    for i, contract in enumerate(evals, start=1):
        drought.record(state, contract, i)
    return state


def test_exploring_iterations_are_not_progress():
    # Other findings (a workspace, documents) grow; the required type does not.
    state = _run([_eval(BENCH, {"repo_workspace": n}) for n in range(1, 5)])

    assert drought.iterations_without_progress(state, 4) == 3
    note = drought.format_drought(state, iteration=4, job_type="research")
    assert note.startswith("NO PROGRESS ON THE CONTRACT FOR 3 ITERATIONS")
    assert "benchmark_measurement" in note
    assert "benchmark_c_snippet" in note, "names the tool that produces it"


def test_quiet_for_fewer_iterations_than_the_threshold():
    state = _run([_eval(BENCH)] * drought.DROUGHT_ITERATIONS)

    assert (
        drought.format_drought(
            state, iteration=drought.DROUGHT_ITERATIONS, job_type="research"
        )
        == ""
    )


def test_one_of_two_measurements_is_progress():
    """`benchmark_measurement>=2` reads the same at one as at none; the count
    is what moved."""
    evals = [_eval(BENCH)] * 3 + [_eval(BENCH, {"benchmark_measurement": 1})]
    state = _run(evals)

    assert drought.iterations_without_progress(state, 4) == 0
    assert drought.format_drought(state, iteration=4, job_type="research") == ""


def test_a_requirement_met_is_progress():
    two = ["finding_type:implementation_verified", *BENCH]
    state = _run([_eval(two)] * 3 + [_eval(BENCH)])

    assert drought.iterations_without_progress(state, 4) == 0


def test_a_satisfied_contract_forgets_the_drought():
    state = _run([_eval(BENCH)] * 4 + [_eval([], satisfied=True)])

    assert drought.STATE_KEY not in state
    assert drought.format_drought(state, iteration=5, job_type="research") == ""


def test_no_contract_says_nothing():
    state = {}
    drought.record(state, {"enabled": False}, 7)
    assert drought.format_drought(state, iteration=9, job_type="research") == ""


def test_a_restart_forgets_when_the_contract_last_improved():
    """A restart resets the iteration count; an old reference would hold the
    note back until the new count caught up with it."""
    state = _run([_eval(BENCH)] * 9)
    agent_loop_policy.forget_stall(state)
    assert drought.STATE_KEY not in state

    state = _run([_eval(BENCH)] * 9)
    agent_loop_policy.clear_give_up_state(state)
    assert drought.STATE_KEY not in state


def test_the_note_reaches_the_volatile_prompt():
    from types import SimpleNamespace

    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    state = _run([_eval(BENCH)] * 4)
    state["goal_contract_last"] = _eval(BENCH)
    executor = AutonomousAgentExecutor.__new__(AutonomousAgentExecutor)
    job = SimpleNamespace(iteration=4, job_type="research", config={}, id="j", goal="g")

    try:
        prompt = executor._build_thinking_prompt_volatile(job, state)
    except (AttributeError, TypeError) as exc:  # pragma: no cover - diagnostic
        pytest.fail(f"volatile prompt needs more of the executor: {exc}")

    assert "NO PROGRESS ON THE CONTRACT FOR 3 ITERATIONS" in prompt
    assert prompt.index("CONTRACT NOT YET SATISFIED") < prompt.index("NO PROGRESS")
