"""Guards on the AXIS tools.

The docker-backed paths are exercised through their preflight and parsing; the
container itself is not run here.
"""

import pytest

from app.services import agent_axis_sandbox as axis


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(axis.agent_sandbox_runtime, "execution_enabled", lambda: True)
    monkeypatch.setattr(
        axis.agent_sandbox_runtime, "allowed_images", lambda: [axis.DEFAULT_IMAGE]
    )


@pytest.mark.asyncio
async def test_an_empty_description_is_rejected_before_a_container_starts(enabled):
    assert "source is required" in (await axis.check_description(source=" "))["error"]


@pytest.mark.asyncio
async def test_execution_must_be_enabled(monkeypatch):
    monkeypatch.setattr(axis.agent_sandbox_runtime, "execution_enabled", lambda: False)

    result = await axis.check_description(source="(defextension foo)")

    assert "ENABLE_UNSAFE_CODE_EXECUTION" in result["error"]


@pytest.mark.asyncio
async def test_an_unlisted_image_is_refused(enabled):
    result = await axis.check_description(
        source="(defextension foo)", image="evil:latest"
    )

    assert "not allowlisted" in result["error"]


@pytest.mark.asyncio
async def test_an_unknown_emit_target_lists_what_is_available(enabled):
    result = await axis.emit_artifact(source="(defextension foo)", target="wat")

    assert "Unknown emit target" in result["error"]
    assert "smt2" in result["error"]


@pytest.mark.asyncio
async def test_a_proof_without_check_sat_asks_the_solver_nothing(enabled):
    """An obligation that never calls check-sat returns no verdict at all."""
    result = await axis.prove_equivalence(
        source="(defextension foo)", obligation="(assert true)"
    )

    assert "(check-sat)" in result["error"]


@pytest.mark.asyncio
async def test_a_missing_obligation_explains_what_unsat_would_mean(enabled):
    result = await axis.prove_equivalence(source="(defextension foo)", obligation="")

    assert "negation" in result["error"]


def test_the_solver_verdict_is_read_from_its_output():
    assert axis.parse_solver_verdict("unsat\n") == "unsat"
    assert axis.parse_solver_verdict("warning: blah\nsat\n") == "sat"
    assert axis.parse_solver_verdict('(error "line 3")\n') == "error"
    assert axis.parse_solver_verdict("") == "error"


def test_unknown_is_not_treated_as_proved():
    """A solver that gave up has neither proved nor disproved the claim."""
    assert axis.parse_solver_verdict("unknown") == "unknown"
    assert axis.parse_solver_verdict("unknown") != "unsat"


def test_every_emit_target_maps_to_a_real_axis_command():
    for target, command in axis.EMIT_TARGETS.items():
        assert command.startswith("emit-"), target
        assert " " not in command, target
