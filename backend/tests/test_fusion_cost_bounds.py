"""What fusing a sequence could save, without inventing the fused instruction.

llvm-mca cannot cost an instruction that does not exist. Naming a real one to
stand for it puts the answer at the mercy of that choice, and two live runs
concluded a candidate was a regression on exactly that basis -- neither
conclusion checkable. A range needs no such choice: the fused form cannot beat
the slowest operation it still has to perform, and must beat the sequence it
replaces to be worth building.
"""

from __future__ import annotations

import pytest

from app.services.agent_compiler_sandbox import (
    cost_fusion_candidate,
    fusion_saving_bounds,
)


def test_the_floor_is_the_slowest_operation_the_fusion_still_performs():
    bounds = fusion_saving_bounds(34.0, {"fsqrt": 17.0, "fdiv": 17.0})

    assert bounds["slowest_constituent"] == "fsqrt"
    assert bounds["slowest_constituent_cycles"] == 17.0
    assert bounds["max_saving_per_occurrence"] == 17.0
    assert bounds["min_saving_per_occurrence"] == 0.0
    assert bounds["worth_pursuing"] is True


def test_a_sequence_no_better_than_its_slowest_part_cannot_pay():
    """A verdict reachable without modelling the instruction at all."""
    bounds = fusion_saving_bounds(17.0, {"fsqrt": 17.0, "fmov": 1.0})

    assert bounds["max_saving_per_occurrence"] == 0.0
    assert bounds["worth_pursuing"] is False


def test_the_saving_never_goes_negative():
    """A negative saving would mean the fused form is slower than the parts it
    replaces, which says the measurement is wrong, not the candidate."""
    bounds = fusion_saving_bounds(3.0, {"fsqrt": 17.0})

    assert bounds["max_saving_per_occurrence"] == 0.0


def test_the_range_is_labelled_as_a_range_and_why():
    bounds = fusion_saving_bounds(10.0, {"fmul": 5.0, "fadd": 5.0})

    assert "cannot cost an instruction that does not exist" in bounds["note"]
    assert "stand-in" in bounds["note"]


def test_no_constituents_is_reported_rather_than_assumed():
    assert "error" in fusion_saving_bounds(10.0, {})


@pytest.mark.asyncio
async def test_a_pattern_is_required():
    result = await cost_fusion_candidate(pattern="", cpu="neoverse-n1")

    assert "pattern is required" in result["error"]


@pytest.mark.asyncio
async def test_a_core_must_be_named():
    """A cycle count is a property of a specific core."""
    result = await cost_fusion_candidate(pattern="fsqrt fdiv | 0>1", cpu="")

    assert "cpu is required" in result["error"]


@pytest.mark.asyncio
async def test_shell_syntax_in_a_pattern_is_refused():
    result = await cost_fusion_candidate(pattern="fsqrt; rm -rf /", cpu="neoverse-n1")

    assert "unsupported characters" in result["error"]


@pytest.mark.asyncio
async def test_the_mode_must_be_one_that_means_something():
    result = await cost_fusion_candidate(
        pattern="fsqrt fdiv | 0>1", cpu="neoverse-n1", mode="fast"
    )

    assert "dependent" in result["error"] and "independent" in result["error"]


@pytest.mark.asyncio
async def test_a_single_instruction_is_not_a_fusion_candidate():
    result = await cost_fusion_candidate(pattern="fsqrt", cpu="neoverse-n1")

    assert "at least two instructions" in result["error"]


def test_the_tool_is_declared_everywhere_it_must_be():
    """A tool wired in one place and missing from another is invisible."""
    from app.agent_core import tool_catalog
    from app.services import agent_job_tool_policy, agent_tools

    tool = agent_tools.get_tool_by_name("cost_fusion_candidate")
    assert tool is not None
    assert set(tool["parameters"]["required"]) == {"pattern", "cpu"}
    # The reason for a range rather than a number belongs where it is chosen.
    assert "stand-in" in tool["description"]

    # Asserted as behaviour, not as text in a particular file. The declaration
    # now lives in agent_core.tool_specs and these registries derive from it,
    # so the file the name appears in is no longer the point -- and a grep
    # would have been satisfied by the name in a comment either way.
    assert tool_catalog.get_tool_metadata("cost_fusion_candidate") is not None
    assert "cost_fusion_candidate" in agent_job_tool_policy.get_tools_for_job_type(
        "research", {}
    )


async def test_dispatch_reaches_the_sandbox_function(monkeypatch):
    from app.services import agent_compiler_sandbox
    from app.services.agent_tool_dispatch import AgentToolExecutionContext
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    seen = {}

    async def _fake(**kwargs):
        seen.update(kwargs)
        return {"success": True, "data": {}}

    monkeypatch.setattr(agent_compiler_sandbox, "cost_fusion_candidate", _fake)

    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id=None, job=None, state={}
    )
    provider = AutonomousAgentExecutor().tool_registry.resolve(
        "cost_fusion_candidate", ctx
    )
    await provider.execute(
        "cost_fusion_candidate",
        {"pattern": "fsqrt fdiv | 0>1", "cpu": "neoverse-n1", "copies": 30},
        ctx,
    )

    assert seen["pattern"] == "fsqrt fdiv | 0>1"
    assert seen["cpu"] == "neoverse-n1"
    assert seen["copies"] == 30
