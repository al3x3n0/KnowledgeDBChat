"""One declaration per tool, and every registry reading it.

Defining a tool meant editing four unrelated files: the schema a model reads,
the governance metadata, the per-job-type allowlist, and the evidence map a
plan is derived from. Sixteen commits in a year touched all four, and the
misses were silent -- a tool absent from a registry is not broken, just
quieter. These tests assert the registries now derive from the spec rather
than agree with it by hand.
"""

import pytest

from app.agent_core import tool_specs
from app.agent_core.tool_catalog import get_tool_metadata
from app.services import agent_evidence_map, agent_job_tool_policy
from app.services.agent_tools import AGENT_TOOLS

SPECS = tool_specs.all_specs()
JOB_TYPES = ("research", "coding", "data_analysis", "analysis", "custom")


def test_there_are_specs_to_check():
    """A guard over an empty registry passes without checking anything."""
    assert len(SPECS) >= 19


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
class TestEveryRegistryReadsTheSpec:
    def test_the_model_is_offered_the_declared_schema(self, spec):
        offered = [t for t in AGENT_TOOLS if t["name"] == spec.name]
        assert len(offered) == 1, f"{spec.name} appears {len(offered)} times"
        assert offered[0]["description"] == spec.description
        assert offered[0]["parameters"] == spec.parameters

    def test_governance_reads_the_declared_classification(self, spec):
        meta = get_tool_metadata(spec.name)
        assert meta is not None, f"{spec.name} has no catalog metadata"
        assert meta.effects == spec.effects
        assert meta.network == spec.network
        assert meta.cost_tier == spec.cost_tier
        assert meta.pii_risk == spec.pii_risk

    def test_the_job_types_that_may_call_it_are_the_declared_ones(self, spec):
        expected = set(spec.job_types) or set(JOB_TYPES)
        for job_type in JOB_TYPES:
            allowed = spec.name in agent_job_tool_policy.get_tools_for_job_type(
                job_type, {}
            )
            assert allowed is (
                job_type in expected
            ), f"{spec.name} on {job_type}: policy says {allowed}"


def test_the_evidence_map_is_exactly_the_specs_that_produce_evidence():
    """Four counter tools were missing from this map, so a contract asking for
    their findings was told no tool produced them. Deriving it removes the
    possibility rather than the instance."""
    mapped = {e.tool for e in agent_evidence_map.EVIDENCE_TOOLS}
    declared = {s.name for s in SPECS if s.produces}
    assert mapped == declared


def test_a_producing_spec_carries_its_evidence_into_the_map():
    for spec in SPECS:
        if not spec.produces:
            continue
        entry = next(
            e for e in agent_evidence_map.EVIDENCE_TOOLS if e.tool == spec.name
        )
        assert entry.produces == spec.produces
        assert entry.requires == spec.requires
        assert entry.consumes == spec.consumes


def test_a_tool_that_produces_nothing_stays_out_of_the_map():
    """Silence is the honest default: a tool with no declared output should not
    appear to promise evidence a planner can chain on."""
    mapped = {e.tool for e in agent_evidence_map.EVIDENCE_TOOLS}
    for spec in SPECS:
        if not spec.produces:
            assert spec.name not in mapped


def test_every_spec_can_actually_be_run():
    """A declaration is not a capability until something answers the call."""
    from tests.test_capability_reachability import dispatchable_tools

    undispatchable = sorted({s.name for s in SPECS} - dispatchable_tools())
    assert not undispatchable, f"declared but nothing handles them: {undispatchable}"


def test_evidence_requirements_name_tools_that_exist():
    """A chain derived backwards from a contract has to be runnable."""
    names = {s.name for s in SPECS}
    for spec in SPECS:
        for required in spec.requires:
            assert required in names, f"{spec.name} requires unknown {required}"


def test_declared_classifications_use_known_values():
    for spec in SPECS:
        assert spec.effects in {"read", "write"}
        assert spec.network in {"none", "egress"}
        assert spec.cost_tier in {"low", "medium", "high"}
        assert spec.pii_risk in {"low", "medium", "high"}


def test_a_spec_is_not_also_a_hand_written_schema():
    """The point of the move is that there is one declaration, not two that
    happen to match today."""
    from app.services import agent_tools

    hand_written = agent_tools.AGENT_TOOLS[: -len(tool_specs.schemas())]
    duplicated = {t["name"] for t in hand_written} & tool_specs.spec_names()
    assert not duplicated, f"declared twice: {sorted(duplicated)}"


def test_one_declaration_is_enough(monkeypatch):
    """The property the move exists for.

    Verified end to end separately, by reloading the four registry modules
    around an invented spec and watching it appear in all of them; that check
    is not safe to keep in a shared suite, because reloading a module other
    tests hold references into swaps the objects under them. What is pinned
    here is the derivation every registry reads at import.
    """
    invented = tool_specs.ToolSpec(
        name="measure_invented_thing",
        description="An invented tool, declared once.",
        parameters={"type": "object", "properties": {}},
        effects="write",
        cost_tier="high",
        produces=("invented_measurement",),
    )
    monkeypatch.setattr(tool_specs, "TOOL_SPECS", SPECS + (invented,))
    monkeypatch.setattr(
        tool_specs, "_BY_NAME", {s.name: s for s in tool_specs.TOOL_SPECS}
    )

    assert any(s["name"] == "measure_invented_thing" for s in tool_specs.schemas())
    assert "measure_invented_thing" in tool_specs.tools_for_job_type("research")
    assert tool_specs.spec_for("measure_invented_thing").effects == "write"
    assert "measure_invented_thing" in tool_specs.spec_names()


def test_a_spec_limited_to_one_job_type_is_offered_only_there(monkeypatch):
    narrow = tool_specs.ToolSpec(
        name="coding_only_thing",
        description="Scoped to one job type.",
        parameters={"type": "object", "properties": {}},
        job_types=("coding",),
    )
    monkeypatch.setattr(tool_specs, "TOOL_SPECS", SPECS + (narrow,))

    assert "coding_only_thing" in tool_specs.tools_for_job_type("coding")
    assert "coding_only_thing" not in tool_specs.tools_for_job_type("research")
