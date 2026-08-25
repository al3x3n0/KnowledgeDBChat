"""Reading a core model's tunable surface, and addressing it correctly.

The paths gem5 prints in config.ini cannot be assigned to, and the difference
is not cosmetic: `FUList03.opList4` must become `FUList[3].opList[4]`, and a
vector printed without a suffix still needs `[0]`. Both forms appear in models
this build ships, and getting either wrong produces a KeyError from deep
inside gem5 that names neither the flag nor the path.
"""

from __future__ import annotations

from app.services import agent_gem5_sandbox as gem5

# Shaped like O3CPU's: one pool, functional units numbered, ops numbered.
O3_CONFIG = """
[system.cpu]
type=DerivO3CPU
issueWidth=8
numROBEntries=192
fetchWidth=8

[system.cpu.instQueues.fuPool.FUList03.opList04]
opClass=FloatSqrt
opLat=24
pipelined=false

[system.cpu.instQueues.fuPool.FUList01.opList0]
opClass=IntMult
opLat=3
pipelined=true
"""

# Shaped like NeoverseV2's: several issue queues, each with a bare FUList.
NEOVERSE_CONFIG = """
[system.cpu]
type=DerivO3CPU
issueWidth=8
numROBEntries=320

[system.cpu.instQueues4.fuPool.FUList.opList26]
opClass=FloatSqrt
opLat=33
pipelined=false

[system.cpu.instQueues5.fuPool.FUList.opList26]
opClass=FloatSqrt
opLat=33
pipelined=false
"""


def test_numbered_vector_members_become_indices():
    assert (
        gem5._addressable("system.cpu.instQueues.fuPool.FUList03.opList4")
        == "system.cpu[0].instQueues[0].fuPool.FUList[3].opList[4]"
    )


def test_bare_vector_names_still_need_an_index():
    """NeoverseV2 prints `FUList` unsuffixed, and the unindexed path is rejected.

    Whether a name carries a number says how many members exist, not whether
    it is a vector -- a distinction that cost a validation round.
    """
    assert (
        gem5._addressable("system.cpu.instQueues4.fuPool.FUList.opList26")
        == "system.cpu[0].instQueues[4].fuPool.FUList[0].opList[26]"
    )


def test_non_vector_segments_are_left_alone():
    """`fuPool` is a plain child; indexing it would break a working path."""
    assert "fuPool." in gem5._addressable(
        "system.cpu.instQueues.fuPool.FUList03.opList4"
    )
    assert "fuPool[0]" not in gem5._addressable(
        "system.cpu.instQueues.fuPool.FUList03.opList4"
    )


def test_op_latencies_are_read_with_addressable_paths():
    parsed = gem5.parse_model_parameters(O3_CONFIG)
    by_class = {row["op_class"]: row for row in parsed["op_latencies"]}

    assert by_class["FloatSqrt"]["op_lat"] == 24
    assert by_class["FloatSqrt"]["pipelined"] is False
    assert by_class["IntMult"]["pipelined"] is True
    assert by_class["FloatSqrt"]["parameters"] == [
        "system.cpu[0].instQueues[0].fuPool.FUList[3].opList[4].opLat"
    ]


def test_an_op_class_in_several_queues_reports_every_path():
    """Setting one of them leaves the model partly untuned."""
    parsed = gem5.parse_model_parameters(NEOVERSE_CONFIG)
    rows = [r for r in parsed["op_latencies"] if r["op_class"] == "FloatSqrt"]

    assert len(rows) == 1, "the same op class should be one entry, not several"
    assert rows[0]["parameters"] == [
        "system.cpu[0].instQueues[4].fuPool.FUList[0].opList[26].opLat",
        "system.cpu[0].instQueues[5].fuPool.FUList[0].opList[26].opLat",
    ]


def test_core_parameters_are_reported_once_each():
    parsed = gem5.parse_model_parameters(O3_CONFIG)
    names = [row["name"] for row in parsed["cpu_parameters"]]

    assert names == sorted(names)
    assert len(names) == len(set(names))
    by_name = {row["name"]: row for row in parsed["cpu_parameters"]}
    assert by_name["issueWidth"]["value"] == 8
    assert by_name["issueWidth"]["parameter"] == "system.cpu[0].issueWidth"
    # `type=DerivO3CPU` is not a tunable number and must not be offered as one.
    assert "type" not in by_name


def test_a_wrong_path_explains_the_indexing_rule():
    """gem5 blames neither the flag nor the path; the tool must."""
    message = gem5.explain_gem5_failure(
        "AttributeError: 'SimObjectVector' object has no attribute '_children'",
        ["system.cpu.instQueues.fuPool.FUList[3].opList[4].opLat=10"],
    )

    assert "without an index" in message
    assert "system.cpu[0]" in message


def test_an_unknown_child_points_at_the_introspection_tool():
    message = gem5.explain_gem5_failure(
        "KeyError: 'FUList03'",
        ["system.cpu[0].instQueues[0].fuPool.FUList03.opList4.opLat=10"],
    )

    assert "describe_model_parameters" in message
    assert "FUList[3]" in message


def test_failures_without_overrides_are_not_blamed_on_overrides():
    message = gem5.explain_gem5_failure("KeyError: 'something else'", [])

    assert "parameter override" not in message


def test_the_tool_is_reachable_and_declared_everywhere():
    """A tool wired in one place and missing from another is invisible.

    Registration used to be spread across the schema list, the dispatch map,
    the catalog and the per-job policy, and an agent could only call a tool
    that appeared in all of them. This one now declares itself once in
    agent_core.tool_specs and the registries derive from it, so what is checked
    here is what each registry answers -- not which file the name appears in,
    which a comment would have satisfied.
    """
    from app.agent_core import tool_catalog
    from app.services import agent_job_tool_policy, agent_tools

    assert agent_tools.get_tool_by_name("describe_model_parameters") is not None
    assert tool_catalog.get_tool_metadata("describe_model_parameters") is not None
    assert "describe_model_parameters" in agent_job_tool_policy.get_tools_for_job_type(
        "research", {}
    )


def test_the_simulator_advertises_tuning_and_the_named_cores():
    """The description is the only place an agent learns these exist."""
    from app.services import agent_tools

    tool = agent_tools.get_tool_by_name("simulate_c_workload")
    properties = tool["parameters"]["properties"]

    assert "param_overrides" in properties
    assert "describe_model_parameters" in properties["param_overrides"]["description"]
    cpu_description = properties["cpu_type"]["description"]
    for model in ("NeoverseV2", "HPI", "ex5_big"):
        assert model in cpu_description
    # The deadlock is silent, so the description must warn about it.
    assert "fused multiply-add" in cpu_description


async def test_dispatch_forwards_overrides_to_the_sandbox(monkeypatch):
    """The parameter existing in the schema but dropped in dispatch is the
    exact defect this gap was: a capability that looks present and is not."""
    from app.services import agent_gem5_sandbox
    from app.services.agent_tool_dispatch import AgentToolExecutionContext
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    seen = {}

    async def _fake(**kwargs):
        seen.update(kwargs)
        return {"success": True, "data": {"cycles": 1.0}}

    monkeypatch.setattr(agent_gem5_sandbox, "simulate_c_workload", _fake)

    executor = AutonomousAgentExecutor()
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id=None, job=None, state={}
    )
    provider = executor.tool_registry.resolve("simulate_c_workload", ctx)
    await provider.execute(
        "simulate_c_workload",
        {
            "code": "int main(void){return 0;}",
            "cpu_type": "O3CPU",
            "param_overrides": ["system.cpu[0].issueWidth=4"],
        },
        ctx,
    )

    assert seen["param_overrides"] == ["system.cpu[0].issueWidth=4"]


async def test_a_single_override_string_is_accepted(monkeypatch):
    from app.services import agent_gem5_sandbox
    from app.services.agent_tool_dispatch import AgentToolExecutionContext
    from app.services.autonomous_agent_executor import AutonomousAgentExecutor

    seen = {}

    async def _fake(**kwargs):
        seen.update(kwargs)
        return {"success": True, "data": {"cycles": 1.0}}

    monkeypatch.setattr(agent_gem5_sandbox, "simulate_c_workload", _fake)

    executor = AutonomousAgentExecutor()
    ctx = AgentToolExecutionContext(
        mode="autonomous", db=None, service=None, user_id=None, job=None, state={}
    )
    provider = executor.tool_registry.resolve("simulate_c_workload", ctx)
    await provider.execute(
        "simulate_c_workload",
        {
            "code": "int main(void){return 0;}",
            "param_overrides": "system.cpu[0].issueWidth=4",
        },
        ctx,
    )

    assert seen["param_overrides"] == ["system.cpu[0].issueWidth=4"]
