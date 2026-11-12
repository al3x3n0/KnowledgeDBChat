from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.models.agent_job import AgentJob
from app.services import agent_pipeline_binding as binding
from app.services import agent_pipeline_spec as spec
from app.services.agent_action_service import AgentActionService
from app.services.agent_chain_orchestration_service import (
    AgentChainOrchestrationService,
)
from app.services.agent_pipeline_tool_bindings import (
    apply_bindings,
    dependency_results,
    pointer_value,
)
from app.services.autonomous_agent_executor import AutonomousAgentExecutor


def pipeline(reference=None):
    return {
        "name": "study",
        "stages": [
            {
                "id": "produce",
                "goal": "Produce",
                "contract": {"required_finding_types": ["algorithm_spec"]},
            },
            {
                "id": "consume",
                "goal": "Check",
                "depends_on": ["produce"],
                "contract": {"required_finding_types": ["implementation_verified"]},
                "tool_bindings": {
                    "check_implementation": {
                        "code": reference
                        or {"stage": "produce", "path": "/structured_output/code"}
                    }
                },
            },
        ],
    }


@pytest.mark.parametrize(
    "value,path,expected",
    [
        ({"a/b": {"~key": [False, 0, None]}}, "/a~1b/~0key/0", False),
        ({"x": [0, None]}, "/x/0", 0),
        ({"x": [0, None]}, "/x/1", None),
        ({"x": 1}, "", {"x": 1}),
    ],
)
def test_json_pointers_preserve_types_and_escaped_keys(value, path, expected):
    assert pointer_value(value, path) == expected


@pytest.mark.parametrize("path", ["x", "/bad~escape", "/x/-1", "/x/01", "/x/9"])
def test_invalid_or_missing_paths_are_refused(path):
    with pytest.raises(ValueError):
        pointer_value({"x": [1]}, path)


def test_bindings_override_explicit_arguments_without_mutating_source_results():
    results = {"produce": {"value": [0, False]}}
    job = SimpleNamespace(
        config={
            "pipeline_stage": "consume",
            "pipeline_depends_on": ["produce"],
            "pipeline_tool_bindings": {
                "tool": {"arg": {"stage": "produce", "path": "/value"}}
            },
            "pipeline_input_results": results,
        }
    )
    output = apply_bindings({"tool": "tool", "params": {"arg": "model value"}}, job)
    assert output["params"]["arg"] == [0, False]
    output["params"]["arg"].append(2)
    assert results["produce"]["value"] == [0, False]


@pytest.mark.parametrize(
    "reference",
    [
        {"stage": "elsewhere", "path": "/x"},
        {"stage": "produce", "path": "x"},
        {"stage": "produce", "path": "/x", "fallback": 1},
    ],
)
def test_pipeline_rejects_sources_outside_dependencies_and_invalid_references(
    reference,
):
    assert spec.validate(spec.normalize(pipeline(reference)))


@pytest.mark.asyncio
async def test_chaining_provides_fresh_results_and_clears_stale_inherited_inputs():
    root = binding.bind(spec.normalize(pipeline())).roots[0]
    parent = AgentJob(
        name="Producer",
        goal="Produce",
        job_type="research",
        config={**root["config"], "pipeline_input_results": {"stale": {"value": 99}}},
        results={"structured_output": {"code": "int main(){}"}},
        chain_config=root["chain_config"],
        chain_depth=0,
    )
    db = SimpleNamespace(add=Mock(), flush=AsyncMock())
    child = await AgentChainOrchestrationService().create_chained_job(
        SimpleNamespace(), parent, root["chain_config"]["child_jobs"][0], db
    )
    assert child.config["pipeline_input_results"] == {"produce": parent.results}
    executor = AutonomousAgentExecutor.__new__(AutonomousAgentExecutor)
    action = executor._apply_default_scope_to_action(
        {"tool": "check_implementation", "params": {"code": "wrong"}}, child
    )
    assert action["params"]["code"] == "int main(){}"
    parent.results["structured_output"]["code"] = "edited"
    assert (
        child.config["pipeline_input_results"]["produce"]["structured_output"]["code"]
        == "int main(){}"
    )
    restarted = await AgentChainOrchestrationService().create_chained_job(
        SimpleNamespace(), parent, root["chain_config"]["child_jobs"][0], db
    )
    assert (
        restarted.config["pipeline_input_results"]["produce"]["structured_output"][
            "code"
        ]
        == "edited"
    )
    assert (
        child.config["pipeline_input_results"]["produce"]["structured_output"]["code"]
        == "int main(){}"
    )
    assert (
        child.config["inherited_data"]["parent_results"]["structured_output"]["code"]
        == "int main(){}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("inherit_config", [False, True])
@pytest.mark.parametrize("fan_in", [False, True])
async def test_child_attempts_isolate_templates_arguments_and_inherited_evidence(
    inherit_config, fan_in
):
    raw = pipeline()
    raw["stages"][1]["tool_defaults"] = {
        "check_implementation": {"cases": [{"stdin": "seed"}]}
    }
    raw["stages"].append(
        {
            "id": "finish",
            "goal": "Finish",
            "depends_on": ["consume"],
            "contract": {"required_finding_types": ["implementation_verified"]},
        }
    )
    root = binding.bind(spec.normalize(raw)).roots[0]
    parent = AgentJob(
        name="Produce",
        goal="Produce",
        job_type="research",
        chain_depth=0,
        config={**root["config"], "shared": {"values": [1]}},
        results={
            "structured_output": {"code": "original code"},
            "findings": [{"value": 1}],
        },
        chain_config={**root["chain_config"], "inherit_config": inherit_config},
    )
    definition = parent.chain_config["child_jobs"][0]
    payload = {
        "sibling_jobs": [
            {"pipeline": "study", "pipeline_stage": "peer", "results": {"value": 7}}
        ]
    }
    if fan_in:
        definition["config"].update(
            origin="swarm_fan_in_aggregator", pipeline_depends_on=["produce", "peer"]
        )
    original_definition, original_config, original_results, original_payload = map(
        deepcopy, [definition, parent.config, parent.results, payload]
    )
    executor = SimpleNamespace(
        _build_swarm_sibling_payload=AsyncMock(return_value=payload)
    )
    db = SimpleNamespace(add=Mock(), flush=AsyncMock())
    service = AgentChainOrchestrationService()
    first = await service.create_chained_job(executor, parent, definition, db)
    first.config["pipeline_tool_defaults"]["check_implementation"]["cases"][0][
        "stdin"
    ] = "child edit"
    first.chain_config["child_jobs"][0]["goal"] = "Changed locally"
    first.config["inherited_data"]["parent_results"]["structured_output"][
        "code"
    ] = "local evidence edit"
    if inherit_config:
        first.config["shared"]["values"].append(2)
    if fan_in:
        first.config["inherited_data"]["swarm"]["sibling_jobs"][0]["results"][
            "value"
        ] = 99
    assert definition == original_definition
    assert parent.config == original_config
    assert parent.results == original_results
    assert payload == original_payload
    parent.results["structured_output"]["code"] = "new code"
    second = await service.create_chained_job(executor, parent, definition, db)
    assert (
        first.config["pipeline_input_results"]["produce"]["structured_output"]["code"]
        == "original code"
    )
    assert (
        second.config["pipeline_input_results"]["produce"]["structured_output"]["code"]
        == "new code"
    )
    assert (
        second.config["pipeline_tool_defaults"]["check_implementation"]["cases"][0][
            "stdin"
        ]
        == "seed"
    )
    assert second.chain_config == definition["chain_config"]
    assert definition == original_definition


@pytest.mark.parametrize("value", [None, "", "   ", False, 0])
def test_an_empty_bound_source_scope_cannot_silently_use_the_default_scope(value):
    current = AgentJob(
        config={
            "pipeline_stage": "consume",
            "source_id": "default-scope",
            "pipeline_depends_on": ["produce"],
            "pipeline_input_results": {"produce": {"source": value}},
            "pipeline_tool_bindings": {
                "search_documents": {
                    "source_id": {"stage": "produce", "path": "/source"}
                }
            },
        }
    )
    executor = AutonomousAgentExecutor.__new__(AutonomousAgentExecutor)
    action = executor._apply_default_scope_to_action(
        {"tool": "search_documents", "params": {}}, current
    )
    assert "bound source scope is empty" in action["_pipeline_binding_error"]
    assert action["params"]["source_id"] == value


def test_fan_in_uses_only_declared_sources_in_the_same_pipeline():
    parent = SimpleNamespace(
        config={"pipeline": "p", "pipeline_stage": "left"}, results={"v": 1}
    )
    config = {
        "pipeline": "p",
        "pipeline_stage": "join",
        "pipeline_depends_on": ["left", "right"],
    }
    peers = {
        "sibling_jobs": [
            {"pipeline": "p", "pipeline_stage": "right", "results": {"v": 2}},
            {"pipeline": "other", "pipeline_stage": "left", "results": {"v": 999}},
            {"pipeline": "p", "pipeline_stage": "unrelated", "results": {"v": 999}},
        ]
    }
    assert dependency_results(parent, config, peers) == {
        "left": {"v": 1},
        "right": {"v": 2},
    }


@pytest.mark.asyncio
async def test_missing_bound_value_does_not_reach_provider_or_fall_back_to_model_arguments():
    child = binding.bind(spec.normalize(pipeline())).roots[0]["chain_config"][
        "child_jobs"
    ][0]
    current = AgentJob(config=child["config"])
    executor = AutonomousAgentExecutor.__new__(AutonomousAgentExecutor)
    executor.tool_registry = SimpleNamespace(try_execute=AsyncMock())
    result = await AgentActionService()._act_unjournaled(
        executor,
        current,
        {"tool": "check_implementation", "params": {"code": "model fallback"}},
        {},
        None,
    )
    assert result["success"] is False
    assert "Cannot bind check_implementation.code" in result["error"]
    executor.tool_registry.try_execute.assert_not_called()


def test_api_checks_bound_parameter_names_before_launch(client, auth_headers):
    raw = pipeline()
    response = client.post(
        "/api/v1/agent-pipelines/check", headers=auth_headers, json={"spec": raw}
    )
    assert response.status_code == 200 and response.json()["valid"], response.text
    raw["stages"][1]["tool_bindings"] = {
        "check_implementation": {"typo": {"stage": "produce", "path": "/value"}}
    }
    response = client.post(
        "/api/v1/agent-pipelines/check", headers=auth_headers, json={"spec": raw}
    )
    assert not response.json()["valid"]
    assert any("unknown argument" in problem for problem in response.json()["problems"])
