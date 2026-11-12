"""Stage controls must hold at dispatch and reach the real Docker adapter."""

import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.agent_core.tool_specs.spec import ToolSpec
from app.core.config import settings
from app.models.agent_job import AgentJob
from app.models.plugin import Plugin, PluginInstallation
from app.schemas.docker_tool import DockerToolExecutionResult
from app.services import agent_pipeline_binding, agent_pipeline_spec, plugin_registry
from app.services.agent_action_service import AgentActionService
from app.services.agent_execution_journal_service import agent_execution_journal_service
from app.services.agent_pipeline_binding_preview import preview
from app.services.agent_plugin_tool_provider import PluginToolProvider
from app.services.agent_tool_dispatch import AgentToolRegistry
from app.services.agent_tool_providers.output_state import (
    build_autonomous_output_state_provider,
)
from app.services.autonomous_agent_executor import AutonomousAgentExecutor
from app.services.docker_tool_executor import docker_executor
from app.services.plugin_manifest import validate_manifest

TOOL = "p_dataset_convert"
IMAGE = "dataset-tools:7"


def executor(specs=(), registry=None):
    runtime = AutonomousAgentExecutor.__new__(AutonomousAgentExecutor)
    runtime._contributed_specs = list(specs)
    runtime.tool_registry = registry or SimpleNamespace(try_execute=AsyncMock())
    runtime._validate_action_scope = Mock(return_value=None)
    runtime._get_scope_guard_config = lambda job: {"enforce": True}
    return runtime


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result_key,tool,params,state_key",
    [
        (
            "structured_output",
            "set_output_schema",
            {"schema": {"code": "checked code"}},
            "output_schema",
        ),
        (
            "formatted_outputs",
            "format_as_table",
            {"title": "Results", "columns": ["Value"], "rows": [["checked"]]},
            "formatted_outputs",
        ),
        (
            "formatted_outputs",
            "format_as_report",
            {
                "title": "Results",
                "sections": [{"heading": "Check", "content": "Passed"}],
            },
            "formatted_outputs",
        ),
    ],
)
async def test_evidence_only_stage_executes_its_required_result_writer(
    result_key, tool, params, state_key, journal
):
    pipeline = agent_pipeline_spec.normalize(
        {
            "stages": [
                {
                    "id": "check",
                    "goal": "Check the implementation and record the result",
                    "tools": [],
                    "tool_defaults": {tool: params},
                    "contract": {
                        "required_finding_types": ["implementation_verified"],
                        "required_result_keys": [result_key],
                    },
                }
            ]
        }
    )
    root = agent_pipeline_binding.bind(pipeline).roots[0]
    runtime = executor()
    runtime.tool_registry = AgentToolRegistry(
        [build_autonomous_output_state_provider(runtime)]
    )
    job = AgentJob(
        name=root["name"],
        goal=root["goal"],
        job_type=root["job_type"],
        config=root["config"],
    )
    state = {}
    result = await AgentActionService().act(
        runtime, job, {"tool": tool, "params": {}}, state, None
    )
    assert result["success"], result
    assert state[state_key]
    assert journal["params"] == params


@pytest.fixture
def journal(monkeypatch):
    seen = {}

    async def begin(**kwargs):
        seen["params"] = deepcopy(kwargs["action"]["params"])
        return None

    monkeypatch.setattr(agent_execution_journal_service, "begin_tool_call", begin)
    monkeypatch.setattr(
        agent_execution_journal_service, "complete_tool_call", AsyncMock()
    )
    return seen


@pytest.fixture
async def container_stage(monkeypatch, db_session, test_user):
    monkeypatch.setattr(settings, "CUSTOM_TOOL_DOCKER_ENABLED", True)
    manifest = validate_manifest(
        {
            "id": "dataset",
            "name": "Dataset tools",
            "version": "1.0.0",
            "contributes": {
                "tools": [
                    {
                        "name": "convert",
                        "description": "Convert a dataset",
                        "tool_type": "docker_container",
                        "job_types": ["research"],
                        "parameters_schema": {
                            "type": "object",
                            "required": ["dataset"],
                            "properties": {
                                "dataset": {"type": "string"},
                                "limit": {"type": "integer"},
                                "mode": {"type": "string", "enum": ["fast", "full"]},
                                "document_id": {"type": "string", "format": "uuid"},
                            },
                        },
                        "config": {
                            "image": IMAGE,
                            "command": ["convert"],
                            "timeout_seconds": 45,
                        },
                    }
                ]
            },
        }
    )
    plugin = Plugin(
        slug="dataset",
        name="Dataset tools",
        version="1.0.0",
        source="user",
        owner_id=test_user.id,
        manifest=manifest,
    )
    db_session.add(plugin)
    await db_session.flush()
    db_session.add(
        PluginInstallation(plugin_id=plugin.id, user_id=test_user.id, is_enabled=True)
    )
    await db_session.flush()
    contributions = await plugin_registry.contributions_for_user(
        db_session, test_user.id
    )
    pipeline = agent_pipeline_spec.normalize(
        {
            "name": "data",
            "stages": [
                {
                    "id": "produce",
                    "goal": "Find data",
                    "contract": {"required_finding_types": ["algorithm_spec"]},
                },
                {
                    "id": "convert",
                    "goal": "Convert data",
                    "depends_on": ["produce"],
                    "contract": {"required_finding_types": ["implementation_verified"]},
                    "tools": [TOOL],
                    "tool_defaults": {TOOL: {"limit": 10}},
                    "tool_bindings": {
                        TOOL: {"dataset": {"stage": "produce", "path": "/dataset"}}
                    },
                },
            ],
        }
    )
    config = agent_pipeline_binding.bind(pipeline).roots[0]["chain_config"][
        "child_jobs"
    ][0]["config"]
    config["pipeline_input_results"] = {
        "produce": {"dataset": "verified upstream data"}
    }
    job = AgentJob(
        name="Convert",
        goal="Convert",
        job_type="research",
        user_id=test_user.id,
        config=config,
    )
    runtime = executor(contributions.specs, AgentToolRegistry([PluginToolProvider()]))
    return runtime, job


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [None, 0, 2])
async def test_bound_inputs_defaults_and_configured_image_reach_docker(
    monkeypatch, container_stage, db_session, test_user, journal, limit
):
    runtime, job = container_stage
    monkeypatch.setattr(
        docker_executor, "docker_available", AsyncMock(return_value=True)
    )
    run = AsyncMock(
        return_value=DockerToolExecutionResult(
            success=True, exit_code=0, stdout="converted", duration_seconds=0.1
        )
    )
    monkeypatch.setattr(docker_executor, "execute", run)
    params = {"dataset": "model guess", "image": "model-image:latest"}
    if limit is not None:
        params["limit"] = limit
    original = deepcopy(params)
    result = await AgentActionService().act(
        runtime, job, {"tool": TOOL, "params": params}, {}, db_session
    )
    assert result["success"] is True
    call = run.await_args.kwargs
    assert call["config"].image == IMAGE
    assert call["config"].timeout_seconds == 45
    assert call["user_id"] == test_user.id
    assert (
        json.loads(call["execution_input"].stdin_data)
        == journal["params"]
        == {
            "dataset": "verified upstream data",
            "limit": 10 if limit is None else limit,
            "image": "model-image:latest",
        }
    )
    assert params == original
    assert job.config["pipeline_tool_defaults"][TOOL] == {"limit": 10}


@pytest.mark.asyncio
@pytest.mark.parametrize("input_source", ["binding", "model"])
@pytest.mark.parametrize(
    "argument,value,message",
    [
        ("limit", True, "field limit should be integer"),
        ("limit", "ten", "field limit should be integer"),
        ("mode", "turbo", "field mode should be one of fast, full"),
        ("document_id", "external-document", "field document_id should be a UUID"),
        ("dataset", "", "required field dataset is empty"),
    ],
)
async def test_plugin_schema_rejects_bad_values_before_docker_dispatch(
    monkeypatch,
    container_stage,
    db_session,
    journal,
    argument,
    value,
    message,
    input_source,
):
    runtime, job = container_stage
    params = {"limit": 3, "dataset": "model guess"}
    if input_source == "binding":
        job.config["pipeline_tool_bindings"][TOOL][argument] = {
            "stage": "produce",
            "path": "/bound_value",
        }
        job.config["pipeline_input_results"]["produce"]["bound_value"] = value
        stage = agent_pipeline_spec.PipelineStage(
            id="convert",
            goal="Convert",
            depends_on=("produce",),
            tool_bindings=job.config["pipeline_tool_bindings"],
            tool_defaults=job.config["pipeline_tool_defaults"],
        )
        source = SimpleNamespace(
            job=SimpleNamespace(
                id=uuid4(),
                status="completed",
                results=job.config["pipeline_input_results"]["produce"],
            ),
            contract_met=True,
        )
        schema = next(
            spec.parameters for spec in runtime._contributed_specs if spec.name == TOOL
        )
        rows = preview(stage, {"produce": source}, {TOOL: {"parameters": schema}})[
            "items"
        ]
        row = next(item for item in rows if item["argument"] == argument)
        assert row["status"] == "invalid" and message in row["error"]
    else:
        job.config["pipeline_tool_bindings"][TOOL].pop(argument, None)
        params[argument] = value
    available = AsyncMock()
    run = AsyncMock()
    monkeypatch.setattr(docker_executor, "docker_available", available)
    monkeypatch.setattr(docker_executor, "execute", run)
    provider = AsyncMock(wraps=runtime.tool_registry.try_execute)
    runtime.tool_registry.try_execute = provider
    result = await AgentActionService().act(
        runtime,
        job,
        {"tool": TOOL, "params": params},
        {},
        db_session,
    )
    assert not result["success"]
    assert message in result["error"]
    assert journal["params"][argument] == value
    provider.assert_not_called()
    available.assert_not_called()
    run.assert_not_called()


@pytest.mark.asyncio
async def test_a_missing_binding_cannot_fall_back_and_start_a_container(
    monkeypatch, container_stage, db_session, journal
):
    runtime, job = container_stage
    job.config["pipeline_input_results"] = {"produce": {}}
    available = AsyncMock()
    monkeypatch.setattr(docker_executor, "docker_available", available)
    result = await AgentActionService().act(
        runtime,
        job,
        {"tool": TOOL, "params": {"dataset": "model guess"}},
        {},
        db_session,
    )
    assert not result["success"]
    assert "does not exist" in result["error"]
    available.assert_not_called()
    runtime._validate_action_scope.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "restriction", ["selection", "denylist", "role", "job_type", "cap"]
)
async def test_a_direct_plugin_call_cannot_bypass_stage_or_job_policy(
    restriction, journal
):
    spec = ToolSpec(TOOL, "Convert", {}, job_types=("research",))
    config = {
        "pipeline_stage": "convert",
        "pipeline_tools": [TOOL],
        "pipeline_tool_defaults": {TOOL: {"dataset": "a default cannot grant access"}},
    }
    state = {}
    if restriction == "selection":
        config["pipeline_tools"] = []
    elif restriction == "denylist":
        config["blocked_tools"] = [TOOL]
    elif restriction == "role":
        state["skill_profile"] = {"blocked_tools": [TOOL]}
    elif restriction == "job_type":
        spec = ToolSpec(TOOL, "Convert", {}, job_types=("coding",))
    else:
        config.update(
            pipeline_tools=[TOOL, "check_implementation"], skill_profile_max_tools=1
        )
    runtime = executor([spec])
    result = await AgentActionService().act(
        runtime,
        AgentJob(job_type="research", config=config),
        {"tool": TOOL, "params": {}},
        state,
        None,
    )
    assert not result["success"]
    assert "not available to pipeline stage 'convert'" in result["error"]
    runtime.tool_registry.try_execute.assert_not_called()
    runtime._validate_action_scope.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool,allowed",
    [
        ("search_web", False),
        ("check_implementation", True),
        ("extract_algorithm_spec", True),
    ],
)
async def test_evidence_tools_and_prerequisites_stay_executable(tool, allowed, journal):
    config = {
        "pipeline_stage": "check",
        "pipeline_tools": [],
        "goal_contract": {"required_finding_types": ["implementation_verified"]},
    }
    runtime = executor()
    runtime.tool_registry.try_execute.return_value = (True, {"success": True})
    params = {"code": "int main(){}", "document_id": "an-ingested-paper"}
    result = await AgentActionService().act(
        runtime,
        AgentJob(job_type="research", config=config),
        {"tool": tool, "params": params},
        {},
        None,
    )
    assert result["success"] is allowed
    assert runtime.tool_registry.try_execute.await_count == int(allowed)
