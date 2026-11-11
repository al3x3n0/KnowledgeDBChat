"""A stage's attached tools reach its runtime without bypassing job policy."""

import json

import pytest

from app.agent_core import research_programs, tool_specs
from app.agent_core.tool_specs.spec import ToolSpec
from app.services import agent_pipeline_binding as binding
from app.services import agent_pipeline_spec as spec
from app.services import agent_pipeline_vocabulary, plugin_registry
from app.services.agent_job_tool_policy import get_tools_for_job_type


def stage(name, **over):
    return {
        "id": name,
        "goal": "Check the implementation",
        "contract": {"required_finding_types": ["implementation_verified"]},
        **over,
    }


def test_stage_selection_survives_binding_and_does_not_leak_to_children():
    pipeline = spec.normalize(
        {
            "stages": [
                stage("first", tools=["search_documents"]),
                stage("second", depends_on=["first"]),
            ]
        }
    )
    root = binding.bind(pipeline).roots[0]
    child = root["chain_config"]["child_jobs"][0]
    assert root["config"]["pipeline_tools"] == ["search_documents"]
    assert child["config"]["pipeline_tools"] is None
    assert "search_documents" in spec.plan(pipeline).stages[0].tools
    assert "check_implementation" in spec.plan(pipeline).stages[0].tools


def test_runtime_keeps_evidence_tools_and_permitted_extras_only():
    plugin = ToolSpec(
        "p_solver_solve", "Solve constraints", {}, job_types=("research",)
    )
    cfg = {
        "pipeline_stage": "check",
        "pipeline_tools": ["p_solver_solve", "apply_patch"],
        "goal_contract": stage("check")["contract"],
    }
    offered = get_tools_for_job_type("research", cfg, contributed=[plugin])
    assert set(offered) == {
        "p_solver_solve",
        "extract_algorithm_spec",
        "check_implementation",
    }
    cfg["blocked_tools"] = ["p_solver_solve"]
    assert set(get_tools_for_job_type("research", cfg, contributed=[plugin])) == {
        "extract_algorithm_spec",
        "check_implementation",
    }
    assert get_tools_for_job_type(
        "research",
        cfg,
        profile={"blocked_tools": ["check_implementation"]},
        contributed=[plugin],
    ) == ["extract_algorithm_spec"]


def test_evidence_only_skill_stage_can_load_its_procedure():
    offered = get_tools_for_job_type(
        "research",
        {
            "pipeline_stage": "process",
            "pipeline_tools": [],
            "goal_contract": {"required_finding_types": ["skill_process_data"]},
        },
    )
    assert set(offered) == {"load_sandbox_skill", "run_sandbox_skill"}


@pytest.mark.parametrize(
    "result_key,writers",
    [
        ("structured_output", {"set_output_schema"}),
        ("formatted_outputs", {"format_as_table", "format_as_report"}),
    ],
)
def test_evidence_only_stage_can_produce_its_required_results(result_key, writers):
    pipeline = spec.normalize(
        {
            "stages": [
                stage(
                    "check",
                    tools=[],
                    contract={
                        "required_finding_types": ["implementation_verified"],
                        "required_result_keys": [result_key],
                    },
                )
            ]
        }
    )
    root = binding.bind(pipeline).roots[0]
    offered = get_tools_for_job_type("research", root["config"])
    assert writers <= set(offered)
    assert writers <= set(spec.plan(pipeline).stages[0].tools)
    assert "search_web" not in offered
    root["config"]["blocked_tools"] = list(writers)
    assert not writers.intersection(get_tools_for_job_type("research", root["config"]))
    root["config"].pop("blocked_tools")
    assert not writers.intersection(
        get_tools_for_job_type(
            "research", root["config"], profile={"blocked_tools": list(writers)}
        )
    )


def test_required_result_writer_accepts_defaults_without_an_extra_selection(
    client, auth_headers
):
    response = client.post(
        "/api/v1/agent-pipelines/check",
        headers=auth_headers,
        json={
            "spec": {
                "stages": [
                    stage(
                        "check",
                        tools=[],
                        contract={
                            "required_finding_types": ["implementation_verified"],
                            "required_result_keys": ["structured_output"],
                        },
                        tool_defaults={"set_output_schema": {"schema": {"code": ""}}},
                    )
                ]
            }
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["valid"], response.json()["problems"]


def test_unset_selection_preserves_existing_job_policy():
    normal = get_tools_for_job_type("research", {})
    assert (
        get_tools_for_job_type(
            "research", {"pipeline_stage": "check", "pipeline_tools": None}
        )
        == normal
    )


def test_restricted_stage_keeps_its_declared_backward_edge_without_inheriting_one():
    pipeline = spec.normalize(
        {
            "name": "study",
            "stages": [
                stage("produce"),
                stage(
                    "check",
                    depends_on=["produce"],
                    may_revisit=["produce"],
                    tools=[],
                ),
                stage("report", depends_on=["check"], tools=[]),
            ],
        }
    )
    root = binding.bind(pipeline).roots[0]
    check = root["chain_config"]["child_jobs"][0]
    report = check["chain_config"]["child_jobs"][0]
    assert "request_stage_rerun" in get_tools_for_job_type("research", check["config"])
    assert "request_stage_rerun" in spec.plan(pipeline).stages[1].tools
    # Chaining merges parent configuration into a child. The child must clear
    # an edge that belongs only to its parent before that merge takes place.
    inherited = {**check["config"], **report["config"]}
    assert inherited["may_revisit"] == []
    assert "request_stage_rerun" not in get_tools_for_job_type("research", inherited)


def test_declared_backward_edge_accepts_rerun_defaults(client, auth_headers):
    response = client.post(
        "/api/v1/agent-pipelines/check",
        headers=auth_headers,
        json={
            "spec": {
                "stages": [
                    stage("produce"),
                    stage(
                        "check",
                        depends_on=["produce"],
                        may_revisit=["produce"],
                        tools=[],
                        tool_defaults={
                            "request_stage_rerun": {
                                "stage": "produce",
                                "reason": "The specification omits the rejection threshold.",
                            }
                        },
                    ),
                ]
            }
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["valid"], response.json()["problems"]


def test_api_refuses_unknown_and_incompatible_tools_before_launch(client, auth_headers):
    for name in ("not_a_tool", "apply_patch", "p_someone_else_private"):
        response = client.post(
            "/api/v1/agent-pipelines/check",
            headers=auth_headers,
            json={"spec": {"stages": [stage("check", tools=[name])]}},
        )
        assert response.status_code == 200
        assert response.json()["valid"] is False
        assert any(name in problem for problem in response.json()["problems"])


def test_api_lists_catalog_and_binds_selected_tools(client, auth_headers):
    response = client.get("/api/v1/agent-pipelines/vocabulary", headers=auth_headers)
    assert response.status_code == 200
    tools = {tool["name"]: tool for tool in response.json()["tools"]}
    assert tools["search_documents"]["source"] == "built-in"
    assert "parameters" in tools["search_documents"]
    assert "network" in tools["search_documents"]
    response = client.post(
        "/api/v1/agent-pipelines/bind",
        headers=auth_headers,
        json={"spec": {"stages": [stage("check", tools=["search_documents"])]}},
    )
    assert response.status_code == 200, response.text
    assert response.json()["chain_config"]["roots"][0]["config"]["pipeline_tools"] == [
        "search_documents"
    ]


@pytest.mark.parametrize("tools", ["search_documents", {}, [1], [""]])
def test_malformed_tool_selection_is_refused(client, auth_headers, tools):
    response = client.post(
        "/api/v1/agent-pipelines/check",
        headers=auth_headers,
        json={"spec": {"stages": [stage("check", tools=tools)]}},
    )
    assert response.status_code == 400


def test_catalog_combines_program_and_plugin_tools_without_chat_only_tools(monkeypatch):
    parameters = {
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"],
    }
    program_spec = ToolSpec(
        "solve",
        "Solve constraints",
        parameters,
        job_types=None,
        produces=("solution",),
        requires=("prepare",),
        consumes="a problem",
    )
    chat_only = ToolSpec("chat_only", "Chat only", {}, job_types=())
    program = research_programs.ResearchProgram(
        name="solvers", tool_specs=(program_spec,), sandbox_images=("solver-base:1",)
    )
    monkeypatch.setattr(research_programs, "installed", lambda: [program])
    monkeypatch.setattr(
        tool_specs, "STATIC_CATALOG", tool_specs.ToolCatalog([program_spec, chat_only])
    )
    plugin_spec = ToolSpec("p_api_fetch", "Fetch data", {}, job_types=("research",))
    plugin = plugin_registry.ContributedTool(
        id="api:fetch",
        name=plugin_spec.name,
        description="Fetch data",
        tool_type="webhook",
        config={
            "headers": {"Authorization": "private-key"},
            "url": "https://example.test/private",
        },
        parameters_schema={},
        plugin_slug="api",
        declared_name="fetch",
    )
    contributions = plugin_registry.ResolvedContributions(
        specs=[plugin_spec], tools={plugin_spec.name: plugin}
    )
    choices = {
        tool["name"]: tool
        for tool in agent_pipeline_vocabulary.tool_choices(contributions)
    }
    assert set(choices) == {"solve", "p_api_fetch"}
    assert choices["solve"]["source"] == "solvers"
    assert choices["p_api_fetch"]["execution"] == "webhook"
    assert choices["solve"]["parameters"] == parameters
    assert choices["solve"]["produces"] == ["solution"]
    assert choices["solve"]["requires"] == ["prepare"]
    assert choices["solve"]["program_images"] == ["solver-base:1"]
    assert choices["solve"]["image"] is None
    # Execution configuration is not part of the inspection response.
    assert "private-key" not in json.dumps(choices)
    assert "example.test/private" not in json.dumps(choices)


def test_container_inspection_exposes_image_without_environment_or_command():
    plugin_spec = ToolSpec("p_container_run", "Process data", {})
    plugin = plugin_registry.ContributedTool(
        id="container:run",
        name=plugin_spec.name,
        description="Process data",
        tool_type="docker_container",
        config={
            "image": "processor:2",
            "environment": {"TOKEN": "secret"},
            "command": "private-command",
        },
        parameters_schema={},
        plugin_slug="container",
        declared_name="run",
    )
    contributions = plugin_registry.ResolvedContributions(
        specs=[plugin_spec], tools={plugin_spec.name: plugin}
    )
    choice = next(
        tool
        for tool in agent_pipeline_vocabulary.tool_choices(contributions)
        if tool["name"] == plugin_spec.name
    )
    assert choice["image"] == "processor:2"
    assert "environment" not in choice
    assert "command" not in choice


@pytest.mark.asyncio
async def test_endpoint_uses_only_the_owners_installed_plugins(
    db_session, test_user, admin_user, client, auth_headers
):
    from app.models.plugin import Plugin, PluginInstallation
    from app.services.plugin_manifest import validate_manifest

    for owner, slug in [(test_user, "mine"), (admin_user, "theirs")]:
        manifest = validate_manifest(
            {
                "id": slug,
                "name": slug,
                "version": "1.0.0",
                "contributes": {
                    "tools": [
                        {
                            "name": "fetch",
                            "description": "Fetch records",
                            "tool_type": "transform",
                            "parameters_schema": {"type": "object", "properties": {}},
                            "config": {"template": "records"},
                            "job_types": ["research"],
                        }
                    ]
                },
            }
        )
        plugin = Plugin(
            slug=slug,
            name=slug,
            version="1.0.0",
            source="user",
            owner_id=owner.id,
            manifest=manifest,
        )
        db_session.add(plugin)
        await db_session.flush()
        db_session.add(
            PluginInstallation(plugin_id=plugin.id, user_id=owner.id, is_enabled=True)
        )
    await db_session.flush()
    response = client.get("/api/v1/agent-pipelines/vocabulary", headers=auth_headers)
    names = {tool["name"] for tool in response.json()["tools"]}
    assert "p_mine_fetch" in names
    assert "p_theirs_fetch" not in names
