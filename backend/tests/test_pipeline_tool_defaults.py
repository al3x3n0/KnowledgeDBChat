from copy import deepcopy
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from app.models.agent_job import AgentJob
from app.services import agent_pipeline_binding, agent_pipeline_spec
from app.services.agent_action_service import AgentActionService
from app.services.agent_pipeline_tool_defaults import apply_defaults
from app.services.agent_tool_dispatch import AgentToolRegistry, FunctionToolProvider
from app.services.agent_tool_validation import validate_tool_params
from app.services.autonomous_agent_executor import (
    AutonomousAgentExecutor,
    _AutonomousRuntimeAdapter,
)


def job(defaults=None):
    return AgentJob(
        name="Defaults",
        goal="Search",
        job_type="research",
        config={
            "pipeline_stage": "search",
            "pipeline_tool_defaults": defaults or {},
        },
    )


def test_explicit_values_win_and_nested_defaults_are_not_mutated():
    configured = {"search_web": {"query": "baseline", "limit": 5, "nested": {"a": [1]}}}
    current = job(configured)
    action = {"tool": "search_web", "params": {"limit": 0}}
    before = deepcopy(action)
    merged = apply_defaults(action, current)
    assert merged["params"]["query"] == "baseline"
    assert merged["params"]["limit"] == 0
    merged["params"]["nested"]["a"].append(2)
    assert configured["search_web"]["nested"] == {"a": [1]}
    assert action == before
    assert (
        apply_defaults({"tool": "search_web", "params": {"query": None}}, current)[
            "params"
        ]["query"]
        is None
    )
    assert apply_defaults({"tool": "other", "params": {}}, current)["params"] == {}


def test_binding_clears_a_parents_defaults_for_the_next_stage():
    pipeline = agent_pipeline_spec.normalize(
        {
            "stages": [
                {
                    "id": "a",
                    "goal": "Check",
                    "contract": {"required_finding_types": ["implementation_verified"]},
                    "tool_defaults": {"check_implementation": {"flags": "-O2"}},
                },
                {
                    "id": "b",
                    "goal": "Check again",
                    "depends_on": ["a"],
                    "contract": {"required_finding_types": ["implementation_verified"]},
                },
            ]
        }
    )
    root = agent_pipeline_binding.bind(pipeline).roots[0]
    assert root["config"]["pipeline_tool_defaults"] == {
        "check_implementation": {"flags": "-O2"}
    }
    assert (
        root["chain_config"]["child_jobs"][0]["config"]["pipeline_tool_defaults"] == {}
    )


def test_partial_validation_allows_omitted_required_arguments_and_rejects_typos():
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "mode": {"type": "string", "enum": ["fast", "full"]},
        },
        "required": ["query"],
    }
    assert (
        validate_tool_params(
            "search", {"mode": "fast"}, schema=schema, partial=True, reject_unknown=True
        )
        is None
    )
    for values in ({"mode": "wrong"}, {"query": 3}, {"qurey": "x"}, {"_hidden": "x"}):
        assert validate_tool_params(
            "search", values, schema=schema, partial=True, reject_unknown=True
        )


@pytest.mark.asyncio
async def test_journal_scope_guard_and_provider_receive_the_effective_arguments(
    monkeypatch,
):
    from app.services.agent_execution_journal_service import (
        agent_execution_journal_service,
    )

    seen = {}

    async def begin(**kwargs):
        seen["journal"] = deepcopy(kwargs["action"]["params"])
        return None

    async def handle(params, context):
        seen["provider"] = deepcopy(params)
        return {"success": True}

    monkeypatch.setattr(agent_execution_journal_service, "begin_tool_call", begin)
    monkeypatch.setattr(
        agent_execution_journal_service, "complete_tool_call", AsyncMock()
    )
    executor = AutonomousAgentExecutor.__new__(AutonomousAgentExecutor)
    executor._validate_action_scope = lambda current, action: seen.update(
        scope=deepcopy(action["params"])
    )
    executor._get_scope_guard_config = lambda current: {"enforce": True}
    executor.tool_registry = AgentToolRegistry(
        [
            FunctionToolProvider(
                name="test", modes={"autonomous"}, handlers={"search_web": handle}
            )
        ]
    )
    result = await AgentActionService().act(
        executor,
        job({"search_web": {"query": "baseline"}}),
        {"tool": "search_web", "params": {}},
        {},
        None,
    )
    assert result["success"] is True
    assert seen == {
        "journal": {"query": "baseline"},
        "scope": {"query": "baseline"},
        "provider": {"query": "baseline"},
    }


@pytest.mark.asyncio
async def test_harness_approval_sees_the_effective_call_before_execution():
    executor = AutonomousAgentExecutor()
    current = job({"search_web": {"query": "baseline"}})
    executor._get_counterfactual_config = lambda current: {"enabled": False}
    executor._build_selection_explainability = lambda **kwargs: {}
    executor._evaluate_approval_checkpoint = Mock(
        side_effect=RuntimeError("stop at approval")
    )
    adapter = _AutonomousRuntimeAdapter(
        executor=executor,
        job=current,
        agent_def=None,
        user_settings=None,
        state={},
        db=AsyncMock(),
        start_time=datetime.utcnow(),
        max_runtime=timedelta(minutes=1),
        progress_callback=None,
    )
    adapter.externally_driven = True
    with pytest.raises(RuntimeError, match="stop at approval"):
        await adapter.act_phase({"action": {"tool": "search_web", "params": {}}})
    assert (
        executor._evaluate_approval_checkpoint.call_args.args[2]["params"]["query"]
        == "baseline"
    )


@pytest.mark.parametrize(
    "values,valid",
    [({"flags": "-O2"}, True), ({"flags": True}, False), ({"flgas": "-O2"}, False)],
)
def test_api_checks_defaults_before_launch(client, auth_headers, values, valid):
    response = client.post(
        "/api/v1/agent-pipelines/check",
        headers=auth_headers,
        json={
            "spec": {
                "stages": [
                    {
                        "id": "check",
                        "goal": "Check",
                        "tools": [],
                        "contract": {
                            "required_finding_types": ["implementation_verified"]
                        },
                        "tool_defaults": {"check_implementation": values},
                    }
                ]
            }
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["valid"] is valid
    if not valid:
        assert any(
            "tool defaults" in problem for problem in response.json()["problems"]
        )


def test_defaults_cannot_attach_a_tool_or_access_another_users_plugin(
    client, auth_headers
):
    for name in ("search_web", "p_private_tool"):
        response = client.post(
            "/api/v1/agent-pipelines/check",
            headers=auth_headers,
            json={
                "spec": {
                    "stages": [
                        {
                            "id": "check",
                            "goal": "Check",
                            "tools": [],
                            "contract": {
                                "required_finding_types": ["implementation_verified"]
                            },
                            "tool_defaults": {name: {"query": "baseline"}},
                        }
                    ]
                }
            },
        )
        assert response.status_code == 200
        assert response.json()["valid"] is False
