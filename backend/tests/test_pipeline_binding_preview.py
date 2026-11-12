from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.models.agent_job import AgentJob
from app.services.agent_pipeline_binding_preview import (
    MAX_ITEMS,
    MAX_VALUE_CHARS,
    preview,
)
from app.services.agent_pipeline_spec import normalize


def spec():
    return {
        "name": "p",
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
                        "code": {"stage": "produce", "path": "/structured_output/code"}
                    }
                },
            },
        ],
    }


def recorded(value):
    return {
        "produce": SimpleNamespace(
            job=SimpleNamespace(
                id=uuid4(),
                results={"structured_output": {"code": value}},
                status="completed",
            ),
            contract_met=False,
        )
    }


TOOLS = {
    "check_implementation": {
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}
    }
}


def test_preview_reports_values_and_source_contract_without_executing_tools():
    entry = preview(
        normalize(spec()).by_id()["consume"], recorded("int main(){}"), TOOLS
    )["items"][0]
    assert entry["status"] == "resolved"
    assert entry["value_json"] == '"int main(){}"'
    assert entry["value_type"] == "string"
    assert entry["source_contract_satisfied"] is False
    assert entry["source_job_id"]
    source = recorded(["int main(){}"])
    normalized = preview(normalize(spec()).by_id()["consume"], source, TOOLS)["items"][
        0
    ]
    assert normalized["status"] == "resolved"
    assert normalized["value_json"] == '"int main(){}"'
    assert source["produce"].job.results["structured_output"]["code"] == [
        "int main(){}"
    ]


def test_missing_results_and_wrong_argument_types_are_visible():
    stage = normalize(spec()).by_id()["consume"]
    missing = preview(stage, {}, TOOLS)["items"][0]
    assert missing["status"] == "invalid" and "no recorded results" in missing["error"]
    wrong = preview(stage, recorded(42), TOOLS)["items"][0]
    assert wrong["status"] == "invalid" and "should be string" in wrong["error"]
    assert preview(stage, recorded("value"), {})["items"][0]["status"] == "invalid"


@pytest.mark.parametrize("value", [None, "", "   ", False, 0])
def test_preview_refuses_empty_bound_source_scope_like_runtime(value):
    raw = spec()
    raw["stages"][1]["tool_bindings"] = {
        "search_documents": {
            "source_id": {"stage": "produce", "path": "/structured_output/code"}
        }
    }
    tools = {
        "search_documents": {
            "parameters": {"properties": {"source_id": {"type": "string"}}}
        }
    }
    entry = preview(normalize(raw).by_id()["consume"], recorded(value), tools)["items"][
        0
    ]
    assert entry["status"] == "invalid"
    assert "bound source scope is empty" in entry["error"]


def test_value_and_row_limits_are_explicit():
    raw = spec()
    raw["stages"][1]["tool_bindings"]["check_implementation"] = {
        f"arg{i}": {"stage": "produce", "path": "/structured_output/code"}
        for i in range(MAX_ITEMS + 1)
    }
    tools = {
        "check_implementation": {
            "parameters": {
                "properties": {
                    f"arg{i}": {"type": "string"} for i in range(MAX_ITEMS + 1)
                }
            }
        }
    }
    response = preview(normalize(raw).by_id()["consume"], recorded("x" * 2000), tools)
    assert len(response["items"]) == MAX_ITEMS and response["omitted"] == 1
    assert response["items"][0]["truncated"] is True
    assert len(response["items"][0]["value_json"]) == MAX_VALUE_CHARS


@pytest.mark.asyncio
async def test_endpoint_uses_latest_owned_attempt_without_changing_the_run(
    db_session, test_user, client, auth_headers
):
    root = AgentJob(
        name="p: produce",
        goal="Produce",
        job_type="research",
        user_id=test_user.id,
        status="completed",
        config={"pipeline": "p", "pipeline_stage": "produce"},
        results={"structured_output": {"code": "old"}},
    )
    db_session.add(root)
    await db_session.flush()
    newer = AgentJob(
        name="p: produce again",
        goal="Produce",
        job_type="research",
        user_id=test_user.id,
        status="completed",
        root_job_id=root.id,
        config={"pipeline": "p", "pipeline_stage": "produce"},
        results={"structured_output": {"code": "new"}},
    )
    db_session.add(newer)
    await db_session.flush()
    response = client.post(
        f"/api/v1/agent-pipelines/runs/{root.id}/bindings/preview",
        headers=auth_headers,
        json={"spec": spec(), "stage": "consume"},
    )
    assert response.status_code == 200, response.text
    entry = response.json()["items"][0]
    assert entry["value_json"] == '"new"'
    assert entry["source_job_id"] == str(newer.id)
    assert root.results["structured_output"]["code"] == "old"


@pytest.mark.asyncio
async def test_another_users_run_is_not_discoverable(
    db_session, admin_user, client, auth_headers
):
    private = AgentJob(
        name="private",
        goal="Private",
        job_type="research",
        user_id=admin_user.id,
        config={"pipeline_stage": "produce"},
        results={"secret": "private-output"},
    )
    db_session.add(private)
    await db_session.flush()
    response = client.post(
        f"/api/v1/agent-pipelines/runs/{private.id}/bindings/preview",
        headers=auth_headers,
        json={"spec": spec(), "stage": "consume"},
    )
    assert response.status_code == 404
    assert "private-output" not in response.text
