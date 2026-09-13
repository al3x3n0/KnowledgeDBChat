import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest

from app.core.config import settings
from app.services.external_agent_gateway_service import (
    ExternalAgentGatewayError,
    ExternalAgentGatewayService,
)
from app.services.secret_service import SecretService


async def _public_resolver(_host, _port):
    return ["93.184.216.34"]


def _tool(**config_overrides):
    config = {
        "endpoint_url": "https://agent.example.test/invoke",
        "capabilities": ["research.summarize"],
        "auth_type": "none",
        "timeout_seconds": 10,
        **config_overrides,
    }
    return SimpleNamespace(
        id=uuid4(),
        name="Research Agent",
        is_enabled=True,
        config=config,
    )


def _compops_tool(**config_overrides):
    return _tool(
        **{
            "provider_type": "compops",
            "endpoint_url": "https://compops.example.test",
            "capabilities": [
                "compops.operators.list",
                "compops.runs.submit",
                "compops.studies.report",
            ],
            **config_overrides,
        }
    )


def _mlflow_tool(**config_overrides):
    return _tool(
        **{
            "provider_type": "mlflow",
            "endpoint_url": "https://mlflow.example.test",
            "capabilities": [
                "mlflow.experiments.search",
                "mlflow.runs.get",
                "mlflow.runs.search",
                "mlflow.artifacts.list",
                "mlflow.model_versions.get",
            ],
            **config_overrides,
        }
    )


@pytest.mark.asyncio
async def test_invokes_allowed_capability_and_returns_provenance():
    received = {}

    def handler(request):
        received["request"] = request
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "claims": [{"id": "claim-1"}],
            },
        )

    service = ExternalAgentGatewayService(
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )
    tool = _tool()

    result = await service.invoke(
        tool=tool,
        user=SimpleNamespace(id=uuid4()),
        db=None,
        capability="research.summarize",
        payload={"topic": "vectorization"},
        request_id="request-1",
    )

    assert received["request"].url.host == "agent.example.test"
    assert received["request"].headers["Idempotency-Key"] == "request-1"
    assert received["request"].headers["X-Request-ID"] == "request-1"
    assert result["output"]["status"] == "completed"
    assert result["provenance"]["external_agent_id"] == str(tool.id)
    assert result["provenance"]["capability"] == "research.summarize"
    assert result["provenance"]["request_id"] == "request-1"
    assert len(result["provenance"]["response_sha256"]) == 64


@pytest.mark.asyncio
async def test_rejects_capability_outside_manifest():
    service = ExternalAgentGatewayService(resolver=_public_resolver)

    with pytest.raises(ExternalAgentGatewayError, match="not allowed"):
        await service.invoke(
            tool=_tool(),
            user=SimpleNamespace(id=uuid4()),
            db=None,
            capability="code.execute",
            payload={},
        )


@pytest.mark.asyncio
async def test_rejects_private_network_resolution():
    async def private_resolver(_host, _port):
        return ["127.0.0.1"]

    service = ExternalAgentGatewayService(resolver=private_resolver)

    with pytest.raises(ExternalAgentGatewayError, match="non-public network"):
        await service.invoke(
            tool=_tool(),
            user=SimpleNamespace(id=uuid4()),
            db=None,
            capability="research.summarize",
            payload={},
        )


@pytest.mark.asyncio
async def test_allows_exact_private_hostname_from_server_allowlist(monkeypatch):
    async def private_resolver(_host, _port):
        return ["10.0.0.25"]

    monkeypatch.setattr(
        settings,
        "EXTERNAL_GATEWAY_PRIVATE_HOST_ALLOWLIST",
        "compops.internal",
    )
    service = ExternalAgentGatewayService(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"status": "ok"})
        ),
        resolver=private_resolver,
    )
    result = await service.invoke(
        tool=_compops_tool(
            endpoint_url="https://compops.internal",
            capabilities=["compops.health"],
        ),
        user=SimpleNamespace(id=uuid4()),
        db=None,
        capability="compops.health",
        payload={},
    )

    assert result["output"]["status"] == "ok"


@pytest.mark.asyncio
async def test_rejects_oversized_request_before_network_call():
    service = ExternalAgentGatewayService(resolver=_public_resolver)
    service.MAX_REQUEST_BYTES = 32

    with pytest.raises(ExternalAgentGatewayError, match="request exceeded size"):
        await service.invoke(
            tool=_tool(),
            user=SimpleNamespace(id=uuid4()),
            db=None,
            capability="research.summarize",
            payload={"content": "x" * 100},
        )


def test_requires_https_and_valid_capability_manifest():
    service = ExternalAgentGatewayService()

    with pytest.raises(ExternalAgentGatewayError, match="HTTPS"):
        service.validate_config(
            {
                "endpoint_url": "http://agent.example.test/invoke",
                "capabilities": ["research.summarize"],
            }
        )

    with pytest.raises(ExternalAgentGatewayError, match="Invalid"):
        service.validate_config(
            {
                "endpoint_url": "https://agent.example.test/invoke",
                "capabilities": ["../../shell"],
            }
        )


@pytest.mark.asyncio
async def test_compops_adapter_maps_read_capability_to_typed_rest_route():
    received = {}

    def handler(request):
        received["request"] = request
        return httpx.Response(200, json={"study_id": "study-1", "metric": "cycles"})

    service = ExternalAgentGatewayService(
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )
    result = await service.invoke(
        tool=_compops_tool(),
        user=SimpleNamespace(id=uuid4()),
        db=None,
        capability="compops.studies.report",
        payload={"study_id": "study-1", "metric": "cycles"},
        request_id="knowledgeops-1",
    )

    request = received["request"]
    assert request.method == "GET"
    assert request.url.path == "/v1/studies/study-1/report"
    assert request.url.params["metric"] == "cycles"
    assert result["provenance"]["provider_type"] == "compops"
    assert result["provenance"]["remote_references"] == {"study_id": "study-1"}


@pytest.mark.asyncio
async def test_compops_adapter_maps_write_capability_without_arbitrary_paths():
    received = {}

    def handler(request):
        received["request"] = request
        return httpx.Response(201, json={"id": "run-1", "status": "pending"})

    service = ExternalAgentGatewayService(
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )
    result = await service.invoke(
        tool=_compops_tool(),
        user=SimpleNamespace(id=uuid4()),
        db=None,
        capability="compops.runs.submit",
        payload={"request": {"workflow_id": "workflow-1", "workflow_version": 1}},
        request_id="knowledgeops-run-1",
    )

    request = received["request"]
    assert request.method == "POST"
    assert request.url.path == "/v1/runs"
    assert (
        json.loads(request.content)["knowledgeops_request_id"] == "knowledgeops-run-1"
    )
    assert result["output"]["id"] == "run-1"


def test_compops_adapter_rejects_capabilities_outside_typed_surface():
    service = ExternalAgentGatewayService()

    with pytest.raises(ExternalAgentGatewayError, match="Unsupported CompOps"):
        service.validate_config(
            {
                "provider_type": "compops",
                "endpoint_url": "https://compops.example.test",
                "capabilities": ["compops.admin.worker.install"],
            }
        )


@pytest.mark.asyncio
async def test_mlflow_adapter_maps_run_and_artifact_reads_to_rest_v2():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"run": {"info": {"run_id": "run-1"}}})

    service = ExternalAgentGatewayService(
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )
    tool = _mlflow_tool()

    result = await service.invoke(
        tool=tool,
        user=SimpleNamespace(id=uuid4()),
        db=None,
        capability="mlflow.runs.get",
        payload={"run_id": "run-1"},
        request_id="mlflow-read-1",
    )
    await service.invoke(
        tool=tool,
        user=SimpleNamespace(id=uuid4()),
        db=None,
        capability="mlflow.artifacts.list",
        payload={"run_id": "run-1", "path": "benchmarks"},
    )

    assert requests[0].method == "GET"
    assert requests[0].url.path == "/api/2.0/mlflow/runs/get"
    assert requests[0].url.params["run_id"] == "run-1"
    assert requests[1].url.path == "/api/2.0/mlflow/artifacts/list"
    assert requests[1].url.params["path"] == "benchmarks"
    assert result["provenance"]["provider_type"] == "mlflow"
    assert result["provenance"]["remote_references"]["run_id"] == "run-1"


@pytest.mark.asyncio
async def test_mlflow_adapter_bounds_run_search_and_forwards_json():
    received = {}

    def handler(request):
        received["request"] = request
        return httpx.Response(200, json={"runs": []})

    service = ExternalAgentGatewayService(
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )
    await service.invoke(
        tool=_mlflow_tool(),
        user=SimpleNamespace(id=uuid4()),
        db=None,
        capability="mlflow.runs.search",
        payload={
            "experiment_ids": ["1", "2"],
            "filter": "metrics.cycles < 1000",
            "max_results": 25,
            "order_by": ["metrics.cycles ASC"],
        },
    )

    request = received["request"]
    assert request.method == "POST"
    assert request.url.path == "/api/2.0/mlflow/runs/search"
    assert json.loads(request.content) == {
        "experiment_ids": ["1", "2"],
        "filter": "metrics.cycles < 1000",
        "max_results": 25,
        "order_by": ["metrics.cycles ASC"],
    }

    with pytest.raises(ExternalAgentGatewayError, match="between 1 and 100"):
        await service.invoke(
            tool=_mlflow_tool(),
            user=SimpleNamespace(id=uuid4()),
            db=None,
            capability="mlflow.runs.search",
            payload={"experiment_ids": ["1"], "max_results": 1000},
        )


def test_mlflow_adapter_rejects_untyped_capabilities():
    service = ExternalAgentGatewayService()

    with pytest.raises(ExternalAgentGatewayError, match="Unsupported MLflow"):
        service.validate_config(
            {
                "provider_type": "mlflow",
                "endpoint_url": "https://mlflow.example.test",
                "capabilities": ["mlflow.artifacts.delete"],
            }
        )


@pytest.mark.asyncio
async def test_mlflow_basic_auth_uses_vault_secret_without_exposing_it():
    received = {}

    def handler(request):
        received["authorization"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"experiments": []})

    secret_id = uuid4()
    user_id = uuid4()
    secret = SimpleNamespace(
        encrypted_value=SecretService().encrypt("researcher:correct-horse")
    )
    db = SimpleNamespace(
        execute=AsyncMock(
            return_value=SimpleNamespace(
                scalar_one_or_none=lambda: secret,
            )
        )
    )
    service = ExternalAgentGatewayService(
        transport=httpx.MockTransport(handler),
        resolver=_public_resolver,
    )
    result = await service.invoke(
        tool=_mlflow_tool(
            capabilities=["mlflow.experiments.search"],
            auth_type="basic",
            secret_id=str(secret_id),
        ),
        user=SimpleNamespace(id=user_id),
        db=db,
        capability="mlflow.experiments.search",
        payload={"max_results": 1},
    )

    assert received["authorization"] == "Basic cmVzZWFyY2hlcjpjb3JyZWN0LWhvcnNl"
    assert "correct-horse" not in repr(result)
