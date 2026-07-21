import json
import sys
import types

import pytest

from app.tasks.agent_job_tasks import _publish_job_progress


class _FakeRedisClient:
    def __init__(self):
        self.published = []
        self.closed = False

    async def publish(self, channel: str, message: str):
        self.published.append((channel, message))

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_publish_job_progress_includes_runtime_execution_fields(monkeypatch):
    fake_client = _FakeRedisClient()

    redis_asyncio_mod = types.ModuleType("redis.asyncio")
    redis_asyncio_mod.from_url = lambda _url: fake_client

    redis_mod = types.ModuleType("redis")
    redis_mod.asyncio = redis_asyncio_mod

    monkeypatch.setitem(sys.modules, "redis", redis_mod)
    monkeypatch.setitem(sys.modules, "redis.asyncio", redis_asyncio_mod)

    await _publish_job_progress(
        job_id="job-123",
        progress=42,
        phase="acting",
        status="running",
        iteration=7,
        phase_details="Executed tool",
        execution_graph_runtime={
            "verification_attempts": 2,
            "verification_successes": 1,
            "graph_health": {"status": "warning", "severity_score": 20},
        },
        scope_observability_runtime={
            "resolved_scope_id": "scope-1",
            "scope_source": "config.source_id",
        },
    )

    assert fake_client.closed is True
    assert len(fake_client.published) == 1

    channel, raw_message = fake_client.published[0]
    payload = json.loads(raw_message)

    assert channel == "agent_job:job-123:progress"
    assert payload["job_id"] == "job-123"
    assert payload["progress"] == 42
    assert payload["phase"] == "acting"
    assert payload["status"] == "running"
    assert payload["iteration"] == 7
    assert payload["phase_details"] == "Executed tool"
    assert payload["execution_graph_runtime"]["verification_attempts"] == 2
    assert payload["execution_graph_runtime"]["graph_health"]["status"] == "warning"
    assert payload["scope_observability_runtime"]["resolved_scope_id"] == "scope-1"
