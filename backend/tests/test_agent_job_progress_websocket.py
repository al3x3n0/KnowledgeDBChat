from app.models.agent_job import AgentJob, AgentJobStatus


class _FakePubSub:
    def __init__(self, messages):
        self._messages = list(messages)
        self.subscribed = []
        self.unsubscribed = []
        self.closed = False

    async def subscribe(self, channel: str):
        self.subscribed.append(channel)

    async def get_message(self, ignore_subscribe_messages=True, timeout=1.0):
        if self._messages:
            return self._messages.pop(0)
        return None

    async def unsubscribe(self, channel: str):
        self.unsubscribed.append(channel)

    async def close(self):
        self.closed = True


class _FakeRedisClient:
    def __init__(self, pubsub):
        self._pubsub = pubsub
        self.closed = False

    def pubsub(self):
        return self._pubsub

    async def close(self):
        self.closed = True


def test_agent_job_progress_websocket_forwards_runtime_payload(
    client,
    db_session,
    test_user,
    auth_headers,
    monkeypatch,
):
    from app.core import database as core_database
    from app.api.endpoints import auth as auth_endpoints

    job = AgentJob(
        name="Runtime Progress Job",
        goal="Observe live graph runtime",
        job_type="research",
        user_id=test_user.id,
        status=AgentJobStatus.RUNNING.value,
        progress=12,
        iteration=3,
        max_iterations=20,
        max_tool_calls=20,
        max_llm_calls=20,
        max_runtime_minutes=30,
    )

    async def _seed():
        db_session.add(job)
        await db_session.commit()
        await db_session.refresh(job)

    import asyncio

    asyncio.get_event_loop().run_until_complete(_seed())

    token = auth_headers["Authorization"].split(" ", 1)[1]

    async def _fake_get_user_from_token(raw_token: str):
        assert raw_token == token
        return test_user

    class _SessionFactory:
        async def __aenter__(self):
            return db_session

        async def __aexit__(self, exc_type, exc, tb):
            return False

    pubsub = _FakePubSub(
        [
            {
                "type": "message",
                "data": (
                    '{"type":"progress","job_id":"%s","progress":55,"phase":"acting","status":"running",'
                    '"iteration":4,"phase_details":"Executed tool",'
                    '"execution_graph_runtime":{"verification_attempts":2,"verification_successes":1,'
                    '"graph_health":{"status":"warning","severity_score":20}},'
                    '"scope_observability_runtime":{"resolved_scope_id":"scope-1","scope_source":"config.source_id"},'
                    '"timestamp":"2026-03-10T00:00:00"}'
                )
                % str(job.id)
            },
            {
                "type": "message",
                "data": (
                    '{"type":"progress","job_id":"%s","progress":100,"phase":"completed","status":"completed",'
                    '"iteration":5,"timestamp":"2026-03-10T00:00:01"}'
                )
                % str(job.id)
            },
        ]
    )
    redis_client = _FakeRedisClient(pubsub)

    monkeypatch.setattr(auth_endpoints, "get_user_from_token", _fake_get_user_from_token)
    monkeypatch.setattr(core_database, "async_session_factory", lambda: _SessionFactory())
    monkeypatch.setattr("app.api.endpoints.agent_jobs.redis.from_url", lambda _url: redis_client)

    with client.websocket_connect(f"/api/v1/agent-jobs/{job.id}/progress?token={token}") as websocket:
        connected = websocket.receive_json()
        forwarded = websocket.receive_json()
        completed = websocket.receive_json()

    assert connected["type"] == "connected"
    assert connected["job_id"] == str(job.id)

    assert forwarded["type"] == "progress"
    assert forwarded["execution_graph_runtime"]["verification_attempts"] == 2
    assert forwarded["execution_graph_runtime"]["graph_health"]["status"] == "warning"
    assert forwarded["scope_observability_runtime"]["resolved_scope_id"] == "scope-1"

    assert completed["status"] == "completed"
    assert pubsub.subscribed == [f"agent_job:{job.id}:progress"]
    assert pubsub.unsubscribed == [f"agent_job:{job.id}:progress"]
    assert pubsub.closed is True
    assert redis_client.closed is True
