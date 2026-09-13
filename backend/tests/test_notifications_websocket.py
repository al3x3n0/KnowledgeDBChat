import threading


class _FakePubSub:
    def __init__(self, messages):
        self._messages = list(messages)
        self.subscribed = []
        self.unsubscribed = []
        self.closed = False
        self.cleanup_completed = threading.Event()

    async def subscribe(self, channel: str):
        self.subscribed.append(channel)

    async def unsubscribe(self, channel: str):
        self.unsubscribed.append(channel)

    async def close(self):
        self.closed = True
        self.cleanup_completed.set()

    async def listen(self):
        for message in self._messages:
            yield message


class _FakeRedisClient:
    def __init__(self, pubsub):
        self._pubsub = pubsub
        self.closed = False

    def pubsub(self):
        return self._pubsub

    async def close(self):
        self.closed = True


def test_notifications_websocket_forwards_experiment_recovery_payload(
    client,
    test_user,
    monkeypatch,
):
    import redis.asyncio as aioredis

    from app.api.endpoints import notifications as notification_endpoints

    async def _fake_require_websocket_auth(websocket):
        await websocket.accept()
        return test_user

    pubsub = _FakePubSub(
        [
            {
                "type": "message",
                "data": (
                    '{"type":"notification","notification":{"id":"notif-1","notification_type":"experiment_run_update",'
                    '"title":"Experiment run failed","message":"Recovery remains open.","priority":"high",'
                    '"related_entity_type":"experiment_run","related_entity_id":"run-1",'
                    '"data":{"agent_job_id":"job-1","note_id":"note-1","launch_mode":"quick_start_claude_backend",'
                    '"final_phase":"fallback","source_name":"Knowledge Repo","fallback_attempted":true,'
                    '"fallback_ok":false,"failed_command_count":2,"first_failed_command":"npm --prefix frontend test",'
                    '"recovery_open":true,"recovery_reason":"fallback verification still failing",'
                    '"recommended_action":"Inspect failing fallback output",'
                    '"latest_operator_action":"restart","latest_operator_note":"Retry after fallback failure",'
                    '"latest_operator_status_before":"failed","latest_operator_status_after":"pending",'
                    '"latest_operator_at":"2026-03-10T01:00:00Z",'
                    '"latest_operator_outcome":"unresolved","latest_operator_outcome_reason":"Job failed after intervention"},'
                    '"action_url":"/autonomous-agents?job=job-1","is_read":false,'
                    '"created_at":"2026-03-12T12:00:00Z"}}'
                ),
            }
        ]
    )
    redis_client = _FakeRedisClient(pubsub)

    monkeypatch.setattr(
        notification_endpoints, "require_websocket_auth", _fake_require_websocket_auth
    )
    monkeypatch.setattr(
        aioredis, "from_url", lambda _url, decode_responses=True: redis_client
    )

    with client.websocket_connect("/api/v1/notifications/ws") as websocket:
        connected = websocket.receive_json()
        forwarded = websocket.receive_json()

    assert connected["type"] == "connected"
    assert connected["channel"] == f"notifications:{test_user.id}"

    assert forwarded["type"] == "notification"
    assert forwarded["notification"]["notification_type"] == "experiment_run_update"
    assert forwarded["notification"]["data"]["agent_job_id"] == "job-1"
    assert forwarded["notification"]["data"]["note_id"] == "note-1"
    assert (
        forwarded["notification"]["data"]["launch_mode"] == "quick_start_claude_backend"
    )
    assert forwarded["notification"]["data"]["recovery_open"] is True
    assert (
        forwarded["notification"]["data"]["recovery_reason"]
        == "fallback verification still failing"
    )
    assert (
        forwarded["notification"]["data"]["recommended_action"]
        == "Inspect failing fallback output"
    )
    assert (
        forwarded["notification"]["data"]["first_failed_command"]
        == "npm --prefix frontend test"
    )
    assert forwarded["notification"]["data"]["latest_operator_action"] == "restart"
    assert (
        forwarded["notification"]["data"]["latest_operator_note"]
        == "Retry after fallback failure"
    )
    assert (
        forwarded["notification"]["data"]["latest_operator_status_before"] == "failed"
    )
    assert (
        forwarded["notification"]["data"]["latest_operator_status_after"] == "pending"
    )
    assert (
        forwarded["notification"]["data"]["latest_operator_at"]
        == "2026-03-10T01:00:00Z"
    )
    assert forwarded["notification"]["data"]["latest_operator_outcome"] == "unresolved"
    assert (
        forwarded["notification"]["data"]["latest_operator_outcome_reason"]
        == "Job failed after intervention"
    )

    assert pubsub.cleanup_completed.wait(timeout=1)
    assert pubsub.subscribed == [f"notifications:{test_user.id}"]
    assert pubsub.unsubscribed == [f"notifications:{test_user.id}"]
    assert pubsub.closed is True
    assert redis_client.closed is True
