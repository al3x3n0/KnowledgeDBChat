"""Reading what waits in the broker, and the sweep that relies on it.

The ``live`` test sends a real message through Celery to a real Redis
(``TEST_REDIS_URL``), because the shape read here is kombu's and a fake
written from the same understanding would agree with it whatever kombu does.
"""

import base64
import json
import os
import uuid

import pytest

from app.services import broker_queue

pytestmark = pytest.mark.unit

TASK = "app.tasks.agent_job_tasks.execute_agent_job_task"


def _message(task, *args):
    body = base64.b64encode(json.dumps([list(args), {}, {}]).encode()).decode()
    return json.dumps(
        {
            "body": body,
            "headers": {"task": task, "id": str(uuid.uuid4())},
            "properties": {"body_encoding": "base64"},
        }
    )


class FakeRedis:
    def __init__(self, lists=None, unacked=None):
        self.lists = lists or {}
        self.unacked = unacked or {}

    def scan_iter(self, pattern):
        prefix = pattern.rstrip("*")
        return [k for k in self.lists if k.startswith(prefix)]

    def lrange(self, key, start, end):
        return list(self.lists.get(key, []))

    def hvals(self, key):
        return list(self.unacked.values()) if key == "unacked" else []


def test_queued_reserved_and_prioritised_messages_are_all_seen():
    client = FakeRedis(
        lists={
            "agents": [_message(TASK, "job-a", "u"), _message("other.task", "job-x")],
            "agents\x06\x163": [_message(TASK, "job-b", "u")],
            "celery": [_message(TASK, "job-elsewhere", "u")],
        },
        unacked={
            "tag1": json.dumps([json.loads(_message(TASK, "job-c", "u")), "", "agents"])
        },
    )

    assert broker_queue.queued_first_args(client, "agents", TASK) == {
        "job-a",
        "job-b",
        "job-c",
    }


def test_a_message_that_cannot_be_read_is_skipped_not_fatal():
    client = FakeRedis(lists={"agents": ["not json", json.dumps({"headers": {}})]})

    assert broker_queue.queued_first_args(client, "agents", TASK) == set()


def test_a_broker_that_is_not_redis_is_unknown(monkeypatch):
    monkeypatch.setattr(broker_queue.settings, "CELERY_BROKER_URL", "memory://")

    assert broker_queue.waiting_agent_job_ids() is None


def test_an_unreachable_redis_is_unknown_not_empty(monkeypatch):
    monkeypatch.setattr(
        broker_queue.settings, "CELERY_BROKER_URL", "redis://127.0.0.1:1/0"
    )

    assert broker_queue.waiting_agent_job_ids() is None


REDIS_URL = os.environ.get("TEST_REDIS_URL", "")


@pytest.mark.skipif(not REDIS_URL, reason="TEST_REDIS_URL is not set")
def test_a_message_celery_really_sent_is_found(monkeypatch):
    import redis
    from celery import Celery

    # Celery prefers this variable to its broker argument, and conftest sets it
    # to memory://, which would send the probe nowhere Redis can see.
    monkeypatch.delenv("CELERY_BROKER_URL", raising=False)
    queue = f"probe-{uuid.uuid4().hex[:8]}"
    app = Celery("probe", broker=REDIS_URL)
    client = redis.Redis.from_url(REDIS_URL)
    try:
        app.send_task(TASK, args=["job-live", "u"], queue=queue)
        assert broker_queue.queued_first_args(client, queue, TASK) == {"job-live"}
    finally:
        client.delete(queue)
        client.close()
