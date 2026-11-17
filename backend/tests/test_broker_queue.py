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
        values = self.lists.get(key, [])
        return list(values[start:] if end == -1 else values[start : end + 1])

    def scan(self, cursor=0, match="*", count=256):
        return 0, self.scan_iter(match)

    def hscan(self, key, cursor=0, count=256):
        return 0, self.unacked if key == "unacked" else {}

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


def test_bounded_presence_finds_queued_prioritised_and_reserved_jobs():
    client = FakeRedis(
        lists={
            "agents": [_message(TASK, "queued")],
            "agents\x06\x163": [_message(TASK, "priority")],
        },
        unacked={
            "tag": json.dumps([json.loads(_message(TASK, "reserved")), "", "agents"])
        },
    )
    assert broker_queue.message_presence(
        client, "agents", TASK, {"queued", "priority", "reserved", "missing"}
    ) == {
        "queued": True,
        "priority": True,
        "reserved": True,
        "missing": False,
    }


def test_scan_budget_keeps_positive_sightings_and_leaves_other_ids_unknown():
    client = FakeRedis(
        lists={"agents": [_message(TASK, "found"), _message(TASK, "later")]}
    )
    assert broker_queue.message_presence(
        client, "agents", TASK, {"found", "later"}, max_messages=1
    ) == {
        "found": True,
        "later": None,
    }


def test_queue_inspection_reads_bounded_pages():
    client = FakeRedis(lists={"agents": [_message(TASK, str(i)) for i in range(600)]})
    calls = []
    original = client.lrange

    def lrange(key, start, end):
        calls.append((start, end))
        return original(key, start, end)

    client.lrange = lrange
    assert broker_queue.message_presence(client, "agents", TASK, {"599"}) == {
        "599": True
    }
    assert calls == [(0, 255), (256, 511), (512, 767)]


def test_broker_failure_does_not_erase_an_already_seen_message():
    client = FakeRedis(lists={"agents": [_message(TASK, "seen")]})

    def failed(**kwargs):
        raise ConnectionError("broker went away")

    client.scan = failed
    assert broker_queue.message_presence(
        client, "agents", TASK, {"seen", "unknown"}
    ) == {
        "seen": True,
        "unknown": None,
    }


def test_presence_for_a_non_redis_broker_is_unknown(monkeypatch):
    monkeypatch.setattr(broker_queue.settings, "CELERY_BROKER_URL", "memory://")
    assert broker_queue.agent_job_message_presence({"job"}) == {"job": None}


def test_reserved_messages_require_kombus_reservation_envelope():
    client = FakeRedis(unacked={"bad": _message(TASK, "job")})
    assert broker_queue.message_presence(client, "agents", TASK, {"job"}) == {
        "job": False
    }


def test_key_scanning_stops_when_its_time_budget_expires(monkeypatch):
    moments = iter([0, 0.1, 0.2, 0.3, 2.1])
    monkeypatch.setattr(broker_queue.time, "monotonic", lambda: next(moments))
    client = FakeRedis()
    client.scan = lambda **kwargs: (1, [])
    assert broker_queue.message_presence(client, "agents", TASK, {"job"}) == {
        "job": None
    }


def test_presence_wrapper_closes_its_redis_connection(monkeypatch):
    import redis

    client = FakeRedis(lists={"agents": [_message(TASK, "job")]})
    closed = []
    client.close = lambda: closed.append(True)
    monkeypatch.setattr(broker_queue.settings, "CELERY_BROKER_URL", "redis://broker/0")
    monkeypatch.setattr(broker_queue.settings, "AGENT_JOB_CELERY_QUEUE", "agents")
    monkeypatch.setattr(redis.Redis, "from_url", lambda *args, **kwargs: client)
    assert broker_queue.agent_job_message_presence({"job"}) == {"job": True}
    assert closed == [True]


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
