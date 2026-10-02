"""One loop forwards a job's progress to a socket, and it always cleans up.

Five handlers each had their own copy. Two leaked a Redis connection whenever
the client left or the job had already finished; three could not notice a
client leaving at all; one polled with the blocking Redis client inside the
event loop. These pin the behaviour of the one that replaced them.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest
from fastapi import WebSocketDisconnect

from app.utils import websocket_progress as progress

pytestmark = pytest.mark.unit


class FakePubSub:
    def __init__(self, messages, log):
        self.messages = list(messages)
        self.log = log

    async def subscribe(self, channel):
        self.log.append(f"subscribe:{channel}")

    async def get_message(self, ignore_subscribe_messages=True, timeout=None):
        if self.messages:
            item = self.messages.pop(0)
            if isinstance(item, Exception):
                raise item
            return {"type": "message", "data": item}
        return None

    async def unsubscribe(self, channel):
        self.log.append("unsubscribe")

    async def close(self):
        self.log.append("pubsub.close")


class FakeRedis:
    def __init__(self, messages, log):
        self._pubsub = FakePubSub(messages, log)
        self.log = log

    def pubsub(self):
        return self._pubsub

    async def close(self):
        self.log.append("redis.close")


class FakeSocket:
    """A client that stays until `leave_after` receives, then disconnects."""

    def __init__(self, log, leave_after=None, says=None):
        self.log = log
        self.sent = []
        self.leave_after = leave_after
        self.says = list(says or [])
        self.receives = 0

    async def send_json(self, payload):
        self.log.append("send")
        self.sent.append(payload)

    async def send_text(self, text):
        self.log.append("send")
        self.sent.append(text)

    async def receive_text(self):
        self.receives += 1
        if self.says:
            return self.says.pop(0)
        if self.leave_after is not None and self.receives > self.leave_after:
            raise WebSocketDisconnect(code=1000)
        await asyncio.sleep(3600)

    async def close(self, code=1000, reason=""):
        self.log.append("socket.close")


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(progress, "CLIENT_CHECK_SECONDS", 0.001)


async def run(messages, socket_kwargs=None, **kwargs):
    log = []
    socket = FakeSocket(log, **(socket_kwargs or {}))
    await progress.forward_progress(
        socket,
        "job:1:progress",
        redis_url="redis://unused",
        redis_factory=lambda _url: FakeRedis(messages, log),
        **kwargs,
    )
    return socket, log


RELEASED = ["unsubscribe", "pubsub.close", "redis.close", "socket.close"]


async def test_it_forwards_until_the_job_is_over_and_then_lets_go():
    socket, log = await run(
        [
            json.dumps({"status": "running", "progress": 40}),
            json.dumps({"status": "completed", "progress": 100}),
            json.dumps({"status": "never sent"}),
        ],
        initial={"type": "progress", "status": "pending"},
    )
    assert socket.sent[0] == {"type": "progress", "status": "pending"}
    assert [json.loads(m)["status"] for m in socket.sent[1:]] == [
        "running",
        "completed",
    ]
    assert log[-4:] == RELEASED


async def test_it_subscribes_before_saying_anything():
    """An update published between reading the job and subscribing is lost
    otherwise, and the client waits on a job that already finished."""
    _, log = await run([json.dumps({"status": "completed"})], initial={"status": "x"})
    assert log.index("subscribe:job:1:progress") < log.index("send")


async def test_a_job_that_already_finished_still_releases_everything():
    """The early return used to skip the cleanup: one leaked connection for
    every socket opened on a finished job."""
    socket, log = await run(
        [json.dumps({"status": "running"})],
        initial={"status": "completed"},
        already_finished=True,
    )
    assert socket.sent == [{"status": "completed"}]
    assert log[-4:] == RELEASED


async def test_a_client_that_leaves_ends_the_stream_and_releases_everything():
    """Waiting on Redis alone cannot see this: a socket watching a job that
    never finishes held its connection until the server restarted."""
    socket, log = await run([], socket_kwargs={"leave_after": 1})
    assert socket.sent == []
    assert log[-4:] == RELEASED


async def test_a_redis_failure_is_reported_and_releases_everything():
    socket, log = await run([RuntimeError("connection reset")])
    assert socket.sent[-1]["type"] == "error"
    assert "connection reset" in socket.sent[-1]["message"]
    assert log[-4:] == RELEASED


async def test_a_ping_is_answered():
    socket, _ = await run(
        [json.dumps({"status": "running"}), json.dumps({"status": "completed"})],
        socket_kwargs={"says": ["ping"]},
    )
    assert "pong" in socket.sent


async def test_the_caller_decides_what_ends_the_stream():
    socket, _ = await run(
        [
            json.dumps({"type": "log", "status": "completed"}),
            json.dumps({"type": "complete"}),
            json.dumps({"type": "after"}),
        ],
        is_terminal=lambda m: m.get("type") == "complete",
    )
    assert [json.loads(m)["type"] for m in socket.sent] == ["log", "complete"]


async def test_a_message_that_is_not_json_is_forwarded_and_does_not_end_it():
    socket, _ = await run(["not json", json.dumps({"status": "failed"})])
    assert socket.sent[0] == "not json" and len(socket.sent) == 2


def test_no_progress_stream_subscribes_by_hand():
    """The loop is easy to write and easy to get slightly wrong, which is how
    there came to be five."""
    endpoints = Path(__file__).resolve().parents[1] / "app" / "api" / "endpoints"
    # A per-user notification feed, not a job's progress: it has no terminal
    # message and lives as long as the session.
    allowed = {"notifications.py"}
    offenders = sorted(
        path.name
        for path in endpoints.glob("*.py")
        if path.name not in allowed
        and re.search(r"\.pubsub\(\)", path.read_text(encoding="utf-8"))
    )
    assert not offenders, f"use websocket_progress.forward_progress: {offenders}"


def test_the_blocking_redis_client_is_not_used_in_a_handler():
    """`import redis` at the top of an endpoint module is the blocking client.
    Polling it from a handler stalled every other request for up to a second
    at a time while someone watched a template job."""
    endpoints = Path(__file__).resolve().parents[1] / "app" / "api" / "endpoints"
    offenders = sorted(
        path.name
        for path in endpoints.glob("*.py")
        if re.search(r"^import redis$", path.read_text(encoding="utf-8"), re.M)
    )
    assert not offenders, f"blocking redis client imported in: {offenders}"
