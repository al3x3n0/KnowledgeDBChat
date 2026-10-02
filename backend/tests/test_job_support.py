"""What every job-shaped task does around its work: report progress, and
record a failure without rewriting a job that already ended.

Three task modules each carried a copy of both. Each publisher closed its
Redis connection on the line after `publish`, not in a `finally`.
"""

import json
from uuid import uuid4

import pytest

from app.tasks import job_support

pytestmark = pytest.mark.unit


class FakeRedis:
    def __init__(self, log, fail=False):
        self.log = log
        self.fail = fail

    async def publish(self, channel, message):
        if self.fail:
            raise RuntimeError("redis down")
        self.log.append(("publish", channel, json.loads(message)))

    async def close(self):
        self.log.append(("close",))


@pytest.fixture
def redis_log(monkeypatch):
    import redis.asyncio as redis

    log, state = [], {"fail": False}
    monkeypatch.setattr(redis, "from_url", lambda _url: FakeRedis(log, state["fail"]))
    return log, state


async def test_a_progress_message_has_the_shape_the_streams_forward(redis_log):
    log, _ = redis_log
    await job_support.publish_progress("export:1:progress", 40, "rendering", "running")
    assert log[0] == (
        "publish",
        "export:1:progress",
        {"type": "progress", "progress": 40, "stage": "rendering", "status": "running"},
    )
    assert log[-1] == ("close",)


async def test_an_error_is_carried_only_when_there_is_one(redis_log):
    log, _ = redis_log
    await job_support.publish_progress("c", 0, "s", "failed", error="boom")
    assert log[0][2]["error"] == "boom"


async def test_a_failed_publish_still_closes_and_does_not_raise(redis_log):
    """A Redis that is down must not fail the job it is reporting on, and must
    not leave a connection behind on every step."""
    log, state = redis_log
    state["fail"] = True
    await job_support.publish_progress("c", 1, "s", "running")
    assert log == [("close",)]


class _Session:
    def __init__(self, job):
        self.job = job
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def execute(self, _statement):
        job = self.job

        class Result:
            def scalar_one_or_none(self):
                return job

        return Result()

    async def commit(self):
        self.committed = True


class _Job:
    id = None

    def __init__(self, status):
        self.status = status
        self.error = None
        self.completed_at = None


class _Model:
    id = "id-column"


@pytest.fixture
def session_for(monkeypatch):
    holder = {}

    def use(job):
        holder["session"] = _Session(job)
        monkeypatch.setattr(
            job_support, "create_celery_session", lambda: (lambda: holder["session"])
        )
        monkeypatch.setattr(job_support, "select", lambda model: _Select())
        return holder["session"]

    class _Select:
        def where(self, *_a):
            return self

    return use


async def test_a_running_job_is_marked_failed(session_for):
    job = _Job("running")
    session = session_for(job)
    assert await job_support.mark_job_failed(_Model, uuid4(), "worker died") is True
    assert job.status == "failed" and job.error == "Task error: worker died"
    assert job.completed_at is not None and session.committed


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
async def test_a_job_that_already_ended_is_left_alone(session_for, status):
    """A late failure -- a retry that dies, a worker killed after the job
    finished -- must not turn a completed job into a failed one."""
    job = _Job(status)
    session = session_for(job)
    assert await job_support.mark_job_failed(_Model, uuid4(), "late") is False
    assert job.status == status and job.error is None and not session.committed


async def test_a_job_that_no_longer_exists_is_not_an_error(session_for):
    session_for(None)
    assert await job_support.mark_job_failed(_Model, uuid4(), "x") is False


def test_the_three_tasks_use_the_shared_helpers():
    import inspect

    from app.tasks import export_tasks, presentation_tasks, repo_report_tasks

    for module in (export_tasks, presentation_tasks, repo_report_tasks):
        source = inspect.getsource(module)
        assert "job_support.publish_progress(" in source, module.__name__
        assert "job_support.mark_job_failed(" in source, module.__name__
        assert "redis.from_url" not in source, module.__name__


class FakeSyncRedis:
    def __init__(self, log, fail=False):
        self.log = log
        self.fail = fail

    def publish(self, channel, message):
        if self.fail:
            raise RuntimeError("redis down")
        self.log.append(("publish", channel, json.loads(message)))

    def close(self):
        self.log.append(("close",))


@pytest.mark.parametrize("fail", [False, True])
def test_a_sync_publish_closes_its_client_either_way(monkeypatch, fail):
    """The copies this replaced opened a client per message and closed none."""
    import redis

    log = []
    monkeypatch.setattr(redis, "from_url", lambda _url, **_k: FakeSyncRedis(log, fail))
    job_support.publish_sync("c:1", {"type": "status", "status": {"ok": True}})
    if not fail:
        assert log[0] == ("publish", "c:1", {"type": "status", "status": {"ok": True}})
    assert log[-1] == ("close",)


def test_a_redis_that_cannot_be_reached_does_not_raise(monkeypatch):
    import redis

    def refuse(_url, **_k):
        raise ConnectionError("nothing listening")

    monkeypatch.setattr(redis, "from_url", refuse)
    job_support.publish_sync("c:1", {"type": "progress"})


def test_task_modules_publish_through_the_shared_helpers():
    """A `_publish_*` wrapper may name its channel and payload; it may not
    open its own client."""
    import ast
    from pathlib import Path

    tasks = Path(job_support.__file__).parent
    own = []
    for path in sorted(tasks.glob("*_tasks.py")):
        source = path.read_text(encoding="utf-8")
        for node in ast.parse(source).body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name.startswith("_publish")
                and "job_support.publish_" not in ast.get_source_segment(source, node)
            ):
                own.append(f"{path.name}:{node.name}")
    assert not own, own


def test_messages_keep_the_shape_their_streams_forward(monkeypatch):
    from app.tasks import ingestion_tasks, synthesis_tasks, template_tasks

    sent = []
    monkeypatch.setattr(
        job_support,
        "publish_sync",
        lambda channel, message: sent.append((channel, message)),
    )
    ingestion_tasks._publish_ing_progress("s1", {"pct": 5})
    template_tasks._publish_progress("j1", {"stage": "analyzing"})
    synthesis_tasks._publish_complete("j2", {"metadata": {"word_count": 7}})
    assert sent == [
        (
            "ingestion_progress:s1",
            {"type": "progress", "document_id": "s1", "progress": {"pct": 5}},
        ),
        (
            "template_progress:j1",
            {"type": "progress", "job_id": "j1", "data": {"stage": "analyzing"}},
        ),
        (
            "synthesis_progress:j2",
            {
                "type": "complete",
                "job_id": "j2",
                "result": {"word_count": 7, "documents_analyzed": 0},
            },
        ),
    ]


async def test_a_task_with_more_to_say_publishes_its_own_message(monkeypatch):
    from app.tasks import agent_job_tasks, training_tasks

    sent = []

    async def capture(channel, message):
        sent.append((channel, dict(message)))

    monkeypatch.setattr(job_support, "publish_message", capture)
    await agent_job_tasks._publish_job_progress(
        "j1", 30, "act", "running", iteration=2, error="boom"
    )
    await training_tasks._publish_training_progress(
        "t1", 10, "training", current_step=0
    )
    (agent_channel, agent), (training_channel, training) = sent
    assert agent_channel == "agent_job:j1:progress"
    assert {k: agent[k] for k in ("type", "job_id", "phase", "iteration", "error")} == {
        "type": "progress",
        "job_id": "j1",
        "phase": "act",
        "iteration": 2,
        "error": "boom",
    }
    assert "phase_details" not in agent and "timestamp" in agent
    assert training_channel == "training_job:t1:progress"
    assert training["current_step"] == 0 and "total_steps" not in training


def test_a_drafting_task_reports_its_owner_with_every_attempt():
    """The polling endpoint refuses a draft that is not the caller's, which it
    can only do if every progress state says whose it is."""
    states = []

    class Task:
        def update_state(self, **kwargs):
            states.append(kwargs)

    notes = ["first"]
    report = job_support.attempt_reporter(Task(), 42)
    report("drafting", 1, notes)
    notes.append("second")
    report("repairing", 2, notes)

    assert [s["state"] for s in states] == ["PROGRESS", "PROGRESS"]
    assert all(s["meta"]["user_id"] == "42" for s in states)
    assert states[0]["meta"] == {
        "user_id": "42",
        "stage": "drafting",
        "attempt": 1,
        "notes": ["first"],
    }


def test_run_async_returns_the_coroutines_result():
    async def answer():
        return 7

    assert job_support.run_async(answer()) == 7


class _FakeRedis:
    def __init__(self, values=None, broken=False):
        self.values, self.broken, self.closed, self.deleted = (
            values or {},
            broken,
            0,
            [],
        )

    def get(self, key):
        if self.broken:
            raise RuntimeError("connection refused")
        return self.values.get(key)

    def delete(self, *keys):
        if self.broken:
            raise RuntimeError("connection refused")
        self.deleted.extend(keys)

    def close(self):
        self.closed += 1


def test_a_polled_flag_closes_its_client_every_time(monkeypatch):
    """Cancellation is polled once per document or training step; each poll
    used to leave a connection pool behind."""
    import redis

    clients = []

    def from_url(*_a, **_k):
        clients.append(_FakeRedis({"job:cancel": "1"}))
        return clients[-1]

    monkeypatch.setattr(redis, "from_url", from_url)

    assert job_support.flag_is_set("job:cancel") is True
    assert job_support.flag_is_set("job:other") is False
    job_support.delete_keys("job:cancel", "job:task")

    assert [c.closed for c in clients] == [1, 1, 1]
    assert clients[-1].deleted == ["job:cancel", "job:task"]


def test_an_unreachable_redis_means_not_cancelled(monkeypatch):
    import redis

    client = _FakeRedis(broken=True)
    monkeypatch.setattr(redis, "from_url", lambda *_a, **_k: client)

    assert job_support.flag_is_set("job:cancel") is False
    job_support.delete_keys("job:cancel")
    assert client.closed == 2
