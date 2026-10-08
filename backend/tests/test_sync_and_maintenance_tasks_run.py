"""The beat-scheduled sync and maintenance Celery tasks, run for real.

None of these task bodies was named in a test, so nothing had checked them,
and a scheduled task that breaks does so silently:

* ``sync_tasks``: ``sync_all_gitlab_sources``, ``sync_all_confluence_sources``,
  ``sync_all_web_sources``, ``sync_all_sources``, ``scan_scheduled_sources``
  -- and the ``ingest_from_source`` body every one of them queues, because
  that is where a sync records what happened (sync log, ``last_sync``,
  ``is_syncing``, ``last_error``);
* ``maintenance_tasks.cleanup_old_data``;
* ``latex_maintenance_tasks.fail_stale_latex_compile_jobs``;
* ``compops_sync_tasks.sync_due_compops_evidence``.

The async bodies run against the in-memory database; each synchronous Celery
function also runs once via ``.run`` against a file database, in its own event
loop, as a worker would. Judgement is on the rows they leave behind.

Only edges that leave the process are replaced, and each replacement binds its
arguments against the real callee, so a call the real thing would refuse fails
here too:

* Redis -- ``job_support`` publishers and flags, and ``cache_service``;
* the vector store -- ``vector_store_service`` (never reached: indexing is
  queued, not run, by a sync);
* Celery's ``.delay`` -- bound against the real task function;
* HTTP (web pages, GitLab, CompOps) -- ``httpx.MockTransport`` behind the
  real connectors and the real external-agent gateway, plus the gateway's DNS
  resolver.

A test marked xfail(strict) is a defect in the application, not in the test.
"""

import asyncio
import inspect
import json
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from celery.app.task import Task as CeleryTask
from sqlalchemy import DateTime, event, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm.attributes import set_committed_value
from sqlalchemy.pool import NullPool

from app.core.cache import cache_service
from app.core.config import settings
from app.core.database import Base
from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.chat import ChatMessage, ChatSession
from app.models.compops_evidence_subscription import CompOpsEvidenceSubscription
from app.models.document import Document, DocumentSource, DocumentSourceSyncLog
from app.models.latex_compile_job import LatexCompileJob
from app.models.tool_audit import ToolExecutionAudit
from app.models.workflow import UserTool
from app.services.auth_service import AuthService
from app.services.external_agent_gateway_service import (
    ExternalAgentGatewayService,
    external_agent_gateway_service,
)
from app.services.vector_store import vector_store_service
from app.tasks import (
    compops_sync_tasks,
    ingestion_tasks,
    job_support,
    latex_maintenance_tasks,
    maintenance_tasks,
    sync_tasks,
)

pytestmark = pytest.mark.unit

TASK_MODULES = (
    sync_tasks,
    maintenance_tasks,
    latex_maintenance_tasks,
    compops_sync_tasks,
    ingestion_tasks,
    job_support,
)

COMPOPS_URL = "https://compops.example.test"
DOCS_URL = "https://docs.example.test"
GITLAB_URL = "https://gitlab.example.test"


# --------------------------------------------------------------------------
# Edges that leave the process
# --------------------------------------------------------------------------


def _binding(real):
    """A checker that raises TypeError exactly when ``real`` would."""
    signature = inspect.signature(real)

    def bind(*args, **kwargs):
        return signature.bind(*args, **kwargs).arguments

    return bind


class FakeTask:
    """The ``self`` a bound Celery task receives; records its state updates."""

    _bind = staticmethod(_binding(CeleryTask.update_state))

    def __init__(self):
        self.states = []

    def update_state(self, *args, **kwargs):
        a = self._bind(self, *args, **kwargs)
        json.dumps(a.get("meta"))
        self.states.append((a.get("state"), a.get("meta")))


class Edges:
    """Redis (publishers, flags, the cache) and the vector store."""

    def __init__(self):
        self.published = []
        self.cache = {}
        self.deleted_keys = []
        self.vector_calls = []

    def install(self, monkeypatch):
        bind_sync = _binding(job_support.publish_sync)
        bind_message = _binding(job_support.publish_message)
        bind_flag = _binding(job_support.flag_is_set)
        bind_keys = _binding(job_support.delete_keys)
        bind_cache_set = _binding(cache_service.set)
        bind_cache_get = _binding(cache_service.get)
        bind_cache_delete = _binding(cache_service.delete)
        edges = self

        def publish_sync(*args, **kwargs):
            a = bind_sync(*args, **kwargs)
            json.dumps(dict(a["message"]))
            edges.published.append((a["channel"], dict(a["message"])))

        async def publish_message(*args, **kwargs):
            a = bind_message(*args, **kwargs)
            json.dumps(dict(a["message"]))
            edges.published.append((a["channel"], dict(a["message"])))

        def flag_is_set(*args, **kwargs):
            a = bind_flag(*args, **kwargs)
            return bool(edges.cache.get(a["key"]))

        def delete_keys(*args, **kwargs):
            a = bind_keys(*args, **kwargs)
            for key in a["keys"]:
                edges.deleted_keys.append(key)
                edges.cache.pop(key, None)

        async def cache_set(*args, **kwargs):
            a = bind_cache_set(*args, **kwargs)
            json.dumps(a["value"])
            edges.cache[a["key"]] = a["value"]
            return True

        async def cache_get(*args, **kwargs):
            a = bind_cache_get(*args, **kwargs)
            return edges.cache.get(a["key"])

        async def cache_delete(*args, **kwargs):
            a = bind_cache_delete(*args, **kwargs)
            edges.cache.pop(a["key"], None)
            return True

        monkeypatch.setattr(job_support, "publish_sync", publish_sync)
        monkeypatch.setattr(job_support, "publish_message", publish_message)
        monkeypatch.setattr(job_support, "flag_is_set", flag_is_set)
        monkeypatch.setattr(job_support, "delete_keys", delete_keys)

        # The scheduled scan records its task id and full-resync flag through
        # the ingestion-state helpers, as plain strings -- which is what the
        # cancel endpoint and the ingestion task read back. Stored here as
        # Redis would hold them, so a writer and a reader that disagree about
        # the format are not hidden by a fake that keeps Python objects.
        async def set_task_mapping(source_id, task_id, ttl=3600):
            edges.cache[f"ingestion:task:{source_id}"] = str(task_id)

        async def set_force_full(source_id, ttl=600):
            edges.cache[f"ingestion:force_full:{source_id}"] = "1"

        monkeypatch.setattr(sync_tasks, "set_ingestion_task_mapping", set_task_mapping)
        monkeypatch.setattr(sync_tasks, "set_force_full_flag", set_force_full)
        monkeypatch.setattr(cache_service, "set", cache_set)
        monkeypatch.setattr(cache_service, "get", cache_get)
        monkeypatch.setattr(cache_service, "delete", cache_delete)

        # A sync queues indexing; it must never reach the vector store itself.
        for name in ("initialize", "add_document_chunks", "delete_document_chunks"):
            real = getattr(vector_store_service, name)

            def refuse(*args, _name=name, _real=real, **kwargs):
                _binding(_real)(*args, **kwargs)
                edges.vector_calls.append(_name)
                raise AssertionError(f"a sync reached the vector store ({_name})")

            monkeypatch.setattr(vector_store_service, name, refuse)
        return self

    def on(self, channel):
        return [m for c, m in self.published if c == channel]


class Web:
    """Every HTTP request any client makes goes to ``routes``."""

    def __init__(self):
        self.routes = {}  # (host, path) -> (status, body, content_type)
        self.requests = []

    def route(self, url, body, status=200, content_type="application/json"):
        parsed = httpx.URL(url)
        if not isinstance(body, str):
            body = json.dumps(body)
        self.routes[(parsed.host, parsed.path)] = (status, body, content_type)

    def handle(self, request):
        self.requests.append(request)
        key = (request.url.host, request.url.path)
        if key not in self.routes:
            return httpx.Response(404, text=f"no route for {request.url}")
        status, body, content_type = self.routes[key]
        return httpx.Response(status, text=body, headers={"content-type": content_type})

    def hits(self, host):
        return [r for r in self.requests if r.url.host == host]


class Queue:
    """A Celery task's ``.delay``, bound against the real task function.

    ``fail_for`` holds first arguments whose enqueue raises, as a broker that
    refuses a message does.
    """

    def __init__(self, task):
        self.signature = inspect.signature(task.run)
        self.calls = []
        self.fail_for = set()

    def delay(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arguments = dict(bound.arguments)
        first = next(iter(arguments.values()), None)
        if first in self.fail_for:
            raise ConnectionError("broker refused the message")
        self.calls.append(arguments)
        return SimpleNamespace(id=f"task-{len(self.calls)}")

    def firsts(self):
        return [next(iter(c.values())) for c in self.calls]


@pytest.fixture
def edges(monkeypatch):
    return Edges().install(monkeypatch)


@pytest.fixture
def web(monkeypatch):
    rec = Web()
    real_client = httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(rec.handle)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)

    # The gateway refuses private addresses after resolving the host; answer
    # the lookup as DNS would for a public host.
    bind_resolve = _binding(ExternalAgentGatewayService._resolve_addresses)

    async def resolve(*args, **kwargs):
        bind_resolve(*args, **kwargs)
        return ["93.184.216.34"]

    monkeypatch.setattr(external_agent_gateway_service, "_resolver", resolve)
    return rec


@pytest.fixture
def ingest_queue(monkeypatch):
    queue = Queue(ingestion_tasks.ingest_from_source)
    monkeypatch.setattr(ingestion_tasks.ingest_from_source, "delay", queue.delay)
    return queue


@pytest.fixture
def process_queue(monkeypatch):
    queue = Queue(ingestion_tasks.process_uploaded_document)
    monkeypatch.setattr(ingestion_tasks.process_uploaded_document, "delay", queue.delay)
    return queue


@pytest.fixture
def task_sessions(db_session, monkeypatch):
    # What create_celery_session builds: expire_on_commit=False.
    factory = async_sessionmaker(
        db_session.bind, class_=AsyncSession, expire_on_commit=False
    )
    for module in TASK_MODULES:
        monkeypatch.setattr(module, "create_celery_session", lambda: factory)
    return factory


@pytest.fixture
def postgres_timestamptz():
    """Load ``DateTime(timezone=True)`` columns the way production does.

    Every such column is ``TIMESTAMP WITH TIME ZONE`` in the migrations
    (``0001_initial_migration``: ``document_sources.last_sync``), and asyncpg
    returns those as aware UTC datetimes. SQLite returns them naive, which
    hides any arithmetic between a loaded value and ``datetime.utcnow()``.
    """
    columns = [
        c.key
        for c in DocumentSource.__table__.columns
        if isinstance(c.type, DateTime) and c.type.timezone
    ]

    def to_aware(target, *_):
        for key in columns:
            value = target.__dict__.get(key)
            if isinstance(value, datetime) and value.tzinfo is None:
                set_committed_value(target, key, value.replace(tzinfo=timezone.utc))

    event.listen(DocumentSource, "load", to_aware)
    event.listen(DocumentSource, "refresh", to_aware)
    yield
    event.remove(DocumentSource, "load", to_aware)
    event.remove(DocumentSource, "refresh", to_aware)


async def _reload(db, model_cls, row_id):
    return (
        await db.execute(
            select(model_cls)
            .where(model_cls.id == row_id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()


async def _all(db, model_cls, *where):
    query = select(model_cls).execution_options(populate_existing=True)
    for clause in where:
        query = query.where(clause)
    return list((await db.execute(query)).scalars())


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------


async def _source(db, source_type="gitlab", config=None, **fields):
    source = DocumentSource(
        name=fields.pop("name", None) or f"{source_type}-{uuid4().hex[:8]}",
        source_type=source_type,
        config=config if config is not None else {},
        **fields,
    )
    db.add(source)
    await db.commit()
    await db.refresh(source)
    return source


def _ago(**delta):
    return datetime.utcnow() - timedelta(**delta)


# ==========================================================================
# sync_all_<type>_sources / sync_all_sources
# ==========================================================================


async def test_type_sweep_queues_every_active_source_of_that_type_only(
    db_session, task_sessions, edges, ingest_queue
):
    first = await _source(db_session, "gitlab")
    second = await _source(db_session, "gitlab")
    await _source(db_session, "gitlab", is_active=False)
    await _source(db_session, "confluence")
    await _source(db_session, "web")

    result = await sync_tasks._async_sync_sources_by_type("gitlab")

    assert sorted(ingest_queue.firsts()) == sorted([str(first.id), str(second.id)])
    assert result["success"] is True
    assert (result["total_sources"], result["synced_sources"]) == (2, 2)
    assert result["failed_sources"] == 0
    assert {r["source_id"] for r in result["results"]} == {
        str(first.id),
        str(second.id),
    }
    assert all(r["status"] == "triggered" and r["task_id"] for r in result["results"])
    # Queuing records nothing on the source: that is ingestion's job.
    assert (await _reload(db_session, DocumentSource, first.id)).last_sync is None


@pytest.mark.parametrize(
    "config, syncing, swept",
    [
        ({}, False, True),  # never configured (created from Documents): swept
        ({"auto_sync": True}, False, True),  # on, but no schedule of its own
        ({"auto_sync": False}, False, False),  # switched off in Admin
        ({"auto_sync": True, "sync_interval_minutes": 30}, False, False),
        ({"auto_sync": True, "cron": "0 * * * *"}, False, False),
        ({}, True, False),  # already syncing: never started twice
    ],
    ids=[
        "unset",
        "on-unscheduled",
        "off",
        "on-interval",
        "on-cron",
        "already-syncing",
    ],
)
async def test_type_sweep_respects_the_sources_own_settings(
    db_session, task_sessions, edges, ingest_queue, config, syncing, swept
):
    # Decided 2026-10-06: per-source settings win over the hourly sweep.
    source = await _source(db_session, "gitlab", config=config, is_syncing=syncing)

    result = await sync_tasks._async_sync_sources_by_type("gitlab")

    assert (str(source.id) in ingest_queue.firsts()) is swept
    if not swept:
        (row,) = result["results"]
        assert row["status"] == "skipped" and row["reason"]


async def test_sync_all_sources_respects_the_sources_own_settings(
    db_session, task_sessions, edges, ingest_queue
):
    kept = await _source(db_session, "gitlab")
    off = await _source(db_session, "web", config={"auto_sync": False})
    busy = await _source(db_session, "confluence", is_syncing=True)

    await sync_tasks._async_sync_all_sources()

    assert ingest_queue.firsts() == [str(kept.id)]
    assert str(off.id) not in ingest_queue.firsts()
    assert str(busy.id) not in ingest_queue.firsts()


async def test_type_sweep_with_no_sources_queues_nothing(
    db_session, task_sessions, edges, ingest_queue
):
    await _source(db_session, "web")

    result = await sync_tasks._async_sync_sources_by_type("confluence")

    assert ingest_queue.calls == []
    assert result == {
        "source_type": "confluence",
        "total_sources": 0,
        "synced_sources": 0,
        "failed_sources": 0,
        "success": True,
    }


async def test_type_sweep_carries_on_past_a_source_that_cannot_be_queued(
    db_session, task_sessions, edges, ingest_queue
):
    refused = await _source(db_session, "web")
    fine = await _source(db_session, "web")
    ingest_queue.fail_for.add(str(refused.id))

    result = await sync_tasks._async_sync_sources_by_type("web")

    assert ingest_queue.firsts() == [str(fine.id)]
    assert (result["synced_sources"], result["failed_sources"]) == (1, 1)
    failed = [r for r in result["results"] if r["status"] == "failed"]
    assert failed == [
        {
            "source_id": str(refused.id),
            "source_name": refused.name,
            "error": "broker refused the message",
            "status": "failed",
        }
    ]


async def test_sync_all_sources_queues_every_active_source_grouped_by_type(
    db_session, task_sessions, edges, ingest_queue
):
    gitlab = await _source(db_session, "gitlab")
    web = await _source(db_session, "web")
    arxiv = await _source(db_session, "arxiv")
    await _source(db_session, "confluence", is_active=False)
    ingest_queue.fail_for.add(str(web.id))

    result = await sync_tasks._async_sync_all_sources()

    assert sorted(ingest_queue.firsts()) == sorted([str(gitlab.id), str(arxiv.id)])
    assert result["total_sources"] == 3
    assert (result["synced_sources"], result["failed_sources"]) == (2, 1)
    by_type = result["results_by_type"]
    assert set(by_type) == {"gitlab", "web", "arxiv"}
    assert by_type["web"][0]["status"] == "failed"
    assert by_type["gitlab"][0]["status"] == "triggered"


async def test_sync_all_sources_with_none_active(
    db_session, task_sessions, edges, ingest_queue
):
    await _source(db_session, "gitlab", is_active=False)

    result = await sync_tasks._async_sync_all_sources()

    assert ingest_queue.calls == []
    assert result["total_sources"] == 0 and result["success"] is True


# ==========================================================================
# What a queued sync records: ingest_from_source, end to end
# ==========================================================================

HOME_PAGE = (
    "<html><head><title>Build Guide</title></head><body><main>"
    "<p>How the compiler-research image is built and tagged.</p>"
    '<a href="/cache">Cache notes</a>'
    "</main></body></html>"
)
CACHE_PAGE = (
    "<html><head><title>Cache Notes</title></head><body><main>"
    "<p>Prefetchers on L2 report pfIssued and pfUseful counters.</p>"
    "</main></body></html>"
)


def _docs_site(web):
    web.route(f"{DOCS_URL}/", HOME_PAGE, content_type="text/html")
    web.route(f"{DOCS_URL}/cache", CACHE_PAGE, content_type="text/html")


def _web_config():
    return {"base_urls": [f"{DOCS_URL}/"], "max_depth": 2, "crawl_delay": 0}


async def test_web_sweep_then_ingestion_records_documents_and_a_sync_log(
    db_session, task_sessions, edges, web, ingest_queue, process_queue
):
    _docs_site(web)
    source = await _source(db_session, "web", config=_web_config())

    await sync_tasks._async_sync_sources_by_type("web")
    assert ingest_queue.firsts() == [str(source.id)]

    task = FakeTask()
    result = await ingestion_tasks._async_ingest_from_source(
        task, ingest_queue.calls[0]["source_id"]
    )

    assert result["success"] is True, result
    assert (result["created"], result["errors"]) == (2, 0)
    docs = await _all(db_session, Document, Document.source_id == source.id)
    assert sorted(d.title for d in docs) == ["Build Guide", "Cache Notes"]
    assert any("pfIssued" in (d.content or "") for d in docs)
    assert sorted(process_queue.firsts()) == sorted(str(d.id) for d in docs)

    row = await _reload(db_session, DocumentSource, source.id)
    assert row.last_sync is not None
    assert row.is_syncing is False
    assert row.last_error is None
    (log,) = await _all(
        db_session,
        DocumentSourceSyncLog,
        DocumentSourceSyncLog.source_id == source.id,
    )
    assert log.status == "success"
    assert log.finished_at is not None
    assert (log.total_documents, log.processed, log.created) == (2, 2, 2)
    assert (log.updated, log.errors) == (0, 0)
    assert edges.vector_calls == []

    # A second sync of an unchanged site creates and re-queues nothing.
    again = await ingestion_tasks._async_ingest_from_source(FakeTask(), str(source.id))
    assert (again["created"], again["updated"], again["errors"]) == (0, 0, 0)
    assert len(process_queue.calls) == 2
    logs = await _all(
        db_session,
        DocumentSourceSyncLog,
        DocumentSourceSyncLog.source_id == source.id,
    )
    assert sorted(entry.status for entry in logs) == ["success", "success"]


async def _rejected_gitlab_sync(db_session, web, ingest_queue):
    web.route(f"{GITLAB_URL}/api/v4/user", {"message": "401 Unauthorized"}, 401)
    source = await _source(
        db_session,
        "gitlab",
        config={"gitlab_url": GITLAB_URL, "token": "revoked", "projects": [1]},
    )
    await sync_tasks._async_sync_sources_by_type("gitlab")
    assert ingest_queue.firsts() == [str(source.id)]
    result = await ingestion_tasks._async_ingest_from_source(FakeTask(), str(source.id))
    return source, result


async def test_a_failed_sync_clears_syncing_and_fails_its_log(
    db_session, task_sessions, edges, web, ingest_queue, process_queue
):
    source, result = await _rejected_gitlab_sync(db_session, web, ingest_queue)

    assert result["success"] is False
    assert web.hits("gitlab.example.test"), "the credentials were never tried"
    row = await _reload(db_session, DocumentSource, source.id)
    assert row.is_syncing is False
    assert row.last_sync is None
    assert row.last_error
    (log,) = await _all(
        db_session,
        DocumentSourceSyncLog,
        DocumentSourceSyncLog.source_id == source.id,
    )
    assert log.status == "failed"
    assert log.error_message == row.last_error
    assert process_queue.calls == []


async def test_a_rejected_token_is_what_the_source_reports(
    db_session, task_sessions, edges, web, ingest_queue, process_queue
):
    source, result = await _rejected_gitlab_sync(db_session, web, ingest_queue)

    row = await _reload(db_session, DocumentSource, source.id)
    assert "HTTP 401" in (row.last_error or ""), row.last_error
    assert "HTTP 401" in result["error"]


# ==========================================================================
# scan_scheduled_sources
# ==========================================================================


async def test_scan_queues_only_sources_that_are_due(
    db_session, task_sessions, edges, ingest_queue
):
    never = await _source(
        db_session, "web", {"auto_sync": True, "sync_interval_minutes": 60}
    )
    overdue = await _source(
        db_session,
        "web",
        {"auto_sync": True, "sync_interval_minutes": 60},
        last_sync=_ago(hours=2),
    )
    cron_due = await _source(
        db_session,
        "gitlab",
        {"auto_sync": True, "cron": "0 * * * *"},
        last_sync=_ago(hours=3),
    )
    full = await _source(
        db_session,
        "confluence",
        {"auto_sync": True, "sync_interval_minutes": 5, "sync_only_changed": False},
    )
    # Not due, or not meant to be scheduled at all.
    await _source(
        db_session,
        "web",
        {"auto_sync": True, "sync_interval_minutes": 60},
        last_sync=_ago(minutes=10),
    )
    await _source(
        db_session,
        "gitlab",
        {"auto_sync": True, "cron": "0 0 1 1 *"},
        last_sync=_ago(minutes=1),
    )
    await _source(db_session, "web", {"auto_sync": False, "sync_interval_minutes": 1})
    await _source(db_session, "web", {"sync_interval_minutes": 1})
    await _source(
        db_session,
        "web",
        {"auto_sync": True, "sync_interval_minutes": 1},
        is_syncing=True,
    )
    await _source(
        db_session,
        "web",
        {"auto_sync": True, "sync_interval_minutes": 1},
        is_active=False,
    )
    await _source(db_session, "web", {"auto_sync": True, "cron": "not a cron"})

    result = await sync_tasks._async_scan_scheduled_sources()

    expected = {str(s.id) for s in (never, overdue, cron_due, full)}
    assert set(ingest_queue.firsts()) == expected
    assert len(ingest_queue.calls) == 4
    assert result["success"] is True and result["count"] == 4
    assert {t["source_id"] for t in result["triggered"]} == expected
    # The task id is recorded for the UI, and a full resync is flagged only
    # where the source asked for one.
    for item in result["triggered"]:
        assert edges.cache[f"ingestion:task:{item['source_id']}"] == item["task_id"]
    assert [k for k in edges.cache if k.startswith("ingestion:force_full:")] == [
        f"ingestion:force_full:{full.id}"
    ]


async def test_scan_with_nothing_scheduled(
    db_session, task_sessions, edges, ingest_queue
):
    await _source(db_session, "web", {})

    result = await sync_tasks._async_scan_scheduled_sources()

    assert result == {"triggered": [], "count": 0, "success": True}
    assert ingest_queue.calls == [] and edges.cache == {}


async def test_scan_carries_on_past_a_source_that_cannot_be_queued(
    db_session, task_sessions, edges, ingest_queue
):
    cfg = {"auto_sync": True, "sync_interval_minutes": 5}
    refused = await _source(db_session, "web", cfg)
    fine = await _source(db_session, "web", cfg)
    ingest_queue.fail_for.add(str(refused.id))

    result = await sync_tasks._async_scan_scheduled_sources()

    assert ingest_queue.firsts() == [str(fine.id)]
    assert [t["source_id"] for t in result["triggered"]] == [str(fine.id)]
    assert f"ingestion:task:{refused.id}" not in edges.cache


@pytest.mark.parametrize(
    "config",
    [
        {"auto_sync": True, "sync_interval_minutes": 60},
        {"auto_sync": True, "cron": "0 * * * *"},
    ],
    ids=["interval", "cron"],
)
async def test_scan_resyncs_a_source_that_has_synced_before_on_postgres(
    db_session, task_sessions, edges, ingest_queue, postgres_timestamptz, config
):
    overdue = await _source(db_session, "web", config, last_sync=_ago(hours=3))

    result = await sync_tasks._async_scan_scheduled_sources()

    assert ingest_queue.firsts() == [str(overdue.id)]
    assert result["count"] == 1


async def test_scan_on_postgres_still_starts_a_never_synced_source(
    db_session, task_sessions, edges, ingest_queue, postgres_timestamptz
):
    """The control for the xfail above: only a loaded last_sync breaks it."""
    fresh = await _source(
        db_session, "web", {"auto_sync": True, "sync_interval_minutes": 60}
    )

    await sync_tasks._async_scan_scheduled_sources()

    assert ingest_queue.firsts() == [str(fresh.id)]


# ==========================================================================
# cleanup_old_data
# ==========================================================================


async def _chat(db, user, *, last_message_at, is_active, messages=1):
    session = ChatSession(
        user_id=user.id,
        title=f"chat-{uuid4().hex[:6]}",
        is_active=is_active,
        last_message_at=last_message_at,
    )
    db.add(session)
    await db.flush()
    for i in range(messages):
        db.add(ChatMessage(session_id=session.id, content=f"m{i}", role="user"))
    await db.commit()
    await db.refresh(session)
    return session


def _log_file(directory, name, days_old, size=2048):
    path = directory / name
    path.write_bytes(b"x" * size)
    stamp = (datetime.now() - timedelta(days=days_old)).timestamp()
    os.utime(path, (stamp, stamp))
    return path


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    """The task cleans ``./data/logs`` relative to the worker's directory."""
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "data" / "logs"
    directory.mkdir(parents=True)
    return directory


async def test_cleanup_deletes_old_logs_and_old_inactive_chats_only(
    db_session, test_user, task_sessions, log_dir
):
    old_log = _log_file(log_dir, "app.2025-01-01.log", days_old=45, size=1024 * 1024)
    new_log = _log_file(log_dir, "app.log", days_old=2)
    (log_dir / "archive").mkdir()

    stale = await _chat(
        db_session, test_user, last_message_at=_ago(days=400), is_active=False
    )
    old_but_active = await _chat(
        db_session, test_user, last_message_at=_ago(days=400), is_active=True
    )
    recent_inactive = await _chat(
        db_session, test_user, last_message_at=_ago(days=30), is_active=False
    )

    result = await maintenance_tasks._async_cleanup_old_data()

    assert result["errors"] == []
    assert result["cleaned_items"] == 2  # one log file, one session
    assert result["freed_space_mb"] == 1.0
    assert not old_log.exists()
    assert new_log.exists() and (log_dir / "archive").is_dir()

    remaining = {s.id for s in await _all(db_session, ChatSession)}
    assert remaining == {old_but_active.id, recent_inactive.id}
    orphans = await _all(db_session, ChatMessage, ChatMessage.session_id == stale.id)
    assert orphans == []
    kept = await _all(
        db_session, ChatMessage, ChatMessage.session_id == recent_inactive.id
    )
    assert len(kept) == 1


async def test_cleanup_keeps_the_memories_of_a_chat_it_deletes(
    db_session, test_user, task_sessions, log_dir
):
    # Decided 2026-10-06: memories outlive the chat they came from. The
    # session's delete-orphan cascade used to take them with it.
    from app.models.memory import ConversationMemory

    stale = await _chat(
        db_session, test_user, last_message_at=_ago(days=500), is_active=False
    )
    memory = ConversationMemory(
        user_id=test_user.id,
        session_id=stale.id,
        memory_type="fact",
        content="prefers concise answers",
    )
    db_session.add(memory)
    await db_session.commit()

    await maintenance_tasks._async_cleanup_old_data()

    assert await _reload(db_session, ChatSession, stale.id) is None
    kept = await _reload(db_session, ConversationMemory, memory.id)
    assert kept is not None and kept.session_id is None
    assert kept.content == "prefers concise answers"


async def test_cleanup_with_nothing_to_clean(
    db_session, test_user, task_sessions, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)  # no ./data/logs at all
    await _chat(db_session, test_user, last_message_at=_ago(days=1), is_active=True)

    result = await maintenance_tasks._async_cleanup_old_data()

    assert "error" not in result
    assert (result["cleaned_items"], result["freed_space_mb"]) == (0, 0)
    assert len(await _all(db_session, ChatSession)) == 1


async def test_cleanup_carries_on_past_a_log_it_cannot_delete(
    db_session, test_user, task_sessions, log_dir, monkeypatch
):
    stuck = _log_file(log_dir, "locked.log", days_old=60)
    gone = _log_file(log_dir, "rotated.log", days_old=60)
    stale = await _chat(
        db_session, test_user, last_message_at=_ago(days=500), is_active=False
    )
    real_remove = os.remove

    def remove(path, *args, **kwargs):
        if os.path.basename(path) == "locked.log":
            raise PermissionError(13, "Permission denied", path)
        return real_remove(path, *args, **kwargs)

    monkeypatch.setattr(os, "remove", remove)

    result = await maintenance_tasks._async_cleanup_old_data()

    assert stuck.exists() and not gone.exists()
    assert len(result["errors"]) == 1
    assert result["errors"][0].startswith("Failed to delete locked.log")
    assert result["cleaned_items"] == 2
    assert await _reload(db_session, ChatSession, stale.id) is None


# ==========================================================================
# fail_stale_latex_compile_jobs
# ==========================================================================


async def _latex_job(db, user, status, *, created_ago, started_ago=None, log=None):
    job = LatexCompileJob(
        user_id=user.id,
        status=status,
        log=log,
        created_at=_ago(seconds=created_ago),
        started_at=_ago(seconds=started_ago) if started_ago is not None else None,
        violations=[],
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def test_stale_latex_jobs_are_failed_and_live_ones_left_alone(
    db_session, test_user, task_sessions
):
    u = test_user
    stuck_queue = await _latex_job(db_session, u, "queued", created_ago=900)
    stuck_run = await _latex_job(
        db_session, u, "running", created_ago=900, started_ago=400
    )
    never_started = await _latex_job(db_session, u, "running", created_ago=400)
    kept_log = await _latex_job(
        db_session, u, "running", created_ago=900, started_ago=400, log="  pass 1\n"
    )
    fresh_queue = await _latex_job(db_session, u, "queued", created_ago=60)
    fresh_run = await _latex_job(
        db_session, u, "running", created_ago=900, started_ago=60
    )
    done = await _latex_job(
        db_session, u, "succeeded", created_ago=9000, started_ago=9000
    )
    failed = await _latex_job(
        db_session, u, "failed", created_ago=9000, log="Undefined control sequence"
    )

    result = await latex_maintenance_tasks._async_fail_stale_latex_compile_jobs()

    assert result == {
        "updated": 4,
        "queued_stale_seconds": settings.LATEX_COMPILER_JOB_QUEUED_STALE_SECONDS,
        "running_stale_seconds": settings.LATEX_COMPILER_JOB_RUNNING_STALE_SECONDS,
    }
    rows = {
        j.id: await _reload(db_session, LatexCompileJob, j.id)
        for j in (
            stuck_queue,
            stuck_run,
            never_started,
            kept_log,
            fresh_queue,
            fresh_run,
            done,
            failed,
        )
    }
    assert rows[stuck_queue.id].status == "failed"
    assert "timed out in queue" in rows[stuck_queue.id].log
    assert rows[stuck_run.id].status == "failed"
    assert "timed out" in rows[stuck_run.id].log
    assert rows[never_started.id].status == "failed"
    assert rows[kept_log.id].status == "failed"
    assert rows[kept_log.id].log == "pass 1"
    for job_id in (stuck_queue.id, stuck_run.id, never_started.id, kept_log.id):
        assert rows[job_id].finished_at is not None
    assert rows[fresh_queue.id].status == "queued"
    assert rows[fresh_run.id].status == "running"
    assert rows[fresh_queue.id].finished_at is None
    assert rows[done.id].status == "succeeded"
    assert rows[failed.id].log == "Undefined control sequence"


async def test_no_stale_latex_jobs(db_session, test_user, task_sessions):
    job = await _latex_job(db_session, test_user, "queued", created_ago=5)

    result = await latex_maintenance_tasks._async_fail_stale_latex_compile_jobs()

    assert result["updated"] == 0
    assert (await _reload(db_session, LatexCompileJob, job.id)).status == "queued"


async def test_latex_staleness_follows_the_settings(
    db_session, test_user, task_sessions, monkeypatch
):
    monkeypatch.setattr(settings, "LATEX_COMPILER_JOB_QUEUED_STALE_SECONDS", 30)
    job = await _latex_job(db_session, test_user, "queued", created_ago=60)

    result = await latex_maintenance_tasks._async_fail_stale_latex_compile_jobs()

    assert result["updated"] == 1 and result["queued_stale_seconds"] == 30
    assert (await _reload(db_session, LatexCompileJob, job.id)).status == "failed"


# ==========================================================================
# sync_due_compops_evidence
# ==========================================================================


def _run_body(run_id, status):
    return {"run_id": run_id, "status": status, "private_log": f"audit-only-{status}"}


async def _compops_job(db, user):
    job = AgentJob(
        name="Track a compiler run",
        goal="Follow a CompOps run to completion",
        job_type="research",
        user_id=user.id,
        status=AgentJobStatus.COMPLETED.value,
        results={"evaluation_outcome": {"claims": [], "evidence": [], "actions": []}},
        output_artifacts=[],
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def _compops_tool(db, user):
    tool = UserTool(
        user_id=user.id,
        name="CompOps",
        tool_type="external_agent",
        parameters_schema={},
        config=external_agent_gateway_service.validate_config(
            {
                "provider_type": "compops",
                "endpoint_url": COMPOPS_URL,
                "capabilities": ["compops.runs.get"],
                "auth_type": "none",
            }
        ),
        is_enabled=True,
    )
    db.add(tool)
    await db.commit()
    await db.refresh(tool)
    return tool


async def _subscription(db, user, job, tool, run_id, **fields):
    row = CompOpsEvidenceSubscription(
        user_id=user.id,
        job_id=job.id,
        tool_id=tool.id,
        capability="compops.runs.get",
        remote_id=run_id,
        payload={"run_id": run_id},
        interval_minutes=fields.pop("interval_minutes", 15),
        is_enabled=fields.pop("is_enabled", True),
        status=fields.pop("status", "active"),
        next_sync_at=fields.pop(
            "next_sync_at", datetime.now(timezone.utc) - timedelta(minutes=1)
        ),
        **fields,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


def _naive(value):
    return value.replace(tzinfo=None) if value and value.tzinfo else value


async def test_due_compops_subscription_links_evidence_and_reschedules(
    db_session, test_user, task_sessions, edges, web
):
    job = await _compops_job(db_session, test_user)
    tool = await _compops_tool(db_session, test_user)
    web.route(f"{COMPOPS_URL}/v1/runs/run-42", _run_body("run-42", "running"))
    sub = await _subscription(db_session, test_user, job, tool, "run-42")
    started = datetime.utcnow()

    summary = await compops_sync_tasks._async_sync_due_compops_evidence()

    assert (summary["checked"], summary["updated"], summary["failed"]) == (1, 1, 0)
    assert [r.url.path for r in web.hits("compops.example.test")] == ["/v1/runs/run-42"]
    row = await _reload(db_session, CompOpsEvidenceSubscription, sub.id)
    assert row.status == "active" and row.last_error is None
    assert row.last_response_sha256
    assert _naive(row.next_sync_at) >= started + timedelta(minutes=14)
    audit = await _reload(db_session, ToolExecutionAudit, row.last_audit_id)
    assert audit.status == "completed"
    assert audit.tool_input["sync_subscription_id"] == str(sub.id)

    parent = await _reload(db_session, AgentJob, job.id)
    evidence = [
        e
        for e in parent.results["evaluation_outcome"]["evidence"]
        if e.get("kind") == "external_system_response"
    ]
    assert [e["id"] for e in evidence] == [f"external-system:{sub.id}"]
    assert "audit-only" not in repr(parent.results)

    # Not due again until its interval has passed.
    again = await compops_sync_tasks._async_sync_due_compops_evidence()
    assert again["checked"] == 0
    assert len(web.hits("compops.example.test")) == 1


async def test_unchanged_compops_response_is_counted_unchanged(
    db_session, test_user, task_sessions, edges, web
):
    job = await _compops_job(db_session, test_user)
    tool = await _compops_tool(db_session, test_user)
    web.route(f"{COMPOPS_URL}/v1/runs/run-7", _run_body("run-7", "running"))
    sub = await _subscription(db_session, test_user, job, tool, "run-7")
    await compops_sync_tasks._async_sync_due_compops_evidence()

    row = await _reload(db_session, CompOpsEvidenceSubscription, sub.id)
    row.next_sync_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    await db_session.commit()

    summary = await compops_sync_tasks._async_sync_due_compops_evidence()

    assert (summary["checked"], summary["updated"], summary["unchanged"]) == (1, 0, 1)
    parent = await _reload(db_session, AgentJob, job.id)
    evidence = [
        e
        for e in parent.results["evaluation_outcome"]["evidence"]
        if e.get("kind") == "external_system_response"
    ]
    assert len(evidence) == 1


async def test_compops_sweep_leaves_subscriptions_that_are_not_due(
    db_session, test_user, task_sessions, edges, web
):
    job = await _compops_job(db_session, test_user)
    tool = await _compops_tool(db_session, test_user)
    future = await _subscription(
        db_session,
        test_user,
        job,
        tool,
        "run-a",
        next_sync_at=datetime.now(timezone.utc) + timedelta(minutes=10),
    )
    disabled = await _subscription(
        db_session, test_user, job, tool, "run-b", is_enabled=False
    )
    unscheduled = await _subscription(
        db_session, test_user, job, tool, "run-c", next_sync_at=None
    )

    summary = await compops_sync_tasks._async_sync_due_compops_evidence()

    assert summary["checked"] == 0
    assert web.requests == []
    for sub in (future, disabled, unscheduled):
        row = await _reload(db_session, CompOpsEvidenceSubscription, sub.id)
        assert row.status == "active" and row.last_attempt_at is None


async def test_a_failing_compops_subscription_does_not_stop_the_others(
    db_session, test_user, task_sessions, edges, web
):
    job = await _compops_job(db_session, test_user)
    tool = await _compops_tool(db_session, test_user)
    web.route(f"{COMPOPS_URL}/v1/runs/run-bad", {"detail": "boom"}, status=500)
    web.route(f"{COMPOPS_URL}/v1/runs/run-ok", _run_body("run-ok", "done"))
    now = datetime.now(timezone.utc)
    bad = await _subscription(
        db_session,
        test_user,
        job,
        tool,
        "run-bad",
        next_sync_at=now - timedelta(minutes=5),
    )
    ok = await _subscription(
        db_session,
        test_user,
        job,
        tool,
        "run-ok",
        next_sync_at=now - timedelta(minutes=1),
    )
    orphan = await _subscription(
        db_session,
        test_user,
        job,
        SimpleNamespace(id=uuid4()),  # its connection has been deleted
        "run-gone",
        next_sync_at=now - timedelta(minutes=3),
    )

    summary = await compops_sync_tasks._async_sync_due_compops_evidence()

    assert summary["checked"] == 3
    assert (summary["updated"], summary["failed"]) == (1, 2)

    bad_row = await _reload(db_session, CompOpsEvidenceSubscription, bad.id)
    assert bad_row.status == "error"
    assert bad_row.last_error == "External agent returned HTTP 500"
    assert bad_row.is_enabled is True
    assert _naive(bad_row.next_sync_at) > datetime.utcnow() + timedelta(minutes=10)
    audit = await _reload(db_session, ToolExecutionAudit, bad_row.last_audit_id)
    assert audit.status == "failed" and "HTTP 500" in audit.error

    orphan_row = await _reload(db_session, CompOpsEvidenceSubscription, orphan.id)
    assert orphan_row.status == "invalid" and orphan_row.is_enabled is False
    assert orphan_row.next_sync_at is None

    ok_row = await _reload(db_session, CompOpsEvidenceSubscription, ok.id)
    assert ok_row.status == "active" and ok_row.last_success_at is not None


# ==========================================================================
# The synchronous Celery functions, each in its own event loop
# ==========================================================================


@pytest.fixture
def file_db(tmp_path, monkeypatch):
    """A file database reachable from any event loop, as Postgres is.

    ``.run`` calls ``asyncio.run``, which makes a new loop; each call gets a
    fresh engine on it, which is what ``create_celery_session`` does.
    """
    url = f"sqlite+aiosqlite:///{tmp_path / 'tasks.db'}"

    def factory():
        engine = create_async_engine(url, poolclass=NullPool)
        return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    for module in TASK_MODULES:
        monkeypatch.setattr(module, "create_celery_session", factory)

    async def create():
        engine = create_async_engine(url, poolclass=NullPool)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        await engine.dispose()

    asyncio.run(create())

    def run(coro_fn):
        async def go():
            async with factory()() as db:
                return await coro_fn(db)

        return asyncio.run(go())

    return run


def _seed_user(run):
    async def make(db):
        return await AuthService().create_user(
            username="beatuser",
            email="beat@example.com",
            password="testpassword123",
            full_name="Beat User",
            db=db,
        )

    return run(make)


def _get(run, model_cls, row_id):
    async def fetch(db):
        return (
            await db.execute(select(model_cls).where(model_cls.id == row_id))
        ).scalar_one_or_none()

    return run(fetch)


@pytest.mark.parametrize(
    "task,source_type",
    [
        (sync_tasks.sync_all_gitlab_sources, "gitlab"),
        (sync_tasks.sync_all_confluence_sources, "confluence"),
        (sync_tasks.sync_all_web_sources, "web"),
    ],
    ids=["gitlab", "confluence", "web"],
)
def test_type_sweep_task_run(file_db, edges, ingest_queue, task, source_type):
    mine = file_db(lambda db: _source(db, source_type))
    file_db(lambda db: _source(db, "arxiv"))

    result = task.run()

    assert ingest_queue.firsts() == [str(mine.id)]
    assert result["synced_sources"] == 1 and result["success"] is True


def test_sync_all_sources_task_run(file_db, edges, ingest_queue):
    a = file_db(lambda db: _source(db, "gitlab"))
    b = file_db(lambda db: _source(db, "arxiv"))

    result = sync_tasks.sync_all_sources.run()

    assert sorted(ingest_queue.firsts()) == sorted([str(a.id), str(b.id)])
    assert result["synced_sources"] == 2


def test_scan_scheduled_sources_task_run(file_db, edges, ingest_queue):
    due = file_db(
        lambda db: _source(db, "web", {"auto_sync": True, "sync_interval_minutes": 5})
    )
    file_db(lambda db: _source(db, "web", {"auto_sync": False}))

    result = sync_tasks.scan_scheduled_sources.run()

    assert ingest_queue.firsts() == [str(due.id)]
    assert result["count"] == 1


def test_cleanup_old_data_task_run(file_db, log_dir):
    user = _seed_user(file_db)
    old = _log_file(log_dir, "old.log", days_old=31)
    stale = file_db(
        lambda db: _chat(db, user, last_message_at=_ago(days=366), is_active=False)
    )
    live = file_db(
        lambda db: _chat(db, user, last_message_at=_ago(days=366), is_active=True)
    )

    result = maintenance_tasks.cleanup_old_data.run()

    assert result["cleaned_items"] == 2 and result["errors"] == []
    assert not old.exists()
    assert _get(file_db, ChatSession, stale.id) is None
    assert _get(file_db, ChatSession, live.id) is not None


def test_fail_stale_latex_compile_jobs_task_run(file_db):
    user = _seed_user(file_db)
    stale = file_db(lambda db: _latex_job(db, user, "queued", created_ago=3600))
    fresh = file_db(lambda db: _latex_job(db, user, "queued", created_ago=1))

    result = latex_maintenance_tasks.fail_stale_latex_compile_jobs.run()

    assert result["updated"] == 1
    assert _get(file_db, LatexCompileJob, stale.id).status == "failed"
    assert _get(file_db, LatexCompileJob, fresh.id).status == "queued"


def test_sync_due_compops_evidence_task_run(file_db, edges, web):
    user = _seed_user(file_db)
    job = file_db(lambda db: _compops_job(db, user))
    tool = file_db(lambda db: _compops_tool(db, user))
    web.route(f"{COMPOPS_URL}/v1/runs/run-9", _run_body("run-9", "queued"))
    sub = file_db(lambda db: _subscription(db, user, job, tool, "run-9"))

    summary = compops_sync_tasks.sync_due_compops_evidence.run()

    assert (summary["checked"], summary["updated"]) == (1, 1)
    row = _get(file_db, CompOpsEvidenceSubscription, sub.id)
    assert row.status == "active" and row.last_success_at is not None
