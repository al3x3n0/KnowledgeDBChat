"""Document, paper, chat and git Celery tasks, run for real.

None of these task bodies was named in a test, so none had ever executed
outside a worker:

* ``chat_tasks.generate_chat_title``
* ``git_compare_tasks.compare_git_branches``
* ``ingestion_tasks.process_uploaded_document`` and ``dry_run_source``
* ``processing_tasks.process_document``
* ``paper_enrichment_tasks.enrich_arxiv_document``
* ``paper_kg_tasks.upsert_paper_insights_to_kg``
* ``research_tasks.generate_literature_review``

The async bodies run against the in-memory database; each synchronous Celery
function runs once via ``.run`` against a file database, in its own event
loop, as a worker would. Judgement is on the rows they leave behind.

Only edges that leave the process are replaced, and each replacement binds its
arguments against the real callee, so a call the real thing would refuse fails
here too:

* Redis -- ``job_support`` publishers, the git-compare cancel flags, and the
  ``cache_service`` the feature flags and document cache read through;
* the vector store -- ``vector_store_service`` methods;
* the model -- ``LLMService.generate_response``;
* HTTP (arXiv, Crossref, GitHub, GitLab) -- ``httpx.MockTransport`` behind the
  real connectors and services;
* Celery's ``.delay`` -- bound against the real task function.

A test marked xfail(strict) is a defect in the application, not in the test.
"""

import asyncio
import hashlib
import inspect
import json
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pytest
from celery.app.task import Task as CeleryTask
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.cache import cache_service
from app.core.config import settings
from app.core.database import Base
from app.models.chat import ChatMessage, ChatSession
from app.models.document import Document, DocumentChunk, DocumentSource, GitBranchDiff
from app.models.knowledge_graph import Entity, EntityMention, Relationship
from app.services.auth_service import AuthService
from app.services.llm_service import LLMService
from app.services.vector_store import vector_store_service
from app.tasks import (
    chat_tasks,
    git_compare_tasks,
    ingestion_tasks,
    job_support,
    paper_enrichment_tasks,
    paper_kg_tasks,
    processing_tasks,
    research_tasks,
)
from app.utils import ingestion_state

pytestmark = pytest.mark.unit

TASK_MODULES = (
    chat_tasks,
    git_compare_tasks,
    ingestion_tasks,
    paper_enrichment_tasks,
    paper_kg_tasks,
    processing_tasks,
    research_tasks,
    job_support,
)

PAPER_TEXT = (
    "Hardware prefetchers predict future cache misses from the history of "
    "past ones. A stride prefetcher detects constant address deltas per load "
    "instruction and issues requests ahead of the demand stream.\n\n"
    "Irregular access patterns defeat stride detection. The irregular stream "
    "buffer linearises correlated miss sequences into a structural address "
    "space, so that temporal streams look sequential to the prefetch engine.\n\n"
    "We evaluate both designs on pointer-chasing and streaming kernels and "
    "report coverage, accuracy and the speedup over a baseline without any "
    "prefetching at all."
)


# --------------------------------------------------------------------------
# Edges that leave the process
# --------------------------------------------------------------------------


def _binding(real):
    """A checker that raises TypeError exactly when ``real`` would."""
    signature = inspect.signature(real)

    def bind(*args, **kwargs):
        return signature.bind(*args, **kwargs).arguments

    return bind


def _recording(real, impl):
    """An async stand-in that refuses arguments ``real`` does not accept."""
    bind = _binding(real)

    async def call(*args, **kwargs):
        bind(*args, **kwargs)
        return impl(*args, **kwargs)

    return call


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
    """Redis, the cache and the vector store, and switches to break them."""

    def __init__(self):
        self.published = []
        self.vector_added = []
        self.vector_deleted = []
        self.fail_vector_add = None
        self.cancelled = set()
        self.cleared = []

    def install(self, monkeypatch):
        bind_sync = _binding(job_support.publish_sync)
        bind_message = _binding(job_support.publish_message)
        bind_flag = _binding(job_support.flag_is_set)
        bind_keys = _binding(job_support.delete_keys)
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
            bind_flag(*args, **kwargs)
            return False

        def delete_keys(*args, **kwargs):
            bind_keys(*args, **kwargs)

        monkeypatch.setattr(job_support, "publish_sync", publish_sync)
        monkeypatch.setattr(job_support, "publish_message", publish_message)
        monkeypatch.setattr(job_support, "flag_is_set", flag_is_set)
        monkeypatch.setattr(job_support, "delete_keys", delete_keys)

        def vector_add(document, chunks):
            if edges.fail_vector_add is not None:
                raise edges.fail_vector_add
            edges.vector_added.append((document.id, [c.content for c in chunks]))
            return [str(uuid4()) for _ in chunks]

        def vector_delete(document_id):
            edges.vector_deleted.append(document_id)

        for obj, name, impl in [
            (vector_store_service, "initialize", lambda *a, **k: None),
            (vector_store_service, "add_document_chunks", vector_add),
            (vector_store_service, "delete_document_chunks", vector_delete),
            (cache_service, "get", lambda key: None),
            (cache_service, "set", lambda key, value, ttl=None: True),
            (cache_service, "delete", lambda key: True),
            (cache_service, "delete_pattern", lambda pattern: 0),
        ]:
            monkeypatch.setattr(obj, name, _recording(getattr(obj, name), impl))

        # The git-compare cancel flags live in Redis.
        bind_cancelled = _binding(ingestion_state.is_git_compare_cancelled)
        bind_clear = _binding(ingestion_state.clear_git_compare_task)

        async def is_cancelled(*args, **kwargs):
            a = bind_cancelled(*args, **kwargs)
            return a["diff_id"] in edges.cancelled

        async def clear(*args, **kwargs):
            a = bind_clear(*args, **kwargs)
            edges.cleared.append(a["diff_id"])

        monkeypatch.setattr(git_compare_tasks, "is_git_compare_cancelled", is_cancelled)
        monkeypatch.setattr(git_compare_tasks, "clear_git_compare_task", clear)

        # Indexing stays on what is under test: chunk rows and the vector store.
        monkeypatch.setattr(settings, "KNOWLEDGE_GRAPH_ENABLED", False, raising=False)
        monkeypatch.setattr(settings, "AUTO_SUMMARIZE_ON_PROCESS", False, raising=False)
        monkeypatch.setattr(settings, "RAG_CHUNKING_STRATEGY", "fixed", raising=False)
        monkeypatch.setattr(settings, "CHUNK_SIZE", 300, raising=False)
        monkeypatch.setattr(settings, "CHUNK_OVERLAP", 40, raising=False)
        return self


class FakeModel:
    """``LLMService.generate_response``; answers with ``reply`` or raises."""

    def __init__(self):
        self.calls = []
        self.reply = "A model answer."
        self.fail_with = None

    def install(self, monkeypatch):
        bind = _binding(LLMService.generate_response)
        model = self

        async def generate_response(*args, **kwargs):
            a = bind(*args, **kwargs)
            prompt = a.get("query") or a.get("prompt") or a.get("user_message")
            assert isinstance(prompt, str) and prompt
            model.calls.append(a)
            if model.fail_with is not None:
                raise model.fail_with
            return model.reply(prompt) if callable(model.reply) else model.reply

        monkeypatch.setattr(LLMService, "generate_response", generate_response)
        return self

    def prompts(self):
        return [c.get("query") or c.get("prompt") or "" for c in self.calls]


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
    """A Celery task's ``.delay``, bound against the real task function."""

    def __init__(self, task):
        self.signature = inspect.signature(task.run)
        self.calls = []

    def delay(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        self.calls.append(dict(bound.arguments))
        return SimpleNamespace(id=str(uuid4()))


@pytest.fixture
def edges(monkeypatch):
    return Edges().install(monkeypatch)


@pytest.fixture
def model(monkeypatch):
    return FakeModel().install(monkeypatch)


@pytest.fixture
def web(monkeypatch):
    rec = Web()
    real_client = httpx.AsyncClient

    def client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(rec.handle)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    # No PDF downloads: the abstract is what an arXiv entry carries here.
    monkeypatch.setattr(settings, "ARXIV_FULL_TEXT_ENABLED", False, raising=False)
    return rec


@pytest.fixture
def ingest_queue(monkeypatch):
    queue = Queue(ingestion_tasks.process_uploaded_document)
    monkeypatch.setattr(ingestion_tasks.process_uploaded_document, "delay", queue.delay)
    return queue


@pytest.fixture
def task_sessions(db_session, monkeypatch):
    factory = async_sessionmaker(
        db_session.bind, class_=AsyncSession, expire_on_commit=False
    )
    for module in TASK_MODULES:
        monkeypatch.setattr(module, "create_celery_session", lambda: factory)
    return factory


async def _reload(db, model_cls, row_id):
    return (
        await db.execute(
            select(model_cls)
            .where(model_cls.id == row_id)
            .execution_options(populate_existing=True)
        )
    ).scalar_one_or_none()


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------


async def _source(db, source_type="file", config=None, name=None):
    source = DocumentSource(
        name=name or f"src-{uuid4().hex[:8]}",
        source_type=source_type,
        config=config or {},
    )
    db.add(source)
    await db.commit()
    await db.refresh(source)
    return source


async def _document(db, source, **overrides):
    content = overrides.pop("content", PAPER_TEXT)
    fields = {
        "title": "Stride and Irregular Prefetching",
        "content": content,
        "content_hash": hashlib.sha256(content.encode()).hexdigest(),
        "source_id": source.id,
        "source_identifier": f"test:{uuid4().hex}",
        "file_type": "txt",
    }
    fields.update(overrides)
    doc = Document(**fields)
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc


async def _arxiv_paper(db, source, arxiv_id="2601.00001v1", **overrides):
    fields = {
        "title": "Irregular Stream Buffer Revisited",
        "source_identifier": f"http://arxiv.org/abs/{arxiv_id}",
        "url": f"http://arxiv.org/abs/{arxiv_id}",
        "author": "A. Author, B. Author",
        "extra_metadata": {
            "authors": ["A. Author", "B. Author"],
            "categories": ["cs.AR", "cs.PF"],
            "primary_category": "cs.AR",
            "doi": None,
        },
    }
    fields.update(overrides)
    return await _document(db, source, **fields)


async def _chunks(db, document_id):
    rows = (
        await db.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
            .execution_options(populate_existing=True)
        )
    ).scalars()
    return list(rows)


async def _count(db, model_cls, *where):
    query = select(func.count(model_cls.id))
    for clause in where:
        query = query.where(clause)
    return (await db.execute(query)).scalar()


# ==========================================================================
# process_uploaded_document / process_document
# ==========================================================================

PROCESSORS = {
    "process_uploaded_document": lambda task, doc_id: (
        ingestion_tasks._async_process_uploaded_document(task, doc_id)
    ),
    "process_document": lambda task, doc_id: (
        processing_tasks._async_process_document(task, doc_id)
    ),
}


@pytest.mark.parametrize("name", ["process_uploaded_document", "process_document"])
async def test_processing_chunks_and_indexes_the_document(
    db_session, task_sessions, edges, name
):
    source = await _source(db_session)
    doc = await _document(db_session, source)
    task = FakeTask()

    result = await PROCESSORS[name](task, str(doc.id))

    row = await _reload(db_session, Document, doc.id)
    assert row.processing_error is None
    assert row.is_processed is True
    chunks = await _chunks(db_session, doc.id)
    assert len(chunks) >= 2
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert "stride prefetcher" in chunks[0].content
    assert "baseline without any" in chunks[-1].content
    # The vector store received exactly the chunks the database holds.
    assert edges.vector_added == [(doc.id, [c.content for c in chunks])]
    assert result["success"] is True and result["document_id"] == str(doc.id)
    assert result["processed"] is True
    if name == "process_document":
        assert result["chunks_count"] == len(chunks)
    assert task.states[-1][1]["status"] == "Processing completed"


@pytest.mark.parametrize("name", list(PROCESSORS))
async def test_processing_a_missing_document_names_it(
    db_session, task_sessions, edges, name
):
    missing = str(uuid4())

    result = await PROCESSORS[name](FakeTask(), missing)

    assert result == {
        "document_id": missing,
        "error": f"Document {missing} not found",
        "success": False,
    }
    assert edges.vector_added == []


@pytest.mark.parametrize("name", ["process_uploaded_document", "process_document"])
async def test_processing_that_fails_to_index_reports_failure(
    db_session, task_sessions, edges, name
):
    edges.fail_vector_add = ConnectionError("qdrant:6333 refused")
    source = await _source(db_session)
    doc = await _document(db_session, source)

    result = await PROCESSORS[name](FakeTask(), str(doc.id))

    row = await _reload(db_session, Document, doc.id)
    assert row.is_processed is False
    assert "qdrant:6333 refused" in (row.processing_error or "")
    assert result["success"] is False
    assert "qdrant:6333 refused" in (result.get("error") or "")


async def test_a_failed_indexing_is_recorded_on_the_row(
    db_session, task_sessions, edges
):
    """Whatever the task reports, the row says what happened."""
    edges.fail_vector_add = ConnectionError("qdrant:6333 refused")
    source = await _source(db_session)
    doc = await _document(db_session, source)

    await ingestion_tasks._async_process_uploaded_document(FakeTask(), str(doc.id))

    row = await _reload(db_session, Document, doc.id)
    assert row.is_processed is False
    assert row.processing_error == "qdrant:6333 refused"


async def test_reprocessing_changed_content_leaves_only_the_new_chunks(
    db_session, task_sessions, edges
):
    source = await _source(db_session)
    doc = await _document(db_session, source)
    await ingestion_tasks._async_process_uploaded_document(FakeTask(), str(doc.id))
    assert len(await _chunks(db_session, doc.id)) >= 2

    # What _update_document does when a source sync sees new content.
    new_text = (
        "Revised paper. The evaluation now covers only graph analytics "
        "workloads, where correlation prefetching is measured against a "
        "next-line baseline on a modern out-of-order core."
    )
    row = await _reload(db_session, Document, doc.id)
    row.content = new_text
    row.content_hash = hashlib.sha256(new_text.encode()).hexdigest()
    row.is_processed = False
    await db_session.commit()

    await ingestion_tasks._async_process_uploaded_document(FakeTask(), str(doc.id))

    chunks = await _chunks(db_session, doc.id)
    assert chunks, "the new content was not chunked"
    assert all("stride prefetcher" not in c.content for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


# ==========================================================================
# dry_run_source
# ==========================================================================


def _arxiv_entry(arxiv_url, title):
    return (
        "<entry>"
        f"<id>{arxiv_url}</id>"
        f"<title>{title}</title>"
        "<summary>An abstract about prefetching.</summary>"
        "<published>2026-01-02T00:00:00Z</published>"
        "<author><name>A. Author</name></author>"
        '<category term="cs.AR"/>'
        '<arxiv:primary_category term="cs.AR"/>'
        "</entry>"
    )


def _arxiv_feed(*entries):
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:arxiv="http://arxiv.org/schemas/atom">' + "".join(entries) + "</feed>"
    )


ARXIV_API = "https://export.arxiv.org/api/query"
TWO_PAPERS = _arxiv_feed(
    _arxiv_entry("http://arxiv.org/abs/2601.00001v1", "Stride prefetching I"),
    _arxiv_entry("http://arxiv.org/abs/2601.00002v1", "Stride prefetching II"),
)


async def test_dry_run_counts_new_and_existing_without_writing(
    db_session, task_sessions, edges, web
):
    web.route(ARXIV_API, TWO_PAPERS, content_type="application/atom+xml")
    source = await _source(
        db_session, "arxiv", {"queries": ["all:prefetching"], "max_results": 5}
    )
    await _arxiv_paper(db_session, source, "2601.00001v1")

    result = await ingestion_tasks._async_dry_run_source(FakeTask(), str(source.id), {})

    assert result["success"] is True, result
    assert result["total"] == 2
    assert result["mode"] == "full"
    assert result["by_type"] == {"unknown": 2}
    assert result["estimated_existing"] == 1
    assert result["estimated_new"] == 1
    assert [s["title"] for s in result["sample"]] == [
        "Stride prefetching I",
        "Stride prefetching II",
    ]
    params = dict(web.hits("export.arxiv.org")[0].url.params)
    assert params["search_query"] == "all:prefetching"
    assert params["max_results"] == "5"
    # A dry run writes nothing.
    assert await _count(db_session, Document) == 1
    row = await _reload(db_session, DocumentSource, source.id)
    assert row.last_sync is None and row.is_syncing is False


async def test_dry_run_of_a_missing_source(db_session, task_sessions, edges):
    missing = str(uuid4())

    result = await ingestion_tasks._async_dry_run_source(FakeTask(), missing, {})

    assert result == {
        "success": False,
        "error": "Source not found",
        "source_id": missing,
    }


async def test_dry_run_of_a_source_without_a_connector(
    db_session, task_sessions, edges
):
    source = await _source(db_session, "file")

    result = await ingestion_tasks._async_dry_run_source(FakeTask(), str(source.id), {})

    assert result["success"] is False
    assert result["error"] == "No connector for type file"


async def test_dry_run_against_a_failing_upstream_names_the_status(
    db_session, task_sessions, edges, web
):
    web.route(ARXIV_API, "Service Unavailable", status=503, content_type="text/plain")
    source = await _source(db_session, "arxiv", {"queries": ["all:prefetching"]})

    result = await ingestion_tasks._async_dry_run_source(FakeTask(), str(source.id), {})

    assert result["success"] is False
    assert "503" in result["error"]


async def test_dry_run_of_a_misconfigured_source_names_the_configuration_error(
    db_session, task_sessions, edges, web
):
    source = await _source(db_session, "arxiv", {"queries": []})

    result = await ingestion_tasks._async_dry_run_source(FakeTask(), str(source.id), {})

    assert result["success"] is False
    assert "search query" in result["error"]


# ==========================================================================
# generate_chat_title
# ==========================================================================


async def _chat(db, user, title="Chat 2026-10-05 09:30", messages=2):
    session = ChatSession(user_id=user.id, title=title)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    turns = [
        ("user", "Why did the ISB prefetcher match the baseline to the cycle?"),
        ("assistant", "Its pfIdentified counter reads 0: it never engaged."),
        ("user", "And the stride prefetcher?"),
        ("assistant", "It issued 63,127 prefetches, 4,171 of them useful."),
    ]
    for role, content in turns[:messages]:
        db.add(ChatMessage(session_id=session.id, role=role, content=content))
        await db.commit()
    return session


async def test_chat_title_is_generated_from_the_conversation(
    db_session, test_user, task_sessions, model
):
    model.reply = '  "Title: ISB Prefetcher Never Engaged"\n'
    session = await _chat(db_session, test_user)

    result = await chat_tasks._async_generate_chat_title(FakeTask(), str(session.id))

    row = await _reload(db_session, ChatSession, session.id)
    date = row.created_at.strftime("%Y-%m-%d")
    assert row.title == f"{date} - ISB Prefetcher Never Engaged"
    assert result == {
        "session_id": str(session.id),
        "success": True,
        "title": row.title,
        "generated_title": "ISB Prefetcher Never Engaged",
    }
    (call,) = model.calls
    assert "User: Why did the ISB prefetcher" in call["query"]
    assert "Assistant: Its pfIdentified counter reads 0" in call["query"]
    assert call["task_type"] == "title_generation"


async def test_a_long_generated_title_is_cut_at_a_word(
    db_session, test_user, task_sessions, model
):
    model.reply = (
        "Why the irregular stream buffer prefetcher never engaged on any kernel"
    )
    session = await _chat(db_session, test_user)

    result = await chat_tasks._async_generate_chat_title(FakeTask(), str(session.id))

    assert result["generated_title"] == (
        "Why the irregular stream buffer prefetcher never engaged on"
    )
    assert len(result["generated_title"]) <= 60


async def test_chat_title_for_a_missing_session(db_session, task_sessions, model):
    missing = str(uuid4())

    result = await chat_tasks._async_generate_chat_title(FakeTask(), missing)

    assert result == {
        "session_id": missing,
        "success": False,
        "error": "Session not found",
    }
    assert model.calls == []


async def test_chat_title_waits_for_an_exchange(
    db_session, test_user, task_sessions, model
):
    session = await _chat(db_session, test_user, messages=1)

    result = await chat_tasks._async_generate_chat_title(FakeTask(), str(session.id))

    assert result["error"] == "Not enough messages"
    assert (await _reload(db_session, ChatSession, session.id)).title == (
        "Chat 2026-10-05 09:30"
    )
    assert model.calls == []


async def test_chat_title_falls_back_when_the_model_fails(
    db_session, test_user, task_sessions, model
):
    model.fail_with = RuntimeError("deepseek: 401 invalid api key")
    session = await _chat(db_session, test_user)

    result = await chat_tasks._async_generate_chat_title(FakeTask(), str(session.id))

    row = await _reload(db_session, ChatSession, session.id)
    date = row.created_at.strftime("%Y-%m-%d")
    assert row.title == f"{date} - Chat"
    assert result["success"] is False
    assert result["error"] == "deepseek: 401 invalid api key"
    assert result["fallback_title"] == row.title


async def test_a_generated_title_is_not_regenerated(
    db_session, test_user, task_sessions, model
):
    session = await _chat(db_session, test_user, title="2026-10-05 - Prefetcher Study")

    result = await chat_tasks._async_generate_chat_title(FakeTask(), str(session.id))

    assert result["message"] == "Title already exists"
    assert model.calls == []


async def test_a_title_the_user_chose_is_kept(
    db_session, test_user, task_sessions, model
):
    model.reply = "Generated Title"
    session = await _chat(db_session, test_user, title="Chat about prefetchers")

    await chat_tasks._async_generate_chat_title(FakeTask(), str(session.id))

    row = await _reload(db_session, ChatSession, session.id)
    assert row.title == "Chat about prefetchers"


# ==========================================================================
# compare_git_branches
# ==========================================================================

GITHUB = "https://api.github.com"
GITHUB_COMPARE = {
    "ahead_by": 2,
    "behind_by": 1,
    "commits": [
        {
            "commit": {
                "message": "Add stride prefetcher",
                "author": {"name": "Ada", "date": "2026-10-01T10:00:00Z"},
            }
        },
        {
            "commit": {
                "message": "Tune degree",
                "author": {"name": "Ada", "date": "2026-10-02T10:00:00Z"},
            }
        },
    ],
    "files": [
        {
            "filename": "README.md",
            "status": "modified",
            "additions": 1,
            "deletions": 0,
            "changes": 1,
        },
        {
            "filename": "src/prefetch/stride.c",
            "status": "added",
            "additions": 120,
            "deletions": 0,
            "changes": 120,
        },
        {
            "filename": "src/prefetch/isb.c",
            "status": "modified",
            "additions": 10,
            "deletions": 30,
            "changes": 40,
        },
    ],
}


def _github_routes(web, compare=GITHUB_COMPARE, compare_status=200):
    web.route(f"{GITHUB}/repos/acme/prefetch", {"full_name": "acme/prefetch"})
    web.route(
        f"{GITHUB}/repos/acme/prefetch/compare/main...stride",
        compare,
        status=compare_status,
    )


async def _diff(db, source, **overrides):
    fields = {
        "source_id": source.id,
        "repository": "acme/prefetch",
        "base_branch": "main",
        "compare_branch": "stride",
        "status": "queued",
        "options": {"include_files": True, "explain": True},
    }
    fields.update(overrides)
    diff = GitBranchDiff(**fields)
    db.add(diff)
    await db.commit()
    await db.refresh(diff)
    return diff


async def _github_source(db, **config):
    return await _source(db, "github", {"repos": ["acme/prefetch"], **config})


async def test_branch_comparison_stores_its_summary_and_explanation(
    db_session, test_user, task_sessions, edges, model, web
):
    _github_routes(web)
    model.reply = "Adds a stride prefetcher; validate on streaming kernels."
    source = await _github_source(db_session)
    diff = await _diff(db_session, source)

    result = await git_compare_tasks._async_compare_git_branches(
        FakeTask(), str(diff.id), str(test_user.id)
    )

    assert result == {"success": True}
    row = await _reload(db_session, GitBranchDiff, diff.id)
    assert row.status == "completed" and row.error is None
    assert row.completed_at is not None
    summary = row.diff_summary
    assert summary["stats"] == {
        "ahead_by": 2,
        "behind_by": 1,
        "total_commits": 2,
        "total_files": 3,
    }
    assert [f["filename"] for f in summary["files"]] == [
        "src/prefetch/stride.c",
        "src/prefetch/isb.c",
        "README.md",
    ]
    assert summary["files"][0]["status"] == "added"
    assert summary["raw"]["commit_messages"][0] == {
        "message": "Add stride prefetcher",
        "author": "Ada",
        "date": "2026-10-01T10:00:00Z",
    }
    assert row.llm_summary == model.reply
    prompt = model.prompts()[0]
    assert "Repository: acme/prefetch" in prompt
    assert "- src/prefetch/stride.c (added, +120/-0)" in prompt
    assert "Commits ahead: 2, behind: 1" in prompt
    assert edges.cleared == [str(diff.id)]


async def test_branch_comparison_without_explanation_calls_no_model(
    db_session, task_sessions, edges, model, web
):
    _github_routes(web)
    source = await _github_source(db_session)
    diff = await _diff(db_session, source, options={"explain": False})

    await git_compare_tasks._async_compare_git_branches(FakeTask(), str(diff.id))

    row = await _reload(db_session, GitBranchDiff, diff.id)
    assert row.status == "completed"
    assert row.diff_summary["stats"]["total_files"] == 3
    assert row.llm_summary is None
    assert model.calls == []


async def test_branch_comparison_survives_a_failing_model(
    db_session, task_sessions, edges, model, web
):
    _github_routes(web)
    model.fail_with = RuntimeError("deepseek: 401 invalid api key")
    source = await _github_source(db_session)
    diff = await _diff(db_session, source)

    await git_compare_tasks._async_compare_git_branches(FakeTask(), str(diff.id))

    row = await _reload(db_session, GitBranchDiff, diff.id)
    assert row.status == "completed"
    assert row.diff_summary["stats"]["total_files"] == 3
    assert not row.llm_summary


async def test_missing_comparison_job_raises(db_session, task_sessions, edges, web):
    with pytest.raises(ValueError, match="Comparison job not found"):
        await git_compare_tasks._async_compare_git_branches(FakeTask(), str(uuid4()))
    assert web.requests == []


async def test_comparison_whose_source_is_gone_fails_naming_it(
    db_session, task_sessions, edges, web
):
    # SQLite does not enforce the foreign key, which stands in for a source
    # deleted between the request and the worker picking the job up.
    diff = await _diff(db_session, SimpleNamespace(id=uuid4()))

    with pytest.raises(ValueError, match="Document source not found"):
        await git_compare_tasks._async_compare_git_branches(FakeTask(), str(diff.id))

    row = await _reload(db_session, GitBranchDiff, diff.id)
    assert row.status == "failed"
    assert row.error == "Document source not found"


async def test_comparison_the_api_refuses_fails_with_its_status(
    db_session, task_sessions, edges, model, web
):
    _github_routes(web, compare={"message": "Not Found"}, compare_status=404)
    source = await _github_source(db_session)
    diff = await _diff(db_session, source)

    with pytest.raises(ValueError, match="GitHub compare API failed: 404"):
        await git_compare_tasks._async_compare_git_branches(FakeTask(), str(diff.id))

    row = await _reload(db_session, GitBranchDiff, diff.id)
    assert row.status == "failed"
    assert "GitHub compare API failed: 404" in row.error
    assert row.completed_at is not None
    assert row.diff_summary is None and model.calls == []
    assert edges.cleared == [str(diff.id)]


async def test_cancelled_comparison_fetches_nothing(
    db_session, task_sessions, edges, model, web
):
    _github_routes(web)
    source = await _github_source(db_session)
    diff = await _diff(db_session, source)
    edges.cancelled.add(str(diff.id))

    result = await git_compare_tasks._async_compare_git_branches(
        FakeTask(), str(diff.id)
    )

    assert result == {"canceled": True}
    row = await _reload(db_session, GitBranchDiff, diff.id)
    assert row.status == "canceled"
    assert row.error == "Canceled before start"
    assert web.requests == [] and model.calls == []


async def test_comparison_with_a_rejected_token_names_the_rejection(
    db_session, task_sessions, edges, model, web
):
    web.route(f"{GITHUB}/user", {"message": "Bad credentials"}, status=401)
    _github_routes(web)
    source = await _github_source(db_session, token="ghp_revoked")
    diff = await _diff(db_session, source)

    with pytest.raises(Exception):
        await git_compare_tasks._async_compare_git_branches(FakeTask(), str(diff.id))

    row = await _reload(db_session, GitBranchDiff, diff.id)
    assert row.status == "failed"
    assert "401" in row.error


GITLAB = "https://gitlab.example.com"
GITLAB_COMPARE = {
    "commits": [
        {
            "title": "Add stride prefetcher",
            "author_name": "Ada",
            "committed_date": "2026-10-01T10:00:00Z",
        }
    ],
    "diffs": [
        {
            "old_path": "src/stride.c",
            "new_path": "src/stride.c",
            "new_file": False,
            "renamed_file": False,
            "deleted_file": False,
            "diff": "@@ -1,2 +1,4 @@\n-int degree = 1;\n+int degree = 4;\n"
            "+int distance = 8;\n+int enabled = 1;\n context\n",
        }
    ],
}


async def _gitlab_comparison(db, web, repository="42"):
    web.route(f"{GITLAB}/api/v4/user", {"id": 1})
    web.route(f"{GITLAB}/api/v4/projects/42/repository/compare", GITLAB_COMPARE)
    source = await _source(
        db,
        "gitlab",
        {
            "gitlab_url": GITLAB,
            "token": "glpat-test",
            "projects": [{"id": 42, "name": "prefetch"}],
        },
    )
    return await _diff(db, source, repository=repository, options={"explain": False})


async def test_gitlab_comparison_completes(db_session, task_sessions, edges, web):
    diff = await _gitlab_comparison(db_session, web)

    await git_compare_tasks._async_compare_git_branches(FakeTask(), str(diff.id))

    row = await _reload(db_session, GitBranchDiff, diff.id)
    assert row.status == "completed", row.error
    (compare,) = [r for r in web.requests if r.url.path.endswith("/compare")]
    assert dict(compare.url.params) == {"from": "main", "to": "stride"}
    assert row.diff_summary["stats"]["total_commits"] == 1
    (entry,) = row.diff_summary["files"]
    assert entry["filename"] == "src/stride.c"
    assert entry["status"] == "modified"
    assert row.diff_summary["raw"]["commit_messages"][0]["message"] == (
        "Add stride prefetcher"
    )


async def test_gitlab_comparison_finds_a_project_by_its_name(
    db_session, task_sessions, edges, web
):
    diff = await _gitlab_comparison(db_session, web, repository="prefetch")

    await git_compare_tasks._async_compare_git_branches(FakeTask(), str(diff.id))

    row = await _reload(db_session, GitBranchDiff, diff.id)
    assert row.status == "completed", row.error


async def test_gitlab_comparison_counts_changed_lines(
    db_session, task_sessions, edges, web
):
    diff = await _gitlab_comparison(db_session, web)

    await git_compare_tasks._async_compare_git_branches(FakeTask(), str(diff.id))

    row = await _reload(db_session, GitBranchDiff, diff.id)
    (entry,) = row.diff_summary["files"]
    assert (entry["additions"], entry["deletions"]) == (3, 1)


# ==========================================================================
# enrich_arxiv_document
# ==========================================================================

DOI = "10.1145/3352460.3358252"
BIBTEX = (
    "@misc{author2026isb,\n"
    "      title={Irregular Stream Buffer Revisited},\n"
    "      author={A. Author and B. Author},\n"
    "      year={2026},\n"
    "      eprint={2601.00001},\n"
    "      archivePrefix={arXiv},\n"
    "}"
)
CROSSREF = {
    "message": {
        "publisher": "ACM",
        "container-title": ["MICRO '26"],
        "subject": ["Computer architecture", "cs.AR"],
        "issued": {"date-parts": [[2026, 10]]},
        "author": [
            {
                "given": "Ada",
                "family": "Author",
                "ORCID": "https://orcid.org/0000-0001",
                "affiliation": [{"name": "Uni A"}, {"name": "uni a"}],
            }
        ],
    }
}


def _scholarly_routes(web, arxiv_id="2601.00001v1"):
    # What arxiv.org actually serves: plain-text BibTeX (checked 2026-10-05:
    # `content-type: text/plain`, the entry itself, no HTML around it).
    web.route(
        f"https://arxiv.org/bibtex/{arxiv_id}",
        BIBTEX,
        content_type="text/plain; charset=utf-8",
    )
    web.route(f"https://api.crossref.org/works/{DOI}", CROSSREF)


async def _arxiv_source(db, **config):
    return await _source(db, "arxiv", {"queries": ["all:prefetching"], **config})


async def test_enrichment_fills_venue_year_keywords_and_affiliations(
    db_session, task_sessions, web
):
    _scholarly_routes(web)
    source = await _arxiv_source(db_session)
    doc = await _arxiv_paper(
        db_session,
        source,
        tags=["prefetching"],
        extra_metadata={"categories": ["cs.AR", "cs.PF"], "doi": DOI},
    )

    result = await paper_enrichment_tasks._async_enrich_document(
        FakeTask(), str(doc.id), False
    )

    assert result == {
        "skipped": False,
        "document_id": str(doc.id),
        "arxiv_id": "2601.00001v1",
        "doi": DOI,
    }
    row = await _reload(db_session, Document, doc.id)
    meta = row.extra_metadata["paper_metadata"]
    assert meta["venue"] == "MICRO '26"
    assert meta["publisher"] == "ACM"
    assert meta["year"] == 2026
    assert meta["doi"] == DOI and meta["arxiv_id"] == "2601.00001v1"
    assert meta["keywords"] == ["Computer architecture", "cs.AR", "cs.PF"]
    assert meta["author_affiliations"] == [
        {
            "name": "Ada Author",
            "orcid": "https://orcid.org/0000-0001",
            "affiliations": ["Uni A"],
        }
    ]
    assert meta["enriched_at"]
    assert row.extra_metadata["categories"] == ["cs.AR", "cs.PF"]  # kept
    assert row.tags == ["prefetching", "Computer architecture", "cs.AR", "cs.PF"]
    assert {r.url.host for r in web.requests} == {"arxiv.org", "api.crossref.org"}


async def test_enrichment_records_the_bibtex_arxiv_serves(
    db_session, task_sessions, web
):
    _scholarly_routes(web)
    source = await _arxiv_source(db_session)
    doc = await _arxiv_paper(db_session, source)

    await paper_enrichment_tasks._async_enrich_document(FakeTask(), str(doc.id), False)

    row = await _reload(db_session, Document, doc.id)
    assert row.extra_metadata["paper_metadata"]["bibtex"] == BIBTEX


async def test_enrichment_is_not_repeated_unless_forced(db_session, task_sessions, web):
    _scholarly_routes(web)
    source = await _arxiv_source(db_session)
    doc = await _arxiv_paper(
        db_session, source, extra_metadata={"categories": ["cs.AR"], "doi": DOI}
    )
    await paper_enrichment_tasks._async_enrich_document(FakeTask(), str(doc.id), False)
    first = len(web.requests)

    again = await paper_enrichment_tasks._async_enrich_document(
        FakeTask(), str(doc.id), False
    )
    forced = await paper_enrichment_tasks._async_enrich_document(
        FakeTask(), str(doc.id), True
    )

    assert again == {"skipped": True, "reason": "already_enriched"}
    assert forced["skipped"] is False
    assert len(web.requests) == first + 2  # only the forced run fetched


async def test_enrichment_skips_a_document_that_is_not_from_arxiv(
    db_session, task_sessions, web
):
    source = await _source(db_session, "file")
    doc = await _document(db_session, source)

    result = await paper_enrichment_tasks._async_enrich_document(
        FakeTask(), str(doc.id), False
    )

    assert result == {"skipped": True, "reason": "not_arxiv"}
    assert web.requests == []


async def test_enrichment_of_a_missing_document(db_session, task_sessions, web):
    missing = str(uuid4())

    result = await paper_enrichment_tasks._async_enrich_document(
        FakeTask(), missing, False
    )

    assert result == {
        "success": False,
        "document_id": missing,
        "error": "Document not found",
    }


async def test_enrichment_during_an_outage_is_retried_later(
    db_session, task_sessions, web
):
    web.route("https://arxiv.org/bibtex/2601.00001v1", "busy", status=503)
    web.route(f"https://api.crossref.org/works/{DOI}", "busy", status=503)
    source = await _arxiv_source(db_session)
    doc = await _arxiv_paper(
        db_session, source, extra_metadata={"categories": ["cs.AR"], "doi": DOI}
    )
    await paper_enrichment_tasks._async_enrich_document(FakeTask(), str(doc.id), False)

    _scholarly_routes(web)  # the outage is over
    result = await paper_enrichment_tasks._async_enrich_document(
        FakeTask(), str(doc.id), False
    )

    assert result.get("reason") != "already_enriched"
    row = await _reload(db_session, Document, doc.id)
    assert row.extra_metadata["paper_metadata"]["venue"] == "MICRO '26"


# ==========================================================================
# upsert_paper_insights_to_kg
# ==========================================================================

INSIGHTS = {
    "key_claims": ["ISB covers 40% more misses than stride"],
    "methods": ["Irregular stream buffer", "stride prefetching", "Stride Prefetching"],
    "datasets": ["SPEC CPU2017"],
    "tasks": ["cache miss reduction"],
}


async def _kg_rows(db, document_id):
    mentions = (
        (
            await db.execute(
                select(EntityMention).where(EntityMention.document_id == document_id)
            )
        )
        .scalars()
        .all()
    )
    rels = (
        (
            await db.execute(
                select(Relationship).where(Relationship.document_id == document_id)
            )
        )
        .scalars()
        .all()
    )
    entities = {e.id: e for e in (await db.execute(select(Entity))).scalars().all()}
    edges_ = sorted(
        (
            r.relation_type,
            entities[r.source_entity_id].canonical_name,
            entities[r.target_entity_id].canonical_name,
        )
        for r in rels
    )
    return mentions, rels, entities, edges_


async def test_paper_insights_become_entities_mentions_and_relationships(
    db_session, task_sessions
):
    source = await _arxiv_source(db_session)
    doc = await _arxiv_paper(
        db_session, source, extra_metadata={"paper_insights": INSIGHTS}
    )

    result = await paper_kg_tasks._async_upsert(FakeTask(), str(doc.id), False)

    assert result["skipped"] is False, result
    assert result["entities_created"] == 4  # the duplicate method folds
    assert result["mentions_created"] == 4
    assert result["relationships_created"] == 4
    mentions, rels, entities, edges_ = await _kg_rows(db_session, doc.id)
    assert edges_ == [
        ("evaluated_on", "2601.00001v1", "SPEC CPU2017"),
        ("targets_task", "2601.00001v1", "cache miss reduction"),
        ("uses_method", "2601.00001v1", "Irregular stream buffer"),
        ("uses_method", "2601.00001v1", "stride prefetching"),
    ]
    paper = entities[rels[0].source_entity_id]
    assert paper.entity_type == "paper"
    assert paper.description == "Irregular Stream Buffer Revisited"
    assert str(paper.id) == result["paper_entity_id"]
    assert json.loads(paper.properties)["document_id"] == str(doc.id)
    assert {m.sentence for m in mentions} == {"paper_insights"}
    assert all(r.inferred and r.evidence == "paper_insights" for r in rels)


async def test_paper_insights_rerun_adds_nothing_twice(db_session, task_sessions):
    source = await _arxiv_source(db_session)
    doc = await _arxiv_paper(
        db_session, source, extra_metadata={"paper_insights": INSIGHTS}
    )
    await paper_kg_tasks._async_upsert(FakeTask(), str(doc.id), False)

    again = await paper_kg_tasks._async_upsert(FakeTask(), str(doc.id), False)

    assert again["entities_created"] == 0
    assert again["mentions_created"] == 0
    assert again["relationships_created"] == 0
    mentions, rels, _, _ = await _kg_rows(db_session, doc.id)
    assert len(mentions) == 4 and len(rels) == 4


async def test_forced_upsert_rebuilds_from_the_current_insights(
    db_session, task_sessions
):
    source = await _arxiv_source(db_session)
    doc = await _arxiv_paper(
        db_session, source, extra_metadata={"paper_insights": INSIGHTS}
    )
    await paper_kg_tasks._async_upsert(FakeTask(), str(doc.id), False)

    row = await _reload(db_session, Document, doc.id)
    row.extra_metadata = {"paper_insights": {"methods": ["Irregular stream buffer"]}}
    await db_session.commit()
    result = await paper_kg_tasks._async_upsert(FakeTask(), str(doc.id), True)

    assert result["relationships_created"] == 1
    mentions, rels, _, edges_ = await _kg_rows(db_session, doc.id)
    assert edges_ == [("uses_method", "2601.00001v1", "Irregular stream buffer")]
    assert [m.text for m in mentions] == ["Irregular stream buffer"]


async def test_two_papers_share_a_method_entity(db_session, task_sessions):
    source = await _arxiv_source(db_session)
    first = await _arxiv_paper(
        db_session, source, extra_metadata={"paper_insights": INSIGHTS}
    )
    second = await _arxiv_paper(
        db_session,
        source,
        "2601.00002v1",
        title="Stride Prefetching at Scale",
        extra_metadata={"paper_insights": {"methods": ["stride prefetching"]}},
    )

    await paper_kg_tasks._async_upsert(FakeTask(), str(first.id), False)
    result = await paper_kg_tasks._async_upsert(FakeTask(), str(second.id), False)

    assert result["entities_created"] == 0  # the method already exists
    assert result["relationships_created"] == 1
    assert await _count(db_session, Entity, Entity.entity_type == "method") == 2
    assert await _count(db_session, Entity, Entity.entity_type == "paper") == 2


@pytest.mark.parametrize(
    "source_type,metadata,reason",
    [
        ("arxiv", {"categories": ["cs.AR"]}, "no_paper_insights"),
        ("arxiv", {"paper_insights": ["not", "a", "dict"]}, "no_paper_insights"),
        ("file", {"paper_insights": INSIGHTS}, "not_arxiv"),
    ],
)
async def test_paper_insights_upsert_skips_what_it_cannot_use(
    db_session, task_sessions, source_type, metadata, reason
):
    source = await _source(db_session, source_type, {"queries": ["x"]})
    doc = await _arxiv_paper(db_session, source, extra_metadata=metadata)

    result = await paper_kg_tasks._async_upsert(FakeTask(), str(doc.id), False)

    assert result == {"skipped": True, "reason": reason}
    assert await _count(db_session, Entity) == 0


async def test_paper_insights_upsert_of_a_missing_document(db_session, task_sessions):
    missing = str(uuid4())

    result = await paper_kg_tasks._async_upsert(FakeTask(), missing, False)

    assert result == {
        "success": False,
        "document_id": missing,
        "error": "Document not found",
    }


# ==========================================================================
# generate_literature_review
# ==========================================================================

REVIEW_MD = (
    "Prefetching research splits into stride and correlation designs.\n\n"
    "| Paper | Key claims | Methods | Limitations | Link |\n"
    "|---|---|---|---|---|\n"
    "| ISB (2601.00001v1) | 40% more coverage | ISB | area | arxiv |\n"
)


async def _papers_source(db):
    source = await _arxiv_source(db, topic="hardware prefetching")
    first = await _arxiv_paper(
        db,
        source,
        summary="ISB linearises irregular streams.",
        extra_metadata={"paper_insights": {"methods": ["ISB"]}},
    )
    second = await _arxiv_paper(
        db, source, "2601.00002v1", title="Stride Prefetching at Scale"
    )
    return source, first, second


async def test_literature_review_is_written_stored_and_queued_for_indexing(
    db_session, test_user, task_sessions, model, ingest_queue
):
    model.reply = f"  {REVIEW_MD}  "
    source, first, second = await _papers_source(db_session)

    result = await research_tasks._async_generate_literature_review(
        FakeTask(), str(source.id), str(test_user.id)
    )

    assert result["success"] is True, result
    assert result["title"] == "Literature Review: hardware prefetching"
    review = await _reload(db_session, Document, UUID(result["document_id"]))
    assert review.content == (
        f"# Literature Review: hardware prefetching\n\n{REVIEW_MD.strip()}\n"
    )
    assert review.content_hash == hashlib.sha256(review.content.encode()).hexdigest()
    assert review.source_id == source.id
    assert review.source_identifier == f"literature_review:{source.id}"
    assert review.extra_metadata == {
        "report_type": "literature_review",
        "source_id": str(source.id),
        "topic": "hardware prefetching",
        "papers_count": 2,
    }
    assert review.is_processed is False
    assert ingest_queue.calls == [{"document_id": result["document_id"]}]
    (call,) = model.calls
    assert call["task_type"] == "summarization"
    papers = json.loads(call["query"].split("Papers JSON:\n", 1)[1])
    assert sorted(p["arxiv_id"] for p in papers) == ["2601.00001v1", "2601.00002v1"]
    by_id = {p["arxiv_id"]: p for p in papers}
    assert by_id["2601.00001v1"]["summary"] == "ISB linearises irregular streams."
    assert by_id["2601.00001v1"]["insights"] == {"methods": ["ISB"]}
    assert "Topic: hardware prefetching" in call["query"]


@pytest.mark.parametrize(
    "setup,error",
    [
        ("missing", "Source not found"),
        ("not_arxiv", "Literature review only supported for arXiv sources"),
    ],
)
async def test_literature_review_refuses_an_unusable_source(
    db_session, task_sessions, model, ingest_queue, setup, error
):
    if setup == "missing":
        source_id = str(uuid4())
    else:
        source_id = str((await _source(db_session, "web")).id)

    result = await research_tasks._async_generate_literature_review(
        FakeTask(), source_id
    )

    assert result == {"success": False, "source_id": source_id, "error": error}
    assert model.calls == [] and ingest_queue.calls == []


async def test_literature_review_of_an_empty_source(
    db_session, task_sessions, model, ingest_queue
):
    source = await _arxiv_source(db_session)

    result = await research_tasks._async_generate_literature_review(
        FakeTask(), str(source.id)
    )

    assert result["error"] == "No documents found for source"
    assert model.calls == []


async def test_literature_review_whose_model_fails_stores_nothing(
    db_session, task_sessions, model, ingest_queue
):
    model.fail_with = RuntimeError("deepseek: 401 invalid api key")
    source, _, _ = await _papers_source(db_session)

    result = await research_tasks._async_generate_literature_review(
        FakeTask(), str(source.id)
    )

    assert result == {
        "success": False,
        "source_id": str(source.id),
        "error": "deepseek: 401 invalid api key",
    }
    assert await _count(db_session, Document) == 2
    assert ingest_queue.calls == []


async def test_a_second_literature_review_reviews_only_the_papers(
    db_session, task_sessions, model, ingest_queue
):
    model.reply = REVIEW_MD
    source, _, _ = await _papers_source(db_session)
    await research_tasks._async_generate_literature_review(FakeTask(), str(source.id))

    second = await research_tasks._async_generate_literature_review(
        FakeTask(), str(source.id)
    )

    papers = json.loads(model.prompts()[-1].split("Papers JSON:\n", 1)[1])
    assert sorted(p["arxiv_id"] for p in papers) == ["2601.00001v1", "2601.00002v1"]
    review = await _reload(db_session, Document, UUID(second["document_id"]))
    assert review.extra_metadata["papers_count"] == 2


# --------------------------------------------------------------------------
# The synchronous Celery functions, each in its own event loop
# --------------------------------------------------------------------------
#
# The two processing tasks call ``self.update_state``, which needs the request
# id a worker gives every task; ``.apply`` runs them eagerly with one, where
# ``.run`` has none and fails before doing anything.


@pytest.fixture
def file_db(tmp_path, monkeypatch):
    """A file database reachable from any event loop, as Postgres is."""
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
            username="syncuser",
            email="sync@example.com",
            password="testpassword123",
            full_name="Sync User",
            db=db,
        )

    return run(make)


def _get(run, model_cls, row_id):
    async def fetch(db):
        return (
            await db.execute(select(model_cls).where(model_cls.id == row_id))
        ).scalar_one_or_none()

    return run(fetch)


def test_process_uploaded_document_run_indexes_a_document(file_db, edges):
    async def seed(db):
        return await _document(db, await _source(db))

    doc = file_db(seed)

    result = ingestion_tasks.process_uploaded_document.apply(args=[str(doc.id)]).get()

    assert result["success"] is True
    row = _get(file_db, Document, doc.id)
    assert row.is_processed is True and row.processing_error is None
    assert len(file_db(lambda db: _chunks(db, doc.id))) >= 2


def test_process_document_run_indexes_a_document(file_db, edges):
    async def seed(db):
        return await _document(db, await _source(db))

    doc = file_db(seed)

    result = processing_tasks.process_document.apply(args=[str(doc.id)]).get()

    row = _get(file_db, Document, doc.id)
    assert row.processing_error is None and row.is_processed is True
    assert result["success"] is True


def test_dry_run_source_run_lists_a_source(file_db, edges, web):
    web.route(ARXIV_API, TWO_PAPERS, content_type="application/atom+xml")

    async def seed(db):
        return await _arxiv_source(db)

    source = file_db(seed)

    result = ingestion_tasks.dry_run_source.run(str(source.id))

    assert result["success"] is True and result["total"] == 2
    assert result["estimated_new"] == 2


def test_generate_chat_title_run_sets_the_title(file_db, model):
    model.reply = "Prefetcher Engagement"
    user = _seed_user(file_db)
    session = file_db(lambda db: _chat(db, user))

    result = chat_tasks.generate_chat_title.run(str(session.id))

    row = _get(file_db, ChatSession, session.id)
    assert result["success"] is True
    assert row.title.endswith(" - Prefetcher Engagement")


def test_compare_git_branches_run_completes(file_db, edges, model, web):
    _github_routes(web)
    model.reply = "Adds a stride prefetcher."
    user = _seed_user(file_db)

    async def seed(db):
        return await _diff(db, await _github_source(db))

    diff = file_db(seed)

    result = git_compare_tasks.compare_git_branches.run(str(diff.id), str(user.id))

    row = _get(file_db, GitBranchDiff, diff.id)
    assert result == {"success": True}
    assert row.status == "completed" and row.llm_summary == "Adds a stride prefetcher."


def test_enrich_arxiv_document_run_enriches(file_db, web):
    _scholarly_routes(web)

    async def seed(db):
        source = await _arxiv_source(db)
        return await _arxiv_paper(db, source, extra_metadata={"doi": DOI})

    doc = file_db(seed)

    result = paper_enrichment_tasks.enrich_arxiv_document.run(str(doc.id))

    assert result["skipped"] is False
    row = _get(file_db, Document, doc.id)
    assert row.extra_metadata["paper_metadata"]["venue"] == "MICRO '26"


def test_upsert_paper_insights_to_kg_run_writes_the_graph(file_db):
    async def seed(db):
        source = await _arxiv_source(db)
        return await _arxiv_paper(
            db, source, extra_metadata={"paper_insights": INSIGHTS}
        )

    doc = file_db(seed)

    result = paper_kg_tasks.upsert_paper_insights_to_kg.run(str(doc.id), force=True)

    assert result["relationships_created"] == 4
    rels = file_db(lambda db: _count(db, Relationship))
    assert rels == 4


def test_generate_literature_review_run_stores_a_review(file_db, model, ingest_queue):
    model.reply = REVIEW_MD
    user = _seed_user(file_db)
    source = file_db(lambda db: _papers_source(db))[0]

    result = research_tasks.generate_literature_review.run(
        str(source.id), user_id=str(user.id)
    )

    assert result["success"] is True, result
    review = _get(file_db, Document, UUID(result["document_id"]))
    assert review.title == "Literature Review: hardware prefetching"
    assert [c["document_id"] for c in ingest_queue.calls] == [result["document_id"]]


def test_tasks_are_registered_with_the_worker():
    from app.core.celery import celery_app

    include = set(celery_app.conf.include or ())
    for module in (
        chat_tasks,
        git_compare_tasks,
        ingestion_tasks,
        paper_enrichment_tasks,
        paper_kg_tasks,
        processing_tasks,
        research_tasks,
    ):
        assert module.__name__ in include, f"{module.__name__} is not in include"
