"""The MCP search, chat and generation tools, called as the server calls them.

None of the three had a test. Real rows; the edges replaced are the vector
store's search (bound against its real signature, and filtering by document as
the store does), the model, and Celery's `.delay`.

What calling them found:

* search loaded the id of every document, capped at 10,000, and dropped any
  result outside that list: past the cap a document could not be found;
* search and chat applied a source filter to a window of the best chunks from
  anywhere, so a source's matches were crowded out by other sources';
* chat treated a filter matching no document as no filter;
* a failed presentation's status read `error_message`, which the model lacks;
* limits and slide counts were whatever number arrived.
"""

import hashlib
import inspect
from uuid import UUID, uuid4

import pytest

from app.mcp.auth import MCPAuthContext
from app.mcp.tools import chat as chat_module
from app.mcp.tools import search as search_module
from app.mcp.tools.chat import ChatTool
from app.mcp.tools.generation import GenerationTool
from app.mcp.tools.search import SearchTool
from app.models.document import Document, DocumentSource
from app.models.presentation import PresentationJob
from app.models.repo_report import RepoReportJob
from app.services.vector_store import VectorStoreService
from app.tasks.presentation_tasks import generate_presentation_task
from app.tasks.repo_report_tasks import generate_repo_report_task

pytestmark = pytest.mark.unit

_SEARCH = inspect.signature(VectorStoreService.search)


class _Key:
    def __init__(self, scopes):
        self.scopes = scopes

    def has_scope(self, scope):
        return scope in self.scopes


def _auth(user, *scopes):
    scopes = list(scopes or ("read", "write", "chat"))
    return MCPAuthContext(user=user, api_key=_Key(scopes), scopes=scopes)


async def _source_with_documents(db, name, titles, file_type="md"):
    source = DocumentSource(
        name=f"{name}-{uuid4().hex[:6]}", source_type="file", config={}
    )
    db.add(source)
    await db.commit()
    await db.refresh(source)
    docs = []
    for title in titles:
        doc = Document(
            title=title,
            content=f"{title} body",
            content_hash=hashlib.sha256(f"{name}{title}".encode()).hexdigest(),
            source_id=source.id,
            source_identifier=f"test:{uuid4().hex}",
            file_type=file_type,
        )
        db.add(doc)
        docs.append(doc)
    await db.commit()
    return source, docs


class _Store:
    """The vector store: best chunks first, filtered by document when asked."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    async def initialize(self, embedding_model=None, background=False):
        return None

    async def search(self, *args, **kwargs):
        bound = _SEARCH.bind(self, *args, **kwargs)
        bound.apply_defaults()
        call = dict(bound.arguments)
        self.calls.append(call)
        wanted = call["document_ids"]
        hits = [
            c
            for c in self.chunks
            if wanted is None or c["metadata"]["document_id"] in set(wanted)
        ]
        return hits[: call["limit"]]


def _chunk(doc, score, file_type="md"):
    return {
        "content": f"{doc.title} body",
        "score": score,
        "metadata": {
            "document_id": str(doc.id),
            "title": doc.title,
            "file_type": file_type,
            "chunk_index": 0,
        },
    }


@pytest.fixture
def store(monkeypatch):
    def install(chunks):
        fake = _Store(chunks)
        monkeypatch.setattr(search_module, "vector_store_service", fake)
        monkeypatch.setattr(chat_module, "vector_store_service", fake)
        return fake

    return install


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------


async def test_search_does_not_filter_when_no_source_is_named(
    db_session, test_user, store
):
    _, docs = await _source_with_documents(db_session, "a", ["one", "two"])
    fake = store([_chunk(docs[0], 0.9), _chunk(docs[1], 0.8)])

    result = await SearchTool().execute(_auth(test_user), db_session, query="body")

    assert [r["title"] for r in result["results"]] == ["one", "two"]
    # No list of every document's id is built, so none can be cut short.
    assert fake.calls[0]["document_ids"] is None


async def test_search_finds_a_document_past_the_old_cap(
    db_session, test_user, store, monkeypatch
):
    """With the cap at 1, the second document used to be unfindable."""
    monkeypatch.setattr(search_module, "MAX_SOURCE_DOCUMENTS", 1)
    _, docs = await _source_with_documents(db_session, "a", ["one", "two"])
    store([_chunk(docs[1], 0.9)])

    result = await SearchTool().execute(_auth(test_user), db_session, query="two")

    assert [r["title"] for r in result["results"]] == ["two"]


async def test_search_in_a_source_is_not_crowded_out_by_other_sources(
    db_session, test_user, store
):
    _, loud = await _source_with_documents(
        db_session, "loud", [f"l{i}" for i in range(6)]
    )
    quiet_source, quiet = await _source_with_documents(db_session, "quiet", ["wanted"])
    store([_chunk(d, 0.9) for d in loud] + [_chunk(quiet[0], 0.2)])

    result = await SearchTool().execute(
        _auth(test_user),
        db_session,
        query="body",
        limit=2,
        source_ids=[str(quiet_source.id)],
    )

    assert [r["title"] for r in result["results"]] == ["wanted"]


async def test_search_refuses_a_source_id_that_is_not_an_id(
    db_session, test_user, store
):
    fake = store([])

    result = await SearchTool().execute(
        _auth(test_user), db_session, query="x", source_ids=["not-a-uuid"]
    )

    assert result["error"] == "source_ids must be UUIDs"
    assert fake.calls == []


async def test_search_says_when_the_named_source_is_empty(db_session, test_user, store):
    source, _ = await _source_with_documents(db_session, "empty", [])
    fake = store([])

    result = await SearchTool().execute(
        _auth(test_user), db_session, query="x", source_ids=[str(source.id)]
    )

    assert result["results"] == []
    assert "no documents" in result["message"]
    assert fake.calls == []


@pytest.mark.parametrize("limit, expected", [(-1, 1), (0, 1), (10**6, 50), ("x", 10)])
async def test_search_bounds_the_limit(db_session, test_user, store, limit, expected):
    fake = store([])

    await SearchTool().execute(_auth(test_user), db_session, query="x", limit=limit)

    assert fake.calls[0]["limit"] == expected


async def test_search_filters_by_file_type(db_session, test_user, store):
    _, docs = await _source_with_documents(db_session, "a", ["note", "paper"])
    store([_chunk(docs[0], 0.9, "md"), _chunk(docs[1], 0.8, "pdf")])

    result = await SearchTool().execute(
        _auth(test_user), db_session, query="x", file_types=["PDF"]
    )

    assert [r["title"] for r in result["results"]] == ["paper"]


# --------------------------------------------------------------------------
# chat
# --------------------------------------------------------------------------


@pytest.fixture
def model(monkeypatch):
    prompts = []

    async def generate_response(self, query=None, **kwargs):
        prompts.append(query)
        return " An answer. "

    monkeypatch.setattr(chat_module.LLMService, "generate_response", generate_response)
    return prompts


async def test_chat_answers_from_the_named_source_only(
    db_session, test_user, store, model
):
    _, loud = await _source_with_documents(
        db_session, "loud", [f"l{i}" for i in range(12)]
    )
    quiet_source, quiet = await _source_with_documents(db_session, "quiet", ["wanted"])
    store([_chunk(d, 0.9) for d in loud] + [_chunk(quiet[0], 0.2)])

    result = await ChatTool().execute(
        _auth(test_user),
        db_session,
        question="what?",
        source_ids=[str(quiet_source.id)],
    )

    assert result["context_found"] is True
    assert result["answer"] == "An answer."
    assert [s["title"] for s in result["sources"]] == ["wanted"]
    assert "wanted body" in model[0]
    assert "l0 body" not in model[0]


async def test_chat_does_not_treat_an_empty_filter_as_no_filter(
    db_session, test_user, store, model
):
    """A source with no documents used to mean "search everything"."""
    _, docs = await _source_with_documents(db_session, "other", ["secret"])
    empty, _ = await _source_with_documents(db_session, "empty", [])
    fake = store([_chunk(docs[0], 0.9)])

    result = await ChatTool().execute(
        _auth(test_user), db_session, question="what?", source_ids=[str(empty.id)]
    )

    assert result["context_found"] is False
    assert fake.calls == []
    assert model == []


@pytest.mark.parametrize("chunks, expected", [(-3, 1), (0, 1), (500, 20)])
async def test_chat_bounds_the_context_size(
    db_session, test_user, store, model, chunks, expected
):
    fake = store([])

    await ChatTool().execute(
        _auth(test_user), db_session, question="what?", max_context_chunks=chunks
    )

    assert fake.calls[0]["limit"] == expected


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


@pytest.fixture
def queued(monkeypatch):
    sent = []

    def recorder(task):
        signature = inspect.signature(task.run)

        def delay(*args, **kwargs):
            signature.bind(*args, **kwargs)
            sent.append((task.name, args))

        return delay

    for task in (generate_presentation_task, generate_repo_report_task):
        monkeypatch.setattr(task, "delay", recorder(task))
    return sent


async def test_create_presentation_stores_a_job_and_queues_it(
    db_session, test_user, queued
):
    result = await GenerationTool().create_presentation(
        _auth(test_user), db_session, topic="Caches", slide_count=10**6
    )

    job = await db_session.get(PresentationJob, UUID(result["job_id"]))
    assert job.user_id == test_user.id
    assert job.slide_count == 30
    assert len(queued) == 1
    assert queued[0][1] == (result["job_id"], str(test_user.id))


async def test_create_presentation_refuses_a_style_nothing_builds(
    db_session, test_user, queued
):
    result = await GenerationTool().create_presentation(
        _auth(test_user), db_session, topic="Caches", style="neon"
    )

    assert "Unknown style" in result["error"]
    assert queued == []


async def test_create_repo_report_stores_a_job_and_queues_it(
    db_session, test_user, queued
):
    result = await GenerationTool().create_repo_report(
        _auth(test_user), db_session, repo_url="https://github.com/acme/tool.git"
    )

    job = await db_session.get(RepoReportJob, UUID(result["job_id"]))
    assert job.repo_name == "acme/tool"
    assert job.repo_type == "github"
    assert queued[0][1] == (result["job_id"], str(test_user.id))


async def test_create_repo_report_refuses_a_format_nothing_builds(
    db_session, test_user, queued
):
    result = await GenerationTool().create_repo_report(
        _auth(test_user),
        db_session,
        repo_url="https://github.com/acme/tool",
        output_format="odt",
    )

    assert "Unknown output_format" in result["error"]
    assert queued == []


async def _presentation(db, user, **fields):
    job = PresentationJob(
        user_id=user.id,
        title="Deck",
        topic="Caches",
        source_document_ids=[],
        slide_count=5,
        style="professional",
        include_diagrams=0,
        status="pending",
        progress=0,
    )
    for key, value in fields.items():
        setattr(job, key, value)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def test_a_failed_presentation_reports_its_own_error(db_session, test_user):
    job = await _presentation(
        db_session, test_user, status="failed", error="model unreachable"
    )

    result = await GenerationTool().get_job_status(
        _auth(test_user), db_session, job_id=str(job.id), job_type="presentation"
    )

    assert result["status"] == "failed"
    assert result["error"] == "model unreachable"


async def test_job_status_is_the_owners_only(db_session, test_user, admin_user):
    job = await _presentation(db_session, admin_user)

    result = await GenerationTool().get_job_status(
        _auth(test_user), db_session, job_id=str(job.id), job_type="presentation"
    )

    assert result["error"] == "Job not found"


@pytest.mark.parametrize("limit", [-1, 0])
async def test_list_jobs_never_runs_without_a_limit(db_session, test_user, limit):
    for _ in range(3):
        await _presentation(db_session, test_user)

    result = await GenerationTool().list_jobs(
        _auth(test_user), db_session, job_type="presentation", limit=limit
    )

    assert "error" not in result
    assert len(result["jobs"]) == 1
