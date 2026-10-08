"""The chat tools that destroy or create documents, called for real.

delete_document, batch_delete_documents, create_document_from_text,
update_document_tags, ingest_url, web_scrape and batch_summarize_documents,
plus `POST /agent/confirm-delete/{id}`. None of them had a test.

Every test calls the real `AgentService._tool_*` method against the in-memory
database with the real `DocumentService`, the real `UrlIngestionService` and
the real `WebScraperService`. Only the edges that leave the process are
replaced, and each replacement refuses a call the real method would refuse:

* the vector store, MinIO and Redis singletons get recording methods bound
  against the real signatures;
* HTTP goes through `httpx.MockTransport`, and DNS through a table, so a
  request that should never be sent is visible as a request;
* Celery's `.delay` is bound against the real task function.

A test marked xfail(strict) is a defect in the application, not in the test.
"""

import hashlib
import inspect
import sys
import types
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import func, select

from app.core.cache import cache_service
from app.core.config import settings
from app.models.document import Document, DocumentChunk, DocumentSource
from app.schemas.agent import AgentToolCall
from app.services import url_ingestion_service as url_ingestion_module
from app.services import web_scraper_service as web_scraper_module
from app.services.agent_service import AgentService
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_agent_service_document_provider,
)
from app.services.document_service import DocumentService
from app.services.storage_service import storage_service
from app.services.text_processor import TextProcessor
from app.services.vector_store import vector_store_service
from app.tasks.summarization_tasks import summarize_document

pytestmark = pytest.mark.unit

PUBLIC_IP = "93.184.216.34"
DNS = {
    "example.com": PUBLIC_IP,
    "other.example.org": "93.184.216.35",
    "www.youtube.com": "142.250.1.1",
    "intranet.corp": "10.0.0.5",
    "wiki.intranet.corp": "10.0.0.6",
    "db.corp": "10.0.0.9",
    "rebind.example.com": "127.0.0.1",
    "metadata.example.com": "169.254.169.254",
}


# --------------------------------------------------------------------------
# The edges
# --------------------------------------------------------------------------


class _Edges:
    """What left the process, and switches to make an edge fail."""

    def __init__(self):
        self.vector_deleted = []
        self.vector_added = []
        self.files_deleted = []
        self.cache_deleted = []
        self.fail_vector_delete = False
        self.fail_vector_add = False
        self.fail_storage = False


def _recording(real, impl):
    """An async stand-in that refuses arguments `real` does not accept."""
    signature = inspect.signature(real)

    async def call(*args, **kwargs):
        signature.bind(*args, **kwargs)
        return impl(*args, **kwargs)

    return call


@pytest.fixture
def edges(monkeypatch):
    rec = _Edges()

    def vector_delete(document_id):
        if rec.fail_vector_delete:
            raise RuntimeError("vector store unreachable")
        rec.vector_deleted.append(document_id)

    def vector_add(document, chunks):
        if rec.fail_vector_add:
            raise RuntimeError("vector store unreachable")
        rec.vector_added.append((document.id, [c.content for c in chunks]))

    def file_delete(object_path):
        if rec.fail_storage:
            raise RuntimeError("minio unreachable")
        rec.files_deleted.append(object_path)
        return True

    def cache_delete(key):
        rec.cache_deleted.append(key)
        return True

    for obj, name, impl in [
        (vector_store_service, "initialize", lambda *a, **k: None),
        (vector_store_service, "delete_document_chunks", vector_delete),
        (vector_store_service, "add_document_chunks", vector_add),
        (storage_service, "delete_file", file_delete),
        (cache_service, "get", lambda key: None),
        (cache_service, "set", lambda key, value, ttl=None: True),
        (cache_service, "delete", cache_delete),
        (cache_service, "delete_pattern", lambda pattern: 0),
    ]:
        monkeypatch.setattr(obj, name, _recording(getattr(obj, name), impl))

    # Keep indexing to the part under test: chunk rows and the vector store.
    monkeypatch.setattr(settings, "KNOWLEDGE_GRAPH_ENABLED", False, raising=False)
    monkeypatch.setattr(settings, "AUTO_SUMMARIZE_ON_PROCESS", False, raising=False)
    monkeypatch.setattr(settings, "RAG_CHUNKING_STRATEGY", "fixed", raising=False)
    return rec


def _document_service():
    """The real DocumentService, minus the constructor's LLM client."""
    docs = DocumentService.__new__(DocumentService)
    docs.vector_store = vector_store_service
    docs.text_processor = TextProcessor()
    docs._vector_store_initialized = False
    return docs


@pytest.fixture
def service(edges, monkeypatch):
    svc = AgentService.__new__(AgentService)
    svc.document_service = _document_service()
    monkeypatch.setattr(
        url_ingestion_module, "DocumentService", lambda: svc.document_service
    )
    return svc


class _Web:
    """Routes for the mock transport, and every request that reached it."""

    def __init__(self):
        self.routes = {}
        self.requests = []
        self.scraper_kwargs = []

    def page(self, url, body, status=200, content_type="text/html", headers=None):
        self.routes[url] = (status, {"content-type": content_type}, body)
        if headers:
            self.routes[url][1].update(headers)

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(url)
        if url not in self.routes:
            return httpx.Response(404, text="not found")
        status, headers, body = self.routes[url]
        return httpx.Response(status, headers=headers, text=body)


@pytest.fixture
def web(monkeypatch):
    """The real WebScraperService over a mock transport and a DNS table."""
    rec = _Web()
    real_scraper = web_scraper_module.WebScraperService

    def scraper_factory(**kwargs):
        rec.scraper_kwargs.append(kwargs)
        client = httpx.AsyncClient(transport=httpx.MockTransport(rec.handler))
        return real_scraper(client=client, **kwargs)

    def getaddrinfo(host, port, *args, **kwargs):
        if host not in DNS:
            raise OSError(f"no such host: {host}")
        return [(2, 1, 6, "", (DNS[host], 0))]

    monkeypatch.setattr(web_scraper_module, "WebScraperService", scraper_factory)
    monkeypatch.setattr(url_ingestion_module, "WebScraperService", scraper_factory)
    monkeypatch.setattr(
        web_scraper_module, "socket", SimpleNamespace(getaddrinfo=getaddrinfo)
    )
    return rec


@pytest.fixture
def queue(monkeypatch):
    """Celery's .delay for the summarization task, bound to the real task."""
    calls = []
    signature = inspect.signature(summarize_document.run)
    state = SimpleNamespace(calls=calls, fail_for=set())

    def delay(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if bound.arguments["document_id"] in state.fail_for:
            raise ConnectionError("broker unreachable")
        calls.append(dict(bound.arguments))
        return SimpleNamespace(id=str(uuid4()))

    monkeypatch.setattr(summarize_document, "delay", delay)
    return state


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------


async def _source(db, name="Uploads", source_type="file", config=None, **extra):
    source = DocumentSource(
        name=name, source_type=source_type, config=config or {}, **extra
    )
    db.add(source)
    await db.commit()
    await db.refresh(source)
    return source


async def _doc(db, source, title, content="body text", chunks=2, **extra):
    doc = Document(
        title=title,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        source_id=source.id,
        source_identifier=f"test:{uuid4().hex}",
        **extra,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    for index in range(chunks):
        text = f"{title} chunk {index}"
        db.add(
            DocumentChunk(
                document_id=doc.id,
                content=text,
                content_hash=hashlib.sha256(text.encode()).hexdigest(),
                chunk_index=index,
            )
        )
    await db.commit()
    return doc


async def _titles(db):
    rows = (await db.execute(select(Document.title))).scalars().all()
    return sorted(rows)


async def _chunk_count(db, document_id=None):
    query = select(func.count(DocumentChunk.id))
    if document_id is not None:
        query = query.where(DocumentChunk.document_id == document_id)
    return (await db.execute(query)).scalar()


async def _tags(db, document_id):
    return (
        await db.execute(select(Document.tags).where(Document.id == document_id))
    ).scalar_one()


async def _attempt(coro):
    """(result, None) or (None, exception): a refusal may take either form."""
    try:
        return await coro, None
    except Exception as exc:  # noqa: BLE001 - the form is what is being observed
        return None, exc


def _produced_nothing(result, exc):
    return exc is not None or "error" in result or not result.get("pages")


# --------------------------------------------------------------------------
# delete_document
# --------------------------------------------------------------------------


@pytest.mark.parametrize("params", [{}, {"document_id": None}, {"document_id": ""}])
async def test_delete_refuses_without_a_document_id(service, db_session, params):
    result = await service._tool_delete_document(
        {**params, "confirm": True}, db_session
    )

    assert "error" in result
    assert "action" not in result


async def test_delete_reports_a_malformed_or_unknown_id(service, db_session, edges):
    source = await _source(db_session)
    await _doc(db_session, source, "keep")

    malformed = await service._tool_delete_document(
        {"document_id": "not-a-uuid", "confirm": True}, db_session
    )
    unknown = await service._tool_delete_document(
        {"document_id": str(uuid4()), "confirm": True}, db_session
    )

    assert "Invalid document ID" in malformed["error"]
    assert "not found" in unknown["error"]
    assert await _titles(db_session) == ["keep"]
    assert edges.vector_deleted == []


async def test_delete_reports_a_non_string_id_instead_of_raising(service, db_session):
    result = await service._tool_delete_document(
        {"document_id": 12345, "confirm": True}, db_session
    )

    assert "error" in result


@pytest.mark.parametrize("confirm", ["absent", False, None, 0])
async def test_delete_without_confirmation_deletes_nothing(
    service, db_session, edges, confirm
):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Quarterly plan", file_path="docs/plan.pdf")
    params = {"document_id": str(doc.id)}
    if confirm != "absent":
        params["confirm"] = confirm

    result = await service._tool_delete_document(params, db_session)

    assert result["action"] == "confirmation_required"
    assert result["document_id"] == str(doc.id)
    assert result["title"] == "Quarterly plan"
    assert await _titles(db_session) == ["Quarterly plan"]
    assert await _chunk_count(db_session, doc.id) == 2
    assert edges.vector_deleted == []
    assert edges.files_deleted == []


@pytest.mark.parametrize("confirm", ["false", "no", "0"])
async def test_delete_is_not_confirmed_by_a_string_saying_no(
    service, db_session, edges, confirm
):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "keep")

    result = await service._tool_delete_document(
        {"document_id": str(doc.id), "confirm": confirm}, db_session
    )

    assert await _titles(db_session) == ["keep"]
    assert result.get("action") != "deleted"


async def test_confirmed_delete_removes_row_chunks_vectors_and_file(
    service, db_session, edges
):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Doomed", file_path="docs/doomed.pdf")
    other = await _doc(db_session, source, "Bystander", file_path="docs/other.pdf")
    doc_id, other_id = doc.id, other.id

    result = await service._tool_delete_document(
        {"document_id": str(doc_id), "confirm": True}, db_session
    )

    assert result["action"] == "deleted"
    assert result["document_id"] == str(doc_id)
    assert result["title"] == "Doomed"
    assert await _titles(db_session) == ["Bystander"]
    assert await _chunk_count(db_session, doc_id) == 0
    assert await _chunk_count(db_session, other_id) == 2
    assert edges.vector_deleted == [doc_id]
    assert edges.files_deleted == ["docs/doomed.pdf"]
    # There is no document cache to invalidate any more: it was written on
    # upload and never read, so the delete no longer touches Redis for it.
    assert edges.cache_deleted == []
    # The source the document lived in is not collateral.
    assert (await db_session.get(DocumentSource, source.id)) is not None


async def test_delete_of_a_document_without_a_file_touches_no_storage(
    service, db_session, edges
):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Note", chunks=0)

    result = await service._tool_delete_document(
        {"document_id": str(doc.id), "confirm": True}, db_session
    )

    assert result["action"] == "deleted"
    assert edges.files_deleted == []
    assert await _titles(db_session) == []


async def test_deleting_twice_reports_the_second_as_not_found(service, db_session):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Once")
    params = {"document_id": str(doc.id), "confirm": True}

    first = await service._tool_delete_document(params, db_session)
    second = await service._tool_delete_document(params, db_session)

    assert first["action"] == "deleted"
    assert "not found" in second["error"]


class _RefusingDeletes(DocumentService):
    """The real service, except deletion fails for the ids named."""

    def __init__(self, mode, ids):
        self.vector_store = vector_store_service
        self.text_processor = None
        self._vector_store_initialized = True
        self._mode = mode
        self._ids = {str(i) for i in ids}

    async def delete_document(self, document_id, db, *, warnings=None):
        if str(document_id) in self._ids:
            if self._mode == "raise":
                raise RuntimeError("database is read-only")
            return False
        return await super().delete_document(document_id, db, warnings=warnings)


@pytest.mark.parametrize("mode", ["false", "raise"])
async def test_a_delete_that_fails_is_reported_as_a_failure(service, db_session, mode):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Stuck")
    service.document_service = _RefusingDeletes(mode, [doc.id])

    result = await service._tool_delete_document(
        {"document_id": str(doc.id), "confirm": True}, db_session
    )

    assert "error" in result
    assert result.get("action") != "deleted"
    assert await _titles(db_session) == ["Stuck"]


@pytest.mark.parametrize("edge", ["fail_vector_delete", "fail_storage"])
async def test_delete_does_not_claim_success_when_vectors_or_file_remain(
    service, db_session, edges, edge
):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Half gone", file_path="docs/half.pdf")
    setattr(edges, edge, True)

    result = await service._tool_delete_document(
        {"document_id": str(doc.id), "confirm": True}, db_session
    )

    row_remains = await _titles(db_session) == ["Half gone"]
    said_so = "error" in result or "warning" in result or "warnings" in result
    assert row_remains or said_so


# --------------------------------------------------------------------------
# Deletion through chat, and the confirm endpoint
# --------------------------------------------------------------------------


async def test_chat_reaches_the_delete_tools_with_the_callers_params(
    service, db_session, test_user
):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Via dispatch")
    provider = build_agent_service_document_provider(service)
    ctx = AgentToolExecutionContext(
        mode="chat", db=db_session, service=service, user_id=test_user.id
    )

    asked = await provider._handlers["delete_document"](
        {"document_id": str(doc.id)}, ctx
    )
    assert asked["action"] == "confirmation_required"
    assert await _titles(db_session) == ["Via dispatch"]

    done = await provider._handlers["batch_delete_documents"](
        {"document_ids": [str(doc.id)], "confirm": True}, ctx
    )
    assert done["deleted_count"] == 1
    assert await _titles(db_session) == []


@pytest.mark.parametrize(
    "tool, params",
    [
        ("delete_document", lambda i: {"document_id": i, "confirm": True}),
        ("batch_delete_documents", lambda i: {"document_ids": [i], "confirm": True}),
    ],
)
async def test_a_chat_turn_cannot_delete_without_an_approval(
    service, db_session, test_user, edges, monkeypatch, tool, params
):
    """The model setting confirm=true is not a person confirming."""
    monkeypatch.setattr(settings, "AGENT_REQUIRE_TOOL_APPROVAL", True)
    assert tool in settings.AGENT_DANGEROUS_TOOLS
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Protected")
    reached = []
    service.tool_registry = SimpleNamespace(
        try_execute=lambda *a, **k: reached.append(a)
    )

    call = await service._execute_tool(
        AgentToolCall(tool_name=tool, tool_input=params(str(doc.id)), status="pending"),
        test_user.id,
        db_session,
    )

    assert call.status == "requires_approval"
    assert call.tool_output["error"] == "approval_required"
    assert reached == []
    assert await _titles(db_session) == ["Protected"]
    assert edges.vector_deleted == []


async def test_confirm_endpoint_requires_a_signed_in_caller(client, db_session, edges):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Private")

    response = client.post(f"/api/v1/agent/confirm-delete/{doc.id}")

    assert response.status_code in (401, 403)
    assert await _titles(db_session) == ["Private"]
    assert edges.vector_deleted == []


async def test_confirm_endpoint_deletes_the_document(
    client, auth_headers, db_session, edges, monkeypatch
):
    """Where approvals are switched off, confirming is enough."""
    monkeypatch.setattr(settings, "AGENT_REQUIRE_TOOL_APPROVAL", False)
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Confirmed", file_path="docs/c.pdf")
    await _doc(db_session, source, "Other")
    doc_id = doc.id

    response = client.post(
        f"/api/v1/agent/confirm-delete/{doc_id}", headers=auth_headers
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["action"] == "deleted"
    assert body["document_id"] == str(doc_id)
    assert await _titles(db_session) == ["Other"]
    assert await _chunk_count(db_session, doc_id) == 0
    assert edges.vector_deleted == [doc_id]
    assert edges.files_deleted == ["docs/c.pdf"]


async def test_confirm_endpoint_answers_a_bad_or_unknown_id_without_deleting(
    client, auth_headers, db_session
):
    source = await _source(db_session)
    await _doc(db_session, source, "Untouched")

    malformed = client.post(
        "/api/v1/agent/confirm-delete/not-a-uuid", headers=auth_headers
    )
    unknown = client.post(
        f"/api/v1/agent/confirm-delete/{uuid4()}", headers=auth_headers
    )

    assert malformed.status_code in (400, 404, 422)
    assert unknown.status_code in (400, 404)
    assert "not found" in unknown.json()["detail"].lower()
    assert await _titles(db_session) == ["Untouched"]


async def test_confirm_endpoint_does_not_bypass_the_approval_gate(
    client, auth_headers, db_session, monkeypatch
):
    monkeypatch.setattr(settings, "AGENT_REQUIRE_TOOL_APPROVAL", True)
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Nobody asked to delete this")

    response = client.post(
        f"/api/v1/agent/confirm-delete/{doc.id}", headers=auth_headers
    )

    assert response.status_code in (400, 403, 404, 409)
    assert await _titles(db_session) == ["Nobody asked to delete this"]


# --------------------------------------------------------------------------
# batch_delete_documents
# --------------------------------------------------------------------------


@pytest.mark.parametrize("params", [{}, {"document_ids": []}, {"document_ids": None}])
async def test_batch_delete_refuses_without_ids(service, db_session, params):
    result = await service._tool_batch_delete_documents(
        {**params, "confirm": True}, db_session
    )

    assert "error" in result


async def test_batch_delete_refuses_more_than_fifty(service, db_session, edges):
    source = await _source(db_session)
    docs = [await _doc(db_session, source, f"d{i:02d}", chunks=0) for i in range(51)]

    over = await service._tool_batch_delete_documents(
        {"document_ids": [str(d.id) for d in docs], "confirm": True}, db_session
    )

    assert "50" in over["error"]
    assert len(await _titles(db_session)) == 51
    assert edges.vector_deleted == []

    at_cap = await service._tool_batch_delete_documents(
        {"document_ids": [str(d.id) for d in docs[:50]], "confirm": True},
        db_session,
    )

    assert at_cap["deleted_count"] == 50
    assert await _titles(db_session) == ["d50"]


async def test_batch_delete_without_confirmation_deletes_nothing(
    service, db_session, edges
):
    source = await _source(db_session)
    one = await _doc(db_session, source, "one")
    two = await _doc(db_session, source, "two")

    result = await service._tool_batch_delete_documents(
        {"document_ids": [str(one.id), str(two.id)]}, db_session
    )

    assert result["action"] == "confirmation_required"
    assert result["count"] == 2
    assert sorted(d["title"] for d in result["documents"]) == ["one", "two"]
    assert await _titles(db_session) == ["one", "two"]
    assert await _chunk_count(db_session) == 4
    assert edges.vector_deleted == []


async def test_batch_delete_with_only_bad_ids_is_an_error(service, db_session):
    source = await _source(db_session)
    await _doc(db_session, source, "keep")

    result = await service._tool_batch_delete_documents(
        {"document_ids": ["nope", str(uuid4())], "confirm": True}, db_session
    )

    assert "error" in result
    assert await _titles(db_session) == ["keep"]


async def test_confirmed_batch_delete_removes_exactly_the_named_documents(
    service, db_session, edges
):
    source = await _source(db_session)
    one = await _doc(db_session, source, "one", file_path="docs/one.pdf")
    two = await _doc(db_session, source, "two")
    await _doc(db_session, source, "three")
    one_id, two_id = one.id, two.id

    result = await service._tool_batch_delete_documents(
        {"document_ids": [str(one_id), str(two_id)], "confirm": True}, db_session
    )

    assert result["action"] == "batch_deleted"
    assert result["deleted_count"] == 2
    assert sorted(result["deleted_ids"]) == sorted([str(one_id), str(two_id)])
    assert result["failed_count"] == 0
    assert await _titles(db_session) == ["three"]
    assert await _chunk_count(db_session) == 2
    assert sorted(edges.vector_deleted, key=str) == sorted([one_id, two_id], key=str)
    assert edges.files_deleted == ["docs/one.pdf"]


async def test_batch_delete_still_deletes_the_good_ids_beside_bad_ones(
    service, db_session
):
    source = await _source(db_session)
    one = await _doc(db_session, source, "one")
    two = await _doc(db_session, source, "two")
    await _doc(db_session, source, "keep")
    ids = [str(one.id), "not-a-uuid", str(uuid4()), str(two.id)]

    result = await service._tool_batch_delete_documents(
        {"document_ids": ids, "confirm": True}, db_session
    )

    assert result["deleted_count"] == 2
    assert await _titles(db_session) == ["keep"]


@pytest.mark.parametrize("confirm", [True, False])
async def test_batch_delete_reports_the_ids_it_could_not_delete(
    service, db_session, confirm
):
    source = await _source(db_session)
    one = await _doc(db_session, source, "one")
    missing = str(uuid4())

    result = await service._tool_batch_delete_documents(
        {"document_ids": [str(one.id), "not-a-uuid", missing], "confirm": confirm},
        db_session,
    )

    assert "not-a-uuid" in repr(result)
    assert missing in repr(result)


async def test_batch_delete_survives_a_non_string_id(service, db_session):
    source = await _source(db_session)
    one = await _doc(db_session, source, "one")

    result = await service._tool_batch_delete_documents(
        {"document_ids": [123, str(one.id)], "confirm": True}, db_session
    )

    assert result["deleted_count"] == 1


async def test_batch_delete_counts_a_repeated_id_once(service, db_session):
    source = await _source(db_session)
    one = await _doc(db_session, source, "one")
    ids = [str(one.id), str(one.id)]

    asked = await service._tool_batch_delete_documents(
        {"document_ids": ids}, db_session
    )
    done = await service._tool_batch_delete_documents(
        {"document_ids": ids, "confirm": True}, db_session
    )

    assert asked["count"] == 1
    assert (done["deleted_count"], done["failed_count"]) == (1, 0)


@pytest.mark.parametrize("mode", ["false", "raise"])
async def test_batch_delete_reports_a_partial_failure_truthfully(
    service, db_session, mode
):
    source = await _source(db_session)
    one = await _doc(db_session, source, "one")
    stuck = await _doc(db_session, source, "stuck")
    three = await _doc(db_session, source, "three")
    ids = [str(one.id), str(stuck.id), str(three.id)]
    service.document_service = _RefusingDeletes(mode, [stuck.id])

    result = await service._tool_batch_delete_documents(
        {"document_ids": ids, "confirm": True}, db_session
    )

    assert result["deleted_count"] == 2
    assert sorted(result["deleted_ids"]) == sorted([ids[0], ids[2]])
    assert result["failed_count"] == 1
    assert [f["id"] for f in result["failed"]] == [ids[1]]
    assert "1 failed" in result["message"]
    assert await _titles(db_session) == ["stuck"]


async def test_batch_delete_is_not_confirmed_by_a_string_saying_no(service, db_session):
    source = await _source(db_session)
    one = await _doc(db_session, source, "one")

    await service._tool_batch_delete_documents(
        {"document_ids": [str(one.id)], "confirm": "false"}, db_session
    )

    assert await _titles(db_session) == ["one"]


# --------------------------------------------------------------------------
# create_document_from_text
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "params, missing",
    [
        ({"content": "body"}, "Title"),
        ({"title": "   ", "content": "body"}, "Title"),
        ({"title": "T"}, "Content"),
        ({"title": "T", "content": " \n "}, "Content"),
    ],
)
async def test_create_refuses_without_title_or_content(
    service, db_session, test_user, params, missing
):
    result = await service._tool_create_document_from_text(
        params, test_user.id, db_session
    )

    assert missing in result["error"]
    assert await _titles(db_session) == []


@pytest.mark.parametrize(
    "params",
    [{"title": None, "content": "body"}, {"title": "T", "content": None}],
)
async def test_create_reports_a_null_field_instead_of_raising(
    service, db_session, test_user, params
):
    result = await service._tool_create_document_from_text(
        params, test_user.id, db_session
    )

    assert "error" in result


async def test_create_writes_a_real_indexed_document(
    service, db_session, test_user, edges
):
    content = "Cache prefetchers issue requests ahead of the demand stream."

    result = await service._tool_create_document_from_text(
        {"title": "  Prefetch notes  ", "content": content, "tags": ["cache", "ml"]},
        test_user.id,
        db_session,
    )

    assert result["action"] == "created"
    (doc,) = await _ingested(db_session)
    assert result["document_id"] == str(doc.id)
    assert doc.title == "Prefetch notes"
    assert doc.content == content
    assert doc.content_hash == hashlib.sha256(content.encode()).hexdigest()
    assert doc.file_size == len(content.encode())
    assert doc.tags == ["cache", "ml"]
    assert result["tags"] == ["cache", "ml"]
    assert doc.author == "Test User"
    assert doc.extra_metadata["origin"] == "agent_created"
    assert doc.source_identifier.startswith("agent_note:")
    source = await db_session.get(DocumentSource, doc.source_id)
    assert source.name == "Agent Notes"
    # Indexed: chunk rows exist, the vector store received them, and the row
    # says so.
    assert await _chunk_count(db_session, doc.id) >= 1
    assert [doc_id for doc_id, _ in edges.vector_added] == [doc.id]
    assert content in " ".join(edges.vector_added[0][1])
    assert doc.is_processed is True


async def test_create_twice_makes_two_documents_in_one_notes_source(
    service, db_session, test_user
):
    for title in ("first", "second"):
        result = await service._tool_create_document_from_text(
            {"title": title, "content": "same body"}, test_user.id, db_session
        )
        assert result["action"] == "created"

    assert await _titles(db_session) == ["first", "second"]
    sources = (await db_session.execute(select(DocumentSource))).scalars().all()
    assert [s.name for s in sources] == ["Agent Notes"]


async def test_create_previews_long_content_and_stores_all_of_it(
    service, db_session, test_user
):
    content = "word " * 400

    result = await service._tool_create_document_from_text(
        {"title": "Long", "content": content}, test_user.id, db_session
    )

    assert result["content_preview"].endswith("...")
    assert len(result["content_preview"]) == 203
    doc = (await db_session.execute(select(Document))).scalar_one()
    assert doc.content == content.strip()


async def test_create_says_so_when_the_document_could_not_be_indexed(
    service, db_session, test_user, edges
):
    edges.fail_vector_add = True

    result = await service._tool_create_document_from_text(
        {"title": "Unindexed", "content": "A note long enough to chunk. " * 4},
        test_user.id,
        db_session,
    )

    doc = (await db_session.execute(select(Document))).scalar_one_or_none()
    assert doc is None or doc.is_processed is False
    said_so = any(k in result for k in ("error", "warning", "warnings", "indexed"))
    assert said_so


async def test_a_short_note_is_still_indexed(service, db_session, test_user, edges):
    await service._tool_create_document_from_text(
        {"title": "Reminder", "content": "The deploy key rotates on Friday."},
        test_user.id,
        db_session,
    )

    (doc,) = await _ingested(db_session)
    assert await _chunk_count(db_session, doc.id) >= 1
    assert edges.vector_added and edges.vector_added[0][1]


async def test_create_failure_is_an_error_and_leaves_the_session_usable(
    service, db_session, test_user
):
    """An insert the database refuses must not poison the caller's session."""
    result = await service._tool_create_document_from_text(
        {"title": "Bad", "content": "body", "tags": [{"unserialisable": {1, 2}}]},
        test_user.id,
        db_session,
    )

    assert "error" in result
    assert result.get("action") != "created"
    # The session a chat turn shares with everything after this tool.
    assert await _titles(db_session) == []


# --------------------------------------------------------------------------
# update_document_tags
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "document_id", [None, "", "not-a-uuid", "00000000-0000-0000-0000-000000000000"]
)
async def test_tags_reports_a_missing_malformed_or_unknown_id(
    service, db_session, document_id
):
    result = await service._tool_update_document_tags(
        {"document_id": document_id, "tags": ["a"]}, db_session
    )

    assert "error" in result


async def test_tags_are_added_by_default_without_duplicates(service, db_session):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "T", tags=["a", "b"])

    result = await service._tool_update_document_tags(
        {"document_id": str(doc.id), "tags": ["b", "c"]}, db_session
    )

    assert sorted(result["previous_tags"]) == ["a", "b"]
    assert sorted(result["current_tags"]) == ["a", "b", "c"]
    assert sorted(await _tags(db_session, doc.id)) == ["a", "b", "c"]


async def test_tags_can_be_added_to_a_document_that_has_none(service, db_session):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "T", tags=None)

    result = await service._tool_update_document_tags(
        {"document_id": str(doc.id), "tags": ["a"], "action": "add"}, db_session
    )

    assert result["previous_tags"] == []
    assert await _tags(db_session, doc.id) == ["a"]


async def test_tags_remove_only_the_named_ones(service, db_session):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "T", tags=["a", "b", "c"])

    result = await service._tool_update_document_tags(
        {"document_id": str(doc.id), "tags": ["b", "zzz"], "action": "remove"},
        db_session,
    )

    assert sorted(result["current_tags"]) == ["a", "c"]
    assert sorted(await _tags(db_session, doc.id)) == ["a", "c"]


async def test_tags_replace_discards_the_old_set(service, db_session):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "T", tags=["a", "b"])
    other = await _doc(db_session, source, "Other", tags=["a"])

    result = await service._tool_update_document_tags(
        {"document_id": str(doc.id), "tags": ["x", "y"], "action": "replace"},
        db_session,
    )

    assert sorted(result["current_tags"]) == ["x", "y"]
    assert sorted(await _tags(db_session, doc.id)) == ["x", "y"]
    assert await _tags(db_session, other.id) == ["a"]


async def test_tags_refuse_an_unknown_action_and_change_nothing(service, db_session):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "T", tags=["a"])

    result = await service._tool_update_document_tags(
        {"document_id": str(doc.id), "tags": ["b"], "action": "toggle"}, db_session
    )

    assert "Invalid action" in result["error"]
    assert await _tags(db_session, doc.id) == ["a"]


@pytest.mark.parametrize("action", ["add", "remove", "replace"])
async def test_tags_are_required(service, db_session, action):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "T", tags=["a", "b"])

    result = await service._tool_update_document_tags(
        {"document_id": str(doc.id), "action": action}, db_session
    )

    assert sorted(await _tags(db_session, doc.id)) == ["a", "b"]
    assert "error" in result


async def test_a_string_of_tags_is_not_split_into_letters(service, db_session):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "T", tags=["a"])

    result = await service._tool_update_document_tags(
        {"document_id": str(doc.id), "tags": "ml", "action": "add"}, db_session
    )

    stored = sorted(await _tags(db_session, doc.id))
    assert stored in (["a"], ["a", "ml"])
    assert "error" in result or sorted(result["current_tags"]) == ["a", "ml"]


async def test_tags_that_are_not_strings_are_refused_not_raised(service, db_session):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "T", tags=["a"])

    result = await service._tool_update_document_tags(
        {"document_id": str(doc.id), "tags": [["nested"]]}, db_session
    )

    assert "error" in result


async def test_replace_stores_the_tags_in_the_order_given(service, db_session):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "T", tags=["old"])
    wanted = [f"tag-{i:02d}" for i in range(20)]

    await service._tool_update_document_tags(
        {"document_id": str(doc.id), "tags": wanted, "action": "replace"}, db_session
    )

    assert await _tags(db_session, doc.id) == wanted


# --------------------------------------------------------------------------
# batch_summarize_documents
# --------------------------------------------------------------------------


@pytest.mark.parametrize("params", [{}, {"document_ids": []}, {"document_ids": None}])
async def test_summarize_refuses_without_ids(service, db_session, queue, params):
    result = await service._tool_batch_summarize_documents(params, db_session)

    assert "error" in result
    assert queue.calls == []


async def test_summarize_refuses_more_than_twenty(service, db_session, queue):
    source = await _source(db_session)
    docs = [await _doc(db_session, source, f"d{i:02d}", chunks=0) for i in range(21)]

    over = await service._tool_batch_summarize_documents(
        {"document_ids": [str(d.id) for d in docs]}, db_session
    )

    assert "20" in over["error"]
    assert queue.calls == []

    at_cap = await service._tool_batch_summarize_documents(
        {"document_ids": [str(d.id) for d in docs[:20]]}, db_session
    )

    assert at_cap["queued_count"] == 20
    assert len(queue.calls) == 20


async def test_summarize_queues_the_real_task_for_documents_lacking_a_summary(
    service, db_session, queue
):
    source = await _source(db_session)
    bare = await _doc(db_session, source, "bare")
    done = await _doc(db_session, source, "done", summary="Already summarised.")
    missing = str(uuid4())

    result = await service._tool_batch_summarize_documents(
        {"document_ids": [str(bare.id), str(done.id), "not-a-uuid", missing]},
        db_session,
    )

    assert queue.calls == [
        {"document_id": str(bare.id), "force": False, "user_id": None}
    ]
    assert result["queued_count"] == 1
    assert result["queued"] == [{"id": str(bare.id), "title": "bare"}]
    assert result["skipped_count"] == 1
    assert result["skipped"][0]["id"] == str(done.id)
    assert result["invalid_count"] == 2
    assert sorted(result["invalid_ids"]) == sorted(["not-a-uuid", missing])
    # Queuing is not summarising: nothing is written by the tool itself.
    summaries = (await db_session.execute(select(Document.summary))).scalars()
    assert [s for s in summaries if s != "Already summarised."] == [None]


async def test_force_regenerate_queues_a_document_that_has_a_summary(
    service, db_session, queue
):
    source = await _source(db_session)
    done = await _doc(db_session, source, "done", summary="Old summary.")

    result = await service._tool_batch_summarize_documents(
        {"document_ids": [str(done.id)], "force_regenerate": True}, db_session
    )

    assert queue.calls == [
        {"document_id": str(done.id), "force": True, "user_id": None}
    ]
    assert result["queued_count"] == 1
    assert result["skipped_count"] == 0


async def test_summarize_with_nothing_to_queue_does_not_claim_work(
    service, db_session, queue
):
    source = await _source(db_session)
    done = await _doc(db_session, source, "done", summary="Have one.")

    result = await service._tool_batch_summarize_documents(
        {"document_ids": [str(done.id)]}, db_session
    )

    assert queue.calls == []
    assert result["queued_count"] == 0
    assert result["queued"] == []
    assert result["skipped_count"] == 1


async def test_summarize_reports_a_queue_failure_without_losing_the_others(
    service, db_session, queue
):
    source = await _source(db_session)
    one = await _doc(db_session, source, "one")
    two = await _doc(db_session, source, "two")
    three = await _doc(db_session, source, "three")
    queue.fail_for = {str(two.id)}

    result = await service._tool_batch_summarize_documents(
        {"document_ids": [str(one.id), str(two.id), str(three.id)]}, db_session
    )

    assert sorted(q["title"] for q in result["queued"]) == ["one", "three"]
    assert result["queued_count"] == 2
    assert str(two.id) in repr({k: v for k, v in result.items() if k != "queued"})


async def test_summarize_counts_a_non_string_id_as_invalid(service, db_session, queue):
    source = await _source(db_session)
    one = await _doc(db_session, source, "one")

    result = await service._tool_batch_summarize_documents(
        {"document_ids": [str(one.id), 123]}, db_session
    )

    assert result["queued_count"] == 1
    assert result["invalid_ids"] == [123]


async def test_summarize_queues_a_repeated_id_once(service, db_session, queue):
    source = await _source(db_session)
    one = await _doc(db_session, source, "one")

    await service._tool_batch_summarize_documents(
        {"document_ids": [str(one.id), str(one.id)]}, db_session
    )

    assert len(queue.calls) == 1


# --------------------------------------------------------------------------
# web_scrape
# --------------------------------------------------------------------------

ARTICLE = """<html><head><title>Stride prefetching</title></head><body>
<nav>menu junk</nav><main><p>Stride prefetchers track deltas.</p>
<a href="/two">two</a> <a href="https://other.example.org/x">elsewhere</a>
</main><script>alert(1)</script></body></html>"""
PAGE_TWO = """<html><head><title>Page two</title></head><body>
<main><p>Second page body.</p><a href="/three">three</a></main></body></html>"""
PAGE_THREE = """<html><head><title>Page three</title></head><body>
<main><p>Third page body.</p></main></body></html>"""


def _site(web):
    web.page("https://example.com/", ARTICLE)
    web.page("https://example.com/two", PAGE_TWO)
    web.page("https://example.com/three", PAGE_THREE)
    web.page("https://other.example.org/x", "<html><body>Elsewhere.</body></html>")


@pytest.mark.parametrize("params", [{}, {"url": ""}, {"url": None}])
async def test_scrape_requires_a_url(service, db_session, test_user, web, params):
    result = await service._tool_web_scrape(params, test_user.id, db_session)

    assert "url" in result["error"]
    assert web.requests == []


async def test_scrape_returns_the_readable_page(service, db_session, test_user, web):
    _site(web)

    result = await service._tool_web_scrape(
        {"url": "https://example.com/"}, test_user.id, db_session
    )

    assert web.requests == ["https://example.com/"]
    assert web.scraper_kwargs == [{"enforce_network_safety": True}]
    assert result["total_pages"] == 1
    page = result["pages"][0]
    assert page["url"] == "https://example.com/"
    assert page["title"] == "Stride prefetching"
    assert "Stride prefetchers track deltas." in page["content"]
    assert "alert(1)" not in page["content"]
    assert "menu junk" not in page["content"]
    assert page["links"] == [
        "https://example.com/two",
        "https://other.example.org/x",
    ]
    assert result["errors"] == []


async def test_scrape_can_omit_links(service, db_session, test_user, web):
    _site(web)

    result = await service._tool_web_scrape(
        {"url": "https://example.com/", "include_links": False},
        test_user.id,
        db_session,
    )

    assert result["pages"][0]["links"] == []


async def test_scrape_truncates_to_max_content_chars(
    service, db_session, test_user, web
):
    web.page("https://example.com/", "<html><body>" + "x" * 5000 + "</body></html>")

    short = await service._tool_web_scrape(
        {"url": "https://example.com/", "max_content_chars": 500},
        test_user.id,
        db_session,
    )
    full = await service._tool_web_scrape(
        {"url": "https://example.com/"}, test_user.id, db_session
    )

    assert len(short["pages"][0]["content"]) <= 500
    assert short["pages"][0]["content"].endswith("[truncated]")
    assert len(full["pages"][0]["content"]) == 5000


async def test_scrape_crawls_only_when_asked_and_within_its_bounds(
    service, db_session, test_user, web
):
    _site(web)
    base = {"url": "https://example.com/", "follow_links": True}

    not_asked = await service._tool_web_scrape(
        {"url": "https://example.com/", "max_pages": 5, "max_depth": 2},
        test_user.id,
        db_session,
    )
    assert [p["url"] for p in not_asked["pages"]] == ["https://example.com/"]

    depth_one = await service._tool_web_scrape(
        {**base, "max_pages": 5, "max_depth": 1}, test_user.id, db_session
    )
    assert [p["url"] for p in depth_one["pages"]] == [
        "https://example.com/",
        "https://example.com/two",
    ]

    depth_two = await service._tool_web_scrape(
        {**base, "max_pages": 5, "max_depth": 2}, test_user.id, db_session
    )
    assert [p["url"] for p in depth_two["pages"]] == [
        "https://example.com/",
        "https://example.com/two",
        "https://example.com/three",
    ]

    capped = await service._tool_web_scrape(
        {**base, "max_pages": 2, "max_depth": 2}, test_user.id, db_session
    )
    assert capped["total_pages"] == 2

    # max_depth defaults to 0: following links needs a depth to follow them to.
    default_depth = await service._tool_web_scrape(
        {**base, "max_pages": 5}, test_user.id, db_session
    )
    assert default_depth["total_pages"] == 1


async def test_scrape_leaves_the_domain_only_when_allowed(
    service, db_session, test_user, web
):
    _site(web)
    params = {
        "url": "https://example.com/",
        "follow_links": True,
        "max_pages": 5,
        "max_depth": 1,
    }

    await service._tool_web_scrape(params, test_user.id, db_session)
    assert "https://other.example.org/x" not in web.requests

    result = await service._tool_web_scrape(
        {**params, "same_domain_only": False}, test_user.id, db_session
    )
    assert "https://other.example.org/x" in [p["url"] for p in result["pages"]]


async def test_scrape_follows_links_even_when_not_returning_them(
    service, db_session, test_user, web
):
    _site(web)

    result = await service._tool_web_scrape(
        {
            "url": "https://example.com/",
            "follow_links": True,
            "include_links": False,
            "max_pages": 5,
            "max_depth": 1,
        },
        test_user.id,
        db_session,
    )

    assert result["total_pages"] == 2


async def test_scrape_reports_a_page_that_failed(service, db_session, test_user, web):
    web.page("https://example.com/", "boom", status=500)

    result = await service._tool_web_scrape(
        {"url": "https://example.com/"}, test_user.id, db_session
    )

    assert result["pages"] == []
    assert result["total_pages"] == 0
    assert result["errors"][0]["url"] == "https://example.com/"
    assert "500" in result["errors"][0]["error"]


BLOCKED_URLS = [
    "http://127.0.0.1/admin",
    "http://localhost/admin",
    "http://app.localhost/",
    "http://printer.local/",
    "http://10.0.0.5/secrets",
    "http://192.168.1.1/",
    "http://169.254.169.254/latest/meta-data/",
    "http://[::1]/",
    "http://0.0.0.0/",
    "http://intranet.corp/wiki",
    "http://rebind.example.com/",
    "http://metadata.example.com/",
    "http://unresolvable.invalid/",
    "ftp://example.com/file",
    "file:///etc/passwd",
    "http://user:pw@example.com/",
    "example.com/no-scheme",
]


@pytest.mark.parametrize("url", BLOCKED_URLS)
async def test_scrape_never_requests_a_private_or_malformed_url(
    service, db_session, test_user, web, url
):
    result, exc = await _attempt(
        service._tool_web_scrape({"url": url}, test_user.id, db_session)
    )

    assert web.requests == []
    assert _produced_nothing(result, exc)


@pytest.mark.parametrize(
    "params",
    [
        {"url": "http://127.0.0.1/admin"},
        {"url": "ftp://example.com/file"},
        {"url": "https://example.com/", "max_pages": 26},
        {"url": "https://example.com/", "max_pages": 0},
        {"url": "https://example.com/", "max_depth": 6},
        {"url": "https://example.com/", "max_content_chars": 0},
        {"url": "https://example.com/", "max_pages": "many"},
    ],
)
async def test_scrape_refusals_are_error_results_not_exceptions(
    service, db_session, test_user, web, params
):
    _site(web)

    result = await service._tool_web_scrape(params, test_user.id, db_session)

    assert "error" in result


@pytest.mark.parametrize(
    "params",
    [
        {"max_pages": 26},
        {"max_pages": 0},
        {"max_depth": 6},
        {"max_depth": -1},
        {"max_content_chars": 500_001},
    ],
)
async def test_scrape_refuses_bounds_beyond_the_declared_maximum(
    service, db_session, test_user, web, params
):
    _site(web)

    result, exc = await _attempt(
        service._tool_web_scrape(
            {"url": "https://example.com/", **params}, test_user.id, db_session
        )
    )

    assert web.requests == []
    assert _produced_nothing(result, exc)


async def test_scrape_does_not_follow_a_redirect_to_a_private_address(
    service, db_session, test_user, web
):
    web.page(
        "https://example.com/",
        "",
        status=302,
        headers={"location": "http://169.254.169.254/latest/meta-data/"},
    )
    web.page("http://169.254.169.254/latest/meta-data/", "secret-credentials")

    result = await service._tool_web_scrape(
        {"url": "https://example.com/"}, test_user.id, db_session
    )

    assert web.requests == ["https://example.com/"]
    assert result["pages"] == []
    assert "secret-credentials" not in repr(result)
    assert result["errors"]


async def test_scrape_does_not_crawl_into_a_private_address(
    service, db_session, test_user, web
):
    web.page(
        "https://example.com/",
        '<html><body><a href="http://10.0.0.5/secrets">x</a>'
        '<a href="http://intranet.corp/wiki">y</a></body></html>',
    )
    web.page("http://10.0.0.5/secrets", "secret")
    web.page("http://intranet.corp/wiki", "secret")

    await service._tool_web_scrape(
        {
            "url": "https://example.com/",
            "follow_links": True,
            "max_pages": 5,
            "max_depth": 2,
            "same_domain_only": False,
        },
        test_user.id,
        db_session,
    )

    assert web.requests == ["https://example.com/"]


async def test_only_an_admin_may_ask_for_private_networks(
    service, db_session, test_user, admin_user, web
):
    web.page("http://intranet.corp/wiki", "<html><body>Internal wiki.</body></html>")
    params = {"url": "http://intranet.corp/wiki", "allow_private_networks": True}

    refused = await service._tool_web_scrape(params, test_user.id, db_session)

    assert "admin" in refused["error"]
    assert web.requests == []

    allowed = await service._tool_web_scrape(params, admin_user.id, db_session)

    assert "Internal wiki." in allowed["pages"][0]["content"]


async def test_even_an_admin_cannot_reach_loopback_or_metadata(
    service, db_session, admin_user, web
):
    for url in ("http://127.0.0.1/", "http://169.254.169.254/", "http://localhost/"):
        result, exc = await _attempt(
            service._tool_web_scrape(
                {"url": url, "allow_private_networks": True},
                admin_user.id,
                db_session,
            )
        )
        assert _produced_nothing(result, exc)

    assert web.requests == []


async def test_an_active_web_source_allowlists_its_own_hosts_only(
    service, db_session, test_user, web
):
    await _source(
        db_session,
        name="Intranet",
        source_type="web",
        config={"allowed_domains": ["intranet.corp"]},
    )
    await _source(
        db_session,
        name="Disabled",
        source_type="web",
        config={"base_urls": ["http://db.corp/"]},
        is_active=False,
    )
    web.page("http://intranet.corp/wiki", "<html><body>Internal wiki.</body></html>")
    web.page("http://wiki.intranet.corp/", "<html><body>Sub wiki.</body></html>")
    web.page("http://db.corp/", "<html><body>Database console.</body></html>")

    exact = await service._tool_web_scrape(
        {"url": "http://intranet.corp/wiki"}, test_user.id, db_session
    )
    sub = await service._tool_web_scrape(
        {"url": "http://wiki.intranet.corp/"}, test_user.id, db_session
    )
    assert "Internal wiki." in exact["pages"][0]["content"]
    assert "Sub wiki." in sub["pages"][0]["content"]

    result, exc = await _attempt(
        service._tool_web_scrape({"url": "http://db.corp/"}, test_user.id, db_session)
    )
    assert "http://db.corp/" not in web.requests
    assert _produced_nothing(result, exc)


async def test_an_allowlisted_page_does_not_open_the_rest_of_the_private_network(
    service, db_session, test_user, web
):
    await _source(
        db_session,
        name="Intranet",
        source_type="web",
        config={"allowed_domains": ["intranet.corp"]},
    )
    web.page(
        "http://intranet.corp/wiki",
        '<html><body>Wiki. <a href="http://db.corp/">db</a></body></html>',
    )
    web.page("http://db.corp/", "<html><body>Database console.</body></html>")

    result = await service._tool_web_scrape(
        {
            "url": "http://intranet.corp/wiki",
            "follow_links": True,
            "max_pages": 5,
            "max_depth": 1,
            "same_domain_only": False,
        },
        test_user.id,
        db_session,
    )

    assert "http://db.corp/" not in web.requests
    assert "Database console." not in repr(result)


async def test_scrape_writes_nothing_to_the_knowledge_base(
    service, db_session, test_user, web, edges
):
    _site(web)

    await service._tool_web_scrape(
        {"url": "https://example.com/"}, test_user.id, db_session
    )

    assert await _titles(db_session) == []
    assert edges.vector_added == []


# --------------------------------------------------------------------------
# ingest_url
# --------------------------------------------------------------------------


async def _ingested(db):
    query = select(Document).order_by(Document.title)
    result = await db.execute(query.execution_options(populate_existing=True))
    return result.scalars().all()


async def test_ingest_refuses_an_unknown_user(service, db_session, web):
    result = await service._tool_ingest_url(
        {"url": "https://example.com/"}, uuid4(), db_session
    )

    assert result == {"error": "User not found"}
    assert web.requests == []


@pytest.mark.parametrize("params", [{}, {"url": ""}, {"url": None}, {"url": "   "}])
async def test_ingest_requires_a_url(service, db_session, test_user, web, params):
    result = await service._tool_ingest_url(params, test_user.id, db_session)

    assert "url" in result["error"]
    assert web.requests == []
    assert await _titles(db_session) == []


async def test_ingest_refuses_an_unknown_mode(service, db_session, test_user, web):
    _site(web)

    result = await service._tool_ingest_url(
        {"url": "https://example.com/", "ingest_mode": "torrent"},
        test_user.id,
        db_session,
    )

    assert "ingest_mode" in result["error"]
    assert web.requests == []
    assert await _titles(db_session) == []


async def test_ingest_stores_the_page_as_an_indexed_document(
    service, db_session, test_user, web, edges
):
    _site(web)

    result = await service._tool_ingest_url(
        {"url": "https://example.com/", "tags": ["cache"]}, test_user.id, db_session
    )

    assert result["action"] == "ingested"
    assert result["errors"] == []
    assert result["total_pages_scraped"] == 1
    assert web.requests == ["https://example.com/"]
    assert web.scraper_kwargs == [{"enforce_network_safety": True}]
    (doc,) = await _ingested(db_session)
    assert result["created"] == [
        {
            "document_id": str(doc.id),
            "url": "https://example.com/",
            "title": "Stride prefetching",
        }
    ]
    assert doc.title == "Stride prefetching"
    assert "Stride prefetchers track deltas." in doc.content
    assert doc.url == "https://example.com/"
    assert doc.source_identifier == "https://example.com/"
    assert doc.tags == ["cache"]
    assert doc.author == "Test User"
    assert doc.extra_metadata["origin"] == "url_ingest"
    assert doc.content_hash == hashlib.sha256(doc.content.encode()).hexdigest()
    source = await db_session.get(DocumentSource, doc.source_id)
    assert (source.name, source.source_type) == ("URL Ingest", "web")
    # Inactive, so ad-hoc ingestion never becomes a scheduled crawl or an
    # allowlist entry for private scraping.
    assert source.is_active is False
    assert await _chunk_count(db_session, doc.id) >= 1
    assert [doc_id for doc_id, _ in edges.vector_added] == [doc.id]
    assert doc.is_processed is True


async def test_ingest_title_overrides_the_page_title(
    service, db_session, test_user, web
):
    _site(web)

    result = await service._tool_ingest_url(
        {"url": "https://example.com/", "title": "My name for it"},
        test_user.id,
        db_session,
    )

    (doc,) = await _ingested(db_session)
    assert doc.title == "My name for it"
    assert result["created"][0]["title"] == "My name for it"


async def test_ingest_honours_max_content_chars(service, db_session, test_user, web):
    web.page("https://example.com/", "<html><body>" + "x" * 5000 + "</body></html>")

    await service._tool_ingest_url(
        {"url": "https://example.com/", "max_content_chars": 500},
        test_user.id,
        db_session,
    )

    (doc,) = await _ingested(db_session)
    assert len(doc.content) <= 500


async def test_ingesting_an_unchanged_page_again_creates_nothing(
    service, db_session, test_user, web, edges
):
    _site(web)
    params = {"url": "https://example.com/"}

    await service._tool_ingest_url(params, test_user.id, db_session)
    again = await service._tool_ingest_url(params, test_user.id, db_session)

    (doc,) = await _ingested(db_session)
    assert again["created"] == []
    assert again["updated"] == []
    assert again["skipped"] == [
        {
            "document_id": str(doc.id),
            "url": "https://example.com/",
            "reason": "unchanged",
        }
    ]
    assert len(edges.vector_added) == 1


async def test_ingesting_a_changed_page_updates_the_same_document(
    service, db_session, test_user, web, edges
):
    _site(web)
    params = {"url": "https://example.com/", "tags": ["v1"]}
    await service._tool_ingest_url(params, test_user.id, db_session)
    (before,) = await _ingested(db_session)
    before_id = before.id
    web.page(
        "https://example.com/",
        "<html><head><title>Rewritten</title></head><body>New words about "
        "tagged geometric branch predictors and their history.</body></html>",
    )

    result = await service._tool_ingest_url(
        {**params, "tags": ["v2"]}, test_user.id, db_session
    )

    (doc,) = await _ingested(db_session)
    assert doc.id == before_id
    assert result["created"] == []
    assert [u["document_id"] for u in result["updated"]] == [str(before_id)]
    assert doc.title == "Rewritten"
    assert "tagged geometric branch predictors" in doc.content
    assert "Stride prefetchers" not in doc.content
    assert doc.tags == ["v2"]
    # The old chunks are replaced, in the database and in the vector store.
    chunks = (
        (
            await db_session.execute(
                select(DocumentChunk.content).where(DocumentChunk.document_id == doc.id)
            )
        )
        .scalars()
        .all()
    )
    assert chunks and all("Stride" not in c for c in chunks)
    assert before_id in edges.vector_deleted
    assert len(edges.vector_added) == 2


async def test_ingest_reports_an_unreachable_page_and_stores_nothing(
    service, db_session, test_user, web
):
    web.page("https://example.com/", "boom", status=500)

    result = await service._tool_ingest_url(
        {"url": "https://example.com/"}, test_user.id, db_session
    )

    assert "No pages scraped" in result["error"]
    assert result.get("action") != "ingested"
    assert await _titles(db_session) == []


@pytest.mark.parametrize("url", BLOCKED_URLS)
async def test_ingest_never_requests_a_private_or_malformed_url(
    service, db_session, test_user, web, url
):
    result, exc = await _attempt(
        service._tool_ingest_url({"url": url}, test_user.id, db_session)
    )

    assert web.requests == []
    assert exc is not None or "error" in result
    assert await _titles(db_session) == []


@pytest.mark.parametrize(
    "params",
    [
        {"url": "http://127.0.0.1/admin"},
        {"url": "file:///etc/passwd"},
        {"url": "https://example.com/", "max_pages": 26},
        {"url": "https://example.com/", "max_depth": 6},
    ],
)
async def test_ingest_refusals_are_error_results_not_exceptions(
    service, db_session, test_user, web, params
):
    _site(web)

    result = await service._tool_ingest_url(params, test_user.id, db_session)

    assert "error" in result


async def test_only_an_admin_may_ingest_from_private_networks(
    service, db_session, test_user, admin_user, web
):
    web.page("http://intranet.corp/wiki", "<html><body>Internal wiki.</body></html>")
    params = {"url": "http://intranet.corp/wiki", "allow_private_networks": True}

    refused = await service._tool_ingest_url(params, test_user.id, db_session)

    assert "admin" in refused["error"]
    assert web.requests == []
    assert await _titles(db_session) == []

    allowed = await service._tool_ingest_url(params, admin_user.id, db_session)

    assert allowed["action"] == "ingested"
    (doc,) = await _ingested(db_session)
    assert "Internal wiki." in doc.content
    assert doc.author == "Admin User"


async def test_ingest_does_not_follow_a_redirect_to_a_private_address(
    service, db_session, test_user, web
):
    web.page(
        "https://example.com/",
        "",
        status=302,
        headers={"location": "http://10.0.0.5/secrets"},
    )
    web.page("http://10.0.0.5/secrets", "<html><body>secret</body></html>")

    result = await service._tool_ingest_url(
        {"url": "https://example.com/"}, test_user.id, db_session
    )

    assert web.requests == ["https://example.com/"]
    assert "error" in result
    assert await _titles(db_session) == []


async def test_ingest_follows_links_into_one_combined_document(
    service, db_session, test_user, web
):
    _site(web)

    result = await service._tool_ingest_url(
        {
            "url": "https://example.com/",
            "follow_links": True,
            "max_pages": 5,
            "max_depth": 2,
        },
        test_user.id,
        db_session,
    )

    assert result["total_pages_scraped"] == 3
    (doc,) = await _ingested(db_session)
    assert "Second page body." in doc.content
    assert "Third page body." in doc.content


async def test_ingest_one_document_per_page(service, db_session, test_user, web):
    _site(web)

    result = await service._tool_ingest_url(
        {
            "url": "https://example.com/",
            "follow_links": True,
            "max_pages": 5,
            "max_depth": 1,
            "one_document_per_page": True,
            "tags": ["crawl"],
        },
        test_user.id,
        db_session,
    )

    docs = await _ingested(db_session)
    assert [d.title for d in docs] == ["Page two", "Stride prefetching"]
    assert sorted(d.url for d in docs) == [
        "https://example.com/",
        "https://example.com/two",
    ]
    assert all(d.tags == ["crawl"] for d in docs)
    assert len(result["created"]) == 2


class _FakeYoutubeDL:
    """yt_dlp's entry point: records how it was asked, then fails to download."""

    seen = []

    def __init__(self, opts):
        type(self).seen.append(opts)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=True):
        raise RuntimeError("video unavailable")


@pytest.fixture
def youtube(monkeypatch):
    _FakeYoutubeDL.seen = []
    module = types.ModuleType("yt_dlp")
    module.YoutubeDL = _FakeYoutubeDL
    monkeypatch.setitem(sys.modules, "yt_dlp", module)
    return _FakeYoutubeDL


async def test_ingest_routes_youtube_urls_to_the_downloader_and_reports_failure(
    service, db_session, test_user, web, youtube
):
    result = await service._tool_ingest_url(
        {"url": "https://www.youtube.com/watch?v=abc"}, test_user.id, db_session
    )

    assert len(youtube.seen) == 1
    assert web.requests == []
    assert "YouTube download failed" in result["error"]
    assert await _titles(db_session) == []


async def test_ingest_mode_decides_the_route_not_the_hostname(
    service, db_session, test_user, web, youtube
):
    web.page(
        "https://www.youtube.com/watch?v=abc",
        "<html><head><title>Watch</title></head><body>Transcript text.</body></html>",
    )
    _site(web)

    as_web = await service._tool_ingest_url(
        {"url": "https://www.youtube.com/watch?v=abc", "ingest_mode": "web"},
        test_user.id,
        db_session,
    )
    assert youtube.seen == []
    assert as_web["action"] == "ingested"

    as_video = await service._tool_ingest_url(
        {"url": "https://example.com/", "ingest_mode": "youtube"},
        test_user.id,
        db_session,
    )
    assert len(youtube.seen) == 1
    assert "error" in as_video
    assert "https://example.com/" not in web.requests


async def test_youtube_audio_only_chooses_what_is_downloaded(
    service, db_session, test_user, web, youtube
):
    url = "https://www.youtube.com/watch?v=abc"

    await service._tool_ingest_url({"url": url}, test_user.id, db_session)
    await service._tool_ingest_url(
        {"url": url, "youtube_audio_only": False}, test_user.id, db_session
    )

    audio, video = (opts["format"] for opts in youtube.seen)
    assert "bestaudio" in audio
    assert "bestaudio" not in video
    assert all(opts["noplaylist"] is True for opts in youtube.seen)


# --------------------------------------------------------------------------
# Workflow events: the editor offers document.uploaded / processed / deleted
# triggers, and nothing published them, so no event workflow ever ran.
# --------------------------------------------------------------------------


@pytest.fixture
def published(monkeypatch):
    import inspect

    from app.tasks import workflow_tasks

    # The message itself: what reaches the broker, whichever helper sent it.
    task = workflow_tasks.trigger_event_workflow
    calls = []

    def publish(*args, **kwargs):
        bound = inspect.signature(task.run).bind(*args, **kwargs)
        calls.append(dict(bound.arguments))

    monkeypatch.setattr(task, "delay", publish)
    return calls


async def test_a_users_delete_publishes_document_deleted(
    db_session, test_user, edges, published
):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Gone")

    assert await _document_service().delete_document(
        doc.id, db_session, user_id=test_user.id
    )

    assert published == [
        {
            "event_name": "document.deleted",
            "event_data": {"document_id": str(doc.id), "title": "Gone"},
            "user_id": str(test_user.id),
        }
    ]


async def test_a_delete_with_no_user_publishes_nothing(db_session, edges, published):
    # A source sync removing documents belongs to nobody's workflows.
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Gone")

    assert await _document_service().delete_document(doc.id, db_session)
    assert published == []


async def test_processing_publishes_document_processed(
    db_session, test_user, edges, published
):
    source = await _source(db_session)
    doc = await _doc(
        db_session, source, "Fresh", content="enough words " * 40, chunks=0
    )

    await _document_service()._process_document_async(
        doc, db_session, user_id=test_user.id
    )

    assert [c["event_name"] for c in published] == ["document.processed"]
    assert published[0]["event_data"]["document_id"] == str(doc.id)


async def test_a_broker_that_refuses_the_event_does_not_fail_the_delete(
    db_session, test_user, edges, monkeypatch
):
    from app.tasks import workflow_tasks

    def refuse(*args, **kwargs):
        raise ConnectionError("broker down")

    monkeypatch.setattr(workflow_tasks.trigger_event_workflow, "delay", refuse)
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Gone")

    assert await _document_service().delete_document(
        doc.id, db_session, user_id=test_user.id
    )
