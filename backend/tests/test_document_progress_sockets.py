"""The document progress sockets, which now share one relay loop.

Four handlers (transcription, summarization, source ingestion, URL ingestion)
each carried the same connect / ping-pong / disconnect loop. These drive the
real routes through the test client; only authentication, the document
lookup and Redis are replaced.
"""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from uuid import uuid4

import pytest
from starlette.websockets import WebSocketDisconnect

from app.api.endpoints import documents as documents_endpoint
from app.utils.websocket_manager import websocket_manager

pytestmark = pytest.mark.unit


def _eventually(condition, timeout=5.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return bool(condition())


@pytest.fixture
def signed_in(monkeypatch, test_user):
    async def _auth(websocket):
        await websocket.accept()
        return test_user

    monkeypatch.setattr(documents_endpoint, "require_websocket_auth", _auth)
    return test_user


@pytest.fixture
def documents(monkeypatch):
    known = set()

    async def get_document(document_id, db):
        return SimpleNamespace(id=document_id) if document_id in known else None

    @asynccontextmanager
    async def session():
        yield None

    monkeypatch.setattr(
        documents_endpoint.document_service, "get_document", get_document
    )
    monkeypatch.setattr(documents_endpoint, "AsyncSessionLocal", session)
    return known


@pytest.mark.parametrize("stream", ["transcription", "summarization"])
def test_a_known_document_answers_pings_and_lets_go(
    client, signed_in, documents, stream
):
    document_id = uuid4()
    documents.add(document_id)

    with client.websocket_connect(
        f"/api/v1/documents/{document_id}/{stream}-progress"
    ) as websocket:
        websocket.send_text("ping")
        assert websocket.receive_text() == "pong"
        assert _eventually(
            lambda: websocket_manager.active_connections.get(str(document_id))
        )

    # The server unregisters after the client has gone, on its own thread:
    # checked at once this raced, and lost on a busy CI runner.
    assert _eventually(
        lambda: not websocket_manager.active_connections.get(str(document_id))
    )


@pytest.mark.parametrize("stream", ["transcription", "summarization"])
def test_an_unknown_document_is_closed(client, signed_in, documents, stream):
    with client.websocket_connect(
        f"/api/v1/documents/{uuid4()}/{stream}-progress"
    ) as websocket:
        with pytest.raises(WebSocketDisconnect) as closed:
            websocket.receive_text()
    assert closed.value.code == 1008


class _Redis:
    def __init__(self, owner):
        self.owner = owner

    async def get(self, key):
        return self.owner.encode() if self.owner else None


def _url_ingest_owner(monkeypatch, owner):
    async def get_redis_client():
        return _Redis(owner)

    from app.core import cache

    monkeypatch.setattr(cache, "get_redis_client", get_redis_client)


def test_url_ingest_progress_reaches_its_owner(client, signed_in, monkeypatch):
    _url_ingest_owner(monkeypatch, str(signed_in.id))

    with client.websocket_connect("/api/v1/documents/ingest-url/job-1/progress") as ws:
        ws.send_text("ping")
        assert ws.receive_text() == "pong"


def test_url_ingest_progress_refuses_a_stranger(client, signed_in, monkeypatch):
    _url_ingest_owner(monkeypatch, str(uuid4()))

    with client.websocket_connect("/api/v1/documents/ingest-url/job-1/progress") as ws:
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()
    assert closed.value.code == 1008


@pytest.fixture
def sources(monkeypatch):
    known = {}

    class _Session:
        async def get(self, model, source_id):
            return known.get(source_id)

    @asynccontextmanager
    async def session():
        yield _Session()

    monkeypatch.setattr(documents_endpoint, "AsyncSessionLocal", session)
    return known


def test_any_signed_in_user_may_watch_a_source_ingest(client, signed_in, sources):
    # Sources are shared: the requester being someone else does not matter.
    source_id = uuid4()
    sources[source_id] = SimpleNamespace(
        id=source_id, config={"requested_by": "someone_else"}
    )

    with client.websocket_connect(
        f"/api/v1/documents/sources/{source_id}/ingestion-progress"
    ) as websocket:
        websocket.send_text("ping")
        assert websocket.receive_text() == "pong"


def test_an_unknown_source_is_closed(client, signed_in, sources):
    with client.websocket_connect(
        f"/api/v1/documents/sources/{uuid4()}/ingestion-progress"
    ) as websocket:
        with pytest.raises(WebSocketDisconnect) as closed:
            websocket.receive_text()
    assert closed.value.code == 1008
