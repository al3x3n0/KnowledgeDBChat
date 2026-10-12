"""One chat turn through `POST /chat/sessions/{id}/messages`, with a model
that answers and a store that returns chunks.

`test_chat.py` accepts any non-empty assistant reply, and the service answers
every failure with an apology stored as an assistant message -- so that test
passes whether or not a turn works. Here the reply must be the model's, the
retrieved chunk must reach the prompt and the sources, and the session must
hold what was said once.
"""

import hashlib
import inspect
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api.endpoints import chat as chat_endpoints
from app.models.document import Document, DocumentSource
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStoreService

pytestmark = pytest.mark.unit

_TRACE = inspect.signature(VectorStoreService.search_with_trace)
_GENERATE = inspect.signature(LLMService.generate_response)


class _Store:
    provider = "fake"

    def __init__(self):
        self.results = []
        self.queries = []

    async def initialize(self, embedding_model=None, background=False):
        return None

    async def search_with_trace(self, *args, **kwargs):
        bound = _TRACE.bind(self, *args, **kwargs)
        self.queries.append(bound.arguments["query"])
        return list(self.results), {"mode": "fake"}


@pytest.fixture
def turn(monkeypatch):
    service = chat_endpoints.chat_service
    store = _Store()
    prompts = []

    async def generate_response(self, *args, **kwargs):
        bound = _GENERATE.bind(self, *args, **kwargs)
        bound.apply_defaults()
        # The answer is one model call among several a turn may make.
        if bound.arguments["task_type"] == "chat":
            prompts.append(dict(bound.arguments))
        return "Prefetchers guess the next address."

    async def no_variations(query, **kwargs):
        return [query]

    async def no_draft(*args, **kwargs):
        return None

    monkeypatch.setattr(service, "vector_store", store)
    monkeypatch.setattr(service, "_vector_store_initialized", False)
    monkeypatch.setattr(LLMService, "generate_response", generate_response)
    monkeypatch.setattr(
        service.query_processor, "generate_multi_queries", no_variations
    )
    monkeypatch.setattr(
        "app.services.agent_campaign_intent.draft_from_message", no_draft
    )
    return type("Turn", (), {"store": store, "prompts": prompts})


def _session(client, headers):
    response = client.post(
        "/api/v1/chat/sessions", json={"title": "Caches"}, headers=headers
    )
    return response.json()["id"]


def _send(client, headers, session_id, text):
    return client.post(
        f"/api/v1/chat/sessions/{session_id}/messages",
        json={"content": text},
        headers=headers,
    )


def test_the_reply_is_the_models_and_not_an_apology(
    client: TestClient, auth_headers, turn
):
    session_id = _session(client, auth_headers)

    response = _send(client, auth_headers, session_id, "What does a prefetcher do?")

    assert response.status_code == 200
    body = response.json()
    assert body["content"] == "Prefetchers guess the next address."
    assert body["message_type"] != "error"
    assert not body.get("processing_error")
    assert turn.prompts[0]["query"] == "What does a prefetcher do?"


def test_a_message_is_stored_once(client: TestClient, auth_headers, turn):
    """The route stored the user's message and so did the service."""
    session_id = _session(client, auth_headers)

    _send(client, auth_headers, session_id, "What does a prefetcher do?")

    session = client.get(
        f"/api/v1/chat/sessions/{session_id}", headers=auth_headers
    ).json()
    assert [(m["role"], m["content"]) for m in session["messages"]] == [
        ("user", "What does a prefetcher do?"),
        ("assistant", "Prefetchers guess the next address."),
    ]


def test_the_second_turn_is_given_the_first_and_not_itself(
    client: TestClient, auth_headers, turn
):
    session_id = _session(client, auth_headers)

    _send(client, auth_headers, session_id, "What does a prefetcher do?")
    _send(client, auth_headers, session_id, "And a stride one?")

    history = turn.prompts[1]["conversation_history"]
    assert history.count("What does a prefetcher do?") == 1
    assert "Prefetchers guess the next address." in history
    assert "And a stride one?" not in history


def test_a_retrieved_chunk_reaches_the_prompt_and_the_sources(
    client: TestClient, auth_headers, turn, db_session, event_loop
):
    async def make_document():
        source = DocumentSource(
            name=f"s-{uuid4().hex[:6]}", source_type="file", config={}
        )
        db_session.add(source)
        await db_session.commit()
        doc = Document(
            title="Prefetching survey",
            content="A stride prefetcher tracks deltas.",
            content_hash=hashlib.sha256(b"survey").hexdigest(),
            source_id=source.id,
            source_identifier=f"test:{uuid4().hex}",
        )
        db_session.add(doc)
        await db_session.commit()
        return str(doc.id)

    doc_id = event_loop.run_until_complete(make_document())
    turn.store.results = [
        {
            "id": "chunk-1",
            "content": "A stride prefetcher tracks deltas.",
            "page_content": "A stride prefetcher tracks deltas.",
            "score": 0.9,
            "metadata": {
                "document_id": doc_id,
                "title": "Prefetching survey",
                "source": "file",
                "chunk_index": 0,
            },
        }
    ]
    session_id = _session(client, auth_headers)

    body = _send(client, auth_headers, session_id, "How does stride work?").json()

    assert body["message_type"] != "error", body
    assert "stride prefetcher tracks deltas" in turn.prompts[0]["context"]
    assert [s["title"] for s in body["source_documents"]] == ["Prefetching survey"]
    assert body["source_documents"][0]["id"] == doc_id
