"""One retrieval for every answer written from the knowledge base.

The chat page rewrote the question, dropped irrelevant passages, fitted the
context to its budget and consulted the knowledge graph. The agent's
`answer_question` tool and the MCP chat tool each searched once and used
whatever came back, so the same question got a worse answer there -- from the
default model, whatever the person had chosen. `services/rag_retrieval.py` is
the chat page's retrieval, moved, and all three call it.

Real rows; the vector store and the model are the replaced edges, each bound
against its real signature.
"""

import hashlib
import inspect
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.core.config import settings
from app.models.document import Document, DocumentSource
from app.models.retrieval_trace import RetrievalTrace
from app.services import rag_retrieval
from app.services.agent_service import AgentService
from app.services.chat_service import ChatService
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStoreService

pytestmark = pytest.mark.unit

_TRACE = inspect.signature(VectorStoreService.search_with_trace)
_GENERATE = inspect.signature(LLMService.generate_response)


class _Store:
    provider = "fake"

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    async def initialize(self, embedding_model=None, background=False):
        return None

    async def search_with_trace(self, *args, **kwargs):
        bound = _TRACE.bind(self, *args, **kwargs)
        bound.apply_defaults()
        call = dict(bound.arguments)
        self.calls.append(call)
        wanted = call["document_ids"]
        hits = [
            h
            for h in self.hits
            if wanted is None or h["metadata"]["document_id"] in set(wanted)
        ]
        return hits[: call["limit"]], {"mode": "fake", "returned": len(hits)}


def _hit(doc, score, text=None):
    return {
        "id": f"chunk-{uuid4().hex[:6]}",
        "content": text or f"{doc.title} says something specific.",
        "page_content": text or f"{doc.title} says something specific.",
        "score": score,
        "metadata": {
            "document_id": str(doc.id),
            "title": doc.title,
            "source": "file",
            "chunk_index": 0,
        },
    }


async def _documents(db, *titles):
    source = DocumentSource(name=f"s-{uuid4().hex[:6]}", source_type="file", config={})
    db.add(source)
    await db.commit()
    docs = []
    for title in titles:
        doc = Document(
            title=title,
            content=f"{title} body",
            content_hash=hashlib.sha256(title.encode()).hexdigest(),
            source_id=source.id,
            source_identifier=f"test:{uuid4().hex}",
        )
        db.add(doc)
        docs.append(doc)
    await db.commit()
    return docs


@pytest.fixture
def model(monkeypatch):
    calls = []

    async def generate_response(self, *args, **kwargs):
        bound = _GENERATE.bind(self, *args, **kwargs)
        bound.apply_defaults()
        if bound.arguments["task_type"] == "chat":
            calls.append(dict(bound.arguments))
        return " The answer. "

    async def one_phrasing(query, **kwargs):
        return [query]

    monkeypatch.setattr(LLMService, "generate_response", generate_response)
    monkeypatch.setattr(
        rag_retrieval._QUERY_PROCESSOR, "generate_multi_queries", one_phrasing
    )
    monkeypatch.setattr(settings, "RAG_KG_CONTEXT_ENABLED", False)
    return calls


# --------------------------------------------------------------------------
# the agent's answer tool
# --------------------------------------------------------------------------


def _agent(store):
    service = AgentService()
    service.vector_store = store
    service._vector_store_initialized = True
    return service


async def test_the_agent_answers_from_relevant_passages_only(
    db_session, test_user, model
):
    good, noise = await _documents(db_session, "Prefetching survey", "Lunch menu")
    store = _Store([_hit(good, 0.82), _hit(noise, 0.05)])

    result = await _agent(store)._tool_answer_question(
        {"question": "How does a stride prefetcher work?"}, test_user.id, db_session
    )

    assert result["answer"] == "The answer."
    # What is below the relevance floor is neither context nor a source. It
    # used to be pasted in beside the real passage.
    assert "Prefetching survey says" in model[0]["context"]
    assert "Lunch menu" not in model[0]["context"]
    assert [s["title"] for s in result["sources"]] == ["Prefetching survey"]
    assert result["confidence"] == "high"
    # The question goes as the question, not buried in a hand-built prompt.
    assert model[0]["query"] == "How does a stride prefetcher work?"


async def test_the_agent_answers_with_the_persons_own_model(
    db_session, test_user, model, monkeypatch
):
    from app.services import llm_service
    from app.services.llm_service import UserLLMSettings

    chosen = UserLLMSettings(provider="anthropic", model="their-model")

    async def settings_for(db, user_id):
        return chosen

    monkeypatch.setattr(llm_service, "load_user_llm_settings", settings_for)
    (doc,) = await _documents(db_session, "Paper")

    await _agent(_Store([_hit(doc, 0.9)]))._tool_answer_question(
        {"question": "What does it say?"}, test_user.id, db_session
    )

    assert model[0]["user_settings"] is chosen


async def test_the_agent_says_so_when_nothing_relevant_is_found(
    db_session, test_user, model
):
    (noise,) = await _documents(db_session, "Lunch menu")

    result = await _agent(_Store([_hit(noise, 0.05)]))._tool_answer_question(
        {"question": "How does a stride prefetcher work?"}, test_user.id, db_session
    )

    assert "couldn't find" in result["answer"]
    assert result["sources"] == [] and result["confidence"] == "low"
    assert model == []


async def test_a_document_is_cited_once_however_many_passages_it_gave(
    db_session, test_user, model
):
    (doc,) = await _documents(db_session, "Long paper")
    store = _Store(
        [_hit(doc, 0.9, "First passage."), _hit(doc, 0.8, "Second passage.")]
    )

    result = await _agent(store)._tool_answer_question(
        {"question": "What does it say?"}, test_user.id, db_session
    )

    assert len(result["sources"]) == 1 and result["source_count"] == 1
    assert "First passage." in model[0]["context"]
    assert "Second passage." in model[0]["context"]


async def test_the_knowledge_graph_reaches_the_agents_answer_too(
    db_session, test_user, model, monkeypatch
):
    from app.services.knowledge_graph_service import KnowledgeGraphService

    async def pack(self, db, search_results, **kwargs):
        return {"kg_context": "Stride -> is_a -> Prefetcher", "entities": []}

    monkeypatch.setattr(settings, "RAG_KG_CONTEXT_ENABLED", True)
    monkeypatch.setattr(KnowledgeGraphService, "build_chat_context_pack", pack)
    (doc,) = await _documents(db_session, "Paper")

    await _agent(_Store([_hit(doc, 0.9)]))._tool_answer_question(
        {"question": "What is stride?"}, test_user.id, db_session
    )

    assert model[0]["kg_context"] == "Stride -> is_a -> Prefetcher"


# --------------------------------------------------------------------------
# the chat page, which this was moved out of
# --------------------------------------------------------------------------


async def _chat(db, user, store, question):
    service = ChatService()
    service.vector_store = store
    service._vector_store_initialized = True

    async def one_phrasing(query, **kwargs):
        return [query]

    service.query_processor.generate_multi_queries = one_phrasing
    session = await service.create_session(user.id, db, title="Caches")
    message = await service.generate_response(
        session_id=session.id, query=question, user_id=user.id, db=db
    )
    return session, message


async def test_a_chat_turn_records_how_it_retrieved(db_session, test_user, model):
    good, noise = await _documents(db_session, "Prefetching survey", "Lunch menu")
    store = _Store([_hit(good, 0.82), _hit(noise, 0.05)])

    session, message = await _chat(
        db_session, test_user, store, "How does stride work?"
    )

    assert message.message_type != "error", message.processing_error
    assert [s["title"] for s in message.source_documents] == ["Prefetching survey"]
    trace = (
        await db_session.execute(
            select(RetrievalTrace).where(
                RetrievalTrace.id == message.retrieval_trace_id
            )
        )
    ).scalar_one()
    assert trace.query == "How does stride work?"
    assert trace.session_id == session.id
    assert [r["title"] for r in trace.trace["selected_results"]] == [
        "Prefetching survey"
    ]
    # Query expansion is on by default, so the search is recorded per phrasing.
    searches = trace.trace["multi_query"]["traces"]
    assert [t["query"] for t in searches] == trace.trace["multi_query"]["queries"]
    assert searches[0]["trace"]["mode"] == "fake"
    assert "context_metrics" in trace.trace


async def test_a_chat_turn_passes_on_what_the_knowledge_graph_says(
    db_session, test_user, model, monkeypatch
):
    from app.services.knowledge_graph_service import KnowledgeGraphService

    async def pack(self, db, search_results, **kwargs):
        return {"kg_context": "Stride -> is_a -> Prefetcher", "entities": [{"n": 1}]}

    monkeypatch.setattr(settings, "RAG_KG_CONTEXT_ENABLED", True)
    monkeypatch.setattr(KnowledgeGraphService, "build_chat_context_pack", pack)
    (doc,) = await _documents(db_session, "Paper")

    _, message = await _chat(
        db_session, test_user, _Store([_hit(doc, 0.9)]), "What is stride?"
    )

    assert message.message_type != "error", message.processing_error
    assert model[0]["kg_context"] == "Stride -> is_a -> Prefetcher"


# --------------------------------------------------------------------------
# the function itself
# --------------------------------------------------------------------------


async def test_a_document_filter_is_applied_by_the_store(db_session, model):
    wanted, other = await _documents(db_session, "Wanted", "Other")
    store = _Store([_hit(other, 0.95), _hit(wanted, 0.6)])

    retrieved = await rag_retrieval.retrieve_for_answer(
        "question",
        db=db_session,
        vector_store=store,
        llm_service=LLMService(),
        document_ids=[str(wanted.id)],
    )

    assert store.calls[0]["document_ids"] == [str(wanted.id)]
    assert [r["metadata"]["title"] for r in retrieved.results] == ["Wanted"]


async def test_the_passage_budget_is_the_callers(db_session, model):
    docs = await _documents(db_session, *[f"Doc {i}" for i in range(6)])
    store = _Store([_hit(d, 0.9 - i * 0.01) for i, d in enumerate(docs)])

    retrieved = await rag_retrieval.retrieve_for_answer(
        "question",
        db=db_session,
        vector_store=store,
        llm_service=LLMService(),
        max_results=2,
    )

    assert store.calls[0]["limit"] == 4
    assert len(retrieved.results) <= 2


def test_nothing_else_writes_its_own_answer_retrieval():
    """The two thin copies searched with `vector_store.search` and pasted the
    hits into a prompt. Neither file does that any more."""
    from pathlib import Path

    app = Path(__file__).resolve().parents[1] / "app"
    mcp_chat = (app / "mcp/tools/chat.py").read_text()
    assert "retrieve_for_answer" in mcp_chat
    assert "vector_store_service.search(" not in mcp_chat

    agent = (app / "services/agent_service.py").read_text()
    start = agent.index("async def _tool_answer_question(")
    body = agent[start : agent.index("    async def ", start + 10)]
    assert "retrieve_for_answer" in body
    assert "vector_store.search(" not in body
