"""The chat tools that read documents, called for real.

search_documents, get_document_details, list_recent_documents,
list_document_sources, list_documents_by_source, search_documents_by_author,
read_document_content, find_similar_documents and compare_documents had no
test at all. These call the real `AgentService._tool_*` handlers against the
in-memory database with real `Document` / `DocumentSource` / `DocumentChunk`
rows and the real `DocumentService`.

Only two edges are replaced, and both fakes refuse a call the real method
would refuse: the vector store (`_Store`, bound against
`VectorStoreService.search` / `.initialize`) and the LLM (`_LLM`, bound
against `LLMService.generate_response`). The store's hits carry the metadata
the real indexer writes (`vector_store.py`: `document_id`, `title`, `source`
= the source's *name*, `source_type`).
"""

import asyncio
import hashlib
import inspect
import json
from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from app.agent_core.tool_specs.documents import SPECS
from app.models.document import Document, DocumentChunk, DocumentSource
from app.services.agent_service import AgentService
from app.services.document_service import DocumentService
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStoreService

pytestmark = pytest.mark.unit

_SEARCH_SIG = inspect.signature(VectorStoreService.search)
_INIT_SIG = inspect.signature(VectorStoreService.initialize)
_LLM_SIG = inspect.signature(LLMService.generate_response)
_SPECS = {spec.name: spec for spec in SPECS}
_T0 = datetime(2026, 1, 1, 12, 0, 0)


class _Store:
    """A vector store that returns canned hits and refuses a wrong call."""

    def __init__(self, hits=(), fail=None):
        self.hits = list(hits)
        self.fail = fail
        self.calls = []
        self.initialized = 0

    async def initialize(self, *args, **kwargs):
        _INIT_SIG.bind(self, *args, **kwargs)
        self.initialized += 1

    async def search(self, *args, **kwargs):
        bound = _SEARCH_SIG.bind(self, *args, **kwargs)
        bound.apply_defaults()
        call = dict(bound.arguments)
        call.pop("self")
        self.calls.append(call)
        if self.fail:
            raise self.fail
        return self.hits[: max(int(call["limit"]), 0)]


class _LLM:
    """An LLM that records the call and refuses unknown keywords."""

    def __init__(self, reply="  They overlap on caching.  ", fail=None):
        self.reply = reply
        self.fail = fail
        self.calls = []

    async def generate_response(self, *args, **kwargs):
        bound = _LLM_SIG.bind(self, *args, **kwargs)
        call = dict(bound.arguments)
        call.pop("self")
        self.calls.append(call)
        if self.fail:
            raise self.fail
        return self.reply


def _service(store=None, llm=None):
    service = AgentService.__new__(AgentService)
    service.document_service = DocumentService.__new__(DocumentService)
    service.vector_store = store or _Store()
    service.llm_service = llm or _LLM()
    service._vector_store_initialized = False
    service._vector_store_init_lock = asyncio.Lock()
    return service


async def _source(db, name="Uploads", source_type="file", **extra):
    source = DocumentSource(name=name, source_type=source_type, config={}, **extra)
    db.add(source)
    await db.commit()
    await db.refresh(source)
    return source


async def _doc(db, source, title, content="body", age=0, **extra):
    """A real document; `age` is how many minutes ago it was last updated."""
    stamp = _T0 - timedelta(minutes=age)
    doc = Document(
        title=title,
        content=content,
        content_hash=hashlib.sha256((content or "").encode()).hexdigest(),
        source_id=source.id,
        source_identifier=f"test:{uuid4().hex}",
        created_at=stamp,
        updated_at=stamp,
        **extra,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc


async def _chunks(db, doc, texts, indexes=None):
    indexes = indexes or list(range(len(texts)))
    for index, text in zip(indexes, texts):
        db.add(
            DocumentChunk(
                document_id=doc.id,
                content=text,
                content_hash=hashlib.sha256(text.encode()).hexdigest(),
                chunk_index=index,
            )
        )
    await db.commit()


def _hit(doc, source, score=0.9, content="chunk text", chunk_index=0):
    """One search hit, with the metadata the real indexer writes."""
    return {
        "id": f"{doc.id}_{chunk_index}",
        "content": content,
        "score": score,
        "metadata": {
            "document_id": str(doc.id),
            "chunk_index": chunk_index,
            "title": doc.title,
            "source": source.name,
            "source_type": source.source_type,
        },
    }


def _plain(value):
    """The value must survive JSON, which is how a tool result reaches chat."""
    return json.loads(json.dumps(value))


def _declared(tool):
    return set(_SPECS[tool].parameters["properties"])


# ---------------------------------------------------------------------------
# search_documents
# ---------------------------------------------------------------------------


async def test_search_documents_returns_one_plain_row_per_document(db_session):
    source = await _source(db_session, name="Team Wiki", source_type="confluence")
    first = await _doc(db_session, source, "Cache design")
    second = await _doc(db_session, source, "Prefetcher notes")
    store = _Store(
        [
            _hit(first, source, 0.91234, "x" * 500),
            _hit(first, source, 0.8, "second chunk", chunk_index=1),
            _hit(second, source, 0.7, "stride prefetching"),
        ]
    )

    result = await _service(store)._tool_search_documents(
        {"query": "cache"}, db_session
    )

    assert _plain(result) == result
    assert [row["id"] for row in result] == [str(first.id), str(second.id)]
    assert result[0]["title"] == "Cache design"
    assert result[0]["score"] == 0.912
    assert result[0]["content_preview"] == "x" * 200
    assert store.initialized == 1
    assert store.calls[0]["query"] == "cache"


async def test_search_documents_source_type_is_the_type_not_the_name(db_session):
    source = await _source(db_session, name="Team Wiki", source_type="confluence")
    doc = await _doc(db_session, source, "Cache design")

    result = await _service(_Store([_hit(doc, source)]))._tool_search_documents(
        {"query": "cache"}, db_session
    )

    assert result[0]["source_type"] == "confluence"


async def test_search_documents_limit_bounds_the_rows(db_session):
    source = await _source(db_session)
    docs = [await _doc(db_session, source, f"Doc {i}") for i in range(6)]
    store = _Store([_hit(doc, source) for doc in docs])

    result = await _service(store)._tool_search_documents(
        {"query": "q", "limit": 2}, db_session
    )

    assert [row["id"] for row in result] == [str(docs[0].id), str(docs[1].id)]


async def test_search_documents_defaults_to_five_and_caps_at_twenty(db_session):
    source = await _source(db_session)
    docs = [await _doc(db_session, source, f"Doc {i}") for i in range(25)]
    store = _Store([_hit(doc, source) for doc in docs])
    service = _service(store)

    default = await service._tool_search_documents({"query": "q"}, db_session)
    huge = await service._tool_search_documents(
        {"query": "q", "limit": 10_000}, db_session
    )

    assert len(default) == 5
    assert len(huge) == 20
    assert max(call["limit"] for call in store.calls) <= 200


@pytest.mark.parametrize("params", [{}, {"query": ""}, {"query": "   "}])
async def test_search_documents_refuses_a_missing_query(db_session, params):
    store = _Store()

    result = await _service(store)._tool_search_documents(params, db_session)

    assert store.calls == []
    assert isinstance(result, dict) and "error" in result


@pytest.mark.parametrize("limit", [0, -1, -10])
async def test_search_documents_never_asks_the_store_for_nothing(db_session, limit):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Doc")
    store = _Store([_hit(doc, source)])

    result = await _service(store)._tool_search_documents(
        {"query": "q", "limit": limit}, db_session
    )

    assert all(call["limit"] > 0 for call in store.calls)
    assert len(result) <= max(limit, 0) or "error" in result


@pytest.mark.parametrize("limit", ["3", "many", None])
async def test_search_documents_survives_a_non_numeric_limit(db_session, limit):
    source = await _source(db_session)
    docs = [await _doc(db_session, source, f"Doc {i}") for i in range(4)]
    store = _Store([_hit(doc, source) for doc in docs])

    result = await _service(store)._tool_search_documents(
        {"query": "q", "limit": limit}, db_session
    )

    assert isinstance(result, dict) and "error" in result or len(result) <= 5


async def test_search_documents_is_not_starved_by_one_documents_chunks(db_session):
    source = await _source(db_session)
    long_doc = await _doc(db_session, source, "Long")
    others = [await _doc(db_session, source, f"Other {i}") for i in range(3)]
    hits = [_hit(long_doc, source, 0.9, chunk_index=i) for i in range(12)]
    hits += [_hit(doc, source, 0.6) for doc in others]

    result = await _service(_Store(hits))._tool_search_documents(
        {"query": "q", "limit": 3}, db_session
    )

    assert len(result) == 3


async def test_search_documents_with_no_hits_is_an_empty_list(db_session):
    result = await _service(_Store())._tool_search_documents(
        {"query": "nothing"}, db_session
    )

    assert result == []


# ---------------------------------------------------------------------------
# get_document_details
# ---------------------------------------------------------------------------


async def test_get_document_details_reads_the_real_row(db_session):
    source = await _source(db_session)
    doc = await _doc(
        db_session,
        source,
        "Cache design",
        content="c" * 1200,
        file_type="pdf",
        file_size=4096,
        author="Ada Lovelace",
        tags=["cache", "l2"],
        summary="A summary.",
        is_processed=True,
    )

    result = await _service()._tool_get_document_details(
        {"document_id": str(doc.id)}, db_session
    )

    assert _plain(result) == result
    assert result["id"] == str(doc.id)
    assert result["title"] == "Cache design"
    assert result["content_preview"] == "c" * 500
    assert result["file_type"] == "pdf"
    assert result["file_size"] == 4096
    assert result["author"] == "Ada Lovelace"
    assert result["tags"] == ["cache", "l2"]
    assert result["summary"] == "A summary."
    assert result["is_processed"] is True
    assert result["created_at"].startswith("2026-01-01T12:00:00")
    assert result["updated_at"].startswith("2026-01-01T12:00:00")


async def test_get_document_details_of_an_empty_document(db_session):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Empty", content=None)

    result = await _service()._tool_get_document_details(
        {"document_id": str(doc.id)}, db_session
    )

    assert _plain(result) == result
    assert result["content_preview"] is None
    assert result["tags"] == []
    assert result["summary"] is None


@pytest.mark.parametrize("bad", [None, "", "not-a-uuid"])
async def test_get_document_details_refuses_a_malformed_id(db_session, bad):
    params = {} if bad is None else {"document_id": bad}

    result = await _service()._tool_get_document_details(params, db_session)

    assert "Invalid document ID" in result["error"]


async def test_get_document_details_of_an_unknown_id(db_session):
    missing = str(uuid4())

    result = await _service()._tool_get_document_details(
        {"document_id": missing}, db_session
    )

    assert result == {"error": f"Document not found: {missing}"}


# ---------------------------------------------------------------------------
# list_recent_documents
# ---------------------------------------------------------------------------


async def test_list_recent_documents_is_newest_first_and_plain(db_session):
    source = await _source(db_session)
    old = await _doc(db_session, source, "Old", age=30)
    new = await _doc(db_session, source, "New", age=1, summary="s", file_type="md")
    mid = await _doc(db_session, source, "Mid", age=10, is_processed=True)

    result = await _service()._tool_list_recent_documents({}, db_session)

    assert _plain(result) == result
    assert [row["id"] for row in result] == [str(new.id), str(mid.id), str(old.id)]
    assert result[0] == {
        "id": str(new.id),
        "title": "New",
        "file_type": "md",
        "is_processed": False,
        "has_summary": True,
        "updated_at": new.updated_at.isoformat(),
    }
    assert result[1]["is_processed"] is True
    assert result[1]["has_summary"] is False


async def test_list_recent_documents_on_an_empty_knowledge_base(db_session):
    assert await _service()._tool_list_recent_documents({}, db_session) == []


async def test_list_recent_documents_limit_default_and_cap(db_session):
    source = await _source(db_session)
    for i in range(55):
        await _doc(db_session, source, f"Doc {i:02d}", age=i)
    service = _service()

    default = await service._tool_list_recent_documents({}, db_session)
    three = await service._tool_list_recent_documents({"limit": 3}, db_session)
    zero = await service._tool_list_recent_documents({"limit": 0}, db_session)
    huge = await service._tool_list_recent_documents({"limit": 10**9}, db_session)

    assert len(default) == 10
    assert [row["title"] for row in three] == ["Doc 00", "Doc 01", "Doc 02"]
    assert zero == []
    assert len(huge) == 50


async def test_list_recent_documents_negative_limit_does_not_lift_the_cap(db_session):
    source = await _source(db_session)
    for i in range(55):
        await _doc(db_session, source, f"Doc {i:02d}", age=i)

    result = await _service()._tool_list_recent_documents({"limit": -1}, db_session)

    assert isinstance(result, dict) and "error" in result or len(result) <= 50


@pytest.mark.parametrize("limit", ["3", "lots", None])
async def test_list_recent_documents_survives_a_non_numeric_limit(db_session, limit):
    source = await _source(db_session)
    await _doc(db_session, source, "Only")

    result = await _service()._tool_list_recent_documents({"limit": limit}, db_session)

    assert isinstance(result, dict) and "error" in result or len(result) == 1


# ---------------------------------------------------------------------------
# list_document_sources
# ---------------------------------------------------------------------------


async def test_list_document_sources_is_ordered_by_name_and_plain(db_session):
    synced = datetime(2026, 2, 3, 4, 5, 6)
    await _source(db_session, name="zeta", source_type="web")
    await _source(
        db_session,
        name="alpha",
        source_type="gitlab",
        is_syncing=True,
        last_sync=synced,
        last_error="401 from gitlab",
    )
    await _source(db_session, name="mid", source_type="file", is_active=False)

    result = await _service()._tool_list_document_sources({}, db_session)

    assert _plain(result) == result
    assert result["active_only"] is False
    assert result["count"] == 3
    assert [s["name"] for s in result["sources"]] == ["alpha", "mid", "zeta"]
    alpha = result["sources"][0]
    assert alpha["source_type"] == "gitlab"
    assert alpha["is_active"] is True
    assert alpha["is_syncing"] is True
    assert alpha["last_sync"].startswith("2026-02-03T04:05:06")
    assert alpha["last_error"] == "401 from gitlab"
    assert result["sources"][1]["is_active"] is False
    assert result["sources"][2]["last_sync"] is None


async def test_list_document_sources_active_only_drops_inactive(db_session):
    await _source(db_session, name="live")
    await _source(db_session, name="retired", is_active=False)

    result = await _service()._tool_list_document_sources(
        {"active_only": True}, db_session
    )

    assert result["active_only"] is True
    assert result["count"] == 1
    assert [s["name"] for s in result["sources"]] == ["live"]


async def test_list_document_sources_never_returns_the_source_config(db_session):
    source = DocumentSource(
        name="gitlab",
        source_type="gitlab",
        config={"token": "glpat-very-secret", "url": "https://gitlab.example"},
    )
    db_session.add(source)
    await db_session.commit()

    result = await _service()._tool_list_document_sources({}, db_session)

    assert "glpat-very-secret" not in json.dumps(result)
    assert "config" not in result["sources"][0]


async def test_list_document_sources_with_none(db_session):
    result = await _service()._tool_list_document_sources({}, db_session)

    assert result == {"active_only": False, "count": 0, "sources": []}


# ---------------------------------------------------------------------------
# list_documents_by_source
# ---------------------------------------------------------------------------


async def _two_sources(db):
    wiki = await _source(db, name="Team Wiki", source_type="confluence")
    repo = await _source(db, name="compiler-repo", source_type="gitlab")
    w1 = await _doc(db, wiki, "Wiki old", age=20, tags=["a"], file_type="html")
    w2 = await _doc(db, wiki, "Wiki new", age=2)
    r1 = await _doc(db, repo, "Repo doc", age=5)
    return wiki, repo, w1, w2, r1


async def test_list_documents_by_source_requires_a_filter(db_session):
    result = await _service()._tool_list_documents_by_source(
        {"limit": 5, "offset": 0}, db_session
    )

    assert result == {"error": "Provide source_id, source_name, or source_type"}


async def test_list_documents_by_source_id(db_session):
    wiki, _repo, w1, w2, _r1 = await _two_sources(db_session)

    result = await _service()._tool_list_documents_by_source(
        {"source_id": str(wiki.id)}, db_session
    )

    assert _plain(result) == result
    assert result["count"] == 2
    assert [d["id"] for d in result["documents"]] == [str(w2.id), str(w1.id)]
    old = result["documents"][1]
    assert old["title"] == "Wiki old"
    assert old["tags"] == ["a"]
    assert old["file_type"] == "html"
    assert old["updated_at"] == w1.updated_at.isoformat()
    assert old["source"] == {
        "id": str(wiki.id),
        "name": "Team Wiki",
        "source_type": "confluence",
    }
    assert result["documents"][0]["tags"] == []
    assert result["filters"]["source_id"] == str(wiki.id)


async def test_list_documents_by_source_name_is_partial_and_case_insensitive(
    db_session,
):
    _wiki, _repo, w1, w2, _r1 = await _two_sources(db_session)

    result = await _service()._tool_list_documents_by_source(
        {"source_name": "team WI"}, db_session
    )

    assert {d["id"] for d in result["documents"]} == {str(w1.id), str(w2.id)}


async def test_list_documents_by_source_type(db_session):
    _wiki, _repo, _w1, _w2, r1 = await _two_sources(db_session)

    result = await _service()._tool_list_documents_by_source(
        {"source_type": "gitlab"}, db_session
    )

    assert [d["id"] for d in result["documents"]] == [str(r1.id)]
    assert result["documents"][0]["source"]["source_type"] == "gitlab"


async def test_list_documents_by_source_filters_combine(db_session):
    wiki, _repo, _w1, _w2, _r1 = await _two_sources(db_session)

    result = await _service()._tool_list_documents_by_source(
        {"source_id": str(wiki.id), "source_type": "gitlab"}, db_session
    )

    assert result["count"] == 0
    assert result["documents"] == []


@pytest.mark.parametrize("bad", ["not-a-uuid", 42, ["x"]])
async def test_list_documents_by_source_refuses_a_malformed_id(db_session, bad):
    await _two_sources(db_session)

    result = await _service()._tool_list_documents_by_source(
        {"source_id": bad, "source_type": "gitlab"}, db_session
    )

    assert "error" in result


async def test_list_documents_by_source_unknown_id_lists_nothing(db_session):
    await _two_sources(db_session)

    result = await _service()._tool_list_documents_by_source(
        {"source_id": str(uuid4())}, db_session
    )

    assert "error" in result or result["documents"] == []


async def test_list_documents_by_source_pages_do_not_overlap(db_session):
    source = await _source(db_session, name="bulk")
    for i in range(7):
        await _doc(db_session, source, f"Doc {i}", age=i)
    service = _service()

    async def page(offset):
        result = await service._tool_list_documents_by_source(
            {"source_name": "bulk", "limit": 3, "offset": offset}, db_session
        )
        return [d["title"] for d in result["documents"]]

    assert await page(0) == ["Doc 0", "Doc 1", "Doc 2"]
    assert await page(3) == ["Doc 3", "Doc 4", "Doc 5"]
    assert await page(6) == ["Doc 6"]
    assert await page(-4) == ["Doc 0", "Doc 1", "Doc 2"]
    assert await page(100) == []


async def test_list_documents_by_source_limit_default_and_cap(db_session):
    source = await _source(db_session, name="bulk")
    for i in range(55):
        await _doc(db_session, source, f"Doc {i:02d}", age=i)
    service = _service()

    async def count(**extra):
        result = await service._tool_list_documents_by_source(
            {"source_name": "bulk", **extra}, db_session
        )
        return result["count"], len(result["documents"])

    assert await count() == (20, 20)
    assert await count(limit=10**9) == (50, 50)
    assert await count(limit=0) == (0, 0)


async def test_list_documents_by_source_negative_limit_keeps_the_cap(db_session):
    source = await _source(db_session, name="bulk")
    for i in range(55):
        await _doc(db_session, source, f"Doc {i:02d}", age=i)

    result = await _service()._tool_list_documents_by_source(
        {"source_name": "bulk", "limit": -1}, db_session
    )

    assert "error" in result or len(result["documents"]) <= 50


@pytest.mark.parametrize(
    "extra",
    [{"limit": "5"}, {"limit": None}, {"offset": "2"}, {"offset": None}],
)
async def test_list_documents_by_source_survives_non_numeric_paging(db_session, extra):
    source = await _source(db_session, name="bulk")
    await _doc(db_session, source, "Only")

    result = await _service()._tool_list_documents_by_source(
        {"source_name": "bulk", **extra}, db_session
    )

    assert "error" in result or isinstance(result["documents"], list)


async def test_list_documents_by_source_name_is_not_a_like_pattern(db_session):
    lookalike = await _source(db_session, name="docsXv2")
    await _doc(db_session, lookalike, "Wrong source")

    result = await _service()._tool_list_documents_by_source(
        {"source_name": "docs_v2"}, db_session
    )

    assert result["documents"] == []


# ---------------------------------------------------------------------------
# search_documents_by_author
# ---------------------------------------------------------------------------


async def _authored(db):
    source = await _source(db)
    ada = await _doc(db, source, "By Ada", author="Ada Lovelace", age=5, tags=["x"])
    adam = await _doc(db, source, "By Adam", author="Adam Smith", age=1)
    grace = await _doc(db, source, "By Grace", author="Grace Hopper (Ada fan)", age=3)
    await _doc(db, source, "Anonymous", author=None)
    return source, ada, adam, grace


@pytest.mark.parametrize("params", [{}, {"author": ""}, {"author": "   "}])
async def test_search_documents_by_author_requires_an_author(db_session, params):
    await _authored(db_session)

    result = await _service()._tool_search_documents_by_author(params, db_session)

    assert result == {"error": "Author is required"}


async def test_search_documents_by_author_contains_is_the_default(db_session):
    source, ada, adam, grace = await _authored(db_session)

    result = await _service()._tool_search_documents_by_author(
        {"author": " ada "}, db_session
    )

    assert _plain(result) == result
    assert result["author_query"] == "ada"
    assert result["match_type"] == "contains"
    assert result["count"] == 3
    assert [d["id"] for d in result["documents"]] == [
        str(adam.id),
        str(grace.id),
        str(ada.id),
    ]
    row = result["documents"][2]
    assert row == {
        "id": str(ada.id),
        "title": "By Ada",
        "author": "Ada Lovelace",
        "tags": ["x"],
        "file_type": None,
        "updated_at": ada.updated_at.isoformat(),
        "source_id": str(source.id),
    }


async def test_search_documents_by_author_starts_with(db_session):
    _source_row, ada, adam, _grace = await _authored(db_session)

    result = await _service()._tool_search_documents_by_author(
        {"author": "ADA", "match_type": "starts_with"}, db_session
    )

    assert {d["id"] for d in result["documents"]} == {str(ada.id), str(adam.id)}


async def test_search_documents_by_author_exact_ignores_case_only(db_session):
    _source_row, ada, _adam, _grace = await _authored(db_session)
    service = _service()

    hit = await service._tool_search_documents_by_author(
        {"author": "ada lovelace", "match_type": "exact"}, db_session
    )
    miss = await service._tool_search_documents_by_author(
        {"author": "ada", "match_type": "exact"}, db_session
    )

    assert [d["id"] for d in hit["documents"]] == [str(ada.id)]
    assert miss["count"] == 0 and miss["documents"] == []


@pytest.mark.parametrize("author", ["Ada L_velace", "%"])
async def test_search_documents_by_author_exact_is_not_a_like_pattern(
    db_session, author
):
    await _authored(db_session)

    result = await _service()._tool_search_documents_by_author(
        {"author": author, "match_type": "exact"}, db_session
    )

    assert result["documents"] == []


async def test_search_documents_by_author_unknown_author(db_session):
    await _authored(db_session)

    result = await _service()._tool_search_documents_by_author(
        {"author": "Nobody"}, db_session
    )

    assert result["count"] == 0
    assert result["documents"] == []


async def test_search_documents_by_author_limit_default_and_cap(db_session):
    source = await _source(db_session)
    for i in range(55):
        await _doc(db_session, source, f"Doc {i:02d}", author="Prolific", age=i)
    service = _service()

    async def titles(**extra):
        result = await service._tool_search_documents_by_author(
            {"author": "prolific", **extra}, db_session
        )
        assert result["count"] == len(result["documents"])
        return [d["title"] for d in result["documents"]]

    assert len(await titles()) == 20
    assert await titles(limit=2) == ["Doc 00", "Doc 01"]
    assert await titles(limit=0) == []
    assert len(await titles(limit=10**9)) == 50


async def test_search_documents_by_author_negative_limit_keeps_the_cap(db_session):
    source = await _source(db_session)
    for i in range(55):
        await _doc(db_session, source, f"Doc {i:02d}", author="Prolific", age=i)

    result = await _service()._tool_search_documents_by_author(
        {"author": "prolific", "limit": -5}, db_session
    )

    assert "error" in result or len(result["documents"]) <= 50


@pytest.mark.parametrize("limit", ["5", None])
async def test_search_documents_by_author_survives_a_non_numeric_limit(
    db_session, limit
):
    await _authored(db_session)

    result = await _service()._tool_search_documents_by_author(
        {"author": "ada", "limit": limit}, db_session
    )

    assert "error" in result or isinstance(result["documents"], list)


async def test_search_documents_by_author_refuses_a_non_string_author(db_session):
    await _authored(db_session)

    result = await _service()._tool_search_documents_by_author(
        {"author": 42}, db_session
    )

    assert "error" in result or result["documents"] == []


# ---------------------------------------------------------------------------
# read_document_content
# ---------------------------------------------------------------------------


async def test_read_document_content_returns_the_whole_text(db_session):
    source = await _source(db_session)
    doc = await _doc(
        db_session, source, "Notes", content="one two three", file_type="md"
    )

    result = await _service()._tool_read_document_content(
        {"document_id": str(doc.id)}, db_session
    )

    assert _plain(result) == result
    assert result == {
        "id": str(doc.id),
        "title": "Notes",
        "file_type": "md",
        "content": "one two three",
        "truncated": False,
        "full_length": 13,
        "word_count": 3,
    }


async def test_read_document_content_truncates_to_max_length(db_session):
    source = await _source(db_session)
    text = "".join(chr(97 + i % 26) for i in range(300))
    doc = await _doc(db_session, source, "Long", content=text)
    service = _service()

    cut = await service._tool_read_document_content(
        {"document_id": str(doc.id), "max_length": 40}, db_session
    )
    exact = await service._tool_read_document_content(
        {"document_id": str(doc.id), "max_length": 300}, db_session
    )
    zero = await service._tool_read_document_content(
        {"document_id": str(doc.id), "max_length": 0}, db_session
    )

    assert cut["content"] == text[:40]
    assert cut["truncated"] is True
    assert cut["full_length"] == 300
    assert exact["content"] == text and exact["truncated"] is False
    assert zero["content"] == "" and zero["truncated"] is True


async def test_read_document_content_default_and_cap(db_session):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Huge", content="z" * 60_000)
    service = _service()

    default = await service._tool_read_document_content(
        {"document_id": str(doc.id)}, db_session
    )
    huge = await service._tool_read_document_content(
        {"document_id": str(doc.id), "max_length": 10**9}, db_session
    )

    assert len(default["content"]) == 10_000 and default["truncated"] is True
    assert len(huge["content"]) == 50_000 and huge["truncated"] is True
    assert huge["full_length"] == 60_000


async def test_read_document_content_negative_max_length_is_not_a_slice(db_session):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Huge", content="z" * 20_000)

    result = await _service()._tool_read_document_content(
        {"document_id": str(doc.id), "max_length": -5}, db_session
    )

    assert "error" in result or len(result["content"]) <= 10_000


@pytest.mark.parametrize("max_length", ["100", None])
async def test_read_document_content_survives_a_non_numeric_max_length(
    db_session, max_length
):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Notes", content="one two three")

    result = await _service()._tool_read_document_content(
        {"document_id": str(doc.id), "max_length": max_length}, db_session
    )

    assert "error" in result or result["content"] == "one two three"


async def test_read_document_content_of_an_empty_document(db_session):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Empty", content=None)

    result = await _service()._tool_read_document_content(
        {"document_id": str(doc.id)}, db_session
    )

    assert result["content"] == ""
    assert result["truncated"] is False
    assert result["full_length"] == 0
    assert result["word_count"] == 0


async def test_read_document_content_chunks_are_in_index_order(db_session):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Chunked", content="alpha beta gamma delta")
    await _chunks(db_session, doc, ["gamma delta", "alpha", "beta"], indexes=[2, 0, 1])

    result = await _service()._tool_read_document_content(
        {"document_id": str(doc.id), "include_chunks": True}, db_session
    )

    assert _plain(result) == result
    assert result["chunks"] == [
        {"index": 0, "content": "alpha", "word_count": 1},
        {"index": 1, "content": "beta", "word_count": 1},
        {"index": 2, "content": "gamma delta", "word_count": 2},
    ]
    assert result["total_chunks"] == 3
    assert result["truncated"] is False
    assert "content" not in result


async def test_read_document_content_chunks_respect_max_length(db_session):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Chunked", content="x")
    await _chunks(db_session, doc, ["a" * 10, "b" * 10, "c" * 10])

    result = await _service()._tool_read_document_content(
        {"document_id": str(doc.id), "include_chunks": True, "max_length": 14},
        db_session,
    )

    assert [c["content"] for c in result["chunks"]] == ["a" * 10, "b" * 4]
    assert sum(len(c["content"]) for c in result["chunks"]) == 14
    assert result["total_chunks"] == 3
    assert result["truncated"] is True


async def test_read_document_content_chunks_that_fit_are_not_truncated(db_session):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Chunked", content="x")
    await _chunks(db_session, doc, ["a" * 10, "b" * 10])

    result = await _service()._tool_read_document_content(
        {"document_id": str(doc.id), "include_chunks": True, "max_length": 20},
        db_session,
    )

    assert [c["content"] for c in result["chunks"]] == ["a" * 10, "b" * 10]
    assert result["truncated"] is False


async def test_read_document_content_chunks_requested_but_absent(db_session):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "Unchunked", content="plain text")

    result = await _service()._tool_read_document_content(
        {"document_id": str(doc.id), "include_chunks": True}, db_session
    )

    assert result["content"] == "plain text"
    assert result["truncated"] is False


@pytest.mark.parametrize("bad", [None, "", "not-a-uuid"])
async def test_read_document_content_refuses_a_malformed_id(db_session, bad):
    params = {} if bad is None else {"document_id": bad}

    result = await _service()._tool_read_document_content(params, db_session)

    assert "Invalid document ID" in result["error"]


async def test_read_document_content_of_an_unknown_id(db_session):
    missing = str(uuid4())

    result = await _service()._tool_read_document_content(
        {"document_id": missing}, db_session
    )

    assert result == {"error": f"Document not found: {missing}"}


# ---------------------------------------------------------------------------
# find_similar_documents
# ---------------------------------------------------------------------------


async def _corpus(db, n=3, content="reference text"):
    source = await _source(db)
    ref = await _doc(db, source, "Reference", content=content)
    others = [await _doc(db, source, f"Other {i}") for i in range(n)]
    return source, ref, others


async def test_find_similar_documents_excludes_the_reference(db_session):
    source, ref, others = await _corpus(db_session, n=2, content="r" * 3000)
    store = _Store(
        [
            _hit(ref, source, 0.99),
            _hit(others[0], source, 0.81234, "y" * 400),
            _hit(others[0], source, 0.7, chunk_index=1),
            _hit(others[1], source, 0.5),
        ]
    )

    result = await _service(store)._tool_find_similar_documents(
        {"document_id": str(ref.id)}, db_session
    )

    assert _plain(result) == result
    assert result["reference_document"] == {"id": str(ref.id), "title": "Reference"}
    assert result["count"] == 2
    assert result["similar_documents"] == [
        {
            "id": str(others[0].id),
            "title": "Other 0",
            "similarity_score": 0.812,
            "content_preview": "y" * 150,
        },
        {
            "id": str(others[1].id),
            "title": "Other 1",
            "similarity_score": 0.5,
            "content_preview": "chunk text",
        },
    ]
    assert store.calls[0]["query"] == "r" * 1000
    assert store.initialized == 1


async def test_find_similar_documents_falls_back_to_the_title(db_session):
    source = await _source(db_session)
    ref = await _doc(db_session, source, "Only a title", content=None)
    store = _Store()

    result = await _service(store)._tool_find_similar_documents(
        {"document_id": str(ref.id)}, db_session
    )

    assert store.calls[0]["query"] == "Only a title"
    assert result["similar_documents"] == [] and result["count"] == 0


async def test_find_similar_documents_limit_default_and_explicit(db_session):
    source, ref, others = await _corpus(db_session, n=30)
    store = _Store([_hit(doc, source) for doc in others])
    service = _service(store)

    async def count(**extra):
        result = await service._tool_find_similar_documents(
            {"document_id": str(ref.id), **extra}, db_session
        )
        assert result["count"] == len(result["similar_documents"])
        return result["count"]

    assert await count() == 5
    assert await count(limit=2) == 2
    assert await count(limit=10**9) <= 25
    assert max(call["limit"] for call in store.calls) <= 200


@pytest.mark.parametrize("limit", [0, -3])
async def test_find_similar_documents_zero_or_negative_limit(db_session, limit):
    source, ref, others = await _corpus(db_session, n=4)
    store = _Store([_hit(doc, source) for doc in others])

    result = await _service(store)._tool_find_similar_documents(
        {"document_id": str(ref.id), "limit": limit}, db_session
    )

    assert "error" in result or result["similar_documents"] == []


@pytest.mark.parametrize("limit", ["2", None])
async def test_find_similar_documents_survives_a_non_numeric_limit(db_session, limit):
    source, ref, others = await _corpus(db_session, n=4)
    store = _Store([_hit(doc, source) for doc in others])

    result = await _service(store)._tool_find_similar_documents(
        {"document_id": str(ref.id), "limit": limit}, db_session
    )

    assert "error" in result or len(result["similar_documents"]) <= 5


async def test_find_similar_documents_excludes_self_however_the_id_is_written(
    db_session,
):
    source, ref, others = await _corpus(db_session, n=1)
    store = _Store([_hit(ref, source, 0.99), _hit(others[0], source, 0.6)])

    result = await _service(store)._tool_find_similar_documents(
        {"document_id": str(ref.id).upper()}, db_session
    )

    assert [d["id"] for d in result["similar_documents"]] == [str(others[0].id)]


async def test_find_similar_documents_is_not_starved_by_its_own_chunks(db_session):
    source, ref, others = await _corpus(db_session, n=3)
    hits = [_hit(ref, source, 0.99, chunk_index=i) for i in range(15)]
    hits += [_hit(doc, source, 0.7) for doc in others]

    result = await _service(_Store(hits))._tool_find_similar_documents(
        {"document_id": str(ref.id), "limit": 3}, db_session
    )

    assert result["count"] == 3


@pytest.mark.parametrize("bad", [None, "", "not-a-uuid"])
async def test_find_similar_documents_refuses_a_malformed_id(db_session, bad):
    store = _Store()
    params = {} if bad is None else {"document_id": bad}

    result = await _service(store)._tool_find_similar_documents(params, db_session)

    assert "Invalid document ID" in result["error"]
    assert store.calls == []


async def test_find_similar_documents_of_an_unknown_id(db_session):
    store = _Store()
    missing = str(uuid4())

    result = await _service(store)._tool_find_similar_documents(
        {"document_id": missing}, db_session
    )

    assert result == {"error": f"Document not found: {missing}"}
    assert store.calls == []


# ---------------------------------------------------------------------------
# compare_documents
# ---------------------------------------------------------------------------


async def _pair(db, one="the cache is fast", two="the cache is small today"):
    source = await _source(db)
    first = await _doc(db, source, "First", content=one, file_type="md")
    second = await _doc(db, source, "Second", content=two, file_type="pdf")
    return source, first, second


def _ids(first, second, **extra):
    return {
        "document_id_1": str(first.id),
        "document_id_2": str(second.id),
        **extra,
    }


async def test_compare_documents_full_comparison(db_session, test_user):
    source, first, second = await _pair(db_session)
    store = _Store([_hit(first, source, 0.99), _hit(second, source, 0.7512)])
    llm = _LLM()

    result = await _service(store, llm)._tool_compare_documents(
        _ids(first, second), test_user.id, db_session
    )

    assert _plain(result) == result
    assert result["document_1"] == {
        "id": str(first.id),
        "title": "First",
        "file_type": "md",
        "word_count": 4,
    }
    assert result["document_2"]["word_count"] == 5
    keywords = result["keyword_analysis"]
    assert keywords["common_word_count"] == 3
    assert keywords["unique_to_doc1_count"] == 1
    assert keywords["unique_to_doc2_count"] == 2
    assert keywords["similarity_score"] == 0.5
    assert sorted(keywords["sample_common_words"]) == ["cache", "is", "the"]
    assert keywords["sample_unique_to_doc1"] == ["fast"]
    assert sorted(keywords["sample_unique_to_doc2"]) == ["small", "today"]
    assert result["semantic_analysis"] == {
        "similarity_score": 0.751,
        "interpretation": "Highly similar - likely related topics",
    }
    assert result["comparison_summary"] == "They overlap on caching."
    assert store.calls[0]["query"] == "the cache is fast"
    call = llm.calls[0]
    assert call["user_id"] == test_user.id
    assert call["db"] is db_session
    assert "First" in call["query"] and "Second" in call["query"]


async def test_compare_documents_keyword_only_does_not_search(db_session, test_user):
    _source_row, first, second = await _pair(db_session)
    store = _Store()

    result = await _service(store)._tool_compare_documents(
        _ids(first, second, comparison_type="keyword"), test_user.id, db_session
    )

    assert "keyword_analysis" in result
    assert "semantic_analysis" not in result
    assert store.calls == [] and store.initialized == 0


async def test_compare_documents_semantic_only_skips_keywords(db_session, test_user):
    source, first, second = await _pair(db_session)
    store = _Store([_hit(second, source, 0.95)])

    result = await _service(store)._tool_compare_documents(
        _ids(first, second, comparison_type="semantic"), test_user.id, db_session
    )

    assert "keyword_analysis" not in result
    assert result["semantic_analysis"] == {
        "similarity_score": 0.95,
        "interpretation": "Nearly identical content",
    }


async def test_compare_documents_refuses_an_unknown_comparison_type(
    db_session, test_user
):
    _source_row, first, second = await _pair(db_session)
    llm = _LLM()

    result = await _service(llm=llm)._tool_compare_documents(
        _ids(first, second, comparison_type="structural"), test_user.id, db_session
    )

    assert "error" in result
    assert llm.calls == []


async def test_compare_documents_semantic_score_however_the_id_is_written(
    db_session, test_user
):
    source, first, second = await _pair(db_session)
    store = _Store([_hit(second, source, 0.95)])

    result = await _service(store)._tool_compare_documents(
        {
            "document_id_1": str(first.id).upper(),
            "document_id_2": str(second.id).upper(),
            "comparison_type": "semantic",
        },
        test_user.id,
        db_session,
    )

    assert result["semantic_analysis"]["similarity_score"] == 0.95


async def test_compare_documents_does_not_invent_a_score_it_never_measured(
    db_session, test_user
):
    source, first, second = await _pair(db_session)
    crowd = [await _doc(db_session, source, f"Crowd {i}") for i in range(3)]
    store = _Store([_hit(doc, source, 0.9) for doc in crowd])

    result = await _service(store)._tool_compare_documents(
        _ids(first, second, comparison_type="semantic"), test_user.id, db_session
    )

    semantic = result["semantic_analysis"]
    asked_for_the_pair = any(
        str(second.id) in (call["document_ids"] or []) for call in store.calls
    )
    assert (
        asked_for_the_pair or "error" in semantic or "similarity_score" not in semantic
    )


async def test_compare_documents_reports_an_unavailable_vector_store(
    db_session, test_user
):
    _source_row, first, second = await _pair(db_session)
    store = _Store(fail=RuntimeError("qdrant unreachable"))

    result = await _service(store)._tool_compare_documents(
        _ids(first, second), test_user.id, db_session
    )

    assert result["semantic_analysis"] == {
        "error": "Semantic comparison unavailable",
        "reason": "qdrant unreachable",
    }
    assert "keyword_analysis" in result
    assert _plain(result) == result


async def test_compare_documents_without_a_model_still_compares(db_session, test_user):
    _source_row, first, second = await _pair(db_session)
    llm = _LLM(fail=RuntimeError("no provider"))

    result = await _service(llm=llm)._tool_compare_documents(
        _ids(first, second, comparison_type="keyword"), test_user.id, db_session
    )

    assert "comparison_summary" not in result
    assert result["keyword_analysis"]["common_word_count"] == 3


async def test_compare_documents_bounds_what_it_sends_to_the_model(
    db_session, test_user
):
    _source_row, first, second = await _pair(
        db_session, one="alpha " * 5000, two="omega " * 5000
    )
    store = _Store()
    llm = _LLM()

    result = await _service(store, llm)._tool_compare_documents(
        _ids(first, second), test_user.id, db_session
    )

    assert len(llm.calls[0]["query"]) < 1500
    assert len(store.calls[0]["query"]) == 1000
    assert result["document_1"]["word_count"] == 5000
    assert result["keyword_analysis"]["similarity_score"] == 0


async def test_compare_documents_of_two_empty_documents(db_session, test_user):
    _source_row, first, second = await _pair(db_session, one=None, two=None)

    result = await _service()._tool_compare_documents(
        _ids(first, second, comparison_type="keyword"), test_user.id, db_session
    )

    assert result["document_1"]["word_count"] == 0
    assert result["keyword_analysis"]["similarity_score"] == 0
    assert result["keyword_analysis"]["sample_common_words"] == []


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"document_id_1": str(uuid4())},
        {"document_id_2": str(uuid4())},
        {"document_id_1": "nope", "document_id_2": str(uuid4())},
    ],
)
async def test_compare_documents_refuses_missing_or_malformed_ids(
    db_session, test_user, params
):
    llm = _LLM()

    result = await _service(llm=llm)._tool_compare_documents(
        params, test_user.id, db_session
    )

    assert "Invalid document ID" in result["error"]
    assert llm.calls == []


async def test_compare_documents_names_the_document_it_could_not_find(
    db_session, test_user
):
    _source_row, first, _second = await _pair(db_session)
    missing = str(uuid4())
    llm = _LLM()
    service = _service(llm=llm)

    as_second = await service._tool_compare_documents(
        {"document_id_1": str(first.id), "document_id_2": missing},
        test_user.id,
        db_session,
    )
    as_first = await service._tool_compare_documents(
        {"document_id_1": missing, "document_id_2": str(first.id)},
        test_user.id,
        db_session,
    )

    assert as_second == {"error": f"Document not found: {missing}"}
    assert as_first == {"error": f"Document not found: {missing}"}
    assert llm.calls == []


# ---------------------------------------------------------------------------
# What the model is told, against what the handlers read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool",
    [
        "search_documents",
        "get_document_details",
        "list_recent_documents",
        "list_document_sources",
        "list_documents_by_source",
        "search_documents_by_author",
        "read_document_content",
        "find_similar_documents",
        "compare_documents",
    ],
)
def test_every_declared_parameter_is_read_by_its_handler(tool):
    source = inspect.getsource(getattr(AgentService, f"_tool_{tool}"))

    unread = {name for name in _declared(tool) if f'"{name}"' not in source}

    assert unread == set()
    assert set(_SPECS[tool].parameters.get("required", [])) <= _declared(tool)


# ---------------------------------------------------------------------------
# An id that is not a string
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool",
    [
        "get_document_details",
        "read_document_content",
        "find_similar_documents",
        "compare_documents",
    ],
)
@pytest.mark.parametrize("bad", [12345, ["x"], {"id": 1}])
async def test_an_id_that_is_not_a_string_is_an_error_result(
    db_session, test_user, tool, bad
):
    handler = getattr(_service(), f"_tool_{tool}")

    if tool == "compare_documents":
        result = await handler(
            {"document_id_1": str(uuid4()), "document_id_2": bad},
            test_user.id,
            db_session,
        )
    else:
        result = await handler({"document_id": bad}, db_session)

    assert "Invalid document ID" in result["error"]
