"""The tag, statistics and knowledge-graph tools, called through their handlers.

Seventeen tools that no test exercised: search_by_tags (and its alias
search_documents_by_tag), list_all_tags, get_knowledge_base_stats,
get_collection_statistics, get_source_analytics, faceted_search,
get_search_suggestions, get_related_searches, get_kg_stats,
get_entity_mentions, get_entity_relationships, find_documents_by_entity,
get_document_knowledge_graph, get_global_knowledge_graph,
rebuild_document_knowledge_graph and delete_entity.

Each is reached the way chat reaches it: the handler registered in the chat
provider (`build_agent_service_*_provider`), which delegates to
`AgentService._tool_<name>`. get_knowledge_base_stats has a second, unrelated
implementation registered for autonomous jobs, tested separately.

Rows are real (in-memory database, the real `KnowledgeGraphService`,
`AnalyticsService` and `SearchService`). Only the vector store is replaced, by
a recording fake that refuses arguments the real one does not accept.

A test marked xfail(strict) states what the tool is supposed to do and
currently does not; the reason names the defect. Line numbers in the reasons
are those of commit 41cdb84.
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select

from app.agent_core.tool_specs import spec_for
from app.models.document import Document, DocumentChunk, DocumentSource
from app.models.knowledge_graph import Entity, EntityMention, Relationship
from app.services.agent_service import AgentService
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_agent_service_analytics_content_provider,
    build_agent_service_document_provider,
    build_agent_service_knowledge_graph_provider,
    build_autonomous_document_provider,
)
from app.services.search_service import search_service

pytestmark = pytest.mark.unit

TOOLS = (
    "search_by_tags",
    "search_documents_by_tag",
    "list_all_tags",
    "get_knowledge_base_stats",
    "get_collection_statistics",
    "get_source_analytics",
    "faceted_search",
    "get_search_suggestions",
    "get_related_searches",
    "get_kg_stats",
    "get_entity_mentions",
    "get_entity_relationships",
    "find_documents_by_entity",
    "get_document_knowledge_graph",
    "get_global_knowledge_graph",
    "rebuild_document_knowledge_graph",
    "delete_entity",
)

TAG_SEARCH_TOOLS = ("search_by_tags", "search_documents_by_tag")

NOW = datetime.utcnow().replace(microsecond=0)


# --------------------------------------------------------------------------
# The one external edge: the vector store
# --------------------------------------------------------------------------


class _VectorStore:
    """Records calls; signatures are the real `VectorStoreService`'s."""

    def __init__(self, hits=(), stats=None, stats_error=None):
        self.hits = list(hits)
        self.stats = stats if stats is not None else {"total_chunks": 0}
        self.stats_error = stats_error
        self.searches = []

    async def initialize(self, embedding_model=None, background=False):
        return None

    async def search(
        self,
        query,
        limit=10,
        filter_metadata=None,
        document_ids=None,
        apply_postprocessing=True,
    ):
        if not isinstance(limit, int) or limit < 1:
            raise ValueError(f"vector store refuses limit={limit!r}")
        self.searches.append(
            {"query": query, "limit": limit, "filter_metadata": filter_metadata}
        )
        wanted = filter_metadata or {}
        hits = [
            hit
            for hit in self.hits
            if all(hit["metadata"].get(key) == value for key, value in wanted.items())
        ]
        hits.sort(key=lambda hit: hit["score"], reverse=True)
        return hits[:limit]

    async def get_collection_stats(self):
        if self.stats_error:
            raise self.stats_error
        return dict(self.stats)


def _hit(document_id, title, score, content="chunk text", **metadata):
    return {
        "id": f"emb-{uuid4().hex}",
        "content": content,
        "score": score,
        "metadata": {"document_id": document_id, "title": title, **metadata},
    }


@pytest.fixture
def search_store(monkeypatch):
    """Give the real search service a vector store that holds known hits."""

    def _install(hits=()):
        store = _VectorStore(hits)
        monkeypatch.setattr(search_service, "vector_store", store)
        monkeypatch.setattr(search_service, "_initialized", True)
        return store

    return _install


# --------------------------------------------------------------------------
# Calling the tools
# --------------------------------------------------------------------------


def _service(vector_store=None):
    service = AgentService.__new__(AgentService)
    service.vector_store = vector_store or _VectorStore()
    service._vector_store_initialized = True
    service._vector_store_init_lock = asyncio.Lock()
    return service


def _chat_handler(tool, service):
    for build in (
        build_agent_service_document_provider,
        build_agent_service_knowledge_graph_provider,
        build_agent_service_analytics_content_provider,
    ):
        provider = build(service)
        if tool in provider._handlers:
            return provider._handlers[tool]
    return None


async def _chat(tool, params, db, user=None, service=None):
    service = service or _service()
    ctx = AgentToolExecutionContext(
        mode="chat",
        db=db,
        service=service,
        user_id=user.id if user is not None else uuid4(),
    )
    return await _chat_handler(tool, service)(params, ctx)


async def _autonomous_stats(params, db):
    provider = build_autonomous_document_provider(
        SimpleNamespace(document_service=None, search_service=None)
    )
    user_id = uuid4()
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=db,
        service=None,
        user_id=str(user_id),
        job=SimpleNamespace(id=uuid4(), user_id=user_id, goal="stats", config={}),
        state={},
    )
    return await provider._handlers["get_knowledge_base_stats"](params, ctx)


def _is_refusal(result):
    """An error result: a dict that says what was wrong and carries no data."""
    return (
        isinstance(result, dict)
        and isinstance(result.get("error"), str)
        and bool(result["error"])
    )


# --------------------------------------------------------------------------
# Real rows
# --------------------------------------------------------------------------


async def _source(db, name="Uploads", source_type="file", **extra):
    source = DocumentSource(name=name, source_type=source_type, config={}, **extra)
    db.add(source)
    await db.commit()
    await db.refresh(source)
    return source


async def _doc(db, source, title, content="body", age_days=0.0, **extra):
    stamp = extra.pop("created_at", None) or NOW - timedelta(days=age_days)
    extra.setdefault("updated_at", stamp)
    extra.setdefault("is_processed", True)
    doc = Document(
        title=title,
        content=content,
        content_hash=hashlib.sha256((content or "").encode()).hexdigest(),
        source_id=source.id,
        source_identifier=f"test:{uuid4().hex}",
        created_at=stamp,
        **extra,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc


async def _chunk(db, doc, content, index=0):
    chunk = DocumentChunk(
        document_id=doc.id,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        chunk_index=index,
    )
    db.add(chunk)
    await db.commit()
    await db.refresh(chunk)
    return chunk


async def _entity(db, name, entity_type="concept", description=None):
    entity = Entity(canonical_name=name, entity_type=entity_type)
    entity.description = description
    db.add(entity)
    await db.commit()
    await db.refresh(entity)
    return entity


async def _mention(db, entity, doc, text="m", sentence=None, age=0):
    mention = EntityMention(
        entity_id=entity.id,
        document_id=doc.id,
        text=text,
        sentence=sentence,
        created_at=NOW - timedelta(seconds=age),
    )
    db.add(mention)
    await db.commit()
    await db.refresh(mention)
    return mention


async def _rel(db, source, target, relation_type="uses", confidence=0.9, doc=None):
    rel = Relationship(
        relation_type=relation_type,
        confidence=confidence,
        source_entity_id=source.id,
        target_entity_id=target.id,
        document_id=doc.id if doc is not None else None,
        evidence=f"{source.canonical_name} {relation_type} {target.canonical_name}",
    )
    db.add(rel)
    await db.commit()
    await db.refresh(rel)
    return rel


async def _count(db, model, *where):
    stmt = select(func.count(model.id))
    for clause in where:
        stmt = stmt.where(clause)
    return int((await db.execute(stmt)).scalar() or 0)


# --------------------------------------------------------------------------
# Every tool is declared and answered
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tool", TOOLS)
def test_every_tool_is_declared_and_answered_in_chat(tool):
    assert spec_for(tool) is not None
    assert _chat_handler(tool, _service()) is not None


def test_only_knowledge_base_stats_is_offered_to_autonomous_jobs():
    for tool in TOOLS:
        job_types = spec_for(tool).job_types
        if tool == "get_knowledge_base_stats":
            assert set(job_types) == {"research", "monitor", "knowledge_expansion"}
        else:
            assert job_types == (), tool
    provider = build_autonomous_document_provider(SimpleNamespace())
    assert "get_knowledge_base_stats" in provider._handlers


# --------------------------------------------------------------------------
# search_by_tags / search_documents_by_tag
# --------------------------------------------------------------------------

_TAG_SEARCH_BROKEN = (
    "agent_service.py:2618-2623: Document.tags is a JSON column, which has no "
    "`overlap` (AttributeError) and whose `contains` is the string LIKE "
    "operator bound to a list. Every call with tags answers 'Search failed'; "
    "the tool has never returned a document."
)

_MATCH_ALL_BROKEN = (
    "agent_service.py:2618: `Document.tags.contains(tags)` on a JSON column "
    "is not containment; it compiles to `tags LIKE '%' || <list> || '%'`. On "
    "SQLite that is a substring match on the serialised list, so the tags "
    "must be stored adjacent and in the order asked for; on PostgreSQL the "
    "statement is `json LIKE ...`, for which no operator exists."
)


async def _tagged_corpus(db):
    source = await _source(db)
    await _doc(db, source, "both", tags=["ml", "cache"], age_days=1)
    await _doc(db, source, "ml only", tags=["ml"], age_days=2)
    await _doc(db, source, "cache only", tags=["cache"], age_days=3)
    await _doc(db, source, "other", tags=["compilers"], age_days=4)
    await _doc(db, source, "untagged", tags=None, age_days=5)
    await _doc(db, source, "empty", tags=[], age_days=6)
    return source


def _titles(result):
    return [doc["title"] for doc in result["documents"]]


@pytest.mark.parametrize("tool", TAG_SEARCH_TOOLS)
@pytest.mark.parametrize("params", [{}, {"tags": []}, {"tags": None}])
async def test_tag_search_refuses_without_tags(db_session, tool, params):
    await _tagged_corpus(db_session)

    result = await _chat(tool, params, db_session)

    assert _is_refusal(result)
    assert "documents" not in result


@pytest.mark.parametrize("tool", TAG_SEARCH_TOOLS)
async def test_tag_search_returns_documents_with_any_tag(db_session, tool):
    await _tagged_corpus(db_session)

    result = await _chat(tool, {"tags": ["ml", "cache"]}, db_session)

    assert "error" not in result
    assert _titles(result) == ["both", "ml only", "cache only"]
    assert result["count"] == 3
    assert result["match_type"] == "any"
    assert result["search_tags"] == ["ml", "cache"]
    assert result["documents"][0]["tags"] == ["ml", "cache"]
    json.dumps(result)


@pytest.mark.parametrize("tool", TAG_SEARCH_TOOLS)
async def test_tag_search_match_all_requires_every_tag(db_session, tool):
    source = await _tagged_corpus(db_session)
    await _doc(db_session, source, "spread", tags=["ml", "extra", "cache"], age_days=7)

    # The stored order is ["ml", "cache"]; a set of tags has no order.
    result = await _chat(tool, {"tags": ["cache", "ml"], "match_all": True}, db_session)

    assert "error" not in result
    assert _titles(result) == ["both", "spread"]
    assert result["match_type"] == "all"


async def test_tag_search_matches_whole_tags_not_substrings(db_session):
    source = await _source(db_session)
    await _doc(db_session, source, "long", tags=["machine-learning"])
    await _doc(db_session, source, "short", tags=["ml"])

    result = await _chat("search_by_tags", {"tags": ["machine"]}, db_session)

    assert "error" not in result
    assert result["documents"] == []
    assert result["count"] == 0


async def test_tag_search_limit_keeps_the_most_recently_updated(db_session):
    source = await _source(db_session)
    for age in range(5):
        await _doc(db_session, source, f"d{age}", tags=["ml"], age_days=age)

    result = await _chat("search_by_tags", {"tags": ["ml"], "limit": 2}, db_session)

    assert _titles(result) == ["d0", "d1"]
    assert result["count"] == 2


async def test_tag_search_caps_a_huge_limit_at_fifty(db_session):
    source = await _source(db_session)
    for index in range(55):
        db_session.add(
            Document(
                title=f"d{index}",
                content_hash="0" * 64,
                source_id=source.id,
                source_identifier=f"bulk:{index}",
                tags=["ml"],
            )
        )
    await db_session.commit()

    result = await _chat(
        "search_by_tags", {"tags": ["ml"], "limit": 10**9}, db_session
    )

    assert result["count"] == 50


async def test_tag_search_non_numeric_limit_is_an_error_result(db_session):
    await _tagged_corpus(db_session)

    result = await _chat(
        "search_by_tags", {"tags": ["ml"], "limit": "many"}, db_session
    )

    # Refused or read as the default: either way an answer, not a crash.
    assert isinstance(result, (dict, list))


async def test_tag_search_negative_limit_does_not_list_everything(db_session):
    await _tagged_corpus(db_session)

    result = await _chat("search_by_tags", {"tags": ["ml"], "limit": -1}, db_session)

    # Either a refusal that names the limit, or a search that returned nothing.
    if "error" in result:
        assert "limit" in result["error"].lower()
    else:
        assert result["documents"] == []


async def test_tag_search_ignores_whitespace_around_a_tag(db_session):
    await _tagged_corpus(db_session)

    result = await _chat("search_by_tags", {"tags": ["  ml  "]}, db_session)

    assert "error" not in result
    assert sorted(_titles(result)) == ["both", "ml only"]


async def test_every_listed_tag_finds_as_many_documents_as_it_counts(db_session):
    await _tagged_corpus(db_session)

    listed = await _chat("list_all_tags", {}, db_session)

    assert listed["tags"]
    for entry in listed["tags"]:
        found = await _chat("search_by_tags", {"tags": [entry["tag"]]}, db_session)
        assert found.get("count") == entry["count"], entry


# --------------------------------------------------------------------------
# list_all_tags
# --------------------------------------------------------------------------


async def test_list_all_tags_counts_documents_per_tag_most_used_first(db_session):
    source = await _source(db_session)
    await _doc(db_session, source, "a", tags=["ml", "cache"])
    await _doc(db_session, source, "b", tags=["ml"])
    await _doc(db_session, source, "c", tags=["ml", "compilers", "cache"])
    await _doc(db_session, source, "untagged", tags=None)
    await _doc(db_session, source, "empty", tags=[])

    result = await _chat("list_all_tags", {}, db_session)

    assert result["total_unique_tags"] == 3
    assert result["tags"] == [
        {"tag": "ml", "count": 3},
        {"tag": "cache", "count": 2},
        {"tag": "compilers", "count": 1},
    ]
    json.dumps(result)


async def test_list_all_tags_on_an_empty_knowledge_base(db_session):
    result = await _chat("list_all_tags", {}, db_session)

    assert result == {"total_unique_tags": 0, "tags": []}


async def test_list_all_tags_counts_beyond_any_page_size(db_session):
    source = await _source(db_session)
    for index in range(120):
        db_session.add(
            Document(
                title=f"d{index}",
                content_hash="0" * 64,
                source_id=source.id,
                source_identifier=f"bulk:{index}",
                tags=["ml"] if index % 2 else ["ml", "odd"],
            )
        )
    await db_session.commit()

    result = await _chat("list_all_tags", {}, db_session)

    assert result["tags"] == [
        {"tag": "ml", "count": 120},
        {"tag": "odd", "count": 60},
    ]


async def test_list_all_tags_does_not_split_a_string_into_characters(db_session):
    source = await _source(db_session)
    await _doc(db_session, source, "legacy row", tags="ml")
    await _doc(db_session, source, "proper row", tags=["cache"])

    result = await _chat("list_all_tags", {}, db_session)

    names = {entry["tag"] for entry in result["tags"]}
    assert "m" not in names and "l" not in names
    assert "cache" in names


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Two tag counters disagree about what a tag is. Chat's list_all_tags "
        "(agent_service.py:2664) counts 'ML' and 'ml' as two tags; the "
        "autonomous get_knowledge_base_stats (agent_tool_dispatch.py:11323) "
        "lower-cases and counts one."
    ),
)
async def test_the_two_tag_counters_agree_on_case(db_session):
    source = await _source(db_session)
    await _doc(db_session, source, "a", tags=["ML"])
    await _doc(db_session, source, "b", tags=["ml"])

    chat = await _chat("list_all_tags", {}, db_session)
    autonomous = await _autonomous_stats({}, db_session)

    chat_counts = {entry["tag"]: entry["count"] for entry in chat["tags"]}
    job_counts = {e["tag"]: e["count"] for e in autonomous["data"]["top_tags"]}
    assert chat_counts == job_counts


# --------------------------------------------------------------------------
# get_knowledge_base_stats (chat)
# --------------------------------------------------------------------------


async def _stats_corpus(db):
    source = await _source(db)
    mib = 1024 * 1024
    await _doc(
        db, source, "p1", file_type="pdf", file_size=mib, summary="s", age_days=1
    )
    await _doc(db, source, "p2", file_type="pdf", file_size=mib // 2, age_days=2)
    await _doc(
        db,
        source,
        "t1",
        file_type="txt",
        file_size=None,
        is_processed=False,
        age_days=30,
    )
    await _doc(db, source, "u1", file_type=None, file_size=0, summary="s", age_days=8)
    return source


async def test_knowledge_base_stats_reports_exact_counts(db_session):
    await _stats_corpus(db_session)
    store = _VectorStore(stats={"total_chunks": 42, "collection_name": "kb"})

    result = await _chat(
        "get_knowledge_base_stats", {}, db_session, service=_service(store)
    )

    assert result == {
        "total_documents": 4,
        "processed_documents": 3,
        "summarized_documents": 2,
        "pending_processing": 1,
        "total_storage_bytes": 1572864,
        "total_storage_mb": 1.5,
        "documents_by_type": {"pdf": 2, "txt": 1, "unknown": 1},
        "documents_last_7_days": 2,
        "vector_store": {"total_chunks": 42, "collection_name": "kb"},
    }
    json.dumps(result)


async def test_knowledge_base_stats_on_an_empty_knowledge_base(db_session):
    result = await _chat("get_knowledge_base_stats", {}, db_session)

    assert result["total_documents"] == 0
    assert result["pending_processing"] == 0
    assert result["total_storage_bytes"] == 0
    assert result["total_storage_mb"] == 0
    assert result["documents_by_type"] == {}
    assert result["documents_last_7_days"] == 0


async def test_knowledge_base_stats_counts_beyond_any_page_size(db_session):
    source = await _source(db_session)
    for index in range(130):
        db_session.add(
            Document(
                title=f"d{index}",
                content_hash="0" * 64,
                source_id=source.id,
                source_identifier=f"bulk:{index}",
                file_type=f"type{index % 13}",
                file_size=10,
                is_processed=True,
            )
        )
    await db_session.commit()

    result = await _chat("get_knowledge_base_stats", {}, db_session)

    assert result["total_documents"] == 130
    assert result["processed_documents"] == 130
    assert result["total_storage_bytes"] == 1300
    # Thirteen types of ten documents each; the breakdown lists the top ten.
    assert len(result["documents_by_type"]) == 10
    assert set(result["documents_by_type"].values()) == {10}


async def test_knowledge_base_stats_survive_an_unreachable_vector_store(db_session):
    await _stats_corpus(db_session)
    store = _VectorStore(stats_error=RuntimeError("qdrant unreachable"))

    result = await _chat(
        "get_knowledge_base_stats", {}, db_session, service=_service(store)
    )

    assert result.get("total_documents") == 4
    assert result.get("processed_documents") == 3


# --------------------------------------------------------------------------
# get_knowledge_base_stats (autonomous jobs: a second implementation)
# --------------------------------------------------------------------------


async def test_autonomous_stats_count_documents_and_sources(db_session):
    first = await _source(db_session, "First")
    second = await _source(db_session, "Second", "web")
    newest = await _doc(db_session, first, "newest", tags=["ML", "cache"], age_days=1)
    await _doc(db_session, first, "middle", tags=["ml"], age_days=2)
    await _doc(db_session, second, "oldest", tags=None, age_days=3)

    result = await _autonomous_stats({}, db_session)

    assert result["success"] is True
    data = result["data"]
    assert data["documents_total"] == 3
    assert data["sources_total"] == 2
    assert data["source_id"] is None
    assert [d["title"] for d in data["recent_documents"]] == [
        "newest",
        "middle",
        "oldest",
    ]
    assert data["recent_documents"][0]["id"] == str(newest.id)
    assert data["top_tags"] == [
        {"tag": "ml", "count": 2},
        {"tag": "cache", "count": 1},
    ]
    json.dumps(result)


async def test_autonomous_stats_can_be_narrowed_to_one_source(db_session):
    first = await _source(db_session, "First")
    second = await _source(db_session, "Second", "web")
    await _doc(db_session, first, "a", tags=["ml"], age_days=1)
    await _doc(db_session, first, "b", tags=["ml"], age_days=2)
    await _doc(db_session, second, "c", tags=["web"], age_days=3)

    result = await _autonomous_stats({"source_id": str(second.id)}, db_session)

    data = result["data"]
    assert data["documents_total"] == 1
    assert data["sources_total"] == 1
    assert data["source_id"] == str(second.id)
    assert [d["title"] for d in data["recent_documents"]] == ["c"]
    assert data["top_tags"] == [{"tag": "web", "count": 1}]


async def test_autonomous_stats_recent_limit_bounds_the_listing(db_session):
    source = await _source(db_session)
    for age in range(4):
        await _doc(db_session, source, f"d{age}", age_days=age)

    two = await _autonomous_stats({"recent_limit": 2}, db_session)
    negative = await _autonomous_stats({"recent_limit": -5}, db_session)

    assert [d["title"] for d in two["data"]["recent_documents"]] == ["d0", "d1"]
    assert two["data"]["documents_total"] == 4
    assert [d["title"] for d in negative["data"]["recent_documents"]] == ["d0"]


async def test_autonomous_stats_cap_the_listing_at_one_hundred(db_session):
    source = await _source(db_session)
    for index in range(105):
        db_session.add(
            Document(
                title=f"d{index}",
                content_hash="0" * 64,
                source_id=source.id,
                source_identifier=f"bulk:{index}",
            )
        )
    await db_session.commit()

    result = await _autonomous_stats({"recent_limit": 10**9}, db_session)

    assert len(result["data"]["recent_documents"]) == 100
    assert result["data"]["documents_total"] == 105


@pytest.mark.xfail(
    strict=True,
    reason=(
        "agent_tool_dispatch.py:11312-11323: `top_tags` is counted over the "
        "`recent_limit` newest rows only, not over the knowledge base, so the "
        "'top' tags are whatever the last page happens to carry (default 25 "
        "documents) while documents_total beside it counts everything."
    ),
)
async def test_autonomous_top_tags_cover_the_whole_knowledge_base(db_session):
    source = await _source(db_session)
    await _doc(db_session, source, "newest", tags=["rare"], age_days=1)
    for age in range(2, 6):
        await _doc(db_session, source, f"old{age}", tags=["common"], age_days=age)

    result = await _autonomous_stats({"recent_limit": 1}, db_session)

    assert result["data"]["documents_total"] == 5
    assert result["data"]["top_tags"][0] == {"tag": "common", "count": 4}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "agent_tool_dispatch.py:11285: `int(params.get('recent_limit', 25) or "
        "25)` is unguarded, so a non-numeric value raises ValueError out of "
        "the handler."
    ),
)
async def test_autonomous_stats_non_numeric_limit_is_an_error_result(db_session):
    result = await _autonomous_stats({"recent_limit": "lots"}, db_session)

    # Refused or read as the default: either way an answer, not a crash.
    assert isinstance(result, (dict, list))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "agent_tool_dispatch.py:11289-11293: a source_id that is not a UUID "
        "is swallowed (`except Exception: source_uuid = None`) and the run is "
        "handed the statistics of the whole knowledge base as though they "
        "were the source's."
    ),
)
async def test_autonomous_stats_refuse_a_malformed_source_id(db_session):
    source = await _source(db_session)
    await _doc(db_session, source, "a")

    result = await _autonomous_stats({"source_id": "not-a-uuid"}, db_session)

    assert _is_refusal(result)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "agent_tool_dispatch.py:11299-11301: `sources_total` is the literal 1 "
        "whenever a source_id parses, so a source that does not exist is "
        "reported as one source holding zero documents."
    ),
)
async def test_autonomous_stats_do_not_invent_a_source(db_session):
    source = await _source(db_session)
    await _doc(db_session, source, "a")

    result = await _autonomous_stats({"source_id": str(uuid4())}, db_session)

    assert _is_refusal(result) or result["data"]["sources_total"] == 0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Two implementations of one tool disagree. The spec "
        "(tool_specs/documents.py:403) promises 'document counts, storage "
        "usage, and processing status' and chat returns them "
        "(agent_service.py:2465); the autonomous handler "
        "(agent_tool_dispatch.py:11325) returns neither storage nor "
        "processing status, under different key names."
    ),
)
async def test_autonomous_stats_report_what_the_spec_promises(db_session):
    await _stats_corpus(db_session)

    data = (await _autonomous_stats({}, db_session))["data"]
    chat = await _chat("get_knowledge_base_stats", {}, db_session)

    assert "storage usage" in spec_for("get_knowledge_base_stats").description
    assert chat["total_storage_bytes"] == 1572864
    assert chat["pending_processing"] == 1
    assert 1572864 in data.values()
    assert any("process" in key or "pending" in key for key in data)


# --------------------------------------------------------------------------
# get_collection_statistics
# --------------------------------------------------------------------------


async def _collection(db):
    files = await _source(db, "Files", "file")
    web = await _source(db, "Web", "web")
    d1 = await _doc(
        db,
        files,
        "d1",
        content="a" * 100,
        file_size=1000,
        file_type="pdf",
        tags=["ml", "cache"],
        author="Ada",
        summary="short",
        age_days=1,
    )
    await _doc(
        db,
        files,
        "d2",
        content="b" * 50,
        file_size=500,
        file_type="pdf",
        tags=["ml"],
        author="Ada",
        age_days=2,
    )
    await _doc(
        db,
        web,
        "d3",
        content="c" * 25,
        file_size=None,
        file_type=None,
        tags=["web"],
        author="Bob",
        age_days=40,
    )
    d4 = await _doc(
        db,
        web,
        "d4 pending",
        content="d" * 999,
        file_size=9999,
        file_type="html",
        tags=["ml"],
        author="Eve",
        is_processed=False,
        age_days=1,
    )
    await _chunk(db, d1, "x" * 10, 0)
    await _chunk(db, d1, "y" * 20, 1)
    await _chunk(db, d4, "z" * 100, 0)
    return files, web


async def test_collection_statistics_report_exact_numbers(db_session):
    await _collection(db_session)

    result = await _chat("get_collection_statistics", {}, db_session)

    assert result["total_documents"] == 3
    assert result["total_file_size_bytes"] == 1500
    assert result["total_content_chars"] == 175
    assert result["estimated_word_count"] == 35
    assert result["total_chunks"] == 2
    assert result["avg_chunk_size_chars"] == 15
    assert result["documents_by_source_type"] == {"file": 2, "web": 1}
    assert result["documents_by_file_type"] == {"pdf": 2, "unknown": 1}
    assert {tag: count for tag, count in result["top_tags"]} == {
        "ml": 2,
        "cache": 1,
        "web": 1,
    }
    assert result["top_tags"][0][0] == "ml"
    assert [list(pair) for pair in result["top_authors"]] == [["Ada", 2], ["Bob", 1]]
    assert result["processing_status"] == {"processed": 3, "pending": 1}
    assert result["summarized_documents"] == 1
    assert result["filters_applied"] == {
        "source_id": None,
        "tag": None,
        "date_from": None,
        "date_to": None,
    }
    json.dumps(result)


async def test_collection_statistics_timeline_covers_the_last_thirty_days(db_session):
    await _collection(db_session)

    result = await _chat("get_collection_statistics", {}, db_session)

    days = [str((NOW - timedelta(days=age)).date()) for age in (2, 1)]
    assert [list(point) for point in result["timeline_last_30_days"]] == [
        [days[0], 1],
        [days[1], 1],
    ]


async def test_collection_statistics_on_an_empty_knowledge_base(db_session):
    result = await _chat("get_collection_statistics", {}, db_session)

    assert result["total_documents"] == 0
    assert result["total_file_size_bytes"] == 0
    assert result["estimated_word_count"] == 0
    assert result["total_chunks"] == 0
    assert result["avg_chunk_size_chars"] == 0
    assert result["top_tags"] == []
    assert result["processing_status"] == {"processed": 0, "pending": 0}


async def test_collection_statistics_filtered_to_one_source(db_session):
    files, _web = await _collection(db_session)

    result = await _chat(
        "get_collection_statistics", {"source_id": str(files.id)}, db_session
    )

    assert result["total_documents"] == 2
    assert result["total_file_size_bytes"] == 1500
    assert result["total_content_chars"] == 150
    assert result["documents_by_source_type"] == {"file": 2}
    assert result["documents_by_file_type"] == {"pdf": 2}
    assert [list(pair) for pair in result["top_authors"]] == [["Ada", 2]]
    assert result["total_chunks"] == 2
    assert result["filters_applied"]["source_id"] == str(files.id)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "analytics_service.py:151-158: `processing_status` is counted without "
        "the filters every other figure uses, so statistics for one source "
        "report the pending documents of every other source."
    ),
)
async def test_collection_processing_status_respects_the_source_filter(db_session):
    files, _web = await _collection(db_session)

    result = await _chat(
        "get_collection_statistics", {"source_id": str(files.id)}, db_session
    )

    assert result["processing_status"] == {"processed": 2, "pending": 0}


async def test_collection_statistics_for_an_unknown_source_are_empty(db_session):
    await _collection(db_session)

    result = await _chat(
        "get_collection_statistics", {"source_id": str(uuid4())}, db_session
    )

    assert result["total_documents"] == 0
    assert result["total_file_size_bytes"] == 0
    assert result["documents_by_file_type"] == {}


async def test_collection_statistics_malformed_source_id_is_an_error_result(
    db_session,
):
    result = await _chat(
        "get_collection_statistics", {"source_id": "not-a-uuid"}, db_session
    )

    assert _is_refusal(result)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "analytics_service.py:66: `Document.tags.contains([tag])` on a JSON "
        "column compiles to `tags LIKE '%' || '[\"ml\"]' || '%'`: on SQLite "
        "a substring match on the serialised list, which only finds "
        "documents whose *whole* tag list is that one tag (1 of the 2 here); "
        "on PostgreSQL `json LIKE ...` has no operator and the tool raises."
    ),
)
async def test_collection_statistics_filtered_to_one_tag(db_session):
    await _collection(db_session)

    result = await _chat("get_collection_statistics", {"tag": "ml"}, db_session)

    assert result["total_documents"] == 2
    assert result["filters_applied"]["tag"] == "ml"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "analytics_service.py:63-68: the tag filter is applied to "
        "`total_documents` alone (and that query is itself wrong). Sizes, "
        "chunks, file types, authors, tags and the timeline are computed "
        "without it, so every other figure would describe the unfiltered "
        "collection."
    ),
)
async def test_collection_tag_filter_applies_to_every_figure(db_session):
    await _collection(db_session)

    result = await _chat("get_collection_statistics", {"tag": "cache"}, db_session)

    assert result["total_file_size_bytes"] == 1000
    assert result["total_content_chars"] == 100
    assert result["documents_by_file_type"] == {"pdf": 1}
    assert [list(pair) for pair in result["top_authors"]] == [["Ada", 1]]


async def test_collection_statistics_date_from_excludes_older_documents(db_session):
    await _collection(db_session)
    cutoff = (NOW - timedelta(days=10)).date().isoformat()

    result = await _chat("get_collection_statistics", {"date_from": cutoff}, db_session)

    assert result["total_documents"] == 2
    assert result["total_content_chars"] == 150
    assert result["documents_by_source_type"] == {"file": 2}
    assert result["filters_applied"]["date_from"].startswith(cutoff)


async def test_collection_statistics_date_to_excludes_newer_documents(db_session):
    await _collection(db_session)
    cutoff = (NOW - timedelta(days=10)).date().isoformat()

    result = await _chat("get_collection_statistics", {"date_to": cutoff}, db_session)

    assert result["total_documents"] == 1
    assert result["total_content_chars"] == 25
    assert result["documents_by_source_type"] == {"web": 1}


async def test_collection_statistics_date_to_includes_the_end_day(db_session):
    source = await _source(db_session)
    await _doc(db_session, source, "noon", created_at=datetime(2026, 1, 31, 12, 0, 0))

    result = await _chat(
        "get_collection_statistics",
        {"date_from": "2026-01-31", "date_to": "2026-01-31"},
        db_session,
    )

    assert result["total_documents"] == 1


@pytest.mark.parametrize("key", ["date_from", "date_to"])
async def test_collection_statistics_refuse_an_unparseable_date(db_session, key):
    await _collection(db_session)

    result = await _chat("get_collection_statistics", {key: "last tuesday"}, db_session)

    assert _is_refusal(result)


# --------------------------------------------------------------------------
# get_source_analytics
# --------------------------------------------------------------------------


async def _sources_for_analytics(db):
    busy = await _source(db, "Busy", "file", last_sync=datetime(2026, 3, 1, 9, 0, 0))
    idle = await _source(db, "Idle", "web", is_active=False)
    await _doc(
        db,
        busy,
        "a",
        content="a" * 30,
        file_size=300,
        summary="s",
        updated_at=datetime(2026, 2, 1, 8, 0, 0),
    )
    await _doc(
        db,
        busy,
        "b",
        content="b" * 20,
        file_size=None,
        updated_at=datetime(2026, 2, 3, 8, 0, 0),
    )
    await _doc(
        db,
        busy,
        "c",
        content="c" * 10,
        file_size=200,
        is_processed=False,
        updated_at=datetime(2026, 2, 2, 8, 0, 0),
    )
    return busy, idle


async def test_source_analytics_report_exact_numbers_per_source(db_session):
    busy, idle = await _sources_for_analytics(db_session)

    result = await _chat("get_source_analytics", {}, db_session)

    by_name = {row["name"]: row for row in result["sources"]}
    assert set(by_name) == {"Busy", "Idle"}
    busy_row = by_name["Busy"]
    assert busy_row["id"] == str(busy.id)
    assert busy_row["source_type"] == "file"
    assert busy_row["document_count"] == 3
    assert busy_row["total_size_bytes"] == 500
    assert busy_row["total_chars"] == 60
    assert busy_row["processing_rate"] == "66.7%"
    assert busy_row["summarization_rate"] == "33.3%"
    assert busy_row["health_status"] == "healthy"
    assert busy_row["is_active"] is True
    assert busy_row["last_sync"].startswith("2026-03-01T09:00:00")
    assert busy_row["last_document_update"].startswith("2026-02-03T08:00:00")
    idle_row = by_name["Idle"]
    assert idle_row["id"] == str(idle.id)
    assert idle_row["document_count"] == 0
    assert idle_row["total_size_bytes"] == 0
    assert idle_row["total_chars"] == 0
    assert idle_row["processing_rate"] == "N/A"
    assert idle_row["summarization_rate"] == "N/A"
    assert idle_row["health_status"] == "inactive"
    assert idle_row["last_sync"] is None
    assert idle_row["last_document_update"] is None
    json.dumps(result)


async def test_source_analytics_can_be_narrowed_to_one_source(db_session):
    _busy, idle = await _sources_for_analytics(db_session)

    result = await _chat(
        "get_source_analytics", {"source_id": str(idle.id)}, db_session
    )

    assert [row["name"] for row in result["sources"]] == ["Idle"]


async def test_source_analytics_with_no_sources(db_session):
    result = await _chat("get_source_analytics", {}, db_session)

    assert result == {"sources": []}


async def test_source_analytics_for_an_unknown_source_invent_nothing(db_session):
    await _sources_for_analytics(db_session)

    result = await _chat(
        "get_source_analytics", {"source_id": str(uuid4())}, db_session
    )

    assert _is_refusal(result) or result == {"sources": []}


async def test_source_analytics_count_beyond_any_page_size(db_session):
    source = await _source(db_session)
    for index in range(150):
        db_session.add(
            Document(
                title=f"d{index}",
                content_hash="0" * 64,
                source_id=source.id,
                source_identifier=f"bulk:{index}",
                file_size=2,
                is_processed=index < 30,
            )
        )
    await db_session.commit()

    result = await _chat("get_source_analytics", {}, db_session)

    (row,) = result["sources"]
    assert row["document_count"] == 150
    assert row["total_size_bytes"] == 300
    assert row["processing_rate"] == "20.0%"
    assert row["summarization_rate"] == "0.0%"


async def test_source_analytics_malformed_source_id_is_an_error_result(db_session):
    result = await _chat(
        "get_source_analytics", {"source_id": "not-a-uuid"}, db_session
    )

    assert _is_refusal(result)


# --------------------------------------------------------------------------
# faceted_search
# --------------------------------------------------------------------------


def _facet_hits():
    """Three documents; the first is indexed as three chunks."""
    return [
        _hit(
            "doc-a",
            "Caching Strategies",
            0.9,
            source_id="s1",
            source_type="file",
            file_type="pdf",
            author="Ada",
            tags=["cache", "perf"],
            created_at="2026-01-15T10:00:00",
        ),
        _hit(
            "doc-a",
            "Caching Strategies",
            0.8,
            source_id="s1",
            source_type="file",
            file_type="pdf",
            author="Ada",
            tags=["cache", "perf"],
            created_at="2026-01-15T10:00:00",
        ),
        _hit(
            "doc-a",
            "Caching Strategies",
            0.7,
            source_id="s1",
            source_type="file",
            file_type="pdf",
            author="Ada",
            tags=["cache", "perf"],
            created_at="2026-01-15T10:00:00",
        ),
        _hit(
            "doc-b",
            "Cache Invalidation",
            0.6,
            source_id="s1",
            source_type="file",
            file_type="txt",
            author="Bob",
            tags=["cache"],
            created_at="2026-02-01T10:00:00Z",
        ),
        _hit(
            "doc-c",
            "Web Caches",
            0.5,
            source_id="s2",
            source_type="web",
            file_type="html",
            author="Ada",
            tags=[],
            created_at="2026-02-20T10:00:00",
        ),
    ]


async def test_faceted_search_returns_one_result_per_document(db_session, search_store):
    store = search_store(_facet_hits())

    result = await _chat("faceted_search", {"query": "cache"}, db_session)

    assert result["query"] == "cache"
    assert [r["id"] for r in result["results"]] == ["doc-a", "doc-b", "doc-c"]
    assert [r["relevance_score"] for r in result["results"]] == [0.9, 0.6, 0.5]
    assert result["results"][0]["title"] == "Caching Strategies"
    assert result["total"] == 3
    assert result["page"] == 1
    assert result["page_size"] == 10
    assert result["filters_applied"] == {}
    assert {call["query"] for call in store.searches} == {"cache"}
    json.dumps(result)


async def test_faceted_search_buckets_dates_by_month_newest_first(
    db_session, search_store
):
    search_store(_facet_hits()[2:])

    result = await _chat("faceted_search", {"query": "cache"}, db_session)

    assert list(result["facets"]["date"].items()) == [
        ("2026-02", 2),
        ("2026-01", 1),
    ]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "search_service.py:346-374: facets are counted per vector-store hit "
        "(chunk), while `results` and `total` are per document. A document "
        "indexed as three chunks counts three times in every facet, so the "
        "facet counts exceed `total`."
    ),
)
async def test_faceted_search_facets_count_documents_not_chunks(
    db_session, search_store
):
    search_store(_facet_hits())

    result = await _chat("faceted_search", {"query": "cache"}, db_session)

    assert result["total"] == 3
    assert result["facets"]["file_type"] == {"pdf": 1, "txt": 1, "html": 1}
    assert result["facets"]["source_type"] == {"file": 2, "web": 1}
    assert result["facets"]["author"] == {"Ada": 2, "Bob": 1}
    assert result["facets"]["tags"] == {"cache": 2, "perf": 1}


async def test_faceted_search_pages_through_the_results(db_session, search_store):
    search_store(_facet_hits())

    first = await _chat(
        "faceted_search", {"query": "cache", "page": 1, "page_size": 2}, db_session
    )
    second = await _chat(
        "faceted_search", {"query": "cache", "page": 2, "page_size": 2}, db_session
    )

    assert [r["id"] for r in first["results"]] == ["doc-a", "doc-b"]
    assert [r["id"] for r in second["results"]] == ["doc-c"]
    assert first["page_size"] == 2
    assert second["page"] == 2


@pytest.mark.xfail(
    strict=True,
    reason=(
        "search_service.py:118-161: `total` is the number of distinct "
        "documents among the `page*page_size + page_size` hits fetched for "
        "this page, not the number that match. It is capped at two pages and "
        "grows as the caller pages forward."
    ),
)
async def test_faceted_search_total_is_not_capped_by_the_page_size(
    db_session, search_store
):
    search_store([_hit(f"doc-{i:02d}", f"Doc {i}", 1.0 - i / 100) for i in range(25)])

    result = await _chat(
        "faceted_search", {"query": "doc", "page": 1, "page_size": 5}, db_session
    )

    assert len(result["results"]) == 5
    assert result["total"] == 25


@pytest.mark.parametrize(
    "filters, expected",
    [
        ({"file_type": "pdf"}, ["doc-a"]),
        ({"source_id": "s2"}, ["doc-c"]),
        ({"source_id": "s1", "file_type": "txt"}, ["doc-b"]),
    ],
)
async def test_faceted_search_applies_source_and_file_type_filters(
    db_session, search_store, filters, expected
):
    search_store(_facet_hits())

    result = await _chat(
        "faceted_search", {"query": "cache", "filters": filters}, db_session
    )

    assert [r["id"] for r in result["results"]] == expected
    assert result["total"] == len(expected)
    assert result["filters_applied"] == filters


@pytest.mark.xfail(
    strict=True,
    reason=(
        "search_service.py:324-331: the spec declares filters {source_id, "
        "file_type, author, tags, date_range} but only source_id and "
        "file_type are read. author, tags and date_range change nothing, and "
        "the result still echoes them under `filters_applied`."
    ),
)
@pytest.mark.parametrize(
    "filters, expected",
    [
        ({"author": "Bob"}, ["doc-b"]),
        ({"tags": ["perf"]}, ["doc-a"]),
        ({"date_range": {"from": "2026-02-10", "to": "2026-02-28"}}, ["doc-c"]),
    ],
)
async def test_faceted_search_applies_every_declared_filter(
    db_session, search_store, filters, expected
):
    search_store(_facet_hits())

    result = await _chat(
        "faceted_search", {"query": "cache", "filters": filters}, db_session
    )

    assert [r["id"] for r in result["results"]] == expected


@pytest.mark.xfail(
    strict=True,
    reason=(
        "search_service.py:345-350: the facet sample is a second, unfiltered "
        "search, so with a filter applied the facets describe documents the "
        "results exclude."
    ),
)
async def test_faceted_search_facets_describe_the_filtered_results(
    db_session, search_store
):
    search_store(_facet_hits())

    result = await _chat(
        "faceted_search",
        {"query": "cache", "filters": {"source_id": "s2"}},
        db_session,
    )

    assert [r["id"] for r in result["results"]] == ["doc-c"]
    assert set(result["facets"]["source_type"]) == {"web"}
    assert set(result["facets"]["file_type"]) == {"html"}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "agent_service.py:4655 / search_service.py:302: `query` is required "
        "by the spec, but a missing or blank one is searched for as the "
        "empty string instead of being refused."
    ),
)
@pytest.mark.parametrize("params", [{}, {"query": ""}, {"query": "   "}])
async def test_faceted_search_refuses_without_a_query(db_session, search_store, params):
    store = search_store(_facet_hits())

    result = await _chat("faceted_search", params, db_session)

    assert _is_refusal(result)
    assert store.searches == []


@pytest.mark.parametrize("key", ["page", "page_size"])
async def test_faceted_search_non_numeric_paging_is_an_error_result(
    db_session, search_store, key
):
    search_store(_facet_hits())

    result = await _chat("faceted_search", {"query": "cache", key: "two"}, db_session)

    # Refused or read as the default: either way an answer, not a crash.
    assert isinstance(result, (dict, list))


@pytest.mark.parametrize("key", ["page", "page_size"])
async def test_faceted_search_negative_paging_is_refused_or_clamped(
    db_session, search_store, key
):
    search_store(_facet_hits())

    result = await _chat("faceted_search", {"query": "cache", key: -1}, db_session)

    assert _is_refusal(result) or len(result["results"]) <= 3


async def test_faceted_search_bounds_a_huge_page_size(db_session, search_store):
    store = search_store(_facet_hits())

    result = await _chat(
        "faceted_search", {"query": "cache", "page_size": 10**9}, db_session
    )

    assert _is_refusal(result) or max(c["limit"] for c in store.searches) <= 10_000


# --------------------------------------------------------------------------
# get_search_suggestions
# --------------------------------------------------------------------------


async def _suggestion_corpus(db):
    source = await _source(db)
    await _doc(db, source, "Caching Strategies", tags=["cache"], author="Cathy")
    await _doc(db, source, "Compiler Design", tags=["llvm"], author="Dan")
    await _doc(db, source, "Notes", tags=["CacheLine"], author="Erin")
    await _doc(
        db,
        source,
        "Cached Draft",
        tags=["cachet"],
        author="Cato",
        is_processed=False,
    )
    return source


async def test_suggestions_come_from_titles_tags_and_authors(db_session):
    await _suggestion_corpus(db_session)

    result = await _chat(
        "get_search_suggestions", {"partial_query": "CAC", "limit": 10}, db_session
    )

    assert result["query"] == "CAC"
    by_type = {}
    for item in result["suggestions"]:
        by_type.setdefault(item["type"], set()).add(item["text"])
    assert by_type == {
        "title": {"Caching Strategies"},
        "tag": {"tag:cache", "tag:CacheLine"},
    }
    json.dumps(result)


async def test_suggestions_include_matching_authors(db_session):
    await _suggestion_corpus(db_session)

    result = await _chat(
        "get_search_suggestions", {"partial_query": "cath"}, db_session
    )

    assert result["suggestions"] == [
        {"type": "author", "text": "author:Cathy", "display": "👤 Cathy"}
    ]


async def test_suggestions_ignore_unprocessed_documents(db_session):
    await _suggestion_corpus(db_session)

    result = await _chat(
        "get_search_suggestions", {"partial_query": "cached"}, db_session
    )
    by_author = await _chat(
        "get_search_suggestions", {"partial_query": "cato"}, db_session
    )

    assert result["suggestions"] == []
    assert by_author["suggestions"] == []


async def test_suggestions_respect_the_limit(db_session):
    source = await _source(db_session)
    for index in range(8):
        await _doc(db_session, source, f"Cache note {index}", tags=[f"cache{index}"])

    three = await _chat(
        "get_search_suggestions", {"partial_query": "cache", "limit": 3}, db_session
    )
    default = await _chat(
        "get_search_suggestions", {"partial_query": "cache"}, db_session
    )

    assert len(three["suggestions"]) == 3
    assert len(default["suggestions"]) == 5


async def test_suggestions_for_a_single_character_are_empty(db_session):
    await _suggestion_corpus(db_session)

    result = await _chat("get_search_suggestions", {"partial_query": "c"}, db_session)

    assert result == {"suggestions": [], "query": "c"}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "agent_service.py:4674: `partial_query` is required by the spec, but "
        "a missing one becomes '' and is answered with an empty suggestion "
        "list -- indistinguishable from a query nothing matched."
    ),
)
async def test_suggestions_refuse_without_a_partial_query(db_session):
    result = await _chat("get_search_suggestions", {}, db_session)

    assert _is_refusal(result)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "search_service.py:455-459: the tag scan takes the first 100 "
        "processed documents (`.limit(100)`) and filters afterwards, so a "
        "matching tag on any later document is never suggested."
    ),
)
async def test_suggestions_find_a_tag_beyond_the_first_hundred_documents(db_session):
    source = await _source(db_session)
    for index in range(100):
        db_session.add(
            Document(
                title=f"d{index}",
                content_hash="0" * 64,
                source_id=source.id,
                source_identifier=f"bulk:{index}",
                tags=["common"],
                is_processed=True,
            )
        )
    await db_session.commit()
    await _doc(db_session, source, "last", tags=["zebra"])

    result = await _chat("get_search_suggestions", {"partial_query": "zeb"}, db_session)

    assert [item["text"] for item in result["suggestions"]] == ["tag:zebra"]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "search_service.py:441, 485: the partial query is interpolated into "
        "an ILIKE pattern without escaping, so `_` and `%` typed by the user "
        "act as wildcards: 'a_c' suggests the title 'abc'."
    ),
)
async def test_suggestions_treat_like_wildcards_literally(db_session):
    source = await _source(db_session)
    await _doc(db_session, source, "abc handbook")
    await _doc(db_session, source, "a_c handbook")

    result = await _chat("get_search_suggestions", {"partial_query": "a_c"}, db_session)

    assert [item["text"] for item in result["suggestions"]] == ["a_c handbook"]


async def test_suggestions_non_numeric_limit_is_an_error_result(db_session):
    await _suggestion_corpus(db_session)

    result = await _chat(
        "get_search_suggestions", {"partial_query": "cac", "limit": "few"}, db_session
    )

    # Refused or read as the default: either way an answer, not a crash.
    assert isinstance(result, (dict, list))


async def test_suggestions_negative_limit_returns_nothing(db_session):
    source = await _source(db_session)
    for index in range(4):
        await _doc(db_session, source, f"Cache note {index}")

    result = await _chat(
        "get_search_suggestions", {"partial_query": "cache", "limit": -1}, db_session
    )

    assert _is_refusal(result) or result["suggestions"] == []


# --------------------------------------------------------------------------
# get_related_searches
# --------------------------------------------------------------------------


def _related_hits():
    return [
        _hit(
            "doc-a",
            "Kubernetes Networking Guide",
            0.9,
            content="Pods talk through services and ingress controllers.",
            tags=["Ingress"],
        ),
        _hit(
            "doc-b",
            "Kubernetes Networking Policies",
            0.8,
            content="Policies restrict which pods may talk.",
            tags=["Ingress", "security"],
        ),
    ]


async def test_related_searches_extend_the_query_with_terms_from_the_hits(
    db_session, search_store
):
    store = search_store(_related_hits())

    result = await _chat(
        "get_related_searches", {"query": "kubernetes", "limit": 2}, db_session
    )

    # 'ingress' is a tag of both hits (5 each, plus one content mention);
    # 'networking' is in both titles (3 each).
    assert result == {
        "related_searches": ["kubernetes ingress", "kubernetes networking"],
        "original_query": "kubernetes",
    }
    assert store.searches[0]["query"] == "kubernetes"
    json.dumps(result)


async def test_related_searches_never_repeat_the_query_or_each_other(
    db_session, search_store
):
    search_store(_related_hits())

    result = await _chat(
        "get_related_searches", {"query": "Kubernetes Networking"}, db_session
    )

    related = result["related_searches"]
    assert related
    assert len(related) == len(set(related)) <= 5
    for suggestion in related:
        extra = suggestion[len("Kubernetes Networking") :].strip()
        assert extra and extra not in {"kubernetes", "networking"}


async def test_related_searches_respect_the_limit(db_session, search_store):
    search_store(_related_hits())

    one = await _chat(
        "get_related_searches", {"query": "kubernetes", "limit": 1}, db_session
    )

    assert one["related_searches"] == ["kubernetes ingress"]


async def test_related_searches_with_no_hits_are_empty(db_session, search_store):
    search_store([])

    result = await _chat("get_related_searches", {"query": "kubernetes"}, db_session)

    assert result == {"related_searches": [], "original_query": "kubernetes"}


async def test_related_searches_negative_limit_returns_nothing(
    db_session, search_store
):
    search_store(_related_hits())

    result = await _chat(
        "get_related_searches", {"query": "kubernetes", "limit": -3}, db_session
    )

    assert _is_refusal(result) or result["related_searches"] == []


@pytest.mark.xfail(
    strict=True,
    reason=(
        "agent_service.py:4690: `query` is required by the spec, but a "
        "missing one becomes '' and is sent to the vector store; the reply "
        "is a list of bare terms presented as searches related to nothing."
    ),
)
async def test_related_searches_refuse_without_a_query(db_session, search_store):
    store = search_store(_related_hits())

    result = await _chat("get_related_searches", {}, db_session)

    assert _is_refusal(result)
    assert store.searches == []


async def test_related_searches_non_numeric_limit_is_an_error_result(
    db_session, search_store
):
    search_store(_related_hits())

    result = await _chat(
        "get_related_searches", {"query": "kubernetes", "limit": "some"}, db_session
    )

    # Refused or read as the default: either way an answer, not a crash.
    assert isinstance(result, (dict, list))


@pytest.mark.xfail(
    strict=True,
    reason=(
        "search_service.py:540-565: the query is split on whitespace only, "
        "so punctuation hides a query word from the exclusion set and "
        "'kubernetes, networking?' is 'related' to itself: "
        "'kubernetes, networking? kubernetes'."
    ),
)
async def test_related_searches_exclude_query_words_despite_punctuation(
    db_session, search_store
):
    search_store(_related_hits())

    result = await _chat(
        "get_related_searches", {"query": "kubernetes, networking?"}, db_session
    )

    for suggestion in result["related_searches"]:
        extra = suggestion[len("kubernetes, networking?") :].strip()
        assert extra not in {"kubernetes", "networking"}


# --------------------------------------------------------------------------
# A small knowledge graph
# --------------------------------------------------------------------------


async def _graph(db):
    """Four entities, six mentions, three relationships, two documents."""
    source = await _source(db)
    paper = await _doc(db, source, "Paper", age_days=1)
    memo = await _doc(db, source, "Memo", age_days=2)
    ada = await _entity(db, "Ada Lovelace", "person", "mathematician")
    bob = await _entity(db, "Bob Stone", "person")
    acme = await _entity(db, "Acme Corp", "org")
    caching = await _entity(db, "Caching", "concept")
    for age, doc in enumerate([paper, paper, memo]):
        await _mention(db, ada, doc, text="Ada", sentence=f"Ada s{age}", age=age)
    await _mention(db, bob, memo, text="Bob", age=10)
    await _mention(db, acme, paper, text="Acme", age=20)
    await _mention(db, acme, memo, text="Acme", age=21)
    works_ada = await _rel(db, ada, acme, "works_for", 0.9, paper)
    works_bob = await _rel(db, bob, acme, "works_for", 0.4, memo)
    knows = await _rel(db, ada, bob, "knows", 0.7, memo)
    return SimpleNamespace(
        paper=paper,
        memo=memo,
        ada=ada,
        bob=bob,
        acme=acme,
        caching=caching,
        works_ada=works_ada,
        works_bob=works_bob,
        knows=knows,
    )


# --------------------------------------------------------------------------
# get_kg_stats
# --------------------------------------------------------------------------


async def test_kg_stats_count_entities_relationships_and_mentions(db_session):
    await _graph(db_session)

    result = await _chat("get_kg_stats", {}, db_session)

    assert result == {"entities": 4, "relationships": 3, "mentions": 6}
    json.dumps(result)


async def test_kg_stats_on_an_empty_graph(db_session):
    result = await _chat("get_kg_stats", {}, db_session)

    assert result == {"entities": 0, "relationships": 0, "mentions": 0}


async def test_kg_stats_count_beyond_any_page_size(db_session):
    source = await _source(db_session)
    doc = await _doc(db_session, source, "big")
    hub = await _entity(db_session, "hub")
    for index in range(260):
        db_session.add(EntityMention(entity_id=hub.id, document_id=doc.id, text="m"))
        db_session.add(Entity(canonical_name=f"e{index}", entity_type="concept"))
    await db_session.commit()

    result = await _chat("get_kg_stats", {}, db_session)

    assert result == {"entities": 261, "relationships": 0, "mentions": 260}


# --------------------------------------------------------------------------
# get_entity_mentions
# --------------------------------------------------------------------------


@pytest.fixture
def uuid_strings_bind(monkeypatch):
    """Let a UUID given as a string reach a query, as it can on PostgreSQL.

    `_tool_get_entity_mentions` validates the id and then hands the *string*
    to `KnowledgeGraphService`. asyncpg binds a string to a uuid column; the
    character-based UUID type the test database uses does not. This converts
    at the service boundary and calls the real methods, so everything else
    about the tool can be tested; `test_entity_mentions_pass_the_parsed_uuid`
    runs without it.
    """
    from app.services.knowledge_graph_service import KnowledgeGraphService

    real_list = KnowledgeGraphService.mentions_for_entity
    real_count = KnowledgeGraphService.mentions_count_for_entity

    async def mentions_for_entity(self, db, entity_id, limit=25, offset=0):
        return await real_list(self, db, UUID(str(entity_id)), limit, offset)

    async def mentions_count_for_entity(self, db, entity_id):
        return await real_count(self, db, UUID(str(entity_id)))

    monkeypatch.setattr(
        KnowledgeGraphService, "mentions_for_entity", mentions_for_entity
    )
    monkeypatch.setattr(
        KnowledgeGraphService, "mentions_count_for_entity", mentions_count_for_entity
    )


async def test_entity_mentions_pass_the_parsed_uuid(db_session):
    graph = await _graph(db_session)

    result = await _chat(
        "get_entity_mentions", {"entity_id": str(graph.ada.id)}, db_session
    )

    assert result.get("total") == 3


@pytest.mark.parametrize("params", [{}, {"entity_id": None}, {"entity_id": "nope"}])
async def test_entity_mentions_refuse_a_missing_or_malformed_id(db_session, params):
    result = await _chat("get_entity_mentions", params, db_session)

    assert _is_refusal(result)
    assert "Invalid entity ID" in result["error"]


async def test_entity_mentions_report_an_unknown_entity(db_session):
    await _graph(db_session)
    missing = str(uuid4())

    result = await _chat("get_entity_mentions", {"entity_id": missing}, db_session)

    assert result == {"error": f"Entity not found: {missing}"}


async def test_entity_mentions_list_newest_first_with_the_total(
    db_session, uuid_strings_bind
):
    graph = await _graph(db_session)

    result = await _chat(
        "get_entity_mentions", {"entity_id": str(graph.ada.id)}, db_session
    )

    assert result["entity"] == {
        "id": str(graph.ada.id),
        "name": "Ada Lovelace",
        "type": "person",
    }
    assert result["total"] == 3
    assert result["limit"] == 25
    assert result["offset"] == 0
    assert [item["sentence"] for item in result["items"]] == [
        "Ada s0",
        "Ada s1",
        "Ada s2",
    ]
    assert [item["document_title"] for item in result["items"]] == [
        "Paper",
        "Paper",
        "Memo",
    ]
    first = result["items"][0]
    assert first["entity_id"] == str(graph.ada.id)
    assert first["document_id"] == str(graph.paper.id)
    assert first["text"] == "Ada"
    json.dumps(result)


async def test_entity_mentions_page_with_limit_and_offset(
    db_session, uuid_strings_bind
):
    graph = await _graph(db_session)

    result = await _chat(
        "get_entity_mentions",
        {"entity_id": str(graph.ada.id), "limit": 1, "offset": 1},
        db_session,
    )
    past_end = await _chat(
        "get_entity_mentions",
        {"entity_id": str(graph.ada.id), "offset": 50},
        db_session,
    )

    assert [item["sentence"] for item in result["items"]] == ["Ada s1"]
    assert result["total"] == 3
    assert result["limit"] == 1
    assert result["offset"] == 1
    assert past_end["items"] == []
    assert past_end["total"] == 3


async def test_entity_mentions_clamp_the_limit_and_offset(
    db_session, uuid_strings_bind
):
    graph = await _graph(db_session)

    result = await _chat(
        "get_entity_mentions",
        {"entity_id": str(graph.ada.id), "limit": 10**9, "offset": -4},
        db_session,
    )

    assert result["limit"] == 200
    assert result["offset"] == 0
    assert len(result["items"]) == 3


async def test_entity_mentions_accept_any_spelling_of_the_uuid(
    db_session, uuid_strings_bind
):
    graph = await _graph(db_session)

    for spelling in (str(graph.ada.id).upper(), graph.ada.id.hex):
        result = await _chat("get_entity_mentions", {"entity_id": spelling}, db_session)

        assert result.get("total") == 3, spelling
        assert len(result["items"]) == 3, spelling


@pytest.mark.parametrize("key", ["limit", "offset"])
async def test_entity_mentions_non_numeric_paging_is_an_error_result(db_session, key):
    graph = await _graph(db_session)

    result = await _chat(
        "get_entity_mentions", {"entity_id": str(graph.ada.id), key: "ten"}, db_session
    )

    # Refused or read as the default: either way an answer, not a crash.
    assert isinstance(result, (dict, list))


async def test_entity_mentions_negative_limit_does_not_list_everything(
    db_session, uuid_strings_bind
):
    graph = await _graph(db_session)

    result = await _chat(
        "get_entity_mentions", {"entity_id": str(graph.ada.id), "limit": -1}, db_session
    )

    assert _is_refusal(result) or len(result["items"]) <= 1


# --------------------------------------------------------------------------
# get_entity_relationships
# --------------------------------------------------------------------------


@pytest.mark.parametrize("params", [{}, {"entity_id": None}, {"entity_id": "nope"}])
async def test_entity_relationships_refuse_a_missing_or_malformed_id(
    db_session, params
):
    result = await _chat("get_entity_relationships", params, db_session)

    assert _is_refusal(result)
    assert "Invalid entity ID" in result["error"]


async def test_entity_relationships_report_an_unknown_entity(db_session):
    await _graph(db_session)
    missing = str(uuid4())

    result = await _chat("get_entity_relationships", {"entity_id": missing}, db_session)

    assert result == {"error": f"Entity not found: {missing}"}


async def test_entity_relationships_list_both_directions(db_session):
    graph = await _graph(db_session)

    result = await _chat(
        "get_entity_relationships", {"entity_id": str(graph.bob.id)}, db_session
    )

    assert result["entity"] == {
        "id": str(graph.bob.id),
        "name": "Bob Stone",
        "type": "person",
    }
    assert result["count"] == 2
    by_direction = {rel["direction"]: rel for rel in result["relationships"]}
    assert by_direction["outgoing"] == {
        "direction": "outgoing",
        "relation_type": "works_for",
        "related_entity": {
            "id": str(graph.acme.id),
            "name": "Acme Corp",
            "type": "org",
        },
        "confidence": 0.4,
        "evidence": "Bob Stone works_for Acme Corp",
    }
    assert by_direction["incoming"]["relation_type"] == "knows"
    assert by_direction["incoming"]["related_entity"]["name"] == "Ada Lovelace"
    assert by_direction["incoming"]["confidence"] == 0.7
    json.dumps(result)


async def test_entity_relationships_of_an_unconnected_entity_are_empty(db_session):
    graph = await _graph(db_session)

    result = await _chat(
        "get_entity_relationships", {"entity_id": str(graph.caching.id)}, db_session
    )

    assert result["relationships"] == []
    assert result["count"] == 0


async def test_entity_relationships_truncate_long_evidence(db_session):
    graph = await _graph(db_session)
    graph.knows.evidence = "e" * 500
    await db_session.commit()

    result = await _chat(
        "get_entity_relationships", {"entity_id": str(graph.bob.id)}, db_session
    )

    incoming = [r for r in result["relationships"] if r["direction"] == "incoming"]
    assert incoming[0]["evidence"] == "e" * 200


async def _hub_with_outgoing(db, count):
    hub = await _entity(db, "hub")
    for index in range(count):
        spoke = await _entity(db, f"spoke{index}")
        await _rel(db, hub, spoke, "uses", 0.5)
    return hub


async def test_entity_relationships_fill_the_limit_from_either_direction(db_session):
    hub = await _hub_with_outgoing(db_session, 15)

    result = await _chat(
        "get_entity_relationships", {"entity_id": str(hub.id)}, db_session
    )

    assert result["count"] == 15


async def test_entity_relationships_limit_of_one_returns_one(db_session):
    hub = await _hub_with_outgoing(db_session, 3)

    result = await _chat(
        "get_entity_relationships", {"entity_id": str(hub.id), "limit": 1}, db_session
    )

    assert result["count"] == 1


async def test_entity_relationships_never_exceed_the_limit(db_session):
    graph = await _graph(db_session)
    for index in range(6):
        spoke = await _entity(db_session, f"spoke{index}")
        await _rel(db_session, graph.acme, spoke, "owns", 0.5)

    result = await _chat(
        "get_entity_relationships",
        {"entity_id": str(graph.acme.id), "limit": 4},
        db_session,
    )

    assert 0 < result["count"] <= 4
    assert len(result["relationships"]) == result["count"]


async def test_entity_relationships_non_numeric_limit_is_an_error_result(db_session):
    graph = await _graph(db_session)

    result = await _chat(
        "get_entity_relationships",
        {"entity_id": str(graph.ada.id), "limit": "all"},
        db_session,
    )

    # Refused or read as the default: either way an answer, not a crash.
    assert isinstance(result, (dict, list))


async def test_entity_relationships_negative_limit_does_not_list_everything(
    db_session,
):
    hub = await _hub_with_outgoing(db_session, 5)

    result = await _chat(
        "get_entity_relationships", {"entity_id": str(hub.id), "limit": -2}, db_session
    )

    assert _is_refusal(result) or result["relationships"] == []


# --------------------------------------------------------------------------
# find_documents_by_entity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("params", [{}, {"entity_id": None}, {"entity_id": "nope"}])
async def test_documents_by_entity_refuse_a_missing_or_malformed_id(db_session, params):
    result = await _chat("find_documents_by_entity", params, db_session)

    assert _is_refusal(result)
    assert "Invalid entity ID" in result["error"]


async def test_documents_by_entity_report_an_unknown_entity(db_session):
    await _graph(db_session)
    missing = str(uuid4())

    result = await _chat("find_documents_by_entity", {"entity_id": missing}, db_session)

    assert result == {"error": f"Entity not found: {missing}"}


async def test_documents_by_entity_group_mentions_per_document(db_session):
    graph = await _graph(db_session)
    await _mention(db_session, graph.ada, graph.paper, text="Ada", sentence="x" * 300)
    await _mention(db_session, graph.ada, graph.paper, text="Ada", sentence=None)

    result = await _chat(
        "find_documents_by_entity", {"entity_id": str(graph.ada.id)}, db_session
    )

    assert result["entity"] == {
        "id": str(graph.ada.id),
        "name": "Ada Lovelace",
        "type": "person",
    }
    assert result["document_count"] == 2
    # Newest document first.
    assert [doc["title"] for doc in result["documents"]] == ["Paper", "Memo"]
    paper, memo = result["documents"]
    assert paper["id"] == str(graph.paper.id)
    assert paper["mention_count"] == 4
    assert len(paper["mentions"]) == 3
    assert all(len(m["sentence"] or "") <= 200 for m in paper["mentions"])
    assert memo["mention_count"] == 1
    assert memo["mentions"] == [{"text": "Ada", "sentence": "Ada s2"}]
    json.dumps(result)


async def test_documents_by_entity_exclude_other_entities(db_session):
    graph = await _graph(db_session)

    result = await _chat(
        "find_documents_by_entity", {"entity_id": str(graph.bob.id)}, db_session
    )
    unmentioned = await _chat(
        "find_documents_by_entity", {"entity_id": str(graph.caching.id)}, db_session
    )

    assert [doc["title"] for doc in result["documents"]] == ["Memo"]
    assert result["documents"][0]["mention_count"] == 1
    assert unmentioned["documents"] == []
    assert unmentioned["document_count"] == 0


async def test_documents_by_entity_limit_keeps_the_newest(db_session):
    graph = await _graph(db_session)

    result = await _chat(
        "find_documents_by_entity",
        {"entity_id": str(graph.acme.id), "limit": 1},
        db_session,
    )

    assert [doc["title"] for doc in result["documents"]] == ["Paper"]
    assert result["document_count"] == 1


async def test_documents_by_entity_limit_counts_documents_not_mentions(db_session):
    source = await _source(db_session)
    heavy = await _doc(db_session, source, "Heavy", age_days=1)
    light = await _doc(db_session, source, "Light", age_days=2)
    entity = await _entity(db_session, "Transformer")
    for _ in range(25):
        db_session.add(
            EntityMention(entity_id=entity.id, document_id=heavy.id, text="T")
        )
    db_session.add(EntityMention(entity_id=entity.id, document_id=light.id, text="T"))
    await db_session.commit()

    result = await _chat(
        "find_documents_by_entity", {"entity_id": str(entity.id)}, db_session
    )

    assert [doc["title"] for doc in result["documents"]] == ["Heavy", "Light"]
    assert result["documents"][0]["mention_count"] == 25


async def test_documents_by_entity_non_numeric_limit_is_an_error_result(db_session):
    graph = await _graph(db_session)

    result = await _chat(
        "find_documents_by_entity",
        {"entity_id": str(graph.ada.id), "limit": "all"},
        db_session,
    )

    # Refused or read as the default: either way an answer, not a crash.
    assert isinstance(result, (dict, list))


async def test_documents_by_entity_negative_limit_does_not_list_everything(
    db_session,
):
    graph = await _graph(db_session)
    source = await _source(db_session, "Extra")
    extra = await _doc(db_session, source, "Extra", age_days=3)
    await _mention(db_session, graph.ada, extra)

    result = await _chat(
        "find_documents_by_entity",
        {"entity_id": str(graph.ada.id), "limit": -1},
        db_session,
    )

    assert _is_refusal(result) or len(result["documents"]) <= 1


# --------------------------------------------------------------------------
# get_document_knowledge_graph
# --------------------------------------------------------------------------


@pytest.mark.parametrize("params", [{}, {"document_id": None}, {"document_id": "nope"}])
async def test_document_graph_refuses_a_missing_or_malformed_id(db_session, params):
    result = await _chat("get_document_knowledge_graph", params, db_session)

    assert _is_refusal(result)
    assert "Invalid document ID" in result["error"]


async def test_document_graph_reports_an_unknown_document(db_session):
    await _graph(db_session)
    missing = str(uuid4())

    result = await _chat(
        "get_document_knowledge_graph", {"document_id": missing}, db_session
    )

    assert result == {"error": f"Document not found: {missing}"}


async def test_document_graph_holds_that_documents_entities_and_relations(db_session):
    graph = await _graph(db_session)

    result = await _chat(
        "get_document_knowledge_graph", {"document_id": str(graph.memo.id)}, db_session
    )

    assert result["document"] == {"id": str(graph.memo.id), "title": "Memo"}
    assert {node["name"] for node in result["nodes"]} == {
        "Ada Lovelace",
        "Bob Stone",
        "Acme Corp",
    }
    assert result["node_count"] == 3
    ada = next(n for n in result["nodes"] if n["name"] == "Ada Lovelace")
    assert ada == {
        "id": str(graph.ada.id),
        "name": "Ada Lovelace",
        "type": "person",
        "description": "mathematician",
    }
    assert {
        (edge["source"], edge["target"], edge["relation_type"], edge["confidence"])
        for edge in result["edges"]
    } == {
        (str(graph.bob.id), str(graph.acme.id), "works_for", 0.4),
        (str(graph.ada.id), str(graph.bob.id), "knows", 0.7),
    }
    assert result["edge_count"] == 2
    json.dumps(result)


async def test_document_graph_lists_an_entity_once_however_often_mentioned(db_session):
    graph = await _graph(db_session)

    result = await _chat(
        "get_document_knowledge_graph", {"document_id": str(graph.paper.id)}, db_session
    )

    assert sorted(node["name"] for node in result["nodes"]) == [
        "Acme Corp",
        "Ada Lovelace",
    ]
    assert result["node_count"] == 2
    assert result["edge_count"] == 1
    assert result["edges"][0]["relation_type"] == "works_for"


async def test_document_graph_of_a_document_with_no_graph_is_empty(db_session):
    graph = await _graph(db_session)
    source = await _source(db_session, "Other")
    bare = await _doc(db_session, source, "Bare")

    result = await _chat(
        "get_document_knowledge_graph", {"document_id": str(bare.id)}, db_session
    )

    assert result["nodes"] == []
    assert result["edges"] == []
    assert result["node_count"] == 0
    assert result["edge_count"] == 0
    assert graph.paper.id != bare.id


async def test_document_graph_truncates_long_evidence(db_session):
    graph = await _graph(db_session)
    graph.works_ada.evidence = "e" * 400
    await db_session.commit()

    result = await _chat(
        "get_document_knowledge_graph", {"document_id": str(graph.paper.id)}, db_session
    )

    assert result["edges"][0]["evidence"] == "e" * 150


async def test_document_graph_edges_only_join_nodes_it_returned(db_session):
    graph = await _graph(db_session)
    await _rel(db_session, graph.ada, graph.caching, "studies", 0.8, graph.paper)

    result = await _chat(
        "get_document_knowledge_graph", {"document_id": str(graph.paper.id)}, db_session
    )

    node_ids = {node["id"] for node in result["nodes"]}
    for edge in result["edges"]:
        assert edge["source"] in node_ids and edge["target"] in node_ids, edge


# --------------------------------------------------------------------------
# get_global_knowledge_graph
# --------------------------------------------------------------------------


def _names(result):
    return [node["name"] for node in result["nodes"]]


def _edge_types(result):
    return [(edge["type"], edge["confidence"]) for edge in result["edges"]]


async def test_global_graph_orders_nodes_by_mentions_and_edges_by_confidence(
    db_session,
):
    graph = await _graph(db_session)

    result = await _chat("get_global_knowledge_graph", {}, db_session)

    assert _names(result) == ["Ada Lovelace", "Acme Corp", "Bob Stone"]
    assert [node["mention_count"] for node in result["nodes"]] == [3, 2, 1]
    assert result["nodes"][0]["id"] == str(graph.ada.id)
    assert _edge_types(result) == [
        ("works_for", 0.9),
        ("knows", 0.7),
        ("works_for", 0.4),
    ]
    metadata = result["metadata"]
    assert metadata["total_entities"] == 4
    assert metadata["total_relationships"] == 3
    assert metadata["filtered_nodes"] == 3
    assert metadata["filtered_edges"] == 3
    assert metadata["entity_types"] == ["org", "person"]
    assert metadata["relation_types"] == ["knows", "works_for"]
    json.dumps(result)


async def test_global_graph_on_an_empty_graph(db_session):
    result = await _chat("get_global_knowledge_graph", {}, db_session)

    assert result["nodes"] == []
    assert result["edges"] == []
    assert result["metadata"]["total_entities"] == 0


async def test_global_graph_filters_by_entity_type(db_session):
    await _graph(db_session)

    result = await _chat(
        "get_global_knowledge_graph", {"entity_types": ["person"]}, db_session
    )

    assert _names(result) == ["Ada Lovelace", "Bob Stone"]
    assert _edge_types(result) == [("knows", 0.7)]


async def test_global_graph_filters_by_relation_type(db_session):
    await _graph(db_session)

    result = await _chat(
        "get_global_knowledge_graph", {"relation_types": ["works_for"]}, db_session
    )

    assert len(result["nodes"]) == 3
    assert _edge_types(result) == [("works_for", 0.9), ("works_for", 0.4)]


async def test_global_graph_filters_by_confidence(db_session):
    await _graph(db_session)

    result = await _chat(
        "get_global_knowledge_graph", {"min_confidence": 0.5}, db_session
    )

    assert _edge_types(result) == [("works_for", 0.9), ("knows", 0.7)]


async def test_global_graph_filters_by_minimum_mentions(db_session):
    await _graph(db_session)

    result = await _chat("get_global_knowledge_graph", {"min_mentions": 2}, db_session)

    assert _names(result) == ["Ada Lovelace", "Acme Corp"]
    assert _edge_types(result) == [("works_for", 0.9)]


async def test_global_graph_min_mentions_zero_includes_unmentioned_entities(
    db_session,
):
    await _graph(db_session)

    result = await _chat("get_global_knowledge_graph", {"min_mentions": 0}, db_session)

    assert "Caching" in _names(result)
    assert len(result["nodes"]) == 4


async def test_global_graph_limits_nodes_to_the_most_mentioned(db_session):
    await _graph(db_session)

    one = await _chat("get_global_knowledge_graph", {"limit_nodes": 1}, db_session)
    two = await _chat("get_global_knowledge_graph", {"limit_nodes": 2}, db_session)

    assert _names(one) == ["Ada Lovelace"]
    assert one["edges"] == []
    assert _names(two) == ["Ada Lovelace", "Acme Corp"]
    assert _edge_types(two) == [("works_for", 0.9)]


async def test_global_graph_limits_edges_to_the_most_confident(db_session):
    await _graph(db_session)

    result = await _chat("get_global_knowledge_graph", {"limit_edges": 2}, db_session)

    assert _edge_types(result) == [("works_for", 0.9), ("knows", 0.7)]
    assert len(result["nodes"]) == 3


async def test_global_graph_searches_entity_names_case_insensitively(db_session):
    await _graph(db_session)

    result = await _chat("get_global_knowledge_graph", {"search": "ACME"}, db_session)

    assert _names(result) == ["Acme Corp"]
    assert result["edges"] == []


async def test_global_graph_caps_huge_limits(db_session):
    await _graph(db_session)

    result = await _chat(
        "get_global_knowledge_graph",
        {"limit_nodes": 10**9, "limit_edges": 10**9},
        db_session,
    )

    assert len(result["nodes"]) == 3
    assert len(result["edges"]) == 3


@pytest.mark.parametrize(
    "key", ["limit_nodes", "limit_edges", "min_mentions", "min_confidence"]
)
async def test_global_graph_non_numeric_parameter_is_an_error_result(db_session, key):
    await _graph(db_session)

    result = await _chat("get_global_knowledge_graph", {key: "many"}, db_session)

    # Refused or read as the default: either way an answer, not a crash.
    assert isinstance(result, (dict, list))


@pytest.mark.parametrize("key, collection", [("limit_nodes", "nodes")])
async def test_global_graph_negative_node_limit_does_not_return_everything(
    db_session, key, collection
):
    await _graph(db_session)

    result = await _chat("get_global_knowledge_graph", {key: -1}, db_session)

    assert _is_refusal(result) or len(result[collection]) <= 1


async def test_global_graph_negative_edge_limit_does_not_return_everything(
    db_session,
):
    await _graph(db_session)

    result = await _chat("get_global_knowledge_graph", {"limit_edges": -1}, db_session)

    assert _is_refusal(result) or len(result["edges"]) <= 1


# --------------------------------------------------------------------------
# rebuild_document_knowledge_graph
# --------------------------------------------------------------------------

_EXTRACTABLE = "Contact alice@example.com or read https://example.com/docs today."


async def _rebuildable(db):
    """A document with a stale graph and a chunk the rule extractor can read."""
    graph = await _graph(db)
    await _chunk(db, graph.paper, _EXTRACTABLE, 0)
    return graph


async def test_rebuild_is_refused_for_a_non_admin_and_changes_nothing(
    db_session, test_user
):
    graph = await _rebuildable(db_session)

    result = await _chat(
        "rebuild_document_knowledge_graph",
        {"document_id": str(graph.paper.id)},
        db_session,
        user=test_user,
    )

    assert result == {"error": "Admin privileges required for this tool"}
    assert await _count(db_session, EntityMention) == 6
    assert await _count(db_session, Relationship) == 3


async def test_rebuild_is_refused_for_an_unknown_user(db_session):
    graph = await _rebuildable(db_session)

    result = await _chat(
        "rebuild_document_knowledge_graph",
        {"document_id": str(graph.paper.id)},
        db_session,
    )

    assert result == {"error": "Admin privileges required for this tool"}
    assert await _count(db_session, EntityMention) == 6


@pytest.mark.parametrize("params", [{}, {"document_id": None}, {"document_id": "nope"}])
async def test_rebuild_refuses_a_missing_or_malformed_id(
    db_session, admin_user, params
):
    await _rebuildable(db_session)

    result = await _chat(
        "rebuild_document_knowledge_graph", params, db_session, user=admin_user
    )

    assert _is_refusal(result)
    assert "Invalid document ID" in result["error"]
    assert await _count(db_session, EntityMention) == 6


async def test_rebuild_replaces_the_documents_graph_and_no_other(
    db_session, admin_user
):
    graph = await _rebuildable(db_session)
    paper_id, memo_id = graph.paper.id, graph.memo.id

    result = await _chat(
        "rebuild_document_knowledge_graph",
        {"document_id": str(paper_id)},
        db_session,
        user=admin_user,
    )

    assert "error" not in result
    assert result["document_id"] == str(paper_id)
    paper_mentions = (
        (
            await db_session.execute(
                select(EntityMention).where(EntityMention.document_id == paper_id)
            )
        )
        .scalars()
        .all()
    )
    # The stale mentions ("Ada", "Acme") are gone; what the chunk says is in.
    texts = {mention.text for mention in paper_mentions}
    assert "Ada" not in texts and "Acme" not in texts
    assert "alice@example.com" in texts
    assert result["mentions"] == len(paper_mentions) > 0
    assert result["relationships"] == await _count(
        db_session, Relationship, Relationship.document_id == paper_id
    )
    assert result["relationships"] == 0
    # The memo's graph is untouched.
    assert (
        await _count(db_session, EntityMention, EntityMention.document_id == memo_id)
        == 3
    )
    assert (
        await _count(db_session, Relationship, Relationship.document_id == memo_id) == 2
    )
    json.dumps(result)


async def test_rebuild_of_a_document_without_chunks_clears_its_graph(
    db_session, admin_user
):
    graph = await _graph(db_session)

    result = await _chat(
        "rebuild_document_knowledge_graph",
        {"document_id": str(graph.memo.id)},
        db_session,
        user=admin_user,
    )

    assert result == {
        "document_id": str(graph.memo.id),
        "mentions": 0,
        "relationships": 0,
    }
    assert (
        await _count(
            db_session, EntityMention, EntityMention.document_id == graph.memo.id
        )
        == 0
    )
    assert await _count(db_session, EntityMention) == 3


async def test_rebuild_reports_an_unknown_document(db_session, admin_user):
    await _graph(db_session)

    result = await _chat(
        "rebuild_document_knowledge_graph",
        {"document_id": str(uuid4())},
        db_session,
        user=admin_user,
    )

    assert _is_refusal(result)
    assert "not found" in result["error"].lower()


# --------------------------------------------------------------------------
# delete_entity
# --------------------------------------------------------------------------


async def _delete(db, user, entity_id, confirm_name=None):
    params = {"entity_id": entity_id}
    if confirm_name is not None:
        params["confirm_name"] = confirm_name
    return await _chat("delete_entity", params, db, user=user)


async def test_delete_entity_is_refused_for_a_non_admin(db_session, test_user):
    graph = await _graph(db_session)

    result = await _delete(db_session, test_user, str(graph.bob.id), "Bob Stone")

    assert result == {"error": "Admin privileges required for this tool"}
    assert await _count(db_session, Entity) == 4
    assert await _count(db_session, EntityMention) == 6
    assert await _count(db_session, Relationship) == 3


@pytest.mark.parametrize("entity_id", [None, "", "nope"])
async def test_delete_entity_refuses_a_missing_or_malformed_id(
    db_session, admin_user, entity_id
):
    await _graph(db_session)

    result = await _chat(
        "delete_entity",
        {"entity_id": entity_id, "confirm_name": "Bob Stone"},
        db_session,
        user=admin_user,
    )

    assert _is_refusal(result)
    assert "Invalid entity ID" in result["error"]
    assert await _count(db_session, Entity) == 4


async def test_delete_entity_reports_an_unknown_entity(db_session, admin_user):
    await _graph(db_session)
    missing = str(uuid4())

    result = await _delete(db_session, admin_user, missing, "Bob Stone")

    assert result == {"error": f"Entity not found: {missing}"}
    assert await _count(db_session, Entity) == 4


@pytest.mark.parametrize("confirm", [None, "", "bob stone", "Bob", " Bob Stone "])
async def test_delete_entity_requires_the_exact_canonical_name(
    db_session, admin_user, confirm
):
    graph = await _graph(db_session)

    result = await _delete(db_session, admin_user, str(graph.bob.id), confirm)

    assert result["error"] == "Confirmation required"
    assert result["entity"] == {
        "id": str(graph.bob.id),
        "canonical_name": "Bob Stone",
        "entity_type": "person",
    }
    assert "entity_deleted" not in result
    assert await _count(db_session, Entity) == 4
    assert await _count(db_session, EntityMention) == 6
    assert await _count(db_session, Relationship) == 3
    json.dumps(result)


async def test_delete_entity_removes_it_with_its_mentions_and_relationships(
    db_session, admin_user
):
    graph = await _graph(db_session)
    bob_id, ada_id, acme_id = graph.bob.id, graph.ada.id, graph.acme.id
    works_ada_id = graph.works_ada.id

    result = await _delete(db_session, admin_user, str(bob_id), "Bob Stone")

    assert result == {
        "entity_id": str(bob_id),
        "canonical_name": "Bob Stone",
        "mentions_deleted": 1,
        "relationships_deleted": 2,
        "entity_deleted": True,
    }
    json.dumps(result)
    db_session.expire_all()
    remaining = (await db_session.execute(select(Entity.canonical_name))).scalars()
    assert sorted(remaining) == ["Acme Corp", "Ada Lovelace", "Caching"]
    # Bob's mention and both relationships touching him are gone...
    assert (
        await _count(db_session, EntityMention, EntityMention.entity_id == bob_id) == 0
    )
    assert (
        await _count(
            db_session,
            Relationship,
            (Relationship.source_entity_id == bob_id)
            | (Relationship.target_entity_id == bob_id),
        )
        == 0
    )
    # ...and nothing else is.
    assert (
        await _count(db_session, EntityMention, EntityMention.entity_id == ada_id) == 3
    )
    assert (
        await _count(db_session, EntityMention, EntityMention.entity_id == acme_id) == 2
    )
    surviving = (await db_session.execute(select(Relationship.id))).scalars().all()
    assert surviving == [works_ada_id]
    assert await _count(db_session, Document) == 2
    assert await _count(db_session, DocumentSource) == 1


async def test_delete_entity_with_nothing_attached(db_session, admin_user):
    graph = await _graph(db_session)

    result = await _delete(db_session, admin_user, str(graph.caching.id), "Caching")

    assert result["entity_deleted"] is True
    assert result["mentions_deleted"] == 0
    assert result["relationships_deleted"] == 0
    assert await _count(db_session, Entity) == 3
    assert await _count(db_session, EntityMention) == 6
    assert await _count(db_session, Relationship) == 3


async def test_delete_entity_counts_a_self_relationship_once(db_session, admin_user):
    graph = await _graph(db_session)
    await _rel(db_session, graph.caching, graph.caching, "related_to", 0.5)

    result = await _delete(db_session, admin_user, str(graph.caching.id), "Caching")

    assert result.get("entity_deleted") is True
    assert result["relationships_deleted"] == 1
    assert await _count(db_session, Relationship) == 3


async def test_delete_entity_twice_says_it_is_gone(db_session, admin_user):
    graph = await _graph(db_session)
    bob_id = str(graph.bob.id)
    await _delete(db_session, admin_user, bob_id, "Bob Stone")

    again = await _delete(db_session, admin_user, bob_id, "Bob Stone")

    assert again == {"error": f"Entity not found: {bob_id}"}
    assert await _count(db_session, Entity) == 3
