"""create_memory, search_memories, recall_memories and get_memory_stats,
called through their real handlers.

The four tools are nested functions registered by
`build_autonomous_memory_provider`. Every test here calls the registered
handler with the real `MemoryService` against the in-memory database and is
judged on the `conversation_memories` rows: a memory is only stored if the row
exists with the owner, type and importance it was asked for, and only useful
if search and recall hand it back to that owner and to nobody else.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.agent_core.tool_specs.memory import SPECS
from app.models.memory import ConversationMemory
from app.schemas.memory import MemoryCreate
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_memory_provider,
)
from app.services.memory_service import MemoryService

pytestmark = pytest.mark.unit

TOOLS = ("create_memory", "search_memories", "recall_memories", "get_memory_stats")
SPEC = {spec.name: spec for spec in SPECS}
CATEGORIES = SPEC["create_memory"].parameters["properties"]["category"]["enum"]
# Written by the agent runtime by constructing ConversationMemory directly;
# MemoryCreate refuses them on purpose.
RUNTIME_TYPES = ("finding", "insight", "pattern", "lesson")

# Distinct vocabularies, so no two generated memories look like duplicates.
WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu"
).split()


def _provider():
    return build_autonomous_memory_provider(
        SimpleNamespace(memory_service=MemoryService())
    )


async def _call(db, user, tool, params):
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=db,
        service=None,
        user_id=str(user.id),
        job=SimpleNamespace(id=uuid4(), user_id=user.id, config={}),
        state={},
    )
    return await _provider()._handlers[tool](params, ctx)


async def _rows(db, user=None):
    query = select(ConversationMemory)
    if user is not None:
        query = query.where(ConversationMemory.user_id == user.id)
    result = await db.execute(query.execution_options(populate_existing=True))
    return result.scalars().all()


async def _store(db, user, content, memory_type="fact", importance=0.5, **fields):
    """Put a row in the table directly, as the agent runtime does."""
    row = ConversationMemory(
        user_id=user.id,
        memory_type=memory_type,
        content=content,
        importance_score=importance,
        **fields,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


def _distinct(index):
    """A sentence sharing no word with any other index's sentence."""
    return f"{WORDS[index % 26]}{index} note{index} about{index} topic{index}"


def _contents(result):
    return [memory["content"] for memory in result["data"]["memories"]]


# ---------------------------------------------------------------------------
# Registration and declaration
# ---------------------------------------------------------------------------


def test_the_provider_answers_all_four_tools_in_autonomous_mode():
    provider = _provider()

    autonomous = AgentToolExecutionContext(mode="autonomous", db=None, service=None)
    chat = AgentToolExecutionContext(mode="chat", db=None, service=None)

    assert set(TOOLS) <= provider.supported_tools
    for tool in TOOLS:
        assert provider.can_handle(tool, autonomous)
        # Chat has no job to take the owner from; this provider is not for it.
        assert not provider.can_handle(tool, chat)


def test_every_tool_is_declared_to_the_model():
    from app.services.agent_tools import AGENT_TOOLS

    declared = {tool["name"] for tool in AGENT_TOOLS}

    assert set(TOOLS) <= declared


def test_only_create_memory_is_declared_as_a_write():
    assert SPEC["create_memory"].effects == "write"
    for name in ("search_memories", "recall_memories", "get_memory_stats"):
        assert SPEC[name].effects != "write"


def test_the_categories_offered_are_exactly_the_ones_the_schema_accepts():
    """An enum wider than MemoryCreate offers a category that is then refused."""
    for category in CATEGORIES:
        assert MemoryCreate(memory_type=category, content="x").memory_type == category

    assert (
        SPEC["search_memories"].parameters["properties"]["category_filter"]["enum"]
        == CATEGORIES
    )
    for runtime_type in RUNTIME_TYPES:
        assert runtime_type not in CATEGORIES
        with pytest.raises(ValueError):
            MemoryCreate(memory_type=runtime_type, content="x")


# ---------------------------------------------------------------------------
# create_memory
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("params", [{}, {"content": ""}, {"content": "   \n"}])
async def test_create_refuses_missing_content_and_stores_nothing(
    db_session, test_user, params
):
    result = await _call(db_session, test_user, "create_memory", params)

    assert result == {"error": "content is required"}
    assert await _rows(db_session) == []


async def test_create_stores_a_row_owned_by_the_jobs_user(db_session, test_user):
    result = await _call(
        db_session,
        test_user,
        "create_memory",
        {"content": "  The API uses OAuth2 for authentication  "},
    )

    assert result["success"] is True
    rows = await _rows(db_session)
    assert len(rows) == 1
    row = rows[0]
    assert result["data"]["memory_id"] == str(row.id)
    assert row.user_id == test_user.id
    assert row.content == "The API uses OAuth2 for authentication"
    assert row.memory_type == "fact"
    assert row.importance_score == 0.5
    assert row.is_active is True
    assert row.access_count == 0


@pytest.mark.parametrize("category", CATEGORIES)
async def test_create_stores_every_advertised_category(db_session, test_user, category):
    result = await _call(
        db_session,
        test_user,
        "create_memory",
        {"content": f"A memory filed under {category}", "category": category},
    )

    assert result.get("success") is True, result
    rows = await _rows(db_session, test_user)
    assert [row.memory_type for row in rows] == [category]


@pytest.mark.parametrize("category", ["pattern", "finding", "nonsense"])
async def test_create_refuses_a_category_outside_the_vocabulary(
    db_session, test_user, category
):
    result = await _call(
        db_session,
        test_user,
        "create_memory",
        {"content": "Something to remember", "category": category},
    )

    assert "success" not in result
    assert "memory_type must be one of" in result["error"]
    assert await _rows(db_session) == []


@pytest.mark.parametrize(
    "given, stored",
    [(0.8, 0.8), (1.0, 1.0), (1.5, 1.0), (-0.3, 0.0), ("0.25", 0.25), (None, 0.5)],
)
async def test_create_stores_importance_clamped_to_the_unit_interval(
    db_session, test_user, given, stored
):
    result = await _call(
        db_session,
        test_user,
        "create_memory",
        {"content": "Importance is recorded", "importance": given},
    )

    assert result.get("success") is True, result
    (row,) = await _rows(db_session)
    assert row.importance_score == stored


async def test_create_stores_an_importance_of_zero(db_session, test_user):
    await _call(
        db_session,
        test_user,
        "create_memory",
        {"content": "Barely worth keeping", "importance": 0.0},
    )

    (row,) = await _rows(db_session)
    assert row.importance_score == 0.0


async def test_create_refuses_a_non_numeric_importance_without_raising(
    db_session, test_user
):
    result = await _call(
        db_session,
        test_user,
        "create_memory",
        {"content": "Importance was a word", "importance": "high"},
    )

    assert "error" in result
    assert await _rows(db_session) == []


async def test_create_stores_metadata_as_context_and_lifts_tags_out(
    db_session, test_user
):
    metadata = {"source": "research", "tags": ["ml", "nlp"]}

    result = await _call(
        db_session,
        test_user,
        "create_memory",
        {"content": "Transformers dominate NLP benchmarks", "metadata": metadata},
    )

    assert result.get("success") is True, result
    (row,) = await _rows(db_session)
    assert row.context == {"source": "research"}
    assert row.tags == ["ml", "nlp"]
    # The caller's own dictionary is not edited on the way.
    assert metadata == {"source": "research", "tags": ["ml", "nlp"]}


async def test_create_ignores_metadata_that_is_not_an_object(db_session, test_user):
    result = await _call(
        db_session,
        test_user,
        "create_memory",
        {"content": "Metadata was a string", "metadata": "not a mapping"},
    )

    assert result.get("success") is True, result
    (row,) = await _rows(db_session)
    assert row.context is None
    assert row.tags is None


async def test_create_reports_the_content_truncated_but_stores_it_whole(
    db_session, test_user
):
    content = " ".join(f"word{i}" for i in range(200))

    result = await _call(db_session, test_user, "create_memory", {"content": content})

    (row,) = await _rows(db_session)
    assert row.content == content
    assert result["data"]["content"] == content[:200]


async def test_creating_the_same_memory_twice_keeps_one_row(db_session, test_user):
    content = "The deploy pipeline requires a signed tag"
    first = await _call(
        db_session, test_user, "create_memory", {"content": content, "importance": 0.9}
    )
    second = await _call(
        db_session, test_user, "create_memory", {"content": content, "importance": 0.2}
    )

    (row,) = await _rows(db_session)
    assert second["data"]["memory_id"] == first["data"]["memory_id"] == str(row.id)
    # A repeat never makes a memory less important than it already was.
    assert row.importance_score == 0.9


async def test_the_same_content_in_another_category_is_a_second_memory(
    db_session, test_user
):
    content = "Ship the report by Friday"
    await _call(
        db_session, test_user, "create_memory", {"content": content, "category": "goal"}
    )
    await _call(
        db_session,
        test_user,
        "create_memory",
        {"content": content, "category": "constraint"},
    )

    rows = await _rows(db_session)
    assert sorted(row.memory_type for row in rows) == ["constraint", "goal"]


async def test_two_users_storing_the_same_content_each_keep_their_own(
    db_session, test_user, admin_user
):
    """Deduplication must not fold one tenant's memory into another's."""
    content = "The staging cluster lives in eu-west"
    mine = await _call(db_session, test_user, "create_memory", {"content": content})
    theirs = await _call(
        db_session,
        admin_user,
        "create_memory",
        {"content": content, "importance": 0.9},
    )

    assert mine["data"]["memory_id"] != theirs["data"]["memory_id"]
    (my_row,) = await _rows(db_session, test_user)
    (their_row,) = await _rows(db_session, admin_user)
    assert my_row.importance_score == 0.5
    assert their_row.importance_score == 0.9


async def test_a_memory_created_by_a_job_names_that_job(db_session, test_user):
    from app.models.agent_job import AgentJob

    job = AgentJob(
        name="Calling job",
        goal="Remember things",
        job_type="research",
        user_id=test_user.id,
        status="running",
    )
    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)
    ctx = AgentToolExecutionContext(
        mode="autonomous",
        db=db_session,
        service=None,
        user_id=str(test_user.id),
        job=job,
        state={},
    )

    result = await _provider()._handlers["create_memory"](
        {"content": "Learned during this run"}, ctx
    )

    assert result.get("success") is True, result
    (row,) = await _rows(db_session)
    assert row.job_id == job.id


# ---------------------------------------------------------------------------
# search_memories
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("params", [{}, {"query": ""}, {"query": "   "}])
async def test_search_refuses_a_missing_query(db_session, test_user, params):
    result = await _call(db_session, test_user, "search_memories", params)

    assert result == {"error": "query is required"}


async def test_search_with_nothing_stored_is_an_empty_success(db_session, test_user):
    result = await _call(db_session, test_user, "search_memories", {"query": "oauth"})

    assert result == {"success": True, "data": {"memories": [], "count": 0}}


async def test_a_created_memory_is_found_by_search(db_session, test_user):
    created = await _call(
        db_session,
        test_user,
        "create_memory",
        {
            "content": "The API uses OAuth2 for authentication",
            "category": "constraint",
            "importance": 0.7,
        },
    )

    result = await _call(
        db_session, test_user, "search_memories", {"query": "authentication"}
    )

    assert result["success"] is True
    assert result["data"]["count"] == 1
    assert result["data"]["memories"] == [
        {
            "id": created["data"]["memory_id"],
            "content": "The API uses OAuth2 for authentication",
            "importance": 0.7,
            "type": "constraint",
            "relevance": 1.0,
        }
    ]


async def test_search_ranks_the_memory_matching_the_query_first(db_session, test_user):
    await _store(db_session, test_user, "Prefers dark roast coffee", importance=0.9)
    await _store(
        db_session, test_user, "The API uses OAuth2 authentication", importance=0.2
    )

    result = await _call(
        db_session, test_user, "search_memories", {"query": "OAuth2 authentication"}
    )

    assert _contents(result)[0] == "The API uses OAuth2 authentication"


async def test_search_says_when_a_memory_does_not_match_the_query(
    db_session, test_user
):
    """Ranking is by shared words, so an unrelated memory can still come back
    on its importance. The result says so rather than passing it off as a
    match."""
    await _store(db_session, test_user, "Prefers dark roast coffee")
    await _store(db_session, test_user, "The ingress runs on kubernetes")

    result = await _call(
        db_session, test_user, "search_memories", {"query": "kubernetes ingress"}
    )

    by_content = {m["content"]: m["relevance"] for m in result["data"]["memories"]}
    assert by_content == {
        "The ingress runs on kubernetes": 1.0,
        "Prefers dark roast coffee": 0.0,
    }
    assert _contents(result)[0] == "The ingress runs on kubernetes"


async def test_search_never_returns_another_users_memories(
    db_session, test_user, admin_user
):
    await _store(db_session, test_user, "My laptop runs macOS", importance=0.3)
    await _store(db_session, admin_user, "Their laptop runs Linux", importance=0.9)

    mine = await _call(db_session, test_user, "search_memories", {"query": "laptop"})
    theirs = await _call(db_session, admin_user, "search_memories", {"query": "laptop"})

    assert _contents(mine) == ["My laptop runs macOS"]
    assert _contents(theirs) == ["Their laptop runs Linux"]


async def test_search_skips_a_deleted_memory(db_session, test_user):
    await _store(db_session, test_user, "Still remembered")
    await _store(db_session, test_user, "Forgotten on purpose", is_active=False)

    result = await _call(db_session, test_user, "search_memories", {"query": "x"})

    assert _contents(result) == ["Still remembered"]


@pytest.mark.parametrize("category", CATEGORIES)
async def test_search_filters_by_each_advertised_category(
    db_session, test_user, category
):
    for index, memory_type in enumerate(CATEGORIES):
        await _store(db_session, test_user, _distinct(index), memory_type=memory_type)

    result = await _call(
        db_session,
        test_user,
        "search_memories",
        {"query": "note", "category_filter": category},
    )

    assert [memory["type"] for memory in result["data"]["memories"]] == [category]


async def test_search_applies_the_minimum_importance(db_session, test_user):
    await _store(db_session, test_user, _distinct(1), importance=0.2)
    await _store(db_session, test_user, _distinct(2), importance=0.7)
    await _store(db_session, test_user, _distinct(3), importance=0.9)

    result = await _call(
        db_session,
        test_user,
        "search_memories",
        {"query": "note", "min_importance": 0.7},
    )

    assert sorted(m["importance"] for m in result["data"]["memories"]) == [0.7, 0.9]


async def test_search_with_a_minimum_importance_of_zero_keeps_everything(
    db_session, test_user
):
    await _store(db_session, test_user, _distinct(1), importance=0.0)
    await _store(db_session, test_user, _distinct(2), importance=0.6)

    result = await _call(
        db_session, test_user, "search_memories", {"query": "note", "min_importance": 0}
    )

    assert result["data"]["count"] == 2


async def test_search_refuses_an_out_of_range_minimum_importance(db_session, test_user):
    await _store(db_session, test_user, _distinct(1))

    result = await _call(
        db_session,
        test_user,
        "search_memories",
        {"query": "note", "min_importance": 1.5},
    )

    assert "success" not in result
    assert result["error"].startswith("Memory search failed")


async def test_search_returns_ten_by_default(db_session, test_user):
    for index in range(12):
        await _store(db_session, test_user, _distinct(index))

    result = await _call(db_session, test_user, "search_memories", {"query": "note"})

    assert result["data"]["count"] == 10
    assert len(result["data"]["memories"]) == 10


async def test_search_honours_a_smaller_limit_and_keeps_the_most_important(
    db_session, test_user
):
    for index in range(6):
        await _store(db_session, test_user, _distinct(index), importance=index / 10)

    result = await _call(
        db_session, test_user, "search_memories", {"query": "note", "limit": 3}
    )

    assert [m["importance"] for m in result["data"]["memories"]] == [0.5, 0.4, 0.3]


async def test_search_caps_the_limit_at_fifty(db_session, test_user):
    for index in range(55):
        await _store(db_session, test_user, _distinct(index))

    result = await _call(
        db_session, test_user, "search_memories", {"query": "note", "limit": 500}
    )

    assert result["data"]["count"] == 50


async def test_search_with_a_negative_limit_still_searches(db_session, test_user):
    await _store(db_session, test_user, _distinct(1))

    result = await _call(
        db_session, test_user, "search_memories", {"query": "note", "limit": -5}
    )

    assert result.get("success") is True, result


async def test_search_with_a_non_numeric_limit_does_not_raise(db_session, test_user):
    await _store(db_session, test_user, _distinct(1))

    result = await _call(
        db_session, test_user, "search_memories", {"query": "note", "limit": "many"}
    )

    assert isinstance(result, dict)


async def test_search_counts_each_returned_memory_as_accessed(db_session, test_user):
    await _store(db_session, test_user, _distinct(1), importance=0.9)
    await _store(db_session, test_user, _distinct(2), importance=0.1)

    await _call(db_session, test_user, "search_memories", {"query": "note", "limit": 1})

    counts = {row.importance_score: row.access_count for row in await _rows(db_session)}
    assert counts == {0.9: 1, 0.1: 0}


# ---------------------------------------------------------------------------
# recall_memories
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("params", [{}, {"topic": ""}, {"topic": "  "}])
async def test_recall_refuses_a_missing_topic(db_session, test_user, params):
    result = await _call(db_session, test_user, "recall_memories", params)

    assert result == {"error": "topic is required"}


async def test_recall_with_nothing_stored_is_an_empty_success(db_session, test_user):
    result = await _call(db_session, test_user, "recall_memories", {"topic": "x"})

    assert result == {"success": True, "data": {"memories": [], "count": 0}}


async def test_a_created_memory_is_recalled(db_session, test_user):
    created = await _call(
        db_session,
        test_user,
        "create_memory",
        {"content": "The team prefers trunk-based development", "category": "context"},
    )

    result = await _call(
        db_session, test_user, "recall_memories", {"topic": "development process"}
    )

    assert result["data"]["memories"] == [
        {
            "id": created["data"]["memory_id"],
            "content": "The team prefers trunk-based development",
            "importance": 0.5,
            "type": "context",
        }
    ]


async def test_recall_never_returns_another_users_memories(
    db_session, test_user, admin_user
):
    await _store(db_session, admin_user, "The admin's private note", importance=1.0)

    result = await _call(db_session, test_user, "recall_memories", {"topic": "note"})

    assert result == {"success": True, "data": {"memories": [], "count": 0}}


@pytest.mark.parametrize("memory_type", RUNTIME_TYPES)
async def test_recall_returns_the_types_only_the_runtime_writes(
    db_session, test_user, memory_type
):
    """A method stored as `pattern` is useless if recall chokes on its type."""
    await _store(
        db_session, test_user, "Bisect the input one element at a time", memory_type
    )

    result = await _call(db_session, test_user, "recall_memories", {"topic": "bisect"})

    assert result.get("success") is True, result
    assert [memory["type"] for memory in result["data"]["memories"]] == [memory_type]


async def test_recall_ignores_the_filters_search_would_apply(db_session, test_user):
    await _store(db_session, test_user, _distinct(1), "fact", importance=0.1)
    await _store(db_session, test_user, _distinct(2), "goal", importance=0.9)

    result = await _call(
        db_session,
        test_user,
        "recall_memories",
        {"topic": "note", "category_filter": "goal", "min_importance": 0.8},
    )

    assert result["data"]["count"] == 2


async def test_recall_returns_ten_by_default_and_caps_at_fifty(db_session, test_user):
    for index in range(55):
        await _store(db_session, test_user, _distinct(index))

    default = await _call(db_session, test_user, "recall_memories", {"topic": "note"})
    capped = await _call(
        db_session, test_user, "recall_memories", {"topic": "note", "limit": 999}
    )
    small = await _call(
        db_session, test_user, "recall_memories", {"topic": "note", "limit": 4}
    )

    assert default["data"]["count"] == 10
    assert capped["data"]["count"] == 50
    assert small["data"]["count"] == 4


async def test_recall_ranks_the_memory_about_the_topic_first(db_session, test_user):
    await _store(db_session, test_user, "Prefers dark roast coffee", importance=0.9)
    await _store(
        db_session, test_user, "gem5 crashes with an L2 prefetcher", importance=0.2
    )

    result = await _call(
        db_session, test_user, "recall_memories", {"topic": "gem5 prefetcher"}
    )

    assert _contents(result)[0] == "gem5 crashes with an L2 prefetcher"


# ---------------------------------------------------------------------------
# get_memory_stats
# ---------------------------------------------------------------------------


async def test_stats_for_a_user_with_no_memories_are_all_zero(db_session, test_user):
    result = await _call(db_session, test_user, "get_memory_stats", {})

    assert result["success"] is True
    assert result["data"]["total_memories"] == 0
    assert result["data"]["memories_by_type"] == {}
    assert result["data"]["recent_memories"] == 0


async def test_stats_count_by_type_and_the_types_sum_to_the_total(
    db_session, test_user
):
    layout = {"fact": 3, "preference": 2, "goal": 1, "pattern": 2}
    index = 0
    for memory_type, count in layout.items():
        for _ in range(count):
            await _store(db_session, test_user, _distinct(index), memory_type)
            index += 1

    result = await _call(db_session, test_user, "get_memory_stats", {})

    data = result["data"]
    assert data["memories_by_type"] == layout
    assert data["total_memories"] == 8
    assert sum(data["memories_by_type"].values()) == data["total_memories"]


async def test_stats_count_memories_made_through_the_tool(db_session, test_user):
    for index, category in enumerate(["fact", "fact", "constraint"]):
        await _call(
            db_session,
            test_user,
            "create_memory",
            {"content": _distinct(index), "category": category},
        )

    result = await _call(db_session, test_user, "get_memory_stats", {})

    assert result["data"]["total_memories"] == 3
    assert result["data"]["memories_by_type"] == {"fact": 2, "constraint": 1}
    assert result["data"]["recent_memories"] == 3


async def test_stats_count_only_the_last_seven_days_as_recent(db_session, test_user):
    now = datetime.utcnow()
    await _store(db_session, test_user, _distinct(1), created_at=now)
    await _store(
        db_session, test_user, _distinct(2), created_at=now - timedelta(days=6)
    )
    await _store(
        db_session, test_user, _distinct(3), created_at=now - timedelta(days=8)
    )
    await _store(
        db_session, test_user, _distinct(4), created_at=now - timedelta(days=90)
    )

    result = await _call(db_session, test_user, "get_memory_stats", {})

    assert result["data"]["total_memories"] == 4
    assert result["data"]["recent_memories"] == 2


async def test_stats_leave_out_deleted_memories(db_session, test_user):
    await _store(db_session, test_user, _distinct(1))
    await _store(db_session, test_user, _distinct(2), "goal", is_active=False)

    result = await _call(db_session, test_user, "get_memory_stats", {})

    assert result["data"]["total_memories"] == 1
    assert result["data"]["memories_by_type"] == {"fact": 1}
    assert result["data"]["recent_memories"] == 1


async def test_stats_count_only_the_callers_memories(db_session, test_user, admin_user):
    await _store(db_session, test_user, _distinct(1), "fact")
    for index in range(2, 6):
        await _store(db_session, admin_user, _distinct(index), "goal")

    mine = await _call(db_session, test_user, "get_memory_stats", {})
    theirs = await _call(db_session, admin_user, "get_memory_stats", {})

    assert mine["data"]["total_memories"] == 1
    assert mine["data"]["memories_by_type"] == {"fact": 1}
    assert theirs["data"]["total_memories"] == 4
    assert theirs["data"]["memories_by_type"] == {"goal": 4}


async def test_stats_report_the_most_accessed_memories(db_session, test_user):
    await _store(db_session, test_user, "Rarely read", access_count=1)
    await _store(db_session, test_user, "Read constantly", access_count=40)

    result = await _call(db_session, test_user, "get_memory_stats", {})

    most_accessed = result["data"]["most_accessed_memories"]
    assert most_accessed[0]["content"] == "Read constantly"
