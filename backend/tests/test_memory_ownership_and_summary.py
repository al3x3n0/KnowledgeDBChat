"""A memory can only be changed by its owner, and a summary can be made.

`MemoryService.update_memory` looked a memory up by id alone, so
`PUT /memory/{id}` let any signed-in user rewrite any memory whose id they
had -- get and delete beside it were scoped. And `generate_memory_summary`
took the `(memories, total)` pair `get_memories` returns as the list itself,
so every summary died on `list.memory_type`.
"""

import pytest
from sqlalchemy import select

from app.models.memory import ConversationMemory
from app.schemas.memory import MemoryCreate, MemorySummaryRequest
from app.services.memory_service import MemoryService

pytestmark = pytest.mark.unit


async def _remember(db, user, content, memory_type="fact"):
    return await MemoryService().create_memory(
        user.id, MemoryCreate(memory_type=memory_type, content=content), db
    )


async def test_another_users_memory_cannot_be_rewritten(
    client, db_session, test_user, admin_user, auth_headers
):
    theirs = await _remember(db_session, admin_user, "The deploy key rotates monthly")

    response = client.put(
        f"/api/v1/memory/{theirs.id}",
        json={"content": "overwritten by a stranger"},
        headers=auth_headers,
    )

    assert response.status_code == 404
    stored = (
        await db_session.execute(
            select(ConversationMemory.content).where(ConversationMemory.id == theirs.id)
        )
    ).scalar_one()
    assert stored == "The deploy key rotates monthly"


async def test_an_owner_can_still_update_their_memory(
    client, db_session, test_user, auth_headers
):
    mine = await _remember(db_session, test_user, "Prefers tabs")

    response = client.put(
        f"/api/v1/memory/{mine.id}",
        json={"content": "Prefers spaces"},
        headers=auth_headers,
    )

    assert response.status_code == 200, response.text
    assert response.json()["content"] == "Prefers spaces"


async def test_a_summary_is_made_from_the_users_memories(db_session, test_user):
    service = MemoryService()
    seen = {}

    class _LLM:
        async def generate_response(self, **kwargs):
            seen.update(kwargs)
            return "They like spaces and work on caches."

    service.llm_service = _LLM()
    await _remember(db_session, test_user, "Works on cache prefetchers")
    await _remember(db_session, test_user, "Prefers spaces", memory_type="preference")

    summary = await service.generate_memory_summary(
        test_user.id, MemorySummaryRequest(), db_session
    )

    assert summary.summary == "They like spaces and work on caches."
    assert summary.memory_count == 2
    assert summary.key_facts == ["Works on cache prefetchers"]
    assert summary.preferences == ["Prefers spaces"]
    assert "cache prefetchers" in seen["query"]


async def test_no_memories_is_said_plainly(db_session, test_user):
    summary = await MemoryService().generate_memory_summary(
        test_user.id, MemorySummaryRequest(), db_session
    )
    assert summary.memory_count == 0
    assert "No memories" in summary.summary
