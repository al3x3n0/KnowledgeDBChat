from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.memory import ConversationMemory
from app.services.agent_job_memory_service import AgentJobMemoryService


def _make_job(
    *, user_id, config=None, status=AgentJobStatus.COMPLETED.value
) -> AgentJob:
    return AgentJob(
        id=uuid4(),
        name="Memory Service Test Job",
        goal="Improve reliability and reduce duplicate memory extraction",
        job_type="research",
        user_id=user_id,
        status=status,
        config=config or {},
        results={"summary": "Job completed", "findings": [], "actions": []},
        execution_log=[],
    )


class _FakeScalars:
    def __init__(self, rows):
        self._rows = list(rows)

    def all(self):
        return list(self._rows)


class _FakeExecuteResult:
    def __init__(self, rows):
        self._rows = list(rows)

    def scalars(self):
        return _FakeScalars(self._rows)


class _FakeAsyncSession:
    def __init__(self, *, existing_memories=None, jobs_by_id=None):
        self._existing_memories = list(existing_memories or [])
        self._jobs_by_id = {str(k): v for k, v in (jobs_by_id or {}).items()}
        self.added = []
        self.commit_calls = 0
        self.rollback_calls = 0
        self.refreshed = []

    async def execute(self, _query):
        return _FakeExecuteResult(self._existing_memories)

    async def get(self, _model, key):
        return self._jobs_by_id.get(str(key))

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commit_calls += 1

    async def refresh(self, obj):
        self.refreshed.append(obj)
        return None

    async def rollback(self):
        self.rollback_calls += 1
        return None


def test_memory_dedup_signature_normalizes_content_scope_and_role():
    service = AgentJobMemoryService()

    left = service._build_memory_dedup_signature(
        memory_type="Insight",
        content="  Use   chunked retries\nfor flaky API calls. ",
        project_scope="Repo Alpha",
        agent_role="Research Engineer",
    )
    right = service._build_memory_dedup_signature(
        memory_type="insight",
        content="Use chunked retries for flaky API calls!",
        project_scope="repo-alpha",
        agent_role="research-engineer",
    )

    assert left
    assert left == right


@pytest.mark.asyncio
async def test_resolve_relaunch_dedup_scope_walks_to_root():
    service = AgentJobMemoryService()
    user_id = uuid4()

    root = _make_job(user_id=user_id, config={})
    parent = _make_job(user_id=user_id, config={"relaunch_from_job_id": str(root.id)})
    child = _make_job(user_id=user_id, config={"relaunch_from_job_id": str(parent.id)})

    db = _FakeAsyncSession(jobs_by_id={root.id: root, parent.id: parent})
    scope = await service._resolve_relaunch_dedup_scope(child, db)

    assert scope["is_relaunch_chain"] is True
    assert scope["root_job_id"] == str(root.id)
    assert {str(v) for v in scope["job_ids"]} == {
        str(child.id),
        str(parent.id),
        str(root.id),
    }


@pytest.mark.asyncio
async def test_extract_memories_deduplicates_existing_and_new_entries():
    service = AgentJobMemoryService()
    user_id = uuid4()

    root = _make_job(user_id=user_id, config={"project": "Repo Alpha"})
    job = _make_job(
        user_id=user_id,
        config={"project": "Repo Alpha", "relaunch_from_job_id": str(root.id)},
    )

    existing = ConversationMemory(
        user_id=user_id,
        job_id=root.id,
        memory_type="insight",
        content="Use chunked retries for flaky API calls.",
        importance_score=0.8,
        tags=["insight"],
        context={"project_scope": "Repo Alpha"},
        is_active=True,
    )

    db = _FakeAsyncSession(existing_memories=[existing], jobs_by_id={root.id: root})

    service.get_user_preferences = AsyncMock(
        return_value=SimpleNamespace(
            auto_extract_job_memories=True,
            agent_job_memory_types=["insight", "lesson"],
        )
    )
    service.llm_service.generate_response = AsyncMock(
        return_value=(
            "TYPE: insight | CONTENT: Use chunked retries for flaky API calls. | IMPORTANCE: 0.9 | TAGS: retry\n"
            "TYPE: lesson | CONTENT: Add exponential backoff after repeated retries. | IMPORTANCE: 0.8 | TAGS: backoff\n"
            "TYPE: lesson | CONTENT: Add exponential backoff after repeated retries. | IMPORTANCE: 0.6 | TAGS: duplicate\n"
        )
    )

    extraction_stats = {}
    created = await service.extract_memories_from_job(
        job=job,
        user_id=str(user_id),
        db=db,
        stats_out=extraction_stats,
    )

    assert len(created) == 1
    assert created[0].memory_type == "lesson"
    assert "exponential backoff" in created[0].content.lower()
    assert created[0].context.get("relaunch_root_job_id") == str(root.id)
    assert job.memories_created_count == 1
    assert len([row for row in db.added if isinstance(row, ConversationMemory)]) == 1
    assert extraction_stats.get("status") == "completed"
    assert extraction_stats.get("parsed_count") == 3
    assert extraction_stats.get("candidate_count") == 3
    assert extraction_stats.get("created_count") == 1
    assert extraction_stats.get("skipped_duplicates") == 2
    assert extraction_stats.get("is_relaunch_chain") is True
    assert extraction_stats.get("relaunch_root_job_id") == str(root.id)


@pytest.mark.asyncio
async def test_create_memory_from_job_returns_existing_duplicate_in_relaunch_scope():
    service = AgentJobMemoryService()
    user_id = uuid4()

    root = _make_job(user_id=user_id, config={"project": "Repo Alpha"})
    job = _make_job(
        user_id=user_id,
        config={"project": "Repo Alpha", "relaunch_from_job_id": str(root.id)},
    )

    existing = ConversationMemory(
        user_id=user_id,
        job_id=root.id,
        memory_type="lesson",
        content="Validate assumptions with a quick dry-run before full execution.",
        importance_score=0.7,
        tags=["lesson"],
        context={"project_scope": "Repo Alpha"},
        is_active=True,
    )

    db = _FakeAsyncSession(existing_memories=[existing], jobs_by_id={root.id: root})
    created = await service.create_memory_from_job(
        job=job,
        memory_type="lesson",
        content="Validate assumptions with a quick dry-run before full execution.",
        user_id=str(user_id),
        db=db,
    )

    assert created is existing
    assert len([row for row in db.added if isinstance(row, ConversationMemory)]) == 0


@pytest.mark.asyncio
async def test_failed_extraction_reloads_job_after_rollback():
    """A rollback expires the caller's job, so the service must reload it.

    Otherwise the next attribute read in the caller happens outside an
    awaitable context and raises MissingGreenlet, which used to abort job
    finalization before chained jobs were triggered.
    """
    service = AgentJobMemoryService()
    user_id = uuid4()
    job = _make_job(user_id=user_id, config={})

    db = _FakeAsyncSession()
    service.get_user_preferences = AsyncMock(
        return_value=SimpleNamespace(
            auto_extract_job_memories=True,
            agent_job_memory_types=["insight", "lesson"],
        )
    )
    service.llm_service.generate_response = AsyncMock(
        side_effect=RuntimeError("LLM service error: Failed to generate response")
    )

    extraction_stats = {}
    created = await service.extract_memories_from_job(
        job=job,
        user_id=str(user_id),
        db=db,
        stats_out=extraction_stats,
    )

    assert created == []
    assert extraction_stats.get("status") == "failed"
    assert db.rollback_calls == 1
    assert job in db.refreshed
