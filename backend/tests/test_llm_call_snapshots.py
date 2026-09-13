"""Tests for LLM call snapshots: capture in LLMService + read-only API."""

import asyncio
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.core.config import settings
from app.models.llm_call_snapshot import LLMCallSnapshot
from app.services.llm_providers.base import LLMCompletion, LLMToolCall
from app.services.llm_service import LLMService
from app.utils.exceptions import LLMServiceError


@pytest.fixture(autouse=True)
def _snapshots_enabled(monkeypatch):
    monkeypatch.setattr(settings, "LLM_CALL_SNAPSHOT_ENABLED", True)


class FakeProvider:
    def __init__(self, completion=None, error=None):
        self.completion = completion
        self.error = error

    async def complete(self, messages, **kwargs):
        if self.error:
            raise self.error
        return self.completion


def _fake_completion():
    return LLMCompletion(
        text='{"ok": true}',
        structured={"ok": True},
        tool_calls=[
            LLMToolCall(id="tc1", name="search_documents", arguments={"q": "x"})
        ],
        provider="anthropic",
        model="claude-opus-4-8",
        prompt_tokens=100,
        completion_tokens=20,
        total_tokens=120,
    )


@pytest.mark.asyncio
async def test_generate_structured_records_snapshot(db_session, test_user, monkeypatch):
    import app.services.llm_providers as providers_module

    monkeypatch.setattr(
        providers_module,
        "build_provider",
        lambda *args, **kwargs: FakeProvider(_fake_completion()),
    )
    service = LLMService()
    job_id = uuid4()

    completion = await service.generate_structured(
        system_prompt="stable system",
        user_message="decide now",
        response_schema={"type": "object"},
        user_id=test_user.id,
        db=db_session,
        snapshot_context={"job_id": str(job_id), "iteration": 4, "phase": "thinking"},
    )
    await db_session.commit()

    assert completion.structured == {"ok": True}
    rows = (await db_session.execute(select(LLMCallSnapshot))).scalars().all()
    assert len(rows) == 1
    snap = rows[0]
    assert snap.job_id == job_id
    assert snap.iteration == 4
    assert snap.phase == "thinking"
    assert snap.provider == "anthropic"
    assert snap.response_text == '{"ok": true}'
    assert snap.structured == {"ok": True}
    assert snap.tool_calls[0]["name"] == "search_documents"
    assert snap.prompt_tokens == 100
    roles = [m["role"] for m in snap.request["messages"]]
    assert roles == ["system", "user"]
    assert snap.request["messages"][0]["content"] == "stable system"


@pytest.mark.asyncio
async def test_generate_structured_records_error_snapshot(
    db_session, test_user, monkeypatch
):
    import app.services.llm_providers as providers_module

    monkeypatch.setattr(
        providers_module,
        "build_provider",
        lambda *args, **kwargs: FakeProvider(error=LLMServiceError("provider down")),
    )
    service = LLMService()

    with pytest.raises(LLMServiceError):
        await service.generate_structured(
            user_message="decide",
            user_id=test_user.id,
            db=db_session,
            snapshot_context={"phase": "thinking"},
        )
    await db_session.commit()

    rows = (await db_session.execute(select(LLMCallSnapshot))).scalars().all()
    assert rows, "error snapshots must be recorded"
    assert "provider down" in rows[0].error
    assert rows[0].response_text is None


@pytest.mark.asyncio
async def test_generate_response_records_snapshot(db_session, test_user, monkeypatch):
    service = LLMService()

    async def _fake_once(**kwargs):
        return "hello world"

    monkeypatch.setattr(service, "_generate_response_once", _fake_once)

    text = await service.generate_response(
        system_prompt="sys",
        user_message="question",
        user_id=test_user.id,
        db=db_session,
        snapshot_context={"phase": "compaction", "iteration": 2},
    )
    await db_session.commit()

    assert text == "hello world"
    rows = (await db_session.execute(select(LLMCallSnapshot))).scalars().all()
    assert len(rows) == 1
    assert rows[0].phase == "compaction"
    assert rows[0].response_text == "hello world"
    assert rows[0].request["system_prompt"] == "sys"
    assert rows[0].request["query"] == "question"


@pytest.mark.asyncio
async def test_snapshots_disabled_records_nothing(db_session, test_user, monkeypatch):
    monkeypatch.setattr(settings, "LLM_CALL_SNAPSHOT_ENABLED", False)
    service = LLMService()

    async def _fake_once(**kwargs):
        return "hello"

    monkeypatch.setattr(service, "_generate_response_once", _fake_once)
    await service.generate_response(
        user_message="q", user_id=test_user.id, db=db_session
    )
    await db_session.commit()

    rows = (await db_session.execute(select(LLMCallSnapshot))).scalars().all()
    assert rows == []


def test_clip_snapshot_truncates(monkeypatch):
    monkeypatch.setattr(settings, "LLM_CALL_SNAPSHOT_MAX_CHARS", 100)
    clipped = LLMService._clip_snapshot("x" * 500)
    assert len(clipped) <= 120
    assert clipped.endswith("[truncated]")
    assert LLMService._clip_snapshot(None) is None


class TestSnapshotEndpoints:
    def _seed(self, db_session, user_id, job_id=None, phase="thinking"):
        snap = LLMCallSnapshot(
            user_id=user_id,
            job_id=job_id,
            iteration=1,
            phase=phase,
            provider="ollama",
            model="llama3.2:3b",
            task_type="chat",
            request={"messages": [{"role": "user", "content": "hi"}]},
            response_text="hello",
        )
        db_session.add(snap)
        return snap

    def test_list_and_detail_owner(self, client, db_session, test_user, auth_headers):
        job_id = uuid4()
        snap = self._seed(db_session, test_user.id, job_id=job_id)
        asyncio.get_event_loop().run_until_complete(db_session.commit())

        listed = client.get(
            f"/api/v1/llm-snapshots/?job_id={job_id}", headers=auth_headers
        )
        assert listed.status_code == 200
        items = listed.json()
        assert len(items) == 1
        assert items[0]["phase"] == "thinking"
        assert "request" not in items[0]  # summaries exclude payloads

        detail = client.get(f"/api/v1/llm-snapshots/{snap.id}", headers=auth_headers)
        assert detail.status_code == 200
        body = detail.json()
        assert body["response_text"] == "hello"
        assert body["request"]["messages"][0]["content"] == "hi"

    def test_other_users_snapshot_hidden(
        self, client, db_session, test_user, admin_user, auth_headers, admin_headers
    ):
        snap = self._seed(db_session, admin_user.id)
        asyncio.get_event_loop().run_until_complete(db_session.commit())

        # Non-owner, non-admin: hidden from both list and detail.
        assert (
            client.get(f"/api/v1/llm-snapshots/{snap.id}", headers=auth_headers)
        ).status_code == 404
        listed = client.get("/api/v1/llm-snapshots/", headers=auth_headers)
        assert listed.status_code == 200
        assert listed.json() == []

        # Admin sees it.
        assert (
            client.get(f"/api/v1/llm-snapshots/{snap.id}", headers=admin_headers)
        ).status_code == 200
