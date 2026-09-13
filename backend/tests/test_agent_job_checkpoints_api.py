"""Focused tests for the modular autonomous-job checkpoint API."""

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.modules.autonomy.api.job_checkpoints import build_job_checkpoint_api


class _ScalarResult:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _Scalars:
    def __init__(self, values):
        self.values = values

    def all(self):
        return self.values


class _ListResult:
    def __init__(self, values):
        self.values = values

    def scalars(self):
        return _Scalars(self.values)


class _Db:
    def __init__(self, job, checkpoints=()):
        self.results = [_ScalarResult(job), _ListResult(checkpoints)]

    async def execute(self, _statement):
        return self.results.pop(0)


@pytest.mark.asyncio
async def test_job_checkpoint_route_rejects_unknown_or_unowned_job():
    api = build_job_checkpoint_api()

    with pytest.raises(HTTPException) as exc_info:
        await api.get_job_checkpoints(
            job_id=uuid4(),
            db=_Db(None),
            current_user=SimpleNamespace(id=uuid4()),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Agent job not found"


@pytest.mark.asyncio
async def test_job_checkpoint_route_returns_projected_checkpoint_history():
    job_id = uuid4()
    checkpoint_id = uuid4()
    checkpoint = SimpleNamespace(
        id=checkpoint_id,
        job_id=job_id,
        iteration=7,
        phase="validation",
        created_at=datetime(2026, 8, 2, tzinfo=timezone.utc),
    )
    api = build_job_checkpoint_api()

    response = await api.get_job_checkpoints(
        job_id=job_id,
        db=_Db(SimpleNamespace(id=job_id), [checkpoint]),
        current_user=SimpleNamespace(id=uuid4()),
    )

    assert len(response) == 1
    assert response[0].id == checkpoint_id
    assert response[0].iteration == 7
    assert response[0].phase == "validation"


@pytest.mark.asyncio
async def test_job_checkpoint_route_returns_empty_owned_history():
    job_id = uuid4()
    api = build_job_checkpoint_api()

    response = await api.get_job_checkpoints(
        job_id=job_id,
        db=_Db(SimpleNamespace(id=job_id)),
        current_user=SimpleNamespace(id=uuid4()),
    )

    assert response == []
