"""Focused tests for the modular autonomous-job execution-log API."""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.modules.autonomy.api.job_logs import build_job_log_api, build_job_log_page


class _Result:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _Db:
    def __init__(self, job):
        self.job = job

    async def execute(self, _statement):
        return _Result(self.job)


def test_job_log_page_paginates_ordered_entries():
    rows = [
        {"phase": "started"},
        {"phase": "research"},
        {"phase": "completed"},
    ]

    assert build_job_log_page(rows, offset=1, limit=1) == {
        "entries": [{"phase": "research"}],
        "total": 3,
        "offset": 1,
        "limit": 1,
        "has_more": True,
    }


def test_job_log_page_normalizes_missing_log_to_empty_page():
    assert build_job_log_page(None, offset=0, limit=50) == {
        "entries": [],
        "total": 0,
        "offset": 0,
        "limit": 50,
        "has_more": False,
    }


@pytest.mark.asyncio
async def test_job_log_route_rejects_unknown_or_unowned_job():
    api = build_job_log_api()

    with pytest.raises(HTTPException) as exc_info:
        await api.get_job_log(
            job_id=uuid4(),
            limit=50,
            offset=0,
            db=_Db(None),
            current_user=SimpleNamespace(id=uuid4()),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Agent job not found"


@pytest.mark.asyncio
async def test_job_log_route_returns_owned_job_page():
    job = SimpleNamespace(execution_log=[{"phase": "started"}, {"phase": "completed"}])
    api = build_job_log_api()

    response = await api.get_job_log(
        job_id=uuid4(),
        limit=1,
        offset=1,
        db=_Db(job),
        current_user=SimpleNamespace(id=uuid4()),
    )

    assert response["entries"] == [{"phase": "completed"}]
    assert response["total"] == 2
    assert response["has_more"] is False
