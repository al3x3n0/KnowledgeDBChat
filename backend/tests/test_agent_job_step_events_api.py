"""Focused tests for the modular autonomous-job step-event API."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.modules.autonomy.api.job_step_events import (
    build_job_step_event_api,
    build_step_event_page,
)


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


def test_step_event_page_prefers_at_least_as_rich_checkpoint_stream():
    page = build_step_event_page(
        job_results={
            "execution_strategy": {
                "step_events": [{"type": "started"}, {"type": "completed"}]
            }
        },
        checkpoint_state={
            "step_events": [
                {"type": "started"},
                "invalid",
                {"type": "checkpoint_waiting"},
            ]
        },
        offset=1,
        limit=1,
    )

    assert page == {
        "items": [{"type": "checkpoint_waiting"}],
        "total": 2,
        "offset": 1,
        "limit": 1,
        "has_more": False,
        "source": "checkpoint_state",
    }


def test_step_event_page_uses_completed_results_when_richer():
    page = build_step_event_page(
        job_results={
            "execution_strategy": {
                "step_events": [{"type": "started"}, {"type": "completed"}]
            }
        },
        checkpoint_state={"step_events": [{"type": "started"}]},
        offset=0,
        limit=1,
    )

    assert page["items"] == [{"type": "started"}]
    assert page["total"] == 2
    assert page["has_more"] is True
    assert page["source"] == "results_execution_strategy"


@pytest.mark.asyncio
async def test_step_event_route_rejects_unknown_or_unowned_job():
    checkpoint_loader = AsyncMock()
    api = build_job_step_event_api(load_latest_checkpoint=checkpoint_loader)

    with pytest.raises(HTTPException) as exc_info:
        await api.get_job_step_events(
            job_id=uuid4(),
            limit=100,
            offset=0,
            db=_Db(None),
            current_user=SimpleNamespace(id=uuid4()),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Agent job not found"
    checkpoint_loader.assert_not_awaited()


@pytest.mark.asyncio
async def test_step_event_route_loads_checkpoint_for_owned_job():
    job = SimpleNamespace(
        id=uuid4(),
        results={"execution_strategy": {"step_events": []}},
    )
    checkpoint_loader = AsyncMock(
        return_value=SimpleNamespace(
            state={"step_events": [{"type": "checkpoint_waiting"}]}
        )
    )
    db = _Db(job)
    api = build_job_step_event_api(load_latest_checkpoint=checkpoint_loader)

    response = await api.get_job_step_events(
        job_id=job.id,
        limit=25,
        offset=0,
        db=db,
        current_user=SimpleNamespace(id=uuid4()),
    )

    checkpoint_loader.assert_awaited_once_with(job.id, db)
    assert response["items"] == [{"type": "checkpoint_waiting"}]
    assert response["source"] == "checkpoint_state"
