from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.endpoints import agent_jobs


class _Scalars:
    def all(self):
        return []


class _Result:
    def scalars(self):
        return _Scalars()


class _EmptyDb:
    async def execute(self, _query):
        return _Result()


@pytest.mark.asyncio
async def test_swarm_analytics_rejects_unknown_preset_before_query():
    with pytest.raises(HTTPException) as exc_info:
        await agent_jobs.get_swarm_analytics(
            source_id=None,
            preset_key="unknown-swarm",
            visibility_scope="mine",
            date_from=None,
            date_to=None,
            db=SimpleNamespace(),
            current_user=SimpleNamespace(id=uuid4()),
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "Unknown coding swarm preset"


@pytest.mark.asyncio
async def test_swarm_analytics_returns_zeroed_rows_for_empty_dataset():
    response = await agent_jobs.get_swarm_analytics(
        source_id=None,
        preset_key=None,
        visibility_scope="mine",
        date_from=None,
        date_to=None,
        db=_EmptyDb(),
        current_user=SimpleNamespace(id=uuid4()),
    )

    assert response.preset_rows
    assert response.totals["total_runs"] == 0
    assert response.totals["avg_confidence"] is None
    assert response.filters["visibility_scope"] == "mine"
