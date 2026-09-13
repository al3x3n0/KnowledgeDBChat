from datetime import datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.modules.autonomy.api.relaunch_lineage import get_agent_job_relaunch_lineage


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


class _CollectionResult:
    def __init__(self, values):
        self.values = values

    def scalars(self):
        return _Scalars(self.values)


class _Db:
    def __init__(self, *results):
        self.results = list(results)

    async def execute(self, _query):
        return self.results.pop(0)


def _job(*, job_id, name, created_at, parent_id=None):
    config = {"launch_mode": "quick_start_claude_backend"}
    if parent_id is not None:
        config["relaunch_from_job_id"] = str(parent_id)
    return SimpleNamespace(
        id=job_id,
        name=name,
        status="completed",
        created_at=created_at,
        config=config,
    )


@pytest.mark.asyncio
async def test_relaunch_lineage_endpoint_returns_user_scoped_graph():
    user = SimpleNamespace(id=uuid4())
    root_id = uuid4()
    child_id = uuid4()
    created_at = datetime.utcnow()
    root = _job(
        job_id=root_id,
        name="root",
        created_at=created_at,
    )
    child = _job(
        job_id=child_id,
        name="child",
        created_at=created_at + timedelta(minutes=1),
        parent_id=root_id,
    )
    db = _Db(_ScalarResult(child), _CollectionResult([root, child]))

    response = await get_agent_job_relaunch_lineage(
        child_id,
        ancestor_limit=100,
        descendant_limit=500,
        db=db,
        current_user=user,
    )

    assert response.job_id == child_id
    assert response.root_job_id == root_id
    assert response.parent_job_id == root_id
    assert [node.id for node in response.ancestors] == [root_id]


@pytest.mark.asyncio
async def test_relaunch_lineage_endpoint_hides_missing_or_foreign_job():
    user = SimpleNamespace(id=uuid4())
    db = _Db(_ScalarResult(None))

    with pytest.raises(HTTPException) as exc_info:
        await get_agent_job_relaunch_lineage(
            uuid4(),
            ancestor_limit=100,
            descendant_limit=500,
            db=db,
            current_user=user,
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Agent job not found"
