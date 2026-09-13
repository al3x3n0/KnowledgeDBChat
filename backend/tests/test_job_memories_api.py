from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.modules.autonomy.api import job_memories


class _Result:
    def __init__(self, value):
        self.value = value

    def scalar_one_or_none(self):
        return self.value


class _Db:
    def __init__(self, value):
        self.value = value

    async def execute(self, _query):
        return _Result(self.value)


@pytest.mark.asyncio
async def test_get_job_memories_hides_missing_or_foreign_job():
    user = SimpleNamespace(id=uuid4())

    with pytest.raises(HTTPException) as exc_info:
        await job_memories.get_job_memories(
            uuid4(),
            db=_Db(None),
            current_user=user,
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Agent job not found"


@pytest.mark.asyncio
async def test_create_job_memory_rejects_invalid_type_before_querying():
    user = SimpleNamespace(id=uuid4())

    with pytest.raises(HTTPException) as exc_info:
        await job_memories.create_job_memory(
            uuid4(),
            memory_type="transient",
            content="discard me",
            importance=0.5,
            tags=None,
            db=_Db(None),
            current_user=user,
        )

    assert exc_info.value.status_code == 400
    assert "Invalid memory type" in exc_info.value.detail
