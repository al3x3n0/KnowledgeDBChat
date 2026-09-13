from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.endpoints import agent_jobs
from app.schemas.agent_job import AgentJobUpdate


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
async def test_update_job_rejects_terminal_status():
    job = SimpleNamespace(status="completed")

    with pytest.raises(HTTPException) as exc_info:
        await agent_jobs.update_agent_job(
            uuid4(),
            AgentJobUpdate(name="renamed"),
            db=_Db(job),
            current_user=SimpleNamespace(id=uuid4()),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Cannot update job in status: completed"


@pytest.mark.asyncio
async def test_delete_job_rejects_running_status():
    job = SimpleNamespace(status="running")

    with pytest.raises(HTTPException) as exc_info:
        await agent_jobs.delete_agent_job(
            uuid4(),
            db=_Db(job),
            current_user=SimpleNamespace(id=uuid4()),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Cannot delete running job. Cancel it first."
