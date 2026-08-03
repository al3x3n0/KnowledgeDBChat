from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from app.api.endpoints import agent_jobs


@pytest.mark.asyncio
async def test_list_jobs_rejects_invalid_relaunch_parent_before_query():
    with pytest.raises(HTTPException) as exc_info:
        await agent_jobs.list_agent_jobs(
            status=None,
            job_type=None,
            launch_mode=None,
            relaunch_from_job_id="not-a-uuid",
            has_relaunch_children=None,
            swarm_only=False,
            swarm_min_consensus=0,
            visibility_scope="mine",
            sort_by="created_desc",
            page=1,
            page_size=20,
            db=SimpleNamespace(),
            current_user=SimpleNamespace(id=uuid4()),
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "Invalid relaunch_from_job_id"
