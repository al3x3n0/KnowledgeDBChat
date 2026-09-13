from uuid import uuid4

import pytest

from app.api.endpoints.agent_jobs import get_chain_status
from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.agent_chain_status_service import (
    AgentChainStatusError,
    agent_chain_status_service,
)


async def _create_status_chain(db_session, test_user):
    chain_definition_id = uuid4()
    root = AgentJob(
        name="Root Research",
        job_type="research",
        goal="Collect compiler evidence",
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
        progress=100,
        chain_depth=0,
        chain_config={"chain_definition_id": str(chain_definition_id)},
    )
    db_session.add(root)
    await db_session.commit()
    await db_session.refresh(root)

    child = AgentJob(
        name="Validate Evidence",
        job_type="analysis",
        goal="Validate compiler evidence",
        user_id=test_user.id,
        status=AgentJobStatus.PENDING.value,
        progress=20,
        parent_job_id=root.id,
        root_job_id=root.id,
        chain_depth=1,
    )
    db_session.add(child)
    await db_session.commit()
    await db_session.refresh(child)
    return root, child, chain_definition_id


@pytest.mark.asyncio
async def test_chain_status_snapshot_works_from_child_job(
    db_session,
    test_user,
):
    root, child, chain_definition_id = await _create_status_chain(db_session, test_user)

    snapshot = await agent_chain_status_service.get_snapshot(
        job_id=child.id,
        user_id=test_user.id,
        db=db_session,
    )

    assert snapshot.root_job_id == root.id
    assert snapshot.chain_definition_id == chain_definition_id
    assert snapshot.total_steps == 2
    assert snapshot.completed_steps == 1
    assert snapshot.current_step == 1
    assert snapshot.overall_progress == 60
    assert snapshot.status == "partially_completed"
    assert [job.id for job in snapshot.jobs] == [root.id, child.id]


@pytest.mark.asyncio
async def test_chain_status_endpoint_serializes_service_snapshot(
    db_session,
    test_user,
):
    root, child, _ = await _create_status_chain(db_session, test_user)

    response = await get_chain_status(
        child.id,
        db=db_session,
        current_user=test_user,
    )

    assert response.root_job_id == root.id
    assert response.total_steps == 2
    assert response.status == "partially_completed"
    assert [job.id for job in response.jobs] == [root.id, child.id]


@pytest.mark.asyncio
async def test_chain_status_rejects_unknown_or_unowned_job(
    db_session,
    test_user,
):
    with pytest.raises(AgentChainStatusError) as exc_info:
        await agent_chain_status_service.get_snapshot(
            job_id=uuid4(),
            user_id=test_user.id,
            db=db_session,
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Agent job not found"
