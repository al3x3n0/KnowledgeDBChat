"""Tests for extracted checkpoint service."""

from uuid import uuid4

import pytest

from app.models.agent_job import AgentJob, AgentJobStatus
from app.services.agent_checkpoint_service import AgentCheckpointService


@pytest.mark.asyncio
async def test_checkpoint_service_round_trip(db_session):
    job = AgentJob(
        name="Checkpoint Test",
        goal="Resume work",
        job_type="research",
        user_id=uuid4(),
        status=AgentJobStatus.RUNNING.value,
        config={},
        max_iterations=20,
        max_tool_calls=20,
        max_llm_calls=20,
        max_runtime_minutes=30,
    )
    job.current_phase = "thinking"
    job.iteration = 3
    job.progress = 45
    db_session.add(job)
    await db_session.commit()

    service = AgentCheckpointService()
    state = {"goal_progress": 45, "findings": [{"id": "f1"}]}

    await service.save_checkpoint(
        job=job, state=state, db=db_session, reason="test_checkpoint"
    )
    checkpoint = await service.load_latest_checkpoint(job_id=job.id, db=db_session)

    assert checkpoint is not None
    assert checkpoint.job_id == job.id
    assert checkpoint.iteration == 3
    assert checkpoint.phase == "thinking"
    assert checkpoint.state["goal_progress"] == 45
    assert checkpoint.context["reason"] == "test_checkpoint"
