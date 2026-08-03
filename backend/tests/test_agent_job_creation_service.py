from datetime import datetime
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.api.endpoints import agent_jobs
from app.models.agent_job import AgentJob, AgentJobStatus
from app.schemas.agent_job import AgentJobCreate, AgentJobFromTemplate
from app.services.agent_job_creation_service import (
    AgentJobCreationError,
    agent_job_creation_service,
)
from app.services.agent_job_templates import (
    CLAUDE_CODE_BACKEND_TEMPLATE_ID,
)


@pytest.mark.asyncio
async def test_job_creation_service_applies_parent_scope_memory_and_launch_log(
    db_session,
    test_user,
):
    parent = AgentJob(
        name="Parent Research",
        job_type="research",
        goal="Collect baseline evidence",
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
        chain_depth=2,
    )
    db_session.add(parent)
    await db_session.commit()
    await db_session.refresh(parent)

    job = await agent_job_creation_service.create_from_request(
        request=AgentJobCreate(
            name="Child Validation",
            job_type="analysis",
            goal="Validate the baseline",
            config={
                "target_source_id": "source-1",
                "enable_memory": False,
                "launch_mode": "manual_validation",
            },
            chain_config={
                "target_source_id": "chain-source",
                "child_jobs": [{"config": {"target_source_id": "nested-source"}}],
            },
            parent_job_id=parent.id,
            start_immediately=False,
        ),
        user_id=test_user.id,
        db=db_session,
    )

    assert job.parent_job_id == parent.id
    assert job.root_job_id == parent.id
    assert job.chain_depth == 3
    assert job.config["source_id"] == "source-1"
    assert "target_source_id" not in job.config
    assert job.enable_memory is False
    assert job.chain_config["source_id"] == "chain-source"
    assert job.chain_config["child_jobs"][0]["config"]["source_id"] == "nested-source"
    launch = job.execution_log[0]
    assert launch["phase"] == "launch"
    assert launch["iteration"] == 0
    assert launch["result"]["launch_mode"] == "manual_validation"


@pytest.mark.asyncio
async def test_job_creation_service_rejects_missing_parent_and_template(
    db_session,
    test_user,
):
    with pytest.raises(AgentJobCreationError) as parent_error:
        await agent_job_creation_service.create_from_request(
            request=AgentJobCreate(
                name="Orphan",
                goal="Should not be created",
                parent_job_id=uuid4(),
                start_immediately=False,
            ),
            user_id=test_user.id,
            db=db_session,
        )
    assert parent_error.value.detail == "Parent job not found"
    assert parent_error.value.status_code == 404

    with pytest.raises(AgentJobCreationError) as template_error:
        await agent_job_creation_service.create_from_template(
            request=AgentJobFromTemplate(
                template_id=uuid4(),
                name="Missing Template",
                start_immediately=False,
            ),
            user_id=test_user.id,
            db=db_session,
        )
    assert template_error.value.detail == "Job template not found or not active"
    assert template_error.value.status_code == 404


@pytest.mark.asyncio
async def test_template_creation_preserves_zero_resource_limits(
    db_session,
    test_user,
):
    job = await agent_job_creation_service.create_from_template(
        request=AgentJobFromTemplate(
            template_id=CLAUDE_CODE_BACKEND_TEMPLATE_ID,
            name="Template Resource Contract",
            goal="Exercise the built-in template",
            start_immediately=False,
        ),
        user_id=test_user.id,
        db=db_session,
    )

    assert job.max_tool_calls == 0
    assert job.max_llm_calls == 2
    assert job.status == AgentJobStatus.PENDING.value


@pytest.mark.asyncio
async def test_create_agent_job_dispatches_and_advances_continuous_schedule(
    db_session,
    test_user,
    monkeypatch,
):
    dispatched = []
    monkeypatch.setattr(
        agent_jobs.execute_agent_job_task,
        "delay",
        lambda *args: dispatched.append(args),
    )
    before = datetime.utcnow()

    response = await agent_jobs.create_agent_job(
        AgentJobCreate(
            name="Continuous Monitor",
            job_type="monitor",
            goal="Watch compiler regressions",
            config={"interval_minutes": 7},
            schedule_type="continuous",
            start_immediately=True,
        ),
        db=db_session,
        current_user=test_user,
    )

    assert dispatched == [(str(response.id), str(test_user.id))]
    persisted = (
        await db_session.execute(select(AgentJob).where(AgentJob.id == response.id))
    ).scalar_one()
    assert persisted.next_run_at is not None
    assert persisted.next_run_at.replace(tzinfo=None) > before
