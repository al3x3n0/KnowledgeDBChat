from uuid import uuid4

import pytest

from app.models.agent_job import AgentJob, AgentJobStatus
from app.schemas.agent_job import AgentJobSaveAsChainRequest
from app.services.agent_playbook_service import (
    AgentPlaybookError,
    agent_playbook_service,
)


@pytest.mark.asyncio
async def test_playbook_service_reconstructs_nested_chain_payload(
    db_session,
    test_user,
):
    root = AgentJob(
        name="Compiler Discovery",
        job_type="research",
        goal="Find optimization opportunities",
        config={"target_source_id": "root-source"},
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
        max_iterations=7,
        max_tool_calls=25,
        max_llm_calls=9,
        max_runtime_minutes=18,
        chain_config={
            "trigger_condition": "on_progress",
            "progress_threshold": 80,
            "child_jobs": [
                {
                    "name": "Validate Candidate",
                    "job_type": "analysis",
                    "goal": "Benchmark the strongest candidate",
                    "config": {"target_source_id": "child-source"},
                }
            ],
        },
    )
    db_session.add(root)
    await db_session.commit()
    await db_session.refresh(root)

    chain = await agent_playbook_service.save(
        job_id=root.id,
        request=AgentJobSaveAsChainRequest(
            name=f"compiler_playbook_{uuid4().hex[:8]}",
            display_name="Compiler Optimization Playbook",
        ),
        user_id=test_user.id,
        db=db_session,
    )

    assert chain.display_name == "Compiler Optimization Playbook"
    assert len(chain.chain_steps) == 2
    first, second = chain.chain_steps
    assert first["step_name"] == "Compiler Discovery"
    assert first["trigger_condition"] == "on_progress"
    assert first["trigger_thresholds"] == {"progress_threshold": 80}
    assert first["config"]["source_id"] == "root-source"
    assert second["step_name"] == "Validate Candidate"
    assert second["config"]["source_id"] == "child-source"
    assert second["trigger_condition"] == "on_complete"
    assert second["trigger_thresholds"] is None
    assert chain.default_settings["max_iterations"] == 7
    assert chain.default_settings["max_tool_calls"] == 25


@pytest.mark.asyncio
async def test_playbook_service_falls_back_to_persisted_job_hierarchy(
    db_session,
    test_user,
):
    root = AgentJob(
        name="Root Experiment",
        job_type="analysis",
        goal="Run baseline",
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
        chain_depth=0,
    )
    db_session.add(root)
    await db_session.commit()
    await db_session.refresh(root)

    child = AgentJob(
        name="Follow-up Experiment",
        job_type="analysis",
        goal="Run optimized candidate",
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
        parent_job_id=root.id,
        root_job_id=root.id,
        chain_depth=1,
    )
    db_session.add(child)
    await db_session.commit()

    chain = await agent_playbook_service.save(
        job_id=root.id,
        request=AgentJobSaveAsChainRequest(
            name=f"persisted_playbook_{uuid4().hex[:8]}"
        ),
        user_id=test_user.id,
        db=db_session,
    )

    assert [step["step_name"] for step in chain.chain_steps] == [
        "Root Experiment",
        "Follow-up Experiment",
    ]


@pytest.mark.asyncio
async def test_playbook_service_preserves_duplicate_name_contract(
    db_session,
    test_user,
):
    root = AgentJob(
        name="Duplicate Playbook Source",
        job_type="analysis",
        goal="Exercise duplicate handling",
        user_id=test_user.id,
        status=AgentJobStatus.COMPLETED.value,
    )
    db_session.add(root)
    await db_session.commit()
    await db_session.refresh(root)
    name = f"duplicate_playbook_{uuid4().hex[:8]}"
    request = AgentJobSaveAsChainRequest(name=name)

    await agent_playbook_service.save(
        job_id=root.id,
        request=request,
        user_id=test_user.id,
        db=db_session,
    )

    with pytest.raises(AgentPlaybookError) as exc_info:
        await agent_playbook_service.save(
            job_id=root.id,
            request=request,
            user_id=test_user.id,
            db=db_session,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Chain definition name already exists"
