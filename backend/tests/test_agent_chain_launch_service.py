from uuid import uuid4

import pytest
from sqlalchemy import select

from app.api.endpoints import agent_jobs
from app.models.agent_job import AgentJob, AgentJobStatus
from app.schemas.agent_job import (
    AgentJobChainDefinitionCreate,
    AgentJobFromChainCreate,
    ChainStepConfig,
)
from app.services.agent_chain_definition_service import (
    agent_chain_definition_service,
)
from app.services.agent_chain_launch_service import agent_chain_launch_service


async def _create_chain(db_session, test_user):
    return await agent_chain_definition_service.create(
        request=AgentJobChainDefinitionCreate(
            name=f"launch_chain_{uuid4().hex[:8]}",
            display_name="Launch Chain",
            description="A two-step launch contract.",
            chain_steps=[
                ChainStepConfig(
                    step_name="Scout",
                    job_type="research",
                    goal_template="Investigate {topic}",
                    config={
                        "persist_artifacts": False,
                        "nested": {"from_step": True},
                    },
                    trigger_condition="on_complete",
                ),
                ChainStepConfig(
                    step_name="Validate",
                    job_type="analysis",
                    goal_template="Validate {topic}",
                    config={"nested": {"validation": True}},
                ),
            ],
            default_settings={
                "persist_artifacts": False,
                "max_iterations": 8,
                "max_tool_calls": 60,
                "max_llm_calls": 20,
                "max_runtime_minutes": 12,
                "inherit_results": True,
                "nested": {"from_default": True},
                "target_source_id": "default-source",
            },
        ),
        user_id=test_user.id,
        db=db_session,
    )


@pytest.mark.asyncio
async def test_chain_launch_builds_root_and_recursive_continuation(
    db_session,
    test_user,
):
    chain = await _create_chain(db_session, test_user)
    request = AgentJobFromChainCreate(
        chain_definition_id=chain.id,
        name_prefix="Compiler R&D",
        variables={"topic": "vectorization"},
        config_overrides={
            "persist_artifacts": True,
            "max_iterations": 4,
            "enable_memory": False,
            "nested": {"from_override": True},
        },
        start_immediately=False,
    )

    job = await agent_chain_launch_service.launch(
        request=request,
        user_id=test_user.id,
        db=db_session,
    )

    assert job.name == "Compiler R&D: Scout"
    assert job.goal == "Investigate vectorization"
    assert job.job_type == "research"
    assert job.status == AgentJobStatus.PENDING.value
    assert job.max_iterations == 4
    assert job.max_tool_calls == 60
    assert job.enable_memory is False
    assert job.config["persist_artifacts"] is True
    assert job.config["source_id"] == "default-source"
    assert job.config["nested"] == {
        "from_default": True,
        "from_step": True,
        "from_override": True,
    }

    chain_config = job.chain_config
    assert chain_config["chain_definition_id"] == str(chain.id)
    assert chain_config["current_step_index"] == 0
    assert chain_config["total_steps"] == 2
    child = chain_config["child_jobs"][0]
    assert child["name"] == "Compiler R&D: Validate"
    assert child["goal"] == "Validate vectorization"
    assert child["max_iterations"] == 4
    assert child["config"]["persist_artifacts"] is True
    assert child["config"]["nested"] == {
        "from_default": True,
        "validation": True,
        "from_override": True,
    }
    assert child["chain_config"]["current_step_index"] == 1


@pytest.mark.asyncio
async def test_create_job_from_chain_returns_job_and_queues_when_requested(
    db_session,
    test_user,
    monkeypatch,
):
    chain = await _create_chain(db_session, test_user)
    dispatched = []
    monkeypatch.setattr(
        agent_jobs.execute_agent_job_task,
        "delay",
        lambda *args: dispatched.append(args),
    )

    response = await agent_jobs.create_job_from_chain(
        AgentJobFromChainCreate(
            chain_definition_id=chain.id,
            name_prefix="Queued Chain",
            variables={"topic": "branch prediction"},
            start_immediately=True,
        ),
        db=db_session,
        current_user=test_user,
    )

    assert response.name == "Queued Chain: Scout"
    assert response.goal == "Investigate branch prediction"
    assert dispatched == [(str(response.id), str(test_user.id))]
    persisted = (
        await db_session.execute(select(AgentJob).where(AgentJob.id == response.id))
    ).scalar_one()
    assert persisted.chain_config["chain_definition_id"] == str(chain.id)
