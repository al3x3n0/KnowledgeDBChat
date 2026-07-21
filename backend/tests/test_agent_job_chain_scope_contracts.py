from uuid import uuid4

import pytest
from sqlalchemy import select

from app.api.endpoints.agent_jobs import (
    create_chain_definition,
    get_chain_definition,
    list_chain_definitions,
    save_job_as_chain_definition,
    update_chain_definition,
)
from app.models.agent_job import AgentJob, AgentJobChainDefinition, AgentJobStatus
from app.schemas.agent_job import (
    AgentJobChainDefinitionCreate,
    AgentJobChainDefinitionUpdate,
    AgentJobSaveAsChainRequest,
    ChainStepConfig,
)


def _assert_no_target_source_id(value):
    if isinstance(value, dict):
        assert "target_source_id" not in value
        for v in value.values():
            _assert_no_target_source_id(v)
    elif isinstance(value, list):
        for item in value:
            _assert_no_target_source_id(item)


@pytest.mark.asyncio
async def test_chain_definition_create_get_list_normalize_scope_keys(db_session, test_user):
    chain_name = f"chain_scope_{uuid4().hex[:8]}"
    payload = AgentJobChainDefinitionCreate(
        name=chain_name,
        display_name="Chain Scope Create",
        chain_steps=[
            ChainStepConfig(
                step_name="Step 1",
                goal_template="Do thing",
                config={
                    "target_source_id": "step-src",
                    "nested": {"target_source_id": "step-nested"},
                },
            )
        ],
        default_settings={
            "target_source_id": "default-src",
            "child": {"target_source_id": "default-nested"},
        },
    )

    created = await create_chain_definition(payload, db=db_session, current_user=test_user)
    assert created.default_settings is not None
    assert created.default_settings["source_id"] == "default-src"
    assert created.chain_steps[0]["config"]["source_id"] == "step-src"
    _assert_no_target_source_id(created.default_settings)
    _assert_no_target_source_id(created.chain_steps)

    got = await get_chain_definition(created.id, db=db_session, current_user=test_user)
    assert got.id == created.id
    _assert_no_target_source_id(got.default_settings)
    _assert_no_target_source_id(got.chain_steps)

    listed = await list_chain_definitions(db=db_session, current_user=test_user)
    found = next((c for c in listed.chains if str(c.id) == str(created.id)), None)
    assert found is not None
    _assert_no_target_source_id(found.default_settings)
    _assert_no_target_source_id(found.chain_steps)

    row = (
        await db_session.execute(
            select(AgentJobChainDefinition).where(AgentJobChainDefinition.id == created.id)
        )
    ).scalar_one()
    _assert_no_target_source_id(row.default_settings)
    _assert_no_target_source_id(row.chain_steps)


@pytest.mark.asyncio
async def test_chain_definition_update_normalizes_scope_keys(db_session, test_user):
    chain_name = f"chain_scope_upd_{uuid4().hex[:8]}"
    base = AgentJobChainDefinitionCreate(
        name=chain_name,
        display_name="Chain Scope Update",
        chain_steps=[ChainStepConfig(step_name="Step 1", goal_template="Do thing")],
        default_settings={},
    )
    created = await create_chain_definition(base, db=db_session, current_user=test_user)

    update_payload = AgentJobChainDefinitionUpdate(
        chain_steps=[
            ChainStepConfig(
                step_name="Step 1",
                goal_template="Updated {x}",
                config={
                    "target_source_id": "updated-src",
                    "nested": {"target_source_id": "updated-nested"},
                },
            )
        ],
        default_settings={
            "target_source_id": "updated-default",
            "child": {"target_source_id": "updated-child"},
        },
    )
    updated = await update_chain_definition(
        created.id,
        update_payload,
        db=db_session,
        current_user=test_user,
    )

    assert updated.default_settings is not None
    assert updated.default_settings["source_id"] == "updated-default"
    assert updated.chain_steps[0]["config"]["source_id"] == "updated-src"
    _assert_no_target_source_id(updated.default_settings)
    _assert_no_target_source_id(updated.chain_steps)


@pytest.mark.asyncio
async def test_save_job_as_chain_definition_brands_failed_job_as_recovery_playbook(db_session, test_user):
    job = AgentJob(
        name="Failed Recovery Job",
        goal="Capture the failed run as a reusable recovery playbook",
        job_type="analysis",
        user_id=test_user.id,
        status=AgentJobStatus.FAILED.value,
        error="Verification failed after fallback run.",
        phase_details="Retry after fallback failure",
        max_iterations=7,
        max_tool_calls=25,
        max_llm_calls=9,
        max_runtime_minutes=18,
        results={
            "execution_strategy": {
                "scheduler_state": {
                    "queue_reason": "execution_failure",
                    "failure_streak": 2,
                }
            }
        },
    )

    db_session.add(job)
    await db_session.commit()
    await db_session.refresh(job)

    created = await save_job_as_chain_definition(
        job.id,
        AgentJobSaveAsChainRequest(),
        db=db_session,
        current_user=test_user,
    )

    assert created.name.startswith("playbook_recovery_")
    assert created.display_name == "Failed Recovery Job (Recovery Playbook)"
    assert created.description is not None
    assert "Saved as a recovery playbook." in created.description
    assert "Recovery reason: Execution failure." in created.description
    assert "Verification failed after fallback run." in created.description
    assert len(created.chain_steps) == 1
    assert created.default_settings is not None
    assert created.default_settings["max_iterations"] == 7
    assert created.default_settings["max_tool_calls"] == 25
