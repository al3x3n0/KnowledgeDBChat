from uuid import uuid4

import pytest
from sqlalchemy import select

from app.models.agent_job import AgentJobChainDefinition
from app.schemas.agent_job import (
    AgentJobChainDefinitionCreate,
    AgentJobChainDefinitionUpdate,
    ChainStepConfig,
)
from app.services.agent_chain_definition_service import (
    AgentChainDefinitionError,
    agent_chain_definition_service,
)


def _chain_create(name: str) -> AgentJobChainDefinitionCreate:
    return AgentJobChainDefinitionCreate(
        name=name,
        display_name="Compiler Research Pipeline",
        chain_steps=[
            ChainStepConfig(
                step_name="Collect evidence",
                goal_template="Investigate {topic}",
                config={
                    "target_source_id": "legacy-source",
                    "nested": {"target_source_id": "legacy-nested"},
                },
            )
        ],
        default_settings={"target_source_id": "legacy-default"},
    )


@pytest.mark.asyncio
async def test_chain_definition_service_lifecycle_normalizes_scope(
    db_session,
    test_user,
):
    name = f"service_chain_{uuid4().hex[:8]}"
    created = await agent_chain_definition_service.create(
        request=_chain_create(name),
        user_id=test_user.id,
        db=db_session,
    )

    assert created.owner_user_id == test_user.id
    assert created.default_settings == {"source_id": "legacy-default"}
    assert created.chain_steps[0]["config"]["source_id"] == "legacy-source"
    assert created.chain_steps[0]["config"]["nested"]["source_id"] == "legacy-nested"

    visible = await agent_chain_definition_service.get_visible(
        chain_id=created.id,
        user_id=test_user.id,
        db=db_session,
    )
    assert visible.id == created.id

    available = await agent_chain_definition_service.list_for_user(
        user_id=test_user.id,
        db=db_session,
    )
    assert created.id in {chain.id for chain in available}

    updated = await agent_chain_definition_service.update_owned(
        chain_id=created.id,
        request=AgentJobChainDefinitionUpdate(
            display_name="Updated Pipeline",
            default_settings={"target_source_id": "updated-source"},
        ),
        user_id=test_user.id,
        db=db_session,
    )
    assert updated.display_name == "Updated Pipeline"
    assert updated.default_settings == {"source_id": "updated-source"}

    await agent_chain_definition_service.delete_owned(
        chain_id=created.id,
        user_id=test_user.id,
        db=db_session,
    )
    persisted = (
        await db_session.execute(
            select(AgentJobChainDefinition).where(
                AgentJobChainDefinition.id == created.id
            )
        )
    ).scalar_one_or_none()
    assert persisted is None


@pytest.mark.asyncio
async def test_chain_definition_service_rejects_duplicate_name(
    db_session,
    test_user,
):
    name = f"duplicate_chain_{uuid4().hex[:8]}"
    request = _chain_create(name)
    await agent_chain_definition_service.create(
        request=request,
        user_id=test_user.id,
        db=db_session,
    )

    with pytest.raises(AgentChainDefinitionError) as exc_info:
        await agent_chain_definition_service.create(
            request=request,
            user_id=test_user.id,
            db=db_session,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == ("Chain definition with this name already exists")


@pytest.mark.asyncio
async def test_chain_definition_service_rejects_non_owned_mutation(
    db_session,
    test_user,
):
    created = await agent_chain_definition_service.create(
        request=_chain_create(f"owned_chain_{uuid4().hex[:8]}"),
        user_id=test_user.id,
        db=db_session,
    )

    with pytest.raises(AgentChainDefinitionError) as exc_info:
        await agent_chain_definition_service.update_owned(
            chain_id=created.id,
            request=AgentJobChainDefinitionUpdate(display_name="Not Allowed"),
            user_id=uuid4(),
            db=db_session,
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == ("Chain definition not found or not editable")
