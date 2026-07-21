from uuid import uuid4

import pytest

from app.api.endpoints.agent_jobs import create_job_from_template
from app.models.agent_job import AgentJobTemplate
from app.schemas.agent_job import AgentJobFromTemplate
from app.services.agent_job_templates import CLAUDE_CODE_BACKEND_TEMPLATE_ID


def _assert_no_target_source_id(value):
    if isinstance(value, dict):
        assert "target_source_id" not in value
        for v in value.values():
            _assert_no_target_source_id(v)
    elif isinstance(value, list):
        for item in value:
            _assert_no_target_source_id(item)


@pytest.mark.asyncio
async def test_create_job_from_template_builtin_path_does_not_require_db_template_fields(db_session, test_user):
    payload = AgentJobFromTemplate(
        template_id=CLAUDE_CODE_BACKEND_TEMPLATE_ID,
        name="Builtin Template Job",
        goal="Fix backend contract tests",
        config={"target_source_id": "cfg-src"},
        chain_config={
            "target_source_id": "chain-src",
            "child_jobs": [{"config": {"target_source_id": "child-src"}}],
        },
        start_immediately=False,
    )

    job = await create_job_from_template(payload, db=db_session, current_user=test_user)

    assert job.config is not None
    assert job.config["source_id"] == "cfg-src"
    _assert_no_target_source_id(job.config)
    assert isinstance(job.chain_config, dict)
    assert job.chain_config["source_id"] == "chain-src"
    assert job.chain_config["child_jobs"][0]["config"]["source_id"] == "child-src"
    _assert_no_target_source_id(job.chain_config)


@pytest.mark.asyncio
async def test_create_job_from_template_db_template_missing_default_chain_config_is_safe(db_session, test_user):
    tpl = AgentJobTemplate(
        name=f"db_tpl_{uuid4().hex[:8]}",
        display_name="DB Template",
        description="",
        category="code",
        job_type="analysis",
        default_goal="DB template goal",
        default_config={"target_source_id": "tpl-src"},
        owner_user_id=test_user.id,
        is_system=False,
        is_active=True,
    )
    db_session.add(tpl)
    await db_session.commit()
    await db_session.refresh(tpl)

    payload = AgentJobFromTemplate(
        template_id=tpl.id,
        name="DB Template Job",
        goal="Use DB template",
        config={"target_source_id": "override-src"},
        start_immediately=False,
    )

    job = await create_job_from_template(payload, db=db_session, current_user=test_user)

    assert job.config is not None
    assert job.config["source_id"] == "override-src"
    _assert_no_target_source_id(job.config)
    assert job.chain_config is None
