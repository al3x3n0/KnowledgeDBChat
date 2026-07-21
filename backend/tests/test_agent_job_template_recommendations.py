from datetime import datetime
from uuid import uuid4

import pytest

from app.api.endpoints.agent_jobs import _score_template_recommendation
from app.api.endpoints.agent_jobs import list_job_templates
from app.models.agent_job import AgentJobTemplate
from app.schemas.agent_job import AgentJobTemplateResponse


def _make_template(*, name: str, category: str, runner: str, display_name: str = "", default_goal: str = "") -> AgentJobTemplateResponse:
    now = datetime.utcnow()
    return AgentJobTemplateResponse(
        id=uuid4(),
        name=name,
        display_name=display_name or name,
        description="",
        category=category,
        job_type="analysis",
        default_goal=default_goal,
        default_config={"deterministic_runner": runner},
        default_chain_config=None,
        agent_definition_id=None,
        default_max_iterations=1,
        default_max_tool_calls=0,
        default_max_llm_calls=0,
        default_max_runtime_minutes=5,
        is_system=True,
        is_active=True,
        owner_user_id=None,
        created_at=now,
        updated_at=now,
    )


def test_template_recommendation_prefers_claude_backend_for_backend_goal():
    tpl = _make_template(
        name="claude_code_backend",
        category="code",
        runner="code_patch_proposer",
        display_name="Code Agent: Claude-Code Backend Loop",
    )
    score, reasons = _score_template_recommendation(
        tpl,
        category="code",
        recommend_goal="Fix backend API test failures",
        recommend_scope="backend",
    )

    assert score >= 100
    assert "backend_loop_specialized" in reasons


def test_template_recommendation_prefers_latex_for_latex_context():
    tpl = _make_template(
        name="latex_reviewer_critic",
        category="latex",
        runner="latex_reviewer_critic",
    )
    score, reasons = _score_template_recommendation(
        tpl,
        category=None,
        recommend_goal="Review LaTeX citations and bibliography",
        recommend_scope=None,
    )

    assert score >= 30
    assert "latex_category_fit" in reasons


def _assert_no_target_source_id(value):
    if isinstance(value, dict):
        assert "target_source_id" not in value
        for v in value.values():
            _assert_no_target_source_id(v)
    elif isinstance(value, list):
        for item in value:
            _assert_no_target_source_id(item)


@pytest.mark.asyncio
async def test_list_job_templates_normalizes_db_template_scope_keys(db_session, test_user):
    tpl = AgentJobTemplate(
        name=f"scope_norm_tpl_{uuid4().hex[:8]}",
        display_name="Scope Normalize Template",
        description="",
        category="code",
        job_type="analysis",
        default_goal="Test normalization",
        default_config={
            "target_source_id": "root-src",
            "nested": {"target_source_id": "nested-src"},
            "items": [{"target_source_id": "item-src"}],
        },
        owner_user_id=test_user.id,
        is_system=False,
        is_active=True,
    )
    db_session.add(tpl)
    await db_session.commit()
    await db_session.refresh(tpl)

    res = await list_job_templates(
        category="code",
        recommend_goal=None,
        recommend_scope=None,
        db=db_session,
        current_user=test_user,
    )

    found = next((t for t in res.templates if str(t.id) == str(tpl.id)), None)
    assert found is not None
    assert found.default_config is not None
    assert found.default_config["source_id"] == "root-src"
    assert found.default_config["nested"]["source_id"] == "nested-src"
    assert found.default_config["items"][0]["source_id"] == "item-src"
    _assert_no_target_source_id(found.default_config)
