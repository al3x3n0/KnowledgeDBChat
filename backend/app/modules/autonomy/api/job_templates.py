"""Template catalog HTTP boundary for autonomous jobs."""

from dataclasses import dataclass
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.agent_job import AgentJobTemplate
from app.models.user import User
from app.modules.autonomy.application.template_recommendations import (
    score_template_recommendation,
)
from app.schemas.agent_job import AgentJobTemplateListResponse, AgentJobTemplateResponse
from app.services.agent_job_templates import list_builtin_agent_job_templates

ScopeNormalizer = Callable[[Any], Any]


@dataclass(frozen=True)
class JobTemplateApi:
    router: APIRouter
    list_job_templates: Callable[..., Any]


def build_job_template_api(
    *,
    router: APIRouter,
    normalize_scope_keys: ScopeNormalizer,
) -> JobTemplateApi:
    """Register the visible built-in and persisted template catalog."""

    @router.get("/templates", response_model=AgentJobTemplateListResponse)
    async def list_job_templates(
        category: Optional[str] = Query(
            None,
            description="Filter by category",
        ),
        recommend_goal: Optional[str] = Query(
            None,
            description="Optional goal text used for relevance ranking",
        ),
        recommend_scope: Optional[str] = Query(
            None,
            description="Optional scope hint (e.g. backend/frontend) for ranking",
        ),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        query = select(AgentJobTemplate).where(
            and_(
                AgentJobTemplate.is_active.is_(True),
                or_(
                    AgentJobTemplate.is_system.is_(True),
                    AgentJobTemplate.owner_user_id == current_user.id,
                ),
            )
        )
        if category:
            query = query.where(AgentJobTemplate.category == category)
        query = query.order_by(
            AgentJobTemplate.is_system.desc(),
            AgentJobTemplate.name,
        )
        persisted = list((await db.execute(query)).scalars().all())

        templates = []
        for row in persisted:
            model = AgentJobTemplateResponse.model_validate(row)
            templates.append(
                model.model_copy(
                    update={
                        "default_config": normalize_scope_keys(model.default_config),
                        "default_chain_config": normalize_scope_keys(
                            model.default_chain_config
                        ),
                    }
                )
            )
        templates.extend(
            AgentJobTemplateResponse(
                id=row.id,
                name=row.name,
                display_name=row.display_name,
                description=row.description,
                category=row.category,
                job_type=row.job_type,
                default_goal=row.default_goal,
                default_config=normalize_scope_keys(row.default_config),
                default_chain_config=normalize_scope_keys(row.default_chain_config),
                agent_definition_id=row.agent_definition_id,
                default_max_iterations=row.default_max_iterations,
                default_max_tool_calls=row.default_max_tool_calls,
                default_max_llm_calls=row.default_max_llm_calls,
                default_max_runtime_minutes=row.default_max_runtime_minutes,
                is_system=row.is_system,
                is_active=row.is_active,
                owner_user_id=row.owner_user_id,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            for row in list_builtin_agent_job_templates(category)
        )

        ranked = []
        for template in templates:
            score, reasons = score_template_recommendation(
                template,
                category=category,
                recommend_goal=recommend_goal,
                recommend_scope=recommend_scope,
            )
            ranked.append(
                (
                    score,
                    template.model_copy(
                        update={
                            "recommended": score > 0,
                            "recommendation_score": score,
                            "recommendation_reasons": reasons,
                        }
                    ),
                )
            )
        ranked.sort(
            key=lambda row: (
                -int(row[0]),
                0 if bool(row[1].is_system) else 1,
                str(row[1].name or "").lower(),
            )
        )
        ordered = [row[1] for row in ranked]
        return AgentJobTemplateListResponse(
            templates=ordered,
            total=len(ordered),
        )

    return JobTemplateApi(
        router=router,
        list_job_templates=list_job_templates,
    )
