"""Promotion boundary from exploratory jobs to durable research operations."""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.domain_research_profile import DomainResearchProfile
from app.models.research_portfolio import ResearchPortfolio
from app.models.user import User
from app.schemas.agent_job import (
    AgentJobPromoteDomainResearchRequest,
    AgentJobPromoteDomainResearchResponse,
    AgentJobQuickStartDomainResearchRequest,
)
from app.schemas.domain_research_profile import DomainResearchProfileCreate
from app.schemas.research_portfolio import ResearchPortfolioCreate
from app.services.autonomy_service import (
    build_domain_profile_compat_policy,
    resolve_domain_profile_automation_contract,
)
from app.services.scientific_validation_service import (
    normalize_portfolio_automation_profile,
    resolve_portfolio_automation_policy,
)

JobVisibility = Callable[[AgentJob, User], bool]
LaunchModeExtractor = Callable[[dict | None], str]
PromotionExtractor = Callable[[AgentJob], dict[str, Any]]
PromotionSeedBuilder = Callable[[AgentJob], dict[str, Any]]
SandboxValidator = Callable[..., Awaitable[None]]
DomainConfigBuilder = Callable[[AgentJobQuickStartDomainResearchRequest], dict]
DomainGoalBuilder = Callable[[AgentJobQuickStartDomainResearchRequest], str]
ProfilePresenter = Callable[..., Awaitable[Any]]
PortfolioPresenter = Callable[..., Awaitable[Any]]
JobPresenter = Callable[..., Any]


@dataclass(frozen=True)
class DomainResearchPromotionApi:
    router: APIRouter
    promote_domain_research_job: Callable[..., Any]


def _promotion_uuid(raw_value: Any, *, fallback: UUID | None) -> UUID | None:
    value = str(raw_value or "").strip()
    if re.fullmatch(r"[0-9a-fA-F-]{36}", value):
        return UUID(value)
    return fallback


def build_domain_research_promotion_api(
    *,
    router: APIRouter,
    is_job_visible: JobVisibility,
    extract_launch_mode: LaunchModeExtractor,
    extract_promotion: PromotionExtractor,
    build_promotion_seed: PromotionSeedBuilder,
    validate_sandbox_profile: SandboxValidator,
    build_domain_config: DomainConfigBuilder,
    build_domain_goal: DomainGoalBuilder,
    present_profile: ProfilePresenter,
    present_portfolio: PortfolioPresenter,
    present_job: JobPresenter,
    execute_job_task: Any,
) -> DomainResearchPromotionApi:
    """Register durable profile/fleet promotion for completed research jobs."""

    @router.post(
        "/{job_id}/promote-domain-research",
        response_model=AgentJobPromoteDomainResearchResponse,
    )
    async def promote_domain_research_job(
        job_id: UUID,
        payload: AgentJobPromoteDomainResearchRequest,
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ):
        job = (
            await db.execute(
                select(AgentJob)
                .options(selectinload(AgentJob.agent_definition))
                .where(AgentJob.id == job_id)
            )
        ).scalar_one_or_none()
        if not job or not is_job_visible(job, current_user):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent job not found",
            )

        config = job.config if isinstance(job.config, dict) else {}
        if extract_launch_mode(config) != "quick_start_domain_research":
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Only domain research quick-start jobs can be promoted",
            )
        if str(job.status or "").strip().lower() != AgentJobStatus.COMPLETED.value:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "Only completed domain research quick-start jobs can be promoted"
                ),
            )
        if str(config.get("profile_id") or "").strip():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Job is already linked to a saved domain research profile",
            )
        existing_promotion = extract_promotion(job)
        if str(
            existing_promotion.get("domain_research_profile_id")
            or existing_promotion.get("promoted_domain_research_profile_id")
            or ""
        ).strip():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Job has already been promoted",
            )

        promotion_seed = build_promotion_seed(job)
        profile_data = dict(promotion_seed["profile"])
        profile_data.update(payload.profile.model_dump(exclude_none=True))
        profile_data["start_immediately"] = False
        profile_request = DomainResearchProfileCreate.model_validate(profile_data)
        await validate_sandbox_profile(
            db,
            sandbox_profile_id=profile_request.sandbox_profile_id,
            track_type=profile_request.track_type,
        )
        (
            profile_automation_profile,
            profile_automation_policy,
        ) = resolve_domain_profile_automation_contract(
            automation_profile=profile_request.automation_profile,
            automation_policy=profile_request.automation_policy,
            explicit_updates=profile_request.model_dump(exclude_none=True),
        )

        if (
            payload.target_mode == "profile_with_portfolio"
            and not payload.portfolio_id
            and payload.portfolio is None
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "Provide an existing portfolio_id or portfolio payload when "
                    "attaching to a fleet"
                ),
            )

        profile = DomainResearchProfile(
            user_id=current_user.id,
            title=profile_request.title,
            domain=profile_request.domain,
            objective=profile_request.objective,
            customer_context=profile_request.customer_context,
            status="draft",
            source_scope=profile_request.source_scope,
            track_type=profile_request.track_type,
            research_mode=profile_request.research_mode,
            monitor_queries=profile_request.monitor_queries,
            repo_source_ids=[
                str(value) for value in (profile_request.repo_source_ids or [])
            ]
            or None,
            benchmark_queries=profile_request.benchmark_queries,
            report_format=profile_request.report_format,
            scoring_policy=profile_request.scoring_policy,
            selection_policy=profile_request.selection_policy,
            validation_policy=build_domain_profile_compat_policy(
                profile_automation_policy
            ),
            automation_profile=profile_automation_profile,
            automation_policy=profile_automation_policy,
            sandbox_profile_id=profile_request.sandbox_profile_id,
            interval_minutes=profile_request.interval_minutes,
            persist_artifacts=profile_request.persist_artifacts,
            auto_launch_follow_up=bool(
                profile_automation_policy.get(
                    "auto_launch_follow_up",
                    profile_request.auto_launch_follow_up,
                )
            ),
            auto_create_experiment_plans=bool(
                profile_automation_policy.get(
                    "auto_create_experiment_plans",
                    profile_request.auto_create_experiment_plans,
                )
            ),
            confidence_threshold=float(
                profile_automation_policy.get(
                    "confidence_threshold",
                    profile_request.confidence_threshold,
                )
            ),
            max_documents=profile_request.max_documents,
            max_papers=profile_request.max_papers,
        )
        db.add(profile)
        await db.flush()

        queued_job_ids: list[str] = []
        if payload.start_profile_now:
            quick_start_request = (
                AgentJobQuickStartDomainResearchRequest.model_validate(
                    {
                        **profile_data,
                        "start_immediately": False,
                        "profile_id": profile.id,
                    }
                )
            )
            profile_job_config = build_domain_config(quick_start_request)
            profile_job_config.update(
                {
                    "profile_id": str(profile.id),
                    "monitor_mode": "profile",
                    "interval_minutes": int(profile.interval_minutes or 1440),
                }
            )
            profile_job = AgentJob(
                user_id=current_user.id,
                name=f"Domain Monitor — {profile.title}",
                goal=build_domain_goal(quick_start_request),
                job_type="research",
                status=AgentJobStatus.PENDING.value,
                progress=0,
                schedule_type="continuous",
                schedule_cron=None,
                next_run_at=datetime.utcnow()
                + timedelta(minutes=int(profile.interval_minutes or 1440)),
                config=profile_job_config,
                max_iterations=6,
                max_tool_calls=20,
                max_llm_calls=12,
                max_runtime_minutes=45,
            )
            db.add(profile_job)
            await db.flush()
            profile.latest_run_job_id = profile_job.id
            profile.active_job_id = profile_job.id
            profile.status = "running"
            profile.started_at = profile.started_at or datetime.utcnow()
            profile.paused_at = None
            queued_job_ids.append(str(profile_job.id))

        portfolio: ResearchPortfolio | None = None
        if payload.target_mode == "profile_with_portfolio":
            if payload.portfolio_id:
                portfolio = await db.get(ResearchPortfolio, payload.portfolio_id)
                if portfolio is None or portfolio.user_id != current_user.id:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Research portfolio not found",
                    )
                linked_profile_ids = [
                    str(value).strip()
                    for value in (portfolio.linked_profile_ids or [])
                    if str(value).strip()
                ]
                if str(profile.id) not in linked_profile_ids:
                    linked_profile_ids.append(str(profile.id))
                portfolio.linked_profile_ids = linked_profile_ids[:24]
            else:
                portfolio_data = dict(promotion_seed["portfolio"])
                if payload.portfolio:
                    portfolio_data.update(
                        payload.portfolio.model_dump(exclude_none=True)
                    )
                portfolio_data.update(
                    {
                        "linked_profile_ids": [profile.id],
                        "start_immediately": False,
                    }
                )
                portfolio_request = ResearchPortfolioCreate.model_validate(
                    portfolio_data
                )
                await validate_sandbox_profile(
                    db,
                    sandbox_profile_id=portfolio_request.sandbox_profile_id,
                    track_type=profile.track_type,
                )
                portfolio = ResearchPortfolio(
                    user_id=current_user.id,
                    title=portfolio_request.title,
                    objective=portfolio_request.objective,
                    status="draft",
                    linked_profile_ids=[str(profile.id)],
                    automation_profile=normalize_portfolio_automation_profile(
                        portfolio_request.automation_profile,
                        default="balanced",
                    ),
                    automation_policy=resolve_portfolio_automation_policy(
                        portfolio_request.automation_profile,
                        portfolio_request.automation_policy,
                    ),
                    sandbox_profile_id=portfolio_request.sandbox_profile_id,
                    opportunities=[],
                    child_job_ids=[],
                )
                db.add(portfolio)
                await db.flush()

            if payload.run_portfolio_now and portfolio is not None:
                portfolio_job = AgentJob(
                    user_id=current_user.id,
                    name=f"Research Fleet — {portfolio.title}",
                    goal=portfolio.objective,
                    job_type="research",
                    status=AgentJobStatus.PENDING.value,
                    progress=0,
                    schedule_type="once",
                    schedule_cron=None,
                    next_run_at=None,
                    config={
                        "launch_mode": "research_fleet_portfolio",
                        "deterministic_runner": "research_fleet_orchestrator",
                        "research_portfolio_id": str(portfolio.id),
                        "linked_profile_ids": list(portfolio.linked_profile_ids or []),
                        "automation_profile": (
                            normalize_portfolio_automation_profile(
                                portfolio.automation_profile,
                                default="balanced",
                            )
                        ),
                        "automation_policy": resolve_portfolio_automation_policy(
                            portfolio.automation_profile,
                            portfolio.automation_policy,
                        ),
                        "sandbox_profile_id": (
                            str(portfolio.sandbox_profile_id or "").strip() or None
                        ),
                        "interval_minutes": 1440,
                    },
                    max_iterations=6,
                    max_tool_calls=24,
                    max_llm_calls=16,
                    max_runtime_minutes=45,
                )
                db.add(portfolio_job)
                await db.flush()
                portfolio.latest_run_job_id = portfolio_job.id
                queued_job_ids.append(str(portfolio_job.id))

        promotion_status = (
            "promoted_to_profile_and_portfolio"
            if portfolio is not None
            else "promoted_to_profile"
        )
        promotion_metadata = {
            "status": promotion_status,
            "promoted_at": datetime.utcnow().isoformat(),
            "source_job_id": str(job.id),
            "domain_research_profile_id": str(profile.id),
            "research_portfolio_id": (
                str(portfolio.id) if portfolio is not None else None
            ),
            "target_mode": payload.target_mode,
            "start_profile_now": bool(payload.start_profile_now),
            "run_portfolio_now": bool(payload.run_portfolio_now),
        }
        next_config = dict(config)
        next_config["promotion"] = promotion_metadata
        quick_start = (
            dict(next_config.get("quick_start"))
            if isinstance(next_config.get("quick_start"), dict)
            else {}
        )
        quick_start["promotion"] = promotion_metadata
        next_config["quick_start"] = quick_start
        job.config = next_config

        await db.commit()
        await db.refresh(job)
        await db.refresh(profile)
        if portfolio is not None:
            await db.refresh(portfolio)

        refreshed_promotion = extract_promotion(job)
        refreshed_status = (
            str(refreshed_promotion.get("status") or promotion_status).strip()
            or promotion_status
        )
        profile_id = _promotion_uuid(
            refreshed_promotion.get("domain_research_profile_id")
            or refreshed_promotion.get("promoted_domain_research_profile_id"),
            fallback=profile.id,
        )
        portfolio_id = _promotion_uuid(
            refreshed_promotion.get("research_portfolio_id")
            or refreshed_promotion.get("promoted_research_portfolio_id"),
            fallback=portfolio.id if portfolio is not None else None,
        )

        for queued_job_id in queued_job_ids:
            execute_job_task.delay(queued_job_id, str(current_user.id))

        return AgentJobPromoteDomainResearchResponse(
            source_job_id=job.id,
            promotion_status=refreshed_status,
            domain_research_profile_id=profile_id or profile.id,
            research_portfolio_id=portfolio_id,
            profile=await present_profile(profile, db),
            portfolio=(
                await present_portfolio(portfolio, db)
                if portfolio is not None
                else None
            ),
            source_job=present_job(
                job,
                current_user_id=str(current_user.id),
            ),
        )

    return DomainResearchPromotionApi(
        router=router,
        promote_domain_research_job=promote_domain_research_job,
    )
