from __future__ import annotations

from datetime import datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.domain_research_profile import DomainResearchProfile
from app.models.experiment import ExperimentPlan
from app.models.research_inbox import ResearchInboxItem
from app.models.research_note import ResearchNote
from app.models.research_portfolio import ResearchPortfolio
from app.models.user import User
from app.schemas.research_portfolio import (
    ResearchPortfolioOpportunityActionRequest,
    ResearchPortfolioActionRequest,
    ResearchPortfolioCreate,
    ResearchPortfolioListResponse,
    ResearchPortfolioResponse,
    ResearchPortfolioUpdate,
)
from app.schemas.experiment import ScientificValidationRunSummaryResponse
from app.services.autonomy_service import build_autonomy_summary
from app.services.scientific_validation_service import (
    get_scientific_sandbox_profile,
    list_scientific_validation_run_summaries,
    normalize_portfolio_automation_profile,
    resolve_portfolio_automation_policy,
)
from app.services.research_opportunity_service import (
    apply_materialized_experiment_metadata,
    build_validation_status_map,
    collect_research_opportunity_linked_ids,
    list_normalized_research_opportunities,
    materialize_research_opportunity_experiment,
    normalize_research_opportunity,
)
from app.api.endpoints.agent_jobs import _relaunch_follow_up_inbox_item
from app.services.agent_job_scheduler_state import (
    extract_scheduler_state as _extract_scheduler_state,
)
from app.services.autonomous_agent_executor import AutonomousAgentExecutor
from app.services.autonomy_event_service import record_autonomy_decision_event
from app.tasks.agent_job_tasks import execute_agent_job_task


router = APIRouter()


def _portfolio_job_name(portfolio: ResearchPortfolio) -> str:
    return f"Research Fleet — {portfolio.title}"


async def _get_portfolio_or_404(db: AsyncSession, portfolio_id: UUID, user_id: UUID) -> ResearchPortfolio:
    portfolio = await db.get(ResearchPortfolio, portfolio_id)
    if not portfolio or portfolio.user_id != user_id:
        raise HTTPException(status_code=404, detail="Research portfolio not found")
    return portfolio


async def _validate_profile_ids(db: AsyncSession, user_id: UUID, profile_ids: list[str]) -> None:
    if not profile_ids:
        return
    stmt = select(DomainResearchProfile.id).where(
        DomainResearchProfile.user_id == user_id,
        DomainResearchProfile.id.in_([UUID(str(v)) for v in profile_ids]),
    )
    found = {str(row) for row in (await db.execute(stmt)).scalars().all()}
    missing = [value for value in profile_ids if str(value) not in found]
    if missing:
        raise HTTPException(status_code=400, detail=f"Unknown domain profile ids: {', '.join(missing[:5])}")


async def _validate_sandbox_profile(db: AsyncSession, sandbox_profile_id: str | None) -> None:
    requested = str(sandbox_profile_id or "").strip()
    if not requested:
        return
    profile = await get_scientific_sandbox_profile(db, requested, include_disabled=False)
    if not isinstance(profile, dict):
        raise HTTPException(status_code=400, detail="Unknown or disabled sandbox profile")


async def _create_portfolio_job(
    *,
    db: AsyncSession,
    portfolio: ResearchPortfolio,
    schedule_type: str,
    start_immediately: bool,
) -> AgentJob:
    config = {
        "launch_mode": "research_fleet_portfolio",
        "deterministic_runner": "research_fleet_orchestrator",
        "research_portfolio_id": str(portfolio.id),
        "linked_profile_ids": list(portfolio.linked_profile_ids or []),
        "automation_profile": normalize_portfolio_automation_profile(portfolio.automation_profile, default="balanced"),
        "automation_policy": resolve_portfolio_automation_policy(portfolio.automation_profile, portfolio.automation_policy),
        "sandbox_profile_id": str(portfolio.sandbox_profile_id or "").strip() or None,
        "interval_minutes": 1440,
    }
    job = AgentJob(
        user_id=portfolio.user_id,
        name=_portfolio_job_name(portfolio),
        goal=portfolio.objective,
        job_type="research",
        status=AgentJobStatus.PENDING.value,
        progress=0,
        schedule_type=schedule_type,
        schedule_cron=None,
        next_run_at=datetime.utcnow() + timedelta(minutes=1440) if schedule_type == "continuous" else None,
        config=config,
        max_iterations=6,
        max_tool_calls=24,
        max_llm_calls=16,
        max_runtime_minutes=45,
    )
    db.add(job)
    await db.flush()
    portfolio.latest_run_job_id = job.id
    if schedule_type == "continuous":
        portfolio.active_job_id = job.id
        portfolio.status = "running"
        portfolio.started_at = portfolio.started_at or datetime.utcnow()
        portfolio.paused_at = None
    await db.commit()
    await db.refresh(job)
    await db.refresh(portfolio)
    if start_immediately:
        execute_agent_job_task.delay(str(job.id), str(portfolio.user_id))
    return job


async def _portfolio_response(portfolio: ResearchPortfolio, db: AsyncSession) -> ResearchPortfolioResponse:
    automation_profile = normalize_portfolio_automation_profile(getattr(portfolio, "automation_profile", None), default="balanced")
    automation_policy = resolve_portfolio_automation_policy(automation_profile, portfolio.automation_policy)
    base = ResearchPortfolioResponse.model_validate(portfolio)
    summaries = await list_scientific_validation_run_summaries(
        db,
        user_id=portfolio.user_id,
        run_ids=portfolio.latest_validation_run_ids or [],
        limit=5,
    )
    typed_summaries = (
        [ScientificValidationRunSummaryResponse.model_validate(summary) for summary in summaries]
        if summaries
        else None
    )
    summary = portfolio.latest_summary if isinstance(portfolio.latest_summary, dict) else {}
    validation_status_by_id = build_validation_status_map(
        latest_validation_runs=[row.model_dump() for row in typed_summaries] if typed_summaries else [],
        summary_validation_runs=summary.get("validation_runs") if isinstance(summary.get("validation_runs"), list) else [],
    )
    opportunities = list_normalized_research_opportunities(portfolio.opportunities, validation_status_by_id=validation_status_by_id)
    summary = build_autonomy_summary(
        raw_summary=summary,
        opportunities=opportunities,
        automation_profile=automation_profile,
        effective_policy=automation_policy,
        sandbox_profile_id=portfolio.sandbox_profile_id,
        config_revision_key="portfolio_config_revision",
    )
    return base.model_copy(
        update={
            "automation_profile": automation_profile,
            "automation_policy": automation_policy,
            "effective_policy": automation_policy,
            "latest_validation_runs": typed_summaries,
            "latest_summary": summary,
            "opportunities": opportunities,
        }
    )


def _find_opportunity(rows: list[dict], opportunity_id: str) -> tuple[int, dict]:
    for idx, row in enumerate(rows):
        if str(row.get("opportunity_id") or "").strip() == opportunity_id:
            return idx, row
    raise HTTPException(status_code=404, detail="Research opportunity not found")


async def _resolve_portfolio_parent_job(db: AsyncSession, portfolio: ResearchPortfolio) -> AgentJob:
    parent_job_id = portfolio.latest_run_job_id or portfolio.active_job_id
    if not parent_job_id:
        raise HTTPException(status_code=400, detail="Portfolio must run at least once before launching downstream actions")
    parent_job = await db.get(AgentJob, parent_job_id)
    if parent_job is None:
        raise HTTPException(status_code=400, detail="Latest portfolio run is unavailable")
    return parent_job


async def _maybe_resolve_portfolio_parent_job(
    db: AsyncSession,
    portfolio: ResearchPortfolio,
) -> AgentJob | None:
    parent_job_id = portfolio.latest_run_job_id or portfolio.active_job_id
    if not parent_job_id:
        return None
    return await db.get(AgentJob, parent_job_id)


async def _create_plan_for_opportunity(
    *,
    db: AsyncSession,
    user_id: UUID,
    note_ids: list[str],
    title: str,
    hypothesis: str,
    generator: str,
    generator_details: dict,
) -> str:
    for existing_id in generator_details.get("existing_plan_ids") or []:
        text = str(existing_id or "").strip()
        if text:
            return text
    recent_plans = list(
        (
            await db.execute(
                select(ExperimentPlan)
                .where(ExperimentPlan.user_id == user_id)
                .order_by(ExperimentPlan.created_at.desc())
                .limit(80)
            )
        ).scalars().all()
    )
    target_opportunity_id = str(generator_details.get("opportunity_id") or "").strip()
    target_title = str(generator_details.get("idea_title") or title).strip().lower()
    for plan in recent_plans:
        details = plan.generator_details if isinstance(plan.generator_details, dict) else {}
        if target_opportunity_id and str(details.get("opportunity_id") or "").strip() == target_opportunity_id:
            return str(plan.id)
        if target_title and str(details.get("idea_title") or "").strip().lower() == target_title:
            return str(plan.id)
    note = None
    for note_id in note_ids:
        try:
            note = await db.get(ResearchNote, UUID(str(note_id)))
        except Exception:
            note = None
        if note is not None and note.user_id == user_id:
            break
    if note is None:
        raise HTTPException(status_code=400, detail="Opportunity is missing a linked research note")
    plan = ExperimentPlan(
        user_id=user_id,
        research_note_id=note.id,
        title=f"Experiment Plan: {title[:460]}",
        hypothesis_text=hypothesis[:4000],
        plan={
            "portfolio_opportunity_title": title,
            "selected_hypothesis_ids": [target_opportunity_id] if target_opportunity_id else [],
            "provenance": {
                "autonomous_origin": {
                    "source_kind": "portfolio",
                    "source_id": str(generator_details.get("portfolio_id") or "").strip() or None,
                    "opportunity_id": target_opportunity_id or None,
                    "evidence_revision_at_launch": str(generator_details.get("evidence_revision_at_launch") or "").strip() or None,
                }
            },
            "recommended_experiments": [
                f"Validate {title[:180]} against current baselines",
                "Define measurable success criteria, instrumentation, and dataset slice",
                "Record outcome, blocked reasons, and next action",
            ],
        },
        generator=generator,
        generator_details={k: v for k, v in generator_details.items() if k != "existing_plan_ids"},
    )
    db.add(plan)
    await db.flush()
    return str(plan.id)


async def _sync_portfolio_opportunities(portfolio: ResearchPortfolio, opportunities: list[dict]) -> None:
    normalized = list_normalized_research_opportunities(opportunities)
    linked_ids = collect_research_opportunity_linked_ids(normalized)
    portfolio.opportunities = normalized
    summary = dict(portfolio.latest_summary) if isinstance(portfolio.latest_summary, dict) else {}
    effective_policy = resolve_portfolio_automation_policy(portfolio.automation_profile, portfolio.automation_policy)
    summary = build_autonomy_summary(
        raw_summary=summary,
        opportunities=normalized,
        automation_profile=normalize_portfolio_automation_profile(portfolio.automation_profile, default="balanced"),
        effective_policy=effective_policy,
        sandbox_profile_id=portfolio.sandbox_profile_id,
        config_revision_key="portfolio_config_revision",
    )
    portfolio.latest_summary = summary
    portfolio.latest_note_ids = list(dict.fromkeys([*([str(v) for v in (portfolio.latest_note_ids or []) if str(v).strip()]), *linked_ids["note_ids"]]))[:30]
    portfolio.latest_experiment_plan_ids = linked_ids["plan_ids"][:30]
    portfolio.latest_validation_run_ids = linked_ids["run_ids"][:30]
    portfolio.child_job_ids = linked_ids["child_job_ids"][:50]


@router.get("", response_model=ResearchPortfolioListResponse)
async def list_research_portfolios(
    status_filter: str | None = Query(None, alias="status"),
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    stmt = select(ResearchPortfolio).where(ResearchPortfolio.user_id == current_user.id)
    if str(status_filter or "").strip():
        stmt = stmt.where(ResearchPortfolio.status == str(status_filter).strip())
    stmt = stmt.order_by(desc(ResearchPortfolio.updated_at)).offset(offset).limit(limit)
    rows = list((await db.execute(stmt)).scalars().all())
    total_stmt = select(ResearchPortfolio.id).where(ResearchPortfolio.user_id == current_user.id)
    if str(status_filter or "").strip():
        total_stmt = total_stmt.where(ResearchPortfolio.status == str(status_filter).strip())
    total = len(list((await db.execute(total_stmt)).scalars().all()))
    return ResearchPortfolioListResponse(
        items=[await _portfolio_response(row, db) for row in rows],
        total=total,
    )


@router.post("", response_model=ResearchPortfolioResponse, status_code=status.HTTP_201_CREATED)
async def create_research_portfolio(
    payload: ResearchPortfolioCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    linked_profile_ids = [str(v) for v in (payload.linked_profile_ids or [])]
    await _validate_profile_ids(db, current_user.id, linked_profile_ids)
    await _validate_sandbox_profile(db, payload.sandbox_profile_id)
    portfolio = ResearchPortfolio(
        user_id=current_user.id,
        title=payload.title,
        objective=payload.objective,
        status="draft",
        linked_profile_ids=linked_profile_ids,
        automation_profile=normalize_portfolio_automation_profile(payload.automation_profile, default="balanced"),
        automation_policy=resolve_portfolio_automation_policy(payload.automation_profile, payload.automation_policy),
        sandbox_profile_id=payload.sandbox_profile_id,
        opportunities=[],
        child_job_ids=[],
    )
    db.add(portfolio)
    await db.flush()
    if payload.start_immediately:
        await _create_portfolio_job(db=db, portfolio=portfolio, schedule_type="continuous", start_immediately=True)
        await db.refresh(portfolio)
        return await _portfolio_response(portfolio, db)
    await db.commit()
    await db.refresh(portfolio)
    return await _portfolio_response(portfolio, db)


@router.get("/{portfolio_id}", response_model=ResearchPortfolioResponse)
async def get_research_portfolio(
    portfolio_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    portfolio = await _get_portfolio_or_404(db, portfolio_id, current_user.id)
    return await _portfolio_response(portfolio, db)


@router.patch("/{portfolio_id}", response_model=ResearchPortfolioResponse)
async def update_research_portfolio(
    portfolio_id: UUID,
    payload: ResearchPortfolioUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    portfolio = await _get_portfolio_or_404(db, portfolio_id, current_user.id)
    updates = payload.model_dump(exclude_none=True)
    await _validate_sandbox_profile(db, updates.get("sandbox_profile_id", portfolio.sandbox_profile_id))
    if "linked_profile_ids" in updates:
        linked_profile_ids = [str(v) for v in (updates.get("linked_profile_ids") or [])]
        await _validate_profile_ids(db, current_user.id, linked_profile_ids)
        updates["linked_profile_ids"] = linked_profile_ids
    if "automation_profile" in updates or "automation_policy" in updates:
        automation_profile = normalize_portfolio_automation_profile(
            updates.get("automation_profile", getattr(portfolio, "automation_profile", None)),
            default="balanced",
        )
        updates["automation_profile"] = automation_profile
        updates["automation_policy"] = resolve_portfolio_automation_policy(
            automation_profile,
            updates.get("automation_policy", portfolio.automation_policy),
        )
    for key, value in updates.items():
        setattr(portfolio, key, value)
    await db.commit()
    await db.refresh(portfolio)
    return await _portfolio_response(portfolio, db)


@router.post("/{portfolio_id}/action", response_model=ResearchPortfolioResponse)
async def act_on_research_portfolio(
    portfolio_id: UUID,
    payload: ResearchPortfolioActionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    portfolio = await _get_portfolio_or_404(db, portfolio_id, current_user.id)
    action = payload.action
    active_job = await db.get(AgentJob, portfolio.active_job_id) if portfolio.active_job_id else None

    if action == "start":
        await _create_portfolio_job(db=db, portfolio=portfolio, schedule_type="continuous", start_immediately=True)
    elif action == "resume":
        if active_job is not None:
            active_job.schedule_type = "continuous"
            active_job.next_run_at = datetime.utcnow()
            portfolio.status = "running"
            portfolio.paused_at = None
            await db.commit()
            execute_agent_job_task.delay(str(active_job.id), str(portfolio.user_id))
        else:
            await _create_portfolio_job(db=db, portfolio=portfolio, schedule_type="continuous", start_immediately=True)
    elif action == "pause":
        portfolio.status = "paused"
        portfolio.paused_at = datetime.utcnow()
        if active_job is not None:
            active_job.schedule_type = None
            active_job.next_run_at = None
        await db.commit()
    elif action == "cancel":
        portfolio.status = "cancelled"
        if active_job is not None:
            active_job.schedule_type = None
            active_job.next_run_at = None
            if active_job.status in {AgentJobStatus.PENDING.value, AgentJobStatus.RUNNING.value, AgentJobStatus.PAUSED.value}:
                active_job.status = AgentJobStatus.CANCELLED.value
        portfolio.active_job_id = None
        await db.commit()
    elif action == "run_now":
        await _create_portfolio_job(db=db, portfolio=portfolio, schedule_type="once", start_immediately=True)
    else:
        raise HTTPException(status_code=400, detail="Unsupported action")

    await db.refresh(portfolio)
    return await _portfolio_response(portfolio, db)


@router.post("/{portfolio_id}/opportunities/{opportunity_id}/action", response_model=ResearchPortfolioResponse)
async def act_on_research_portfolio_opportunity(
    portfolio_id: UUID,
    opportunity_id: str,
    payload: ResearchPortfolioOpportunityActionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    portfolio = await _get_portfolio_or_404(db, portfolio_id, current_user.id)
    parent_job = await _maybe_resolve_portfolio_parent_job(db, portfolio)
    opportunities = list_normalized_research_opportunities(portfolio.opportunities)
    idx, opportunity = _find_opportunity(opportunities, opportunity_id)
    opportunity_before = dict(opportunity)
    action = str(payload.action or "").strip().lower()
    operator_note = str(payload.operator_note or "").strip() or None
    now_iso = datetime.utcnow().isoformat()

    if action == "suppress" and not operator_note:
        raise HTTPException(status_code=400, detail="operator_note is required when suppressing an opportunity")

    existing_plan_ids = [str(v) for v in (opportunity.get("linked_experiment_plan_ids") or []) if str(v).strip()]
    existing_run_ids = [str(v) for v in (opportunity.get("linked_validation_run_ids") or []) if str(v).strip()]
    existing_child_job_ids = [str(v) for v in (opportunity.get("child_job_ids") or []) if str(v).strip()]

    async def _ensure_plan_ids(current_plan_ids: list[str]) -> list[str]:
        plan_ids = list(current_plan_ids or [])
        if plan_ids:
            return plan_ids
        plan_id = await _create_plan_for_opportunity(
            db=db,
            user_id=current_user.id,
            note_ids=[str(v) for v in (opportunity.get("source_note_ids") or []) if str(v).strip()] or [str(v) for v in (portfolio.latest_note_ids or []) if str(v).strip()],
            title=str(opportunity.get("title") or "Research opportunity"),
            hypothesis=str(opportunity.get("hypothesis") or opportunity.get("title") or ""),
            generator="research_fleet_operator_action",
            generator_details={
                "origin": "research_portfolio_action",
                "portfolio_id": str(portfolio.id),
                "opportunity_id": opportunity["opportunity_id"],
                "idea_title": str(opportunity.get("title") or ""),
                "supporting_sources": opportunity.get("supporting_sources") if isinstance(opportunity.get("supporting_sources"), list) else [],
                "selected_hypothesis_ids": [opportunity["opportunity_id"]],
                "evidence_revision_at_launch": str(opportunity.get("evidence_revision") or "").strip() or None,
                "existing_plan_ids": plan_ids,
                "created_at": now_iso,
            },
        )
        return [plan_id]

    opportunity["updated_at"] = now_iso
    opportunity["decision_source"] = "operator"
    if operator_note is not None:
        opportunity["operator_note"] = operator_note

    if action == "accept":
        opportunity["decision_state"] = "accepted"
        opportunity["stage"] = "accepted"
    elif action == "suppress":
        opportunity["decision_state"] = "suppressed"
        opportunity["stage"] = "suppressed"
    elif action == "reopen":
        opportunity["decision_state"] = "pending_review"
        opportunity["stage"] = "discovered"
    elif action == "create_plan":
        plan_ids = await _ensure_plan_ids(existing_plan_ids)
        opportunity["linked_experiment_plan_ids"] = plan_ids
        opportunity["latest_experiment_plan_id"] = plan_ids[-1] if plan_ids else None
        opportunity["decision_state"] = "accepted"
        opportunity["stage"] = "planned"
    elif action == "launch_validation":
        parent_job = await _resolve_portfolio_parent_job(db, portfolio)
        materialization = await materialize_research_opportunity_experiment(
            db=db,
            parent_job=parent_job,
            owner_kind="portfolio",
            owner_id=str(portfolio.id),
            user_id=str(current_user.id),
            opportunity=opportunity,
            title=str(opportunity.get("title") or ""),
            hypothesis=str(opportunity.get("hypothesis") or opportunity.get("title") or ""),
            note_ids=[str(v) for v in (opportunity.get("source_note_ids") or []) if str(v).strip()] or [str(v) for v in (portfolio.latest_note_ids or []) if str(v).strip()],
            track_type=str(opportunity.get("track_type") or "generic"),
            objective=portfolio.objective,
            validation_policy=portfolio.automation_policy if isinstance(portfolio.automation_policy, dict) else {},
            sandbox_profile_id=portfolio.sandbox_profile_id,
            repo_source_ids=[str(v) for v in (opportunity.get("source_repo_ids") or []) if str(v).strip()],
            benchmark_queries=[],
            ensure_plan_ids=_ensure_plan_ids,
            portfolio_id=str(portfolio.id),
            originating_job_id=str(parent_job.id),
            start_immediately=False,
        )
        opportunity = apply_materialized_experiment_metadata(
            opportunity,
            owner_kind="portfolio",
            owner_id=str(portfolio.id),
            plan_ids=materialization["plan_ids"],
            run_id=materialization.get("run_id"),
            job_id=materialization.get("job_id"),
            validation_status=materialization.get("validation_status"),
            blocked_reason_code=materialization.get("blocked_reason_code"),
            materialized_at=now_iso,
        )
    elif action == "materialize_experiment":
        parent_job = await _resolve_portfolio_parent_job(db, portfolio)
        materialization = await materialize_research_opportunity_experiment(
            db=db,
            parent_job=parent_job,
            owner_kind="portfolio",
            owner_id=str(portfolio.id),
            user_id=str(current_user.id),
            opportunity=opportunity,
            title=str(opportunity.get("title") or ""),
            hypothesis=str(opportunity.get("hypothesis") or opportunity.get("title") or ""),
            note_ids=[str(v) for v in (opportunity.get("source_note_ids") or []) if str(v).strip()] or [str(v) for v in (portfolio.latest_note_ids or []) if str(v).strip()],
            track_type=str(opportunity.get("track_type") or "generic"),
            objective=portfolio.objective,
            validation_policy=portfolio.automation_policy if isinstance(portfolio.automation_policy, dict) else {},
            sandbox_profile_id=portfolio.sandbox_profile_id,
            repo_source_ids=[str(v) for v in (opportunity.get("source_repo_ids") or []) if str(v).strip()],
            benchmark_queries=[],
            ensure_plan_ids=_ensure_plan_ids,
            portfolio_id=str(portfolio.id),
            originating_job_id=str(parent_job.id),
            start_immediately=payload.start_immediately is not False,
        )
        opportunity = apply_materialized_experiment_metadata(
            opportunity,
            owner_kind="portfolio",
            owner_id=str(portfolio.id),
            plan_ids=materialization["plan_ids"],
            run_id=materialization.get("run_id"),
            job_id=materialization.get("job_id"),
            validation_status=materialization.get("validation_status"),
            blocked_reason_code=materialization.get("blocked_reason_code"),
            materialized_at=now_iso,
        )
    elif action == "launch_follow_up":
        if existing_child_job_ids:
            opportunity["decision_state"] = "accepted"
            opportunity["stage"] = "validating"
            opportunity["follow_up_review_status"] = "approved_launch"
            opportunity["follow_up_review_evidence_revision"] = opportunity.get("evidence_revision")
            opportunities[idx] = normalize_research_opportunity(opportunity)
            await _sync_portfolio_opportunities(portfolio, opportunities)
            await record_autonomy_decision_event(
                db,
                user_id=current_user.id,
                event_type="follow_up_launched",
                event_time=datetime.utcnow(),
                source_kind="portfolio",
                source_id=str(portfolio.id),
                source_label=str(portfolio.title or "Research fleet").strip(),
                decision_type="follow_up_launched",
                reason_code="existing_follow_up_job",
                status=str(opportunities[idx].get("stage") or "").strip() or None,
                severity="medium",
                actor_mode="operator",
                summary=f"{str(portfolio.title or 'Research fleet').strip()}: reused existing follow-up for {str(opportunity.get('title') or 'opportunity').strip()}",
                operator_note=operator_note,
                reason_label="Follow-up launched",
                scheduler_state=_extract_scheduler_state(parent_job),
                before_state=opportunity_before,
                after_state=opportunities[idx],
                deep_link={"target_tab": "fleet", "params": {"tab": "fleet", "fleetId": str(portfolio.id)}, "label": "Open Research Fleet"},
                metadata={"opportunity_id": opportunity_id},
            )
            await db.commit()
            await db.refresh(portfolio)
            return await _portfolio_response(portfolio, db)
        parent_job = await _resolve_portfolio_parent_job(db, portfolio)
        executor = AutonomousAgentExecutor()
        child_job = await executor._create_domain_research_follow_up_job(
            db=db,
            job=parent_job,
            domain=str(opportunity.get("title") or portfolio.title),
            objective=portfolio.objective,
            customer_context="research_portfolio",
            track_type=str(opportunity.get("track_type") or "generic"),
            source_scope="kb_plus_arxiv_plus_repo" if opportunity.get("source_repo_ids") else "kb_plus_arxiv",
            top_idea=opportunity,
            docs=[],
            repo_documents=[],
            papers=[],
            repo_source_ids=[str(v) for v in (opportunity.get("source_repo_ids") or []) if str(v).strip()],
            benchmark_queries=[],
            automation_profile=portfolio.automation_profile,
            automation_policy=portfolio.automation_policy if isinstance(portfolio.automation_policy, dict) else {},
            sandbox_profile_id=portfolio.sandbox_profile_id,
        )
        if child_job is None:
            raise HTTPException(status_code=400, detail="Failed to launch follow-up job")
        opportunity["child_job_ids"] = list(dict.fromkeys([*([str(v) for v in (opportunity.get("child_job_ids") or []) if str(v).strip()]), str(child_job.id)]))[:8]
        opportunity["decision_state"] = "accepted"
        opportunity["stage"] = "validating"
        opportunity["follow_up_review_status"] = "approved_launch"
        opportunity["follow_up_review_evidence_revision"] = opportunity.get("evidence_revision")
        execute_agent_job_task.delay(str(child_job.id), str(portfolio.user_id))
    elif action == "relaunch_follow_up":
        follow_up_last_job_id = str(opportunity.get("follow_up_last_job_id") or "").strip()
        follow_up_outcome_status = str(opportunity.get("follow_up_outcome_status") or "").strip().lower()
        if follow_up_outcome_status not in {"failed", "cancelled"}:
            raise HTTPException(status_code=400, detail="Only failed or cancelled follow-ups can be relaunched")
        if not follow_up_last_job_id:
            raise HTTPException(status_code=422, detail="Opportunity is missing its last follow-up job identifier")
        try:
            follow_up_job_uuid = UUID(follow_up_last_job_id)
        except Exception as exc:
            raise HTTPException(status_code=422, detail="Opportunity has an invalid last follow-up job identifier") from exc
        inbox_item = (
            await db.execute(
                select(ResearchInboxItem).where(
                    ResearchInboxItem.user_id == current_user.id,
                    ResearchInboxItem.follow_up_job_id == follow_up_job_uuid,
                )
            )
        ).scalar_one_or_none()
        if inbox_item is None:
            raise HTTPException(status_code=422, detail="Could not resolve a relaunchable inbox follow-up for this opportunity")
        await _relaunch_follow_up_inbox_item(
            item=inbox_item,
            operator_note=operator_note,
            db=db,
            current_user=current_user,
        )
        parent_job = await _resolve_portfolio_parent_job(db, portfolio)
        opportunities = list_normalized_research_opportunities(portfolio.opportunities)
        idx, opportunity = _find_opportunity(opportunities, opportunity_id)
    else:
        raise HTTPException(status_code=400, detail="Unsupported action")

    opportunities[idx] = normalize_research_opportunity(opportunity)
    await _sync_portfolio_opportunities(portfolio, opportunities)
    decision_event_type = {
        "accept": "opportunity_accepted",
        "suppress": "opportunity_suppressed",
        "reopen": "opportunity_reopened",
        "create_plan": "experiment_plan_created",
        "launch_validation": "validation_requeued" if opportunity.get("linked_validation_run_ids") else "validation_blocked",
        "materialize_experiment": "experiment_materialized",
        "launch_follow_up": "follow_up_launched",
        "relaunch_follow_up": "follow_up_launched",
    }.get(action, "opportunity_updated")
    await record_autonomy_decision_event(
        db,
        user_id=current_user.id,
        event_type=decision_event_type,
        event_time=datetime.utcnow(),
        source_kind="portfolio",
        source_id=str(portfolio.id),
        source_label=str(portfolio.title or "Research fleet").strip(),
        decision_type=decision_event_type,
        reason_code=(
            "operator_relaunched_follow_up"
            if action == "relaunch_follow_up"
            else str(opportunities[idx].get("last_decision_reason_code") or opportunities[idx].get("last_blocked_reason_code") or action).strip() or None
        ),
        status=str(opportunities[idx].get("stage") or opportunities[idx].get("autonomy_state") or "").strip() or None,
        severity="high" if str(opportunities[idx].get("stage") or "").strip().lower() == "blocked" else "medium",
        actor_mode="operator",
        summary=f"{str(portfolio.title or 'Research fleet').strip()}: {action.replace('_', ' ')} {str(opportunity.get('title') or 'opportunity').strip()}",
        operator_note=operator_note,
        reason_label=str(decision_event_type).replace("_", " ").strip().capitalize(),
        scheduler_state=_extract_scheduler_state(parent_job),
        before_state=opportunity_before,
        after_state=opportunities[idx],
        deep_link={"target_tab": "fleet", "params": {"tab": "fleet", "fleetId": str(portfolio.id)}, "label": "Open Research Fleet"},
        metadata={"opportunity_id": opportunity_id},
    )
    await db.commit()
    await db.refresh(portfolio)
    return await _portfolio_response(portfolio, db)
