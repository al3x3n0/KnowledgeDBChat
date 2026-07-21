from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.agent_jobs import (
    _build_domain_research_goal,
    _build_quick_start_domain_research_config,
    _relaunch_follow_up_inbox_item,
)
from app.services.agent_job_scheduler_state import (
    extract_scheduler_state as _extract_scheduler_state,
)
from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.experiment import ExperimentPlan
from app.models.research_note import ResearchNote
from app.models.domain_research_profile import DomainResearchProfile
from app.models.research_inbox import ResearchInboxItem
from app.models.user import User
from app.schemas.agent_job import AgentJobQuickStartDomainResearchRequest
from app.schemas.domain_research_profile import (
    DomainResearchProfileActionRequest,
    ResearchOpportunityActionRequest,
    DomainResearchProfileCreate,
    DomainResearchProfileListResponse,
    DomainResearchProfileResponse,
    DomainResearchProfileUpdate,
)
from app.schemas.experiment import ScientificValidationRunSummaryResponse
from app.services.autonomy_service import (
    PROFILE_POLICY_COMPAT_FIELDS,
    build_autonomy_summary,
    build_domain_profile_compat_policy,
    current_domain_profile_policy_snapshot,
    resolve_domain_profile_automation_contract,
)
from app.services.scientific_validation_service import (
    get_scientific_sandbox_profile,
    list_scientific_validation_run_summaries,
)
from app.services.research_opportunity_service import (
    apply_materialized_experiment_metadata,
    build_validation_status_map,
    collect_research_opportunity_linked_ids,
    list_normalized_research_opportunities,
    materialize_research_opportunity_experiment,
    normalize_research_opportunity,
)
from app.services.autonomous_agent_executor import AutonomousAgentExecutor
from app.services.autonomy_event_service import record_autonomy_decision_event
from app.tasks.agent_job_tasks import execute_agent_job_task


router = APIRouter()

def _profile_to_quick_start_request(profile: DomainResearchProfile) -> AgentJobQuickStartDomainResearchRequest:
    automation_profile, effective_policy = resolve_domain_profile_automation_contract(
        automation_profile=getattr(profile, "automation_profile", None),
        automation_policy=getattr(profile, "automation_policy", None),
        current_snapshot=current_domain_profile_policy_snapshot(profile),
    )
    return AgentJobQuickStartDomainResearchRequest(
        domain=profile.domain,
        objective=profile.objective,
        customer_context=profile.customer_context,
        source_scope=profile.source_scope,
        track_type=profile.track_type,
        research_mode=profile.research_mode,
        monitor_queries=profile.monitor_queries or None,
        repo_source_ids=profile.repo_source_ids or None,
        benchmark_queries=profile.benchmark_queries or None,
        sandbox_profile_id=profile.sandbox_profile_id,
        report_format=profile.report_format,
        scoring_policy=profile.scoring_policy if isinstance(profile.scoring_policy, dict) else None,
        selection_policy=profile.selection_policy if isinstance(profile.selection_policy, dict) else None,
        persist_artifacts=bool(profile.persist_artifacts),
        auto_launch_follow_up=bool(effective_policy.get("auto_launch_follow_up", profile.auto_launch_follow_up)),
        auto_create_experiment_plans=bool(effective_policy.get("auto_create_experiment_plans", profile.auto_create_experiment_plans)),
        max_documents=int(profile.max_documents or 10),
        max_papers=int(profile.max_papers or 8),
        confidence_threshold=float(effective_policy.get("confidence_threshold", profile.confidence_threshold or 0.7)),
        start_immediately=True,
        profile_id=profile.id,
    )


def _profile_job_name(profile: DomainResearchProfile) -> str:
    return f"Domain Monitor — {profile.title}"


async def _get_profile_or_404(db: AsyncSession, profile_id: UUID, user_id: UUID) -> DomainResearchProfile:
    profile = await db.get(DomainResearchProfile, profile_id)
    if not profile or profile.user_id != user_id:
        raise HTTPException(status_code=404, detail="Domain research profile not found")
    return profile


async def _validate_sandbox_profile(
    db: AsyncSession,
    *,
    sandbox_profile_id: str | None,
    track_type: str,
) -> None:
    requested = str(sandbox_profile_id or "").strip()
    if not requested:
        return
    profile = await get_scientific_sandbox_profile(db, requested, track_type=track_type, include_disabled=False)
    if not isinstance(profile, dict):
        raise HTTPException(status_code=400, detail="Unknown or disabled sandbox profile")


async def _create_profile_job(
    *,
    db: AsyncSession,
    profile: DomainResearchProfile,
    schedule_type: str,
    start_immediately: bool,
) -> AgentJob:
    request = _profile_to_quick_start_request(profile)
    config = _build_quick_start_domain_research_config(request)
    config["profile_id"] = str(profile.id)
    config["monitor_mode"] = "profile"
    config["interval_minutes"] = int(profile.interval_minutes or 1440)
    automation_profile, effective_policy = resolve_domain_profile_automation_contract(
        automation_profile=getattr(profile, "automation_profile", None),
        automation_policy=getattr(profile, "automation_policy", None),
        current_snapshot=current_domain_profile_policy_snapshot(profile),
    )
    config["automation_profile"] = automation_profile
    config["automation_policy"] = effective_policy
    config["validation_policy"] = build_domain_profile_compat_policy(effective_policy)
    config["auto_create_experiment_plans"] = bool(effective_policy.get("auto_create_experiment_plans", profile.auto_create_experiment_plans))

    job = AgentJob(
        user_id=profile.user_id,
        name=_profile_job_name(profile),
        goal=_build_domain_research_goal(request),
        job_type="research",
        status=AgentJobStatus.PENDING.value,
        progress=0,
        schedule_type=schedule_type,
        schedule_cron=None,
        next_run_at=datetime.utcnow() + timedelta(minutes=int(profile.interval_minutes or 1440)) if schedule_type == "continuous" else None,
        config=config,
        max_iterations=6,
        max_tool_calls=20,
        max_llm_calls=12,
        max_runtime_minutes=45,
    )
    db.add(job)
    await db.flush()

    profile.latest_run_job_id = job.id
    if schedule_type == "continuous":
        profile.active_job_id = job.id
        profile.status = "running"
        profile.started_at = profile.started_at or datetime.utcnow()
        profile.paused_at = None
    await db.commit()
    await db.refresh(job)
    await db.refresh(profile)

    if start_immediately:
        execute_agent_job_task.delay(str(job.id), str(profile.user_id))
    return job


async def _profile_response(profile: DomainResearchProfile, db: AsyncSession) -> DomainResearchProfileResponse:
    automation_profile, effective_policy = resolve_domain_profile_automation_contract(
        automation_profile=getattr(profile, "automation_profile", None),
        automation_policy=getattr(profile, "automation_policy", None),
        current_snapshot=current_domain_profile_policy_snapshot(profile),
    )
    base = DomainResearchProfileResponse.model_validate(profile)
    summaries = await list_scientific_validation_run_summaries(
        db,
        user_id=profile.user_id,
        run_ids=profile.latest_validation_run_ids or [],
        limit=5,
    )
    typed_summaries = (
        [ScientificValidationRunSummaryResponse.model_validate(summary) for summary in summaries]
        if summaries
        else None
    )
    summary = profile.latest_summary if isinstance(profile.latest_summary, dict) else {}
    validation_status_by_id = build_validation_status_map(
        latest_validation_runs=[row.model_dump() for row in typed_summaries] if typed_summaries else [],
        summary_validation_runs=summary.get("validation_runs") if isinstance(summary.get("validation_runs"), list) else [],
    )
    opportunities = list_normalized_research_opportunities(
        summary.get("opportunities") if isinstance(summary.get("opportunities"), list) else summary.get("idea_candidates"),
        validation_status_by_id=validation_status_by_id,
    )
    summary = build_autonomy_summary(
        raw_summary=summary,
        opportunities=opportunities,
        automation_profile=automation_profile,
        effective_policy=effective_policy,
        sandbox_profile_id=profile.sandbox_profile_id,
        config_revision_key="profile_config_revision",
    )
    return base.model_copy(
        update={
            "automation_profile": automation_profile,
            "automation_policy": effective_policy,
            "effective_policy": effective_policy,
            "validation_policy": build_domain_profile_compat_policy(effective_policy),
            "auto_launch_follow_up": bool(effective_policy.get("auto_launch_follow_up", profile.auto_launch_follow_up)),
            "auto_create_experiment_plans": bool(effective_policy.get("auto_create_experiment_plans", profile.auto_create_experiment_plans)),
            "confidence_threshold": float(effective_policy.get("confidence_threshold", profile.confidence_threshold or 0.7)),
            "latest_validation_runs": typed_summaries,
            "latest_summary": summary,
            "opportunities": opportunities or None,
        }
    )


def _find_opportunity(rows: list[dict[str, Any]], opportunity_id: str) -> tuple[int, dict[str, Any]]:
    for idx, row in enumerate(rows):
        if str(row.get("opportunity_id") or "").strip() == opportunity_id:
            return idx, row
    raise HTTPException(status_code=404, detail="Research opportunity not found")


async def _resolve_profile_parent_job(db: AsyncSession, profile: DomainResearchProfile) -> AgentJob:
    parent_job_id = profile.latest_run_job_id or profile.active_job_id
    if not parent_job_id:
        raise HTTPException(status_code=400, detail="Profile must run at least once before launching downstream actions")
    parent_job = await db.get(AgentJob, parent_job_id)
    if parent_job is None:
        raise HTTPException(status_code=400, detail="Latest profile run is unavailable")
    return parent_job


async def _maybe_resolve_profile_parent_job(
    db: AsyncSession,
    profile: DomainResearchProfile,
) -> AgentJob | None:
    parent_job_id = profile.latest_run_job_id or profile.active_job_id
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
    generator_details: dict[str, Any],
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
            "opportunity_title": title,
            "supporting_evidence": generator_details.get("supporting_evidence") if isinstance(generator_details.get("supporting_evidence"), list) else [],
            "selected_hypothesis_ids": [target_opportunity_id] if target_opportunity_id else [],
            "provenance": {
                "autonomous_origin": {
                    "source_kind": "profile",
                    "source_id": str(generator_details.get("profile_id") or "").strip() or None,
                    "opportunity_id": target_opportunity_id or None,
                    "evidence_revision_at_launch": str(generator_details.get("evidence_revision_at_launch") or "").strip() or None,
                }
            },
            "recommended_experiments": [
                f"Validate {title[:180]} against current baselines",
                "Define measurable success criteria, scope, and instrumentation",
                "Record outcome, counterexamples, and next action",
            ],
        },
        generator=generator,
        generator_details={k: v for k, v in generator_details.items() if k != "existing_plan_ids"},
    )
    db.add(plan)
    await db.flush()
    return str(plan.id)


async def _sync_profile_opportunities(profile: DomainResearchProfile, opportunities: list[dict[str, Any]]) -> None:
    automation_profile, effective_policy = resolve_domain_profile_automation_contract(
        automation_profile=getattr(profile, "automation_profile", None),
        automation_policy=getattr(profile, "automation_policy", None),
        current_snapshot=current_domain_profile_policy_snapshot(profile),
    )
    summary = dict(profile.latest_summary) if isinstance(profile.latest_summary, dict) else {}
    normalized = list_normalized_research_opportunities(opportunities)
    linked_ids = collect_research_opportunity_linked_ids(normalized)
    summary = build_autonomy_summary(
        raw_summary=summary,
        opportunities=normalized,
        automation_profile=automation_profile,
        effective_policy=effective_policy,
        sandbox_profile_id=profile.sandbox_profile_id,
        config_revision_key="profile_config_revision",
    )
    profile.latest_summary = summary
    profile.latest_note_ids = list(dict.fromkeys([*([str(v) for v in (profile.latest_note_ids or []) if str(v).strip()]), *linked_ids["note_ids"]]))[:20]
    profile.latest_experiment_plan_ids = linked_ids["plan_ids"][:20]
    profile.latest_validation_run_ids = linked_ids["run_ids"][:20]


@router.get("", response_model=DomainResearchProfileListResponse)
async def list_domain_research_profiles(
    status_filter: str | None = Query(None, alias="status"),
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    stmt = select(DomainResearchProfile).where(DomainResearchProfile.user_id == current_user.id)
    if str(status_filter or "").strip():
        stmt = stmt.where(DomainResearchProfile.status == str(status_filter).strip())
    stmt = stmt.order_by(desc(DomainResearchProfile.updated_at)).offset(offset).limit(limit)
    rows = list((await db.execute(stmt)).scalars().all())
    total_stmt = select(DomainResearchProfile.id).where(DomainResearchProfile.user_id == current_user.id)
    if str(status_filter or "").strip():
        total_stmt = total_stmt.where(DomainResearchProfile.status == str(status_filter).strip())
    total = len(list((await db.execute(total_stmt)).scalars().all()))
    return DomainResearchProfileListResponse(
        items=[await _profile_response(row, db) for row in rows],
        total=total,
    )


@router.post("", response_model=DomainResearchProfileResponse, status_code=status.HTTP_201_CREATED)
async def create_domain_research_profile(
    payload: DomainResearchProfileCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    await _validate_sandbox_profile(
        db,
        sandbox_profile_id=payload.sandbox_profile_id,
        track_type=payload.track_type,
    )
    automation_profile, automation_policy = resolve_domain_profile_automation_contract(
        automation_profile=payload.automation_profile,
        automation_policy=payload.automation_policy,
        explicit_updates=payload.model_dump(exclude_none=True),
    )
    profile = DomainResearchProfile(
        user_id=current_user.id,
        title=payload.title,
        domain=payload.domain,
        objective=payload.objective,
        customer_context=payload.customer_context,
        status="draft",
        source_scope=payload.source_scope,
        track_type=payload.track_type,
        research_mode=payload.research_mode,
        monitor_queries=payload.monitor_queries,
        repo_source_ids=[str(v) for v in (payload.repo_source_ids or [])] or None,
        benchmark_queries=payload.benchmark_queries,
        report_format=payload.report_format,
        scoring_policy=payload.scoring_policy,
        selection_policy=payload.selection_policy,
        validation_policy=build_domain_profile_compat_policy(automation_policy),
        automation_profile=automation_profile,
        automation_policy=automation_policy,
        sandbox_profile_id=payload.sandbox_profile_id,
        interval_minutes=payload.interval_minutes,
        persist_artifacts=payload.persist_artifacts,
        auto_launch_follow_up=bool(automation_policy.get("auto_launch_follow_up", payload.auto_launch_follow_up)),
        auto_create_experiment_plans=bool(automation_policy.get("auto_create_experiment_plans", payload.auto_create_experiment_plans)),
        confidence_threshold=float(automation_policy.get("confidence_threshold", payload.confidence_threshold)),
        max_documents=payload.max_documents,
        max_papers=payload.max_papers,
    )
    db.add(profile)
    await db.flush()
    if payload.start_immediately:
        await _create_profile_job(db=db, profile=profile, schedule_type="continuous", start_immediately=True)
        await db.refresh(profile)
        return await _profile_response(profile, db)
    await db.commit()
    await db.refresh(profile)
    return await _profile_response(profile, db)


@router.get("/{profile_id}", response_model=DomainResearchProfileResponse)
async def get_domain_research_profile(
    profile_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    profile = await _get_profile_or_404(db, profile_id, current_user.id)
    return await _profile_response(profile, db)


@router.patch("/{profile_id}", response_model=DomainResearchProfileResponse)
async def update_domain_research_profile(
    profile_id: UUID,
    payload: DomainResearchProfileUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    profile = await _get_profile_or_404(db, profile_id, current_user.id)
    updates = payload.model_dump(exclude_none=True)
    next_automation_profile, next_automation_policy = resolve_domain_profile_automation_contract(
        automation_profile=updates.get("automation_profile", getattr(profile, "automation_profile", None)),
        automation_policy=updates.get("automation_policy", getattr(profile, "automation_policy", None)),
        current_snapshot=current_domain_profile_policy_snapshot(profile),
        explicit_updates=updates,
    )
    await _validate_sandbox_profile(
        db,
        sandbox_profile_id=updates.get("sandbox_profile_id", profile.sandbox_profile_id),
        track_type=str(updates.get("track_type", profile.track_type) or "generic"),
    )
    updates["automation_profile"] = next_automation_profile
    updates["automation_policy"] = next_automation_policy
    updates["validation_policy"] = build_domain_profile_compat_policy(next_automation_policy)
    updates["auto_launch_follow_up"] = bool(next_automation_policy.get("auto_launch_follow_up", profile.auto_launch_follow_up))
    updates["auto_create_experiment_plans"] = bool(next_automation_policy.get("auto_create_experiment_plans", profile.auto_create_experiment_plans))
    updates["confidence_threshold"] = float(next_automation_policy.get("confidence_threshold", profile.confidence_threshold or 0.7))
    for key, value in updates.items():
        setattr(profile, key, value)
    await db.commit()
    await db.refresh(profile)
    return await _profile_response(profile, db)


@router.post("/{profile_id}/action", response_model=DomainResearchProfileResponse)
async def act_on_domain_research_profile(
    profile_id: UUID,
    payload: DomainResearchProfileActionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    profile = await _get_profile_or_404(db, profile_id, current_user.id)
    action = payload.action
    active_job = await db.get(AgentJob, profile.active_job_id) if profile.active_job_id else None

    if action == "start":
        await _create_profile_job(db=db, profile=profile, schedule_type="continuous", start_immediately=True)
    elif action == "resume":
        if active_job is not None:
            active_job.schedule_type = "continuous"
            active_job.config = {**(active_job.config or {}), "interval_minutes": int(profile.interval_minutes or 1440)}
            active_job.next_run_at = datetime.utcnow()
            profile.status = "running"
            profile.paused_at = None
            await db.commit()
            execute_agent_job_task.delay(str(active_job.id), str(profile.user_id))
        else:
            await _create_profile_job(db=db, profile=profile, schedule_type="continuous", start_immediately=True)
    elif action == "pause":
        profile.status = "paused"
        profile.paused_at = datetime.utcnow()
        if active_job is not None:
            active_job.schedule_type = None
            active_job.next_run_at = None
        await db.commit()
    elif action == "cancel":
        profile.status = "cancelled"
        if active_job is not None:
            active_job.schedule_type = None
            active_job.next_run_at = None
            if active_job.status in {
                AgentJobStatus.PENDING.value,
                AgentJobStatus.RUNNING.value,
                AgentJobStatus.PAUSED.value,
            }:
                active_job.status = AgentJobStatus.CANCELLED.value
        profile.active_job_id = None
        await db.commit()
    elif action == "run_now":
        await _create_profile_job(db=db, profile=profile, schedule_type="once", start_immediately=True)
    else:
        raise HTTPException(status_code=400, detail="Unsupported action")

    await db.refresh(profile)
    return await _profile_response(profile, db)


@router.post("/{profile_id}/opportunities/{opportunity_id}/action", response_model=DomainResearchProfileResponse)
async def act_on_domain_research_opportunity(
    profile_id: UUID,
    opportunity_id: str,
    payload: ResearchOpportunityActionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    profile = await _get_profile_or_404(db, profile_id, current_user.id)
    parent_job = await _maybe_resolve_profile_parent_job(db, profile)
    summary = profile.latest_summary if isinstance(profile.latest_summary, dict) else {}
    opportunities = list_normalized_research_opportunities(summary.get("opportunities") or summary.get("idea_candidates"))
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
            note_ids=[str(v) for v in (opportunity.get("source_note_ids") or []) if str(v).strip()] or [str(v) for v in (profile.latest_note_ids or []) if str(v).strip()],
            title=str(opportunity.get("title") or "Research opportunity"),
            hypothesis=str(opportunity.get("hypothesis") or opportunity.get("title") or ""),
            generator="domain_research_operator_action",
            generator_details={
                "origin": "domain_research_profile_action",
                "profile_id": str(profile.id),
                "opportunity_id": opportunity["opportunity_id"],
                "idea_title": str(opportunity.get("title") or ""),
                "supporting_evidence": opportunity.get("supporting_evidence") if isinstance(opportunity.get("supporting_evidence"), list) else [],
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
        parent_job = await _resolve_profile_parent_job(db, profile)
        _automation_profile, effective_policy = resolve_domain_profile_automation_contract(
            automation_profile=getattr(profile, "automation_profile", None),
            automation_policy=getattr(profile, "automation_policy", None),
            current_snapshot=current_domain_profile_policy_snapshot(profile),
        )
        materialization = await materialize_research_opportunity_experiment(
            db=db,
            parent_job=parent_job,
            owner_kind="profile",
            owner_id=str(profile.id),
            user_id=str(current_user.id),
            opportunity=opportunity,
            title=str(opportunity.get("title") or ""),
            hypothesis=str(opportunity.get("hypothesis") or opportunity.get("title") or ""),
            note_ids=[str(v) for v in (opportunity.get("source_note_ids") or []) if str(v).strip()] or [str(v) for v in (profile.latest_note_ids or []) if str(v).strip()],
            track_type=str(profile.track_type or "generic"),
            objective=profile.objective,
            validation_policy=effective_policy,
            sandbox_profile_id=profile.sandbox_profile_id,
            repo_source_ids=[str(v) for v in (profile.repo_source_ids or []) if str(v).strip()],
            benchmark_queries=[str(v) for v in (profile.benchmark_queries or []) if str(v).strip()],
            ensure_plan_ids=_ensure_plan_ids,
            profile_id=str(profile.id),
            originating_job_id=str(parent_job.id),
            start_immediately=False,
        )
        opportunity = apply_materialized_experiment_metadata(
            opportunity,
            owner_kind="profile",
            owner_id=str(profile.id),
            plan_ids=materialization["plan_ids"],
            run_id=materialization.get("run_id"),
            job_id=materialization.get("job_id"),
            validation_status=materialization.get("validation_status"),
            blocked_reason_code=materialization.get("blocked_reason_code"),
            materialized_at=now_iso,
        )
    elif action == "materialize_experiment":
        parent_job = await _resolve_profile_parent_job(db, profile)
        _automation_profile, effective_policy = resolve_domain_profile_automation_contract(
            automation_profile=getattr(profile, "automation_profile", None),
            automation_policy=getattr(profile, "automation_policy", None),
            current_snapshot=current_domain_profile_policy_snapshot(profile),
        )
        materialization = await materialize_research_opportunity_experiment(
            db=db,
            parent_job=parent_job,
            owner_kind="profile",
            owner_id=str(profile.id),
            user_id=str(current_user.id),
            opportunity=opportunity,
            title=str(opportunity.get("title") or ""),
            hypothesis=str(opportunity.get("hypothesis") or opportunity.get("title") or ""),
            note_ids=[str(v) for v in (opportunity.get("source_note_ids") or []) if str(v).strip()] or [str(v) for v in (profile.latest_note_ids or []) if str(v).strip()],
            track_type=str(profile.track_type or "generic"),
            objective=profile.objective,
            validation_policy=effective_policy,
            sandbox_profile_id=profile.sandbox_profile_id,
            repo_source_ids=[str(v) for v in (profile.repo_source_ids or []) if str(v).strip()],
            benchmark_queries=[str(v) for v in (profile.benchmark_queries or []) if str(v).strip()],
            ensure_plan_ids=_ensure_plan_ids,
            profile_id=str(profile.id),
            originating_job_id=str(parent_job.id),
            start_immediately=payload.start_immediately is not False,
        )
        opportunity = apply_materialized_experiment_metadata(
            opportunity,
            owner_kind="profile",
            owner_id=str(profile.id),
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
            opportunities[idx] = normalize_research_opportunity(opportunity)
            await _sync_profile_opportunities(profile, opportunities)
            await record_autonomy_decision_event(
                db,
                user_id=current_user.id,
                event_type="follow_up_launched",
                event_time=datetime.utcnow(),
                source_kind="domain_profile",
                source_id=str(profile.id),
                source_label=str(profile.title or "Domain profile").strip(),
                customer=str(profile.customer_context or "").strip() or None,
                decision_type="follow_up_launched",
                reason_code="existing_follow_up_job",
                status=str(opportunities[idx].get("stage") or "").strip() or None,
                severity="medium",
                actor_mode="operator",
                summary=f"{str(profile.title or 'Domain profile').strip()}: reused existing follow-up for {str(opportunity.get('title') or 'opportunity').strip()}",
                operator_note=operator_note,
                reason_label="Follow-up launched",
                scheduler_state=_extract_scheduler_state(parent_job),
                before_state=opportunity_before,
                after_state=opportunities[idx],
                deep_link={"target_tab": "domain", "params": {"tab": "domain"}, "label": "Open Domain Profiles"},
                metadata={"opportunity_id": opportunity_id},
            )
            await db.commit()
            await db.refresh(profile)
            return await _profile_response(profile, db)
        parent_job = await _resolve_profile_parent_job(db, profile)
        executor = AutonomousAgentExecutor()
        child_job = await executor._create_domain_research_follow_up_job(
            db=db,
            job=parent_job,
            domain=profile.domain,
            objective=profile.objective,
            customer_context=str(profile.customer_context or ""),
            track_type=str(profile.track_type or "generic"),
            source_scope=str(profile.source_scope or "kb_plus_arxiv"),
            top_idea=opportunity,
            docs=[],
            repo_documents=[],
            papers=[],
            repo_source_ids=[str(v) for v in (profile.repo_source_ids or []) if str(v).strip()],
            benchmark_queries=[str(v) for v in (profile.benchmark_queries or []) if str(v).strip()],
            automation_profile=profile.automation_profile,
            automation_policy=profile.automation_policy if isinstance(profile.automation_policy, dict) else {},
            sandbox_profile_id=profile.sandbox_profile_id,
            profile_id=str(profile.id),
        )
        if child_job is None:
            raise HTTPException(status_code=400, detail="Failed to launch follow-up job")
        opportunity["child_job_ids"] = list(dict.fromkeys([*([str(v) for v in (opportunity.get("child_job_ids") or []) if str(v).strip()]), str(child_job.id)]))[:8]
        opportunity["decision_state"] = "accepted"
        opportunity["stage"] = "validating"
        opportunity["follow_up_review_status"] = "approved_launch"
        execute_agent_job_task.delay(str(child_job.id), str(profile.user_id))
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
        parent_job = await _resolve_profile_parent_job(db, profile)
        refreshed_summary = profile.latest_summary if isinstance(profile.latest_summary, dict) else {}
        refreshed_rows = list_normalized_research_opportunities(
            refreshed_summary.get("opportunities") if isinstance(refreshed_summary.get("opportunities"), list) else refreshed_summary.get("idea_candidates")
        )
        idx, opportunity = _find_opportunity(refreshed_rows, opportunity_id)
        opportunities = refreshed_rows
    else:
        raise HTTPException(status_code=400, detail="Unsupported action")

    opportunities[idx] = normalize_research_opportunity(opportunity)
    await _sync_profile_opportunities(profile, opportunities)
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
        source_kind="domain_profile",
        source_id=str(profile.id),
        source_label=str(profile.title or "Domain profile").strip(),
        customer=str(profile.customer_context or "").strip() or None,
        decision_type=decision_event_type,
        reason_code=(
            "operator_relaunched_follow_up"
            if action == "relaunch_follow_up"
            else str(opportunities[idx].get("last_decision_reason_code") or opportunities[idx].get("last_blocked_reason_code") or action).strip() or None
        ),
        status=str(opportunities[idx].get("stage") or opportunities[idx].get("autonomy_state") or "").strip() or None,
        severity="high" if str(opportunities[idx].get("stage") or "").strip().lower() == "blocked" else "medium",
        actor_mode="operator",
        summary=f"{str(profile.title or 'Domain profile').strip()}: {action.replace('_', ' ')} {str(opportunity.get('title') or 'opportunity').strip()}",
        operator_note=operator_note,
        reason_label=str(decision_event_type).replace("_", " ").strip().capitalize(),
        scheduler_state=_extract_scheduler_state(parent_job),
        before_state=opportunity_before,
        after_state=opportunities[idx],
        deep_link={"target_tab": "domain", "params": {"tab": "domain"}, "label": "Open Domain Profiles"},
        metadata={"opportunity_id": opportunity_id},
    )
    await db.commit()
    await db.refresh(profile)
    return await _profile_response(profile, db)
