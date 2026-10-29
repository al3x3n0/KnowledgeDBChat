"""Advance a campaign by one step, from whatever state it is in.

The whole design is in that sentence. There is no long-running process holding
a campaign together: every call reads the current state, does the next thing
that is due, writes it down and returns. A scheduler calling this every few
minutes runs a week-long programme, and a machine that reboots halfway loses
nothing but the time it was off.

That makes idempotence the property to protect. Calling it twice in a row must
not launch the same work twice, so an item moves to `running` in the same
transaction that creates its job, and reconciliation keys on the job's terminal
status rather than on anything remembered in a process.

What it does per step:

  reconcile  finished jobs settle their item, and their findings may add work
  launch     one pending item becomes a job, budget permitting
  conclude   nothing pending and nothing running means the campaign is done

One item per step on purpose. A campaign that fans out ten jobs at once has
stopped being a line of enquiry and become a batch, and the reason to have a
campaign at all is that each result should be able to change what is asked
next.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence
from uuid import UUID

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.research_campaign import (
    CampaignItemStatus,
    CampaignStatus,
    ResearchCampaign,
    ResearchCampaignItem,
)
from app.services import agent_campaign_priority as priority
from app.services.agent_loop_policy import finding_identity
from app.utils.datetimes import age

TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled", "stopped"}
DEFAULT_MAX_JOBS = 10
# A campaign may not spawn unlimited work from one job's findings. Ten
# candidates in a report are a reading list, not a plan.
MAX_ITEMS_FROM_ONE_JOB = 5
_JOB_NAME_LENGTH = AgentJob.__table__.c.name.type.length
# A stage that has just completed may not have created the stage after it
# yet: completing and starting the next are two writes. For this long, a
# completed job with follow-up work it has not started is not "finished".
STAGE_HANDOVER_GRACE = timedelta(minutes=5)

WAITING_ON_JOB = "job"
WAITING_ON_APPROVAL = "approval"
WAITING_ON_PERSON = "person"


async def create_campaign(
    db: AsyncSession,
    *,
    user_id: Any,
    name: str,
    goal: str,
    items: Sequence[Mapping[str, Any]] = (),
    max_jobs: int = DEFAULT_MAX_JOBS,
    job_template: Optional[Mapping[str, Any]] = None,
    plan_approval: Optional[Mapping[str, Any]] = None,
) -> ResearchCampaign:
    """Start a campaign with the work already known about."""
    if not str(goal or "").strip():
        raise ValueError(
            "a campaign needs a goal: it is what completion is judged against"
        )
    policy = checked_plan_approval(plan_approval)

    campaign = ResearchCampaign(
        user_id=user_id,
        name=str(name or "campaign")[:300],
        goal=str(goal),
        status=CampaignStatus.ACTIVE,
        max_jobs=max(1, min(int(max_jobs or DEFAULT_MAX_JOBS), 500)),
        job_template=dict(job_template or {}),
        plan_approval=policy,
    )
    db.add(campaign)
    await db.flush()

    seeded = 0
    for item in items:
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        db.add(
            ResearchCampaignItem(
                campaign_id=campaign.id,
                title=title[:300],
                detail=str(item.get("detail") or "") or None,
                origin="seed",
            )
        )
        seeded += 1

    if not seeded:
        # A campaign with nothing pending is COMPLETED by its first `advance`
        # -- the `not pending` branch cannot tell "finished the work" from
        # "never had any" -- so it would report success having launched no
        # job at all. The goal is always a serviceable first question, and
        # findings from it spawn the rest, so start there rather than create
        # a campaign that is over before the beat task sees it.
        db.add(
            ResearchCampaignItem(
                campaign_id=campaign.id,
                title=str(goal).strip()[:300],
                detail=None,
                origin="goal",
            )
        )

    await db.flush()
    return campaign


async def _items(
    db: AsyncSession, campaign: ResearchCampaign, status: Optional[str] = None
) -> List[ResearchCampaignItem]:
    query = select(ResearchCampaignItem).where(
        ResearchCampaignItem.campaign_id == campaign.id
    )
    if status:
        query = query.where(ResearchCampaignItem.status == status)
    return list(
        (await db.execute(query.order_by(ResearchCampaignItem.created_at))).scalars()
    )


def _findings_of(job: AgentJob) -> List[Dict[str, Any]]:
    """What this job established, excluding evidence borrowed from its ancestors.

    The campaign walks every stage, so the original producer contributes the
    finding. Counting downstream copies or repeated calls again would inflate
    its evidence and feed one measurement back into discovery as several results.
    """
    results = job.results if isinstance(job.results, dict) else {}
    findings = results.get("findings")
    unique = {}
    for finding in findings if isinstance(findings, list) else []:
        if isinstance(finding, dict) and finding.get("inherited") is not True:
            unique.setdefault(finding_identity(finding), finding)
    return list(unique.values())


def _contract_of(job: AgentJob) -> Dict[str, Any]:
    results = job.results if isinstance(job.results, dict) else {}
    contract = results.get("goal_contract")
    return contract if isinstance(contract, dict) else {}


async def _run_of(db: AsyncSession, job: AgentJob) -> List[AgentJob]:
    """The job an item launched and every stage that followed from it.

    An item's job may propose a pipeline, and once someone approves it the
    work is done by the stages, not by the job the campaign started. Reading
    that one job alone would settle the item before its work had begun and
    conclude from none of what the stages found.
    """
    stages = (
        (
            await db.execute(
                select(AgentJob)
                .where(AgentJob.root_job_id == job.id, AgentJob.id != job.id)
                .order_by(AgentJob.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    return [job, *stages]


def _hands_over(job: AgentJob) -> bool:
    """True while a completed job may still be about to start its successor."""
    chain = job.chain_config if isinstance(job.chain_config, dict) else {}
    if not chain.get("child_jobs") or getattr(job, "chain_triggered", False):
        return False
    if str(job.status or "") != "completed":
        return False
    since = age(job.completed_at)
    return since is not None and since < STAGE_HANDOVER_GRACE


def _waiting_on(run: Sequence[AgentJob]) -> Optional[str]:
    """What an item's run is waiting for, or None when all of it has ended.

    A run paused for a person is named apart from one that is working: a
    campaign that has stopped until somebody approves a plan looks, from its
    counters, exactly like one whose job is still running.
    """
    waiting: Optional[str] = None
    for job in run:
        status = str(job.status or "")
        if status in TERMINAL_JOB_STATUSES:
            if _hands_over(job):
                waiting = waiting or WAITING_ON_JOB
            continue
        if status == AgentJobStatus.PAUSED.value:
            if str(job.current_phase or "") == "awaiting_approval":
                return WAITING_ON_APPROVAL
            waiting = WAITING_ON_PERSON
        elif waiting != WAITING_ON_PERSON:
            waiting = WAITING_ON_JOB
    return waiting


def _stage_name(job: AgentJob) -> str:
    config = job.config if isinstance(job.config, dict) else {}
    return str(config.get("pipeline_stage") or job.name or "")[:120]


def _ceilings() -> Dict[str, int]:
    from app.core.config import settings

    return {
        "max_stages": int(getattr(settings, "CAMPAIGN_PLAN_APPROVAL_MAX_STAGES", 4)),
        "max_minutes": int(
            getattr(settings, "CAMPAIGN_PLAN_APPROVAL_MAX_MINUTES", 120)
        ),
    }


def checked_plan_approval(
    policy: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, int]]:
    """The limits under which a campaign approves its own plans, or None.

    None (or an empty mapping) is "a person approves every plan". A limit
    over the deployment's ceiling is refused with the figure rather than cut
    down, because the person setting it should know they did not get it.
    """
    if not policy:
        return None
    ceilings = _ceilings()
    checked: Dict[str, int] = {}
    for key, ceiling in ceilings.items():
        try:
            value = int(policy.get(key))
        except (TypeError, ValueError):
            raise ValueError(f"plan_approval.{key} must be a whole number")
        if value < 1:
            raise ValueError(f"plan_approval.{key} must be at least 1")
        if value > ceiling:
            raise ValueError(
                f"plan_approval.{key} is {value}; this deployment allows at most "
                f"{ceiling} (CAMPAIGN_PLAN_APPROVAL_{key.upper()})"
            )
        checked[key] = value
    return checked


def _effective_limits(policy: Any) -> Optional[Dict[str, int]]:
    """The stored limits under today's ceilings. A ceiling an admin lowered
    after the campaign was made binds it; a stored value that is not a
    number is no permission at all."""
    if not isinstance(policy, dict) or not policy:
        return None
    limits: Dict[str, int] = {}
    for key, ceiling in _ceilings().items():
        try:
            value = int(policy.get(key))
        except (TypeError, ValueError):
            return None
        if value < 1:
            return None
        limits[key] = min(value, ceiling)
    return limits


def _held(job: AgentJob, reason: str) -> Dict[str, Any]:
    return {"job": str(job.id), "approved": False, "reason": reason}


#: ``approve(job, db, *, approver, note)``: how a campaign approves a plan.
#: Passed in by the caller (the scheduled task passes
#: ``plan_approval.approve_proposed_plan``) rather than imported here: the
#: decision machinery imports the finaliser, which reaches the planner,
#: which reads this module's budget, and importing it from here closed that
#: loop through thirteen modules.
PlanApprover = Callable[..., Awaitable[Any]]


async def _approve_own_plan(
    db: AsyncSession,
    campaign: ResearchCampaign,
    run: Sequence[AgentJob],
    approve: Optional[PlanApprover],
) -> Optional[Dict[str, Any]]:
    """Approve the plan this run's job proposed, if the campaign's limits allow.

    None when the campaign has no limits (a person approves, as before) or
    nothing in the run is waiting for approval. Otherwise what was decided
    and why. Only the gate in front of a plan the run proposed itself is
    the campaign's to pass: an approval a tool asked for mid-run, or a
    checkpoint the plan declares between its own stages, waits for a person.
    """
    limits = _effective_limits(campaign.plan_approval)
    if limits is None or approve is None:
        return None
    from app.services import agent_self_planned_pipeline
    from app.services.agent_runtime_finalizer import CHAIN_GATE_CHECKPOINT

    for job in run:
        if str(job.status or "") != AgentJobStatus.PAUSED.value:
            continue
        if str(job.current_phase or "") != "awaiting_approval":
            continue
        results = job.results if isinstance(job.results, dict) else {}
        checkpoint = results.get("approval_checkpoint")
        checkpoint = checkpoint if isinstance(checkpoint, dict) else {}
        proposal = agent_self_planned_pipeline.proposal_of(job)
        if checkpoint.get("checkpoint_type") != CHAIN_GATE_CHECKPOINT or not proposal:
            return _held(job, "this approval is not a plan the run proposed")

        stages = len(proposal.get("stages") or [])
        minutes = -(-int(proposal.get("estimated_seconds") or 0) // 60)
        room = int(campaign.max_jobs or 0) - int(campaign.jobs_launched or 0)
        if stages > limits["max_stages"]:
            return _held(
                job,
                f"the plan has {stages} stages and this campaign approves at "
                f"most {limits['max_stages']}",
            )
        if minutes > limits["max_minutes"]:
            return _held(
                job,
                f"the plan is estimated at {minutes} min and this campaign "
                f"approves at most {limits['max_minutes']}",
            )
        if stages > room:
            return _held(
                job,
                f"the plan has {stages} stages and the campaign has {room} "
                "job(s) left",
            )

        from app.models.user import User

        approver = await db.get(User, campaign.user_id)
        if approver is None or not approver.is_active:
            return _held(job, "the campaign's owner cannot approve anything")
        note = (
            f"Approved by campaign '{campaign.name}' within its limits "
            f"({limits['max_stages']} stages, {limits['max_minutes']} min): "
            f"{stages} stage(s), about {minutes} min."
        )
        await approve(job, db, approver=approver, note=note)
        logger.info(f"Campaign {campaign.id} approved job {job.id}'s plan: {note}")
        return {
            "job": str(job.id),
            "approved": True,
            "stages": stages,
            "minutes": minutes,
            "reason": note,
        }
    return None


async def _reconcile(
    db: AsyncSession,
    campaign: ResearchCampaign,
    approve: Optional[PlanApprover] = None,
) -> Dict[str, Any]:
    """Settle items whose run has finished, and harvest any work it revealed."""
    settled = 0
    discovered = 0
    waiting_on: Optional[str] = None
    plan_decisions: List[Dict[str, Any]] = []
    template = campaign.job_template if isinstance(campaign.job_template, dict) else {}
    spawn_from = template.get("spawn_items_from")
    spawn_from = (
        {str(x) for x in spawn_from} if isinstance(spawn_from, (list, tuple)) else set()
    )

    for item in await _items(db, campaign, CampaignItemStatus.RUNNING):
        if item.job_id is None:
            continue
        job = (
            await db.execute(select(AgentJob).where(AgentJob.id == item.job_id))
        ).scalar_one_or_none()
        if job is None:
            # The job is gone. The item cannot be waiting on it forever, and
            # pretending it succeeded would be worse than recording that it
            # cannot be settled.
            item.status = CampaignItemStatus.FAILED
            item.outcome = {"error": "the job for this item no longer exists"}
            settled += 1
            continue
        run = await _run_of(db, job)
        waiting = _waiting_on(run)
        if waiting == WAITING_ON_APPROVAL:
            decision = await _approve_own_plan(db, campaign, run, approve)
            if decision is not None:
                # Kept on the item, so the page can say who approved what,
                # or why the campaign left it for a person.
                item.outcome = {
                    **(item.outcome if isinstance(item.outcome, dict) else {}),
                    "plan_approval": decision,
                }
                plan_decisions.append(decision)
                if decision["approved"]:
                    waiting = _waiting_on(await _run_of(db, job))
        if waiting:
            waiting_on = waiting
            continue

        contract = _contract_of(job)
        findings = [finding for member in run for finding in _findings_of(member)]
        stages = run[1:]
        item.status = (
            CampaignItemStatus.DONE
            if str(job.status) == "completed"
            else CampaignItemStatus.FAILED
        )
        earlier = item.outcome if isinstance(item.outcome, dict) else {}
        item.outcome = {
            **(
                {"plan_approval": earlier["plan_approval"]}
                if "plan_approval" in earlier
                else {}
            ),
            "job_status": str(job.status or ""),
            "iterations": int(job.iteration or 0),
            "contract_satisfied": bool(contract.get("satisfied")),
            "contract_missing": (contract.get("missing") or [])[:6],
            "finding_counts": _counts(findings),
            "conclusion": str((job.results or {}).get("conclusion") or "")[:600],
        }
        if stages:
            # The stages are jobs this campaign caused, so they come out of
            # its budget, counted here because only now is the number known.
            campaign.jobs_launched = int(campaign.jobs_launched or 0) + len(stages)
            item.outcome["stages"] = [
                {
                    "stage": _stage_name(stage),
                    "job_status": str(stage.status or ""),
                    "contract_satisfied": bool(_contract_of(stage).get("satisfied")),
                }
                for stage in stages[:12]
            ]
        settled += 1

        if spawn_from:
            discovered += await _spawn_items(db, campaign, findings, spawn_from, item)

    return {
        "settled": settled,
        "discovered": discovered,
        "waiting_on": waiting_on,
        "plan_decisions": plan_decisions,
    }


def _counts(findings: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for finding in findings:
        name = str(finding.get("type") or "").strip() or "untyped"
        counts[name] = counts.get(name, 0) + 1
    return counts


async def _spawn_items(
    db: AsyncSession,
    campaign: ResearchCampaign,
    findings: Sequence[Mapping[str, Any]],
    spawn_from: set,
    parent: Optional[ResearchCampaignItem] = None,
) -> int:
    """Turn findings of the named types into further work, within a limit."""
    existing = {item.title for item in await _items(db, campaign)}
    added = 0
    for finding in findings:
        if added >= MAX_ITEMS_FROM_ONE_JOB:
            break
        if str(finding.get("type") or "") not in spawn_from:
            continue
        title = str(finding.get("title") or "").strip()[:300]
        if not title or title in existing:
            continue
        db.add(
            ResearchCampaignItem(
                campaign_id=campaign.id,
                title=title,
                detail=str(finding.get("content") or finding.get("example") or "")
                or None,
                origin="discovered",
                # Where this came from, so a line can be followed back and a
                # run of unproductive ancestors can be recognised as one.
                parent_item_id=parent.id if parent is not None else None,
                generation=(int(parent.generation or 0) + 1) if parent else 1,
            )
        )
        existing.add(title)
        added += 1
    if added:
        await db.flush()
    return added


async def _launch(
    db: AsyncSession,
    campaign: ResearchCampaign,
    item: ResearchCampaignItem,
    chose: Optional[Mapping[str, Any]] = None,
) -> AgentJob:
    """Create the job for one item, and mark the item running with it."""
    template = campaign.job_template if isinstance(campaign.job_template, dict) else {}
    config = dict(template.get("config") or {})
    config["campaign_id"] = str(campaign.id)
    config["campaign_item_id"] = str(item.id)

    goal = f"{campaign.goal}\n\n" f"This job's part of that: {item.title}" + (
        f"\n{item.detail}" if item.detail else ""
    )
    job = AgentJob(
        # The job's column is narrower than the campaign's name and the item's
        # title, and SQLite does not enforce a width: on Postgres a long
        # question failed every tick at launch.
        name=f"{campaign.name}: {item.title}"[:_JOB_NAME_LENGTH],
        goal=goal,
        job_type=str(template.get("job_type") or "research"),
        user_id=campaign.user_id,
        status=AgentJobStatus.PENDING.value,
        config=config,
        max_iterations=int(template.get("max_iterations") or 12),
        max_tool_calls=int(template.get("max_tool_calls") or 40),
        max_llm_calls=int(template.get("max_llm_calls") or 60),
        max_runtime_minutes=int(template.get("max_runtime_minutes") or 45),
    )
    db.add(job)
    await db.flush()

    # In the same transaction as the job: a step that created work without
    # recording it would launch the same item again on the next call.
    item.status = CampaignItemStatus.RUNNING
    item.job_id = job.id
    item.launched_at = datetime.utcnow()
    if chose:
        item.priority = float(chose.get("score") or 0.0)
        item.priority_reason = str(chose.get("reason") or "")[:400]
    campaign.jobs_launched = int(campaign.jobs_launched or 0) + 1
    await db.flush()
    return job


def _target_types(campaign: ResearchCampaign) -> List[str]:
    """The finding types this campaign counts as a result.

    Deliberately *not* defaulted to `spawn_items_from`. Those answer different
    questions -- what is worth looking into next, versus what would count as
    having found something -- and conflating them makes abandonment impossible:
    any parent that spawned a child produced findings of exactly the spawn
    types, so no line could ever read as cold.

    Leaving it undeclared is therefore how a campaign opts out of having its
    lines abandoned, which is the right default. A campaign that never said
    what it was looking for has not earned the right to give up on anything.
    """
    template = campaign.job_template if isinstance(campaign.job_template, dict) else {}
    declared = template.get("target_finding_types")
    if isinstance(declared, (list, tuple)):
        return [str(x) for x in declared]
    return []


def _view(
    item: ResearchCampaignItem, by_id: Mapping[str, ResearchCampaignItem]
) -> Dict[str, Any]:
    """What the scorer is allowed to see about one pending item."""
    parent = by_id.get(str(item.parent_item_id)) if item.parent_item_id else None
    siblings = 0
    if parent is not None:
        siblings = sum(
            1
            for other in by_id.values()
            if str(other.parent_item_id or "") == str(parent.id)
            and other.id != item.id
            and other.job_id is not None
        )
    return {
        "origin": item.origin,
        "generation": int(item.generation or 0),
        "parent_outcome": parent.outcome if parent is not None else None,
        "siblings_launched": siblings,
    }


def _ancestry(
    item: ResearchCampaignItem, by_id: Mapping[str, ResearchCampaignItem]
) -> List[Optional[Dict[str, Any]]]:
    """Outcomes of this item's ancestors, nearest first."""
    chain: List[Optional[Dict[str, Any]]] = []
    seen = {str(item.id)}
    current = by_id.get(str(item.parent_item_id)) if item.parent_item_id else None
    while current is not None and str(current.id) not in seen:
        seen.add(str(current.id))
        chain.append(current.outcome if isinstance(current.outcome, dict) else None)
        current = (
            by_id.get(str(current.parent_item_id)) if current.parent_item_id else None
        )
    return chain


def _recent_origins(items: Sequence[ResearchCampaignItem]) -> List[str]:
    """Origins of items in the order their jobs actually started.

    Launch order, not creation order: an item spawned early may run late, and
    the guard against a campaign chasing its own tail cares what ran.
    """
    launched = [item for item in items if item.launched_at is not None]
    launched.sort(key=lambda item: item.launched_at)
    return [str(item.origin or "seed") for item in launched]


async def _triage(
    db: AsyncSession,
    campaign: ResearchCampaign,
    items: Sequence[ResearchCampaignItem],
) -> Dict[str, Any]:
    """Score every pending item, and abandon the ones on lines gone cold.

    Both are written down: the score so a choice can be read afterwards, the
    drop with the reason for it, because a campaign that quietly stops doing
    something looks the same as one that finished it.
    """
    targets = _target_types(campaign)
    by_id = {str(item.id): item for item in items}
    dropped = 0

    for item in items:
        if item.status != CampaignItemStatus.PENDING:
            continue
        # Seeds are never dropped, whatever their line's record: they are what
        # a person actually asked for, and a cold line may be cold because the
        # questions were hard rather than wrong.
        if item.origin == "discovered" and priority.is_cold(
            _ancestry(item, by_id), targets
        ):
            item.status = CampaignItemStatus.DROPPED
            item.priority_reason = (
                f"line abandoned: {priority.COLD_RUN_LIMIT} consecutive ancestors "
                "settled without meeting a contract or finding anything wanted"
            )
            item.outcome = {"dropped": True, "reason": item.priority_reason}
            dropped += 1
            continue

        value, reason = priority.score(_view(item, by_id), targets)
        item.priority = value
        item.priority_reason = reason[:400]

    if dropped:
        await db.flush()
    return {"dropped": dropped, "targets": targets, "by_id": by_id}


async def _all_findings(
    db: AsyncSession, campaign: ResearchCampaign
) -> List[Dict[str, Any]]:
    """Every finding this campaign's jobs produced, with its item attached.

    Carrying the item title matters: a campaign's findings come from several
    jobs that each answered a different question, and a flat list of numbers
    with no indication of what each was measuring is not something a
    conclusion can be drawn from honestly.
    """
    collected: List[Dict[str, Any]] = []
    for item in await _items(db, campaign):
        if item.job_id is None:
            continue
        job = (
            await db.execute(select(AgentJob).where(AgentJob.id == item.job_id))
        ).scalar_one_or_none()
        if job is None:
            continue
        for member in await _run_of(db, job):
            for finding in _findings_of(member):
                enriched = dict(finding)
                enriched.setdefault("subject", item.title)
                collected.append(enriched)
    return collected


async def conclude_campaign(
    db: AsyncSession, campaign: ResearchCampaign
) -> Dict[str, Any]:
    """Answer the campaign's goal from what its jobs found, and store it.

    A campaign exists to settle a goal. Ending without saying what the answer
    turned out to be makes it a batch runner with extra steps -- and that is
    what it was: `conclusion` was declared on the model, returned by the API,
    and written by nothing, so every campaign that ever completed did so
    silently.

    Never raises. A campaign that cannot be concluded is still a campaign that
    finished, and losing the whole finalization over the summary would be the
    wrong trade.
    """
    from app.services.agent_run_synthesis_service import conclude, conclusion_line
    from app.services.llm_service import LLMService, load_user_llm_settings

    findings = await _all_findings(db, campaign)

    # The budget running out is something the caller knows and no individual
    # finding records. Saying it here is what keeps "we did not find out" from
    # being confused with "we found out that no".
    extra_gaps: List[str] = []
    if str(campaign.status) == CampaignStatus.EXHAUSTED:
        pending = len(await _items(db, campaign, CampaignItemStatus.PENDING))
        extra_gaps.append(
            f"The campaign ran out of budget after {campaign.jobs_launched} job(s) "
            f"with {pending} question(s) still unanswered, so this is partial."
        )

    result = await conclude(
        LLMService(),
        goal=str(campaign.goal or ""),
        findings=findings,
        db=db,
        snapshot_context={"campaign_id": str(campaign.id), "phase": "conclusion"},
        extra_gaps=extra_gaps,
        user_settings=await load_user_llm_settings(db, campaign.user_id),
        user_id=campaign.user_id,
    )

    campaign.conclusion = conclusion_line(result) or None
    campaign.conclusion_detail = result
    return result


async def advance(
    db: AsyncSession,
    campaign: ResearchCampaign,
    approve: Optional[PlanApprover] = None,
) -> Dict[str, Any]:
    """Do the next thing this campaign is due, and say what that was.

    Safe to call at any time and from anywhere: everything it needs is read
    from the database, and nothing is held between calls.
    """
    if str(campaign.status) != CampaignStatus.ACTIVE:
        return {
            "campaign": str(campaign.id),
            "action": "none",
            "status": campaign.status,
        }

    reconciled = await _reconcile(db, campaign, approve)

    all_items = await _items(db, campaign)
    triage = await _triage(db, campaign, all_items)

    running = [i for i in all_items if i.status == CampaignItemStatus.RUNNING]
    pending = [i for i in all_items if i.status == CampaignItemStatus.PENDING]

    action = "waiting"
    launched_job: Optional[AgentJob] = None
    chose: Optional[Dict[str, Any]] = None

    if running:
        # Nothing starts while something is running. The reason to have a
        # campaign rather than a batch is that each result can change what is
        # asked next, and a job launched before the previous one lands cannot
        # have been informed by it.
        action = "waiting"
    elif pending and int(campaign.jobs_launched or 0) < int(campaign.max_jobs or 0):
        chose = priority.choose(
            [_view(item, triage["by_id"]) for item in pending],
            target_types=triage["targets"],
            recent_origins=_recent_origins(all_items),
        )
        launched_job = await _launch(db, campaign, pending[chose["index"]], chose)
        action = "launched"
    elif not pending:
        campaign.status = CampaignStatus.COMPLETED
        campaign.completed_at = datetime.utcnow()
        action = "completed"
    else:
        # Out of budget with work still on the list. Said plainly, because a
        # campaign that stopped early looks exactly like one that finished
        # unless someone writes down which it was.
        campaign.status = CampaignStatus.EXHAUSTED
        campaign.completed_at = datetime.utcnow()
        action = "exhausted"

    if action in ("completed", "exhausted"):
        # The last thing a campaign does is say what it found out. Both endings
        # get one: "we ran out of budget" is a different answer from "we
        # finished", and neither is the same as no answer at all.
        await conclude_campaign(db, campaign)

    campaign.updated_at = datetime.utcnow()
    await db.flush()

    return {
        "campaign": str(campaign.id),
        "action": action,
        "status": campaign.status,
        # Why nothing moved, when nothing did: a job at work, a plan waiting
        # for approval, or a run that stopped to ask a person something.
        "waiting_on": reconciled["waiting_on"] if action == "waiting" else None,
        "settled": reconciled["settled"],
        "discovered": reconciled["discovered"],
        "plan_decisions": reconciled["plan_decisions"],
        "dropped": triage["dropped"],
        "launched_job": str(launched_job.id) if launched_job else None,
        "chose": (
            {
                "title": pending[chose["index"]].title,
                "score": chose["score"],
                "why": chose["reason"],
            }
            if chose
            else None
        ),
        "pending": len(pending) - (1 if launched_job else 0),
        "running": len(running) + (1 if launched_job else 0),
        "jobs_launched": int(campaign.jobs_launched or 0),
        "budget": int(campaign.max_jobs or 0),
        "conclusion": campaign.conclusion,
    }


async def _what_it_waits_on(
    db: AsyncSession, items: Sequence[ResearchCampaignItem]
) -> Optional[str]:
    """What the running item's run is waiting for, read without settling it."""
    for item in items:
        if item.status != CampaignItemStatus.RUNNING or item.job_id is None:
            continue
        job = (
            await db.execute(select(AgentJob).where(AgentJob.id == item.job_id))
        ).scalar_one_or_none()
        if job is not None:
            return _waiting_on(await _run_of(db, job))
    return None


async def remaining_jobs(db: AsyncSession, campaign_id: Any) -> Optional[int]:
    """How many more jobs this campaign may cause, or None if it is not one."""
    try:
        campaign = await db.get(ResearchCampaign, UUID(str(campaign_id)))
    except (ValueError, TypeError, AttributeError):
        return None
    if campaign is None:
        return None
    return max(0, int(campaign.max_jobs or 0) - int(campaign.jobs_launched or 0))


async def summarize(db: AsyncSession, campaign: ResearchCampaign) -> Dict[str, Any]:
    """What the campaign has done, for an operator or a report."""
    items = await _items(db, campaign)
    by_status: Dict[str, int] = {}
    for item in items:
        by_status[item.status] = by_status.get(item.status, 0) + 1

    settled = [item for item in items if item.status == CampaignItemStatus.DONE]
    with_contract = [
        item
        for item in settled
        if isinstance(item.outcome, dict) and item.outcome.get("contract_satisfied")
    ]
    return {
        "name": campaign.name,
        "status": campaign.status,
        "goal": campaign.goal,
        "waiting_on": await _what_it_waits_on(db, items),
        # The limits under which it approves its own plans; None means a
        # person approves every one.
        "plan_approval": (
            campaign.plan_approval if isinstance(campaign.plan_approval, dict) else None
        ),
        # Why a plan is waiting for a person although the campaign could
        # approve plans, when one is.
        "plan_held": next(
            (
                item.outcome["plan_approval"]["reason"]
                for item in items
                if item.status == CampaignItemStatus.RUNNING
                and isinstance(item.outcome, dict)
                and isinstance(item.outcome.get("plan_approval"), dict)
                and not item.outcome["plan_approval"].get("approved")
            ),
            None,
        ),
        "items": len(items),
        "by_status": by_status,
        "discovered_items": sum(1 for item in items if item.origin == "discovered"),
        "jobs_launched": int(campaign.jobs_launched or 0),
        "budget": int(campaign.max_jobs or 0),
        "items_meeting_contract": len(with_contract),
        "dropped_items": sum(
            1 for item in items if item.status == CampaignItemStatus.DROPPED
        ),
        "max_generation": max((int(i.generation or 0) for i in items), default=0),
        # What it would do next, in the order it would do it. An operator
        # should be able to disagree with a campaign before it spends a job.
        "next_up": [
            {
                "title": item.title,
                "origin": item.origin,
                "generation": int(item.generation or 0),
                "priority": item.priority,
                "why": item.priority_reason,
            }
            for item in sorted(
                (i for i in items if i.status == CampaignItemStatus.PENDING),
                key=lambda i: (-(i.priority or 0.0), i.created_at),
            )[:5]
        ],
    }


async def active_campaigns(db: AsyncSession, limit: int = 50) -> List[ResearchCampaign]:
    """Campaigns a scheduler should advance."""
    rows = await db.execute(
        select(ResearchCampaign)
        .where(ResearchCampaign.status == CampaignStatus.ACTIVE)
        .order_by(ResearchCampaign.updated_at)
        .limit(max(1, min(int(limit), 200)))
    )
    return list(rows.scalars())


async def advance_all(
    db: AsyncSession,
    limit: int = 50,
    approve: Optional[PlanApprover] = None,
) -> List[Dict[str, Any]]:
    """Advance every active campaign one step. One failing must not stop the rest.

    Each step is committed on its own. A failed flush leaves the session
    unusable until it is rolled back, so the next campaign used to fail too,
    and so did the handler, reading ``campaign.id`` from the broken session.
    The rollback expires every loaded object, which is why the ids are read
    first and each campaign is loaded afresh.
    """
    results: List[Dict[str, Any]] = []
    ids = [c.id for c in await active_campaigns(db, limit=limit)]
    for campaign_id in ids:
        try:
            campaign = await db.get(
                ResearchCampaign, campaign_id, populate_existing=True
            )
            if campaign is None:
                continue
            results.append(await advance(db, campaign, approve))
            await db.commit()
        except Exception as exc:
            await db.rollback()
            logger.warning(f"Could not advance campaign {campaign_id}: {exc}")
            results.append(
                {
                    "campaign": str(campaign_id),
                    "action": "error",
                    "error": str(exc)[:200],
                }
            )
    return results
