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

from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence

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

TERMINAL_JOB_STATUSES = {"completed", "failed", "cancelled", "stopped"}
DEFAULT_MAX_JOBS = 10
# A campaign may not spawn unlimited work from one job's findings. Ten
# candidates in a report are a reading list, not a plan.
MAX_ITEMS_FROM_ONE_JOB = 5


async def create_campaign(
    db: AsyncSession,
    *,
    user_id: Any,
    name: str,
    goal: str,
    items: Sequence[Mapping[str, Any]] = (),
    max_jobs: int = DEFAULT_MAX_JOBS,
    job_template: Optional[Mapping[str, Any]] = None,
) -> ResearchCampaign:
    """Start a campaign with the work already known about."""
    if not str(goal or "").strip():
        raise ValueError(
            "a campaign needs a goal: it is what completion is judged against"
        )

    campaign = ResearchCampaign(
        user_id=user_id,
        name=str(name or "campaign")[:300],
        goal=str(goal),
        status=CampaignStatus.ACTIVE,
        max_jobs=max(1, min(int(max_jobs or DEFAULT_MAX_JOBS), 500)),
        job_template=dict(job_template or {}),
    )
    db.add(campaign)
    await db.flush()

    for item in items:
        db.add(
            ResearchCampaignItem(
                campaign_id=campaign.id,
                title=str(item.get("title") or "")[:300] or "untitled",
                detail=str(item.get("detail") or "") or None,
                origin="seed",
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
    results = job.results if isinstance(job.results, dict) else {}
    findings = results.get("findings")
    return (
        [f for f in findings if isinstance(f, dict)]
        if isinstance(findings, list)
        else []
    )


def _contract_of(job: AgentJob) -> Dict[str, Any]:
    results = job.results if isinstance(job.results, dict) else {}
    contract = results.get("goal_contract")
    return contract if isinstance(contract, dict) else {}


async def _reconcile(db: AsyncSession, campaign: ResearchCampaign) -> Dict[str, Any]:
    """Settle items whose job has finished, and harvest any work it revealed."""
    settled = 0
    discovered = 0
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
        if str(job.status or "") not in TERMINAL_JOB_STATUSES:
            continue

        contract = _contract_of(job)
        findings = _findings_of(job)
        item.status = (
            CampaignItemStatus.DONE
            if str(job.status) == "completed"
            else CampaignItemStatus.FAILED
        )
        item.outcome = {
            "job_status": str(job.status or ""),
            "iterations": int(job.iteration or 0),
            "contract_satisfied": bool(contract.get("satisfied")),
            "contract_missing": (contract.get("missing") or [])[:6],
            "finding_counts": _counts(findings),
            "conclusion": str((job.results or {}).get("conclusion") or "")[:600],
        }
        settled += 1

        if spawn_from:
            discovered += await _spawn_items(db, campaign, findings, spawn_from, item)

    return {"settled": settled, "discovered": discovered}


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
        name=f"{campaign.name}: {item.title}"[:300],
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


async def advance(db: AsyncSession, campaign: ResearchCampaign) -> Dict[str, Any]:
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

    reconciled = await _reconcile(db, campaign)

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

    campaign.updated_at = datetime.utcnow()
    await db.flush()

    return {
        "campaign": str(campaign.id),
        "action": action,
        "status": campaign.status,
        "settled": reconciled["settled"],
        "discovered": reconciled["discovered"],
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
    }


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


async def advance_all(db: AsyncSession, limit: int = 50) -> List[Dict[str, Any]]:
    """Advance every active campaign one step. One failing must not stop the rest."""
    results: List[Dict[str, Any]] = []
    for campaign in await active_campaigns(db, limit=limit):
        try:
            results.append(await advance(db, campaign))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Could not advance campaign {campaign.id}: {exc}")
            results.append(
                {
                    "campaign": str(campaign.id),
                    "action": "error",
                    "error": str(exc)[:200],
                }
            )
    return results
