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
            discovered += await _spawn_items(db, campaign, findings, spawn_from)

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
            )
        )
        existing.add(title)
        added += 1
    if added:
        await db.flush()
    return added


async def _launch(
    db: AsyncSession, campaign: ResearchCampaign, item: ResearchCampaignItem
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
    campaign.jobs_launched = int(campaign.jobs_launched or 0) + 1
    await db.flush()
    return job


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

    running = await _items(db, campaign, CampaignItemStatus.RUNNING)
    pending = await _items(db, campaign, CampaignItemStatus.PENDING)

    action = "waiting"
    launched_job: Optional[AgentJob] = None

    if running:
        # Nothing starts while something is running. The reason to have a
        # campaign rather than a batch is that each result can change what is
        # asked next, and a job launched before the previous one lands cannot
        # have been informed by it.
        action = "waiting"
    elif pending and int(campaign.jobs_launched or 0) < int(campaign.max_jobs or 0):
        launched_job = await _launch(db, campaign, pending[0])
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
        "launched_job": str(launched_job.id) if launched_job else None,
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
