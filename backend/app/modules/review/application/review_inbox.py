"""Everything waiting on one person, in one answer.

A decision a person owes the lab lived on five pages under three doors: runs
that stopped for sign-off (Control Plane), drafts in review and proposed
patches (Synthesis), the research inbox (Library), skills a run proposed
(Tools) -- and tool approvals on the Admin page only, so someone who was not
an admin had nowhere to approve an action waiting on *them*.

This composes them and owns none of them. Each section is read the way its
own page reads it, and for runs that means the operator queue's composer
itself, so the two cannot disagree about what a queue row is. A section that
cannot be read says so and the others still answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, List, Optional

from loguru import logger
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.artifact_draft import ArtifactDraft
from app.models.code_patch_proposal import CodePatchProposal
from app.models.research_inbox import ResearchInboxItem
from app.models.sandbox_skill import SandboxSkill
from app.models.tool_audit import ToolExecutionAudit
from app.models.user import User
from app.modules.autonomy.application.checkpoint_queue_builder import (
    build_checkpoint_queue_items,
)
from app.modules.autonomy.application.checkpoint_queue_candidates import load_queue_jobs
from app.services.auth_service import is_admin

#: How many of a section's items are named; the count covers all of them.
SHOWN_PER_SECTION = 5
#: Queue rows about one run. The queue also carries monitor and portfolio
#: rows, which are standing configuration and have their own page.
RUN_ITEM_TYPES = frozenset(
    {"approval_checkpoint", "blocked_run", "contract_unmet", "job_recovery"}
)
#: Approvals are decided in Python (two people may each owe one), so at most
#: this many undecided records are read.
MAX_APPROVALS_READ = 500


@dataclass
class ReviewItem:
    id: str
    title: str
    detail: Optional[str] = None
    waiting_since: Optional[datetime] = None
    #: Where in the application this one item is handled, when it has a place.
    link: Optional[str] = None
    kind: Optional[str] = None


@dataclass
class ReviewSection:
    key: str
    title: str
    #: What a person does here, in a sentence.
    description: str
    #: The page that owns these.
    link: str
    count: int = 0
    oldest: Optional[datetime] = None
    items: List[ReviewItem] = field(default_factory=list)
    #: Set when the section could not be read. A count of 0 beside it means
    #: "unknown", not "nothing waiting".
    error: Optional[str] = None


def _take(section: ReviewSection, items: List[ReviewItem]) -> None:
    """Fill a section from every waiting item, oldest first."""
    dated = [i.waiting_since for i in items if i.waiting_since is not None]
    section.count = len(items)
    section.oldest = min(dated) if dated else None
    section.items = sorted(
        items, key=lambda i: (i.waiting_since is None, i.waiting_since or 0)
    )[:SHOWN_PER_SECTION]


async def _runs(db: AsyncSession, user: User, section: ReviewSection) -> None:
    jobs = await load_queue_jobs(db, user.id)
    rows = [
        row
        for row in build_checkpoint_queue_items(jobs, [], monitor_health_rows=[])
        if str(row.item_type or "") in RUN_ITEM_TYPES
    ]
    _take(
        section,
        [
            ReviewItem(
                id=str(row.job_id or row.queue_key),
                title=str(row.title or row.job_name or "Run"),
                detail=row.summary or row.reason_label,
                waiting_since=row.created_at,
                link=(f"/autonomous-agents?job={row.job_id}" if row.job_id else None),
                kind=str(row.item_type),
            )
            for row in rows
        ],
    )


def approval_waits_on(row: ToolExecutionAudit, user: User) -> bool:
    """Whether this undecided approval needs *this* person.

    `owner_and_admin` wants one of each; the legacy mode wants either. An
    owner who has already approved is not asked again while the admin's half
    is outstanding.
    """
    if not row.approval_required:
        return False
    if str(row.approval_status or "") in {"approved", "rejected"}:
        return False
    owner = str(row.user_id) == str(user.id)
    admin = is_admin(user)
    if str(row.approval_mode or "").strip().lower() == "owner_or_admin":
        return owner or admin
    return (owner and row.owner_approved_at is None) or (
        admin and row.admin_approved_at is None
    )


async def _tool_approvals(db: AsyncSession, user: User, section: ReviewSection) -> None:
    query = (
        select(ToolExecutionAudit)
        .where(
            ToolExecutionAudit.approval_required.is_(True),
            or_(
                ToolExecutionAudit.approval_status.is_(None),
                ToolExecutionAudit.approval_status.notin_(["approved", "rejected"]),
            ),
        )
        .order_by(ToolExecutionAudit.created_at)
        .limit(MAX_APPROVALS_READ)
    )
    if not is_admin(user):
        query = query.where(ToolExecutionAudit.user_id == user.id)
    rows = (await db.execute(query)).scalars().all()
    _take(
        section,
        [
            ReviewItem(
                id=str(row.id),
                title=str(row.tool_name),
                detail=_approval_detail(row, user),
                waiting_since=row.created_at,
                kind="tool_approval",
            )
            for row in rows
            if approval_waits_on(row, user)
        ],
    )


def _approval_detail(row: ToolExecutionAudit, user: User) -> str:
    whose = "yours" if str(row.user_id) == str(user.id) else "another user's"
    return f"Requested in {whose} session"


async def _counted(
    db: AsyncSession,
    section: ReviewSection,
    model: Any,
    conditions: List[Any],
    when: Any,
    item: Callable[[Any], ReviewItem],
) -> None:
    """A section whose waiting items are rows matching `conditions`."""
    section.count = int(
        await db.scalar(select(func.count()).select_from(model).where(*conditions)) or 0
    )
    if section.count:
        rows = (
            (
                await db.execute(
                    select(model)
                    .where(*conditions)
                    .order_by(when)
                    .limit(SHOWN_PER_SECTION)
                )
            )
            .scalars()
            .all()
        )
        section.items = [item(row) for row in rows]
        section.oldest = section.items[0].waiting_since if section.items else None


async def _drafts(db: AsyncSession, user: User, section: ReviewSection) -> None:
    await _counted(
        db,
        section,
        ArtifactDraft,
        [ArtifactDraft.user_id == user.id, ArtifactDraft.status == "in_review"],
        ArtifactDraft.created_at,
        lambda row: ReviewItem(
            id=str(row.id),
            title=str(row.title),
            waiting_since=row.created_at,
            link="/artifact-drafts",
            kind="draft",
        ),
    )


async def _patches(db: AsyncSession, user: User, section: ReviewSection) -> None:
    await _counted(
        db,
        section,
        CodePatchProposal,
        [CodePatchProposal.user_id == user.id, CodePatchProposal.status == "proposed"],
        CodePatchProposal.created_at,
        lambda row: ReviewItem(
            id=str(row.id),
            title=str(row.title),
            detail=row.summary,
            waiting_since=row.created_at,
            link="/patch-prs",
            kind="patch",
        ),
    )


async def _inbox(db: AsyncSession, user: User, section: ReviewSection) -> None:
    await _counted(
        db,
        section,
        ResearchInboxItem,
        [ResearchInboxItem.user_id == user.id, ResearchInboxItem.status == "new"],
        ResearchInboxItem.discovered_at,
        lambda row: ReviewItem(
            id=str(row.id),
            title=str(row.title),
            waiting_since=row.discovered_at,
            link="/research/inbox",
            kind=str(row.item_type or "inbox_item"),
        ),
    )


async def _skills(db: AsyncSession, user: User, section: ReviewSection) -> None:
    await _counted(
        db,
        section,
        SandboxSkill,
        [
            SandboxSkill.user_id == user.id,
            SandboxSkill.status == "draft",
            SandboxSkill.origin == "agent",
        ],
        SandboxSkill.created_at,
        lambda row: ReviewItem(
            id=str(row.id),
            title=str(row.name),
            waiting_since=row.created_at,
            link="/tools",
            kind="skill",
        ),
    )


Reader = Callable[[AsyncSession, User, ReviewSection], Awaitable[None]]

#: In the order a person should look: what is blocking work comes first.
#: (key, title, what a person does there, the page that owns it, its reader)
SECTIONS: List[tuple[str, str, str, str, Reader]] = [
    (
        "runs",
        "Runs",
        "Runs that stopped for sign-off, are blocked, or finished without "
        "meeting their contract.",
        "/agent-control-plane",
        _runs,
    ),
    (
        "tool_approvals",
        "Tool approvals",
        "Actions an agent or chat asked to take that need your approval "
        "before they run.",
        "/review",
        _tool_approvals,
    ),
    (
        "patches",
        "Proposed patches",
        "Code changes a run proposed. Nothing is applied until you accept one.",
        "/patch-prs",
        _patches,
    ),
    (
        "drafts",
        "Drafts in review",
        "Generated documents waiting for approval before they are published.",
        "/artifact-drafts",
        _drafts,
    ),
    (
        "skills",
        "Skills a run proposed",
        "Sandbox skills written by a run. None is offered to any run until "
        "you activate it.",
        "/tools",
        _skills,
    ),
    (
        "inbox",
        "Research inbox",
        "Papers and documents the monitors surfaced, waiting to be kept or "
        "rejected.",
        "/research/inbox",
        _inbox,
    ),
]


async def build_review_inbox(db: AsyncSession, user: User) -> List[ReviewSection]:
    """Every section, each read on its own.

    A savepoint around each: on Postgres one failed statement poisons the
    transaction, and the sections after it would fail for a reason that was
    not theirs.
    """
    sections: List[ReviewSection] = []
    for key, title, description, link, read in SECTIONS:
        section = ReviewSection(
            key=key, title=title, description=description, link=link
        )
        try:
            async with db.begin_nested():
                await read(db, user, section)
        except Exception as exc:  # noqa: BLE001 - reported on the section
            logger.warning(f"Review section {key} could not be read: {exc}")
            section = ReviewSection(
                key=key,
                title=title,
                description=description,
                link=link,
                error="This section could not be read.",
            )
        sections.append(section)
    return sections


__all__ = [
    "ReviewItem",
    "ReviewSection",
    "approval_waits_on",
    "build_review_inbox",
]
