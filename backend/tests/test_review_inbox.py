"""`GET /review`: everything waiting on one person, read from real rows.

The decisions a person owes were on five pages under three doors, and tool
approvals were on the Admin page only. These check that each kind is counted
by the rule its own page uses, that nobody is shown what waits on someone
else, and that one section failing does not hide the rest.
"""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from app.api.endpoints import review as review_endpoint
from app.models.agent_job import AgentJob, AgentJobStatus
from app.models.artifact_draft import ArtifactDraft
from app.models.code_patch_proposal import CodePatchProposal
from app.models.research_inbox import ResearchInboxItem
from app.models.sandbox_skill import SandboxSkill
from app.models.tool_audit import ToolExecutionAudit
from app.modules.review.application import review_inbox
from app.modules.review.application.review_inbox import (
    SECTIONS,
    SHOWN_PER_SECTION,
    build_review_inbox,
)

pytestmark = pytest.mark.unit


def _blocked_job(user, **fields):
    values = dict(
        name="attention-survey: writeup",
        goal="Write up the survey",
        job_type="research",
        user_id=user.id,
        status=AgentJobStatus.PAUSED.value,
        current_phase="blocked_needs_input",
        created_at=datetime.utcnow() - timedelta(days=2),
        last_activity_at=datetime.utcnow() - timedelta(days=2),
        results={
            "blocked": {
                "reason": "3 consecutive rounds produced no new findings",
                "missing": ["finding_type:synthesis_document"],
                "resumable": True,
            }
        },
    )
    values.update(fields)
    return AgentJob(**values)


def _approval(user, **fields):
    values = dict(
        user_id=user.id,
        tool_name="delete_document",
        status="requires_approval",
        approval_required=True,
        approval_mode="owner_and_admin",
        approval_status="pending_owner",
    )
    values.update(fields)
    return ToolExecutionAudit(**values)


def _draft(user, status="in_review", title="Quarterly deck"):
    return ArtifactDraft(
        user_id=user.id,
        artifact_type="presentation",
        title=title,
        status=status,
        draft_payload={},
    )


def _patch(user, status="proposed"):
    return CodePatchProposal(
        user_id=user.id,
        title="Fix the off-by-one",
        diff_unified="--- a\n+++ b\n",
        status=status,
    )


def _inbox(user, status="new", n=0):
    return ResearchInboxItem(
        user_id=user.id,
        item_type="arxiv",
        item_key=f"2401.{n:05d}-{uuid4().hex[:6]}",
        title=f"Paper {n}",
        status=status,
        discovered_at=datetime.utcnow() - timedelta(hours=n),
    )


def _skill(user, origin="agent", status="draft"):
    return SandboxSkill(
        user_id=user.id,
        slug=f"s{uuid4().hex[:8]}",
        name="Count symbols",
        origin=origin,
        status=status,
    )


async def _sections(db, user):
    return {s.key: s for s in await build_review_inbox(db, user)}


async def test_nothing_waiting_is_six_empty_sections(db_session, test_user):
    sections = await build_review_inbox(db_session, test_user)

    assert [s.key for s in sections] == [spec[0] for spec in SECTIONS]
    assert all(s.count == 0 and s.items == [] and s.error is None for s in sections)


async def test_each_kind_of_waiting_item_is_counted_where_it_belongs(
    db_session, test_user
):
    db_session.add_all(
        [
            _blocked_job(test_user),
            _approval(test_user),
            _draft(test_user),
            _patch(test_user),
            _inbox(test_user),
            _skill(test_user),
        ]
    )
    await db_session.commit()

    sections = await _sections(db_session, test_user)

    assert {key: s.count for key, s in sections.items()} == {
        "runs": 1,
        "tool_approvals": 1,
        "patches": 1,
        "drafts": 1,
        "skills": 1,
        "inbox": 1,
    }
    run = sections["runs"].items[0]
    assert run.kind == "blocked_run"
    assert "no new findings" in (run.detail or "")
    assert run.link.startswith("/autonomous-agents?job=")
    assert sections["tool_approvals"].items[0].title == "delete_document"


async def test_what_is_already_decided_is_not_waiting(db_session, test_user):
    db_session.add_all(
        [
            _blocked_job(
                test_user,
                status=AgentJobStatus.COMPLETED.value,
                current_phase="completed",
                results={},
            ),
            _approval(test_user, approval_status="approved"),
            _approval(test_user, approval_status="rejected"),
            _approval(test_user, approval_required=False, approval_status=None),
            _draft(test_user, status="draft"),
            _draft(test_user, status="approved"),
            _patch(test_user, status="applied"),
            _inbox(test_user, status="accepted"),
            _skill(test_user, origin="manual"),
            _skill(test_user, status="active"),
        ]
    )
    await db_session.commit()

    sections = await build_review_inbox(db_session, test_user)

    assert sum(s.count for s in sections) == 0


async def test_nobody_is_shown_what_waits_on_someone_else(
    db_session, test_user, admin_user
):
    other = await _someone_else(db_session)
    db_session.add_all(
        [
            _blocked_job(other),
            _approval(other),
            _draft(other),
            _patch(other),
            _inbox(other),
            _skill(other),
        ]
    )
    await db_session.commit()

    mine = await build_review_inbox(db_session, test_user)

    assert sum(s.count for s in mine) == 0


async def _someone_else(db):
    from app.services.auth_service import AuthService

    return await AuthService().create_user(
        username="someone_else",
        email="else@example.com",
        password="otherpassword123",
        full_name="Someone Else",
        db=db,
    )


async def test_an_approval_waits_on_each_person_who_still_owes_one(
    db_session, test_user, admin_user
):
    now = datetime.utcnow()
    db_session.add_all(
        [
            # Nobody has approved: both the owner and an admin are asked.
            _approval(test_user, tool_name="both"),
            # The owner has; only the admin's half is outstanding.
            _approval(
                test_user,
                tool_name="admin_only",
                owner_approved_at=now,
                owner_approved_by=test_user.id,
                approval_status="pending_admin",
            ),
            # Legacy mode: either of them settles it.
            _approval(
                test_user,
                tool_name="either",
                approval_mode="owner_or_admin",
                approval_status="pending",
            ),
        ]
    )
    await db_session.commit()

    owner = (await _sections(db_session, test_user))["tool_approvals"]
    admin = (await _sections(db_session, admin_user))["tool_approvals"]

    assert sorted(i.title for i in owner.items) == ["both", "either"]
    assert sorted(i.title for i in admin.items) == ["admin_only", "both", "either"]


async def test_a_section_names_its_oldest_items_and_counts_them_all(
    db_session, test_user
):
    db_session.add_all([_inbox(test_user, n=n) for n in range(SHOWN_PER_SECTION + 3)])
    await db_session.commit()

    inbox = (await _sections(db_session, test_user))["inbox"]

    assert inbox.count == SHOWN_PER_SECTION + 3
    assert len(inbox.items) == SHOWN_PER_SECTION
    # Oldest first: what has waited longest is what to look at.
    assert inbox.items[0].title == f"Paper {SHOWN_PER_SECTION + 2}"
    assert inbox.oldest == inbox.items[0].waiting_since


async def test_one_section_failing_does_not_hide_the_others(
    db_session, test_user, monkeypatch
):
    db_session.add_all([_patch(test_user), _draft(test_user)])
    await db_session.commit()

    async def broken(db, user, section):
        raise RuntimeError("the queue could not be composed")

    specs = [
        (key, title, text, link, broken if key == "runs" else read)
        for key, title, text, link, read in SECTIONS
    ]
    monkeypatch.setattr(review_inbox, "SECTIONS", specs)

    sections = await _sections(db_session, test_user)

    assert sections["runs"].error
    assert sections["runs"].title == "Runs"
    assert sections["patches"].count == 1 and sections["patches"].error is None
    assert sections["drafts"].count == 1


async def test_the_route_totals_what_the_sections_count(db_session, test_user):
    db_session.add_all([_patch(test_user), _draft(test_user), _inbox(test_user)])
    await db_session.commit()

    response = await review_endpoint.get_review_inbox(
        current_user=test_user, db=db_session
    )

    assert response.total == 3
    assert [s.key for s in response.sections] == [spec[0] for spec in SECTIONS]
    patches = next(s for s in response.sections if s.key == "patches")
    assert patches.items[0].title == "Fix the off-by-one"
    assert patches.link == "/patch-prs"
