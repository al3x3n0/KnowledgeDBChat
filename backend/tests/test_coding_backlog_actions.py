"""What each operator action does to a coding backlog item.

The action route was thirteen branches in one 650-line handler with no test.
These run the route against a database and pin what an operator sees: what is
refused and with which status, what the item and its slice say afterwards,
and which jobs are created and queued. They were written against the handler
as it stood, before any of it moved, so a move can be checked against them.

The only seam is the task's `.delay`: jobs are really created and really
handed to the dispatcher, and what reaches the broker is recorded.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import select

from app.api.endpoints import coding_backlog
from app.models.agent_job import AgentJob
from app.models.code_patch_proposal import CodePatchProposal
from app.models.coding_backlog import CodingBacklogItem
from app.models.patch_pr import PatchPR
from app.schemas.coding_backlog import CodingBacklogItemActionRequest
from app.services.auth_service import AuthService

SLICE_ACTIONS = [
    "apply_override",
    "create_patch_pr",
    "keep_proposal_only",
    "relaunch_slice",
    "skip_slice",
]


@pytest.fixture(autouse=True)
def queued(monkeypatch):
    """Job ids handed to the broker, in order."""
    from app.tasks.agent_job_tasks import execute_agent_job_task

    sent = []
    monkeypatch.setattr(
        execute_agent_job_task, "delay", lambda job_id, user_id: sent.append(job_id)
    )
    return sent


async def _person(db, name):
    return await AuthService().create_user(
        username=name,
        email=f"{name}@example.com",
        password="testpassword123",
        full_name=name,
        db=db,
    )


def _slice(slice_id="s1", **fields):
    row = {
        "slice_id": slice_id,
        "title": f"Slice {slice_id}",
        "goal": f"Fix {slice_id}",
        "status": "blocked",
        "awaiting_operator_action": True,
        "allowed_slice_actions": list(SLICE_ACTIONS),
    }
    row.update(fields)
    return row


async def _item(db, owner, *slices, **fields):
    item = CodingBacklogItem(
        id=uuid4(),
        user_id=owner.id,
        title="Parser crash",
        portfolio_goal="Stop the parser crashing on empty input",
        status=fields.pop("status", "draft"),
        decomposition={"planned_slices": list(slices)} if slices else None,
        **fields,
    )
    db.add(item)
    await db.commit()
    return item


async def _proposal(db, owner):
    proposal = CodePatchProposal(
        id=uuid4(), user_id=owner.id, title="Guard empty input", diff_unified="--- a\n"
    )
    db.add(proposal)
    await db.commit()
    return proposal


async def _act(db, user, item, action, **fields):
    return await coding_backlog.act_on_coding_backlog_item(
        item_id=item.id if hasattr(item, "id") else item,
        payload=CodingBacklogItemActionRequest(action=action, **fields),
        current_user=user,
        db=db,
    )


async def _refused(db, user, item, action, **fields):
    with pytest.raises(HTTPException) as refusal:
        await _act(db, user, item, action, **fields)
    return refusal.value.status_code, refusal.value.detail


def _slice_of(response, slice_id="s1"):
    return next(
        row
        for row in response.decomposition["planned_slices"]
        if row["slice_id"] == slice_id
    )


def _timeline_actions(response):
    return [row.get("action") for row in response.decomposition["backlog_timeline"]]


async def _jobs(db):
    return (await db.execute(select(AgentJob))).scalars().all()


# --- what is refused -------------------------------------------------------


class TestRefusals:
    @pytest.mark.asyncio
    async def test_an_action_that_does_not_exist(self, db_session, test_user):
        item = await _item(db_session, test_user)

        assert await _refused(db_session, test_user, item, "explode") == (
            400,
            "Unsupported action",
        )

    @pytest.mark.asyncio
    async def test_an_item_the_caller_may_not_see(self, db_session, test_user):
        stranger = await _person(db_session, "stranger")
        item = await _item(db_session, stranger)

        assert (await _refused(db_session, test_user, item, "pause"))[0] == 404

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", ["cancel", "close", *SLICE_ACTIONS])
    async def test_a_collaborator_cannot_do_what_only_the_owner_may(
        self, db_session, test_user, action
    ):
        owner = await _person(db_session, "owner")
        item = await _item(
            db_session,
            owner,
            _slice(),
            visibility="shared",
            shared_with_user_ids=[str(test_user.id)],
        )

        status, detail = await _refused(
            db_session,
            test_user,
            item,
            action,
            slice_id="s1",
            closure_reason="duplicate",
        )

        assert (status, detail) == (
            403,
            "Only the backlog owner can perform this action",
        )

    @pytest.mark.asyncio
    async def test_a_collaborator_may_pause(self, db_session, test_user):
        owner = await _person(db_session, "owner")
        item = await _item(
            db_session,
            owner,
            visibility="shared",
            shared_with_user_ids=[str(test_user.id)],
        )

        assert (await _act(db_session, test_user, item, "pause")).status == "paused"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", SLICE_ACTIONS)
    async def test_a_slice_action_needs_a_slice(self, db_session, test_user, action):
        item = await _item(db_session, test_user, _slice())

        for slice_id in (None, "no-such-slice"):
            assert await _refused(
                db_session, test_user, item, action, slice_id=slice_id
            ) == (400, "slice_id is required for this action")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "action", ["apply_override", "create_patch_pr", "keep_proposal_only"]
    )
    async def test_a_slice_nobody_is_waiting_on(self, db_session, test_user, action):
        item = await _item(
            db_session,
            test_user,
            _slice(awaiting_operator_action=False, allowed_slice_actions=[]),
        )

        assert await _refused(db_session, test_user, item, action, slice_id="s1") == (
            409,
            "Slice is not awaiting operator action",
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", ["cancel", "close"])
    @pytest.mark.parametrize("reason", [None, "because"])
    async def test_ending_an_item_needs_a_reason_from_the_list(
        self, db_session, test_user, action, reason
    ):
        item = await _item(db_session, test_user)

        assert await _refused(
            db_session, test_user, item, action, closure_reason=reason
        ) == (400, f"closure_reason is required for {action}")

    @pytest.mark.asyncio
    async def test_assigning_to_nobody_real(self, db_session, test_user):
        item = await _item(db_session, test_user)

        assert await _refused(
            db_session, test_user, item, "assign_backlog", assigned_user_id=uuid4()
        ) == (422, "Assigned user not found")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", ["apply_override", "create_patch_pr"])
    async def test_a_slice_with_no_proposal(self, db_session, test_user, action):
        item = await _item(db_session, test_user, _slice())

        status, detail = await _refused(
            db_session, test_user, item, action, slice_id="s1"
        )

        assert status == 400 and "valid proposal" in detail

    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", ["apply_override", "create_patch_pr"])
    async def test_somebody_elses_proposal(self, db_session, test_user, action):
        other = await _person(db_session, "other")
        proposal = await _proposal(db_session, other)
        item = await _item(
            db_session, test_user, _slice(selected_proposal_id=str(proposal.id))
        )

        assert await _refused(db_session, test_user, item, action, slice_id="s1") == (
            404,
            "Proposal not found",
        )
        assert await _jobs(db_session) == []


# --- who it belongs to -----------------------------------------------------


class TestAssignment:
    @pytest.mark.asyncio
    async def test_assigning_to_somebody_else_shares_it_with_them(
        self, db_session, test_user
    ):
        reviewer = await _person(db_session, "reviewer")
        item = await _item(db_session, test_user)

        done = await _act(
            db_session,
            test_user,
            item,
            "assign_backlog",
            assigned_user_id=reviewer.id,
            operator_note="yours",
        )

        assert done.assigned_user_id == reviewer.id
        assert done.assigned_by_user_id == test_user.id
        assert done.visibility == "shared"
        assert done.shared_with_user_ids == [reviewer.id]
        assert done.collaboration["note"] == "yours"
        assert _timeline_actions(done) == ["assign_backlog"]

    @pytest.mark.asyncio
    async def test_assigning_with_nobody_named_takes_it_yourself(
        self, db_session, test_user
    ):
        item = await _item(db_session, test_user)

        done = await _act(db_session, test_user, item, "assign_backlog")

        assert done.assigned_user_id == test_user.id
        assert done.visibility == "private"

    @pytest.mark.asyncio
    async def test_clearing_keeps_who_it_was_shared_with(self, db_session, test_user):
        reviewer = await _person(db_session, "reviewer")
        item = await _item(db_session, test_user)
        await _act(
            db_session, test_user, item, "assign_backlog", assigned_user_id=reviewer.id
        )

        done = await _act(db_session, test_user, item, "clear_backlog_assignment")

        assert done.assigned_user_id is None
        assert done.assigned_by_user_id is None
        assert done.shared_with_user_ids == [reviewer.id]
        assert _timeline_actions(done) == [
            "assign_backlog",
            "clear_backlog_assignment",
        ]

    @pytest.mark.asyncio
    async def test_a_note_is_kept(self, db_session, test_user):
        item = await _item(db_session, test_user)

        done = await _act(
            db_session, test_user, item, "update_backlog_note", operator_note="check CI"
        )

        assert done.collaboration["note"] == "check CI"
        assert done.latest_summary["operator_note"] == "check CI"

    @pytest.mark.asyncio
    async def test_a_second_note_replaces_the_first(self, db_session, test_user):
        item = await _item(db_session, test_user)
        await _act(
            db_session, test_user, item, "update_backlog_note", operator_note="first"
        )

        done = await _act(
            db_session, test_user, item, "update_backlog_note", operator_note="second"
        )

        assert done.collaboration["note"] == "second"
        assert done.latest_summary["operator_note"] == "second"


# --- its life --------------------------------------------------------------


class TestLifecycle:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("action", ["start", "resume"])
    async def test_starting_creates_the_orchestrator_and_queues_it(
        self, db_session, test_user, queued, action
    ):
        item = await _item(db_session, test_user)

        done = await _act(db_session, test_user, item, action)

        jobs = await _jobs(db_session)
        assert done.status == "running"
        assert [job.id for job in jobs] == [done.orchestrator_job_id]
        assert jobs[0].config["deterministic_runner"] == "coding_backlog_orchestrator"
        assert queued == [str(jobs[0].id)]
        assert _timeline_actions(done) == ["orchestrator_started"]

    @pytest.mark.asyncio
    async def test_pausing(self, db_session, test_user, queued):
        item = await _item(db_session, test_user, status="running")

        done = await _act(db_session, test_user, item, "pause", operator_note="hold")

        assert done.status == "paused"
        entry = done.decomposition["backlog_timeline"][-1]
        assert (entry["action"], entry["previous_status"], entry["new_status"]) == (
            "pause",
            "running",
            "paused",
        )
        assert queued == []

    @pytest.mark.asyncio
    async def test_cancelling(self, db_session, test_user):
        item = await _item(db_session, test_user, status="running")

        done = await _act(
            db_session, test_user, item, "cancel", closure_reason="Duplicate"
        )

        assert done.status == "cancelled"
        assert done.completed_at is not None
        assert done.latest_summary["status"] == "cancelled"
        assert done.latest_summary["closure_reason"] == "duplicate"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "reason, status",
        [
            ("fixed_through_backlog", "completed"),
            ("promoted_to_repair", "completed"),
            ("false_alarm", "cancelled"),
            ("blocked_external", "cancelled"),
        ],
    )
    async def test_closing_ends_it_by_its_reason(
        self, db_session, test_user, reason, status
    ):
        item = await _item(db_session, test_user)

        done = await _act(db_session, test_user, item, "close", closure_reason=reason)

        assert done.status == status
        assert done.latest_summary["status"] == "closed"
        assert done.latest_summary["closure_reason"] == reason


# --- what an operator decides about a slice --------------------------------


class TestSliceDecisions:
    @pytest.mark.asyncio
    async def test_apply_override_starts_an_apply_job(
        self, db_session, test_user, queued
    ):
        proposal = await _proposal(db_session, test_user)
        item = await _item(
            db_session, test_user, _slice(selected_proposal_id=str(proposal.id))
        )

        done = await _act(
            db_session,
            test_user,
            item,
            "apply_override",
            slice_id="s1",
            operator_note="ship it",
        )

        (job,) = await _jobs(db_session)
        row = _slice_of(done)
        assert job.config["deterministic_runner"] == "code_patch_apply_to_kb"
        assert job.config["proposal_id"] == str(proposal.id)
        assert job.config["coding_backlog_slice_id"] == "s1"
        assert queued == [str(job.id)]
        assert done.status == "running"
        assert done.current_job_id == job.id
        assert done.latest_apply_job_id == job.id
        assert done.child_job_ids == [str(job.id)]
        assert (row["status"], row["apply_job_id"]) == ("applying", str(job.id))
        assert row["awaiting_operator_action"] is False
        assert row["allowed_slice_actions"] == []
        assert row["operator_decision"] == "apply_override"
        assert done.decomposition["active_slice_id"] == "s1"
        assert done.latest_summary["status"] == "apply_started"
        assert _timeline_actions(done) == ["apply_override"]

    @pytest.mark.asyncio
    async def test_create_patch_pr_opens_a_draft_and_completes_the_item(
        self, db_session, test_user, queued
    ):
        proposal = await _proposal(db_session, test_user)
        item = await _item(
            db_session, test_user, _slice(selected_proposal_id=str(proposal.id))
        )

        done = await _act(db_session, test_user, item, "create_patch_pr", slice_id="s1")

        (pr,) = (await db_session.execute(select(PatchPR))).scalars().all()
        row = _slice_of(done)
        assert (pr.status, pr.selected_proposal_id) == ("draft", proposal.id)
        assert pr.title == "Parser crash: Slice s1"
        assert pr.checks["coding_backlog"]["slice_id"] == "s1"
        assert (row["status"], row["patch_pr_id"]) == ("patch_pr", str(pr.id))
        assert row["promotion_decision"] == "patch_pr"
        assert done.status == "completed"
        assert done.completed_at is not None
        assert done.decomposition["completed_slices"] == ["s1"]
        assert done.decomposition["active_slice_id"] is None
        decision = done.decomposition["promotion_decisions"][-1]
        assert (decision["decision"], decision["patch_pr_id"]) == (
            "patch_pr",
            str(pr.id),
        )
        assert done.latest_summary["patch_pr_id"] == str(pr.id)
        assert queued == [] and await _jobs(db_session) == []

    @pytest.mark.asyncio
    async def test_keep_proposal_only_completes_without_a_job(
        self, db_session, test_user, queued
    ):
        item = await _item(db_session, test_user, _slice())

        done = await _act(
            db_session, test_user, item, "keep_proposal_only", slice_id="s1"
        )

        row = _slice_of(done)
        assert (row["status"], row["promotion_decision"]) == (
            "proposal_only",
            "proposal_only",
        )
        assert row["manual_promotion_history"][-1]["action"] == "keep_proposal_only"
        assert done.status == "completed"
        assert done.decomposition["completed_slices"] == ["s1"]
        assert done.latest_summary["promotion_decision"] == "proposal_only"
        assert queued == []

    @pytest.mark.asyncio
    async def test_relaunch_starts_a_repair_job_and_counts_the_retry(
        self, db_session, test_user, queued
    ):
        earlier = str(uuid4())
        item = await _item(
            db_session,
            test_user,
            _slice(status="failed", child_job_id=earlier, retry_count=1),
        )

        done = await _act(db_session, test_user, item, "relaunch_slice", slice_id="s1")

        (job,) = await _jobs(db_session)
        row = _slice_of(done)
        assert job.config["coding_backlog_child_kind"] == "repair"
        assert job.config["coding_backlog_slice_id"] == "s1"
        assert job.goal == "Fix s1"
        assert queued == [str(job.id)]
        assert (row["status"], row["retry_count"]) == ("retrying", 2)
        assert row["child_job_id"] == str(job.id)
        assert row["job_lineage"]["retry_from_job_ids"] == [earlier]
        assert done.status == "running"
        assert done.current_job_id == job.id
        assert done.latest_summary["status"] == "repair_started"
        assert done.latest_summary["current_child_job_id"] == str(job.id)
        # The run it retries, not the run it just started.
        assert done.latest_summary["retry_from_job_id"] == earlier

    @pytest.mark.asyncio
    async def test_relaunch_is_allowed_on_a_slice_nobody_is_waiting_on(
        self, db_session, test_user
    ):
        item = await _item(
            db_session,
            test_user,
            _slice(awaiting_operator_action=False, allowed_slice_actions=[]),
        )

        done = await _act(db_session, test_user, item, "relaunch_slice", slice_id="s1")

        assert _slice_of(done)["status"] == "retrying"

    @pytest.mark.asyncio
    async def test_skipping_moves_on_to_the_next_pending_slice(
        self, db_session, test_user, queued
    ):
        item = await _item(
            db_session,
            test_user,
            _slice("s1"),
            _slice(
                "s2",
                status="pending",
                awaiting_operator_action=False,
                allowed_slice_actions=[],
            ),
        )

        done = await _act(db_session, test_user, item, "skip_slice", slice_id="s1")

        (job,) = await _jobs(db_session)
        assert _slice_of(done, "s1")["status"] == "deferred"
        assert _slice_of(done, "s2")["status"] == "repairing"
        assert _slice_of(done, "s2")["child_job_id"] == str(job.id)
        assert job.config["coding_backlog_slice_id"] == "s2"
        assert done.status == "running"
        assert done.decomposition["active_slice_id"] == "s2"
        assert queued == [str(job.id)]

    @pytest.mark.asyncio
    async def test_skipping_the_last_slice_completes_the_item(
        self, db_session, test_user, queued
    ):
        item = await _item(db_session, test_user, _slice("s1"))

        done = await _act(db_session, test_user, item, "skip_slice", slice_id="s1")

        assert _slice_of(done)["status"] == "deferred"
        assert done.status == "completed"
        assert done.completed_at is not None
        assert done.decomposition["active_slice_id"] is None
        assert queued == []


# --- the rules as data ------------------------------------------------------


class TestTheActionTable:
    def test_it_offers_exactly_what_the_request_schema_describes(self):
        from app.modules.coding_backlog.application.backlog_actions import ACTIONS

        described = CodingBacklogItemActionRequest.model_fields["action"].description
        named = {name.strip() for name in described.split("|")}

        assert set(ACTIONS) == named

    def test_who_may_do_what(self):
        from app.modules.coding_backlog.application.backlog_actions import ACTIONS

        owner_only = {name for name, action in ACTIONS.items() if action.owner_only}
        on_slice = {name for name, action in ACTIONS.items() if action.on_slice}

        assert owner_only == {"cancel", "close", *SLICE_ACTIONS}
        assert on_slice == set(SLICE_ACTIONS)
        # A decision about a slice is always the owner's.
        assert on_slice <= owner_only

    def test_every_refusal_kind_has_a_status(self):
        from app.modules.coding_backlog.application.errors import KINDS

        assert set(coding_backlog._REFUSAL_STATUS) == set(KINDS)

    def test_the_application_layer_does_not_speak_http(self):
        import ast
        from pathlib import Path

        package = (
            Path(coding_backlog.__file__).resolve().parents[2]
            / "modules/coding_backlog/application"
        )
        imported = set()
        for path in package.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    imported.add(str(node.module))
                elif isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)

        assert imported, "nothing was read"
        assert not [name for name in imported if name.split(".")[0] == "fastapi"]
        assert not [name for name in imported if name.startswith("app.api")]
