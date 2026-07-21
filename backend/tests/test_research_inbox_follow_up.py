from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4
from datetime import datetime, timedelta, timezone

import pytest

from app.api.endpoints import agent_jobs as agent_jobs_endpoint
from app.api.endpoints import research_inbox as research_inbox_endpoint
from app.api.endpoints import research_monitor_profiles as research_monitor_profiles_endpoint
from app.models.agent_job import AgentJob
from app.models.domain_research_profile import DomainResearchProfile
from app.models.research_inbox import ResearchInboxItem
from app.models.research_portfolio import ResearchPortfolio
from app.models.research_monitor_profile import ResearchMonitorProfile
from app.models.notification import NotificationType
from app.schemas.research_inbox import (
    ResearchInboxBulkFollowUpRelaunchRequest,
    ResearchInboxFollowUpRelaunchRequest,
)
from app.schemas.agent_job import AgentCheckpointQueueFollowUpActionRequest, AgentCheckpointQueueBulkFollowUpActionRequest
from app.schemas.research_monitor_profile import (
    ResearchMonitorCustomerRebalanceApplyMonitorRequest,
    ResearchMonitorCustomerRebalanceApplyRequest,
    ResearchMonitorBudgetUpdateRequest,
    ResearchMonitorPolicyRollbackRequest,
    ResearchMonitorPolicyUpdateRequest,
)
from app.services.research_inbox_follow_up_service import sync_follow_up_outcome_for_job
from app.services.research_monitor_profile_service import research_monitor_profile_service


class _FakeScalarResult:
    def __init__(self, values):
        self._values = values

    def all(self):
        return list(self._values)


class _FakeExecuteResult:
    def __init__(self, *, rows=None, scalar=None):
        self._rows = list(rows or [])
        self._scalar = scalar

    def all(self):
        return list(self._rows)

    def scalar_one_or_none(self):
        return self._scalar

    def scalar_one(self):
        return self._scalar

    def scalars(self):
        return _FakeScalarResult(self._rows)


class _FakeProfileSession:
    def __init__(self, results):
        self._results = list(results)
        self.added = []
        self.commits = 0
        self.refreshed = []

    async def execute(self, _stmt):
        if not self._results:
            raise AssertionError("Unexpected execute call")
        return self._results.pop(0)

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1

    async def refresh(self, obj):
        self.refreshed.append(obj)


class _FakeInboxOutcomeSession:
    def __init__(self, items, *, profile=None, portfolio=None):
        self.items = list(items)
        self.notifications = []
        self.profile = profile
        self.portfolio = portfolio

    async def execute(self, _stmt):
        model_name = None
        try:
            model_name = _stmt.column_descriptions[0]["entity"].__name__
        except Exception:
            model_name = None
        if model_name == "Notification":
            return _FakeExecuteResult(rows=self.notifications)
        if model_name == "NotificationPreferences":
            return _FakeExecuteResult(scalar=None)
        return _FakeExecuteResult(rows=self.items)

    def add(self, obj):
        self.notifications.append(obj)

    async def flush(self):
        return None


class _FakeInboxSerializeDb:
    def __init__(self, jobs_by_id=None):
        self.jobs_by_id = dict(jobs_by_id or {})

    async def get(self, model, lookup_id):
        if model is AgentJob:
            return self.jobs_by_id.get(lookup_id)
        return None

    async def refresh(self, _obj):
        return None

    async def get(self, model, lookup_id):
        if model is DomainResearchProfile and self.profile is not None and self.profile.id == lookup_id:
            return self.profile
        if model is ResearchPortfolio and self.portfolio is not None and self.portfolio.id == lookup_id:
            return self.portfolio
        return None


@pytest.mark.asyncio
async def test_apply_follow_up_policy_blocks_manual_mode():
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=uuid4(),
        item_type="document",
        item_key="doc-1",
        title="Accepted note",
        status="accepted",
        job_id=uuid4(),
    )
    source_job = AgentJob(
        id=item.job_id,
        user_id=item.user_id,
        name="Research Inbox Monitor",
        goal="Monitor for updates",
        job_type="monitor",
        status="pending",
        config={"follow_up_autonomy": {"mode": "manual_only"}},
    )

    async def fake_get(_model, lookup_id):
        if lookup_id == item.job_id:
            return source_job
        return None

    db = SimpleNamespace(get=fake_get)
    current_user = SimpleNamespace(id=item.user_id)

    await agent_jobs_endpoint._apply_follow_up_policy_on_accept(
        item=item,
        current_user=current_user,
        db=db,
    )

    assert item.follow_up_policy_mode == "manual_only"
    assert item.follow_up_decision == "manual"
    assert item.follow_up_launch_status == "blocked"
    assert "manual follow-up launches" in str(item.follow_up_block_reason or "")


@pytest.mark.asyncio
async def test_apply_follow_up_policy_queues_safe_follow_up_for_approval():
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=uuid4(),
        item_type="document",
        item_key="doc-2",
        title="Accepted note",
        status="accepted",
        job_id=uuid4(),
    )
    source_job = AgentJob(
        id=item.job_id,
        user_id=item.user_id,
        name="Research Inbox Monitor",
        goal="Monitor for updates",
        job_type="monitor",
        status="pending",
        config={"follow_up_autonomy": {"mode": "queue_for_approval"}},
    )

    async def fake_get(_model, lookup_id):
        if lookup_id == item.job_id:
            return source_job
        return None

    db = SimpleNamespace(get=fake_get)
    current_user = SimpleNamespace(id=item.user_id)

    await agent_jobs_endpoint._apply_follow_up_policy_on_accept(
        item=item,
        current_user=current_user,
        db=db,
    )

    assert item.follow_up_policy_mode == "queue_for_approval"
    assert item.follow_up_decision == "queued_for_approval"
    assert item.follow_up_launch_status == "pending_approval"
    assert item.follow_up_recommendation_key == "deep_dive_chain"


@pytest.mark.asyncio
async def test_apply_follow_up_policy_auto_launches_safe_chain(monkeypatch):
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=uuid4(),
        item_type="document",
        item_key="doc-3",
        title="Accepted note",
        status="accepted",
        job_id=uuid4(),
    )
    source_job = AgentJob(
        id=item.job_id,
        user_id=item.user_id,
        name="Research Inbox Monitor",
        goal="Monitor for updates",
        job_type="monitor",
        status="pending",
        config={
            "follow_up_autonomy": {
                "mode": "auto_launch_safe",
                "allowed_recommendations": ["deep_dive_chain"],
            }
        },
    )

    async def fake_get(_model, lookup_id):
        if lookup_id == item.job_id:
            return source_job
        return None

    async def fake_create_job_from_chain(request, db, current_user):
        return SimpleNamespace(id=uuid4(), chain_definition_id=request.chain_definition_id)

    monkeypatch.setattr(agent_jobs_endpoint, "create_job_from_chain", fake_create_job_from_chain)

    db = SimpleNamespace(get=fake_get)
    current_user = SimpleNamespace(id=item.user_id)

    await agent_jobs_endpoint._apply_follow_up_policy_on_accept(
        item=item,
        current_user=current_user,
        db=db,
    )

    assert item.follow_up_policy_mode == "auto_launch_safe"
    assert item.follow_up_decision == "auto_launched"
    assert item.follow_up_launch_status == "launched"
    assert item.follow_up_job_id is not None
    assert item.follow_up_chain_definition_id is not None
    assert item.follow_up_launched_at is not None


@pytest.mark.asyncio
async def test_apply_follow_up_policy_blocks_non_allowlisted_safe_recommendation():
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=uuid4(),
        item_type="arxiv",
        item_key="paper-1",
        title="Paper with repo",
        status="accepted",
        job_id=uuid4(),
        item_metadata={"repos": [{"provider": "github", "repo": "acme/demo"}]},
    )
    source_job = AgentJob(
        id=item.job_id,
        user_id=item.user_id,
        name="Research Inbox Monitor",
        goal="Monitor for updates",
        job_type="monitor",
        status="pending",
        config={
            "follow_up_autonomy": {
                "mode": "auto_launch_safe",
                "allowed_recommendations": ["repo_patch_chain"],
            }
        },
    )

    async def fake_get(_model, lookup_id):
        if lookup_id == item.job_id:
            return source_job
        return None

    db = SimpleNamespace(get=fake_get)
    current_user = SimpleNamespace(id=item.user_id)

    await agent_jobs_endpoint._apply_follow_up_policy_on_accept(
        item=item,
        current_user=current_user,
        db=db,
    )

    assert item.follow_up_recommendation_key == "deep_dive_chain"
    assert item.follow_up_launch_status == "blocked"
    assert item.follow_up_decision == "manual"
    assert "allowlisted" in str(item.follow_up_block_reason or "")


@pytest.mark.asyncio
async def test_perform_follow_up_queue_action_approves_and_launches(monkeypatch):
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=uuid4(),
        item_type="document",
        item_key="doc-4",
        title="Pending approval",
        status="accepted",
        follow_up_launch_status="pending_approval",
        follow_up_recommendation_key="deep_dive_chain",
    )

    async def fake_create_job_from_chain(request, db, current_user):
        return SimpleNamespace(id=uuid4(), chain_definition_id=request.chain_definition_id)

    monkeypatch.setattr(agent_jobs_endpoint, "create_job_from_chain", fake_create_job_from_chain)

    response = await agent_jobs_endpoint._perform_follow_up_queue_action(
        item=item,
        action="approve_launch",
        operator_note="Looks bounded and safe.",
        db=SimpleNamespace(),
        current_user=SimpleNamespace(id=item.user_id),
    )

    assert response.follow_up_launch_status == "launched"
    assert response.follow_up_operator_decision == "approved_launch"
    assert response.follow_up_job_id is not None
    assert item.follow_up_launch_status == "launched"
    assert item.follow_up_operator_decision == "approved_launch"
    assert item.follow_up_operator_note == "Looks bounded and safe."


@pytest.mark.asyncio
async def test_perform_follow_up_queue_action_rejects_without_launch():
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=uuid4(),
        item_type="document",
        item_key="doc-5",
        title="Pending approval",
        status="accepted",
        follow_up_launch_status="pending_approval",
        follow_up_recommendation_key="deep_dive_chain",
    )

    response = await agent_jobs_endpoint._perform_follow_up_queue_action(
        item=item,
        action="reject_launch",
        operator_note="Not worth running.",
        db=SimpleNamespace(),
        current_user=SimpleNamespace(id=item.user_id),
    )

    assert response.follow_up_launch_status == "rejected"
    assert response.follow_up_operator_decision == "rejected"
    assert item.follow_up_launch_status == "rejected"
    assert item.follow_up_operator_decision == "rejected"
    assert item.follow_up_operator_note == "Not worth running."


@pytest.mark.asyncio
async def test_checkpoint_queue_follow_up_action_threads_source_job_scheduler_state(monkeypatch):
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=uuid4(),
        item_type="document",
        item_key="doc-queue-1",
        title="Queued follow-up",
        status="accepted",
        job_id=uuid4(),
        follow_up_launch_status="pending_approval",
        follow_up_recommendation_key="deep_dive_chain",
    )
    source_job = AgentJob(
        id=item.job_id,
        user_id=item.user_id,
        name="Inbox Monitor",
        goal="Monitor for updates",
        job_type="monitor",
        status="failed",
        results={
            "execution_strategy": {
                "scheduler_state": {
                    "queue_reason": "execution_failure",
                    "last_scheduled_at": "2026-03-16T09:00:00Z",
                    "last_dispatched_at": "2026-03-16T09:05:00Z",
                }
            }
        },
    )
    captured = {}

    class _Db:
        async def execute(self, _stmt):
            return _FakeExecuteResult(scalar=item)

        async def get(self, model, lookup_id):
            if model is AgentJob and lookup_id == source_job.id:
                return source_job
            return None

        async def commit(self):
            return None

    async def _fake_follow_up_action(*args, **kwargs):
        return SimpleNamespace(
            follow_up_launch_status="launched",
            follow_up_operator_decision="approved_launch",
        )

    async def _fake_record(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr(agent_jobs_endpoint, "_perform_follow_up_queue_action", _fake_follow_up_action)
    monkeypatch.setattr(agent_jobs_endpoint, "record_autonomy_decision_event", _fake_record)

    response = await agent_jobs_endpoint.checkpoint_queue_follow_up_action(
        ResearchInboxFollowUpRelaunchRequest(
            inbox_item_id=item.id,
            action="approve_launch",
            operator_note="Looks safe to launch.",
        ),
        current_user=SimpleNamespace(id=item.user_id),
        db=_Db(),
    )

    assert response.follow_up_launch_status == "launched"
    assert captured["reason_label"] == "Deep dive chain"
    assert captured["scheduler_state"] == {
        "queue_reason": "execution_failure",
        "last_scheduled_at": "2026-03-16T09:00:00Z",
        "last_dispatched_at": "2026-03-16T09:05:00Z",
    }


@pytest.mark.asyncio
async def test_checkpoint_queue_follow_up_action_omits_malformed_source_job_scheduler_state(monkeypatch):
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=uuid4(),
        item_type="document",
        item_key="doc-queue-2",
        title="Queued follow-up",
        status="accepted",
        job_id=uuid4(),
        follow_up_launch_status="pending_approval",
        follow_up_recommendation_key="deep_dive_chain",
    )
    source_job = AgentJob(
        id=item.job_id,
        user_id=item.user_id,
        name="Inbox Monitor",
        goal="Monitor for updates",
        job_type="monitor",
        status="failed",
        results={"execution_strategy": {"scheduler_state": "bad-payload"}},
    )
    captured = {}

    class _Db:
        async def execute(self, _stmt):
            return _FakeExecuteResult(scalar=item)

        async def get(self, model, lookup_id):
            if model is AgentJob and lookup_id == source_job.id:
                return source_job
            return None

        async def commit(self):
            return None

    async def _fake_follow_up_action(*args, **kwargs):
        return SimpleNamespace(
            follow_up_launch_status="rejected",
            follow_up_operator_decision="rejected",
        )

    async def _fake_record(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr(agent_jobs_endpoint, "_perform_follow_up_queue_action", _fake_follow_up_action)
    monkeypatch.setattr(agent_jobs_endpoint, "record_autonomy_decision_event", _fake_record)

    response = await agent_jobs_endpoint.checkpoint_queue_follow_up_action(
        ResearchInboxFollowUpRelaunchRequest(
            inbox_item_id=item.id,
            action="reject_launch",
            operator_note="Not safe enough.",
        ),
        current_user=SimpleNamespace(id=item.user_id),
        db=_Db(),
    )

    assert response.follow_up_launch_status == "rejected"
    assert captured["reason_label"] == "Deep dive chain"
    assert captured["scheduler_state"] is None


@pytest.mark.asyncio
async def test_checkpoint_queue_follow_up_action_threads_profile_parent_job_scheduler_state(monkeypatch):
    profile = DomainResearchProfile(
        id=uuid4(),
        user_id=uuid4(),
        title="Compiler Frontier",
        domain="Compiler",
        objective="Track compiler opportunities",
        status="running",
        latest_run_job_id=uuid4(),
        latest_summary={
            "opportunities": [
                {
                    "opportunity_id": "profile-opp-1",
                    "canonical_key": "compiler_follow_up",
                    "title": "Compiler follow-up",
                    "hypothesis": "Reuse a bounded follow-up job",
                    "decision_state": "pending_review",
                    "stage": "discovered",
                    "child_job_ids": [],
                }
            ]
        },
    )
    parent_job = AgentJob(
        id=profile.latest_run_job_id,
        user_id=profile.user_id,
        name="Compiler Monitor",
        goal="Track compiler opportunities",
        job_type="monitor",
        status="failed",
        results={
            "execution_strategy": {
                "scheduler_state": {
                    "queue_reason": "scheduled_recovery",
                    "last_scheduled_at": "2026-03-16T09:00:00Z",
                    "last_dispatched_at": "2026-03-16T09:05:00Z",
                }
            }
        },
    )
    captured = {}

    class _Db:
        async def execute(self, _stmt):
            return _FakeExecuteResult(scalar=profile)

        async def get(self, model, lookup_id):
            if model is AgentJob and lookup_id == parent_job.id:
                return parent_job
            return None

        async def commit(self):
            return None

    async def _fake_follow_up_action(*args, **kwargs):
        return SimpleNamespace(
            follow_up_launch_status="launched",
            follow_up_operator_decision="approved_launch",
        )

    async def _fake_record(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr(agent_jobs_endpoint, "_perform_follow_up_queue_action", _fake_follow_up_action)
    monkeypatch.setattr(agent_jobs_endpoint, "record_autonomy_decision_event", _fake_record)

    response = await agent_jobs_endpoint.checkpoint_queue_follow_up_action(
        AgentCheckpointQueueFollowUpActionRequest(
            domain_research_profile_id=profile.id,
            profile_opportunity_id="profile-opp-1",
            action="approve_launch",
            operator_note="Looks good to launch.",
        ),
        current_user=SimpleNamespace(id=profile.user_id),
        db=_Db(),
    )

    assert response.follow_up_launch_status == "launched"
    assert captured["reason_label"] == "Compiler follow up"
    assert captured["scheduler_state"] == {
        "queue_reason": "scheduled_recovery",
        "last_scheduled_at": "2026-03-16T09:00:00Z",
        "last_dispatched_at": "2026-03-16T09:05:00Z",
    }


@pytest.mark.asyncio
async def test_checkpoint_queue_follow_up_action_omits_malformed_profile_parent_job_scheduler_state(monkeypatch):
    profile = DomainResearchProfile(
        id=uuid4(),
        user_id=uuid4(),
        title="Compiler Frontier",
        domain="Compiler",
        objective="Track compiler opportunities",
        status="running",
        latest_run_job_id=uuid4(),
        latest_summary={
            "opportunities": [
                {
                    "opportunity_id": "profile-opp-2",
                    "canonical_key": "compiler_follow_up",
                    "title": "Compiler follow-up",
                    "hypothesis": "Reuse a bounded follow-up job",
                    "decision_state": "pending_review",
                    "stage": "discovered",
                    "child_job_ids": [],
                }
            ]
        },
    )
    parent_job = AgentJob(
        id=profile.latest_run_job_id,
        user_id=profile.user_id,
        name="Compiler Monitor",
        goal="Track compiler opportunities",
        job_type="monitor",
        status="failed",
        results={"execution_strategy": {"scheduler_state": "bad-payload"}},
    )
    captured = {}

    class _Db:
        async def execute(self, _stmt):
            return _FakeExecuteResult(scalar=profile)

        async def get(self, model, lookup_id):
            if model is AgentJob and lookup_id == parent_job.id:
                return parent_job
            return None

        async def commit(self):
            return None

    async def _fake_follow_up_action(*args, **kwargs):
        return SimpleNamespace(
            follow_up_launch_status="rejected",
            follow_up_operator_decision="rejected",
        )

    async def _fake_record(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr(agent_jobs_endpoint, "_perform_follow_up_queue_action", _fake_follow_up_action)
    monkeypatch.setattr(agent_jobs_endpoint, "record_autonomy_decision_event", _fake_record)

    response = await agent_jobs_endpoint.checkpoint_queue_follow_up_action(
        AgentCheckpointQueueFollowUpActionRequest(
            domain_research_profile_id=profile.id,
            profile_opportunity_id="profile-opp-2",
            action="reject_launch",
            operator_note="Not safe enough.",
        ),
        current_user=SimpleNamespace(id=profile.user_id),
        db=_Db(),
    )

    assert response.follow_up_launch_status == "rejected"
    assert captured["reason_label"] == "Compiler follow up"
    assert captured["scheduler_state"] is None


@pytest.mark.asyncio
async def test_checkpoint_queue_bulk_follow_up_action_processes_profile_rows(monkeypatch):
    profile = DomainResearchProfile(
        id=uuid4(),
        user_id=uuid4(),
        title="Compiler Frontier",
        domain="Compiler",
        objective="Track compiler opportunities",
        status="running",
        latest_run_job_id=uuid4(),
    )
    parent_job = AgentJob(
        id=profile.latest_run_job_id,
        user_id=profile.user_id,
        name="Compiler Monitor",
        goal="Track compiler opportunities",
        job_type="monitor",
        status="failed",
        results={"execution_strategy": {"scheduler_state": {"queue_reason": "scheduled_recovery"}}},
    )
    captured_actions = []
    captured_events = []

    class _Db:
        async def execute(self, _stmt):
            return _FakeExecuteResult(scalar=profile)

        async def get(self, model, lookup_id):
            if model is AgentJob and lookup_id == parent_job.id:
                return parent_job
            return None

        async def commit(self):
            return None

    async def _fake_follow_up_action(*args, **kwargs):
        opportunity_id = kwargs.get("profile_opportunity_id")
        captured_actions.append(opportunity_id)
        if opportunity_id == "opp-fail":
            raise agent_jobs_endpoint.HTTPException(status_code=400, detail="Already launched")
        return SimpleNamespace(
            follow_up_launch_status="launched",
            follow_up_operator_decision="approved_launch",
            follow_up_job_id=uuid4(),
            detail="Follow-up launched from queue approval",
        )

    async def _fake_record(*args, **kwargs):
        captured_events.append(kwargs)
        return None

    monkeypatch.setattr(agent_jobs_endpoint, "_perform_follow_up_queue_action", _fake_follow_up_action)
    monkeypatch.setattr(agent_jobs_endpoint, "_record_follow_up_queue_decision_event", _fake_record)

    response = await agent_jobs_endpoint.checkpoint_queue_bulk_follow_up_action(
        AgentCheckpointQueueBulkFollowUpActionRequest(
            domain_research_profile_id=profile.id,
            profile_opportunity_ids=["opp-1", "opp-fail"],
            action="approve_launch",
            operator_note="Bulk ship",
        ),
        current_user=SimpleNamespace(id=profile.user_id),
        db=_Db(),
    )

    assert captured_actions == ["opp-1", "opp-fail"]
    assert response.requested_count == 2
    assert response.applied == 1
    assert response.failed == 1
    assert response.results[0].ok is True
    assert response.results[0].profile_opportunity_id == "opp-1"
    assert response.results[1].ok is False
    assert response.results[1].profile_opportunity_id == "opp-fail"
    assert response.results[1].error == "Already launched"
    assert len(captured_events) == 1
    assert captured_events[0]["reason_code"] == "opp-1"


@pytest.mark.asyncio
async def test_checkpoint_queue_follow_up_action_threads_portfolio_parent_job_scheduler_state(monkeypatch):
    portfolio = ResearchPortfolio(
        id=uuid4(),
        user_id=uuid4(),
        title="Scientific Fleet",
        objective="Track scientific opportunities",
        status="running",
        latest_run_job_id=uuid4(),
        opportunities=[
            {
                "opportunity_id": "portfolio-opp-1",
                "canonical_key": "fleet_follow_up",
                "title": "Fleet follow-up",
                "hypothesis": "Reuse a bounded follow-up job",
                "decision_state": "pending_review",
                "stage": "discovered",
                "child_job_ids": [],
            }
        ],
    )
    parent_job = AgentJob(
        id=portfolio.latest_run_job_id,
        user_id=portfolio.user_id,
        name="Fleet Monitor",
        goal="Track scientific opportunities",
        job_type="research",
        status="failed",
        results={
            "execution_strategy": {
                "scheduler_state": {
                    "queue_reason": "execution_failure",
                    "last_scheduled_at": "2026-03-16T10:00:00Z",
                    "last_dispatched_at": "2026-03-16T10:05:00Z",
                }
            }
        },
    )
    captured = {}

    class _Db:
        async def execute(self, _stmt):
            return _FakeExecuteResult(scalar=portfolio)

        async def get(self, model, lookup_id):
            if model is AgentJob and lookup_id == parent_job.id:
                return parent_job
            return None

        async def commit(self):
            return None

    async def _fake_follow_up_action(*args, **kwargs):
        return SimpleNamespace(
            follow_up_launch_status="launched",
            follow_up_operator_decision="approved_launch",
        )

    async def _fake_record(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr(agent_jobs_endpoint, "_perform_follow_up_queue_action", _fake_follow_up_action)
    monkeypatch.setattr(agent_jobs_endpoint, "record_autonomy_decision_event", _fake_record)

    response = await agent_jobs_endpoint.checkpoint_queue_follow_up_action(
        AgentCheckpointQueueFollowUpActionRequest(
            portfolio_id=portfolio.id,
            portfolio_opportunity_id="portfolio-opp-1",
            action="approve_launch",
            operator_note="Looks good to launch.",
        ),
        current_user=SimpleNamespace(id=portfolio.user_id),
        db=_Db(),
    )

    assert response.follow_up_launch_status == "launched"
    assert captured["reason_label"] == "Fleet follow up"
    assert captured["scheduler_state"] == {
        "queue_reason": "execution_failure",
        "last_scheduled_at": "2026-03-16T10:00:00Z",
        "last_dispatched_at": "2026-03-16T10:05:00Z",
    }


@pytest.mark.asyncio
async def test_checkpoint_queue_bulk_follow_up_action_processes_portfolio_rows(monkeypatch):
    portfolio = ResearchPortfolio(
        id=uuid4(),
        user_id=uuid4(),
        title="Scientific Fleet",
        objective="Track scientific opportunities",
        status="running",
        latest_run_job_id=uuid4(),
    )
    parent_job = AgentJob(
        id=portfolio.latest_run_job_id,
        user_id=portfolio.user_id,
        name="Fleet Monitor",
        goal="Track scientific opportunities",
        job_type="research",
        status="failed",
        results={"execution_strategy": {"scheduler_state": {"queue_reason": "execution_failure"}}},
    )
    captured_actions = []
    captured_events = []

    class _Db:
        async def execute(self, _stmt):
            return _FakeExecuteResult(scalar=portfolio)

        async def get(self, model, lookup_id):
            if model is AgentJob and lookup_id == parent_job.id:
                return parent_job
            return None

        async def commit(self):
            return None

    async def _fake_follow_up_action(*args, **kwargs):
        opportunity_id = kwargs.get("portfolio_opportunity_id")
        captured_actions.append(opportunity_id)
        return SimpleNamespace(
            follow_up_launch_status="rejected",
            follow_up_operator_decision="rejected",
            follow_up_job_id=None,
            detail="Operator rejected the queued follow-up launch.",
        )

    async def _fake_record(*args, **kwargs):
        captured_events.append(kwargs)
        return None

    monkeypatch.setattr(agent_jobs_endpoint, "_perform_follow_up_queue_action", _fake_follow_up_action)
    monkeypatch.setattr(agent_jobs_endpoint, "_record_follow_up_queue_decision_event", _fake_record)

    response = await agent_jobs_endpoint.checkpoint_queue_bulk_follow_up_action(
        AgentCheckpointQueueBulkFollowUpActionRequest(
            portfolio_id=portfolio.id,
            portfolio_opportunity_ids=["opp-1", "opp-2"],
            action="reject_launch",
            operator_note="Bulk reject",
        ),
        current_user=SimpleNamespace(id=portfolio.user_id),
        db=_Db(),
    )

    assert captured_actions == ["opp-1", "opp-2"]
    assert response.requested_count == 2
    assert response.applied == 2
    assert response.failed == 0
    assert all(row.ok for row in response.results)
    assert {row.portfolio_opportunity_id for row in response.results} == {"opp-1", "opp-2"}
    assert len(captured_events) == 2


@pytest.mark.asyncio
async def test_checkpoint_queue_follow_up_action_omits_malformed_portfolio_parent_job_scheduler_state(monkeypatch):
    portfolio = ResearchPortfolio(
        id=uuid4(),
        user_id=uuid4(),
        title="Scientific Fleet",
        objective="Track scientific opportunities",
        status="running",
        latest_run_job_id=uuid4(),
        opportunities=[
            {
                "opportunity_id": "portfolio-opp-2",
                "canonical_key": "fleet_follow_up",
                "title": "Fleet follow-up",
                "hypothesis": "Reuse a bounded follow-up job",
                "decision_state": "pending_review",
                "stage": "discovered",
                "child_job_ids": [],
            }
        ],
    )
    parent_job = AgentJob(
        id=portfolio.latest_run_job_id,
        user_id=portfolio.user_id,
        name="Fleet Monitor",
        goal="Track scientific opportunities",
        job_type="research",
        status="failed",
        results={"execution_strategy": {"scheduler_state": "bad-payload"}},
    )
    captured = {}

    class _Db:
        async def execute(self, _stmt):
            return _FakeExecuteResult(scalar=portfolio)

        async def get(self, model, lookup_id):
            if model is AgentJob and lookup_id == parent_job.id:
                return parent_job
            return None

        async def commit(self):
            return None

    async def _fake_follow_up_action(*args, **kwargs):
        return SimpleNamespace(
            follow_up_launch_status="rejected",
            follow_up_operator_decision="rejected",
        )

    async def _fake_record(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr(agent_jobs_endpoint, "_perform_follow_up_queue_action", _fake_follow_up_action)
    monkeypatch.setattr(agent_jobs_endpoint, "record_autonomy_decision_event", _fake_record)

    response = await agent_jobs_endpoint.checkpoint_queue_follow_up_action(
        AgentCheckpointQueueFollowUpActionRequest(
            portfolio_id=portfolio.id,
            portfolio_opportunity_id="portfolio-opp-2",
            action="reject_launch",
            operator_note="Not safe enough.",
        ),
        current_user=SimpleNamespace(id=portfolio.user_id),
        db=_Db(),
    )

    assert response.follow_up_launch_status == "rejected"
    assert captured["reason_label"] == "Fleet follow up"
    assert captured["scheduler_state"] is None


@pytest.mark.asyncio
async def test_checkpoint_queue_follow_up_action_omits_scheduler_state_without_source_job(monkeypatch):
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=uuid4(),
        item_type="document",
        item_key="doc-queue-3",
        title="Queued follow-up",
        status="accepted",
        job_id=None,
        follow_up_launch_status="pending_approval",
        follow_up_recommendation_key="deep_dive_chain",
    )
    captured = {}

    class _Db:
        async def execute(self, _stmt):
            return _FakeExecuteResult(scalar=item)

        async def get(self, model, lookup_id):
            return None

        async def commit(self):
            return None

    async def _fake_follow_up_action(*args, **kwargs):
        return SimpleNamespace(
            follow_up_launch_status="rejected",
            follow_up_operator_decision="rejected",
        )

    async def _fake_record(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr(agent_jobs_endpoint, "_perform_follow_up_queue_action", _fake_follow_up_action)
    monkeypatch.setattr(agent_jobs_endpoint, "record_autonomy_decision_event", _fake_record)

    response = await agent_jobs_endpoint.checkpoint_queue_follow_up_action(
        AgentCheckpointQueueFollowUpActionRequest(
            inbox_item_id=item.id,
            action="reject_launch",
            operator_note="Not safe enough.",
        ),
        current_user=SimpleNamespace(id=item.user_id),
        db=_Db(),
    )

    assert response.follow_up_launch_status == "rejected"
    assert captured["reason_label"] == "Deep dive chain"
    assert captured["scheduler_state"] is None


@pytest.mark.asyncio
async def test_sync_follow_up_outcome_for_job_records_completed_summary():
    job = AgentJob(
        id=uuid4(),
        user_id=uuid4(),
        name="Deep Dive Follow-up",
        goal="Investigate the accepted signal",
        job_type="research",
        status="completed",
        results={"summary": "Produced a concise follow-up report with concrete next steps."},
    )
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=job.user_id,
        item_type="document",
        item_key="doc-6",
        title="Accepted note",
        status="accepted",
        follow_up_job_id=job.id,
        follow_up_launch_status="launched",
    )

    updated = await sync_follow_up_outcome_for_job(_FakeInboxOutcomeSession([item]), job)

    assert updated == 1
    assert item.follow_up_outcome_status == "completed"
    assert item.follow_up_outcome_recorded_at is not None
    assert item.follow_up_outcome_summary == "Produced a concise follow-up report with concrete next steps."


@pytest.mark.asyncio
async def test_sync_follow_up_outcome_for_job_emits_notification():
    job = AgentJob(
        id=uuid4(),
        user_id=uuid4(),
        name="Deep Dive Follow-up",
        goal="Investigate the accepted signal",
        job_type="research",
        status="failed",
        error="Compilation failed in verification step.",
    )
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=job.user_id,
        item_type="document",
        item_key="doc-7",
        title="Accepted note",
        status="accepted",
        customer="Acme",
        follow_up_job_id=job.id,
        follow_up_launch_status="launched",
        follow_up_recommendation_key="single_research_job",
    )
    session = _FakeInboxOutcomeSession([item])

    updated = await sync_follow_up_outcome_for_job(session, job)

    assert updated == 1
    assert len(session.notifications) == 1
    notification = session.notifications[0]
    assert notification.notification_type == NotificationType.FOLLOW_UP_OUTCOME_ALERT
    assert notification.priority == "high"
    assert notification.related_entity_id == item.id
    assert notification.data["follow_up_outcome_status"] == "failed"
    assert notification.data["inbox_item_id"] == str(item.id)
    assert notification.action_url == f"/autonomous-agents?tab=inbox&inbox={item.id}"


@pytest.mark.asyncio
async def test_sync_follow_up_outcome_for_job_emits_profile_deep_link_notification():
    profile = DomainResearchProfile(
        id=uuid4(),
        user_id=uuid4(),
        title="Compiler Frontier",
        domain="Compiler",
        objective="Track compiler opportunities",
    )
    profile.latest_summary = {
        "opportunities": [
            {
                "opportunity_id": "opp-profile-1",
                "title": "Compiler hotspot",
                "child_job_ids": [],
            }
        ]
    }
    job = AgentJob(
        id=uuid4(),
        user_id=profile.user_id,
        name="Deep Dive Follow-up",
        goal="Investigate compiler hotspot",
        job_type="research",
        status="completed",
        completed_at=datetime.now(timezone.utc),
        results={"summary": "Validated the hotspot and documented next steps."},
        config={
            "domain_research_follow_up": {
                "idea": {
                    "opportunity_id": "opp-profile-1",
                    "autonomous_origin": {
                        "source_kind": "profile",
                        "source_id": str(profile.id),
                        "opportunity_id": "opp-profile-1",
                    },
                }
            }
        },
    )
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=profile.user_id,
        item_type="document",
        item_key="doc-profile-link-1",
        title="Accepted note",
        status="accepted",
        follow_up_job_id=job.id,
        follow_up_launch_status="launched",
    )
    session = _FakeInboxOutcomeSession([item], profile=profile)

    updated = await sync_follow_up_outcome_for_job(session, job)

    assert updated == 1
    notification = session.notifications[0]
    assert notification.action_url == (
        f"/autonomous-agents?tab=domain&profileId={profile.id}&opportunityId=opp-profile-1"
    )
    assert notification.data["origin_source_kind"] == "profile"
    assert notification.data["origin_source_id"] == str(profile.id)
    assert notification.data["origin_opportunity_id"] == "opp-profile-1"
    assert notification.data["follow_up_last_job_id"] == str(job.id)


@pytest.mark.asyncio
async def test_sync_follow_up_outcome_for_job_emits_portfolio_deep_link_notification():
    portfolio = ResearchPortfolio(
        id=uuid4(),
        user_id=uuid4(),
        title="Scientific Fleet",
        objective="Track scientific opportunities",
    )
    portfolio.opportunities = [
        {
            "opportunity_id": "opp-portfolio-1",
            "title": "Prefetch gap",
            "child_job_ids": [],
        }
    ]
    job = AgentJob(
        id=uuid4(),
        user_id=portfolio.user_id,
        name="Deep Dive Follow-up",
        goal="Investigate prefetch gap",
        job_type="research",
        status="failed",
        completed_at=datetime.now(timezone.utc),
        error="Verification failed in the benchmark harness.",
        config={
            "domain_research_follow_up": {
                "idea": {
                    "opportunity_id": "opp-portfolio-1",
                    "autonomous_origin": {
                        "source_kind": "portfolio",
                        "source_id": str(portfolio.id),
                        "opportunity_id": "opp-portfolio-1",
                    },
                }
            }
        },
    )
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=portfolio.user_id,
        item_type="document",
        item_key="doc-portfolio-link-1",
        title="Accepted note",
        status="accepted",
        follow_up_job_id=job.id,
        follow_up_launch_status="launched",
    )
    session = _FakeInboxOutcomeSession([item], portfolio=portfolio)

    updated = await sync_follow_up_outcome_for_job(session, job)

    assert updated == 1
    notification = session.notifications[0]
    assert notification.action_url == (
        f"/autonomous-agents?tab=fleet&fleetId={portfolio.id}&opportunityId=opp-portfolio-1"
    )
    assert notification.data["origin_source_kind"] == "portfolio"
    assert notification.data["origin_source_id"] == str(portfolio.id)
    assert notification.data["origin_opportunity_id"] == "opp-portfolio-1"
    assert notification.data["follow_up_last_job_id"] == str(job.id)


@pytest.mark.asyncio
async def test_sync_follow_up_outcome_for_job_projects_completed_profile_opportunity():
    profile = DomainResearchProfile(
        id=uuid4(),
        user_id=uuid4(),
        title="Compiler Frontier",
        domain="Compiler",
        objective="Track compiler opportunities",
    )
    profile.latest_summary = {
        "opportunities": [
            {
                "opportunity_id": "opp-profile-1",
                "title": "Compiler hotspot",
                "child_job_ids": [],
                "autonomous_origin": {
                    "source_kind": "profile",
                    "source_id": str(profile.id),
                    "opportunity_id": "opp-profile-1",
                },
            }
        ]
    }
    job = AgentJob(
        id=uuid4(),
        user_id=profile.user_id,
        name="Deep Dive Follow-up",
        goal="Investigate compiler hotspot",
        job_type="research",
        status="completed",
        completed_at=datetime.now(timezone.utc),
        results={"summary": "Validated the hotspot and documented next steps."},
        config={
            "domain_research_follow_up": {
                "idea": {
                    "opportunity_id": "opp-profile-1",
                    "autonomous_origin": {
                        "source_kind": "profile",
                        "source_id": str(profile.id),
                        "opportunity_id": "opp-profile-1",
                    },
                }
            }
        },
    )
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=profile.user_id,
        item_type="document",
        item_key="doc-profile-1",
        title="Accepted note",
        status="accepted",
        follow_up_job_id=job.id,
        follow_up_launch_status="launched",
    )

    updated = await sync_follow_up_outcome_for_job(
        _FakeInboxOutcomeSession([item], profile=profile),
        job,
    )

    assert updated == 1
    row = profile.latest_summary["opportunities"][0]
    assert row["follow_up_outcome_status"] == "completed"
    assert row["follow_up_outcome_summary"] == "Validated the hotspot and documented next steps."
    assert row["follow_up_last_job_id"] == str(job.id)
    assert row["last_decision_type"] == "follow_up_completed"
    assert row["last_decision_reason_code"] == "follow_up_completed"


@pytest.mark.asyncio
async def test_sync_follow_up_outcome_for_job_projects_failed_portfolio_opportunity():
    portfolio = ResearchPortfolio(
        id=uuid4(),
        user_id=uuid4(),
        title="Scientific Fleet",
        objective="Track scientific opportunities",
    )
    portfolio.opportunities = [
        {
            "opportunity_id": "opp-portfolio-1",
            "title": "Prefetch gap",
            "child_job_ids": ["job-old"],
        }
    ]
    job = AgentJob(
        id=uuid4(),
        user_id=portfolio.user_id,
        name="Deep Dive Follow-up",
        goal="Investigate prefetch gap",
        job_type="research",
        status="failed",
        completed_at=datetime.now(timezone.utc),
        error="Verification failed in the benchmark harness.",
        config={
            "domain_research_follow_up": {
                "idea": {
                    "opportunity_id": "opp-portfolio-1",
                    "autonomous_origin": {
                        "source_kind": "portfolio",
                        "source_id": str(portfolio.id),
                        "opportunity_id": "opp-portfolio-1",
                    },
                }
            }
        },
    )
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=portfolio.user_id,
        item_type="document",
        item_key="doc-portfolio-1",
        title="Accepted note",
        status="accepted",
        follow_up_job_id=job.id,
        follow_up_launch_status="launched",
    )

    updated = await sync_follow_up_outcome_for_job(
        _FakeInboxOutcomeSession([item], portfolio=portfolio),
        job,
    )

    assert updated == 1
    row = portfolio.opportunities[0]
    assert row["follow_up_outcome_status"] == "failed"
    assert row["follow_up_outcome_summary"] == "Verification failed in the benchmark harness."
    assert row["follow_up_last_job_id"] == str(job.id)
    assert row["last_decision_type"] == "follow_up_failed"


@pytest.mark.asyncio
async def test_recompute_profile_rewards_completed_follow_up_outcomes():
    session = _FakeProfileSession(
        [
            _FakeExecuteResult(rows=[("accepted", "Latency regression note", "API latency regression found")]),
            _FakeExecuteResult(
                rows=[
                    ("single_research_job", "launched", "approved_launch", "document", "completed"),
                    ("deep_dive_chain", "launched", "approved_launch", "document", "failed"),
                ]
            ),
            _FakeExecuteResult(scalar=None),
        ]
    )

    profile = await research_monitor_profile_service.recompute_profile(
        db=session,
        user_id=uuid4(),
        customer="Acme",
    )

    assert profile.recommendation_scores["single_research_job"] > profile.recommendation_scores["deep_dive_chain"]
    assert profile.outcome_counters["completed_follow_up"] == 1
    assert profile.outcome_counters["failed_follow_up"] == 1


def test_build_effectiveness_snapshot_scores_monitor_health_and_recommendations():
    monitor_id = uuid4()
    weak_monitor_id = uuid4()
    user_id = uuid4()
    items = [
        ResearchInboxItem(
            id=uuid4(),
            user_id=user_id,
            job_id=monitor_id,
            customer="Acme",
            item_type="document",
            item_key="doc-1",
            title="Accepted success",
            status="accepted",
            follow_up_policy_mode="auto_launch_safe",
            follow_up_decision="auto_launched",
            follow_up_launch_status="launched",
            follow_up_recommendation_key="deep_dive_chain",
            follow_up_outcome_status="completed",
        ),
        ResearchInboxItem(
            id=uuid4(),
            user_id=user_id,
            job_id=monitor_id,
            customer="Acme",
            item_type="document",
            item_key="doc-2",
            title="Accepted success again",
            status="accepted",
            follow_up_policy_mode="auto_launch_safe",
            follow_up_decision="auto_launched",
            follow_up_launch_status="launched",
            follow_up_recommendation_key="deep_dive_chain",
            follow_up_outcome_status="completed",
        ),
        ResearchInboxItem(
            id=uuid4(),
            user_id=user_id,
            job_id=monitor_id,
            customer="Acme",
            item_type="document",
            item_key="doc-3",
            title="Rejected signal",
            status="rejected",
        ),
        ResearchInboxItem(
            id=uuid4(),
            user_id=user_id,
            job_id=weak_monitor_id,
            customer="Beta",
            item_type="document",
            item_key="doc-4",
            title="Weak accepted signal",
            status="accepted",
            follow_up_policy_mode="manual_only",
            follow_up_decision="manual",
            follow_up_launch_status="blocked",
            follow_up_recommendation_key="single_research_job",
        ),
        ResearchInboxItem(
            id=uuid4(),
            user_id=user_id,
            job_id=weak_monitor_id,
            customer="Beta",
            item_type="document",
            item_key="doc-5",
            title="Another rejected signal",
            status="rejected",
        ),
    ]

    snapshot = research_monitor_profile_service.build_effectiveness_snapshot(
        items=items,
        jobs_by_id={
            monitor_id: AgentJob(
                id=monitor_id,
                user_id=user_id,
                name="Acme Monitor",
                goal="Monitor Acme updates",
                job_type="monitor",
                status="completed",
                results={
                    "follow_up_policy_history": [
                        {
                            "id": "history-1",
                            "at": "2026-03-16T09:00:00Z",
                            "actor_user_id": str(user_id),
                            "change_source": "guided_recommendation",
                            "previous_follow_up_autonomy": {
                                "mode": "manual_only",
                                "allowed_recommendations": ["deep_dive_chain", "single_research_job"],
                            },
                            "next_follow_up_autonomy": {
                                "mode": "auto_launch_safe",
                                "allowed_recommendations": ["deep_dive_chain", "single_research_job"],
                            },
                            "analytics_context": {"health_bucket": "strong"},
                        }
                    ],
                    "autonomy_budget_history": [
                        {
                            "id": "budget-1",
                            "at": "2026-03-17T08:00:00Z",
                            "actor_user_id": str(user_id),
                            "change_source": "customer_rebalance_guidance",
                            "change_reason": "Customer rebalance guidance for Acme",
                            "previous_autonomy_budget": {
                                "auto_launch_limit_24h": 2,
                                "approval_queue_limit_24h": 5,
                                "alert_limit_24h": 3,
                                "queue_backlog_cap": 7,
                            },
                            "next_autonomy_budget": {
                                "auto_launch_limit_24h": 3,
                                "approval_queue_limit_24h": 6,
                                "alert_limit_24h": 4,
                                "queue_backlog_cap": 8,
                            },
                            "guidance_context": {"customer": "Acme"},
                        }
                    ],
                },
            ),
            weak_monitor_id: AgentJob(
                id=weak_monitor_id,
                user_id=user_id,
                name="Beta Watch",
                goal="Monitor Beta updates",
                job_type="monitor",
                status="completed",
            ),
        },
    )

    assert snapshot["totals"]["total_monitors"] == 2
    assert snapshot["totals"]["strong_monitors"] == 1
    assert snapshot["totals"]["weak_monitors"] == 1
    assert snapshot["recommendations"][0]["recommendation_key"] == "deep_dive_chain"
    assert snapshot["recommendations"][0]["score_trend"] == "positive"

    strong_monitor = next(row for row in snapshot["monitors"] if row["monitor_name"] == "Acme Monitor")
    weak_monitor = next(row for row in snapshot["monitors"] if row["monitor_name"] == "Beta Watch")

    assert strong_monitor["health_bucket"] == "strong"
    assert strong_monitor["follow_up_completed_count"] == 2
    assert strong_monitor["top_recommendations"][0]["recommendation_key"] == "deep_dive_chain"
    assert strong_monitor["recommended_policy_mode"] == "auto_launch_safe"
    assert strong_monitor["policy_history_count"] == 1
    assert strong_monitor["budget_history_count"] == 1
    assert strong_monitor["latest_budget_change_source"] == "customer_rebalance_guidance"
    assert strong_monitor["latest_policy_change_source"] == "guided_recommendation"
    assert strong_monitor["recent_policy_history"][0]["id"] == "history-1"
    assert "deep_dive_chain" in strong_monitor["recommended_allowed_recommendations"]
    assert weak_monitor["health_bucket"] == "weak"
    assert weak_monitor["blocked_count"] == 1
    assert "manual_only" in weak_monitor["policy_mode_counts"]
    assert weak_monitor["recommended_policy_mode"] == "manual_only"

    acme_customer = next(row for row in snapshot["customers"] if row["customer"] == "Acme")
    beta_customer = next(row for row in snapshot["customers"] if row["customer"] == "Beta")

    assert acme_customer["monitor_count"] == 1
    assert acme_customer["strong_monitor_count"] == 1
    assert acme_customer["portfolio_status"] == "normal"
    assert acme_customer["auto_launch_used_24h"] == 2
    assert acme_customer["top_launch_monitors"][0]["monitor_name"] == "Acme Monitor"

    assert beta_customer["monitor_count"] == 1
    assert beta_customer["weak_monitor_count"] == 1
    assert beta_customer["blocked_count"] == 1
    assert beta_customer["portfolio_status"] == "normal"
    assert beta_customer["top_backlog_monitors"] == []


def test_customer_rebalance_guidance_identifies_pressure_and_relief_monitors():
    customer_row = {
        "customer": "Acme",
        "portfolio_status": "monitor_throttled",
        "customer_budget_throttle_state": "manual_only_clamped",
    }
    monitor_rows = [
        {
            "monitor_job_id": uuid4(),
            "monitor_name": "Pressure Monitor",
            "customer": "Acme",
            "health_bucket": "weak",
            "accepted_count": 5,
            "budget_usage": {
                "auto_launch_count_24h": 3,
                "approval_queue_count_24h": 5,
                "alert_count_24h": 3,
                "queue_backlog_count": 7,
            },
            "autonomy_budget": {
                "auto_launch_limit_24h": 3,
                "approval_queue_limit_24h": 6,
                "alert_limit_24h": 4,
                "queue_backlog_cap": 8,
            },
        },
        {
            "monitor_job_id": uuid4(),
            "monitor_name": "Relief Monitor",
            "customer": "Acme",
            "health_bucket": "strong",
            "accepted_count": 4,
            "budget_usage": {
                "auto_launch_count_24h": 0,
                "approval_queue_count_24h": 1,
                "alert_count_24h": 0,
                "queue_backlog_count": 1,
            },
            "autonomy_budget": {
                "auto_launch_limit_24h": 3,
                "approval_queue_limit_24h": 6,
                "alert_limit_24h": 4,
                "queue_backlog_cap": 8,
            },
        },
    ]

    status, reasons, summary, changes = research_monitor_profile_service._build_customer_rebalance_guidance(
        customer_row=customer_row,
        monitor_rows=monitor_rows,
    )

    assert status == "actionable"
    assert "Pressure Monitor" in summary
    assert len(changes) == 2
    pressure_change = next(change for change in changes if change["monitor_name"] == "Pressure Monitor")
    relief_change = next(change for change in changes if change["monitor_name"] == "Relief Monitor")
    assert pressure_change["proposed_budget"]["auto_launch_limit_24h"] < pressure_change["current_budget"]["auto_launch_limit_24h"]
    assert relief_change["proposed_budget"]["auto_launch_limit_24h"] > relief_change["current_budget"]["auto_launch_limit_24h"]


def test_build_effectiveness_snapshot_evaluates_policy_change_before_and_after():
    monitor_id = uuid4()
    user_id = uuid4()
    changed_at = datetime(2026, 3, 16, 9, 0, tzinfo=timezone.utc)
    items = [
        ResearchInboxItem(
            id=uuid4(),
            user_id=user_id,
            job_id=monitor_id,
            customer="Acme",
            item_type="document",
            item_key="before-1",
            title="Before policy success",
            status="accepted",
            follow_up_policy_mode="manual_only",
            follow_up_decision="manual",
            follow_up_launch_status="launched",
            follow_up_recommendation_key="deep_dive_chain",
            follow_up_outcome_status="completed",
            updated_at=changed_at - timedelta(days=2),
        ),
        ResearchInboxItem(
            id=uuid4(),
            user_id=user_id,
            job_id=monitor_id,
            customer="Acme",
            item_type="document",
            item_key="before-2",
            title="Before policy blocked",
            status="accepted",
            follow_up_policy_mode="manual_only",
            follow_up_decision="manual",
            follow_up_launch_status="blocked",
            follow_up_recommendation_key="deep_dive_chain",
            updated_at=changed_at - timedelta(days=1),
        ),
        ResearchInboxItem(
            id=uuid4(),
            user_id=user_id,
            job_id=monitor_id,
            customer="Acme",
            item_type="document",
            item_key="after-1",
            title="After policy success",
            status="accepted",
            follow_up_policy_mode="auto_launch_safe",
            follow_up_decision="auto_launched",
            follow_up_launch_status="launched",
            follow_up_recommendation_key="deep_dive_chain",
            follow_up_outcome_status="completed",
            updated_at=changed_at + timedelta(hours=1),
        ),
        ResearchInboxItem(
            id=uuid4(),
            user_id=user_id,
            job_id=monitor_id,
            customer="Acme",
            item_type="document",
            item_key="after-2",
            title="After policy success again",
            status="accepted",
            follow_up_policy_mode="auto_launch_safe",
            follow_up_decision="auto_launched",
            follow_up_launch_status="launched",
            follow_up_recommendation_key="deep_dive_chain",
            follow_up_outcome_status="completed",
            updated_at=changed_at + timedelta(days=1),
        ),
    ]

    snapshot = research_monitor_profile_service.build_effectiveness_snapshot(
        items=items,
        jobs_by_id={
            monitor_id: AgentJob(
                id=monitor_id,
                user_id=user_id,
                name="Acme Monitor",
                goal="Monitor Acme updates",
                job_type="monitor",
                status="completed",
                config={"follow_up_autonomy": {"mode": "auto_launch_safe", "allowed_recommendations": ["deep_dive_chain"]}},
                results={
                    "follow_up_policy_history": [
                        {
                            "id": "history-1",
                            "at": changed_at.isoformat(),
                            "actor_user_id": str(user_id),
                            "change_source": "guided_recommendation",
                            "previous_follow_up_autonomy": {
                                "mode": "manual_only",
                                "allowed_recommendations": ["deep_dive_chain", "single_research_job"],
                            },
                            "next_follow_up_autonomy": {
                                "mode": "auto_launch_safe",
                                "allowed_recommendations": ["deep_dive_chain"],
                            },
                            "analytics_context": {"health_bucket": "strong"},
                            "evaluation_target_count": 2,
                            "evaluation_state": "active",
                        }
                    ]
                },
            ),
        },
    )

    monitor = snapshot["monitors"][0]
    assert monitor["latest_policy_evaluation_status"] == "improving"
    assert monitor["latest_policy_evaluation_sample_count"] == 2
    assert monitor["recent_policy_history"][0]["evaluation_status"] == "improving"
    assert monitor["recent_policy_history"][0]["after_counts"]["follow_up_completed_count"] == 2
    assert monitor["recent_policy_history"][0]["before_counts"]["blocked_count"] == 1
    assert monitor["recent_policy_history"][0]["delta_counts"]["follow_up_completed_count"] == 1
    assert any("Completion rate improved" in reason for reason in monitor["latest_policy_evaluation_reasons"])
    assert monitor["recent_policy_history"][0]["sample_items"][0]["period"] == "before"
    assert monitor["recent_policy_history"][0]["sample_items"][-1]["period"] == "after"


@pytest.mark.asyncio
async def test_build_customer_budget_snapshot_uses_customer_profile_caps():
    user_id = uuid4()
    profile = ResearchMonitorProfile(
        id=uuid4(),
        user_id=user_id,
        customer="Acme",
        customer_budget_config={
            "auto_launch_limit_24h": 0,
            "approval_queue_limit_24h": 1,
            "alert_limit_24h": 0,
            "queue_backlog_cap": 1,
        },
    )
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=user_id,
        customer="Acme",
        item_type="document",
        item_key="doc-1",
        title="Queued follow-up",
        status="accepted",
        follow_up_decision="queued_for_approval",
        follow_up_launch_status="pending_approval",
    )
    db = _FakeProfileSession(
        [
            _FakeExecuteResult(scalar=profile),
            _FakeExecuteResult(rows=[item]),
            _FakeExecuteResult(rows=[]),
        ]
    )

    snapshot = await research_monitor_profile_service.build_customer_budget_snapshot(
        db=db,
        user_id=user_id,
        customer="Acme",
    )

    assert snapshot["customer"] == "Acme"
    assert snapshot["customer_budget"]["approval_queue_limit_24h"] == 1
    assert snapshot["customer_budget_usage"]["approval_queue_count_24h"] == 1
    assert snapshot["customer_budget_usage"]["queue_backlog_count"] == 1
    assert snapshot["customer_budget_throttle_state"] == "manual_only_clamped"


def test_build_policy_simulation_snapshot_estimates_policy_impact():
    monitor_id = uuid4()
    user_id = uuid4()
    monitor_job = AgentJob(
        id=monitor_id,
        user_id=user_id,
        name="Acme Monitor",
        goal="Monitor Acme updates",
        job_type="monitor",
        status="completed",
        config={"follow_up_autonomy": {"mode": "manual_only", "allowed_recommendations": ["deep_dive_chain", "single_research_job"]}},
    )
    items = [
        ResearchInboxItem(
            id=uuid4(),
            user_id=user_id,
            job_id=monitor_id,
            customer="Acme",
            item_type="document",
            item_key="doc-1",
            title="Document signal",
            summary="Deep dive on new customer launch",
            status="accepted",
        ),
        ResearchInboxItem(
            id=uuid4(),
            user_id=user_id,
            job_id=monitor_id,
            customer="Acme",
            item_type="arxiv",
            item_key="paper-1",
            title="Paper signal",
            summary="A paper with implementation guidance",
            status="accepted",
        ),
    ]

    snapshot = research_monitor_profile_service.build_policy_simulation_snapshot(
        monitor_job=monitor_job,
        items=items,
        proposed_policy={"mode": "auto_launch_safe", "allowed_recommendations": ["deep_dive_chain", "single_research_job"]},
        learning_profile={
            "token_scores": {},
            "phrase_scores": {},
            "recommendation_scores": {},
            "source_type_scores": {},
        },
        history_limit=25,
    )

    assert snapshot["current_policy"]["mode"] == "manual_only"
    assert snapshot["proposed_policy"]["mode"] == "auto_launch_safe"
    assert snapshot["baseline_counts"]["manual_only_count"] == 2
    assert snapshot["simulated_counts"]["auto_launch_safe_count"] == 2
    assert snapshot["delta_counts"]["auto_launch_safe_count"] == 2
    assert snapshot["sample_items"]


@pytest.mark.asyncio
async def test_relaunch_inbox_follow_up_relaunches_failed_safe_recommendation(monkeypatch):
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=uuid4(),
        item_type="document",
        item_key="doc-8",
        title="Failed follow-up",
        status="accepted",
        discovered_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        follow_up_launch_status="launched",
        follow_up_outcome_status="failed",
        follow_up_recommendation_key="deep_dive_chain",
    )

    async def fake_create_job_from_chain(request, db, current_user):
        return SimpleNamespace(id=uuid4(), chain_definition_id=request.chain_definition_id)

    class _RelaunchDb:
        async def get(self, model, lookup_id):
            if model is ResearchInboxItem and lookup_id == item.id:
                return item
            return None

        async def commit(self):
            return None

        async def refresh(self, _obj):
            return None

    monkeypatch.setattr(agent_jobs_endpoint, "_load_follow_up_learning_profile", AsyncMock(return_value={
        "token_scores": {},
        "phrase_scores": {},
        "recommendation_scores": {},
        "source_type_scores": {},
        "outcome_counters": {},
    }))
    monkeypatch.setattr(agent_jobs_endpoint, "_launch_follow_up_action", AsyncMock(return_value=SimpleNamespace(id=uuid4())))

    response = await research_inbox_endpoint.relaunch_inbox_follow_up(
        str(item.id),
        ResearchInboxFollowUpRelaunchRequest(operator_note="Retry this bounded follow-up."),
        current_user=SimpleNamespace(id=item.user_id),
        db=_RelaunchDb(),
    )

    assert response.follow_up_launch_status == "launched"
    assert response.follow_up_job_id is not None
    assert response.follow_up_outcome_status is None


@pytest.mark.asyncio
async def test_relaunch_inbox_follow_up_projects_relaunch_to_profile_opportunity(monkeypatch):
    profile = DomainResearchProfile(
        id=uuid4(),
        user_id=uuid4(),
        title="Compiler Frontier",
        domain="Compiler",
        objective="Track compiler opportunities",
    )
    profile.latest_summary = {
        "opportunities": [
            {
                "opportunity_id": "opp-profile-relaunch-1",
                "title": "Compiler hotspot",
                "child_job_ids": ["job-old"],
                "follow_up_outcome_status": "failed",
                "follow_up_outcome_recorded_at": "2026-03-25T12:00:00Z",
                "follow_up_outcome_summary": "Benchmark verification failed.",
                "follow_up_last_job_id": "job-old",
                "follow_up_launched_at": "2026-03-24T12:00:00Z",
                "last_decision_type": "follow_up_failed",
                "last_decision_reason_code": "follow_up_failed",
                "autonomy_state": "cooldown",
                "stage": "accepted",
            }
        ]
    }
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=profile.user_id,
        item_type="document",
        item_key="doc-profile-relaunch-1",
        title="Failed follow-up",
        status="accepted",
        discovered_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        follow_up_launch_status="launched",
        follow_up_outcome_status="failed",
        follow_up_recommendation_key="deep_dive_chain",
    )
    launched_job = SimpleNamespace(
        id=uuid4(),
        config={
            "domain_research_follow_up": {
                "idea": {
                    "opportunity_id": "opp-profile-relaunch-1",
                    "autonomous_origin": {
                        "source_kind": "profile",
                        "source_id": str(profile.id),
                        "opportunity_id": "opp-profile-relaunch-1",
                    },
                }
            }
        },
    )

    class _RelaunchDb:
        async def get(self, model, lookup_id):
            if model is ResearchInboxItem and lookup_id == item.id:
                return item
            if model is DomainResearchProfile and lookup_id == profile.id:
                return profile
            return None

        async def commit(self):
            return None

        async def refresh(self, _obj):
            return None

    monkeypatch.setattr(agent_jobs_endpoint, "_load_follow_up_learning_profile", AsyncMock(return_value={
        "token_scores": {},
        "phrase_scores": {},
        "recommendation_scores": {},
        "source_type_scores": {},
        "outcome_counters": {},
    }))
    monkeypatch.setattr(agent_jobs_endpoint, "_launch_follow_up_action", AsyncMock(return_value=launched_job))

    response = await research_inbox_endpoint.relaunch_inbox_follow_up(
        str(item.id),
        ResearchInboxFollowUpRelaunchRequest(operator_note="Retry this bounded follow-up."),
        current_user=SimpleNamespace(id=item.user_id),
        db=_RelaunchDb(),
    )

    assert response.follow_up_launch_status == "launched"
    row = profile.latest_summary["opportunities"][0]
    assert row["follow_up_last_job_id"] == str(launched_job.id)
    assert row["follow_up_outcome_status"] is None
    assert row["follow_up_outcome_recorded_at"] is None
    assert row["follow_up_outcome_summary"] is None
    assert row["last_decision_type"] == "follow_up_launched"
    assert row["last_decision_reason_code"] == "follow_up_relaunched"
    assert row["autonomy_state"] == "active"
    assert str(launched_job.id) in row["child_job_ids"]
    assert row["child_job_ids"].count(str(launched_job.id)) == 1


@pytest.mark.asyncio
async def test_relaunch_inbox_follow_up_projects_relaunch_to_portfolio_opportunity(monkeypatch):
    portfolio = ResearchPortfolio(
        id=uuid4(),
        user_id=uuid4(),
        title="Scientific Fleet",
        objective="Track scientific opportunities",
    )
    portfolio.opportunities = [
        {
            "opportunity_id": "opp-portfolio-relaunch-1",
            "title": "Prefetch gap",
            "child_job_ids": ["job-old"],
            "follow_up_outcome_status": "cancelled",
            "follow_up_outcome_recorded_at": "2026-03-25T12:00:00Z",
            "follow_up_outcome_summary": "Operator cancelled verification.",
            "follow_up_last_job_id": "job-old",
            "follow_up_launched_at": "2026-03-24T12:00:00Z",
            "last_decision_type": "follow_up_cancelled",
            "last_decision_reason_code": "follow_up_cancelled",
            "autonomy_state": "cooldown",
            "stage": "accepted",
        }
    ]
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=portfolio.user_id,
        item_type="document",
        item_key="doc-portfolio-relaunch-1",
        title="Cancelled follow-up",
        status="accepted",
        discovered_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        follow_up_launch_status="launched",
        follow_up_outcome_status="cancelled",
        follow_up_recommendation_key="deep_dive_chain",
    )
    launched_job = SimpleNamespace(
        id=uuid4(),
        config={
            "domain_research_follow_up": {
                "idea": {
                    "opportunity_id": "opp-portfolio-relaunch-1",
                    "autonomous_origin": {
                        "source_kind": "portfolio",
                        "source_id": str(portfolio.id),
                        "opportunity_id": "opp-portfolio-relaunch-1",
                    },
                }
            }
        },
    )

    class _RelaunchDb:
        async def get(self, model, lookup_id):
            if model is ResearchInboxItem and lookup_id == item.id:
                return item
            if model is ResearchPortfolio and lookup_id == portfolio.id:
                return portfolio
            return None

        async def commit(self):
            return None

        async def refresh(self, _obj):
            return None

    monkeypatch.setattr(agent_jobs_endpoint, "_load_follow_up_learning_profile", AsyncMock(return_value={
        "token_scores": {},
        "phrase_scores": {},
        "recommendation_scores": {},
        "source_type_scores": {},
        "outcome_counters": {},
    }))
    monkeypatch.setattr(agent_jobs_endpoint, "_launch_follow_up_action", AsyncMock(return_value=launched_job))

    response = await research_inbox_endpoint.relaunch_inbox_follow_up(
        str(item.id),
        ResearchInboxFollowUpRelaunchRequest(operator_note="Retry this bounded follow-up."),
        current_user=SimpleNamespace(id=item.user_id),
        db=_RelaunchDb(),
    )

    assert response.follow_up_launch_status == "launched"
    row = portfolio.opportunities[0]
    assert row["follow_up_last_job_id"] == str(launched_job.id)
    assert row["follow_up_outcome_status"] is None
    assert row["follow_up_outcome_recorded_at"] is None
    assert row["follow_up_outcome_summary"] is None
    assert row["last_decision_type"] == "follow_up_launched"
    assert row["last_decision_reason_code"] == "follow_up_relaunched"
    assert row["autonomy_state"] == "active"
    assert str(launched_job.id) in row["child_job_ids"]


@pytest.mark.asyncio
async def test_bulk_relaunch_inbox_follow_up_relaunches_multiple_items(monkeypatch):
    user_id = uuid4()
    item_a = ResearchInboxItem(
        id=uuid4(),
        user_id=user_id,
        item_type="follow_up_recommendation",
        item_key="item-a",
        title="Failed follow-up A",
        status="accepted",
        discovered_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        follow_up_launch_status="launched",
        follow_up_outcome_status="failed",
        follow_up_recommendation_key="deep_dive_chain",
    )
    item_b = ResearchInboxItem(
        id=uuid4(),
        user_id=user_id,
        item_type="follow_up_recommendation",
        item_key="item-b",
        title="Cancelled follow-up B",
        status="accepted",
        discovered_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        follow_up_launch_status="launched",
        follow_up_outcome_status="cancelled",
        follow_up_recommendation_key="deep_dive_chain",
    )

    class _BulkRelaunchDb:
        def __init__(self, items):
            self.items = {item.id: item for item in items}
            self.commits = 0
            self.refreshed = []

        async def get(self, model, lookup_id):
            if model is ResearchInboxItem:
                return self.items.get(lookup_id)
            return None

        async def commit(self):
            self.commits += 1

        async def refresh(self, obj):
            self.refreshed.append(obj)

    db = _BulkRelaunchDb([item_a, item_b])
    monkeypatch.setattr(
        research_inbox_endpoint,
        "_relaunch_follow_up_inbox_item",
        AsyncMock(side_effect=[
            SimpleNamespace(follow_up_job_id=uuid4()),
            SimpleNamespace(follow_up_job_id=uuid4()),
        ]),
    )

    response = await research_inbox_endpoint.bulk_relaunch_inbox_follow_up(
        ResearchInboxBulkFollowUpRelaunchRequest(item_ids=[item_a.id, item_b.id], operator_note="Retry in bulk."),
        current_user=SimpleNamespace(id=user_id),
        db=db,
    )

    assert response.requested_count == 2
    assert response.applied == 2
    assert response.failed == 0
    assert [row.item_id for row in response.results] == [item_a.id, item_b.id]
    assert all(row.ok for row in response.results)
    assert all(row.follow_up_job_id is not None for row in response.results)
    assert db.commits == 1


@pytest.mark.asyncio
async def test_bulk_relaunch_inbox_follow_up_reports_partial_failures(monkeypatch):
    user_id = uuid4()
    item_ok = ResearchInboxItem(
        id=uuid4(),
        user_id=user_id,
        item_type="follow_up_recommendation",
        item_key="item-ok",
        title="Failed follow-up",
        status="accepted",
        discovered_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        follow_up_launch_status="launched",
        follow_up_outcome_status="failed",
        follow_up_recommendation_key="deep_dive_chain",
    )
    item_bad = ResearchInboxItem(
        id=uuid4(),
        user_id=user_id,
        item_type="follow_up_recommendation",
        item_key="item-bad",
        title="Completed follow-up",
        status="accepted",
        discovered_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        follow_up_launch_status="launched",
        follow_up_outcome_status="completed",
        follow_up_recommendation_key="deep_dive_chain",
    )

    class _BulkRelaunchDb:
        def __init__(self, items):
            self.items = {item.id: item for item in items}
            self.commits = 0

        async def get(self, model, lookup_id):
            if model is ResearchInboxItem:
                return self.items.get(lookup_id)
            return None

        async def commit(self):
            self.commits += 1

        async def refresh(self, _obj):
            return None

    db = _BulkRelaunchDb([item_ok, item_bad])
    monkeypatch.setattr(
        research_inbox_endpoint,
        "_relaunch_follow_up_inbox_item",
        AsyncMock(side_effect=[
            SimpleNamespace(follow_up_job_id=uuid4()),
            agent_jobs_endpoint.HTTPException(status_code=400, detail="Only failed or cancelled launched follow-ups can be relaunched"),
        ]),
    )

    response = await research_inbox_endpoint.bulk_relaunch_inbox_follow_up(
        ResearchInboxBulkFollowUpRelaunchRequest(item_ids=[item_ok.id, item_bad.id]),
        current_user=SimpleNamespace(id=user_id),
        db=db,
    )

    assert response.requested_count == 2
    assert response.applied == 1
    assert response.failed == 1
    assert response.results[0].ok is True
    assert response.results[1].ok is False
    assert response.results[1].error == "Only failed or cancelled launched follow-ups can be relaunched"


@pytest.mark.asyncio
async def test_serialize_research_inbox_item_exposes_profile_origin_fields():
    follow_up_job = AgentJob(
        id=uuid4(),
        user_id=uuid4(),
        name="Profile Follow-up",
        goal="Investigate compiler hotspot",
        job_type="research",
        status="completed",
        config={
            "domain_research_follow_up": {
                "idea": {
                    "opportunity_id": "opp-profile-serialize-1",
                    "autonomous_origin": {
                        "source_kind": "profile",
                        "source_id": "profile-serialize-1",
                        "opportunity_id": "opp-profile-serialize-1",
                    },
                }
            }
        },
    )
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=uuid4(),
        item_type="document",
        item_key="doc-profile-serialize-1",
        title="Profile serializer source",
        status="accepted",
        discovered_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        follow_up_job_id=follow_up_job.id,
    )

    response = await research_inbox_endpoint._serialize_research_inbox_item(
        item,
        _FakeInboxSerializeDb({follow_up_job.id: follow_up_job}),
        follow_up_job=follow_up_job,
    )

    assert str(response.follow_up_last_job_id) == str(follow_up_job.id)
    assert response.origin_source_kind == "profile"
    assert response.origin_source_id == "profile-serialize-1"
    assert response.origin_opportunity_id == "opp-profile-serialize-1"


@pytest.mark.asyncio
async def test_serialize_research_inbox_item_exposes_portfolio_origin_fields():
    follow_up_job = AgentJob(
        id=uuid4(),
        user_id=uuid4(),
        name="Portfolio Follow-up",
        goal="Investigate fleet hotspot",
        job_type="research",
        status="failed",
        config={
            "domain_research_follow_up": {
                "idea": {
                    "opportunity_id": "opp-portfolio-serialize-1",
                    "autonomous_origin": {
                        "source_kind": "portfolio",
                        "source_id": "portfolio-serialize-1",
                        "opportunity_id": "opp-portfolio-serialize-1",
                    },
                }
            }
        },
    )
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=uuid4(),
        item_type="document",
        item_key="doc-portfolio-serialize-1",
        title="Portfolio serializer source",
        status="accepted",
        discovered_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        follow_up_job_id=follow_up_job.id,
    )

    response = await research_inbox_endpoint._serialize_research_inbox_item(
        item,
        _FakeInboxSerializeDb({follow_up_job.id: follow_up_job}),
        follow_up_job=follow_up_job,
    )

    assert str(response.follow_up_last_job_id) == str(follow_up_job.id)
    assert response.origin_source_kind == "portfolio"
    assert response.origin_source_id == "portfolio-serialize-1"
    assert response.origin_opportunity_id == "opp-portfolio-serialize-1"


@pytest.mark.asyncio
async def test_serialize_research_inbox_item_leaves_origin_empty_without_linked_job():
    item = ResearchInboxItem(
        id=uuid4(),
        user_id=uuid4(),
        item_type="document",
        item_key="doc-no-origin-serialize-1",
        title="Detached serializer source",
        status="accepted",
        discovered_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    response = await research_inbox_endpoint._serialize_research_inbox_item(
        item,
        _FakeInboxSerializeDb(),
    )

    assert response.follow_up_last_job_id is None
    assert response.origin_source_kind is None
    assert response.origin_source_id is None
    assert response.origin_opportunity_id is None


@pytest.mark.asyncio
async def test_update_monitor_policy_updates_follow_up_autonomy(monkeypatch):
    job = AgentJob(
        id=uuid4(),
        user_id=uuid4(),
        name="Research Inbox Monitor",
        goal="Monitor updates",
        job_type="monitor",
        status="running",
        config={"follow_up_autonomy": {"mode": "manual_only", "allowed_recommendations": ["deep_dive_chain"]}},
        results={
            "execution_strategy": {
                "scheduler_state": {
                    "queue_reason": "execution_failure",
                    "last_scheduled_at": "2026-03-16T09:00:00Z",
                    "last_dispatched_at": "2026-03-16T09:05:00Z",
                }
            }
        },
    )
    captured = {}

    class _PolicyDb:
        async def get(self, model, lookup_id):
            if model is AgentJob and lookup_id == job.id:
                return job
            return None

        async def commit(self):
            return None

    async def _fake_record(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr(research_monitor_profiles_endpoint, "record_autonomy_decision_event", _fake_record)

    response = await research_monitor_profiles_endpoint.update_monitor_policy(
        str(job.id),
        ResearchMonitorPolicyUpdateRequest(
            mode="queue_for_approval",
            allowed_recommendations=["single_research_job"],
            change_source="guided_recommendation",
            change_reason="Monitor performance improved",
            analytics_context={"health_bucket": "mixed", "accepted_count": 3},
        ),
        current_user=SimpleNamespace(id=job.user_id),
        db=_PolicyDb(),
    )

    assert response.monitor_job_id == job.id
    assert response.automation_profile == "balanced"
    assert response.automation_policy["follow_up_review_mode"] == "queue_for_approval"
    assert response.automation_policy["allowed_recommendations"] == ["single_research_job"]
    assert response.effective_policy["follow_up_review_mode"] == "queue_for_approval"
    assert response.follow_up_autonomy.mode == "queue_for_approval"
    assert response.follow_up_autonomy.allowed_recommendations == ["single_research_job"]
    assert response.policy_history_count == 1
    assert response.latest_history_entry is not None
    assert response.latest_history_entry.change_source == "guided_recommendation"
    assert response.latest_history_entry.change_reason == "Monitor performance improved"
    assert job.config["automation_profile"] == "balanced"
    assert job.config["automation_policy"]["follow_up_review_mode"] == "queue_for_approval"
    assert job.config["automation_policy"]["allowed_recommendations"] == ["single_research_job"]
    assert job.config["follow_up_autonomy"]["mode"] == "queue_for_approval"
    assert job.results["follow_up_policy_history"][0]["next_automation_profile"] == "balanced"
    assert (
        job.results["follow_up_policy_history"][0]["next_automation_policy"]["follow_up_review_mode"]
        == "queue_for_approval"
    )
    assert job.results["follow_up_policy_history"][0]["next_follow_up_autonomy"]["mode"] == "queue_for_approval"
    assert captured["reason_label"] == "Guided recommendation"
    assert captured["scheduler_state"] == {
        "queue_reason": "execution_failure",
        "last_scheduled_at": "2026-03-16T09:00:00Z",
        "last_dispatched_at": "2026-03-16T09:05:00Z",
    }


@pytest.mark.asyncio
async def test_rollback_monitor_policy_restores_previous_follow_up_autonomy(monkeypatch):
    job = AgentJob(
        id=uuid4(),
        user_id=uuid4(),
        name="Research Inbox Monitor",
        goal="Monitor updates",
        job_type="monitor",
        status="running",
        config={
            "automation_profile": "max_autonomy",
            "automation_policy": {
                "follow_up_review_mode": "auto_launch_safe",
                "allowed_recommendations": ["deep_dive_chain"],
            },
            "follow_up_autonomy": {"mode": "auto_launch_safe", "allowed_recommendations": ["deep_dive_chain"]},
        },
        results={
            "execution_strategy": {
                "scheduler_state": {
                    "queue_reason": "scheduled_recovery",
                    "last_scheduled_at": "2026-03-16T10:00:00Z",
                    "last_dispatched_at": "2026-03-16T10:05:00Z",
                }
            },
            "follow_up_policy_history": [
                {
                    "id": "history-1",
                    "at": "2026-03-16T09:00:00Z",
                    "actor_user_id": str(uuid4()),
                    "change_source": "guided_recommendation",
                    "previous_automation_profile": "balanced",
                    "previous_automation_policy": {
                        "follow_up_review_mode": "manual_only",
                        "allowed_recommendations": ["deep_dive_chain", "single_research_job"],
                    },
                    "next_automation_profile": "max_autonomy",
                    "next_automation_policy": {
                        "follow_up_review_mode": "auto_launch_safe",
                        "allowed_recommendations": ["deep_dive_chain"],
                    },
                    "analytics_context": {},
                }
            ],
        },
    )
    captured = {}

    class _PolicyDb:
        async def get(self, model, lookup_id):
            if model is AgentJob and lookup_id == job.id:
                return job
            return None

        async def commit(self):
            return None

    async def _fake_record(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr(research_monitor_profiles_endpoint, "record_autonomy_decision_event", _fake_record)

    response = await research_monitor_profiles_endpoint.rollback_monitor_policy(
        str(job.id),
        ResearchMonitorPolicyRollbackRequest(history_entry_id="history-1"),
        current_user=SimpleNamespace(id=job.user_id),
        db=_PolicyDb(),
    )

    assert response.automation_profile == "balanced"
    assert response.automation_policy["follow_up_review_mode"] == "manual_only"
    assert response.effective_policy["follow_up_review_mode"] == "manual_only"
    assert response.follow_up_autonomy.mode == "manual_only"
    assert response.policy_history_count == 2
    assert response.latest_history_entry is not None
    assert response.latest_history_entry.change_source == "rollback"
    assert job.config["automation_profile"] == "balanced"
    assert job.config["automation_policy"]["follow_up_review_mode"] == "manual_only"
    assert job.config["follow_up_autonomy"]["mode"] == "manual_only"
    assert captured["reason_label"] == "Policy rollback"
    assert captured["scheduler_state"] == {
        "queue_reason": "scheduled_recovery",
        "last_scheduled_at": "2026-03-16T10:00:00Z",
        "last_dispatched_at": "2026-03-16T10:05:00Z",
    }


@pytest.mark.asyncio
async def test_update_monitor_budget_omits_malformed_scheduler_state(monkeypatch):
    job = AgentJob(
        id=uuid4(),
        user_id=uuid4(),
        name="Research Inbox Monitor",
        goal="Monitor updates",
        job_type="monitor",
        status="running",
        config={"autonomy_budget": {"auto_launch_limit_24h": 3}},
        results={"execution_strategy": {"scheduler_state": "bad-payload"}},
    )
    captured = {}

    class _BudgetDb:
        async def get(self, model, lookup_id):
            if model is AgentJob and lookup_id == job.id:
                return job
            return None

        async def commit(self):
            return None

    async def _fake_record(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr(research_monitor_profiles_endpoint, "record_autonomy_decision_event", _fake_record)

    response = await research_monitor_profiles_endpoint.update_monitor_budget(
        str(job.id),
        ResearchMonitorBudgetUpdateRequest(
            auto_launch_limit_24h=4,
            approval_queue_limit_24h=6,
            alert_limit_24h=4,
            queue_backlog_cap=8,
            change_source="manual_override",
            change_reason="Increase budget",
        ),
        current_user=SimpleNamespace(id=job.user_id),
        db=_BudgetDb(),
    )

    assert response.monitor_job_id == job.id
    assert captured["reason_label"] == "Manual override"
    assert captured["scheduler_state"] is None


def test_customer_rebalance_evaluation_scores_before_and_after():
    user_id = uuid4()
    monitor_id = uuid4()
    changed_at = datetime(2026, 3, 17, 9, 30, tzinfo=timezone.utc)
    items = [
        ResearchInboxItem(
            id=uuid4(),
            user_id=user_id,
            job_id=monitor_id,
            customer="Beta",
            item_type="document",
            item_key="before-1",
            title="Before pressure",
            status="accepted",
            follow_up_launch_status="blocked",
            follow_up_recommendation_key="single_research_job",
            updated_at=changed_at - timedelta(days=1),
        ),
        ResearchInboxItem(
            id=uuid4(),
            user_id=user_id,
            job_id=monitor_id,
            customer="Beta",
            item_type="document",
            item_key="after-1",
            title="After recovery",
            status="accepted",
            follow_up_launch_status="launched",
            follow_up_outcome_status="completed",
            follow_up_recommendation_key="single_research_job",
            updated_at=changed_at + timedelta(hours=1),
        ),
        ResearchInboxItem(
            id=uuid4(),
            user_id=user_id,
            job_id=monitor_id,
            customer="Beta",
            item_type="document",
            item_key="after-2",
            title="After recovery two",
            status="accepted",
            follow_up_launch_status="launched",
            follow_up_outcome_status="completed",
            follow_up_recommendation_key="single_research_job",
            updated_at=changed_at + timedelta(hours=2),
        ),
    ]
    history_entry = {
        "id": "rebalance-1",
        "at": changed_at,
        "changes": [{"monitor_job_id": monitor_id, "monitor_name": "Beta Watch"}],
        "evaluation_target_count": 3,
    }
    monitor_rows = [
        {
            "monitor_job_id": monitor_id,
            "budget_usage": {
                "auto_launch_count_24h": 0,
                "approval_queue_count_24h": 1,
                "alert_count_24h": 1,
                "queue_backlog_count": 1,
            },
            "budget_throttle_state": "normal",
        }
    ]

    detail = research_monitor_profile_service.build_customer_rebalance_evaluation_detail(
        customer="Beta",
        history_entry=history_entry,
        items=items,
        monitor_rows=monitor_rows,
        jobs_by_id={monitor_id: AgentJob(id=monitor_id, user_id=user_id, name="Beta Watch", goal="", job_type="monitor", status="completed")},
    )

    assert detail["evaluation_status"] in {"improving", "mixed"}
    assert detail["after_counts"]["follow_up_completed_count"] >= detail["before_counts"]["follow_up_completed_count"]
    assert detail["history_entry_id"] == "rebalance-1"


@pytest.mark.asyncio
async def test_apply_customer_rebalance_records_customer_history():
    user_id = uuid4()
    profile = ResearchMonitorProfile(
        id=uuid4(),
        user_id=user_id,
        customer="Beta",
        token_scores={},
        phrase_scores={},
        recommendation_scores={},
        source_type_scores={},
        outcome_counters={},
        customer_budget_config={},
        customer_rebalance_history=[],
    )
    job = AgentJob(
        id=uuid4(),
        user_id=user_id,
        name="Beta Watch",
        goal="Monitor Beta",
        job_type="monitor",
        status="running",
        config={"autonomy_budget": {"auto_launch_limit_24h": 3, "approval_queue_limit_24h": 6, "alert_limit_24h": 4, "queue_backlog_cap": 8}},
    )

    class _ScalarResult:
        def __init__(self, value):
            self._value = value

        def scalar_one_or_none(self):
            return self._value

        def scalars(self):
            return self

        def all(self):
            return [self._value] if self._value is not None else []

    class _Db:
        async def execute(self, stmt):
            text = str(stmt)
            if "research_monitor_profiles" in text:
                return _ScalarResult(profile)
            if "research_inbox_items" in text:
                return _ScalarResult(None)
            if "agent_jobs" in text:
                return _ScalarResult(job)
            return _ScalarResult(None)

        async def get(self, model, lookup_id):
            if model is AgentJob and lookup_id == job.id:
                return job
            return None

        async def flush(self):
            return None

        async def commit(self):
            return None

    async def _fake_preview(**kwargs):
        return {
            "customer": "Beta",
            "guidance_status": "actionable",
            "guidance_summary": "Shift budget headroom from Beta Watch.",
            "guidance_reasons": [],
            "before_capacity": {"auto_launch_limit_24h": 3, "approval_queue_limit_24h": 6, "alert_limit_24h": 4, "queue_backlog_cap": 8},
            "after_capacity": {"auto_launch_limit_24h": 2, "approval_queue_limit_24h": 5, "alert_limit_24h": 3, "queue_backlog_cap": 7},
            "changes": [
                {
                    "monitor_job_id": job.id,
                    "monitor_name": "Beta Watch",
                    "customer": "Beta",
                    "current_budget": {"auto_launch_limit_24h": 3, "approval_queue_limit_24h": 6, "alert_limit_24h": 4, "queue_backlog_cap": 8},
                    "proposed_budget": {"auto_launch_limit_24h": 2, "approval_queue_limit_24h": 5, "alert_limit_24h": 3, "queue_backlog_cap": 7},
                    "delta_budget": {"auto_launch_limit_24h": 1, "approval_queue_limit_24h": 1, "alert_limit_24h": 1, "queue_backlog_cap": 1},
                    "reasons": ["Reduce pressure"],
                }
            ],
        }

    captured = {}

    async def _fake_record(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id=uuid4())

    original_preview = research_monitor_profile_service.build_customer_rebalance_preview
    research_monitor_profile_service.build_customer_rebalance_preview = _fake_preview
    monkeypatch.setattr(research_monitor_profiles_endpoint, "record_autonomy_decision_event", _fake_record)
    try:
        response = await research_monitor_profiles_endpoint.apply_customer_rebalance(
            ResearchMonitorCustomerRebalanceApplyRequest(
                customer="Beta",
                monitor_budget_updates=[
                    ResearchMonitorCustomerRebalanceApplyMonitorRequest(
                        monitor_job_id=job.id,
                        auto_launch_limit_24h=2,
                        approval_queue_limit_24h=5,
                        alert_limit_24h=3,
                        queue_backlog_cap=7,
                    )
                ],
                change_reason="Reduce customer pressure",
            ),
            current_user=SimpleNamespace(id=user_id),
            db=_Db(),
        )
    finally:
        research_monitor_profile_service.build_customer_rebalance_preview = original_preview

    assert response.customer == "Beta"
    assert profile.customer_rebalance_history
    assert profile.customer_rebalance_history[0]["change_reason"] == "Reduce customer pressure"
    assert captured["event_type"] == "customer_rebalanced"
    assert captured["reason_label"] == "Customer rebalance guidance"
    assert captured["scheduler_state"] is None
    assert captured["metadata"] == {"updated_monitor_ids": [str(job.id)]}
