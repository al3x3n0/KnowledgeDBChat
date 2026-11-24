"""Regression coverage for the modular checkpoint-queue HTTP boundary."""

from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import APIRouter

from app.api.endpoints import agent_jobs
from app.modules.autonomy.api.checkpoint_queue import build_checkpoint_queue_api
from app.schemas.agent_job import AgentCheckpointQueueItemResponse


class _Scalars:
    def all(self):
        return []


class _Result:
    def scalars(self):
        return _Scalars()


class _EmptyDb:
    async def execute(self, _query):
        return _Result()


@pytest.mark.asyncio
async def test_checkpoint_queue_returns_an_empty_operator_snapshot():
    response = await agent_jobs.get_checkpoint_queue(
        item_type=None,
        status=None,
        customer=None,
        job_type=None,
        sla_bucket=None,
        escalation_level=None,
        overdue_only=False,
        sort_by="priority_score_desc",
        limit=100,
        offset=0,
        db=_EmptyDb(),
        current_user=SimpleNamespace(id=uuid4()),
    )

    assert response.items == []
    assert response.total == 0
    assert response.approvals == 0
    assert response.recoveries == 0
    assert response.follow_ups == 0
    assert response.policy_reviews == 0
    assert response.budget_reviews == 0
    assert response.by_type == {}
    assert response.by_status == {}
    assert response.by_customer == {}
    assert response.by_sla_bucket == {}
    assert response.by_escalation_level == {}


@pytest.mark.asyncio
async def test_job_filter_selects_the_checkpoint_before_pagination():
    target = uuid4()
    approvals = [
        AgentCheckpointQueueItemResponse(
            queue_key=f"approval_checkpoint:{job_id}",
            item_type="approval_checkpoint",
            priority=priority,
            priority_score=priority,
            title=f"Plan {job_id}",
            job_id=job_id,
        )
        for job_id, priority in ((uuid4(), 10), (target, 1))
    ]
    api = build_checkpoint_queue_api(
        router=APIRouter(),
        build_queue_items=lambda *args, **kwargs: approvals,
        customer_profile_key=lambda value: value,
        load_learning_profile=None,
        build_monitor_snapshot=lambda **kwargs: {},
    )

    response = await api.get_checkpoint_queue(
        item_type="approval_checkpoint",
        status=None,
        customer=None,
        job_type=None,
        sla_bucket=None,
        escalation_level=None,
        overdue_only=False,
        sort_by="priority_score_desc",
        limit=1,
        offset=0,
        db=_EmptyDb(),
        current_user=SimpleNamespace(id=uuid4()),
        job_id=target,
    )

    assert response.total == response.approvals == 1
    assert response.items[0].job_id == target
