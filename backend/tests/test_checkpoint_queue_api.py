"""Regression coverage for the modular checkpoint-queue HTTP boundary."""

from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.api.endpoints import agent_jobs


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
