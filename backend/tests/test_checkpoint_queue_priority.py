"""Policy tests for modular checkpoint queue prioritization."""

from datetime import datetime, timedelta

import pytest

from app.modules.autonomy.application.checkpoint_queue_priority import (
    queue_priority_fields,
)


@pytest.mark.parametrize(
    ("item_type", "age_minutes", "sla_bucket", "escalation_level", "overdue"),
    [
        ("approval_checkpoint", 30, "normal", "normal", False),
        ("approval_checkpoint", 90, "at_risk", "medium", False),
        ("approval_checkpoint", 300, "overdue", "high", True),
        ("policy_review", 45, "at_risk", "medium", False),
        ("policy_review", 200, "overdue", "high", True),
        ("budget_review", 90, "at_risk", "medium", False),
        ("budget_review", 300, "overdue", "high", True),
        ("follow_up_recommendation", 1500, "at_risk", "medium", False),
    ],
)
def test_queue_priority_sla_thresholds(
    item_type,
    age_minutes,
    sla_bucket,
    escalation_level,
    overdue,
):
    now = datetime(2026, 3, 16, 12, 0, 0)

    fields = queue_priority_fields(
        item_type=item_type,
        reason_code=None,
        created_at=now - timedelta(minutes=age_minutes),
        next_run_at=None,
        backoff_until=None,
        stale=False,
        now=now,
    )

    assert fields["sla_bucket"] == sla_bucket
    assert fields["escalation_level"] == escalation_level
    assert fields["is_overdue"] is overdue
    assert fields["age_minutes"] == age_minutes


def test_queue_priority_recovery_combines_failure_staleness_and_due_time():
    now = datetime(2026, 3, 16, 12, 0, 0)
    fields = queue_priority_fields(
        item_type="job_recovery",
        reason_code="execution_failure",
        created_at=now - timedelta(minutes=30),
        next_run_at=now - timedelta(minutes=1),
        backoff_until=now - timedelta(minutes=2),
        stale=True,
        now=now,
    )

    assert fields["priority_score"] == 138.5
    assert fields["sla_bucket"] == "overdue"
    assert fields["escalation_level"] == "high"
    assert fields["is_overdue"] is True
    assert fields["is_stale"] is True
