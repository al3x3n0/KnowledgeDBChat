from app.tasks.monitoring_tasks import (
    _build_policy_guardrail_notification_action_url,
    _build_queue_urgency_notification_action_url,
    _summarize_policy_guardrail_notification,
    _queue_alert_should_emit,
    _build_experiment_run_notification_action_url,
    _summarize_queue_urgency_notification,
    _summarize_experiment_run_notification,
)
from app.models.notification import Notification
from datetime import datetime, timezone


def test_summarize_experiment_run_notification_tracks_open_recovery():
    summary = _summarize_experiment_run_notification(
        {
            "final_phase": "fallback",
            "source_name": "Knowledge Repo",
            "source_id": "repo-4",
            "bootstrap_attempted": True,
            "bootstrap_ok": True,
            "fallback_attempted": True,
            "fallback_ok": False,
            "failed_commands": ["pytest -q backend/tests"],
            "execution_strategy": {
                "operator_interventions": [
                    {
                        "action": "restart",
                        "note": "Retry after fallback failure",
                        "job_status_before": "failed",
                        "job_status_after": "pending",
                        "at": "2026-03-10T01:00:00Z",
                        "outcome_status": "unresolved",
                        "outcome_reason": "Job failed after intervention",
                    }
                ],
                "execution_graph": {
                    "graph_health": {"reasons": ["fallback verification still failing"]},
                    "recommended_actions": ["Inspect failing fallback output"],
                }
            },
        },
        "failed",
        launch_mode="quick_start_claude_backend",
    )

    assert summary["message_suffix"] == (
        "failed · phase fallback · repo Knowledge Repo · recovery open · last operator restart · operator unresolved · "
        "fallback verification still failing"
    )
    assert summary["data"]["final_phase"] == "fallback"
    assert summary["data"]["source_name"] == "Knowledge Repo"
    assert summary["data"]["source_id"] == "repo-4"
    assert summary["data"]["bootstrap_ok"] is True
    assert summary["data"]["fallback_attempted"] is True
    assert summary["data"]["fallback_ok"] is False
    assert summary["data"]["failed_command_count"] == 1
    assert summary["data"]["first_failed_command"] == "pytest -q backend/tests"
    assert summary["data"]["recovery_open"] is True
    assert summary["data"]["recovery_reason"] == "fallback verification still failing"
    assert summary["data"]["recommended_action"] == "Inspect failing fallback output"
    assert summary["data"]["launch_mode"] == "quick_start_claude_backend"
    assert summary["data"]["latest_operator_action"] == "restart"
    assert summary["data"]["latest_operator_note"] == "Retry after fallback failure"
    assert summary["data"]["latest_operator_status_before"] == "failed"
    assert summary["data"]["latest_operator_status_after"] == "pending"
    assert summary["data"]["latest_operator_at"] == "2026-03-10T01:00:00Z"
    assert summary["data"]["latest_operator_outcome"] == "unresolved"
    assert summary["data"]["latest_operator_outcome_reason"] == "Job failed after intervention"


def test_summarize_experiment_run_notification_tracks_successful_fallback():
    summary = _summarize_experiment_run_notification(
        {
            "final_phase": "retry_primary",
            "source_name": "Frontend Repo",
            "fallback_attempted": True,
            "fallback_ok": True,
            "failed_commands": ["npm test"],
            "execution_strategy": {
                "operator_interventions": [
                    {
                        "action": "resume",
                        "job_status_before": "paused",
                        "job_status_after": "running",
                        "at": "2026-03-10T02:00:00Z",
                        "outcome_status": "resolved",
                        "outcome_reason": "Job completed after intervention",
                    }
                ]
            },
        },
        "completed",
        launch_mode=None,
    )

    assert summary["message_suffix"] == (
        "completed · phase retry_primary · repo Frontend Repo · fallback ok · last operator resume · operator resolved"
    )
    assert summary["data"]["recovery_open"] is False
    assert summary["data"]["fallback_ok"] is True
    assert summary["data"]["failed_command_count"] == 1
    assert summary["data"]["first_failed_command"] == "npm test"
    assert summary["data"]["recovery_reason"] is None
    assert summary["data"]["recommended_action"] is None
    assert summary["data"]["launch_mode"] is None
    assert summary["data"]["latest_operator_action"] == "resume"
    assert summary["data"]["latest_operator_note"] is None
    assert summary["data"]["latest_operator_status_before"] == "paused"
    assert summary["data"]["latest_operator_status_after"] == "running"
    assert summary["data"]["latest_operator_at"] == "2026-03-10T02:00:00Z"
    assert summary["data"]["latest_operator_outcome"] == "resolved"
    assert summary["data"]["latest_operator_outcome_reason"] == "Job completed after intervention"


def test_build_experiment_run_notification_action_url_prefers_agent_job():
    action_url = _build_experiment_run_notification_action_url(
        note_id="note-9",
        agent_job_id="job-42",
    )

    assert action_url == "/autonomous-agents?job=job-42"


def test_build_experiment_run_notification_action_url_falls_back_to_note():
    action_url = _build_experiment_run_notification_action_url(
        note_id="note-9",
        agent_job_id=None,
    )

    assert action_url == "/research-notes?note=note-9"


def test_build_queue_urgency_notification_action_url_includes_queue_context():
    action_url = _build_queue_urgency_notification_action_url(
        job_id="job-42",
        item_type="approval_checkpoint",
        sla_bucket="overdue",
    )

    assert (
        action_url
        == "/autonomous-agents?tab=queue&job=job-42&queue_item_type=approval_checkpoint&queue_sla=overdue"
    )


def test_build_policy_guardrail_notification_action_url_includes_review_context():
    action_url = _build_policy_guardrail_notification_action_url(
        job_id="job-42",
        history_entry_id="history-2",
    )

    assert action_url == "/autonomous-agents?tab=queue&queue_item_type=policy_review&job=job-42&policy_history=history-2"


def test_summarize_queue_urgency_notification_builds_payload():
    summary = _summarize_queue_urgency_notification(
        {
            "queue_key": "approval:job-42:checkpoint-9",
            "job_id": "job-42",
            "title": "Approval Required Job",
            "item_type": "approval_checkpoint",
            "reason_label": "Approval required",
            "sla_bucket": "overdue",
            "escalation_level": "high",
            "recommended_action": "approve",
            "customer": "Acme",
            "age_minutes": 185,
            "priority_score": 142,
            "is_overdue": True,
            "is_stale": True,
            "evidence_summary": "Human approval required before next action.",
            "scheduler_state": {
                "queue_reason": "execution_failure",
                "last_run_status": "failed",
                "failure_streak": 3,
                "last_scheduled_at": "2026-03-16T09:00:00Z",
                "last_dispatched_at": "2026-03-16T09:05:00Z",
                "backoff_until": "2026-03-16T10:00:00Z",
            },
        }
    )

    assert summary["title"] == "Queue alert: Approval Required Job"
    assert summary["message"] == (
        "approval checkpoint · overdue · escalation high · Approval required · stale · age 185m"
    )
    assert summary["priority"] == "high"
    assert summary["data"]["queue_key"] == "approval:job-42:checkpoint-9"
    assert summary["data"]["queue_item_type"] == "approval_checkpoint"
    assert summary["data"]["job_id"] == "job-42"
    assert summary["data"]["sla_bucket"] == "overdue"
    assert summary["data"]["escalation_level"] == "high"
    assert summary["data"]["priority_score"] == 142
    assert summary["data"]["recommended_action"] == "approve"
    assert summary["data"]["reason_label"] == "Approval required"
    assert summary["data"]["customer"] == "Acme"
    assert summary["data"]["age_minutes"] == 185
    assert summary["data"]["is_overdue"] is True
    assert summary["data"]["is_stale"] is True
    assert summary["data"]["evidence_summary"] == "Human approval required before next action."
    assert summary["data"]["scheduler_state"] is not None
    assert summary["data"]["scheduler_state"]["queue_reason"] == "execution_failure"
    assert summary["data"]["scheduler_state"]["failure_streak"] == 3


def test_summarize_queue_urgency_notification_omits_malformed_scheduler_state():
    summary = _summarize_queue_urgency_notification(
        {
            "queue_key": "approval:job-42:checkpoint-9",
            "job_id": "job-42",
            "title": "Approval Required Job",
            "item_type": "approval_checkpoint",
            "reason_label": "Approval required",
            "scheduler_state": "bad-payload",
        }
    )

    assert summary["data"]["scheduler_state"] is None


def test_summarize_policy_guardrail_notification_builds_payload():
    summary = _summarize_policy_guardrail_notification(
        {
            "queue_key": "policy_review:job-42:history-2",
            "job_id": "job-42",
            "title": "Beta Watch",
            "policy_guardrail_action": "rollback",
            "policy_guardrail_target_history_entry_id": "history-2",
            "policy_guardrail_reasons": ["More accepted items are getting blocked by policy"],
            "customer": "Beta",
        }
    )

    assert summary["title"] == "Policy safeguard: Beta Watch"
    assert summary["message"] == "degrading policy evaluation · suggested rollback · customer Beta · More accepted items are getting blocked by policy"
    assert summary["priority"] == "high"
    assert summary["data"]["monitor_job_id"] == "job-42"
    assert summary["data"]["history_entry_id"] == "history-2"
    assert summary["data"]["policy_guardrail_action"] == "rollback"


def test_queue_alert_should_emit_for_first_at_risk_notification():
    should_emit = _queue_alert_should_emit(
        item={"queue_key": "job:1", "sla_bucket": "at_risk"},
        existing_notifications=[],
        reminder_cooldown_hours=6,
        now=datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
    )

    assert should_emit is True


def test_queue_alert_should_emit_on_escalation_to_overdue():
    existing = Notification(
        user_id="00000000-0000-0000-0000-000000000001",
        notification_type="queue_urgency_alert",
        title="Queue alert",
        message="Alert",
        priority="normal",
        data={"queue_key": "job:1", "sla_bucket": "at_risk"},
        created_at=datetime(2026, 3, 17, 10, 0, tzinfo=timezone.utc),
    )

    should_emit = _queue_alert_should_emit(
        item={"queue_key": "job:1", "sla_bucket": "overdue"},
        existing_notifications=[existing],
        reminder_cooldown_hours=6,
        now=datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
    )

    assert should_emit is True


def test_queue_alert_should_not_emit_duplicate_unchanged_state():
    existing = Notification(
        user_id="00000000-0000-0000-0000-000000000001",
        notification_type="queue_urgency_alert",
        title="Queue alert",
        message="Alert",
        priority="normal",
        data={"queue_key": "job:1", "sla_bucket": "at_risk"},
        created_at=datetime(2026, 3, 17, 10, 0, tzinfo=timezone.utc),
    )

    should_emit = _queue_alert_should_emit(
        item={"queue_key": "job:1", "sla_bucket": "at_risk"},
        existing_notifications=[existing],
        reminder_cooldown_hours=6,
        now=datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
    )

    assert should_emit is False


def test_policy_guardrail_alert_should_not_emit_duplicate_state():
    existing = Notification(
        user_id="00000000-0000-0000-0000-000000000001",
        notification_type="policy_guardrail_alert",
        title="Policy safeguard",
        message="Rollback recommended",
        priority="high",
        data={"queue_key": "policy_review:job-1:history-2"},
        created_at=datetime(2026, 3, 17, 10, 0, tzinfo=timezone.utc),
    )

    should_emit = _queue_alert_should_emit(
        item={"queue_key": "policy_review:job-1:history-2", "item_type": "policy_review", "sla_bucket": "overdue"},
        existing_notifications=[existing],
        reminder_cooldown_hours=6,
        now=datetime(2026, 3, 17, 18, 0, tzinfo=timezone.utc),
    )

    assert should_emit is False


def test_queue_alert_should_emit_overdue_reminder_after_cooldown():
    existing = Notification(
        user_id="00000000-0000-0000-0000-000000000001",
        notification_type="queue_urgency_alert",
        title="Queue alert",
        message="Alert",
        priority="high",
        data={"queue_key": "job:1", "sla_bucket": "overdue"},
        created_at=datetime(2026, 3, 17, 4, 0, tzinfo=timezone.utc),
    )

    should_emit = _queue_alert_should_emit(
        item={"queue_key": "job:1", "sla_bucket": "overdue"},
        existing_notifications=[existing],
        reminder_cooldown_hours=6,
        now=datetime(2026, 3, 17, 12, 30, tzinfo=timezone.utc),
    )

    assert should_emit is True


def test_queue_alert_should_not_emit_overdue_reminder_before_cooldown():
    existing = Notification(
        user_id="00000000-0000-0000-0000-000000000001",
        notification_type="queue_urgency_alert",
        title="Queue alert",
        message="Alert",
        priority="high",
        data={"queue_key": "job:1", "sla_bucket": "overdue"},
        created_at=datetime(2026, 3, 17, 8, 30, tzinfo=timezone.utc),
    )

    should_emit = _queue_alert_should_emit(
        item={"queue_key": "job:1", "sla_bucket": "overdue"},
        existing_notifications=[existing],
        reminder_cooldown_hours=6,
        now=datetime(2026, 3, 17, 12, 0, tzinfo=timezone.utc),
    )

    assert should_emit is False
