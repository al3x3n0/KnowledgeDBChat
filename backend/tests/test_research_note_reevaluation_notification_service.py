from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.models.notification import Notification, NotificationType
from app.services.research_note_reevaluation_notification_service import maybe_emit_reevaluation_notification, resolve_reevaluation_notifications


class _EmptyResult:
    class _Scalars:
        @staticmethod
        def all():
            return []

    @staticmethod
    def scalars():
        return _EmptyResult._Scalars()


class _FakeDb:
    def __init__(self, notifications=None):
        self.notifications = notifications or []
        self.committed = False

    async def execute(self, _query):
        notifications = self.notifications

        class _Result:
            class _Scalars:
                @staticmethod
                def all():
                    return notifications

            @staticmethod
            def scalars():
                return _Result._Scalars()

        return _Result()

    async def commit(self):
        self.committed = True


@pytest.mark.asyncio
async def test_maybe_emit_reevaluation_notification_builds_origin_link(monkeypatch):
    captured = {}

    async def _fake_create_notification(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr(
        "app.services.research_note_reevaluation_notification_service.notification_service.create_notification",
        _fake_create_notification,
    )

    note = SimpleNamespace(
        id=uuid4(),
        title="Compiler note",
        structured_payload={
            "hypotheses": [
                {
                    "id": "hyp-1",
                    "autonomous_origin": {
                        "source_kind": "profile",
                        "source_id": "profile-1",
                        "opportunity_id": "opp-1",
                    },
                }
            ]
        },
    )

    result = await maybe_emit_reevaluation_notification(
        _FakeDb(),
        note=note,
        user_id=uuid4(),
        reevaluation_job_id="syn-1",
        status="completed",
        summary="Hypothesis A moved ahead.",
        source_run_ids=["run-1"],
        created_at="2026-03-12T02:00:00Z",
        completed_at="2026-03-12T03:00:00Z",
        commit=False,
        push=False,
    )

    assert result is not None
    assert captured["notification_type"] == "hypothesis_reevaluation_update"
    assert captured["action_url"] == f"/research-notes?note={note.id}"
    assert captured["data"]["origin_action_url"] == "/autonomous-agents?tab=domain&profileId=profile-1&opportunityId=opp-1"
    assert captured["data"]["source_run_ids"] == ["run-1"]
    assert captured["title"] == "Reevaluation ready: Compiler note"


@pytest.mark.asyncio
async def test_resolve_reevaluation_notifications_dismisses_matching_rows():
    note_id = uuid4()
    user_id = uuid4()
    notification = Notification(
        user_id=user_id,
        notification_type=NotificationType.HYPOTHESIS_REEVALUATION_UPDATE,
        title="Reevaluation ready: Compiler note",
        message="Ready to review.",
        priority="normal",
        related_entity_type="research_note",
        related_entity_id=note_id,
        data={
            "note_id": str(note_id),
            "reevaluation_job_id": "syn-1",
            "reevaluation_status": "completed",
        },
        action_url=f"/research-notes?note={note_id}",
    )
    other = Notification(
        user_id=user_id,
        notification_type=NotificationType.HYPOTHESIS_REEVALUATION_UPDATE,
        title="Other",
        message="Other.",
        priority="normal",
        related_entity_type="research_note",
        related_entity_id=note_id,
        data={
            "note_id": str(note_id),
            "reevaluation_job_id": "syn-2",
            "reevaluation_status": "completed",
        },
        action_url=f"/research-notes?note={note_id}",
    )
    db = _FakeDb([notification, other])

    count = await resolve_reevaluation_notifications(
        db,
        user_id=user_id,
        reevaluation_job_id="syn-1",
        note_id=note_id,
        review_outcome_status="dismissed",
        review_recorded_at="2026-04-20T10:00:00Z",
        review_note="Superseded by newer evidence.",
        commit=True,
    )

    assert count == 1
    assert db.committed is True
    assert notification.is_dismissed is True
    assert notification.is_read is True
    assert notification.data["review_outcome_status"] == "dismissed"
    assert notification.data["resolved_by_review"] is True
    assert notification.data["review_note"] == "Superseded by newer evidence."
    assert other.is_dismissed is False
