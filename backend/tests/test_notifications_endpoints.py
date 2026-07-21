from app.models.notification import NotificationPreferences


def test_get_notification_preferences_creates_defaults(
    client,
    db_session,
    test_user,
    auth_headers,
):
    response = client.get("/api/v1/notifications/preferences", headers=auth_headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == str(test_user.id)
    assert payload["notify_experiment_run_updates"] is True
    assert payload["notify_research_note_citation_issues"] is True
    assert payload["notify_queue_urgency_alerts"] is True
    assert payload["notify_follow_up_outcome_alerts"] is True
    assert payload["queue_urgency_alert_reminder_cooldown_hours"] == 6


def test_update_notification_preferences_can_disable_experiment_run_updates(
    client,
    db_session,
    test_user,
    auth_headers,
):
    import asyncio
    from sqlalchemy import select

    response = client.put(
        "/api/v1/notifications/preferences",
        headers=auth_headers,
        json={
            "notify_experiment_run_updates": False,
            "notify_queue_urgency_alerts": False,
            "notify_follow_up_outcome_alerts": False,
            "queue_urgency_alert_reminder_cooldown_hours": 10,
            "show_desktop_notification": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == str(test_user.id)
    assert payload["notify_experiment_run_updates"] is False
    assert payload["notify_queue_urgency_alerts"] is False
    assert payload["notify_follow_up_outcome_alerts"] is False
    assert payload["queue_urgency_alert_reminder_cooldown_hours"] == 10
    assert payload["show_desktop_notification"] is True

    async def _load_preferences():
        result = await db_session.execute(
            select(NotificationPreferences).where(NotificationPreferences.user_id == test_user.id)
        )
        return result.scalar_one()

    prefs = asyncio.get_event_loop().run_until_complete(_load_preferences())
    assert prefs.notify_experiment_run_updates is False
    assert prefs.notify_queue_urgency_alerts is False
    assert prefs.notify_follow_up_outcome_alerts is False
    assert prefs.queue_urgency_alert_reminder_cooldown_hours == 10
    assert prefs.show_desktop_notification is True
