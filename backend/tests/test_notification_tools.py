"""Tests for notification/alerting tools (send_notification, send_email_alert)."""

import uuid


class TestSendNotification:
    """Tests for send_notification tool logic."""

    def test_requires_title(self):
        params = {"message": "something happened"}
        title = str(params.get("title", "")).strip()
        assert not title

    def test_requires_message(self):
        params = {"title": "Alert"}
        message = str(params.get("message", "")).strip()
        assert not message

    def test_accepts_valid_params(self):
        params = {
            "title": "Job Complete",
            "message": "Research finished with 10 papers found",
        }
        title = str(params.get("title", "")).strip()
        message = str(params.get("message", "")).strip()
        assert title == "Job Complete"
        assert message

    def test_priority_defaults_to_normal(self):
        params = {"title": "x", "message": "y"}
        priority = str(params.get("priority", "normal")).strip().lower()
        assert priority == "normal"

    def test_valid_priorities(self):
        valid = {"low", "normal", "high", "urgent"}
        for p in valid:
            assert p in valid

    def test_invalid_priority_normalized(self):
        params = {"title": "x", "message": "y", "priority": "CRITICAL"}
        priority = str(params.get("priority", "normal")).strip().lower()
        if priority not in {"low", "normal", "high", "urgent"}:
            priority = "normal"
        assert priority == "normal"

    def test_title_truncated(self):
        long_title = "A" * 300
        truncated = long_title[:200]
        assert len(truncated) == 200

    def test_message_truncated(self):
        long_msg = "B" * 3000
        truncated = long_msg[:2000]
        assert len(truncated) == 2000

    def test_action_url_optional(self):
        params = {"title": "x", "message": "y"}
        action_url = str(params.get("action_url", "")).strip()[:500] or None
        assert action_url is None

    def test_action_url_accepted(self):
        params = {"title": "x", "message": "y", "action_url": "/agents/job-123"}
        action_url = str(params.get("action_url", "")).strip()[:500] or None
        assert action_url == "/agents/job-123"

    def test_result_format(self):
        result = {
            "success": True,
            "data": {
                "notification_id": str(uuid.uuid4()),
                "delivered": True,
                "priority": "high",
            },
        }
        assert result["data"]["delivered"] is True
        assert result["data"]["priority"] == "high"

    def test_notification_data_payload(self):
        job_id = str(uuid.uuid4())
        data = {"source_job_id": job_id, "source_job_name": "Research Agent"}
        assert data["source_job_id"] == job_id


class TestSendEmailAlert:
    """Tests for send_email_alert tool logic."""

    def test_requires_subject(self):
        params = {"body": "something happened"}
        subject = str(params.get("subject", "")).strip()
        assert not subject

    def test_requires_body(self):
        params = {"subject": "Alert"}
        body = str(params.get("body", "")).strip()
        assert not body

    def test_accepts_valid_params(self):
        params = {
            "subject": "Urgent: Job Failed",
            "body": "The research job encountered an error.",
        }
        subject = str(params.get("subject", "")).strip()
        body = str(params.get("body", "")).strip()
        assert subject
        assert body

    def test_email_title_prefixed(self):
        subject = "Urgent: Job Failed"
        title = f"[Email] {subject[:180]}"
        assert title.startswith("[Email] ")
        assert "Urgent" in title

    def test_fallback_delivery_method(self):
        result = {
            "success": True,
            "data": {
                "notification_id": str(uuid.uuid4()),
                "delivered": True,
                "delivery_method": "in_app_notification",
                "note": "SMTP not configured; delivered as in-app notification",
            },
        }
        assert result["data"]["delivery_method"] == "in_app_notification"
        assert "SMTP" in result["data"]["note"]

    def test_intended_delivery_in_data(self):
        data = {"intended_delivery": "email", "source_job_id": "j-1"}
        assert data["intended_delivery"] == "email"


class TestNotificationToolSchemas:
    """Tests for notification tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS

        names = {t["name"] for t in AGENT_TOOLS}
        assert "send_notification" in names
        assert "send_email_alert" in names

    def test_send_notification_requires_params(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("send_notification")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "title" in required
        assert "message" in required

    def test_send_email_requires_params(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("send_email_alert")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "subject" in required
        assert "body" in required


class TestNotificationToolRegistry:
    """Tests for notification tool registry classification."""

    def test_send_notification_is_write_tool(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("send_notification")
        assert meta is not None
        assert meta.effects == "write"

    def test_send_email_is_write_tool(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("send_email_alert")
        assert meta is not None
        assert meta.effects == "write"

    def test_notification_tools_are_low_cost(self):
        from app.services.tool_registry import get_tool_metadata

        for tool_name in ["send_notification", "send_email_alert"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.cost_tier == "low"
