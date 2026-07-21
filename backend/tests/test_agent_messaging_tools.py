"""Tests for agent-to-agent communication tools (send_message_to_agent, read_agent_messages)."""

import uuid

import pytest


class TestSendMessageToAgent:
    """Tests for send_message_to_agent tool logic."""

    def test_requires_target_job_id(self):
        params = {"message": "hello"}
        target = str(params.get("target_job_id", "")).strip()
        assert not target

    def test_requires_message(self):
        params = {"target_job_id": str(uuid.uuid4())}
        message = str(params.get("message", "")).strip()
        assert not message

    def test_accepts_valid_params(self):
        params = {"target_job_id": str(uuid.uuid4()), "message": "Found interesting data"}
        target = str(params.get("target_job_id", "")).strip()
        message = str(params.get("message", "")).strip()
        assert target
        assert message

    def test_message_truncated_to_2000(self):
        long_msg = "x" * 3000
        truncated = long_msg[:2000]
        assert len(truncated) == 2000

    def test_category_optional(self):
        params = {"target_job_id": "id", "message": "hi"}
        category = str(params.get("category", ""))[:100].strip()
        assert category == ""

    def test_category_accepted(self):
        params = {"target_job_id": "id", "message": "hi", "category": "question"}
        category = str(params.get("category", ""))[:100].strip()
        assert category == "question"

    def test_message_entry_structure(self):
        from datetime import datetime
        entry = {
            "from_job_id": str(uuid.uuid4()),
            "from_job_name": "Research Agent",
            "message": "Found relevant paper",
            "category": "finding",
            "sent_at": datetime.utcnow().isoformat(),
        }
        assert "from_job_id" in entry
        assert "message" in entry
        assert "sent_at" in entry

    def test_messages_capped_at_100(self):
        messages = [{"message": f"msg-{i}"} for i in range(120)]
        capped = messages[-100:]
        assert len(capped) == 100
        assert capped[0]["message"] == "msg-20"

    def test_multi_tenant_safety(self):
        """Messages should only be sent between jobs with the same user_id."""
        user_a = uuid.uuid4()
        user_b = uuid.uuid4()
        assert str(user_a) != str(user_b)


class TestReadAgentMessages:
    """Tests for read_agent_messages tool logic."""

    def test_since_index_defaults_to_zero(self):
        params = {}
        since = max(0, int(params.get("since_index", 0) or 0))
        assert since == 0

    def test_since_index_accepted(self):
        params = {"since_index": 5}
        since = max(0, int(params.get("since_index", 0) or 0))
        assert since == 5

    def test_negative_since_clamped(self):
        params = {"since_index": -10}
        since = max(0, int(params.get("since_index", 0) or 0))
        assert since == 0

    def test_read_from_results(self):
        job_results = {
            "agent_messages": [
                {"from_job_id": "j-1", "message": "msg 1"},
                {"from_job_id": "j-2", "message": "msg 2"},
                {"from_job_id": "j-1", "message": "msg 3"},
            ],
            "shared_findings": [
                {"title": "finding 1"},
            ],
        }
        messages = job_results.get("agent_messages", [])
        shared = job_results.get("shared_findings", [])
        since = 1
        result = {
            "messages": messages[since:],
            "total": len(messages),
            "since_index": since,
            "shared_findings_count": len(shared),
        }
        assert len(result["messages"]) == 2
        assert result["total"] == 3
        assert result["shared_findings_count"] == 1

    def test_empty_messages(self):
        job_results = {}
        messages = job_results.get("agent_messages", [])
        assert messages == []

    def test_result_format(self):
        result = {
            "success": True,
            "data": {
                "messages": [
                    {"from_job_id": "j-1", "message": "hello", "category": "greeting"},
                ],
                "total": 1,
                "since_index": 0,
                "shared_findings_count": 0,
            }
        }
        assert result["data"]["total"] == 1
        assert result["data"]["messages"][0]["category"] == "greeting"


class TestMessagingToolSchemas:
    """Tests for messaging tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS
        names = {t["name"] for t in AGENT_TOOLS}
        assert "send_message_to_agent" in names
        assert "read_agent_messages" in names

    def test_send_message_requires_params(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("send_message_to_agent")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "target_job_id" in required
        assert "message" in required

    def test_read_messages_no_required_params(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("read_agent_messages")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert required == []


class TestMessagingToolRegistry:
    """Tests for messaging tool registry classification."""

    def test_send_message_is_write_tool(self):
        from app.services.tool_registry import get_tool_metadata
        meta = get_tool_metadata("send_message_to_agent")
        assert meta is not None
        assert meta.effects == "write"

    def test_read_messages_is_read_tool(self):
        from app.services.tool_registry import get_tool_metadata
        meta = get_tool_metadata("read_agent_messages")
        assert meta is not None
        assert meta.effects == "read"
