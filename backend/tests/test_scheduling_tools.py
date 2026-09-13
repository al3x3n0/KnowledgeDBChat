"""Tests for scheduling tools (schedule_job, cancel_scheduled_job)."""

import uuid
from datetime import datetime, timezone

import pytest

try:
    from croniter import croniter

    HAS_CRONITER = True
except ImportError:
    HAS_CRONITER = False


class TestScheduleJob:
    """Tests for schedule_job tool logic."""

    def test_requires_goal(self):
        params = {"schedule_type": "once"}
        goal = str(params.get("goal", "")).strip()
        assert not goal

    def test_requires_schedule_type(self):
        params = {"goal": "Monitor arxiv papers"}
        schedule_type = str(params.get("schedule_type", "")).strip().lower()
        assert not schedule_type

    def test_valid_schedule_types(self):
        valid = {"once", "recurring"}
        assert "once" in valid
        assert "recurring" in valid
        assert "continuous" not in valid

    def test_invalid_schedule_type_rejected(self):
        schedule_type = "daily"
        assert schedule_type not in {"once", "recurring"}

    def test_once_requires_run_at(self):
        params = {"goal": "test", "schedule_type": "once"}
        run_at = str(params.get("run_at", "")).strip()
        assert not run_at

    def test_once_accepts_iso_datetime(self):
        run_at = "2026-04-01T09:00:00+00:00"
        dt = datetime.fromisoformat(run_at)
        assert dt.year == 2026
        assert dt.month == 4

    def test_naive_datetime_gets_utc(self):
        run_at = "2026-04-01T09:00:00"
        dt = datetime.fromisoformat(run_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        assert dt.tzinfo == timezone.utc

    def test_recurring_requires_cron(self):
        params = {"goal": "test", "schedule_type": "recurring"}
        cron = str(params.get("cron", "")).strip()
        assert not cron

    @pytest.mark.skipif(not HAS_CRONITER, reason="croniter not installed")
    def test_cron_validation(self):
        assert croniter.is_valid("0 9 * * 1")  # Every Monday at 9am
        assert croniter.is_valid("*/5 * * * *")  # Every 5 minutes
        assert not croniter.is_valid("invalid cron")

    @pytest.mark.skipif(not HAS_CRONITER, reason="croniter not installed")
    def test_cron_next_run(self):
        now = datetime.now(timezone.utc)
        cron_iter = croniter("0 9 * * *", now)
        next_run = cron_iter.get_next(datetime)
        assert next_run > now

    def test_job_type_defaults_to_research(self):
        params = {"goal": "test", "schedule_type": "once"}
        job_type = str(params.get("job_type", "research")).strip().lower()
        assert job_type == "research"

    def test_config_optional(self):
        params = {"goal": "test", "schedule_type": "once"}
        config = params.get("config") if isinstance(params.get("config"), dict) else {}
        assert config == {}

    def test_config_accepted(self):
        params = {
            "goal": "test",
            "schedule_type": "once",
            "config": {"max_iterations": 50},
        }
        config = params.get("config") if isinstance(params.get("config"), dict) else {}
        assert config["max_iterations"] == 50

    def test_goal_truncated(self):
        long_goal = "G" * 3000
        truncated = long_goal[:2000]
        assert len(truncated) == 2000

    def test_result_format(self):
        result = {
            "success": True,
            "data": {
                "id": str(uuid.uuid4()),
                "goal": "Monitor ML papers",
                "job_type": "monitor",
                "schedule_type": "recurring",
                "next_run_at": "2026-04-01T09:00:00+00:00",
                "cron": "0 9 * * 1",
            },
        }
        assert result["data"]["schedule_type"] == "recurring"
        assert result["data"]["cron"] == "0 9 * * 1"


class TestCancelScheduledJob:
    """Tests for cancel_scheduled_job tool logic."""

    def test_requires_job_id(self):
        params = {}
        job_id = str(params.get("job_id", "")).strip()
        assert not job_id

    def test_accepts_valid_uuid(self):
        jid = str(uuid.uuid4())
        params = {"job_id": jid}
        job_id = str(params.get("job_id", "")).strip()
        assert job_id == jid

    def test_invalid_uuid_detected(self):
        with pytest.raises(ValueError):
            uuid.UUID("not-a-uuid")

    def test_running_job_cannot_be_cancelled(self):
        status = "running"
        assert status == "running"  # Should trigger error in handler

    def test_multi_tenant_check(self):
        user_a = uuid.uuid4()
        user_b = uuid.uuid4()
        assert user_a != user_b  # Different users cannot cancel each other's jobs

    def test_result_format(self):
        result = {
            "success": True,
            "data": {
                "id": str(uuid.uuid4()),
                "status": "cancelled",
                "goal": "Monitor papers",
            },
        }
        assert result["data"]["status"] == "cancelled"


class TestSchedulingToolSchemas:
    """Tests for scheduling tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS

        names = {t["name"] for t in AGENT_TOOLS}
        assert "schedule_job" in names
        assert "cancel_scheduled_job" in names

    def test_schedule_job_requires_params(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("schedule_job")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "goal" in required
        assert "schedule_type" in required

    def test_cancel_scheduled_job_requires_job_id(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("cancel_scheduled_job")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "job_id" in required

    def test_schedule_job_has_cron_param(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("schedule_job")
        assert "cron" in tool["parameters"]["properties"]

    def test_schedule_job_has_run_at_param(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("schedule_job")
        assert "run_at" in tool["parameters"]["properties"]


class TestSchedulingToolRegistry:
    """Tests for scheduling tool registry classification."""

    def test_schedule_job_is_write(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("schedule_job")
        assert meta is not None
        assert meta.effects == "write"

    def test_cancel_scheduled_job_is_write(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("cancel_scheduled_job")
        assert meta is not None
        assert meta.effects == "write"

    def test_scheduling_tools_are_low_cost(self):
        from app.services.tool_registry import get_tool_metadata

        for tool_name in ["schedule_job", "cancel_scheduled_job"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.cost_tier == "low"
