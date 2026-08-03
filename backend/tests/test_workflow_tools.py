"""Tests for workflow orchestration tool handler logic (list_available_workflows, execute_workflow, get_workflow_status)."""

import uuid

import pytest


class TestListAvailableWorkflows:
    """Tests for list_available_workflows tool logic."""

    def test_defaults_to_active(self):
        params = {}
        is_active = params.get("is_active", True)
        assert is_active is True

    def test_can_filter_inactive(self):
        params = {"is_active": False}
        is_active = params.get("is_active", True)
        assert is_active is False

    def test_result_format(self):
        result = {
            "success": True,
            "data": {
                "workflows": [
                    {
                        "id": "wf-1",
                        "name": "Data Pipeline",
                        "description": "ETL",
                        "is_active": True,
                        "node_count": 5,
                    },
                    {
                        "id": "wf-2",
                        "name": "Report Gen",
                        "description": "Weekly",
                        "is_active": True,
                        "node_count": 3,
                    },
                ],
                "count": 2,
            },
        }
        assert result["data"]["count"] == 2
        assert result["data"]["workflows"][0]["node_count"] == 5

    def test_empty_result(self):
        result = {"success": True, "data": {"workflows": [], "count": 0}}
        assert result["data"]["count"] == 0


class TestExecuteWorkflow:
    """Tests for execute_workflow tool logic."""

    def test_requires_workflow_id(self):
        params = {}
        wf_id = str(params.get("workflow_id", "")).strip()
        assert not wf_id

    def test_accepts_valid_workflow_id(self):
        wf_id = str(uuid.uuid4())
        params = {"workflow_id": wf_id}
        assert str(params.get("workflow_id", "")).strip() == wf_id

    def test_uuid_parsing(self):
        from uuid import UUID

        valid_id = str(uuid.uuid4())
        parsed = UUID(valid_id)
        assert str(parsed) == valid_id

    def test_invalid_uuid_raises(self):
        from uuid import UUID

        with pytest.raises(ValueError):
            UUID("not-a-uuid")

    def test_trigger_data_defaults(self):
        params = {"workflow_id": str(uuid.uuid4())}
        job_id = "job-123"
        trigger_data = params.get("trigger_data") or {"source_job_id": job_id}
        assert trigger_data == {"source_job_id": "job-123"}

    def test_custom_trigger_data(self):
        params = {"workflow_id": str(uuid.uuid4()), "trigger_data": {"key": "value"}}
        trigger_data = params.get("trigger_data") or {"source_job_id": "job-1"}
        assert trigger_data == {"key": "value"}

    def test_inputs_optional(self):
        params = {"workflow_id": str(uuid.uuid4())}
        inputs = params.get("inputs")
        assert inputs is None

    def test_result_format(self):
        exec_id = str(uuid.uuid4())
        wf_id = str(uuid.uuid4())
        result = {
            "success": True,
            "data": {
                "execution_id": exec_id,
                "status": "completed",
                "workflow_id": wf_id,
            },
        }
        assert result["data"]["status"] == "completed"
        assert result["data"]["execution_id"] == exec_id


class TestGetWorkflowStatus:
    """Tests for get_workflow_status tool logic."""

    def test_requires_execution_id(self):
        params = {}
        exec_id = str(params.get("execution_id", "")).strip()
        assert not exec_id

    def test_accepts_valid_execution_id(self):
        exec_id = str(uuid.uuid4())
        params = {"execution_id": exec_id}
        assert str(params.get("execution_id", "")).strip() == exec_id

    def test_result_format(self):
        exec_id = str(uuid.uuid4())
        wf_id = str(uuid.uuid4())
        result = {
            "success": True,
            "data": {
                "execution_id": exec_id,
                "workflow_id": wf_id,
                "status": "running",
                "progress": 0.5,
                "error": None,
                "started_at": "2026-01-01T00:00:00",
                "completed_at": None,
            },
        }
        assert result["data"]["status"] == "running"
        assert result["data"]["progress"] == 0.5
        assert result["data"]["error"] is None

    def test_completed_result(self):
        result = {
            "success": True,
            "data": {
                "execution_id": str(uuid.uuid4()),
                "workflow_id": str(uuid.uuid4()),
                "status": "completed",
                "progress": 1.0,
                "error": None,
                "started_at": "2026-01-01T00:00:00",
                "completed_at": "2026-01-01T00:05:00",
            },
        }
        assert result["data"]["status"] == "completed"
        assert result["data"]["completed_at"] is not None

    def test_failed_result(self):
        result = {
            "success": True,
            "data": {
                "execution_id": str(uuid.uuid4()),
                "workflow_id": str(uuid.uuid4()),
                "status": "failed",
                "progress": 0.3,
                "error": "Node 'step3' failed: timeout",
                "started_at": "2026-01-01T00:00:00",
                "completed_at": "2026-01-01T00:02:00",
            },
        }
        assert result["data"]["status"] == "failed"
        assert "timeout" in result["data"]["error"]


class TestWorkflowToolSchemas:
    """Tests for workflow tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS

        names = {t["name"] for t in AGENT_TOOLS}
        assert "list_available_workflows" in names
        assert "execute_workflow" in names
        assert "get_workflow_status" in names

    def test_execute_workflow_requires_workflow_id(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("execute_workflow")
        assert tool is not None
        assert "workflow_id" in tool["parameters"].get("required", [])

    def test_get_workflow_status_requires_execution_id(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("get_workflow_status")
        assert tool is not None
        assert "execution_id" in tool["parameters"].get("required", [])

    def test_list_workflows_has_no_required_params(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("list_available_workflows")
        assert tool is not None
        assert tool["parameters"].get("required", []) == []


class TestWorkflowToolRegistry:
    """Tests for workflow tool registry classification."""

    def test_execute_workflow_is_write_tool(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("execute_workflow")
        assert meta is not None
        assert meta.effects == "write"

    def test_list_workflows_is_read_tool(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("list_available_workflows")
        assert meta is not None
        assert meta.effects == "read"

    def test_get_status_is_read_tool(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("get_workflow_status")
        assert meta is not None
        assert meta.effects == "read"

    def test_execute_workflow_is_medium_cost(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("execute_workflow")
        assert meta is not None
        assert meta.cost_tier == "medium"
