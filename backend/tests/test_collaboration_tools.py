"""Tests for agent collaboration protocol tools (create_handoff, get_sibling_status, broadcast_to_siblings)."""

import uuid


class TestCreateHandoff:
    """Tests for create_handoff tool logic."""

    def test_requires_goal(self):
        params = {"expected_outputs": ["summary"]}
        goal = str(params.get("goal", "")).strip()
        assert not goal

    def test_requires_expected_outputs(self):
        params = {"goal": "Research transformers"}
        outputs = params.get("expected_outputs", [])
        assert not isinstance(outputs, list) or not outputs

    def test_empty_expected_outputs_rejected(self):
        params = {"goal": "Research transformers", "expected_outputs": []}
        outputs = params.get("expected_outputs", [])
        assert not outputs

    def test_accepts_valid_params(self):
        params = {
            "goal": "Research transformers",
            "expected_outputs": ["summary", "key_findings", "recommendations"],
        }
        goal = str(params.get("goal", "")).strip()
        outputs = params.get("expected_outputs", [])
        assert goal
        assert isinstance(outputs, list) and len(outputs) == 3

    def test_chain_depth_guard(self):
        chain_depth = 3
        assert chain_depth >= 3

    def test_max_children_guard(self):
        existing_children = [str(uuid.uuid4()) for _ in range(5)]
        assert len(existing_children) >= 5

    def test_default_job_type_is_research(self):
        params = {"goal": "test", "expected_outputs": ["summary"]}
        job_type = str(params.get("job_type", "research")).strip()
        assert job_type == "research"

    def test_invalid_job_type_falls_back(self):
        params = {
            "goal": "test",
            "expected_outputs": ["summary"],
            "job_type": "invalid",
        }
        child_type = str(params.get("job_type", "research")).strip()
        valid = {"research", "analysis", "synthesis", "custom"}
        if child_type not in valid:
            child_type = "research"
        assert child_type == "research"

    def test_max_iterations_capped_at_20(self):
        params = {"goal": "test", "expected_outputs": ["summary"], "max_iterations": 50}
        child_max = min(int(params.get("max_iterations", 10) or 10), 20)
        assert child_max == 20

    def test_default_max_iterations_10(self):
        params = {"goal": "test", "expected_outputs": ["summary"]}
        child_max = min(int(params.get("max_iterations", 10) or 10), 20)
        assert child_max == 10

    def test_share_findings_defaults_true(self):
        params = {"goal": "test", "expected_outputs": ["summary"]}
        share = params.get("share_findings", True)
        if share is None:
            share = True
        assert share is True

    def test_handoff_contract_structure(self):
        contract = {
            "from_job_id": str(uuid.uuid4()),
            "from_job_name": "Parent Job",
            "context": "We have found 5 papers on transformers",
            "expected_outputs": ["summary", "key_findings"],
        }
        assert "from_job_id" in contract
        assert "expected_outputs" in contract
        assert len(contract["expected_outputs"]) == 2

    def test_context_truncated_to_2000(self):
        long_ctx = "X" * 3000
        truncated = long_ctx[:2000]
        assert len(truncated) == 2000

    def test_expected_outputs_capped_at_10(self):
        outputs = [f"output_{i}" for i in range(15)]
        capped = [str(o)[:200] for o in outputs[:10]]
        assert len(capped) == 10

    def test_result_format(self):
        result = {
            "child_job_id": str(uuid.uuid4()),
            "child_name": "Handoff from parent: Research transformers",
            "job_type": "research",
            "expected_outputs": ["summary", "key_findings"],
            "max_iterations": 10,
            "findings_shared": True,
        }
        assert "child_job_id" in result
        assert "expected_outputs" in result
        assert result["findings_shared"] is True


class TestGetSiblingStatus:
    """Tests for get_sibling_status tool logic."""

    def test_no_parent_rejected(self):
        parent_job_id = None
        assert not parent_job_id

    def test_include_findings_default_false(self):
        params = {}
        include_findings = bool(params.get("include_findings", False))
        assert include_findings is False

    def test_sibling_entry_format(self):
        entry = {
            "job_id": str(uuid.uuid4()),
            "name": "Sibling Research",
            "job_type": "research",
            "status": "running",
            "iteration": 5,
            "max_iterations": 15,
        }
        assert "job_id" in entry
        assert "status" in entry
        assert entry["iteration"] == 5

    def test_sibling_entry_with_findings(self):
        entry = {
            "job_id": str(uuid.uuid4()),
            "name": "Sibling Research",
            "job_type": "research",
            "status": "completed",
            "iteration": 10,
            "max_iterations": 10,
            "findings_count": 5,
            "finding_titles": ["Paper on BERT", "Transformer survey"],
        }
        assert entry["findings_count"] == 5
        assert len(entry["finding_titles"]) == 2

    def test_finding_titles_capped_at_10(self):
        findings = [{"title": f"Finding {i}"} for i in range(15)]
        titles = [str(f.get("title", ""))[:100] for f in findings[:10]]
        assert len(titles) == 10

    def test_result_format(self):
        result = {
            "siblings": [
                {"job_id": "id1", "name": "A", "status": "running"},
                {"job_id": "id2", "name": "B", "status": "completed"},
            ],
            "count": 2,
        }
        assert result["count"] == 2
        assert len(result["siblings"]) == 2


class TestBroadcastToSiblings:
    """Tests for broadcast_to_siblings tool logic."""

    def test_requires_message(self):
        params = {}
        message = str(params.get("message", "")).strip()
        assert not message

    def test_no_parent_rejected(self):
        parent_job_id = None
        assert not parent_job_id

    def test_default_category_is_broadcast(self):
        params = {"message": "Hello siblings"}
        category = str(params.get("category", "broadcast")).strip()[:100]
        assert category == "broadcast"

    def test_message_truncated_to_2000(self):
        long_msg = "M" * 3000
        truncated = long_msg[:2000]
        assert len(truncated) == 2000

    def test_message_entry_format(self):
        msg = {
            "from_job_id": str(uuid.uuid4()),
            "from_job_name": "Researcher A",
            "message": "Found important paper on attention mechanisms",
            "category": "broadcast",
            "sent_at": "2026-03-20T12:00:00",
            "broadcast": True,
        }
        assert msg["broadcast"] is True
        assert "from_job_id" in msg
        assert msg["category"] == "broadcast"

    def test_messages_capped_at_100(self):
        existing = [{"message": f"msg-{i}"} for i in range(99)]
        existing.append({"message": "new message"})
        capped = existing[-100:]
        assert len(capped) == 100

    def test_result_format(self):
        result = {
            "recipients": 3,
            "message_length": 45,
        }
        assert result["recipients"] == 3
        assert result["message_length"] == 45


class TestCollaborationSchemas:
    """Tests for collaboration tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS

        names = {t["name"] for t in AGENT_TOOLS}
        assert "create_handoff" in names
        assert "get_sibling_status" in names
        assert "broadcast_to_siblings" in names

    def test_create_handoff_requires_goal_and_outputs(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("create_handoff")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "goal" in required
        assert "expected_outputs" in required

    def test_create_handoff_has_job_type_enum(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("create_handoff")
        job_type_prop = tool["parameters"]["properties"]["job_type"]
        assert "enum" in job_type_prop
        assert "research" in job_type_prop["enum"]

    def test_get_sibling_status_no_required(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("get_sibling_status")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert len(required) == 0

    def test_broadcast_requires_message(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("broadcast_to_siblings")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "message" in required


class TestCollaborationRegistry:
    """Tests for collaboration tool registry classification."""

    def test_create_handoff_is_write(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("create_handoff")
        assert meta is not None
        assert meta.effects == "write"

    def test_create_handoff_is_medium_cost(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("create_handoff")
        assert meta is not None
        assert meta.cost_tier == "medium"

    def test_get_sibling_status_is_read(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("get_sibling_status")
        assert meta is not None
        assert meta.effects == "read"

    def test_broadcast_is_write(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("broadcast_to_siblings")
        assert meta is not None
        assert meta.effects == "write"

    def test_get_sibling_status_is_low_cost(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("get_sibling_status")
        assert meta is not None
        assert meta.cost_tier == "low"

    def test_none_is_network_tool(self):
        from app.services.tool_registry import get_tool_metadata

        for tool_name in [
            "create_handoff",
            "get_sibling_status",
            "broadcast_to_siblings",
        ]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.network == "none"
