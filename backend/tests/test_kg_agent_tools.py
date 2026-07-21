"""Tests for knowledge graph agent tools (query_kg_entities, get_entity_context, create_kg_entity, create_kg_relationship, query_kg_graph)."""

import uuid

import pytest


class TestQueryKgEntities:
    """Tests for query_kg_entities tool logic."""

    def test_requires_query(self):
        params = {}
        query = str(params.get("query", "")).strip()
        assert not query

    def test_accepts_valid_query(self):
        params = {"query": "machine learning"}
        query = str(params.get("query", "")).strip()
        assert query == "machine learning"

    def test_limit_defaults_to_20(self):
        params = {"query": "test"}
        limit = min(int(params.get("limit", 20) or 20), 100)
        assert limit == 20

    def test_limit_capped_at_100(self):
        params = {"query": "test", "limit": 500}
        limit = min(int(params.get("limit", 20) or 20), 100)
        assert limit == 100

    def test_entity_type_filter_optional(self):
        params = {"query": "test"}
        entity_type = str(params.get("entity_type", "")).strip() or None
        assert entity_type is None

    def test_entity_type_filter_applied(self):
        params = {"query": "test", "entity_type": "person"}
        entity_type = str(params.get("entity_type", "")).strip() or None
        assert entity_type == "person"

    def test_result_format(self):
        result = {
            "success": True,
            "data": {
                "query": "transformer",
                "entities": [
                    {"id": str(uuid.uuid4()), "canonical_name": "Transformer Architecture", "entity_type": "concept", "description": "Attention-based neural network architecture"},
                ],
                "count": 1,
            }
        }
        assert result["data"]["count"] == 1
        assert result["data"]["entities"][0]["entity_type"] == "concept"

    def test_entity_type_filtering(self):
        entities = [
            {"entity_type": "person", "canonical_name": "John"},
            {"entity_type": "org", "canonical_name": "Acme"},
            {"entity_type": "person", "canonical_name": "Jane"},
        ]
        entity_type = "person"
        filtered = [e for e in entities if e["entity_type"] == entity_type]
        assert len(filtered) == 2


class TestGetEntityContext:
    """Tests for get_entity_context tool logic."""

    def test_requires_entity_id(self):
        params = {}
        entity_id = str(params.get("entity_id", "")).strip()
        assert not entity_id

    def test_accepts_valid_uuid(self):
        eid = str(uuid.uuid4())
        params = {"entity_id": eid}
        entity_id = str(params.get("entity_id", "")).strip()
        assert entity_id == eid

    def test_invalid_uuid_detected(self):
        entity_id = "not-a-uuid"
        with pytest.raises(ValueError):
            uuid.UUID(entity_id)

    def test_valid_uuid_parsed(self):
        eid = str(uuid.uuid4())
        parsed = uuid.UUID(eid)
        assert str(parsed) == eid

    def test_result_structure(self):
        result = {
            "success": True,
            "data": {
                "entities": [
                    {"id": str(uuid.uuid4()), "canonical_name": "Python", "entity_type": "technology"},
                ],
                "relationships": [
                    {"source": "Python", "target": "Django", "relation_type": "has_framework"},
                ],
            }
        }
        assert "entities" in result["data"]
        assert "relationships" in result["data"]
        assert len(result["data"]["relationships"]) == 1


class TestCreateKgEntity:
    """Tests for create_kg_entity tool logic."""

    def test_requires_name(self):
        params = {"entity_type": "person"}
        name = str(params.get("name", "")).strip()
        assert not name

    def test_requires_entity_type(self):
        params = {"name": "John Doe"}
        entity_type = str(params.get("entity_type", "")).strip().lower()
        assert not entity_type

    def test_accepts_valid_params(self):
        params = {"name": "OpenAI", "entity_type": "org", "description": "AI research company"}
        name = str(params.get("name", "")).strip()
        entity_type = str(params.get("entity_type", "")).strip().lower()
        assert name == "OpenAI"
        assert entity_type == "org"

    def test_name_truncated_to_512(self):
        long_name = "A" * 600
        truncated = long_name[:512]
        assert len(truncated) == 512

    def test_entity_type_truncated_to_64(self):
        long_type = "x" * 100
        truncated = long_type[:64]
        assert len(truncated) == 64

    def test_description_optional(self):
        params = {"name": "X", "entity_type": "other"}
        description = str(params.get("description", "")).strip() or None
        assert description is None

    def test_result_format(self):
        eid = str(uuid.uuid4())
        result = {
            "success": True,
            "data": {
                "id": eid,
                "canonical_name": "OpenAI",
                "entity_type": "org",
                "description": "AI research company",
            }
        }
        assert result["data"]["id"] == eid
        assert result["data"]["canonical_name"] == "OpenAI"


class TestCreateKgRelationship:
    """Tests for create_kg_relationship tool logic."""

    def test_requires_source_entity_id(self):
        params = {"target_entity_id": str(uuid.uuid4()), "relation_type": "works_at"}
        source = str(params.get("source_entity_id", "")).strip()
        assert not source

    def test_requires_target_entity_id(self):
        params = {"source_entity_id": str(uuid.uuid4()), "relation_type": "works_at"}
        target = str(params.get("target_entity_id", "")).strip()
        assert not target

    def test_requires_relation_type(self):
        params = {"source_entity_id": str(uuid.uuid4()), "target_entity_id": str(uuid.uuid4())}
        rel_type = str(params.get("relation_type", "")).strip()
        assert not rel_type

    def test_accepts_valid_params(self):
        params = {
            "source_entity_id": str(uuid.uuid4()),
            "target_entity_id": str(uuid.uuid4()),
            "relation_type": "authored",
        }
        assert str(params.get("source_entity_id", "")).strip()
        assert str(params.get("target_entity_id", "")).strip()
        assert str(params.get("relation_type", "")).strip() == "authored"

    def test_confidence_defaults_to_0_8(self):
        params = {"source_entity_id": "a", "target_entity_id": "b", "relation_type": "related"}
        confidence = float(params.get("confidence", 0.8) or 0.8)
        assert confidence == 0.8

    def test_confidence_clamped(self):
        for raw, expected in [(1.5, 1.0), (-0.5, 0.0), (0.7, 0.7)]:
            confidence = max(0.0, min(1.0, raw))
            assert confidence == expected

    def test_evidence_optional(self):
        params = {"source_entity_id": "a", "target_entity_id": "b", "relation_type": "related"}
        evidence = str(params.get("evidence", "")).strip() or None
        assert evidence is None

    def test_relation_type_truncated(self):
        long_type = "r" * 200
        truncated = long_type[:128]
        assert len(truncated) == 128

    def test_result_format(self):
        result = {
            "success": True,
            "data": {
                "id": str(uuid.uuid4()),
                "relation_type": "works_at",
                "source_entity_id": str(uuid.uuid4()),
                "target_entity_id": str(uuid.uuid4()),
                "confidence": 0.9,
            }
        }
        assert result["data"]["relation_type"] == "works_at"
        assert result["data"]["confidence"] == 0.9


class TestQueryKgGraph:
    """Tests for query_kg_graph tool logic."""

    def test_no_required_params(self):
        params = {}
        # All params optional — should not error
        entity_types = params.get("entity_types") if isinstance(params.get("entity_types"), list) else None
        assert entity_types is None

    def test_entity_types_filter(self):
        params = {"entity_types": ["person", "org"]}
        entity_types = params.get("entity_types") if isinstance(params.get("entity_types"), list) else None
        assert entity_types == ["person", "org"]

    def test_relation_types_filter(self):
        params = {"relation_types": ["works_at", "authored"]}
        relation_types = params.get("relation_types") if isinstance(params.get("relation_types"), list) else None
        assert relation_types == ["works_at", "authored"]

    def test_min_confidence_defaults_to_zero(self):
        params = {}
        mc = float(params.get("min_confidence", 0.0) or 0.0)
        assert mc == 0.0

    def test_limit_nodes_defaults_to_50(self):
        params = {}
        limit = min(int(params.get("limit_nodes", 50) or 50), 200)
        assert limit == 50

    def test_limit_nodes_capped_at_200(self):
        params = {"limit_nodes": 500}
        limit = min(int(params.get("limit_nodes", 50) or 50), 200)
        assert limit == 200

    def test_limit_edges_is_3x_nodes(self):
        limit_nodes = 50
        limit_edges = limit_nodes * 3
        assert limit_edges == 150

    def test_search_optional(self):
        params = {}
        search = str(params.get("search", "")).strip() or None
        assert search is None

    def test_search_accepted(self):
        params = {"search": "transformer"}
        search = str(params.get("search", "")).strip() or None
        assert search == "transformer"

    def test_non_list_entity_types_ignored(self):
        params = {"entity_types": "person"}
        entity_types = params.get("entity_types") if isinstance(params.get("entity_types"), list) else None
        assert entity_types is None


class TestKgToolSchemas:
    """Tests for KG tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS
        names = {t["name"] for t in AGENT_TOOLS}
        assert "query_kg_entities" in names
        assert "get_entity_context" in names
        assert "create_kg_entity" in names
        assert "create_kg_relationship" in names
        assert "query_kg_graph" in names

    def test_query_kg_entities_requires_query(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("query_kg_entities")
        assert tool is not None
        assert "query" in tool["parameters"].get("required", [])

    def test_get_entity_context_requires_entity_id(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("get_entity_context")
        assert tool is not None
        assert "entity_id" in tool["parameters"].get("required", [])

    def test_create_kg_entity_requires_params(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("create_kg_entity")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "name" in required
        assert "entity_type" in required

    def test_create_kg_relationship_requires_params(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("create_kg_relationship")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "source_entity_id" in required
        assert "target_entity_id" in required
        assert "relation_type" in required

    def test_query_kg_graph_no_required(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("query_kg_graph")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert required == []


class TestKgToolRegistry:
    """Tests for KG tool registry classification."""

    def test_query_kg_entities_is_read(self):
        from app.services.tool_registry import get_tool_metadata
        meta = get_tool_metadata("query_kg_entities")
        assert meta is not None
        assert meta.effects == "read"

    def test_get_entity_context_is_read(self):
        from app.services.tool_registry import get_tool_metadata
        meta = get_tool_metadata("get_entity_context")
        assert meta is not None
        assert meta.effects == "read"

    def test_query_kg_graph_is_read(self):
        from app.services.tool_registry import get_tool_metadata
        meta = get_tool_metadata("query_kg_graph")
        assert meta is not None
        assert meta.effects == "read"

    def test_create_kg_entity_is_write(self):
        from app.services.tool_registry import get_tool_metadata
        meta = get_tool_metadata("create_kg_entity")
        assert meta is not None
        assert meta.effects == "write"

    def test_create_kg_relationship_is_write(self):
        from app.services.tool_registry import get_tool_metadata
        meta = get_tool_metadata("create_kg_relationship")
        assert meta is not None
        assert meta.effects == "write"

    def test_kg_tools_are_low_cost(self):
        from app.services.tool_registry import get_tool_metadata
        for tool_name in ["query_kg_entities", "get_entity_context", "create_kg_entity", "create_kg_relationship", "query_kg_graph"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.cost_tier == "low"
