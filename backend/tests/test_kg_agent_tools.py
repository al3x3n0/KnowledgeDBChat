"""The knowledge-graph agent tools, called through their real handlers.

query_kg_entities, get_entity_context, create_kg_entity,
create_kg_relationship and query_kg_graph are nested functions registered by
`build_autonomous_kg_provider`. Every test here calls the registered handler
against the in-memory database; nothing restates what a handler does.
"""

import json
import uuid
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.models.knowledge_graph import Entity, EntityMention, Relationship
from app.services.agent_tool_dispatch import (
    AgentToolExecutionContext,
    build_autonomous_kg_provider,
)

pytestmark = pytest.mark.unit


def _ctx(db):
    user_id = uuid.uuid4()
    return AgentToolExecutionContext(
        mode="autonomous",
        db=db,
        service=None,
        user_id=str(user_id),
        job=SimpleNamespace(id=uuid.uuid4(), user_id=user_id, config={}),
        state={},
    )


async def _call(db, tool, params):
    provider = build_autonomous_kg_provider(SimpleNamespace())
    return await provider._handlers[tool](params, _ctx(db))


async def _entity(db, name, entity_type="concept", description=None, age=0):
    """Store an entity; `age` (seconds) orders it behind newer ones."""
    stamp = datetime(2026, 1, 1) - timedelta(seconds=age)
    entity = Entity(
        canonical_name=name,
        entity_type=entity_type,
        description=description,
        created_at=stamp,
        updated_at=stamp,
    )
    db.add(entity)
    await db.commit()
    return entity


async def _relationship(db, source, target, relation_type="uses", confidence=0.9):
    """Store an extracted relationship directly (it has a document)."""
    rel = Relationship(
        relation_type=relation_type,
        confidence=confidence,
        source_entity_id=source.id,
        target_entity_id=target.id,
        document_id=uuid.uuid4(),
        evidence="seeded",
    )
    db.add(rel)
    await db.commit()
    return rel


async def _mention(db, entity):
    db.add(EntityMention(entity_id=entity.id, document_id=uuid.uuid4(), text="m"))
    await db.commit()


class TestQueryKgEntities:
    async def test_refuses_a_missing_or_blank_query(self, db_session):
        await _entity(db_session, "Transformer")
        for params in ({}, {"query": "   "}):
            result = await _call(db_session, "query_kg_entities", params)
            assert result == {"error": "query is required"}

    async def test_finds_entities_by_name_case_insensitively(self, db_session):
        hit = await _entity(db_session, "Transformer Architecture", "concept", "attn")
        await _entity(db_session, "Django", "technology")

        result = await _call(
            db_session, "query_kg_entities", {"query": " transformer "}
        )

        assert result["success"] is True
        assert result["data"]["query"] == "transformer"
        assert result["data"]["count"] == 1
        assert result["data"]["entities"] == [
            {
                "id": str(hit.id),
                "canonical_name": "Transformer Architecture",
                "entity_type": "concept",
                "description": "attn",
            }
        ]

    async def test_no_match_is_an_empty_success(self, db_session):
        await _entity(db_session, "Django")
        result = await _call(db_session, "query_kg_entities", {"query": "zzz"})
        assert result["success"] is True
        assert result["data"]["entities"] == []
        assert result["data"]["count"] == 0

    async def test_result_is_json_serialisable(self, db_session):
        await _entity(db_session, "Django")
        result = await _call(db_session, "query_kg_entities", {"query": "django"})
        json.dumps(result)

    async def test_matches_descriptions_as_the_schema_promises(self, db_session):
        await _entity(db_session, "BERT", "concept", "a bidirectional encoder model")
        result = await _call(
            db_session, "query_kg_entities", {"query": "bidirectional"}
        )
        assert [e["canonical_name"] for e in result["data"]["entities"]] == ["BERT"]

    async def test_entity_type_narrows_the_matches(self, db_session):
        await _entity(db_session, "Apple Inc", "org")
        await _entity(db_session, "Apple pie", "product")
        result = await _call(
            db_session, "query_kg_entities", {"query": "apple", "entity_type": "org"}
        )
        assert [e["canonical_name"] for e in result["data"]["entities"]] == [
            "Apple Inc"
        ]
        assert result["data"]["count"] == 1

    async def test_entity_type_filter_matches_what_create_stored(self, db_session):
        created = await _call(
            db_session, "create_kg_entity", {"name": "Ada", "entity_type": "Person"}
        )
        assert created["data"]["entity_type"] == "person"
        result = await _call(
            db_session, "query_kg_entities", {"query": "ada", "entity_type": "Person"}
        )
        assert [e["canonical_name"] for e in result["data"]["entities"]] == ["Ada"]

    async def test_entity_type_filter_sees_matches_beyond_the_limit(self, db_session):
        for i in range(5):
            await _entity(db_session, f"alpha concept {i}", "concept", age=i)
        await _entity(db_session, "alpha person", "person", age=100)

        result = await _call(
            db_session,
            "query_kg_entities",
            {"query": "alpha", "entity_type": "person", "limit": 3},
        )

        assert [e["canonical_name"] for e in result["data"]["entities"]] == [
            "alpha person"
        ]

    async def test_limit_is_honoured(self, db_session):
        for i in range(5):
            await _entity(db_session, f"node {i}", age=i)
        result = await _call(
            db_session, "query_kg_entities", {"query": "node", "limit": 2}
        )
        assert result["data"]["count"] == 2
        assert [e["canonical_name"] for e in result["data"]["entities"]] == [
            "node 0",
            "node 1",
        ]

    async def test_limit_defaults_to_20_and_is_capped_at_100(self, db_session):
        db_session.add_all(
            [
                Entity(canonical_name=f"node {i}", entity_type="concept")
                for i in range(105)
            ]
        )
        await db_session.commit()

        default = await _call(db_session, "query_kg_entities", {"query": "node"})
        capped = await _call(
            db_session, "query_kg_entities", {"query": "node", "limit": 500}
        )

        assert default["data"]["count"] == 20
        assert capped["data"]["count"] == 100

    async def test_a_non_numeric_limit_is_an_error_not_a_crash(self, db_session):
        result = await _call(
            db_session, "query_kg_entities", {"query": "node", "limit": "many"}
        )
        assert "error" in result and "success" not in result


class TestGetEntityContext:
    async def test_refuses_a_missing_entity_id(self, db_session):
        for params in ({}, {"entity_id": "  "}):
            result = await _call(db_session, "get_entity_context", params)
            assert result == {"error": "entity_id is required"}

    async def test_refuses_a_malformed_entity_id(self, db_session):
        result = await _call(
            db_session, "get_entity_context", {"entity_id": "not-a-uuid"}
        )
        assert result == {"error": "Invalid entity_id format: not-a-uuid"}

    async def test_returns_the_entity_its_relationships_and_neighbours(
        self, db_session
    ):
        python = await _entity(db_session, "Python", "technology")
        django = await _entity(db_session, "Django", "technology")
        await _entity(db_session, "Unrelated", "concept")
        rel = await _relationship(db_session, python, django, "has_framework")

        result = await _call(
            db_session, "get_entity_context", {"entity_id": str(python.id)}
        )

        assert result["success"] is True
        data = result["data"]
        # Whatever the element shape, the right rows must be in it.
        assert {str(getattr(e, "id", None) or e["id"]) for e in data["entities"]} == {
            str(python.id),
            str(django.id),
        }
        assert [
            str(getattr(r, "id", None) or r["id"]) for r in data["relationships"]
        ] == [str(rel.id)]

    async def test_result_is_plain_data_a_model_can_read(self, db_session):
        python = await _entity(db_session, "Python", "technology")
        django = await _entity(db_session, "Django", "technology")
        await _relationship(db_session, python, django, "has_framework")

        result = await _call(
            db_session, "get_entity_context", {"entity_id": str(python.id)}
        )

        text = json.dumps(result)
        assert "Python" in text and "Django" in text and "has_framework" in text

    async def test_relationships_are_capped_at_30(self, db_session):
        hub = await _entity(db_session, "Hub")
        for i in range(35):
            spoke = await _entity(db_session, f"spoke {i}")
            await _relationship(db_session, hub, spoke)

        result = await _call(
            db_session, "get_entity_context", {"entity_id": str(hub.id)}
        )

        assert len(result["data"]["relationships"]) == 30
        assert len(result["data"]["entities"]) == 31

    async def test_an_unknown_entity_is_not_reported_as_found(self, db_session):
        result = await _call(
            db_session, "get_entity_context", {"entity_id": str(uuid.uuid4())}
        )
        assert "error" in result


class TestCreateKgEntity:
    async def test_refuses_a_missing_name(self, db_session):
        for params in (
            {"entity_type": "person"},
            {"name": " ", "entity_type": "person"},
        ):
            result = await _call(db_session, "create_kg_entity", params)
            assert result == {"error": "name is required"}
        assert (await db_session.execute(select(Entity))).scalars().all() == []

    async def test_refuses_a_missing_entity_type(self, db_session):
        result = await _call(db_session, "create_kg_entity", {"name": "John Doe"})
        assert result == {"error": "entity_type is required"}
        assert (await db_session.execute(select(Entity))).scalars().all() == []

    async def test_writes_the_entity_row(self, db_session):
        result = await _call(
            db_session,
            "create_kg_entity",
            {
                "name": "  OpenAI ",
                "entity_type": " ORG ",
                "description": " AI research company ",
            },
        )

        assert result["success"] is True
        rows = (await db_session.execute(select(Entity))).scalars().all()
        assert len(rows) == 1
        row = rows[0]
        assert (row.canonical_name, row.entity_type, row.description) == (
            "OpenAI",
            "org",
            "AI research company",
        )
        assert result["data"] == {
            "id": str(row.id),
            "canonical_name": "OpenAI",
            "entity_type": "org",
            "description": "AI research company",
        }

    async def test_description_is_optional(self, db_session):
        result = await _call(
            db_session, "create_kg_entity", {"name": "X", "entity_type": "other"}
        )
        assert result["data"]["description"] == ""
        row = (await db_session.execute(select(Entity))).scalar_one()
        assert row.description is None

    async def test_name_and_type_are_cut_to_their_columns(self, db_session):
        result = await _call(
            db_session,
            "create_kg_entity",
            {"name": "A" * 600, "entity_type": "x" * 100},
        )
        row = (await db_session.execute(select(Entity))).scalar_one()
        assert len(row.canonical_name) == 512
        assert len(row.entity_type) == 64
        assert result["data"]["canonical_name"] == "A" * 512

    async def test_the_created_entity_is_committed(self, db_session):
        """A reported id must survive a later rollback of the borrowed session."""
        result = await _call(
            db_session,
            "create_kg_entity",
            {"name": "Durable", "entity_type": "concept"},
        )
        assert result["success"] is True

        await db_session.rollback()

        row = await db_session.get(Entity, uuid.UUID(result["data"]["id"]))
        assert row is not None, "the tool reported an id for a row it never committed"

    async def test_the_created_entity_is_found_by_the_read_tools(self, db_session):
        created = await _call(
            db_session,
            "create_kg_entity",
            {"name": "Graphene", "entity_type": "concept"},
        )
        found = await _call(db_session, "query_kg_entities", {"query": "graphene"})
        assert [e["id"] for e in found["data"]["entities"]] == [created["data"]["id"]]

    async def test_the_created_entity_appears_in_the_graph(self, db_session):
        created = await _call(
            db_session,
            "create_kg_entity",
            {"name": "Graphene", "entity_type": "concept"},
        )
        graph = await _call(db_session, "query_kg_graph", {"search": "graphene"})
        assert [n["id"] for n in graph["data"]["nodes"]] == [created["data"]["id"]]


class TestCreateKgRelationship:
    async def _pair(self, db):
        return await _entity(db, "Ada", "person"), await _entity(db, "Acme", "org")

    async def test_refuses_each_missing_required_parameter(self, db_session):
        a, b = await self._pair(db_session)
        full = {
            "source_entity_id": str(a.id),
            "target_entity_id": str(b.id),
            "relation_type": "works_at",
        }
        for key in full:
            params = {k: v for k, v in full.items() if k != key}
            result = await _call(db_session, "create_kg_relationship", params)
            assert result == {"error": f"{key} is required"}
        assert (await db_session.execute(select(Relationship))).scalars().all() == []

    async def test_refuses_an_entity_that_does_not_exist(self, db_session):
        a, _ = await self._pair(db_session)
        missing = str(uuid.uuid4())
        for params in (
            {"source_entity_id": missing, "target_entity_id": str(a.id)},
            {"source_entity_id": str(a.id), "target_entity_id": missing},
        ):
            result = await _call(
                db_session,
                "create_kg_relationship",
                {**params, "relation_type": "works_at"},
            )
            assert "success" not in result
            assert missing in result["error"] and "not found" in result["error"]
        assert (await db_session.execute(select(Relationship))).scalars().all() == []

    async def test_refuses_a_malformed_entity_id(self, db_session):
        a, _ = await self._pair(db_session)
        result = await _call(
            db_session,
            "create_kg_relationship",
            {
                "source_entity_id": "nope",
                "target_entity_id": str(a.id),
                "relation_type": "works_at",
            },
        )
        assert "error" in result and "success" not in result

    async def test_writes_the_relationship_row(self, db_session):
        a, b = await self._pair(db_session)

        result = await _call(
            db_session,
            "create_kg_relationship",
            {
                "source_entity_id": str(a.id),
                "target_entity_id": str(b.id),
                "relation_type": "Works At",
                "evidence": " her contract ",
            },
        )

        assert result.get("success") is True, result
        row = (await db_session.execute(select(Relationship))).scalar_one()
        assert (row.source_entity_id, row.target_entity_id) == (a.id, b.id)
        assert row.relation_type == "works_at"
        assert row.confidence == 0.8
        assert row.evidence == "her contract"
        assert result["data"] == {
            "id": str(row.id),
            "relation_type": "works_at",
            "source_entity_id": str(a.id),
            "target_entity_id": str(b.id),
            "confidence": 0.8,
        }

    async def test_confidence_is_clamped_to_the_unit_interval(self, db_session):
        a, b = await self._pair(db_session)
        seen = {}
        for name, raw in (("high", 1.5), ("low", -0.5), ("mid", 0.7)):
            result = await _call(
                db_session,
                "create_kg_relationship",
                {
                    "source_entity_id": str(a.id),
                    "target_entity_id": str(b.id),
                    "relation_type": name,
                    "confidence": raw,
                },
            )
            assert result.get("success") is True, result
            seen[name] = result["data"]["confidence"]
        assert seen == {"high": 1.0, "low": 0.0, "mid": 0.7}

    async def test_the_same_relationship_is_not_created_twice(self, db_session):
        a, b = await self._pair(db_session)
        params = {
            "source_entity_id": str(a.id),
            "target_entity_id": str(b.id),
            "relation_type": "works_at",
        }
        first = await _call(db_session, "create_kg_relationship", params)
        assert first.get("success") is True, first

        second = await _call(db_session, "create_kg_relationship", params)

        assert "already exists" in second["error"]
        rows = (await db_session.execute(select(Relationship))).scalars().all()
        assert len(rows) == 1

    async def test_relation_type_is_cut_to_its_column(self, db_session):
        """The column is String(64); Postgres refuses anything longer."""
        a, b = await self._pair(db_session)
        result = await _call(
            db_session,
            "create_kg_relationship",
            {
                "source_entity_id": str(a.id),
                "target_entity_id": str(b.id),
                "relation_type": "r" * 200,
            },
        )
        assert result.get("success") is True, result
        limit = Relationship.__table__.c.relation_type.type.length
        assert len(result["data"]["relation_type"]) <= limit

    async def test_the_created_relationship_shows_in_entity_context(self, db_session):
        a, b = await self._pair(db_session)
        created = await _call(
            db_session,
            "create_kg_relationship",
            {
                "source_entity_id": str(a.id),
                "target_entity_id": str(b.id),
                "relation_type": "works_at",
            },
        )
        assert created.get("success") is True, created
        context = await _call(
            db_session, "get_entity_context", {"entity_id": str(a.id)}
        )
        assert len(context["data"]["relationships"]) == 1


class TestQueryKgGraph:
    async def _graph(self, db):
        """Three mentioned entities and two edges of differing confidence."""
        ada = await _entity(db, "Ada", "person")
        acme = await _entity(db, "Acme", "org")
        paris = await _entity(db, "Paris", "location")
        for e in (ada, acme, paris):
            await _mention(db, e)
        strong = await _relationship(db, ada, acme, "works_at", 0.9)
        weak = await _relationship(db, acme, paris, "located_in", 0.3)
        return ada, acme, paris, strong, weak

    async def test_no_parameters_returns_the_whole_graph(self, db_session):
        ada, acme, paris, strong, weak = await self._graph(db_session)

        result = await _call(db_session, "query_kg_graph", {})

        assert result["success"] is True
        data = result["data"]
        assert {n["id"] for n in data["nodes"]} == {
            str(ada.id),
            str(acme.id),
            str(paris.id),
        }
        assert {e["id"] for e in data["edges"]} == {str(strong.id), str(weak.id)}
        edge = next(e for e in data["edges"] if e["id"] == str(strong.id))
        assert (edge["source"], edge["target"], edge["type"]) == (
            str(ada.id),
            str(acme.id),
            "works_at",
        )
        assert data["metadata"]["total_entities"] == 3
        json.dumps(result)

    async def test_an_empty_graph_is_an_empty_success(self, db_session):
        result = await _call(db_session, "query_kg_graph", {})
        assert result["success"] is True
        assert result["data"]["nodes"] == [] and result["data"]["edges"] == []

    async def test_entity_types_filter_nodes_and_their_edges(self, db_session):
        ada, acme, _, strong, _ = await self._graph(db_session)
        result = await _call(
            db_session, "query_kg_graph", {"entity_types": ["person", "org"]}
        )
        assert {n["id"] for n in result["data"]["nodes"]} == {str(ada.id), str(acme.id)}
        assert [e["id"] for e in result["data"]["edges"]] == [str(strong.id)]

    async def test_relation_types_filter_edges(self, db_session):
        *_, weak = await self._graph(db_session)
        result = await _call(
            db_session, "query_kg_graph", {"relation_types": ["located_in"]}
        )
        assert [e["id"] for e in result["data"]["edges"]] == [str(weak.id)]
        assert len(result["data"]["nodes"]) == 3

    async def test_min_confidence_drops_weaker_edges(self, db_session):
        _, _, _, strong, _ = await self._graph(db_session)
        result = await _call(db_session, "query_kg_graph", {"min_confidence": 0.5})
        assert [e["id"] for e in result["data"]["edges"]] == [str(strong.id)]

    async def test_search_filters_on_entity_name(self, db_session):
        ada, *_ = await self._graph(db_session)
        result = await _call(db_session, "query_kg_graph", {"search": " ada "})
        assert [n["id"] for n in result["data"]["nodes"]] == [str(ada.id)]
        assert result["data"]["edges"] == []

    async def test_limit_nodes_is_honoured_and_bounds_edges(self, db_session):
        hub = await _entity(db_session, "Hub")
        await _mention(db_session, hub)
        await _mention(db_session, hub)
        for i in range(4):
            spoke = await _entity(db_session, f"spoke {i}")
            await _mention(db_session, spoke)
            await _relationship(db_session, hub, spoke)

        result = await _call(db_session, "query_kg_graph", {"limit_nodes": 2})

        nodes = result["data"]["nodes"]
        assert len(nodes) == 2
        assert nodes[0]["id"] == str(hub.id), "most-mentioned entity comes first"
        assert len(result["data"]["edges"]) == 1

    async def test_limit_nodes_defaults_to_50_and_is_capped_at_200(self, db_session):
        entities = [
            Entity(canonical_name=f"node {i}", entity_type="concept")
            for i in range(205)
        ]
        db_session.add_all(entities)
        await db_session.flush()
        db_session.add_all(
            [
                EntityMention(entity_id=e.id, document_id=uuid.uuid4(), text="m")
                for e in entities
            ]
        )
        await db_session.commit()

        default = await _call(db_session, "query_kg_graph", {})
        capped = await _call(db_session, "query_kg_graph", {"limit_nodes": 5000})

        assert len(default["data"]["nodes"]) == 50
        assert len(capped["data"]["nodes"]) == 200

    async def test_a_non_numeric_limit_is_an_error_not_a_crash(self, db_session):
        result = await _call(db_session, "query_kg_graph", {"limit_nodes": "lots"})
        assert "error" in result and "success" not in result


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

        for tool_name in [
            "query_kg_entities",
            "get_entity_context",
            "create_kg_entity",
            "create_kg_relationship",
            "query_kg_graph",
        ]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.cost_tier == "low"


async def test_the_service_refuses_a_relation_type_the_column_cannot_hold(db_session):
    """Checked on Postgres: 65 characters is refused by the database with an
    error naming no field. SQLite, which these tests run on, would accept it."""
    from app.services.knowledge_graph_service import KnowledgeGraphService

    a = Entity(canonical_name="A", entity_type="concept")
    b = Entity(canonical_name="B", entity_type="concept")
    db_session.add_all([a, b])
    await db_session.commit()

    with pytest.raises(ValueError, match="64 characters"):
        await KnowledgeGraphService().create_relationship(
            db_session, str(a.id), str(b.id), "x" * 65
        )
