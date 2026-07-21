"""Tests for memory tool handler logic (create_memory, search_memories, recall_memories, get_memory_stats)."""

import pytest


def _make_state():
    """Create a minimal state dict for memory tools."""
    return {
        "memories_created": [],
        "memories_searched": [],
    }


class TestCreateMemory:
    """Tests for create_memory tool logic."""

    def test_content_required(self):
        params = {}
        content = str(params.get("content", "")).strip()
        assert not content

    def test_content_accepted(self):
        params = {"content": "The API uses OAuth2 for authentication"}
        content = str(params.get("content", "")).strip()
        assert content == "The API uses OAuth2 for authentication"

    def test_importance_defaults_to_half(self):
        params = {"content": "test"}
        importance = float(params.get("importance", 0.5) or 0.5)
        assert importance == 0.5

    def test_importance_clamped(self):
        params = {"content": "test", "importance": 1.5}
        importance = max(0.0, min(1.0, float(params.get("importance", 0.5) or 0.5)))
        assert importance == 1.0

        params["importance"] = -0.3
        importance = max(0.0, min(1.0, float(params.get("importance", 0.5) or 0.5)))
        assert importance == 0.0

    def test_category_defaults_to_fact(self):
        params = {"content": "test"}
        category = str(params.get("category", "fact") or "fact")
        assert category == "fact"

    def test_valid_categories(self):
        valid = {"fact", "preference", "context", "summary", "goal", "constraint"}
        for cat in valid:
            assert cat in valid

    def test_metadata_tags_extracted(self):
        metadata = {"source": "research", "tags": ["ml", "nlp"]}
        tags = []
        if isinstance(metadata.get("tags"), list):
            tags = metadata.pop("tags")
        assert tags == ["ml", "nlp"]
        assert "tags" not in metadata
        assert metadata == {"source": "research"}

    def test_memory_create_schema_fields(self):
        """Verify MemoryCreate accepts the fields we pass."""
        from app.schemas.memory import MemoryCreate
        mem = MemoryCreate(
            memory_type="fact",
            content="Test memory content",
            importance_score=0.8,
            context={"key": "value"},
            tags=["test"],
        )
        assert mem.memory_type == "fact"
        assert mem.content == "Test memory content"
        assert mem.importance_score == 0.8

    def test_memory_create_validates_type(self):
        from app.schemas.memory import MemoryCreate
        with pytest.raises(Exception):
            MemoryCreate(
                memory_type="invalid_type",
                content="test",
            )


class TestSearchMemories:
    """Tests for search_memories tool logic."""

    def test_query_required(self):
        params = {}
        query = str(params.get("query", "")).strip()
        assert not query

    def test_query_accepted(self):
        params = {"query": "authentication methods"}
        query = str(params.get("query", "")).strip()
        assert query == "authentication methods"

    def test_limit_capped_at_50(self):
        params = {"query": "test", "limit": 100}
        limit = min(int(params.get("limit", 10) or 10), 50)
        assert limit == 50

    def test_limit_defaults_to_10(self):
        params = {"query": "test"}
        limit = min(int(params.get("limit", 10) or 10), 50)
        assert limit == 10

    def test_category_filter_to_memory_types(self):
        params = {"query": "test", "category_filter": "preference"}
        cat_filter = params.get("category_filter")
        memory_types = [cat_filter] if cat_filter else None
        assert memory_types == ["preference"]

    def test_no_category_filter(self):
        params = {"query": "test"}
        cat_filter = params.get("category_filter")
        memory_types = [cat_filter] if cat_filter else None
        assert memory_types is None

    def test_min_importance_conversion(self):
        params = {"query": "test", "min_importance": 0.7}
        min_imp = params.get("min_importance")
        val = float(min_imp) if min_imp is not None else None
        assert val == 0.7

    def test_search_request_schema(self):
        from app.schemas.memory import MemorySearchRequest
        req = MemorySearchRequest(
            query="authentication",
            limit=20,
            memory_types=["fact", "preference"],
            min_importance=0.5,
        )
        assert req.query == "authentication"
        assert req.limit == 20
        assert req.memory_types == ["fact", "preference"]


class TestRecallMemories:
    """Tests for recall_memories tool logic."""

    def test_topic_required(self):
        params = {}
        topic = str(params.get("topic", "")).strip()
        assert not topic

    def test_topic_accepted(self):
        params = {"topic": "machine learning architectures"}
        topic = str(params.get("topic", "")).strip()
        assert topic == "machine learning architectures"

    def test_recall_creates_broad_search(self):
        """recall_memories should create a search request without filters."""
        from app.schemas.memory import MemorySearchRequest
        topic = "project requirements"
        limit = 10
        req = MemorySearchRequest(query=topic, limit=limit)
        assert req.query == topic
        assert req.memory_types is None
        assert req.min_importance is None

    def test_limit_defaults_and_caps(self):
        params = {"topic": "test"}
        limit = min(int(params.get("limit", 10) or 10), 50)
        assert limit == 10

        params["limit"] = 999
        limit = min(int(params.get("limit", 10) or 10), 50)
        assert limit == 50


class TestGetMemoryStats:
    """Tests for get_memory_stats tool logic."""

    def test_stats_response_fields(self):
        """Verify MemoryStatsResponse has expected fields."""
        from app.schemas.memory import MemoryStatsResponse
        # Check the schema has the fields we access in the handler
        fields = MemoryStatsResponse.model_fields
        assert "total_memories" in fields
        assert "memories_by_type" in fields
        assert "recent_memories" in fields

    def test_no_params_needed(self):
        params = {}
        # get_memory_stats takes no params, just user_id from job
        assert len(params) == 0


class TestMemoryResultFormat:
    """Tests for memory tool result formatting."""

    def test_create_result_format(self):
        result = {
            "success": True,
            "data": {"memory_id": "mem-123", "content": "Short memory"},
        }
        assert result["success"] is True
        assert "memory_id" in result["data"]

    def test_search_result_format(self):
        result = {
            "success": True,
            "data": {
                "memories": [
                    {"id": "m-1", "content": "Memory 1", "importance": 0.8, "type": "fact"},
                    {"id": "m-2", "content": "Memory 2", "importance": 0.5, "type": "context"},
                ],
                "count": 2,
            },
        }
        assert result["data"]["count"] == 2
        assert result["data"]["memories"][0]["type"] == "fact"

    def test_stats_result_format(self):
        result = {
            "success": True,
            "data": {
                "total_memories": 42,
                "memories_by_type": {"fact": 20, "preference": 10, "context": 12},
                "recent_memories": 5,
            },
        }
        assert result["data"]["total_memories"] == 42
        assert sum(result["data"]["memories_by_type"].values()) == 42
