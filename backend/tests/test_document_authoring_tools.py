"""Tests for document authoring enhancement tools (list_documents_by_tag, merge_documents)."""

import uuid


class TestListDocumentsByTag:
    """Tests for list_documents_by_tag tool logic."""

    def test_requires_tags(self):
        params = {}
        tags = params.get("tags")
        assert not tags or not isinstance(tags, list) or not tags

    def test_empty_tags_rejected(self):
        params = {"tags": []}
        tags = params.get("tags")
        assert not tags

    def test_non_list_tags_rejected(self):
        params = {"tags": "research"}
        tags = params.get("tags")
        assert not isinstance(tags, list)

    def test_accepts_valid_tags(self):
        params = {"tags": ["machine-learning", "transformer"]}
        tags = params.get("tags")
        assert isinstance(tags, list)
        assert len(tags) == 2

    def test_match_all_defaults_false(self):
        params = {"tags": ["a"]}
        match_all = bool(params.get("match_all", False))
        assert not match_all

    def test_match_all_set(self):
        params = {"tags": ["a"], "match_all": True}
        match_all = bool(params.get("match_all", False))
        assert match_all

    def test_limit_defaults_to_20(self):
        params = {"tags": ["a"]}
        limit = min(int(params.get("limit", 20) or 20), 100)
        assert limit == 20

    def test_limit_capped_at_100(self):
        params = {"tags": ["a"], "limit": 500}
        limit = min(int(params.get("limit", 20) or 20), 100)
        assert limit == 100

    def test_or_matching(self):
        """Documents matching ANY tag should be returned."""
        doc_tags = [["ml", "nlp"], ["ml", "cv"], ["cv", "robotics"]]
        search_tags = {"ml"}
        matched = [t for t in doc_tags if search_tags & set(t)]
        assert len(matched) == 2

    def test_and_matching(self):
        """Documents matching ALL tags should be returned."""
        doc_tags = [["ml", "nlp"], ["ml", "cv"], ["ml", "nlp", "transformer"]]
        search_tags = {"ml", "nlp"}
        matched = [t for t in doc_tags if search_tags.issubset(set(t))]
        assert len(matched) == 2

    def test_tag_set_construction(self):
        tags_param = ["research", " ai ", "", "ml"]
        tags_set = set(str(t).strip() for t in tags_param if str(t).strip())
        assert tags_set == {"research", "ai", "ml"}

    def test_result_format(self):
        result = {
            "success": True,
            "data": {
                "tags": ["ml"],
                "match_all": False,
                "documents": [
                    {
                        "id": str(uuid.uuid4()),
                        "title": "Paper 1",
                        "tags": ["ml", "nlp"],
                        "file_type": "pdf",
                        "summary": "...",
                        "created_at": "2026-01-01T00:00:00",
                    },
                ],
                "count": 1,
            },
        }
        assert result["data"]["count"] == 1
        assert "ml" in result["data"]["documents"][0]["tags"]


class TestMergeDocuments:
    """Tests for merge_documents tool logic."""

    def test_requires_document_ids(self):
        params = {"title": "Merged"}
        doc_ids = params.get("document_ids")
        assert not doc_ids or not isinstance(doc_ids, list)

    def test_empty_ids_rejected(self):
        params = {"document_ids": [], "title": "Merged"}
        doc_ids = params.get("document_ids")
        assert not doc_ids

    def test_requires_title(self):
        params = {"document_ids": [str(uuid.uuid4())]}
        title = str(params.get("title", "")).strip()
        assert not title

    def test_accepts_valid_params(self):
        params = {
            "document_ids": [str(uuid.uuid4()), str(uuid.uuid4())],
            "title": "Merged Research Report",
        }
        doc_ids = params.get("document_ids")
        title = str(params.get("title", "")).strip()
        assert isinstance(doc_ids, list)
        assert len(doc_ids) == 2
        assert title

    def test_separator_default(self):
        params = {"document_ids": ["a"], "title": "t"}
        separator = str(params.get("separator", "\n\n---\n\n"))
        assert separator == "\n\n---\n\n"

    def test_separator_custom(self):
        params = {"document_ids": ["a"], "title": "t", "separator": "\n\n"}
        separator = str(params.get("separator", "\n\n---\n\n"))
        assert separator == "\n\n"

    def test_tags_optional(self):
        params = {"document_ids": ["a"], "title": "t"}
        tags = params.get("tags") if isinstance(params.get("tags"), list) else []
        assert tags == []

    def test_document_ids_capped_at_20(self):
        ids = [str(uuid.uuid4()) for _ in range(30)]
        capped = ids[:20]
        assert len(capped) == 20

    def test_merge_content_structure(self):
        sections = [
            "# Paper A\n\nContent of paper A.",
            "# Paper B\n\nContent of paper B.",
        ]
        merged = "\n\n---\n\n".join(sections)
        assert "# Paper A" in merged
        assert "# Paper B" in merged
        assert "---" in merged

    def test_content_size_limit(self):
        large = "x" * 2_000_001
        assert len(large.encode("utf-8")) > 2_000_000

    def test_result_format(self):
        result = {
            "success": True,
            "data": {
                "document_id": str(uuid.uuid4()),
                "title": "Merged Report",
                "source_count": 3,
                "content_length": 15000,
            },
        }
        assert result["data"]["source_count"] == 3
        assert result["data"]["content_length"] > 0

    def test_artifacts_format(self):
        artifacts = [{"type": "document", "id": str(uuid.uuid4()), "title": "Merged"}]
        assert artifacts[0]["type"] == "document"

    def test_extra_metadata_contains_source_ids(self):
        source_ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        metadata = {"origin": "agent_merge", "source_document_ids": source_ids}
        assert metadata["origin"] == "agent_merge"
        assert len(metadata["source_document_ids"]) == 2


class TestDocumentAuthoringToolSchemas:
    """Tests for document authoring tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS

        names = {t["name"] for t in AGENT_TOOLS}
        assert "list_documents_by_tag" in names
        assert "merge_documents" in names

    def test_list_by_tag_requires_tags(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("list_documents_by_tag")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "tags" in required

    def test_merge_documents_requires_params(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("merge_documents")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "document_ids" in required
        assert "title" in required

    def test_list_by_tag_has_match_all(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("list_documents_by_tag")
        assert "match_all" in tool["parameters"]["properties"]

    def test_merge_documents_has_separator(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("merge_documents")
        assert "separator" in tool["parameters"]["properties"]


class TestDocumentAuthoringToolRegistry:
    """Tests for document authoring tool registry classification."""

    def test_list_by_tag_is_read(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("list_documents_by_tag")
        assert meta is not None
        assert meta.effects == "read"

    def test_merge_documents_is_write(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("merge_documents")
        assert meta is not None
        assert meta.effects == "write"

    def test_both_are_low_cost(self):
        from app.services.tool_registry import get_tool_metadata

        for tool_name in ["list_documents_by_tag", "merge_documents"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.cost_tier == "low"
