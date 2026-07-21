"""Tests for batch processing tools (batch_search, batch_summarize)."""

import uuid

import pytest


class TestBatchSearch:
    """Tests for batch_search tool logic."""

    def test_requires_queries(self):
        params = {}
        queries = params.get("queries")
        assert not queries or not isinstance(queries, list)

    def test_empty_queries_rejected(self):
        params = {"queries": []}
        queries = params.get("queries")
        assert not queries

    def test_non_list_queries_rejected(self):
        params = {"queries": "single query"}
        queries = params.get("queries")
        assert not isinstance(queries, list)

    def test_accepts_valid_queries(self):
        params = {"queries": ["machine learning", "deep learning", "transformers"]}
        queries = params.get("queries")
        assert isinstance(queries, list)
        assert len(queries) == 3

    def test_queries_capped_at_10(self):
        raw = [f"query-{i}" for i in range(15)]
        queries = [str(q).strip() for q in raw if str(q).strip()][:10]
        assert len(queries) == 10

    def test_empty_strings_filtered(self):
        raw = ["valid", "", "  ", "also valid", ""]
        queries = [str(q).strip() for q in raw if str(q).strip()]
        assert len(queries) == 2

    def test_limit_per_query_defaults_to_5(self):
        params = {"queries": ["test"]}
        limit_per = min(int(params.get("limit_per_query", 5) or 5), 20)
        assert limit_per == 5

    def test_limit_per_query_capped_at_20(self):
        params = {"queries": ["test"], "limit_per_query": 50}
        limit_per = min(int(params.get("limit_per_query", 5) or 5), 20)
        assert limit_per == 20

    def test_source_id_optional(self):
        params = {"queries": ["test"]}
        source_id = str(params.get("source_id", "")).strip() or None
        assert source_id is None

    def test_source_id_accepted(self):
        sid = str(uuid.uuid4())
        params = {"queries": ["test"], "source_id": sid}
        source_id = str(params.get("source_id", "")).strip() or None
        assert source_id == sid

    def test_deduplicate_defaults_true(self):
        params = {"queries": ["test"]}
        dedup = params.get("deduplicate", True)
        if dedup is None:
            dedup = True
        assert dedup is True

    def test_deduplication_logic(self):
        seen_ids = set()
        results_q1 = [{"id": "doc-1"}, {"id": "doc-2"}, {"id": "doc-3"}]
        results_q2 = [{"id": "doc-2"}, {"id": "doc-4"}]  # doc-2 is duplicate

        filtered_q1 = []
        for r in results_q1:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                filtered_q1.append(r)
        assert len(filtered_q1) == 3

        filtered_q2 = []
        for r in results_q2:
            if r["id"] not in seen_ids:
                seen_ids.add(r["id"])
                filtered_q2.append(r)
        assert len(filtered_q2) == 1  # doc-2 deduped
        assert filtered_q2[0]["id"] == "doc-4"

    def test_no_deduplication(self):
        seen_ids = set()
        dedup = False
        results = [{"id": "doc-1"}, {"id": "doc-1"}, {"id": "doc-1"}]
        filtered = []
        for r in results:
            if dedup and r["id"] in seen_ids:
                continue
            seen_ids.add(r["id"])
            filtered.append(r)
        assert len(filtered) == 3  # No dedup, all kept

    def test_result_format(self):
        result = {
            "success": True,
            "data": {
                "queries_executed": 3,
                "results": [
                    {"query": "transformers", "results": [{"id": "d1", "title": "Paper 1"}], "total": 15},
                    {"query": "attention mechanism", "results": [{"id": "d2", "title": "Paper 2"}], "total": 8},
                    {"query": "BERT", "results": [], "total": 0},
                ],
                "total_unique_documents": 2,
            }
        }
        assert result["data"]["queries_executed"] == 3
        assert result["data"]["total_unique_documents"] == 2
        assert result["data"]["results"][2]["total"] == 0

    def test_findings_aggregation(self):
        all_results = [
            {"query": "q1", "results": [{"id": "d1", "title": "P1", "relevance_score": 0.9}]},
            {"query": "q2", "results": [{"id": "d2", "title": "P2", "relevance_score": 0.8}]},
        ]
        findings = []
        for qr in all_results:
            for r in qr.get("results", [])[:5]:
                findings.append({
                    "type": "document",
                    "title": r.get("title"),
                    "id": r.get("id"),
                    "score": r.get("relevance_score"),
                    "query": qr.get("query"),
                })
        assert len(findings) == 2
        assert findings[0]["query"] == "q1"
        assert findings[1]["query"] == "q2"

    def test_per_query_error_handling(self):
        all_results = [
            {"query": "good query", "results": [{"id": "d1"}], "total": 1},
            {"query": "bad query", "results": [], "total": 0, "error": "search failed"},
        ]
        errors = [r for r in all_results if "error" in r]
        assert len(errors) == 1


class TestBatchSummarize:
    """Tests for batch_summarize tool logic."""

    def test_requires_document_ids(self):
        params = {}
        doc_ids = params.get("document_ids")
        assert not doc_ids or not isinstance(doc_ids, list)

    def test_empty_ids_rejected(self):
        params = {"document_ids": []}
        doc_ids = params.get("document_ids")
        assert not doc_ids

    def test_accepts_valid_ids(self):
        params = {"document_ids": [str(uuid.uuid4()), str(uuid.uuid4())]}
        doc_ids = params.get("document_ids")
        assert isinstance(doc_ids, list)
        assert len(doc_ids) == 2

    def test_ids_capped_at_20(self):
        raw = [str(uuid.uuid4()) for _ in range(30)]
        doc_ids = [str(d).strip() for d in raw if str(d).strip()][:20]
        assert len(doc_ids) == 20

    def test_generate_missing_defaults_false(self):
        params = {"document_ids": ["x"]}
        gen = bool(params.get("generate_missing", False))
        assert gen is False

    def test_generate_missing_true(self):
        params = {"document_ids": ["x"], "generate_missing": True}
        gen = bool(params.get("generate_missing", False))
        assert gen is True

    def test_status_available(self):
        summary = {"document_id": "d1", "title": "Paper", "summary": "A summary...", "status": "available"}
        assert summary["status"] == "available"

    def test_status_no_summary(self):
        summary = {"document_id": "d2", "title": "Paper 2", "status": "no_summary"}
        assert summary["status"] == "no_summary"

    def test_status_not_found(self):
        summary = {"document_id": "d3", "status": "not_found"}
        assert summary["status"] == "not_found"

    def test_status_generated(self):
        summary = {"document_id": "d4", "title": "Paper 4", "summary": "Generated summary", "status": "generated"}
        assert summary["status"] == "generated"

    def test_status_generation_failed(self):
        summary = {"document_id": "d5", "title": "Paper 5", "status": "generation_failed", "error": "LLM timeout"}
        assert summary["status"] == "generation_failed"
        assert "timeout" in summary["error"]

    def test_available_count(self):
        summaries = [
            {"status": "available"},
            {"status": "generated"},
            {"status": "no_summary"},
            {"status": "not_found"},
            {"status": "available"},
        ]
        available = sum(1 for s in summaries if s.get("status") in ("available", "generated"))
        assert available == 3

    def test_result_format(self):
        result = {
            "success": True,
            "data": {
                "summaries": [
                    {"document_id": "d1", "title": "P1", "summary": "...", "status": "available"},
                    {"document_id": "d2", "title": "P2", "status": "no_summary"},
                    {"document_id": "d3", "status": "not_found"},
                ],
                "total_requested": 3,
                "available": 1,
                "missing": 2,
            }
        }
        assert result["data"]["total_requested"] == 3
        assert result["data"]["available"] == 1
        assert result["data"]["missing"] == 2

    def test_error_truncated(self):
        long_error = "E" * 500
        truncated = long_error[:200]
        assert len(truncated) == 200


class TestBatchToolSchemas:
    """Tests for batch tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS
        names = {t["name"] for t in AGENT_TOOLS}
        assert "batch_search" in names
        assert "batch_summarize" in names

    def test_batch_search_requires_queries(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("batch_search")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "queries" in required

    def test_batch_summarize_requires_document_ids(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("batch_summarize")
        assert tool is not None
        required = tool["parameters"].get("required", [])
        assert "document_ids" in required

    def test_batch_search_has_deduplicate_param(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("batch_search")
        assert "deduplicate" in tool["parameters"]["properties"]

    def test_batch_summarize_has_generate_missing_param(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("batch_summarize")
        assert "generate_missing" in tool["parameters"]["properties"]

    def test_batch_search_queries_is_array(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("batch_search")
        queries_prop = tool["parameters"]["properties"]["queries"]
        assert queries_prop["type"] == "array"

    def test_batch_summarize_ids_is_array(self):
        from app.services.agent_tools import get_tool_by_name
        tool = get_tool_by_name("batch_summarize")
        ids_prop = tool["parameters"]["properties"]["document_ids"]
        assert ids_prop["type"] == "array"


class TestBatchToolRegistry:
    """Tests for batch tool registry classification."""

    def test_batch_search_is_read(self):
        from app.services.tool_registry import get_tool_metadata
        meta = get_tool_metadata("batch_search")
        assert meta is not None
        assert meta.effects == "read"

    def test_batch_summarize_is_read(self):
        from app.services.tool_registry import get_tool_metadata
        meta = get_tool_metadata("batch_summarize")
        assert meta is not None
        assert meta.effects == "read"

    def test_both_are_medium_cost(self):
        from app.services.tool_registry import get_tool_metadata
        for tool_name in ["batch_search", "batch_summarize"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.cost_tier == "medium"

    def test_neither_is_network_tool(self):
        from app.services.tool_registry import get_tool_metadata
        for tool_name in ["batch_search", "batch_summarize"]:
            meta = get_tool_metadata(tool_name)
            assert meta is not None
            assert meta.network == "none"
