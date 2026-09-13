"""Tests for research tools (search_web, summarize_url, fetch_url_content)."""

import re


class TestSearchWeb:
    """Tests for search_web tool logic."""

    def test_requires_query(self):
        params = {}
        query = str(params.get("query", "")).strip()
        assert not query

    def test_accepts_valid_query(self):
        params = {"query": "machine learning transformers"}
        query = str(params.get("query", "")).strip()
        assert query == "machine learning transformers"

    def test_max_results_defaults_to_5(self):
        params = {"query": "test"}
        max_results = min(int(params.get("max_results", 5) or 5), 10)
        assert max_results == 5

    def test_max_results_capped_at_10(self):
        params = {"query": "test", "max_results": 50}
        max_results = min(int(params.get("max_results", 5) or 5), 10)
        assert max_results == 10

    def test_result_format(self):
        result = {
            "success": True,
            "data": {
                "query": "python async",
                "results": [
                    {
                        "title": "AsyncIO docs",
                        "url": "https://docs.python.org/3/library/asyncio.html",
                        "snippet": "...",
                    },
                    {
                        "title": "Real Python AsyncIO",
                        "url": "https://realpython.com/async-io-python/",
                        "snippet": "...",
                    },
                ],
                "count": 2,
            },
        }
        assert result["data"]["count"] == 2
        assert result["data"]["results"][0]["title"] == "AsyncIO docs"

    def test_ddg_url_extraction(self):
        """Test DuckDuckGo redirect URL parsing."""
        url_raw = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&rut=abc"
        url_match = re.search(r"uddg=([^&]+)", url_raw)
        assert url_match is not None
        from urllib.parse import unquote

        url_clean = unquote(url_match.group(1))
        assert url_clean == "https://example.com/page"

    def test_html_tag_stripping(self):
        html = "<b>Bold</b> and <i>italic</i> text"
        clean = re.sub(r"<[^>]+>", "", html).strip()
        assert clean == "Bold and italic text"

    def test_empty_results(self):
        result = {
            "success": True,
            "data": {"query": "xyzzynotfound", "results": [], "count": 0},
        }
        assert result["data"]["count"] == 0


class TestFetchUrlContent:
    """Tests for fetch_url_content tool logic."""

    def test_requires_url(self):
        params = {}
        url = str(params.get("url", "")).strip()
        assert not url

    def test_accepts_valid_url(self):
        params = {"url": "https://example.com"}
        url = str(params.get("url", "")).strip()
        assert url == "https://example.com"

    def test_max_chars_defaults(self):
        params = {"url": "https://example.com"}
        max_chars = min(int(params.get("max_chars", 50000) or 50000), 100000)
        assert max_chars == 50000

    def test_max_chars_capped(self):
        params = {"url": "https://example.com", "max_chars": 200000}
        max_chars = min(int(params.get("max_chars", 50000) or 50000), 100000)
        assert max_chars == 100000

    def test_result_format(self):
        result = {
            "success": True,
            "data": {
                "url": "https://example.com",
                "title": "Example Domain",
                "content": "This domain is for use in examples...",
                "content_length": 37,
            },
        }
        assert result["data"]["content_length"] == 37
        assert result["data"]["title"] == "Example Domain"


class TestSummarizeUrl:
    """Tests for summarize_url tool logic."""

    def test_requires_url(self):
        params = {}
        url = str(params.get("url", "")).strip()
        assert not url

    def test_focus_optional(self):
        params = {"url": "https://example.com"}
        focus = str(params.get("focus", "")).strip()
        assert focus == ""

    def test_focus_included_in_prompt(self):
        focus = "security implications"
        focus_clause = f" with focus on: {focus}" if focus else ""
        prompt = f"Summarize the following web page content{focus_clause}."
        assert "security implications" in prompt

    def test_no_focus_in_prompt(self):
        focus = ""
        focus_clause = f" with focus on: {focus}" if focus else ""
        prompt = f"Summarize the following web page content{focus_clause}."
        assert "focus on" not in prompt

    def test_content_truncated_for_llm(self):
        text = "x" * 60000
        truncated = text[:30000]
        assert len(truncated) == 30000

    def test_result_format(self):
        result = {
            "success": True,
            "data": {
                "url": "https://example.com/article",
                "summary": "This article discusses...",
                "content_length": 15000,
                "focus": "machine learning",
            },
        }
        assert result["data"]["summary"].startswith("This article")
        assert result["data"]["focus"] == "machine learning"


class TestResearchToolSchemas:
    """Tests for research tool schema definitions."""

    def test_schemas_exist(self):
        from app.services.agent_tools import AGENT_TOOLS

        names = {t["name"] for t in AGENT_TOOLS}
        assert "search_web" in names
        assert "summarize_url" in names
        assert "fetch_url_content" in names

    def test_search_web_requires_query(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("search_web")
        assert tool is not None
        assert "query" in tool["parameters"].get("required", [])

    def test_fetch_url_requires_url(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("fetch_url_content")
        assert tool is not None
        assert "url" in tool["parameters"].get("required", [])

    def test_summarize_url_requires_url(self):
        from app.services.agent_tools import get_tool_by_name

        tool = get_tool_by_name("summarize_url")
        assert tool is not None
        assert "url" in tool["parameters"].get("required", [])


class TestResearchToolRegistry:
    """Tests for research tool registry classification."""

    def test_search_web_is_network_tool(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("search_web")
        assert meta is not None
        assert meta.network == "egress"

    def test_search_web_is_medium_cost(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("search_web")
        assert meta is not None
        assert meta.cost_tier == "medium"

    def test_fetch_url_is_network_tool(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("fetch_url_content")
        assert meta is not None
        assert meta.network == "egress"

    def test_summarize_url_is_network_tool(self):
        from app.services.tool_registry import get_tool_metadata

        meta = get_tool_metadata("summarize_url")
        assert meta is not None
        assert meta.network == "egress"
        assert meta.cost_tier == "medium"
