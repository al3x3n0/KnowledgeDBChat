"""Autonomous-job tools: the ``web_research`` provider.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

from typing import Any, Dict

from app.services.agent_tool_providers.base import (
    AgentToolExecutionContext,
    FunctionToolProvider,
)
from app.services.agent_tool_providers.common import _tool_snapshot_context


def build_autonomous_web_research_provider(executor: Any) -> FunctionToolProvider:
    """External web research helpers for AutonomousAgentExecutor."""

    async def _search_web(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        import html as _html
        import re as _re
        from urllib.parse import unquote

        import httpx

        query = str(params.get("query", "")).strip()
        if not query:
            return {"error": "query is required"}
        try:
            max_results = max(1, min(int(params.get("max_results", 5) or 5), 10))
            async with httpx.AsyncClient(
                timeout=15.0,
                headers={"User-Agent": "Mozilla/5.0 (compatible; KnowledgeDBChat/1.0)"},
                follow_redirects=True,
            ) as client:
                resp = await client.get(
                    "https://html.duckduckgo.com/html/", params={"q": query}
                )
                resp.raise_for_status()
            # One result at a time: a snippet is looked for only between a
            # title and the next title. A single pattern spanning title to
            # snippet gave a result that had no snippet the next result's,
            # and swallowed that result.
            titles = list(
                _re.finditer(
                    r'<a[^>]+class="result__a"[^>]+href="([^"]*)"[^>]*>(.*?)</a>',
                    resp.text,
                    _re.DOTALL,
                )
            )
            result_blocks = []
            for index, match in enumerate(titles):
                end = (
                    titles[index + 1].start()
                    if index + 1 < len(titles)
                    else len(resp.text)
                )
                snippet = _re.search(
                    r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
                    resp.text[match.end() : end],
                    _re.DOTALL,
                )
                result_blocks.append(
                    (
                        match.group(1),
                        match.group(2),
                        snippet.group(1) if snippet else "",
                    )
                )
            results_list = []
            for url_raw, title_raw, snippet_raw in result_blocks[:max_results]:
                title_clean = _html.unescape(_re.sub(r"<[^>]+>", "", title_raw)).strip()
                snippet_clean = _html.unescape(
                    _re.sub(r"<[^>]+>", "", snippet_raw)
                ).strip()
                url_raw = _html.unescape(url_raw)
                url_match = _re.search(r"uddg=([^&]+)", url_raw)
                url_clean = unquote(url_match.group(1) if url_match else url_raw)
                if title_clean:
                    results_list.append(
                        {
                            "title": title_clean[:200],
                            "url": url_clean[:500],
                            "snippet": snippet_clean[:500],
                        }
                    )
            return {
                "success": True,
                "data": {
                    "query": query,
                    "results": results_list,
                    "count": len(results_list),
                },
            }
        except Exception as exc:
            return {"error": f"Web search failed: {exc}"}

    async def _scrape_one_page(url: str, max_chars: int) -> Dict[str, Any]:
        """Fetch a single page.

        WebScraperService exposes `scrape`, which crawls and returns a "pages"
        list; there is no scrape_url. Both handlers below want one page.
        """
        from app.services.web_scraper_service import WebScraperService

        scraper = WebScraperService()
        try:
            result = await scraper.scrape(
                url,
                follow_links=False,
                max_pages=1,
                include_links=False,
                max_content_chars=max_chars,
            )
        finally:
            await scraper.aclose()
        pages = (result or {}).get("pages") or []
        if not pages:
            # Say why. Indexing the empty list reported every 404, timeout
            # and refused connection as "list index out of range", with the
            # real cause sitting unread beside it.
            errors = (result or {}).get("errors") or []
            first = errors[0] if errors else {}
            reason = first.get("error") if isinstance(first, dict) else first
            raise ValueError(str(reason or "the page could not be fetched"))
        return pages[0] if isinstance(pages[0], dict) else {}

    async def _fetch_url_content(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        url = str(params.get("url", "")).strip()
        if not url:
            return {"error": "url is required"}
        try:
            max_chars = min(int(params.get("max_chars", 50000) or 50000), 100000)
            page = await _scrape_one_page(url, max_chars)
            content = str(page.get("content", ""))[:max_chars]
            title = str(page.get("title", ""))[:200]
            if not content.strip():
                return {"error": f"No content extracted from {url}"}
            return {
                "success": True,
                "data": {
                    "url": url,
                    "title": title,
                    "content": content,
                    "content_length": len(content),
                },
            }
        except Exception as exc:
            return {"error": f"Failed to fetch URL: {exc}"}

    async def _summarize_url(
        params: Dict[str, Any], ctx: AgentToolExecutionContext
    ) -> Any:
        url = str(params.get("url", "")).strip()
        if not url:
            return {"error": "url is required"}
        try:
            page = await _scrape_one_page(url, 100000)
            full_text = str(page.get("content", ""))
            # What the model is shown is what is reported as summarised.
            text = full_text[:30000]
            if not text.strip():
                return {"error": f"No content extracted from {url}"}
            focus = str(params.get("focus", "")).strip()
            focus_clause = f" with focus on: {focus}" if focus else ""
            # LLMService has no `generate`; the text entry point is
            # generate_response(system_prompt=..., user_message=...).
            summary = await executor.llm_service.generate_response(
                system_prompt=(
                    "Summarize the web page content the user provides"
                    f"{focus_clause}. Be concise and extract key information."
                ),
                user_message=text,
                max_tokens=1000,
                db=ctx.db,
                snapshot_context=_tool_snapshot_context(ctx, "summarize_url"),
            )
            if not str(summary or "").strip():
                return {"error": f"The model returned no summary for {url}"}
            return {
                "success": True,
                "data": {
                    "url": url,
                    "summary": summary,
                    "content_length": len(text),
                    "truncated": len(full_text) > len(text),
                    "focus": focus or None,
                },
            }
        except Exception as exc:
            return {"error": f"URL summarization failed: {exc}"}

    return FunctionToolProvider(
        name="autonomous_web_research_tools",
        modes={"autonomous"},
        handlers={
            "search_web": _search_web,
            "fetch_url_content": _fetch_url_content,
            "summarize_url": _summarize_url,
        },
    )
