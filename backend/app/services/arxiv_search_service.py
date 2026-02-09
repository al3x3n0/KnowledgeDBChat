"""
ArXiv search service (interactive research use-cases).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx
import xml.etree.ElementTree as ET


@dataclass
class ArxivSearchResult:
    total_results: int
    start: int
    max_results: int
    items: List[Dict[str, Any]]


class ArxivSearchService:
    def __init__(self, api_url: str = "https://export.arxiv.org/api/query") -> None:
        self.api_url = api_url.rstrip("/")

    async def search(
        self,
        *,
        query: str,
        start: int = 0,
        max_results: int = 10,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> ArxivSearchResult:
        q = query.strip()
        if not q:
            raise ValueError("Query is required")

        params = {
            "search_query": q,
            "start": max(0, int(start)),
            "max_results": max(1, min(int(max_results), 50)),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        async with httpx.AsyncClient(
            headers={"User-Agent": "KnowledgeDBChat-ArxivSearch"},
            timeout=30.0,
        ) as client:
            resp = await client.get(self.api_url, params=params)
            resp.raise_for_status()

        total, items = self._parse_feed(resp.text)
        return ArxivSearchResult(
            total_results=total,
            start=params["start"],
            max_results=params["max_results"],
            items=items,
        )

    @staticmethod
    def _parse_feed(feed_xml: str) -> Tuple[int, List[Dict[str, Any]]]:
        items: List[Dict[str, Any]] = []
        try:
            root = ET.fromstring(feed_xml)
        except ET.ParseError:
            return 0, items

        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
            "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
        }

        total_text = root.findtext("opensearch:totalResults", default="0", namespaces=ns) or "0"
        try:
            total = int(total_text)
        except Exception:
            total = 0

        for entry in root.findall("atom:entry", ns):
            entry_url = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
            if not entry_url:
                continue

            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
            updated = (entry.findtext("atom:updated", default="", namespaces=ns) or "").strip()

            authors = [
                (author.findtext("atom:name", default="", namespaces=ns) or "").strip()
                for author in entry.findall("atom:author", ns)
            ]
            authors = [a for a in authors if a]

            categories = [
                cat.get("term") for cat in entry.findall("atom:category", ns)
                if cat.get("term")
            ]

            primary_category = None
            primary_node = entry.find("arxiv:primary_category", ns)
            if primary_node is not None:
                primary_category = primary_node.get("term")

            doi = (entry.findtext("arxiv:doi", default="", namespaces=ns) or "").strip() or None
            comments = (entry.findtext("arxiv:comment", default="", namespaces=ns) or "").strip() or None

            pdf_url = None
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                    pdf_url = link.get("href")

            # Derive a stable arXiv id (best-effort)
            paper_id = entry_url.rstrip("/").split("/")[-1]

            items.append(
                {
                    "id": paper_id,
                    "entry_url": entry_url,
                    "pdf_url": pdf_url,
                    "title": title or paper_id,
                    "summary": summary,
                    "authors": authors,
                    "published": published,
                    "updated": updated,
                    "categories": categories,
                    "primary_category": primary_category,
                    "doi": doi,
                    "comments": comments,
                }
            )

        return total, items

