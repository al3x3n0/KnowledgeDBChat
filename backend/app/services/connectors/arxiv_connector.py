"""
Connector for ingesting content from the ArXiv API.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
import httpx
import xml.etree.ElementTree as ET

from loguru import logger

from .base_connector import BaseConnector


class ArxivConnector(BaseConnector):
    """Connector that pulls paper metadata and abstracts from ArXiv."""

    def __init__(self) -> None:
        super().__init__()
        self.api_url: str = "https://export.arxiv.org/api/query"
        self.queries: List[str] = []
        self.paper_ids: List[str] = []
        self.max_results: int = 50
        self.start: int = 0
        self.sort_by: str = "submittedDate"
        self.sort_order: str = "descending"
        self.session: Optional[httpx.AsyncClient] = None
        self.entry_cache: Dict[str, Dict[str, Any]] = {}

    async def initialize(self, config: Dict[str, Any]) -> bool:
        try:
            cfg = config or {}
            self.api_url = (cfg.get("api_url") or self.api_url).rstrip("/")
            self.queries = [q.strip() for q in (cfg.get("queries") or []) if isinstance(q, str) and q.strip()]
            categories = [c.strip() for c in (cfg.get("categories") or []) if isinstance(c, str) and c.strip()]
            self.paper_ids = [self._normalize_id(pid) for pid in (cfg.get("paper_ids") or []) if isinstance(pid, str) and pid.strip()]
            self.max_results = int(max(1, min(int(cfg.get("max_results", self.max_results)), 200)))
            self.start = int(max(0, min(int(cfg.get("start", self.start)), 1000)))
            self.sort_by = cfg.get("sort_by", self.sort_by)
            self.sort_order = cfg.get("sort_order", self.sort_order)

            if not self.queries and categories:
                combined = " OR ".join(f"cat:{cat}" for cat in categories)
                self.queries = [combined]

            if not self.queries and not self.paper_ids:
                raise ValueError("ArXiv connector requires at least one search query or paper id.")

            self.session = httpx.AsyncClient(
                headers={"User-Agent": "KnowledgeDBChat-ArxivConnector"},
                timeout=30.0,
            )
            self.is_initialized = True
            return True
        except Exception as exc:
            logger.error(f"Failed to initialize ArXiv connector: {exc}")
            return False

    async def test_connection(self) -> bool:
        try:
            if self.paper_ids:
                params = {"id_list": ",".join(self.paper_ids[:5])}
            else:
                query = self.queries[0]
                params = {
                    "search_query": query,
                    "start": self.start,
                    "max_results": 1,
                    "sortBy": self.sort_by,
                    "sortOrder": self.sort_order,
                }
            await self._fetch_entries(params)
            return True
        except Exception as exc:
            logger.error(f"ArXiv connection test failed: {exc}")
            return False

    async def list_documents(self) -> List[Dict[str, Any]]:
        self._ensure_initialized()
        documents: List[Dict[str, Any]] = []
        seen: set[str] = set()

        # Run queries
        for query in self.queries:
            params = {
                "search_query": query,
                "start": self.start,
                "max_results": self.max_results,
                "sortBy": self.sort_by,
                "sortOrder": self.sort_order,
            }
            docs = await self._fetch_entries(params)
            for doc in docs:
                identifier = doc["identifier"]
                if identifier in seen:
                    continue
                seen.add(identifier)
                documents.append(doc)

        # Fetch explicit IDs
        if self.paper_ids:
            params = {"id_list": ",".join(self.paper_ids)}
            docs = await self._fetch_entries(params)
            for doc in docs:
                identifier = doc["identifier"]
                if identifier in seen:
                    continue
                seen.add(identifier)
                documents.append(doc)

        return documents

    async def list_changed_documents(self, since: datetime) -> List[Dict[str, Any]]:
        docs = await self.list_documents()
        changed = []
        for doc in docs:
            last_modified: Optional[datetime] = doc.get("last_modified")
            if last_modified and last_modified <= since:
                continue
            changed.append(doc)
        return changed

    async def get_document_content(self, identifier: str) -> str:
        self._ensure_initialized()
        entry = self.entry_cache.get(identifier)
        if not entry:
            await self._hydrate_entry(identifier)
            entry = self.entry_cache.get(identifier)
        if not entry:
            raise ValueError(f"ArXiv entry {identifier} not found")

        sections = [
            entry.get("title"),
            f"Authors: {', '.join(entry.get('authors', []))}" if entry.get("authors") else None,
            f"Categories: {', '.join(entry.get('categories', []))}" if entry.get("categories") else None,
            f"Primary Category: {entry.get('primary_category')}" if entry.get("primary_category") else None,
            f"DOI: {entry.get('doi')}" if entry.get("doi") else None,
            f"Comments: {entry.get('comments')}" if entry.get("comments") else None,
            "Abstract:",
            entry.get("summary"),
            f"PDF: {entry.get('pdf_url')}" if entry.get("pdf_url") else None,
            f"Link: {entry.get('entry_url')}" if entry.get("entry_url") else None,
        ]
        content = "\n".join([part for part in sections if part])
        return content.strip()

    async def get_document_metadata(self, identifier: str) -> Dict[str, Any]:
        entry = self.entry_cache.get(identifier)
        if not entry:
            await self._hydrate_entry(identifier)
            entry = self.entry_cache.get(identifier)
        if not entry:
            raise ValueError(f"ArXiv entry {identifier} not found")
        return {
            "authors": entry.get("authors"),
            "categories": entry.get("categories"),
            "primary_category": entry.get("primary_category"),
            "doi": entry.get("doi"),
            "comments": entry.get("comments"),
            "links": {
                "pdf": entry.get("pdf_url"),
                "html": entry.get("entry_url"),
            },
        }

    async def cleanup(self):
        if self.session:
            await self.session.aclose()
            self.session = None

    async def _fetch_entries(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.session:
            raise RuntimeError("HTTP session not initialized")
        response = await self.session.get(self.api_url, params=params)
        response.raise_for_status()
        return self._parse_feed(response.text)

    async def _hydrate_entry(self, identifier: str):
        if not identifier.startswith("http"):
            identifier = f"http://arxiv.org/abs/{identifier}"
        params = {"id_list": identifier.split("/")[-1]}
        try:
            await self._fetch_entries(params)
        except Exception as exc:
            logger.warning(f"Failed to hydrate ArXiv entry {identifier}: {exc}")

    def _parse_feed(self, feed_xml: str) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        try:
            root = ET.fromstring(feed_xml)
        except ET.ParseError as exc:
            logger.error(f"Failed to parse ArXiv feed: {exc}")
            return entries

        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        for entry in root.findall("atom:entry", ns):
            identifier = entry.findtext("atom:id", default="", namespaces=ns)
            if not identifier:
                continue
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            published_text = entry.findtext("atom:published", default="", namespaces=ns) or ""
            updated_text = entry.findtext("atom:updated", default="", namespaces=ns) or ""
            published = self._parse_datetime(updated_text or published_text)
            authors = [
                (author.findtext("atom:name", default="", namespaces=ns) or "").strip()
                for author in entry.findall("atom:author", ns)
            ]
            categories = [
                cat.get("term") for cat in entry.findall("atom:category", ns)
                if cat.get("term")
            ]
            pdf_url = None
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf" or link.get("type") == "application/pdf":
                    pdf_url = link.get("href")
            doi = (entry.findtext("arxiv:doi", default="", namespaces=ns) or "").strip() or None
            comments = (entry.findtext("arxiv:comment", default="", namespaces=ns) or "").strip() or None
            primary_cat = None
            primary_node = entry.find("arxiv:primary_category", ns)
            if primary_node is not None:
                primary_cat = primary_node.get("term")

            doc = {
                "identifier": identifier,
                "title": title or identifier,
                "url": identifier,
                "last_modified": published,
                "author": ", ".join([a for a in authors if a]),
                "metadata": {
                    "authors": authors,
                    "categories": categories,
                    "primary_category": primary_cat,
                    "doi": doi,
                }
            }
            entries.append(doc)
            self.entry_cache[identifier] = {
                "title": doc["title"],
                "summary": summary,
                "authors": authors,
                "categories": categories,
                "primary_category": primary_cat,
                "doi": doi,
                "comments": comments,
                "pdf_url": pdf_url,
                "entry_url": identifier,
            }
        return entries

    @staticmethod
    def _parse_datetime(value: str) -> Optional[datetime]:
        if not value:
            return None
        try:
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            return datetime.fromisoformat(value)
        except Exception:
            try:
                return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S%z")
            except Exception:
                return None

    @staticmethod
    def _normalize_id(value: str) -> str:
        value = value.strip()
        if value.lower().startswith("arxiv:"):
            value = value.split(":", 1)[1]
        return value
