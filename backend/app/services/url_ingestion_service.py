"""
Service for scraping URLs and ingesting them into the KnowledgeDB.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentSource
from app.models.user import User
from app.services.document_service import DocumentService
from app.services.web_scraper_service import WebScraperService


class UrlIngestionService:
    def __init__(self):
        self.document_service = DocumentService()

    async def ingest_url(
        self,
        *,
        db: AsyncSession,
        user: User,
        url: str,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        follow_links: bool = False,
        max_pages: int = 1,
        max_depth: int = 0,
        same_domain_only: bool = True,
        one_document_per_page: bool = False,
        allow_private_networks: bool = False,
        max_content_chars: int = 50_000,
        publish: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        url = (url or "").strip()
        if not url:
            return {"error": "Missing required parameter: url"}

        publish = publish or (lambda _t, _p: None)

        is_allowlisted = await self._is_url_allowlisted_for_internal_scrape(url, db)

        allow_private_effective = False
        if allow_private_networks:
            if user.role == "admin":
                allow_private_effective = True
            elif is_allowlisted:
                allow_private_effective = True
            else:
                return {"error": "allow_private_networks requires admin role (or an active web source allowlist)"}
        else:
            allow_private_effective = bool(is_allowlisted)

        cancel_check = cancel_check or (lambda: False)

        if cancel_check():
            return {"error": "canceled"}

        # Scrape
        publish("status", {"stage": "scraping", "status": "Scraping pages...", "progress": 5})
        scraper = WebScraperService(enforce_network_safety=True)
        try:
            scrape_result = await scraper.scrape(
                url,
                follow_links=follow_links,
                max_pages=max_pages,
                max_depth=max_depth,
                same_domain_only=same_domain_only,
                include_links=False,
                allow_private_networks=allow_private_effective,
                max_content_chars=max_content_chars,
            )
        finally:
            await scraper.aclose()

        pages = scrape_result.get("pages") or []
        if not pages:
            return {"error": "No pages scraped (blocked, unreachable, or empty)"}

        ingest_source = await self.document_service._get_or_create_url_ingest_source(db)
        owner_display_name = user.full_name or user.username or user.email

        created: List[Dict[str, Any]] = []
        updated: List[Dict[str, Any]] = []
        skipped: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        async def upsert_page(page: Dict[str, Any]) -> None:
            page_url = page.get("url") or ""
            page_title = (page.get("title") or "").strip() or page_url
            page_content = (page.get("content") or "").strip()
            if not page_url or not page_content:
                skipped.append({"url": page_url, "reason": "missing_url_or_content"})
                return

            content_hash = hashlib.sha256(page_content.encode("utf-8")).hexdigest()
            existing_res = await db.execute(
                select(Document).where(
                    Document.source_id == ingest_source.id,
                    Document.source_identifier == page_url,
                )
            )
            existing = existing_res.scalar_one_or_none()

            if existing and existing.content_hash == content_hash:
                skipped.append({"document_id": str(existing.id), "url": page_url, "reason": "unchanged"})
                return

            try:
                if existing:
                    existing.title = page_title
                    existing.content = page_content
                    existing.content_hash = content_hash
                    existing.url = page_url
                    existing.file_type = "text/html"
                    existing.file_size = len(page_content.encode("utf-8"))
                    if tags is not None:
                        existing.tags = tags
                    existing.author = existing.author or owner_display_name
                    existing.extra_metadata = existing.extra_metadata or {}
                    existing.extra_metadata.update(
                        {
                            "origin": "url_ingest",
                            "source_url": page_url,
                            "scraped_at": datetime.utcnow().isoformat(),
                            "content_type": page.get("content_type"),
                            "status_code": page.get("status_code"),
                        }
                    )
                    existing.is_processed = False
                    await db.commit()
                    await self.document_service.reprocess_document(existing.id, db, user_id=user.id)
                    updated.append({"document_id": str(existing.id), "url": page_url, "title": existing.title})
                else:
                    doc = Document(
                        title=page_title,
                        content=page_content,
                        content_hash=content_hash,
                        url=page_url,
                        file_path=None,
                        file_type="text/html",
                        file_size=len(page_content.encode("utf-8")),
                        source_id=ingest_source.id,
                        source_identifier=page_url,
                        author=owner_display_name,
                        tags=tags,
                        extra_metadata={
                            "origin": "url_ingest",
                            "source_url": page_url,
                            "scraped_at": datetime.utcnow().isoformat(),
                            "content_type": page.get("content_type"),
                            "status_code": page.get("status_code"),
                        },
                        is_processed=False,
                    )
                    db.add(doc)
                    await db.commit()
                    await db.refresh(doc)
                    await self.document_service.reprocess_document(doc.id, db, user_id=user.id)
                    created.append({"document_id": str(doc.id), "url": page_url, "title": doc.title})
            except Exception as e:
                await db.rollback()
                errors.append({"url": page_url, "error": str(e)})

        publish("status", {"stage": "ingesting", "status": "Saving pages into KnowledgeDB...", "progress": 50})

        if follow_links and one_document_per_page:
            total = len(pages)
            for i, page in enumerate(pages, start=1):
                if cancel_check():
                    publish("status", {"stage": "canceled", "status": "Canceled", "progress": 100})
                    return {"error": "canceled"}
                publish("progress", {"stage": "ingesting", "current": i, "total": total, "progress": 50 + int(50 * (i / max(1, total)))})
                await upsert_page(page)
        else:
            combined_title = (title or "").strip() or (pages[0].get("title") or "").strip() or url
            combined_url = pages[0].get("url") or url
            combined_parts: List[str] = []
            for p in pages:
                p_title = (p.get("title") or "").strip()
                p_url = (p.get("url") or "").strip()
                p_content = (p.get("content") or "").strip()
                if not p_content:
                    continue
                header = p_title or p_url or "Page"
                if p_url and p_url != combined_url:
                    header = f"{header}\n{p_url}"
                combined_parts.append(f"{header}\n\n{p_content}")
            combined_content = "\n\n---\n\n".join(combined_parts).strip()

            publish("progress", {"stage": "ingesting", "current": 1, "total": 1, "progress": 90})
            if cancel_check():
                publish("status", {"stage": "canceled", "status": "Canceled", "progress": 100})
                return {"error": "canceled"}
            await upsert_page(
                {
                    "url": combined_url,
                    "title": combined_title,
                    "content": combined_content,
                    "content_type": "text/html",
                    "status_code": pages[0].get("status_code"),
                }
            )

        result = {
            "action": "ingested",
            "root_url": scrape_result.get("root_url"),
            "total_pages_scraped": int(scrape_result.get("total_pages") or len(pages)),
            "created": created,
            "updated": updated,
            "skipped": skipped,
            "errors": errors,
        }
        publish("complete", {"progress": 100, **result})
        return result

    async def _is_url_allowlisted_for_internal_scrape(self, url: str, db: AsyncSession) -> bool:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if not host:
            return False

        res = await db.execute(
            select(DocumentSource).where(
                DocumentSource.source_type == "web",
                DocumentSource.is_active == True,
            )
        )
        sources = res.scalars().all()

        def host_matches(allowed: str) -> bool:
            allowed = (allowed or "").strip().lower()
            if not allowed:
                return False
            return host == allowed or host.endswith("." + allowed)

        for source in sources:
            cfg = source.config or {}
            for d in (cfg.get("allowed_domains") or []):
                if host_matches(d):
                    return True
            for base in (cfg.get("base_urls") or []):
                try:
                    base_host = (urlparse(str(base)).hostname or "").lower()
                except Exception:
                    base_host = ""
                if base_host and host_matches(base_host):
                    return True

        return False
