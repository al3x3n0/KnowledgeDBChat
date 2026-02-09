"""
Metadata enrichment for scientific papers (arXiv).

Adds structured fields like BibTeX, DOI metadata (venue/publisher/year), and keywords.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple
from uuid import UUID

import httpx
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentSource


def _arxiv_id_from_source_identifier(source_identifier: Optional[str]) -> Optional[str]:
    if not source_identifier:
        return None
    parts = source_identifier.strip().rstrip("/").split("/")
    if not parts:
        return None
    last = parts[-1]
    return last or None


def _merge_unique_strings(existing: Any, additions: Any, *, limit: int = 50) -> list[str]:
    out: list[str] = []
    for src in (existing, additions):
        if not src:
            continue
        if isinstance(src, str):
            src = [src]
        if not isinstance(src, list):
            continue
        for v in src:
            if not isinstance(v, str):
                continue
            s = v.strip()
            if not s:
                continue
            out.append(s)
    seen = set()
    uniq: list[str] = []
    for s in out:
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(s)
        if len(uniq) >= limit:
            break
    return uniq


class PaperEnrichmentService:
    async def enrich_arxiv_document(self, db: AsyncSession, document_id: UUID, force: bool = False) -> Dict[str, Any]:
        doc = await db.get(Document, document_id)
        if not doc:
            raise ValueError("Document not found")
        src = await db.get(DocumentSource, doc.source_id)
        if not src or src.source_type != "arxiv":
            return {"skipped": True, "reason": "not_arxiv"}

        meta = doc.extra_metadata if isinstance(doc.extra_metadata, dict) else {}
        paper_meta = meta.get("paper_metadata") if isinstance(meta, dict) else None
        paper_meta = paper_meta if isinstance(paper_meta, dict) else {}

        if paper_meta.get("enriched_at") and not force:
            return {"skipped": True, "reason": "already_enriched"}

        arxiv_id = _arxiv_id_from_source_identifier(doc.source_identifier)
        doi = (meta.get("doi") or paper_meta.get("doi")) if isinstance(meta, dict) else None

        bibtex = None
        if arxiv_id and (force or not paper_meta.get("bibtex")):
            bibtex = await self._fetch_arxiv_bibtex(arxiv_id)

        crossref = None
        if doi and (force or not paper_meta.get("venue")):
            crossref = await self._fetch_crossref(doi)

        if crossref:
            venue, publisher, year, keywords, author_affiliations = self._normalize_crossref(crossref)
        else:
            venue, publisher, year, keywords, author_affiliations = (None, None, None, [], [])

        # Keywords: arXiv categories + Crossref subject
        categories = meta.get("categories") if isinstance(meta, dict) else None
        keywords_merged = _merge_unique_strings(paper_meta.get("keywords"), keywords)
        keywords_merged = _merge_unique_strings(keywords_merged, categories)

        from datetime import datetime as _dt
        updated_paper_meta = {
            **paper_meta,
            "arxiv_id": arxiv_id or paper_meta.get("arxiv_id"),
            "doi": doi or paper_meta.get("doi"),
            "bibtex": bibtex or paper_meta.get("bibtex"),
            "venue": venue or paper_meta.get("venue"),
            "publisher": publisher or paper_meta.get("publisher"),
            "year": year or paper_meta.get("year"),
            "keywords": keywords_merged,
            "author_affiliations": author_affiliations or paper_meta.get("author_affiliations") or [],
            "enriched_at": _dt.utcnow().isoformat(),
        }

        # Write back
        meta = dict(meta) if isinstance(meta, dict) else {}
        meta["paper_metadata"] = updated_paper_meta
        doc.extra_metadata = meta

        # Also keep tags somewhat aligned
        if keywords_merged:
            doc.tags = _merge_unique_strings(doc.tags, keywords_merged, limit=50)

        await db.commit()
        return {"skipped": False, "document_id": str(doc.id), "arxiv_id": arxiv_id, "doi": doi}

    async def _fetch_arxiv_bibtex(self, arxiv_id: str) -> Optional[str]:
        url = f"https://arxiv.org/bibtex/{arxiv_id}"
        try:
            async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": "KnowledgeDBChat/1.0"}) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                html = resp.text or ""
        except Exception as exc:
            logger.debug(f"arXiv bibtex fetch failed for {arxiv_id}: {exc}")
            return None

        # Page contains bibtex inside <pre>...</pre>
        start = html.find("<pre")
        if start == -1:
            return None
        start = html.find(">", start)
        end = html.find("</pre>", start)
        if start == -1 or end == -1:
            return None
        pre = html[start + 1:end]
        pre = pre.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        bib = pre.strip()
        return bib or None

    async def _fetch_crossref(self, doi: str) -> Optional[Dict[str, Any]]:
        doi = doi.strip()
        if not doi:
            return None
        url = f"https://api.crossref.org/works/{doi}"
        try:
            async with httpx.AsyncClient(timeout=20.0, headers={"User-Agent": "KnowledgeDBChat/1.0"}) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                payload = resp.json()
            msg = payload.get("message") if isinstance(payload, dict) else None
            return msg if isinstance(msg, dict) else None
        except Exception as exc:
            logger.debug(f"Crossref fetch failed for DOI {doi}: {exc}")
            return None

    def _normalize_crossref(
        self, msg: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[str], Optional[int], list[str], list[Dict[str, Any]]]:
        venue = None
        publisher = None
        year = None
        keywords: list[str] = []
        author_affiliations: list[Dict[str, Any]] = []

        try:
            publisher = msg.get("publisher")
            container = msg.get("container-title")
            if isinstance(container, list) and container:
                venue = container[0]
            elif isinstance(container, str):
                venue = container

            subjects = msg.get("subject")
            if isinstance(subjects, list):
                keywords = [s for s in subjects if isinstance(s, str) and s.strip()]

            authors = msg.get("author")
            if isinstance(authors, list):
                for a in authors[:50]:
                    if not isinstance(a, dict):
                        continue
                    given = (a.get("given") or "").strip()
                    family = (a.get("family") or "").strip()
                    name = " ".join([p for p in [given, family] if p]) or (a.get("name") or "").strip()
                    if not name:
                        continue
                    affs = a.get("affiliation")
                    aff_names: list[str] = []
                    if isinstance(affs, list):
                        for aff in affs:
                            if isinstance(aff, dict) and isinstance(aff.get("name"), str) and aff.get("name").strip():
                                aff_names.append(aff["name"].strip())
                    # Dedup affiliations
                    seen = set()
                    uniq_affs: list[str] = []
                    for s in aff_names:
                        k = s.lower()
                        if k in seen:
                            continue
                        seen.add(k)
                        uniq_affs.append(s)
                    author_affiliations.append(
                        {
                            "name": name,
                            "orcid": a.get("ORCID"),
                            "affiliations": uniq_affs,
                        }
                    )

            issued = msg.get("issued", {})
            parts = issued.get("date-parts") if isinstance(issued, dict) else None
            if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
                y = parts[0][0]
                year = int(y) if isinstance(y, (int, float, str)) and str(y).isdigit() else None
        except Exception:
            pass

        return venue, publisher, year, keywords, author_affiliations
