"""
Services for extracting structured paper data from arXiv-backed documents.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.research_note import ResearchNote
from app.models.research_paper import PaperClaim, PaperExtractionJob, ResearchPaper
from app.services.llm_service import LLMService


class PaperExtractionService:
    EXTRACTOR_VERSION = "paper_extraction_v1"
    VALID_CLAIM_KINDS = {
        "performance",
        "compile_time",
        "code_size",
        "energy",
        "correctness",
        "robustness",
        "other",
    }
    VALID_TARGET_LAYERS = {
        "source",
        "ir",
        "midend",
        "backend",
        "runtime",
        "hardware",
        "unknown",
    }

    def __init__(self) -> None:
        self.llm = LLMService()

    async def queue_document_extraction_job(
        self,
        db: AsyncSession,
        *,
        document: Document,
        user_id: UUID,
        force: bool = False,
        dispatch_task: Optional[Callable[[str], Any]] = None,
    ) -> PaperExtractionJob:
        await db.refresh(document, ["source"])
        self._validate_document(document)

        existing_stmt = select(ResearchPaper).where(
            ResearchPaper.document_id == document.id
        )
        existing = (await db.execute(existing_stmt)).scalar_one_or_none()
        had_completed_extraction = bool(
            existing and existing.extracted_at and existing.raw_extraction_payload
        )

        if existing and not force and existing.extraction_status == "completed":
            raise PaperExtractionConflictError(
                f"Document {document.id} already has extracted structure"
            )

        job = PaperExtractionJob(
            user_id=user_id,
            document_id=document.id,
            source_id=document.source_id,
            paper_id=existing.id if existing else None,
            status="pending",
            extractor_version=self.EXTRACTOR_VERSION,
            request_payload={
                "force": force,
                "previous_extraction_status": existing.extraction_status
                if existing
                else None,
                "had_completed_extraction": had_completed_extraction,
            },
        )
        db.add(job)
        if existing:
            existing.extraction_status = "pending"
            existing.extractor_version = self.EXTRACTOR_VERSION
        await db.commit()
        await db.refresh(job)

        if dispatch_task is not None:
            dispatch_task(str(job.id))
        return job

    async def extract_document(
        self,
        db: AsyncSession,
        *,
        document: Document,
        user_id: UUID,
        job: PaperExtractionJob,
    ) -> ResearchPaper:
        self._validate_document(document)

        job.status = "running"
        job.started_at = datetime.utcnow()
        job.error = None
        await db.commit()

        existing_stmt = select(ResearchPaper).where(
            ResearchPaper.document_id == document.id
        )
        existing = (await db.execute(existing_stmt)).scalar_one_or_none()

        if existing:
            existing.extraction_status = "running"
            existing.extractor_version = self.EXTRACTOR_VERSION
            await db.commit()

        prompt = self._build_prompt(document)
        raw = await self.llm.generate_response(
            query=prompt,
            context=None,
            temperature=0.1,
            max_tokens=2200,
            task_type="workflow_synthesis",
            user_id=user_id,
            db=db,
        )
        parsed = self._parse_json(raw)

        paper = await self._upsert_research_paper(
            db,
            document=document,
            user_id=user_id,
            payload=parsed,
            existing=existing,
        )

        job.status = "completed"
        job.paper_id = paper.id
        job.extractor_version = self.EXTRACTOR_VERSION
        job.result_summary = {
            "paper_id": str(paper.id),
            "claims_count": len(paper.claims),
            "mechanisms_count": len(paper.mechanisms or []),
        }
        job.completed_at = datetime.utcnow()
        await db.commit()
        await db.refresh(paper)
        return paper

    async def mark_failed(
        self, db: AsyncSession, *, job: PaperExtractionJob, error: str
    ) -> None:
        job.status = "failed"
        job.error = error[:20000]
        job.completed_at = datetime.utcnow()
        request_payload = (
            job.request_payload if isinstance(job.request_payload, dict) else {}
        )
        previous_status = str(
            request_payload.get("previous_extraction_status") or ""
        ).strip()
        had_completed_extraction = bool(request_payload.get("had_completed_extraction"))
        if job.paper_id:
            paper = await db.get(ResearchPaper, job.paper_id)
            if paper:
                if (
                    had_completed_extraction
                    and paper.extracted_at
                    and paper.raw_extraction_payload
                ):
                    paper.extraction_status = previous_status or "completed"
                else:
                    paper.extraction_status = "failed"
        else:
            stmt = select(ResearchPaper).where(
                ResearchPaper.document_id == job.document_id
            )
            paper = (await db.execute(stmt)).scalar_one_or_none()
            if paper:
                if (
                    had_completed_extraction
                    and paper.extracted_at
                    and paper.raw_extraction_payload
                ):
                    paper.extraction_status = previous_status or "completed"
                else:
                    paper.extraction_status = "failed"
        await db.commit()

    async def create_research_note(
        self,
        db: AsyncSession,
        *,
        paper: ResearchPaper,
        user_id: UUID,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> ResearchNote:
        claims = sorted(
            list(paper.claims or []),
            key=lambda item: (
                item.rank is None,
                item.rank if item.rank is not None else 999999,
            ),
        )
        lines = [
            f"# {title or f'Paper Extraction: {paper.title}'}",
            "",
            f"- arXiv ID: `{paper.arxiv_id}`",
            f"- Extraction status: `{paper.extraction_status}`",
        ]
        if paper.paper_url:
            lines.append(f"- Paper URL: {paper.paper_url}")
        if paper.summary:
            lines.extend(["", "## Summary", "", paper.summary.strip()])
        for heading, values in (
            ("Mechanisms", paper.mechanisms or []),
            ("Assumptions", paper.assumptions or []),
            ("Benchmarks", paper.benchmarks or []),
            ("Metrics", paper.metrics or []),
            ("Limitations", paper.limitations or []),
        ):
            if values:
                lines.extend(["", f"## {heading}", ""])
                lines.extend(
                    [
                        f"- {str(value).strip()}"
                        for value in values
                        if str(value).strip()
                    ]
                )
        if claims:
            lines.extend(["", "## Claims", ""])
            for claim in claims:
                label = f"[{claim.kind}] " if claim.kind else ""
                lines.append(f"- {label}{claim.statement}")
                if claim.evidence_summary:
                    lines.append(f"  Evidence: {claim.evidence_summary}")
        note = ResearchNote(
            user_id=user_id,
            title=title or f"Paper Extraction: {paper.title}",
            content_markdown="\n".join(lines).strip(),
            tags=tags or ["paper-extraction", "arxiv"],
            source_document_ids=[str(paper.document_id)],
            structured_payload={
                "artifact_type": "paper_extraction",
                "paper_id": str(paper.id),
                "arxiv_id": paper.arxiv_id,
                "summary": paper.summary,
                "mechanisms": paper.mechanisms or [],
                "assumptions": paper.assumptions or [],
                "benchmarks": paper.benchmarks or [],
                "metrics": paper.metrics or [],
                "limitations": paper.limitations or [],
                "claims": [
                    {
                        "id": str(claim.id),
                        "kind": claim.kind,
                        "statement": claim.statement,
                        "mechanism": claim.mechanism,
                        "target_layer": claim.target_layer,
                        "confidence": claim.confidence,
                        "evidence_summary": claim.evidence_summary,
                        "rank": claim.rank,
                    }
                    for claim in claims
                ],
            },
        )
        db.add(note)
        await db.commit()
        await db.refresh(note)
        return note

    def _validate_document(self, document: Document) -> None:
        source = getattr(document, "source", None)
        if not source or getattr(source, "source_type", None) != "arxiv":
            raise UnsupportedPaperSourceError(
                "Only arXiv documents are supported for paper extraction"
            )

    def _build_prompt(self, document: Document) -> str:
        metadata = (
            document.extra_metadata if isinstance(document.extra_metadata, dict) else {}
        )
        abstract = (document.summary or document.content or "").strip()
        if len(abstract) > 18000:
            abstract = abstract[:18000]
        return (
            "Extract structured paper metadata and claims from this arXiv paper.\n"
            "Return valid JSON only.\n"
            "Schema:\n"
            "{"
            '"summary": string,'
            '"mechanisms": [string],'
            '"assumptions": [string],'
            '"benchmarks": [string],'
            '"metrics": [string],'
            '"limitations": [string],'
            '"claims": ['
            "{"
            '"kind": "performance|compile_time|code_size|energy|correctness|robustness|other",'
            '"statement": string,'
            '"mechanism": string|null,'
            '"target_layer": "source|ir|midend|backend|runtime|hardware|unknown",'
            '"conditions": [string],'
            '"assumptions": [string],'
            '"expected_effect": string|null,'
            '"evidence_summary": string|null,'
            '"confidence": number|null,'
            '"tags": [string],'
            '"rank": integer'
            "}"
            "]"
            "}\n"
            "Rules:\n"
            "- Keep claims concrete and falsifiable.\n"
            "- Produce 5 to 15 claims.\n"
            "- Prefer short normalized phrases in list fields.\n"
            "- If uncertain, put the uncertainty in evidence_summary instead of inventing facts.\n\n"
            f"Title: {document.title}\n"
            f"arXiv ID: {document.source_identifier}\n"
            f"Authors: {metadata.get('authors') or ([document.author] if document.author else [])}\n"
            f"Categories: {metadata.get('categories') or []}\n"
            f"Document text:\n{abstract}"
        )

    def _parse_json(self, raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            data = raw
        else:
            text = str(raw or "").strip()
            if text.startswith("```"):
                text = re.sub(r"^```[a-zA-Z0-9_-]*", "", text).strip()
                if text.endswith("```"):
                    text = text[:-3].strip()
            try:
                data = json.loads(text)
            except Exception:
                match = re.search(r"\{.*\}", text, flags=re.DOTALL)
                if not match:
                    raise ValueError("Model did not return valid JSON")
                data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise ValueError("Model did not return an object")
        if not isinstance(data.get("claims"), list) or not data["claims"]:
            raise ValueError("Extraction did not include claims")
        return data

    async def _upsert_research_paper(
        self,
        db: AsyncSession,
        *,
        document: Document,
        user_id: UUID,
        payload: Dict[str, Any],
        existing: Optional[ResearchPaper],
    ) -> ResearchPaper:
        metadata = (
            document.extra_metadata if isinstance(document.extra_metadata, dict) else {}
        )
        paper = existing or ResearchPaper(
            user_id=user_id,
            document_id=document.id,
            source_id=document.source_id,
            arxiv_id=document.source_identifier,
            title=document.title,
        )
        if not existing:
            db.add(paper)
            await db.flush()

        paper.source_id = document.source_id
        paper.title = document.title
        paper.arxiv_id = document.source_identifier
        paper.authors = self._normalize_string_list(
            metadata.get("authors") or ([document.author] if document.author else [])
        )
        paper.abstract = (
            (document.summary or document.content or "")[:20000]
            if (document.summary or document.content)
            else None
        )
        paper.categories = self._normalize_string_list(metadata.get("categories"))
        paper.paper_url = (
            document.url or f"https://arxiv.org/abs/{document.source_identifier}"
        )
        paper.pdf_url = f"https://arxiv.org/pdf/{document.source_identifier}.pdf"
        paper.summary = self._normalize_optional_string(payload.get("summary"))
        paper.mechanisms = self._normalize_string_list(payload.get("mechanisms"))
        paper.assumptions = self._normalize_string_list(payload.get("assumptions"))
        paper.benchmarks = self._normalize_string_list(payload.get("benchmarks"))
        paper.metrics = self._normalize_string_list(payload.get("metrics"))
        paper.limitations = self._normalize_string_list(payload.get("limitations"))
        paper.raw_extraction_payload = payload
        paper.extraction_status = "completed"
        paper.extracted_at = datetime.utcnow()
        paper.extractor_version = self.EXTRACTOR_VERSION

        await db.execute(delete(PaperClaim).where(PaperClaim.paper_id == paper.id))
        await db.flush()

        claims = []
        for index, claim_payload in enumerate(payload.get("claims") or [], start=1):
            if not isinstance(claim_payload, dict):
                continue
            statement = self._normalize_optional_string(claim_payload.get("statement"))
            if not statement:
                continue
            kind = str(claim_payload.get("kind") or "other").strip().lower()
            if kind not in self.VALID_CLAIM_KINDS:
                kind = "other"
            target_layer = (
                str(claim_payload.get("target_layer") or "unknown").strip().lower()
            )
            if target_layer not in self.VALID_TARGET_LAYERS:
                target_layer = "unknown"
            confidence = claim_payload.get("confidence")
            try:
                confidence_value = (
                    max(0.0, min(1.0, float(confidence)))
                    if confidence is not None
                    else None
                )
            except Exception:
                confidence_value = None
            claim = PaperClaim(
                paper_id=paper.id,
                kind=kind,
                statement=statement,
                mechanism=self._normalize_optional_string(
                    claim_payload.get("mechanism")
                ),
                target_layer=target_layer,
                conditions=self._normalize_string_list(claim_payload.get("conditions")),
                assumptions=self._normalize_string_list(
                    claim_payload.get("assumptions")
                ),
                expected_effect=self._normalize_optional_string(
                    claim_payload.get("expected_effect")
                ),
                evidence_summary=self._normalize_optional_string(
                    claim_payload.get("evidence_summary")
                ),
                confidence=confidence_value,
                tags=self._normalize_string_list(claim_payload.get("tags")),
                rank=int(claim_payload.get("rank") or index),
            )
            claims.append(claim)
        paper.claims = claims
        await db.commit()
        await db.refresh(paper)
        return paper

    def _normalize_string_list(self, value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        cleaned = []
        seen = set()
        for item in value:
            text = str(item or "").strip()
            if not text:
                continue
            if text in seen:
                continue
            seen.add(text)
            cleaned.append(text[:1000])
        return cleaned

    def _normalize_optional_string(self, value: Any) -> Optional[str]:
        text = str(value or "").strip()
        return text[:20000] if text else None


paper_extraction_service = PaperExtractionService()


class PaperExtractionError(Exception):
    pass


class UnsupportedPaperSourceError(PaperExtractionError):
    pass


class PaperExtractionConflictError(PaperExtractionError):
    pass
