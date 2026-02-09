"""
Research notes API endpoints.

Research-native notes intended for labs: hypotheses, experiment plans, insights.
"""

from datetime import datetime
import json
import re
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.document import Document
from app.models.memory import UserPreferences
from app.models.research_note import ResearchNote
from app.models.synthesis_job import SynthesisJob
from app.models.user import User
from app.schemas.research_note import (
    ResearchNoteCreate,
    ResearchNoteEnforceCitationsRequest,
    ResearchNoteLintCitationsRequest,
    ResearchNotesLintRecentRequest,
    ResearchNotesLintRecentResponse,
    ResearchNoteUpdate,
    ResearchNoteResponse,
    ResearchNoteListResponse,
)
from app.services.auth_service import get_current_user
from app.services.llm_service import LLMService, UserLLMSettings
from app.services.vector_store import vector_store_service

router = APIRouter()


def _trim_attribution(attribution: Optional[Dict[str, Any]], *, include_details: bool) -> Optional[Dict[str, Any]]:
    if not isinstance(attribution, dict):
        return None

    if include_details:
        return attribution

    # List view should be lightweight: keep only summary fields and drop large blobs.
    keep_root_keys = {
        "generated_at",
        "policy",
        "update_content",
        "append_bibliography",
        "strict",
        "use_vector_snippets",
        "chunks_per_source",
        "chunk_max_chars",
        "max_sources",
        "max_source_chars",
        "chunk_query_used",
        "document_ids_used",
        "sources",
        "coverage",
        "total_citable_lines",
        "cited_citable_lines",
        "line_citation_coverage",
        "used_citation_keys",
        "unknown_citation_keys",
        "lint",
    }
    trimmed: Dict[str, Any] = {k: attribution.get(k) for k in keep_root_keys if k in attribution}

    # Trim lint too (avoid long example lists in list view).
    lint = trimmed.get("lint")
    if isinstance(lint, dict):
        keep_lint_keys = {
            "generated_at",
            "document_ids_used",
            "bibliography_present",
            "used_citation_keys",
            "unknown_citation_keys",
            "total_citable_lines",
            "cited_citable_lines",
            "line_citation_coverage",
            "notified_at",
            "notified_reasons",
        }
        trimmed["lint"] = {k: lint.get(k) for k in keep_lint_keys if k in lint}

    return trimmed


def _to_response(note: ResearchNote, *, include_attribution_details: bool = True) -> ResearchNoteResponse:
    source_doc_ids = None
    if isinstance(note.source_document_ids, list):
        source_doc_ids = []
        for x in note.source_document_ids:
            try:
                source_doc_ids.append(UUID(str(x)))
            except Exception:
                pass

    tags = note.tags if isinstance(note.tags, list) else None
    attribution = _trim_attribution(
        note.attribution if isinstance(note.attribution, dict) else None,
        include_details=include_attribution_details,
    )
    return ResearchNoteResponse(
        id=note.id,
        user_id=note.user_id,
        title=note.title,
        content_markdown=note.content_markdown,
        tags=tags,
        attribution=attribution,
        source_synthesis_job_id=note.source_synthesis_job_id,
        source_document_ids=source_doc_ids,
        created_at=note.created_at,
        updated_at=note.updated_at,
    )


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*", "", cleaned).strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in response")
    payload = cleaned[start : end + 1]
    return json.loads(payload)


async def _build_sources_payload(
    documents: List[Document],
    *,
    max_source_chars: int,
    use_vector_snippets: bool,
    chunks_per_source: int,
    chunk_max_chars: int,
    chunk_query: str,
) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for doc in documents:
        evidence: List[Dict[str, Any]] = []
        snippet = ""

        if use_vector_snippets:
            try:
                results = await vector_store_service.search(
                    query=chunk_query,
                    limit=chunks_per_source,
                    document_ids=[str(doc.id)],
                    apply_postprocessing=False,
                )
                for r in results:
                    md = r.get("metadata") or {}
                    text = (r.get("content") or r.get("page_content") or "").strip()
                    text = re.sub(r"\s+", " ", text)[:chunk_max_chars]
                    if not text:
                        continue
                    evidence.append(
                        {
                            "chunk_id": md.get("chunk_id"),
                            "chunk_index": md.get("chunk_index"),
                            "score": r.get("score"),
                            "excerpt": text,
                        }
                    )
                    snippet += (text + " ")
                snippet = snippet.strip()[:max_source_chars]
            except Exception as exc:
                logger.warning(f"Vector snippet search failed for doc {doc.id}: {exc}")

        if not snippet:
            raw = (doc.summary or doc.content or "").strip()
            snippet = re.sub(r"\s+", " ", raw)[:max_source_chars]

        sources.append(
            {
                "id": str(doc.id),
                "title": doc.title,
                "url": doc.url,
                "file_path": doc.file_path,
                "file_type": doc.file_type,
                "snippet": snippet,
                "metadata": doc.extra_metadata or {},
                "evidence": evidence,
            }
        )
    return sources


@router.get("", response_model=ResearchNoteListResponse)
async def list_research_notes(
    q: Optional[str] = Query(None, description="Search in title/content"),
    tag: Optional[str] = Query(None, description="Filter by a single tag"),
    include_attribution_details: bool = Query(False, description="Include full attribution payloads (larger responses)"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        base = select(ResearchNote).where(ResearchNote.user_id == current_user.id)

        if q:
            like = f"%{q.strip()}%"
            base = base.where(
                (ResearchNote.title.ilike(like)) | (ResearchNote.content_markdown.ilike(like))
            )

        if tag:
            # Tags are stored as JSON array; use a simple textual match for portability.
            base = base.where(ResearchNote.tags.cast(str).ilike(f"%{tag.strip()}%"))

        total = int((await db.execute(select(func.count()).select_from(base.subquery()))).scalar() or 0)
        result = await db.execute(base.order_by(desc(ResearchNote.updated_at)).offset(offset).limit(limit))
        items = [_to_response(n, include_attribution_details=include_attribution_details) for n in result.scalars().all()]
        return ResearchNoteListResponse(items=items, total=total, limit=limit, offset=offset)
    except Exception as exc:
        logger.error(f"Failed to list research notes: {exc}")
        raise HTTPException(status_code=500, detail="Failed to list research notes")


@router.post("", response_model=ResearchNoteResponse, status_code=201)
async def create_research_note(
    payload: ResearchNoteCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        # Optional: verify synthesis job ownership for provenance
        synthesis_job_id = payload.source_synthesis_job_id
        if synthesis_job_id:
            job = await db.get(SynthesisJob, synthesis_job_id)
            if not job or job.user_id != current_user.id:
                raise HTTPException(status_code=404, detail="Synthesis job not found")

        note = ResearchNote(
            user_id=current_user.id,
            title=payload.title,
            content_markdown=payload.content_markdown,
            tags=payload.tags,
            source_synthesis_job_id=synthesis_job_id,
            source_document_ids=[str(x) for x in (payload.source_document_ids or [])] or None,
        )
        db.add(note)
        await db.commit()
        await db.refresh(note)
        return _to_response(note)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to create research note: {exc}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create research note")


@router.get("/{note_id}", response_model=ResearchNoteResponse)
async def get_research_note(
    note_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    note = await db.get(ResearchNote, note_id)
    if not note or note.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Research note not found")
    return _to_response(note)


@router.patch("/{note_id}", response_model=ResearchNoteResponse)
async def update_research_note(
    note_id: UUID,
    payload: ResearchNoteUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    note = await db.get(ResearchNote, note_id)
    if not note or note.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Research note not found")

    if payload.title is not None:
        note.title = payload.title
    if payload.content_markdown is not None:
        note.content_markdown = payload.content_markdown
    if payload.tags is not None:
        note.tags = payload.tags

    await db.commit()
    await db.refresh(note)
    return _to_response(note)


@router.delete("/{note_id}", status_code=204)
async def delete_research_note(
    note_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    note = await db.get(ResearchNote, note_id)
    if not note or note.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Research note not found")
    await db.delete(note)
    await db.commit()


@router.post("/{note_id}/enforce-citations", response_model=ResearchNoteResponse)
async def enforce_research_note_citations(
    note_id: UUID,
    payload: ResearchNoteEnforceCitationsRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    note = await db.get(ResearchNote, note_id)
    if not note or note.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Research note not found")

    if payload.document_ids:
        doc_ids: List[UUID] = payload.document_ids
    else:
        doc_ids = []
        if isinstance(note.source_document_ids, list) and note.source_document_ids:
            for x in note.source_document_ids:
                try:
                    doc_ids.append(UUID(str(x)))
                except Exception:
                    pass

        if not doc_ids and note.source_synthesis_job_id:
            job = await db.get(SynthesisJob, note.source_synthesis_job_id)
            if job and job.user_id == current_user.id and isinstance(job.document_ids, list):
                for x in job.document_ids:
                    try:
                        doc_ids.append(UUID(str(x)))
                    except Exception:
                        pass

    if not doc_ids:
        raise HTTPException(
            status_code=400,
            detail="No source documents available for this note. Provide document_ids or set source_document_ids.",
        )

    doc_ids = doc_ids[: payload.max_sources]
    result = await db.execute(select(Document).where(Document.id.in_(doc_ids)))
    documents_by_id: Dict[str, Document] = {str(d.id): d for d in result.scalars().all()}
    documents: List[Document] = []
    for doc_id in doc_ids:
        doc = documents_by_id.get(str(doc_id))
        if doc:
            documents.append(doc)

    if not documents:
        raise HTTPException(status_code=400, detail="No source documents found for provided IDs")

    chunk_query = (payload.chunk_query or "").strip()
    if not chunk_query:
        chunk_query = f"{note.title}\n{note_markdown[:1200]}"

    sources_payload = await _build_sources_payload(
        documents,
        max_source_chars=payload.max_source_chars,
        use_vector_snippets=payload.use_vector_snippets,
        chunks_per_source=payload.chunks_per_source,
        chunk_max_chars=payload.chunk_max_chars,
        chunk_query=chunk_query,
    )
    sources_index = [
        {
            "key": f"S{i + 1}",
            "doc_id": s["id"],
            "title": s.get("title"),
            "url": s.get("url"),
        }
        for i, s in enumerate(sources_payload)
    ]

    evidence_index = [
        {"key": f"S{i + 1}", "doc_id": s["id"], "evidence": s.get("evidence") or []}
        for i, s in enumerate(sources_payload)
    ]
    note_markdown = (note.content_markdown or "").strip()
    if len(note_markdown) > payload.max_note_chars:
        note_markdown = note_markdown[: payload.max_note_chars].rstrip() + "\n\n[TRUNCATED]\n"

    strict_block = ""
    if payload.strict:
        strict_block = (
            "STRICT MODE:\n"
            "- Do not keep any sentence with a factual/technical claim unless it is directly supported by the provided snippets.\n"
            "- Every non-heading, non-code sentence/paragraph that contains a claim must include at least one [[S#]] citation.\n\n"
        )

    prompt = (
        "You are a citation-aware research writing assistant.\n"
        "Rewrite the NOTE MARKDOWN so that factual/technical claims are supported by the provided SOURCES.\n"
        "You MUST NOT invent sources or cite anything not in SOURCES.\n"
        "If a claim cannot be supported by SOURCES, list it under unsupported_claims and revise the output to remove or qualify it.\n\n"
        "Citation format:\n"
        "- Assign short keys S1..Sn to the provided sources (in the same order as SOURCES).\n"
        "- Add citations as [[S#]] at the end of the sentence (policy=sentence) or paragraph (policy=paragraph).\n\n"
        f"{strict_block}"
        "Output ONLY a JSON object with this shape:\n"
        "{\n"
        '  \"content_markdown_cited\": \"string\",\n'
        '  \"sources\": [ { \"key\": \"S1\", \"doc_id\": \"uuid\", \"title\": \"string\", \"url\": \"string|null\" } ],\n'
        '  \"unsupported_claims\": [ { \"claim\": \"string\", \"reason\": \"string\", \"suggested_fix\": \"string\" } ],\n'
        '  \"coverage\": number\n'
        "}\n\n"
        f"policy: {payload.policy}\n\n"
        f"SOURCES (JSON):\n{json.dumps(sources_payload, ensure_ascii=True)}\n\n"
        f"NOTE MARKDOWN:\n{note_markdown}\n"
    )

    user_settings: Optional[UserLLMSettings] = None
    try:
        prefs_result = await db.execute(select(UserPreferences).where(UserPreferences.user_id == current_user.id))
        user_prefs = prefs_result.scalar_one_or_none()
        if user_prefs:
            user_settings = UserLLMSettings.from_preferences(user_prefs)
    except Exception as exc:
        logger.warning(f"Could not load user LLM preferences: {exc}")

    llm = LLMService()
    try:
        response_text = await llm.generate_response(
            query=prompt,
            user_settings=user_settings,
            task_type="summarization",
            user_id=current_user.id,
            db=db,
        )
        data = _extract_json(response_text)
    except Exception as exc:
        logger.error(f"Citation enforcement failed: {exc}")
        raise HTTPException(status_code=500, detail="Failed to enforce citations") from exc

    generated_markdown = (data.get("content_markdown_cited") or "").strip()

    def _is_line_citable(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        if s.startswith("#"):
            return False
        if s.startswith("```") or s.startswith(">"):
            return False
        return bool(re.search(r"[A-Za-z0-9]", s))

    uncited_examples: List[Dict[str, Any]] = []
    total_citable_lines = 0
    cited_citable_lines = 0
    used_citation_keys: List[str] = []
    unknown_citation_keys: List[str] = []
    if generated_markdown:
        for idx, line in enumerate(generated_markdown.splitlines(), start=1):
            if not _is_line_citable(line):
                continue
            total_citable_lines += 1
            if "[[S" in line:
                cited_citable_lines += 1
                continue
            if payload.max_uncited_examples > 0:
                uncited_examples.append({"line_no": idx, "line": line[:500]})
                if len(uncited_examples) >= payload.max_uncited_examples:
                    break

    # Always validate citation keys (even if not strict)
    if generated_markdown:
        max_key_num = len(sources_index)
        seen = set()
        for m in re.finditer(r"\[\[S(\d+)\]\]", generated_markdown):
            num = int(m.group(1))
            key = f"S{num}"
            if key in seen:
                continue
            seen.add(key)
            used_citation_keys.append(key)
            if num < 1 or num > max_key_num:
                unknown_citation_keys.append(key)

    bibliography_markdown = ""
    if sources_index:
        lines = ["## Sources", ""]
        for s in sources_index:
            key = s.get("key") or ""
            title = (s.get("title") or s.get("doc_id") or "").strip()
            url = (s.get("url") or "").strip()
            if url:
                lines.append(f"- **{key}**: {title} ({url})")
            else:
                lines.append(f"- **{key}**: {title}")
        bibliography_markdown = "\n".join(lines).strip() + "\n"

    final_markdown = generated_markdown
    if payload.append_bibliography and bibliography_markdown:
        if final_markdown and not final_markdown.endswith("\n"):
            final_markdown += "\n"
        final_markdown += "\n" + bibliography_markdown

    report: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(),
        "policy": payload.policy,
        "update_content": payload.update_content,
        "append_bibliography": payload.append_bibliography,
        "strict": payload.strict,
        "use_vector_snippets": payload.use_vector_snippets,
        "chunks_per_source": payload.chunks_per_source,
        "chunk_max_chars": payload.chunk_max_chars,
        "max_sources": payload.max_sources,
        "max_source_chars": payload.max_source_chars,
        "chunk_query_used": (chunk_query or "")[:500],
        "document_ids_used": [str(x) for x in doc_ids],
        "sources": sources_index,
        "evidence": evidence_index,
        "unsupported_claims": data.get("unsupported_claims") or [],
        "uncited_examples": uncited_examples,
        "total_citable_lines": total_citable_lines or None,
        "cited_citable_lines": cited_citable_lines or None,
        "line_citation_coverage": (float(cited_citable_lines) / float(total_citable_lines)) if total_citable_lines else None,
        "used_citation_keys": used_citation_keys,
        "unknown_citation_keys": unknown_citation_keys,
        "coverage": data.get("coverage"),
        "bibliography_markdown": bibliography_markdown or None,
        "generated_markdown": final_markdown,
    }

    note.attribution = report
    if payload.update_content and final_markdown:
        note.content_markdown = final_markdown

    await db.commit()
    await db.refresh(note)
    return _to_response(note)


@router.post("/{note_id}/lint-citations", response_model=ResearchNoteResponse)
async def lint_research_note_citations(
    note_id: UUID,
    payload: ResearchNoteLintCitationsRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    note = await db.get(ResearchNote, note_id)
    if not note or note.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Research note not found")

    if payload.document_ids:
        doc_ids: List[UUID] = payload.document_ids
    else:
        doc_ids = []
        if isinstance(note.source_document_ids, list) and note.source_document_ids:
            for x in note.source_document_ids:
                try:
                    doc_ids.append(UUID(str(x)))
                except Exception:
                    pass

        if not doc_ids and note.source_synthesis_job_id:
            job = await db.get(SynthesisJob, note.source_synthesis_job_id)
            if job and job.user_id == current_user.id and isinstance(job.document_ids, list):
                for x in job.document_ids:
                    try:
                        doc_ids.append(UUID(str(x)))
                    except Exception:
                        pass

    if not doc_ids:
        raise HTTPException(
            status_code=400,
            detail="No source documents available for this note. Provide document_ids or set source_document_ids.",
        )

    doc_ids = doc_ids[: payload.max_sources]
    result = await db.execute(select(Document).where(Document.id.in_(doc_ids)))
    documents_by_id: Dict[str, Document] = {str(d.id): d for d in result.scalars().all()}
    documents: List[Document] = []
    for doc_id in doc_ids:
        doc = documents_by_id.get(str(doc_id))
        if doc:
            documents.append(doc)

    sources_index = []
    for i, doc in enumerate(documents):
        sources_index.append(
            {
                "key": f"S{i + 1}",
                "doc_id": str(doc.id),
                "title": doc.title,
                "url": doc.url,
            }
        )

    # Lint current note markdown (not the generated markdown).
    markdown = (note.content_markdown or "").strip()

    def _is_line_citable(line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        if s.startswith("#"):
            return False
        if s.startswith("```") or s.startswith(">"):
            return False
        return bool(re.search(r"[A-Za-z0-9]", s))

    used_citation_keys: List[str] = []
    unknown_citation_keys: List[str] = []
    total_citable_lines = 0
    cited_citable_lines = 0
    uncited_examples: List[Dict[str, Any]] = []

    max_key_num = len(sources_index)
    seen_keys = set()
    for m in re.finditer(r"\[\[S(\d+)\]\]", markdown):
        num = int(m.group(1))
        key = f"S{num}"
        if key in seen_keys:
            continue
        seen_keys.add(key)
        used_citation_keys.append(key)
        if num < 1 or num > max_key_num:
            unknown_citation_keys.append(key)

    for idx, line in enumerate(markdown.splitlines(), start=1):
        if not _is_line_citable(line):
            continue
        total_citable_lines += 1
        if "[[S" in line:
            cited_citable_lines += 1
            continue
        if payload.max_uncited_examples > 0:
            uncited_examples.append({"line_no": idx, "line": line[:500]})
            if len(uncited_examples) >= payload.max_uncited_examples:
                break

    bibliography_present = bool(re.search(r"^##\\s+Sources\\s*$", markdown, flags=re.MULTILINE))
    bibliography_keys: List[str] = []
    if bibliography_present:
        for m in re.finditer(r"\\*\\*S(\\d+)\\*\\*", markdown):
            bibliography_keys.append(f"S{m.group(1)}")
        bibliography_keys = sorted(set(bibliography_keys))

    lint_report: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(),
        "document_ids_used": [str(x) for x in doc_ids],
        "sources": sources_index,
        "bibliography_present": bibliography_present,
        "bibliography_keys": bibliography_keys,
        "used_citation_keys": used_citation_keys,
        "unknown_citation_keys": unknown_citation_keys,
        "total_citable_lines": total_citable_lines or None,
        "cited_citable_lines": cited_citable_lines or None,
        "line_citation_coverage": (float(cited_citable_lines) / float(total_citable_lines)) if total_citable_lines else None,
        "uncited_examples": uncited_examples,
    }

    existing = note.attribution if isinstance(note.attribution, dict) else {}
    existing = {**existing, "lint": lint_report}
    note.attribution = existing
    await db.commit()
    await db.refresh(note)
    return _to_response(note)


@router.post("/lint-recent", response_model=ResearchNotesLintRecentResponse)
async def lint_recent_research_notes(
    payload: ResearchNotesLintRecentRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Lint citations for recently updated notes for the current user (no LLM).

    This updates `note.attribution.lint` for eligible notes.
    """
    from datetime import timedelta

    now = datetime.utcnow()
    window = timedelta(hours=int(payload.window_hours))

    processed = 0
    updated = 0
    skipped = 0
    missing_sources = 0

    stmt = (
        select(ResearchNote)
        .where(
            ResearchNote.user_id == current_user.id,
            ResearchNote.updated_at >= (now - window),
        )
        .order_by(desc(ResearchNote.updated_at))
        .limit(int(payload.max_notes))
    )
    res = await db.execute(stmt)
    notes = list(res.scalars().all())

    def _parse_ts(v: str | None) -> datetime | None:
        if not v:
            return None
        try:
            return datetime.fromisoformat(v.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            return None

    def _is_line_citable(line: str) -> bool:
        s = (line or "").strip()
        if not s:
            return False
        if s.startswith("#"):
            return False
        if s.startswith("```") or s.startswith(">"):
            return False
        return bool(re.search(r"[A-Za-z0-9]", s))

    for note in notes:
        processed += 1
        attribution = note.attribution if isinstance(note.attribution, dict) else {}
        lint = attribution.get("lint") if isinstance(attribution.get("lint"), dict) else None
        last_linted_at = _parse_ts(str(lint.get("generated_at")) if lint else None)
        note_updated_at = note.updated_at.replace(tzinfo=None) if note.updated_at else None
        if last_linted_at and note_updated_at and last_linted_at >= note_updated_at:
            skipped += 1
            continue

        doc_ids: List[UUID] = []
        if isinstance(note.source_document_ids, list) and note.source_document_ids:
            for x in note.source_document_ids:
                try:
                    doc_ids.append(UUID(str(x)))
                except Exception:
                    pass
        if not doc_ids and note.source_synthesis_job_id:
            job = await db.get(SynthesisJob, note.source_synthesis_job_id)
            if job and job.user_id == current_user.id and isinstance(job.document_ids, list):
                for x in job.document_ids:
                    try:
                        doc_ids.append(UUID(str(x)))
                    except Exception:
                        pass

        if not doc_ids:
            missing_sources += 1
            continue

        doc_ids = doc_ids[: int(payload.max_sources)]
        doc_res = await db.execute(select(Document).where(Document.id.in_(doc_ids)))
        documents_by_id: Dict[str, Document] = {str(d.id): d for d in doc_res.scalars().all()}
        documents: List[Document] = []
        for did in doc_ids:
            d = documents_by_id.get(str(did))
            if d:
                documents.append(d)

        sources_index = [
            {"key": f"S{i2 + 1}", "doc_id": str(d.id), "title": d.title, "url": d.url}
            for i2, d in enumerate(documents)
        ]
        max_key_num = len(sources_index)

        markdown = (note.content_markdown or "").strip()
        used_citation_keys: List[str] = []
        unknown_citation_keys: List[str] = []
        seen_keys = set()
        for m in re.finditer(r"\[\[S(\d+)\]\]", markdown):
            num = int(m.group(1))
            key = f"S{num}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            used_citation_keys.append(key)
            if num < 1 or num > max_key_num:
                unknown_citation_keys.append(key)

        total_citable_lines = 0
        cited_citable_lines = 0
        uncited_examples: List[Dict[str, Any]] = []
        for line_no, line in enumerate(markdown.splitlines(), start=1):
            if not _is_line_citable(line):
                continue
            total_citable_lines += 1
            if "[[S" in line:
                cited_citable_lines += 1
                continue
            if payload.max_uncited_examples > 0:
                uncited_examples.append({"line_no": line_no, "line": line[:500]})
                if len(uncited_examples) >= int(payload.max_uncited_examples):
                    break

        bibliography_present = bool(re.search(r"^##\\s+Sources\\s*$", markdown, flags=re.MULTILINE))
        lint_report = {
            "generated_at": now.isoformat(),
            "document_ids_used": [str(x) for x in doc_ids],
            "sources": sources_index,
            "bibliography_present": bibliography_present,
            "used_citation_keys": used_citation_keys,
            "unknown_citation_keys": unknown_citation_keys,
            "total_citable_lines": total_citable_lines or None,
            "cited_citable_lines": cited_citable_lines or None,
            "line_citation_coverage": (float(cited_citable_lines) / float(total_citable_lines)) if total_citable_lines else None,
            "uncited_examples": uncited_examples,
        }

        note.attribution = {**attribution, "lint": lint_report}
        updated += 1

    await db.commit()
    return ResearchNotesLintRecentResponse(
        processed=processed,
        updated=updated,
        skipped=skipped,
        missing_sources=missing_sources,
    )
