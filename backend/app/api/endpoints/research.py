"""
Research endpoints (external sources).
"""

import httpx
from pydantic import BaseModel, Field
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
import sqlalchemy as sa
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID

from app.core.database import get_db
from app.models.document import Document
from app.models.document import DocumentSource
from app.models.user import User
from app.services.auth_service import get_current_user
from app.services.arxiv_search_service import ArxivSearchService
from app.services.llm_service import LLMService

router = APIRouter()


@router.get("/arxiv/search")
async def search_arxiv(
    q: str = Query(..., min_length=2, max_length=500, description="arXiv search query (API syntax)"),
    start: int = Query(0, ge=0, le=1000),
    max_results: int = Query(10, ge=1, le=50),
    sort_by: str = Query("relevance", pattern="^(relevance|lastUpdatedDate|submittedDate)$"),
    sort_order: str = Query("descending", pattern="^(ascending|descending)$"),
    current_user: User = Depends(get_current_user),
):
    """
    Search scientific papers on arXiv and return metadata + abstracts.

    The `q` parameter uses arXiv API query syntax, e.g.:
    - `all:transformers AND cat:cs.CL`
    - `ti:\"diffusion models\"`
    """
    _ = current_user  # authenticated access required
    q_clean = (q or "").strip()
    # Prevent common placeholder queries like "all:" which cause arXiv 400s.
    if q_clean.endswith(":"):
        raise HTTPException(status_code=422, detail="Invalid arXiv query: missing term after ':' (e.g. all:transformers)")
    try:
        service = ArxivSearchService()
        result = await service.search(
            query=q_clean,
            start=start,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
        )
        return {
            "total_results": result.total_results,
            "start": result.start,
            "max_results": result.max_results,
            "items": result.items,
        }
    except httpx.HTTPStatusError as exc:
        # Map arXiv errors to user-facing validation errors when possible.
        status = getattr(exc.response, "status_code", None)
        if status and 400 <= int(status) < 500:
            logger.warning(f"ArXiv search rejected query '{q_clean}': {exc}")
            raise HTTPException(status_code=422, detail="arXiv rejected the query (check syntax)")
        logger.warning(f"ArXiv search failed: {exc}")
        raise HTTPException(status_code=502, detail="Failed to query arXiv")
    except httpx.HTTPError as exc:
        logger.warning(f"ArXiv search failed: {exc}")
        raise HTTPException(status_code=502, detail="Failed to query arXiv")
    except Exception as exc:
        logger.error(f"ArXiv search error: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))


class ArxivTranslateQueryRequest(BaseModel):
    text: str = Field(..., min_length=2, max_length=500, description="Natural language query to translate")
    categories: Optional[list[str]] = Field(default=None, description="Optional arXiv categories like cs.CL, cs.CV")


class ArxivTranslateQueryResponse(BaseModel):
    query: str


@router.post("/arxiv/translate-query", response_model=ArxivTranslateQueryResponse)
async def translate_arxiv_query(
    payload: ArxivTranslateQueryRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Translate a natural language request into arXiv API query syntax.

    Returns an arXiv query string suitable for /arxiv/search `q=...`.
    """
    _ = current_user  # authenticated access required
    text = (payload.text or "").strip()
    categories = [c.strip() for c in (payload.categories or []) if isinstance(c, str) and c.strip()]

    cat_hint = ""
    if categories:
        cat_hint = " If categories are provided, constrain with AND (cat:... OR cat:...)."

    prompt = (
        "Convert the user's request into a valid arXiv API query string for the `search_query` parameter.\n"
        "Rules:\n"
        "- Output ONLY the query string (no JSON, no quotes).\n"
        "- Use arXiv fields like: all:, ti:, abs:, au:, cat:, id:.\n"
        "- Use AND/OR, parentheses, and quotes for phrases when needed.\n"
        "- Do NOT invent unsupported filters (no date ranges); rely on sort_by/submittedDate instead.\n"
        "- The query must not end with ':'.\n"
        "- Keep it concise.\n"
        f"{cat_hint}\n\n"
        "Examples:\n"
        "User: \"papers about diffusion models for image segmentation\"\n"
        "Query: all:\"diffusion model\" AND (all:segmentation OR all:\"image segmentation\")\n"
        "User: \"transformer pruning in vision\"\n"
        "Query: (ti:transformer OR abs:transformer) AND (all:pruning OR all:sparsity) AND (all:vision OR cat:cs.CV)\n\n"
        f"User request: {text}\n"
        f"Categories: {categories}\n"
        "Query:"
    )

    try:
        llm = LLMService()
        raw = await llm.generate_response(
            query=prompt,
            context=None,
            temperature=0.1,
            max_tokens=120,
            task_type="query_expansion",
        )
        query = (raw or "").strip().strip('"').strip()
        # Basic sanitation
        if query.endswith(":") or len(query) < 2:
            raise ValueError("LLM returned an invalid arXiv query")
        return ArxivTranslateQueryResponse(query=query)
    except Exception as exc:
        logger.warning(f"Failed to translate arXiv query: {exc}")
        raise HTTPException(status_code=500, detail="Failed to translate query")


@router.get("/arxiv/imports")
async def list_arxiv_imports(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List recent arXiv ingestion sources created by the current user.

    Ownership is determined via `config.requested_by` / `config.requested_by_user_id`.
    """
    try:
        result = await db.execute(
            select(DocumentSource)
            .where(DocumentSource.source_type == "arxiv")
            .order_by(desc(DocumentSource.created_at))
            .limit(500)
        )
        sources = list(result.scalars().all())

        def is_owned_by_user(source: DocumentSource) -> bool:
            cfg = source.config or {}
            if not isinstance(cfg, dict):
                return False
            requested_by = cfg.get("requested_by") or cfg.get("requestedBy")
            requested_by_user_id = cfg.get("requested_by_user_id") or cfg.get("requestedByUserId")
            return requested_by in {current_user.username, str(current_user.id)} or requested_by_user_id == str(current_user.id)

        owned = [s for s in sources if is_owned_by_user(s)]
        total = len(owned)
        page_items = owned[offset:offset + limit]

        items = []
        for s in page_items:
            cfg = s.config if isinstance(s.config, dict) else {}
            display = cfg.get("display") if isinstance(cfg, dict) else None
            doc_count = 0
            review_doc_id = None
            review_doc_title = None
            try:
                count_result = await db.execute(select(sa.func.count()).select_from(Document).where(Document.source_id == s.id))
                doc_count = int(count_result.scalar() or 0)
            except Exception:
                pass
            try:
                review_result = await db.execute(
                    select(Document.id, Document.title)
                    .where(
                        Document.source_id == s.id,
                        Document.source_identifier.like("literature_review:%"),
                    )
                    .order_by(desc(Document.created_at))
                    .limit(1)
                )
                row = review_result.first()
                if row:
                    review_doc_id = str(row[0])
                    review_doc_title = row[1]
            except Exception:
                pass
            items.append(
                {
                    "id": str(s.id),
                    "name": s.name,
                    "source_type": s.source_type,
                    "is_active": bool(s.is_active),
                    "is_syncing": bool(getattr(s, "is_syncing", False)),
                    "last_error": getattr(s, "last_error", None),
                    "last_sync": s.last_sync.isoformat() if s.last_sync else None,
                    "created_at": s.created_at.isoformat() if s.created_at else None,
                    "updated_at": s.updated_at.isoformat() if s.updated_at else None,
                    "display": display,
                    "document_count": doc_count,
                    "review_document_id": review_doc_id,
                    "review_document_title": review_doc_title,
                }
            )

        return {"items": items, "total": total, "limit": limit, "offset": offset}
    except Exception as exc:
        logger.error(f"Failed to list arXiv imports: {exc}")
        raise HTTPException(status_code=500, detail="Failed to list arXiv imports")


def _is_source_owned_by_user(source: DocumentSource, user: User) -> bool:
    cfg = source.config or {}
    if not isinstance(cfg, dict):
        return False
    requested_by = cfg.get("requested_by") or cfg.get("requestedBy")
    requested_by_user_id = cfg.get("requested_by_user_id") or cfg.get("requestedByUserId")
    return requested_by in {user.username, str(user.id)} or requested_by_user_id == str(user.id)


class ImportSummarizeRequest(BaseModel):
    force: bool = False
    limit: int = Field(default=200, ge=1, le=2000)
    only_missing: bool = True


@router.post("/arxiv/imports/{source_id}/summarize")
async def summarize_arxiv_import(
    source_id: str,
    payload: ImportSummarizeRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Queue summarization for documents in an arXiv import source."""
    try:
        src_uuid = UUID(source_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid source id")

    src = await db.get(DocumentSource, src_uuid)
    if not src or src.source_type != "arxiv" or not _is_source_owned_by_user(src, current_user):
        raise HTTPException(status_code=404, detail="Import not found")

    from app.tasks.summarization_tasks import summarize_document as summarize_task

    stmt = (
        select(Document.id, Document.summary)
        .where(Document.source_id == src.id)
        .order_by(desc(Document.created_at))
        .limit(payload.limit)
    )
    rows = (await db.execute(stmt)).all()
    queued = 0
    for doc_id, summary in rows:
        if payload.only_missing and summary and not payload.force:
            continue
        summarize_task.delay(str(doc_id), payload.force, user_id=str(current_user.id))
        queued += 1

    return {"message": "queued", "source_id": source_id, "queued": queued, "considered": len(rows)}


class ImportReviewRequest(BaseModel):
    topic: str | None = None


@router.post("/arxiv/imports/{source_id}/generate-review")
async def generate_review_for_arxiv_import(
    source_id: str,
    payload: ImportReviewRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        src_uuid = UUID(source_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid source id")

    src = await db.get(DocumentSource, src_uuid)
    if not src or src.source_type != "arxiv" or not _is_source_owned_by_user(src, current_user):
        raise HTTPException(status_code=404, detail="Import not found")

    if payload.topic and isinstance(src.config, dict):
        cfg = dict(src.config)
        cfg["topic"] = payload.topic.strip()
        src.config = cfg
        await db.commit()

    from app.tasks.research_tasks import generate_literature_review
    task = generate_literature_review.delay(source_id, user_id=str(current_user.id))
    return {"message": "queued", "source_id": source_id, "task_id": task.id}


class ImportSlidesRequest(BaseModel):
    title: str | None = None
    topic: str | None = None
    slide_count: int = Field(default=10, ge=3, le=40)
    style: str = "professional"
    include_diagrams: bool = True
    prefer_review_document: bool = True


@router.post("/arxiv/imports/{source_id}/generate-slides")
async def generate_slides_for_arxiv_import(
    source_id: str,
    payload: ImportSlidesRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        src_uuid = UUID(source_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid source id")

    src = await db.get(DocumentSource, src_uuid)
    if not src or src.source_type != "arxiv" or not _is_source_owned_by_user(src, current_user):
        raise HTTPException(status_code=404, detail="Import not found")

    title = payload.title or f"Slides: {src.name}"
    topic = payload.topic
    if not topic and isinstance(src.config, dict):
        topic = src.config.get("topic")
    topic = topic or src.name

    review_doc_id = None
    if payload.prefer_review_document:
        review_result = await db.execute(
            select(Document.id)
            .where(
                Document.source_id == src.id,
                Document.source_identifier.like("literature_review:%"),
            )
            .order_by(desc(Document.created_at))
            .limit(1)
        )
        review_doc_id = review_result.scalar_one_or_none()

    if review_doc_id:
        doc_ids = [str(review_doc_id)]
    else:
        docs_result = await db.execute(
            select(Document.id)
            .where(Document.source_id == src.id)
            .order_by(desc(Document.created_at))
            .limit(8)
        )
        doc_ids = [str(r[0]) for r in docs_result.all()]
        if not doc_ids:
            raise HTTPException(status_code=400, detail="No documents found for import")

    from app.models.presentation import PresentationJob
    from app.tasks.presentation_tasks import generate_presentation_task

    job = PresentationJob(
        user_id=current_user.id,
        title=title,
        topic=topic,
        source_document_ids=doc_ids,
        slide_count=payload.slide_count,
        style=payload.style,
        include_diagrams=1 if payload.include_diagrams else 0,
        status="pending",
        progress=0,
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    generate_presentation_task.delay(str(job.id), str(current_user.id))
    return {"message": "queued", "source_id": source_id, "presentation_job_id": str(job.id)}


class ImportEnrichRequest(BaseModel):
    force: bool = False
    limit: int = Field(default=500, ge=1, le=5000)


@router.post("/arxiv/imports/{source_id}/enrich-metadata")
async def enrich_metadata_for_arxiv_import(
    source_id: str,
    payload: ImportEnrichRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        src_uuid = UUID(source_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid source id")

    src = await db.get(DocumentSource, src_uuid)
    if not src or src.source_type != "arxiv" or not _is_source_owned_by_user(src, current_user):
        raise HTTPException(status_code=404, detail="Import not found")

    from app.tasks.paper_enrichment_tasks import enrich_arxiv_source
    task = enrich_arxiv_source.delay(source_id, payload.force, payload.limit)
    return {"message": "queued", "source_id": source_id, "task_id": task.id}
