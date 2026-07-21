"""
Structured paper extraction endpoints.
"""

from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.models.document import Document
from app.models.research_paper import PaperExtractionJob, ResearchPaper
from app.models.user import User
from app.schemas.research_paper import (
    PaperExtractionJobResponse,
    PaperExtractionRequest,
    ResearchPaperListResponse,
    ResearchPaperResponse,
    SaveResearchPaperAsNoteRequest,
)
from app.services.auth_service import get_current_user
from app.services.paper_extraction_service import (
    PaperExtractionConflictError,
    UnsupportedPaperSourceError,
    paper_extraction_service,
)
from app.tasks.paper_extraction_tasks import extract_paper_job

router = APIRouter()


def _job_to_response(job: PaperExtractionJob) -> PaperExtractionJobResponse:
    return PaperExtractionJobResponse.model_validate(job)


def _paper_to_response(paper: ResearchPaper, latest_job: Optional[PaperExtractionJob] = None) -> ResearchPaperResponse:
    return ResearchPaperResponse(
        id=paper.id,
        user_id=paper.user_id,
        document_id=paper.document_id,
        source_id=paper.source_id,
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        authors=paper.authors if isinstance(paper.authors, list) else None,
        abstract=paper.abstract,
        published_at=paper.published_at,
        categories=paper.categories if isinstance(paper.categories, list) else None,
        paper_url=paper.paper_url,
        pdf_url=paper.pdf_url,
        extraction_status=paper.extraction_status,
        extracted_at=paper.extracted_at,
        extractor_version=paper.extractor_version,
        summary=paper.summary,
        mechanisms=paper.mechanisms if isinstance(paper.mechanisms, list) else None,
        assumptions=paper.assumptions if isinstance(paper.assumptions, list) else None,
        benchmarks=paper.benchmarks if isinstance(paper.benchmarks, list) else None,
        metrics=paper.metrics if isinstance(paper.metrics, list) else None,
        limitations=paper.limitations if isinstance(paper.limitations, list) else None,
        raw_extraction_payload=paper.raw_extraction_payload if isinstance(paper.raw_extraction_payload, dict) else None,
        claims=list(sorted(paper.claims or [], key=lambda item: (item.rank is None, item.rank if item.rank is not None else 999999))),
        latest_job=_job_to_response(latest_job) if latest_job else None,
        created_at=paper.created_at,
        updated_at=paper.updated_at,
    )


async def _get_latest_jobs_for_papers(db: AsyncSession, paper_ids: List[UUID]) -> dict[UUID, PaperExtractionJob]:
    if not paper_ids:
        return {}
    stmt = (
        select(PaperExtractionJob)
        .where(PaperExtractionJob.paper_id.in_(paper_ids))
        .order_by(PaperExtractionJob.paper_id, desc(PaperExtractionJob.created_at))
    )
    jobs = (await db.execute(stmt)).scalars().all()
    latest: dict[UUID, PaperExtractionJob] = {}
    for job in jobs:
        if job.paper_id and job.paper_id not in latest:
            latest[job.paper_id] = job
    return latest


@router.get("", response_model=ResearchPaperListResponse)
async def list_research_papers(
    source_id: Optional[UUID] = Query(None),
    arxiv_id: Optional[str] = Query(None),
    arxiv_ids: Optional[str] = Query(None, description="Comma-separated arXiv IDs"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    base_stmt = select(ResearchPaper).where(ResearchPaper.user_id == current_user.id)
    if source_id:
        base_stmt = base_stmt.where(ResearchPaper.source_id == source_id)
    if arxiv_id:
        base_stmt = base_stmt.where(ResearchPaper.arxiv_id == arxiv_id.strip())
    if arxiv_ids:
        values = [item.strip() for item in arxiv_ids.split(",") if item.strip()]
        if values:
            base_stmt = base_stmt.where(ResearchPaper.arxiv_id.in_(values))
    count_stmt = select(func.count()).select_from(base_stmt.subquery())
    total = int((await db.execute(count_stmt)).scalar() or 0)
    stmt = base_stmt.options(selectinload(ResearchPaper.claims)).order_by(desc(ResearchPaper.updated_at))
    papers = (await db.execute(stmt.offset(offset).limit(limit))).scalars().unique().all()
    latest_jobs = await _get_latest_jobs_for_papers(db, [paper.id for paper in papers])
    return ResearchPaperListResponse(
        items=[_paper_to_response(paper, latest_jobs.get(paper.id)) for paper in papers],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/jobs", response_model=list[PaperExtractionJobResponse])
async def list_paper_extraction_jobs(
    source_id: Optional[UUID] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(PaperExtractionJob).where(PaperExtractionJob.user_id == current_user.id).order_by(desc(PaperExtractionJob.created_at))
    if source_id:
        stmt = stmt.where(PaperExtractionJob.source_id == source_id)
    jobs = (await db.execute(stmt.limit(100))).scalars().all()
    return [_job_to_response(job) for job in jobs]


@router.post("/extract", response_model=list[PaperExtractionJobResponse], status_code=status.HTTP_202_ACCEPTED)
async def extract_research_papers(
    payload: PaperExtractionRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    document_ids = list(payload.document_ids or [])
    if payload.source_id:
        source_docs = (
            await db.execute(
                select(Document)
                .where(Document.source_id == payload.source_id)
                .order_by(desc(Document.created_at))
                .limit(payload.limit)
            )
        ).scalars().all()
        document_ids.extend([doc.id for doc in source_docs])
    seen = set()
    ordered_ids = []
    for document_id in document_ids:
        if document_id in seen:
            continue
        seen.add(document_id)
        ordered_ids.append(document_id)
    if not ordered_ids:
        raise HTTPException(status_code=400, detail="Provide at least one document_id or a source_id")

    queued: list[PaperExtractionJobResponse] = []
    for document_id in ordered_ids:
        document = await db.get(Document, document_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
        try:
            job = await paper_extraction_service.queue_document_extraction_job(
                db,
                document=document,
                user_id=current_user.id,
                force=payload.force,
                dispatch_task=lambda job_id: extract_paper_job.delay(job_id),
            )
        except UnsupportedPaperSourceError:
            raise HTTPException(status_code=422, detail=f"Document {document.id} is not an arXiv document")
        except PaperExtractionConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        queued.append(_job_to_response(job))
    return queued


@router.get("/{paper_id}", response_model=ResearchPaperResponse)
async def get_research_paper(
    paper_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(ResearchPaper).where(ResearchPaper.id == paper_id).options(selectinload(ResearchPaper.claims))
    paper = (await db.execute(stmt)).scalar_one_or_none()
    if not paper or paper.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Research paper not found")
    latest_jobs = await _get_latest_jobs_for_papers(db, [paper.id])
    return _paper_to_response(paper, latest_jobs.get(paper.id))


@router.post("/{paper_id}/reextract", response_model=PaperExtractionJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def reextract_research_paper(
    paper_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(ResearchPaper).where(ResearchPaper.id == paper_id).options(selectinload(ResearchPaper.claims))
    paper = (await db.execute(stmt)).scalar_one_or_none()
    if not paper or paper.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Research paper not found")
    document = await db.get(Document, paper.document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        job = await paper_extraction_service.queue_document_extraction_job(
            db,
            document=document,
            user_id=current_user.id,
            force=True,
            dispatch_task=lambda job_id: extract_paper_job.delay(job_id),
        )
    except UnsupportedPaperSourceError:
        raise HTTPException(status_code=422, detail=f"Document {document.id} is not an arXiv document")
    return _job_to_response(job)


@router.post("/{paper_id}/save-as-note", status_code=status.HTTP_201_CREATED)
async def save_research_paper_as_note(
    paper_id: UUID,
    payload: SaveResearchPaperAsNoteRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(ResearchPaper).where(ResearchPaper.id == paper_id).options(selectinload(ResearchPaper.claims))
    paper = (await db.execute(stmt)).scalar_one_or_none()
    if not paper or paper.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Research paper not found")
    note = await paper_extraction_service.create_research_note(
        db,
        paper=paper,
        user_id=current_user.id,
        title=payload.title,
        tags=payload.tags,
    )
    return {"id": str(note.id), "title": note.title}
