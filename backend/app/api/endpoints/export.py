"""
API endpoints for DOCX/PDF export functionality.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from loguru import logger

from app.core.database import get_db
from app.api.endpoints.users import get_current_user
from app.models.user import User
from app.models.export_job import ExportJob
from app.models.chat import ChatSession
from app.models.document import Document
from app.schemas.export import (
    ExportChatRequest,
    ExportDocumentSummaryRequest,
    ExportCustomRequest,
    ExportJobResponse,
    ExportJobCreatedResponse,
    ExportJobStatusResponse,
    ExportJobListResponse,
)
from app.services.export_service import export_service


router = APIRouter()


def _job_to_response(job: ExportJob) -> ExportJobResponse:
    """Convert ExportJob model to response schema."""
    return ExportJobResponse(
        id=job.id,
        user_id=job.user_id,
        export_type=job.export_type,
        output_format=job.output_format,
        source_type=job.source_type,
        source_id=job.source_id,
        title=job.title,
        style=job.style,
        custom_theme=job.custom_theme,
        status=job.status,
        progress=job.progress,
        current_stage=job.current_stage,
        file_path=job.file_path,
        file_size=job.file_size,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.post("/chat/{session_id}", response_model=ExportJobCreatedResponse, status_code=status.HTTP_201_CREATED)
async def export_chat_session(
    session_id: UUID,
    request: ExportChatRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Export a chat session to DOCX or PDF.

    The export will be processed asynchronously via Celery.
    Use GET /export/{job_id} to check status.
    """
    # Verify chat session exists and belongs to user
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == current_user.id
        )
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found"
        )

    # Determine title
    title = request.title or session.topic or f"Chat Export - {session.created_at.strftime('%Y-%m-%d')}"

    # Create export job
    job = await export_service.create_export_job(
        db=db,
        user_id=current_user.id,
        export_type="chat",
        output_format=request.format,
        source_type="chat_session",
        source_id=session_id,
        title=title,
        style=request.style,
        custom_theme=request.custom_theme.model_dump() if request.custom_theme else None
    )

    # Dispatch Celery task
    from app.tasks.export_tasks import process_export_task
    process_export_task.delay(str(job.id))

    logger.info(f"Created chat export job {job.id} for session {session_id}")

    return ExportJobCreatedResponse(
        job_id=job.id,
        status="pending",
        message="Export job created. Use GET /export/{job_id} to check status."
    )


@router.post("/document/{document_id}/summary", response_model=ExportJobCreatedResponse, status_code=status.HTTP_201_CREATED)
async def export_document_summary(
    document_id: UUID,
    request: ExportDocumentSummaryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Export a document summary to DOCX or PDF.

    The export will be processed asynchronously via Celery.
    Use GET /export/{job_id} to check status.
    """
    # Verify document exists
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    # Determine title
    title = request.title or document.title or f"Document Summary - {document.created_at.strftime('%Y-%m-%d')}"

    # Create export job
    job = await export_service.create_export_job(
        db=db,
        user_id=current_user.id,
        export_type="document_summary",
        output_format=request.format,
        source_type="document",
        source_id=document_id,
        title=title,
        style=request.style,
        custom_theme=request.custom_theme.model_dump() if request.custom_theme else None
    )

    # Dispatch Celery task
    from app.tasks.export_tasks import process_export_task
    process_export_task.delay(str(job.id))

    logger.info(f"Created document export job {job.id} for document {document_id}")

    return ExportJobCreatedResponse(
        job_id=job.id,
        status="pending",
        message="Export job created. Use GET /export/{job_id} to check status."
    )


@router.post("/custom", response_model=ExportJobCreatedResponse, status_code=status.HTTP_201_CREATED)
async def export_custom_content(
    request: ExportCustomRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Export custom/LLM-generated content to DOCX or PDF.

    Accepts markdown or HTML content and converts it to the specified format.
    The export will be processed asynchronously via Celery.
    Use GET /export/{job_id} to check status.
    """
    # Create export job
    job = await export_service.create_export_job(
        db=db,
        user_id=current_user.id,
        export_type="custom",
        output_format=request.format,
        source_type="llm_content",
        title=request.title,
        content=request.content,
        content_format=request.content_format,
        style=request.style,
        custom_theme=request.custom_theme.model_dump() if request.custom_theme else None
    )

    # Dispatch Celery task
    from app.tasks.export_tasks import process_export_task
    process_export_task.delay(str(job.id))

    logger.info(f"Created custom export job {job.id} for user {current_user.id}")

    return ExportJobCreatedResponse(
        job_id=job.id,
        status="pending",
        message="Export job created. Use GET /export/{job_id} to check status."
    )


@router.get("/{job_id}", response_model=ExportJobResponse)
async def get_export_job(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get export job status and details.
    """
    job = await export_service.get_export_job(db, job_id, current_user.id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Export job not found"
        )

    return _job_to_response(job)


@router.get("/{job_id}/status", response_model=ExportJobStatusResponse)
async def get_export_status(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get minimal export job status (for polling).
    """
    job = await export_service.get_export_job(db, job_id, current_user.id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Export job not found"
        )

    return ExportJobStatusResponse(
        id=job.id,
        status=job.status,
        progress=job.progress,
        current_stage=job.current_stage,
        error=job.error,
        file_path=job.file_path,
        file_size=job.file_size,
        completed_at=job.completed_at
    )


@router.get("/{job_id}/download")
async def download_export(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Download the exported file.

    Only available for completed export jobs.
    """
    job = await export_service.get_export_job(db, job_id, current_user.id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Export job not found"
        )

    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Export job is not completed (status: {job.status})"
        )

    if not job.file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Export file not found"
        )

    # Get file content
    content = await export_service.get_download_content(db, job_id, current_user.id)

    if not content:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Export file not found in storage"
        )

    # Determine content type and filename
    if job.output_format == "docx":
        content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        extension = "docx"
    else:
        content_type = "application/pdf"
        extension = "pdf"

    # Sanitize filename
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in job.title)
    filename = f"{safe_title}.{extension}"

    return Response(
        content=content,
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(content))
        }
    )


@router.get("", response_model=ExportJobListResponse)
async def list_export_jobs(
    page: int = 1,
    page_size: int = 20,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List export jobs for the current user.
    """
    # Get total count
    count_result = await db.execute(
        select(func.count()).select_from(ExportJob).where(ExportJob.user_id == current_user.id)
    )
    total = count_result.scalar()

    # Get jobs with pagination
    offset = (page - 1) * page_size
    jobs = await export_service.get_user_export_jobs(
        db=db,
        user_id=current_user.id,
        limit=page_size,
        offset=offset
    )

    total_pages = (total + page_size - 1) // page_size if total > 0 else 0

    return ExportJobListResponse(
        items=[_job_to_response(job) for job in jobs],
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages
    )


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_export_job(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete an export job and its associated file.
    """
    job = await export_service.get_export_job(db, job_id, current_user.id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Export job not found"
        )

    # Delete file from storage if exists
    if job.file_path:
        try:
            from app.services.storage_service import StorageService
            storage = StorageService()
            await storage.initialize()
            await storage.delete_file(job.file_path)
        except Exception as e:
            logger.warning(f"Failed to delete export file {job.file_path}: {e}")

    # Delete job from database
    await db.delete(job)
    await db.commit()

    logger.info(f"Deleted export job {job_id}")
