"""
Template filling API endpoints.
"""

import json
from typing import List, Optional
from uuid import UUID, uuid4
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from loguru import logger
import redis

from app.core.database import get_db
from app.core.config import settings
from app.models.user import User
from app.models.template import TemplateJob
from app.services.auth_service import get_current_user
from app.services.storage_service import storage_service
from app.schemas.template import (
    TemplateJobCreate,
    TemplateJobResponse,
    TemplateJobListResponse,
)
from app.tasks.template_tasks import fill_template


router = APIRouter()


ALLOWED_TEMPLATE_EXTENSIONS = {'.docx', '.doc'}
MAX_TEMPLATE_SIZE = 50 * 1024 * 1024  # 50MB


def _validate_template_file(filename: str, file_size: int) -> None:
    """Validate template file."""
    from pathlib import Path
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_TEMPLATE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_TEMPLATE_EXTENSIONS)}"
        )
    if file_size > MAX_TEMPLATE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_TEMPLATE_SIZE // (1024*1024)}MB"
        )


@router.post("/fill", response_model=TemplateJobResponse)
async def create_template_fill_job(
    template: UploadFile = File(..., description="Template document (docx)"),
    source_document_ids: str = Form(..., description="Comma-separated document UUIDs"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a template filling job.

    Upload a template document and specify source documents to extract
    information from. The system will analyze the template, extract
    relevant information from sources, and generate a filled document.

    Args:
        template: Template file (docx format)
        source_document_ids: Comma-separated list of document UUIDs to use as sources
        current_user: Current authenticated user
        db: Database session

    Returns:
        Created template job with status
    """
    try:
        # Parse source document IDs
        try:
            doc_ids = [UUID(id.strip()) for id in source_document_ids.split(',') if id.strip()]
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid document ID format: {e}")

        if not doc_ids:
            raise HTTPException(status_code=400, detail="At least one source document ID is required")

        # Validate template file
        content = await template.read()
        _validate_template_file(template.filename, len(content))

        # Generate job ID
        job_id = uuid4()

        # Upload template to MinIO
        await storage_service.initialize()
        template_path = await storage_service.upload_file(
            job_id,
            template.filename,
            content,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        # Create job record
        job = TemplateJob(
            id=job_id,
            user_id=current_user.id,
            template_file_path=template_path,
            template_filename=template.filename,
            source_document_ids=[str(doc_id) for doc_id in doc_ids],
            status="pending",
            progress=0
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)

        # Trigger background task
        fill_template.delay(str(job_id))

        logger.info(f"Created template fill job {job_id} for user {current_user.id}")

        return TemplateJobResponse(
            id=job.id,
            template_filename=job.template_filename,
            sections=job.sections,
            source_document_ids=job.source_document_ids,
            status=job.status,
            progress=job.progress,
            current_section=job.current_section,
            filled_filename=job.filled_filename,
            error_message=job.error_message,
            created_at=job.created_at,
            updated_at=job.updated_at,
            completed_at=job.completed_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create template fill job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create job: {str(e)}")


@router.get("/{job_id}", response_model=TemplateJobResponse)
async def get_template_job(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get template job status and details.

    Args:
        job_id: Template job UUID
        current_user: Current authenticated user
        db: Database session

    Returns:
        Template job details
    """
    job = await db.get(TemplateJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Template job not found")

    if job.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")

    # Generate download URL if completed
    download_url = None
    if job.status == "completed" and job.filled_file_path:
        try:
            download_url = await storage_service.get_presigned_download_url(
                job.filled_file_path
            )
        except Exception as e:
            logger.warning(f"Failed to generate download URL: {e}")

    return TemplateJobResponse(
        id=job.id,
        template_filename=job.template_filename,
        sections=job.sections,
        source_document_ids=job.source_document_ids,
        status=job.status,
        progress=job.progress,
        current_section=job.current_section,
        filled_filename=job.filled_filename,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        completed_at=job.completed_at,
        download_url=download_url
    )


@router.get("/{job_id}/download")
async def download_filled_template(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Download the filled template document.

    Args:
        job_id: Template job UUID
        current_user: Current authenticated user
        db: Database session

    Returns:
        Streaming response with the filled document
    """
    job = await db.get(TemplateJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Template job not found")

    if job.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")

    if job.status != "completed" or not job.filled_file_path:
        raise HTTPException(status_code=400, detail="Filled document not available")

    try:
        # Get file content directly from storage
        await storage_service.initialize()
        content = await storage_service.get_file_content(job.filled_file_path)

        # Determine filename
        filename = job.filled_filename or f"filled_{job.template_filename}"

        return Response(
            content=content,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(content)),
            }
        )

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Filled document file not found in storage")
    except Exception as e:
        logger.error(f"Failed to download filled template: {e}")
        raise HTTPException(status_code=500, detail="Failed to download file")


@router.get("/", response_model=TemplateJobListResponse)
async def list_template_jobs(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List user's template filling jobs.

    Args:
        skip: Number of records to skip
        limit: Maximum number of records to return
        status: Optional status filter
        current_user: Current authenticated user
        db: Database session

    Returns:
        List of template jobs with total count
    """
    # Build query
    query = select(TemplateJob).where(TemplateJob.user_id == current_user.id)

    if status:
        query = query.where(TemplateJob.status == status)

    # Get total count
    count_query = select(func.count()).select_from(TemplateJob).where(
        TemplateJob.user_id == current_user.id
    )
    if status:
        count_query = count_query.where(TemplateJob.status == status)

    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Get jobs
    query = query.order_by(desc(TemplateJob.created_at)).offset(skip).limit(limit)
    result = await db.execute(query)
    jobs = result.scalars().all()

    return TemplateJobListResponse(
        jobs=[
            TemplateJobResponse(
                id=job.id,
                template_filename=job.template_filename,
                sections=job.sections,
                source_document_ids=job.source_document_ids,
                status=job.status,
                progress=job.progress,
                current_section=job.current_section,
                filled_filename=job.filled_filename,
                error_message=job.error_message,
                created_at=job.created_at,
                updated_at=job.updated_at,
                completed_at=job.completed_at
            )
            for job in jobs
        ],
        total=total
    )


@router.delete("/{job_id}")
async def delete_template_job(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a template job and its associated files.

    Args:
        job_id: Template job UUID
        current_user: Current authenticated user
        db: Database session
    """
    job = await db.get(TemplateJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Template job not found")

    if job.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        # Delete files from MinIO
        await storage_service.initialize()

        if job.template_file_path:
            try:
                await storage_service.delete_file(job.template_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete template file: {e}")

        if job.filled_file_path:
            try:
                await storage_service.delete_file(job.filled_file_path)
            except Exception as e:
                logger.warning(f"Failed to delete filled file: {e}")

        # Delete job record
        await db.delete(job)
        await db.commit()

        logger.info(f"Deleted template job {job_id}")
        return {"message": "Template job deleted successfully"}

    except Exception as e:
        logger.error(f"Failed to delete template job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete job: {str(e)}")


@router.websocket("/{job_id}/progress")
async def template_job_progress(
    websocket: WebSocket,
    job_id: str,
    token: str = Query(...)
):
    """
    WebSocket endpoint for real-time progress updates.

    Connect to receive progress updates for a template filling job.

    Args:
        websocket: WebSocket connection
        job_id: Template job UUID
        token: Authentication token
    """
    await websocket.accept()

    try:
        # Verify token using websocket auth utility
        from app.utils.websocket_auth import authenticate_websocket
        user = await authenticate_websocket(websocket, token)
        if not user:
            await websocket.close(code=4001, reason="Invalid token")
            return

        # Subscribe to Redis channel
        redis_client = redis.from_url(settings.REDIS_URL)
        pubsub = redis_client.pubsub()
        channel = f"template_progress:{job_id}"
        pubsub.subscribe(channel)

        logger.info(f"WebSocket connected for template job {job_id}")

        # Listen for messages
        try:
            while True:
                message = pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    data = message['data']
                    if isinstance(data, bytes):
                        data = data.decode('utf-8')
                    await websocket.send_text(data)

                    # Check if job is complete or failed
                    try:
                        msg_data = json.loads(data)
                        if msg_data.get('type') in ('complete', 'error'):
                            break
                    except json.JSONDecodeError:
                        pass

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for template job {job_id}")
        finally:
            pubsub.unsubscribe(channel)
            pubsub.close()
            redis_client.close()

    except Exception as e:
        logger.error(f"WebSocket error for template job {job_id}: {e}")
        try:
            await websocket.close(code=4000, reason=str(e))
        except Exception:
            pass
