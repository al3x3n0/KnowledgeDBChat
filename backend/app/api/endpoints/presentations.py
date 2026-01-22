"""
API endpoints for AI-powered presentation generation.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import RedirectResponse
import tempfile
import os
import re
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from loguru import logger

from app.core.database import get_db
from app.api.endpoints.users import get_current_user
from app.models.user import User
from app.models.presentation import PresentationJob, PresentationTemplate
from app.schemas.presentation import (
    PresentationJobCreate,
    PresentationJobResponse,
    PresentationJobUpdate,
    PresentationTemplateCreate,
    PresentationTemplateUpdate,
    PresentationTemplateResponse,
)
from app.services.storage_service import StorageService


router = APIRouter()


@router.post("", response_model=PresentationJobResponse, status_code=status.HTTP_201_CREATED)
async def create_presentation(
    request: PresentationJobCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new presentation generation job.

    The presentation will be generated asynchronously via Celery.
    Use GET /presentations/{job_id} to check status or WebSocket for real-time updates.
    """
    # Validate template if provided
    if request.template_id:
        result = await db.execute(
            select(PresentationTemplate).where(
                PresentationTemplate.id == request.template_id,
                PresentationTemplate.is_active == True
            )
        )
        template = result.scalar_one_or_none()
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        # Check access: user owns it, it's public, or it's a system template
        if not (template.is_system or template.is_public or template.user_id == current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this template"
            )

    # Create job in database
    job = PresentationJob(
        user_id=current_user.id,
        title=request.title,
        topic=request.topic,
        source_document_ids=[str(doc_id) for doc_id in request.source_document_ids] if request.source_document_ids else [],
        slide_count=request.slide_count,
        style=request.style,
        include_diagrams=1 if request.include_diagrams else 0,
        template_id=request.template_id,
        custom_theme=request.custom_theme.model_dump() if request.custom_theme else None,
        status="pending",
        progress=0,
    )

    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Dispatch Celery task
    from app.tasks.presentation_tasks import generate_presentation_task
    generate_presentation_task.delay(str(job.id), str(current_user.id))

    logger.info(f"Created presentation job {job.id} for user {current_user.id}")

    return _job_to_response(job)


@router.get("", response_model=List[PresentationJobResponse])
async def list_presentations(
    limit: int = 20,
    offset: int = 0,
    status_filter: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List user's presentation generation jobs.

    Results are ordered by creation date (newest first).
    """
    query = select(PresentationJob).where(
        PresentationJob.user_id == current_user.id
    )

    if status_filter:
        query = query.where(PresentationJob.status == status_filter)

    query = query.order_by(PresentationJob.created_at.desc())
    query = query.offset(offset).limit(limit)

    result = await db.execute(query)
    jobs = result.scalars().all()

    return [_job_to_response(job) for job in jobs]


@router.get("/{job_id}", response_model=PresentationJobResponse)
async def get_presentation_job(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get presentation job status and details.
    """
    job = await _get_user_job(job_id, current_user.id, db)
    return _job_to_response(job)


@router.get("/{job_id}/download")
async def download_presentation(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Download the generated presentation.

    Redirects to a presigned MinIO URL for direct download.
    """
    job = await _get_user_job(job_id, current_user.id, db)

    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Presentation is not ready. Current status: {job.status}"
        )

    if not job.file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Presentation file not found"
        )

    # Generate presigned URL for download
    storage = StorageService()
    try:
        download_url = await storage.get_presigned_url(
            file_path=job.file_path,
            expires_in=3600  # 1 hour
        )
        return RedirectResponse(url=download_url, status_code=status.HTTP_302_FOUND)
    except Exception as e:
        logger.error(f"Failed to generate download URL for job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate download URL"
        )


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_presentation(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a presentation job and its associated file.
    """
    job = await _get_user_job(job_id, current_user.id, db)

    # Delete file from MinIO if exists
    if job.file_path:
        storage = StorageService()
        try:
            await storage.delete_file(job.file_path)
        except Exception as e:
            logger.warning(f"Failed to delete presentation file {job.file_path}: {e}")

    # Delete job from database
    await db.delete(job)
    await db.commit()

    logger.info(f"Deleted presentation job {job_id}")


@router.post("/{job_id}/cancel", response_model=PresentationJobResponse)
async def cancel_presentation(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Cancel a pending or in-progress presentation generation job.
    """
    job = await _get_user_job(job_id, current_user.id, db)

    if job.status in ("completed", "failed", "cancelled"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job with status: {job.status}"
        )

    job.status = "cancelled"
    job.error = "Cancelled by user"
    await db.commit()
    await db.refresh(job)

    logger.info(f"Cancelled presentation job {job_id}")

    return _job_to_response(job)


@router.websocket("/{job_id}/progress")
async def presentation_progress(
    websocket: WebSocket,
    job_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """
    WebSocket endpoint for real-time progress updates.

    Sends JSON messages with format:
    {
        "type": "progress",
        "progress": 50,
        "stage": "generating_slides",
        "status": "generating"
    }

    Connection closes automatically when job completes or fails.
    """
    await websocket.accept()

    # Verify job exists
    result = await db.execute(
        select(PresentationJob).where(PresentationJob.id == job_id)
    )
    job = result.scalar_one_or_none()

    if not job:
        await websocket.close(code=4004, reason="Job not found")
        return

    import redis.asyncio as redis
    from app.core.config import settings

    try:
        # Subscribe to Redis channel for this job
        redis_client = redis.from_url(settings.REDIS_URL)
        pubsub = redis_client.pubsub()
        channel = f"presentation:{job_id}:progress"
        await pubsub.subscribe(channel)

        # Send initial status
        await websocket.send_json({
            "type": "progress",
            "progress": job.progress,
            "stage": job.current_stage or "pending",
            "status": job.status,
        })

        # If already completed/failed, close immediately
        if job.status in ("completed", "failed", "cancelled"):
            await websocket.close()
            return

        # Listen for updates
        async for message in pubsub.listen():
            if message["type"] == "message":
                import json
                data = json.loads(message["data"])
                await websocket.send_json(data)

                # Close on completion
                if data.get("status") in ("completed", "failed", "cancelled"):
                    break

        await pubsub.unsubscribe(channel)
        await redis_client.close()

    except WebSocketDisconnect:
        logger.debug(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass


async def _get_user_job(
    job_id: UUID,
    user_id: UUID,
    db: AsyncSession
) -> PresentationJob:
    """Get a job ensuring it belongs to the user."""
    result = await db.execute(
        select(PresentationJob).where(
            PresentationJob.id == job_id,
            PresentationJob.user_id == user_id
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Presentation job not found"
        )

    return job


def _job_to_response(job: PresentationJob) -> PresentationJobResponse:
    """Convert database model to response schema."""
    template_name = None
    if job.template:
        template_name = job.template.name

    return PresentationJobResponse(
        id=job.id,
        user_id=job.user_id,
        title=job.title,
        topic=job.topic,
        source_document_ids=[str(doc_id) for doc_id in (job.source_document_ids or [])],
        slide_count=job.slide_count,
        style=job.style,
        include_diagrams=bool(job.include_diagrams),
        template_id=job.template_id,
        template_name=template_name,
        custom_theme=job.custom_theme,
        status=job.status,
        progress=job.progress,
        current_stage=job.current_stage,
        generated_outline=job.generated_outline,
        file_path=job.file_path,
        file_size=job.file_size,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


# =============================================================================
# Template Endpoints
# =============================================================================

@router.get("/templates", response_model=List[PresentationTemplateResponse])
async def list_templates(
    include_system: bool = True,
    include_public: bool = True,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List available presentation templates.

    Returns user's own templates plus system and public templates.
    """
    from sqlalchemy import or_

    conditions = [PresentationTemplate.user_id == current_user.id]

    if include_system:
        conditions.append(PresentationTemplate.is_system == True)
    if include_public:
        conditions.append(PresentationTemplate.is_public == True)

    query = select(PresentationTemplate).where(
        or_(*conditions),
        PresentationTemplate.is_active == True
    ).order_by(PresentationTemplate.is_system.desc(), PresentationTemplate.name)

    result = await db.execute(query)
    templates = result.scalars().all()

    return [_template_to_response(t) for t in templates]


@router.post("/templates", response_model=PresentationTemplateResponse, status_code=status.HTTP_201_CREATED)
async def create_template(
    request: PresentationTemplateCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new presentation template (theme).

    For PPTX file templates, use the upload endpoint instead.
    """
    template = PresentationTemplate(
        user_id=current_user.id,
        name=request.name,
        description=request.description,
        template_type=request.template_type,
        theme_config=request.theme_config.model_dump() if request.theme_config else None,
        is_public=request.is_public,
        is_system=False,
    )

    db.add(template)
    await db.commit()
    await db.refresh(template)

    logger.info(f"Created template {template.id} for user {current_user.id}")
    return _template_to_response(template)


@router.post("/templates/upload", response_model=PresentationTemplateResponse, status_code=status.HTTP_201_CREATED)
async def upload_pptx_template(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(None),
    is_public: bool = Form(False),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a PPTX file as a presentation template.

    The uploaded file will be validated and stored in MinIO.
    It can then be used as a base template when generating presentations.
    """
    # Validate file extension
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required"
        )

    if not file.filename.lower().endswith('.pptx'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .pptx files are allowed"
        )

    # Validate content type
    valid_content_types = [
        'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'application/octet-stream',  # Some browsers send this
    ]
    if file.content_type and file.content_type not in valid_content_types:
        logger.warning(f"Unexpected content type: {file.content_type}")

    # Read file content
    content = await file.read()

    # Validate file size (max 50MB)
    max_size = 50 * 1024 * 1024  # 50MB
    if len(content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size exceeds maximum allowed (50MB)"
        )

    # Save to temp file and validate PPTX structure
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.pptx', delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        # Validate PPTX structure using python-pptx
        from pptx import Presentation as PptxPresentation
        from pptx.exc import PackageNotFoundError

        try:
            prs = PptxPresentation(temp_path)
            slide_count = len(prs.slides)
            layout_count = len(prs.slide_layouts)
            logger.info(f"Valid PPTX uploaded: {slide_count} slides, {layout_count} layouts")
        except PackageNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or corrupted PPTX file"
            )
        except Exception as e:
            logger.error(f"Error parsing PPTX: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error parsing PPTX file: {str(e)}"
            )

        # Create template record first to get the ID
        from uuid import uuid4
        template_id = uuid4()

        # Sanitize filename
        safe_filename = re.sub(r'[^\w\-.]', '_', file.filename)

        # Upload to MinIO
        storage = StorageService()
        object_path = f"templates/{template_id}/{safe_filename}"

        try:
            await storage.upload_file(
                document_id=str(template_id),
                filename=safe_filename,
                content=content,
                content_type='application/vnd.openxmlformats-officedocument.presentationml.presentation',
                prefix="templates"
            )
            logger.info(f"Uploaded template file to {object_path}")
        except Exception as e:
            logger.error(f"Failed to upload template to MinIO: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to store template file"
            )

        # Create template record
        template = PresentationTemplate(
            id=template_id,
            user_id=current_user.id,
            name=name,
            description=description,
            template_type="pptx",
            file_path=object_path,
            is_public=is_public,
            is_system=False,
        )

        db.add(template)
        await db.commit()
        await db.refresh(template)

        logger.info(f"Created PPTX template {template.id} for user {current_user.id}")
        return _template_to_response(template)

    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@router.get("/templates/{template_id}", response_model=PresentationTemplateResponse)
async def get_template(
    template_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific template."""
    template = await _get_accessible_template(template_id, current_user.id, db)
    return _template_to_response(template)


@router.put("/templates/{template_id}", response_model=PresentationTemplateResponse)
async def update_template(
    template_id: UUID,
    request: PresentationTemplateUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update a presentation template.

    Only the template owner can update it. System templates cannot be modified.
    """
    template = await _get_user_template(template_id, current_user.id, db)

    if template.is_system:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot modify system templates"
        )

    if request.name is not None:
        template.name = request.name
    if request.description is not None:
        template.description = request.description
    if request.theme_config is not None:
        template.theme_config = request.theme_config.model_dump()
    if request.is_public is not None:
        template.is_public = request.is_public
    if request.is_active is not None:
        template.is_active = request.is_active

    await db.commit()
    await db.refresh(template)

    logger.info(f"Updated template {template_id}")
    return _template_to_response(template)


@router.delete("/templates/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_template(
    template_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a presentation template.

    Only the template owner can delete it. System templates cannot be deleted.
    """
    template = await _get_user_template(template_id, current_user.id, db)

    if template.is_system:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot delete system templates"
        )

    # Delete associated file if it's a PPTX template
    if template.file_path:
        storage = StorageService()
        try:
            await storage.delete_file(template.file_path)
        except Exception as e:
            logger.warning(f"Failed to delete template file {template.file_path}: {e}")

    await db.delete(template)
    await db.commit()

    logger.info(f"Deleted template {template_id}")


async def _get_accessible_template(
    template_id: UUID,
    user_id: UUID,
    db: AsyncSession
) -> PresentationTemplate:
    """Get a template that the user can access (own, public, or system)."""
    from sqlalchemy import or_

    result = await db.execute(
        select(PresentationTemplate).where(
            PresentationTemplate.id == template_id,
            PresentationTemplate.is_active == True,
            or_(
                PresentationTemplate.user_id == user_id,
                PresentationTemplate.is_system == True,
                PresentationTemplate.is_public == True
            )
        )
    )
    template = result.scalar_one_or_none()

    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Template not found"
        )

    return template


async def _get_user_template(
    template_id: UUID,
    user_id: UUID,
    db: AsyncSession
) -> PresentationTemplate:
    """Get a template owned by the user."""
    result = await db.execute(
        select(PresentationTemplate).where(
            PresentationTemplate.id == template_id,
            PresentationTemplate.user_id == user_id
        )
    )
    template = result.scalar_one_or_none()

    if not template:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Template not found or you don't own it"
        )

    return template


def _template_to_response(template: PresentationTemplate) -> PresentationTemplateResponse:
    """Convert database model to response schema."""
    return PresentationTemplateResponse(
        id=template.id,
        user_id=template.user_id,
        name=template.name,
        description=template.description,
        template_type=template.template_type,
        theme_config=template.theme_config,
        preview_url=None,  # TODO: Generate preview URL if preview_path exists
        is_system=template.is_system,
        is_public=template.is_public,
        is_active=template.is_active,
        created_at=template.created_at,
        updated_at=template.updated_at,
    )
