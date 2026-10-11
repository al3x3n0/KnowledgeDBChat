"""
API endpoints for AI-powered presentation generation.

The queries and writes are in
``modules/presentations/application/presentation_store.py``; this module maps
them to HTTP, queues the generation task and builds the responses.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    WebSocket,
    status,
)
from loguru import logger
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import NO_VALUE

from app.core.database import get_db
from app.models.presentation import PresentationJob, PresentationTemplate
from app.models.user import User
from app.modules.presentations.application import presentation_store as store
from app.schemas.presentation import (
    PresentationJobCreate,
    PresentationJobResponse,
    PresentationTemplateCreate,
    PresentationTemplateResponse,
    PresentationTemplateUpdate,
)
from app.services.auth_service import get_current_user
from app.utils.http_headers import content_disposition

router = APIRouter()

#: How each kind of refusal is told to a client. The store names the kind;
#: this is the only place that knows the numbers.
_REFUSAL_STATUS = {
    "invalid": 400,
    "forbidden": 403,
    "not_found": 404,
    "unavailable": 500,
}


def _http(refusal: store.Refused) -> HTTPException:
    return HTTPException(
        status_code=_REFUSAL_STATUS[refusal.kind], detail=refusal.detail
    )


def _queue(job: PresentationJob, user_id: UUID) -> None:
    from app.tasks.presentation_tasks import generate_presentation_task

    generate_presentation_task.delay(str(job.id), str(user_id))


@router.post(
    "", response_model=PresentationJobResponse, status_code=status.HTTP_201_CREATED
)
async def create_presentation(
    request: PresentationJobCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new presentation generation job.

    The presentation will be generated asynchronously via Celery.
    Use GET /presentations/{job_id} to check status or WebSocket for real-time updates.
    """
    template_name: Optional[str] = None
    try:
        if request.template_id:
            template = await store.usable_template(
                db, request.template_id, current_user.id
            )
            template_name = template.name
    except store.Refused as refusal:
        raise _http(refusal)

    job = await store.create_job(
        db,
        current_user.id,
        title=request.title,
        topic=request.topic,
        source_document_ids=[str(doc_id) for doc_id in request.source_document_ids]
        if request.source_document_ids
        else [],
        slide_count=request.slide_count,
        style=request.style,
        include_diagrams=1 if request.include_diagrams else 0,
        template_id=request.template_id,
        custom_theme=request.custom_theme.model_dump()
        if request.custom_theme
        else None,
    )
    _queue(job, current_user.id)

    logger.info(f"Created presentation job {job.id} for user {current_user.id}")

    return _job_to_response(job, template_name=template_name)


@router.post(
    "/from-research",
    response_model=PresentationJobResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_research_presentation(
    topic: str = Form(..., description="Research topic for the presentation"),
    slide_count: int = Form(10, ge=5, le=25, description="Number of slides"),
    include_arxiv: bool = Form(True, description="Search arXiv for papers"),
    arxiv_max_papers: int = Form(
        5, ge=0, le=10, description="Max arXiv papers to ingest"
    ),
    style: str = Form("technical", description="Presentation style"),
    include_diagrams: bool = Form(True, description="Include diagrams"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a research presentation from a topic.

    This endpoint:
    1. Searches the knowledge base for relevant documents
    2. Optionally searches arXiv and ingests relevant papers (instant ingest)
    3. Creates a presentation generation job

    The arXiv papers are ingested synchronously (fast-path) so they're
    immediately available for the presentation.
    """
    # Read now. A failed ingest rolls back, which expires every object in the
    # session, the caller among them, and reading an expired attribute
    # afterwards is a database call from a place that cannot make one.
    user_id = current_user.id

    source_document_ids: List[str] = []
    if include_arxiv and arxiv_max_papers > 0:
        source_document_ids = await store.ingest_arxiv_sources(
            db, user_id, topic, arxiv_max_papers
        )

    job = await store.create_job(
        db,
        user_id,
        title=f"Research: {topic}",
        topic=topic,
        source_document_ids=source_document_ids,
        slide_count=slide_count,
        style=style,
        include_diagrams=include_diagrams,
        current_stage="queued",
    )
    _queue(job, user_id)

    logger.info(
        f"Created research presentation job {job.id} with {len(source_document_ids)} source documents"
    )

    return _job_to_response(job)


@router.get("", response_model=List[PresentationJobResponse])
async def list_presentations(
    # Bounded: a negative limit is "no limit" to the database, and this
    # returned every job the user had.
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status_filter: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List user's presentation generation jobs.

    Results are ordered by creation date (newest first).
    """
    jobs = await store.list_jobs(
        db, current_user.id, limit=limit, offset=offset, status=status_filter
    )
    return [_job_to_response(job) for job in jobs]


def _job_to_response(
    job: PresentationJob, template_name: Optional[str] = None
) -> PresentationJobResponse:
    """Convert database model to response schema."""
    if template_name is None:
        # Avoid async lazy-loading (MissingGreenlet) by only reading the relationship
        # if it's already loaded.
        try:
            insp = sa_inspect(job)
            if (
                insp.attrs.template.loaded_value is not NO_VALUE
                and job.template is not None
            ):
                template_name = job.template.name
        except Exception:
            template_name = None

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
        retrieval_trace_id=getattr(job, "retrieval_trace_id", None),
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
    db: AsyncSession = Depends(get_db),
):
    """
    List available presentation templates.

    Returns user's own templates plus system and public templates.
    """
    templates = await store.list_templates(
        db,
        current_user.id,
        include_system=include_system,
        include_public=include_public,
    )
    return [_template_to_response(t) for t in templates]


@router.post(
    "/templates",
    response_model=PresentationTemplateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_template(
    request: PresentationTemplateCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new presentation template (theme).

    For PPTX file templates, use the upload endpoint instead.
    """
    template = await store.create_template(
        db,
        current_user.id,
        name=request.name,
        description=request.description,
        template_type=request.template_type,
        theme_config=request.theme_config.model_dump()
        if request.theme_config
        else None,
        is_public=request.is_public,
    )

    logger.info(f"Created template {template.id} for user {current_user.id}")
    return _template_to_response(template)


@router.post(
    "/templates/upload",
    response_model=PresentationTemplateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_pptx_template(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(None),
    is_public: bool = Form(False),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a PPTX file as a presentation template.

    The uploaded file will be validated and stored in MinIO.
    It can then be used as a base template when generating presentations.
    """
    try:
        template = await store.store_uploaded_template(
            db,
            current_user.id,
            filename=file.filename,
            content=await file.read(),
            name=name,
            description=description,
            is_public=is_public,
        )
    except store.Refused as refusal:
        raise _http(refusal)

    logger.info(f"Created PPTX template {template.id} for user {current_user.id}")
    return _template_to_response(template)


@router.get("/templates/{template_id}", response_model=PresentationTemplateResponse)
async def get_template(
    template_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a specific template."""
    try:
        template = await store.get_accessible_template(db, template_id, current_user.id)
    except store.Refused as refusal:
        raise _http(refusal)
    return _template_to_response(template)


@router.put("/templates/{template_id}", response_model=PresentationTemplateResponse)
async def update_template(
    template_id: UUID,
    request: PresentationTemplateUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update a presentation template.

    Only the template owner can update it. System templates cannot be modified.
    """
    try:
        template = await store.update_template(
            db,
            template_id,
            current_user.id,
            {
                "name": request.name,
                "description": request.description,
                "theme_config": request.theme_config.model_dump()
                if request.theme_config is not None
                else None,
                "is_public": request.is_public,
                "is_active": request.is_active,
            },
        )
    except store.Refused as refusal:
        raise _http(refusal)

    logger.info(f"Updated template {template_id}")
    return _template_to_response(template)


@router.delete("/templates/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_template(
    template_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a presentation template.

    Only the template owner can delete it. System templates cannot be deleted.
    """
    try:
        await store.delete_template(db, template_id, current_user.id)
    except store.Refused as refusal:
        raise _http(refusal)

    logger.info(f"Deleted template {template_id}")


# =============================================================================
# Job Detail Endpoints
# =============================================================================


async def _get_user_job(
    job_id: UUID, user_id: UUID, db: AsyncSession
) -> PresentationJob:
    """Get a job ensuring it belongs to the user."""
    try:
        return await store.get_job(db, job_id, user_id)
    except store.Refused as refusal:
        raise _http(refusal)


@router.get("/{job_id}", response_model=PresentationJobResponse)
async def get_presentation_job(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
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
    db: AsyncSession = Depends(get_db),
):
    """
    Download the generated presentation.
    """
    job = await _get_user_job(job_id, current_user.id, db)
    try:
        filename, content = await store.read_deck(job)
    except store.Refused as refusal:
        raise _http(refusal)

    return Response(
        content=content,
        media_type=store.PPTX_MEDIA_TYPE,
        headers={
            "Content-Disposition": content_disposition(filename),
            "Content-Length": str(len(content)),
        },
    )


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_presentation(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a presentation job and its associated file.
    """
    job = await _get_user_job(job_id, current_user.id, db)
    await store.delete_job(db, job)

    logger.info(f"Deleted presentation job {job_id}")


@router.post("/{job_id}/cancel", response_model=PresentationJobResponse)
async def cancel_presentation(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Cancel a pending or in-progress presentation generation job.
    """
    job = await _get_user_job(job_id, current_user.id, db)
    try:
        job = await store.cancel_job(db, job)
    except store.Refused as refusal:
        raise _http(refusal)

    logger.info(f"Cancelled presentation job {job_id}")

    return _job_to_response(job)


@router.websocket("/{job_id}/progress")
async def presentation_progress(
    websocket: WebSocket,
    job_id: UUID,
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
    from app.core.database import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(PresentationJob).where(PresentationJob.id == job_id)
        )
        job = result.scalar_one_or_none()

    if not job:
        await websocket.close(code=4004, reason="Job not found")
        return

    from app.utils.websocket_auth import authorize_owner

    if await authorize_owner(websocket, job.user_id) is None:
        return

    from app.utils.websocket_progress import TERMINAL_STATUSES, forward_progress

    await forward_progress(
        websocket,
        f"presentation:{job_id}:progress",
        initial={
            "type": "progress",
            "progress": job.progress,
            "stage": job.current_stage or "pending",
            "status": job.status,
        },
        already_finished=job.status in TERMINAL_STATUSES,
    )


def _template_to_response(
    template: PresentationTemplate,
) -> PresentationTemplateResponse:
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
