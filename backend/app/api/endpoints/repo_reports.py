"""
API endpoints for repository report and presentation generation.
"""

import re
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from loguru import logger

from app.core.database import get_db
from app.api.endpoints.users import get_current_user
from app.models.user import User
from app.models.repo_report import RepoReportJob
from app.models.document import DocumentSource
from app.schemas.repo_report import (
    RepoReportJobCreate,
    RepoReportJobResponse,
    RepoReportJobListItem,
    RepoReportJobListResponse,
    AvailableSectionsResponse,
    AVAILABLE_SECTIONS,
)
from app.services.storage_service import StorageService


router = APIRouter()


def _parse_repo_url(url: str) -> tuple[str, str, str]:
    """
    Parse a repository URL to extract type, owner, and repo name.

    Returns:
        Tuple of (repo_type, owner, repo_name)
    """
    # GitHub patterns
    github_patterns = [
        r"github\.com[:/]([^/]+)/([^/.]+)",
        r"api\.github\.com/repos/([^/]+)/([^/]+)",
    ]
    for pattern in github_patterns:
        match = re.search(pattern, url)
        if match:
            return ("github", match.group(1), match.group(2).rstrip(".git"))

    # GitLab patterns
    gitlab_patterns = [
        r"gitlab\.com[:/]([^/]+)/([^/.]+)",
        r"gitlab\.[^/]+[:/]([^/]+)/([^/.]+)",
    ]
    for pattern in gitlab_patterns:
        match = re.search(pattern, url)
        if match:
            return ("gitlab", match.group(1), match.group(2).rstrip(".git"))

    raise ValueError(f"Could not parse repository URL: {url}")


@router.get("/sections", response_model=AvailableSectionsResponse)
async def list_available_sections():
    """
    List available sections for repository reports/presentations.

    Returns information about each section and whether it's included by default.
    """
    return AvailableSectionsResponse()


@router.post("", response_model=RepoReportJobResponse, status_code=status.HTTP_201_CREATED)
async def create_repo_report(
    request: RepoReportJobCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new repository report or presentation generation job.

    You can provide either:
    - `source_id`: ID of an existing DocumentSource (GitHub/GitLab repo)
    - `repo_url`: Repository URL for ad-hoc analysis

    The report/presentation will be generated asynchronously via Celery.
    Use GET /repo-reports/{job_id} to check status.
    """
    repo_name = ""
    repo_url = ""
    repo_type = ""

    # Validate source or URL
    if request.source_id:
        # Load source from database
        result = await db.execute(
            select(DocumentSource).where(DocumentSource.id == request.source_id)
        )
        source = result.scalar_one_or_none()
        if not source:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="DocumentSource not found"
            )

        source_type = source.source_type.lower()
        if source_type not in ("github", "gitlab"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Source type '{source_type}' is not supported. Only 'github' and 'gitlab' sources are supported."
            )

        config = source.config or {}
        repo_type = source_type

        # Extract repo info from config
        if source_type == "github":
            repos = config.get("repos", [])
            if repos:
                if isinstance(repos[0], str) and "/" in repos[0]:
                    repo_name = repos[0]
                    repo_url = f"https://github.com/{repos[0]}"
                else:
                    owner = repos[0].get("owner", "")
                    repo = repos[0].get("repo", "")
                    repo_name = f"{owner}/{repo}"
                    repo_url = f"https://github.com/{owner}/{repo}"
        elif source_type == "gitlab":
            projects = config.get("projects", [])
            gitlab_base = config.get("gitlab_url", "https://gitlab.com")
            if projects:
                project_id = projects[0].get("id") or projects[0].get("name", "")
                repo_name = str(project_id)
                repo_url = f"{gitlab_base}/{project_id}"

        if not repo_name:
            repo_name = source.name

    elif request.repo_url:
        try:
            logger.info(f"Parsing repo URL: '{request.repo_url}' (len={len(request.repo_url)})")
            repo_type, owner, repo = _parse_repo_url(request.repo_url)
            logger.info(f"Parsed: type={repo_type}, owner={owner}, repo={repo}")
            repo_name = f"{owner}/{repo}"
            repo_url = request.repo_url
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either source_id or repo_url must be provided"
        )

    # Determine title
    title = request.title or f"{repo_name} Report"

    # Create job in database
    job = RepoReportJob(
        user_id=current_user.id,
        source_id=request.source_id,
        adhoc_url=request.repo_url if not request.source_id else None,
        adhoc_token=request.repo_token if not request.source_id else None,
        repo_name=repo_name,
        repo_url=repo_url,
        repo_type=repo_type,
        output_format=request.output_format,
        title=title,
        sections=request.sections,
        slide_count=request.slide_count if request.output_format == "pptx" else None,
        include_diagrams=request.include_diagrams,
        style=request.style,
        custom_theme=request.custom_theme.model_dump() if request.custom_theme else None,
        status="pending",
        progress=0,
    )

    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Dispatch Celery task
    from app.tasks.repo_report_tasks import generate_repo_report_task
    generate_repo_report_task.delay(str(job.id), str(current_user.id))

    logger.info(f"Created repo report job {job.id} for user {current_user.id}, repo: {repo_name}")

    return _job_to_response(job)


@router.get("", response_model=RepoReportJobListResponse)
async def list_repo_reports(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    output_format: Optional[str] = Query(None, description="Filter by output format"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List user's repository report generation jobs.

    Results are ordered by creation date (newest first).
    """
    # Build query
    query = select(RepoReportJob).where(
        RepoReportJob.user_id == current_user.id
    )

    if status_filter:
        query = query.where(RepoReportJob.status == status_filter)
    if output_format:
        query = query.where(RepoReportJob.output_format == output_format)

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Get paginated results
    query = query.order_by(RepoReportJob.created_at.desc())
    query = query.offset(offset).limit(limit)

    result = await db.execute(query)
    jobs = result.scalars().all()

    return RepoReportJobListResponse(
        jobs=[_job_to_list_item(job) for job in jobs],
        total=total
    )


@router.get("/{job_id}", response_model=RepoReportJobResponse)
async def get_repo_report(
    job_id: UUID,
    include_analysis: bool = Query(False, description="Include cached analysis data"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get details of a repository report job.

    Set include_analysis=true to include the full analysis data (can be large).
    """
    job = await _get_user_job(job_id, current_user.id, db)
    return _job_to_response(job, include_analysis=include_analysis)


@router.get("/{job_id}/download")
async def download_repo_report(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Download the generated report or presentation.

    Returns the file as a direct download.
    """
    job = await _get_user_job(job_id, current_user.id, db)

    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Report is not ready. Current status: {job.status}"
        )

    if not job.file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report file not found"
        )

    # Get file from MinIO
    storage = StorageService()
    try:
        await storage.initialize()
        content = await storage.get_file_content(job.file_path)

        # Determine filename and content type based on format
        output_format = job.output_format.lower()
        safe_title = job.title.replace(" ", "_").replace("/", "_")[:50]

        if output_format == "pptx":
            filename = f"{safe_title}.pptx"
            media_type = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        elif output_format == "pdf":
            filename = f"{safe_title}.pdf"
            media_type = "application/pdf"
        else:  # docx
            filename = f"{safe_title}.docx"
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

        return Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(content)),
            }
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report file not found in storage"
        )
    except Exception as e:
        logger.error(f"Failed to download report for job {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download report"
        )


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_repo_report(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a repository report job and its associated file.
    """
    job = await _get_user_job(job_id, current_user.id, db)

    # Delete file from MinIO if exists
    if job.file_path:
        storage = StorageService()
        try:
            await storage.initialize()
            await storage.delete_file(job.file_path)
        except Exception as e:
            logger.warning(f"Failed to delete report file {job.file_path}: {e}")

    # Delete job from database
    await db.delete(job)
    await db.commit()

    logger.info(f"Deleted repo report job {job_id}")


@router.post("/{job_id}/cancel", response_model=RepoReportJobResponse)
async def cancel_repo_report(
    job_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Cancel a pending or in-progress report generation job.
    """
    job = await _get_user_job(job_id, current_user.id, db)

    if job.status in ("completed", "failed", "cancelled"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job with status: {job.status}"
        )

    job.status = "cancelled"
    job.current_stage = "Cancelled by user"
    await db.commit()
    await db.refresh(job)

    logger.info(f"Cancelled repo report job {job_id}")

    return _job_to_response(job)


@router.websocket("/{job_id}/progress")
async def repo_report_progress(
    websocket: WebSocket,
    job_id: UUID,
):
    """
    WebSocket endpoint for real-time progress updates.

    Sends JSON messages with format:
    {
        "type": "progress",
        "progress": 50,
        "stage": "Fetching repository info",
        "status": "analyzing"
    }

    Connection closes automatically when job completes or fails.
    """
    await websocket.accept()

    # Verify job exists
    from app.core.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(RepoReportJob).where(RepoReportJob.id == job_id)
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
        channel = f"repo_report:{job_id}:progress"
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
        logger.debug(f"WebSocket disconnected for repo report job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for repo report job {job_id}: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# =============================================================================
# Helper Functions
# =============================================================================

async def _get_user_job(
    job_id: UUID,
    user_id: UUID,
    db: AsyncSession
) -> RepoReportJob:
    """Get a job ensuring it belongs to the user."""
    result = await db.execute(
        select(RepoReportJob).where(
            RepoReportJob.id == job_id,
            RepoReportJob.user_id == user_id
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository report job not found"
        )

    return job


def _job_to_response(job: RepoReportJob, include_analysis: bool = False) -> RepoReportJobResponse:
    """Convert database model to response schema."""
    from app.schemas.repo_report import RepoAnalysisResult

    analysis_data = None
    if include_analysis and job.analysis_data:
        try:
            analysis_data = RepoAnalysisResult.model_validate(job.analysis_data)
        except Exception:
            pass

    return RepoReportJobResponse(
        id=job.id,
        user_id=job.user_id,
        source_id=job.source_id,
        adhoc_url=job.adhoc_url,
        repo_name=job.repo_name,
        repo_url=job.repo_url,
        repo_type=job.repo_type,
        output_format=job.output_format,
        title=job.title,
        sections=job.sections,
        slide_count=job.slide_count,
        include_diagrams=job.include_diagrams,
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
        analysis_data=analysis_data,
    )


def _job_to_list_item(job: RepoReportJob) -> RepoReportJobListItem:
    """Convert database model to list item schema."""
    return RepoReportJobListItem(
        id=job.id,
        user_id=job.user_id,
        repo_name=job.repo_name,
        repo_url=job.repo_url,
        repo_type=job.repo_type,
        output_format=job.output_format,
        title=job.title,
        status=job.status,
        progress=job.progress,
        file_size=job.file_size,
        error=job.error,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )
