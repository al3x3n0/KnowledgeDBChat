from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger
from pydantic import BaseModel
import base64

from app.core.database import get_db
from app.models.document import DocumentSource, GitBranchDiff
from app.schemas.git import GitBranchResponse, GitCompareRequest, GitCompareJobResponse
from app.services.auth_service import get_current_user
from app.models.user import User
from app.services.git_service import GitService
from app.tasks.git_compare_tasks import compare_git_branches
from celery.result import AsyncResult
from app.core.celery import celery_app
from app.utils.ingestion_state import (
    set_git_compare_task,
    get_git_compare_task,
    set_git_compare_cancel_flag,
    clear_git_compare_task,
)

router = APIRouter()
git_service = GitService()


async def _get_source_or_404(db: AsyncSession, source_id: UUID) -> DocumentSource:
    source = await db.get(DocumentSource, source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Document source not found")
    if source.source_type not in ("github", "gitlab"):
        raise HTTPException(status_code=400, detail="Source does not support git operations")
    return source


def _check_source_access(source: DocumentSource, user: User) -> None:
    if user.is_admin():
        return
    config = source.config or {}
    requested_by = config.get("requested_by") or config.get("requestedBy")
    if requested_by != user.username:
        raise HTTPException(status_code=403, detail="Not authorized for this source")


def _default_repository(source: DocumentSource) -> Optional[str]:
    config = source.config or {}
    if source.source_type == "github":
        repos = config.get("repos") or []
        if repos:
            first = repos[0]
            if isinstance(first, str):
                return first
            owner = first.get("owner")
            repo = first.get("repo")
            if owner and repo:
                return f"{owner}/{repo}"
    else:
        projects = config.get("projects") or []
        if projects:
            proj = projects[0]
            return str(proj.get("id") or proj.get("path") or proj.get("name"))
    return None


@router.get("/sources/{source_id}/branches", response_model=List[GitBranchResponse])
async def list_git_branches(
    source_id: UUID,
    repository: Optional[str] = Query(default=None, description="Repository identifier"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    source = await _get_source_or_404(db, source_id)
    _check_source_access(source, current_user)
    repository = repository or _default_repository(source)
    if not repository:
        raise HTTPException(status_code=400, detail="Repository parameter required")
    branches = await git_service.list_branches(source, repository)
    return [
        GitBranchResponse(
            repository=repository,
            name=branch.get("name"),
            commit_sha=branch.get("commit_sha"),
            commit_message=branch.get("commit_message"),
            commit_author=branch.get("commit_author"),
            commit_date=branch.get("commit_date"),
            protected=branch.get("protected"),
        )
        for branch in branches
    ]


@router.post("/sources/{source_id}/compare", response_model=GitCompareJobResponse)
async def start_branch_comparison(
    source_id: UUID,
    request: GitCompareRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    source = await _get_source_or_404(db, source_id)
    _check_source_access(source, current_user)

    diff = GitBranchDiff(
        source_id=source.id,
        repository=request.repository,
        base_branch=request.base_branch,
        compare_branch=request.compare_branch,
        status="queued",
        options={
            "include_files": request.include_files,
            "explain": request.explain,
        },
    )
    db.add(diff)
    await db.commit()
    await db.refresh(diff)

    task = compare_git_branches.delay(str(diff.id), user_id=str(current_user.id))
    diff.task_id = task.id
    await db.commit()
    await set_git_compare_task(str(diff.id), task.id)

    return GitCompareJobResponse.from_orm(diff)


@router.get("/compare/", response_model=List[GitCompareJobResponse])
async def list_compare_jobs(
    source_id: Optional[UUID] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(GitBranchDiff, DocumentSource).join(DocumentSource, GitBranchDiff.source_id == DocumentSource.id)
    if source_id:
        stmt = stmt.where(GitBranchDiff.source_id == source_id)
    result = await db.execute(stmt.order_by(GitBranchDiff.created_at.desc()).limit(50))
    jobs = []
    for diff, source in result.all():
        try:
            _check_source_access(source, current_user)
            jobs.append(GitCompareJobResponse.from_orm(diff))
        except HTTPException:
            continue
    return jobs


@router.get("/compare/{diff_id}", response_model=GitCompareJobResponse)
async def get_compare_job(
    diff_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    diff = await db.get(GitBranchDiff, diff_id)
    if not diff:
        raise HTTPException(status_code=404, detail="Comparison not found")
    source = await db.get(DocumentSource, diff.source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Document source missing")
    _check_source_access(source, current_user)
    return GitCompareJobResponse.from_orm(diff)


@router.post("/compare/{diff_id}/cancel")
async def cancel_compare_job(
    diff_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    diff = await db.get(GitBranchDiff, diff_id)
    if not diff:
        raise HTTPException(status_code=404, detail="Comparison not found")
    source = await db.get(DocumentSource, diff.source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source missing")
    _check_source_access(source, current_user)

    task_id = diff.task_id or await get_git_compare_task(str(diff_id))
    await set_git_compare_cancel_flag(str(diff_id), ttl=600)
    if task_id:
        try:
            AsyncResult(task_id, app=celery_app).revoke(terminate=True)
        except Exception as exc:
            logger.warning(f"Failed to revoke git compare task {task_id}: {exc}")
    if diff.status not in ("completed", "failed", "canceled"):
        diff.status = "cancel_requested"
        await db.commit()
    return {"message": "Cancellation requested", "task_id": task_id}


# ============================================================================
# Architecture Diagram Generation
# ============================================================================


class ArchitectureDiagramRequest(BaseModel):
    """Request model for generating architecture diagrams."""
    project_id: str
    branch: Optional[str] = None
    diagram_type: str = "auto"
    focus: Optional[str] = None
    detail_level: str = "medium"


class ArchitectureDiagramResponse(BaseModel):
    """Response model for architecture diagram generation."""
    project_name: str
    project_url: Optional[str] = None
    mermaid_code: str
    diagram_type: str
    focus: Optional[str] = None
    analysis_summary: dict
    has_png: bool = False
    has_svg: bool = False
    render_error: Optional[str] = None


@router.post("/sources/{source_id}/architecture", response_model=ArchitectureDiagramResponse)
async def generate_architecture_diagram(
    source_id: UUID,
    request: ArchitectureDiagramRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate an architecture diagram from a GitLab repository.

    Analyzes the repository structure, README, config files, and code
    to understand the system architecture and generate a Mermaid diagram.
    """
    from app.services.gitlab_architecture_service import get_gitlab_architecture_service

    source = await _get_source_or_404(db, source_id)
    _check_source_access(source, current_user)

    if source.source_type != "gitlab":
        raise HTTPException(
            status_code=400,
            detail="Architecture diagrams are only supported for GitLab sources"
        )

    config = source.config or {}
    gitlab_url = config.get("gitlab_url")
    token = config.get("token")

    if not gitlab_url or not token:
        raise HTTPException(
            status_code=400,
            detail="GitLab source is missing URL or token configuration"
        )

    try:
        service = get_gitlab_architecture_service()
        result = await service.generate_architecture_diagram(
            gitlab_url=gitlab_url,
            token=token,
            project_id=request.project_id,
            branch=request.branch,
            diagram_type=request.diagram_type,
            focus=request.focus,
            detail_level=request.detail_level,
        )

        return ArchitectureDiagramResponse(
            project_name=result["project"]["name"],
            project_url=result["project"].get("web_url"),
            mermaid_code=result["mermaid_code"],
            diagram_type=result["diagram_type"],
            focus=result["focus"],
            analysis_summary=result["analysis_summary"],
            has_png=result.get("png_base64") is not None,
            has_svg=result.get("svg") is not None,
            render_error=result.get("render_error"),
        )

    except Exception as e:
        logger.error(f"Error generating architecture diagram: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sources/{source_id}/architecture/png")
async def generate_architecture_diagram_png(
    source_id: UUID,
    request: ArchitectureDiagramRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate an architecture diagram as PNG image.

    Returns the rendered PNG image directly.
    """
    from app.services.gitlab_architecture_service import get_gitlab_architecture_service

    source = await _get_source_or_404(db, source_id)
    _check_source_access(source, current_user)

    if source.source_type != "gitlab":
        raise HTTPException(
            status_code=400,
            detail="Architecture diagrams are only supported for GitLab sources"
        )

    config = source.config or {}
    gitlab_url = config.get("gitlab_url")
    token = config.get("token")

    if not gitlab_url or not token:
        raise HTTPException(
            status_code=400,
            detail="GitLab source is missing URL or token configuration"
        )

    try:
        service = get_gitlab_architecture_service()
        result = await service.generate_architecture_diagram(
            gitlab_url=gitlab_url,
            token=token,
            project_id=request.project_id,
            branch=request.branch,
            diagram_type=request.diagram_type,
            focus=request.focus,
            detail_level=request.detail_level,
        )

        if not result.get("png_base64"):
            raise HTTPException(
                status_code=500,
                detail=result.get("render_error", "Failed to render PNG")
            )

        png_bytes = base64.b64decode(result["png_base64"])
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={
                "Content-Disposition": f"inline; filename=architecture_{request.project_id}.png"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating architecture PNG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sources/{source_id}/architecture/svg")
async def generate_architecture_diagram_svg(
    source_id: UUID,
    request: ArchitectureDiagramRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate an architecture diagram as SVG.

    Returns the rendered SVG directly.
    """
    from app.services.gitlab_architecture_service import get_gitlab_architecture_service

    source = await _get_source_or_404(db, source_id)
    _check_source_access(source, current_user)

    if source.source_type != "gitlab":
        raise HTTPException(
            status_code=400,
            detail="Architecture diagrams are only supported for GitLab sources"
        )

    config = source.config or {}
    gitlab_url = config.get("gitlab_url")
    token = config.get("token")

    if not gitlab_url or not token:
        raise HTTPException(
            status_code=400,
            detail="GitLab source is missing URL or token configuration"
        )

    try:
        service = get_gitlab_architecture_service()
        result = await service.generate_architecture_diagram(
            gitlab_url=gitlab_url,
            token=token,
            project_id=request.project_id,
            branch=request.branch,
            diagram_type=request.diagram_type,
            focus=request.focus,
            detail_level=request.detail_level,
        )

        if not result.get("svg"):
            raise HTTPException(
                status_code=500,
                detail=result.get("render_error", "Failed to render SVG")
            )

        return Response(
            content=result["svg"],
            media_type="image/svg+xml",
            headers={
                "Content-Disposition": f"inline; filename=architecture_{request.project_id}.svg"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating architecture SVG: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sources/{source_id}/analyze")
async def analyze_repository_structure(
    source_id: UUID,
    project_id: str = Query(..., description="GitLab project ID or path"),
    branch: Optional[str] = Query(None, description="Branch to analyze"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Analyze a GitLab repository structure without generating a diagram.

    Returns the analysis data including components, dependencies, and file structure.
    """
    from app.services.gitlab_architecture_service import get_gitlab_architecture_service

    source = await _get_source_or_404(db, source_id)
    _check_source_access(source, current_user)

    if source.source_type != "gitlab":
        raise HTTPException(
            status_code=400,
            detail="Repository analysis is only supported for GitLab sources"
        )

    config = source.config or {}
    gitlab_url = config.get("gitlab_url")
    token = config.get("token")

    if not gitlab_url or not token:
        raise HTTPException(
            status_code=400,
            detail="GitLab source is missing URL or token configuration"
        )

    try:
        service = get_gitlab_architecture_service()
        analysis = await service.analyze_repository(
            gitlab_url=gitlab_url,
            token=token,
            project_id=project_id,
            branch=branch,
        )

        return analysis

    except Exception as e:
        logger.error(f"Error analyzing repository: {e}")
        raise HTTPException(status_code=500, detail=str(e))
