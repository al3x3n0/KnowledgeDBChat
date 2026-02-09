"""
Synthesis API endpoints.

Provides endpoints for multi-document synthesis, comparative analysis,
theme extraction, and report generation.
"""

from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from loguru import logger

from app.core.database import get_db
from app.services.auth_service import get_current_user
from app.models.user import User
from app.models.synthesis_job import SynthesisJob, SynthesisJobType, SynthesisJobStatus
from app.services.synthesis_service import synthesis_service
from app.services.storage_service import storage_service

router = APIRouter()


# ==================== Schemas ====================

class SynthesisJobCreate(BaseModel):
    """Request to create a synthesis job."""
    job_type: str = Field(..., description="Type of synthesis: multi_doc_summary, comparative_analysis, theme_extraction, knowledge_synthesis, research_report, executive_brief, gap_analysis_hypotheses")
    title: str = Field(..., description="Title for the synthesis")
    document_ids: List[str] = Field(..., description="List of document IDs to synthesize")
    description: Optional[str] = Field(None, description="Optional description")
    search_query: Optional[str] = Field(None, description="Optional search query for additional documents")
    topic: Optional[str] = Field(None, description="Focus topic for synthesis")
    options: Optional[dict] = Field(None, description="Synthesis options")
    output_format: str = Field("markdown", description="Output format: markdown, docx, pdf, pptx")
    output_style: str = Field("professional", description="Style: professional, technical, casual")


class SynthesisJobResponse(BaseModel):
    """Synthesis job response."""
    id: str
    user_id: str
    job_type: str
    title: str
    description: Optional[str]
    document_ids: List[str]
    search_query: Optional[str]
    topic: Optional[str]
    options: Optional[dict]
    output_format: str
    output_style: str
    status: str
    progress: int
    current_stage: Optional[str]
    result_content: Optional[str]
    result_metadata: Optional[dict]
    artifacts: Optional[List[dict]]
    file_path: Optional[str]
    file_size: Optional[int]
    error: Optional[str]
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]

    class Config:
        from_attributes = True


class SynthesisJobListResponse(BaseModel):
    """List of synthesis jobs."""
    jobs: List[SynthesisJobResponse]
    total: int
    page: int
    page_size: int


# ==================== Endpoints ====================

@router.post("", response_model=SynthesisJobResponse)
async def create_synthesis_job(
    request: SynthesisJobCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Create a new document synthesis job.

    Job types:
    - multi_doc_summary: Summarize multiple documents into one cohesive summary
    - comparative_analysis: Compare and contrast documents
    - theme_extraction: Extract common themes across documents
    - knowledge_synthesis: Synthesize new knowledge from sources
    - research_report: Generate formal research report
    - executive_brief: Create executive briefing
    - gap_analysis_hypotheses: Identify research gaps and propose testable hypotheses + experiments
    """
    # Validate job type
    valid_types = [t.value for t in SynthesisJobType]
    if request.job_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid job_type. Must be one of: {', '.join(valid_types)}"
        )

    # Validate output format
    valid_formats = ["markdown", "docx", "pdf", "pptx"]
    if request.output_format not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid output_format. Must be one of: {', '.join(valid_formats)}"
        )

    # Validate document_ids
    if not request.document_ids:
        raise HTTPException(status_code=400, detail="At least one document_id is required")

    if len(request.document_ids) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 documents allowed")

    # Create job
    job = await synthesis_service.create_job(
        db=db,
        user_id=current_user.id,
        job_type=request.job_type,
        title=request.title,
        document_ids=request.document_ids,
        description=request.description,
        search_query=request.search_query,
        topic=request.topic,
        options=request.options,
        output_format=request.output_format,
        output_style=request.output_style,
    )

    # Queue background task
    from app.tasks.synthesis_tasks import execute_synthesis_task
    execute_synthesis_task.delay(str(job.id), str(current_user.id))

    return SynthesisJobResponse(
        id=str(job.id),
        user_id=str(job.user_id),
        job_type=job.job_type,
        title=job.title,
        description=job.description,
        document_ids=job.document_ids,
        search_query=job.search_query,
        topic=job.topic,
        options=job.options,
        output_format=job.output_format,
        output_style=job.output_style,
        status=job.status,
        progress=job.progress,
        current_stage=job.current_stage,
        result_content=job.result_content,
        result_metadata=job.result_metadata,
        artifacts=job.artifacts,
        file_path=job.file_path,
        file_size=job.file_size,
        error=job.error,
        created_at=job.created_at.isoformat() if job.created_at else None,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


@router.get("", response_model=SynthesisJobListResponse)
async def list_synthesis_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    job_type: Optional[str] = Query(None, description="Filter by job type"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List user's synthesis jobs."""
    query = select(SynthesisJob).where(SynthesisJob.user_id == current_user.id)

    if status:
        query = query.where(SynthesisJob.status == status)
    if job_type:
        query = query.where(SynthesisJob.job_type == job_type)

    query = query.order_by(desc(SynthesisJob.created_at))

    # Count total
    from sqlalchemy import func
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Paginate
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    jobs = result.scalars().all()

    return SynthesisJobListResponse(
        jobs=[
            SynthesisJobResponse(
                id=str(j.id),
                user_id=str(j.user_id),
                job_type=j.job_type,
                title=j.title,
                description=j.description,
                document_ids=j.document_ids,
                search_query=j.search_query,
                topic=j.topic,
                options=j.options,
                output_format=j.output_format,
                output_style=j.output_style,
                status=j.status,
                progress=j.progress,
                current_stage=j.current_stage,
                result_content=None,  # Don't include full content in list
                result_metadata=j.result_metadata,
                artifacts=None,  # Don't include artifacts in list
                file_path=j.file_path,
                file_size=j.file_size,
                error=j.error,
                created_at=j.created_at.isoformat() if j.created_at else None,
                started_at=j.started_at.isoformat() if j.started_at else None,
                completed_at=j.completed_at.isoformat() if j.completed_at else None,
            )
            for j in jobs
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{job_id}", response_model=SynthesisJobResponse)
async def get_synthesis_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a synthesis job by ID."""
    result = await db.execute(
        select(SynthesisJob).where(
            SynthesisJob.id == UUID(job_id),
            SynthesisJob.user_id == current_user.id
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Synthesis job not found")

    return SynthesisJobResponse(
        id=str(job.id),
        user_id=str(job.user_id),
        job_type=job.job_type,
        title=job.title,
        description=job.description,
        document_ids=job.document_ids,
        search_query=job.search_query,
        topic=job.topic,
        options=job.options,
        output_format=job.output_format,
        output_style=job.output_style,
        status=job.status,
        progress=job.progress,
        current_stage=job.current_stage,
        result_content=job.result_content,
        result_metadata=job.result_metadata,
        artifacts=job.artifacts,
        file_path=job.file_path,
        file_size=job.file_size,
        error=job.error,
        created_at=job.created_at.isoformat() if job.created_at else None,
        started_at=job.started_at.isoformat() if job.started_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


@router.get("/{job_id}/download")
async def download_synthesis_result(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Download the generated synthesis file."""
    result = await db.execute(
        select(SynthesisJob).where(
            SynthesisJob.id == UUID(job_id),
            SynthesisJob.user_id == current_user.id
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Synthesis job not found")

    if job.status != SynthesisJobStatus.COMPLETED.value:
        raise HTTPException(status_code=400, detail="Job is not completed")

    if not job.file_path:
        raise HTTPException(status_code=400, detail="No file available for download")

    try:
        # Get file from MinIO
        file_obj = storage_service.get_file_stream(job.file_path)

        # Determine content type
        ext = job.file_path.split(".")[-1].lower()
        content_types = {
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "pdf": "application/pdf",
            "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        }
        content_type = content_types.get(ext, "application/octet-stream")

        filename = f"{job.title}.{ext}"

        return StreamingResponse(
            file_obj,
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )

    except Exception as e:
        logger.error(f"Failed to download synthesis file: {e}")
        raise HTTPException(status_code=500, detail="Failed to download file")


@router.delete("/{job_id}")
async def delete_synthesis_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a synthesis job."""
    result = await db.execute(
        select(SynthesisJob).where(
            SynthesisJob.id == UUID(job_id),
            SynthesisJob.user_id == current_user.id
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Synthesis job not found")

    # Delete file if exists
    if job.file_path:
        try:
            await storage_service.delete_file(job.file_path)
        except Exception as e:
            logger.warning(f"Failed to delete synthesis file: {e}")

    await db.delete(job)
    await db.commit()

    return {"success": True, "message": "Job deleted"}


@router.post("/{job_id}/cancel")
async def cancel_synthesis_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Cancel a running synthesis job."""
    result = await db.execute(
        select(SynthesisJob).where(
            SynthesisJob.id == UUID(job_id),
            SynthesisJob.user_id == current_user.id
        )
    )
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Synthesis job not found")

    if job.status in [SynthesisJobStatus.COMPLETED.value, SynthesisJobStatus.FAILED.value, SynthesisJobStatus.CANCELLED.value]:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")

    job.status = SynthesisJobStatus.CANCELLED.value
    await db.commit()

    return {"success": True, "message": "Job cancelled"}


# ==================== Quick Synthesis Endpoints ====================

class QuickSynthesisRequest(BaseModel):
    """Request for quick synthesis without creating a job."""
    document_ids: List[str] = Field(..., description="Document IDs to synthesize")
    topic: Optional[str] = Field(None, description="Focus topic")
    max_length: int = Field(500, description="Maximum word count")


@router.post("/quick/summary")
async def quick_summary(
    request: QuickSynthesisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Generate a quick multi-document summary without creating a job.
    For smaller document sets that can be processed synchronously.
    """
    if len(request.document_ids) > 5:
        raise HTTPException(
            status_code=400,
            detail="Quick summary limited to 5 documents. Use job endpoint for more."
        )

    from app.services.content_generation_service import content_generation_service

    document_ids = [UUID(doc_id) for doc_id in request.document_ids]

    result = await content_generation_service.generate_executive_summary(
        db=db,
        document_ids=document_ids,
        topic=request.topic,
        max_length=request.max_length,
        include_recommendations=True,
        include_metrics=True,
    )

    return result


@router.post("/quick/compare")
async def quick_compare(
    request: QuickSynthesisRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Generate a quick document comparison without creating a job.
    For smaller document sets.
    """
    if len(request.document_ids) > 3:
        raise HTTPException(
            status_code=400,
            detail="Quick comparison limited to 3 documents. Use job endpoint for more."
        )

    if len(request.document_ids) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 documents required for comparison"
        )

    from app.services.content_generation_service import content_generation_service

    document_ids = [UUID(doc_id) for doc_id in request.document_ids]

    result = await content_generation_service.generate_report(
        db=db,
        report_type="analysis",
        document_ids=document_ids,
        title=f"Comparison: {request.topic}" if request.topic else "Document Comparison",
        sections=["Executive Summary", "Document Overview", "Similarities", "Differences", "Conclusions"],
    )

    return result


# ==================== Job Types Info ====================

@router.get("/types/info")
async def get_synthesis_types_info(
    current_user: User = Depends(get_current_user),
):
    """Get information about available synthesis types."""
    return {
        "types": [
            {
                "value": "multi_doc_summary",
                "label": "Multi-Document Summary",
                "description": "Synthesize multiple documents into a comprehensive summary",
                "max_documents": 50,
                "typical_output_length": "500-2000 words",
            },
            {
                "value": "comparative_analysis",
                "label": "Comparative Analysis",
                "description": "Compare and contrast documents to identify similarities and differences",
                "max_documents": 20,
                "typical_output_length": "1000-3000 words",
            },
            {
                "value": "theme_extraction",
                "label": "Theme Extraction",
                "description": "Extract and analyze common themes across documents",
                "max_documents": 50,
                "typical_output_length": "1000-2500 words",
            },
            {
                "value": "knowledge_synthesis",
                "label": "Knowledge Synthesis",
                "description": "Synthesize knowledge from sources into new insights",
                "max_documents": 30,
                "typical_output_length": "1500-3000 words",
            },
            {
                "value": "research_report",
                "label": "Research Report",
                "description": "Generate formal research report from documents",
                "max_documents": 50,
                "typical_output_length": "2000-5000 words",
            },
            {
                "value": "executive_brief",
                "label": "Executive Brief",
                "description": "Create concise executive briefing for leadership",
                "max_documents": 20,
                "typical_output_length": "300-800 words",
            },
            {
                "value": "gap_analysis_hypotheses",
                "label": "Gap Analysis & Hypotheses",
                "description": "Identify research gaps and propose testable hypotheses, novel solution directions, and experiment plans",
                "max_documents": 50,
                "typical_output_length": "1500-4000 words",
            },
        ],
        "output_formats": ["markdown", "docx", "pdf", "pptx"],
        "output_styles": ["professional", "technical", "casual"],
    }
