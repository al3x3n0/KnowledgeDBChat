"""HTTP boundary for exporting autonomous-job results."""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from loguru import logger
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.endpoints.auth import get_current_active_user
from app.core.database import get_db
from app.models.agent_job import AgentJob
from app.models.user import User

ExporterFactory = Callable[..., Any]
UserSettingsLoader = Callable[..., Awaitable[Any]]

SUPPORTED_FORMATS = frozenset({"docx", "pdf", "pptx"})
SUPPORTED_STYLES = frozenset({"professional", "technical", "casual"})
CONTENT_TYPES = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pdf": "application/pdf",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


@dataclass(frozen=True)
class JobExportApi:
    router: APIRouter
    export_job_results: Callable[..., Any]
    export_job_transcript: Callable[..., Any]


def _default_exporter_factory(*, style: str) -> Any:
    from app.services.job_results_exporter import JobResultsExporter

    return JobResultsExporter(style=style)


async def _load_user_settings(*, db: AsyncSession, user_id: UUID) -> Any:
    try:
        from app.models.memory import UserPreferences
        from app.services.llm_service import UserLLMSettings

        result = await db.execute(
            select(UserPreferences).where(UserPreferences.user_id == user_id)
        )
        preferences = result.scalar_one_or_none()
        return UserLLMSettings.from_preferences(preferences) if preferences else None
    except Exception:
        return None


def build_job_export_api(
    *,
    exporter_factory: ExporterFactory = _default_exporter_factory,
    load_user_settings: UserSettingsLoader = _load_user_settings,
) -> JobExportApi:
    """Build the job-results export route with generation dependencies injected."""
    router = APIRouter()

    @router.get("/{job_id}/export")
    async def export_job_results(
        job_id: UUID,
        format: str = Query("docx", description="Export format: docx, pdf, or pptx"),
        style: str = Query(
            "professional",
            description="Visual style: professional, technical, or casual",
        ),
        include_log: bool = Query(False, description="Include execution log in export"),
        include_metadata: bool = Query(
            True, description="Include job metadata in export"
        ),
        enhance: bool = Query(
            False,
            description="Use LLM to generate executive summary and insights",
        ),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ) -> Response:
        """Export job results as a document or presentation attachment."""
        if format not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported format: {format}. Use docx, pdf, or pptx.",
            )
        if style not in SUPPORTED_STYLES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Unsupported style: {style}. "
                    "Use professional, technical, or casual."
                ),
            )

        result = await db.execute(
            select(AgentJob).where(
                and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id)
            )
        )
        job = result.scalar_one_or_none()
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent job not found",
            )

        try:
            exporter = exporter_factory(style=style)
            if enhance:
                user_settings = await load_user_settings(
                    db=db,
                    user_id=current_user.id,
                )
                file_bytes = await exporter.export_enhanced(
                    job=job,
                    format=format,
                    include_log=include_log,
                    include_metadata=include_metadata,
                    user_id=current_user.id,
                    user_settings=user_settings,
                )
            else:
                file_bytes = exporter.export(
                    job=job,
                    format=format,
                    include_log=include_log,
                    include_metadata=include_metadata,
                )
        except Exception as error:
            logger.error(f"Failed to export job {job_id}: {error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Export failed: {str(error)}",
            )

        filename = f"{_safe_job_name(job.name)}_report.{format}"
        logger.info(f"Exported job {job_id} as {format} for user {current_user.id}")
        return Response(
            content=file_bytes,
            media_type=CONTENT_TYPES[format],
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @router.get("/{job_id}/export/transcript")
    async def export_job_transcript(
        job_id: UUID,
        include_prompts: bool = Query(
            False,
            description=(
                "Include verbatim prompt and reply text. Only available for runs "
                "made with LLM_CALL_SNAPSHOT_ENABLED."
            ),
        ),
        db: AsyncSession = Depends(get_db),
        current_user: User = Depends(get_current_active_user),
    ) -> Dict[str, Any]:
        """Export what a job decided, why, and what it said, as JSON.

        The DOCX export above is a report for a reader; this is the record of
        the run itself, for inspecting or replaying an agent's behaviour.
        """
        from app.models.agent_job import AgentJobCheckpoint
        from app.models.llm_call_snapshot import LLMCallSnapshot
        from app.services.agent_job_transcript_service import build_job_transcript

        result = await db.execute(
            select(AgentJob).where(
                and_(AgentJob.id == job_id, AgentJob.user_id == current_user.id)
            )
        )
        job = result.scalar_one_or_none()
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent job not found",
            )

        checkpoints = (
            (
                await db.execute(
                    select(AgentJobCheckpoint)
                    .where(AgentJobCheckpoint.job_id == job_id)
                    .order_by(
                        AgentJobCheckpoint.iteration,
                        AgentJobCheckpoint.created_at,
                    )
                )
            )
            .scalars()
            .all()
        )
        snapshots = (
            (
                await db.execute(
                    select(LLMCallSnapshot)
                    .where(LLMCallSnapshot.job_id == job_id)
                    .order_by(LLMCallSnapshot.created_at)
                )
            )
            .scalars()
            .all()
        )

        return build_job_transcript(
            job,
            checkpoints,
            snapshots,
            include_prompts=include_prompts,
        )

    return JobExportApi(
        router=router,
        export_job_results=export_job_results,
        export_job_transcript=export_job_transcript,
    )


def _safe_job_name(name: Any) -> str:
    safe_name = "".join(
        character
        for character in str(name or "")
        if character.isalnum() or character in (" ", "-", "_")
    ).strip()
    return safe_name[:50] or "agent_job"
