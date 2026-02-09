"""
Celery tasks for LaTeX Studio (async compilation).

These tasks are meant to run in a dedicated worker/container with strict resource limits.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict
from uuid import UUID

from loguru import logger

from app.core.celery_latex import celery_app
from app.core.config import settings
from app.core.database import create_celery_session
from app.models.latex_compile_job import LatexCompileJob
from app.models.latex_project import LatexProject
from app.models.latex_project_file import LatexProjectFile
from app.services.latex_compiler_service import LatexSafetyError, latex_compiler_service
from app.services.storage_service import storage_service
from sqlalchemy import select


@celery_app.task(bind=True, name="app.tasks.latex_tasks.compile_latex_project_job")
def compile_latex_project_job(self, job_id: str) -> Dict[str, Any]:
    return asyncio.run(_async_compile_latex_project_job(self, job_id))


async def _async_compile_latex_project_job(task, job_id: str) -> Dict[str, Any]:
    async with create_celery_session()() as db:
        job = await db.get(LatexCompileJob, UUID(job_id))
        if not job:
            raise ValueError(f"LatexCompileJob {job_id} not found")

        if job.status not in ("queued", "running"):
            return {"job_id": job_id, "status": job.status}

        project = await db.get(LatexProject, job.project_id) if job.project_id else None
        if not project:
            job.status = "failed"
            job.log = "LaTeX project not found."
            job.finished_at = datetime.utcnow()
            await db.commit()
            return {"job_id": job_id, "status": job.status}

        # Mark running
        job.status = "running"
        job.started_at = job.started_at or datetime.utcnow()
        await db.commit()

        task.update_state(state="PROGRESS", meta={"status": "Fetching project files"})

        additional_files: Dict[str, bytes] = {}
        try:
            files_result = await db.execute(select(LatexProjectFile).where(LatexProjectFile.project_id == project.id))
            for f in files_result.scalars().all():
                name = (f.filename or "").strip()
                if not name or "/" in name or "\\" in name:
                    continue
                try:
                    additional_files[name] = await storage_service.get_file_content(f.file_path)
                except Exception:
                    continue
        except Exception:
            additional_files = {}

        task.update_state(state="PROGRESS", meta={"status": "Compiling LaTeX"})

        try:
            result = await asyncio.to_thread(
                latex_compiler_service.compile_to_pdf,
                tex_source=project.tex_source,
                timeout_seconds=int(settings.LATEX_COMPILER_TIMEOUT_SECONDS),
                max_source_chars=int(settings.LATEX_COMPILER_MAX_SOURCE_CHARS),
                safe_mode=bool(job.safe_mode),
                preferred_engine=job.preferred_engine,
                additional_files=additional_files or None,
            )
            job.engine = result.engine
            job.log = result.log
            job.violations = list(result.violations or [])

            project.last_compile_engine = result.engine
            project.last_compile_log = result.log
            project.last_compiled_at = datetime.utcnow()

            if not result.success or not result.pdf_bytes:
                job.status = "failed"
                job.finished_at = datetime.utcnow()
                await db.commit()
                return {"job_id": job_id, "status": job.status, "success": False}

            task.update_state(state="PROGRESS", meta={"status": "Uploading PDF"})
            pdf_path = await storage_service.upload_file(
                document_id=project.id,
                filename="paper.pdf",
                content=result.pdf_bytes,
                content_type="application/pdf",
            )
            job.pdf_file_path = pdf_path
            job.status = "succeeded"
            job.finished_at = datetime.utcnow()

            project.pdf_file_path = pdf_path
            await db.commit()

            return {"job_id": job_id, "status": job.status, "success": True, "engine": result.engine}

        except LatexSafetyError as exc:
            job.engine = None
            job.status = "failed"
            job.log = str(exc)
            job.violations = list(getattr(exc, "violations", []) or [])
            job.finished_at = datetime.utcnow()
            project.last_compile_log = job.log
            project.last_compiled_at = datetime.utcnow()
            await db.commit()
            return {"job_id": job_id, "status": job.status, "success": False}
        except Exception as exc:
            logger.error(f"LaTeX compile job failed ({job_id}): {exc}")
            job.status = "failed"
            job.log = "Compilation failed due to a server error."
            job.finished_at = datetime.utcnow()
            project.last_compile_log = job.log
            project.last_compiled_at = datetime.utcnow()
            await db.commit()
            return {"job_id": job_id, "status": job.status, "success": False}
