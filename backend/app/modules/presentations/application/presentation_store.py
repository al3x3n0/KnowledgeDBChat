"""Reading and writing a user's presentation jobs and templates.

This was inline in ``api/endpoints/presentations.py``, which now only maps
these results to HTTP. A refusal is raised as ``Refused(kind, detail)``;
nothing here knows a status code.

Two rules about who sees a template live here and are deliberately not the
same. To *use* or *read* a template it must be active and yours, public or a
system one. To *change* or *delete* one it must be yours, active or not, so a
template its owner switched off can be switched back on.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from typing import Any, Optional, Sequence
from uuid import UUID, uuid4

from loguru import logger
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.presentation import PresentationJob, PresentationTemplate
from app.services import arxiv_instant_ingest
from app.services.connectors import arxiv_connector
from app.services.storage_service import StorageService

PPTX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.presentationml.presentation"
)
MAX_TEMPLATE_BYTES = 50 * 1024 * 1024
FINISHED_STATUSES = ("completed", "failed", "cancelled")

#: The kinds of refusal. The endpoint maps each to a status code.
KINDS = (
    "invalid",  # the request does not make sense as asked
    "forbidden",  # this person may not do this
    "not_found",  # nothing of that id exists for them
    "unavailable",  # storage did not do what was asked
)


class Refused(Exception):
    def __init__(self, kind: str, detail: str):
        if kind not in KINDS:
            raise ValueError(f"unknown refusal kind: {kind}")
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


# -- jobs ---------------------------------------------------------------------


async def usable_template(
    db: AsyncSession, template_id: UUID, user_id: UUID
) -> PresentationTemplate:
    """The active template a new job names, if this person may use it."""
    result = await db.execute(
        select(PresentationTemplate).where(
            PresentationTemplate.id == template_id,
            PresentationTemplate.is_active.is_(True),
        )
    )
    template = result.scalar_one_or_none()
    if not template:
        raise Refused("not_found", "Template not found")
    if not (template.is_system or template.is_public or template.user_id == user_id):
        raise Refused("forbidden", "You don't have access to this template")
    return template


async def create_job(db: AsyncSession, user_id: UUID, **fields: Any) -> PresentationJob:
    """Store a pending job. Queuing its task is the caller's, after this."""
    job = PresentationJob(user_id=user_id, status="pending", progress=0, **fields)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def ingest_arxiv_sources(
    db: AsyncSession, user_id: UUID, topic: str, max_papers: int
) -> list[str]:
    """Search arXiv for a topic and ingest what it finds; the document ids.

    Never raises: a search that fails yields no sources, and the presentation
    is made from the knowledge base alone. ``user_id`` is taken as a value
    because a rollback here expires every object in the session, the caller
    among them.
    """
    document_ids: list[str] = []
    connector = arxiv_connector.ArxivConnector()
    try:
        await connector.initialize(
            {"queries": [topic], "max_results": max_papers, "sort_by": "relevance"}
        )
        docs = await connector.list_documents()
        logger.info(f"Found {len(docs)} arXiv papers for topic: {topic}")

        for doc_info in docs[:max_papers]:
            arxiv_id = doc_info["identifier"]
            # One paper at a time: a paper that cannot be fetched or indexed
            # is left out, and the ones after it still go in. One `try`
            # around the whole loop lost every later paper to the first
            # failure.
            try:
                paper = await arxiv_instant_ingest.ingest_paper(
                    db,
                    connector,
                    arxiv_id,
                    doc_info,
                    user_id=user_id,
                    marker={"research_presentation": True},
                )
                document_ids.append(str(paper.document.id))
            except Exception as e:
                logger.warning(f"Could not ingest arXiv paper {arxiv_id}: {e}")
                # Whatever the failure left half-written must not ride along
                # into the next paper's commit, or the job's.
                await db.rollback()
    except Exception as e:
        logger.warning(f"Failed to search arXiv for '{topic}': {e}")
        await db.rollback()
    finally:
        await connector.cleanup()
    return document_ids


async def list_jobs(
    db: AsyncSession,
    user_id: UUID,
    *,
    limit: int,
    offset: int,
    status: Optional[str] = None,
) -> Sequence[PresentationJob]:
    """A page of this person's jobs, newest first."""
    query = (
        select(PresentationJob)
        .where(PresentationJob.user_id == user_id)
        .options(selectinload(PresentationJob.template))
    )
    if status:
        query = query.where(PresentationJob.status == status)
    query = query.order_by(PresentationJob.created_at.desc())
    result = await db.execute(query.offset(offset).limit(limit))
    return result.scalars().all()


async def get_job(db: AsyncSession, job_id: UUID, user_id: UUID) -> PresentationJob:
    """A job, if it belongs to this person."""
    result = await db.execute(
        select(PresentationJob)
        .options(selectinload(PresentationJob.template))
        .where(PresentationJob.id == job_id, PresentationJob.user_id == user_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise Refused("not_found", "Presentation job not found")
    return job


async def read_deck(job: PresentationJob) -> tuple[str, bytes]:
    """The finished deck's download name and bytes."""
    if job.status != "completed":
        raise Refused(
            "invalid", f"Presentation is not ready. Current status: {job.status}"
        )
    if not job.file_path:
        raise Refused("not_found", "Presentation file not found")

    try:
        content = await StorageService().get_file_content(job.file_path)
    except FileNotFoundError:
        logger.error(
            f"Presentation file not found in storage for job {job.id}: {job.file_path}"
        )
        raise Refused("not_found", "Presentation file not found in storage")
    except Exception as e:
        logger.error(f"Failed to download presentation for job {job.id}: {e}")
        raise Refused("unavailable", "Failed to download presentation")

    filename = job.file_path.split("/")[-1]
    if not filename.endswith(".pptx"):
        filename = f"{job.title or 'presentation'}.pptx"
    return filename, content


async def _remove_file(path: str, what: str) -> None:
    """Best effort: a file that stays behind does not keep its row alive."""
    try:
        # It answers False instead of raising when the file is still there.
        if not await StorageService().delete_file(path):
            logger.warning(f"{what} file {path} was not removed")
    except Exception as e:
        logger.warning(f"Failed to delete {what.lower()} file {path}: {e}")


async def delete_job(db: AsyncSession, job: PresentationJob) -> None:
    if job.file_path:
        await _remove_file(job.file_path, "Presentation")
    await db.delete(job)
    await db.commit()


async def cancel_job(db: AsyncSession, job: PresentationJob) -> PresentationJob:
    if job.status in FINISHED_STATUSES:
        raise Refused("invalid", f"Cannot cancel job with status: {job.status}")
    job.status = "cancelled"
    job.error = "Cancelled by user"
    await db.commit()
    await db.refresh(job)
    return job


# -- templates ----------------------------------------------------------------


async def list_templates(
    db: AsyncSession,
    user_id: UUID,
    *,
    include_system: bool = True,
    include_public: bool = True,
) -> Sequence[PresentationTemplate]:
    """This person's active templates, plus the shared ones asked for."""
    conditions = [PresentationTemplate.user_id == user_id]
    if include_system:
        conditions.append(PresentationTemplate.is_system.is_(True))
    if include_public:
        conditions.append(PresentationTemplate.is_public.is_(True))

    result = await db.execute(
        select(PresentationTemplate)
        .where(or_(*conditions), PresentationTemplate.is_active.is_(True))
        .order_by(PresentationTemplate.is_system.desc(), PresentationTemplate.name)
    )
    return result.scalars().all()


async def create_template(
    db: AsyncSession, user_id: UUID, **fields: Any
) -> PresentationTemplate:
    template = PresentationTemplate(user_id=user_id, is_system=False, **fields)
    db.add(template)
    await db.commit()
    await db.refresh(template)
    return template


async def _check_deck(content: bytes) -> None:
    """Refuse bytes python-pptx cannot open as a presentation."""
    from pptx import Presentation as PptxPresentation
    from pptx.exc import PackageNotFoundError

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        try:
            # Parsing a 50 MB deck is seconds of CPU; not on the event loop.
            prs = await asyncio.to_thread(PptxPresentation, temp_path)
            logger.info(
                f"Valid PPTX uploaded: {len(prs.slides)} slides, "
                f"{len(prs.slide_layouts)} layouts"
            )
        except PackageNotFoundError:
            raise Refused("invalid", "Invalid or corrupted PPTX file")
        except Exception as e:
            logger.error(f"Error parsing PPTX: {e}")
            raise Refused("invalid", f"Error parsing PPTX file: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


async def store_uploaded_template(
    db: AsyncSession,
    user_id: UUID,
    *,
    filename: Optional[str],
    content: bytes,
    name: str,
    description: Optional[str],
    is_public: bool,
) -> PresentationTemplate:
    """Check an uploaded deck, store it, and record it as a template."""
    if not filename:
        raise Refused("invalid", "Filename is required")
    if not filename.lower().endswith(".pptx"):
        raise Refused("invalid", "Only .pptx files are allowed")
    if len(content) > MAX_TEMPLATE_BYTES:
        raise Refused("invalid", "File size exceeds maximum allowed (50MB)")

    await _check_deck(content)

    # The id comes first: the file is stored under it.
    template_id = uuid4()
    safe_filename = re.sub(r"[^\w\-.]", "_", filename)
    object_path = f"templates/{template_id}/{safe_filename}"

    try:
        # To the path recorded on the template below. This called
        # `upload_file` with a `prefix` argument it does not take, so every
        # upload raised TypeError here and answered "Failed to store template
        # file": no PPTX template was ever stored.
        await StorageService().upload_to_path(
            object_path=object_path, content=content, content_type=PPTX_MEDIA_TYPE
        )
        logger.info(f"Uploaded template file to {object_path}")
    except Exception as e:
        logger.error(f"Failed to upload template to MinIO: {e}")
        raise Refused("unavailable", "Failed to store template file")

    template = PresentationTemplate(
        id=template_id,
        user_id=user_id,
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
    return template


async def get_accessible_template(
    db: AsyncSession, template_id: UUID, user_id: UUID
) -> PresentationTemplate:
    """An active template this person may read: their own, public or system."""
    result = await db.execute(
        select(PresentationTemplate).where(
            PresentationTemplate.id == template_id,
            PresentationTemplate.is_active.is_(True),
            or_(
                PresentationTemplate.user_id == user_id,
                PresentationTemplate.is_system.is_(True),
                PresentationTemplate.is_public.is_(True),
            ),
        )
    )
    template = result.scalar_one_or_none()
    if not template:
        raise Refused("not_found", "Template not found")
    return template


async def _changeable_template(
    db: AsyncSession, template_id: UUID, user_id: UUID, verb: str
) -> PresentationTemplate:
    result = await db.execute(
        select(PresentationTemplate).where(
            PresentationTemplate.id == template_id,
            PresentationTemplate.user_id == user_id,
        )
    )
    template = result.scalar_one_or_none()
    if not template:
        raise Refused("not_found", "Template not found or you don't own it")
    if template.is_system:
        raise Refused("forbidden", f"Cannot {verb} system templates")
    return template


async def update_template(
    db: AsyncSession, template_id: UUID, user_id: UUID, changes: dict[str, Any]
) -> PresentationTemplate:
    """Apply the fields that were given; ``None`` means "leave it"."""
    template = await _changeable_template(db, template_id, user_id, "modify")
    for field in ("name", "description", "theme_config", "is_public", "is_active"):
        if changes.get(field) is not None:
            setattr(template, field, changes[field])
    await db.commit()
    await db.refresh(template)
    return template


async def delete_template(db: AsyncSession, template_id: UUID, user_id: UUID) -> None:
    template = await _changeable_template(db, template_id, user_id, "delete")
    if template.file_path:
        await _remove_file(template.file_path, "Template")
    await db.delete(template)
    await db.commit()


__all__ = [
    "KINDS",
    "PPTX_MEDIA_TYPE",
    "Refused",
    "cancel_job",
    "create_job",
    "create_template",
    "delete_job",
    "delete_template",
    "get_accessible_template",
    "get_job",
    "ingest_arxiv_sources",
    "list_jobs",
    "list_templates",
    "read_deck",
    "store_uploaded_template",
    "update_template",
    "usable_template",
]
