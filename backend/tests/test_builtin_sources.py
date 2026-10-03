"""The built-in document sources DocumentService creates on first use."""

import pytest
from sqlalchemy import func, select

from app.models.document import DocumentSource
from app.services.document_service import DocumentService

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "method,name,source_type",
    [
        ("_get_or_create_upload_source", "File Upload", "file"),
        ("_get_or_create_agent_notes_source", "Agent Notes", "file"),
        ("_get_or_create_latex_projects_source", "LaTeX Projects", "file"),
        ("_get_or_create_url_ingest_source", "URL Ingest", "web"),
    ],
)
async def test_created_once_then_reused(db_session, method, name, source_type):
    service = DocumentService.__new__(DocumentService)

    first = await getattr(service, method)(db_session)
    second = await getattr(service, method)(db_session)

    assert first.id == second.id
    assert (first.name, first.source_type) == (name, source_type)
    count = await db_session.scalar(
        select(func.count())
        .select_from(DocumentSource)
        .where(DocumentSource.name == name)
    )
    assert count == 1


async def test_url_ingest_stays_inactive_so_no_sync_crawls_it(db_session):
    service = DocumentService.__new__(DocumentService)
    source = await service._get_or_create_url_ingest_source(db_session)
    assert source.is_active is False

    source.is_active = True
    await db_session.commit()

    again = await service._get_or_create_url_ingest_source(db_session)
    assert again.is_active is False


async def test_other_builtin_sources_are_active(db_session):
    service = DocumentService.__new__(DocumentService)
    assert (await service._get_or_create_upload_source(db_session)).is_active is True
