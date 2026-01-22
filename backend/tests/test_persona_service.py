"""
Tests for persona service helpers.
"""

import pytest
from sqlalchemy import select

from app.models.document import Document, DocumentSource
from app.models.persona import DocumentPersonaDetection
from app.services.persona_service import persona_service
from app.services.document_service import DocumentService
from tests.factories import create_test_document_source, create_test_document


@pytest.mark.asyncio
async def test_assign_owner_from_user(db_session, test_user):
    source = DocumentSource(name="Upload", source_type="file", config={})
    doc = Document(
        title="Owned Doc",
        content="",
        content_hash="abc",
        source=source,
        source_identifier="doc-1",
    )
    db_session.add_all([source, doc])
    await db_session.commit()
    await db_session.refresh(doc)

    await persona_service.assign_document_owner(
        db_session,
        doc,
        user=test_user,
        platform_scope="file-upload",
    )
    await db_session.commit()

    assert doc.owner_persona_id is not None
    result = await db_session.execute(
        select(DocumentPersonaDetection).where(
            DocumentPersonaDetection.document_id == doc.id,
            DocumentPersonaDetection.role == "owner",
        )
    )
    detection = result.scalar_one()
    assert detection.persona_id == doc.owner_persona_id


@pytest.mark.asyncio
async def test_record_sentence_speakers(db_session):
    source = DocumentSource(name="Video", source_type="file", config={})
    doc = Document(
        title="Video",
        content="",
        content_hash="xyz",
        source=source,
        source_identifier="video-1",
    )
    db_session.add_all([source, doc])
    await db_session.commit()
    await db_session.refresh(doc)

    segments = [
        {"start": 0, "end": 5, "text": "Hello world", "speaker": "Speaker 1"},
        {"start": 5, "end": 10, "text": "Another line", "speaker": "Speaker 2"},
    ]
    await persona_service.record_sentence_speakers(
        db_session,
        document=doc,
        sentence_segments=segments,
        base_document_id=doc.id,
    )
    await db_session.commit()

    result = await db_session.execute(
        select(DocumentPersonaDetection).where(
            DocumentPersonaDetection.document_id == doc.id,
            DocumentPersonaDetection.role == "speaker",
        )
    )
    detections = result.scalars().all()
    assert len(detections) == 2
    labels = {det.details.get("text") for det in detections if det.details}
    assert labels == {"Hello world", "Another line"}


@pytest.mark.asyncio
async def test_document_service_filters_personas(db_session):
    source = await create_test_document_source(db_session)
    document = await create_test_document(db_session, source, title="Persona Doc")
    owner = await persona_service.ensure_persona(
        db_session,
        name="Owner Persona",
        platform_id="owner-123",
    )
    document.owner_persona_id = owner.id
    await db_session.commit()

    service = DocumentService()
    docs, total = await service.get_documents(db_session, owner_persona_id=owner.id)
    assert total == 1
    assert docs[0].id == document.id

    await persona_service.record_detection(
        db_session,
        document_id=document.id,
        persona=owner,
        role="speaker",
        detection_type="diarization",
        start_time=0,
        end_time=5,
    )
    await db_session.commit()

    filtered, filtered_total = await service.get_documents(
        db_session,
        persona_id=owner.id,
        persona_role="speaker",
    )
    assert filtered_total == 1
    assert filtered[0].id == document.id
