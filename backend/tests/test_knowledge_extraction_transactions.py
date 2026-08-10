"""Transaction behaviour of KG extraction.

A duplicate relationship is routine — the same fact usually appears in more
than one chunk of a document — so hitting the unique constraint must cost
nothing beyond the duplicate row itself.
"""

import hashlib

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentChunk, DocumentSource
from app.models.knowledge_graph import Entity, Relationship
from app.services.knowledge_extraction import extractor

# Matches the rule extractor's works_for pattern.
SENTENCE = "Alice Smith at Acme Corp leads the compiler team."


async def _make_document(db: AsyncSession) -> Document:
    source = DocumentSource(
        name="kg-transaction-source",
        source_type="file",
        config={},
    )
    db.add(source)
    await db.flush()

    document = Document(
        title="KG Transaction Doc",
        content=SENTENCE,
        content_hash="kg-transaction-doc",
        source_id=source.id,
        source_identifier="kg-transaction-doc",
    )
    db.add(document)
    await db.flush()
    return document


async def _make_chunk(db: AsyncSession, document: Document, index: int):
    chunk = DocumentChunk(
        document_id=document.id,
        content=SENTENCE,
        content_hash=hashlib.sha256(f"{index}".encode()).hexdigest(),
        chunk_index=index,
    )
    db.add(chunk)
    await db.flush()
    return chunk


@pytest.mark.asyncio
async def test_duplicate_relationship_does_not_roll_back_the_transaction(
    db_session: AsyncSession,
):
    """The duplicate is skipped without discarding surrounding work.

    Rolling the whole session back here used to throw away every entity and
    mention extracted for the document since the last commit, and expired the
    caller's objects, so the next attribute read raised MissingGreenlet.
    """
    document = await _make_document(db_session)
    first_chunk = await _make_chunk(db_session, document, 0)
    second_chunk = await _make_chunk(db_session, document, 1)

    mentions, relations = await extractor.index_chunk(db_session, document, first_chunk)
    assert relations == 1, "expected the works_for relation from the first chunk"
    await db_session.commit()

    # Uncommitted work that a full rollback would destroy.
    sentinel = Entity(canonical_name="sentinel-entity", entity_type="other")
    db_session.add(sentinel)
    await db_session.flush()

    # Same sentence, same document: the relationship is a duplicate.
    mentions, relations = await extractor.index_chunk(
        db_session, document, second_chunk
    )
    assert relations == 0, "the duplicate relationship should be skipped"
    assert mentions > 0, "mentions from this chunk are still new rows"

    surviving = (
        await db_session.execute(
            select(Entity).where(Entity.canonical_name == "sentinel-entity")
        )
    ).scalar_one_or_none()
    assert surviving is not None, "the duplicate took the transaction with it"

    # The caller's objects must still be usable, not expired by a rollback.
    assert document.title == "KG Transaction Doc"
    assert second_chunk.chunk_index == 1

    await db_session.commit()
    stored = (
        (
            await db_session.execute(
                select(Relationship).where(Relationship.document_id == document.id)
            )
        )
        .scalars()
        .all()
    )
    assert len(stored) == 1
