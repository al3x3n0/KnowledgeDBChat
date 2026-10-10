"""Put one arXiv paper into the knowledge base now, inside the caller's request.

The ordinary route to a paper is a document source and a Celery task. Two
places want the paper before they answer -- "ingest this so I can ask about
it" and "make a presentation from research on a topic" -- and each carried its
own copy of an inline ingest. Neither copy could store a new paper:

- the document was built without ``source_id``, which the column requires;
- ``TextProcessor.split_text`` is a coroutine function and was called without
  ``await``, so what was iterated was a coroutine;
- the chunks were built without ``content_hash``, which that column requires.

The first is refused by the database, so the other two were never reached.
One route answered 500. The other caught the error, then failed on the
session the error had poisoned.

Here the document is filed under a built-in source and handed to
``DocumentService._process_document_async``, the same pipeline an upload goes
through, so there is one place that knows what a chunk needs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentChunk, DocumentSource
from app.services.document_service import DocumentService

SOURCE_NAME = "ArXiv Instant"


class PaperNotIndexed(RuntimeError):
    """The paper was fetched and stored but could not be made searchable."""


@dataclass
class IngestedPaper:
    document: Document
    #: Chunks made by this call; 0 when the paper was already there.
    chunks_created: int
    already_indexed: bool
    content: str


async def _source(db: AsyncSession, documents: DocumentService) -> DocumentSource:
    # Inactive: it has no queries, so a sync sweep has nothing to do with it,
    # and being active is what makes the sweeps look.
    return await documents._get_or_create_builtin_source(
        db,
        SOURCE_NAME,
        source_type="arxiv",
        config={
            "type": "arxiv_instant",
            "description": "Papers ingested one at a time, on request",
        },
        active=False,
    )


async def existing_paper(db: AsyncSession, arxiv_id: str) -> Optional[Document]:
    """The stored document for this paper, a searchable one if there is one.

    More than one row can carry an arXiv id -- an ordinary arXiv source may
    have ingested the same paper -- so this never assumes there is exactly one.
    """
    rows = (
        (
            await db.execute(
                select(Document)
                .where(Document.source_identifier == arxiv_id)
                .order_by(Document.is_processed.desc(), Document.created_at)
            )
        )
        .scalars()
        .all()
    )
    return rows[0] if rows else None


async def ingest_paper(
    db: AsyncSession,
    connector: Any,
    arxiv_id: str,
    info: Dict[str, Any],
    *,
    user_id: Optional[UUID] = None,
    marker: Optional[Dict[str, Any]] = None,
    document_service: Optional[DocumentService] = None,
) -> IngestedPaper:
    """Fetch `arxiv_id` through `connector` and make it searchable.

    `info` is the connector's listing row for the paper (title, url, author).
    `marker` is merged into the document's metadata, to say who asked.
    Raises `PaperNotIndexed` when the paper is stored but not searchable; the
    row is kept, marked unprocessed with the reason, and a later call retries.
    """
    documents = document_service or DocumentService()
    document = await existing_paper(db, arxiv_id)
    if document is not None and document.is_processed:
        return IngestedPaper(document, 0, True, document.content or "")

    content = await connector.get_document_content(arxiv_id)
    metadata = await connector.get_document_metadata(arxiv_id)
    extra = {
        "arxiv": True,
        "authors": metadata.get("authors", []),
        "categories": metadata.get("categories", []),
        "primary_category": metadata.get("primary_category"),
        "doi": metadata.get("doi"),
        **(marker or {}),
    }
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    if document is None:
        source = await _source(db, documents)
        document = Document(
            title=info["title"],
            content=content,
            content_hash=content_hash,
            source_id=source.id,
            source_identifier=arxiv_id,
            url=info.get("url"),
            author=info.get("author"),
            extra_metadata=extra,
        )
        db.add(document)
    else:
        # Stored before and never indexed: start again from what arXiv has.
        document.title = info["title"]
        document.content = content
        document.content_hash = content_hash
        document.url = info.get("url")
        document.author = info.get("author")
        document.extra_metadata = extra
        await db.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == document.id)
        )
    await db.commit()
    await db.refresh(document)

    # It records a failure on the document instead of raising.
    await documents._process_document_async(document, db, user_id=user_id)
    if not document.is_processed:
        raise PaperNotIndexed(
            str(document.processing_error or "the paper could not be indexed")
        )

    chunks = int(
        (
            await db.execute(
                select(func.count(DocumentChunk.id)).where(
                    DocumentChunk.document_id == document.id
                )
            )
        ).scalar()
        or 0
    )
    return IngestedPaper(document, chunks, False, content)


__all__ = [
    "SOURCE_NAME",
    "IngestedPaper",
    "PaperNotIndexed",
    "existing_paper",
    "ingest_paper",
]
