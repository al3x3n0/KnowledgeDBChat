"""Finding documents by their tags.

`Document.tags` is a plain JSON column, which has no portable "contains" or
"overlaps" operator. The chat tool called `.overlap()` on it, which does not
exist, so `search_by_tags` had never returned a document; its match-all branch
compiled to a substring match on the serialised list. The autonomous tool did
the matching in Python and worked. This is that implementation, shared.
"""

from __future__ import annotations

from typing import Any, List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document

PAGE_SIZE = 500


def clean_tags(value: Any) -> List[str]:
    """Non-empty stripped strings, first occurrence kept. A bare string is
    one tag, not a sequence of letters."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return list(
        dict.fromkeys(t.strip() for t in value if isinstance(t, str) and t.strip())
    )


def document_tags(document: Document) -> List[str]:
    tags = document.tags
    if isinstance(tags, str):
        return [tags] if tags.strip() else []
    return [t for t in tags if isinstance(t, str)] if isinstance(tags, list) else []


async def documents_with_tags(
    db: AsyncSession,
    tags: List[str],
    *,
    match_all: bool,
    limit: int,
    newest_by: str = "updated_at",
) -> List[Document]:
    """Newest first (by `newest_by`), a page at a time, until `limit` match."""
    order_column = getattr(Document, newest_by)
    # Tags are compared without regard to case, here and in the counters:
    # "ML" and "ml" are one tag to the person who typed them.
    wanted = {tag.lower() for tag in tags}
    matched: List[Document] = []
    if not wanted or limit < 1:
        return matched
    offset = 0
    while len(matched) < limit:
        rows = (
            (
                await db.execute(
                    select(Document)
                    .where(Document.tags.isnot(None))
                    .order_by(order_column.desc(), Document.id)
                    .offset(offset)
                    .limit(PAGE_SIZE)
                )
            )
            .scalars()
            .all()
        )
        for document in rows:
            have = {tag.lower() for tag in document_tags(document)}
            if wanted.issubset(have) if match_all else wanted & have:
                matched.append(document)
        if len(rows) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return matched[:limit]
