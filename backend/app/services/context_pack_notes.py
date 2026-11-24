"""Build a research note from the evidence saved in a retrieval trace."""

from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.research_note import ResearchNote
from app.models.retrieval_trace import RetrievalTrace
from app.services.config_values import parse_uuid

MAX_CITATIONS = 100


def _records(value: Any) -> list[dict]:
    return (
        [row for row in value if isinstance(row, dict)]
        if isinstance(value, list)
        else []
    )


async def build_context_pack_note(
    db: AsyncSession, trace: RetrievalTrace, pack: dict, owner_id: UUID
) -> ResearchNote:
    evidence = [
        (entry, entry.get("text"))
        for entity in _records(pack.get("entities"))
        for entry in _records(entity.get("evidence"))
    ]
    evidence.extend(
        (entry, entry.get("evidence")) for entry in _records(pack.get("relationships"))
    )
    evidence.extend(
        (entry, None) for entry in _records(trace.trace.get("selected_results"))
    )

    citations = {}
    for entry, text in evidence:
        document_id = parse_uuid(entry.get("document_id"))
        if document_id is None:
            continue
        chunk_id = parse_uuid(entry.get("chunk_id"))
        excerpt = " ".join(text.split())[:500] if isinstance(text, str) else ""
        citations.setdefault(
            (document_id, chunk_id), {"excerpt": excerpt, "title": entry.get("title")}
        )
        if len(citations) >= MAX_CITATIONS:
            break

    document_ids = list(dict.fromkeys(document_id for document_id, _ in citations))
    titles = (
        dict(
            (
                await db.execute(
                    select(Document.id, Document.title).where(
                        Document.id.in_(document_ids)
                    )
                )
            ).all()
        )
        if document_ids
        else {}
    )
    keys = {
        document_id: f"S{index}" for index, document_id in enumerate(document_ids, 1)
    }
    trace_titles = {
        document_id: citation["title"]
        for (document_id, _), citation in citations.items()
        if isinstance(citation["title"], str) and citation["title"].strip()
    }
    sources = []
    for document_id in document_ids:
        title = (
            titles.get(document_id)
            or trace_titles.get(document_id)
            or "Source document"
        )
        sources.append(
            {
                "key": keys[document_id],
                "doc_id": str(document_id),
                "title": " ".join(title.split())[:200],
            }
        )

    markdown = [
        f"Retrieval trace: `{trace.id}`",
        f"**Research question**\n\n{trace.query.strip()[:2000]}",
        "## Retrieved context\n\n" + pack["kg_context"].strip()[:8000],
    ]
    source_evidence = []
    if citations:
        evidence_lines = ["## Evidence"]
        for (document_id, chunk_id), citation in citations.items():
            key = keys[document_id]
            excerpt = citation["excerpt"]
            location = f" (chunk `{chunk_id}`)" if chunk_id else ""
            evidence_lines.append(
                f"- {excerpt or 'Retrieved source'} [[{key}]]{location}"
            )
            source_evidence.append(
                {
                    "key": key,
                    "doc_id": str(document_id),
                    "chunk_id": str(chunk_id) if chunk_id else None,
                    "excerpt": excerpt,
                }
            )
        markdown.append("\n".join(evidence_lines))
        markdown.append(
            "\n".join(
                [
                    "## Sources",
                    *(
                        f'- **{source["key"]}** {source["title"]} (document `{source["doc_id"]}`)'
                        for source in sources
                    ),
                ]
            )
        )

    title = " ".join(trace.query.split())[:480]
    return ResearchNote(
        user_id=owner_id,
        title=f"Context pack: {title}" if title else "Context pack",
        content_markdown="\n\n".join(markdown),
        tags=["context-pack"],
        source_document_ids=[str(document_id) for document_id in document_ids] or None,
        attribution={
            "sources": sources,
            "saved_from_context_pack": {
                "trace_id": str(trace.id),
                "citations": source_evidence,
            },
        },
    )
