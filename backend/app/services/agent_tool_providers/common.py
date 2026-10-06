"""Helpers more than one tool provider uses.

Split out of ``agent_tool_dispatch``, which re-exports every name here.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.services.agent_tool_providers.base import AgentToolExecutionContext


async def _load_documents_for_analysis(
    ctx: AgentToolExecutionContext, document_ids: Any, *, max_docs: int = 8
) -> tuple[list[Any], Optional[str]]:
    """Load documents by id for an analysis tool, or return why it cannot run.

    Bounded deliberately: these tools feed document text to a model, and an
    unbounded list would blow the context window rather than fail cleanly.
    """
    from uuid import UUID as _UUID

    from app.services.document_service import DocumentService

    if not isinstance(document_ids, list) or not document_ids:
        return [], "document_ids must be a non-empty list"

    service = DocumentService()
    documents: list[Any] = []
    missing: list[str] = []
    for raw in document_ids[:max_docs]:
        try:
            doc = await service.get_document(_UUID(str(raw)), ctx.db)
        except (TypeError, ValueError):
            missing.append(str(raw))
            continue
        if doc is None:
            missing.append(str(raw))
        else:
            documents.append(doc)

    if not documents:
        return [], f"No documents found for: {', '.join(missing) or document_ids}"
    return documents, None


def _tool_snapshot_context(ctx: Any, tool: str) -> Dict[str, Any]:
    """Attribute a tool's own LLM call to the job phase that made it.

    Without db and this context the snapshot recorder returns early, so these
    calls are absent from a run's export and its captured total falls short of
    the calls the job reports making.
    """
    job = getattr(ctx, "job", None)
    return {
        "job_id": str(getattr(job, "id", "") or "") or None,
        "iteration": int(getattr(job, "iteration", 0) or 0),
        "phase": f"tool:{tool}",
    }
