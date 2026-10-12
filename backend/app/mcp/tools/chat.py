"""
MCP Chat tool for RAG-powered Q&A.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.mcp.auth import MCPAuthContext
from app.models.document import Document
from app.services.config_values import bounded_int
from app.services.llm_service import LLMService, UserLLMSettings, load_user_llm_settings
from app.services.rag_retrieval import record_trace, retrieve_for_answer
from app.services.vector_store import vector_store_service

MAX_CONTEXT_CHUNKS = 20


def _preview(content: str, chars: int = 200) -> str:
    content = content or ""
    return content[:chars] + "..." if len(content) > chars else content


class ChatTool:
    """
    Chat tool for MCP.

    Provides RAG-powered question answering over the knowledge base.
    """

    name = "chat"
    description = "Ask questions and get AI-powered answers based on the knowledge base"

    input_schema = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question to ask"},
            "source_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional: Limit search to specific source IDs",
            },
            "document_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional: Limit search to specific document IDs",
            },
            "max_context_chunks": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
                "description": "Maximum number of context chunks to use",
            },
            "include_sources": {
                "type": "boolean",
                "default": True,
                "description": "Include source references in response",
            },
        },
        "required": ["question"],
    }

    def __init__(self):
        self.llm_service = LLMService()

    async def execute(
        self,
        auth: MCPAuthContext,
        db: AsyncSession,
        question: str,
        source_ids: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
        max_context_chunks: int = 5,
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG over the knowledge base.

        Args:
            auth: MCP authentication context
            db: Database session
            question: The question to answer
            source_ids: Optional source ID filter
            document_ids: Optional document ID filter
            max_context_chunks: Maximum context chunks to retrieve
            include_sources: Whether to include source references

        Returns:
            Answer with optional source references
        """
        auth.require_scope("chat")

        logger.info(
            f"MCP chat: question='{question[:50]}...', user={auth.user.username}"
        )

        max_context_chunks = bounded_int(max_context_chunks, 5, 1, MAX_CONTEXT_CHUNKS)

        try:
            # Load user LLM settings
            user_settings = await self._load_user_settings(auth.user_id, db)

            # Resolve the filter first and hand it to the store. It used to be
            # applied to a window of the best chunks from *anywhere*, so a
            # question about one source was answered "nothing relevant"
            # whenever other sources matched better; and a filter matching no
            # document was treated as no filter at all.
            filtering = bool(source_ids or document_ids)
            accessible_doc_ids = await self._get_accessible_doc_ids(
                auth, db, source_ids, document_ids
            )
            doc_id_set = (
                set(str(doc_id) for doc_id in accessible_doc_ids or [])
                if filtering
                else None
            )

            if doc_id_set is not None and not doc_id_set:
                # A filter that matches no document: nothing to answer from.
                return {
                    "answer": "I couldn't find any relevant information in the knowledge base to answer your question.",
                    "question": question,
                    "sources": [],
                    "context_found": False,
                }

            # The chat page's own retrieval (services/rag_retrieval.py). This
            # tool used to search once and use whatever came back.
            retrieved = await retrieve_for_answer(
                question,
                db=db,
                vector_store=vector_store_service,
                llm_service=self.llm_service,
                user_settings=user_settings,
                document_ids=sorted(doc_id_set) if doc_id_set else None,
                max_results=max_context_chunks,
            )
            results = retrieved.results

            if not results:
                return {
                    "answer": "I couldn't find any relevant information in the knowledge base to answer your question.",
                    "question": question,
                    "sources": [],
                    "context_found": False,
                }

            answer = await self.llm_service.generate_response(
                query=question,
                context=retrieved.context,
                kg_context=retrieved.kg_context,
                temperature=0.3,
                max_tokens=1500,
                user_settings=user_settings,
            )

            response = {
                "answer": answer.strip(),
                "question": question,
                "context_found": True,
                "chunks_used": len(results),
            }

            try:
                trace = await record_trace(
                    db,
                    retrieved,
                    query=question,
                    user_id=auth.user_id,
                    vector_store=vector_store_service,
                    trace_type="mcp",
                )
                response["retrieval_trace_id"] = str(trace.id)
            except Exception as exc:  # noqa: BLE001 - the answer stands
                logger.warning(f"MCP chat: trace not recorded: {exc}")

            if include_sources:
                response["sources"] = [
                    {
                        "document_id": str(
                            (r.get("metadata") or {}).get("document_id") or ""
                        ),
                        "title": (r.get("metadata") or {}).get("title", "Unknown"),
                        "score": r.get("score", 0),
                        "chunk_preview": _preview(
                            r.get("content", r.get("page_content", ""))
                        ),
                    }
                    for r in results
                ]

            return response

        except Exception as e:
            logger.error(f"MCP chat error: {e}")
            return {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "question": question,
                "error": str(e),
            }

    async def _load_user_settings(
        self, user_id: UUID, db: AsyncSession
    ) -> Optional[UserLLMSettings]:
        """Load user's LLM preferences."""
        return await load_user_llm_settings(db, user_id)

    async def _get_accessible_doc_ids(
        self,
        auth: MCPAuthContext,
        db: AsyncSession,
        source_ids: Optional[List[str]] = None,
        document_ids: Optional[List[str]] = None,
    ) -> Optional[List[UUID]]:
        """Get document IDs accessible to user based on filters."""
        # If specific document IDs are provided, use those
        if document_ids:
            return [UUID(did) for did in document_ids]

        # If source IDs are provided, get documents from those sources
        if source_ids:
            source_uuids = [UUID(sid) for sid in source_ids]
            result = await db.execute(
                select(Document.id).where(Document.source_id.in_(source_uuids))
            )
            return [row[0] for row in result.fetchall()]

        # No filter - return None to indicate all documents accessible
        return None
