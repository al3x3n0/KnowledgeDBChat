"""
MCP Chat tool for RAG-powered Q&A.
"""

from typing import List, Optional, Any, Dict
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.mcp.auth import MCPAuthContext
from app.models.document import Document
from app.models.memory import UserPreferences
from app.services.chat_service import ChatService
from app.services.llm_service import LLMService, UserLLMSettings
from app.services.vector_store import vector_store_service


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
            "question": {
                "type": "string",
                "description": "The question to ask"
            },
            "source_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional: Limit search to specific source IDs"
            },
            "document_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional: Limit search to specific document IDs"
            },
            "max_context_chunks": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
                "description": "Maximum number of context chunks to use"
            },
            "include_sources": {
                "type": "boolean",
                "default": True,
                "description": "Include source references in response"
            }
        },
        "required": ["question"]
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

        logger.info(f"MCP chat: question='{question[:50]}...', user={auth.user.username}")

        try:
            # Load user LLM settings
            user_settings = await self._load_user_settings(auth.user_id, db)

            # Initialize vector store
            await vector_store_service.initialize(background=True)

            # Search for relevant context
            raw_results = await vector_store_service.search(
                query=question,
                limit=max_context_chunks * 2  # Get more to filter
            )

            # Filter results by accessible documents
            accessible_doc_ids = await self._get_accessible_doc_ids(
                auth, db, source_ids, document_ids
            )
            doc_id_set = set(str(doc_id) for doc_id in accessible_doc_ids) if accessible_doc_ids else None

            filtered_results = []
            sources_used = []

            for result in raw_results:
                metadata = result.get("metadata", {})
                result_doc_id = metadata.get("document_id", "")

                # If we have a filter, apply it
                if doc_id_set is not None and result_doc_id not in doc_id_set:
                    continue

                content = result.get("content", result.get("page_content", ""))
                filtered_results.append(content)

                if include_sources:
                    sources_used.append({
                        "document_id": result_doc_id,
                        "title": metadata.get("title", "Unknown"),
                        "score": result.get("score", 0),
                        "chunk_preview": content[:200] + "..." if len(content) > 200 else content,
                    })

                if len(filtered_results) >= max_context_chunks:
                    break

            if not filtered_results:
                return {
                    "answer": "I couldn't find any relevant information in the knowledge base to answer your question.",
                    "question": question,
                    "sources": [],
                    "context_found": False,
                }

            # Build context for LLM
            context = "\n\n---\n\n".join(filtered_results)

            # Generate answer using LLM
            prompt = f"""Based on the following context from the knowledge base, answer the user's question.
If the context doesn't contain enough information to answer the question, say so.
Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""

            answer = await self.llm_service.generate_response(
                query=prompt,
                temperature=0.3,
                max_tokens=1500,
                user_settings=user_settings,
            )

            response = {
                "answer": answer.strip(),
                "question": question,
                "context_found": True,
                "chunks_used": len(filtered_results),
            }

            if include_sources:
                response["sources"] = sources_used

            return response

        except Exception as e:
            logger.error(f"MCP chat error: {e}")
            return {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "question": question,
                "error": str(e),
            }

    async def _load_user_settings(
        self,
        user_id: UUID,
        db: AsyncSession
    ) -> Optional[UserLLMSettings]:
        """Load user's LLM preferences."""
        try:
            result = await db.execute(
                select(UserPreferences).where(UserPreferences.user_id == user_id)
            )
            prefs = result.scalar_one_or_none()
            if prefs:
                return UserLLMSettings.from_preferences(prefs)
        except Exception as e:
            logger.warning(f"Failed to load user LLM settings: {e}")
        return None

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
