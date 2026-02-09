"""
Chat service for handling chat sessions and messages.
"""

import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload, noload
from loguru import logger

from app.models.chat import ChatSession, ChatMessage
from app.models.user import User
from app.services.llm_service import LLMService, UserLLMSettings
from app.services.vector_store import vector_store_service
from app.services.memory_service import MemoryService
from app.services.query_processor import QueryProcessor
from app.services.context_manager import ContextManager
from app.schemas.memory import MemorySearchRequest, MemoryCreate
from app.core.config import settings


class ChatService:
    """Service for managing chat sessions and messages."""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.vector_store = vector_store_service
        self.memory_service = MemoryService()
        self.query_processor = QueryProcessor()
        self.context_manager = ContextManager()
        self._vector_store_initialized = False
    
    async def _ensure_vector_store_initialized(self):
        """Ensure vector store is initialized."""
        if not self._vector_store_initialized:
            await self.vector_store.initialize(background=True)
            self._vector_store_initialized = True
    
    async def create_session(
        self,
        user_id: UUID,
        db: AsyncSession,
        title: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatSession:
        """
        Create a new chat session for a user.
        
        Args:
            user_id: User ID
            title: Optional session title
            db: Database session
            
        Returns:
            Created ChatSession object
        """
        session = ChatSession(
            user_id=user_id,
            title=title or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            extra_metadata=extra_metadata,
        )
        
        db.add(session)
        await db.commit()
        await db.refresh(session)
        
        # Re-query the session with noload to prevent loading messages
        # This ensures messages relationship is not loaded
        result = await db.execute(
            select(ChatSession)
            .options(noload(ChatSession.messages))
            .where(ChatSession.id == session.id)
        )
        session = result.scalar_one()
        
        logger.info(f"Created chat session {session.id} for user {user_id} with title '{session.title}'")
        return session
    
    async def get_session(
        self,
        session_id: UUID,
        user_id: UUID,
        db: AsyncSession
    ) -> Optional[ChatSession]:
        """
        Get a chat session by ID, ensuring it belongs to the user.
        
        Args:
            session_id: Chat session ID
            user_id: User ID to verify ownership
            db: Database session
            
        Returns:
            ChatSession object if found and belongs to user, None otherwise
        """
        result = await db.execute(
            select(ChatSession).where(
                and_(
                    ChatSession.id == session_id,
                    ChatSession.user_id == user_id
                )
            )
        )
        return result.scalar_one_or_none()
    
    async def get_session_with_messages(
        self,
        session_id: UUID,
        user_id: UUID,
        db: AsyncSession
    ) -> Optional[ChatSession]:
        """
        Get a chat session with its messages, ensuring it belongs to the user.
        
        Args:
            session_id: Chat session ID
            user_id: User ID to verify ownership
            db: Database session
            
        Returns:
            ChatSession object with messages loaded, None if not found or doesn't belong to user
            
        Note:
            Messages are sorted by creation time
        """
        result = await db.execute(
            select(ChatSession)
            .options(selectinload(ChatSession.messages))
            .where(
                and_(
                    ChatSession.id == session_id,
                    ChatSession.user_id == user_id
                )
            )
        )
        session = result.scalar_one_or_none()
        
        if session and session.messages:
            # Sort messages by creation time
            session.messages.sort(key=lambda m: m.created_at)
        
        return session
    
    async def get_user_sessions(
        self,
        user_id: UUID,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 50
    ) -> tuple[List[ChatSession], int]:
        """
        Get paginated chat sessions for a user with total count.
        
        Args:
            user_id: User ID
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            Tuple of (sessions list, total count)
        """
        from sqlalchemy import func
        
        # Base query for user sessions - only active sessions
        base_query = select(ChatSession).where(
            and_(
                ChatSession.user_id == user_id,
                ChatSession.is_active == True
            )
        )
        
        # Get total count
        count_query = select(func.count()).select_from(base_query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0
        
        # Get paginated results
        # Use noload for messages to prevent lazy loading (messages not needed in list view)
        # Order by last_message_at (most recent activity), then by created_at (newest first) as fallback
        # Use COALESCE to handle NULL last_message_at values
        from sqlalchemy import desc, func as sql_func
        query = base_query.options(noload(ChatSession.messages)).order_by(
            desc(sql_func.coalesce(ChatSession.last_message_at, ChatSession.created_at)),
            desc(ChatSession.created_at)
        ).offset(skip).limit(limit)
        result = await db.execute(query)
        sessions = result.scalars().all()
        
        logger.debug(f"Retrieved {len(sessions)} sessions for user {user_id} (skip={skip}, limit={limit}, total={total})")
        
        return sessions, total
    
    async def create_message(
        self,
        session_id: UUID,
        content: str,
        role: str,
        db: AsyncSession,
        **kwargs
    ) -> ChatMessage:
        """
        Create a new chat message.
        
        Args:
            session_id: Chat session ID
            content: Message content
            role: Message role ('user' or 'assistant')
            db: Database session
            **kwargs: Additional message fields (model_used, response_time, source_documents, etc.)
            
        Returns:
            Created ChatMessage object
            
        Note:
            Automatically updates the session's last_message_at timestamp
        """
        message = ChatMessage(
            session_id=session_id,
            content=content,
            role=role,
            **kwargs
        )
        
        db.add(message)
        
        # Update session's last_message_at
        session_result = await db.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        session = session_result.scalar_one_or_none()
        if session:
            session.last_message_at = datetime.utcnow()
        
        await db.commit()
        await db.refresh(message)
        
        logger.debug(f"Created {role} message in session {session_id}, updated last_message_at")
        
        return message
    
    async def generate_response(
        self,
        session_id: UUID,
        query: str,
        user_id: UUID,
        db: AsyncSession,
        user_settings: Optional["UserLLMSettings"] = None,
    ) -> ChatMessage:
        """
        Generate an AI response to a user query using RAG (Retrieval-Augmented Generation).

        This method:
        1. Creates a user message from the query
        2. Retrieves relevant documents from the vector store
        3. Retrieves relevant memories for context
        4. Builds conversation history
        5. Generates response using LLM with all context
        6. Creates and returns assistant message

        Args:
            session_id: Chat session ID
            query: User's question or query
            user_id: User ID for memory retrieval
            db: Database session
            user_settings: Optional user-specific LLM settings

        Returns:
            ChatMessage object containing the AI response with metadata
            
        Raises:
            Exception: If response generation fails
        """
        start_time = time.time()
        
        try:
            # Create user message first
            user_message = await self.create_message(
                session_id=session_id,
                content=query,
                role="user",
                db=db
            )
            
            # Get conversation history
            session = await self.get_session_with_messages(
                session_id=session_id,
                user_id=user_id,
                db=db
            )

            # Apply per-session LLM overrides (stored in session.extra_metadata)
            effective_user_settings = user_settings or UserLLMSettings()
            try:
                meta = session.extra_metadata or {}
                # Supported keys (intentionally simple for "plugin" consumers):
                # - llm_provider, llm_model
                # - llm_task_models: {"chat": "..."}
                # - llm_task_providers: {"chat": "..."}
                if isinstance(meta, dict):
                    llm_provider = meta.get("llm_provider")
                    llm_model = meta.get("llm_model")
                    llm_task_models = meta.get("llm_task_models")
                    llm_task_providers = meta.get("llm_task_providers")

                    # Merge, preferring session overrides when present.
                    merged_task_models = dict(effective_user_settings.task_models or {})
                    if isinstance(llm_task_models, dict):
                        merged_task_models.update({k: v for k, v in llm_task_models.items() if v})

                    merged_task_providers = dict(effective_user_settings.task_providers or {})
                    if isinstance(llm_task_providers, dict):
                        merged_task_providers.update({k: v for k, v in llm_task_providers.items() if v})

                    effective_user_settings = UserLLMSettings(
                        provider=(llm_provider or effective_user_settings.provider),
                        model=(llm_model or effective_user_settings.model),
                        api_url=effective_user_settings.api_url,
                        api_key=effective_user_settings.api_key,
                        temperature=effective_user_settings.temperature,
                        max_tokens=effective_user_settings.max_tokens,
                        task_models=merged_task_models or None,
                        task_providers=merged_task_providers or None,
                    )
            except Exception as e:
                logger.warning(f"Failed to apply session LLM overrides: {e}")
            
            # Get relevant memories for context
            relevant_memories = await self.memory_service.search_memories(
                user_id=user_id,
                search_request=MemorySearchRequest(
                    query=query,
                    limit=5,
                    min_importance=0.3
                ),
                db=db
            )
            
            # Process query
            processed_query = self.query_processor.process_query(
                query,
                expand=settings.RAG_QUERY_EXPANSION_ENABLED,
                rewrite=True,
                normalize=True
            )
            
            # Use processed query for search
            search_query = processed_query.get("processed", query)
            
            # Ensure vector store is initialized
            await self._ensure_vector_store_initialized()
            
            # Generate multi-queries if enabled for better recall
            all_search_results = []
            retrieval_trace_payload: dict[str, Any] = {
                "original_query": query,
                "processed_query": search_query,
                "query_processor": {
                    "intent": processed_query.get("intent"),
                    "key_terms": processed_query.get("key_terms"),
                },
            }
            if settings.RAG_QUERY_EXPANSION_ENABLED:
                try:
                    query_variations = await self.query_processor.generate_multi_queries(
                        search_query,
                        llm_service=self.llm_service,
                        num_queries=3,
                        user_settings=effective_user_settings,
                    )
                    
                    # Search with all variations
                    traces: list[dict[str, Any]] = []
                    for q_var in query_variations:
                        results, one_trace = await self.vector_store.search_with_trace(
                            query=q_var,
                            limit=settings.MAX_SEARCH_RESULTS * 2
                        )
                        traces.append({"query": q_var, "trace": one_trace})
                        all_search_results.extend(results)

                    retrieval_trace_payload["multi_query"] = {
                        "queries": query_variations,
                        "traces": traces,
                    }
                    
                    # Deduplicate by ID and merge scores
                    seen_ids = {}
                    for result in all_search_results:
                        result_id = result.get("id")
                        if result_id in seen_ids:
                            # Merge scores (take max)
                            seen_ids[result_id]["score"] = max(
                                seen_ids[result_id]["score"],
                                result.get("score", 0.0)
                            )
                        else:
                            seen_ids[result_id] = result
                    
                    search_results = list(seen_ids.values())
                    # Sort by score
                    search_results = sorted(search_results, key=lambda x: x.get("score", 0.0), reverse=True)
                except Exception as e:
                    logger.warning(f"Multi-query generation failed: {e}, using single query")
                    search_results, one_trace = await self.vector_store.search_with_trace(
                        query=search_query,
                        limit=settings.MAX_SEARCH_RESULTS * 2
                    )
                    retrieval_trace_payload["single_query"] = {"query": search_query, "trace": one_trace}
            else:
                # Single query search
                search_results, one_trace = await self.vector_store.search_with_trace(
                    query=search_query,
                    limit=settings.MAX_SEARCH_RESULTS * 2  # Get more for filtering/reranking
                )
                retrieval_trace_payload["single_query"] = {"query": search_query, "trace": one_trace}
            
            # Filter by relevance
            search_results = self.context_manager.filter_by_relevance(search_results)
            
            # Select most relevant parts
            search_results = self.context_manager.select_relevant_parts(
                search_results,
                query=search_query,
                max_results=settings.MAX_SEARCH_RESULTS
            )

            retrieval_trace_payload["selected_results"] = [
                {
                    "id": r.get("id"),
                    "score": r.get("score"),
                    "document_id": (r.get("metadata") or {}).get("document_id"),
                    "chunk_id": (r.get("metadata") or {}).get("chunk_id"),
                    "title": (r.get("metadata") or {}).get("title"),
                    "source": (r.get("metadata") or {}).get("source") or (r.get("metadata") or {}).get("source_type"),
                }
                for r in (search_results or [])[:settings.MAX_SEARCH_RESULTS]
            ]
            
            # Build context with token management
            context = await self.context_manager.compress_context(
                search_results,
                llm_service=self.llm_service
            )
            
            # Get context metrics for logging
            context_metrics = self.context_manager.get_context_metrics(search_results)
            retrieval_trace_payload["context_metrics"] = context_metrics

            # Inject Knowledge Graph context if enabled
            kg_context = ""
            if settings.RAG_KG_CONTEXT_ENABLED:
                try:
                    from app.services.knowledge_graph_service import KnowledgeGraphService
                    kg_service = KnowledgeGraphService()

                    # Extract potential entity names from query and search results
                    entity_names = self.context_manager.extract_entity_names_from_results(search_results)

                    # Also add key terms from the processed query
                    key_terms = processed_query.get("key_terms", [])
                    for term in key_terms:
                        if len(term) > 2 and term not in entity_names:
                            entity_names.append(term)

                    # Search for matching entities in KG
                    if entity_names:
                        entities = await kg_service.search_entities_by_names(
                            names=entity_names[:20],  # Limit search candidates
                            db=db,
                            limit=settings.RAG_KG_MAX_ENTITIES
                        )

                        if entities:
                            # Get relationships between found entities
                            entity_ids = [e.id for e in entities]
                            kg_data = await kg_service.get_entity_context(
                                entity_ids=entity_ids,
                                db=db,
                                max_relationships=settings.RAG_KG_MAX_RELATIONSHIPS
                            )

                            # Build KG context string
                            kg_context = self.context_manager.build_kg_context(
                                entities=kg_data["entities"],
                                relationships=kg_data["relationships"]
                            )
                            logger.debug(f"KG context injected: {len(kg_data['entities'])} entities, {len(kg_data['relationships'])} relationships")
                except Exception as e:
                    logger.warning(f"KG context injection failed: {e}")
                    # Continue without KG context - don't break the chat flow

            memory_context = self._build_memory_context(relevant_memories)
            conversation_history = self._build_conversation_history(session.messages[:-1])  # Exclude current user message

            # Generate response using LLM with memory and KG context
            response_content = await self.llm_service.generate_response(
                query=query,
                context=context,
                conversation_history=conversation_history,
                memory_context=memory_context,
                kg_context=kg_context,
                user_settings=effective_user_settings,
                task_type="chat",
                user_id=user_id,
                db=db,
            )
            
            response_time = time.time() - start_time
            
            # Log RAG performance metrics
            logger.info(
                f"RAG metrics for session {session_id}: "
                f"results={context_metrics['total_results']}, "
                f"tokens={context_metrics['total_tokens']}, "
                f"avg_score={context_metrics['avg_score']:.2f}, "
                f"coverage={context_metrics['coverage']:.1f}%, "
                f"intent={processed_query.get('intent')}, "
                f"response_time={response_time:.2f}s"
            )
            
            # Create assistant message
            # search_results is a list of dicts with "metadata" key (dict), not objects
            from app.services.storage_service import storage_service
            from app.models.document import Document
            from app.models.retrieval_trace import RetrievalTrace
            from sqlalchemy import select as sql_select

            settings_snapshot = {
                "provider": getattr(self.vector_store, "provider", None),
                "hybrid_enabled": bool(getattr(settings, "RAG_HYBRID_SEARCH_ENABLED", False)),
                "hybrid_alpha": float(getattr(settings, "RAG_HYBRID_SEARCH_ALPHA", 0.0)),
                "rerank_enabled": bool(getattr(settings, "RAG_RERANKING_ENABLED", False)),
                "rerank_model": getattr(settings, "RAG_RERANKING_MODEL", None),
                "mmr_enabled": bool(getattr(settings, "RAG_MMR_ENABLED", False)),
                "dedup_enabled": bool(getattr(settings, "RAG_DEDUPLICATION_ENABLED", False)),
                "min_relevance": float(getattr(settings, "RAG_MIN_RELEVANCE_SCORE", 0.0)),
                "max_search_results": int(getattr(settings, "MAX_SEARCH_RESULTS", 0)),
            }
            retrieval_trace = RetrievalTrace(
                user_id=user_id,
                session_id=session_id,
                trace_type="chat",
                query=query,
                processed_query=search_query,
                provider=getattr(self.vector_store, "provider", None),
                settings_snapshot=settings_snapshot,
                trace=retrieval_trace_payload,
            )
            db.add(retrieval_trace)
            await db.commit()
            await db.refresh(retrieval_trace)

            # Batch fetch documents to avoid N+1 queries
            doc_ids = set()
            for doc in search_results:
                doc_id = doc.get("metadata", {}).get("document_id", doc.get("id"))
                if doc_id:
                    try:
                        doc_ids.add(UUID(doc_id) if isinstance(doc_id, str) else doc_id)
                    except (ValueError, TypeError):
                        pass

            # Fetch all documents in one query
            documents_map = {}
            if doc_ids:
                stmt = sql_select(Document).where(Document.id.in_(list(doc_ids)))
                result = await db.execute(stmt)
                for document in result.scalars().all():
                    documents_map[document.id] = document

            source_docs = []
            for doc in search_results:
                doc_id = doc.get("metadata", {}).get("document_id", doc.get("id"))
                doc_metadata = doc.get("metadata", {})

                # Generate download URL if document has file_path
                download_url = None
                if doc_id:
                    try:
                        doc_uuid = UUID(doc_id) if isinstance(doc_id, str) else doc_id
                        document = documents_map.get(doc_uuid)
                        if document and document.file_path:
                            download_url = await storage_service.get_presigned_download_url(document.file_path)
                    except Exception as e:
                        logger.warning(f"Failed to generate download URL for document {doc_id}: {e}")

                # Get content snippet for preview (first 200 chars)
                content = doc.get("content") or doc.get("page_content", "")
                snippet = content[:200].strip() if content else None

                source_docs.append({
                    "id": doc_id,
                    "title": doc_metadata.get("title", "Unknown"),
                    "score": doc_metadata.get("score", doc.get("score", 0.0)),
                    "source": doc_metadata.get("source", "Unknown"),
                    "download_url": download_url,
                    "chunk_id": doc_metadata.get("chunk_id"),
                    "chunk_index": doc_metadata.get("chunk_index"),
                    "snippet": snippet,
                })

            # Heuristic groundedness score (0..1): combines retrieval confidence with context coverage.
            try:
                scores = []
                for r in search_results:
                    md = r.get("metadata") or {}
                    s = md.get("score", r.get("score", 0.0))
                    try:
                        scores.append(float(s))
                    except Exception:
                        pass
                avg_score = (sum(scores) / len(scores)) if scores else 0.0
                coverage01 = float(context_metrics.get("coverage", 0.0) or 0.0) / 100.0
                groundedness_score = 0.7 * avg_score + 0.3 * coverage01
                groundedness_score = max(0.0, min(1.0, groundedness_score))
            except Exception:
                groundedness_score = None
            
            assistant_message = await self.create_message(
                session_id=session_id,
                content=response_content,
                role="assistant",
                model_used=effective_user_settings.get_model_for_task("chat") or self.llm_service.get_active_model(),
                response_time=response_time,
                source_documents=source_docs,
                context_used=context,
                search_query=query,
                groundedness_score=groundedness_score,
                retrieval_trace_id=retrieval_trace.id,
                db=db
            )
            
            # Extract memories from this conversation turn
            await self._extract_memories_from_turn(
                user_id=user_id,
                session_id=session_id,
                user_message=user_message,
                assistant_message=assistant_message,
                db=db
            )
            
            # Trigger async title generation if this is the first assistant message
            # Check if session has default title (starts with "Chat ") or is None
            session_result = await db.execute(
                select(ChatSession).where(ChatSession.id == session_id)
            )
            session = session_result.scalar_one_or_none()
            if session and (not session.title or session.title.startswith("Chat ")):
                try:
                    from app.tasks.chat_tasks import generate_chat_title
                    # Trigger async title generation
                    generate_chat_title.delay(str(session_id))
                    logger.info(f"Triggered title generation for session {session_id}")
                except Exception as e:
                    logger.warning(f"Failed to trigger title generation for session {session_id}: {e}")
            
            logger.info(f"Generated response for session {session_id} in {response_time:.2f}s")
            return assistant_message
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            
            # Create error message
            error_message = await self.create_message(
                session_id=session_id,
                content="I apologize, but I encountered an error while processing your request. Please try again.",
                role="assistant",
                message_type="error",
                processing_error=str(e),
                response_time=time.time() - start_time,
                db=db
            )
            
            return error_message
    
    def _build_context(self, search_results: List[Any]) -> str:
        """
        Build context string from vector search results.
        
        Args:
            search_results: List of search result dictionaries from vector store
            
        Returns:
            Formatted context string with document excerpts and metadata
        """
        if not search_results:
            return ""
        
        context_parts = []
        for i, doc in enumerate(search_results, 1):
            # search_results are dicts with "content" or "page_content" key and "metadata" key (dict)
            content = (doc.get("content") or doc.get("page_content", ""))[:settings.CHUNK_SIZE]
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Unknown")
            title = metadata.get("title", "Unknown Document")
            
            context_parts.append(f"Source {i} ({source} - {title}):\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _build_conversation_history(self, messages: List[ChatMessage]) -> str:
        """
        Build conversation history string from chat messages.
        
        Args:
            messages: List of ChatMessage objects
            
        Returns:
            Formatted conversation history string
        """
        if not messages:
            return ""
        
        # Get last few messages to fit within context window
        max_messages = 10
        recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
        
        history_parts = []
        for message in recent_messages:
            role = "Human" if message.role == "user" else "Assistant"
            history_parts.append(f"{role}: {message.content}")
        
        return "\n".join(history_parts)
    
    async def delete_session(
        self,
        session_id: UUID,
        user_id: UUID,
        db: AsyncSession
    ) -> bool:
        """
        Soft delete a chat session (set is_active to False), ensuring it belongs to the user.
        
        Args:
            session_id: Chat session ID
            user_id: User ID to verify ownership
            db: Database session
            
        Returns:
            True if session was deleted, False if not found or doesn't belong to user
        """
        try:
            # Get session without is_active filter (so we can delete already-deleted sessions if needed)
            result = await db.execute(
                select(ChatSession).where(
                    and_(
                        ChatSession.id == session_id,
                        ChatSession.user_id == user_id
                    )
                )
            )
            session = result.scalar_one_or_none()
            
            if not session:
                logger.warning(f"Session {session_id} not found or doesn't belong to user {user_id}")
                return False
            
            # Check if already deleted
            if not session.is_active:
                logger.info(f"Session {session_id} is already deleted")
                return True  # Consider it successful if already deleted
            
            # Soft delete: set is_active to False
            session.is_active = False
            session.updated_at = datetime.utcnow()
            await db.commit()
            await db.refresh(session)
            
            logger.info(f"Soft deleted chat session {session_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            await db.rollback()
            raise
    
    async def update_message_feedback(
        self,
        message_id: UUID,
        rating: int,
        feedback: Optional[str],
        db: AsyncSession
    ) -> bool:
        """
        Update feedback for a chat message.
        
        Args:
            message_id: Chat message ID
            rating: User rating (typically 1-5)
            feedback: Optional text feedback
            db: Database session
            
        Returns:
            True if feedback was updated, False if message not found
        """
        result = await db.execute(
            select(ChatMessage).where(ChatMessage.id == message_id)
        )
        message = result.scalar_one_or_none()
        
        if not message:
            return False
        
        message.user_rating = rating
        message.user_feedback = feedback
        message.updated_at = datetime.utcnow()
        
        await db.commit()
        
        logger.info(f"Updated feedback for message {message_id}: rating={rating}")
        return True
    
    def _build_memory_context(self, memories: List[Any]) -> str:
        """
        Build memory context string from relevant memories.
        
        Args:
            memories: List of memory response objects
            
        Returns:
            Formatted memory context string
        """
        if not memories:
            return ""
        
        context_parts = ["Relevant memories from previous conversations:"]
        for i, memory in enumerate(memories, 1):
            memory_type = memory.memory_type.upper()
            context_parts.append(f"{i}. [{memory_type}] {memory.content}")
        
        return "\n".join(context_parts)
    
    async def _extract_memories_from_turn(
        self,
        user_id: UUID,
        session_id: UUID,
        user_message: ChatMessage,
        assistant_message: ChatMessage,
        db: AsyncSession
    ) -> None:
        """Extract memories from a conversation turn."""
        try:
            # Combine user and assistant messages for memory extraction
            conversation_text = f"User: {user_message.content}\nAssistant: {assistant_message.content}"
            
            # Use LLM to extract memories
            extracted_memories = await self.memory_service._extract_memories_with_llm(
                conversation_text=conversation_text,
                user_id=user_id,
                session_id=session_id
            )
            
            # Create memories
            for memory_data in extracted_memories:
                memory_data.source_message_id = user_message.id
                await self.memory_service.create_memory(
                    user_id=user_id,
                    memory_data=memory_data,
                    db=db
                )
            
            # Note: Memory interactions are recorded when memories are retrieved in generate_response
            # This function only extracts new memories from the conversation
            
        except Exception as e:
            logger.error(f"Error extracting memories from turn: {e}")
            # Don't raise - memory extraction failure shouldn't break the chat
