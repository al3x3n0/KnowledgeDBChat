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
from app.services.llm_service import LLMService
from app.services.vector_store import VectorStoreService
from app.services.memory_service import MemoryService
from app.services.query_processor import QueryProcessor
from app.services.context_manager import ContextManager
from app.schemas.memory import MemorySearchRequest, MemoryCreate
from app.core.config import settings


class ChatService:
    """Service for managing chat sessions and messages."""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.vector_store = VectorStoreService()
        self.memory_service = MemoryService()
        self.query_processor = QueryProcessor()
        self.context_manager = ContextManager()
        self._vector_store_initialized = False
    
    async def _ensure_vector_store_initialized(self):
        """Ensure vector store is initialized."""
        if not self._vector_store_initialized:
            await self.vector_store.initialize()
            self._vector_store_initialized = True
    
    async def create_session(
        self,
        user_id: UUID,
        db: AsyncSession,
        title: Optional[str] = None
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
            title=title or f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )
        
        db.add(session)
        await db.commit()
        
        # Re-query the session with noload to prevent loading messages
        # This ensures messages relationship is not loaded
        result = await db.execute(
            select(ChatSession)
            .options(noload(ChatSession.messages))
            .where(ChatSession.id == session.id)
        )
        session = result.scalar_one()
        
        logger.info(f"Created chat session {session.id} for user {user_id}")
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
        
        # Base query for user sessions
        base_query = select(ChatSession).where(ChatSession.user_id == user_id)
        
        # Get total count
        count_query = select(func.count()).select_from(base_query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0
        
        # Get paginated results
        # Use noload for messages to prevent lazy loading (messages not needed in list view)
        query = base_query.options(noload(ChatSession.messages)).order_by(ChatSession.last_message_at.desc()).offset(skip).limit(limit)
        result = await db.execute(query)
        sessions = result.scalars().all()
        
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
        await db.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        session = await db.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        session = session.scalar_one_or_none()
        if session:
            session.last_message_at = datetime.utcnow()
        
        await db.commit()
        await db.refresh(message)
        
        return message
    
    async def generate_response(
        self,
        session_id: UUID,
        query: str,
        user_id: UUID,
        db: AsyncSession
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
            if settings.RAG_QUERY_EXPANSION_ENABLED:
                try:
                    query_variations = await self.query_processor.generate_multi_queries(
                        search_query,
                        llm_service=self.llm_service,
                        num_queries=3
                    )
                    
                    # Search with all variations
                    for q_var in query_variations:
                        results = await self.vector_store.search(
                            query=q_var,
                            limit=settings.MAX_SEARCH_RESULTS * 2
                        )
                        all_search_results.extend(results)
                    
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
                    search_results = await self.vector_store.search(
                        query=search_query,
                        limit=settings.MAX_SEARCH_RESULTS * 2
                    )
            else:
                # Single query search
                search_results = await self.vector_store.search(
                    query=search_query,
                    limit=settings.MAX_SEARCH_RESULTS * 2  # Get more for filtering/reranking
                )
            
            # Filter by relevance
            search_results = self.context_manager.filter_by_relevance(search_results)
            
            # Select most relevant parts
            search_results = self.context_manager.select_relevant_parts(
                search_results,
                query=search_query,
                max_results=settings.MAX_SEARCH_RESULTS
            )
            
            # Build context with token management
            context = await self.context_manager.compress_context(
                search_results,
                llm_service=self.llm_service
            )
            
            # Get context metrics for logging
            context_metrics = self.context_manager.get_context_metrics(search_results)
            
            memory_context = self._build_memory_context(relevant_memories)
            conversation_history = self._build_conversation_history(session.messages[:-1])  # Exclude current user message
            
            # Generate response using LLM with memory context
            response_content = await self.llm_service.generate_response(
                query=query,
                context=context,
                conversation_history=conversation_history,
                memory_context=memory_context
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
            assistant_message = await self.create_message(
                session_id=session_id,
                content=response_content,
                role="assistant",
                model_used=settings.DEFAULT_MODEL,
                response_time=response_time,
                source_documents=[{
                    "id": doc.get("metadata", {}).get("document_id", doc.get("id")),
                    "title": doc.get("metadata", {}).get("title", "Unknown"),
                    "score": doc.get("metadata", {}).get("score", doc.get("score", 0.0)),
                    "source": doc.get("metadata", {}).get("source", "Unknown")
                } for doc in search_results],
                context_used=context,
                search_query=query,
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
        Delete a chat session, ensuring it belongs to the user.
        
        Args:
            session_id: Chat session ID
            user_id: User ID to verify ownership
            db: Database session
            
        Returns:
            True if session was deleted, False if not found or doesn't belong to user
        """
        session = await self.get_session(session_id, user_id, db)
        if not session:
            return False
        
        await db.delete(session)
        await db.commit()
        
        logger.info(f"Deleted chat session {session_id}")
        return True
    
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
            
            # Record memory interactions for the memories used
            if hasattr(assistant_message, 'source_documents'):
                for memory in memories:
                    await self.memory_service.create_memory_interaction(
                        memory_id=memory.id,
                        session_id=session_id,
                        interaction_type="retrieved",
                        relevance_score=0.8,  # Default relevance
                        message_id=assistant_message.id,
                        db=db
                    )
            
        except Exception as e:
            logger.error(f"Error extracting memories from turn: {e}")
            # Don't raise - memory extraction failure shouldn't break the chat
