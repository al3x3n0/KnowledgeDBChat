"""
Memory service for conversation context retention.
"""

import json
import re
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, desc, asc
from sqlalchemy.orm import selectinload
from loguru import logger

from app.models.memory import ConversationMemory, MemoryInteraction, UserPreferences
from app.models.user import User
from app.models.chat import ChatSession, ChatMessage
from app.schemas.memory import (
    MemoryCreate, MemoryUpdate, MemoryResponse, MemorySearchRequest,
    MemorySummaryRequest, MemorySummaryResponse, MemoryStatsResponse
)
from app.services.llm_service import LLMService
from app.services.text_processor import TextProcessor

class MemoryService:
    """Service for managing conversation memories."""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.text_processor = TextProcessor()
    
    async def create_memory(
        self,
        user_id: UUID,
        memory_data: MemoryCreate,
        db: AsyncSession
    ) -> MemoryResponse:
        """Create a new conversation memory."""
        try:
            # Check if similar memory already exists
            existing_memory = await self._find_similar_memory(
                user_id, memory_data.content, memory_data.memory_type, db
            )
            
            if existing_memory:
                # Update existing memory instead of creating duplicate
                return await self.update_memory(
                    existing_memory.id, 
                    MemoryUpdate(
                        content=memory_data.content,
                        importance_score=max(existing_memory.importance_score, memory_data.importance_score),
                        context=memory_data.context,
                        tags=memory_data.tags
                    ),
                    db
                )
            
            # Create new memory
            memory = ConversationMemory(
                user_id=user_id,
                session_id=memory_data.session_id,
                memory_type=memory_data.memory_type,
                content=memory_data.content,
                importance_score=memory_data.importance_score,
                context=memory_data.context,
                tags=memory_data.tags,
                source_message_id=memory_data.source_message_id
            )
            
            db.add(memory)
            await db.commit()
            await db.refresh(memory)
            
            logger.info(f"Created memory {memory.id} for user {user_id}")
            return MemoryResponse.from_orm(memory)
            
        except Exception as e:
            logger.error(f"Error creating memory: {e}")
            await db.rollback()
            raise
    
    async def get_memories(
        self,
        user_id: UUID,
        session_id: Optional[UUID] = None,
        memory_types: Optional[List[str]] = None,
        skip: int = 0,
        limit: int = 10,
        db: AsyncSession = None
    ) -> tuple[List[MemoryResponse], int]:
        """
        Get paginated memories for a user with total count.
        
        Args:
            user_id: User ID
            session_id: Optional session ID filter
            memory_types: Optional list of memory types to filter by
            skip: Number of records to skip
            limit: Maximum number of records to return
            db: Database session
            
        Returns:
            Tuple of (memories list, total count)
        """
        from sqlalchemy import func
        
        try:
            # Build base query
            base_query = select(ConversationMemory).where(
                and_(
                    ConversationMemory.user_id == user_id,
                    ConversationMemory.is_active == True
                )
            )
            
            if session_id:
                base_query = base_query.where(ConversationMemory.session_id == session_id)
            
            if memory_types:
                base_query = base_query.where(ConversationMemory.memory_type.in_(memory_types))
            
            # Get total count
            count_query = select(func.count()).select_from(base_query.subquery())
            total_result = await db.execute(count_query)
            total = total_result.scalar() or 0
            
            # Get paginated results
            query = base_query.order_by(
                desc(ConversationMemory.importance_score),
                desc(ConversationMemory.last_accessed_at)
            ).offset(skip).limit(limit)
            
            result = await db.execute(query)
            memories = result.scalars().all()
            
            # Update access count and last accessed time
            for memory in memories:
                memory.access_count += 1
                memory.last_accessed_at = datetime.utcnow()
            
            await db.commit()
            
            return [MemoryResponse.from_orm(memory) for memory in memories], total
            
        except Exception as e:
            logger.error(f"Error getting memories: {e}")
            raise
    
    async def get_memory_by_id(
        self,
        memory_id: UUID,
        user_id: UUID,
        db: AsyncSession
    ) -> Optional[MemoryResponse]:
        """
        Get a specific memory by ID, ensuring it belongs to the user.
        
        Args:
            memory_id: Memory UUID
            user_id: User ID to verify ownership
            db: Database session
            
        Returns:
            MemoryResponse if found and belongs to user, None otherwise
        """
        try:
            result = await db.execute(
                select(ConversationMemory).where(
                    and_(
                        ConversationMemory.id == memory_id,
                        ConversationMemory.user_id == user_id,
                        ConversationMemory.is_active == True
                    )
                )
            )
            memory = result.scalar_one_or_none()
            
            if memory:
                # Update access count
                memory.access_count += 1
                memory.last_accessed_at = datetime.utcnow()
                await db.commit()
                return MemoryResponse.from_orm(memory)
            
            return None
        except Exception as e:
            logger.error(f"Error getting memory by ID: {e}")
            raise
    
    async def search_memories(
        self,
        user_id: UUID,
        search_request: MemorySearchRequest,
        db: AsyncSession
    ) -> List[MemoryResponse]:
        """Search memories using semantic similarity."""
        try:
            # Get user preferences for search
            prefs = await self.get_user_preferences(user_id, db)
            
            # Build base query
            conditions = [
                ConversationMemory.user_id == user_id,
                ConversationMemory.is_active == True
            ]
            
            # Add importance score filter only if min_importance is not None
            if search_request.min_importance is not None:
                conditions.append(ConversationMemory.importance_score >= search_request.min_importance)
            
            query = select(ConversationMemory).where(and_(*conditions))
            
            # Apply filters
            if search_request.memory_types:
                query = query.where(ConversationMemory.memory_type.in_(search_request.memory_types))
            
            if search_request.tags:
                query = query.where(
                    or_(*[ConversationMemory.tags.contains([tag]) for tag in search_request.tags])
                )
            
            # Get all matching memories
            result = await db.execute(query)
            all_memories = result.scalars().all()
            
            if not all_memories:
                return []
            
            # Use LLM to rank memories by relevance
            ranked_memories = await self._rank_memories_by_relevance(
                all_memories, search_request.query
            )
            
            # Return top results
            top_memories = ranked_memories[:search_request.limit]
            
            # Update access counts
            for memory in top_memories:
                memory.access_count += 1
                memory.last_accessed_at = datetime.utcnow()
            
            await db.commit()
            
            return [MemoryResponse.from_orm(memory) for memory in top_memories]
            
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            raise
    
    async def update_memory(
        self,
        memory_id: UUID,
        memory_update: MemoryUpdate,
        db: AsyncSession
    ) -> MemoryResponse:
        """Update an existing memory."""
        try:
            result = await db.execute(
                select(ConversationMemory).where(ConversationMemory.id == memory_id)
            )
            memory = result.scalar_one_or_none()
            
            if not memory:
                raise ValueError("Memory not found")
            
            # Update fields
            if memory_update.content is not None:
                memory.content = memory_update.content
            if memory_update.importance_score is not None:
                memory.importance_score = memory_update.importance_score
            if memory_update.context is not None:
                memory.context = memory_update.context
            if memory_update.tags is not None:
                memory.tags = memory_update.tags
            if memory_update.is_active is not None:
                memory.is_active = memory_update.is_active
            
            memory.last_accessed_at = datetime.utcnow()
            
            await db.commit()
            await db.refresh(memory)
            
            logger.info(f"Updated memory {memory_id}")
            return MemoryResponse.from_orm(memory)
            
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            await db.rollback()
            raise
    
    async def delete_memory(
        self,
        memory_id: UUID,
        user_id: UUID,
        db: AsyncSession
    ) -> bool:
        """Delete a memory (soft delete)."""
        try:
            result = await db.execute(
                select(ConversationMemory).where(
                    and_(
                        ConversationMemory.id == memory_id,
                        ConversationMemory.user_id == user_id
                    )
                )
            )
            memory = result.scalar_one_or_none()
            
            if not memory:
                return False
            
            memory.is_active = False
            await db.commit()
            
            logger.info(f"Deleted memory {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting memory: {e}")
            await db.rollback()
            raise
    
    async def create_memory_interaction(
        self,
        memory_id: UUID,
        session_id: UUID,
        interaction_type: str,
        relevance_score: Optional[float] = None,
        usage_context: Optional[Dict[str, Any]] = None,
        message_id: Optional[UUID] = None,
        db: AsyncSession = None
    ) -> None:
        """Record a memory interaction."""
        try:
            interaction = MemoryInteraction(
                memory_id=memory_id,
                session_id=session_id,
                message_id=message_id,
                interaction_type=interaction_type,
                relevance_score=relevance_score,
                usage_context=usage_context
            )
            
            db.add(interaction)
            await db.commit()
            
        except Exception as e:
            logger.error(f"Error creating memory interaction: {e}")
            await db.rollback()
            raise
    
    async def extract_memories_from_conversation(
        self,
        session_id: UUID,
        user_id: UUID,
        db: AsyncSession
    ) -> List[MemoryResponse]:
        """Extract memories from a conversation session."""
        try:
            # Get recent messages from the session
            result = await db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(desc(ChatMessage.created_at))
                .limit(20)  # Last 20 messages
            )
            messages = result.scalars().all()
            
            if not messages:
                return []
            
            # Combine messages into conversation text
            conversation_text = "\n".join([
                f"{msg.role}: {msg.content}" for msg in reversed(messages)
            ])
            
            # Use LLM to extract memories
            extracted_memories = await self._extract_memories_with_llm(
                conversation_text, user_id, session_id
            )
            
            # Create memories
            created_memories = []
            for memory_data in extracted_memories:
                memory = await self.create_memory(user_id, memory_data, db)
                created_memories.append(memory)
            
            logger.info(f"Extracted {len(created_memories)} memories from session {session_id}")
            return created_memories
            
        except Exception as e:
            logger.error(f"Error extracting memories from conversation: {e}")
            raise
    
    async def generate_memory_summary(
        self,
        user_id: UUID,
        summary_request: MemorySummaryRequest,
        db: AsyncSession
    ) -> MemorySummaryResponse:
        """Generate a summary of user's memories."""
        try:
            # Get memories based on request
            memories = await self.get_memories(
                user_id=user_id,
                session_id=summary_request.session_id,
                memory_types=summary_request.include_types,
                limit=50,  # Get more for summary
                db=db
            )
            
            if not memories:
                return MemorySummaryResponse(
                    summary="No memories found for the specified criteria.",
                    key_facts=[],
                    preferences=[],
                    context_items=[],
                    memory_count=0,
                    time_range=f"Last {summary_request.time_range_days} days"
                )
            
            # Categorize memories
            facts = [m for m in memories if m.memory_type == 'fact']
            preferences = [m for m in memories if m.memory_type == 'preference']
            context_items = [m for m in memories if m.memory_type == 'context']
            
            # Generate summary using LLM
            summary_text = await self._generate_memory_summary_with_llm(memories)
            
            return MemorySummaryResponse(
                summary=summary_text,
                key_facts=[m.content for m in facts[:10]],  # Top 10 facts
                preferences=[m.content for m in preferences[:10]],  # Top 10 preferences
                context_items=[m.content for m in context_items[:10]],  # Top 10 context items
                memory_count=len(memories),
                time_range=f"Last {summary_request.time_range_days} days"
            )
            
        except Exception as e:
            logger.error(f"Error generating memory summary: {e}")
            raise
    
    async def get_memory_stats(
        self,
        user_id: UUID,
        db: AsyncSession
    ) -> MemoryStatsResponse:
        """Get memory statistics for a user."""
        try:
            # Total memories
            total_result = await db.execute(
                select(func.count(ConversationMemory.id))
                .where(
                    and_(
                        ConversationMemory.user_id == user_id,
                        ConversationMemory.is_active == True
                    )
                )
            )
            total_memories = total_result.scalar()
            
            # Memories by type
            type_result = await db.execute(
                select(
                    ConversationMemory.memory_type,
                    func.count(ConversationMemory.id)
                )
                .where(
                    and_(
                        ConversationMemory.user_id == user_id,
                        ConversationMemory.is_active == True
                    )
                )
                .group_by(ConversationMemory.memory_type)
            )
            memories_by_type = dict(type_result.fetchall())
            
            # Recent memories (last 7 days)
            recent_date = datetime.utcnow() - timedelta(days=7)
            recent_result = await db.execute(
                select(func.count(ConversationMemory.id))
                .where(
                    and_(
                        ConversationMemory.user_id == user_id,
                        ConversationMemory.is_active == True,
                        ConversationMemory.created_at >= recent_date
                    )
                )
            )
            recent_memories = recent_result.scalar()
            
            # Most accessed memories
            accessed_result = await db.execute(
                select(ConversationMemory)
                .where(
                    and_(
                        ConversationMemory.user_id == user_id,
                        ConversationMemory.is_active == True
                    )
                )
                .order_by(desc(ConversationMemory.access_count))
                .limit(5)
            )
            most_accessed = [MemoryResponse.from_orm(m) for m in accessed_result.scalars().all()]
            
            # Memory usage trend (last 30 days)
            trend_data = await self._get_memory_usage_trend(user_id, db)
            
            return MemoryStatsResponse(
                total_memories=total_memories,
                memories_by_type=memories_by_type,
                recent_memories=recent_memories,
                most_accessed_memories=most_accessed,
                memory_usage_trend=trend_data
            )
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            raise
    
    async def get_user_preferences(
        self,
        user_id: UUID,
        db: AsyncSession
    ) -> UserPreferences:
        """Get or create user preferences."""
        try:
            result = await db.execute(
                select(UserPreferences).where(UserPreferences.user_id == user_id)
            )
            preferences = result.scalar_one_or_none()
            
            if not preferences:
                # Create default preferences
                preferences = UserPreferences(user_id=user_id)
                db.add(preferences)
                await db.commit()
                await db.refresh(preferences)
            
            return preferences
            
        except Exception as e:
            logger.error(f"Error getting user preferences: {e}")
            raise
    
    async def _find_similar_memory(
        self,
        user_id: UUID,
        content: str,
        memory_type: str,
        db: AsyncSession
    ) -> Optional[ConversationMemory]:
        """Find similar existing memory to avoid duplicates."""
        try:
            # Simple similarity check - could be enhanced with embeddings
            result = await db.execute(
                select(ConversationMemory)
                .where(
                    and_(
                        ConversationMemory.user_id == user_id,
                        ConversationMemory.memory_type == memory_type,
                        ConversationMemory.is_active == True
                    )
                )
            )
            memories = result.scalars().all()
            
            # Check for similar content (simple string similarity)
            for memory in memories:
                if self._calculate_similarity(content, memory.content) > 0.8:
                    return memory
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding similar memory: {e}")
            return None
    
    async def _rank_memories_by_relevance(
        self,
        memories: List[ConversationMemory],
        query: str
    ) -> List[ConversationMemory]:
        """Rank memories by relevance to query using LLM."""
        try:
            if not memories:
                return []
            
            # Create a prompt for ranking
            memory_texts = [f"{i+1}. {mem.content}" for i, mem in enumerate(memories)]
            prompt = f"""
            Given this query: "{query}"
            
            Rank these memories by relevance (most relevant first):
            
            {chr(10).join(memory_texts)}
            
            Return only the numbers in order of relevance, separated by commas.
            """
            
            # Use LLM to rank (simplified - in production, use proper ranking)
            # For now, return sorted by importance score
            return sorted(memories, key=lambda m: m.importance_score, reverse=True)
            
        except Exception as e:
            logger.error(f"Error ranking memories: {e}")
            return sorted(memories, key=lambda m: m.importance_score, reverse=True)
    
    async def _extract_memories_with_llm(
        self,
        conversation_text: str,
        user_id: UUID,
        session_id: UUID
    ) -> List[MemoryCreate]:
        """Use LLM to extract memories from conversation."""
        try:
            prompt = f"""
            Analyze this conversation and extract important information that should be remembered for future interactions.
            
            Conversation:
            {conversation_text}
            
            Extract memories in this format:
            TYPE: [fact|preference|context|summary]
            CONTENT: [the memory content]
            IMPORTANCE: [0.0-1.0]
            TAGS: [comma-separated tags]
            
            Focus on:
            - Personal preferences and settings
            - Important facts about the user or their work
            - Context that would be useful in future conversations
            - Goals or objectives mentioned
            - Constraints or limitations discussed
            
            Return up to 5 memories, one per line.
            """
            
            # Use LLM service to extract memories
            response = await self.llm_service.generate_response(prompt)
            
            # Parse the response
            memories = []
            for line in response.split('\n'):
                if line.strip() and 'TYPE:' in line:
                    try:
                        parts = line.split('CONTENT:')
                        if len(parts) == 2:
                            type_part = parts[0].replace('TYPE:', '').strip()
                            content_part = parts[1].split('IMPORTANCE:')[0].strip()
                            
                            # Extract importance and tags if present
                            importance = 0.5
                            tags = []
                            
                            if 'IMPORTANCE:' in line:
                                imp_part = line.split('IMPORTANCE:')[1]
                                if 'TAGS:' in imp_part:
                                    imp_val = imp_part.split('TAGS:')[0].strip()
                                    tags_part = imp_part.split('TAGS:')[1].strip()
                                    tags = [t.strip() for t in tags_part.split(',') if t.strip()]
                                else:
                                    imp_val = imp_part.strip()
                                
                                try:
                                    importance = float(imp_val)
                                except ValueError:
                                    importance = 0.5
                            
                            memories.append(MemoryCreate(
                                memory_type=type_part,
                                content=content_part,
                                importance_score=importance,
                                tags=tags,
                                session_id=session_id
                            ))
                    except Exception as e:
                        logger.warning(f"Error parsing memory line: {e}")
                        continue
            
            return memories
            
        except Exception as e:
            logger.error(f"Error extracting memories with LLM: {e}")
            return []
    
    async def _generate_memory_summary_with_llm(
        self,
        memories: List[MemoryResponse]
    ) -> str:
        """Generate a summary of memories using LLM."""
        try:
            memory_texts = []
            for mem in memories:
                memory_texts.append(f"[{mem.memory_type.upper()}] {mem.content}")
            
            prompt = f"""
            Create a comprehensive summary of these user memories:
            
            {chr(10).join(memory_texts)}
            
            The summary should:
            - Highlight key facts about the user
            - Note important preferences
            - Identify recurring themes or patterns
            - Be concise but informative
            - Be written in a natural, conversational tone
            
            Keep it under 300 words.
            """
            
            return await self.llm_service.generate_response(prompt)
            
        except Exception as e:
            logger.error(f"Error generating memory summary: {e}")
            return "Unable to generate summary at this time."
    
    async def _get_memory_usage_trend(
        self,
        user_id: UUID,
        db: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Get memory usage trend over time."""
        try:
            # Get memory creation counts by day for last 30 days
            trend_data = []
            for i in range(30):
                date = datetime.utcnow() - timedelta(days=i)
                start_date = date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_date = start_date + timedelta(days=1)
                
                result = await db.execute(
                    select(func.count(ConversationMemory.id))
                    .where(
                        and_(
                            ConversationMemory.user_id == user_id,
                            ConversationMemory.created_at >= start_date,
                            ConversationMemory.created_at < end_date
                        )
                    )
                )
                count = result.scalar()
                
                trend_data.append({
                    "date": start_date.strftime("%Y-%m-%d"),
                    "count": count
                })
            
            return list(reversed(trend_data))
            
        except Exception as e:
            logger.error(f"Error getting memory usage trend: {e}")
            return []
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity (0.0 to 1.0)."""
        try:
            # Simple Jaccard similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            if not union:
                return 0.0
            
            return len(intersection) / len(union)
            
        except Exception:
            return 0.0


