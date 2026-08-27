"""
Agent Memory Integration Service.

Integrates the memory system with agent processing by:
- Fetching relevant memories for context injection
- Formatting memories for agent prompts
- Extracting and storing memories from conversations
- Tracking memory usage in conversations
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_definition import AgentMemoryInjection
from app.models.memory import ConversationMemory, UserPreferences
from app.services.llm_service import LLMService
from app.services.memory_service import MemoryService


class AgentMemoryIntegration:
    """
    Integrates memory system with agent processing.

    Handles:
    - Retrieving relevant memories based on conversation context
    - Formatting memories for injection into agent prompts
    - Extracting and storing new memories from conversations
    - Tracking which memories were injected
    """

    def __init__(self, memory_service: Optional[MemoryService] = None):
        self.memory_service = memory_service or MemoryService()
        self.llm_service = LLMService()

    async def get_relevant_memories(
        self,
        user_id: UUID,
        message: str,
        conversation_id: Optional[UUID],
        preferences: UserPreferences,
        db: AsyncSession,
    ) -> List[ConversationMemory]:
        """
        Fetch memories relevant to the current message context.

        Args:
            user_id: User whose memories to search
            message: Current user message to match against
            conversation_id: Current conversation ID (for tracking)
            preferences: User's memory preferences
            db: Database session

        Returns:
            List of relevant memories, sorted by relevance
        """
        if not preferences.enable_agent_memory:
            return []

        try:
            # Get memory types to inject
            allowed_types = preferences.memory_injection_types or [
                "fact",
                "preference",
                "context",
            ]
            max_memories = preferences.max_injected_memories or 5

            # Search memories using the memory service
            memories = await self.memory_service.search_memories(
                user_id=user_id,
                query=message,
                memory_types=allowed_types,
                limit=max_memories * 2,  # Get extra for relevance filtering
                db=db,
            )

            # Filter to top N by relevance (already sorted by search_memories)
            relevant_memories = memories[:max_memories]

            logger.info(
                f"Found {len(relevant_memories)} relevant memories for user {user_id}"
            )

            return relevant_memories

        except Exception as e:
            logger.error(f"Error fetching relevant memories: {e}")
            return []

    def format_memories_for_prompt(
        self, memories: List[ConversationMemory], include_metadata: bool = False
    ) -> str:
        """
        Format memories as a context section for agent prompts.

        Args:
            memories: List of memories to format
            include_metadata: Whether to include importance scores and timestamps

        Returns:
            Formatted string to inject into agent prompt
        """
        if not memories:
            return ""

        lines = ["## User Context from Previous Conversations\n"]

        for mem in memories:
            # Format type as uppercase tag
            type_tag = f"[{mem.memory_type.upper()}]"

            # Build the line
            line = f"- {type_tag} {mem.content}"

            # Optionally add metadata
            if include_metadata:
                line += f" (importance: {mem.importance_score:.1f})"

            lines.append(line)

        # Add instruction for agent
        lines.append("")
        lines.append("*Use this context to personalize your responses when relevant.*")

        return "\n".join(lines)

    async def extract_and_store_memories(
        self,
        user_id: UUID,
        conversation_id: UUID,
        messages: List[Dict[str, Any]],
        preferences: UserPreferences,
        db: AsyncSession,
    ) -> List[ConversationMemory]:
        """
        Extract and store new memories from conversation messages.

        This runs asynchronously after the agent responds, not blocking the response.

        Args:
            user_id: User ID
            conversation_id: Conversation ID for context
            messages: Recent messages to extract from (user + assistant)
            preferences: User's memory preferences
            db: Database session

        Returns:
            List of newly created memories
        """
        if not preferences.enable_agent_memory:
            return []

        if not preferences.allow_personal_data_storage:
            return []

        try:
            # Format messages for extraction
            message_texts = []
            for msg in messages[-10:]:  # Last 10 messages
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if content:
                    message_texts.append(f"{role}: {content}")

            if not message_texts:
                return []

            # Use memory service to extract memories
            new_memories = await self.memory_service.extract_memories_from_conversation(
                user_id=user_id, messages=message_texts, db=db
            )

            logger.info(
                f"Extracted {len(new_memories)} memories from conversation {conversation_id}"
            )

            return new_memories

        except Exception as e:
            logger.error(f"Error extracting memories: {e}")
            return []

    async def record_memory_injection(
        self,
        conversation_id: UUID,
        memory_id: UUID,
        turn_number: int,
        relevance_score: float,
        injection_type: str,
        db: AsyncSession,
    ) -> AgentMemoryInjection:
        """
        Record that a memory was injected into a conversation turn.

        Args:
            conversation_id: Conversation the memory was injected into
            memory_id: ID of the injected memory
            turn_number: Which turn in the conversation
            relevance_score: How relevant the memory was scored
            injection_type: "automatic", "manual", or "shared"
            db: Database session

        Returns:
            The created injection record
        """
        try:
            injection = AgentMemoryInjection(
                conversation_id=conversation_id,
                memory_id=memory_id,
                turn_number=turn_number,
                relevance_score=relevance_score,
                injection_type=injection_type,
            )
            db.add(injection)
            await db.commit()
            await db.refresh(injection)

            # Update memory access tracking
            await self._update_memory_access(memory_id, db)

            return injection

        except Exception as e:
            logger.error(f"Error recording memory injection: {e}")
            await db.rollback()
            raise

    async def record_memory_injections_batch(
        self,
        conversation_id: UUID,
        memories: List[ConversationMemory],
        turn_number: int,
        injection_type: str,
        db: AsyncSession,
    ) -> List[AgentMemoryInjection]:
        """
        Record multiple memory injections at once.

        Args:
            conversation_id: Conversation ID
            memories: List of memories that were injected
            turn_number: Turn number in conversation
            injection_type: Type of injection
            db: Database session

        Returns:
            List of created injection records
        """
        injections = []

        for i, mem in enumerate(memories):
            # Use list position as proxy for relevance (already sorted)
            relevance = 1.0 - (i * 0.1)  # First is 1.0, second is 0.9, etc.
            relevance = max(0.1, relevance)

            injection = AgentMemoryInjection(
                conversation_id=conversation_id,
                memory_id=mem.id,
                turn_number=turn_number,
                relevance_score=relevance,
                injection_type=injection_type,
            )
            db.add(injection)
            injections.append(injection)

            # Update access tracking
            mem.last_accessed_at = datetime.utcnow()
            mem.access_count = (mem.access_count or 0) + 1

        try:
            await db.commit()
            return injections
        except Exception as e:
            logger.error(f"Error recording batch memory injections: {e}")
            await db.rollback()
            return []

    async def _update_memory_access(self, memory_id: UUID, db: AsyncSession) -> None:
        """Update memory access timestamp and count."""
        try:
            result = await db.execute(
                select(ConversationMemory).where(ConversationMemory.id == memory_id)
            )
            memory = result.scalar_one_or_none()

            if memory:
                memory.last_accessed_at = datetime.utcnow()
                memory.access_count = (memory.access_count or 0) + 1
                await db.commit()

        except Exception as e:
            logger.error(f"Error updating memory access: {e}")

    async def manually_inject_memory(
        self, conversation_id: UUID, memory_id: UUID, turn_number: int, db: AsyncSession
    ) -> Optional[AgentMemoryInjection]:
        """
        Manually inject a specific memory into the conversation context.

        Used when user explicitly wants to add a memory to context.
        """
        return await self.record_memory_injection(
            conversation_id=conversation_id,
            memory_id=memory_id,
            turn_number=turn_number,
            relevance_score=1.0,  # Manual = maximum relevance
            injection_type="manual",
            db=db,
        )
