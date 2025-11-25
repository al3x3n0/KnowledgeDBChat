"""
Background tasks for chat-related operations.
"""

import asyncio
from typing import Dict, Any
from uuid import UUID
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.celery import celery_app
from app.core.database import AsyncSessionLocal
from app.core.config import settings
from app.models.chat import ChatSession, ChatMessage
from app.services.chat_service import ChatService
from app.services.llm_service import LLMService
from sqlalchemy import select, and_
from sqlalchemy.orm import selectinload


@celery_app.task(bind=True, name="app.tasks.chat_tasks.generate_chat_title")
def generate_chat_title(self, session_id: str) -> Dict[str, Any]:
    """
    Generate a chat session title using LLM based on the first few messages.
    
    Args:
        session_id: UUID of the chat session
        
    Returns:
        Dict with generation results
    """
    return asyncio.run(_async_generate_chat_title(self, session_id))


async def _async_generate_chat_title(task, session_id: str) -> Dict[str, Any]:
    """Async implementation of chat title generation."""
    async with AsyncSessionLocal() as db:
        try:
            logger.info(f"Starting title generation for session {session_id}")
            
            # Get the session
            result = await db.execute(
                select(ChatSession).where(ChatSession.id == UUID(session_id))
            )
            session = result.scalar_one_or_none()
            
            if not session:
                logger.warning(f"Session {session_id} not found for title generation")
                return {
                    "session_id": session_id,
                    "success": False,
                    "error": "Session not found"
                }
            
            # Check if title is already generated (not a default date-based title)
            # Allow regeneration if title starts with "Chat " (default format) or matches date pattern
            if session.title and not session.title.startswith("Chat ") and " - " in session.title:
                # Title already has date format with generated content, skip
                logger.info(f"Session {session_id} already has a generated title: {session.title}")
                return {
                    "session_id": session_id,
                    "success": True,
                    "title": session.title,
                    "message": "Title already exists"
                }
            
            # Get first few messages (user + assistant pairs)
            messages_result = await db.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == UUID(session_id))
                .order_by(ChatMessage.created_at)
                .limit(6)  # Get first 3 exchanges (6 messages)
            )
            messages = messages_result.scalars().all()
            
            if len(messages) < 2:
                logger.info(f"Not enough messages in session {session_id} for title generation")
                return {
                    "session_id": session_id,
                    "success": False,
                    "error": "Not enough messages"
                }
            
            # Build conversation context for title generation
            conversation_text = "\n".join([
                f"{msg.role.capitalize()}: {msg.content[:200]}"  # Limit to first 200 chars per message
                for msg in messages
            ])
            
            # Generate title using LLM
            llm_service = LLMService()
            title_prompt = f"""Generate a concise title (3-6 words) for this conversation. Only return the title, nothing else.

Conversation:
{conversation_text}

Title:"""
            
            try:
                generated_title = await llm_service.generate_response(
                    query=title_prompt,
                    model=settings.DEFAULT_MODEL,
                    temperature=0.5,  # Lower temperature for more consistent titles
                    max_tokens=30
                )
                
                # Clean up the title (remove quotes, extra whitespace, newlines, etc.)
                generated_title = generated_title.strip()
                # Remove quotes if present
                if generated_title.startswith('"') and generated_title.endswith('"'):
                    generated_title = generated_title[1:-1].strip()
                if generated_title.startswith("'") and generated_title.endswith("'"):
                    generated_title = generated_title[1:-1].strip()
                
                # Remove "Title:" prefix if present
                if generated_title.lower().startswith("title:"):
                    generated_title = generated_title[6:].strip()
                
                # Remove newlines and extra spaces
                generated_title = " ".join(generated_title.split())
                
                # Limit title length to 60 characters
                if len(generated_title) > 60:
                    generated_title = generated_title[:60].rsplit(' ', 1)[0]  # Cut at word boundary
                
                # If title is empty or too short, use fallback
                if not generated_title or len(generated_title) < 3:
                    raise ValueError("Generated title is too short or empty")
                
                # Format as "date + generated title"
                date_str = session.created_at.strftime('%Y-%m-%d')
                formatted_title = f"{date_str} - {generated_title}"
                
                # Update session title
                session.title = formatted_title
                await db.commit()
                await db.refresh(session)
                
                logger.info(f"Generated title for session {session_id}: {formatted_title}")
                
                return {
                    "session_id": session_id,
                    "success": True,
                    "title": formatted_title,
                    "generated_title": generated_title
                }
                
            except Exception as e:
                logger.error(f"Error generating title with LLM for session {session_id}: {e}", exc_info=True)
                # Fallback to date-based title if LLM fails
                date_str = session.created_at.strftime('%Y-%m-%d')
                fallback_title = f"{date_str} - Chat"
                session.title = fallback_title
                await db.commit()
                await db.refresh(session)
                
                return {
                    "session_id": session_id,
                    "success": False,
                    "error": str(e),
                    "fallback_title": fallback_title
                }
            
        except Exception as e:
            logger.error(f"Error generating chat title for session {session_id}: {e}", exc_info=True)
            return {
                "session_id": session_id,
                "success": False,
                "error": str(e)
            }

