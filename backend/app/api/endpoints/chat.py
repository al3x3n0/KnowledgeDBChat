"""
Chat-related API endpoints.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.database import get_db
from app.core.rate_limit import limiter, CHAT_LIMIT
from app.models.chat import ChatSession, ChatMessage
from app.models.user import User
from app.models.memory import UserPreferences
from app.services.chat_service import ChatService
from app.services.llm_service import UserLLMSettings
from app.services.auth_service import get_current_user
from sqlalchemy import select
from app.schemas.chat import (
    ChatSessionCreate,
    ChatSessionResponse,
    ChatMessageCreate,
    ChatMessageResponse,
    ChatQuery
)
from app.schemas.common import PaginatedResponse

router = APIRouter()
chat_service = ChatService()


@router.post("/sessions", response_model=ChatSessionResponse)
@limiter.limit(CHAT_LIMIT)
async def create_chat_session(
    request: Request,
    session_data: ChatSessionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new chat session.
    
    Args:
        request: FastAPI request object (for rate limiting)
        session_data: Chat session creation data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Created chat session response
    """
    from app.core.logging import log_error
    
    try:
        session = await chat_service.create_session(
            user_id=current_user.id,
            title=session_data.title,
            db=db
        )
        # Set messages to empty list for new session (prevents lazy loading issues)
        session.messages = []
        return ChatSessionResponse.from_orm(session)
    except Exception as e:
        log_error(e, context={"user_id": str(current_user.id)})
        raise HTTPException(status_code=500, detail="Failed to create chat session")


@router.get("/sessions", response_model=PaginatedResponse[ChatSessionResponse])
async def get_chat_sessions(
    page: int = 1,
    page_size: int = 20,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get paginated list of chat sessions for the current user.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page (1-100)
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Paginated response with chat sessions
    """
    from app.utils.exceptions import ValidationError
    from app.core.logging import log_error
    
    try:
        # Validate pagination parameters
        if page < 1:
            raise ValidationError("Page must be >= 1", field="page")
        if page_size < 1 or page_size > 100:
            raise ValidationError("Page size must be between 1 and 100", field="page_size")
        
        # Calculate skip
        skip = (page - 1) * page_size
        
        # Get sessions with total count
        sessions, total = await chat_service.get_user_sessions(
            user_id=current_user.id,
            db=db,
            skip=skip,
            limit=page_size
        )
        
        logger.info(f"Retrieved {len(sessions)} sessions for user {current_user.id} (total: {total}, page: {page}, page_size: {page_size})")
        
        # Convert to response models
        # For list view, set messages to empty list to prevent lazy loading issues
        items = []
        for session in sessions:
            session.messages = []
            items.append(ChatSessionResponse.from_orm(session))
        
        logger.debug(f"Converted {len(items)} sessions to response models")
        
        # Return paginated response
        return PaginatedResponse.create(
            items=items,
            total=total,
            page=page,
            page_size=page_size
        )
    except ValidationError:
        raise
    except Exception as e:
        log_error(e, context={"user_id": str(current_user.id), "page": page})
        raise HTTPException(status_code=500, detail="Failed to retrieve chat sessions")


@router.get("/sessions/{session_id}", response_model=ChatSessionResponse)
async def get_chat_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific chat session with messages."""
    try:
        session = await chat_service.get_session_with_messages(
            session_id=session_id,
            user_id=current_user.id,
            db=db
        )
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        return ChatSessionResponse.from_orm(session)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving chat session: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat session")


@router.post("/sessions/{session_id}/messages", response_model=ChatMessageResponse)
@limiter.limit(CHAT_LIMIT)
async def send_message(
    request: Request,
    session_id: UUID,
    message_data: ChatMessageCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Send a message in a chat session and generate AI response.
    
    Args:
        request: FastAPI request object (for rate limiting)
        session_id: Chat session ID
        message_data: Message creation data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Chat message response with AI-generated reply
    """
    from app.core.logging import log_error
    
    try:
        # Verify session belongs to user
        session = await chat_service.get_session(
            session_id=session_id,
            user_id=current_user.id,
            db=db
        )
        if not session:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        # Create user message
        user_message = await chat_service.create_message(
            session_id=session_id,
            content=message_data.content,
            role="user",
            db=db
        )

        # Load user LLM preferences
        prefs_result = await db.execute(
            select(UserPreferences).where(UserPreferences.user_id == current_user.id)
        )
        user_prefs = prefs_result.scalar_one_or_none()
        user_settings = UserLLMSettings.from_preferences(user_prefs)

        # Generate AI response
        ai_response = await chat_service.generate_response(
            session_id=session_id,
            query=message_data.content,
            user_id=current_user.id,
            db=db,
            user_settings=user_settings,
        )
        
        return ChatMessageResponse.from_orm(ai_response)
    
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, context={"session_id": str(session_id), "user_id": str(current_user.id)})
        raise HTTPException(status_code=500, detail="Failed to send message")


@router.websocket("/sessions/{session_id}/ws")
async def websocket_chat(
    websocket: WebSocket,
    session_id: UUID,
    db: AsyncSession = Depends(get_db)
):
    """WebSocket endpoint for real-time chat."""
    # Authenticate WebSocket connection
    from app.utils.websocket_auth import require_websocket_auth
    
    try:
        user = await require_websocket_auth(websocket)
        logger.info(f"WebSocket authenticated for user {user.id}, session {session_id}")
    except WebSocketDisconnect:
        logger.warning(f"WebSocket authentication failed for session {session_id}")
        return
    
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            query = data.get("message", "")
            
            if not query:
                await websocket.send_json({
                    "type": "error",
                    "message": "Empty message received"
                })
                continue
            
            # Send typing indicator
            await websocket.send_json({
                "type": "typing",
                "message": "AI is thinking..."
            })
            
            try:
                # Verify session belongs to authenticated user
                session = await chat_service.get_session(
                    session_id=session_id,
                    user_id=user.id,
                    db=db
                )
                if not session:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Chat session not found or access denied"
                    })
                    continue

                # Load user LLM preferences
                prefs_result = await db.execute(
                    select(UserPreferences).where(UserPreferences.user_id == user.id)
                )
                user_prefs = prefs_result.scalar_one_or_none()
                user_settings = UserLLMSettings.from_preferences(user_prefs)

                # Generate response
                response = await chat_service.generate_response(
                    session_id=session_id,
                    query=query,
                    user_id=user.id,
                    db=db,
                    user_settings=user_settings,
                )
                
                # Send response
                await websocket.send_json({
                    "type": "message",
                    "data": {
                        "id": str(response.id),
                        "content": response.content,
                        "role": response.role,
                        "created_at": response.created_at.isoformat(),
                        "source_documents": response.source_documents,
                        "response_time": response.response_time
                    }
                })
                
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": "Failed to generate response"
                })
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a chat session (soft delete)."""
    from app.core.logging import log_error
    
    try:
        logger.info(f"Delete request for session {session_id} by user {current_user.id}")
        success = await chat_service.delete_session(
            session_id=session_id,
            user_id=current_user.id,
            db=db
        )
        
        if not success:
            logger.warning(f"Session {session_id} not found or doesn't belong to user {current_user.id}")
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        logger.info(f"Successfully deleted session {session_id}")
        return {"message": "Chat session deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, context={"session_id": str(session_id), "user_id": str(current_user.id)})
        raise HTTPException(status_code=500, detail="Failed to delete chat session")


@router.put("/messages/{message_id}/feedback")
async def provide_feedback(
    message_id: UUID,
    rating: int,
    feedback: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Provide feedback on a chat message."""
    try:
        success = await chat_service.update_message_feedback(
            message_id=message_id,
            rating=rating,
            feedback=feedback,
            db=db
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Message not found")
        
        return {"message": "Feedback submitted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback")
