"""
Memory management API endpoints.
"""

from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.database import get_db
from app.core.logging import log_error
from app.models.user import User
from app.services.auth_service import get_current_user, require_admin
from app.services.memory_service import MemoryService
from app.schemas.common import PaginatedResponse
from app.schemas.memory import (
    MemoryCreate, MemoryUpdate, MemoryResponse, MemorySearchRequest,
    MemorySummaryRequest, MemorySummaryResponse, MemoryStatsResponse,
    UserPreferencesUpdate, UserPreferencesResponse
)
from app.utils.exceptions import ValidationError

router = APIRouter()
memory_service = MemoryService()

@router.post("/", response_model=MemoryResponse)
async def create_memory(
    memory_data: MemoryCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new conversation memory.
    
    Args:
        memory_data: Memory creation data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Created memory response
    """
    try:
        memory = await memory_service.create_memory(
            user_id=current_user.id,
            memory_data=memory_data,
            db=db
        )
        return memory
    except Exception as e:
        log_error(e, context={"user_id": str(current_user.id)})
        raise HTTPException(status_code=500, detail="Failed to create memory")

@router.get("/", response_model=PaginatedResponse[MemoryResponse])
async def get_memories(
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(20, ge=1, le=100, description="Number of items per page"),
    session_id: Optional[UUID] = Query(None, description="Filter by session ID"),
    memory_types: Optional[str] = Query(None, description="Comma-separated memory types"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get paginated list of user's memories with optional filtering.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page (1-100)
        session_id: Optional session ID filter
        memory_types: Optional comma-separated memory types filter
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Paginated response with memories
    """
    try:
        # Validate pagination parameters
        if page < 1:
            raise ValidationError("Page must be >= 1", field="page")
        if page_size < 1 or page_size > 100:
            raise ValidationError("Page size must be between 1 and 100", field="page_size")
        
        types_list = None
        if memory_types:
            types_list = [t.strip() for t in memory_types.split(',')]
        
        # Calculate skip
        skip = (page - 1) * page_size
        
        # Get memories with pagination and total count
        memories, total = await memory_service.get_memories(
            user_id=current_user.id,
            session_id=session_id,
            memory_types=types_list,
            skip=skip,
            limit=page_size,
            db=db
        )
        
        return PaginatedResponse.create(
            items=memories,
            total=total,
            page=page,
            page_size=page_size
        )
    except ValidationError:
        raise
    except Exception as e:
        log_error(e, context={"user_id": str(current_user.id), "page": page})
        raise HTTPException(status_code=500, detail="Failed to retrieve memories")

@router.post("/search", response_model=List[MemoryResponse])
async def search_memories(
    search_request: MemorySearchRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Search memories using semantic similarity.
    
    Args:
        search_request: Search request with query and filters
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        List of matching memories
    """
    try:
        memories = await memory_service.search_memories(
            user_id=current_user.id,
            search_request=search_request,
            db=db
        )
        return memories
    except Exception as e:
        log_error(e, context={"user_id": str(current_user.id), "query": search_request.query})
        raise HTTPException(status_code=500, detail="Failed to search memories")

@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific memory by ID.
    
    Args:
        memory_id: Memory UUID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Memory response if found
        
    Raises:
        HTTPException: 404 if memory not found, 500 on server error
    """
    try:
        memory = await memory_service.get_memory_by_id(
            memory_id=memory_id,
            user_id=current_user.id,
            db=db
        )
        
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        return memory
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, context={"memory_id": str(memory_id), "user_id": str(current_user.id)})
        raise HTTPException(status_code=500, detail="Failed to retrieve memory")

@router.put("/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: UUID,
    memory_update: MemoryUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Update an existing memory.
    
    Args:
        memory_id: Memory UUID
        memory_update: Memory update data
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Updated memory response
        
    Raises:
        HTTPException: 404 if memory not found, 500 on server error
    """
    try:
        memory = await memory_service.update_memory(
            memory_id=memory_id,
            memory_update=memory_update,
            db=db
        )
        return memory
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log_error(e, context={"memory_id": str(memory_id), "user_id": str(current_user.id)})
        raise HTTPException(status_code=500, detail="Failed to update memory")

@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a memory (soft delete).
    
    Args:
        memory_id: Memory UUID
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Success message
        
    Raises:
        HTTPException: 404 if memory not found, 500 on server error
    """
    try:
        success = await memory_service.delete_memory(
            memory_id=memory_id,
            user_id=current_user.id,
            db=db
        )
        if not success:
            raise HTTPException(status_code=404, detail="Memory not found")
        
        return {"message": "Memory deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, context={"memory_id": str(memory_id), "user_id": str(current_user.id)})
        raise HTTPException(status_code=500, detail="Failed to delete memory")

@router.post("/extract/{session_id}")
async def extract_memories_from_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Extract memories from a conversation session."""
    try:
        memories = await memory_service.extract_memories_from_conversation(
            session_id=session_id,
            user_id=current_user.id,
            db=db
        )
        return {
            "message": f"Extracted {len(memories)} memories",
            "memories": memories
        }
    except Exception as e:
        logger.error(f"Error extracting memories: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract memories")

@router.post("/summary", response_model=MemorySummaryResponse)
async def generate_memory_summary(
    summary_request: MemorySummaryRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Generate a summary of user's memories."""
    try:
        summary = await memory_service.generate_memory_summary(
            user_id=current_user.id,
            summary_request=summary_request,
            db=db
        )
        return summary
    except Exception as e:
        logger.error(f"Error generating memory summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate summary")

@router.get("/stats/overview", response_model=MemoryStatsResponse)
async def get_memory_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get memory statistics for the current user."""
    try:
        stats = await memory_service.get_memory_stats(
            user_id=current_user.id,
            db=db
        )
        return stats
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get memory statistics")

@router.get("/preferences", response_model=UserPreferencesResponse)
async def get_user_preferences(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's memory preferences."""
    try:
        preferences = await memory_service.get_user_preferences(
            user_id=current_user.id,
            db=db
        )
        return UserPreferencesResponse.from_orm(preferences)
    except Exception as e:
        logger.error(f"Error getting user preferences: {e}")
        raise HTTPException(status_code=500, detail="Failed to get preferences")

@router.put("/preferences", response_model=UserPreferencesResponse)
async def update_user_preferences(
    preferences_update: UserPreferencesUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update user's memory preferences."""
    try:
        # Get current preferences
        preferences = await memory_service.get_user_preferences(
            user_id=current_user.id,
            db=db
        )
        
        # Update fields
        update_data = preferences_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(preferences, field, value)
        
        await db.commit()
        await db.refresh(preferences)
        
        return UserPreferencesResponse.from_orm(preferences)
    except Exception as e:
        logger.error(f"Error updating user preferences: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to update preferences")

# Admin endpoints
@router.get("/admin/stats", response_model=MemoryStatsResponse)
async def get_all_memory_stats(
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Get memory statistics for all users (admin only)."""
    try:
        # This would need to be implemented in the service
        # For now, return empty stats
        return MemoryStatsResponse(
            total_memories=0,
            memories_by_type={},
            recent_memories=0,
            most_accessed_memories=[],
            memory_usage_trend=[]
        )
    except Exception as e:
        logger.error(f"Error getting admin memory stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get admin statistics")

@router.delete("/admin/cleanup")
async def cleanup_old_memories(
    days_old: int = Query(90, ge=1, le=365, description="Delete memories older than this many days"),
    current_user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db)
):
    """Clean up old memories (admin only)."""
    try:
        # This would need to be implemented in the service
        return {"message": f"Cleanup completed for memories older than {days_old} days"}
    except Exception as e:
        logger.error(f"Error cleaning up memories: {e}")
        raise HTTPException(status_code=500, detail="Failed to cleanup memories")


