"""
API endpoints for user-defined custom tools.
"""

from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from loguru import logger

from app.core.database import get_db
from app.models.user import User
from app.models.workflow import UserTool
from app.services.auth_service import get_current_user
from app.services.custom_tool_service import CustomToolService, ToolExecutionError
from app.schemas.workflow import (
    UserToolCreate,
    UserToolUpdate,
    UserToolResponse,
    UserToolListResponse,
    UserToolTestRequest,
    UserToolTestResponse,
)


router = APIRouter()


@router.get("", response_model=UserToolListResponse)
async def list_user_tools(
    tool_type: Optional[str] = Query(None, description="Filter by tool type"),
    enabled_only: bool = Query(False, description="Only return enabled tools"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """List all custom tools for the current user."""
    try:
        query = select(UserTool).where(UserTool.user_id == current_user.id)

        if tool_type:
            query = query.where(UserTool.tool_type == tool_type)
        if enabled_only:
            query = query.where(UserTool.is_enabled == True)

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar() or 0

        # Get paginated results
        query = query.order_by(UserTool.name).offset(offset).limit(limit)
        result = await db.execute(query)
        tools = result.scalars().all()

        return UserToolListResponse(
            tools=[UserToolResponse.model_validate(t) for t in tools],
            total=total
        )

    except Exception as e:
        logger.error(f"Error listing user tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=UserToolResponse, status_code=201)
async def create_user_tool(
    tool_data: UserToolCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new custom tool."""
    try:
        # Check for duplicate name
        existing = await db.execute(
            select(UserTool).where(
                UserTool.user_id == current_user.id,
                UserTool.name == tool_data.name
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=400,
                detail=f"Tool with name '{tool_data.name}' already exists"
            )

        # Validate tool type
        valid_types = ["webhook", "transform", "python", "llm_prompt"]
        if tool_data.tool_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid tool type. Must be one of: {valid_types}"
            )

        # Create tool
        tool = UserTool(
            user_id=current_user.id,
            name=tool_data.name,
            description=tool_data.description,
            tool_type=tool_data.tool_type,
            parameters_schema=tool_data.parameters_schema,
            config=tool_data.config,
            is_enabled=tool_data.is_enabled,
        )

        db.add(tool)
        await db.commit()
        await db.refresh(tool)

        logger.info(f"Created tool '{tool.name}' for user {current_user.id}")
        return UserToolResponse.model_validate(tool)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user tool: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{tool_id}", response_model=UserToolResponse)
async def get_user_tool(
    tool_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific custom tool by ID."""
    result = await db.execute(
        select(UserTool).where(
            UserTool.id == tool_id,
            UserTool.user_id == current_user.id
        )
    )
    tool = result.scalar_one_or_none()

    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")

    return UserToolResponse.model_validate(tool)


@router.put("/{tool_id}", response_model=UserToolResponse)
async def update_user_tool(
    tool_id: UUID,
    tool_data: UserToolUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a custom tool."""
    try:
        result = await db.execute(
            select(UserTool).where(
                UserTool.id == tool_id,
                UserTool.user_id == current_user.id
            )
        )
        tool = result.scalar_one_or_none()

        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")

        # Check for name conflict if name is being changed
        if tool_data.name and tool_data.name != tool.name:
            existing = await db.execute(
                select(UserTool).where(
                    UserTool.user_id == current_user.id,
                    UserTool.name == tool_data.name
                )
            )
            if existing.scalar_one_or_none():
                raise HTTPException(
                    status_code=400,
                    detail=f"Tool with name '{tool_data.name}' already exists"
                )

        # Update fields
        update_data = tool_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(tool, field, value)

        # Increment version
        tool.version += 1

        await db.commit()
        await db.refresh(tool)

        logger.info(f"Updated tool '{tool.name}' (version {tool.version})")
        return UserToolResponse.model_validate(tool)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user tool: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{tool_id}", status_code=204)
async def delete_user_tool(
    tool_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a custom tool."""
    try:
        result = await db.execute(
            select(UserTool).where(
                UserTool.id == tool_id,
                UserTool.user_id == current_user.id
            )
        )
        tool = result.scalar_one_or_none()

        if not tool:
            raise HTTPException(status_code=404, detail="Tool not found")

        await db.delete(tool)
        await db.commit()

        logger.info(f"Deleted tool '{tool.name}' for user {current_user.id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user tool: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{tool_id}/test", response_model=UserToolTestResponse)
async def test_user_tool(
    tool_id: UUID,
    test_request: UserToolTestRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Test a custom tool with sample inputs."""
    result = await db.execute(
        select(UserTool).where(
            UserTool.id == tool_id,
            UserTool.user_id == current_user.id
        )
    )
    tool = result.scalar_one_or_none()

    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")

    tool_service = CustomToolService()

    try:
        result = await tool_service.execute_tool(
            tool=tool,
            inputs=test_request.inputs,
            user=current_user,
            db=db
        )

        return UserToolTestResponse(
            success=True,
            output=result.get("output"),
            error=None,
            execution_time_ms=result.get("execution_time_ms", 0)
        )

    except ToolExecutionError as e:
        return UserToolTestResponse(
            success=False,
            output=None,
            error=str(e),
            execution_time_ms=0
        )
    except Exception as e:
        logger.error(f"Tool test failed: {e}")
        return UserToolTestResponse(
            success=False,
            output=None,
            error=f"Unexpected error: {str(e)}",
            execution_time_ms=0
        )


@router.post("/{tool_id}/duplicate", response_model=UserToolResponse, status_code=201)
async def duplicate_user_tool(
    tool_id: UUID,
    new_name: str = Query(..., min_length=1, max_length=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Duplicate an existing tool with a new name."""
    try:
        # Get original tool
        result = await db.execute(
            select(UserTool).where(
                UserTool.id == tool_id,
                UserTool.user_id == current_user.id
            )
        )
        original = result.scalar_one_or_none()

        if not original:
            raise HTTPException(status_code=404, detail="Tool not found")

        # Check for name conflict
        existing = await db.execute(
            select(UserTool).where(
                UserTool.user_id == current_user.id,
                UserTool.name == new_name
            )
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=400,
                detail=f"Tool with name '{new_name}' already exists"
            )

        # Create duplicate
        duplicate = UserTool(
            user_id=current_user.id,
            name=new_name,
            description=original.description,
            tool_type=original.tool_type,
            parameters_schema=original.parameters_schema,
            config=original.config,
            is_enabled=original.is_enabled,
        )

        db.add(duplicate)
        await db.commit()
        await db.refresh(duplicate)

        logger.info(f"Duplicated tool '{original.name}' as '{new_name}'")
        return UserToolResponse.model_validate(duplicate)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error duplicating tool: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
