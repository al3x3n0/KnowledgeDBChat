"""
MCP Configuration API endpoints.

Allows users to configure MCP tools and access control for their API keys.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.database import get_db
from app.models.api_key import APIKey
from app.models.mcp_config import MCPToolConfig, MCPSourceAccess, MCP_TOOLS
from app.models.document import DocumentSource
from app.models.user import User
from app.api.endpoints.users import get_current_user


router = APIRouter()


# =============================================================================
# Pydantic Models
# =============================================================================

class MCPToolInfo(BaseModel):
    """Information about an MCP tool."""
    name: str
    display_name: str
    description: str
    category: str
    required_scope: str
    config_schema: dict


class MCPToolConfigUpdate(BaseModel):
    """Update tool configuration for an API key."""
    tool_name: str
    is_enabled: bool = True
    config: Optional[dict] = None


class MCPToolConfigResponse(BaseModel):
    """Tool configuration response."""
    tool_name: str
    display_name: str
    description: str
    category: str
    is_enabled: bool
    config: Optional[dict]

    class Config:
        from_attributes = True


class MCPSourceAccessUpdate(BaseModel):
    """Update source access for an API key."""
    source_id: UUID
    can_read: bool = True
    can_search: bool = True
    can_chat: bool = True


class MCPSourceAccessResponse(BaseModel):
    """Source access configuration response."""
    source_id: UUID
    source_name: str
    source_type: str
    can_read: bool
    can_search: bool
    can_chat: bool

    class Config:
        from_attributes = True


class MCPKeyConfigUpdate(BaseModel):
    """Update MCP configuration for an API key."""
    mcp_enabled: Optional[bool] = None
    allowed_tools: Optional[List[str]] = None  # None = all tools, empty list = no tools
    source_access_mode: Optional[str] = None  # "all" or "restricted"


class MCPKeyConfigResponse(BaseModel):
    """Full MCP configuration for an API key."""
    api_key_id: UUID
    api_key_name: str
    mcp_enabled: bool
    allowed_tools: Optional[List[str]]
    source_access_mode: str
    tool_configs: List[MCPToolConfigResponse]
    source_access: List[MCPSourceAccessResponse]


# =============================================================================
# Endpoints
# =============================================================================

@router.get("/tools", response_model=List[MCPToolInfo])
async def list_available_tools():
    """
    List all available MCP tools.

    Returns tool definitions with their configuration schemas.
    """
    return [
        MCPToolInfo(
            name=tool["name"],
            display_name=tool["display_name"],
            description=tool["description"],
            category=tool["category"],
            required_scope=tool["required_scope"],
            config_schema=tool["config_schema"],
        )
        for tool in MCP_TOOLS.values()
    ]


@router.get("/keys/{key_id}/config", response_model=MCPKeyConfigResponse)
async def get_mcp_key_config(
    key_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get MCP configuration for a specific API key.
    """
    # Get the API key
    result = await db.execute(
        select(APIKey).where(
            APIKey.id == key_id,
            APIKey.user_id == current_user.id,
        )
    )
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )

    # Get tool configs
    tool_configs_result = await db.execute(
        select(MCPToolConfig).where(MCPToolConfig.api_key_id == key_id)
    )
    tool_configs_db = {tc.tool_name: tc for tc in tool_configs_result.scalars().all()}

    # Build tool configs list
    tool_configs = []
    for tool_name, tool_info in MCP_TOOLS.items():
        tc = tool_configs_db.get(tool_name)
        tool_configs.append(MCPToolConfigResponse(
            tool_name=tool_name,
            display_name=tool_info["display_name"],
            description=tool_info["description"],
            category=tool_info["category"],
            is_enabled=tc.is_enabled if tc else True,
            config=tc.config if tc else None,
        ))

    # Get source access
    source_access_result = await db.execute(
        select(MCPSourceAccess, DocumentSource)
        .join(DocumentSource, MCPSourceAccess.source_id == DocumentSource.id)
        .where(MCPSourceAccess.api_key_id == key_id)
    )
    source_access = [
        MCPSourceAccessResponse(
            source_id=sa.source_id,
            source_name=src.name,
            source_type=src.source_type,
            can_read=sa.can_read,
            can_search=sa.can_search,
            can_chat=sa.can_chat,
        )
        for sa, src in source_access_result.all()
    ]

    return MCPKeyConfigResponse(
        api_key_id=api_key.id,
        api_key_name=api_key.name,
        mcp_enabled=api_key.mcp_enabled if api_key.mcp_enabled is not None else True,
        allowed_tools=api_key.allowed_tools,
        source_access_mode=api_key.source_access_mode or "all",
        tool_configs=tool_configs,
        source_access=source_access,
    )


@router.patch("/keys/{key_id}/config", response_model=MCPKeyConfigResponse)
async def update_mcp_key_config(
    key_id: UUID,
    config: MCPKeyConfigUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update MCP configuration for an API key.
    """
    # Get the API key
    result = await db.execute(
        select(APIKey).where(
            APIKey.id == key_id,
            APIKey.user_id == current_user.id,
        )
    )
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )

    # Update fields
    if config.mcp_enabled is not None:
        api_key.mcp_enabled = config.mcp_enabled

    if config.allowed_tools is not None:
        # Validate tool names
        invalid_tools = [t for t in config.allowed_tools if t not in MCP_TOOLS]
        if invalid_tools:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid tool names: {invalid_tools}"
            )
        api_key.allowed_tools = config.allowed_tools if config.allowed_tools else None

    if config.source_access_mode is not None:
        if config.source_access_mode not in ("all", "restricted"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="source_access_mode must be 'all' or 'restricted'"
            )
        api_key.source_access_mode = config.source_access_mode

    await db.commit()
    await db.refresh(api_key)

    logger.info(f"Updated MCP config for API key {key_id}")

    # Return full config
    return await get_mcp_key_config(key_id, current_user, db)


@router.put("/keys/{key_id}/tools/{tool_name}", response_model=MCPToolConfigResponse)
async def update_tool_config(
    key_id: UUID,
    tool_name: str,
    config: MCPToolConfigUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update configuration for a specific tool on an API key.
    """
    # Verify API key ownership
    result = await db.execute(
        select(APIKey).where(
            APIKey.id == key_id,
            APIKey.user_id == current_user.id,
        )
    )
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )

    # Validate tool name
    if tool_name not in MCP_TOOLS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tool name: {tool_name}"
        )

    # Get or create tool config
    result = await db.execute(
        select(MCPToolConfig).where(
            MCPToolConfig.api_key_id == key_id,
            MCPToolConfig.tool_name == tool_name,
        )
    )
    tool_config = result.scalar_one_or_none()

    if tool_config:
        tool_config.is_enabled = config.is_enabled
        tool_config.config = config.config
    else:
        tool_config = MCPToolConfig(
            api_key_id=key_id,
            tool_name=tool_name,
            is_enabled=config.is_enabled,
            config=config.config,
        )
        db.add(tool_config)

    await db.commit()
    await db.refresh(tool_config)

    tool_info = MCP_TOOLS[tool_name]
    return MCPToolConfigResponse(
        tool_name=tool_name,
        display_name=tool_info["display_name"],
        description=tool_info["description"],
        category=tool_info["category"],
        is_enabled=tool_config.is_enabled,
        config=tool_config.config,
    )


@router.get("/keys/{key_id}/sources", response_model=List[MCPSourceAccessResponse])
async def list_source_access(
    key_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List source access configuration for an API key.
    """
    # Verify API key ownership
    result = await db.execute(
        select(APIKey).where(
            APIKey.id == key_id,
            APIKey.user_id == current_user.id,
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )

    # Get source access
    result = await db.execute(
        select(MCPSourceAccess, DocumentSource)
        .join(DocumentSource, MCPSourceAccess.source_id == DocumentSource.id)
        .where(MCPSourceAccess.api_key_id == key_id)
    )

    return [
        MCPSourceAccessResponse(
            source_id=sa.source_id,
            source_name=src.name,
            source_type=src.source_type,
            can_read=sa.can_read,
            can_search=sa.can_search,
            can_chat=sa.can_chat,
        )
        for sa, src in result.all()
    ]


@router.put("/keys/{key_id}/sources/{source_id}", response_model=MCPSourceAccessResponse)
async def update_source_access(
    key_id: UUID,
    source_id: UUID,
    access: MCPSourceAccessUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update or create source access for an API key.
    """
    # Verify API key ownership
    result = await db.execute(
        select(APIKey).where(
            APIKey.id == key_id,
            APIKey.user_id == current_user.id,
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )

    # Verify source exists
    result = await db.execute(
        select(DocumentSource).where(DocumentSource.id == source_id)
    )
    source = result.scalar_one_or_none()
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document source not found"
        )

    # Get or create source access
    result = await db.execute(
        select(MCPSourceAccess).where(
            MCPSourceAccess.api_key_id == key_id,
            MCPSourceAccess.source_id == source_id,
        )
    )
    source_access = result.scalar_one_or_none()

    if source_access:
        source_access.can_read = access.can_read
        source_access.can_search = access.can_search
        source_access.can_chat = access.can_chat
    else:
        source_access = MCPSourceAccess(
            api_key_id=key_id,
            source_id=source_id,
            can_read=access.can_read,
            can_search=access.can_search,
            can_chat=access.can_chat,
        )
        db.add(source_access)

    await db.commit()

    return MCPSourceAccessResponse(
        source_id=source_id,
        source_name=source.name,
        source_type=source.source_type,
        can_read=source_access.can_read,
        can_search=source_access.can_search,
        can_chat=source_access.can_chat,
    )


@router.delete("/keys/{key_id}/sources/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_source_access(
    key_id: UUID,
    source_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Remove source access restriction for an API key.
    """
    # Verify API key ownership
    result = await db.execute(
        select(APIKey).where(
            APIKey.id == key_id,
            APIKey.user_id == current_user.id,
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )

    # Delete source access
    await db.execute(
        delete(MCPSourceAccess).where(
            MCPSourceAccess.api_key_id == key_id,
            MCPSourceAccess.source_id == source_id,
        )
    )
    await db.commit()
