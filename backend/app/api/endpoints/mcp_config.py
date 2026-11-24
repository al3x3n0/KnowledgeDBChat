"""
MCP Configuration API endpoints.

Allows users to configure MCP tools and access control for their API keys.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent_core import research_programs
from app.core.database import get_db
from app.mcp.tools.research_notes import note_config_query, selected_note_ids
from app.models.api_key import APIKey
from app.models.document import DocumentSource
from app.models.mcp_config import (
    MCP_PROFILE_TOOLS,
    MCP_TOOLS,
    RESEARCH_NOTES_TOOL,
    RESEARCH_PROGRAM_TOOL,
    MCPSourceAccess,
    MCPToolConfig,
)
from app.models.research_note import ResearchNote
from app.models.user import User
from app.schemas.mcp_prompt import BUILT_IN_PROMPT_NAMES, MCPCustomPrompt
from app.services.api_key_service import api_key_service
from app.services.auth_service import get_current_user

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

    model_config = ConfigDict(from_attributes=True)


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

    model_config = ConfigDict(from_attributes=True)


class MCPKeyConfigUpdate(BaseModel):
    """Update MCP configuration for an API key."""

    mcp_enabled: Optional[bool] = None
    allowed_tools: Optional[List[str]] = None  # None = all tools, empty list = no tools
    source_access_mode: Optional[str] = None  # "all" or "restricted"


class MCPNoteOption(BaseModel):
    id: UUID
    title: str


class MCPKeyConfigResponse(BaseModel):
    """Full MCP configuration for an API key."""

    api_key_id: UUID
    api_key_name: str
    mcp_enabled: bool
    allowed_tools: Optional[List[str]]
    source_access_mode: str
    tool_configs: List[MCPToolConfigResponse]
    source_access: List[MCPSourceAccessResponse]
    mcp_programs: Optional[List[str]] = None
    research_notes: List[MCPNoteOption] = Field(default_factory=list)
    prompts: List[MCPCustomPrompt] = Field(default_factory=list)


class MCPProgramOption(BaseModel):
    name: str
    version: str
    description: str
    tools: List[str]


class MCPProfileOptions(BaseModel):
    programs: List[MCPProgramOption]
    tools: List[MCPToolInfo]


class MCPProfileCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=100)
    programs: List[str] = Field(default_factory=list, max_length=100)
    tools: List[str] = Field(default_factory=list, max_length=len(MCP_PROFILE_TOOLS))
    source_ids: Optional[List[UUID]] = Field(default=None, max_length=500)
    note_ids: Optional[List[UUID]] = Field(default=None, max_length=50)
    prompts: Optional[List[MCPCustomPrompt]] = Field(default=None, max_length=10)

    @field_validator("prompts")
    @classmethod
    def validate_prompt_names(
        cls, prompts: Optional[List[MCPCustomPrompt]]
    ) -> Optional[List[MCPCustomPrompt]]:
        names = [prompt.name for prompt in prompts or []]
        if len(set(names)) != len(names):
            raise ValueError("Prompt names must be unique")
        if BUILT_IN_PROMPT_NAMES.intersection(names):
            raise ValueError("Use a name distinct from the built-in prompts")
        return prompts


class MCPProfileCreated(BaseModel):
    id: UUID
    api_key: str


class MCPProfileRotation(BaseModel):
    key_prefix: str = Field(min_length=1, max_length=16)


class MCPProfileRotated(MCPProfileCreated):
    key_prefix: str


async def lock_owned_profile(db: AsyncSession, key_id: UUID, user_id: UUID) -> APIKey:
    api_key = await db.scalar(
        select(APIKey)
        .where(
            APIKey.id == key_id,
            APIKey.user_id == user_id,
            APIKey.revoked_at.is_(None),
            APIKey.mcp_programs.is_not(None),
        )
        .with_for_update()
    )
    if api_key is None:
        raise HTTPException(404, "MCP profile not found")
    return api_key


async def validate_profile_selection(
    profile: MCPProfileCreate,
    db: AsyncSession,
    note_ids: List[UUID],
    has_prompts: bool,
    existing: Optional[APIKey] = None,
) -> tuple[List[str], List[str], Optional[List[UUID]]]:
    programs = list(dict.fromkeys(profile.programs))
    tools = list(dict.fromkeys(profile.tools))
    if not programs and not tools and not note_ids and not has_prompts:
        raise HTTPException(400, "Select at least one program, note, tool, or prompt")
    available_programs = {program.name for program in research_programs.installed()}
    if existing is not None:
        available_programs.update(existing.mcp_programs)
    if set(programs) - available_programs:
        raise HTTPException(400, "One or more research programs are unavailable")
    if set(tools) - set(MCP_PROFILE_TOOLS):
        raise HTTPException(400, "One or more tools are unavailable for MCP profiles")

    sources = None
    if profile.source_ids is not None:
        sources = list(dict.fromkeys(profile.source_ids))
        active_sources = select(DocumentSource.id).where(
            DocumentSource.id.in_(sources), DocumentSource.is_active.is_(True)
        )
        available_sources = set(await db.scalars(active_sources))
        if existing is not None:
            available_sources.update(
                await db.scalars(
                    select(MCPSourceAccess.source_id).where(
                        MCPSourceAccess.api_key_id == existing.id
                    )
                )
            )
        if set(sources) - available_sources:
            raise HTTPException(400, "One or more document sources are unavailable")
    if programs:
        tools.append(RESEARCH_PROGRAM_TOOL)
    if note_ids:
        tools.append(RESEARCH_NOTES_TOOL)
    return programs, tools, sources


async def validate_note_selection(
    note_ids: List[UUID], user_id: UUID, db: AsyncSession
) -> List[UUID]:
    ids = list(dict.fromkeys(note_ids))
    if ids:
        result = await db.scalars(
            select(ResearchNote.id).where(
                ResearchNote.user_id == user_id, ResearchNote.id.in_(ids)
            )
        )
        if set(result) != set(ids):
            raise HTTPException(400, "One or more research notes are unavailable")
    return ids


def profile_scopes(tools: List[str], has_prompts: bool) -> List[str]:
    scopes = {MCP_TOOLS[name]["required_scope"] for name in tools}
    if has_prompts:
        scopes.add("read")
    return sorted(scopes)


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/profile-options", response_model=MCPProfileOptions)
async def list_mcp_profile_options(current_user: User = Depends(get_current_user)):
    return MCPProfileOptions(
        programs=[
            MCPProgramOption(
                name=program.name,
                version=program.version,
                description=program.description,
                tools=[spec.name for spec in program.tool_specs],
            )
            for program in research_programs.installed()
        ],
        tools=[MCPToolInfo(**MCP_TOOLS[name]) for name in MCP_PROFILE_TOOLS],
    )


@router.get("/profile-notes", response_model=List[MCPNoteOption])
async def list_mcp_profile_notes(
    q: str = Query("", max_length=100),
    note_id: Optional[UUID] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    notes = select(ResearchNote.id, ResearchNote.title).where(
        ResearchNote.user_id == current_user.id
    )
    if q.strip():
        notes = notes.where(ResearchNote.title.icontains(q.strip(), autoescape=True))
    if note_id is not None:
        notes = notes.where(ResearchNote.id == note_id)
    rows = await db.execute(
        notes.order_by(ResearchNote.updated_at.desc(), ResearchNote.id).limit(30)
    )
    return [MCPNoteOption(id=row.id, title=row.title) for row in rows]


@router.post(
    "/profiles", response_model=MCPProfileCreated, status_code=status.HTTP_201_CREATED
)
async def create_mcp_profile(
    profile: MCPProfileCreate,
    request: Request,
    response: Response,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if request.headers.get("X-API-Key"):
        raise HTTPException(403, "Sign in to create an MCP profile")

    note_ids = await validate_note_selection(
        profile.note_ids or [], current_user.id, db
    )
    prompts = [prompt.model_dump() for prompt in profile.prompts or []]
    programs, tools, sources = await validate_profile_selection(
        profile, db, note_ids, bool(prompts)
    )

    api_key, secret = await api_key_service.create_api_key(
        db=db,
        user_id=current_user.id,
        name=profile.name,
        scopes=profile_scopes(tools, bool(prompts)),
        allowed_tools=tools,
        mcp_programs=programs,
        mcp_source_ids=sources,
        mcp_note_ids=note_ids,
        mcp_prompts=prompts,
    )
    response.headers["Cache-Control"] = "no-store"
    return MCPProfileCreated(id=api_key.id, api_key=secret)


@router.put("/profiles/{key_id}", response_model=MCPKeyConfigResponse)
async def update_mcp_profile(
    key_id: UUID,
    profile: MCPProfileCreate,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if request.headers.get("X-API-Key"):
        raise HTTPException(403, "Sign in to edit an MCP profile")

    api_key = await lock_owned_profile(db, key_id, current_user.id)
    notes_config = await db.scalar(note_config_query(key_id))
    note_ids = (
        selected_note_ids(notes_config)
        if profile.note_ids is None
        else await validate_note_selection(profile.note_ids, current_user.id, db)
    )
    prompts = (
        api_key.mcp_prompts
        if profile.prompts is None
        else [prompt.model_dump() for prompt in profile.prompts]
    )
    programs, tools, sources = await validate_profile_selection(
        profile, db, note_ids, bool(prompts), api_key
    )
    api_key.name = profile.name
    api_key.mcp_programs = programs
    api_key.allowed_tools = tools
    api_key.scopes = profile_scopes(tools, bool(prompts))
    if profile.prompts is not None:
        api_key.mcp_prompts = prompts
    if profile.note_ids is not None and (note_ids or notes_config is not None):
        if notes_config is None:
            notes_config = MCPToolConfig(
                api_key_id=key_id, tool_name=RESEARCH_NOTES_TOOL
            )
            db.add(notes_config)
        notes_config.config = {"note_ids": [str(note_id) for note_id in note_ids]}
    if "source_ids" in profile.model_fields_set:
        api_key.source_access_mode = "restricted" if sources is not None else "all"
    if sources is not None:
        current_sources = set(
            await db.scalars(
                select(MCPSourceAccess.source_id).where(
                    MCPSourceAccess.api_key_id == key_id
                )
            )
        )
        removed_sources = current_sources.difference(sources)
        if removed_sources:
            await db.execute(
                delete(MCPSourceAccess).where(
                    MCPSourceAccess.api_key_id == key_id,
                    MCPSourceAccess.source_id.in_(removed_sources),
                )
            )
        db.add_all(
            MCPSourceAccess(api_key_id=key_id, source_id=source_id)
            for source_id in sources
            if source_id not in current_sources
        )
    await db.commit()
    return await get_mcp_key_config(key_id, current_user, db)


@router.post("/profiles/{key_id}/rotate", response_model=MCPProfileRotated)
async def rotate_mcp_profile_key(
    key_id: UUID,
    rotation: MCPProfileRotation,
    request: Request,
    response: Response,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if request.headers.get("X-API-Key"):
        raise HTTPException(403, "Sign in to rotate an MCP profile key")

    api_key = await lock_owned_profile(db, key_id, current_user.id)
    if not api_key.is_valid():
        raise HTTPException(409, "Rotation requires an active, unexpired API key")
    if api_key.key_prefix != rotation.key_prefix:
        raise HTTPException(
            409, "The key has changed. Refresh the profile and try again."
        )

    secret, prefix, key_hash = api_key_service.generate_key()
    while prefix == api_key.key_prefix:
        secret, prefix, key_hash = api_key_service.generate_key()
    api_key.key_prefix = prefix
    api_key.key_hash = key_hash
    await db.commit()

    logger.info(f"Rotated MCP profile key {key_id} for user {current_user.id}")
    response.headers["Cache-Control"] = "no-store"
    return MCPProfileRotated(id=key_id, api_key=secret, key_prefix=prefix)


@router.get("/tools", response_model=List[MCPToolInfo])
async def list_available_tools(
    current_user: User = Depends(get_current_user),
):
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
            status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
        )

    # Get tool configs
    tool_configs_result = await db.execute(
        select(MCPToolConfig).where(MCPToolConfig.api_key_id == key_id)
    )
    tool_configs_db = {tc.tool_name: tc for tc in tool_configs_result.scalars().all()}
    note_ids = selected_note_ids(tool_configs_db.get(RESEARCH_NOTES_TOOL))
    research_notes = []
    if api_key.mcp_programs is not None and note_ids:
        rows = await db.execute(
            select(ResearchNote.id, ResearchNote.title).where(
                ResearchNote.user_id == current_user.id, ResearchNote.id.in_(note_ids)
            )
        )
        titles = {row.id: row.title for row in rows}
        research_notes = [
            MCPNoteOption(id=note_id, title=titles[note_id])
            for note_id in note_ids
            if note_id in titles
        ]

    # Build tool configs list
    tool_configs = []
    for tool_name, tool_info in MCP_TOOLS.items():
        if tool_name == RESEARCH_PROGRAM_TOOL and not api_key.mcp_programs:
            continue
        if tool_name == RESEARCH_NOTES_TOOL and not research_notes:
            continue
        tc = tool_configs_db.get(tool_name)
        tool_configs.append(
            MCPToolConfigResponse(
                tool_name=tool_name,
                display_name=tool_info["display_name"],
                description=tool_info["description"],
                category=tool_info["category"],
                is_enabled=tc.is_enabled if tc else True,
                config=tc.config if tc else None,
            )
        )

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
        mcp_programs=api_key.mcp_programs,
        research_notes=research_notes,
        prompts=api_key.mcp_prompts or [],
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
            status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
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
                detail=f"Invalid tool names: {invalid_tools}",
            )
        if api_key.mcp_programs is not None:
            selectable = set(MCP_PROFILE_TOOLS)
            if api_key.mcp_programs:
                selectable.add(RESEARCH_PROGRAM_TOOL)
            if RESEARCH_NOTES_TOOL in config.allowed_tools:
                if selected_note_ids(await db.scalar(note_config_query(key_id))):
                    selectable.add(RESEARCH_NOTES_TOOL)
            if set(config.allowed_tools) - selectable:
                raise HTTPException(400, "Tool unavailable for this MCP profile")
        api_key.allowed_tools = config.allowed_tools

    if config.source_access_mode is not None:
        if config.source_access_mode not in ("all", "restricted"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="source_access_mode must be 'all' or 'restricted'",
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
            status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
        )

    # Validate tool name
    if tool_name not in MCP_TOOLS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tool name: {tool_name}",
        )

    # Get or create tool config
    result = await db.execute(
        select(MCPToolConfig).where(
            MCPToolConfig.api_key_id == key_id,
            MCPToolConfig.tool_name == tool_name,
        )
    )
    tool_config = result.scalar_one_or_none()

    if tool_name == RESEARCH_NOTES_TOOL and (
        api_key.mcp_programs is None
        or not selected_note_ids(tool_config)
        or config.config is not None
    ):
        raise HTTPException(400, "Edit the MCP profile to select its research notes")

    if tool_config:
        tool_config.is_enabled = config.is_enabled
        if tool_name != RESEARCH_NOTES_TOOL:
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
            status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
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


@router.put(
    "/keys/{key_id}/sources/{source_id}", response_model=MCPSourceAccessResponse
)
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
            status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
        )

    # Verify source exists
    result = await db.execute(
        select(DocumentSource).where(DocumentSource.id == source_id)
    )
    source = result.scalar_one_or_none()
    if not source:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document source not found"
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


@router.delete(
    "/keys/{key_id}/sources/{source_id}", status_code=status.HTTP_204_NO_CONTENT
)
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
            status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
        )

    # Delete source access
    await db.execute(
        delete(MCPSourceAccess).where(
            MCPSourceAccess.api_key_id == key_id,
            MCPSourceAccess.source_id == source_id,
        )
    )
    await db.commit()
