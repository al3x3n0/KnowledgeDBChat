"""MCP resources backed by the profile's read-only research tools."""

import json
from urllib.parse import quote

from sqlalchemy.ext.asyncio import AsyncSession

from app.agent_core import research_programs
from app.mcp.auth import MCPAuthContext
from app.mcp.executor import research_program_context
from app.mcp.tools.research_notes import read_research_notes
from app.models.mcp_config import RESEARCH_NOTES_TOOL, RESEARCH_PROGRAM_TOOL
from app.services.config_values import parse_uuid
from app.services.tool_policy_engine import evaluate_tool_policy

NOTE_URI_PREFIX = "kdbc://research-notes/"
PROGRAM_URI_PREFIX = "kdbc://research-programs/"
MARKDOWN = "text/markdown"
JSON_TYPE = "application/json"


def _program_uri(name: str) -> str:
    return f"{PROGRAM_URI_PREFIX}{quote(name, safe='')}.json"


async def _policy_allows_resource(
    auth: MCPAuthContext,
    db: AsyncSession,
    tool_name: str,
    tool_args: dict | None = None,
) -> bool:
    decision = await evaluate_tool_policy(
        db=db,
        tool_name=f"mcp:{tool_name}",
        tool_args=tool_args,
        user=auth.user,
        api_key_id=auth.api_key.id,
    )
    # Clients have no approval interaction for resource browsing.
    return decision.allowed and not decision.require_approval


async def list_profile_resources(auth: MCPAuthContext, db: AsyncSession) -> list[dict]:
    allowed = await auth.allowed_tools(db)
    resources = []
    if RESEARCH_NOTES_TOOL in allowed and await _policy_allows_resource(
        auth, db, RESEARCH_NOTES_TOOL
    ):
        selected = await read_research_notes(db, auth.api_key.id, auth.user_id, None)
        resources.extend(
            {
                "uri": f"{NOTE_URI_PREFIX}{note['id']}",
                "name": note["title"],
                "mimeType": MARKDOWN,
            }
            for note in selected["notes"]
        )
    if RESEARCH_PROGRAM_TOOL in allowed and await _policy_allows_resource(
        auth, db, RESEARCH_PROGRAM_TOOL
    ):
        selected_names = set(auth.api_key.mcp_programs or ())
        resources.extend(
            {
                "uri": _program_uri(program.name),
                "name": program.name,
                "description": program.description,
                "mimeType": JSON_TYPE,
            }
            for program in research_programs.installed()
            if program.name in selected_names
        )
    return resources


async def _read_program_resource(
    auth: MCPAuthContext, db: AsyncSession, uri: str
) -> dict | None:
    name = next(
        (name for name in auth.api_key.mcp_programs or () if _program_uri(name) == uri),
        None,
    )
    if name is None or RESEARCH_PROGRAM_TOOL not in await auth.allowed_tools(db):
        return None
    if not await _policy_allows_resource(auth, db, RESEARCH_PROGRAM_TOOL, {}):
        return None
    selected = research_program_context([name])["programs"]
    if not selected:
        return None
    return {
        "uri": uri,
        "mimeType": JSON_TYPE,
        "text": json.dumps(selected[0], ensure_ascii=False, indent=2),
    }


async def read_profile_resource(
    auth: MCPAuthContext, db: AsyncSession, uri: str
) -> dict | None:
    if uri.startswith(PROGRAM_URI_PREFIX):
        return await _read_program_resource(auth, db, uri)
    note_id = parse_uuid(uri.removeprefix(NOTE_URI_PREFIX))
    if note_id is None or uri != f"{NOTE_URI_PREFIX}{note_id}":
        return None
    if RESEARCH_NOTES_TOOL not in await auth.allowed_tools(db):
        return None
    if not await _policy_allows_resource(
        auth, db, RESEARCH_NOTES_TOOL, {"note_id": str(note_id)}
    ):
        return None
    result = await read_research_notes(db, auth.api_key.id, auth.user_id, str(note_id))
    if "error" in result:
        return None
    text = result["content_markdown"]
    if result["content_truncated"]:
        text += "\n\n[Research note truncated]"
    return {"uri": uri, "mimeType": MARKDOWN, "text": text}
