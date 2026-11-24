"""Read the owner's research notes explicitly selected for an MCP profile."""

from uuid import UUID

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.mcp_config import RESEARCH_NOTES_TOOL, MCPToolConfig
from app.models.research_note import ResearchNote
from app.services.config_values import parse_uuid

MAX_NOTE_CONTENT = 100_000
RESEARCH_NOTES_SCHEMA = {
    "type": "object",
    "properties": {
        "note_id": {
            "type": "string",
            "format": "uuid",
            "description": "Read this note. Omit to list the profile's selected notes.",
        }
    },
}


def note_config_query(key_id: UUID) -> Select[tuple[MCPToolConfig]]:
    return select(MCPToolConfig).where(
        MCPToolConfig.api_key_id == key_id,
        MCPToolConfig.tool_name == RESEARCH_NOTES_TOOL,
    )


def selected_note_ids(config: MCPToolConfig | None) -> list[UUID]:
    settings = config.config if config else None
    values = settings.get("note_ids") if isinstance(settings, dict) else None
    if not isinstance(values, list):
        return []
    return [note_id for value in values if (note_id := parse_uuid(value)) is not None]


async def read_research_notes(
    db: AsyncSession, key_id: UUID, user_id: UUID, note_id: str | None
) -> dict:
    allowed_ids = selected_note_ids(await db.scalar(note_config_query(key_id)))
    owned_notes = (
        ResearchNote.user_id == user_id,
        ResearchNote.id.in_(allowed_ids),
    )
    if note_id is None:
        rows = await db.execute(
            select(ResearchNote.id, ResearchNote.title, ResearchNote.updated_at)
            .where(*owned_notes)
            .order_by(ResearchNote.updated_at.desc(), ResearchNote.id)
        )
        return {
            "notes": [
                {
                    "id": str(row.id),
                    "title": row.title,
                    "updated_at": row.updated_at.isoformat()
                    if row.updated_at
                    else None,
                }
                for row in rows
            ]
        }

    parsed_id = parse_uuid(note_id)
    if parsed_id is None:
        return {"error": "note_id must be a UUID"}
    note = await db.scalar(
        select(ResearchNote).where(*owned_notes, ResearchNote.id == parsed_id)
    )
    if note is None:
        return {"error": "Research note not found"}
    return {
        "id": str(note.id),
        "title": note.title,
        "content_markdown": note.content_markdown[:MAX_NOTE_CONTENT],
        "content_truncated": len(note.content_markdown) > MAX_NOTE_CONTENT,
        "updated_at": note.updated_at.isoformat() if note.updated_at else None,
    }
