"""Finding or creating a knowledge-graph entity by its type and name.

Three copies existed (two in `knowledge_extraction`, one in `paper_kg_service`)
and the third looked an entity up by its full name but stored it truncated to
the column. A name longer than the column never matched its own row, so every
call made another; with no unique constraint the copies accumulated, and the
next exact lookup of that name raised MultipleResultsFound.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.knowledge_graph import Entity

NAME_MAX = 512
TYPE_MAX = 64


async def get_or_create_entity(
    db: AsyncSession,
    name: str,
    entity_type: str,
    *,
    description: Optional[str] = None,
    properties: Optional[Dict[str, Any]] = None,
) -> Tuple[Entity, bool]:
    """The entity, and whether it was created. Flushed, not committed.

    The key is cut to the column widths *before* the lookup, so a long name
    finds the row it was stored as. Rows duplicated before this existed are
    tolerated: the oldest wins.
    """
    name = name[:NAME_MAX]
    entity_type = entity_type[:TYPE_MAX]
    existing = (
        (
            await db.execute(
                select(Entity)
                .where(Entity.canonical_name == name, Entity.entity_type == entity_type)
                .order_by(Entity.created_at)
                .limit(1)
            )
        )
        .scalars()
        .first()
    )
    if existing is not None:
        return existing, False
    entity = Entity(
        canonical_name=name,
        entity_type=entity_type,
        description=description,
        properties=json.dumps(properties, ensure_ascii=False) if properties else None,
    )
    db.add(entity)
    await db.flush()
    return entity, True
