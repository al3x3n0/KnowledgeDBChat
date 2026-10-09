"""Which backlog items a person may see, and reading them.

This was inline in ``api/endpoints/coding_backlog.py``. The list route asked
the database for every item anyone had shared with anyone, decided in Python
which of them this caller could see, applied the caller's filters in Python
too, and returned one page of what was left. Here the database does what it
can decide exactly, and the rule below has the last word on the rest.

The rule: an item is visible to its owner, to whoever it is assigned to, and
-- when it is shared -- to the people it is shared with. ``visibility ==
"shared"`` alone shows it to nobody else: the list of people is what grants.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple
from uuid import UUID

from sqlalchemy import desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.coding_backlog import CodingBacklogItem
from app.services.collaboration_service import normalize_collaboration_visibility
from app.services.config_values import uuid_list

SCOPES = ("mine", "shared", "all")


class BacklogItemNotFound(LookupError):
    """No such item, or not one this person may see. The two look the same."""


def is_visible_to(item: CodingBacklogItem, user_id: Any) -> bool:
    if str(item.user_id) == str(user_id):
        return True
    if str(getattr(item, "assigned_user_id", "") or "").strip() == str(user_id):
        return True
    visibility = normalize_collaboration_visibility(
        getattr(item, "visibility", "private")
    )
    if visibility != "shared":
        return False
    return str(user_id) in uuid_list(getattr(item, "shared_with_user_ids", None), 200)


async def get_visible_item(
    db: AsyncSession, item_id: UUID, user_id: UUID
) -> CodingBacklogItem:
    item = await db.get(CodingBacklogItem, item_id)
    if item is None or not is_visible_to(item, user_id):
        raise BacklogItemNotFound(str(item_id))
    return item


def normalize_scope(scope: Any) -> str:
    return str(scope or "mine").strip().lower() or "mine"


async def list_visible_items(
    db: AsyncSession,
    user_id: UUID,
    *,
    scope: Any = "mine",
    status: Optional[str] = None,
    assigned_user_id: Optional[UUID] = None,
    limit: int = 50,
    offset: int = 0,
) -> Tuple[List[CodingBacklogItem], int]:
    """One page of the items this person may see, newest first, and the total.

    A person's own items are selected, counted and paged by the database.
    Other people's are visible through a list held as JSON, which the
    database is not asked to read: it narrows to the items that could
    qualify, and `is_visible_to` decides.
    """
    scope = normalize_scope(scope)
    theirs = CodingBacklogItem.user_id != user_id
    could_be_visible = or_(
        CodingBacklogItem.assigned_user_id == user_id,
        CodingBacklogItem.visibility == "shared",
    )

    statement = select(CodingBacklogItem)
    if scope == "mine":
        statement = statement.where(CodingBacklogItem.user_id == user_id)
    elif scope == "shared":
        statement = statement.where(theirs, could_be_visible)
    else:
        statement = statement.where(
            or_(CodingBacklogItem.user_id == user_id, could_be_visible)
        )
    if status:
        statement = statement.where(
            CodingBacklogItem.status == str(status).strip().lower()
        )
    if assigned_user_id:
        statement = statement.where(
            CodingBacklogItem.assigned_user_id == assigned_user_id
        )
    ordered = statement.order_by(desc(CodingBacklogItem.updated_at))

    if scope == "mine":
        total = int(
            (
                await db.execute(
                    statement.with_only_columns(func.count(CodingBacklogItem.id))
                )
            ).scalar()
            or 0
        )
        page = (await db.execute(ordered.limit(limit).offset(offset))).scalars().all()
        return list(page), total

    visible = [
        item
        for item in (await db.execute(ordered)).scalars().all()
        if is_visible_to(item, user_id)
    ]
    return visible[offset : offset + limit], len(visible)


__all__ = [
    "SCOPES",
    "BacklogItemNotFound",
    "get_visible_item",
    "is_visible_to",
    "list_visible_items",
    "normalize_scope",
]
