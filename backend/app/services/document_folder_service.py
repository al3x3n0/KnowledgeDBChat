"""Folders over the document corpus: the tree, and the one filter vocabulary.

Every node in the tree — user folder or system folder — is addressed by a
single opaque string, its `key`:

    all                     the whole corpus
    unfiled                 in none of this user's folders
    user:<uuid>             one of this user's folders
    source:<uuid>           everything from one document source
    type:<file_type>        everything of one file type
    recent:today|week|month ingested within that window
    tag:<tag>               carrying that tag

That vocabulary exists so the tree and the document list agree by
construction. The alternative — a `folder_id` param plus a `file_type` param
plus a `since` param plus a `tag` param — lets the two drift, and puts the
definition of "what is in this folder" in two places. Here, `resolve_filter`
is the only thing that knows, and both the tree's counts and the list's rows
come out of it.

System folders are computed rather than stored, so a source that syncs for the
first time appears in the tree immediately and a folder can never claim a
document that no longer matches it.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Optional
from uuid import UUID

from loguru import logger
from sqlalchemy import Select, Text, and_, func, not_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document, DocumentSource
from app.models.document_folder import DocumentFolder, DocumentFolderItem

#: How deep a user may nest folders. Not a storage limit — a limit on how much
#: rope the UI has to draw and the user has to navigate.
MAX_FOLDER_DEPTH = 8

#: The recency windows offered as system folders, in days.
RECENT_WINDOWS = {"today": 1, "week": 7, "month": 30}


class FolderError(Exception):
    """A folder operation the caller got wrong, with a reason worth showing."""

    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class DocumentFolderService:
    # ---------------------------------------------------------------- filters

    def resolve_filter(self, key: str, user_id: UUID) -> Optional[Any]:
        """Turn a folder key into a SQLAlchemy predicate over `Document`.

        Returns None for the whole corpus. Raises for a key that is not part
        of the vocabulary, rather than silently returning everything — a typo
        in a filter should not look like a successful query.
        """
        raw = (key or "all").strip()
        if raw in ("", "all"):
            return None

        if raw == "unfiled":
            # Not in ANY of this user's folders. Someone else's filing does not
            # make a document filed for you.
            mine = (
                select(DocumentFolderItem.document_id)
                .join(
                    DocumentFolder,
                    DocumentFolder.id == DocumentFolderItem.folder_id,
                )
                .where(DocumentFolder.user_id == user_id)
            )
            return not_(Document.id.in_(mine))

        prefix, _, value = raw.partition(":")
        if not value:
            raise FolderError(f"Not a folder key: {raw!r}")

        if prefix == "user":
            return Document.id.in_(
                select(DocumentFolderItem.document_id)
                .join(DocumentFolder, DocumentFolder.id == DocumentFolderItem.folder_id)
                .where(
                    and_(
                        DocumentFolder.user_id == user_id,
                        DocumentFolderItem.folder_id == self._as_uuid(value),
                    )
                )
            )
        if prefix == "source":
            return Document.source_id == self._as_uuid(value)
        if prefix == "type":
            return Document.file_type == value
        if prefix == "recent":
            days = RECENT_WINDOWS.get(value)
            if days is None:
                raise FolderError(f"Unknown recency window: {value!r}")
            return Document.created_at >= datetime.utcnow() - timedelta(days=days)
        if prefix == "tag":
            # `tags` is a JSON list. Postgres can ask whether it contains a
            # value; SQLite (the test database) cannot, so fall back to a
            # substring match on the serialised list, which is good enough for
            # a filter the user drove by clicking a tag we ourselves listed.
            return func.cast(Document.tags, Text).like(f'%"{value}"%')

        raise FolderError(f"Not a folder key: {raw!r}")

    def apply_filter(self, stmt: Select, key: str, user_id: UUID) -> Select:
        """Narrow a `Document` select to one folder."""
        predicate = self.resolve_filter(key, user_id)
        return stmt if predicate is None else stmt.where(predicate)

    @staticmethod
    def _as_uuid(value: str) -> UUID:
        try:
            return UUID(value)
        except (ValueError, AttributeError, TypeError) as exc:
            raise FolderError(f"Not an id: {value!r}") from exc

    # ------------------------------------------------------------------- tree

    async def user_tree(
        self, db: AsyncSession, user_id: UUID, *, with_counts: bool = True
    ) -> list[dict]:
        """This user's folders, nested, each with how many documents it holds.

        One query for the folders and one for the counts, then assembled in
        memory. A recursive CTE would push the nesting into the database, but
        `MAX_FOLDER_DEPTH` bounds the tree at eight levels and a user's folder
        count is small — two flat queries are simpler to read and cheaper than
        N+1 by a wide margin.
        """
        rows = (
            (
                await db.execute(
                    select(DocumentFolder)
                    .where(DocumentFolder.user_id == user_id)
                    .order_by(DocumentFolder.position, DocumentFolder.name)
                )
            )
            .scalars()
            .all()
        )

        counts: dict[UUID, int] = {}
        if with_counts and rows:
            count_rows = await db.execute(
                select(
                    DocumentFolderItem.folder_id,
                    func.count(DocumentFolderItem.document_id),
                )
                .where(DocumentFolderItem.folder_id.in_([r.id for r in rows]))
                .group_by(DocumentFolderItem.folder_id)
            )
            counts = {fid: int(n) for fid, n in count_rows.all()}

        nodes: dict[UUID, dict] = {
            r.id: {
                "key": f"user:{r.id}",
                "id": str(r.id),
                "name": r.name,
                "description": r.description,
                "color": r.color,
                "position": r.position,
                "kind": "user",
                "document_count": counts.get(r.id, 0),
                "children": [],
            }
            for r in rows
        }

        roots: list[dict] = []
        for r in rows:
            node = nodes[r.id]
            parent = nodes.get(r.parent_id) if r.parent_id else None
            if parent is not None:
                parent["children"].append(node)
            else:
                # A folder whose parent is missing (or not ours) is shown at the
                # root rather than dropped: an invisible folder is worse than a
                # misplaced one.
                roots.append(node)

        # A folder's total includes what its descendants hold, which is what
        # someone reading a tree expects a number beside a closed folder to mean.
        def rollup(node: dict) -> int:
            node["subtree_count"] = node["document_count"] + sum(
                rollup(child) for child in node["children"]
            )
            return node["subtree_count"]

        for root in roots:
            rollup(root)
        return roots

    async def system_tree(self, db: AsyncSession, user_id: UUID) -> list[dict]:
        """The computed folders: by source, by type, recent, tagged, unfiled.

        Every count here is a real count, and every group is derived from the
        documents themselves — so this cannot be stale, and it needs nothing
        seeded when a new source is added.
        """
        total = int(
            (await db.execute(select(func.count(Document.id)))).scalar_one_or_none()
            or 0
        )

        by_source = (
            await db.execute(
                select(
                    DocumentSource.id,
                    DocumentSource.name,
                    DocumentSource.source_type,
                    func.count(Document.id),
                )
                .join(Document, Document.source_id == DocumentSource.id)
                .group_by(
                    DocumentSource.id, DocumentSource.name, DocumentSource.source_type
                )
                .order_by(func.count(Document.id).desc())
            )
        ).all()

        by_type = (
            await db.execute(
                select(Document.file_type, func.count(Document.id))
                .where(Document.file_type.isnot(None))
                .group_by(Document.file_type)
                .order_by(func.count(Document.id).desc())
            )
        ).all()

        unfiled_predicate = self.resolve_filter("unfiled", user_id)
        unfiled = int(
            (
                await db.execute(
                    select(func.count(Document.id)).where(unfiled_predicate)
                )
            ).scalar_one_or_none()
            or 0
        )

        groups: list[dict] = [
            {
                "key": "all",
                "name": "All documents",
                "kind": "system",
                "document_count": total,
                "children": [],
            }
        ]

        if by_source:
            groups.append(
                {
                    "key": "group:source",
                    "name": "By source",
                    "kind": "group",
                    "document_count": 0,
                    "children": [
                        {
                            "key": f"source:{sid}",
                            "name": name,
                            "kind": "system",
                            "icon": source_type,
                            "document_count": int(n),
                            "children": [],
                        }
                        for sid, name, source_type, n in by_source
                    ],
                }
            )

        if by_type:
            groups.append(
                {
                    "key": "group:type",
                    "name": "By type",
                    "kind": "group",
                    "document_count": 0,
                    "children": [
                        {
                            "key": f"type:{ftype}",
                            "name": ftype,
                            "kind": "system",
                            "document_count": int(n),
                            "children": [],
                        }
                        for ftype, n in by_type
                    ],
                }
            )

        recent_children = []
        for window in ("today", "week", "month"):
            predicate = self.resolve_filter(f"recent:{window}", user_id)
            n = int(
                (
                    await db.execute(select(func.count(Document.id)).where(predicate))
                ).scalar_one_or_none()
                or 0
            )
            recent_children.append(
                {
                    "key": f"recent:{window}",
                    "name": {
                        "today": "Today",
                        "week": "This week",
                        "month": "This month",
                    }[window],
                    "kind": "system",
                    "document_count": n,
                    "children": [],
                }
            )
        groups.append(
            {
                "key": "group:recent",
                "name": "Recent",
                "kind": "group",
                "document_count": 0,
                "children": recent_children,
            }
        )

        groups.append(
            {
                "key": "unfiled",
                "name": "Unfiled",
                "kind": "system",
                "document_count": unfiled,
                "children": [],
            }
        )

        for group in groups:
            group["subtree_count"] = group["document_count"] + sum(
                child["document_count"] for child in group["children"]
            )
        return groups

    # -------------------------------------------------------------------- CRUD

    async def create(
        self,
        db: AsyncSession,
        user_id: UUID,
        *,
        name: str,
        parent_id: Optional[UUID] = None,
        description: Optional[str] = None,
        color: Optional[str] = None,
        position: int = 0,
    ) -> DocumentFolder:
        clean = (name or "").strip()
        if not clean:
            raise FolderError("A folder needs a name")

        if parent_id is not None:
            parent = await self._owned(db, user_id, parent_id)
            depth = await self._depth(db, parent)
            if depth + 1 >= MAX_FOLDER_DEPTH:
                raise FolderError(
                    f"Folders nest at most {MAX_FOLDER_DEPTH} deep; "
                    f"'{parent.name}' is already at {depth + 1}"
                )

        await self._refuse_duplicate(db, user_id, parent_id, clean)

        folder = DocumentFolder(
            user_id=user_id,
            parent_id=parent_id,
            name=clean,
            description=(description or None),
            color=(color or None),
            position=position,
        )
        db.add(folder)
        try:
            await db.commit()
        except IntegrityError as exc:
            await db.rollback()
            raise FolderError("A folder with that name is already here", 409) from exc
        await db.refresh(folder)
        return folder

    async def update(
        self,
        db: AsyncSession,
        user_id: UUID,
        folder_id: UUID,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        color: Optional[str] = None,
        position: Optional[int] = None,
        parent_id: Optional[UUID] = None,
        reparent: bool = False,
    ) -> DocumentFolder:
        folder = await self._owned(db, user_id, folder_id)

        if reparent:
            await self._check_reparent(db, user_id, folder, parent_id)
            folder.parent_id = parent_id

        if name is not None:
            clean = name.strip()
            if not clean:
                raise FolderError("A folder needs a name")
            if clean != folder.name:
                await self._refuse_duplicate(db, user_id, folder.parent_id, clean)
            folder.name = clean
        if description is not None:
            folder.description = description or None
        if color is not None:
            folder.color = color or None
        if position is not None:
            folder.position = position

        try:
            await db.commit()
        except IntegrityError as exc:
            await db.rollback()
            raise FolderError("A folder with that name is already here", 409) from exc
        await db.refresh(folder)
        return folder

    async def delete(
        self,
        db: AsyncSession,
        user_id: UUID,
        folder_id: UUID,
        *,
        recursive: bool = False,
    ) -> dict:
        """Delete a folder. Refuses to take a subtree with it silently.

        The documents themselves are never touched: a folder is a view, and
        deleting a view must not delete what it was looking at.
        """
        folder = await self._owned(db, user_id, folder_id)

        children = int(
            (
                await db.execute(
                    select(func.count(DocumentFolder.id)).where(
                        DocumentFolder.parent_id == folder_id
                    )
                )
            ).scalar_one_or_none()
            or 0
        )
        if children and not recursive:
            raise FolderError(
                f"'{folder.name}' has {children} subfolder(s). "
                "Pass recursive=true to delete them too.",
                409,
            )

        name = folder.name
        await db.delete(folder)
        await db.commit()
        logger.info(f"Deleted folder {folder_id} ('{name}') for user {user_id}")
        return {"deleted": str(folder_id), "name": name, "subfolders_deleted": children}

    # ------------------------------------------------------------------- items

    async def add_documents(
        self, db: AsyncSession, user_id: UUID, folder_id: UUID, document_ids: list[UUID]
    ) -> dict:
        """File documents into a folder, idempotently."""
        await self._owned(db, user_id, folder_id)
        if not document_ids:
            return {"added": 0, "already_present": 0, "not_found": 0}

        wanted = list(dict.fromkeys(document_ids))
        exists = set(
            (await db.execute(select(Document.id).where(Document.id.in_(wanted))))
            .scalars()
            .all()
        )
        present = set(
            (
                await db.execute(
                    select(DocumentFolderItem.document_id).where(
                        and_(
                            DocumentFolderItem.folder_id == folder_id,
                            DocumentFolderItem.document_id.in_(wanted),
                        )
                    )
                )
            )
            .scalars()
            .all()
        )

        added = 0
        for doc_id in wanted:
            if doc_id not in exists or doc_id in present:
                continue
            db.add(DocumentFolderItem(folder_id=folder_id, document_id=doc_id))
            added += 1
        if added:
            await db.commit()
        return {
            "added": added,
            "already_present": len(present),
            "not_found": len([d for d in wanted if d not in exists]),
        }

    async def remove_documents(
        self, db: AsyncSession, user_id: UUID, folder_id: UUID, document_ids: list[UUID]
    ) -> dict:
        """Unfile documents. The documents themselves are untouched."""
        await self._owned(db, user_id, folder_id)
        if not document_ids:
            return {"removed": 0}

        rows = (
            (
                await db.execute(
                    select(DocumentFolderItem).where(
                        and_(
                            DocumentFolderItem.folder_id == folder_id,
                            DocumentFolderItem.document_id.in_(list(set(document_ids))),
                        )
                    )
                )
            )
            .scalars()
            .all()
        )
        for row in rows:
            await db.delete(row)
        if rows:
            await db.commit()
        return {"removed": len(rows)}

    async def folders_for_document(
        self, db: AsyncSession, user_id: UUID, document_id: UUID
    ) -> list[dict]:
        """Which of this user's folders hold a given document."""
        rows = (
            (
                await db.execute(
                    select(DocumentFolder)
                    .join(
                        DocumentFolderItem,
                        DocumentFolderItem.folder_id == DocumentFolder.id,
                    )
                    .where(
                        and_(
                            DocumentFolder.user_id == user_id,
                            DocumentFolderItem.document_id == document_id,
                        )
                    )
                    .order_by(DocumentFolder.name)
                )
            )
            .scalars()
            .all()
        )
        return [
            {"key": f"user:{r.id}", "id": str(r.id), "name": r.name, "color": r.color}
            for r in rows
        ]

    # ------------------------------------------------------------------ guards

    async def _owned(
        self, db: AsyncSession, user_id: UUID, folder_id: UUID
    ) -> DocumentFolder:
        """Fetch a folder, or refuse. Never reveals another user's folder."""
        folder = (
            await db.execute(
                select(DocumentFolder).where(
                    and_(
                        DocumentFolder.id == folder_id,
                        DocumentFolder.user_id == user_id,
                    )
                )
            )
        ).scalar_one_or_none()
        if folder is None:
            # 404 rather than 403: a folder you do not own should not be
            # distinguishable from one that does not exist.
            raise FolderError("Folder not found", 404)
        return folder

    async def _refuse_duplicate(
        self,
        db: AsyncSession,
        user_id: UUID,
        parent_id: Optional[UUID],
        name: str,
    ) -> None:
        """The unique constraint cannot cover root folders, because Postgres
        treats NULL parents as distinct. This closes that gap."""
        clause = (
            DocumentFolder.parent_id.is_(None)
            if parent_id is None
            else DocumentFolder.parent_id == parent_id
        )
        clash = (
            await db.execute(
                select(DocumentFolder.id).where(
                    and_(
                        DocumentFolder.user_id == user_id,
                        clause,
                        func.lower(DocumentFolder.name) == name.lower(),
                    )
                )
            )
        ).scalar_one_or_none()
        if clash is not None:
            raise FolderError("A folder with that name is already here", 409)

    async def _check_reparent(
        self,
        db: AsyncSession,
        user_id: UUID,
        folder: DocumentFolder,
        new_parent_id: Optional[UUID],
    ) -> None:
        """Refuse a move that would detach the tree from its root.

        A folder cannot become its own parent, nor a child of one of its own
        descendants — that severs the subtree from every root and makes it
        unreachable, while leaving perfectly valid-looking rows behind.
        """
        if new_parent_id is None:
            return
        if new_parent_id == folder.id:
            raise FolderError("A folder cannot be its own parent")

        parent = await self._owned(db, user_id, new_parent_id)

        # Walk up from the proposed parent: if we meet this folder, the move
        # would create a cycle.
        seen: set[UUID] = set()
        cursor: Optional[DocumentFolder] = parent
        while cursor is not None:
            if cursor.id == folder.id:
                raise FolderError("A folder cannot be moved inside itself")
            if cursor.id in seen:
                # Defensive: existing data should never contain a cycle, but
                # looping for ever while checking for one would be worse.
                logger.warning(f"Cycle already present above folder {cursor.id}")
                break
            seen.add(cursor.id)
            cursor = (
                None
                if cursor.parent_id is None
                else (
                    await db.execute(
                        select(DocumentFolder).where(
                            DocumentFolder.id == cursor.parent_id
                        )
                    )
                ).scalar_one_or_none()
            )

        depth = await self._depth(db, parent)
        subtree = await self._subtree_height(db, folder)
        if depth + 1 + subtree > MAX_FOLDER_DEPTH:
            raise FolderError(
                f"That move would nest folders {depth + 1 + subtree} deep; "
                f"the limit is {MAX_FOLDER_DEPTH}"
            )

    async def _depth(self, db: AsyncSession, folder: DocumentFolder) -> int:
        """How many ancestors a folder has. A root folder is depth 0."""
        depth = 0
        cursor = folder
        seen: set[UUID] = {folder.id}
        while cursor.parent_id is not None and depth < MAX_FOLDER_DEPTH + 1:
            parent = (
                await db.execute(
                    select(DocumentFolder).where(DocumentFolder.id == cursor.parent_id)
                )
            ).scalar_one_or_none()
            if parent is None or parent.id in seen:
                break
            seen.add(parent.id)
            cursor = parent
            depth += 1
        return depth

    async def _subtree_height(self, db: AsyncSession, folder: DocumentFolder) -> int:
        """How many levels sit below a folder. A leaf is 0."""
        level = [folder.id]
        height = 0
        while level and height < MAX_FOLDER_DEPTH + 1:
            children = (
                (
                    await db.execute(
                        select(DocumentFolder.id).where(
                            DocumentFolder.parent_id.in_(level)
                        )
                    )
                )
                .scalars()
                .all()
            )
            if not children:
                break
            level = list(children)
            height += 1
        return height


document_folder_service = DocumentFolderService()
