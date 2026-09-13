"""Request and response shapes for document folders."""

from __future__ import annotations

from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentFolderCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    parent_id: Optional[UUID] = None
    description: Optional[str] = Field(None, max_length=2000)
    color: Optional[str] = Field(None, max_length=32)
    position: int = 0


class DocumentFolderUpdate(BaseModel):
    """Every field optional, so a rename does not have to restate the rest.

    `parent_id` is ambiguous on its own: `None` could mean "make this a root
    folder" or "leave the parent alone". `reparent` disambiguates — the move
    happens only when it is true, and then `parent_id=None` means the root.
    """

    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    color: Optional[str] = Field(None, max_length=32)
    position: Optional[int] = None
    parent_id: Optional[UUID] = None
    reparent: bool = False


class DocumentFolderNode(BaseModel):
    """One node of either tree.

    `key` is the whole addressing story: it is what the documents list takes
    as a filter, and it is present on system nodes that have no `id`.
    """

    key: str
    name: str
    kind: Literal["user", "system", "group"]
    document_count: int = 0
    subtree_count: int = 0
    children: list["DocumentFolderNode"] = Field(default_factory=list)

    id: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None
    icon: Optional[str] = None
    position: Optional[int] = None


DocumentFolderNode.model_rebuild()


class DocumentFolderTree(BaseModel):
    """Both halves of the tree, kept apart because they behave differently:
    system nodes are read-only and user nodes can be renamed, moved, filled."""

    system: list[DocumentFolderNode] = Field(default_factory=list)
    folders: list[DocumentFolderNode] = Field(default_factory=list)


class DocumentFolderItemsRequest(BaseModel):
    document_ids: list[UUID] = Field(..., min_length=1, max_length=1000)


class DocumentFolderItemsResult(BaseModel):
    added: int = 0
    already_present: int = 0
    not_found: int = 0
    removed: int = 0


class DocumentFolderDeleteResult(BaseModel):
    deleted: str
    name: str
    subfolders_deleted: int = 0


class DocumentFolderRef(BaseModel):
    """A folder a given document sits in."""

    key: str
    id: str
    name: str
    color: Optional[str] = None


class DocumentFolderResponse(BaseModel):
    id: str
    name: str
    key: str
    parent_id: Optional[str] = None
    description: Optional[str] = None
    color: Optional[str] = None
    position: int = 0

    @classmethod
    def of(cls, folder: Any) -> "DocumentFolderResponse":
        return cls(
            id=str(folder.id),
            name=folder.name,
            key=f"user:{folder.id}",
            parent_id=str(folder.parent_id) if folder.parent_id else None,
            description=folder.description,
            color=folder.color,
            position=folder.position or 0,
        )
