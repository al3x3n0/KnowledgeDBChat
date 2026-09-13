"""Document folder endpoints.

The tree is per-user and so is every write here. `FolderError` carries its own
status code, so the service decides what a bad request means and this module
only translates — which keeps "a folder you do not own is a 404, not a 403" in
one place rather than repeated at each route.
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.user import User
from app.schemas.document_folder import (
    DocumentFolderCreate,
    DocumentFolderDeleteResult,
    DocumentFolderItemsRequest,
    DocumentFolderItemsResult,
    DocumentFolderRef,
    DocumentFolderResponse,
    DocumentFolderTree,
    DocumentFolderUpdate,
)
from app.services.auth_service import get_current_user
from app.services.document_folder_service import FolderError, document_folder_service

router = APIRouter()


def _http(error: FolderError) -> HTTPException:
    return HTTPException(status_code=error.status_code, detail=error.detail)


@router.get("/tree", response_model=DocumentFolderTree)
async def get_folder_tree(
    include_system: bool = Query(True, description="Include the computed folders"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """The whole tree: computed system folders, then this user's own.

    Counts are real counts taken at read time, which is the point of computing
    the system half rather than storing it.
    """
    folders = await document_folder_service.user_tree(db, current_user.id)
    system = (
        await document_folder_service.system_tree(db, current_user.id)
        if include_system
        else []
    )
    return DocumentFolderTree(system=system, folders=folders)


@router.post("", response_model=DocumentFolderResponse, status_code=201)
async def create_folder(
    payload: DocumentFolderCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        folder = await document_folder_service.create(
            db,
            current_user.id,
            name=payload.name,
            parent_id=payload.parent_id,
            description=payload.description,
            color=payload.color,
            position=payload.position,
        )
    except FolderError as error:
        raise _http(error) from error
    return DocumentFolderResponse.of(folder)


@router.patch("/{folder_id}", response_model=DocumentFolderResponse)
async def update_folder(
    folder_id: UUID,
    payload: DocumentFolderUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Rename, recolour, reorder, or move a folder.

    A move only happens when `reparent` is true, because `parent_id: null`
    otherwise cannot be told apart from "not supplied".
    """
    try:
        folder = await document_folder_service.update(
            db,
            current_user.id,
            folder_id,
            name=payload.name,
            description=payload.description,
            color=payload.color,
            position=payload.position,
            parent_id=payload.parent_id,
            reparent=payload.reparent,
        )
    except FolderError as error:
        raise _http(error) from error
    return DocumentFolderResponse.of(folder)


@router.delete("/{folder_id}", response_model=DocumentFolderDeleteResult)
async def delete_folder(
    folder_id: UUID,
    recursive: bool = Query(False, description="Also delete subfolders"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete a folder. The documents in it are never touched."""
    try:
        result = await document_folder_service.delete(
            db, current_user.id, folder_id, recursive=recursive
        )
    except FolderError as error:
        raise _http(error) from error
    return DocumentFolderDeleteResult(**result)


@router.post("/{folder_id}/documents", response_model=DocumentFolderItemsResult)
async def add_documents_to_folder(
    folder_id: UUID,
    payload: DocumentFolderItemsRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """File documents into a folder. Idempotent, and reports what it found."""
    try:
        result = await document_folder_service.add_documents(
            db, current_user.id, folder_id, payload.document_ids
        )
    except FolderError as error:
        raise _http(error) from error
    return DocumentFolderItemsResult(**result)


@router.delete("/{folder_id}/documents", response_model=DocumentFolderItemsResult)
async def remove_documents_from_folder(
    folder_id: UUID,
    payload: DocumentFolderItemsRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Unfile documents from a folder, leaving the documents in place."""
    try:
        result = await document_folder_service.remove_documents(
            db, current_user.id, folder_id, payload.document_ids
        )
    except FolderError as error:
        raise _http(error) from error
    return DocumentFolderItemsResult(**result)


@router.get("/for-document/{document_id}", response_model=list[DocumentFolderRef])
async def folders_for_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Which of this user's folders hold a document — for the document detail
    view, so a document can show where it is filed."""
    return await document_folder_service.folders_for_document(
        db, current_user.id, document_id
    )
