"""
Reading lists API endpoints.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.database import get_db
from app.models.user import User
from app.models.document import Document, DocumentSource
from app.models.reading_list import ReadingList, ReadingListItem
from app.schemas.reading_list import (
    ReadingListCreate,
    ReadingListUpdate,
    ReadingListResponse,
    ReadingListListResponse,
    ReadingListItemCreate,
    ReadingListItemUpdate,
    ReadingListItemResponse,
)
from app.services.auth_service import get_current_user


router = APIRouter()


def _to_list_response(rl: ReadingList) -> ReadingListResponse:
    return ReadingListResponse(
        id=rl.id,
        user_id=rl.user_id,
        name=rl.name,
        description=rl.description,
        source_id=rl.source_id,
        created_at=rl.created_at,
        updated_at=rl.updated_at,
    )


@router.get("", response_model=ReadingListListResponse)
async def list_reading_lists(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        base = select(ReadingList).where(ReadingList.user_id == current_user.id)
        total = int((await db.execute(select(func.count()).select_from(base.subquery()))).scalar() or 0)
        result = await db.execute(
            base.order_by(desc(ReadingList.updated_at)).offset(offset).limit(limit)
        )
        items = [_to_list_response(x) for x in result.scalars().all()]
        return ReadingListListResponse(items=items, total=total, limit=limit, offset=offset)
    except Exception as exc:
        logger.error(f"Failed to list reading lists: {exc}")
        raise HTTPException(status_code=500, detail="Failed to list reading lists")


@router.post("", response_model=ReadingListResponse, status_code=201)
async def create_reading_list(
    payload: ReadingListCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    try:
        source: Optional[DocumentSource] = None
        if payload.source_id:
            source = await db.get(DocumentSource, payload.source_id)
            if not source:
                raise HTTPException(status_code=404, detail="Source not found")

        rl = ReadingList(
            user_id=current_user.id,
            name=payload.name,
            description=payload.description,
            source_id=payload.source_id,
        )
        db.add(rl)
        await db.flush()

        if payload.auto_populate_from_source and source:
            result = await db.execute(
                select(Document.id).where(Document.source_id == source.id).order_by(Document.created_at.asc())
            )
            doc_ids = [row[0] for row in result.all()]
            for idx, doc_id in enumerate(doc_ids):
                db.add(
                    ReadingListItem(
                        reading_list_id=rl.id,
                        document_id=doc_id,
                        status="to-read",
                        priority=0,
                        position=idx,
                    )
                )

        await db.commit()
        await db.refresh(rl)
        return _to_list_response(rl)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to create reading list: {exc}")
        await db.rollback()
        raise HTTPException(status_code=500, detail="Failed to create reading list")


@router.get("/{reading_list_id}", response_model=ReadingListResponse)
async def get_reading_list(
    reading_list_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rl = await db.get(ReadingList, reading_list_id)
    if not rl or rl.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Reading list not found")

    result = await db.execute(
        select(ReadingListItem, Document.title)
        .join(Document, Document.id == ReadingListItem.document_id)
        .where(ReadingListItem.reading_list_id == rl.id)
        .order_by(ReadingListItem.position.asc(), ReadingListItem.created_at.asc())
    )
    items = []
    for item, title in result.all():
        items.append(
            ReadingListItemResponse(
                id=item.id,
                reading_list_id=item.reading_list_id,
                document_id=item.document_id,
                document_title=title,
                status=item.status,
                priority=item.priority,
                position=item.position,
                notes=item.notes,
                created_at=item.created_at,
                updated_at=item.updated_at,
            )
        )
    return ReadingListResponse(
        id=rl.id,
        user_id=rl.user_id,
        name=rl.name,
        description=rl.description,
        source_id=rl.source_id,
        created_at=rl.created_at,
        updated_at=rl.updated_at,
        items=items,
    )


@router.put("/{reading_list_id}", response_model=ReadingListResponse)
async def update_reading_list(
    reading_list_id: UUID,
    payload: ReadingListUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rl = await db.get(ReadingList, reading_list_id)
    if not rl or rl.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Reading list not found")
    if payload.name is not None:
        rl.name = payload.name
    if payload.description is not None:
        rl.description = payload.description
    await db.commit()
    await db.refresh(rl)
    return _to_list_response(rl)


@router.post("/{reading_list_id}/items", response_model=ReadingListItemResponse, status_code=201)
async def add_reading_list_item(
    reading_list_id: UUID,
    payload: ReadingListItemCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rl = await db.get(ReadingList, reading_list_id)
    if not rl or rl.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Reading list not found")

    doc = await db.get(Document, payload.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if payload.position is None:
        max_pos = int(
            (await db.execute(select(func.max(ReadingListItem.position)).where(ReadingListItem.reading_list_id == rl.id))).scalar() or 0
        )
        position = max_pos + 1
    else:
        position = payload.position

    item = ReadingListItem(
        reading_list_id=rl.id,
        document_id=payload.document_id,
        status=payload.status,
        priority=payload.priority,
        position=position,
        notes=payload.notes,
    )
    db.add(item)
    await db.commit()
    await db.refresh(item)
    return ReadingListItemResponse(
        id=item.id,
        reading_list_id=item.reading_list_id,
        document_id=item.document_id,
        document_title=doc.title,
        status=item.status,
        priority=item.priority,
        position=item.position,
        notes=item.notes,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


@router.put("/{reading_list_id}/items/{item_id}", response_model=ReadingListItemResponse)
async def update_reading_list_item(
    reading_list_id: UUID,
    item_id: UUID,
    payload: ReadingListItemUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rl = await db.get(ReadingList, reading_list_id)
    if not rl or rl.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Reading list not found")
    item = await db.get(ReadingListItem, item_id)
    if not item or item.reading_list_id != rl.id:
        raise HTTPException(status_code=404, detail="Item not found")

    if payload.status is not None:
        item.status = payload.status
    if payload.priority is not None:
        item.priority = payload.priority
    if payload.position is not None:
        item.position = payload.position
    if payload.notes is not None:
        item.notes = payload.notes

    doc = await db.get(Document, item.document_id)
    await db.commit()
    await db.refresh(item)

    return ReadingListItemResponse(
        id=item.id,
        reading_list_id=item.reading_list_id,
        document_id=item.document_id,
        document_title=doc.title if doc else None,
        status=item.status,
        priority=item.priority,
        position=item.position,
        notes=item.notes,
        created_at=item.created_at,
        updated_at=item.updated_at,
    )


@router.delete("/{reading_list_id}/items/{item_id}")
async def delete_reading_list_item(
    reading_list_id: UUID,
    item_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    rl = await db.get(ReadingList, reading_list_id)
    if not rl or rl.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Reading list not found")
    item = await db.get(ReadingListItem, item_id)
    if not item or item.reading_list_id != rl.id:
        raise HTTPException(status_code=404, detail="Item not found")
    await db.delete(item)
    await db.commit()
    return {"message": "deleted"}

