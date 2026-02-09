"""
Analytics API endpoints for data analysis and visualization.
"""

from datetime import datetime
from typing import Optional, List
from uuid import UUID
from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.services.auth_service import get_current_user
from app.models.user import User
from app.services.analytics_service import analytics_service

router = APIRouter()


@router.get("/statistics")
async def get_collection_statistics(
    source_id: Optional[str] = Query(None, description="Filter by source ID"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get comprehensive statistics for document collection."""
    source_uuid = UUID(source_id) if source_id else None

    date_from_dt = None
    date_to_dt = None
    if date_from:
        try:
            date_from_dt = datetime.fromisoformat(date_from)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date_from format")
    if date_to:
        try:
            date_to_dt = datetime.fromisoformat(date_to)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date_to format")

    return await analytics_service.get_collection_statistics(
        db=db,
        source_id=source_uuid,
        tag=tag,
        date_from=date_from_dt,
        date_to=date_to_dt,
    )


@router.get("/sources")
async def get_source_analytics(
    source_id: Optional[str] = Query(None, description="Specific source ID"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get analytics for document sources."""
    source_uuid = UUID(source_id) if source_id else None
    sources = await analytics_service.get_source_analytics(db=db, source_id=source_uuid)
    return {"sources": sources}


@router.get("/trending")
async def get_trending_topics(
    days: int = Query(7, ge=1, le=90, description="Days to look back"),
    limit: int = Query(10, ge=1, le=50, description="Maximum topics"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get trending topics based on recent documents."""
    topics = await analytics_service.get_trending_topics(db=db, days=days, limit=limit)
    return {"trending_topics": topics, "period_days": days}


@router.get("/chart")
async def get_chart_data(
    metric: str = Query(..., description="Metric: document_count, file_size, content_size"),
    group_by: str = Query(..., description="Group by: source_type, file_type, author, date"),
    chart_type: str = Query("bar", description="Chart type: bar, line, pie, area"),
    date_from: Optional[str] = Query(None, description="Start date"),
    date_to: Optional[str] = Query(None, description="End date"),
    limit: int = Query(10, ge=1, le=100, description="Max data points"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Generate chart data for visualizations."""
    if metric not in ["document_count", "file_size", "content_size"]:
        raise HTTPException(status_code=400, detail="Invalid metric")
    if group_by not in ["source_type", "file_type", "author", "date"]:
        raise HTTPException(status_code=400, detail="Invalid group_by")

    date_from_dt = None
    date_to_dt = None
    if date_from:
        try:
            date_from_dt = datetime.fromisoformat(date_from)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date_from format")
    if date_to:
        try:
            date_to_dt = datetime.fromisoformat(date_to)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date_to format")

    return await analytics_service.generate_chart_data(
        db=db,
        chart_type=chart_type,
        metric=metric,
        group_by=group_by,
        date_from=date_from_dt,
        date_to=date_to_dt,
        limit=limit,
    )


@router.get("/export")
async def export_data(
    format: str = Query("json", description="Export format: json, csv, jsonl"),
    source_id: Optional[str] = Query(None, description="Filter by source"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    include_content: bool = Query(False, description="Include full content"),
    include_chunks: bool = Query(False, description="Include chunks"),
    limit: int = Query(1000, ge=1, le=10000, description="Max documents"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Export document data to file."""
    if format not in ["json", "csv", "jsonl"]:
        raise HTTPException(status_code=400, detail="Invalid format")

    source_uuid = UUID(source_id) if source_id else None

    content, filename, content_type = await analytics_service.export_data(
        db=db,
        format=format,
        source_id=source_uuid,
        tag=tag,
        include_content=include_content,
        include_chunks=include_chunks,
        limit=limit,
    )

    return Response(
        content=content,
        media_type=content_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
