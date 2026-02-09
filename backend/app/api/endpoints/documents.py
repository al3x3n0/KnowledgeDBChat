"""
Document-related API endpoints.
"""

import os
import hashlib
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.database import get_db, AsyncSessionLocal
from app.core.rate_limit import limiter, UPLOAD_LIMIT
from app.models.user import User
from app.services.auth_service import get_current_user
from app.services.document_service import DocumentService
from app.services.text_processor import TextProcessor
from app.services.transcription_service import TranscriptionService
from app.services.storage_service import storage_service
from app.utils.exceptions import DocumentNotFoundError, ValidationError
from app.utils.validators import validate_file_type
from app.core.logging import log_error
from app.schemas.document import (
    DocumentResponse,
    DocumentSourceResponse,
    DocumentSourceCreate,
    DocumentUpload,
    GitRepoSourceRequest,
    ArxivSourceRequest,
    ActiveSourceStatus,
    IngestUrlRequest,
    IngestUrlResponse,
    IngestUrlJobResponse,
    InstantArxivIngestRequest,
    InstantArxivIngestResponse,
    ResearchPresentationRequest,
)
from app.schemas.common import PaginatedResponse
from app.tasks.summarization_tasks import summarize_document as summarize_task
from sqlalchemy import select as sql_select
from app.models.document import Document as _Document, DocumentSource as _DocumentSource
from app.tasks.ingestion_tasks import ingest_from_source
from app.core.celery import celery_app
from celery.result import AsyncResult
from app.utils.ingestion_state import (
    set_ingestion_task_mapping,
    get_ingestion_task_mapping,
    set_ingestion_cancel_flag,
)
from datetime import datetime
from uuid import uuid4
import re
import uuid

router = APIRouter()
document_service = DocumentService()


@router.get("/sources/{source_id}/download-zip")
async def download_document_source_zip(
    source_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Download all Documents in a DocumentSource as a ZIP.

    Intended for generated projects and repo sources. Access is restricted to:
    - admins, or
    - sources whose config indicates they were requested by the current user.
    """
    import io
    import zipfile

    try:
        source = await db.get(_DocumentSource, source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        cfg = source.config or {}
        requested_user_id = str(cfg.get("requested_by_user_id") or cfg.get("requestedByUserId") or "").strip()
        requested_by = str(cfg.get("requested_by") or cfg.get("requestedBy") or "").strip()

        if not current_user.is_admin():
            if requested_user_id and requested_user_id != str(current_user.id):
                raise HTTPException(status_code=403, detail="Not authorized for this source")
            if not requested_user_id and requested_by and requested_by != current_user.username:
                raise HTTPException(status_code=403, detail="Not authorized for this source")
            if not requested_user_id and not requested_by:
                raise HTTPException(status_code=403, detail="Not authorized for this source")

        res = await db.execute(sql_select(_Document).where(_Document.source_id == source_id))
        docs = res.scalars().all()
        if not docs:
            raise HTTPException(status_code=404, detail="No documents found for this source")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for d in docs:
                p = (d.file_path or d.source_identifier or d.title or str(d.id) or "").replace("\\", "/").strip()
                p = re.sub(r"^/+", "", p)
                while p.startswith("./"):
                    p = p[2:]
                parts = [x for x in p.split("/") if x not in {"", ".", ".."}]
                safe_path = "/".join(parts) or f"file_{str(d.id)}.txt"
                if len(safe_path) > 240:
                    safe_path = safe_path[-240:]
                zf.writestr(safe_path, d.content or "")

            manifest = {
                "source_id": str(source.id),
                "source_name": source.name,
                "source_type": source.source_type,
                "generated_at": datetime.utcnow().isoformat(),
                "documents_count": len(docs),
                "config": cfg,
            }
            zf.writestr("_manifest.json", json.dumps(manifest, indent=2))

        buf.seek(0)
        filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.name or "source")[:80] + ".zip"
        headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
        return StreamingResponse(buf, media_type="application/zip", headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating zip for source {source_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create ZIP")
text_processor_service = TextProcessor()


@router.get("/search")
async def search_documents(
    q: str = Query(..., min_length=1, description="Search query"),
    mode: str = Query("smart", regex="^(smart|keyword|exact)$", description="Search mode"),
    sort_by: str = Query("relevance", regex="^(relevance|date|title)$", description="Sort field"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Results per page"),
    source_id: Optional[str] = Query(None, description="Filter by source ID"),
    file_type: Optional[str] = Query(None, description="Filter by file type"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Search documents with multiple modes.

    Modes:
    - smart: Hybrid semantic + keyword search with reranking (recommended)
    - keyword: BM25 keyword search only
    - exact: SQL LIKE matching for exact phrases
    """
    from app.services.search_service import search_service
    from app.schemas.search import SearchResponse, SearchResult

    try:
        results, total, took_ms = await search_service.search(
            query=q,
            mode=mode,
            page=page,
            page_size=page_size,
            sort_by=sort_by,
            sort_order=sort_order,
            source_id=source_id,
            file_type=file_type,
            db=db,
        )

        return SearchResponse(
            results=[SearchResult(**r) for r in results],
            total=total,
            page=page,
            page_size=page_size,
            query=q,
            mode=mode,
            took_ms=took_ms,
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=PaginatedResponse[DocumentResponse])
async def get_documents(
    page: int = 1,
    page_size: int = 20,
    skip: Optional[int] = Query(None, ge=0, description="Optional offset (alternative to page/page_size)"),
    limit: Optional[int] = Query(None, ge=1, le=2000, description="Optional limit (alternative to page_size)"),
    source_id: Optional[UUID] = None,
    search: Optional[str] = None,
    order_by: Optional[str] = "updated_at",
    order: Optional[str] = "desc",
    owner_persona_id: Optional[UUID] = Query(None),
    persona_id: Optional[UUID] = Query(None),
    persona_role: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get paginated list of documents.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page (1-2000)
        source_id: Optional source ID filter
        search: Optional search term
        order_by: Field to order by (default: updated_at)
        order: Sort order (asc or desc, default: desc)
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Paginated response with documents
    """
    
    try:
        # Validate pagination parameters
        if page < 1:
            raise ValidationError("Page must be >= 1", field="page")
        if page_size < 1 or page_size > 2000:
            raise ValidationError("Page size must be between 1 and 2000", field="page_size")
        if limit is not None and (limit < 1 or limit > 2000):
            raise ValidationError("Limit must be between 1 and 2000", field="limit")
        if order not in ["asc", "desc"]:
            raise ValidationError("Order must be 'asc' or 'desc'", field="order")
        
        # Support legacy skip/limit query params used by parts of the frontend.
        effective_page_size = limit or page_size
        effective_skip = skip if skip is not None else (page - 1) * effective_page_size
        
        # Get documents with total count
        documents, total = await document_service.get_documents(
            skip=effective_skip,
            limit=effective_page_size,
            source_id=source_id,
            search=search,
            order_by=order_by,
            order=order,
            owner_persona_id=owner_persona_id,
            persona_id=persona_id,
            persona_role=persona_role,
            db=db
        )
        logger.debug(
            "Fetched documents for listing",
            extra={
                "skip": effective_skip,
                "limit": effective_page_size,
                "result_count": len(documents),
                "total": total,
                "source_id": str(source_id) if source_id else None,
                "search": search,
                "order_by": order_by,
                "order": order,
            }
        )
        
        # Convert to response models
        # For list view, we don't need chunks, so set to empty list to avoid lazy loading issues
        items = []
        for doc in documents:
            # Set chunks to empty list for list view (chunks are only needed in detail view)
            # This prevents SQLAlchemy from trying to lazy load the relationship in async context
            doc.chunks = []
            doc.persona_detections = []
            try:
                items.append(DocumentResponse.from_orm(doc))
            except Exception as exc:
                logger.error(
                    "Failed to serialize document for response",
                    extra={
                        "document_id": getattr(doc, "id", None),
                        "title": getattr(doc, "title", None),
                        "source_id": getattr(doc, "source_id", None),
                    },
                    exc_info=exc,
                )
                raise
        
        # Return paginated response
        return PaginatedResponse.create(
            items=items,
            total=total,
            page=page,
            page_size=effective_page_size
        )
    except ValidationError:
        raise
    except Exception as e:
        logger.exception(
            "Unhandled error retrieving documents",
            extra={
                "page": page,
                "page_size": page_size,
                "source_id": str(source_id) if source_id else None,
                "search": search,
                "order_by": order_by,
                "order": order,
            }
        )
        log_error(e, context={"page": page, "page_size": page_size})
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")


@router.get("/{document_id}/related")
async def get_related_documents(
    document_id: UUID,
    limit: int = Query(8, ge=1, le=25),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Recommend related documents based on KG overlap + embedding similarity."""
    _ = current_user  # authenticated access required
    try:
        doc = await document_service.get_document(document_id, db)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        from app.services.paper_recommendation_service import PaperRecommendationService

        svc = PaperRecommendationService()
        items = await svc.related_documents(db, document_id, limit=limit)
        return {"items": items, "limit": limit}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to get related documents: {exc}")
        raise HTTPException(status_code=500, detail="Failed to get related documents")


async def get_current_user_optional(
    request: Request,
    token: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """
    Get current user with optional authentication.
    Supports both Authorization header and token query parameter.
    """
    # Try token from query parameter first (for video players)
    if token:
        try:
            from jose import jwt
            from app.core.config import settings
            from sqlalchemy import select
            
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            user_id: str = payload.get("sub")
            if user_id:
                result = await db.execute(select(User).where(User.id == user_id))
                user = result.scalar_one_or_none()
                if user and user.is_active:
                    return user
        except Exception as e:
            logger.debug(f"Token query param authentication failed: {e}")
    
    # Try Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        try:
            token = auth_header.replace("Bearer ", "")
            from jose import jwt
            from app.core.config import settings
            from sqlalchemy import select
            
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            user_id: str = payload.get("sub")
            if user_id:
                result = await db.execute(select(User).where(User.id == user_id))
                user = result.scalar_one_or_none()
                if user and user.is_active:
                    return user
        except Exception as e:
            logger.debug(f"Authorization header authentication failed: {e}")
    
    return None


@router.head("/{document_id}/download")
async def download_document_head(
    document_id: UUID,
    request: Request,
    token: Optional[str] = Query(None),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
    use_proxy: bool = Query(True),
):
    """
    Handle HEAD request for video streaming (used by players to check availability).
    Returns metadata without the actual file content.
    """
    # Require authentication
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        from app.services.storage_service import storage_service
        
        document = await document_service.get_document(document_id, db)
        if not document or not document.file_path:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Normalize file path
        file_path = document.file_path
        if file_path.startswith('./'):
            file_path = file_path[2:]
        if file_path.startswith('documents/'):
            file_path = file_path[10:]
        
        # Check if file exists
        file_exists = await storage_service.file_exists(file_path)
        if not file_exists:
            raise HTTPException(status_code=404, detail="File not found in storage")
        
        if use_proxy:
            # Get file metadata
            await storage_service.initialize()
            metadata = await storage_service.get_file_metadata(file_path)
            file_size = metadata.get("size", 0)
            content_type = metadata.get("content_type") or "application/octet-stream"
            filename = os.path.basename(file_path) or document.title or f"document_{document_id}"
            
            from fastapi.responses import Response
            response = Response(
                status_code=200,
                headers={
                    "Content-Type": content_type,
                    "Content-Length": str(file_size),
                    "Accept-Ranges": "bytes",
                    "Content-Disposition": f'inline; filename="{filename}"',
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
                    "Access-Control-Allow-Headers": "Range, Authorization, Content-Type",
                    "Access-Control-Expose-Headers": "Content-Range, Content-Length, Accept-Ranges",
                }
            )
            return response
        else:
            raise HTTPException(status_code=400, detail="HEAD requests only supported with use_proxy=true")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in HEAD request for document {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get file metadata")


@router.get("/{document_id}/download")
async def download_document(
    document_id: UUID,
    request: Request,  # Added to access headers for Range requests
    token: Optional[str] = Query(None, description="JWT token for authentication (alternative to Authorization header)"),
    current_user: Optional[User] = Depends(get_current_user_optional),
    db: AsyncSession = Depends(get_db),
    use_proxy: bool = Query(True, description="If True, stream file through backend; if False, return presigned URL")
):
    """
    Download a document file.
    
    If use_proxy=True (default): Streams file through backend (avoids signature issues)
    If use_proxy=False: Returns presigned URL for direct download
    
    Authentication can be provided via:
    - Authorization header (Bearer token) - standard method
    - token query parameter - for video players that don't support custom headers
    """
    # Require authentication
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    """
    Download a document file.
    
    If use_proxy=True (default): Streams file through backend (avoids signature issues)
    If use_proxy=False: Returns presigned URL for direct download
    """
    try:
        from app.services.storage_service import storage_service
        
        logger.info(f"Download request for document {document_id} by user {current_user.id}")
        document = await document_service.get_document(document_id, db)
        if not document:
            logger.warning(f"Document {document_id} not found")
            raise HTTPException(status_code=404, detail="Document not found")
        
        logger.info(f"Document {document_id} found, file_path: {document.file_path}")
        if not document.file_path:
            logger.warning(f"Document {document_id} has no file_path")
            raise HTTPException(status_code=404, detail="Document file not found")
        
        # Normalize file_path - handle old format (data/documents/...) and new format (documents/{id}/...)
        file_path = document.file_path
        original_path = file_path
        
        # Remove leading './' if present (common in old file paths)
        if file_path.startswith('./'):
            file_path = file_path[2:]  # Remove './' prefix
            logger.info(f"Removed './' prefix from file_path: {original_path} -> {file_path}")
        
        # Remove 'documents/' prefix if present (bucket name is already 'documents')
        # This handles cases where file_path is stored as "documents/{id}/{filename}"
        if file_path.startswith('documents/'):
            file_path = file_path[10:]  # Remove 'documents/' prefix (10 chars)
            logger.info(f"Removed 'documents/' prefix from file_path, new path: {file_path}")
        
        # Try to find the file in MinIO
        from app.services.storage_service import storage_service
        
        # First, try the path as-is (after removing ./ and documents/)
        file_exists = await storage_service.file_exists(file_path)
        
        # If not found and path starts with 'data/documents/', try new format
        if not file_exists and file_path.startswith('data/documents/'):
            # Old format: try to find in new format location
            filename = file_path.split('/')[-1]
            # New format: {document_id}/{filename} (no 'documents/' prefix since bucket name is 'documents')
            new_path = f"{document_id}/{filename}"
            logger.info(f"File not found at old path {file_path}, trying new format: {new_path}")
            
            if await storage_service.file_exists(new_path):
                file_path = new_path
                file_exists = True
                logger.info(f"Found file at new location: {file_path}")
        
        # If still not found, try with 'documents/' prefix (some files were uploaded with prefix)
        if not file_exists:
            path_with_prefix = f"documents/{file_path}" if not file_path.startswith('documents/') else file_path
            logger.info(f"File not found at {file_path}, trying with 'documents/' prefix: {path_with_prefix}")
            if await storage_service.file_exists(path_with_prefix):
                file_path = path_with_prefix
                file_exists = True
                logger.info(f"Found file with 'documents/' prefix: {file_path}")
        
        # If file still doesn't exist, return 404
        if not file_exists:
            logger.error(f"File not found in MinIO at any path: original={original_path}, tried={file_path}")
            raise HTTPException(
                status_code=404, 
                detail=f"Document file not found in MinIO storage. The file may have been deleted."
            )
        
        # If use_proxy is True, stream the file through the backend
        # This avoids presigned URL signature issues
        if use_proxy:
            try:
                # Ensure MinIO is initialized
                await storage_service.initialize()
                
                # Get file metadata for headers
                metadata = await storage_service.get_file_metadata(file_path)
                file_size = metadata.get("size", 0)
                
                # Get filename from document title or file path
                filename = os.path.basename(file_path) or document.title or f"document_{document_id}"
                # Ensure filename has extension if file_path has one
                if '.' not in filename and '.' in file_path:
                    ext = os.path.splitext(file_path)[1]
                    filename = f"{filename}{ext}"
                
                # Determine content type
                content_type = metadata.get("content_type") or "application/octet-stream"
                
                # Handle HTTP Range requests for video streaming
                # Check both "Range" and "range" headers (browsers may send either)
                range_header = request.headers.get("Range") or request.headers.get("range")
                
                if range_header:
                    # Parse Range header (e.g., "bytes=0-1023" or "bytes=1024-")
                    try:
                        range_match = range_header.replace("bytes=", "").split("-")
                        start = int(range_match[0]) if range_match[0] else 0
                        end = int(range_match[1]) if range_match[1] and range_match[1] else file_size - 1
                        
                        # Validate range
                        if start < 0 or end >= file_size or start > end:
                            from fastapi.responses import Response
                            return Response(status_code=416, headers={"Content-Range": f"bytes */{file_size}"})
                        
                        # Calculate content length for partial content
                        content_length = end - start + 1
                        
                        logger.info(f"Range request for document {document_id}: bytes {start}-{end} of {file_size}")
                        
                        # Get partial file stream
                        file_stream = storage_service.get_file_stream_range(file_path, start, end)
                        
                        response = StreamingResponse(
                            file_stream,
                            status_code=206,  # Partial Content
                            media_type=content_type,
                            headers={
                                "Content-Range": f"bytes {start}-{end}/{file_size}",
                                "Content-Length": str(content_length),
                                "Accept-Ranges": "bytes",
                                "Content-Disposition": f'inline; filename="{filename}"',
                                "Cache-Control": "public, max-age=3600",
                            }
                        )
                        # Add CORS headers for video streaming
                        response.headers["Access-Control-Allow-Origin"] = "*"
                        response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, OPTIONS"
                        response.headers["Access-Control-Allow-Headers"] = "Range, Authorization, Content-Type"
                        response.headers["Access-Control-Expose-Headers"] = "Content-Range, Content-Length, Accept-Ranges"
                        return response
                    except (ValueError, IndexError):
                        # Invalid range header, fall through to full file
                        logger.warning(f"Invalid Range header: {range_header}")
                
                # No range request or invalid range - return full file
                logger.info(f"Streaming full file download for document {document_id}: {filename} ({file_size} bytes)")
                file_stream = storage_service.get_file_stream(file_path)
                
                response = StreamingResponse(
                    file_stream,
                    media_type=content_type,
                    headers={
                        "Content-Disposition": f'inline; filename="{filename}"',  # Changed to inline for video playback
                        "Content-Length": str(file_size),
                        "Accept-Ranges": "bytes",  # Enable range requests
                        "Cache-Control": "public, max-age=3600",  # Cache for 1 hour
                    }
                )
                # Add CORS headers for video streaming
                response.headers["Access-Control-Allow-Origin"] = "*"
                response.headers["Access-Control-Allow-Methods"] = "GET, HEAD, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = "Range, Authorization, Content-Type"
                response.headers["Access-Control-Expose-Headers"] = "Content-Range, Content-Length, Accept-Ranges"
                return response
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="File not found in storage")
            except Exception as e:
                logger.error(f"Error streaming file for document {document_id}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to stream file: {str(e)}")
        else:
            # Legacy mode: return presigned URL
            # Generate fresh presigned URL (always generate new one to avoid expiry)
            download_url = await storage_service.get_presigned_download_url(file_path)
            
            logger.debug(f"Generated download URL for document {document_id}")
            return JSONResponse(content={"download_url": download_url})
    
    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"Invalid file path for document {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid file path: {str(e)}")
    except Exception as e:
        logger.error(f"Error generating download URL for document {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate download URL")


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific document by ID."""
    try:
        document = await document_service.get_document(document_id, db)
        if not document:
            raise DocumentNotFoundError(str(document_id))
        return DocumentResponse.from_orm(document)
    except DocumentNotFoundError:
        raise
    except Exception as e:
        log_error(e, context={"document_id": str(document_id)})
        raise HTTPException(status_code=500, detail="Failed to retrieve document")


@router.post("/upload")
@limiter.limit(UPLOAD_LIMIT)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Upload a document file."""
    try:
        # Validate file type
        if not validate_file_type(file.filename, file.content_type):
            raise ValidationError(
                f"File type not allowed: {file.content_type}",
                field="file"
            )
        
        # Validate file size (configurable, larger limit for videos)
        from app.core.config import settings
        from app.utils.validators import ALLOWED_EXTENSIONS
        
        # Check if it's a video/audio file
        file_ext = os.path.splitext(file.filename or "")[1].lower()
        is_video_audio = file_ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv', '.wmv', 
                                      '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac']
        
        # Use larger limit for video/audio files
        max_size = settings.MAX_VIDEO_SIZE if is_video_audio else settings.MAX_FILE_SIZE
        
        file_content = await file.read()
        file_size = len(file_content)
        if file_size > max_size:
            size_mb = file_size / (1024*1024)
            max_mb = max_size / (1024*1024)
            raise ValidationError(
                f"File size ({size_mb:.2f}MB) exceeds maximum allowed size of {max_mb:.0f}MB",
                field="file"
            )
        
        # Reset file pointer and create a new file-like object with the content for the service
        from io import BytesIO
        # Reset the file pointer to the beginning
        await file.seek(0)
        # Create a new BytesIO object with the content
        file.file = BytesIO(file_content)
        # Do not assign to file.content_type: it's a read-only property on Starlette's UploadFile.
        
        # Parse tags if provided
        tag_list = []
        if tags:
            from app.utils.validators import validate_tags
            tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
            if not validate_tags(tag_list):
                raise ValidationError("Invalid tag format", field="tags")
        
        document = await document_service.upload_file(
            file=file,
            title=title,
            tags=tag_list,
            db=db,
            owner_user=current_user
        )
        
        return {"message": "Document uploaded successfully", "document_id": document.id}
    
    except (ValidationError, DocumentNotFoundError):
        raise
    except Exception as e:
        error_detail = str(e)
        logger.error(f"Error uploading document {file.filename if file else 'unknown'}: {error_detail}", exc_info=True)
        log_error(e, context={"filename": file.filename if file else None, "file_type": file.content_type if file else None})
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {error_detail}")


@router.post("/ingest-url", response_model=IngestUrlResponse)
async def ingest_url(
    request: IngestUrlRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Scrape a URL and ingest the extracted text into the KnowledgeDB as document(s).

    Convenience API for the platform UI; reuses built-in `ingest_url` tool logic.
    """
    try:
        from app.services.agent_service import AgentService

        agent_service = AgentService()
        result = await agent_service.execute_tool(
            tool_name="ingest_url",
            tool_input=request.model_dump(exclude_none=True),
            user_id=current_user.id,
            db=db,
        )
        if isinstance(result, dict) and result.get("error"):
            raise HTTPException(status_code=400, detail=result.get("error"))
        return IngestUrlResponse(**(result or {}))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting URL: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to ingest URL")


@router.post("/ingest-url/async", response_model=IngestUrlJobResponse)
async def ingest_url_async(
    request: IngestUrlRequest,
    current_user: User = Depends(get_current_user),
):
    """Schedule URL ingestion as a background job with progress over WebSocket."""
    try:
        from app.tasks.url_ingestion_tasks import ingest_url as ingest_url_task
        from app.core.cache import get_redis_client

        task_res = ingest_url_task.delay(request.model_dump(exclude_none=True), str(current_user.id))
        job_id = task_res.id
        progress_key = f"url_ingest:{job_id}"

        # Store owner mapping for progress authorization (best-effort).
        try:
            client = await get_redis_client()
            await client.setex(f"url_ingest:owner:{job_id}", 7200, str(current_user.id).encode("utf-8"))
        except Exception:
            pass

        return IngestUrlJobResponse(job_id=job_id, progress_key=progress_key)
    except Exception as e:
        logger.error(f"Failed to schedule URL ingestion: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to schedule URL ingestion")


@router.delete("/{document_id}")
async def delete_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a document."""
    try:
        logger.info(f"Delete request for document {document_id} by user {current_user.id}")
        # Prevent deletion while transcoding
        from sqlalchemy import select as sql_select
        from app.models.document import Document
        res = await db.execute(sql_select(Document).where(Document.id == document_id))
        doc = res.scalar_one_or_none()
        if doc and isinstance(doc.extra_metadata, dict) and doc.extra_metadata.get("is_transcoding"):
            logger.info(f"Delete denied for {document_id}: currently transcoding")
            raise HTTPException(status_code=409, detail="Document is being converted. Please wait until it finishes.")
        success = await document_service.delete_document(document_id, db)
        if not success:
            logger.warning(f"Document {document_id} not found or deletion failed")
            raise HTTPException(status_code=404, detail="Document not found")
        
        logger.info(f"Successfully deleted document {document_id}")
        return {"message": "Document deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}", exc_info=True)
        error_detail = str(e) if str(e) else "Failed to delete document"
        raise HTTPException(status_code=500, detail=error_detail)


@router.delete("/sources/{source_id}/documents")
async def delete_documents_for_source(
    source_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete all documents for a specific Git source (GitHub/GitLab)."""
    try:
        source = await db.get(_DocumentSource, source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Document source not found")
        if source.source_type not in ("github", "gitlab"):
            raise HTTPException(status_code=400, detail="Bulk deletion only supported for Git sources")

        config = source.config or {}
        requested_by = config.get("requested_by") or config.get("requestedBy")
        if not current_user.is_admin() and requested_by != current_user.username:
            raise HTTPException(status_code=403, detail="Not authorized to modify this source")

        deleted = await document_service.delete_documents_by_source(source_id, db)
        logger.info(f"Deleted {deleted} documents for source {source_id}")
        return {"message": f"Deleted {deleted} documents", "deleted": deleted}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting documents for source {source_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete documents for source")


@router.post("/{document_id}/presentation/audio")
async def attach_presentation_audio(
    document_id: UUID,
    audio: UploadFile = File(...),
    language: str = Form("ru"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Attach an audio narration to a presentation document and align it with slides."""
    try:
        result = await db.execute(sql_select(_Document).where(_Document.id == document_id))
        document = result.scalar_one_or_none()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        presentation_meta = await _ensure_presentation_metadata(document, db)
        if not presentation_meta or not presentation_meta.get("slides"):
            raise HTTPException(status_code=400, detail="Presentation metadata unavailable for this document")
        if not audio or not audio.filename:
            raise HTTPException(status_code=400, detail="Audio file required")
        audio_ext = os.path.splitext(audio.filename)[1].lower()
        allowed_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
        if audio_ext not in allowed_exts:
            raise HTTPException(status_code=400, detail="Unsupported audio format")
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=audio_ext or ".tmp")
        temp_audio.write(audio_bytes)
        temp_audio_path = Path(temp_audio.name)
        temp_audio.close()
        try:
            object_path = await storage_service.upload_file(
                document.id,
                f"presentation_audio_{uuid.uuid4().hex}{audio_ext}",
                audio_bytes,
                audio.content_type or "audio/mpeg"
            )
            transcription_service = TranscriptionService()
            transcript_text, metadata = transcription_service.transcribe_file(
                temp_audio_path,
                language=language or "ru"
            )
            segments = metadata.get("sentence_segments") or metadata.get("segments")
            if not segments:
                raise HTTPException(status_code=500, detail="Unable to extract speech segments from audio")
            alignment = _align_segments_to_slides(presentation_meta.get("slides", []), segments)
            audio_duration = metadata.get("duration")
            if not audio_duration and segments:
                audio_duration = segments[-1].get("end")
            transcript_doc = _Document(
                title=f"{document.title or 'Presentation'} - Narration Transcript",
                content=transcript_text,
                content_hash=hashlib.sha256(transcript_text.encode("utf-8")).hexdigest(),
                url=None,
                file_path=None,
                file_type="text/plain",
                file_size=len(transcript_text.encode("utf-8")),
                source_id=document.source_id,
                source_identifier=f"{document.source_identifier}:audio_transcript:{uuid.uuid4().hex[:8]}",
                extra_metadata={
                    "doc_type": "transcript",
                    "parent_document_id": str(document.id),
                    "presentation_audio": True,
                    "transcription_metadata": metadata,
                },
            )
            db.add(transcript_doc)
            await db.commit()
            await db.refresh(transcript_doc)
            await document_service._process_document_async(transcript_doc, db)
            presentation_meta["audio_track"] = {
                "object_path": object_path,
                "file_name": audio.filename,
                "content_type": audio.content_type or "audio/mpeg",
                "duration": audio_duration,
                "alignment": alignment,
                "transcript_document_id": str(transcript_doc.id),
                "language": language,
                "created_at": datetime.utcnow().isoformat(),
            }
            document.extra_metadata = document.extra_metadata or {}
            document.extra_metadata["presentation"] = presentation_meta
            await db.commit()
            await db.refresh(document)
            audio_url = await storage_service.get_presigned_download_url(object_path)
            return {
                "message": "Audio narration synchronized",
                "audio_url": audio_url,
                "alignment": alignment,
                "duration": audio_duration,
                "audio_track": presentation_meta["audio_track"],
                "transcript_document_id": str(transcript_doc.id),
            }
        finally:
            try:
                if temp_audio_path.exists():
                    temp_audio_path.unlink()
            except Exception:
                pass
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error attaching audio to presentation {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to attach audio narration")


@router.get("/{document_id}/presentation/audio")
async def get_presentation_audio(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Return signed URL and alignment metadata for presentation audio."""
    try:
        result = await db.execute(sql_select(_Document).where(_Document.id == document_id))
        document = result.scalar_one_or_none()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        presentation_meta = (document.extra_metadata or {}).get("presentation") or {}
        audio_track = presentation_meta.get("audio_track")
        if not audio_track:
            raise HTTPException(status_code=404, detail="Audio track not available")
        object_path = audio_track.get("object_path")
        if not object_path:
            raise HTTPException(status_code=404, detail="Audio file missing")
        audio_url = await storage_service.get_presigned_download_url(object_path)
        return {
            "audio_url": audio_url,
            "alignment": audio_track.get("alignment") or [],
            "duration": audio_track.get("duration"),
            "transcript_document_id": audio_track.get("transcript_document_id"),
            "file_name": audio_track.get("file_name"),
            "content_type": audio_track.get("content_type"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving presentation audio for {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve audio narration")


@router.post("/sources/{source_id}/cancel")
async def cancel_user_source_ingestion(
    source_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Allow source owners to request cancellation of an active/pending ingestion."""
    try:
        source = await db.get(_DocumentSource, source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Document source not found")
        config = source.config or {}
        requested_by = config.get("requested_by") or config.get("requestedBy")
        if not current_user.is_admin() and requested_by != current_user.username:
            raise HTTPException(status_code=403, detail="Not authorized to cancel this source")

        source_id_str = str(source_id)
        task_id = await get_ingestion_task_mapping(source_id_str)

        await set_ingestion_cancel_flag(source_id_str, ttl=600)

        if task_id:
            try:
                AsyncResult(task_id, app=celery_app).revoke(terminate=True)
            except Exception as revoke_err:
                logger.warning(f"Failed to revoke task {task_id} for source {source_id}: {revoke_err}")

        return {"message": "Cancellation requested", "task_id": task_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling ingestion for source {source_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel ingestion for source")


@router.post("/reprocess/{document_id}")
async def reprocess_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Reprocess a document for indexing."""
    try:
        success = await document_service.reprocess_document(document_id, db, user_id=current_user.id)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": "Document reprocessing started"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing document: {e}")
        raise HTTPException(status_code=500, detail="Failed to reprocess document")


@router.post("/{document_id}/transcribe")
async def retrigger_transcription(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Manually retrigger transcription for a document.
    If the document is a non-MP4 video and not transcoded, schedule transcode first.
    """
    try:
        # Load document
        from sqlalchemy import select
        from app.models.document import Document as _Document
        result = await db.execute(select(_Document).where(_Document.id == document_id))
        document = result.scalar_one_or_none()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        # Clear previous errors
        document.processing_error = None
        document.extra_metadata = document.extra_metadata or {}
        document.extra_metadata.pop("transcription_error", None)
        document.extra_metadata.pop("transcription_traceback", None)

        # Detect type
        ext = (document.file_path or "").split("/")[-1].lower()
        ext = ('.' + ext.split('.')[-1]) if '.' in ext else ''
        video_exts = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv"}
        is_video = (document.file_type or "").startswith("video/") or ext in video_exts
        is_mp4 = (document.file_type == "video/mp4") or ext == ".mp4"

        # Publish status helper
        try:
            from app.tasks.transcription_tasks import _publish_status as publish_status_trans
            from app.tasks.transcode_tasks import _publish_status as publish_status_transcode
        except Exception:
            publish_status_trans = publish_status_transcode = None

        if is_video and not is_mp4 and not (document.extra_metadata or {}).get("is_transcoded"):
            # Transcode first
            document.extra_metadata["is_transcoding"] = True
            document.extra_metadata["is_transcribing"] = False
            await db.commit()
            if publish_status_transcode:
                publish_status_transcode(str(document.id), {
                    "is_transcoding": True,
                    "is_transcribing": False,
                    "is_transcribed": False,
                })
            from app.tasks.transcode_tasks import transcode_to_mp4
            transcode_to_mp4.delay(str(document.id))
            return {"message": "Transcode scheduled, transcription will start after conversion"}
        else:
            # Transcribe now
            document.extra_metadata["is_transcribing"] = True
            await db.commit()
            if publish_status_trans:
                publish_status_trans(str(document.id), {
                    "is_transcoding": False,
                    "is_transcribing": True,
                    "is_transcribed": False,
                })
            from app.tasks.transcription_tasks import transcribe_document
            transcribe_document.delay(str(document.id))
            return {"message": "Transcription scheduled"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error re-triggering transcription for {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to schedule transcription")


@router.websocket("/{document_id}/transcription-progress")
async def transcription_progress_websocket(
    websocket: WebSocket,
    document_id: UUID
):
    """WebSocket endpoint for real-time transcription progress updates."""
    from app.utils.websocket_auth import require_websocket_auth
    from app.utils.websocket_manager import websocket_manager
    from app.services.document_service import DocumentService
    
    # Authenticate WebSocket connection
    try:
        user = await require_websocket_auth(websocket)
        logger.info(f"Transcription progress WebSocket authenticated for user {user.id}, document {document_id}")
    except WebSocketDisconnect:
        logger.warning(f"Transcription progress WebSocket authentication failed for document {document_id}")
        return
    
    # Verify user has access to this document
    from app.core.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        document = await document_service.get_document(document_id, db)
        
        if not document:
            await websocket.close(code=1008, reason="Document not found")
            return
    
    # Connect to WebSocket manager
    await websocket_manager.connect(websocket, str(document_id))
    
    try:
        # Keep connection alive and wait for messages (client can send ping)
        while True:
            try:
                # Wait for messages from client (ping/pong or close)
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break
    except Exception as e:
        logger.error(f"Error in transcription progress WebSocket: {e}")
    finally:
        websocket_manager.disconnect(websocket, str(document_id))


@router.websocket("/{document_id}/summarization-progress")
async def summarization_progress_websocket(
    websocket: WebSocket,
    document_id: UUID
):
    """WebSocket endpoint for real-time summarization progress updates."""
    from app.utils.websocket_auth import require_websocket_auth
    from app.utils.websocket_manager import websocket_manager
    
    # Authenticate WebSocket connection
    try:
        user = await require_websocket_auth(websocket)
        logger.info(f"Summarization progress WebSocket authenticated for user {user.id}, document {document_id}")
    except WebSocketDisconnect:
        logger.warning(f"Summarization progress WebSocket authentication failed for document {document_id}")
        return
    
    # Verify user has access to this document
    from app.core.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        document = await document_service.get_document(document_id, db)
        if not document:
            await websocket.close(code=1008, reason="Document not found")
            return
    
    await websocket_manager.connect(websocket, str(document_id))
    try:
        while True:
            try:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break
    except Exception as e:
        logger.error(f"Error in summarization progress WebSocket: {e}")
    finally:
        websocket_manager.disconnect(websocket, str(document_id))


@router.post("/summarize-missing")
async def summarize_missing_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    limit: int = Query(500, ge=1, le=5000)
):
    """Admin: queue summaries for processed documents lacking a summary."""
    try:
        if not current_user.is_admin():
            raise HTTPException(status_code=403, detail="Admin privileges required")
        from app.core.config import settings as _settings
        if not _settings.SUMMARIZATION_ENABLED:
            raise HTTPException(status_code=400, detail="Summarization disabled")
        res = await db.execute(
            sql_select(_Document.id).where((_Document.is_processed == True) & ((_Document.summary == None) | (_Document.summary == ''))).limit(limit)
        )
        ids = [str(r[0]) for r in res.all()]
        for doc_id in ids:
            try:
                summarize_task.delay(doc_id, False, user_id=str(current_user.id))
            except Exception:
                pass
        return {"queued": len(ids)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error queuing summarization: {e}")
        raise HTTPException(status_code=500, detail="Failed to queue summarization")


# Document Sources endpoints
@router.get("/sources/", response_model=List[DocumentSourceResponse])
async def get_document_sources(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get all document sources."""
    try:
        sources = await document_service.get_document_sources(db)
        return [DocumentSourceResponse.from_orm(source) for source in sources]
    except Exception as e:
        logger.error(f"Error retrieving document sources: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve document sources")


@router.get("/sources/git-active", response_model=List[ActiveSourceStatus])
async def get_active_git_sources(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Return git sources currently syncing or queued for ingestion for the requesting user."""
    try:
        from app.core.cache import get_redis_client
        
        stmt = sql_select(_DocumentSource).where(_DocumentSource.source_type.in_(["github", "gitlab"]))
        result = await db.execute(stmt)
        sources = result.scalars().all()
        active_items: List[ActiveSourceStatus] = []
        for source in sources:
            config = source.config or {}
            requested_by = config.get("requested_by") or config.get("requestedBy")
            if not current_user.is_admin() and requested_by != current_user.username:
                continue
            
            # Check if source is canceled
            source_id_str = str(source.id)
            is_canceled = False
            try:
                rc = await get_redis_client()
                if rc:
                    cancel_flag = await rc.get(f"ingestion:cancel:{source_id_str}")
                    if cancel_flag:
                        is_canceled = True
            except Exception:
                pass
            
            # Skip canceled sources
            if is_canceled:
                continue
            
            pending = False
            task_id = None
            try:
                cached_task = await get_ingestion_task_mapping(str(source.id))
                if cached_task:
                    pending = True
                    task_id = cached_task
            except Exception:
                pending = False
                task_id = None
            if source.is_syncing or pending:
                active_items.append(
                    ActiveSourceStatus(
                        source=DocumentSourceResponse.from_orm(source),
                        pending=bool(pending and not source.is_syncing),
                        task_id=task_id,
                    )
                )
        return active_items
    except Exception as e:
        logger.error(f"Error retrieving active git sources: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve active git sources")


@router.get("/sources/git-search", response_model=List[DocumentSourceResponse])
async def search_git_sources(
    q: Optional[str] = Query(None, description="Case-insensitive search over source name"),
    limit: int = Query(25, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Search git DocumentSources (github/gitlab).

    For non-admin users, restrict to sources whose config indicates they were requested by the user.
    """
    try:
        stmt = sql_select(_DocumentSource).where(_DocumentSource.source_type.in_(["github", "gitlab"]))

        q_norm = (q or "").strip()

        # Prefer most recently updated sources.
        # We do search + ownership filtering in Python to support config fields.
        fetch_limit = 400 if not current_user.is_admin() else 500
        stmt = stmt.order_by(_DocumentSource.updated_at.desc()).limit(fetch_limit)

        result = await db.execute(stmt)
        sources_all = list(result.scalars().all())

        if not current_user.is_admin():
            filtered: list[_DocumentSource] = []
            for src in sources_all:
                cfg = src.config or {}
                requested_user_id = str(cfg.get("requested_by_user_id") or cfg.get("requestedByUserId") or "").strip()
                requested_by = str(cfg.get("requested_by") or cfg.get("requestedBy") or "").strip()
                if requested_user_id and requested_user_id == str(current_user.id):
                    filtered.append(src)
                elif not requested_user_id and requested_by and requested_by == current_user.username:
                    filtered.append(src)
            sources = filtered
        else:
            sources = sources_all

        def _matches_query(src: _DocumentSource, qtext: str) -> bool:
            if not qtext:
                return True
            ql = qtext.lower()
            if ql in (src.name or "").lower():
                return True
            cfg = src.config or {}
            if not isinstance(cfg, dict):
                return False
            display = cfg.get("display")
            if isinstance(display, dict):
                for k in ("url", "repo_url", "repository_url", "html_url", "web_url"):
                    v = str(display.get(k) or "").strip()
                    if v and ql in v.lower():
                        return True
            # github repos
            repos = cfg.get("repos") or cfg.get("repositories")
            if isinstance(repos, list):
                for r in repos:
                    rs = str(r or "").strip()
                    if rs and ql in rs.lower():
                        return True
            # gitlab projects
            gitlab_url = str(cfg.get("gitlab_url") or "").strip()
            if gitlab_url and ql in gitlab_url.lower():
                return True
            projects = cfg.get("projects")
            if isinstance(projects, list):
                for p in projects:
                    if isinstance(p, dict):
                        pid = str(p.get("id") or "").strip()
                        if pid and ql in pid.lower():
                            return True
            return False

        if q_norm:
            sources = [s for s in sources if _matches_query(s, q_norm)]

        sources = sources[:limit]

        return [DocumentSourceResponse.from_orm(source) for source in sources]
    except Exception as e:
        logger.error(f"Error searching git sources: {e}")
        raise HTTPException(status_code=500, detail="Failed to search git sources")


def _token_count(text: Optional[str]) -> int:
    if not text:
        return 0
    return len(re.findall(r"\w+", text.lower()))


def _combine_slide_text(slide: Dict[str, Any]) -> str:
    parts = []
    for key in ("title", "text", "notes"):
        value = slide.get(key)
        if value:
            parts.append(value)
    return "\n".join(parts)


def _align_segments_to_slides(slides: List[Dict[str, Any]], segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not slides or not segments:
        return []
    alignments: List[Dict[str, Any]] = []
    seg_index = 0
    total_segments = len(segments)
    for idx, slide in enumerate(slides):
        if seg_index >= total_segments:
            last_end = alignments[-1]["end"] if alignments else 0.0
            alignments.append({
                "slide_index": slide.get("index", idx + 1),
                "start": last_end,
                "end": last_end,
            })
            continue
        slide_tokens = max(_token_count(_combine_slide_text(slide)), 5)
        start_time = float(segments[seg_index].get("start", 0.0))
        end_time = start_time
        consumed_tokens = 0
        while seg_index < total_segments:
            segment = segments[seg_index]
            seg_tokens = max(_token_count(segment.get("text")), 1)
            consumed_tokens += seg_tokens
            end_time = float(segment.get("end", end_time))
            seg_index += 1
            if consumed_tokens >= slide_tokens or seg_index >= total_segments:
                break
        alignments.append({
            "slide_index": slide.get("index", idx + 1),
            "start": start_time,
            "end": end_time if end_time >= start_time else start_time,
        })
    if alignments and alignments[-1]["end"] < alignments[-1]["start"]:
        alignments[-1]["end"] = alignments[-1]["start"]
    return alignments


async def _ensure_presentation_metadata(document: _Document, db: AsyncSession) -> Optional[Dict[str, Any]]:
    metadata = (document.extra_metadata or {}).get("presentation")
    if metadata and metadata.get("slides"):
        return metadata
    if not document.file_path:
        return None
    suffix = os.path.splitext(document.file_path)[1] or ".pptx"
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file_path = temp_file.name
    temp_file.close()
    try:
        downloaded = await storage_service.download_file(document.file_path, temp_file_path)
        if not downloaded:
            return None
        _, extraction_meta = await text_processor_service.extract_text(temp_file_path, document.file_type)
        presentation_meta = extraction_meta.get("presentation") if extraction_meta else None
        if presentation_meta:
            document.extra_metadata = document.extra_metadata or {}
            document.extra_metadata["presentation"] = presentation_meta
            await db.commit()
            await db.refresh(document)
        return presentation_meta
    finally:
        try:
            os.unlink(temp_file_path)
        except Exception:
            pass


@router.post("/sources/", response_model=DocumentSourceResponse)
async def create_document_source(
    source_data: DocumentSourceCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new document source."""
    try:
        # Check admin privileges for creating sources
        if not current_user.is_admin():
            raise HTTPException(status_code=403, detail="Admin privileges required")
        
        source = await document_service.create_document_source(
            name=source_data.name,
            source_type=source_data.source_type,
            config=source_data.config,
            db=db
        )
        
        return DocumentSourceResponse.from_orm(source)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating document source: {e}")
        raise HTTPException(status_code=500, detail="Failed to create document source")


@router.post("/sources/git-repo", response_model=DocumentSourceResponse)
async def submit_git_repository(
    request: GitRepoSourceRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Allow authenticated users to request processing of Git-compatible repositories.
    Creates a GitHub or GitLab document source scoped to the provided configuration.
    """
    try:
        provider = request.provider.lower()
        repos = request.repositories
        if not repos:
            raise HTTPException(status_code=400, detail="At least one repository must be provided")

        config: Dict[str, Any] = {
            "include_files": request.include_files,
            "include_issues": request.include_issues,
            "include_wiki": request.include_wiki,
            "include_pull_requests": request.include_pull_requests,
            "incremental_files": request.incremental_files,
            "use_gitignore": request.use_gitignore,
            "max_pages": request.max_pages,
            "requested_by": current_user.username,
        }
        if request.token:
            config["token"] = request.token

        source_type = provider
        if provider == "github":
            config["repos"] = repos
        elif provider == "gitlab":
            if not request.token:
                raise HTTPException(status_code=400, detail="GitLab access token is required for repository ingestion")
            gitlab_url = request.gitlab_url or getattr(settings, "GITLAB_URL", None)
            if not gitlab_url:
                raise HTTPException(status_code=400, detail="GitLab URL must be provided for GitLab repositories")
            config["gitlab_url"] = gitlab_url.rstrip("/")
            # GitLab connector expects project dictionaries
            config["projects"] = [
                {
                    "id": repo,
                    "include_files": request.include_files,
                    "include_wikis": request.include_wiki,
                    "include_issues": request.include_issues,
                    "include_merge_requests": request.include_pull_requests,
                }
                for repo in repos
            ]
        else:
            raise HTTPException(status_code=400, detail="Unsupported provider")

        auto_name = request.name or f"{provider.title()} repo ({current_user.username})"
        unique_suffix = uuid4().hex[:6]
        source_name = f"{auto_name.strip()} #{unique_suffix}"

        source = await document_service.create_document_source(
            name=source_name,
            source_type=source_type,
            config=config,
            db=db
        )

        if request.auto_sync:
            try:
                task = ingest_from_source.delay(str(source.id))
                await set_ingestion_task_mapping(str(source.id), task.id, ttl=3600)
            except Exception as sync_err:
                logger.warning(f"Failed to trigger ingestion for {source.id}: {sync_err}")

        return DocumentSourceResponse.from_orm(source)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating git repository source: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit repository for processing")


@router.post("/sources/arxiv", response_model=DocumentSourceResponse)
async def submit_arxiv_request(
    request: ArxivSourceRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Allow authenticated users to ingest papers from ArXiv searches or explicit IDs.
    """
    try:
        config = {
            "queries": request.search_queries or [],
            "paper_ids": request.paper_ids or [],
            "categories": request.categories or [],
            "max_results": request.max_results,
            "start": request.start,
            "sort_by": request.sort_by,
            "sort_order": request.sort_order,
            "requested_by": current_user.username,
            "requested_by_user_id": str(current_user.id),
            "auto_summarize": bool(getattr(request, "auto_summarize", True)),
            "auto_literature_review": bool(getattr(request, "auto_literature_review", False)),
            "auto_enrich_metadata": bool(getattr(request, "auto_enrich_metadata", True)),
            "topic": getattr(request, "topic", None),
            "display": {
                "queries": request.search_queries or [],
                "paper_ids": request.paper_ids or [],
                "categories": request.categories or [],
                "max_results": request.max_results,
            }
        }

        base_name = request.name or "ArXiv search"
        source_name = f"{base_name.strip()} #{uuid4().hex[:6]}"

        source = await document_service.create_document_source(
            name=source_name,
            source_type="arxiv",
            config=config,
            db=db
        )

        if request.auto_sync:
            try:
                ingest_from_source.delay(str(source.id))
            except Exception as sync_err:
                logger.warning(f"Failed to trigger ArXiv ingestion for {source.id}: {sync_err}")

        return DocumentSourceResponse.from_orm(source)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating ArXiv source: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit ArXiv ingestion request")


@router.post("/ingest-arxiv-instant", response_model=InstantArxivIngestResponse)
async def ingest_arxiv_instant(
    request: InstantArxivIngestRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Instantly ingest a single arXiv paper for immediate chat.

    This endpoint performs synchronous processing (~2-3 seconds) so the paper
    is immediately searchable and available for Q&A. Background tasks for
    summarization and metadata enrichment can be optionally queued.

    Accepts arXiv URLs or paper IDs:
    - https://arxiv.org/abs/2401.12345
    - https://arxiv.org/pdf/2401.12345.pdf
    - arxiv:2401.12345
    - 2401.12345
    """
    from app.services.connectors.arxiv_connector import ArxivConnector
    from app.services.vector_store import vector_store_service
    from app.models.document import Document, DocumentChunk
    from app.tasks.summarization_tasks import summarize_document as summarize_task

    arxiv_id = request.arxiv_input  # Already validated/extracted by schema

    try:
        # Check if paper already exists
        existing = await db.execute(
            sql_select(_Document).where(_Document.source_identifier == arxiv_id)
        )
        existing_doc = existing.scalar_one_or_none()

        if existing_doc and existing_doc.is_processed:
            # Paper already indexed, return existing
            metadata = existing_doc.extra_metadata or {}
            return InstantArxivIngestResponse(
                document_id=existing_doc.id,
                arxiv_id=arxiv_id,
                title=existing_doc.title,
                authors=metadata.get("authors", [existing_doc.author] if existing_doc.author else []),
                abstract=existing_doc.content[:1000] if existing_doc.content else "",
                categories=metadata.get("categories", []),
                url=existing_doc.url or f"https://arxiv.org/abs/{arxiv_id}",
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                chunks_created=0,
                ready_for_chat=True,
                background_tasks=["already_indexed"]
            )

        # Fetch paper from arXiv
        connector = ArxivConnector()
        await connector.initialize({"paper_ids": [arxiv_id]})

        docs = await connector.list_documents()
        if not docs:
            raise HTTPException(status_code=404, detail=f"Paper not found on arXiv: {arxiv_id}")

        doc_info = docs[0]
        content = await connector.get_document_content(arxiv_id)
        metadata = await connector.get_document_metadata(arxiv_id)

        await connector.cleanup()

        # Create or update document
        if existing_doc:
            document = existing_doc
            document.title = doc_info["title"]
            document.content = content
            document.content_hash = hashlib.sha256(content.encode()).hexdigest()
            document.url = doc_info.get("url")
            document.author = doc_info.get("author")
            document.extra_metadata = {
                "arxiv": True,
                "authors": metadata.get("authors", []),
                "categories": metadata.get("categories", []),
                "primary_category": metadata.get("primary_category"),
                "doi": metadata.get("doi"),
                "instant_ingest": True,
            }
        else:
            document = _Document(
                id=uuid4(),
                title=doc_info["title"],
                content=content,
                content_hash=hashlib.sha256(content.encode()).hexdigest(),
                source_identifier=arxiv_id,
                url=doc_info.get("url"),
                author=doc_info.get("author"),
                extra_metadata={
                    "arxiv": True,
                    "authors": metadata.get("authors", []),
                    "categories": metadata.get("categories", []),
                    "primary_category": metadata.get("primary_category"),
                    "doi": metadata.get("doi"),
                    "instant_ingest": True,
                }
            )
            db.add(document)

        await db.commit()
        await db.refresh(document)

        # Process chunks synchronously
        chunks_data = text_processor_service.split_text(
            content,
            chunk_size=1000,
            chunk_overlap=200
        )

        # Delete existing chunks if updating
        if existing_doc:
            from sqlalchemy import delete
            await db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
            await db.commit()

        # Create new chunks
        chunks = []
        for idx, chunk_text in enumerate(chunks_data):
            chunk = DocumentChunk(
                id=uuid4(),
                document_id=document.id,
                content=chunk_text,
                chunk_index=idx,
            )
            chunks.append(chunk)
            db.add(chunk)

        await db.commit()

        # Add to vector store synchronously
        await vector_store_service.initialize()
        await vector_store_service.add_document_chunks(document, chunks)

        # Mark as processed
        document.is_processed = True
        await db.commit()

        # Queue background tasks (non-blocking)
        background_tasks = []
        if request.auto_summarize:
            try:
                summarize_task.delay(str(document.id), force=True, user_id=str(current_user.id))
                background_tasks.append("summarization_queued")
            except Exception as e:
                logger.warning(f"Failed to queue summarization: {e}")

        if request.auto_enrich:
            try:
                from app.tasks.paper_enrichment_tasks import enrich_arxiv_document
                enrich_arxiv_document.delay(str(document.id))
                background_tasks.append("enrichment_queued")
            except Exception as e:
                logger.warning(f"Failed to queue enrichment: {e}")

        doc_metadata = document.extra_metadata or {}

        return InstantArxivIngestResponse(
            document_id=document.id,
            arxiv_id=arxiv_id,
            title=document.title,
            authors=doc_metadata.get("authors", []),
            abstract=content[:1500] if content else "",
            categories=doc_metadata.get("categories", []),
            url=document.url or f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            chunks_created=len(chunks),
            ready_for_chat=True,
            background_tasks=background_tasks
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in instant arXiv ingest: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to ingest arXiv paper: {str(e)}")


@router.websocket("/sources/{source_id}/ingestion-progress")
async def document_source_ingestion_progress(
    websocket: WebSocket,
    source_id: UUID,
):
    """WebSocket endpoint for ingestion progress updates (requesting user or admin)."""
    from app.utils.websocket_auth import require_websocket_auth
    from app.utils.websocket_manager import websocket_manager

    try:
        user = await require_websocket_auth(websocket)
    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close(code=1008, reason="Authentication failed")
        return

    # Verify access rights to the source
    try:
        async with AsyncSessionLocal() as session:
            source = await session.get(_DocumentSource, source_id)
    except Exception:
        source = None

    if not source:
        await websocket.close(code=1008, reason="Source not found")
        return

    config = source.config or {}
    requested_by = None
    if isinstance(config, dict):
        requested_by = config.get("requested_by") or config.get("requestedBy")

    if not user.is_admin() and requested_by and requested_by != user.username:
        await websocket.close(code=1008, reason="Not authorized to view this source")
        return
    if not user.is_admin() and not requested_by:
        await websocket.close(code=1008, reason="Progress available only to source owner")
        return

    await websocket_manager.connect(websocket, str(source_id))
    try:
        while True:
            try:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break
    finally:
        websocket_manager.disconnect(websocket, str(source_id))


@router.websocket("/ingest-url/{job_id}/progress")
async def url_ingest_progress(
    websocket: WebSocket,
    job_id: str,
):
    """WebSocket endpoint for ad-hoc URL ingestion progress updates."""
    from app.utils.websocket_auth import require_websocket_auth
    from app.utils.websocket_manager import websocket_manager
    from app.core.cache import get_redis_client

    try:
        user = await require_websocket_auth(websocket)
    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close(code=1008, reason="Authentication failed")
        return

    # Verify ownership (best-effort; requires Redis mapping)
    if not user.is_admin():
        try:
            client = await get_redis_client()
            raw = await client.get(f"url_ingest:owner:{job_id}")
            owner_id = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else None
        except Exception:
            owner_id = None

        if not owner_id or owner_id != str(user.id):
            await websocket.close(code=1008, reason="Not authorized to view this job")
            return

    key = f"url_ingest:{job_id}"
    await websocket_manager.connect(websocket, key)
    try:
        while True:
            try:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break
    finally:
        websocket_manager.disconnect(websocket, key)


@router.post("/ingest-url/{job_id}/cancel")
async def cancel_url_ingest(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Cancel a running URL ingestion job (best-effort)."""
    from app.core.cache import get_redis_client
    from celery.result import AsyncResult

    if not job_id:
        raise HTTPException(status_code=400, detail="job_id required")

    # Verify ownership (best-effort; requires Redis mapping)
    if not current_user.is_admin():
        try:
            client = await get_redis_client()
            raw = await client.get(f"url_ingest:owner:{job_id}")
            owner_id = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else None
        except Exception:
            owner_id = None

        if not owner_id or owner_id != str(current_user.id):
            raise HTTPException(status_code=403, detail="Not authorized to cancel this job")

    # Set cancel flag so cooperative workers can stop between steps
    try:
        client = await get_redis_client()
        await client.setex(f"url_ingest:cancel:{job_id}", 3600, b"1")
    except Exception:
        pass

    # Revoke task (may terminate depending on worker config)
    try:
        AsyncResult(job_id, app=celery_app).revoke(terminate=True)
    except Exception as e:
        logger.warning(f"Failed to revoke URL ingest job {job_id}: {e}")

    # Publish status update to connected clients (best-effort)
    try:
        client = await get_redis_client()
        channel = f"ingestion_progress:url_ingest:{job_id}"
        msg = {
            "type": "status",
            "document_id": f"url_ingest:{job_id}",
            "status": {"stage": "canceled", "status": "Canceled", "progress": 100},
        }
        await client.publish(channel, json.dumps(msg).encode("utf-8"))
    except Exception:
        pass

    return {"message": "Cancel requested", "job_id": job_id}


@router.put("/sources/{source_id}", response_model=DocumentSourceResponse)
async def update_document_source(
    source_id: UUID,
    source_data: DocumentSourceCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update a document source."""
    try:
        if not current_user.is_admin():
            raise HTTPException(status_code=403, detail="Admin privileges required")
        
        source = await document_service.update_document_source(
            source_id=source_id,
            name=source_data.name,
            source_type=source_data.source_type,
            config=source_data.config,
            db=db
        )
        
        if not source:
            raise HTTPException(status_code=404, detail="Document source not found")
        
        return DocumentSourceResponse.from_orm(source)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document source: {e}")
        raise HTTPException(status_code=500, detail="Failed to update document source")


@router.post("/sources/{source_id}/sync")
async def sync_document_source(
    source_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Trigger synchronization for a document source."""
    try:
        if not current_user.is_admin():
            raise HTTPException(status_code=403, detail="Admin privileges required")
        
        success = await document_service.sync_document_source(source_id, db)
        if not success:
            raise HTTPException(status_code=404, detail="Document source not found")
        
        return {"message": "Document source synchronization started"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error syncing document source: {e}")
        raise HTTPException(status_code=500, detail="Failed to sync document source")


@router.delete("/sources/{source_id}")
async def delete_document_source(
    source_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a document source."""
    try:
        if not current_user.is_admin():
            raise HTTPException(status_code=403, detail="Admin privileges required")
        
        success = await document_service.delete_document_source(source_id, db)
        if not success:
            raise HTTPException(status_code=404, detail="Document source not found")
        
        return {"message": "Document source deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document source: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document source")


@router.post("/{document_id}/summarize")
async def summarize_document_endpoint(
    document_id: UUID,
    force: bool = Query(False),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Trigger summarization for a document (background task)."""
    try:
        doc = await document_service.get_document(document_id, db)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        task = summarize_task.delay(str(document_id), force, user_id=str(current_user.id))
        return {"message": "summarization_started", "task_id": task.id}
    except HTTPException:
        raise
    except Exception as e:
        log_error(e, context={"document_id": str(document_id)})
        raise HTTPException(status_code=500, detail="Failed to start summarization")
