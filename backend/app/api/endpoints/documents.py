"""
Document-related API endpoints.
"""

import os
from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from app.core.database import get_db
from app.core.rate_limit import limiter, UPLOAD_LIMIT
from app.models.user import User
from app.services.auth_service import get_current_user
from app.services.document_service import DocumentService
from app.utils.exceptions import DocumentNotFoundError, ValidationError
from app.utils.validators import validate_file_type
from app.core.logging import log_error
from app.schemas.document import (
    DocumentResponse,
    DocumentSourceResponse,
    DocumentSourceCreate,
    DocumentUpload
)
from app.schemas.common import PaginatedResponse

router = APIRouter()
document_service = DocumentService()


@router.get("/", response_model=PaginatedResponse[DocumentResponse])
async def get_documents(
    page: int = 1,
    page_size: int = 20,
    source_id: Optional[UUID] = None,
    search: Optional[str] = None,
    order_by: Optional[str] = "updated_at",
    order: Optional[str] = "desc",
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get paginated list of documents.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of items per page (1-100)
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
        if page_size < 1 or page_size > 100:
            raise ValidationError("Page size must be between 1 and 100", field="page_size")
        if order not in ["asc", "desc"]:
            raise ValidationError("Order must be 'asc' or 'desc'", field="order")
        
        # Calculate skip
        skip = (page - 1) * page_size
        
        # Get documents with total count
        documents, total = await document_service.get_documents(
            skip=skip,
            limit=page_size,
            source_id=source_id,
            search=search,
            order_by=order_by,
            order=order,
            db=db
        )
        
        # Convert to response models
        # For list view, we don't need chunks, so set to empty list to avoid lazy loading issues
        items = []
        for doc in documents:
            # Set chunks to empty list for list view (chunks are only needed in detail view)
            # This prevents SQLAlchemy from trying to lazy load the relationship in async context
            doc.chunks = []
            items.append(DocumentResponse.from_orm(doc))
        
        # Return paginated response
        return PaginatedResponse.create(
            items=items,
            total=total,
            page=page,
            page_size=page_size
        )
    except ValidationError:
        raise
    except Exception as e:
        log_error(e, context={"page": page, "page_size": page_size})
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")


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
        # Reset the filename and content_type attributes
        file.filename = file.filename
        file.content_type = file.content_type
        
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
            db=db
        )
        
        return {"message": "Document uploaded successfully", "document_id": document.id}
    
    except (ValidationError, DocumentNotFoundError):
        raise
    except Exception as e:
        error_detail = str(e)
        logger.error(f"Error uploading document {file.filename if file else 'unknown'}: {error_detail}", exc_info=True)
        log_error(e, context={"filename": file.filename if file else None, "file_type": file.content_type if file else None})
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {error_detail}")


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


@router.post("/reprocess/{document_id}")
async def reprocess_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Reprocess a document for indexing."""
    try:
        success = await document_service.reprocess_document(document_id, db)
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
