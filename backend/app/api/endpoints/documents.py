"""
Document-related API endpoints.
"""

from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
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
        
        # Validate file size (max 50MB)
        MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
        file_content = await file.read()
        file_size = len(file_content)
        if file_size > MAX_FILE_SIZE:
            raise ValidationError(
                f"File size ({file_size / (1024*1024):.2f}MB) exceeds maximum allowed size of {MAX_FILE_SIZE / (1024*1024)}MB",
                field="file"
            )
        
        # Create a new file-like object with the content for the service
        from io import BytesIO
        file.file = BytesIO(file_content)
        
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
        log_error(e, context={"filename": file.filename if file else None})
        raise HTTPException(status_code=500, detail="Failed to upload document")


@router.delete("/{document_id}")
async def delete_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Delete a document."""
    try:
        success = await document_service.delete_document(document_id, db)
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {"message": "Document deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")


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


