"""
API endpoints for DOCX document editing.
"""

import tempfile
import hashlib
from uuid import UUID
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from loguru import logger

from app.core.database import get_db
from app.models.document import Document
from app.services.storage_service import storage_service
from app.services.docx_editor_service import docx_editor_service
from app.schemas.docx_editor import DocxEditResponse, DocxEditRequest, DocxSaveResponse


router = APIRouter()


@router.get("/{document_id}/edit", response_model=DocxEditResponse)
async def get_document_for_editing(
    document_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a DOCX document converted to HTML for editing.

    Returns the document content as HTML that can be edited in a rich text editor.
    """
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    # Fetch document from database
    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    # Check if it's a DOCX file
    if not document.file_path:
        raise HTTPException(status_code=400, detail="Document has no file attached")

    file_ext = Path(document.file_path).suffix.lower()
    if file_ext not in [".docx", ".doc"]:
        raise HTTPException(
            status_code=400,
            detail=f"Only DOCX files can be edited. This file is: {file_ext}"
        )

    # Download file from MinIO to temp location
    temp_file = None
    try:
        temp_file = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        # Download from storage
        file_data = await storage_service.download_file(document.file_path)
        if not file_data:
            raise HTTPException(status_code=404, detail="Document file not found in storage")

        with open(temp_path, "wb") as f:
            f.write(file_data)

        # Convert to HTML
        conversion_result = await docx_editor_service.docx_to_html(temp_path)

        return DocxEditResponse(
            html_content=conversion_result["html_content"],
            document_title=document.title or "Untitled",
            document_id=str(document.id),
            version=conversion_result["version"],
            editable=True,
            warnings=conversion_result.get("warnings")
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to prepare document for editing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load document: {str(e)}")
    finally:
        # Cleanup temp file
        if temp_file:
            try:
                Path(temp_file.name).unlink(missing_ok=True)
            except Exception:
                pass


@router.put("/{document_id}/edit", response_model=DocxSaveResponse)
async def save_document_edits(
    document_id: str,
    request: DocxEditRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Save edited HTML content back to DOCX format.

    Converts the HTML from the editor back to DOCX and uploads to storage.
    """
    try:
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")

    # Fetch document from database
    result = await db.execute(select(Document).where(Document.id == doc_uuid))
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    if not document.file_path:
        raise HTTPException(status_code=400, detail="Document has no file attached")

    # Download original to use as template (preserves styles)
    temp_original = None
    temp_new = None

    try:
        # Download original file
        temp_original = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
        temp_original_path = temp_original.name
        temp_original.close()

        original_data = await storage_service.download_file(document.file_path)
        if original_data:
            with open(temp_original_path, "wb") as f:
                f.write(original_data)
        else:
            temp_original_path = None

        # Create backup if requested
        backup_path = None
        if request.create_backup and original_data:
            backup_path = f"{document.file_path}.backup"
            await storage_service.upload_file(
                backup_path,
                original_data,
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
            logger.info(f"Created backup at {backup_path}")

        # Convert HTML to DOCX
        docx_bytes = await docx_editor_service.html_to_docx(
            request.html_content,
            original_path=temp_original_path
        )

        # Upload new version to MinIO
        await storage_service.upload_file(
            document.file_path,
            docx_bytes,
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )

        # Update document metadata
        new_content_hash = hashlib.sha256(docx_bytes).hexdigest()
        document.content_hash = new_content_hash
        document.file_size = len(docx_bytes)

        # Extract plain text for search indexing (optional background task)
        # background_tasks.add_task(reindex_document, document.id)

        await db.commit()

        # Calculate new version hash
        new_version = hashlib.sha256(request.html_content.encode()).hexdigest()[:16]

        return DocxSaveResponse(
            success=True,
            document_id=str(document.id),
            new_version=new_version,
            message="Document saved successfully",
            backup_path=backup_path
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save document edits: {e}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save document: {str(e)}")
    finally:
        # Cleanup temp files
        if temp_original:
            try:
                Path(temp_original.name).unlink(missing_ok=True)
            except Exception:
                pass
