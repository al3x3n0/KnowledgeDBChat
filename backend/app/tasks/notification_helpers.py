"""
Helper functions for emitting notifications from Celery tasks.

These functions are synchronous wrappers around the async notification service,
designed to be called from Celery task implementations.
"""

import asyncio
from typing import Optional, Dict, Any
from uuid import UUID
from loguru import logger

from app.core.database import create_celery_session
from app.services.notification_service import notification_service
from app.models.notification import NotificationType


def emit_document_processing_notification(
    user_id: str,
    document_id: str,
    document_title: str,
    success: bool,
    error: Optional[str] = None,
    chunks_count: int = 0,
) -> None:
    """
    Emit notification for document processing completion.

    Args:
        user_id: User ID to notify
        document_id: Document ID that was processed
        document_title: Title of the document
        success: Whether processing succeeded
        error: Error message if failed
        chunks_count: Number of chunks created
    """
    try:
        asyncio.run(_async_emit_document_processing_notification(
            user_id, document_id, document_title, success, error, chunks_count
        ))
    except Exception as e:
        logger.warning(f"Failed to emit document processing notification: {e}")


async def _async_emit_document_processing_notification(
    user_id: str,
    document_id: str,
    document_title: str,
    success: bool,
    error: Optional[str] = None,
    chunks_count: int = 0,
) -> None:
    """Async implementation of document processing notification."""
    async with create_celery_session()() as db:
        try:
            if success:
                await notification_service.create_notification(
                    db=db,
                    user_id=UUID(user_id),
                    notification_type=NotificationType.DOCUMENT_PROCESSING_COMPLETE,
                    title="Document Processed",
                    message=f"'{document_title}' has been processed successfully with {chunks_count} chunks.",
                    priority="normal",
                    related_entity_type="document",
                    related_entity_id=UUID(document_id),
                    data={"document_title": document_title, "chunks_count": chunks_count},
                    action_url=f"/documents",
                )
            else:
                await notification_service.create_notification(
                    db=db,
                    user_id=UUID(user_id),
                    notification_type=NotificationType.DOCUMENT_PROCESSING_ERROR,
                    title="Document Processing Failed",
                    message=f"Failed to process '{document_title}': {error or 'Unknown error'}",
                    priority="high",
                    related_entity_type="document",
                    related_entity_id=UUID(document_id),
                    data={"document_title": document_title, "error": error},
                    action_url=f"/documents",
                )
        except Exception as e:
            logger.error(f"Failed to emit document processing notification: {e}")


def emit_ingestion_notification(
    user_id: str,
    source_id: str,
    source_name: str,
    success: bool,
    total: int = 0,
    created: int = 0,
    updated: int = 0,
    errors: int = 0,
    error_message: Optional[str] = None,
) -> None:
    """
    Emit notification for source ingestion completion.

    Args:
        user_id: User ID to notify
        source_id: Source ID that was synced
        source_name: Name of the source
        success: Whether ingestion succeeded
        total: Total documents processed
        created: Documents created
        updated: Documents updated
        errors: Number of errors
        error_message: Error message if failed
    """
    try:
        asyncio.run(_async_emit_ingestion_notification(
            user_id, source_id, source_name, success, total, created, updated, errors, error_message
        ))
    except Exception as e:
        logger.warning(f"Failed to emit ingestion notification: {e}")


async def _async_emit_ingestion_notification(
    user_id: str,
    source_id: str,
    source_name: str,
    success: bool,
    total: int = 0,
    created: int = 0,
    updated: int = 0,
    errors: int = 0,
    error_message: Optional[str] = None,
) -> None:
    """Async implementation of ingestion notification."""
    async with create_celery_session()() as db:
        try:
            if success:
                await notification_service.create_notification(
                    db=db,
                    user_id=UUID(user_id),
                    notification_type=NotificationType.DOCUMENT_INGESTION_COMPLETE,
                    title="Sync Complete",
                    message=f"'{source_name}' sync completed: {created} new, {updated} updated, {errors} errors.",
                    priority="normal",
                    related_entity_type="source",
                    related_entity_id=UUID(source_id),
                    data={
                        "source_name": source_name,
                        "total": total,
                        "created": created,
                        "updated": updated,
                        "errors": errors,
                    },
                    action_url="/admin",
                )
            else:
                await notification_service.create_notification(
                    db=db,
                    user_id=UUID(user_id),
                    notification_type=NotificationType.DOCUMENT_INGESTION_ERROR,
                    title="Sync Failed",
                    message=f"'{source_name}' sync failed: {error_message or 'Unknown error'}",
                    priority="high",
                    related_entity_type="source",
                    related_entity_id=UUID(source_id),
                    data={"source_name": source_name, "error": error_message},
                    action_url="/admin",
                )
        except Exception as e:
            logger.error(f"Failed to emit ingestion notification: {e}")


def emit_transcription_notification(
    user_id: str,
    document_id: str,
    document_title: str,
    success: bool,
    duration_seconds: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """
    Emit notification for transcription completion.

    Args:
        user_id: User ID to notify
        document_id: Document ID that was transcribed
        document_title: Title of the document
        success: Whether transcription succeeded
        duration_seconds: Duration of the transcription
        error: Error message if failed
    """
    try:
        asyncio.run(_async_emit_transcription_notification(
            user_id, document_id, document_title, success, duration_seconds, error
        ))
    except Exception as e:
        logger.warning(f"Failed to emit transcription notification: {e}")


async def _async_emit_transcription_notification(
    user_id: str,
    document_id: str,
    document_title: str,
    success: bool,
    duration_seconds: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    """Async implementation of transcription notification."""
    async with create_celery_session()() as db:
        try:
            if success:
                duration_str = ""
                if duration_seconds:
                    minutes = int(duration_seconds // 60)
                    seconds = int(duration_seconds % 60)
                    duration_str = f" ({minutes}m {seconds}s)"

                await notification_service.create_notification(
                    db=db,
                    user_id=UUID(user_id),
                    notification_type=NotificationType.DOCUMENT_TRANSCRIPTION_COMPLETE,
                    title="Transcription Complete",
                    message=f"'{document_title}' has been transcribed{duration_str}.",
                    priority="normal",
                    related_entity_type="document",
                    related_entity_id=UUID(document_id),
                    data={"document_title": document_title, "duration_seconds": duration_seconds},
                    action_url=f"/documents",
                )
            else:
                await notification_service.create_notification(
                    db=db,
                    user_id=UUID(user_id),
                    notification_type=NotificationType.DOCUMENT_TRANSCRIPTION_ERROR,
                    title="Transcription Failed",
                    message=f"Failed to transcribe '{document_title}': {error or 'Unknown error'}",
                    priority="high",
                    related_entity_type="document",
                    related_entity_id=UUID(document_id),
                    data={"document_title": document_title, "error": error},
                    action_url=f"/documents",
                )
        except Exception as e:
            logger.error(f"Failed to emit transcription notification: {e}")


def emit_summarization_notification(
    user_id: str,
    document_id: str,
    document_title: str,
    success: bool,
    error: Optional[str] = None,
) -> None:
    """
    Emit notification for summarization completion.

    Args:
        user_id: User ID to notify
        document_id: Document ID that was summarized
        document_title: Title of the document
        success: Whether summarization succeeded
        error: Error message if failed
    """
    try:
        asyncio.run(_async_emit_summarization_notification(
            user_id, document_id, document_title, success, error
        ))
    except Exception as e:
        logger.warning(f"Failed to emit summarization notification: {e}")


async def _async_emit_summarization_notification(
    user_id: str,
    document_id: str,
    document_title: str,
    success: bool,
    error: Optional[str] = None,
) -> None:
    """Async implementation of summarization notification."""
    async with create_celery_session()() as db:
        try:
            if success:
                await notification_service.create_notification(
                    db=db,
                    user_id=UUID(user_id),
                    notification_type=NotificationType.DOCUMENT_SUMMARIZATION_COMPLETE,
                    title="Summary Generated",
                    message=f"Summary for '{document_title}' has been generated.",
                    priority="low",
                    related_entity_type="document",
                    related_entity_id=UUID(document_id),
                    data={"document_title": document_title},
                    action_url=f"/documents",
                )
            else:
                # Use processing error type for summarization errors
                await notification_service.create_notification(
                    db=db,
                    user_id=UUID(user_id),
                    notification_type=NotificationType.DOCUMENT_PROCESSING_ERROR,
                    title="Summarization Failed",
                    message=f"Failed to summarize '{document_title}': {error or 'Unknown error'}",
                    priority="normal",
                    related_entity_type="document",
                    related_entity_id=UUID(document_id),
                    data={"document_title": document_title, "error": error},
                    action_url=f"/documents",
                )
        except Exception as e:
            logger.error(f"Failed to emit summarization notification: {e}")


def emit_sync_notification(
    user_id: str,
    source_id: str,
    source_name: str,
    success: bool,
    documents_synced: int = 0,
    error: Optional[str] = None,
) -> None:
    """
    Emit notification for source sync completion.

    Args:
        user_id: User ID to notify
        source_id: Source ID that was synced
        source_name: Name of the source
        success: Whether sync succeeded
        documents_synced: Number of documents synced
        error: Error message if failed
    """
    try:
        asyncio.run(_async_emit_sync_notification(
            user_id, source_id, source_name, success, documents_synced, error
        ))
    except Exception as e:
        logger.warning(f"Failed to emit sync notification: {e}")


async def _async_emit_sync_notification(
    user_id: str,
    source_id: str,
    source_name: str,
    success: bool,
    documents_synced: int = 0,
    error: Optional[str] = None,
) -> None:
    """Async implementation of sync notification."""
    async with create_celery_session()() as db:
        try:
            if success:
                await notification_service.create_notification(
                    db=db,
                    user_id=UUID(user_id),
                    notification_type=NotificationType.DOCUMENT_SYNC_COMPLETE,
                    title="Source Synced",
                    message=f"'{source_name}' has been synced. {documents_synced} documents updated.",
                    priority="normal",
                    related_entity_type="source",
                    related_entity_id=UUID(source_id),
                    data={"source_name": source_name, "documents_synced": documents_synced},
                    action_url="/admin",
                )
            else:
                await notification_service.create_notification(
                    db=db,
                    user_id=UUID(user_id),
                    notification_type=NotificationType.DOCUMENT_SYNC_ERROR,
                    title="Sync Failed",
                    message=f"Failed to sync '{source_name}': {error or 'Unknown error'}",
                    priority="high",
                    related_entity_type="source",
                    related_entity_id=UUID(source_id),
                    data={"source_name": source_name, "error": error},
                    action_url="/admin",
                )
        except Exception as e:
            logger.error(f"Failed to emit sync notification: {e}")


def emit_admin_broadcast(
    title: str,
    message: str,
    priority: str = "normal",
    action_url: Optional[str] = None,
    target_roles: Optional[list] = None,
) -> None:
    """
    Emit a broadcast notification to all users or specific roles.

    Args:
        title: Notification title
        message: Notification message
        priority: Priority level (low, normal, high, urgent)
        action_url: Optional URL to navigate to
        target_roles: Optional list of roles to target (None = all users)
    """
    try:
        asyncio.run(_async_emit_admin_broadcast(
            title, message, priority, action_url, target_roles
        ))
    except Exception as e:
        logger.warning(f"Failed to emit admin broadcast: {e}")


async def _async_emit_admin_broadcast(
    title: str,
    message: str,
    priority: str = "normal",
    action_url: Optional[str] = None,
    target_roles: Optional[list] = None,
) -> None:
    """Async implementation of admin broadcast."""
    async with create_celery_session()() as db:
        try:
            await notification_service.create_broadcast_notification(
                db=db,
                notification_type=NotificationType.ADMIN_BROADCAST,
                title=title,
                message=message,
                priority=priority,
                action_url=action_url,
                target_roles=target_roles,
            )
        except Exception as e:
            logger.error(f"Failed to emit admin broadcast: {e}")
