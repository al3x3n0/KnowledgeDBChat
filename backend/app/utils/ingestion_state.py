"""
Utility helpers for managing ingestion state flags in Redis.
"""

from typing import Optional
from loguru import logger
from app.core.cache import get_redis_client


async def _decode(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode()
        except Exception:
            try:
                return value.decode("utf-8", errors="ignore")
            except Exception:
                return None
    return value


async def set_ingestion_task_mapping(source_id: str, task_id: str, ttl: int = 3600) -> None:
    """Store the celery task id for an ingestion request."""
    try:
        client = await get_redis_client()
        await client.setex(f"ingestion:task:{source_id}", ttl, str(task_id))
    except Exception as exc:
        logger.warning(f"Failed to set ingestion task mapping for {source_id}: {exc}")


async def get_ingestion_task_mapping(source_id: str) -> Optional[str]:
    """Fetch a previously stored celery task id for the given source."""
    try:
        client = await get_redis_client()
        value = await client.get(f"ingestion:task:{source_id}")
        return await _decode(value)
    except Exception as exc:
        logger.warning(f"Failed to read ingestion task mapping for {source_id}: {exc}")
        return None


async def set_ingestion_cancel_flag(source_id: str, ttl: int = 600) -> None:
    """Mark an ingestion task as canceled so the worker loop can exit."""
    try:
        client = await get_redis_client()
        await client.setex(f"ingestion:cancel:{source_id}", ttl, "1")
    except Exception as exc:
        logger.warning(f"Failed to set ingestion cancel flag for {source_id}: {exc}")


async def set_force_full_flag(source_id: str, ttl: int = 600) -> None:
    """Indicate that the next ingestion should run as a full sync."""
    try:
        client = await get_redis_client()
        await client.setex(f"ingestion:force_full:{source_id}", ttl, "1")
    except Exception as exc:
        logger.warning(f"Failed to set force-full flag for {source_id}: {exc}")


# Git comparison helpers -----------------------------------------------------

async def set_git_compare_task(diff_id: str, task_id: str, ttl: int = 3600) -> None:
    """Store Celery task id for a git branch comparison job."""
    try:
        client = await get_redis_client()
        await client.setex(f"gitcompare:task:{diff_id}", ttl, str(task_id))
    except Exception as exc:
        logger.warning(f"Failed to set git compare task mapping for {diff_id}: {exc}")


async def get_git_compare_task(diff_id: str) -> Optional[str]:
    """Return Celery task id for a git branch comparison job."""
    try:
        client = await get_redis_client()
        value = await client.get(f"gitcompare:task:{diff_id}")
        return await _decode(value)
    except Exception as exc:
        logger.warning(f"Failed to fetch git compare task mapping for {diff_id}: {exc}")
        return None


async def set_git_compare_cancel_flag(diff_id: str, ttl: int = 600) -> None:
    """Mark a git comparison job as canceled."""
    try:
        client = await get_redis_client()
        await client.setex(f"gitcompare:cancel:{diff_id}", ttl, "1")
    except Exception as exc:
        logger.warning(f"Failed to set git compare cancel flag for {diff_id}: {exc}")


async def is_git_compare_cancelled(diff_id: str) -> bool:
    """Check if a cancel flag was raised for the given job."""
    try:
        client = await get_redis_client()
        value = await client.get(f"gitcompare:cancel:{diff_id}")
        return bool(value)
    except Exception as exc:
        logger.warning(f"Failed to read git compare cancel flag for {diff_id}: {exc}")
        return False


async def clear_git_compare_task(diff_id: str) -> None:
    """Remove git comparison task mapping and cancel flag."""
    try:
        client = await get_redis_client()
        await client.delete(f"gitcompare:task:{diff_id}")
        await client.delete(f"gitcompare:cancel:{diff_id}")
    except Exception as exc:
        logger.warning(f"Failed to clear git compare state for {diff_id}: {exc}")
