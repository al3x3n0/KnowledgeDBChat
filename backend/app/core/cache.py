"""
Caching utilities using Redis.
"""

import asyncio
import pickle
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

try:
    import redis.asyncio as aioredis
except ImportError:
    import aioredis

from loguru import logger

from app.core.config import settings

T = TypeVar("T")

#: One client per event loop, not one per process.
#:
#: An asyncio Redis client binds to the loop that created it, and celery runs
#: each task in a FRESH loop -- the same reason `create_celery_session` builds
#: an engine per invocation rather than sharing one. A process-global client
#: therefore worked for exactly one task per worker and then failed every
#: call with "Event loop is closed". It failed quietly: `CacheService.get`
#: catches and returns None, so feature flags silently stopped consulting
#: Redis in workers and fell back to settings, which looks like flags simply
#: not taking effect.
#:
#: Keyed by id(loop) and swept of closed loops on each access, so a worker
#: that runs thousands of tasks does not accumulate thousands of clients.
_redis_clients: Dict[int, aioredis.Redis] = {}


async def get_redis_client() -> aioredis.Redis:
    """
    Get or create the Redis client for the running event loop.

    Returns:
        Redis client instance bound to this loop
    """
    loop = asyncio.get_running_loop()
    for key in [k for k, c in _redis_clients.items() if k != id(loop)]:
        # Its loop is gone; the client cannot be used or cleanly closed from
        # here, so drop the reference and let it be collected.
        _redis_clients.pop(key, None)

    client = _redis_clients.get(id(loop))
    if client is None:
        client = aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=False,  # We'll handle encoding ourselves
        )
        _redis_clients[id(loop)] = client
    return client


async def close_redis_client():
    """Close the client for the running loop, if there is one."""
    try:
        loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_id = None
    client = _redis_clients.pop(loop_id, None) if loop_id is not None else None
    if client:
        await client.close()


class CacheService:
    """Service for caching data in Redis."""

    def __init__(self):
        # Deliberately holds no client. `cache_service` is a module singleton
        # shared by every request and every celery task, so caching one here
        # would pin the first loop's client for the life of the process --
        # exactly the bug get_redis_client was changed to avoid, one level up.
        pass

    async def _get_client(self) -> aioredis.Redis:
        """The client for the loop this call is running on."""
        return await get_redis_client()

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        try:
            client = await self._get_client()
            data = await client.get(key)
            if data is None:
                return None
            return pickle.loads(data)
        except Exception as e:
            logger.warning(f"Cache get error for key {key}: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None for no expiration)

        Returns:
            True if successful, False otherwise
        """
        try:
            client = await self._get_client()
            data = pickle.dumps(value)
            if ttl:
                await client.setex(key, ttl, data)
            else:
                await client.set(key, data)
            return True
        except Exception as e:
            logger.warning(f"Cache set error for key {key}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            client = await self._get_client()
            await client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a pattern.

        Args:
            pattern: Pattern to match (e.g., "user:*")

        Returns:
            Number of keys deleted
        """
        try:
            client = await self._get_client()
            keys = []
            async for key in client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                return await client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Cache delete pattern error for {pattern}: {e}")
            return 0

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """
        try:
            client = await self._get_client()
            return await client.exists(key) > 0
        except Exception as e:
            logger.warning(f"Cache exists check error for key {key}: {e}")
            return False


# Global cache service instance
cache_service = CacheService()


def cached(
    key_prefix: str, ttl: int = 300, key_func: Optional[Callable[..., str]] = None
):
    """
    Decorator for caching function results.

    Args:
        key_prefix: Prefix for cache keys
        ttl: Time to live in seconds (default: 5 minutes)
        key_func: Optional function to generate cache key from arguments

    Example:
        @cached("user", ttl=600)
        async def get_user(user_id: str):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default: use args and kwargs
                key_parts = [key_prefix]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)

            # Try to get from cache
            cached_value = await cache_service.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for key: {cache_key}")
                return cached_value

            # Call function and cache result
            logger.debug(f"Cache miss for key: {cache_key}")
            result = await func(*args, **kwargs)
            await cache_service.set(cache_key, result, ttl=ttl)

            return result

        return wrapper

    return decorator
