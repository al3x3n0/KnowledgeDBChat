"""
Caching utilities using Redis.

Values are stored as JSON. They used to be pickled, and reading a pickle runs
whatever it describes: anything able to write to Redis could execute code in
every API and worker process. Nothing cached needs more than JSON -- feature
flags, small payload dicts, ids -- now that the document cache is gone (it
pickled whole ORM rows, content included, and nothing ever read them back).

Values written by the old code are still read, through an unpickler that
refuses every class and function, so it can only build plain data. That
matters for feature flags, which have no expiry: without it every flag set at
runtime would silently fall back to its default on the deploy that changed
the format.
"""

import io
import json
import pickle
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

try:
    import redis.asyncio as aioredis
except ImportError:
    import aioredis

from loguru import logger

from app.core.config import settings
from app.utils.per_loop import PerLoop

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
#: One per loop, dropped when its loop closes (`utils.per_loop`). It used to
#: drop every client but the caller's on each access, which is right only
#: while a process has one live loop at a time: two jobs in two threads each
#: evicted the other's client on every call.
def _new_redis_client() -> aioredis.Redis:
    return aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=False,  # We'll handle encoding ourselves
    )


_redis_clients: PerLoop[aioredis.Redis] = PerLoop(_new_redis_client)


async def get_redis_client() -> aioredis.Redis:
    """
    Get or create the Redis client for the running event loop.

    Returns:
        Redis client instance bound to this loop
    """
    return _redis_clients.get()


async def close_redis_client():
    """Close the client for the running loop, if there is one."""
    try:
        client = _redis_clients.pop()
    except RuntimeError:
        client = None
    if client:
        await client.close()


class _PlainDataUnpickler(pickle.Unpickler):
    """Reads a pickle of plain data and nothing else.

    Every way a pickle runs code goes through ``find_class`` (to fetch the
    class or function to call). Refusing it leaves the opcodes that build
    None, booleans, numbers, strings, bytes, lists, tuples, dicts and sets.
    """

    def find_class(self, module: str, name: str):
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")


def _decode(data: Any) -> Any:
    """A cached value: JSON, or plain data pickled by an earlier version."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    try:
        return json.loads(data)
    except (ValueError, UnicodeDecodeError):
        return _PlainDataUnpickler(io.BytesIO(data)).load()


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
            return _decode(data)
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
            # Strict: a value JSON cannot represent is a caller's mistake to
            # see, not something to coerce into a string and read back wrong.
            data = json.dumps(value)
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
