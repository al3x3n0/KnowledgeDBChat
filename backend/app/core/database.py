"""
Database configuration and connection management.
"""

import asyncio

from loguru import logger
from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from .config import settings

# Convert sync database URL to async for async operations
async_database_url = settings.DATABASE_URL.replace(
    "postgresql://", "postgresql+asyncpg://"
)

# Create async engine
engine = create_async_engine(
    async_database_url,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    pool_recycle=settings.DB_POOL_RECYCLE_SECONDS,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT_SECONDS,
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Alias used by callers that need a bare session factory (e.g. WebSocket
# endpoints that manage their own session lifecycle outside the get_db
# dependency).
async_session_factory = AsyncSessionLocal

# Create declarative base
Base = declarative_base()

# Metadata for table creation
metadata = MetaData()

_db_session_semaphore: asyncio.Semaphore | None = None


def _get_db_session_semaphore() -> asyncio.Semaphore:
    global _db_session_semaphore
    if _db_session_semaphore is None:
        limit = settings.DB_SESSION_CONCURRENCY_LIMIT
        if limit is None:
            limit = max(1, int(settings.DB_POOL_SIZE) + int(settings.DB_MAX_OVERFLOW))
        else:
            limit = max(1, int(limit))
        _db_session_semaphore = asyncio.Semaphore(limit)
        logger.info(f"DB session concurrency limit: {limit}")
    return _db_session_semaphore


async def get_db() -> AsyncSession:
    """
    Get database session.

    Yields:
        AsyncSession: Database session
    """
    semaphore = _get_db_session_semaphore()
    try:
        await asyncio.wait_for(
            semaphore.acquire(),
            timeout=float(settings.DB_SESSION_ACQUIRE_TIMEOUT_SECONDS),
        )
    except asyncio.TimeoutError:
        # Avoid dogpiling the pool; fail fast under load.
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server is busy (DB concurrency limit reached). Please retry in a moment.",
            headers={"Retry-After": "2"},
        )

    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            # Don't log HTTPException as database errors - they're expected API responses
            from fastapi import HTTPException
            from fastapi.exceptions import RequestValidationError

            if not isinstance(e, (HTTPException, RequestValidationError)):
                logger.opt(exception=True).error("Database session error: {}", str(e))
            await session.rollback()
            raise
        finally:
            await session.close()
            try:
                semaphore.release()
            except Exception:
                pass


async def create_tables():
    """Create tables straight from the models, for tests and throwaway databases.

    This is NOT how a real database is built. Alembic owns the schema; use
    ``alembic upgrade head`` (``make db-migrate``). Creating tables from model
    metadata skips every migration, leaves ``alembic_version`` unstamped, and so
    silently omits anything a migration adds that metadata cannot express —
    triggers, partial indexes, backfills.
    """
    try:
        async with engine.begin() as conn:
            # Import all models to ensure they're registered
            import app.models  # noqa: F401

            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created from model metadata")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


async def drop_tables():
    """Drop all database tables."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Error dropping database tables: {e}")
        raise


def create_celery_session():
    """
    Create a fresh async session for Celery tasks.

    Celery workers fork from the main process, and the global engine's connection pool
    is bound to the parent's event loop. When asyncio.run() creates a new event loop
    in the worker, the old engine is incompatible, causing "Future attached to a
    different loop" errors.

    This function creates a fresh engine and session factory for each task invocation.

    Returns:
        AsyncSession factory (sessionmaker instance)
    """
    if settings.CELERY_DB_USE_NULLPOOL:
        # NullPool doesn't support QueuePool tuning kwargs like pool_size/max_overflow/pool_timeout.
        # It's also pointless to pre-ping/recycle when every checkout creates a fresh connection.
        kwargs = dict(
            echo=settings.DEBUG,
            poolclass=NullPool,
        )
    else:
        kwargs = dict(
            echo=settings.DEBUG,
            pool_pre_ping=True,
            pool_recycle=settings.DB_POOL_RECYCLE_SECONDS,
            pool_timeout=settings.CELERY_DB_POOL_TIMEOUT_SECONDS,
            pool_size=settings.CELERY_DB_POOL_SIZE,
            max_overflow=settings.CELERY_DB_MAX_OVERFLOW,
        )

    fresh_engine = create_async_engine(async_database_url, **kwargs)
    return sessionmaker(
        fresh_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
