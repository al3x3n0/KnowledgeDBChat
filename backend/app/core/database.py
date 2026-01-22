"""
Database configuration and connection management.
"""

from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from loguru import logger

from .config import settings

# Convert sync database URL to async for async operations
async_database_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Create async engine
engine = create_async_engine(
    async_database_url,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    pool_recycle=300,
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Create declarative base
Base = declarative_base()

# Metadata for table creation
metadata = MetaData()


async def get_db() -> AsyncSession:
    """
    Get database session.
    
    Yields:
        AsyncSession: Database session
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            # Don't log HTTPException as database errors - they're expected API responses
            from fastapi import HTTPException
            if not isinstance(e, HTTPException):
                # Use % formatting to avoid issues with curly braces in exception messages
                logger.error("Database session error: %s", str(e), exc_info=True)
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_tables():
    """Create database tables."""
    try:
        async with engine.begin() as conn:
            # Import all models to ensure they're registered
            from app.models import document, chat, user, upload_session, knowledge_graph
            
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")

            # Apply lightweight migrations for columns added after initial release
            await _apply_minimal_migrations(conn)
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
    fresh_engine = create_async_engine(
        async_database_url,
        echo=settings.DEBUG,
        pool_pre_ping=True,
        pool_recycle=300,
    )
    return sessionmaker(
        fresh_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def _apply_minimal_migrations(conn) -> None:
    """
    Apply idempotent schema updates that are safe to run on every startup.
    This supplements Alembic for environments that only rely on create_all.
    """
    statements = [
        # Document summarization columns (introduced after initial schema)
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary TEXT",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary_model VARCHAR(100)",
        "ALTER TABLE documents ADD COLUMN IF NOT EXISTS summary_generated_at TIMESTAMPTZ",
    ]

    for stmt in statements:
        try:
            await conn.execute(text(stmt))
        except Exception as exc:
            logger.warning(f"Minimal migration statement failed ({stmt}): {exc}")




