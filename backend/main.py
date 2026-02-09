"""
Main FastAPI application entry point for Knowledge Database Chat.
"""

import os
import logging
from contextlib import asynccontextmanager
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
import uvicorn

# Disable ChromaDB/PostHog telemetry as early as possible (before any chromadb import)
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "0")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "0")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "0")
os.environ.setdefault("POSTHOG_DISABLED", "1")

# Silence noisy telemetry logger in case dependencies still attempt it
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

from app.core.config import settings
from app.core.exceptions import (
    knowledge_db_exception_handler,
    validation_exception_handler,
    generic_exception_handler,
)
from app.core.middleware import LoggingMiddleware, SecurityHeadersMiddleware
from app.core.rate_limit import limiter
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from app.api.routes import api_router
from app.mcp.server import mcp_router
from app.utils.exceptions import KnowledgeDBException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Knowledge Database Chat application")

    async def _start_init_task(name: str, coro, timeout_s: float = 20.0) -> None:
        """
        Start an initialization coroutine in the background and (optionally) wait briefly.

        Important for hot-reload: we keep the Task handle so it can be cancelled on shutdown/reload.
        Avoid creating shielded coroutines without tracking them, which can keep the loop alive and
        make uvicorn reload hang while waiting for background tasks.
        """
        task = asyncio.create_task(coro, name=f"init:{name}")
        getattr(app.state, "background_tasks", []).append(task)
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout_s)
            logger.info(f"{name} initialized")
        except asyncio.TimeoutError:
            logger.warning(f"{name} init timed out after {timeout_s}s (continuing startup)")
        except Exception as e:
            logger.warning(f"{name} init failed: {e} (continuing startup)")

    # Apply idempotent minimal migrations for environments without Alembic.
    # Don't block the server indefinitely; allow UI to load even if DB is still coming up.
    try:
        from app.core.database import apply_minimal_migrations
        await asyncio.wait_for(apply_minimal_migrations(), timeout=30.0)
    except asyncio.TimeoutError:
        logger.warning("Minimal migrations timed out after 30s (continuing startup)")
    except Exception as e:
        logger.warning(f"Minimal migrations failed: {e} (continuing startup)")

    # Kick off heavy/optional initializations in background so the server can start serving immediately.
    app.state.background_tasks = []

    # Vector store initialization can be slow (model load). Most services also lazy-init it on demand.
    from app.services.vector_store import vector_store_service
    await _start_init_task("Vector store", vector_store_service.initialize(background=True), timeout_s=2.0)

    # MinIO storage service (uploads/downloads)
    from app.services.storage_service import storage_service
    await _start_init_task("Storage service", storage_service.initialize(), timeout_s=20.0)

    # Redis subscriber for progress updates (transcription/summarization/ingestion)
    from app.utils.redis_subscriber import redis_subscriber
    # This is a long-running loop; start it and rely on explicit stop/cancel on shutdown.
    try:
        sub_task = asyncio.create_task(redis_subscriber.start(), name="bg:redis_subscriber")
        app.state.background_tasks.append(sub_task)
        logger.info("Redis subscriber started")
    except Exception as e:
        logger.warning(f"Redis subscriber start failed: {e} (continuing startup)")
    
    logger.info("Application startup complete")
    logger.info(
        "READY_TO_SERVE http://{}:{} (pid={})",
        settings.HOST,
        settings.PORT,
        os.getpid(),
    )
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    
    # Stop Redis subscriber
    try:
        await redis_subscriber.stop()
    except Exception as e:
        logger.warning(f"Error stopping Redis subscriber: {e}")
    
    # Cancel any background tasks we started (best-effort).
    tasks = list(getattr(app.state, "background_tasks", []) or [])
    for t in tasks:
        try:
            if not t.done():
                t.cancel()
        except Exception:
            pass
    if tasks:
        try:
            # Don't hang shutdown/reload forever in dev.
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=2.0)
        except Exception:
            pass


# Create FastAPI app
app = FastAPI(
    title="Knowledge Database Chat API",
    description="LLM-powered chat interface for organizational knowledge",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add custom middleware (order matters - first added is outermost)
app.add_middleware(LoggingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add exception handlers
app.add_exception_handler(KnowledgeDBException, knowledge_db_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Include MCP server routes (for external AI agents)
app.include_router(mcp_router, prefix="/api/v1")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Knowledge Database Chat API",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS
    )
